//! Shared application state: one pooled HTTP client, the artifact cache, and the sync tools, wired
//! from `Config` and cloned (behind `Arc`) into every connection.

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{Semaphore, SemaphorePermit};

/// Admission and the 64 MiB memory-cache cap are one 512 MiB cgroup budget. Three concurrent
/// Tier-1 children measure about 200 MiB total; CI additionally forces 112 MiB resident for a full
/// cache (including allocator overhead) plus retained inputs/outputs and requires at least 112 MiB
/// cgroup headroom with the real service running. Tier 2 takes two units, leaving one for Tier 1 so
/// an audio decode cannot stop every cheap reference alignment. That other maximum is measured too:
/// CI runs one real partial-audio Tier-2 job (ffmpeg extraction followed by alass, on a two-hour 5.1
/// soundtrack) beside back-to-back Tier-1 jobs, the service and the same 112 MiB, in a fresh 512 MiB
/// cgroup, against the same 400 MiB ceiling. The two units conservatively preserve the measured
/// headroom while the extraction or alignment subprocess is resident.
const SYNC_MEMORY_UNITS: usize = 3;
const TIER1_SLOTS: usize = 3;
const TIER2_SLOTS: usize = 1;
const TIER1_WEIGHT: u32 = 1;
const TIER2_WEIGHT: u32 = 2;

use crate::cache::Cache;
use crate::config::Config;
use crate::inflight::{InFlight, Progress};
use crate::logging::LogGate;
use crate::seal::Keyring;
use crate::sync::SyncTools;
use crate::userconfig::{self, Rejected, UserConfig};

static INSTALL_REVOKED: LogGate = LogGate::new();
static INSTALL_TOO_OLD: LogGate = LogGate::new();
static INSTALL_NO_IID: LogGate = LogGate::new();
static PLAINTEXT_REFUSED: LogGate = LogGate::new();

pub struct AppState {
    pub cfg: Config,
    /// Decrypts a sealed config path segment (den-scout/docs/SEALED-CONFIG.md). `None` = sealed URLs
    /// disabled (legacy plaintext still works); the current key's public half is served at `/config-key`.
    pub config_keyring: Option<Keyring>,
    /// The pooled HTTP client. `None` if TLS init failed at boot — health/manifest/configure still
    /// serve; the subtitle/translate routes 503 instead of the whole process refusing to boot.
    pub http: Option<reqwest::Client>,
    pub cache: Cache,
    /// One worker per cache key for the two expensive jobs (an LLM translation, a sync subprocess).
    /// The cache only collapses work that has already finished; this collapses work in progress.
    pub inflight: Arc<InFlight>,
    /// How far a running translation has got, for the `.status` endpoint.
    pub progress: Progress,
    /// Tier binaries allowed to run at once. Partial-audio extraction and alignment together get up
    /// to 90 seconds across progressive probes; the runtime has one thread and the container is a
    /// homelab box. The
    /// single-flight map collapses duplicates of the SAME alignment, but distinct ones — twenty
    /// picker URLs, or a client naming twenty different `?ref=` values — are distinct keys and would
    /// all spawn.
    pub sync_admission: SyncAdmission,
    pub sync: SyncTools,
    /// Consecutive OpenSubtitles search failures — surfaced as `degraded` on /health (ADDON-02).
    pub os_fails: AtomicU32,
    /// What OpenSubtitles has said about each credential's rate limit and download quota, so one
    /// refusal holds every title rather than only the one that heard it.
    pub os_limits: crate::opensubtitles::Limits,
    /// The same for translation providers, per provider and key, across batches and runs.
    pub llm_pauses: crate::ratelimit::Pauses,
}

impl AppState {
    pub fn new(cfg: Config) -> Arc<AppState> {
        // Bounded so an upstream (OpenSubtitles / LLM) that never responds can't pin a request task
        // forever. The connect bound is tight; the overall bound is generous because a translation
        // batch on a slow model is legitimately slow (per-request LLM calls override it upward).
        let http = match reqwest::Client::builder()
            .user_agent(concat!("den-subtitles/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(60))
            .build()
        {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("warning: HTTP client init failed ({e}) — subtitle/translate routes will 503");
                None
            }
        };
        let sync = SyncTools {
            alass: cfg.alass.clone(),
            ffmpeg: cfg.ffmpeg.clone(),
            work_dir: cfg.cache_dir.join("sync"),
        };
        // Disk tier under CACHE_DIR/store so a restart/redeploy doesn't cold-start the cache.
        let cache = Cache::new(cfg.cache_max_bytes as usize, Some(cfg.cache_dir.join("store")));
        // A malformed key disables sealed URLs (legacy plaintext keeps working) rather than crashing.
        let config_keyring = match Keyring::from_env(&cfg.config_key, &cfg.config_keys_prev) {
            Ok(kr) => kr,
            Err(e) => {
                eprintln!("warning: CONFIG_KEY invalid ({e}) — sealed configs disabled");
                None
            }
        };
        Arc::new(AppState {
            cfg,
            config_keyring,
            http,
            cache,
            inflight: Arc::new(InFlight::default()),
            progress: Progress::default(),
            sync_admission: SyncAdmission::new(),
            sync,
            os_fails: AtomicU32::new(0),
            os_limits: Default::default(),
            llm_pauses: Default::default(),
        })
    }

    /// The install config in `blob`, if it decodes and its install is still admitted. Every route
    /// that reads a config comes through here, so a revocation reaches every URL that embeds the
    /// config, including the `/subtitle` and `/translate` URLs handed out before it. A refused
    /// install gets the same answer as an undecodable segment; only the log line tells them apart,
    /// once a minute per reason, because a revoked install keeps polling.
    pub fn decode_config(&self, blob: &str) -> Option<UserConfig> {
        let why = match userconfig::decode_checked(self.config_keyring.as_ref(), &self.cfg.revocation, blob) {
            Ok(cfg) => return Some(cfg),
            Err(why) => why,
        };
        let gate = match why {
            Rejected::Undecodable => return None,
            Rejected::Plaintext => &PLAINTEXT_REFUSED,
            Rejected::Revoked { .. } => &INSTALL_REVOKED,
            Rejected::EpochTooOld { .. } => &INSTALL_TOO_OLD,
            Rejected::NoInstallId => &INSTALL_NO_IID,
        };
        if gate.allow() {
            eprintln!("bad_config: {why}");
        }
        None
    }

    /// Record a successful OpenSubtitles search. Logs, and returns true, only when it recovers
    /// /health: that is the state change, and the successes around it are not news.
    pub fn search_succeeded(&self) -> bool {
        let recovered = self.os_fails.swap(0, Ordering::Relaxed) >= crate::HEALTH_FAIL_THRESHOLD;
        if recovered {
            eprintln!("opensubtitles: searching again — /health ok");
        }
        recovered
    }

    /// Record a failed OpenSubtitles search. Logs, and returns true, only for the failure that turns
    /// /health degraded; the ones after it are the same state.
    pub fn search_failed(&self) -> bool {
        let fails = self.os_fails.fetch_add(1, Ordering::Relaxed) + 1;
        let flipped = fails == crate::HEALTH_FAIL_THRESHOLD;
        if flipped {
            eprintln!(
                "opensubtitles: {fails} searches failed in a row — /health degraded (upstream_unavailable)"
            );
        }
        flipped
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SyncTier {
    Tier1,
    Tier2,
}

#[derive(Clone, Copy, Debug)]
pub enum SyncOutcome {
    Completed,
    TimedOut,
    Failed,
}

#[derive(Default)]
struct TierStats {
    running: AtomicUsize,
    queued: AtomicU64,
    queue_ns: AtomicU64,
    completed: AtomicU64,
    timed_out: AtomicU64,
    failed: AtomicU64,
    cancelled: AtomicU64,
}

#[derive(Clone, Copy)]
pub struct SyncStats {
    pub running: usize,
    pub queued: u64,
    pub queue_ns: u64,
    pub completed: u64,
    pub timed_out: u64,
    pub failed: u64,
    pub cancelled: u64,
}

impl TierStats {
    fn snapshot(&self) -> SyncStats {
        SyncStats {
            running: self.running.load(Ordering::Relaxed),
            queued: self.queued.load(Ordering::Relaxed),
            queue_ns: self.queue_ns.load(Ordering::Relaxed),
            completed: self.completed.load(Ordering::Relaxed),
            timed_out: self.timed_out.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
            cancelled: self.cancelled.load(Ordering::Relaxed),
        }
    }
}

/// Independent FIFO tier queues backed by one weighted memory budget. A Tier-2 job cannot consume
/// the final unit, and therefore cannot starve Tier 1; FIFO semaphores prevent either tier from being
/// overtaken indefinitely by later arrivals of its own kind.
pub struct SyncAdmission {
    memory: Semaphore,
    // Taken before a request downloads or retains either subtitle body. These are deliberately
    // separate: a queue of slow Tier-2 preparations must not occupy every place a cheap Tier-1
    // request can use. Requests outside these gates retain only their small request metadata.
    tier1_preparation: Semaphore,
    tier2_preparation: Semaphore,
    tier1_stats: TierStats,
    tier2_stats: TierStats,
}

impl SyncAdmission {
    fn new() -> Self {
        Self {
            memory: Semaphore::new(SYNC_MEMORY_UNITS),
            tier1_preparation: Semaphore::new(TIER1_SLOTS),
            tier2_preparation: Semaphore::new(TIER2_SLOTS),
            tier1_stats: TierStats::default(),
            tier2_stats: TierStats::default(),
        }
    }

    fn parts(&self, tier: SyncTier) -> (&Semaphore, &TierStats, u32) {
        match tier {
            SyncTier::Tier1 => (&self.tier1_preparation, &self.tier1_stats, TIER1_WEIGHT),
            SyncTier::Tier2 => (&self.tier2_preparation, &self.tier2_stats, TIER2_WEIGHT),
        }
    }

    /// Reserve a bounded body-retention place before fetching either sync input. The weighted
    /// subprocess budget is intentionally acquired later, after network preparation, but at most
    /// three Tier-1 jobs and one Tier-2 job can reach that point with bodies resident.
    pub async fn reserve(&self, tier: SyncTier) -> SyncReservation<'_> {
        let queued_at = Instant::now();
        let (preparation, stats, weight) = self.parts(tier);
        let preparation_permit = preparation.acquire().await.expect("sync tier semaphore is never closed");
        let elapsed = queued_at.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        stats.queued.fetch_add(1, Ordering::Relaxed);
        stats.queue_ns.fetch_add(elapsed, Ordering::Relaxed);
        SyncReservation { stats, weight, _preparation_permit: preparation_permit }
    }

    /// Admit a prepared job to the weighted subprocess budget. The reservation remains held until
    /// the subprocess finishes, so another request cannot start retaining bodies behind it early.
    pub async fn acquire<'a>(&'a self, reservation: SyncReservation<'a>) -> SyncPermit<'a> {
        let queued_at = Instant::now();
        let memory_permit = self
            .memory
            .acquire_many(reservation.weight)
            .await
            .expect("sync memory semaphore is never closed");
        let elapsed = queued_at.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        reservation.stats.queue_ns.fetch_add(elapsed, Ordering::Relaxed);
        let stats = reservation.stats;
        stats.running.fetch_add(1, Ordering::Relaxed);
        SyncPermit { stats, _reservation: reservation, _memory_permit: memory_permit, finished: false }
    }

    pub fn stats(&self, tier: SyncTier) -> SyncStats {
        self.parts(tier).1.snapshot()
    }
}

pub struct SyncReservation<'a> {
    stats: &'a TierStats,
    weight: u32,
    _preparation_permit: SemaphorePermit<'a>,
}

pub struct SyncPermit<'a> {
    stats: &'a TierStats,
    _reservation: SyncReservation<'a>,
    _memory_permit: SemaphorePermit<'a>,
    finished: bool,
}

impl SyncPermit<'_> {
    pub fn finish(&mut self, outcome: SyncOutcome) {
        if self.finished {
            return;
        }
        match outcome {
            SyncOutcome::Completed => &self.stats.completed,
            SyncOutcome::TimedOut => &self.stats.timed_out,
            SyncOutcome::Failed => &self.stats.failed,
        }
        .fetch_add(1, Ordering::Relaxed);
        self.stats.running.fetch_sub(1, Ordering::Relaxed);
        self.finished = true;
    }
}

impl Drop for SyncPermit<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.stats.cancelled.fetch_add(1, Ordering::Relaxed);
            self.stats.running.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod sync_admission_tests {
    use super::*;

    #[tokio::test]
    async fn tier2_always_leaves_capacity_for_tier1() {
        let admission = SyncAdmission::new();
        let tier2_reservation = admission.reserve(SyncTier::Tier2).await;
        let mut tier2 = admission.acquire(tier2_reservation).await;
        let tier1_reservation = admission.reserve(SyncTier::Tier1).await;
        let mut tier1 = admission.acquire(tier1_reservation).await;
        let second = admission.reserve(SyncTier::Tier1).await;
        assert!(tokio::time::timeout(Duration::from_millis(10), admission.acquire(second)).await.is_err());
        tier1.finish(SyncOutcome::Completed);
        tier2.finish(SyncOutcome::Completed);
        let one = admission.stats(SyncTier::Tier1);
        let two = admission.stats(SyncTier::Tier2);
        assert_eq!((one.running, one.completed), (0, 1));
        assert_eq!((two.running, two.completed), (0, 1));
    }

    #[tokio::test]
    async fn queued_tier2_is_not_starved_by_later_tier1_work() {
        let admission = Arc::new(SyncAdmission::new());
        let mut held = Vec::new();
        for _ in 0..TIER1_SLOTS {
            let reservation = admission.reserve(SyncTier::Tier1).await;
            held.push(admission.acquire(reservation).await);
        }

        let (acquired_tx, acquired_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let waiting_tier2 = {
            let admission = admission.clone();
            tokio::spawn(async move {
                let reservation = admission.reserve(SyncTier::Tier2).await;
                let mut permit = admission.acquire(reservation).await;
                let _ = acquired_tx.send(());
                let _ = release_rx.await;
                permit.finish(SyncOutcome::Completed);
            })
        };
        tokio::task::yield_now().await;
        let later_tier1 = {
            let admission = admission.clone();
            tokio::spawn(async move {
                let reservation = admission.reserve(SyncTier::Tier1).await;
                let mut permit = admission.acquire(reservation).await;
                permit.finish(SyncOutcome::Completed);
            })
        };
        tokio::task::yield_now().await;

        // Releasing two units satisfies the older weighted Tier-2 waiter. The later one-unit Tier-1
        // request must not jump the FIFO memory queue while the final original Tier-1 job is running.
        for mut permit in held.drain(..TIER2_WEIGHT as usize) {
            permit.finish(SyncOutcome::Completed);
        }
        tokio::time::timeout(Duration::from_secs(1), acquired_rx).await.unwrap().unwrap();
        assert!(!later_tier1.is_finished());

        let _ = release_tx.send(());
        tokio::time::timeout(Duration::from_secs(1), waiting_tier2).await.unwrap().unwrap();
        tokio::time::timeout(Duration::from_secs(1), later_tier1).await.unwrap().unwrap();
        held.pop().unwrap().finish(SyncOutcome::Completed);
    }

    #[tokio::test]
    async fn dropping_a_running_job_records_cancellation_and_releases_memory() {
        let admission = SyncAdmission::new();
        let reservation = admission.reserve(SyncTier::Tier1).await;
        drop(admission.acquire(reservation).await);
        let stats = admission.stats(SyncTier::Tier1);
        assert_eq!((stats.running, stats.cancelled), (0, 1));
        let reservation = admission.reserve(SyncTier::Tier1).await;
        let mut replacement = admission.acquire(reservation).await;
        replacement.finish(SyncOutcome::TimedOut);
        assert_eq!(admission.stats(SyncTier::Tier1).timed_out, 1);
    }

    #[tokio::test]
    async fn queued_requests_do_not_pass_the_body_retention_bound() {
        let admission = Arc::new(SyncAdmission::new());
        let mut held = Vec::new();
        for _ in 0..TIER1_SLOTS {
            held.push(admission.reserve(SyncTier::Tier1).await);
        }
        let (acquired_tx, acquired_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let waiting = {
            let admission = admission.clone();
            tokio::spawn(async move {
                let reservation = admission.reserve(SyncTier::Tier1).await;
                let _ = acquired_tx.send(());
                let _ = release_rx.await;
                drop(reservation);
            })
        };
        tokio::task::yield_now().await;
        assert!(!waiting.is_finished(), "a fourth Tier-1 request passed the three-body bound");
        drop(held.pop());
        tokio::time::timeout(Duration::from_secs(1), acquired_rx).await.unwrap().unwrap();
        let _ = release_tx.send(());
        waiting.await.unwrap();
    }
}
