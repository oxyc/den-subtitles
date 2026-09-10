//! Shared application state: one pooled HTTP client, the artifact cache, and the sync tools, wired
//! from `Config` and cloned (behind `Arc`) into every connection.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Semaphore;

/// Tier binaries (`alass`/`ffsubsync`, each its own process, each up to 90s) allowed to run at once.
/// Two keeps a burst of requests from turning into a burst of ffmpeg decodes on a homelab box.
const MAX_CONCURRENT_SYNCS: usize = 2;

use crate::cache::Cache;
use crate::config::Config;
use crate::inflight::{InFlight, Progress};
use crate::seal::Keyring;
use crate::sync::SyncTools;

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
    pub inflight: InFlight,
    /// How far a running translation has got, for the `.status` endpoint.
    pub progress: Progress,
    /// Tier binaries allowed to run at once. `alass` decodes audio through ffmpeg and gets up to 90
    /// seconds; the runtime has one thread and the container is a homelab box. The single-flight map
    /// collapses duplicates of the SAME alignment, but distinct ones — twenty picker URLs, or a
    /// client naming twenty different `?ref=` values — are distinct keys and would all spawn.
    pub sync_slots: Semaphore,
    pub sync: SyncTools,
    /// Consecutive OpenSubtitles search failures — surfaced as `degraded` on /health (ADDON-02).
    pub os_fails: AtomicU32,
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
            ffsubsync: cfg.ffsubsync.clone(),
            alass: cfg.alass.clone(),
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
            inflight: InFlight::default(),
            progress: Progress::default(),
            sync_slots: Semaphore::new(MAX_CONCURRENT_SYNCS),
            sync,
            os_fails: AtomicU32::new(0),
        })
    }

    /// Tier binaries running now: the slots taken out of `sync_slots`. Callers still waiting for a
    /// slot are not counted — they have not spawned anything.
    pub fn syncs_running(&self) -> usize {
        MAX_CONCURRENT_SYNCS - self.sync_slots.available_permits()
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
