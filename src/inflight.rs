//! Single-flight: one worker per cache key, and everyone else waits for it.
//!
//! The cache collapses repeated work into one lookup, but only once the first worker has FINISHED.
//! Until then every arrival is a miss and starts its own copy of the job — and the two jobs this
//! addon runs on a miss are the two expensive ones: a whole film's LLM translation, charged to the
//! viewer's own provider account, and an `alass`/`ffsubsync` subprocess on a runtime with one thread.
//!
//! The app's own `.json`-then-`.srt` flow is sequential, so the duplicate is not the everyday case.
//! Two devices on the same title, or a second tap during a run that legitimately takes minutes, is —
//! and that window is exactly as long as the work is expensive.
//!
//! A caller MUST re-check the cache after acquiring: the whole point is that the wait may have been
//! spent watching someone else produce the answer.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

#[derive(Default)]
pub struct InFlight {
    /// Sync mutex around the map — held for a lookup only, never across an await. The per-key async
    /// mutex inside it is what a waiter actually blocks on.
    keys: Mutex<HashMap<String, Arc<AsyncMutex<()>>>>,
}

/// The right to produce one key. Releases it on drop, and takes the key back out of the map once
/// nobody is waiting for it.
pub struct Guard<'a> {
    owner: &'a InFlight,
    key: String,
    /// Struct fields drop AFTER `Drop::drop` runs, so this Arc is still counted by the cleanup below.
    _permit: OwnedMutexGuard<()>,
}

impl InFlight {
    /// Wait for the exclusive right to produce `key`.
    pub async fn acquire(&self, key: &str) -> Guard<'_> {
        let lock = {
            let mut keys = self.keys.lock().unwrap();
            // Sweep entries nobody holds any more. `Guard::drop` keeps an entry alive while a waiter
            // still needs it, and relies on that waiter to remove it later — but a waiter whose
            // request is CANCELLED never reaches `Guard::drop`, and leaves the entry behind with only
            // the map referencing it. Keys here are request-shaped and unbounded, so those orphans
            // accumulate for the life of the process. A count of one means map-only: nobody holds it,
            // nobody is waiting on it.
            keys.retain(|_, lock| Arc::strong_count(lock) > 1);
            keys.entry(key.to_string()).or_default().clone()
        };
        let permit = lock.lock_owned().await;
        Guard { owner: self, key: key.to_string(), _permit: permit }
    }

    #[cfg(test)]
    fn tracked(&self) -> usize {
        self.keys.lock().unwrap().len()
    }
}

impl Drop for Guard<'_> {
    fn drop(&mut self) {
        let mut keys = self.owner.keys.lock().unwrap();
        let Some(lock) = keys.get(&self.key) else { return };
        // The map's own copy, plus the one held inside our permit. Anything beyond those two is a
        // waiter that still needs this entry: dropping it would let that waiter and the next arrival
        // each build a fresh lock and run at the same time, which is the one thing this prevents.
        //
        // Checked under the map lock, and a newcomer needs that same lock to clone the Arc, so there
        // is no window between the count and the removal.
        if Arc::strong_count(lock) <= 2 {
            keys.remove(&self.key);
        }
    }
}

/// How far a long job has got, so a client can show a bar rather than a spinner.
///
/// Keyed by the job, not by the artifact: the key has to be computable from the request alone, or a
/// poll would cost a search of its own — and a poll happens every second while the expensive thing
/// is running.
///
/// Entries are live only while a job is; a missing one means "not running here", which is also the
/// honest answer after a restart or from a second instance.
#[derive(Default)]
pub struct Progress {
    jobs: Mutex<HashMap<String, (usize, usize)>>,
}

impl Progress {
    /// Start reporting for `key`. The returned guard clears the entry when it drops.
    ///
    /// A guard rather than a matching `clear` call, because the third way out of a translation is
    /// neither success nor failure: the client disconnects, hyper drops the handler future
    /// mid-await, and a hand-written `clear` after that await never runs. The entry would then
    /// outlive the process's interest in it — and since `handle_translate_status` checks progress
    /// before the failure marker, a stale entry does not merely go stale, it MASKS the failure and
    /// leaves a poller waiting on a run that is not happening.
    pub fn start(&self, key: &str, total: usize) -> Reporter<'_> {
        self.jobs.lock().unwrap().insert(key.to_string(), (0, total));
        Reporter { owner: self, key: key.to_string() }
    }

    pub fn get(&self, key: &str) -> Option<(usize, usize)> {
        self.jobs.lock().unwrap().get(key).copied()
    }

    #[cfg(test)]
    fn tracked(&self) -> usize {
        self.jobs.lock().unwrap().len()
    }
}

/// Reports progress for one job, and stops reporting however the job ends — returned, failed, or
/// dropped underneath us.
pub struct Reporter<'a> {
    owner: &'a Progress,
    key: String,
}

impl Reporter<'_> {
    pub fn set(&self, done: usize, total: usize) {
        self.owner.jobs.lock().unwrap().insert(self.key.clone(), (done, total));
    }
}

impl Drop for Reporter<'_> {
    fn drop(&mut self) {
        self.owner.jobs.lock().unwrap().remove(&self.key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// Two callers on one key do not run at the same time, and a third key is not held up by either.
    #[tokio::test]
    async fn one_key_runs_one_worker_at_a_time() {
        let flight = Arc::new(InFlight::default());
        let running = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));

        let mut tasks = Vec::new();
        for _ in 0..8 {
            let (flight, running, peak) = (flight.clone(), running.clone(), peak.clone());
            tasks.push(tokio::spawn(async move {
                let _guard = flight.acquire("same").await;
                let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                // Long enough that a missing lock would overlap.
                tokio::time::sleep(Duration::from_millis(5)).await;
                running.fetch_sub(1, Ordering::SeqCst);
            }));
        }
        for t in tasks {
            t.await.unwrap();
        }
        assert_eq!(peak.load(Ordering::SeqCst), 1, "two workers held the same key at once");
        // And the map does not grow a permanent entry per key ever seen.
        assert_eq!(flight.tracked(), 0, "the key outlived the last guard");
    }

    /// A reporter that is DROPPED rather than finished stops reporting. This is the case a matching
    /// `clear` call after the await cannot cover — the request is cancelled mid-run and that line
    /// never executes — and a stale entry does not merely go stale: `handle_translate_status` checks
    /// progress before the failure marker, so it MASKS the failure and leaves a client polling a run
    /// that is not happening.
    #[test]
    fn a_dropped_reporter_stops_reporting() {
        let progress = Progress::default();
        {
            let reporter = progress.start("job", 100);
            reporter.set(40, 100);
            assert_eq!(progress.get("job"), Some((40, 100)));
        }
        assert_eq!(progress.get("job"), None, "a cancelled run went on reporting itself as working");
        assert_eq!(progress.tracked(), 0);
    }

    /// A waiter that is CANCELLED never reaches `Guard::drop`, so it cannot remove the entry it was
    /// keeping alive. Nothing else was watching that entry, and the keys here are request-shaped and
    /// unbounded, so the orphans accumulate for the life of the process.
    #[tokio::test]
    async fn a_cancelled_waiter_does_not_orphan_its_key() {
        let flight = Arc::new(InFlight::default());
        let first = flight.acquire("k").await;

        let waiter = {
            let flight = flight.clone();
            tokio::spawn(async move {
                let _held = flight.acquire("k").await;
                tokio::time::sleep(Duration::from_secs(30)).await;
            })
        };
        // Let the waiter reach the lock and queue behind `first`.
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(flight.tracked(), 1);

        // `first` releases while the waiter is still queued, so the entry is deliberately kept for
        // it — and then the waiter is cancelled before it can ever take, or release, the key.
        drop(first);
        waiter.abort();
        let _ = waiter.await;

        // Nothing is left holding it, so the next acquire must not find it still there.
        let unrelated = flight.acquire("other").await;
        drop(unrelated);
        assert_eq!(flight.tracked(), 0, "a cancelled waiter left its key in the map forever");
    }

    /// Different keys are independent — a slow translation must not block an unrelated one.
    #[tokio::test]
    async fn different_keys_do_not_block_each_other() {
        let flight = InFlight::default();
        let held = flight.acquire("a").await;
        // Would hang if this waited on "a".
        let other = tokio::time::timeout(Duration::from_secs(5), flight.acquire("b")).await;
        assert!(other.is_ok(), "an unrelated key waited on a held one");
        drop(other);
        drop(held);
        assert_eq!(flight.tracked(), 0);
    }

    /// The entry survives exactly as long as someone needs it: released while a waiter is queued, it
    /// must NOT be removed, or the waiter and the next arrival would each get a fresh lock.
    #[tokio::test]
    async fn a_waiting_caller_keeps_the_key_alive() {
        let flight = Arc::new(InFlight::default());
        let first = flight.acquire("k").await;

        let waiter = {
            let flight = flight.clone();
            tokio::spawn(async move {
                let _g = flight.acquire("k").await;
                tokio::time::sleep(Duration::from_millis(5)).await;
            })
        };
        // Let the waiter reach the lock.
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(flight.tracked(), 1);
        drop(first);
        // Still tracked: the waiter holds it now.
        assert_eq!(flight.tracked(), 1, "the key was dropped out from under a waiter");

        waiter.await.unwrap();
        assert_eq!(flight.tracked(), 0, "the key outlived the last guard");
    }
}
