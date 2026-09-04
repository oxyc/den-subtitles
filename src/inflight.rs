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
