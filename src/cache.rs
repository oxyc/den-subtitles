//! In-memory TTL + LRU cache bounded by BYTES (a count cap lets a big value blow the heap and OOM a
//! homelab container). Holds the finished artifacts — translated and synced SRTs — keyed so each
//! film is paid for once: `translate:<hash|imdb>:<lang>:<provider>:<model>`. Thread-safe.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

struct Entry {
    value: String,
    size: usize,
    expires: Instant,
    /// Monotonic touch counter for LRU (cheaper than reordering a list under a mutex).
    used: u64,
}

pub struct Cache {
    inner: Mutex<Inner>,
    max_bytes: usize,
}

struct Inner {
    map: HashMap<String, Entry>,
    bytes: usize,
    tick: u64,
}

impl Cache {
    pub fn new(max_bytes: usize) -> Cache {
        Cache {
            inner: Mutex::new(Inner { map: HashMap::new(), bytes: 0, tick: 0 }),
            max_bytes,
        }
    }

    pub fn get(&self, key: &str) -> Option<String> {
        let mut g = self.inner.lock().unwrap();
        g.tick += 1;
        let tick = g.tick;
        match g.map.get_mut(key) {
            Some(e) if e.expires > Instant::now() => {
                e.used = tick;
                Some(e.value.clone())
            }
            Some(_) => {
                let size = g.map.remove(key).map(|e| e.size).unwrap_or(0);
                g.bytes -= size;
                None
            }
            None => None,
        }
    }

    pub fn put(&self, key: String, value: String, ttl: Duration) {
        let mut g = self.inner.lock().unwrap();
        g.tick += 1;
        let size = key.len() + value.len();
        if let Some(old) = g.map.remove(&key) {
            g.bytes -= old.size;
        }
        let entry = Entry { value, size, expires: Instant::now() + ttl, used: g.tick };
        g.bytes += size;
        g.map.insert(key, entry);
        // Evict least-recently-used until under budget.
        while g.bytes > self.max_bytes {
            let Some(victim) = g.map.iter().min_by_key(|(_, e)| e.used).map(|(k, _)| k.clone()) else {
                break;
            };
            if let Some(e) = g.map.remove(&victim) {
                g.bytes -= e.size;
            }
        }
    }
}
