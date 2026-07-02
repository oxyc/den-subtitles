//! Two-tier artifact cache: a byte-bounded in-memory LRU backed by an optional disk store, so a
//! container restart/redeploy doesn't cold-start (SRTs are tiny and worth persisting). Holds the
//! finished artifacts — search results, downloaded SRTs, translated SRTs — keyed so each is fetched
//! once. Thread-safe.
//!
//! Memory is the hot tier (bounded, LRU). Disk is the durable tier: `put` writes through to a file,
//! and a memory miss lazily reads it back (repopulating memory). Disk is best-effort — a write
//! failure or unwritable dir just disables persistence and logs; memory still serves. Expiry on disk
//! is wall-clock (survives restarts); in memory it's monotonic.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use base64::Engine;

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
    /// Disk tier directory, or `None` when persistence is off (memory-only).
    dir: Option<PathBuf>,
}

struct Inner {
    map: HashMap<String, Entry>,
    bytes: usize,
    tick: u64,
}

impl Cache {
    /// `dir = Some` enables the disk tier (created if missing; falls back to memory-only on failure).
    pub fn new(max_bytes: usize, dir: Option<PathBuf>) -> Cache {
        let dir = dir.and_then(|d| match std::fs::create_dir_all(&d) {
            Ok(()) => Some(d),
            Err(e) => {
                eprintln!("warning: cache store {} not writable ({e}) — disk persistence off", d.display());
                None
            }
        });
        Cache {
            inner: Mutex::new(Inner { map: HashMap::new(), bytes: 0, tick: 0 }),
            max_bytes,
            dir,
        }
    }

    pub fn get(&self, key: &str) -> Option<String> {
        if let Some(v) = self.mem_get(key) {
            return Some(v);
        }
        // Memory miss: fall back to disk, then repopulate memory for the next hit.
        if let Some((value, remaining)) = self.disk_get(key) {
            self.mem_put(key.to_string(), value.clone(), remaining);
            return Some(value);
        }
        None
    }

    pub fn put(&self, key: String, value: String, ttl: Duration) {
        self.disk_put(&key, &value, ttl);
        self.mem_put(key, value, ttl);
    }

    // ---- memory tier -------------------------------------------------------

    fn mem_get(&self, key: &str) -> Option<String> {
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

    fn mem_put(&self, key: String, value: String, ttl: Duration) {
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

    // ---- disk tier (best-effort) ------------------------------------------

    /// Filename for a key: url-safe base64 so any key (colons, slashes) is a valid single filename.
    fn disk_path(&self, key: &str) -> Option<PathBuf> {
        let dir = self.dir.as_ref()?;
        let name = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(key.as_bytes());
        Some(dir.join(name))
    }

    /// Returns (value, remaining ttl) if a fresh entry exists on disk; removes it if expired.
    fn disk_get(&self, key: &str) -> Option<(String, Duration)> {
        let path = self.disk_path(key)?;
        let raw = std::fs::read_to_string(&path).ok()?;
        // Format: "<expiry-unix-secs>\n<value>".
        let (head, value) = raw.split_once('\n')?;
        let expiry: u64 = head.parse().ok()?;
        let now = unix_now();
        if expiry <= now {
            let _ = std::fs::remove_file(&path);
            return None;
        }
        Some((value.to_string(), Duration::from_secs(expiry - now)))
    }

    fn disk_put(&self, key: &str, value: &str, ttl: Duration) {
        let Some(path) = self.disk_path(key) else { return };
        let expiry = unix_now().saturating_add(ttl.as_secs());
        // Best-effort: a write failure just means this entry isn't persisted (memory still holds it).
        let body = format!("{expiry}\n{value}");
        let _ = std::fs::write(&path, body);
    }
}

fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}
