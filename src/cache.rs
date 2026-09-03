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
use std::sync::atomic::{AtomicU64, Ordering};
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
        let cache = Cache {
            inner: Mutex::new(Inner { map: HashMap::new(), bytes: 0, tick: 0 }),
            max_bytes,
            dir,
        };
        cache.sweep();
        cache
    }

    /// Bound the disk tier. `max_bytes` caps memory only, and nothing else reclaims disk: an entry
    /// is deleted lazily by `disk_get`, which needs someone to ask for that exact key again — and
    /// search keys carry a per-file hash, so once a title is watched its entry is never re-read.
    /// On a persistent volume that grows forever.
    ///
    /// Expired first, then oldest-written until under budget. Cheap enough to run on a timer: one
    /// readdir plus a stat per file, over a directory of small text entries.
    pub fn sweep(&self) {
        let Some(dir) = self.dir.as_ref() else { return };
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        let now = unix_now();
        let mut live: Vec<(u64, u64, PathBuf)> = Vec::new(); // (mtime, size, path)
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(meta) = entry.metadata() else { continue };
            if !meta.is_file() {
                continue;
            }
            // Read only the header line: the value can be megabytes and is not needed to judge it.
            let expired = match std::fs::read_to_string(&path) {
                Ok(raw) => Self::parse_entry(&raw).is_none(),
                Err(_) => true,
            };
            if expired {
                let _ = std::fs::remove_file(&path);
                continue;
            }
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map_or(now, |d| d.as_secs());
            live.push((mtime, meta.len(), path));
        }
        let mut total: u64 = live.iter().map(|(_, size, _)| size).sum();
        let budget = self.max_bytes as u64;
        if total <= budget {
            return;
        }
        live.sort_by_key(|(mtime, _, _)| *mtime);
        for (_, size, path) in live {
            if total <= budget {
                break;
            }
            if std::fs::remove_file(&path).is_ok() {
                total -= size;
            }
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

    /// Returns (value, remaining ttl) if a fresh, intact entry exists on disk; removes it otherwise.
    fn disk_get(&self, key: &str) -> Option<(String, Duration)> {
        let path = self.disk_path(key)?;
        let raw = std::fs::read_to_string(&path).ok()?;
        match Self::parse_entry(&raw) {
            Some(fresh) => Some(fresh),
            // Expired, torn, or written by an older format. All three are unusable, and all three
            // are dropped — an unreadable file left in place is never reclaimed by anything.
            None => {
                let _ = std::fs::remove_file(&path);
                None
            }
        }
    }

    /// Format: "<expiry-unix-secs> <value-bytes>\n<value>". The length is what makes a torn file
    /// detectable: without it any prefix carrying a newline parses as a complete entry, so a
    /// half-written file is hoisted into memory and served for the rest of its TTL.
    fn parse_entry(raw: &str) -> Option<(String, Duration)> {
        let (head, value) = raw.split_once('\n')?;
        let (expiry, len) = head.split_once(' ')?;
        let expiry: u64 = expiry.parse().ok()?;
        if value.len() != len.parse::<usize>().ok()? {
            return None;
        }
        let now = unix_now();
        (expiry > now).then(|| (value.to_string(), Duration::from_secs(expiry - now)))
    }

    fn disk_put(&self, key: &str, value: &str, ttl: Duration) {
        let Some(path) = self.disk_path(key) else { return };
        let expiry = unix_now().saturating_add(ttl.as_secs());
        let body = format!("{expiry} {}\n{value}", value.len());
        // Write then rename, because a truncated file is indistinguishable from a complete one on
        // read: any prefix carrying a newline parses as a valid entry, and would then be hoisted into
        // memory and served for the whole TTL. ENOSPC, a kill mid-write, and two writers on one key
        // all produce that. The temp name is unique so concurrent writers don't share one.
        let tmp = path.with_extension(format!("t{}", next_temp_id()));
        if std::fs::write(&tmp, body).is_ok() && std::fs::rename(&tmp, &path).is_ok() {
            return;
        }
        let _ = std::fs::remove_file(&tmp);
    }
}

fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn next_temp_id() -> u64 {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    SEQ.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("den-subs-cache-{name}-{}", next_temp_id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    const HOUR: Duration = Duration::from_secs(3600);

    #[test]
    fn a_value_survives_a_restart_via_the_disk_tier() {
        let dir = tmpdir("restart");
        let c = Cache::new(1 << 20, Some(dir.clone()));
        c.put("k".into(), "v".into(), HOUR);
        // A fresh Cache over the same dir is what a redeploy looks like: memory empty, disk warm.
        let c2 = Cache::new(1 << 20, Some(dir));
        assert_eq!(c2.get("k"), Some("v".into()));
    }

    #[test]
    fn an_expired_disk_entry_is_not_served() {
        let dir = tmpdir("expired");
        let c = Cache::new(1 << 20, Some(dir.clone()));
        c.put("k".into(), "v".into(), Duration::from_secs(0));
        assert_eq!(Cache::new(1 << 20, Some(dir)).get("k"), None);
    }

    /// The whole reason writes go through a rename: a half-written file parses fine (any prefix with
    /// a newline splits into expiry + value), so a torn write is served as a complete value for the
    /// full TTL. Nothing may be left behind that reads as a valid entry.
    #[test]
    fn a_torn_write_never_becomes_a_readable_entry() {
        let dir = tmpdir("torn");
        let c = Cache::new(1 << 20, Some(dir.clone()));
        c.put("k".into(), "the whole subtitle".into(), HOUR);
        let path = c.disk_path("k").unwrap();

        // Truncate the file the way ENOSPC or a kill mid-write would. It still parses — that is the
        // hazard — so the recorded length is what has to reject it.
        let raw = std::fs::read_to_string(&path).unwrap();
        std::fs::write(&path, &raw[..raw.len() - 6]).unwrap();
        assert_eq!(Cache::new(1 << 20, Some(dir.clone())).get("k"), None, "served a truncated entry");
        assert!(!path.exists(), "a torn entry should be dropped, not left to be re-read");

        // A completed put must leave exactly one file: the entry. A temp left behind would be
        // unreadable garbage that nothing ever reclaims. (The torn file was already dropped above.)
        c.put("k".into(), "v2".into(), HOUR);
        let files: Vec<_> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).collect();
        assert_eq!(files.len(), 1, "a put left scratch behind: {:?}", files.iter().map(|f| f.file_name()).collect::<Vec<_>>());
        assert_eq!(Cache::new(1 << 20, Some(dir)).get("k"), Some("v2".into()));
    }

    /// An entry this build cannot read is deleted, not left behind. The format changed to carry a
    /// length, so every pre-upgrade file is unreadable — on a persistent volume with no disk budget,
    /// leaving them is a leak that nothing ever reclaims.
    #[test]
    fn an_unreadable_entry_is_reclaimed() {
        let dir = tmpdir("stale");
        let c = Cache::new(1 << 20, Some(dir.clone()));
        c.put("k".into(), "v".into(), HOUR);
        let path = c.disk_path("k").unwrap();
        for raw in ["999999999999\nold format, no length", "", "garbage"] {
            std::fs::write(&path, raw).unwrap();
            assert_eq!(Cache::new(1 << 20, Some(dir.clone())).get("k"), None, "served {raw:?}");
            assert!(!path.exists(), "left an unreadable entry on disk: {raw:?}");
        }
    }

    /// The disk tier has to be bounded too. Memory's cap never touched it, and lazy expiry needs
    /// someone to re-request the exact key — which search entries, keyed by a per-file hash, never
    /// get. Expired entries go first; if that isn't enough, the oldest do.
    #[test]
    fn the_sweep_bounds_the_disk_tier() {
        let dir = tmpdir("sweep");
        // Budget fits roughly two entries, so a third must push the oldest out.
        let c = Cache::new(120, Some(dir.clone()));
        c.put("expired".into(), "x".repeat(40), Duration::from_secs(0));
        c.put("old".into(), "x".repeat(40), HOUR);
        std::thread::sleep(std::time::Duration::from_millis(1100)); // mtime has 1s resolution
        c.put("new".into(), "x".repeat(40), HOUR);

        c.sweep();
        let fresh = Cache::new(120, Some(dir.clone()));
        assert_eq!(fresh.get("expired"), None, "an expired entry must not survive a sweep");
        assert_eq!(fresh.get("new"), Some("x".repeat(40)), "the newest entry must survive");
        let bytes: u64 = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.metadata().unwrap().len()).sum();
        assert!(bytes <= 120, "the sweep left {bytes} bytes on disk, over the 120 budget");
    }

    /// A sweep must never remove a live entry while it is still under budget — evicting eagerly
    /// would re-download subtitles the cache exists to keep.
    #[test]
    fn the_sweep_keeps_everything_that_fits() {
        let dir = tmpdir("sweep-fits");
        let c = Cache::new(1 << 20, Some(dir.clone()));
        c.put("a".into(), "aaa".into(), HOUR);
        c.put("b".into(), "bbb".into(), HOUR);
        c.sweep();
        let fresh = Cache::new(1 << 20, Some(dir));
        assert_eq!(fresh.get("a"), Some("aaa".into()));
        assert_eq!(fresh.get("b"), Some("bbb".into()));
    }

    #[test]
    fn memory_evicts_least_recently_used_under_the_byte_cap() {
        let c = Cache::new(24, None);
        c.put("a".into(), "0123456789".into(), HOUR);
        c.put("b".into(), "0123456789".into(), HOUR);
        assert_eq!(c.get("a"), Some("0123456789".into())); // touch a, so b is the victim
        c.put("c".into(), "0123456789".into(), HOUR);
        assert_eq!(c.get("b"), None, "the least recently used entry should have gone");
        assert_eq!(c.get("a"), Some("0123456789".into()));
    }

    #[test]
    fn an_unwritable_dir_falls_back_to_memory_only() {
        let dir = tmpdir("file-not-dir");
        std::fs::write(&dir, "not a directory").unwrap();
        let c = Cache::new(1 << 20, Some(dir));
        c.put("k".into(), "v".into(), HOUR);
        assert_eq!(c.get("k"), Some("v".into()), "memory must still serve");
    }
}
