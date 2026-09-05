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
    /// Expired first, then oldest-written until under budget.
    pub fn sweep(&self) {
        /// Longer than any write takes, far shorter than the hourly sweep interval.
        const TEMP_GRACE: Duration = Duration::from_secs(60);

        // The memory tier first, and unconditionally — it has to be reclaimed even when there is no
        // disk tier to walk. An expired memory entry is otherwise dropped only when someone reads
        // that exact key again, or when LRU pressure evicts it. Entries written and never re-read
        // are the norm now that a film leaves ~150 batch entries behind, and every one of them holds
        // its bytes against the budget until something else needs the room.
        {
            let mut g = self.inner.lock().unwrap();
            let now = Instant::now();
            let dead: Vec<String> = g
                .map
                .iter()
                .filter(|(_, e)| e.expires <= now)
                .map(|(k, _)| k.clone())
                .collect();
            for key in dead {
                if let Some(e) = g.map.remove(&key) {
                    g.bytes -= e.size;
                }
            }
        }

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
            // A leftover temp from a kill between write and rename. Nothing addresses it, and
            // counting it as live lets it push a real entry out during eviction.
            //
            // Only once it is old enough to be abandoned: `disk_put` writes to exactly this name
            // before renaming it into place, so unlinking one on sight can delete an in-flight
            // write and make a healthy disk report itself degraded.
            if path.extension().is_some_and(|e| e.to_string_lossy().starts_with('t')) {
                match age(&meta) {
                    // Abandoned by a kill between write and rename.
                    Some(age) if age > TEMP_GRACE => {
                        let _ = std::fs::remove_file(&path);
                        continue;
                    }
                    // Young enough to still be in flight. Leave it, and don't budget it either —
                    // it is about to become a real entry or be cleaned up by its own writer.
                    Some(_) => continue,
                    // Undatable: `elapsed()` errors when the mtime reads as being in the future —
                    // a backward clock step, or a filesystem that doesn't report mtime. Deleting on
                    // that signal would kill in-flight writes on sight, which is what the grace
                    // period exists to stop; keeping it forever is a leak nothing else reclaims.
                    //
                    // So it goes into the budget as the OLDEST thing there: never deleted on sight,
                    // but first to go once the store is over budget. Stamping it `now` instead made
                    // it the last — so a sweep under pressure evicted live, in-TTL subtitles, each
                    // costing a re-download against the viewer's quota, to keep a scratch file of
                    // unknown provenance. It skips the expiry check below, which would read a
                    // half-written file as garbage and delete it outright.
                    None => {
                        live.push((0, meta.len(), path));
                        continue;
                    }
                }
            }
            match Self::read_expiry(&path) {
                // Unreadable is not expired. Treating it as such deleted the whole store the first
                // time a redeploy shifted the volume's uid — `disk_get` treats the same error as a
                // benign miss and leaves the file be.
                Err(_) => continue,
                Ok(None) | Ok(Some(0)) => {
                    let _ = std::fs::remove_file(&path);
                    continue;
                }
                Ok(Some(expiry)) if expiry <= now => {
                    let _ = std::fs::remove_file(&path);
                    continue;
                }
                Ok(Some(_)) => {}
            }
            let mtime = modified_secs(&meta).unwrap_or(now);
            live.push((mtime, meta.len(), path));
        }
        let mut total: u64 = live.iter().map(|(_, size, _)| size).sum();
        let budget = self.max_bytes as u64;
        if total <= budget {
            return;
        }
        // Cheapest-to-rebuild first, and only then oldest-first.
        //
        // Age alone is the wrong question here. `disk_get` never touches mtime, so the disk tier
        // evicts in WRITE order, not use order: a translation served every evening for two months
        // looks exactly as old as one watched once and abandoned, and gets deleted just as readily.
        // And the entries are nowhere near equal in what losing them costs — a search is a free
        // round trip to rebuild, a downloaded subtitle is one metered credit, a translation is a
        // whole film's LLM bill against the viewer's own provider account.
        live.sort_by_key(|(mtime, _, path)| (Self::evict_rank(path), *mtime));
        for (_, size, path) in live {
            if total <= budget {
                break;
            }
            if std::fs::remove_file(&path).is_ok() {
                total -= size;
            }
        }
    }

    /// Eviction order for one stored file: what it costs to lose, cheapest first.
    ///
    /// Read back off the filename, which is the key base64'd — and for an over-long key, the base64
    /// of its first 32 characters plus a hash. Either way the leading namespace survives, which is
    /// all this needs. Anything unreadable sorts first: a file this cache cannot name is a file it
    /// cannot serve.
    fn evict_rank(path: &std::path::Path) -> u8 {
        // A scratch file goes first, whatever key it was named after. Only an UNDATABLE temp reaches
        // the budget at all (see the sweep), and it is a half-written file of unknown provenance —
        // decoding its name would hand it the rank of the real entry it was becoming, so a temp
        // named after a translation would be kept while live entries that cost credits were evicted.
        if path.extension().is_some_and(|e| e.to_string_lossy().starts_with('t')) {
            return 0;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else { return 0 };
        let head = name.split('~').next().unwrap_or(name);
        let Ok(bytes) = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(head) else { return 0 };
        let key = String::from_utf8_lossy(&bytes);
        if key.starts_with("search:") {
            0 // a free round trip to rebuild
        } else if key.starts_with("os:") {
            1 // one metered download credit
        } else if key.starts_with("translate:") {
            2 // a whole film's LLM bill
        } else {
            // Pins, allowance counters, failure markers. All tiny, so evicting them frees nothing —
            // and a lost pin can cost a re-translation of every language of a film.
            3
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

    /// Remember this in memory only.
    ///
    /// `put` writes through to disk synchronously — an `fs::write` plus a rename, on the one runtime
    /// thread that also serves every connection. That is the right trade for an artifact fetched once
    /// and served for sixty days. It is the wrong one for something written a hundred and fifty times
    /// per film, which exists only to rescue a retry a few minutes later and is redundant the moment
    /// the finished artifact is cached.
    ///
    /// The saving is not really the write itself — it is that ~150 files per film, at up to fifty
    /// films a day, would leave the hourly `sweep` opening and header-reading thousands of extra
    /// files on that same thread.
    ///
    /// The cost is real and worth naming: a restart mid-run — a deploy, an OOM, a host reboot —
    /// now re-buys the whole film, where a disk-backed entry would have rescued the batches already
    /// paid for. That is the trade. A retry by the same process, which is the common case and the
    /// one the viewer is sitting in front of, is unaffected.
    pub fn put_mem(&self, key: String, value: String, ttl: Duration) {
        self.mem_put(key, value, ttl);
    }

    /// Forget a key in both tiers. For an entry that has been proven wrong rather than merely stale
    /// — the alternative, writing it back with a zero TTL, means a blocking disk write to say
    /// "delete this".
    pub fn remove(&self, key: &str) {
        {
            let mut g = self.inner.lock().unwrap();
            if let Some(e) = g.map.remove(key) {
                g.bytes -= e.size;
            }
        }
        if let Some(path) = self.disk_path(key) {
            let _ = std::fs::remove_file(path);
        }
    }

    /// Forget a memory-only key. The companion to `put_mem` that `remove` would be wrong for: that
    /// one also unlinks a file, which for a key nothing ever wrote to disk is a syscall to delete
    /// something that was never there.
    pub fn remove_mem(&self, key: &str) {
        let mut g = self.inner.lock().unwrap();
        if let Some(e) = g.map.remove(key) {
            g.bytes -= e.size;
        }
    }

    /// Read from memory only. The companion to `put_mem`: going through `get` would fall through to
    /// a disk probe that, for a key only ever written by `put_mem`, cannot hit — a blocking `open`
    /// per lookup, ~150 of them per resumed film, all of them ENOENT.
    pub fn get_mem(&self, key: &str) -> Option<String> {
        self.mem_get(key)
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
    ///
    /// Base64 is 4 bytes out per 3 in, and NAME_MAX is 255 on every filesystem this runs on — so a
    /// key past ~190 bytes produces a name the OS refuses. That is reachable: a translate key carries
    /// a model name and a language, each bounded but generously. The write then fails for every such
    /// key forever, and the first one spends the process's one-shot "persistence degraded" warning,
    /// so the next real disk problem says nothing.
    ///
    /// Long keys therefore get a hashed name instead. The prefix is kept readable so the store is
    /// still greppable by namespace, and the hash is over the WHOLE key, so two long keys sharing a
    /// prefix stay distinct.
    fn disk_path(&self, key: &str) -> Option<PathBuf> {
        /// Bytes of key that base64 to a comfortably-legal filename.
        const MAX_PLAIN: usize = 180;

        let dir = self.dir.as_ref()?;
        let encode = |b: &[u8]| base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b);
        let name = if key.len() <= MAX_PLAIN {
            encode(key.as_bytes())
        } else {
            // A FIXED hash, not `DefaultHasher`: this names a file on disk that outlives the
            // process, and std may change that algorithm between compiler releases — which would
            // rename every hashed entry at once, orphaning them all.
            let digest = crate::httputil::stable_hash(key.as_bytes());
            // On a char boundary — a key is UTF-8 and a byte slice through a multibyte char panics.
            let head: String = key.chars().take(32).collect();
            // `~` separates the two namespaces because it is not in the base64url alphabet, so a
            // plain name can never equal a hashed one. A `-` would not do: `-` and the hex digits
            // are all legal base64url output, so some short key could encode to exactly a long key's
            // hashed name and the two would share a file. Nor a `.`, which `disk_put`'s
            // `with_extension` would then treat as the extension and overwrite.
            format!("{}~{digest:016x}", encode(head.as_bytes()))
        };
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

    /// Just the expiry, without pulling the value into memory — a sweep over a full store would
    /// otherwise read every entry whole (up to MAX_BODY each) to ask one question. `Ok(None)` means
    /// the header is not ours: an older format, or garbage.
    fn read_expiry(path: &std::path::Path) -> std::io::Result<Option<u64>> {
        use std::io::Read;
        let mut head = [0u8; 48];
        let n = std::fs::File::open(path)?.read(&mut head)?;
        let text = String::from_utf8_lossy(&head[..n]);
        let Some((line, _)) = text.split_once('\n') else { return Ok(None) };
        let Some((expiry, len)) = line.split_once(' ') else { return Ok(None) };
        if len.parse::<usize>().is_err() {
            return Ok(None);
        }
        Ok(expiry.parse::<u64>().ok())
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
        let wrote = std::fs::write(&tmp, body).and_then(|()| std::fs::rename(&tmp, &path));
        if let Err(e) = wrote {
            // Once per process: a full or read-only volume degrades the cache to memory-only, which
            // survives a restart as a cold cache and used to say nothing at all.
            static WARNED: AtomicU64 = AtomicU64::new(0);
            if WARNED.fetch_add(1, Ordering::Relaxed) == 0 {
                eprintln!("warning: cache store write failed ({e}) — persistence degraded");
            }
            let _ = std::fs::remove_file(&tmp);
        }
    }
}

fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn modified_secs(meta: &std::fs::Metadata) -> Option<u64> {
    meta.modified().ok()?.duration_since(UNIX_EPOCH).ok().map(|d| d.as_secs())
}

/// How long ago this file was written, or `None` if the filesystem won't say.
fn age(meta: &std::fs::Metadata) -> Option<Duration> {
    meta.modified().ok()?.elapsed().ok()
}

fn next_temp_id() -> u64 {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    SEQ.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn tmpdir(name: &str) -> PathBuf {
        // The counter restarts per process, so include the pid: two concurrent `cargo test` runs
        // otherwise pick the same directory and delete each other's files.
        let d = std::env::temp_dir()
            .join(format!("den-subs-cache-{name}-{}-{}", std::process::id(), next_temp_id()));
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
        // Budget fits one entry, so after the expired one is dropped the oldest must still go.
        let c = Cache::new(60, Some(dir.clone()));
        c.put("expired".into(), "x".repeat(40), Duration::from_secs(0));
        c.put("old".into(), "x".repeat(40), HOUR);
        std::thread::sleep(std::time::Duration::from_millis(1100)); // mtime has 1s resolution
        c.put("new".into(), "x".repeat(40), HOUR);

        let expired_path = c.disk_path("expired").unwrap();
        let old_path = c.disk_path("old").unwrap();
        c.sweep();
        // Both branches must have run: dropping the expired entry alone left the store under
        // budget, so this test used to return before the eviction loop it exists to cover.
        assert!(!old_path.exists(), "the eviction loop did not run — the oldest entry survived");
        // Assert the FILE is gone, not that `get` misses — `disk_get` expires lazily on read, so a
        // `get` returning None passes whether or not the sweep did anything at all.
        assert!(!expired_path.exists(), "the sweep left an expired entry on disk");
        let fresh = Cache::new(60, Some(dir.clone()));
        assert_eq!(fresh.get("new"), Some("x".repeat(40)), "the newest entry must survive");
        let bytes: u64 = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.metadata().unwrap().len()).sum();
        assert!(bytes <= 60, "the sweep left {bytes} bytes on disk, over the 60 budget");
    }

    /// Under pressure the sweep spends the cheap entries first. Age alone made it evict in WRITE
    /// order — `disk_get` never touches mtime — so a translation served every evening looked exactly
    /// as old as one watched once, and a free-to-refetch search entry written a minute later
    /// outlived it. What the entries cost to lose is nowhere near equal: a search is a round trip, a
    /// downloaded subtitle is one metered credit, a translation is a whole film's LLM bill.
    #[test]
    fn the_sweep_spends_the_cheap_entries_first() {
        let dir = tmpdir("sweep-rank");
        // The expensive entry is written FIRST, so age alone would take it.
        let c = Cache::new(120, Some(dir.clone()));
        c.put("translate:500:SV:openai:m".into(), "x".repeat(40), HOUR);
        std::thread::sleep(std::time::Duration::from_millis(1100)); // mtime has 1s resolution
        c.put("os:777".into(), "x".repeat(40), HOUR);
        c.put("search:tt0111161:0:0:".into(), "x".repeat(40), HOUR);

        let translation = c.disk_path("translate:500:SV:openai:m").unwrap();
        let search = c.disk_path("search:tt0111161:0:0:").unwrap();
        c.sweep();

        assert!(!search.exists(), "the free-to-rebuild entry survived");
        assert!(translation.exists(), "a film's LLM bill was evicted before a search round trip");
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

    /// Unreadable is not expired. A redeploy that shifts the volume's uid makes every file
    /// unreadable at once, and a sweep that reads that as "expired" deletes the entire store —
    /// where `disk_get` treats the same error as a benign miss and leaves the file alone.
    #[test]
    fn the_sweep_keeps_an_entry_it_cannot_read() {
        let dir = tmpdir("sweep-unreadable");
        let c = Cache::new(1 << 20, Some(dir.clone()));
        c.put("k".into(), "v".into(), HOUR);
        let path = c.disk_path("k").unwrap();
        let mut perms = std::fs::metadata(&path).unwrap().permissions();
        perms.set_mode(0o000);
        std::fs::set_permissions(&path, perms).unwrap();

        c.sweep();
        assert!(path.exists(), "the sweep deleted an entry it merely could not read");

        let mut perms = std::fs::metadata(&path).unwrap().permissions();
        perms.set_mode(0o644);
        std::fs::set_permissions(&path, perms).unwrap();
        assert_eq!(Cache::new(1 << 20, Some(dir)).get("k"), Some("v".into()), "and it still reads");
    }

    /// A temp left by a kill between write and rename is addressed by nothing, but parses fine —
    /// so a sweep counted it as live and could evict a real entry to make room for it.
    #[test]
    fn the_sweep_reclaims_leftover_temps() {
        let dir = tmpdir("sweep-temp");
        let c = Cache::new(1 << 20, Some(dir.clone()));
        c.put("k".into(), "v".into(), HOUR);
        let stray = c.disk_path("k").unwrap().with_extension("t99");
        std::fs::write(&stray, format!("{} 1\nv", unix_now() + 3600)).unwrap();
        backdate(&stray, Duration::from_secs(3600));
        c.sweep();
        assert!(!stray.exists(), "a leftover temp survived the sweep");
        assert_eq!(Cache::new(1 << 20, Some(dir)).get("k"), Some("v".into()), "the real entry stayed");
    }

    /// `disk_put` writes to a `.tN` name and then renames it into place. A sweep that unlinks those
    /// on sight can delete an in-flight write, which fails the rename and makes a healthy disk
    /// report itself degraded — so a temp too young to be abandoned is left alone.
    #[test]
    fn the_sweep_leaves_an_in_flight_temp_alone() {
        let dir = tmpdir("sweep-inflight");
        let c = Cache::new(1 << 20, Some(dir.clone()));
        c.put("k".into(), "v".into(), HOUR);
        let in_flight = c.disk_path("k").unwrap().with_extension("t42");
        std::fs::write(&in_flight, "half-written").unwrap();
        c.sweep();
        assert!(in_flight.exists(), "the sweep deleted a temp that was still being written");
    }

    /// An undatable temp must not pile up forever either: nothing else reclaims a `.tN` file, so it
    /// is counted toward the budget and evicted under pressure like anything else.
    #[test]
    fn an_undatable_temp_is_still_bounded_by_the_budget() {
        let dir = tmpdir("sweep-undatable-budget");
        let c = Cache::new(60, Some(dir.clone()));
        c.put("k".into(), "x".repeat(40), HOUR);
        let stray = c.disk_path("k").unwrap().with_extension("t3");
        std::fs::write(&stray, "x".repeat(80)).unwrap();
        shift_mtime(&stray, std::time::SystemTime::now() + Duration::from_secs(5));
        let entry = c.disk_path("k").unwrap();
        c.sweep();
        let bytes: u64 = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.metadata().unwrap().len()).sum();
        assert!(bytes <= 60, "an undatable temp escaped the budget: {bytes} bytes left");
        // And it goes BEFORE live data: evicting a real subtitle to keep a scratch file of unknown
        // provenance costs a re-download against the viewer's quota.
        assert!(!stray.exists(), "the temp outlived the entry it displaced");
        assert!(entry.exists(), "a live entry was evicted to make room for an undatable temp");
    }

    /// A temp whose mtime reads as being in the FUTURE — an NTP step backwards, or a filesystem
    /// that does not report mtime — makes `elapsed()` fail. Treating an unknown age as "old enough"
    /// deletes the in-flight write the grace period exists to protect, and on a filesystem without
    /// mtime it disables the grace period permanently.
    #[test]
    fn the_sweep_leaves_a_temp_it_cannot_date() {
        let dir = tmpdir("sweep-undatable");
        let c = Cache::new(1 << 20, Some(dir.clone()));
        c.put("k".into(), "v".into(), HOUR);
        let in_flight = c.disk_path("k").unwrap().with_extension("t7");
        std::fs::write(&in_flight, "half-written").unwrap();
        shift_mtime(&in_flight, std::time::SystemTime::now() + Duration::from_secs(5));
        c.sweep();
        assert!(in_flight.exists(), "the sweep deleted a temp whose age it could not determine");
    }

    /// Push a file's modification time into the past.
    fn backdate(path: &std::path::Path, by: Duration) {
        shift_mtime(path, std::time::SystemTime::now() - by);
    }

    fn shift_mtime(path: &std::path::Path, when: std::time::SystemTime) {
        let f = std::fs::File::options().write(true).open(path).unwrap();
        f.set_times(std::fs::FileTimes::new().set_modified(when)).unwrap();
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
