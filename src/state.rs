//! Shared application state: one pooled HTTP client, the artifact cache, and the sync tools, wired
//! from `Config` and cloned (behind `Arc`) into every connection.

use std::sync::Arc;
use std::time::Duration;

use crate::cache::Cache;
use crate::config::Config;
use crate::sync::SyncTools;

pub struct AppState {
    pub cfg: Config,
    /// The pooled HTTP client. `None` if TLS init failed at boot — health/manifest/configure still
    /// serve; the subtitle/translate routes 503 instead of the whole process refusing to boot.
    pub http: Option<reqwest::Client>,
    pub cache: Cache,
    pub sync: SyncTools,
}

impl AppState {
    pub fn new(cfg: Config) -> Arc<AppState> {
        // Bounded so an upstream (OpenSubtitles / LLM) that never responds can't pin a request task
        // forever. The connect bound is tight; the overall bound is generous because a translation
        // batch on a slow model is legitimately slow (per-request LLM calls override it upward).
        let http = match reqwest::Client::builder()
            .user_agent("den-subtitles/0.1")
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
        Arc::new(AppState { cfg, http, cache, sync })
    }
}
