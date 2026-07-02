//! Shared application state: one pooled HTTP client, the artifact cache, and the sync tools, wired
//! from `Config` and cloned (behind `Arc`) into every connection.

use std::sync::Arc;

use crate::cache::Cache;
use crate::config::Config;
use crate::sync::SyncTools;

pub struct AppState {
    pub cfg: Config,
    pub http: reqwest::Client,
    pub cache: Cache,
    pub sync: SyncTools,
}

impl AppState {
    pub fn new(cfg: Config) -> Arc<AppState> {
        let http = reqwest::Client::builder()
            .user_agent("den-subtitles/0.1")
            .build()
            .expect("http client");
        let sync = SyncTools {
            ffsubsync: cfg.ffsubsync.clone(),
            alass: cfg.alass.clone(),
            work_dir: cfg.cache_dir.join("sync"),
        };
        let cache = Cache::new(cfg.cache_max_bytes as usize);
        Arc::new(AppState { cfg, http, cache, sync })
    }
}
