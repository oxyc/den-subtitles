//! Runtime configuration from the environment. No user credentials live here — both the OpenSubtitles
//! key and the LLM key are BYOK and ride in the install URL (see `userconfig.rs`). What lives here
//! is addon-level infrastructure: binary paths, cache sizing, the public origin, and the addon's own
//! sealing key and metrics token.

use std::env;
use std::path::PathBuf;

pub struct Config {
    pub port: u16,
    pub cache_dir: PathBuf,
    pub cache_max_bytes: u64,
    /// Fixed public origin for building `/subtitle/…` URLs we hand back; falls back to forwarded
    /// headers when unset.
    pub public_base_url: Option<String>,
    pub ffsubsync: String,
    pub alass: String,
    /// Sealed config-in-URL (den-scout/docs/SEALED-CONFIG.md). `config_key` = current X25519 private key
    /// (base64); `config_keys_prev` = comma-separated prior keys (rotation). Empty → sealed URLs disabled.
    pub config_key: String,
    pub config_keys_prev: String,
    /// Bearer token for `/metrics`. Empty → the route answers 404, like any unknown path.
    pub metrics_token: String,
    /// OpenSubtitles API root. A field so tests can point the request path at a local server instead
    /// of the live API — several of them reach `handle_translate`, which searches before it can do
    /// anything else, and without this they made a real request to api.opensubtitles.com on every
    /// `cargo test`. They passed either way, which is what made it easy to miss.
    pub os_api_base: String,
}

fn env_opt(key: &str) -> Option<String> {
    match env::var(key) {
        Ok(v) if !v.is_empty() => Some(v),
        _ => None,
    }
}

impl Config {
    pub fn from_env() -> Config {
        let cache_dir = env_opt("CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| env::temp_dir().join("den-subtitles-cache"));
        Config {
            port: env_opt("PORT").and_then(|v| v.parse().ok()).unwrap_or(8093),
            cache_dir,
            cache_max_bytes: env_opt("CACHE_MAX_BYTES")
                .and_then(|v| v.parse().ok())
                .unwrap_or(256 * 1024 * 1024), // 256 MB — SRTs are tiny, this holds a lot of films
            public_base_url: env_opt("PUBLIC_BASE_URL"),
            ffsubsync: env_opt("FFSUBSYNC_PATH").unwrap_or_else(|| "ffsubsync".to_string()),
            alass: env_opt("ALASS_PATH").unwrap_or_else(|| "alass".to_string()),
            config_key: env_opt("CONFIG_KEY").unwrap_or_default(),
            config_keys_prev: env_opt("CONFIG_KEYS_PREV").unwrap_or_default(),
            metrics_token: env_opt("METRICS_TOKEN").unwrap_or_default(),
            os_api_base: crate::opensubtitles::API.to_string(),
        }
    }
}
