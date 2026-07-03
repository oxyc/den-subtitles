//! den-subtitles — a self-hosted Stremio subtitles addon for Den in one small binary:
//!
//! 1. FETCH — OpenSubtitles, hash-matched (the app sends the file's OSHash as `videoHash`), served
//!    from our own cache/origin so results are fast and dodge the per-IP download quota.
//! 2. TRANSLATE — BYOK LLM translation (gpt-4o-mini default) via a chunk→same-length→retry harness
//!    that survives a full film without losing sync between cue count and timing.
//! 3. SYNC — an auto-sync ladder (hash → reference-align → alass audio VAD) so subtitles line up.
//!
//! Both credentials (OpenSubtitles key, LLM key) are BYOK and ride in the install URL, Keychain-
//! stored by the app. Nothing credential-shaped lives in the environment.

mod addon;
mod cache;
mod config;
mod fetch;
mod httputil;
mod opensubtitles;
mod seal;
mod srt;
mod state;
mod sync;
mod translate;
mod userconfig;

use std::convert::Infallible;
use std::sync::Arc;

use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;

use crate::config::Config;
use crate::httputil::Body;
use crate::state::AppState;

/// The /configure page, embedded so the binary is self-contained.
const CONFIGURE_PAGE: &str = include_str!("configure.html");

/// Consecutive OpenSubtitles failures before /health reports `degraded` (ADDON-02).
const HEALTH_FAIL_THRESHOLD: u32 = 3;

/// The /health response body for a given consecutive-failure count (ADDON-02). Pure so the
/// degraded/ok decision is unit-testable without standing up the HTTP server.
fn health_body(os_fails: u32) -> serde_json::Value {
    if os_fails >= HEALTH_FAIL_THRESHOLD {
        serde_json::json!({"status": "degraded", "reason": "upstream_unavailable", "detail": "OpenSubtitles has been failing"})
    } else {
        serde_json::json!({"status": "ok"})
    }
}

pub async fn handle_request(state: Arc<AppState>, req: Request<hyper::body::Incoming>) -> Response<Body> {
    let (parts, _body) = req.into_parts();
    let path = parts.uri.path();

    match path {
        // Standard Den addon health (ADDON-02): 200 for liveness, `degraded` when OpenSubtitles has been
        // failing so the app's Plugins screen can surface it.
        "/health" => {
            let body = health_body(state.os_fails.load(std::sync::atomic::Ordering::Relaxed));
            return httputil::json(StatusCode::OK, &body, "no-store");
        }
        "/manifest.json" => return httputil::json(StatusCode::OK, &addon::manifest(false), "public, max-age=3600"),
        "/" | "/configure" | "/configure/" => {
            return httputil::html(StatusCode::OK, CONFIGURE_PAGE, "public, max-age=3600")
        }
        // The current X25519 public key (base64) so /configure can seal the config to it; 404 when
        // sealed configs are disabled (no key) — the page then keeps plaintext (SEALED-CONFIG.md).
        "/config-key" => {
            return match state.config_keyring.as_ref().map(|kr| kr.current_pub_b64()) {
                Some(pub_b64) if !pub_b64.is_empty() => {
                    httputil::json(StatusCode::OK, &serde_json::json!({"key": pub_b64}), "public, max-age=3600")
                }
                _ => httputil::json(StatusCode::NOT_FOUND, &serde_json::json!({"error": "no_key"}), "no-store"),
            };
        }
        _ => {}
    }

    let segs = split_path(path);
    let config = segs.first().copied().unwrap_or("");
    let resource = segs.get(1).copied().unwrap_or("");

    match resource {
        "manifest.json" => match userconfig::decode(state.config_keyring.as_ref(), config) {
            Some(_) => httputil::json(StatusCode::OK, &addon::manifest(true), "public, max-age=3600"),
            None => httputil::json(StatusCode::BAD_REQUEST, &serde_json::json!({"error": "bad_config"}), "no-store"),
        },
        "subtitles" => {
            // /<config>/subtitles/<type>/<id>[/<extra>].json
            let (id_seg, extra) = match segs.len() {
                5 => (segs[3], segs[4]),
                4 => (segs[3], ""),
                _ => return httputil::text(StatusCode::NOT_FOUND, "not found"),
            };
            let id = strip_json(id_seg).unwrap_or(id_seg);
            let extra = strip_json(extra).unwrap_or(extra);
            addon::handle_subtitles(&state, &parts.headers, config, id, extra).await
        }
        "subtitle" => {
            // /<config>/subtitle/<file_id>.srt[?ref=<id>|?resync=<stream-url>]
            let file = segs.get(2).copied().unwrap_or("");
            match file.strip_suffix(".srt").and_then(|n| n.parse::<i64>().ok()) {
                Some(file_id) => {
                    let query = parts.uri.query().unwrap_or("");
                    let ref_id = query_get(query, "ref").and_then(|v| v.parse().ok());
                    let resync = query_get(query, "resync");
                    addon::handle_subtitle_file(&state, config, file_id, ref_id, resync).await
                }
                None => httputil::text(StatusCode::BAD_REQUEST, "bad file id"),
            }
        }
        "translate" => {
            // /<config>/translate/<type>/<id>/<lang>.(json|srt)
            if segs.len() != 5 {
                return httputil::text(StatusCode::NOT_FOUND, "not found");
            }
            let id = segs[3];
            let last = segs[4];
            let (lang, want_json) = if let Some(l) = last.strip_suffix(".json") {
                (l, true)
            } else if let Some(l) = last.strip_suffix(".srt") {
                (l, false)
            } else {
                return httputil::text(StatusCode::NOT_FOUND, "not found");
            };
            addon::handle_translate(&state, &parts.headers, config, id, lang, want_json).await
        }
        _ => httputil::text(StatusCode::NOT_FOUND, "not found"),
    }
}

fn split_path(path: &str) -> Vec<&str> {
    path.split('/').filter(|s| !s.is_empty()).collect()
}

/// Percent-decoded value of a query parameter, or None.
fn query_get(query: &str, key: &str) -> Option<String> {
    let prefix = format!("{key}=");
    query
        .split('&')
        .find_map(|p| p.strip_prefix(&prefix))
        .map(httputil::percent_decode)
}

fn strip_json(seg: &str) -> Option<&str> {
    seg.strip_suffix(".json")
}

async fn run(cfg: Config) -> std::io::Result<()> {
    // Don't refuse to boot on an unwritable cache mount — the artifact cache is fully in-memory and
    // the sync scratch dir creates itself lazily. A failed pre-create is logged, not fatal (a hard
    // exit here would crash-loop the container and the app would see only connection-refused).
    if let Err(e) = std::fs::create_dir_all(&cfg.cache_dir) {
        eprintln!("warning: cache dir {} not writable ({e}) — sync tiers will retry lazily", cfg.cache_dir.display());
    }
    let port = cfg.port;
    let state = AppState::new(cfg);
    let listener = TcpListener::bind(("0.0.0.0", port)).await?;
    println!("den-subtitles on :{port} (keys are per-install; build one at /configure)");

    loop {
        let (stream, _) = match listener.accept().await {
            Ok(pair) => pair,
            Err(e) => {
                eprintln!("accept: {e}");
                continue;
            }
        };
        let state = state.clone();
        tokio::spawn(async move {
            let io = TokioIo::new(stream);
            let service = service_fn(move |req| {
                let state = state.clone();
                async move { Ok::<_, Infallible>(handle_request(state, req).await) }
            });
            let _ = hyper::server::conn::http1::Builder::new()
                .serve_connection(io, service)
                .await;
        });
    }
}

/// `den-subtitles healthcheck` — used by the container HEALTHCHECK so the slim image needs no curl.
async fn healthcheck(port: u16) -> i32 {
    match reqwest::get(format!("http://127.0.0.1:{port}/health")).await {
        Ok(r) if r.status().is_success() => 0,
        _ => 1,
    }
}

fn main() {
    let cfg = Config::from_env();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    if std::env::args().nth(1).as_deref() == Some("healthcheck") {
        std::process::exit(rt.block_on(healthcheck(cfg.port)));
    }
    if let Err(e) = rt.block_on(run(cfg)) {
        eprintln!("fatal: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_ok_below_threshold() {
        // Fewer than HEALTH_FAIL_THRESHOLD consecutive failures → healthy liveness.
        for fails in 0..HEALTH_FAIL_THRESHOLD {
            let body = health_body(fails);
            assert_eq!(body["status"], "ok", "fails={fails} should be ok");
            // No degraded fields leak into the healthy body.
            assert!(body.get("reason").is_none());
            assert!(body.get("detail").is_none());
        }
    }

    #[test]
    fn health_degraded_at_and_above_threshold() {
        // At the threshold and beyond, OpenSubtitles is treated as down (ADDON-02).
        for fails in [HEALTH_FAIL_THRESHOLD, HEALTH_FAIL_THRESHOLD + 1, 100] {
            let body = health_body(fails);
            assert_eq!(body["status"], "degraded", "fails={fails} should be degraded");
            assert_eq!(body["reason"], "upstream_unavailable");
            assert_eq!(body["detail"], "OpenSubtitles has been failing");
        }
    }
}
