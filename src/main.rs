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
mod httputil;
mod opensubtitles;
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

pub async fn handle_request(state: Arc<AppState>, req: Request<hyper::body::Incoming>) -> Response<Body> {
    let (parts, _body) = req.into_parts();
    let path = parts.uri.path();

    match path {
        "/health" => return httputil::text(StatusCode::OK, "ok"),
        "/manifest.json" => return httputil::json(StatusCode::OK, &addon::manifest(false), "public, max-age=3600"),
        "/" | "/configure" | "/configure/" => {
            return httputil::html(StatusCode::OK, CONFIGURE_PAGE, "public, max-age=3600")
        }
        _ => {}
    }

    let segs = split_path(path);
    let config = segs.first().copied().unwrap_or("");
    let resource = segs.get(1).copied().unwrap_or("");

    match resource {
        "manifest.json" => match userconfig::decode(config) {
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
            // /<config>/subtitle/<file_id>.srt
            let file = segs.get(2).copied().unwrap_or("");
            match file.strip_suffix(".srt").and_then(|n| n.parse::<i64>().ok()) {
                Some(file_id) => addon::handle_subtitle_file(&state, config, file_id).await,
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

fn strip_json(seg: &str) -> Option<&str> {
    seg.strip_suffix(".json")
}

async fn run(cfg: Config) -> std::io::Result<()> {
    std::fs::create_dir_all(&cfg.cache_dir)?;
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
