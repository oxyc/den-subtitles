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

// Generic over the request body: this handler routes on path/query only and discards the body, so tests
// can drive it with a `Request<()>` while `run()` passes the real `Request<Incoming>`.
pub async fn handle_request<B>(state: Arc<AppState>, req: Request<B>) -> Response<Body> {
    let (parts, _body) = req.into_parts();
    let resp = route(&state, &parts).await;
    // Honor conditional GET/HEAD: any cacheable 200 carries an ETag, so an `If-None-Match` hit
    // collapses to a 304. Unsafe methods (none served today) keep their full response.
    if matches!(parts.method, hyper::Method::GET | hyper::Method::HEAD) {
        let resp = httputil::apply_conditional(resp, &parts.headers);
        // HEAD must not carry a body (the router builds one regardless of method).
        if parts.method == hyper::Method::HEAD {
            httputil::strip_body(resp)
        } else {
            resp
        }
    } else {
        resp
    }
}

async fn route(state: &Arc<AppState>, parts: &hyper::http::request::Parts) -> Response<Body> {
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
            addon::handle_subtitles(state, &parts.headers, config, id, extra).await
        }
        "subtitle" => {
            // /<config>/subtitle/<file_id>.srt[?ref=<id>|?resync=<stream-url>]
            let file = segs.get(2).copied().unwrap_or("");
            match file.strip_suffix(".srt").and_then(|n| n.parse::<i64>().ok()) {
                Some(file_id) => {
                    let query = parts.uri.query().unwrap_or("");
                    let ref_id = query_get(query, "ref").and_then(|v| v.parse().ok());
                    let resync = query_get(query, "resync");
                    addon::handle_subtitle_file(state, config, file_id, ref_id, resync).await
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
            addon::handle_translate(state, &parts.headers, config, id, lang, want_json).await
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
    use base64::Engine;

    // Same fixed vector key + browser-minted sealed segment the seal/userconfig unit tests use, so the
    // HTTP layer is exercised against a real crypto_box_seal blob, not a mock.
    const VEC_PRIV_B64: &str = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=";
    const VEC_PUB_B64: &str = "j0DFrbaPJWJK5bIU6nZ6bslNgp09e14a0bpvPiE4KF8=";
    const JS_SEG: &str = "Ac3WWHzRZKV9OjdSgIPaNFFhaE9UY0vwxgSO6F5Ghug1nyjlKUodEQmhhlPhX-j1KffJnpj58HPhlpePWcbnuX9GL9rGMsdki1hGXSzRG94ON_aYocvFkl9bSU2QZa8o3waeHHm9wmjLQg";

    fn test_state(config_key: &str) -> Arc<AppState> {
        let cfg = Config {
            port: 0,
            cache_dir: std::env::temp_dir().join("den-subtitles-test-cache"),
            cache_max_bytes: 8 * 1024 * 1024,
            public_base_url: None,
            ffsubsync: "ffsubsync".to_string(),
            alass: "alass".to_string(),
            config_key: config_key.to_string(),
            config_keys_prev: String::new(),
        };
        AppState::new(cfg)
    }

    async fn body_string(resp: Response<Body>) -> String {
        use http_body_util::BodyExt;
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    // The HTTP-level mirror of den-scout's TestRoutesSealedConfig: drive the real router so a future
    // refactor that returns the wrong status/body for the sealed, legacy, or /config-key arms fails CI.

    #[tokio::test]
    async fn config_key_serves_the_pubkey_when_keyring_set() {
        let resp = handle_request(test_state(VEC_PRIV_B64), Request::builder().uri("/config-key").body(()).unwrap()).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(body_string(resp).await.contains(VEC_PUB_B64), "/config-key must serve the derived pubkey");
    }

    #[tokio::test]
    async fn config_key_404s_when_sealing_disabled() {
        let resp = handle_request(test_state(""), Request::builder().uri("/config-key").body(()).unwrap()).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn sealed_manifest_resolves_end_to_end() {
        let uri = format!("/{JS_SEG}/manifest.json");
        let resp = handle_request(test_state(VEC_PRIV_B64), Request::builder().uri(uri).body(()).unwrap()).await;
        assert_eq!(resp.status(), StatusCode::OK, "a sealed URL must resolve the manifest");
    }

    #[tokio::test]
    async fn sealed_manifest_fails_closed_without_a_keyring() {
        let uri = format!("/{JS_SEG}/manifest.json");
        let resp = handle_request(test_state(""), Request::builder().uri(uri).body(()).unwrap()).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "sealed URL with no key must fail closed, not open");
    }

    #[tokio::test]
    async fn legacy_plaintext_manifest_still_resolves_with_a_keyring_present() {
        let seg = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(r#"{"osKey":"os-legacy"}"#);
        let uri = format!("/{seg}/manifest.json");
        let resp = handle_request(test_state(VEC_PRIV_B64), Request::builder().uri(uri).body(()).unwrap()).await;
        assert_eq!(resp.status(), StatusCode::OK, "legacy plaintext config must still resolve (back-compat)");
    }

    #[tokio::test]
    async fn configure_page_renders_with_the_seal_bundle() {
        // Guards against a truncated/corrupt include_str! shipping silently (audit finding B).
        let resp = handle_request(test_state(VEC_PRIV_B64), Request::builder().uri("/configure").body(()).unwrap()).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let page = body_string(resp).await;
        assert!(page.contains("DenSeal"), "the /configure page must inline the seal bundle");
    }

    #[tokio::test]
    async fn manifest_carries_a_strong_etag_and_honors_if_none_match() {
        use hyper::header::{ETAG, IF_NONE_MATCH};
        // A cacheable 200 must carry a strong (quoted, unweakened) ETag.
        let resp = handle_request(
            test_state(VEC_PRIV_B64),
            Request::builder().uri("/manifest.json").body(()).unwrap(),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let etag = resp.headers().get(ETAG).expect("manifest must carry an ETag").clone();
        let etag_str = etag.to_str().unwrap().to_string();
        assert!(etag_str.starts_with('"') && etag_str.ends_with('"'), "ETag must be quoted/strong");

        // Re-request with that ETag → 304 Not Modified, no body, same validator + Cache-Control.
        let resp = handle_request(
            test_state(VEC_PRIV_B64),
            Request::builder()
                .uri("/manifest.json")
                .header(IF_NONE_MATCH, &etag_str)
                .body(())
                .unwrap(),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::NOT_MODIFIED);
        assert_eq!(resp.headers().get(ETAG).unwrap(), &etag);
        assert!(resp.headers().get(hyper::header::CACHE_CONTROL).is_some());
        assert!(body_string(resp).await.is_empty(), "a 304 has an empty body");
    }

    #[tokio::test]
    async fn no_store_replies_carry_no_etag_and_never_304() {
        use hyper::header::{ETAG, IF_NONE_MATCH};
        // /health is no-store — it must not get an ETag, and a wildcard If-None-Match can't 304 it.
        let resp = handle_request(
            test_state(VEC_PRIV_B64),
            Request::builder().uri("/health").header(IF_NONE_MATCH, "*").body(()).unwrap(),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp.headers().get(ETAG).is_none(), "no-store bodies get no ETag");
    }

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
