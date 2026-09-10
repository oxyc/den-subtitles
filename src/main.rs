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
mod inflight;
mod opensubtitles;
mod seal;
mod srt;
mod state;
mod sync;
mod translate;
mod userconfig;

use std::convert::Infallible;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use hyper_util::server::graceful::GracefulShutdown;
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
        "/manifest.json" => {
            return httputil::json(StatusCode::OK, &addon::manifest(false), "public, max-age=3600")
        }
        "/" | "/configure" | "/configure/" => {
            return httputil::html(StatusCode::OK, CONFIGURE_PAGE, "public, max-age=3600")
        }
        // The current X25519 public key (base64) so /configure can seal the config to it; 404 when
        // sealed configs are disabled (no key) — the page then keeps plaintext (SEALED-CONFIG.md).
        "/config-key" => {
            return match state.config_keyring.as_ref().map(|kr| kr.current_pub_b64()) {
                Some(pub_b64) if !pub_b64.is_empty() => httputil::json(
                    StatusCode::OK,
                    &serde_json::json!({"key": pub_b64}),
                    "public, max-age=3600",
                ),
                _ => {
                    httputil::json(StatusCode::NOT_FOUND, &serde_json::json!({"error": "no_key"}), "no-store")
                }
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
            None => httputil::json(
                StatusCode::BAD_REQUEST,
                &serde_json::json!({"error": "bad_config"}),
                "no-store",
            ),
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
            // /<config>/subtitle/<file_id>.(srt|vtt)[?ref=<id>|?resync=<stream-url>]
            let file = segs.get(2).copied().unwrap_or("");
            let (stem, want_vtt) = split_subtitle_format(file);
            match stem.and_then(|n| n.parse::<i64>().ok()) {
                Some(file_id) => {
                    let query = parts.uri.query().unwrap_or("");
                    let ref_id = query_get(query, "ref").and_then(|v| v.parse().ok());
                    let resync = query_get(query, "resync");
                    let resp = addon::handle_subtitle_file(state, config, file_id, ref_id, resync).await;
                    if want_vtt {
                        httputil::to_vtt(resp, &parts.headers).await
                    } else {
                        resp
                    }
                }
                None => httputil::text(StatusCode::BAD_REQUEST, "bad file id"),
            }
        }
        "translate" => {
            // /<config>/translate/<type>/<id>[/<extra>]/<lang>.(json|srt)
            //
            // The <extra> segment carries this stream's videoHash, which is what lets a translation
            // find a Tier-1 anchor. It is optional so installs built against the older five-segment
            // shape keep resolving — they just get no anchor, which is where every translation was.
            let (id, extra, last) = match segs.len() {
                6 => (segs[3], segs[4], segs[5]),
                5 => (segs[3], "", segs[4]),
                _ => return httputil::text(StatusCode::NOT_FOUND, "not found"),
            };
            // `.status` is answered from the request alone, so it is dispatched before anything that
            // could make a poll expensive.
            if let Some(lang) = last.strip_suffix(".status") {
                return addon::handle_translate_status(state, config, id, lang).await;
            }
            let (lang, want_json, want_vtt) = if let Some(l) = last.strip_suffix(".json") {
                (l, true, false)
            } else if let Some(l) = last.strip_suffix(".srt") {
                (l, false, false)
            } else if let Some(l) = last.strip_suffix(".vtt") {
                (l, false, true)
            } else {
                return httputil::text(StatusCode::NOT_FOUND, "not found");
            };
            let resync = query_get(parts.uri.query().unwrap_or(""), "resync");
            let resp =
                addon::handle_translate(state, &parts.headers, config, id, extra, lang, want_json, resync)
                    .await;
            if want_vtt {
                httputil::to_vtt(resp, &parts.headers).await
            } else {
                resp
            }
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
    query.split('&').find_map(|p| p.strip_prefix(&prefix)).map(httputil::percent_decode)
}

fn strip_json(seg: &str) -> Option<&str> {
    seg.strip_suffix(".json")
}

/// Split a subtitle filename into its stem and the format asked for. SRT is what the engine takes;
/// VTT is the same document for a client that needs a `<track>`, and is rendered from the finished
/// SRT at the response rather than anywhere inside the sync ladder.
fn split_subtitle_format(file: &str) -> (Option<&str>, bool) {
    match (file.strip_suffix(".srt"), file.strip_suffix(".vtt")) {
        (Some(stem), _) => (Some(stem), false),
        (_, Some(stem)) => (Some(stem), true),
        _ => (None, false),
    }
}

async fn run(cfg: Config) -> std::io::Result<()> {
    // Don't refuse to boot on an unwritable cache mount — the artifact cache is fully in-memory and
    // the sync scratch dir creates itself lazily. A failed pre-create is logged, not fatal (a hard
    // exit here would crash-loop the container and the app would see only connection-refused).
    if let Err(e) = std::fs::create_dir_all(&cfg.cache_dir) {
        eprintln!(
            "warning: cache dir {} not writable ({e}) — sync tiers will retry lazily",
            cfg.cache_dir.display()
        );
    }
    let port = cfg.port;
    let state = AppState::new(cfg);
    let listener = TcpListener::bind(("0.0.0.0", port)).await?;
    println!("den-subtitles on :{port} (keys are per-install; build one at /configure)");

    // Reclaim the disk cache hourly. `Cache::new` sweeps at boot, which bounds the store across
    // restarts but not within one — a container that stays up keeps writing entries that only a
    // repeat request for the same key would ever expire.
    {
        let state = state.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(std::time::Duration::from_secs(3600));
            tick.tick().await; // fires immediately; the boot sweep already ran
            loop {
                tick.tick().await;
                // Off the runtime thread. The sweep stats every file in the store and header-reads
                // most of them — thousands of blocking opens at the default budget — and this
                // runtime has ONE thread, which is also the one serving every connection. Fast on a
                // local SSD; on a container bind-mount at a millisecond an open it is seconds of
                // total stall, every hour.
                let swept = state.clone();
                let done = tokio::task::spawn_blocking(move || {
                    swept.cache.sweep();
                    // The sync work dir too. `finish` is the only cleanup there and it is skipped
                    // when a request is cancelled mid-alignment, so without this each one leaks its
                    // scratch files permanently.
                    swept.sync.sweep_scratch();
                })
                .await;
                if let Err(e) = done {
                    eprintln!("cache sweep did not run: {e}");
                }
            }
        });
    }

    serve_until(listener, state, shutdown_signal(), DRAIN_GRACE).await;
    Ok(())
}

/// How long in-flight requests get to finish after SIGTERM. Under podman's default 10s stop timeout,
/// as den-atlas's and den-embed's are, so the drain works whether or not the Quadlet's --stop-timeout
/// has reached the box.
const DRAIN_GRACE: Duration = Duration::from_secs(8);

/// Serve until `shutdown` resolves, then let in-flight requests finish for at most `grace`. Without
/// it a redeploy killed the process outright, cutting every fetch and translation mid-response.
///
/// The bound is the point: a graceful shutdown waits for every connection, and a client that sends
/// half a request head and stops would otherwise decide how long a restart takes. Sync subprocesses
/// are `kill_on_drop`, so any still running at the deadline die with their tasks.
async fn serve_until(
    listener: TcpListener,
    state: Arc<AppState>,
    shutdown: impl Future<Output = ()>,
    grace: Duration,
) {
    let graceful = GracefulShutdown::new();
    tokio::pin!(shutdown);
    loop {
        let (stream, _) = tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok(pair) => pair,
                Err(e) => {
                    eprintln!("accept: {e}");
                    continue;
                }
            },
            _ = &mut shutdown => break,
        };
        let state = state.clone();
        let service = service_fn(move |req| {
            let state = state.clone();
            async move { Ok::<_, Infallible>(handle_request(state, req).await) }
        });
        let conn = hyper::server::conn::http1::Builder::new().serve_connection(TokioIo::new(stream), service);
        let conn = graceful.watch(conn);
        tokio::spawn(async move {
            let _ = conn.await;
        });
    }
    drop(listener);
    eprintln!("den-subtitles: shutting down — draining in-flight requests");
    tokio::select! {
        _ = graceful.shutdown() => {}
        _ = tokio::time::sleep(grace) => {
            eprintln!("den-subtitles: drain deadline ({grace:?}) reached with requests still in flight");
        }
    }
}

/// Resolves on SIGTERM (a redeploy) or SIGINT (a terminal).
async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};
    let mut term = match signal(SignalKind::terminate()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot listen for SIGTERM ({e}); a redeploy will cut in-flight requests");
            return std::future::pending().await;
        }
    };
    tokio::select! {
        _ = term.recv() => {},
        _ = tokio::signal::ctrl_c() => {},
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
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().expect("tokio runtime");

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
            // Never the live API from a test: port 1 refuses instantly.
            os_api_base: "http://127.0.0.1:1".to_string(),
        };
        AppState::new(cfg)
    }

    async fn body_string(resp: Response<Body>) -> String {
        use http_body_util::BodyExt;
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// `serve_until` on a loopback port, stopped by the returned sender instead of a signal.
    async fn start_serve(
        grace: Duration,
    ) -> (std::net::SocketAddr, tokio::sync::oneshot::Sender<()>, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let stop = async move {
            let _ = rx.await;
        };
        (addr, tx, tokio::spawn(serve_until(listener, test_state(""), stop, grace)))
    }

    #[tokio::test]
    async fn an_idle_server_stops_at_once() {
        let (_, stop, server) = start_serve(Duration::from_secs(5)).await;
        stop.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(1), server)
            .await
            .expect("an idle server waited out the grace instead of stopping")
            .unwrap();
    }

    /// A graceful shutdown waits for every connection, so without a deadline a client that sends half
    /// a request head and goes quiet would hold the stop open for as long as it liked.
    #[tokio::test]
    async fn a_half_sent_request_cannot_hold_the_stop_open() {
        use tokio::io::AsyncWriteExt;
        let (addr, stop, server) = start_serve(Duration::from_millis(300)).await;
        let mut sock = tokio::net::TcpStream::connect(addr).await.unwrap();
        sock.write_all(b"GET /health HTTP/1.1\r\nHost: x\r\n").await.unwrap(); // no terminating blank line
        tokio::time::sleep(Duration::from_millis(50)).await; // let the server accept it first
        stop.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(3), server).await.expect("the drain is unbounded").unwrap();
    }

    // The HTTP-level mirror of den-scout's TestRoutesSealedConfig: drive the real router so a future
    // refactor that returns the wrong status/body for the sealed, legacy, or /config-key arms fails CI.

    #[tokio::test]
    async fn config_key_serves_the_pubkey_when_keyring_set() {
        let resp =
            handle_request(test_state(VEC_PRIV_B64), Request::builder().uri("/config-key").body(()).unwrap())
                .await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(body_string(resp).await.contains(VEC_PUB_B64), "/config-key must serve the derived pubkey");
    }

    #[tokio::test]
    async fn config_key_404s_when_sealing_disabled() {
        let resp =
            handle_request(test_state(""), Request::builder().uri("/config-key").body(()).unwrap()).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn sealed_manifest_resolves_end_to_end() {
        let uri = format!("/{JS_SEG}/manifest.json");
        let resp =
            handle_request(test_state(VEC_PRIV_B64), Request::builder().uri(uri).body(()).unwrap()).await;
        assert_eq!(resp.status(), StatusCode::OK, "a sealed URL must resolve the manifest");
    }

    #[tokio::test]
    async fn sealed_manifest_fails_closed_without_a_keyring() {
        let uri = format!("/{JS_SEG}/manifest.json");
        let resp = handle_request(test_state(""), Request::builder().uri(uri).body(()).unwrap()).await;
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "sealed URL with no key must fail closed, not open"
        );
    }

    #[tokio::test]
    async fn legacy_plaintext_manifest_still_resolves_with_a_keyring_present() {
        let seg = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(r#"{"osKey":"os-legacy"}"#);
        let uri = format!("/{seg}/manifest.json");
        let resp =
            handle_request(test_state(VEC_PRIV_B64), Request::builder().uri(uri).body(()).unwrap()).await;
        assert_eq!(resp.status(), StatusCode::OK, "legacy plaintext config must still resolve (back-compat)");
    }

    #[tokio::test]
    async fn configure_page_renders_with_the_seal_bundle() {
        // Guards against a truncated/corrupt include_str! shipping silently (audit finding B).
        let resp =
            handle_request(test_state(VEC_PRIV_B64), Request::builder().uri("/configure").body(()).unwrap())
                .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let page = body_string(resp).await;
        // `contains("DenSeal")` alone passed on a page cut to 4% of its length — the marker sits in
        // the first few KB. The page must carry the seal implementation AND the form that uses it.
        assert!(page.contains("DenSeal"), "the /configure page must inline the seal bundle");
        assert!(page.contains("denSeal"), "the seal implementation itself must be present");
        assert!(page.contains("</html>"), "the page was truncated before the end");
        assert!(page.len() > 100_000, "the bundle looks truncated: {} bytes", page.len());
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
            Request::builder().uri("/manifest.json").header(IF_NONE_MATCH, &etag_str).body(()).unwrap(),
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
