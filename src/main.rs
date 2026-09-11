//! den-subtitles — a self-hosted Stremio subtitles addon for Den in one small binary:
//!
//! 1. FETCH — OpenSubtitles, hash-matched (the app sends the file's OSHash as `videoHash`), served
//!    from our own cache/origin so results are fast and dodge the per-IP download quota.
//! 2. TRANSLATE — BYOK LLM translation (gpt-4o-mini default) via a chunk→same-length→retry harness
//!    that survives a full film without losing sync between cue count and timing.
//! 3. SYNC — an auto-sync ladder (hash → reference-align → alass audio VAD) so subtitles line up.
//!
//! Both credentials (OpenSubtitles key, LLM key) are BYOK and ride in the install URL, Keychain-
//! stored by the app. No user credential lives in the environment.

mod addon;
mod cache;
mod config;
mod fetch;
mod httputil;
mod inflight;
mod logging;
mod metrics;
mod opensubtitles;
mod resync;
mod seal;
mod srt;
mod state;
mod sync;
mod translate;
mod userconfig;

use std::convert::Infallible;
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use hyper::header::{HeaderValue, ACCESS_CONTROL_ALLOW_ORIGIN};
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::server::graceful::GracefulShutdown;
use tokio::net::TcpListener;

use crate::config::Config;
use crate::httputil::Body;
use crate::state::AppState;

/// The /configure page, embedded so the binary is self-contained.
const CONFIGURE_PAGE: &str = include_str!("configure.html");

/// The manifest and /configure change only on a redeploy. The stale-while-revalidate window lets a
/// client answer from its copy while it re-checks, the same as every other den addon.
const STATIC_CACHE: &str = "public, max-age=3600, stale-while-revalidate=600";

/// The sealing key can rotate, so its freshness window is short; the ETag still busts a stale copy.
const KEY_CACHE: &str = "public, max-age=300";

/// How long a client gets to send a complete request head. Without it a socket that sends half a head
/// holds a connection task open indefinitely.
const HEADER_READ_TIMEOUT: Duration = Duration::from_secs(10);

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
    let started = Instant::now();
    let (parts, _body) = req.into_parts();
    let mut resp = match parts.method {
        // A CORS preflight is answered for any path, before routing.
        hyper::Method::OPTIONS => httputil::preflight(),
        // Honor conditional GET/HEAD: any cacheable 200 carries an ETag, so an `If-None-Match` hit
        // collapses to a 304.
        hyper::Method::GET | hyper::Method::HEAD => {
            let resp = httputil::apply_conditional(route(&state, &parts).await, &parts.headers);
            // HEAD must not carry a body (the router builds one regardless of method).
            if parts.method == hyper::Method::HEAD {
                httputil::strip_body(resp)
            } else {
                resp
            }
        }
        // Every route is a read. Routing on the path alone let a POST reach the same handlers — and a
        // subtitle or translate route spends a metered download or an LLM bill.
        _ => httputil::error(StatusCode::METHOD_NOT_ALLOWED, "method_not_allowed"),
    };
    // `total` closes the header on any response whose handler named its phases, in tenths of a
    // millisecond like every other den addon.
    if resp.headers().contains_key(httputil::SERVER_TIMING) {
        let total = format!("total;dur={:.1}", started.elapsed().as_secs_f64() * 1000.0);
        resp = httputil::add_timing(resp, &total);
    }
    // Every reply is readable from a browser-based Stremio client. Nothing here rides on a cookie —
    // an install's credentials are in its path — so a wildcard origin grants a page nothing it could
    // not already fetch. Added here, last, so the 304 and VTT paths that rebuild a response keep it.
    resp.headers_mut().insert(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
    // The debug headers readable too: a cross-origin fetch sees only the CORS-safelisted headers unless
    // Expose-Headers names more, and Resource Timing hides Server-Timing without Timing-Allow-Origin.
    resp.headers_mut()
        .insert("access-control-expose-headers", HeaderValue::from_static("Server-Timing, X-Den-Degraded"));
    resp.headers_mut().insert("timing-allow-origin", HeaderValue::from_static("*"));
    // Off by default, and then this bool is the whole cost. The path is redacted and the query left
    // out: a config segment is an install's credentials, and `?resync=` carries a stream URL. The
    // caller's X-Request-Id rides along so the line can be matched to the app's.
    if state.cfg.log_requests {
        eprintln!(
            "{}",
            logging::request_line(
                parts.method.as_str(),
                parts.uri.path(),
                resp.status().as_u16(),
                started.elapsed().as_millis(),
                logging::request_id(&parts.headers).as_deref(),
            )
        );
    }
    resp
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
        // Prometheus text behind `METRICS_TOKEN`; 404 without it (see metrics.rs).
        "/metrics" => return metrics::handle(state, &parts.headers),
        "/manifest.json" => return httputil::json(StatusCode::OK, &addon::manifest(false), STATIC_CACHE),
        "/" | "/configure" | "/configure/" => {
            return httputil::html(StatusCode::OK, CONFIGURE_PAGE, STATIC_CACHE)
        }
        // The current X25519 public key (base64) so /configure can seal the config to it; 404 when
        // sealed configs are disabled (no key) — the page then keeps plaintext (SEALED-CONFIG.md).
        // Both answers carry CONFIG_EPOCH, which the page stamps into every link it builds, sealed or
        // not: a link stamped below it would be refused the moment it was built.
        "/config-key" => {
            let epoch = state.cfg.revocation.epoch();
            return match state.config_keyring.as_ref().map(|kr| kr.current_pub_b64()) {
                Some(pub_b64) if !pub_b64.is_empty() => httputil::json(
                    StatusCode::OK,
                    &serde_json::json!({"key": pub_b64, "epoch": epoch}),
                    KEY_CACHE,
                ),
                _ => httputil::json(
                    StatusCode::NOT_FOUND,
                    &serde_json::json!({"error": "no_key", "epoch": epoch}),
                    "no-store",
                ),
            };
        }
        _ => {}
    }

    let segs = split_path(path);
    let config = segs.first().copied().unwrap_or("");
    let resource = segs.get(1).copied().unwrap_or("");

    match resource {
        "manifest.json" => match state.decode_config(config) {
            Some(_) => httputil::json(StatusCode::OK, &addon::manifest(true), STATIC_CACHE),
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
                _ => return httputil::not_found(),
            };
            let id = strip_json(id_seg).unwrap_or(id_seg);
            let extra = strip_json(extra).unwrap_or(extra);
            addon::handle_subtitles(state, &parts.headers, config, id, extra).await
        }
        "subtitle" => {
            // /<config>/subtitle/<file_id>.(srt|vtt)[?ref=<id>|?resync=<stream-url>][&lang=<code>]
            let file = segs.get(2).copied().unwrap_or("");
            let (stem, want_vtt) = split_subtitle_format(file);
            match stem.and_then(|n| n.parse::<i64>().ok()) {
                Some(file_id) => {
                    let query = parts.uri.query().unwrap_or("");
                    let ref_id = query_get(query, "ref").and_then(|v| v.parse().ok());
                    let resync = query_get(query, "resync");
                    let lang = query_get(query, "lang");
                    let resp =
                        addon::handle_subtitle_file(state, config, file_id, ref_id, resync, lang.as_deref())
                            .await;
                    if want_vtt {
                        httputil::to_vtt(resp, &parts.headers).await
                    } else {
                        resp
                    }
                }
                None => httputil::error(StatusCode::BAD_REQUEST, "bad_file_id"),
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
                _ => return httputil::not_found(),
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
                return httputil::not_found();
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
        _ => httputil::not_found(),
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
    // Registered before the process says it is up, so a stop that arrives from here on drains.
    let shutdown = shutdown_signal();
    // What this process is running with, minus anything secret: the key and the token appear only as
    // on/off, and the origin is one the addon hands to every client anyway. Same shape as every other
    // den addon's line.
    let on_off = |on: bool| if on { "on" } else { "off" };
    eprintln!(
        "den-subtitles {} listening on :{port} — metrics={} log_requests={} sealed={} revoked={} epoch={} \
         require_iid={} cache_dir={} cache_max_mib={} public_base={} resync={}",
        env!("CARGO_PKG_VERSION"),
        on_off(!state.cfg.metrics_token.is_empty()),
        on_off(state.cfg.log_requests),
        on_off(state.config_keyring.is_some()),
        state.cfg.revocation.revoked_count(),
        state.cfg.revocation.epoch(),
        on_off(state.cfg.revocation.requires_install_id()),
        state.cfg.cache_dir.display(),
        state.cfg.cache_max_bytes / (1024 * 1024),
        state.cfg.public_base_url.as_deref().unwrap_or("derived"),
        // Off says why a resync falls back to the raw sub: no SCOUT_ORIGINS.
        match state.cfg.scout_origins.is_empty() {
            true => "off".to_string(),
            false => state.cfg.scout_origins.iter().map(ToString::to_string).collect::<Vec<_>>().join(","),
        },
    );

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

    serve_until(listener, state, shutdown, DRAIN_GRACE).await;
    Ok(())
}

/// How long in-flight requests get to finish after SIGTERM. Under podman's default 10s stop timeout,
/// as den-atlas's and den-embed's are, so the drain works whether or not the Quadlet's --stop-timeout
/// has reached the box.
const DRAIN_GRACE: Duration = Duration::from_secs(8);

static ACCEPT_FAILED: logging::LogGate = logging::LogGate::new();

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
    let mut http = hyper::server::conn::http1::Builder::new();
    http.timer(TokioTimer::new()).header_read_timeout(HEADER_READ_TIMEOUT);
    tokio::pin!(shutdown);
    loop {
        let (stream, _) = tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok(pair) => pair,
                Err(e) => {
                    // Out of file descriptors fails every accept in a row; say so once a minute.
                    if ACCEPT_FAILED.allow() {
                        eprintln!("accept: {e}");
                    }
                    // And back off before trying again. The listener stays readable while the process is
                    // out of descriptors, so an immediate retry fails at once and the loop spins a core
                    // at 100% until one frees up — on the one runtime thread that serves every request.
                    tokio::time::sleep(Duration::from_millis(100)).await;
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
        let conn = http.serve_connection(TokioIo::new(stream), service);
        let conn = graceful.watch(conn);
        tokio::spawn(async move {
            let _ = conn.await;
        });
    }
    drop(listener);
    eprintln!("shutting down — draining in-flight requests");
    tokio::select! {
        _ = graceful.shutdown() => eprintln!("shut down cleanly"),
        _ = tokio::time::sleep(grace) => {
            eprintln!("drain deadline ({grace:?}) reached with requests still in flight");
        }
    }
}

/// Resolves on SIGTERM (a redeploy) or SIGINT (a terminal). The same handler den-atlas and den-embed
/// use.
fn shutdown_signal() -> impl Future<Output = ()> {
    use tokio::signal::unix::{signal, SignalKind};
    // Registered NOW, by the caller, not lazily when the future is first polled. Polling starts once
    // the server is already accepting, and until then SIGTERM keeps its default disposition — so a
    // stop arriving in that window killed the process outright rather than draining it.
    let term = signal(SignalKind::terminate());
    let int = signal(SignalKind::interrupt());
    async move {
        tokio::select! {
            _ = wait_for(term, "SIGTERM") => {}
            _ = wait_for(int, "SIGINT") => {}
        }
        // A SECOND signal ends it now. Both handles above are dropped by here, and tokio does not
        // restore the default disposition when a `Signal` drops — so every later SIGTERM and ^C would
        // be caught and discarded, and an operator could not get out of the drain short of SIGKILL.
        // Exit 0, because asking twice is a deliberate choice, not a failure.
        tokio::spawn(async move {
            tokio::select! {
                _ = quietly(signal(SignalKind::terminate())) => {}
                _ = quietly(signal(SignalKind::interrupt())) => {}
            }
            eprintln!("second signal — exiting without finishing the drain");
            std::process::exit(0);
        });
    }
}

/// Like `wait_for`, but says nothing — for a caller that prints its own, different message.
async fn quietly(registered: std::io::Result<tokio::signal::unix::Signal>) {
    match registered {
        Ok(mut sig) => {
            sig.recv().await;
        }
        Err(_) => std::future::pending::<()>().await,
    }
}

/// Resolve when this signal arrives, or never if it could not be registered — returning at once would
/// shut the server down the moment it started. The two signals are registered independently, so one
/// failing does not take the other with it.
async fn wait_for(registered: std::io::Result<tokio::signal::unix::Signal>, name: &str) {
    match registered {
        Ok(mut sig) => {
            sig.recv().await;
            eprintln!("{name} — draining in-flight requests");
        }
        Err(e) => {
            eprintln!("{name} handler unavailable ({e}); it will be a hard kill");
            std::future::pending::<()>().await;
        }
    }
}

fn main() {
    let cfg = Config::from_env();
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().expect("tokio runtime");

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
        test_state_with(config_key, "")
    }

    fn test_state_with(config_key: &str, metrics_token: &str) -> Arc<AppState> {
        AppState::new(test_config(config_key, metrics_token))
    }

    fn test_config(config_key: &str, metrics_token: &str) -> Config {
        Config {
            port: 0,
            cache_dir: std::env::temp_dir().join("den-subtitles-test-cache"),
            cache_max_bytes: 8 * 1024 * 1024,
            public_base_url: None,
            ffsubsync: "ffsubsync".to_string(),
            alass: "alass".to_string(),
            config_key: config_key.to_string(),
            config_keys_prev: String::new(),
            metrics_token: metrics_token.to_string(),
            log_requests: false,
            // Never the live API from a test: port 1 refuses instantly.
            os_api_base: "http://127.0.0.1:1".to_string(),
            scout_origins: Vec::new(),
            revocation: Default::default(),
        }
    }

    // Issue #8 R3: an install id as /configure mints it (bytes 0..16).
    const IID: &str = "AAECAwQFBgcICQoLDA0ODw";

    fn plain_segment(json: &str) -> String {
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(json)
    }

    /// A state refusing the installs in `revoked` and every link stamped below `epoch`, with a fixed
    /// public origin so the URLs it hands back can be requested again.
    fn revoking_state(revoked: &str, epoch: &str) -> Arc<AppState> {
        let mut cfg = test_config("", "");
        cfg.revocation = userconfig::Revocation::from_env(revoked, Some(epoch));
        cfg.public_base_url = Some("http://subs.test".to_string());
        AppState::new(cfg)
    }

    /// Every route that reads a config refuses a revoked install, with exactly the answer an
    /// undecodable segment gets on that route.
    #[tokio::test]
    async fn a_revoked_install_is_refused_on_every_config_route() {
        let seg = plain_segment(&format!(r#"{{"osKey":"os-test","iid":"{IID}"}}"#));
        let routes = [
            "manifest.json",
            "subtitles/movie/tt0000001.json",
            "subtitle/42.srt",
            "translate/movie/tt0000001/Swedish.json",
            "translate/movie/tt0000001/Swedish.srt",
            "translate/movie/tt0000001/Swedish.status",
        ];
        for route in routes {
            let revoked = handle_request(
                revoking_state(IID, "0"),
                request(hyper::Method::GET, &format!("/{seg}/{route}")),
            )
            .await;
            let garbage = handle_request(
                revoking_state(IID, "0"),
                request(hyper::Method::GET, &format!("/not-a-config/{route}")),
            )
            .await;
            assert_eq!(revoked.status(), StatusCode::BAD_REQUEST, "{route}");
            assert_eq!(revoked.status(), garbage.status(), "{route}");
            let body = body_string(revoked).await;
            assert_eq!(body, r#"{"error":"bad_config"}"#, "{route}");
            assert_eq!(body, body_string(garbage).await, "{route}");
        }
        // The same install, not revoked, is served.
        let resp = handle_request(
            revoking_state("", "0"),
            request(hyper::Method::GET, &format!("/{seg}/manifest.json")),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn an_install_stamped_below_config_epoch_is_refused() {
        for (json, want) in [
            (r#"{"osKey":"o","ep":1}"#, StatusCode::BAD_REQUEST),
            (r#"{"osKey":"o"}"#, StatusCode::BAD_REQUEST), // no epoch reads as 0
            (r#"{"osKey":"o","ep":2}"#, StatusCode::OK),
            (r#"{"osKey":"o","ep":3}"#, StatusCode::OK),
        ] {
            let uri = format!("/{}/manifest.json", plain_segment(json));
            let resp = handle_request(revoking_state("", "2"), request(hyper::Method::GET, &uri)).await;
            assert_eq!(resp.status(), want, "{json}");
        }
    }

    /// Links built before install ids existed carry neither field, and keep working until an epoch
    /// is raised.
    #[tokio::test]
    async fn a_config_without_an_install_id_works_at_epoch_zero() {
        let uri = format!("/{}/manifest.json", plain_segment(r#"{"osKey":"o"}"#));
        let resp = handle_request(revoking_state(IID, "0"), request(hyper::Method::GET, &uri)).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    /// The subtitle URLs the subtitles resource hands back embed the install's config, so a
    /// revocation made after they were issued still reaches them.
    #[tokio::test]
    async fn a_revoked_install_s_issued_subtitle_url_is_refused() {
        let seg = plain_segment(&format!(r#"{{"osKey":"os-test","iid":"{IID}"}}"#));
        let before = revoking_state("", "0");
        let sub = crate::opensubtitles::Subtitle {
            file_id: 4242,
            lang: "en".into(),
            hash_match: false,
            downloads: 1,
            release: "Some.Film.2020.1080p".into(),
            hd: true,
            fps: 0.0,
            from_trusted: false,
            machine_translated: false,
            ai_translated: false,
            ratings: 0.0,
        };
        let key = format!("{}tt0000093:0:0:", crate::cache::SEARCH_NS);
        before.cache.put(key, serde_json::to_string(&vec![sub]).unwrap(), Duration::from_secs(60));
        let list = handle_request(
            before,
            request(hyper::Method::GET, &format!("/{seg}/subtitles/movie/tt0000093.json")),
        )
        .await;
        let list: serde_json::Value = serde_json::from_str(&body_string(list).await).unwrap();
        let url = list["subtitles"][0]["url"].as_str().expect("a subtitle url").to_string();
        let path = url.strip_prefix("http://subs.test").expect("the fixed public origin");
        assert!(path.starts_with(&format!("/{seg}/subtitle/4242.srt")), "{url}");

        let resp = handle_request(revoking_state(IID, "0"), request(hyper::Method::GET, path)).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(body_string(resp).await, r#"{"error":"bad_config"}"#);
    }

    /// /configure stamps the epoch into every link, sealed or not, so both answers carry it.
    #[tokio::test]
    async fn config_key_carries_the_config_epoch() {
        for (key, want) in [(VEC_PRIV_B64, StatusCode::OK), ("", StatusCode::NOT_FOUND)] {
            let mut cfg = test_config(key, "");
            cfg.revocation = userconfig::Revocation::from_env("", Some("4"));
            let resp = handle_request(AppState::new(cfg), request(hyper::Method::GET, "/config-key")).await;
            assert_eq!(resp.status(), want);
            let body: serde_json::Value = serde_json::from_str(&body_string(resp).await).unwrap();
            assert_eq!(body["epoch"], 4, "{body}");
        }
    }

    /// Every link the page builds names its install and the epoch it was minted in, and both are
    /// inside what gets sealed.
    #[tokio::test]
    async fn configure_page_mints_an_install_id_and_stamps_the_epoch() {
        let page =
            body_string(handle_request(test_state(""), request(hyper::Method::GET, "/configure")).await)
                .await;
        assert!(
            page.contains("crypto.getRandomValues(bytes)"),
            "the install id is not minted from the CSPRNG"
        );
        assert!(page.contains("iid: mintInstallId(), ep: configEpoch"), "the link is not stamped");
        assert!(page.contains("toSegment(install)"), "the stamped config is not what gets sealed");
        assert!(page.contains("configEpoch = j.epoch"), "the epoch is not read from /config-key");
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
    async fn a_plaintext_config_is_refused_while_sealing_is_on() {
        // Every install /configure issues is sealed once a key is set, and /config-key is public: a
        // plaintext segment is one anyone could have minted.
        let seg = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(r#"{"osKey":"os-plain"}"#);
        let uri = format!("/{seg}/manifest.json");
        let resp =
            handle_request(test_state(VEC_PRIV_B64), Request::builder().uri(uri).body(()).unwrap()).await;
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "a plaintext config must be refused with a key set"
        );
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

    /// A plaintext config segment carrying only an OpenSubtitles key, enough for `handle_subtitles`.
    fn os_only_segment() -> String {
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(r#"{"osKey":"os-test"}"#)
    }

    fn server_timing(resp: &Response<Body>) -> String {
        resp.headers().get(httputil::SERVER_TIMING).expect("Server-Timing").to_str().unwrap().to_string()
    }

    /// An empty list because the search failed must not read like a title with no subtitles.
    #[tokio::test]
    async fn a_failed_search_is_marked_degraded_and_timed() {
        let uri = format!("/{}/subtitles/movie/tt0000001.json", os_only_segment());
        let resp = handle_request(test_state(""), request(hyper::Method::GET, &uri)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers().get(httputil::X_DEN_DEGRADED).unwrap(), "upstream_unavailable");
        let timing = server_timing(&resp);
        assert!(timing.starts_with("opensubtitles;dur="), "{timing}");
        assert!(timing.contains(", total;dur="), "{timing}");
        assert_eq!(body_string(resp).await, r#"{"subtitles":[]}"#);
    }

    #[tokio::test]
    async fn a_cached_search_is_timed_as_a_hit_and_not_degraded() {
        let state = test_state("");
        let key = format!("{}tt0000002:0:0:", crate::cache::SEARCH_NS);
        state.cache.put(key, "[]".into(), Duration::from_secs(60));
        let uri = format!("/{}/subtitles/movie/tt0000002.json", os_only_segment());
        let resp = handle_request(state, request(hyper::Method::GET, &uri)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp.headers().get(httputil::X_DEN_DEGRADED).is_none(), "a normal answer is not degraded");
        let timing = server_timing(&resp);
        assert!(timing.starts_with("cache;desc=hit, total;dur="), "{timing}");
    }

    /// /health changing state is logged once each way, not once per search.
    #[test]
    fn a_health_flip_is_reported_once_each_way() {
        let state = test_state("");
        for _ in 1..HEALTH_FAIL_THRESHOLD {
            assert!(!state.search_failed(), "still ok below the threshold");
        }
        assert!(state.search_failed(), "the failure that crosses the threshold flips /health");
        assert!(!state.search_failed(), "staying degraded is not a change");
        assert!(state.search_succeeded(), "the first success recovers /health");
        assert!(!state.search_succeeded(), "staying ok is not a change");
        assert!(!state.search_failed(), "one failure after a recovery does not flip it back");
    }

    fn metrics_request(auth: Option<&str>) -> Request<()> {
        let mut b = Request::builder().uri("/metrics");
        if let Some(a) = auth {
            b = b.header(hyper::header::AUTHORIZATION, a);
        }
        b.body(()).unwrap()
    }

    /// Unset means off: not an empty 200, and not a route an empty bearer header can open.
    #[tokio::test]
    async fn metrics_is_404_without_a_configured_token() {
        for auth in [None, Some("Bearer "), Some("Bearer anything")] {
            let resp = handle_request(test_state(""), metrics_request(auth)).await;
            assert_eq!(
                resp.status(),
                StatusCode::NOT_FOUND,
                "auth {auth:?} reached an unconfigured /metrics"
            );
        }
    }

    #[tokio::test]
    async fn metrics_is_404_for_a_wrong_or_missing_token() {
        for auth in [None, Some("Bearer wrong"), Some("Bearer s3cre"), Some("s3cret")] {
            let resp = handle_request(test_state_with("", "s3cret"), metrics_request(auth)).await;
            assert_eq!(resp.status(), StatusCode::NOT_FOUND, "auth {auth:?} was let in");
        }
    }

    fn request(method: hyper::Method, uri: &str) -> Request<()> {
        Request::builder().method(method).uri(uri).body(()).unwrap()
    }

    #[tokio::test]
    async fn options_is_a_preflight_on_any_path() {
        use hyper::header::{
            ACCESS_CONTROL_ALLOW_HEADERS, ACCESS_CONTROL_ALLOW_METHODS, ACCESS_CONTROL_MAX_AGE,
        };
        for uri in ["/manifest.json", "/no/such/path", "/cfg/subtitle/1.srt"] {
            let resp = handle_request(test_state(""), request(hyper::Method::OPTIONS, uri)).await;
            assert_eq!(resp.status(), StatusCode::NO_CONTENT, "{uri}");
            let h = resp.headers();
            assert_eq!(h.get(ACCESS_CONTROL_ALLOW_ORIGIN).unwrap(), "*", "{uri}");
            assert_eq!(h.get(ACCESS_CONTROL_ALLOW_METHODS).unwrap(), "GET, HEAD, OPTIONS");
            assert_eq!(h.get(ACCESS_CONTROL_ALLOW_HEADERS).unwrap(), "*");
            assert_eq!(h.get(ACCESS_CONTROL_MAX_AGE).unwrap(), "86400");
            assert!(body_string(resp).await.is_empty());
        }
    }

    /// The header is added after every path that rebuilds a response — a 304, a HEAD, an error —
    /// so none of them can drop it.
    #[tokio::test]
    async fn every_reply_carries_the_wildcard_origin() {
        use hyper::header::{ETAG, IF_NONE_MATCH};
        let ok = handle_request(test_state(""), request(hyper::Method::GET, "/manifest.json")).await;
        assert_eq!(ok.status(), StatusCode::OK);
        assert_eq!(ok.headers().get(ACCESS_CONTROL_ALLOW_ORIGIN).unwrap(), "*");
        let etag = ok.headers().get(ETAG).unwrap().clone();

        let not_modified = handle_request(
            test_state(""),
            Request::builder().uri("/manifest.json").header(IF_NONE_MATCH, etag).body(()).unwrap(),
        )
        .await;
        assert_eq!(not_modified.status(), StatusCode::NOT_MODIFIED);

        let head = handle_request(test_state(""), request(hyper::Method::HEAD, "/health")).await;
        let missing = handle_request(test_state(""), request(hyper::Method::GET, "/no/such/path")).await;
        for resp in [not_modified, head, missing] {
            assert_eq!(resp.headers().get(ACCESS_CONTROL_ALLOW_ORIGIN).unwrap(), "*", "{}", resp.status());
        }
    }

    /// An unknown path, a known resource with the wrong shape, and a refused /metrics all give the
    /// same JSON 404.
    #[tokio::test]
    async fn an_unknown_path_is_a_json_404() {
        for uri in
            ["/no/such/path", "/cfg/subtitles/movie", "/cfg/translate/movie/tt1/Swedish.txt", "/metrics"]
        {
            let resp = handle_request(test_state(""), request(hyper::Method::GET, uri)).await;
            assert_eq!(resp.status(), StatusCode::NOT_FOUND, "{uri}");
            assert_eq!(resp.headers().get(hyper::header::CONTENT_TYPE).unwrap(), "application/json", "{uri}");
            assert_eq!(resp.headers().get(hyper::header::CACHE_CONTROL).unwrap(), "no-store", "{uri}");
            assert_eq!(body_string(resp).await, r#"{"error":"not_found"}"#, "{uri}");
        }
    }

    #[tokio::test]
    async fn metrics_serves_prometheus_text_for_the_right_token() {
        let resp =
            handle_request(test_state_with("", "s3cret"), metrics_request(Some("Bearer s3cret"))).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get(hyper::header::CONTENT_TYPE).unwrap(),
            "text/plain; version=0.0.4; charset=utf-8"
        );
        let body = body_string(resp).await;
        let build = format!("subtitles_build_info{{version=\"{}\"}} 1\n", env!("CARGO_PKG_VERSION"));
        assert!(body.contains(&build), "no build_info line in:\n{body}");
        assert!(body.contains("\nsubtitles_consecutive_failures{kind=\"opensubtitles\"} 0\n"), "{body}");
        assert!(body.contains("# TYPE subtitles_cache_disk_write_failures_total counter\n"), "{body}");
    }

    /// Every route is a read; anything else is refused before it can reach a handler that spends a
    /// metered download or an LLM bill.
    #[tokio::test]
    async fn a_write_method_is_a_json_405() {
        for method in [hyper::Method::POST, hyper::Method::PUT, hyper::Method::DELETE] {
            let resp = handle_request(test_state(""), request(method.clone(), "/manifest.json")).await;
            assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED, "{method}");
            assert_eq!(resp.headers().get(hyper::header::CACHE_CONTROL).unwrap(), "no-store", "{method}");
            assert_eq!(resp.headers().get(ACCESS_CONTROL_ALLOW_ORIGIN).unwrap(), "*", "{method}");
            assert_eq!(body_string(resp).await, r#"{"error":"method_not_allowed"}"#, "{method}");
        }
    }

    /// An error is the fleet's JSON shape, not a text body.
    #[tokio::test]
    async fn an_error_reply_is_json() {
        let resp = handle_request(test_state(""), request(hyper::Method::GET, "/cfg/subtitle/abc.srt")).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(resp.headers().get(hyper::header::CONTENT_TYPE).unwrap(), "application/json");
        assert_eq!(resp.headers().get(hyper::header::CACHE_CONTROL).unwrap(), "no-store");
        assert_eq!(body_string(resp).await, r#"{"error":"bad_file_id"}"#);
    }

    #[tokio::test]
    async fn static_routes_carry_the_fleet_cache_policy() {
        let seg = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(r#"{"osKey":"os-test"}"#);
        // The configured manifest runs with sealing off: with it on, only a sealed segment is an install.
        let cases = [
            ("/manifest.json".to_string(), STATIC_CACHE, VEC_PRIV_B64),
            (format!("/{seg}/manifest.json"), STATIC_CACHE, ""),
            ("/configure".to_string(), STATIC_CACHE, VEC_PRIV_B64),
            ("/config-key".to_string(), KEY_CACHE, VEC_PRIV_B64),
        ];
        for (uri, want, key) in cases {
            let resp = handle_request(test_state(key), request(hyper::Method::GET, &uri)).await;
            assert_eq!(resp.status(), StatusCode::OK, "{uri}");
            assert_eq!(resp.headers().get(hyper::header::CACHE_CONTROL).unwrap(), want, "{uri}");
        }
        assert_eq!(STATIC_CACHE, "public, max-age=3600, stale-while-revalidate=600");
        assert_eq!(KEY_CACHE, "public, max-age=300");
    }

    /// Tenths of a millisecond, the format every den addon's `total` uses.
    #[tokio::test]
    async fn server_timing_total_has_tenths_of_a_millisecond() {
        let uri = format!("/{}/subtitles/movie/tt0000001.json", os_only_segment());
        let resp = handle_request(test_state(""), request(hyper::Method::GET, &uri)).await;
        let timing = server_timing(&resp);
        let total = timing.rsplit("total;dur=").next().unwrap();
        assert!(total.contains('.') && total.split('.').nth(1).unwrap().len() == 1, "{timing}");
    }
}
