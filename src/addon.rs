//! Addon request path: the Stremio `subtitles` resource plus the Den-specific translate/serve
//! endpoints.
//!
//!   GET /manifest.json                                  unconfigured manifest
//!   GET /<config>/manifest.json                         configured manifest
//!   GET /<config>/subtitles/<type>/<id>/<extra>.json    native subs (OpenSubtitles, hash-matched)
//!   GET /<config>/subtitle/<file_id>.srt                proxy+cache one OpenSubtitles file
//!   GET /<config>/translate/<type>/<id>/<lang>.json     app-driven: kick off/await a translation → { url }
//!   GET /<config>/translate/<type>/<id>/<lang>.srt      the translated SRT (cache hit after the .json warmed it)
//!
//! `<id>` is `tt<digits>` or `tt<digits>:<season>:<episode>`. `<extra>` is the Stremio query blob
//! carrying `videoHash`/`videoSize` (the OSHash the app computed).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use hyper::header::HeaderMap;
use hyper::{Response, StatusCode};
use serde_json::{json, Value};

use crate::httputil::{self, percent_decode, Body};
use crate::opensubtitles;
use crate::state::AppState;
use crate::userconfig::{self, LlmConfig, UserConfig};
use crate::{srt, translate};

const CACHE_TTL: Duration = Duration::from_secs(60 * 60 * 24 * 60); // 60 days — mirrors the app cache
// Search results turn over as new subs are uploaded, so a short TTL — enough to spare repeated
// round-trips when the app reopens a title, not so long that fresh uploads stay hidden.
const SEARCH_TTL: Duration = Duration::from_secs(60 * 60 * 6); // 6 hours

/// Monotonic counter making sync scratch-file names unique per invocation.
static SYNC_SEQ: AtomicU64 = AtomicU64::new(0);

pub fn manifest(configured: bool) -> Value {
    json!({
        "id": "fi.oxy.den-subtitles",
        "version": "0.1.0",
        "name": "Den Subtitles",
        "description": if configured {
            "OpenSubtitles (hash-matched + auto-synced) with optional BYOK AI translation for Den."
        } else {
            "Self-hosted subtitles for Den — configure with your OpenSubtitles key (AI translation optional)."
        },
        "resources": ["subtitles"],
        "types": ["movie", "series"],
        "idPrefixes": ["tt"],
        "catalogs": [],
        "behaviorHints": { "configurable": true, "configurationRequired": !configured },
    })
}

/// The base URL this server is reachable at (for the `url` we hand back to the client/app).
pub fn self_base(state: &AppState, headers: &HeaderMap, config: &str) -> String {
    let root = if let Some(b) = &state.cfg.public_base_url {
        b.trim_end_matches('/').to_string()
    } else {
        let hdr = |name: &str| headers.get(name).and_then(|v| v.to_str().ok());
        let proto = hdr("x-forwarded-proto")
            .map(|p| p.split(',').next().unwrap_or("http").trim().to_string())
            .unwrap_or_else(|| "http".to_string());
        // Reflecting a client-controlled Host into a URL we hand back is a poisoning vector; only
        // accept a sane host charset, else fall back. Set PUBLIC_BASE_URL in deploy to avoid this
        // path entirely (see .env.example).
        let host = hdr("x-forwarded-host")
            .or_else(|| hdr("host"))
            .filter(|h| is_sane_host(h))
            .unwrap_or("localhost")
            .to_string();
        format!("{proto}://{host}")
    };
    format!("{root}/{config}")
}

/// A hostname/authority we're willing to reflect into a returned URL: letters, digits, and the few
/// punctuation chars a host+port legitimately uses. Rejects spaces, slashes, `@`, etc.
fn is_sane_host(h: &str) -> bool {
    !h.is_empty()
        && h.len() <= 255
        && h.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b':' | b'_'))
}

/// `tt123` or `tt123:1:2` → (imdb, season, episode).
fn parse_id(id: &str) -> Option<(String, Option<i64>, Option<i64>)> {
    let mut parts = id.split(':');
    let imdb = parts.next()?;
    if !imdb.starts_with("tt") || imdb.len() < 3 {
        return None;
    }
    let season = parts.next().and_then(|s| s.parse().ok());
    let episode = parts.next().and_then(|s| s.parse().ok());
    Some((imdb.to_string(), season, episode))
}

/// Pull a `key=value` field out of the Stremio extra-args blob (already `.json`-stripped). The
/// client sends `videoHash`, `videoSize`, and `filename` here.
fn extra_field(extra: &str, key: &str) -> Option<String> {
    let decoded = percent_decode(extra);
    let prefix = format!("{key}=");
    for pair in decoded.split('&') {
        if let Some(v) = pair.strip_prefix(&prefix) {
            if !v.is_empty() {
                return Some(v.to_string());
            }
        }
    }
    None
}

/// GET /<config>/subtitles/<type>/<id>/<extra>.json — native OpenSubtitles list. Each `url` points
/// at our own `/subtitle/<file_id>.srt` proxy so results are cached and served from our origin
/// (dodging the per-IP download quota). Hash matches are floated to the top by `search`.
pub async fn handle_subtitles(
    state: &Arc<AppState>,
    headers: &HeaderMap,
    config: &str,
    id: &str,
    extra: &str,
) -> Response<Body> {
    let Some(cfg) = userconfig::decode(config) else {
        return httputil::json(StatusCode::BAD_REQUEST, &json!({"error": "bad_config"}), "no-store");
    };
    let Some((imdb, season, episode)) = parse_id(id) else {
        return httputil::json(StatusCode::BAD_REQUEST, &json!({"error": "bad_id"}), "no-store");
    };
    let Some(http) = state.http.as_ref() else {
        eprintln!("subtitles: http client unavailable");
        return httputil::json(StatusCode::OK, &json!({"subtitles": []}), "no-store");
    };

    // Cache the search result itself (config-independent: file_ids/langs are the same for everyone),
    // keyed by the query params incl. the file hash. Short TTL — new subs get uploaded — but enough
    // to spare a live round-trip every time the app reopens a title. Ranking is filename-specific, so
    // it is NOT baked into the cached list — we rank per request below.
    let hash = extra_field(extra, "videoHash");
    let filename = extra_field(extra, "filename");
    let search_key = format!(
        "search:{imdb}:{}:{}:{}",
        season.unwrap_or(0),
        episode.unwrap_or(0),
        hash.as_deref().unwrap_or("")
    );
    let mut subs: Vec<opensubtitles::Subtitle> = if let Some(hit) = state
        .cache
        .get(&search_key)
        .and_then(|h| serde_json::from_str(&h).ok())
    {
        hit
    } else {
        let client = opensubtitles::Client {
            http,
            api_key: &cfg.opensubtitles_key,
            token: cfg.opensubtitles_token.as_deref(),
        };
        // Ask for everything; the app filters/selects by its own preferred-language rules.
        match client.search(&imdb, season, episode, "all", hash.as_deref()).await {
            Ok(s) => {
                state.os_fails.store(0, std::sync::atomic::Ordering::Relaxed);
                if let Ok(json) = serde_json::to_string(&s) {
                    state.cache.put(search_key, json, SEARCH_TTL);
                }
                s
            }
            // Empty-200 is the correct Stremio shape for "nothing", but log the cause (our error
            // strings carry no key) and count it so /health can report `degraded` (ADDON-02).
            Err(e) => {
                eprintln!("subtitles: opensubtitles search failed for {imdb}: {e}");
                state.os_fails.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return httputil::json(StatusCode::OK, &json!({"subtitles": []}), "no-store");
            }
        }
    };

    // Rank for THIS stream: hash-match, then release/filename fit, then quality; grouped by language
    // best-first. Machine/AI subs sink to the bottom.
    opensubtitles::rank(&mut subs, filename.as_deref());

    let base = self_base(state, headers, config);
    // Tier 1 auto-sync reference: the (any-language) hash-matched sub is authored against this exact
    // file, so it's a trusted timing reference for aligning the non-hash-matched subs.
    let reference = subs.iter().find(|s| s.hash_match).map(|s| s.file_id);
    let out: Vec<Value> = subs
        .iter()
        .map(|s| {
            // Non-hash subs get `?ref=<id>` so the proxy reference-aligns them on fetch; a hash match
            // is already in sync and served as-is.
            let mut url = format!("{base}/subtitle/{}.srt", s.file_id);
            if !s.hash_match {
                if let Some(ref_id) = reference.filter(|&r| r != s.file_id) {
                    url = format!("{url}?ref={ref_id}");
                }
            }
            // Standard Stremio fields (id/url/lang) plus Den-specific detail the app renders in the
            // subtitle picker; a generic client ignores the unknown fields.
            json!({
                "id": format!("os-{}", s.file_id),
                "url": url,
                "lang": s.lang,
                "release": s.release,
                "hd": s.hd,
                "fps": s.fps,
                "hashMatch": s.hash_match,
                "trusted": s.from_trusted,
                "downloads": s.downloads,
                "machineTranslated": s.machine_translated,
                "aiTranslated": s.ai_translated,
            })
        })
        .collect();
    httputil::json(StatusCode::OK, &json!({"subtitles": out}), "public, max-age=3600")
}

/// GET /<config>/subtitle/<file_id>.srt[?ref=<id>|?resync=<url>] — download one OpenSubtitles file,
/// optionally auto-sync it (Tier 1 against a reference sub, or Tier 2 against the stream audio),
/// cache, serve. Any sync failure falls back to the raw sub — a slightly-off sub beats none.
pub async fn handle_subtitle_file(
    state: &Arc<AppState>,
    config: &str,
    file_id: i64,
    ref_id: Option<i64>,
    resync_url: Option<String>,
) -> Response<Body> {
    let Some(cfg) = userconfig::decode(config) else {
        return httputil::text(StatusCode::BAD_REQUEST, "bad_config");
    };
    // Cache identity depends on the sync mode so the raw and aligned variants don't collide.
    let cache_key = match (&resync_url, ref_id) {
        (Some(url), _) => format!("os:{file_id}:resync:{}", short_hash(url)),
        (None, Some(r)) => format!("os:{file_id}:ref:{r}"),
        (None, None) => format!("os:{file_id}"),
    };
    if let Some(hit) = state.cache.get(&cache_key) {
        return httputil::srt(hit);
    }
    let Some(http) = state.http.as_ref() else {
        return httputil::text(StatusCode::SERVICE_UNAVAILABLE, "subtitle service unavailable");
    };
    let client = opensubtitles::Client {
        http,
        api_key: &cfg.opensubtitles_key,
        token: cfg.opensubtitles_token.as_deref(),
    };

    let target = match subtitle_srt(state, &client, file_id).await {
        Ok(body) => body,
        Err(e) => {
            eprintln!("subtitle: download of file {file_id} failed: {e}");
            return httputil::text(StatusCode::BAD_GATEWAY, "upstream subtitle fetch failed");
        }
    };

    // Per-invocation unique temp tag: two concurrent requests for the same file must not share
    // scratch paths (one would read the other's half-written output and cache it for 60 days).
    let tag = format!("{}-{}", cache_key.replace(':', "-"), SYNC_SEQ.fetch_add(1, Ordering::Relaxed));
    let synced: Option<String> = if let Some(url) = resync_url.filter(|u| is_safe_resync_url(u)) {
        // Tier 2 — audio VAD against the playing stream (opt-in; alass pulls the audio via ffmpeg).
        match state.sync.sync_to_audio(&target, &url, &tag).await {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("subtitle: resync {file_id} failed: {e}");
                None
            }
        }
    } else if let Some(r) = ref_id {
        // Tier 1 — reference-align against the hash-matched sub (no audio needed).
        match subtitle_srt(state, &client, r).await {
            Ok(reference) => state.sync.sync_to_reference(&target, &reference, &tag).await.ok(),
            Err(_) => None,
        }
    } else {
        None
    };

    let body = synced.unwrap_or(target);
    state.cache.put(cache_key, body.clone(), CACHE_TTL);
    httputil::srt(body)
}

/// Fetch a subtitle's SRT, cached by file id (the raw, un-synced text — reused as a sync input).
async fn subtitle_srt(state: &Arc<AppState>, client: &opensubtitles::Client<'_>, file_id: i64) -> Result<String, String> {
    let key = format!("os:{file_id}");
    if let Some(hit) = state.cache.get(&key) {
        return Ok(hit);
    }
    let body = client.download(file_id).await?;
    state.cache.put(key, body.clone(), CACHE_TTL);
    Ok(body)
}

fn is_http_url(u: &str) -> bool {
    u.starts_with("http://") || u.starts_with("https://")
}

/// A resync target we're willing to fetch server-side (SSRF guard). The stream lives on the LAN
/// (den-scout on a private IP), so we can't blanket-deny private ranges — but we DO deny loopback
/// and link-local, which blocks the cloud-metadata endpoint (169.254.169.254) and localhost probes
/// while still allowing the user's own 192.168/10/172.16 stream host.
fn is_safe_resync_url(u: &str) -> bool {
    if !is_http_url(u) {
        return false;
    }
    let Some((_, rest)) = u.split_once("://") else { return false };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or(rest);
    let host = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    // Strip a :port (naive; fine for the hostnames/IPv4 we see — bracketed IPv6 handled below).
    let host = host.trim_start_matches('[');
    let host = host.split([']', ':']).next().unwrap_or(host);
    let host_lc = host.to_ascii_lowercase();
    if host_lc == "localhost" || host_lc.ends_with(".localhost") {
        return false;
    }
    match host_lc.parse::<std::net::IpAddr>() {
        Ok(std::net::IpAddr::V4(v4)) => !(v4.is_loopback() || v4.is_link_local()),
        Ok(std::net::IpAddr::V6(v6)) => !v6.is_loopback(),
        Err(_) => true, // a hostname (not a literal IP) — allowed; we don't resolve it here
    }
}

fn short_hash(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

/// GET /<config>/translate/<type>/<id>/<lang>.(json|srt) — the app-driven translation flow. The app
/// calls the `.json` form (showing its own "Translating…" wait), which does the work and returns the
/// `.srt` URL; it then hands that URL to the engine, which fetches the now-cached `.srt` instantly.
pub async fn handle_translate(
    state: &Arc<AppState>,
    headers: &HeaderMap,
    config: &str,
    id: &str,
    lang: &str,
    want_json: bool,
) -> Response<Body> {
    let Some(cfg) = userconfig::decode(config) else {
        return httputil::text(StatusCode::BAD_REQUEST, "bad_config");
    };
    // Translation needs the (optional) LLM credential — a subtitles-only install has none.
    let Some(llm) = &cfg.llm else {
        return httputil::text(StatusCode::BAD_REQUEST, "no AI provider configured for translation");
    };
    let Some((imdb, season, episode)) = parse_id(id) else {
        return httputil::text(StatusCode::BAD_REQUEST, "bad_id");
    };
    let cache_key = format!(
        "translate:{imdb}:{}:{}:{lang}:{}:{}",
        season.unwrap_or(0),
        episode.unwrap_or(0),
        provider_tag(llm),
        llm.model,
    );

    // Warm the cache if needed (both the .json and .srt forms share it).
    if state.cache.get(&cache_key).is_none() {
        if let Err(e) = produce_translation(state, &cfg, llm, &imdb, season, episode, lang, &cache_key).await {
            // Log the detail (no key in these strings); hand the client a generic message rather than
            // echoing a raw upstream error body.
            eprintln!("translate: {imdb} → {lang} failed: {e}");
            return httputil::text(StatusCode::BAD_GATEWAY, "translation failed");
        }
    }

    if want_json {
        let base = self_base(state, headers, config);
        let url = format!("{base}/translate/{}/{}/{}.srt", type_of(season), id, lang);
        return httputil::json(StatusCode::OK, &json!({ "url": url }), "no-store");
    }
    match state.cache.get(&cache_key) {
        Some(body) => httputil::srt(body),
        None => httputil::text(StatusCode::NOT_FOUND, "not translated"),
    }
}

/// Fetch a source subtitle (prefer English), translate it, and store the result in the cache.
#[allow(clippy::too_many_arguments)]
async fn produce_translation(
    state: &Arc<AppState>,
    cfg: &UserConfig,
    llm: &LlmConfig,
    imdb: &str,
    season: Option<i64>,
    episode: Option<i64>,
    lang: &str,
    cache_key: &str,
) -> Result<(), String> {
    let http = state.http.as_ref().ok_or("http client unavailable")?;
    let client = opensubtitles::Client {
        http,
        api_key: &cfg.opensubtitles_key,
        token: cfg.opensubtitles_token.as_deref(),
    };
    // Prefer an English source (best-resourced), else whatever exists.
    let subs = client.search(imdb, season, episode, "en", None).await?;
    let source = opensubtitles::best_for(&subs, "en")
        .or_else(|| subs.first())
        .ok_or("no source subtitle to translate")?;
    let raw = client.download(source.file_id).await?;
    let cues = srt::parse(&raw);
    if cues.is_empty() {
        return Err("source subtitle was empty".into());
    }
    let translated = translate::translate(http, llm, &cues, lang).await?;
    state.cache.put(cache_key.to_string(), srt::serialize(&translated), CACHE_TTL);
    Ok(())
}

fn provider_tag(llm: &LlmConfig) -> &'static str {
    use crate::userconfig::Provider::*;
    match llm.provider {
        OpenAI => "openai",
        Anthropic => "anthropic",
        Google => "google",
        Xai => "xai",
        OpenRouter => "openrouter",
        DeepL => "deepl",
    }
}

fn type_of(season: Option<i64>) -> &'static str {
    if season.is_some() {
        "series"
    } else {
        "movie"
    }
}
