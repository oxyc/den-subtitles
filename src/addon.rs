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
    let Some(cfg) = userconfig::decode(state.config_keyring.as_ref(), config) else {
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
    // Tier-1 auto-sync anchor: any hash-matched sub is a trusted, already-in-sync timing reference
    // for the rest. `None` when auto-sync is off for this install, or when the search returned no
    // hash match at all — the out-of-sync gap Tier-1 can't close (see `tier1_reference`); either way
    // those subs are served as-is and only the Tier-2 audio resync (a user action) can fix them.
    let reference = if cfg.auto_sync { tier1_reference(&subs) } else { None };
    let out: Vec<Value> = subs
        .iter()
        .map(|s| {
            // A sub that needs Tier-1 gets `?ref=<id>` so the proxy reference-aligns it on fetch; a
            // hash match (or anything, when there's no anchor) is served as-is.
            let mut url = format!("{base}/subtitle/{}.srt", s.file_id);
            if let Some(ref_id) = tier1_ref_for(s, reference) {
                url = format!("{url}?ref={ref_id}");
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

/// The `file_id` of a trusted timing reference for Tier-1 reference alignment: any hash-matched sub,
/// authored against this exact encode, so its timing is correct by construction (language doesn't
/// matter — timing is language-independent).
///
/// `None` when the search produced no hash match at all. That is the gap behind the out-of-sync
/// complaint: with no trusted anchor we deliberately do NOT align to an untrusted sub (that could
/// make timing worse), so those subs are served as-is and only the Tier-2 audio resync — a user
/// action, see `is_safe_resync_url` — can fix them.
fn tier1_reference(subs: &[opensubtitles::Subtitle]) -> Option<i64> {
    subs.iter().find(|s| s.hash_match).map(|s| s.file_id)
}

/// The reference `s` should be Tier-1 aligned against on fetch (`?ref=`), or `None` when it needs no
/// alignment: a hash match is already in sync, a sub can't be aligned to itself, and with no anchor
/// nothing is aligned.
fn tier1_ref_for(s: &opensubtitles::Subtitle, reference: Option<i64>) -> Option<i64> {
    if s.hash_match {
        return None;
    }
    reference.filter(|&r| r != s.file_id)
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
    let Some(cfg) = userconfig::decode(state.config_keyring.as_ref(), config) else {
        return httputil::text(StatusCode::BAD_REQUEST, "bad_config");
    };
    // Cache identity depends on the sync mode so the raw and aligned variants don't collide.
    let cache_key = sync_cache_key(file_id, &resync_url, ref_id);
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
    // Unwrap the host from an optional `[IPv6]:port` / `host:port`. A bracketed literal must be read
    // to its closing `]` (an IPv6 address is full of colons); only a bare host/IPv4 splits on `:`.
    let host = if let Some(rest) = host.strip_prefix('[') {
        rest.split(']').next().unwrap_or(rest)
    } else {
        host.split(':').next().unwrap_or(host)
    };
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

/// Cache key for one proxied subtitle, namespaced by sync mode so the raw / reference-aligned /
/// audio-resynced variants of the same `file_id` never collide (else an aligned sub would be served
/// from the raw entry, or vice versa). A resync URL takes precedence over a `ref` — it's the
/// stronger Tier-2 correction — mirroring the dispatch in `handle_subtitle_file`.
fn sync_cache_key(file_id: i64, resync_url: &Option<String>, ref_id: Option<i64>) -> String {
    match (resync_url, ref_id) {
        (Some(url), _) => format!("os:{file_id}:resync:{}", short_hash(url)),
        (None, Some(r)) => format!("os:{file_id}:ref:{r}"),
        (None, None) => format!("os:{file_id}"),
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
    let Some(cfg) = userconfig::decode(state.config_keyring.as_ref(), config) else {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opensubtitles::Subtitle;

    fn sub(file_id: i64, hash_match: bool) -> Subtitle {
        Subtitle {
            file_id,
            lang: "en".into(),
            hash_match,
            downloads: 0,
            release: String::new(),
            hd: false,
            fps: 0.0,
            from_trusted: false,
            machine_translated: false,
            ai_translated: false,
            ratings: 0.0,
        }
    }

    #[test]
    fn tier1_reference_is_the_hash_match_or_none() {
        // No hash match anywhere → no anchor. This is the out-of-sync gap: Tier-1 can't run.
        assert_eq!(tier1_reference(&[sub(1, false), sub(2, false)]), None);
        // A hash match becomes the anchor (the first one encountered).
        assert_eq!(tier1_reference(&[sub(1, false), sub(2, true), sub(3, true)]), Some(2));
    }

    #[test]
    fn tier1_ref_for_skips_hash_matches_self_and_missing_anchor() {
        let anchor = Some(2);
        // A non-hash sub aligns to the anchor.
        assert_eq!(tier1_ref_for(&sub(1, false), anchor), Some(2));
        // The hash-matched anchor is already in sync — no alignment.
        assert_eq!(tier1_ref_for(&sub(2, true), anchor), None);
        // A sub is never aligned to itself (its own id as the anchor is a no-op).
        assert_eq!(tier1_ref_for(&sub(2, false), anchor), None);
        // With no anchor, nothing is aligned — the gap case, served as-is.
        assert_eq!(tier1_ref_for(&sub(1, false), None), None);
    }

    #[test]
    fn sync_cache_key_separates_raw_ref_and_resync() {
        let raw = sync_cache_key(5, &None, None);
        let aligned = sync_cache_key(5, &None, Some(9));
        let resynced = sync_cache_key(5, &Some("http://host/s.mkv".into()), None);
        assert_eq!(raw, "os:5");
        assert_eq!(aligned, "os:5:ref:9");
        assert!(resynced.starts_with("os:5:resync:"));
        // The three variants must never collide — else an aligned sub is served from the raw entry.
        assert_ne!(raw, aligned);
        assert_ne!(raw, resynced);
        assert_ne!(aligned, resynced);
        // resync (Tier-2) takes precedence over a ref (Tier-1) when both are present.
        let both = sync_cache_key(5, &Some("http://host/s.mkv".into()), Some(9));
        assert_eq!(both, resynced);
    }

    #[test]
    fn resync_guard_allows_lan_streams() {
        // The stream lives on the user's LAN (den-scout on a private IP), so private ranges and
        // ordinary hostnames must be reachable.
        assert!(is_safe_resync_url("http://192.168.1.10:8080/stream.mkv"));
        assert!(is_safe_resync_url("http://10.0.0.5/a.mkv"));
        assert!(is_safe_resync_url("http://172.16.3.4/a.mkv"));
        assert!(is_safe_resync_url("https://stream.example.com/a.mkv"));
        // A public IPv6 stream host, bracketed with a port, is still reachable.
        assert!(is_safe_resync_url("http://[2001:db8::1]:8080/a.mkv"));
    }

    #[test]
    fn resync_guard_blocks_loopback_linklocal_and_bad_schemes() {
        // Loopback / localhost / link-local (incl. the cloud-metadata endpoint) are refused.
        assert!(!is_safe_resync_url("http://127.0.0.1/a.mkv"));
        assert!(!is_safe_resync_url("http://localhost:8080/a.mkv"));
        assert!(!is_safe_resync_url("http://sub.localhost/a.mkv"));
        assert!(!is_safe_resync_url("http://169.254.169.254/latest/meta-data/"));
        // IPv6 loopback, bracketed — the case the old naive `:`-split let through.
        assert!(!is_safe_resync_url("http://[::1]/a.mkv"));
        assert!(!is_safe_resync_url("http://[::1]:8080/a.mkv"));
        // userinfo must not smuggle a blocked host past the check.
        assert!(!is_safe_resync_url("http://user@127.0.0.1/a.mkv"));
        // Only http(s); no file/ftp/empty.
        assert!(!is_safe_resync_url("file:///etc/passwd"));
        assert!(!is_safe_resync_url("ftp://host/a.mkv"));
        assert!(!is_safe_resync_url(""));
    }
}
