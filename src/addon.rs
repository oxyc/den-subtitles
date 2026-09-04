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

use crate::httputil::{self, Body};
use crate::opensubtitles;
use crate::state::AppState;
use crate::userconfig::{self, LlmConfig, UserConfig};
use crate::{srt, translate};

const CACHE_TTL: Duration = Duration::from_secs(60 * 60 * 24 * 60); // 60 days — mirrors the app cache
/// How long a raw sub stands in for an alignment that failed. Long enough that a retrying client
/// doesn't re-spawn the tier binary per request, short enough that a fixed deploy heals itself.
const SYNC_RETRY_TTL: Duration = Duration::from_secs(600);
/// Namespace for the "this sync just failed" marker. Kept off every prefix a BODY is stored under
/// (`BODY_PREFIXES`) so a marker can never be read back and served as a subtitle.
const SYNCFAIL: &str = "syncfail:";
/// Digits in an IMDb id. Real ones run to seven or eight; ten leaves room and still bounds the key.
const MAX_IMDB_DIGITS: usize = 10;
/// Longest a target-language name may be. The longest real one is a couple of dozen characters.
const MAX_LANG: usize = 64;
#[cfg(test)]
const BODY_PREFIXES: [&str; 3] = ["os:", "search:", "translate:"];
// Search results turn over as new subs are uploaded, so a short TTL — enough to spare repeated
// round-trips when the app reopens a title, not so long that fresh uploads stay hidden.
const SEARCH_TTL: Duration = Duration::from_secs(60 * 60 * 6); // 6 hours

/// Monotonic counter making sync scratch-file names unique per invocation.
static SYNC_SEQ: AtomicU64 = AtomicU64::new(0);

pub fn manifest(configured: bool) -> Value {
    json!({
        "id": "fi.oxy.den-subtitles",
        // Single source of truth: the Cargo package version (CI asserts it == the v* tag). So the
        // manifest can't drift from Cargo.toml, nor the tag from either.
        "version": env!("CARGO_PKG_VERSION"),
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
    // `tt` and digits, and not many of them. It goes into a cache key that becomes a filename —
    // the same hazard `search_hash` bounds `videoHash` against and `MAX_LANG` bounds `lang`
    // against. An id long enough to overflow NAME_MAX fails the disk write and burns the process's
    // one-shot "persistence degraded" warning on a request that was never a real title.
    let digits = imdb.strip_prefix("tt")?;
    if digits.is_empty() || digits.len() > MAX_IMDB_DIGITS || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let season = parts.next().and_then(|s| s.parse().ok());
    let episode = parts.next().and_then(|s| s.parse().ok());
    Some((imdb.to_string(), season, episode))
}

/// The client's OSHash, if it really is one: 16 hex digits. Anything else is not a hash, and this
/// value goes into a cache key that becomes a filename — an over-long one fails the disk write and
/// burns the process's one-shot "persistence degraded" warning on a request that was never valid.
fn search_hash(extra: &str) -> Option<String> {
    extra_field(extra, "videoHash").filter(|h| h.len() == 16 && h.bytes().all(|b| b.is_ascii_hexdigit()))
}

/// Pull a `key=value` field out of the Stremio extra-args blob (already `.json`-stripped). The
/// client sends `videoHash`, `videoSize`, and `filename` here.
fn extra_field(extra: &str, key: &str) -> Option<String> {
    // Split first, decode second. Decoding the whole blob turned a %26 inside a value into a real
    // separator, so "Fast %26 Furious 6.mkv" arrived as the filename "Fast " — and a filename
    // carrying %26videoHash%3D... replaced the hash the client actually sent, which then went to
    // OpenSubtitles as the moviehash and silently disabled the whole sync ladder.
    let prefix = format!("{key}=");
    for pair in extra.split('&') {
        if let Some(v) = pair.strip_prefix(&prefix) {
            let v = httputil::percent_decode_path(v);
            if !v.is_empty() {
                return Some(v);
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
    // An OSHash is 16 hex digits. Anything else is not one, and this value goes into a cache key
    // that becomes a filename — an over-long one fails the disk write and burns the process's
    // one-shot "persistence degraded" warning on a request that was never valid.
    let hash = search_hash(extra);
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
            api_base: opensubtitles::API,
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
    // A strong ETag (hash of this serialized ranked list) is attached by `json()`; add
    // stale-while-revalidate so a client can serve the last list instantly while refreshing.
    httputil::json(StatusCode::OK, &json!({"subtitles": out}), "public, max-age=3600, stale-while-revalidate=3600")
}

/// The `file_id` of a trusted timing reference for Tier-1 reference alignment: the BEST hash-matched
/// sub, authored against this exact encode, so its timing is correct by construction (language
/// doesn't matter — timing is language-independent).
///
/// Best, not first. This runs after `rank`, which sorts by language ascending, so taking the first
/// hash match took the one belonging to the alphabetically-lowest language code — Arabic before
/// English. Every hash match is equally correct about timing, but they are not equally useful AS a
/// reference: a hash-matched forced/signs-only track is a handful of cues, and handing ffsubsync a
/// near-empty reference produces a bad alignment that then caches for sixty days. `fit_score` has no
/// cue count to work from, but trust/ratings/downloads separate a full dialogue track from a signs
/// track well enough.
///
/// `Reverse` + `min_by_key` rather than `max_by_key`: both pick a highest score, but `max_by_key`
/// returns the LAST of equal maxima and `min_by_key` the first. Ties are common here (two untrusted,
/// unrated hash matches), and the anchor decides the `?ref=` in every URL we hand back — so it has to
/// be the same choice on every request over the same cached list, not merely a valid one.
///
/// `None` when the search produced no hash match at all. That is the gap behind the out-of-sync
/// complaint: with no trusted anchor we deliberately do NOT align to an untrusted sub (that could
/// make timing worse), so those subs are served as-is and only the Tier-2 audio resync — a user
/// action, see `is_safe_resync_url` — can fix them.
fn tier1_reference(subs: &[opensubtitles::Subtitle]) -> Option<i64> {
    subs.iter()
        .filter(|s| s.hash_match)
        .min_by_key(|s| std::cmp::Reverse(opensubtitles::fit_score(s, None)))
        .map(|s| s.file_id)
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
    // Vet BEFORE keying: a rejected URL changes which tier runs, so keying off the raw one filed a
    // reference-aligned (or raw) body under a `resync` key. Everything downstream reads the VETTED
    // value, so a refused target is simply a request that asked for no sync.
    let resync_url = match resync_url {
        Some(u) if is_safe_resync_url(&u).await => Some(u),
        // Never the URL itself: a stream target carries the provider's token.
        Some(_) => {
            eprintln!("subtitle: refusing unsafe resync target for {file_id}");
            None
        }
        None => None,
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
        api_base: opensubtitles::API,
    };

    let target = match subtitle_srt(state, &client, file_id).await {
        Ok(body) => body,
        Err(e) => {
            eprintln!("subtitle: download of file {file_id} failed: {e}");
            return httputil::text(StatusCode::BAD_GATEWAY, "upstream subtitle fetch failed");
        }
    };

    // A sync that just failed is not retried on every request — the binary spawn, or a 90s alass
    // timeout, would be paid again per request. The marker is separate from `cache_key` so that key
    // never holds anything but a settled answer.
    let wanted_sync = resync_url.is_some() || ref_id.is_some();
    let retry_marker = format!("{SYNCFAIL}{cache_key}");
    if wanted_sync && state.cache.get(&retry_marker).is_some() {
        return httputil::srt_provisional(target);
    }

    // Per-invocation unique temp tag: two concurrent requests for the same file must not share
    // scratch paths (one would read the other's half-written output and cache it for 60 days).
    let tag = format!("{}-{}", cache_key.replace(':', "-"), SYNC_SEQ.fetch_add(1, Ordering::Relaxed));
    let synced: Option<String> = if let Some(url) = resync_url {
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
            Ok(reference) => match state.sync.sync_to_reference(&target, &reference, &tag).await {
                Ok(s) => Some(s),
                Err(e) => {
                    eprintln!("subtitle: align {file_id} to {r} failed: {e}");
                    None
                }
            },
            Err(e) => {
                eprintln!("subtitle: reference {r} for {file_id} unavailable: {e}");
                None
            }
        }
    } else {
        None
    };

    // A sync that was ASKED FOR and did not happen is not the answer to this key — it is the raw sub
    // standing in. Caching it for 60 days makes one broken afternoon permanent: the URL is
    // deterministic, so every later request is a cache hit and the sync never runs again. Remember it
    // briefly so a retry loop doesn't re-spawn the binary per request, and let it heal.
    match synced {
        // The alignment happened: this body IS the answer to this key.
        Some(body) => {
            state.cache.put(cache_key, body.clone(), CACHE_TTL);
            httputil::srt(body)
        }
        // Asked for and didn't happen. The raw sub stands in, and is NOT written to `cache_key` —
        // caching it there made one broken afternoon permanent, since the URL is deterministic and
        // every later request became a hit that never retried. Only the "don't re-spawn for a
        // moment" marker is remembered; the raw body is already cached under `os:{file_id}`.
        None if wanted_sync => {
            state.cache.put(retry_marker, "1".into(), SYNC_RETRY_TTL);
            httputil::srt_provisional(target)
        }
        // No sync was asked for, so the raw sub is the answer. `subtitle_srt` already cached it.
        None => httputil::srt(target),
    }
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
async fn is_safe_resync_url(u: &str) -> bool {
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
        Ok(ip) => !is_blocked_ip(ip),
        // A HOSTNAME (not a literal IP): RESOLVE it and refuse if ANY resolved address is internal —
        // otherwise an attacker-controlled name that resolves to 169.254.169.254 (cloud metadata) or
        // 127.0.0.1 (DNS rebinding) sails through. Fail closed on a resolve error. (LAN/private ranges stay
        // allowed — the stream legitimately lives on the LAN; only loopback/link-local/unspecified are refused.)
        Err(_) => match tokio::net::lookup_host((host_lc.as_str(), 0u16)).await {
            Ok(addrs) => {
                let ips: Vec<std::net::IpAddr> = addrs.map(|a| a.ip()).collect();
                !ips.is_empty() && !ips.iter().any(|ip| is_blocked_ip(*ip))
            }
            Err(_) => false,
        },
    }
}

/// Addresses we refuse to fetch server-side (SSRF): loopback, link-local (incl. the 169.254 cloud-metadata
/// endpoint), and unspecified — plus their IPv4-mapped-IPv6 forms. Private LAN ranges (RFC1918 / IPv6 ULA)
/// are DELIBERATELY allowed: Tier-2 resync fetches the user's stream, which lives on the LAN.
fn is_blocked_ip(ip: std::net::IpAddr) -> bool {
    use std::net::IpAddr;
    match ip {
        IpAddr::V4(v4) => v4.is_loopback() || v4.is_link_local() || v4.is_unspecified(),
        IpAddr::V6(v6) => {
            if v6.is_loopback() || v6.is_unspecified() {
                return true;
            }
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_blocked_ip(IpAddr::V4(v4));
            }
            v6.segments()[0] & 0xffc0 == 0xfe80 // link-local fe80::/10
        }
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
    // A language name, not an essay. This lands in a cache key that becomes a filename — the same
    // hazard `search_hash` already bounds `videoHash` against — and it is interpolated into the
    // prompt of every batch, so its length multiplies the bill against the viewer's own key.
    if lang.is_empty() || lang.len() > MAX_LANG {
        return httputil::text(StatusCode::BAD_REQUEST, "bad_lang");
    }
    let cache_key = format!(
        "translate:{imdb}:{}:{}:{lang}:{}:{}",
        season.unwrap_or(0),
        episode.unwrap_or(0),
        provider_tag(llm),
        llm.model,
    );

    // Warm the cache if needed (both the .json and .srt forms share it).
    let failed_recently = format!("{SYNCFAIL}{cache_key}");
    if state.cache.get(&cache_key).is_none() {
        // A whole film's LLM bill is not something to re-pay on every tap. The `.json` and `.srt`
        // forms are two requests, so the app's own flow retries once by design — and nothing was
        // remembered about the failure, so each attempt paid in full again, with no backoff. Same
        // marker the sync path uses, in the same namespace.
        // Said differently from a fresh failure on purpose: from the outside the two are the same
        // 502, and the difference — "we just tried" versus "we are backing off" — is the first
        // thing you want to know when a translation stops working.
        if state.cache.get(&failed_recently).is_some() {
            return httputil::text(StatusCode::BAD_GATEWAY, "translation failed recently");
        }
        if let Err(e) = produce_translation(state, &cfg, llm, &imdb, season, episode, lang, &cache_key).await {
            // Log the detail (no key in these strings); hand the client a generic message rather than
            // echoing a raw upstream error body.
            eprintln!("translate: {imdb} → {lang} failed: {e}");
            state.cache.put(failed_recently, "1".into(), SYNC_RETRY_TTL);
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
        api_base: opensubtitles::API,
    };
    // Prefer an English source (best-resourced), else whatever exists.
    //
    // Searching "en" made that "else" unreachable: the list it picks from could only ever hold
    // English, so `.or_else` was dead code and a film with a perfectly good Spanish or French source
    // returned "no source subtitle to translate". Asking for every language costs the same one round
    // trip; the preference is expressed in the pick below, not in the query.
    let subs = client.search(imdb, season, episode, "all", None).await?;
    let source = opensubtitles::best_for(&subs, "en")
        // No English at all. Take the best of what there is, by the same score and the same
        // first-wins tie-break `tier1_reference` uses — `subs` arrives unranked from the API, so
        // `.first()` was whichever result OpenSubtitles happened to list first. `fit_score` also
        // sinks machine/AI-translated subs, which is what keeps us from translating a translation.
        .or_else(|| {
            subs.iter()
                .min_by_key(|s| std::cmp::Reverse(opensubtitles::fit_score(s, None)))
        })
        .ok_or("no source subtitle to translate")?;
    // Through the cache, not straight at the API. A `/download` call spends one of the viewer's
    // daily OpenSubtitles credits on the CALL, not on the file fetch — and dodging that quota is the
    // reason the proxy-and-cache design exists at all. Going direct re-paid for a file already
    // sitting under `os:{file_id}`: once per retry after the ten-minute backoff, and once more for
    // every additional target language of the same film.
    let raw = subtitle_srt(state, &client, source.file_id).await?;
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
        // A hash match becomes the anchor, and a non-match never does.
        assert_eq!(tier1_reference(&[sub(1, false), sub(2, true), sub(3, true)]), Some(2));
        // Among equally-scoring hash matches the choice is the FIRST, and it is stable: the anchor
        // decides the `?ref=` in every URL handed back, so it must be the same answer every time the
        // same cached list is ranked, not merely a valid one.
        assert_eq!(tier1_reference(&[sub(3, true), sub(2, true), sub(1, true)]), Some(3));
    }

    /// The anchor is the BEST hash match, not the first one in language order. `tier1_reference`
    /// runs after `rank` sorts by language ascending, so `find` handed ffsubsync whichever hash match
    /// belonged to the alphabetically-lowest language code — and a hash-matched forced/signs-only
    /// track is a handful of cues, which makes a poor reference and a bad alignment that then caches
    /// for sixty days.
    #[test]
    fn the_anchor_is_the_best_hash_match_not_the_first() {
        // A sparse signs track (first in language order) must lose to a well-used dialogue track.
        let signs = Subtitle { downloads: 3, ..sub(1, true) };
        let dialogue = Subtitle { downloads: 50_000, from_trusted: true, ..sub(2, true) };
        assert_eq!(tier1_reference(&[signs.clone(), dialogue.clone()]), Some(2));
        // And the order it arrives in doesn't change the answer.
        assert_eq!(tier1_reference(&[dialogue, signs]), Some(2));

        // A non-hash sub never becomes the anchor, however popular — being the same encode is the
        // whole claim, and downloads are not evidence of that.
        let popular_guess = Subtitle { downloads: 999_999, from_trusted: true, ..sub(9, false) };
        assert_eq!(tier1_reference(&[popular_guess, sub(4, true)]), Some(4));
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

    #[tokio::test]
    async fn resync_guard_allows_lan_and_public_streams() {
        // The stream lives on the user's LAN (den-scout on a private IP), so private ranges stay reachable;
        // a public literal is fine too. (Hostnames are vetted by resolution at runtime, not asserted here.)
        assert!(is_safe_resync_url("http://192.168.1.10:8080/stream.mkv").await);
        assert!(is_safe_resync_url("http://10.0.0.5/a.mkv").await);
        assert!(is_safe_resync_url("http://172.16.3.4/a.mkv").await);
        assert!(is_safe_resync_url("http://8.8.8.8/a.mkv").await); // public literal
        assert!(is_safe_resync_url("http://[2001:db8::1]:8080/a.mkv").await);
        assert!(is_safe_resync_url("http://[fc00::1]/a.mkv").await); // IPv6 ULA = LAN, allowed
    }

    #[tokio::test]
    async fn resync_guard_blocks_internal_and_bad_schemes() {
        // Loopback / localhost / link-local (incl. the 169.254 cloud-metadata endpoint) / unspecified.
        assert!(!is_safe_resync_url("http://127.0.0.1/a.mkv").await);
        assert!(!is_safe_resync_url("http://localhost:8080/a.mkv").await);
        assert!(!is_safe_resync_url("http://sub.localhost/a.mkv").await);
        assert!(!is_safe_resync_url("http://169.254.169.254/latest/meta-data/").await);
        assert!(!is_safe_resync_url("http://0.0.0.0/a.mkv").await); // unspecified
        // IPv6 loopback / link-local / IPv4-mapped-loopback, bracketed.
        assert!(!is_safe_resync_url("http://[::1]/a.mkv").await);
        assert!(!is_safe_resync_url("http://[::1]:8080/a.mkv").await);
        assert!(!is_safe_resync_url("http://[fe80::1]/a.mkv").await);
        assert!(!is_safe_resync_url("http://[::ffff:127.0.0.1]/a.mkv").await);
        // userinfo must not smuggle a blocked host past the check.
        assert!(!is_safe_resync_url("http://user@127.0.0.1/a.mkv").await);
        // Only http(s); no file/ftp/empty.
        assert!(!is_safe_resync_url("file:///etc/passwd").await);
        assert!(!is_safe_resync_url("ftp://host/a.mkv").await);
        assert!(!is_safe_resync_url("").await);
    }
}

#[cfg(test)]
mod id_tests {
    use super::*;

    /// The id is `tt` and digits, and it lands in a cache key that becomes a filename. Left
    /// unbounded it overflowed NAME_MAX, so the entry never persisted AND the process's one-shot
    /// "persistence degraded" warning was spent on it — the next real disk problem then said
    /// nothing. Two siblings of this value were already bounded for exactly that reason; this one
    /// was not, which is the shape of gap a per-field fix leaves behind.
    #[test]
    fn an_id_is_tt_and_a_sane_number_of_digits() {
        assert_eq!(parse_id("tt0111161").unwrap().0, "tt0111161");
        assert_eq!(parse_id("tt0111161:2:5").unwrap(), ("tt0111161".into(), Some(2), Some(5)));

        for bad in [
            "tt",                                  // no digits
            "tt12a4",                              // not digits
            "nope0111161",                         // no prefix
            "",
        ] {
            assert!(parse_id(bad).is_none(), "accepted {bad:?}");
        }
        // Long enough to overflow a filename once base64'd into a cache key.
        let huge = format!("tt{}", "1".repeat(2000));
        assert!(parse_id(&huge).is_none(), "accepted a 2000-digit id");
        assert!(parse_id(&format!("tt{}", "1".repeat(MAX_IMDB_DIGITS + 1))).is_none());
        assert!(parse_id(&format!("tt{}", "1".repeat(MAX_IMDB_DIGITS))).is_some());
    }
}

#[cfg(test)]
mod sync_fallback_tests {
    use super::*;
    use hyper::header::CACHE_CONTROL;

    /// The two response shapes must differ in the one way that matters: `immutable` tells a client
    /// never to come back, which is a year of an out-of-sync subtitle if the body is a stand-in.
    #[test]
    fn a_stand_in_is_revalidated_and_a_settled_body_is_not() {
        let cc = |r: Response<Body>| r.headers().get(CACHE_CONTROL).unwrap().to_str().unwrap().to_string();
        let provisional = cc(httputil::srt_provisional("x".into()));
        assert!(!provisional.contains("immutable"), "a stand-in must not be immutable: {provisional}");
        assert!(provisional.contains("must-revalidate"), "the client has to come back: {provisional}");
        assert!(cc(httputil::srt("x".into())).contains("immutable"));
    }

    /// A rejected resync target leaves a request that asked for no sync at all: the key must not
    /// keep claiming a resync, and `os:5` — the same key `subtitle_srt` fills — must not be
    /// downgraded to a stand-in, or the raw sub is re-downloaded on the next request.
    #[test]
    fn a_rejected_resync_target_leaves_a_plain_request() {
        let vetted: Option<String> = None; // what the SSRF guard leaves behind
        assert_eq!(sync_cache_key(5, &vetted, Some(9)), "os:5:ref:9");
        assert_eq!(sync_cache_key(5, &vetted, None), "os:5");
        // Two different targets stay distinct, so one stream's alignment is never served for another.
        assert_ne!(
            sync_cache_key(5, &Some("http://host/a.mkv".into()), None),
            sync_cache_key(5, &Some("http://host/b.mkv".into()), None)
        );
    }

    /// The invariant the whole fix rests on: a retry marker can never be read back as a body. Built
    /// from the real key builders rather than literals, so a change to any of them is caught here.
    #[test]
    fn a_retry_marker_can_never_be_read_as_a_body() {
        // Every body key in the service starts with one of these; the marker starts with none.
        for prefix in BODY_PREFIXES {
            assert!(!prefix.starts_with(SYNCFAIL), "{SYNCFAIL} shadows the body prefix {prefix}");
            assert!(!SYNCFAIL.starts_with(prefix), "a {prefix} read would match {SYNCFAIL}");
        }
        // And the key builder really does produce keys inside that namespace, for every tier.
        for key in [
            sync_cache_key(5, &None, None),
            sync_cache_key(5, &None, Some(9)),
            sync_cache_key(5, &Some("http://host/a.mkv".into()), None),
        ] {
            assert!(BODY_PREFIXES.iter().any(|p| key.starts_with(p)), "unnamespaced body key: {key}");
            assert!(!key.starts_with(SYNCFAIL));
        }
    }

    /// The extras blob is `key=value` pairs joined by `&`, each value percent-encoded. Decoding the
    /// blob before splitting let a value's own `%26` become a separator.
    #[test]
    fn an_encoded_ampersand_stays_inside_its_value() {
        let extra = "videoHash=8e24&videoSize=734003200&filename=Fast%20%26%20Furious%206%20(2013).mkv";
        assert_eq!(extra_field(extra, "filename").as_deref(), Some("Fast & Furious 6 (2013).mkv"));
        assert_eq!(extra_field(extra, "videoHash").as_deref(), Some("8e24"));

        // And a value cannot forge a field the client never sent.
        let forged = "filename=movie%26videoHash%3Dcafebabecafebabe.mkv";
        assert_eq!(extra_field(forged, "videoHash"), None, "a filename forged a videoHash");

        // A videoHash is 16 hex digits. Anything else is not one — and it lands in a cache key
        // that becomes a filename, where an over-long value fails the write and consumes the
        // process's only "persistence degraded" warning on a request that was never valid.
        assert_eq!(extra_field("videoHash=8e245d9679d31e12", "videoHash").as_deref(), Some("8e245d9679d31e12"));
        for bad in ["videoHash=8e24", "videoHash=zzzzzzzzzzzzzzzz", &format!("videoHash={}", "a".repeat(300))] {
            assert_eq!(search_hash(bad), None, "accepted a non-hash: {bad}");
        }

        // The extras arrive as a PATH segment, where `+` is a literal plus. Form-decoding it turned
        // every HDR10+ / DTS-HD 7.1+ release name into one the ranker scores differently.
        let plus = "filename=Movie.2013.2160p.HDR10+.WEB-DL.mkv";
        assert_eq!(
            extra_field(plus, "filename").as_deref(),
            Some("Movie.2013.2160p.HDR10+.WEB-DL.mkv")
        );
    }
}

#[cfg(test)]
mod translate_retry_tests {
    use super::*;
    use crate::config::Config;
    use crate::state::AppState;

    /// A state with its own cache directory. The disk tier is real and persists between runs, so a
    /// shared directory would carry one test's markers into another's preconditions.
    fn state(name: &str) -> Arc<AppState> {
        let dir = std::env::temp_dir().join(format!("den-subs-translate-retry-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        AppState::new(Config {
            port: 0,
            cache_dir: dir,
            cache_max_bytes: 1 << 20,
            public_base_url: None,
            ffsubsync: "ffsubsync".into(),
            alass: "alass".into(),
            config_key: String::new(),
            config_keys_prev: String::new(),
        })
    }

    /// A config segment with an LLM key, so `handle_translate` gets past its own guards.
    fn config_segment() -> String {
        use base64::Engine;
        base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(r#"{"osKey":"os-test","provider":"openai","apiKey":"llm-test","model":"m"}"#)
    }

    /// A film's LLM bill must not be re-paid on every tap. The `.json` and `.srt` forms are two
    /// requests, so the app retries once by design, and nothing was remembered about a failure.
    /// The marker has to short-circuit — and it must never be mistaken for the subtitle itself:
    /// its key is in the `syncfail:` namespace, and every body read uses a `translate:` key.
    #[tokio::test]
    async fn a_remembered_failure_short_circuits_without_being_served() {
        let state = state("short-circuit");
        let config = config_segment();
        let cfg = userconfig::decode(state.config_keyring.as_ref(), &config).expect("test config decodes");
        let llm = cfg.llm.as_ref().expect("test config carries an llm");
        let cache_key = format!("translate:tt0111161:0:0:Swedish:{}:{}", provider_tag(llm), llm.model);
        state.cache.put(format!("{SYNCFAIL}{cache_key}"), "1".into(), SYNC_RETRY_TTL);

        // No HTTP client is configured in this state, so reaching the upstream would fail
        // differently — a 502 here means the marker short-circuited before any of that.
        let resp = handle_translate(
            &state,
            &HeaderMap::new(),
            &config,
            "tt0111161",
            "Swedish",
            false,
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

        // The marker's value must never reach the client as a subtitle.
        let body = String::from_utf8_lossy(
            &http_body_util::BodyExt::collect(resp.into_body()).await.unwrap().to_bytes(),
        )
        .to_string();
        assert_ne!(body.trim(), "1", "the retry marker was served as the response body");
        // "recently" is what says the marker short-circuited rather than a fresh attempt failing:
        // with no HTTP client both paths end in a 502, so the status alone proves nothing.
        assert!(body.contains("recently"), "the marker did not short-circuit; body: {body}");
    }

    /// An over-long language name is refused before it can become a filename or a prompt. The same
    /// file already bounds `videoHash` for the first reason; the second is that `lang` goes into
    /// every batch's prompt, so its length multiplies the bill against the viewer's own key.
    #[tokio::test]
    async fn an_over_long_language_is_refused() {
        let state = state("long-lang");
        let config = config_segment();
        let long = "x".repeat(MAX_LANG + 1);
        let resp = handle_translate(&state, &HeaderMap::new(), &config, "tt0111161", &long, false).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "an over-long language was accepted");

        // Nothing was written for it — not the body key, and not the failure marker either.
        let store = state.cfg.cache_dir.join("store");
        let files = std::fs::read_dir(&store).map(|d| d.count()).unwrap_or(0);
        assert_eq!(files, 0, "an invalid request left {files} cache files behind");

        // A real language still works its way through to the upstream check.
        let resp = handle_translate(&state, &HeaderMap::new(), &config, "tt0111161", "Swedish", false).await;
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY, "a valid language must not be refused");
    }

    /// And the marker is scoped to the translation it belongs to: another language is a different
    /// job and must still be attempted. Proven by the marker the ATTEMPT leaves behind — this state
    /// has no HTTP client, so a Finnish run gets as far as failing on that and recording it, which
    /// a short-circuit would never do.
    #[tokio::test]
    async fn a_remembered_failure_does_not_block_a_different_translation() {
        let state = state("scoping");
        let config = config_segment();
        let cfg = userconfig::decode(state.config_keyring.as_ref(), &config).unwrap();
        let llm = cfg.llm.as_ref().unwrap();
        let key_for = |lang: &str| format!("translate:tt0111161:0:0:{lang}:{}:{}", provider_tag(llm), llm.model);
        state.cache.put(format!("{SYNCFAIL}{}", key_for("Swedish")), "1".into(), SYNC_RETRY_TTL);

        let finnish_marker = format!("{SYNCFAIL}{}", key_for("Finnish"));
        assert!(state.cache.get(&finnish_marker).is_none(), "precondition: Finnish is unmarked");
        let resp = handle_translate(&state, &HeaderMap::new(), &config, "tt0111161", "Finnish", false).await;
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        assert!(
            state.cache.get(&finnish_marker).is_some(),
            "Finnish was short-circuited by Swedish's marker instead of being attempted"
        );
    }
}
