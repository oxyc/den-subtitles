//! Addon request path: the Stremio `subtitles` resource plus the Den-specific translate/serve
//! endpoints.
//!
//!   GET /manifest.json                                  unconfigured manifest
//!   GET /<config>/manifest.json                         configured manifest
//!   GET /<config>/subtitles/<type>/<id>/<extra>.json    native subs (OpenSubtitles, hash-matched)
//!   GET /<config>/subtitle/<file_id>.(srt|vtt)          proxy+cache one OpenSubtitles file
//!   GET /<config>/translate/<type>/<id>/<extra>/<lang>.json  app-driven: kick off/await a translation → { url }
//!   GET /<config>/translate/<type>/<id>/<extra>/<lang>.srt   the translated SRT (cache hit after the .json warmed it)
//!   GET /<config>/translate/<type>/<id>/<extra>/<lang>.status how far a running translation has got
//!
//! `<id>` is `tt<digits>` or `tt<digits>:<season>:<episode>`. `<extra>` is the Stremio query blob
//! carrying `videoHash`/`videoSize` (the OSHash the app computed). On the translate routes it is
//! optional — the older five-segment form still resolves — but without it a translation can find no
//! Tier-1 anchor and is served on whatever timing its source happened to have.

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

    // An OSHash is 16 hex digits. Anything else is not one, and this value goes into a cache key
    // that becomes a filename — an over-long one fails the disk write and burns the process's
    // one-shot "persistence degraded" warning on a request that was never valid.
    let hash = search_hash(extra);
    let filename = extra_field(extra, "filename");
    let client = os_client(state, http, &cfg);
    let mut subs = match cached_search(state, &client, &imdb, season, episode, hash.as_deref()).await {
        Ok(s) => s,
        // Empty-200 is the correct Stremio shape for "nothing"; `cached_search` has already logged
        // the cause and counted it for /health.
        Err(_) => return httputil::json(StatusCode::OK, &json!({"subtitles": []}), "no-store"),
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

/// The OpenSubtitles client for one request, from that install's BYOK credentials.
fn os_client<'a>(state: &'a AppState, http: &'a reqwest::Client, cfg: &'a UserConfig) -> opensubtitles::Client<'a> {
    opensubtitles::Client {
        http,
        api_key: &cfg.opensubtitles_key,
        token: cfg.opensubtitles_token.as_deref(),
        api_base: &state.cfg.os_api_base,
    }
}

/// The search for one title, cached. Config-independent (file_ids and languages are the same for
/// everyone), keyed by the query params including the file hash. Short TTL — new subs get uploaded —
/// but enough to spare a live round-trip every time the app reopens a title.
///
/// Always asks for every language: the app filters by its own preferred-language rules, and the
/// translate path needs to see every candidate source.
///
/// Entries are per hash, so the picker (which sends one) and the translate path's source lookup
/// (which deliberately does not — see `handle_translate`) do NOT share an entry. A translate on a
/// cold title therefore pays a second search. Searches spend no download credit, and the alternative
/// is choosing the source from a list whose contents depend on the encode, which costs a second
/// full-price translation.
///
/// Returned UNRANKED, and cached that way: ranking is filename-specific, so each caller ranks the
/// list for its own request.
async fn cached_search(
    state: &Arc<AppState>,
    client: &opensubtitles::Client<'_>,
    imdb: &str,
    season: Option<i64>,
    episode: Option<i64>,
    hash: Option<&str>,
) -> Result<Vec<opensubtitles::Subtitle>, String> {
    let search_key = format!(
        "search:{imdb}:{}:{}:{}",
        season.unwrap_or(0),
        episode.unwrap_or(0),
        hash.unwrap_or("")
    );
    if let Some(hit) = state.cache.get(&search_key).and_then(|h| serde_json::from_str(&h).ok()) {
        return Ok(hit);
    }
    match client.search(imdb, season, episode, "all", hash).await {
        Ok(s) => {
            state.os_fails.store(0, Ordering::Relaxed);
            if let Ok(json) = serde_json::to_string(&s) {
                state.cache.put(search_key, json, SEARCH_TTL);
            }
            Ok(s)
        }
        // Log the cause (our error strings carry no key) and count it so /health can report
        // `degraded` (ADDON-02). Counted HERE rather than at one call site, so a translation that
        // cannot reach OpenSubtitles is visible on /health too — it was not before.
        Err(e) => {
            eprintln!("search: opensubtitles failed for {imdb}: {e}");
            state.os_fails.fetch_add(1, Ordering::Relaxed);
            Err(e)
        }
    }
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
    // A subtitle is never its own timing reference. `tier1_ref_for` refuses that when it BUILDS a
    // URL, but this value arrives on the query string and anyone can name it: `?ref=5` on file 5
    // spawned a tier binary to align a file to itself — a subprocess, on the one runtime thread, for
    // a guaranteed no-op — and filed the result under `os:5:ref:5`.
    let ref_id = ref_id.filter(|&r| r != file_id);
    // Cache identity depends on the sync mode so the raw and aligned variants don't collide.
    let cache_key = sync_cache_key(&os_base_key(file_id), &resync_url, ref_id);
    if let Some(hit) = state.cache.get(&cache_key) {
        return httputil::srt(hit);
    }
    let Some(http) = state.http.as_ref() else {
        return httputil::text(StatusCode::SERVICE_UNAVAILABLE, "subtitle service unavailable");
    };
    let client = os_client(state, http, &cfg);

    let target = match subtitle_srt(state, &client, file_id).await {
        Ok(body) => body,
        Err(e) => {
            eprintln!("subtitle: download of file {file_id} failed: {e}");
            return httputil::text(StatusCode::BAD_GATEWAY, "upstream subtitle fetch failed");
        }
    };

    sync_and_cache(state, &client, cache_key, target, ref_id, resync_url, &format!("subtitle {file_id}")).await
}

/// Run the sync ladder over an SRT we already hold, and apply the caching rules that go with it.
///
/// Shared by the two things that produce a servable subtitle: the OpenSubtitles proxy and the
/// translation endpoint. A translated body is an SRT with the source's timings (the harness never
/// lets the model touch a timecode), so it aligns exactly like a downloaded one — and it needs to,
/// because it inherits whatever offset the source it was translated from had.
///
/// `cache_key` must already be namespaced for the tier being run (see `sync_cache_key`), and
/// `resync_url` must already be vetted. `what` names the job in log lines only.
#[allow(clippy::too_many_arguments)]
async fn sync_and_cache(
    state: &Arc<AppState>,
    client: &opensubtitles::Client<'_>,
    cache_key: String,
    target: String,
    ref_id: Option<i64>,
    resync_url: Option<String>,
    what: &str,
) -> Response<Body> {
    // A sync that just failed is not retried on every request — the binary spawn, or a 90s alass
    // timeout, would be paid again per request. The marker is separate from `cache_key` so that key
    // never holds anything but a settled answer.
    let wanted_sync = resync_url.is_some() || ref_id.is_some();
    let retry_marker = format!("{SYNCFAIL}{cache_key}");
    if wanted_sync && state.cache.get(&retry_marker).is_some() {
        return httputil::srt_provisional(target);
    }

    // One tier binary per key at a time. Concurrent requests for the same alignment were each
    // spawning their own — a 90s alass run against the same stream, on a runtime with one thread —
    // and the scratch-file tag below only made that safe, never rare.
    let _flight = if wanted_sync {
        let guard = state.inflight.acquire(&cache_key).await;
        // Settled while we waited: the alignment we were about to run has already been run.
        if let Some(hit) = state.cache.get(&cache_key) {
            return httputil::srt(hit);
        }
        // And it may have failed while we waited, in which case re-running it now is the retry the
        // marker exists to prevent.
        if state.cache.get(&retry_marker).is_some() {
            return httputil::srt_provisional(target);
        }
        Some(guard)
    } else {
        None
    };

    // Per-invocation unique temp tag: two concurrent requests for the same file must not share
    // scratch paths (one would read the other's half-written output and cache it for 60 days).
    // Bounded, for the reason `Cache::disk_path` is: this becomes `{tag}-reference.srt` in the work
    // dir, and a translate cache key already runs to ~255 bytes with a long model name and language.
    // Past NAME_MAX the temp write fails, so the alignment fails, so the key earns a `syncfail:`
    // marker — a long model name silently disabling Tier-1 for that install.
    let tag = format!(
        "{}-{}",
        cache_key.replace(':', "-").chars().take(80).collect::<String>(),
        SYNC_SEQ.fetch_add(1, Ordering::Relaxed)
    );
    let synced: Option<String> = if let Some(url) = resync_url {
        // Tier 2 — audio VAD against the playing stream (opt-in; alass pulls the audio via ffmpeg).
        match state.sync.sync_to_audio(&target, &url, &tag).await {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("sync: resync of {what} failed: {e}");
                None
            }
        }
    } else if let Some(r) = ref_id {
        // Tier 1 — reference-align against the hash-matched sub (no audio needed).
        match subtitle_srt(state, client, r).await {
            Ok(reference) => match state.sync.sync_to_reference(&target, &reference, &tag).await {
                Ok(s) => Some(s),
                Err(e) => {
                    eprintln!("sync: aligning {what} to {r} failed: {e}");
                    None
                }
            },
            Err(e) => {
                eprintln!("sync: reference {r} for {what} unavailable: {e}");
                None
            }
        }
    } else {
        None
    };

    // A sync that was ASKED FOR and did not happen is not the answer to this key — it is the
    // unaligned body standing in. Caching it for 60 days makes one broken afternoon permanent: the
    // URL is deterministic, so every later request is a cache hit and the sync never runs again.
    // Remember it briefly so a retry loop doesn't re-spawn the binary per request, and let it heal.
    match synced {
        // The alignment happened: this body IS the answer to this key.
        Some(body) => {
            state.cache.put(cache_key, body.clone(), CACHE_TTL);
            httputil::srt(body)
        }
        // Asked for and didn't happen. The unaligned body stands in, and is NOT written to
        // `cache_key` — caching it there made one broken afternoon permanent, since the URL is
        // deterministic and every later request became a hit that never retried. Only the "don't
        // re-spawn for a moment" marker is remembered; the unaligned body is already cached under
        // its own base key.
        None if wanted_sync => {
            state.cache.put(retry_marker, "1".into(), SYNC_RETRY_TTL);
            httputil::srt_provisional(target)
        }
        // No sync was asked for, so the body we have is the answer, and it is already cached.
        None => httputil::srt(target),
    }
}

/// Fetch a subtitle's SRT, cached by file id (the raw, un-synced text — reused as a sync input).
async fn subtitle_srt(state: &Arc<AppState>, client: &opensubtitles::Client<'_>, file_id: i64) -> Result<String, String> {
    let key = os_base_key(file_id);
    if let Some(hit) = state.cache.get(&key) {
        return Ok(hit);
    }
    // Single-flighted, because this is the one call that spends a METERED credit: OpenSubtitles
    // charges the viewer's daily download allowance on the API call itself, and the anonymous
    // allowance is a handful per day. Two devices opening the same title, or the twenty picker URLs
    // that all carry the same `?ref=`, would each buy the same file.
    let _flight = state.inflight.acquire(&key).await;
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

/// The cache key for one raw OpenSubtitles file: the unaligned body that every sync variant of it
/// hangs off. Also what `subtitle_srt` files it under.
fn os_base_key(file_id: i64) -> String {
    format!("os:{file_id}")
}

/// Cache key for one servable subtitle, namespaced by sync mode so the unaligned / reference-aligned
/// / audio-resynced variants of the same body never collide (else an aligned sub would be served
/// from the unaligned entry, or vice versa). A resync URL takes precedence over a `ref` — it's the
/// stronger Tier-2 correction — mirroring the dispatch in `sync_and_cache`.
///
/// `base` is the identity of the UNALIGNED body: `os:{file_id}` for a proxied subtitle, or the
/// translate key for a translated one. Taking a base rather than a file id is what lets a translated
/// body carry the same tier namespacing, on the same rules, without a second copy of them.
fn sync_cache_key(base: &str, resync_url: &Option<String>, ref_id: Option<i64>) -> String {
    match (resync_url, ref_id) {
        (Some(url), _) => format!("{base}:resync:{}", short_hash(url)),
        (None, Some(r)) => format!("{base}:ref:{r}"),
        (None, None) => base.to_string(),
    }
}

/// Key of the "this translation failed recently" marker: title-scoped, and answerable before any
/// network call. That is the whole point of it — a whole film's LLM bill must not be re-paid on the
/// app's next tap, and the app's own `.json`-then-`.srt` flow retries once by design.
///
/// Deliberately NOT the key the translated body lives under (`translate_body_key`): that one names a
/// source file, which is only knowable after a search, and a marker that can only be consulted after
/// a search cannot do this job. Built here rather than inline so the handler and the tests cannot
/// drift apart on it — the same reason `sync_cache_key` exists, and `lang` in particular is
/// canonicalized rather than taken verbatim (see `translate::canonical_lang`).
///
/// Scoped to the INSTALL, unlike the body key. A translated body is the same bytes whoever asked for
/// it and is deliberately shared; a failure is not. One install with a dead provider key would
/// otherwise hand every other install on the same provider and model a 502 and a `.status` of
/// `failed` for ten minutes, and show them its progress bar — the cheapest cross-tenant lever in the
/// service, from something that was only ever meant to protect one viewer's bill.
fn translate_fail_key(
    config: &str,
    imdb: &str,
    season: Option<i64>,
    episode: Option<i64>,
    lang_key: &str,
    llm: &LlmConfig,
) -> String {
    format!(
        "translate:{:016x}:{imdb}:{}:{}:{lang_key}:{}:{}",
        short_hash(config),
        season.unwrap_or(0),
        episode.unwrap_or(0),
        llm.provider.tag(),
        llm.model,
    )
}

/// Cache key for the translated TEXT of one source subtitle.
///
/// Keyed by the SOURCE FILE, not by the title. The translated text depends only on what was
/// translated, so every encode of a film that resolves to the same source shares one translation —
/// and one LLM bill, the only cost here charged to the viewer's own provider account. The timing
/// differences between encodes are not this key's business: the aligned variants hang off it through
/// `sync_cache_key`, exactly as they do off `os:{file_id}`.
///
/// Keyed by provider+model too: stepping up to a bigger model to re-translate a title that read badly
/// is meant to overwrite, and it can only do that if the model is part of the identity.
fn translate_body_key(source_file_id: i64, lang_key: &str, llm: &LlmConfig) -> String {
    format!("translate:{source_file_id}:{lang_key}:{}:{}", llm.provider.tag(), llm.model)
}

/// The subtitle to translate FROM: English for preference, best-quality otherwise.
///
/// Judged as text, so the pick does not depend on which encode is playing — see
/// `opensubtitles::text_score` for why that matters to the bill.
///
/// The English preference is a bonus rather than a filter-then-fall-back, because "the best English
/// sub, else anything" picks a machine-translated English one over a good human Spanish one. Nothing
/// downstream can tell that it is translating a translation, and the errors compound instead of
/// cancelling.
fn translation_source(subs: &[opensubtitles::Subtitle]) -> Option<&opensubtitles::Subtitle> {
    // Enough to prefer any English source over an equally good one in another language, and nowhere
    // near enough to rescue a machine-translated one: `text_score` sinks those by two million.
    const ENGLISH: i64 = 10_000;
    subs.iter().min_by_key(|s| {
        let mut score = opensubtitles::text_score(s);
        if s.lang.eq_ignore_ascii_case("en") {
            score += ENGLISH;
        }
        // Highest score wins, and `file_id` breaks a tie. Taking "the first of equal scores" was not
        // good enough: first means first IN THE LIST, and the list arrives in an order OpenSubtitles
        // chooses. The chosen source is part of the translation's cache key, so a pick that depends
        // on the order buys the same film twice.
        (std::cmp::Reverse(score), s.file_id)
    })
}

/// Translations one install may START in a day.
///
/// A viewer watches a film, or an evening of episodes. This sits far above that and far below what
/// an install URL in the wrong hands could spend — and it is the viewer's own provider account that
/// gets spent, which is why the ceiling exists at all. Every other bound here caps ONE run:
/// `MAX_CUES` its size, `RUN_DEADLINE` its wall clock, the ten-minute marker its retries. None of
/// them bounds breadth — distinct titles times distinct languages, each one a fresh paid film.
const DAILY_TRANSLATIONS: u64 = 50;
/// Two days, so a day's counter falls out of the store on its own rather than needing a sweep.
const QUOTA_TTL: Duration = Duration::from_secs(2 * 24 * 60 * 60);

/// Key for one install's translation count today. The config segment is a bearer secret and this key
/// becomes a filename in the disk tier, so the segment is hashed rather than named.
fn quota_key(config: &str, now: std::time::SystemTime) -> String {
    let day = now
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() / 86_400)
        .unwrap_or(0);
    format!("quota:{}:{day}", short_hash(config))
}

/// Take one from today's allowance, or refuse. Charged only where a run actually BEGINS: a cache
/// hit, a resumed batch, and the `.srt` half of the app's own two-request flow are all free, because
/// none of them spends anything.
fn charge_translation(state: &Arc<AppState>, config: &str) -> bool {
    let key = quota_key(config, std::time::SystemTime::now());
    let used: u64 = state.cache.get(&key).and_then(|v| v.parse().ok()).unwrap_or(0);
    if used >= DAILY_TRANSLATIONS {
        return false;
    }
    // Read-modify-write, and deliberately not atomic: two runs starting in the same instant can both
    // read `used` and both write `used + 1`, so the count can undershoot by the number of concurrent
    // starts. This is an abuse ceiling, not an accounting record — being a few under on a burst
    // costs nothing, and the alternative is holding a lock over the whole cache for a counter.
    state.cache.put(key, (used + 1).to_string(), QUOTA_TTL);
    true
}

fn short_hash(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

/// GET /<config>/translate/<type>/<id>[/<extra>]/<lang>.(json|srt) — the app-driven translation
/// flow. The app calls the `.json` form (showing its own "Translating…" wait), which does the work
/// and returns the `.srt` URL; it then hands that URL to the engine, which fetches the now-cached
/// `.srt` instantly.
///
/// `<extra>` is the same Stremio blob the `subtitles` resource takes, carrying this stream's OSHash.
/// It is optional — the older five-segment form still resolves — but without it a translation gets
/// no Tier-1 anchor, which is the state every translation used to be in: the timing came from
/// whatever source happened to be picked and nothing ever corrected it.
///
/// A `?resync=` on the `.srt` form runs Tier 2, mirroring the subtitle proxy. It is not read on the
/// `.json` form: that one's job is to warm and hand back a URL, and Tier 2 is a user action taken
/// against the track itself.
#[allow(clippy::too_many_arguments)]
pub async fn handle_translate(
    state: &Arc<AppState>,
    headers: &HeaderMap,
    config: &str,
    id: &str,
    extra: &str,
    lang: &str,
    want_json: bool,
    resync_url: Option<String>,
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
    // The path segment arrives percent-encoded ("Brazilian%20Portuguese"), and nothing decoded it.
    // So the escape sequence went into the cache key AND was interpolated into the prompt of every
    // batch — the model was asked to translate into "Brazilian%20Portuguese", and the encoded and
    // decoded spellings of one language bought the film twice.
    let lang_seg = lang;
    let lang = httputil::percent_decode_path(lang_seg);
    // A language name, not an essay. This lands in a cache key that becomes a filename — the same
    // hazard `search_hash` already bounds `videoHash` against — and it is interpolated into the
    // prompt of every batch, so its length multiplies the bill against the viewer's own key.
    if lang.is_empty() || lang.len() > MAX_LANG {
        return httputil::text(StatusCode::BAD_REQUEST, "bad_lang");
    }
    // Case and spacing are not different languages, but they were different cache keys, and a key
    // here is a whole film's LLM bill. Canonicalized separately from the name we send the model: the
    // prompt reads better with the viewer's own spelling, and the key only has to be stable.
    let lang_key = translate::canonical_lang(&lang);
    // All-whitespace survives the length check and normalizes to nothing, which would file every such
    // request under one shared key.
    if lang_key.is_empty() {
        return httputil::text(StatusCode::BAD_REQUEST, "bad_lang");
    }
    // Checked before any network call, and scoped to the title rather than to the source file: a
    // whole film's LLM bill is not something to re-pay on every tap. The `.json` and `.srt` forms are
    // two requests, so the app's own flow retries once by design — and nothing was remembered about a
    // failure, so each attempt paid in full again, with no backoff. Same marker the sync path uses,
    // in the same namespace.
    //
    // Said differently from a fresh failure on purpose: from the outside the two are the same 502,
    // and the difference — "we just tried" versus "we are backing off" — is the first thing you want
    // to know when a translation stops working.
    let job_key = translate_fail_key(config, &imdb, season, episode, &lang_key, llm);
    let failed_recently = format!("{SYNCFAIL}{job_key}");
    if state.cache.get(&failed_recently).is_some() {
        return httputil::text(StatusCode::BAD_GATEWAY, "translation failed recently");
    }

    let Some(http) = state.http.as_ref() else {
        return httputil::text(StatusCode::SERVICE_UNAVAILABLE, "translation service unavailable");
    };
    let client = os_client(state, http, &cfg);

    // TWO searches, and the difference between them is the whole cost model of this endpoint.
    //
    // The hashed one exists to make `moviehash_match` flags appear, which is what makes a Tier-1
    // anchor findable. The UNHASHED one is what the source is chosen from, and it has to be unhashed:
    // OpenSubtitles floats hash matches to the top and returns one page, so the same title yields a
    // differently-ordered and differently-truncated list per encode. Choosing from that resolves two
    // encodes of one film to two sources, which is two `translate_body_key`s and two full-price
    // translations of the same dialogue — the exact cost keying by source file was meant to avoid.
    //
    // Both are cached under `search:`, keyed per hash — so they are two different entries and a cold
    // title pays two searches, not one. Neither spends a download credit, and the alternative is
    // choosing the source from an encode-dependent list, which spends a whole second translation.
    let hash = search_hash(extra);
    // A failed search does NOT set the marker. The marker means "a whole film's LLM bill was just
    // spent and lost, do not spend it again for ten minutes" — a search that failed cost nothing,
    // spent no tokens, and is usually a blip. Marking it made a moment's upstream trouble outlive
    // itself by ten minutes across every language the viewer tried, and made `.status` report
    // `failed` for a run that was never attempted.
    let Ok(candidates) = cached_search(state, &client, &imdb, season, episode, None).await else {
        return httputil::text(StatusCode::BAD_GATEWAY, "translation failed");
    };
    // Which file this title translates FROM is pinned once and reused.
    //
    // `translation_source` reads download counts, ratings and the trusted flag, and all three drift —
    // one new trusted upload is +400 and flips the pick outright. The search behind it is cached for
    // six hours, so a run the next day re-picks from a refreshed list, lands on a different file, and
    // that is a different `translate_body_key`: a second full-price translation of dialogue already
    // bought, with the first left orphaned under a key nothing will ask for again.
    //
    // The pin is per title rather than per language, so every language of a film translates from the
    // same source, and it is honoured only while that file is still among the candidates — if the
    // upload disappears, the next pick stands in and is pinned in its place.
    let pin_key = format!("source:{imdb}:{}:{}", season.unwrap_or(0), episode.unwrap_or(0));
    let pinned = state.cache.get(&pin_key).and_then(|v| v.parse::<i64>().ok());
    let source = pinned
        .and_then(|id| candidates.iter().find(|s| s.file_id == id))
        .or_else(|| translation_source(&candidates));
    let Some(source) = source else {
        return httputil::text(StatusCode::NOT_FOUND, "no source subtitle to translate");
    };
    let source_id = source.file_id;
    state.cache.put(pin_key, source_id.to_string(), CACHE_TTL);
    let body_key = translate_body_key(source_id, &lang_key, llm);

    // The hashed list is only worth asking for when there is a hash AND auto-sync is on: without
    // either there is no anchor to find, and the answer would be the list we already have.
    let anchored = match hash.as_deref().filter(|_| cfg.auto_sync) {
        // Propagated, not swallowed. Treating a failed search as "no anchor" makes `ref_id` depend
        // on whether a round trip happened to succeed — so the `.json` call could align and cache
        // under `body_key:ref:R` while the `.srt` call that follows computes `None`, keys on
        // `body_key`, and is served the UNALIGNED body while the aligned one sits unused.
        Some(h) => match cached_search(state, &client, &imdb, season, episode, Some(h)).await {
            Ok(s) => s,
            Err(_) => return httputil::text(StatusCode::BAD_GATEWAY, "translation failed"),
        },
        None => Vec::new(),
    };
    // Whether the SOURCE is hash-matched to this encode is a fact about the hashed list, and the
    // source was deliberately chosen from the other one — so it is looked up rather than read off
    // the candidate, whose `hash_match` is false by construction.
    let source_in_sync = anchored.iter().any(|s| s.file_id == source_id && s.hash_match);

    // Tier 2 is a user action against the track itself, so it is only read on the `.srt` form — the
    // `.json` form's job is to warm and hand back a URL. Vetted before it can reach a cache key, for
    // the reason `handle_subtitle_file` gives.
    let resync_url = match resync_url.filter(|_| !want_json) {
        Some(u) if is_safe_resync_url(&u).await => Some(u),
        // Never the URL itself: a stream target carries the provider's token.
        Some(_) => {
            eprintln!("translate: refusing unsafe resync target for {imdb}");
            None
        }
        None => None,
    };
    // The translated body inherits its source's timing, so it needs the same Tier-1 correction the
    // source itself would get from the picker: nothing when the source is already hash-matched to
    // this encode, and otherwise the anchor.
    let ref_id = match source_in_sync {
        true => None,
        false => tier1_reference(&anchored).filter(|&r| r != source_id),
    };
    let cache_key = sync_cache_key(&body_key, &resync_url, ref_id);

    // Read once, not twice. Asking again on the settled path could miss what the first read saw —
    // LRU eviction and TTL expiry both happen between two reads — and answer "not translated" for a
    // translation that exists and had just been confirmed.
    if let Some(settled) = state.cache.get(&cache_key) {
        if !want_json {
            return httputil::srt(settled);
        }
    } else {
        // The expensive half: the translated text, cached against the source file so every encode of
        // this film reuses it.
        let translated = match state.cache.get(&body_key) {
            Some(body) => body,
            None => {
                // One film, one bill. Two devices on the same title — or a second tap during a run
                // that legitimately takes minutes — each used to start their own full translation,
                // because a cache only collapses work that has already finished.
                let _flight = state.inflight.acquire(&body_key).await;
                // The marker was read at the top of this handler, before the wait. Everything queued
                // behind a run that then FAILED would sail past that stale read and each start its
                // own full-price translation, with no backoff — N waiters, N films, on the viewer's
                // own key. The marker exists to stop exactly that, so it is read again now that we
                // know how the wait ended. `sync_and_cache` re-checks its marker for this reason.
                if state.cache.get(&failed_recently).is_some() {
                    return httputil::text(StatusCode::BAD_GATEWAY, "translation failed recently");
                }
                match state.cache.get(&body_key) {
                    // Produced while we waited. This is the branch the whole guard exists for.
                    Some(body) => body,
                    // Charged here and nowhere else: this is the one path that starts a run, and it
                    // is already behind the cache check and the single-flight guard, so nothing that
                    // merely waited for someone else's work is counted against the allowance.
                    None if !charge_translation(state, config) => {
                        eprintln!("translate: {imdb} → {lang} refused, install is over its daily allowance");
                        return httputil::text(
                            StatusCode::TOO_MANY_REQUESTS,
                            "translation allowance for today is used up",
                        );
                    }
                    None => match produce_translation(state, &client, llm, source.file_id, &lang, &body_key, &job_key).await {
                        Ok(body) => body,
                        Err(e) => {
                            // Log the detail (no key in these strings); hand the client a generic
                            // message rather than echoing a raw upstream error body.
                            eprintln!("translate: {imdb} → {lang} failed: {e}");
                            state.cache.put(failed_recently, "1".into(), SYNC_RETRY_TTL);
                            return httputil::text(StatusCode::BAD_GATEWAY, "translation failed");
                        }
                    },
                }
            }
        };
        // The cheap half. Run here even for the `.json` form so the engine's follow-up fetch is a
        // cache hit rather than an ffsubsync spawn with the viewer waiting on it.
        let resp = sync_and_cache(
            state,
            &client,
            cache_key,
            translated,
            ref_id,
            resync_url,
            &format!("translation of {imdb} → {lang_key}"),
        )
        .await;
        if !want_json {
            return resp;
        }
    }

    let base = self_base(state, headers, config);
    // The segment as it arrived, not the decoded name: this is a URL, and handing back a decoded
    // "Brazilian Portuguese" would put a raw space in it. The client fetching this lands on the same
    // cache key it just warmed, whichever spelling it used — and it must carry the same extras, or
    // that fetch would resolve a different source and a different anchor.
    let url = match extra.is_empty() {
        true => format!("{base}/translate/{}/{}/{}.srt", type_of(season), id, lang_seg),
        false => format!("{base}/translate/{}/{}/{}/{}.srt", type_of(season), id, extra, lang_seg),
    };
    httputil::json(StatusCode::OK, &json!({ "url": url }), "no-store")
}

/// GET /<config>/translate/<type>/<id>[/<extra>]/<lang>.status — how far a running translation has
/// got, so the app can draw a bar instead of a spinner while the `.json` call it made on another
/// connection is still working.
///
/// Answered from the request alone: no search, no upstream call, no cache write. A poll arrives every
/// second or so while the expensive thing runs, and it must not itself become expensive — which is
/// why progress is filed under the title-scoped job key rather than the key the translation lands on.
///
/// `working` means a run is in progress in THIS process. Anything else is `idle`, `done` or `failed`,
/// and a client that gets `idle` should simply keep waiting on its `.json`: a run that has not
/// reached its first completed batch, or one being carried out by another instance, looks the same
/// from here and is not worth inventing a state for.
pub async fn handle_translate_status(
    state: &Arc<AppState>,
    config: &str,
    id: &str,
    lang: &str,
) -> Response<Body> {
    let Some(cfg) = userconfig::decode(state.config_keyring.as_ref(), config) else {
        return httputil::json(StatusCode::BAD_REQUEST, &json!({"error": "bad_config"}), "no-store");
    };
    let Some(llm) = &cfg.llm else {
        return httputil::json(StatusCode::BAD_REQUEST, &json!({"error": "no_llm"}), "no-store");
    };
    let Some((imdb, season, episode)) = parse_id(id) else {
        return httputil::json(StatusCode::BAD_REQUEST, &json!({"error": "bad_id"}), "no-store");
    };
    let lang = httputil::percent_decode_path(lang);
    if lang.is_empty() || lang.len() > MAX_LANG {
        return httputil::json(StatusCode::BAD_REQUEST, &json!({"error": "bad_lang"}), "no-store");
    }
    let lang_key = translate::canonical_lang(&lang);
    if lang_key.is_empty() {
        return httputil::json(StatusCode::BAD_REQUEST, &json!({"error": "bad_lang"}), "no-store");
    }
    let job_key = translate_fail_key(config, &imdb, season, episode, &lang_key, llm);

    let body = if let Some((done, total)) = state.progress.get(&job_key) {
        json!({"state": "working", "done": done, "total": total})
    } else if state.cache.get(&format!("{SYNCFAIL}{job_key}")).is_some() {
        json!({"state": "failed"})
    } else {
        json!({"state": "idle"})
    };
    httputil::json(StatusCode::OK, &body, "no-store")
}

/// Translate one source subtitle into `lang` and cache the result under `body_key`. Returns the
/// translated SRT — the caller then runs the sync ladder over it.
#[allow(clippy::too_many_arguments)]
async fn produce_translation(
    state: &Arc<AppState>,
    client: &opensubtitles::Client<'_>,
    llm: &LlmConfig,
    source_file_id: i64,
    lang: &str,
    body_key: &str,
    job_key: &str,
) -> Result<String, String> {
    // Through the cache, not straight at the API. A `/download` call spends one of the viewer's
    // daily OpenSubtitles credits on the CALL, not on the file fetch — and dodging that quota is the
    // reason the proxy-and-cache design exists at all. Going direct re-paid for a file already
    // sitting under `os:{file_id}`: once per retry after the ten-minute backoff, and once more for
    // every additional target language of the same film.
    let raw = subtitle_srt(state, client, source_file_id).await?;
    let cues = srt::parse(&raw);
    if cues.is_empty() {
        return Err("source subtitle was empty".into());
    }
    // The cache doubles as the batch store: a run that dies at cue 1100 of 1200 leaves the 1100
    // behind, so the retry the viewer is about to make re-buys only what actually failed.
    //
    // Progress is published under the JOB key rather than the body key, so `.status` can answer a
    // poll from the request alone — deriving the body key needs a search, and a poll happens every
    // second while the expensive thing runs.
    let reporter = state.progress.start(job_key, cues.len());
    let translated = translate::translate(client.http, llm, &cues, lang, &state.cache, &|done, total| {
        reporter.set(done, total);
    })
    .await;
    // Dropped however this ends — returned, failed, or the request cancelled out from under us
    // mid-run. That third case is the one a matching `clear` call could not cover, and it is the one
    // that leaves `.status` insisting a dead run is still working.
    drop(reporter);

    let body = srt::serialize(&translated?);
    state.cache.put(body_key.to_string(), body.clone(), CACHE_TTL);
    Ok(body)
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

    /// The translation source is chosen as TEXT, not as a fit to this encode. If the hash match got
    /// a vote here, two encodes of one film would resolve to two different sources and each would buy
    /// a separate full-price translation of the same dialogue — against the viewer's own key.
    #[test]
    fn the_translation_source_does_not_depend_on_the_encode() {
        let good = Subtitle { downloads: 50_000, from_trusted: true, ..sub(1, false) };
        let poor = Subtitle { downloads: 2, ..sub(2, false) };
        // One encode hash-matches the poor sub; another hash-matches nothing. Same answer both times.
        let hashes_the_poor_one = [Subtitle { hash_match: true, ..poor.clone() }, good.clone()];
        let hashes_nothing = [poor.clone(), good.clone()];
        assert_eq!(translation_source(&hashes_the_poor_one).unwrap().file_id, 1);
        assert_eq!(translation_source(&hashes_nothing).unwrap().file_id, 1);

        // With no English at all, the best of what exists — and a machine/AI sub is the last resort,
        // never the first, so we don't translate a translation.
        let spanish = Subtitle { lang: "es".into(), downloads: 10, ..sub(3, false) };
        let ai_english = Subtitle { ai_translated: true, downloads: 999_999, ..sub(4, false) };
        assert_eq!(translation_source(&[spanish.clone(), ai_english.clone()]).unwrap().file_id, 3);
        // An English sub still wins when there is one, even a modest one.
        assert_eq!(translation_source(&[spanish, ai_english, poor]).unwrap().file_id, 2);

        assert!(translation_source(&[]).is_none());
    }

    /// A translated body carries the same tier namespacing a downloaded one does, hanging off the
    /// translate key instead of `os:{id}`. The unaligned translation and its aligned variants must
    /// not collide, and none of them may look like a retry marker.
    #[test]
    fn a_translated_body_carries_the_same_tier_namespacing() {
        let llm = LlmConfig {
            provider: userconfig::Provider::OpenAI,
            model: "gpt-4o-mini".into(),
            api_key: "k".into(),
        };
        let base = translate_body_key(42, "SV", &llm);
        let raw = sync_cache_key(&base, &None, None);
        let aligned = sync_cache_key(&base, &None, Some(9));
        let resynced = sync_cache_key(&base, &Some("http://host/s.mkv".into()), None);
        assert_eq!(raw, base, "the unaligned body is the base key itself");
        assert_ne!(raw, aligned);
        assert_ne!(raw, resynced);
        assert_ne!(aligned, resynced);
        for key in [&raw, &aligned, &resynced] {
            assert!(BODY_PREFIXES.iter().any(|p| key.starts_with(p)), "unnamespaced body key: {key}");
            assert!(!key.starts_with(SYNCFAIL), "a body key must never look like a marker: {key}");
        }

        // Two encodes of one film: the same translated text, two different anchors. One LLM bill and
        // two cheap alignments — which is the whole reason the text key names a source file rather
        // than a title.
        let encode_a = sync_cache_key(&base, &None, Some(11));
        let encode_b = sync_cache_key(&base, &None, Some(22));
        assert_ne!(encode_a, encode_b);
        assert!(encode_a.starts_with(&base) && encode_b.starts_with(&base));

        // A different source, or a different model, is a different translation.
        assert_ne!(translate_body_key(43, "SV", &llm), base);
        let bigger = LlmConfig { model: "gpt-4o".into(), ..llm.clone() };
        assert_ne!(translate_body_key(42, "SV", &bigger), base);
    }

    /// A subtitle is never its own reference. `tier1_ref_for` refuses that when it builds a URL, but
    /// `ref` arrives on the query string and anyone can name it — and `?ref=5` on file 5 spawned a
    /// tier binary to align a file to itself: a subprocess on the one runtime thread, for a
    /// guaranteed no-op, cached afterwards under a key that claims an alignment happened.
    #[test]
    fn a_subtitle_is_never_its_own_reference() {
        // What the handler does with the query value before it reaches a key.
        let vetted = |file_id: i64, r: Option<i64>| r.filter(|&r| r != file_id);
        assert_eq!(vetted(5, Some(5)), None, "a file was accepted as its own reference");
        assert_eq!(vetted(5, Some(9)), Some(9), "a real reference was rejected");
        assert_eq!(vetted(5, None), None);
        // And with it refused, the request keys on the plain body rather than claiming an alignment.
        assert_eq!(sync_cache_key(&os_base_key(5), &None, vetted(5, Some(5))), "os:5");
    }

    #[test]
    fn sync_cache_key_separates_raw_ref_and_resync() {
        let raw = sync_cache_key(&os_base_key(5), &None, None);
        let aligned = sync_cache_key(&os_base_key(5), &None, Some(9));
        let resynced = sync_cache_key(&os_base_key(5), &Some("http://host/s.mkv".into()), None);
        assert_eq!(raw, "os:5");
        assert_eq!(aligned, "os:5:ref:9");
        assert!(resynced.starts_with("os:5:resync:"));
        // The three variants must never collide — else an aligned sub is served from the raw entry.
        assert_ne!(raw, aligned);
        assert_ne!(raw, resynced);
        assert_ne!(aligned, resynced);
        // resync (Tier-2) takes precedence over a ref (Tier-1) when both are present.
        let both = sync_cache_key(&os_base_key(5), &Some("http://host/s.mkv".into()), Some(9));
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
        assert_eq!(sync_cache_key(&os_base_key(5), &vetted, Some(9)), "os:5:ref:9");
        assert_eq!(sync_cache_key(&os_base_key(5), &vetted, None), "os:5");
        // Two different targets stay distinct, so one stream's alignment is never served for another.
        assert_ne!(
            sync_cache_key(&os_base_key(5), &Some("http://host/a.mkv".into()), None),
            sync_cache_key(&os_base_key(5), &Some("http://host/b.mkv".into()), None)
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
            sync_cache_key(&os_base_key(5), &None, None),
            sync_cache_key(&os_base_key(5), &None, Some(9)),
            sync_cache_key(&os_base_key(5), &Some("http://host/a.mkv".into()), None),
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
            // Port 1 refuses instantly. These cases are about the handler's own guards — the marker,
            // the language bound, the allowance — and every one of them has to get past a search
            // first. Pointed at the real API root they made a live request to api.opensubtitles.com
            // on every `cargo test`, which passed whether it 401'd or the network was down, so the
            // dependency was invisible.
            os_api_base: "http://127.0.0.1:1".into(),
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
        // From the real key builder, not a literal: `lang` is canonicalized on the way into the key,
        // so a literal silently stopped addressing the same entry the handler reads.
        let cache_key = translate_fail_key(&config, "tt0111161", None, None, &translate::canonical_lang("Swedish"), llm);
        state.cache.put(format!("{SYNCFAIL}{cache_key}"), "1".into(), SYNC_RETRY_TTL);

        let resp = handle_translate(
            &state,
            &HeaderMap::new(),
            &config,
            "tt0111161",
            "",
            "Swedish",
            false,
            None,
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

        // The marker's value must never reach the client as a subtitle.
        let body = String::from_utf8_lossy(
            &http_body_util::BodyExt::collect(resp.into_body()).await.unwrap().to_bytes(),
        )
        .to_string();
        assert_ne!(body.trim(), "1", "the retry marker was served as the response body");
        // "recently" is what says the marker short-circuited rather than a fresh attempt failing.
        // The config's OpenSubtitles key is a test string, so an attempt that got as far as the
        // upstream would ALSO come back 502 — the status alone proves nothing, only the body does.
        assert!(body.contains("recently"), "the marker did not short-circuit; body: {body}");
    }

    /// Spellings of one language are one cache key. A translate key is a whole film's LLM bill
    /// against the viewer's own provider account, and "Swedish" / "swedish" / " Swedish " / "sv"
    /// each bought that film separately — no repeat request could ever hit the cache. The encoded
    /// spelling counted too: the segment was never percent-decoded, so "Brazilian%20Portuguese" was
    /// a fifth key AND the language name the model was handed in every batch prompt.
    #[test]
    fn spellings_of_one_language_share_a_cache_key() {
        let llm = LlmConfig {
            provider: userconfig::Provider::OpenAI,
            model: "gpt-4o-mini".into(),
            api_key: "k".into(),
        };
        let config = config_segment();
        let key = |lang: &str| {
            let decoded = httputil::percent_decode_path(lang);
            translate_fail_key(&config, "tt0111161", None, None, &translate::canonical_lang(&decoded), &llm)
        };
        // The body key is keyed by source file rather than by title, but it carries the same
        // canonicalized language, so the collapsing has to hold there too — that is the key the
        // film's text actually lives under.
        let body_key = |lang: &str| {
            let decoded = httputil::percent_decode_path(lang);
            translate_body_key(77, &translate::canonical_lang(&decoded), &llm)
        };
        assert_eq!(body_key("sv"), body_key("Swedish"));
        assert_ne!(body_key("Swedish"), body_key("Finnish"));

        let swedish = key("Swedish");
        for same in ["swedish", "SWEDISH", "  Swedish  ", "sv", "SV"] {
            assert_eq!(key(same), swedish, "{same:?} was billed as a separate film");
        }
        // Percent-encoding is a spelling of the same language too.
        assert_eq!(key("Brazilian%20Portuguese"), key("brazilian portuguese"));

        // Different languages must still be different films.
        assert_ne!(key("Finnish"), swedish);
        // And a language the table doesn't carry keeps its own identity rather than collapsing into
        // some neighbour — the whole reason `canonical_lang` refuses to guess a code.
        assert_ne!(key("Hebrew"), key("Thai"));
        assert_ne!(key("Brazilian Portuguese"), key("Portuguese"));
    }

    /// The allowance bounds BREADTH, which nothing else did: every other limit caps one run, and an
    /// install URL in the wrong hands spends the viewer's own provider account one fresh film at a
    /// time. It is charged per install per day, and two installs do not share one.
    #[test]
    fn the_daily_allowance_is_per_install_and_per_day() {
        let state = state("quota");
        let one = config_segment();
        let two = format!("{}x", config_segment());

        for i in 0..DAILY_TRANSLATIONS {
            assert!(charge_translation(&state, &one), "refused run {i} inside the allowance");
        }
        assert!(!charge_translation(&state, &one), "the allowance did not stop anything");

        // A different install has its own, and is not held back by the first one's spending.
        assert!(charge_translation(&state, &two), "one install's allowance blocked another's");

        // And the counter is per day: yesterday's key is a different key, so a spent day does not
        // carry over into the next one.
        let now = std::time::SystemTime::now();
        let yesterday = now - Duration::from_secs(86_400);
        assert_ne!(quota_key(&one, now), quota_key(&one, yesterday));
        // The hash is of the config, so the secret itself never lands in a filename.
        assert!(!quota_key(&one, now).contains(&one));
    }

    /// `.status` answers from the request alone — no search, no upstream call — because it is polled
    /// every second or so while the expensive thing runs. It reports what the run is doing, and it
    /// reports a remembered failure rather than leaving a client polling a job that will not start.
    #[tokio::test]
    async fn status_reports_the_run_without_touching_the_upstream() {
        let state = state("status");
        let config = config_segment();
        let cfg = userconfig::decode(state.config_keyring.as_ref(), &config).unwrap();
        let llm = cfg.llm.as_ref().unwrap();
        let job_key = translate_fail_key(&config, "tt0111161", None, None, &translate::canonical_lang("Swedish"), llm);

        let read = |resp: Response<Body>| async {
            let bytes = http_body_util::BodyExt::collect(resp.into_body()).await.unwrap().to_bytes();
            serde_json::from_slice::<Value>(&bytes).expect("status is json")
        };

        // Nothing running, nothing remembered.
        let resp = handle_translate_status(&state, &config, "tt0111161", "Swedish").await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(read(resp).await["state"], "idle");

        // A run in progress reports how far it has got.
        let reporter = state.progress.start(&job_key, 1200);
        reporter.set(340, 1200);
        let body = read(handle_translate_status(&state, &config, "tt0111161", "Swedish").await).await;
        assert_eq!(body["state"], "working");
        assert_eq!(body["done"], 340);
        assert_eq!(body["total"], 1200);

        // Another language is a different job and must not read this one's progress.
        let other = read(handle_translate_status(&state, &config, "tt0111161", "Finnish").await).await;
        assert_eq!(other["state"], "idle", "a different language saw Swedish's progress");

        // However the run ends — including cancelled mid-await, which is what the guard is for — the
        // entry goes. A remembered failure is then reported as one, so a client stops waiting for a
        // run that is being backed off rather than polling a stale "working" forever.
        drop(reporter);
        state.cache.put(format!("{SYNCFAIL}{job_key}"), "1".into(), SYNC_RETRY_TTL);
        let body = read(handle_translate_status(&state, &config, "tt0111161", "Swedish").await).await;
        assert_eq!(body["state"], "failed");

        // Spellings of one language are one job here too, or a poll would never find its own run.
        let reporter = state.progress.start(&job_key, 2);
        reporter.set(1, 2);
        let by_code = read(handle_translate_status(&state, &config, "tt0111161", "sv").await).await;
        assert_eq!(by_code["state"], "working", "a code and a name polled different jobs");
    }

    /// An over-long language name is refused before it can become a filename or a prompt. The same
    /// file already bounds `videoHash` for the first reason; the second is that `lang` goes into
    /// every batch's prompt, so its length multiplies the bill against the viewer's own key.
    #[tokio::test]
    async fn an_over_long_language_is_refused() {
        let state = state("long-lang");
        let config = config_segment();
        let long = "x".repeat(MAX_LANG + 1);
        let resp = handle_translate(&state, &HeaderMap::new(), &config, "tt0111161", "", &long, false, None).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "an over-long language was accepted");

        // Nothing was written for it — not the body key, and not the failure marker either.
        // `expect`, not `unwrap_or(0)`: a missing directory would otherwise score zero and pass this
        // assertion without ever testing anything.
        let store = state.cfg.cache_dir.join("store");
        let files = std::fs::read_dir(&store).expect("the store directory is created at boot").count();
        assert_eq!(files, 0, "an invalid request left {files} cache files behind");

        // A real language still works its way through to the upstream check.
        let resp = handle_translate(&state, &HeaderMap::new(), &config, "tt0111161", "", "Swedish", false, None).await;
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY, "a valid language must not be refused");
    }

    /// And the marker is scoped to the translation it belongs to: another language is a different
    /// job and must still be attempted.
    ///
    /// Proven by the response body, not by a marker the attempt leaves behind — a failed SEARCH no
    /// longer writes one, deliberately: a search costs nothing, and marking it made a moment's
    /// upstream trouble outlive itself by ten minutes. The body is what separates the two outcomes
    /// anyway, since both are a 502.
    #[tokio::test]
    async fn a_remembered_failure_does_not_block_a_different_translation() {
        let state = state("scoping");
        let config = config_segment();
        let cfg = userconfig::decode(state.config_keyring.as_ref(), &config).unwrap();
        let llm = cfg.llm.as_ref().unwrap();
        // Through the real builder and the real canonicalizer, so the keys the test addresses are
        // the keys the handler writes.
        let key_for =
            |lang: &str| translate_fail_key(&config, "tt0111161", None, None, &translate::canonical_lang(lang), llm);
        state.cache.put(format!("{SYNCFAIL}{}", key_for("Swedish")), "1".into(), SYNC_RETRY_TTL);

        let body_of = |resp: Response<Body>| async {
            let bytes = http_body_util::BodyExt::collect(resp.into_body()).await.unwrap().to_bytes();
            String::from_utf8_lossy(&bytes).into_owned()
        };

        // Finnish is a different job: it must get past the marker and fail on its own merits.
        let resp = handle_translate(&state, &HeaderMap::new(), &config, "tt0111161", "", "Finnish", false, None).await;
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        let finnish = body_of(resp).await;
        assert!(
            !finnish.contains("recently"),
            "Finnish was short-circuited by Swedish's marker instead of being attempted: {finnish}"
        );

        // Swedish, the marked one, is still short-circuited.
        let resp = handle_translate(&state, &HeaderMap::new(), &config, "tt0111161", "", "Swedish", false, None).await;
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        let swedish = body_of(resp).await;
        assert!(swedish.contains("recently"), "the marked language was attempted anyway: {swedish}");
    }
}
