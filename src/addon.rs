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

use crate::cache;
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
const BODY_PREFIXES: [&str; 3] = [cache::OS_NS, cache::SEARCH_NS, cache::TRANSLATE_NS];
// Search results turn over as new subs are uploaded, so a short TTL — enough to spare repeated
// round-trips when the app reopens a title, not so long that fresh uploads stay hidden.
const SEARCH_TTL: Duration = Duration::from_secs(60 * 60 * 6); // 6 hours
/// How long a failed search is remembered as failed. Short — this is a blip, not a verdict — but
/// long enough that the queue behind a single-flighted miss does not run one live search each,
/// serially, at up to the client timeout apiece.
const SEARCH_FAIL_TTL: Duration = Duration::from_secs(30);
/// How long a file the API says does not exist is remembered as not existing. A day: long enough
/// that a dead track in a picker list stops costing a download credit per playback, short enough
/// that an upload restored tomorrow is picked up.
const DEAD_FILE_TTL: Duration = Duration::from_secs(60 * 60 * 24);
/// Lifetime of a pinned translation source. Long because the pin is what keeps every language of one
/// film reading from ONE source file: that is a single metered download for the title rather than one
/// per language, and it is the whole job now.
///
/// It used to be the bill that was at stake — the body key named the source, so a re-pick produced a
/// different key and re-bought a film that was still in the cache. That is no longer true, and the
/// value did not change: `translate_body_key` names the title, so a lost pin costs a download and
/// nothing more. Kept long anyway; the pin is fifty bytes and the download is metered.
const SOURCE_PIN_TTL: Duration = Duration::from_secs(60 * 60 * 24 * 180);

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
    let mut subs = match cached_search(state, &client, config, &imdb, season, episode, hash.as_deref()).await {
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
/// is choosing the source from a list whose contents depend on the encode, which resolves two
/// encodes of one film to two sources: a second metered download, and a pin that moves under the
/// other encode every time the viewer switches between them.
///
/// Returned UNRANKED, and cached that way: ranking is filename-specific, so each caller ranks the
/// list for its own request.
#[allow(clippy::too_many_arguments)]
async fn cached_search(
    state: &Arc<AppState>,
    client: &opensubtitles::Client<'_>,
    config: &str,
    imdb: &str,
    season: Option<i64>,
    episode: Option<i64>,
    hash: Option<&str>,
) -> Result<Vec<opensubtitles::Subtitle>, String> {
    let search_key = format!(
        "{}{imdb}:{}:{}:{}",
        cache::SEARCH_NS,
        season.unwrap_or(0),
        episode.unwrap_or(0),
        hash.unwrap_or("")
    );
    if let Some(hit) = state.cache.get(&search_key).and_then(|h| serde_json::from_str(&h).ok()) {
        return Ok(hit);
    }
    // Single-flighted. Two requests arriving on a cold entry would each run a live search, and
    // OpenSubtitles returns one ordered page — so the two lists can differ and `translation_source`
    // can pick differently from each. One flight means one list, so both pick the same source.
    //
    // SHARED, deliberately — not scoped to the install like the failure marker below. Scoping it
    // seemed tidier and was the same bug at install granularity: two installs on a cold title each
    // ran their own live search and could each land on a different source.
    //
    // What that costs is now a DOWNLOAD, not a film. It used to cost the film: the body key named
    // the source, so two picks meant two keys and no later guard could collapse them. Since the
    // body is keyed by title, the guard on `body_key` is taken before any source is resolved and
    // collapses the two runs whatever they pick — so the exposure is one extra metered credit and a
    // pin that flips between two sources, which on a free tier of a handful a day is still worth a
    // flight, and one the plain picker's own downloads are drawn from.
    //
    // The cost of keeping it shared is that during an outage installs queue behind one guard and
    // then each miss their own marker, so they fail one after another rather than together. That is
    // latency, during an outage, bounded by install count. A metered quota still outranks it — but
    // it is a closer call than it was, so anyone widening this should re-weigh it rather than cite
    // a film's LLM bill, which is no longer what is at stake.
    let _flight = state.inflight.acquire(&search_key).await;
    if let Some(hit) = state.cache.get(&search_key).and_then(|h| serde_json::from_str(&h).ok()) {
        return Ok(hit);
    }
    // The flight ahead of us may have failed. Nothing caches a failed search, so without this the
    // queue behind one miss ran a live search EACH, one after another, at up to the client timeout
    // apiece — the tenth caller waiting ten times as long as it used to for the same failure.
    // Scoped to the INSTALL, unlike the result. A successful search is install-independent — the
    // same file ids for everyone — which is why the positive entry is shared. A FAILURE is not: it
    // depends on the caller's key, token and rate limit, and a 401 from one revoked key would
    // otherwise deny that title to every other install for the marker's lifetime, re-armed every
    // time the bad install polled. `translate_fail_key` learned this already.
    //
    // `get_mem` to match the `put_mem` below: going through `get` would fall through to a disk probe
    // for a key nothing ever writes to disk — a blocking ENOENT `open` per cache-miss search.
    let fail_key = format!("{SYNCFAIL}{:016x}:{search_key}", short_hash(config));
    if state.cache.get_mem(&fail_key).is_some() {
        return Err("search failed recently".to_string());
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
            state.cache.put_mem(fail_key, "1".into(), SEARCH_FAIL_TTL);
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
/// The anchor decides the `?ref=` in every URL we hand back, so it has to be the same choice on
/// every request over the same title — not merely a valid one. Ties are common here (two untrusted,
/// unrated hash matches), so the score alone does not settle it and `file_id` breaks the tie.
///
/// It used to rest on list order instead, with `Reverse` + `min_by_key` to take the first of equal
/// maxima. That held only for a list someone had already ranked, and it was a precondition the two
/// callers did not both meet: the picker ranks, and the translate path reads `cached_search`, which
/// caches deliberately UNRANKED. So one title could anchor two different ways — a second metered
/// download, a second alignment and a duplicate `:ref:` entry — and the anchor could flip again
/// whenever the six-hour search entry lapsed and the API answered in a different order. Ordering by
/// something intrinsic to the subtitle costs nothing and removes the precondition.
///
/// `None` when the search produced no hash match at all. That is the gap behind the out-of-sync
/// complaint: with no trusted anchor we deliberately do NOT align to an untrusted sub (that could
/// make timing worse), so those subs are served as-is and only the Tier-2 audio resync — a user
/// action, see `is_safe_resync_url` — can fix them.
fn tier1_reference(subs: &[opensubtitles::Subtitle]) -> Option<i64> {
    subs.iter()
        .filter(|s| s.hash_match)
        .min_by_key(|s| (std::cmp::Reverse(opensubtitles::fit_score(s, None)), s.file_id))
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
    let ref_id = vetted_ref(file_id, ref_id);
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
            eprintln!("subtitle: download of file {file_id} failed: {}", e.message());
            return httputil::text(StatusCode::BAD_GATEWAY, "upstream subtitle fetch failed");
        }
    };

    // Settled: this handler is told which reference to use, so "no ref" here means none was asked
    // for, never that we failed to find out.
    let what = format!("subtitle {file_id}");
    sync_and_cache(state, &client, cache_key, target, ref_id, resync_url, &what, true).await
}

/// A cache key turned into one safe filename component, plus a per-invocation sequence number.
///
/// Two concurrent requests for the same file must not share scratch paths — one would read the
/// other's half-written output and cache it for sixty days — which is what the sequence number is
/// for. The rest is making a filename out of something that was never one.
///
/// Every byte that is not `[A-Za-z0-9]` becomes `-`, rather than the `:` this replaced. `:` was the
/// separator the keys are built from, so it looked like the whole story, and it is not: an
/// OpenRouter model id is `openai/gpt-4o-mini`, and `translate_body_key` interpolates it verbatim.
/// The slash landed in the middle of `{tag}-target.srt`, `write_temp` creates only `work_dir` and
/// not the parent of the file, and every alignment for the install failed ENOENT — silently, since
/// a failed alignment is a `syncfail:` marker and a provisional body. That is Tier 1 and Tier 2
/// dead for one of the six providers, on its DEFAULT model, while the viewer still paid the whole
/// LLM bill for a translation served with the source's uncorrected timings.
///
/// The bound is the other half, and it is why this counts bytes rather than chars: the result is
/// `{tag}-reference.srt` in one directory, a translate key already runs to ~255 bytes with a long
/// model name, and past NAME_MAX the write fails exactly the same silent way. Mapping to ASCII
/// first makes the two agree — a multibyte model name counted 80 chars and wrote up to 320 bytes.
fn scratch_tag(cache_key: &str, seq: u64) -> String {
    let safe: String = cache_key
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .take(80)
        .collect();
    format!("{safe}-{seq}")
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
    settled: bool,
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

    let tag = scratch_tag(&cache_key, SYNC_SEQ.fetch_add(1, Ordering::Relaxed));
    // A permit, held only around work that actually spawns a binary.
    //
    // The single-flight guard above collapses two requests for the SAME alignment; twenty different
    // ones — the picker hands back a URL per subtitle — are twenty keys and were twenty concurrent
    // subprocesses on a one-thread runtime. Acquired inside these branches rather than above the
    // dispatch: the semaphore is FIFO with two permits, so taking it on the no-sync path made a
    // plain cache-miss fetch queue behind every pending alignment for work it was never going to do.
    let synced: Option<String> = if let Some(url) = resync_url {
        let _slot = state.sync_slots.acquire().await;
        // Tier 2 — audio VAD against the playing stream (opt-in; alass pulls the audio via ffmpeg).
        match state.sync.sync_to_audio(&target, &url, &tag).await {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("sync: resync of {what} failed: {e}");
                None
            }
        }
    } else if let Some(r) = ref_id {
        // Tier 1 — reference-align against the hash-matched sub (no audio needed). The reference is
        // fetched BEFORE taking a permit: that is a network download, and a permit meant to bound
        // subprocesses should not be spent waiting on OpenSubtitles.
        match subtitle_srt(state, client, r).await {
            Ok(reference) => {
                let aligned = {
                    let _slot = state.sync_slots.acquire().await;
                    state.sync.sync_to_reference(&target, &reference, &tag).await
                };
                match aligned {
                    Ok(s) => Some(s),
                    Err(e) => {
                        eprintln!("sync: aligning {what} to {r} failed: {e}");
                        None
                    }
                }
            }
            Err(e) => {
                eprintln!("sync: reference {r} for {what} unavailable: {}", e.message());
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
        // No sync was asked for, so the body we have is the answer, and it is already cached —
        // unless the caller could not determine whether an alignment was owed at all, in which case
        // this is a stand-in and must revalidate rather than being pinned for a year.
        None if settled => httputil::srt(target),
        None => httputil::srt_provisional(target),
    }
}

/// Fetch a subtitle's SRT, cached by file id (the raw, un-synced text — reused as a sync input).
async fn subtitle_srt(
    state: &Arc<AppState>,
    client: &opensubtitles::Client<'_>,
    file_id: i64,
) -> Result<String, opensubtitles::DownloadError> {
    let key = os_base_key(file_id);
    if let Some(hit) = state.cache.get(&key) {
        return Ok(hit);
    }
    // Single-flighted, because this is the one call that spends a METERED credit: OpenSubtitles
    // charges the viewer's daily download allowance on the API call itself, and the anonymous
    // allowance is a handful per day. Two devices opening the same title, or the twenty picker URLs
    // that all carry the same `?ref=`, would each buy the same file.
    // A download that just failed is not retried on every request, because every attempt SPENDS.
    // OpenSubtitles charges the daily allowance on the API call, and for a dead upload the 404
    // arrives on the CDN fetch afterwards — so a request for a file that will never resolve cost a
    // credit and made no progress, for as long as anything kept asking. The picker hands back a URL
    // per subtitle and re-ranks the same dead track every playback, so a handful of clicks emptied
    // the day's allowance and then broke every other download with it.
    if let Some(e) = remembered_failure(state, client, file_id) {
        return Err(e);
    }
    let _flight = state.inflight.acquire(&key).await;
    if let Some(hit) = state.cache.get(&key) {
        return Ok(hit);
    }
    // Re-read after the wait, the way every other flight here does. Twenty picker URLs all carry the
    // same `?ref=R` under twenty different cache keys, so all twenty flights are distinct and all
    // twenty call this for the SAME reference — they clear the check above together, queue on
    // `os:R`, and then every one of them re-issues the download on release. Twenty metered credits
    // out of a daily handful, for a file already known to be failing.
    if let Some(e) = remembered_failure(state, client, file_id) {
        return Err(e);
    }
    let body = match client.download(file_id).await {
        Ok(body) => {
            // A success clears the record. Without this the counter tallies failures across a whole
            // day with no credit for the successes between them, so a file that works nine times out
            // of ten still gets convicted on its second bad afternoon — and the verdict is the one
            // thing trusted to drop a pin shared by every install.
            forget_failure(state, file_id);
            body
        }
        Err(e) => {
            remember_failure(state, client, file_id, &e);
            return Err(e);
        }
    };
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

/// Key of the pinned translation source for one title.
///
/// Scoped to the title and nothing else: every language and every install translates the same film
/// from the same file, which is what keeps one title to one paid translation per language rather
/// than one per re-pick. Deliberately NOT under `translate:`, since it holds a file id rather than a
/// subtitle and must never be reachable as a body.
fn source_pin_key(imdb: &str, season: Option<i64>, episode: Option<i64>) -> String {
    format!("source:{imdb}:{}:{}", season.unwrap_or(0), episode.unwrap_or(0))
}

/// "This file will not download." A fact about the file, so shared by every install — and its own
/// namespace inside `syncfail:`, since `sync_and_cache`'s retry marker is `syncfail:{cache_key}` and
/// for a request that asked for no sync that key IS `os:{id}`.
fn dead_file_key(file_id: i64) -> String {
    format!("{SYNCFAIL}dl:gone:{}", os_base_key(file_id))
}

/// "This file is too big to translate." Also a fact about the file and shared, but distinct from
/// `dead_file_key`: this one downloads perfectly well, so it stays a fine subtitle for the plain
/// picker to hand over. It is only unusable as a translation SOURCE, and only the re-pick reads it.
fn oversized_file_key(file_id: i64) -> String {
    format!("{SYNCFAIL}dl:oversized:{}", os_base_key(file_id))
}

/// "Stop asking about this file for a moment." Short, shared, and it DOES gate — which is what keeps
/// a burst of requests for one failing file to a single spent credit.
fn suspect_backoff_key(file_id: i64) -> String {
    format!("{SYNCFAIL}dl:backoff:{}", os_base_key(file_id))
}

/// "This file failed at this time." The promotion counter, holding the unix second of the first
/// failure — deliberately NOT the same key as the gate above.
///
/// One key tried to be both and could not. Gating on the counter made the promotion unreachable,
/// since the attempt whose failure would confirm the strike was the attempt being refused. Not
/// gating on it made the promotion instant: twenty picker URLs all fetch the same reference, so
/// waiter one wrote the strike and waiter two confirmed it milliseconds later — two failures from
/// one incident, fabricating a verdict that is supposed to mean two occasions.
fn suspect_strike_key(file_id: i64) -> String {
    format!("{SYNCFAIL}dl:strike:{}", os_base_key(file_id))
}

/// "This credential could not fetch this file just now." Quota, a revoked key, a blip — facts about
/// one install's OpenSubtitles credential and nobody else's, so keyed by the credential rather than
/// by the install: two installs sharing a key really do share the quota that produced it.
fn unavailable_file_key(client: &opensubtitles::Client<'_>, file_id: i64) -> String {
    format!("{SYNCFAIL}dl:{:016x}:{}", short_hash(client.api_key), os_base_key(file_id))
}

/// Rebuild the failure a previous attempt recorded, so a request that cannot succeed does not spend
/// a metered credit finding that out again.
fn remembered_failure(
    state: &Arc<AppState>,
    client: &opensubtitles::Client<'_>,
    file_id: i64,
) -> Option<opensubtitles::DownloadError> {
    if state.cache.get(&dead_file_key(file_id)).is_some() {
        return Some(opensubtitles::DownloadError::Gone(format!("file {file_id} is gone (remembered)")));
    }
    // The BACKOFF gates; the strike counter deliberately does not. A burst of requests for one
    // failing file must cost one credit, not one each — but the attempt that confirms a strike has
    // to be allowed through, and it is, ten minutes later when this lapses.
    //
    // `get_mem` to match `put_mem` — going through `get` would probe a disk tier nothing writes to.
    if state.cache.get_mem(&suspect_backoff_key(file_id)).is_some() {
        return Some(opensubtitles::DownloadError::Suspect(format!(
            "file {file_id} would not download (remembered)"
        )));
    }
    if state.cache.get_mem(&unavailable_file_key(client, file_id)).is_some() {
        return Some(opensubtitles::DownloadError::Unavailable(format!(
            "file {file_id} unavailable (remembered)"
        )));
    }
    None
}

/// Seconds since the epoch. Wall clock rather than a monotonic instant because it is stored and
/// compared across requests; a backward clock step just delays a promotion.
fn unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Drop the suspicion record for a file that has just downloaded cleanly.
///
/// Only the two `Suspect` keys. A `Gone` verdict is not forgotten here: it is the API's own answer
/// about the id, or a confirmed pair, and it gates this function's caller — so reaching a success
/// with one live is not something a clean download should quietly overwrite.
fn forget_failure(state: &Arc<AppState>, file_id: i64) {
    state.cache.remove_mem(&suspect_backoff_key(file_id));
    state.cache.remove_mem(&suspect_strike_key(file_id));
}

/// Remember a download failure so the next request does not spend a credit rediscovering it.
///
/// A `Suspect` escalates on repeat: the first one is a short shared strike, and a second inside that
/// window promotes to the day-long `Gone` marker. That is what separates a CDN interstitial — which
/// clears on its own and must not touch the shared source pin — from an upload that really is junk,
/// which would otherwise be re-fetched every ten minutes for a metered credit each time.
fn remember_failure(
    state: &Arc<AppState>,
    client: &opensubtitles::Client<'_>,
    file_id: i64,
    e: &opensubtitles::DownloadError,
) {
    use opensubtitles::DownloadError::*;
    match e {
        Gone(_) => state.cache.put(dead_file_key(file_id), "1".into(), DEAD_FILE_TTL),
        // Two failures on two OCCASIONS promote. The gate is always refreshed, so a burst costs one
        // credit; the counter promotes only when the earlier failure is at least a backoff window
        // old, so the confirming attempt is one that genuinely had to be let through rather than a
        // sibling from the same instant.
        Suspect(_) => {
            state.cache.put_mem(suspect_backoff_key(file_id), "1".into(), SYNC_RETRY_TTL);
            let strike = suspect_strike_key(file_id);
            let first = state.cache.get_mem(&strike).and_then(|v| v.parse::<u64>().ok());
            match first {
                Some(then) if unix_seconds().saturating_sub(then) >= SYNC_RETRY_TTL.as_secs() => {
                    state.cache.put(dead_file_key(file_id), "1".into(), DEAD_FILE_TTL)
                }
                // Already counted, too recent to confirm — leave it, so the clock keeps running from
                // the FIRST failure rather than being pushed forward by every sibling in a burst.
                Some(_) => {}
                None => state.cache.put_mem(strike, unix_seconds().to_string(), DEAD_FILE_TTL),
            }
        }
        Unavailable(_) => {
            state.cache.put_mem(unavailable_file_key(client, file_id), "1".into(), SYNC_RETRY_TTL)
        }
    }
}

/// The `ref` a request may actually use. `tier1_ref_for` refuses a self-reference when it BUILDS a
/// URL, but this value arrives on the query string where anyone can name it: `?ref=5` on file 5
/// spawned a tier binary to align a file to itself — a subprocess, on the one runtime thread, for a
/// guaranteed no-op — and filed the result under `os:5:ref:5` as though an alignment had happened.
fn vetted_ref(file_id: i64, ref_id: Option<i64>) -> Option<i64> {
    ref_id.filter(|&r| r != file_id)
}

/// The cache key for one raw OpenSubtitles file: the unaligned body that every sync variant of it
/// hangs off. Also what `subtitle_srt` files it under.
fn os_base_key(file_id: i64) -> String {
    format!("{}{file_id}", cache::OS_NS)
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
/// Deliberately NOT the key the translated body lives under (`translate_body_key`), for the scoping
/// reason below rather than a structural one: both are answerable from the request alone now that the
/// body is keyed by title, so either could be consulted this early — but they must not be the same
/// key, because one is shared and the other cannot be. Built here rather than inline so the handler
/// and the tests cannot drift apart on it — the same reason `sync_cache_key` exists, and `lang` in
/// particular is canonicalized rather than taken verbatim (see `translate::canonical_lang`).
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
        "{}{:016x}:{imdb}:{}:{}:{lang_key}:{}:{}",
        cache::TRANSLATE_NS,
        short_hash(config),
        season.unwrap_or(0),
        episode.unwrap_or(0),
        llm.provider.tag(),
        llm.model,
    )
}

/// Cache key for one title's translated text.
///
/// Keyed by the TITLE, not by the source file it was translated from — and that is a correction of
/// the obvious design, so it is worth saying why.
///
/// Source-keying was there to stop two encodes of one film resolving to two sources and buying the
/// dialogue twice. The source pin already guarantees that: one title translates from one file, and
/// every encode and every language follows the pin. So source-keying bought nothing the pin was not
/// already providing, and it cost something serious — the pin is the ONLY pointer to that source id,
/// so if it ever moved, every language already paid for became unreachable. A source dying while
/// adding a second language re-bought the first one at full price, from a cache that still held it.
///
/// With the title as the key, a re-pick changes which file a future untranslated language reads from
/// and nothing else. Paid bodies stay reachable, whatever happens to the pin.
///
/// Keyed by provider+model too: stepping up to a bigger model to re-translate a title that read badly
/// is meant to overwrite, and it can only do that if the model is part of the identity. NOT keyed by
/// install: these are the same bytes whoever asked for them, and sharing them is deliberate.
fn translate_body_key(
    imdb: &str,
    season: Option<i64>,
    episode: Option<i64>,
    lang_key: &str,
    llm: &LlmConfig,
) -> String {
    format!(
        "{}{imdb}:{}:{}:{lang_key}:{}:{}",
        cache::TRANSLATE_NS,
        season.unwrap_or(0),
        episode.unwrap_or(0),
        llm.provider.tag(),
        llm.model,
    )
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
        // chooses. An order-dependent pick moves the pin between two sources of one title, so each
        // move is another metered download — and the two encodes of a film a viewer switches
        // between would keep taking it off each other.
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

/// Give a slot back, for a run that was charged and then spent nothing at all.
///
/// "Nothing at all" is both resources, not just the model: a request that bought a metered
/// OpenSubtitles credit keeps its slot even though no token was sent, because the daily allowance is
/// what bounds how many distinct titles can spend credits. The caller decides — see the two
/// conditions at the one call site.
fn refund_translation(state: &Arc<AppState>, config: &str) {
    let key = quota_key(config, std::time::SystemTime::now());
    let used: u64 = state.cache.get(&key).and_then(|v| v.parse().ok()).unwrap_or(0);
    state.cache.put(key, used.saturating_sub(1).to_string(), QUOTA_TTL);
}

/// "This install's provider credential was refused." Install-, provider- and model-scoped, because
/// that is the scope of the fact: the same key will refuse the next title identically.
///
/// Without it, a wrong or expired LLM key — the likeliest misconfiguration in a BYOK product, since
/// the key rides in the install URL — cost a metered subtitle download per title browsed and would
/// have drained the whole daily allowance, all for runs that sent no tokens at all. The
/// title-scoped marker cannot help: it is a different key for every film.
fn credential_refused_key(config: &str, llm: &LlmConfig) -> String {
    format!("{SYNCFAIL}llm:{:016x}:{}:{}", short_hash(config), llm.provider.tag(), llm.model)
}

/// Which title last hit an AMBIGUOUS refusal on this credential — a 400 or a 403 that might be the
/// key and might be this film's dialogue tripping a content filter.
///
/// Two such refusals on two DIFFERENT titles is the thing that tells them apart: a content filter is
/// a fact about one film, a dead key is not. Guessing per provider got it wrong twice in two rounds
/// in both directions, so this measures it instead. One title alone never arms the install-wide
/// block, so a filtered series cannot take the rest of the library down with it.
fn ambiguous_refusal_key(config: &str, llm: &LlmConfig) -> String {
    format!("{SYNCFAIL}llm-amb:{:016x}:{}:{}", short_hash(config), llm.provider.tag(), llm.model)
}

/// One film or one episode, as anything counting distinct titles has to count them.
///
/// `parse_id` splits `tt123:2:5` into three values and the first is the SERIES, so building this
/// from `imdb` alone makes every episode of a show one title — which is how the escalation below
/// came to be unable to fire across a season at all. A named function because that mistake was
/// invisible at the call site and, spelled inline, no test could catch its return.
fn title_id(imdb: &str, season: Option<i64>, episode: Option<i64>) -> String {
    format!("{imdb}:{}:{}", season.unwrap_or(0), episode.unwrap_or(0))
}

/// "This series keeps being refused." Between the per-episode marker and the install-wide one.
fn series_refused_key(config: &str, llm: &LlmConfig, imdb: &str) -> String {
    format!("{SYNCFAIL}llm-series:{:016x}:{}:{}:{imdb}", short_hash(config), llm.provider.tag(), llm.model)
}

/// How widely a credential-class refusal should be remembered.
///
/// Three scopes because there are three situations, and the middle one had no home. A dead key is
/// the install. A content filter is the SERIES — that is where it recurs, since the thing it objects
/// to is the show's dialogue — and giving it the install was collateral while giving it only the
/// episode was no throttle at all: every next episode paid a metered download, an allowance slot and
/// half a film in tokens before being refused, which is worse than the block it replaced.
enum RefusalScope {
    Install,
    Series,
    TitleOnly,
}

fn refusal_scope(
    state: &Arc<AppState>,
    config: &str,
    llm: &LlmConfig,
    title: &str,
    key_certain: bool,
    spent: bool,
) -> RefusalScope {
    if key_certain {
        return RefusalScope::Install;
    }
    // Billed work in this same run is proof the provider accepted this credential, so whatever it
    // has now objected to is the content. Never the install; the series is where it will recur.
    if spent {
        return RefusalScope::Series;
    }
    // Nothing billed and the status is ambiguous, so this rests on corroboration — and `spent` is a
    // weaker witness here than it looks. A content filter that objects to a show's dialogue objects
    // to the GLOSSARY sample too, which is drawn from that same dialogue, so the filtered case often
    // reaches here having billed nothing at all rather than arriving above.
    match ambiguous_refusal_escalates(state, config, llm, title) {
        Corroboration::Unrelated => RefusalScope::Install,
        Corroboration::SameSeries => RefusalScope::Series,
        Corroboration::None => RefusalScope::TitleOnly,
    }
}

/// Should an ambiguous refusal block the whole install? Only once a SECOND, different title has
/// refused the same way inside the window.
///
/// The title is the episode, not the series. Keyed on the series id, every episode of one show read
/// as the same title and the escalation could never fire — while each episode still resolved its own
/// source and spent its own metered download, so a dead key on a season was unbounded. That is the
/// same browsing pattern the ambiguity itself is modelled on, and it needs to cut both ways: one
/// film's dialogue must not block the library, and a key that fails across two episodes must.
///
/// The memory of the last refusal deliberately outlives the block it can arm. Sharing a lifetime
/// with it meant that every time the block lapsed the evidence had lapsed too, so re-arming took two
/// fresh titles and two more metered downloads, every window.
///
/// What the second title IS decides how far it reaches. Two episodes of one show corroborate
/// nothing about the key — a content filter objects to a show, and a season is the browsing pattern
/// that produces two refusals fastest — so they arm the series. It takes an unrelated title to
/// implicate the credential. Reading any second title as the install let a filtered series take the
/// library down ten minutes at a time, which is what the doc above says must not happen; reading a
/// season as one title, which the previous shape did, left it with no throttle at all.
fn ambiguous_refusal_escalates(
    state: &Arc<AppState>,
    config: &str,
    llm: &LlmConfig,
    title: &str,
) -> Corroboration {
    let seen = ambiguous_refusal_key(config, llm);
    let verdict = match state.cache.get_mem(&seen) {
        // The same title repeating is still one film's dialogue, however many times it repeats.
        Some(prev) if prev == title => Corroboration::None,
        Some(prev) if series_of(&prev) == series_of(title) => Corroboration::SameSeries,
        Some(_) => Corroboration::Unrelated,
        None => Corroboration::None,
    };
    state.cache.put_mem(seen, title.to_string(), 2 * SYNC_RETRY_TTL);
    verdict
}

/// What a second ambiguous refusal corroborates.
enum Corroboration {
    /// Nothing yet, or the same title again.
    None,
    /// Another episode of the show that refused last time.
    SameSeries,
    /// A different show or a film — the part that implicates the credential rather than a script.
    Unrelated,
}

/// The series half of a `title_id`. `imdb` ids carry no colon, so the first segment is the show.
fn series_of(title: &str) -> &str {
    title.split(':').next().unwrap_or(title)
}

/// Is today's allowance already gone? A read, not a charge.
///
/// Consulted before the source download so a refusal is free. Charging only after the download —
/// which is right, since a source that will not fetch must not cost a slot — meant every request
/// past the fiftieth bought a metered OpenSubtitles credit and then returned 429. Nothing caps the
/// number of distinct titles a viewer browses, so that was unbounded in the dimension that matters,
/// and it emptied the download quota that the plain subtitle picker also depends on.
fn allowance_used_up(state: &Arc<AppState>, config: &str) -> bool {
    let key = quota_key(config, std::time::SystemTime::now());
    let used: u64 = state.cache.get(&key).and_then(|v| v.parse().ok()).unwrap_or(0);
    used >= DAILY_TRANSLATIONS
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

/// A stable digest for values that go into keys which outlive the process — the daily allowance, the
/// per-install failure marker, the resync variant of a cache key. Deliberately not `DefaultHasher`:
/// std may change that algorithm between compiler releases, and a toolchain bump would then reset
/// every allowance and move every marker at once. See `httputil::stable_hash`.
fn short_hash(s: &str) -> u64 {
    httputil::stable_hash(s.as_bytes())
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

    let hash = search_hash(extra);

    // Which file this title translates FROM is decided once and then pinned.
    //
    // `translation_source` reads download counts, ratings and the trusted flag, and all three drift —
    // one new trusted upload is +400 and flips the pick outright — while the search behind it is only
    // cached for six hours. The pin keeps every language of a film reading from one file, which is
    // one download rather than one per language, and keeps the choice stable rather than following
    // whatever the list looked like that afternoon.
    //
    // It is no longer load-bearing for the LLM bill: the body is keyed by title, so losing the pin
    // costs a re-pick and a download, never a re-translation.
    //
    // READ here, but RESOLVED only if a translation actually has to be produced. Since the body is
    // keyed by title, nothing on the serving path needs a source id except the decision of whether to
    // align — so a pin miss no longer drags a live search, fifty dead-file probes and a blocking pin
    // write onto a request whose translation is already bought and cached, and can no longer refuse
    // one with a 502 when that search fails or a 404 when every candidate is filtered out.
    let pin_key = source_pin_key(&imdb, season, episode);
    let pinned = state.cache.get(&pin_key).and_then(|v| v.parse::<i64>().ok());
    let body_key = translate_body_key(&imdb, season, episode, &lang_key, llm);

    // The hashed list is only worth asking for when there is a hash AND auto-sync is on: without
    // either there is no anchor to find, and the answer would be the list we already have.
    // A failed search here means no anchor, and no anchor means carry on without one: `ref_id`
    // becomes `None`, `sync_cache_key` keys on `body_key` itself, and that IS the honest identity of
    // an unaligned body — not a key an aligned one should have owned. Nothing is mis-filed, and a
    // later request that does find an anchor misses `body_key:ref:R` and aligns then.
    //
    // Two stricter versions of this were both worse. Refusing outright returned 502 for a title
    // already bought and cached; returning early with whatever happened to be cached could not
    // translate a cold title at all during a blip. Neither wrote a backoff, so every client retry
    // was another live search.
    // `anchor_unknown` is the difference between "there is no anchor" and "we could not find out
    // whether there is one". Both produce `ref_id = None` and both key on `body_key`, which is the
    // honest identity of an unaligned body — but only the first is SETTLED. Serving the second as
    // `immutable` pins the client to a mis-timed track for a year at a deterministic URL, which is
    // the failure `srt_provisional` exists for, moved from the server cache to the client's.
    let mut anchor_unknown = false;
    let anchored = match hash.as_deref().filter(|_| cfg.auto_sync) {
        Some(h) => match cached_search(state, &client, config, &imdb, season, episode, Some(h)).await {
            Ok(subs) => subs,
            Err(_) => {
                anchor_unknown = true;
                Vec::new()
            }
        },
        None => Vec::new(),
    };
    // Which alignment a given source needs. A source that is itself hash-matched to this encode is
    // already correctly timed, so the translated body carrying its timings is too.
    //
    // Taken as a function because it has to be answered twice. Before a source is known — a pin
    // miss — the honest answer is "assume not aligned", which is what lets the cached-body probe
    // below happen without resolving a source at all. But answering it that way and then STOPPING
    // there bought real work for nothing: `ref_id` became `Some(R)`, and `sync_and_cache` fetches
    // that reference through the one call in the program that spends a metered download credit,
    // then runs a tier binary. When the source turned out to be hash-matched after all, both were
    // no-ops, and the entry they produced was filed under a key the very next request — now pinned,
    // now answering `true` — never asks for again.
    let align_for = |source: Option<i64>| -> Option<i64> {
        let in_sync = source.is_some_and(|id| anchored.iter().any(|s| s.file_id == id && s.hash_match));
        match in_sync {
            true => None,
            false => tier1_reference(&anchored).filter(|&r| Some(r) != source),
        }
    };

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
    // source itself would get from the picker.
    let ref_id = align_for(pinned);
    let cache_key = sync_cache_key(&body_key, &resync_url, ref_id);

    // Read once, not twice. Asking again on the settled path could miss what the first read saw —
    // LRU eviction and TTL expiry both happen between two reads — and answer "not translated" for a
    // translation that exists and had just been confirmed.
    if let Some(settled) = state.cache.get(&cache_key) {
        if !want_json {
            // `immutable` only when this really is the answer. With the anchor merely unknown the
            // body may well be superseded within the thirty seconds the search marker lasts, and a
            // client told `immutable` will not come back for a year.
            return match anchor_unknown {
                true => httputil::srt_provisional(settled),
                false => httputil::srt(settled),
            };
        }
    } else {
        // The expensive half: the translated text. `used_source` carries back which file it came
        // from, so the alignment decision can be re-answered from a source that is actually known.
        let mut used_source = pinned;
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
                    // Only here, with every cache read behind us, is a source actually needed — so
                    // only here is one resolved. A pin miss costs a live search, a probe per
                    // candidate and a blocking pin write, and none of that belongs on a request whose
                    // translation already exists.
                    None => {
                        // Refuse before resolving a source, not just before downloading one.
                        // Resolution is two live searches, a blocking disk probe per candidate — the
                        // list is a page of fifty — and a blocking pin write, all for a request that
                        // is about to be refused, on the one thread serving every connection. The
                        // charge itself still happens after the download, so a source that will not
                        // fetch continues to cost no slot.
                        if allowance_used_up(state, config) {
                            eprintln!("translate: {imdb} → {lang} refused, install is over its daily allowance");
                            return httputil::text(
                                StatusCode::TOO_MANY_REQUESTS,
                                "translation allowance for today is used up",
                            );
                        }
                        // A refused credential is about the install, not this title, so the
                        // title-scoped marker cannot catch it — every film is a fresh key.
                        //
                        // But it belongs HERE, beside the allowance check and behind every cache
                        // read, for the same reason that one does. Checked at the top of the handler
                        // it refused translations that were already bought and cached: serving one
                        // makes no provider call at all, so a dead key was making the install's
                        // entire existing library unavailable, re-armed by every new title browsed.
                        if state.cache.get_mem(&credential_refused_key(config, llm)).is_some() {
                            return httputil::text(
                                StatusCode::BAD_GATEWAY,
                                "the AI provider refused this key",
                            );
                        }
                        // And the series-scoped one, for the refusal that billed work before it
                        // arrived — a content filter, not a credential. Same placement and the same
                        // reason: behind the cache reads, so episodes already bought still serve.
                        if state.cache.get_mem(&series_refused_key(config, llm, &imdb)).is_some() {
                            return httputil::text(
                                StatusCode::BAD_GATEWAY,
                                "the AI provider refused to translate this title",
                            );
                        }
                        let source_id = match pinned {
                            Some(id) => id,
                            None => {
                                // Unhashed deliberately. OpenSubtitles floats hash matches up and
                                // returns one page, so a hashed list is ordered and truncated
                                // differently per encode — choosing from it resolves two encodes of
                                // one film to two sources, so the pin flips whenever the viewer
                                // switches encode and each flip is another metered download.
                                //
                                // A failed search does NOT set the failure marker: a search costs
                                // nothing, and marking it made a blip outlive itself by ten minutes
                                // across every language the viewer tried.
                                let Ok(candidates) =
                                    cached_search(state, &client, config, &imdb, season, episode, None).await
                                else {
                                    return httputil::text(StatusCode::BAD_GATEWAY, "translation failed");
                                };
                                // Skip what is already known not to download, or the loop has no
                                // exit: a dead source is unpinned, the deterministic re-pick chooses
                                // the same file, pins it again, and refuses again. Same for one too
                                // big to translate — it downloads, so it is a fine subtitle to serve
                                // as-is, but as a source it fails identically every time.
                                let usable: Vec<opensubtitles::Subtitle> = candidates
                                    .into_iter()
                                    .filter(|s| state.cache.get(&dead_file_key(s.file_id)).is_none())
                                    .filter(|s| state.cache.get(&oversized_file_key(s.file_id)).is_none())
                                    .collect();
                                let Some(source) = translation_source(&usable) else {
                                    return httputil::text(
                                        StatusCode::NOT_FOUND,
                                        "no source subtitle to translate",
                                    );
                                };
                                state.cache.put(
                                    pin_key.clone(),
                                    source.file_id.to_string(),
                                    SOURCE_PIN_TTL,
                                );
                                source.file_id
                            }
                        };
                        used_source = Some(source_id);
                        // A source already known to be failing must not cost an allowance slot to
                        // rediscover that, and one confirmed dead must not stay pinned — otherwise
                        // the ten-minute marker lapses, the same source is tried again, and the title
                        // burns one of the fifty daily translations per attempt.
                        if let Some(e) = remembered_failure(state, &client, source_id) {
                            // `Gone` means the API named the id, or two separate occasions confirmed
                            // it. A single `Suspect` is not `Gone`, so a transient cannot unpin.
                            if matches!(e, opensubtitles::DownloadError::Gone(_)) {
                                eprintln!("translate: unpinning {imdb} — source {source_id} will not download");
                                state.cache.remove(&pin_key);
                            }
                            // Backed off like any other failure, or the refusal is free to repeat and
                            // each repeat re-runs the search and the pin write.
                            state.cache.put(failed_recently, "1".into(), SYNC_RETRY_TTL);
                            return httputil::text(StatusCode::BAD_GATEWAY, "translation source unavailable");
                        }
                        match produce_translation(state, &client, llm, config, source_id, &lang, &body_key, &job_key).await {
                        Ok(body) => {
                            // Refresh the pin on the path that actually produced a translation, so
                            // its lifetime and mtime do not stay frozen at the first one. Losing it
                            // is no longer expensive — the body is keyed by title, so a re-pick
                            // cannot orphan anything paid for — but a stable pin still keeps every
                            // language of a film reading from one source, which is one download
                            // rather than several. One blocking write per real translation, not per
                            // request, which is what the per-request version cost.
                            state.cache.put(pin_key.clone(), source_id.to_string(), SOURCE_PIN_TTL);
                            body
                        }
                        // Nothing was spent — the allowance was checked at the last moment before
                        // the first token, so this costs neither a slot nor a backoff marker.
                        Err(TranslationFailure::Allowance) => {
                            eprintln!("translate: {imdb} → {lang} refused, install is over its daily allowance");
                            return httputil::text(
                                StatusCode::TOO_MANY_REQUESTS,
                                "translation allowance for today is used up",
                            );
                        }
                        // The provider refused the key itself. Remembered install-wide rather than
                        // per title, because that is the scope of the fact — and no per-title
                        // marker, since the title is innocent and will work once the key does.
                        Err(TranslationFailure::Credential { message, key_certain, spent }) => {
                            eprintln!("translate: provider refused this install's credential: {message}");
                            // The install-wide block only when the status can ONLY mean the key.
                            // A 400 or a 403 might be this film's dialogue tripping a content
                            // filter, and blocking the install on that lets one series take every
                            // other title down with it, ten minutes at a time, as the viewer works
                            // through the episodes. The per-title marker below covers that case.
                            // Certain: block the install at once. Ambiguous: block it only once a
                            // SECOND title has refused the same way, which is what separates a dead
                            // key from one film's dialogue upsetting a content filter.
                            // A refusal that arrives AFTER a billed batch cannot be about the key:
                            // the provider accepted this credential minutes ago, in this run. That
                            // is a measurement, where `key_certain` is a guess and the two-title
                            // heuristic is a proxy — so it settles the ambiguous case outright.
                            //
                            // It settles it as the SERIES, not as nothing. Demoting it to the
                            // per-title marker alone left a moderation-filtered show with no
                            // throttle at all: every next episode paid a metered download, an
                            // allowance slot and half a film in tokens to rediscover the same
                            // refusal — dearer than the install-wide block it was meant to spare.
                            match refusal_scope(
                                state,
                                config,
                                llm,
                                &title_id(&imdb, season, episode),
                                key_certain,
                                spent,
                            ) {
                                RefusalScope::Install => state.cache.put_mem(
                                    credential_refused_key(config, llm),
                                    "1".into(),
                                    SYNC_RETRY_TTL,
                                ),
                                RefusalScope::Series => state.cache.put_mem(
                                    series_refused_key(config, llm, &imdb),
                                    "1".into(),
                                    SYNC_RETRY_TTL,
                                ),
                                RefusalScope::TitleOnly => {}
                            }
                            // The per-title marker too, even though this is meant to be a fact about
                            // the install. `is_credential_refusal` reads a 400/403 as one, and those
                            // are not exclusively about credentials — OpenRouter answers 403 when a
                            // model's moderation trips, and this file already notes that film
                            // dialogue trips content filters routinely. Without a per-title marker
                            // such a title re-arms the install-wide one every ten minutes forever,
                            // and takes every other title down with it each time.
                            state.cache.put(failed_recently, "1".into(), SYNC_RETRY_TTL);
                            return httputil::text(
                                StatusCode::BAD_GATEWAY,
                                "the AI provider refused this key",
                            );
                        }
                        Err(e) => {
                            // Log the detail (no key in these strings); hand the client a generic
                            // message rather than echoing a raw upstream error body.
                            eprintln!("translate: {imdb} → {lang} failed: {}", e.message());
                            state.cache.put(failed_recently, "1".into(), SYNC_RETRY_TTL);
                            // Unpin ONLY when the source is what failed. A pin naming a file that
                            // holds no cues would otherwise be honoured forever — the marker expires
                            // after ten minutes, the same pin is read, the same failure follows, and
                            // the title is stuck for every install and language, since the pin is
                            // scoped to neither.
                            //
                            // For everything else the pin is innocent, and dropping it spends a
                            // metered download to learn nothing: a provider timeout or a rate limit
                            // says nothing about the file, so the re-pick lands on the same title
                            // needing the same source fetched again.
                            if matches!(e, TranslationFailure::Source(_)) {
                                state.cache.remove(&pin_key);
                            }
                            return httputil::text(StatusCode::BAD_GATEWAY, "translation failed");
                        }
                        }
                    }
                }
            }
        };
        // Re-answered from the source we actually used. The version above was computed before a
        // source was known, so it assumed one was needed; if the source turns out to be hash-matched
        // to this encode, that assumption would spend a metered download on the reference and a tier
        // binary on an alignment that is a no-op by construction — and file the result under a key
        // the next request, now pinned, never asks for.
        let ref_id = align_for(used_source);
        let cache_key = sync_cache_key(&body_key, &resync_url, ref_id);
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
            // Settled only if we actually know whether an anchor exists. If the hashed search failed
            // we are serving an unaligned body that a working search might have aligned, so the
            // client has to come back rather than caching it for a year.
            !anchor_unknown,
        )
        .await;
        if !want_json {
            return resp;
        }
    }

    translate_url_response(state, headers, config, season, id, extra, lang_seg)
}

/// The `.json` form's answer: the `.srt` URL the engine should fetch.
///
/// Factored out because every exit from the `.json` path has to produce this shape. One of them
/// didn't — the failed-anchor branch returned an SRT body regardless of which form had been asked
/// for, and the app calls `.json` first.
#[allow(clippy::too_many_arguments)]
fn translate_url_response(
    state: &Arc<AppState>,
    headers: &HeaderMap,
    config: &str,
    season: Option<i64>,
    id: &str,
    extra: &str,
    lang_seg: &str,
) -> Response<Body> {
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

/// Why a translation did not happen, split by whether the SOURCE is to blame.
///
/// The distinction decides whether the pin survives. `Source` means this file cannot be translated
/// from — it will not download, or it parses to nothing — so the pin naming it is wrong and has to
/// go. `Model` means the provider timed out, refused, or produced junk: nothing to do with which
/// file was chosen, so dropping the pin buys a metered download and learns nothing. And the pin is
/// scoped to neither install nor language, so one install's rate limit would spend that credit on
/// behalf of everybody.
enum TranslationFailure {
    Source(String),
    Model(String),
    /// The install has used up today's translations. Distinct because it is the one failure the
    /// client should see as "not now" rather than "something broke".
    Allowance,
    /// The provider refused the credential itself. Distinct because a refusal that really is about
    /// the key will happen identically for the next title, so it can earn a backoff wider than the
    /// one title — see `refusal_scope`, which decides how much wider from the two fields below.
    ///
    /// It is NOT a free failure, and the two fields are there because it used to be treated as one.
    /// The per-title marker is written as well as whatever wider scope is armed, and the allowance
    /// slot comes back only when this request spent nothing at all — neither tokens nor a metered
    /// download.
    ///
    /// `key_certain` says whether the status can only be about the credential. A 401 or an empty
    /// balance can; a 400 or a 403 might instead be this film's dialogue tripping a content filter,
    /// and blocking a whole install on one of those lets a series take every other title down.
    ///
    /// `spent` is the strongest signal of the three and is a measurement rather than a guess: if a
    /// batch was billed before the refusal, the provider accepted this credential in this very run,
    /// so whatever it has now objected to is about the CONTENT, not the key.
    Credential { message: String, key_certain: bool, spent: bool },
}

/// Carry a download failure's own verdict through to the pin.
///
/// Only `Gone` — the API's own 404/410 on the id, or a `Suspect` that has now failed twice and been
/// promoted — is allowed to drop a pin shared by every install and every language. A single
/// `Suspect` is not: an expired one-shot CDN link and an interstitial served as a 200 both look like
/// that, both clear on their own, and unpinning on one re-picks against a drifted candidate list and
/// spends another metered credit on a source that was fine.
fn classify_download(e: opensubtitles::DownloadError) -> TranslationFailure {
    match e {
        // The API says the id does not exist, or a repeat has confirmed the file will not download.
        opensubtitles::DownloadError::Gone(m) => TranslationFailure::Source(m),
        // One failed link fetch, or quota, a revoked key, transport. The pin is innocent and must
        // survive all of them: blaming it here would spend a download credit re-picking a source,
        // because the viewer ran out of download credits for the day.
        opensubtitles::DownloadError::Suspect(m) | opensubtitles::DownloadError::Unavailable(m) => {
            TranslationFailure::Model(m)
        }
    }
}

impl TranslationFailure {
    fn message(&self) -> &str {
        match self {
            TranslationFailure::Source(m)
            | TranslationFailure::Model(m)
            | TranslationFailure::Credential { message: m, .. } => m.as_str(),
            TranslationFailure::Allowance => "daily translation allowance used up",
        }
    }
}

/// Translate one source subtitle into `lang` and cache the result under `body_key`. Returns the
/// translated SRT — the caller then runs the sync ladder over it.
#[allow(clippy::too_many_arguments)]
async fn produce_translation(
    state: &Arc<AppState>,
    client: &opensubtitles::Client<'_>,
    llm: &LlmConfig,
    config: &str,
    source_file_id: i64,
    lang: &str,
    body_key: &str,
    job_key: &str,
) -> Result<String, TranslationFailure> {
    // Through the cache, not straight at the API. A `/download` call spends one of the viewer's
    // daily OpenSubtitles credits on the CALL, not on the file fetch — and dodging that quota is the
    // reason the proxy-and-cache design exists at all. Going direct re-paid for a file already
    // sitting under `os:{file_id}`: once per retry after the ten-minute backoff, and once more for
    // every additional target language of the same film.
    //
    // Refuse for free BEFORE spending a credit on the source. The charge itself is further down, on
    // purpose — a source that will not fetch must not cost a slot — but with only that check, every
    // request past the fiftieth bought a metered download and then returned 429. Nothing caps how
    // many distinct titles a viewer browses, so that was unbounded, and it drained the same quota
    // the plain subtitle picker spends.
    if allowance_used_up(state, config) {
        return Err(TranslationFailure::Allowance);
    }
    // Whether this request is about to spend a metered OpenSubtitles credit, asked BEFORE the fetch
    // because afterwards the answer is always "it is cached now". Read pessimistically: a source
    // already on disk costs nothing, and anything else is assumed to have cost a credit, which is
    // the safe direction for a quota whose free tier is a handful a day.
    let source_was_cached = state.cache.get(&os_base_key(source_file_id)).is_some();
    // A download failure is NOT charged to the source. An exhausted daily credit and a dead upload
    // look identical from here, and on the free tier the first is an ordinary evening — so unpinning
    // on it would re-pick the source and owe another credit for the replacement, because the viewer
    // ran out of credits.
    let raw = subtitle_srt(state, client, source_file_id)
        .await
        .map_err(classify_download)?;
    let cues = srt::parse(&raw);
    if cues.is_empty() {
        // Unreachable in practice — `download` gates on `has_a_cue`, and a differential test pins
        // that gate to agree with `parse` in both directions. Kept because "it parsed to nothing"
        // is a statement about the file either way, and the classification should not depend on
        // which of two agreeing checks happened to run first.
        return Err(TranslationFailure::Source("source subtitle was empty".into()));
    }
    // The size gate, before the charge rather than inside `translate`. It is a refusal, not a run:
    // `translate` returns on it without making a single call, so charging first spent a slot on a
    // film that sent nothing — and the per-title marker throttles that to six an hour rather than
    // stopping it, so anything that kept asking ate the day's allowance. The cue count is known
    // three lines above; this is the same argument the download already gets.
    //
    // `Source`, not `Model`, and both halves of it. Size is a fact about the file that no retry and
    // no other install will change, and `Model` keeps the pin — so an oversized source pinned itself
    // in front of a title for the pin's full 180 days, for every install and every language, while
    // the perfectly good smaller candidates behind it were never reachable. `Source` drops the pin.
    //
    // Dropping the pin is only half an answer, because the re-pick is deterministic and would choose
    // this same file again. So the file is marked as well, which is what actually retires it, and the
    // marker is separate from `dead_file_key`: this file downloads fine and remains a good subtitle
    // to hand over as-is. It is unusable only as something to translate.
    let dialogue: usize = cues.iter().map(|c| c.text.len()).sum();
    let too_big = match (cues.len() > translate::MAX_CUES, dialogue > translate::MAX_DIALOGUE_BYTES) {
        (true, _) => Some(format!(
            "subtitle too large: {} cues (max {})",
            cues.len(),
            translate::MAX_CUES
        )),
        (_, true) => Some(format!(
            "subtitle too large: {dialogue} bytes of dialogue (max {})",
            translate::MAX_DIALOGUE_BYTES
        )),
        _ => None,
    };
    if let Some(why) = too_big {
        state.cache.put(oversized_file_key(source_file_id), "1".into(), SOURCE_PIN_TTL);
        return Err(TranslationFailure::Source(why));
    }
    // Charged HERE — the last point before the first token is spent, and after everything that can
    // fail without spending one.
    //
    // Charged before the download, it took a slot for every attempt whose source would not fetch,
    // having sent nothing. And the guards around it are per file and per title-language, so distinct
    // titles are not gated against each other: an evening browsing ten episodes in two languages
    // burned twenty slots in under a minute, and about twenty-five titles emptied the day — turning
    // an OpenSubtitles quota that resets at midnight into a self-inflicted day-long lockout of the
    // paid feature.
    if !charge_translation(state, config) {
        return Err(TranslationFailure::Allowance);
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

    // None of the harness's failures says anything about which file was chosen, so none of them
    // touches the pin.
    let translated = translated.map_err(|e| {
        // Refunded only when this request spent NOTHING — no tokens on the viewer's key and no
        // metered OpenSubtitles credit. Both halves are load-bearing and each was wrong on its own.
        //
        // Asking only about the failure's shape missed the resumed run whose batches all came back
        // from the store and then tripped the ratio gate: a slot for a run that never called the
        // provider. Asking only about tokens gave the slot back for a request that had just bought
        // a metered download — and the daily allowance is the ONLY thing bounding how many distinct
        // titles a viewer can burn credits on, since a `Model` failure arms no install-wide marker
        // the way a credential refusal does. A typo'd model name answers 404 for every title, so a
        // twenty-episode season became twenty credits out of a free tier of a handful, with the
        // counter back where it started each time.
        if !e.spent && source_was_cached {
            refund_translation(state, config);
        }
        match e.credential_refused {
            // A key the provider refuses outright will refuse the next film identically.
            true => TranslationFailure::Credential {
                message: e.message,
                key_certain: e.key_certain,
                spent: e.spent,
            },
            false => TranslationFailure::Model(e.message),
        }
    })?;
    let body = srt::serialize(&translated);
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
        // Among equally-scoring hash matches the choice is stable AND independent of list order —
        // the anchor decides the `?ref=` in every URL handed back, so it must be the same answer
        // every time, not merely a valid one. Pinning it to list order was the bug: the picker ranks
        // its list and the translate path does not, so one title anchored two ways and paid a second
        // metered download for it.
        assert_eq!(tier1_reference(&[sub(3, true), sub(2, true), sub(1, true)]), Some(1));
        assert_eq!(tier1_reference(&[sub(1, true), sub(2, true), sub(3, true)]), Some(1));
        // Every permutation, since "some order happened to agree" is what the old assertion proved.
        for order in [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]] {
            let subs: Vec<Subtitle> = order.iter().map(|&i| sub(i, true)).collect();
            assert_eq!(tier1_reference(&subs), Some(1), "order {order:?} chose a different anchor");
        }
        // The tie-break is the LAST word, not the first: a better score still wins over a lower id.
        let better = Subtitle { downloads: 50_000, from_trusted: true, ..sub(9, true) };
        assert_eq!(tier1_reference(&[sub(1, true), better]), Some(9));
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
    /// a vote here, two encodes of one film would resolve to two different sources, so the pin would
    /// flip every time the viewer switched between them and each flip is another metered download
    /// out of a free tier of a handful a day.
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
        let base = translate_body_key("tt0111161", None, None, "SV", &llm);
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
        // two cheap alignments hanging off it.
        let encode_a = sync_cache_key(&base, &None, Some(11));
        let encode_b = sync_cache_key(&base, &None, Some(22));
        assert_ne!(encode_a, encode_b);
        assert!(encode_a.starts_with(&base) && encode_b.starts_with(&base));

        // A different title, or a different model, is a different translation. A different SOURCE is
        // not — the pin decides which file a title reads from, and a body already paid for stays
        // reachable when that decision changes. Were the source in this key, one language's source
        // dying would have re-bought every other language of the film.
        assert_ne!(translate_body_key("tt0068646", None, None, "SV", &llm), base);
        assert_ne!(translate_body_key("tt0111161", Some(1), Some(2), "SV", &llm), base);
        let bigger = LlmConfig { model: "gpt-4o".into(), ..llm.clone() };
        assert_ne!(translate_body_key("tt0111161", None, None, "SV", &bigger), base);
    }

    /// A subtitle is never its own reference. `tier1_ref_for` refuses that when it builds a URL, but
    /// `ref` arrives on the query string and anyone can name it — and `?ref=5` on file 5 spawned a
    /// tier binary to align a file to itself: a subprocess on the one runtime thread, for a
    /// guaranteed no-op, cached afterwards under a key that claims an alignment happened.
    #[test]
    fn a_subtitle_is_never_its_own_reference() {
        // Through the real function the handler calls, not a re-typed copy of the rule — a copy
        // cannot catch the handler dropping the check.
        assert_eq!(vetted_ref(5, Some(5)), None, "a file was accepted as its own reference");
        assert_eq!(vetted_ref(5, Some(9)), Some(9), "a real reference was rejected");
        assert_eq!(vetted_ref(5, None), None);
        // And with it refused, the request keys on the plain body rather than claiming an alignment.
        assert_eq!(sync_cache_key(&os_base_key(5), &None, vetted_ref(5, Some(5))), "os:5");
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
        // The body key carries the same canonicalized language, so the collapsing has to hold there
        // too — that is the key the film's text actually lives under, and the one whose duplicates
        // are paid for in full.
        let body_key = |lang: &str| {
            let decoded = httputil::percent_decode_path(lang);
            translate_body_key("tt0111161", None, None, &translate::canonical_lang(&decoded), &llm)
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

    /// A refused credential is a fact about the install, and has to be remembered as one.
    ///
    /// The title-scoped marker can never catch it — every film is a different key — so a wrong or
    /// expired provider key, the likeliest misconfiguration in a BYOK product since the key rides in
    /// the install URL, cost a metered subtitle download per title browsed and would have drained the
    /// whole daily allowance on runs that sent no tokens at all.
    #[test]
    fn a_refused_key_is_remembered_for_the_install_not_the_title() {
        let llm = LlmConfig {
            provider: userconfig::Provider::OpenAI,
            model: "gpt-4o-mini".into(),
            api_key: "k".into(),
        };
        let one = credential_refused_key("install-one", &llm);

        // Not scoped to a title, so the next film is short-circuited too.
        assert!(!one.contains("tt"), "the credential marker names a title: {one}");
        // Another install is unaffected, and so is the same install on another provider or model —
        // switching either is the fix, and it must take effect at once.
        assert_ne!(one, credential_refused_key("install-two", &llm));
        let elsewhere = LlmConfig { model: "gpt-4o".into(), ..llm.clone() };
        assert_ne!(one, credential_refused_key("install-one", &elsewhere));
        // And it never leaks the config segment itself, which is a bearer secret.
        assert!(!one.contains("install-one"));

        // A refund gives back exactly the slot the refused run took, and cannot run the counter
        // below zero if it is somehow called twice.
        let state = state("refund");
        assert!(charge_translation(&state, "install-one"));
        refund_translation(&state, "install-one");
        refund_translation(&state, "install-one");
        for i in 0..DAILY_TRANSLATIONS {
            assert!(charge_translation(&state, "install-one"), "the refund lost a slot at {i}");
        }
    }

    /// An ambiguous refusal blocks the install only once a SECOND title has hit it.
    ///
    /// A 400 or a 403 might be the key and might be one film's dialogue tripping a content filter.
    /// Guessing per provider got that wrong twice in two rounds, in both directions — first blocking
    /// the whole library on one filtered series, then letting a dead key spend a metered download per
    /// title browsed. Two different titles refusing the same way is what actually distinguishes them.
    #[test]
    fn an_ambiguous_refusal_needs_two_titles_to_block_the_install() {
        let films = state("ambiguous");
        let llm = LlmConfig {
            provider: userconfig::Provider::OpenRouter,
            model: "m".into(),
            api_key: "k".into(),
        };
        // Through the real function the handler calls — a re-typed copy of the rule cannot catch the
        // handler diverging from it, which is how the series case got in.
        let refuse = |title: &str| ambiguous_refusal_escalates(&films, "install-one", &llm, title);

        assert!(matches!(refuse("tt0111161:0:0"), Corroboration::None), "one title corroborated");
        // The same title again is still one film — a filtered film must not escalate by repeating.
        assert!(
            matches!(refuse("tt0111161:0:0"), Corroboration::None),
            "the same title twice corroborated itself"
        );
        // An UNRELATED one is the signal that this is the key, not the dialogue.
        assert!(
            matches!(refuse("tt0068646:0:0"), Corroboration::Unrelated),
            "two unrelated films did not implicate the credential"
        );

        // Episodes of one series are DIFFERENT titles here. `parse_id` returns the SERIES id, so
        // building this from that alone made every episode read as one title and the escalation
        // could never fire across a season. Asserted against the real builder: spelled inline in the
        // handler, reverting it compiled and passed.
        assert_ne!(
            title_id("tt1234567", Some(2), Some(5)),
            title_id("tt1234567", Some(2), Some(6)),
            "two episodes of one series count as the same title"
        );
        assert_ne!(title_id("tt0111161", None, None), title_id("tt0068646", None, None));
        // And a film is stable across requests, so repeating it never escalates on its own.
        assert_eq!(title_id("tt0111161", None, None), title_id("tt0111161", None, None));

        let series = state("ambiguous-series");
        let refuse = |title: &str| ambiguous_refusal_escalates(&series, "install-one", &llm, title);
        let ep = |s, e| title_id("tt1234567", Some(s), Some(e));
        assert!(matches!(refuse(&ep(2, 5)), Corroboration::None), "one episode corroborated");
        // A second EPISODE is a second title, so it is evidence — but only about the show. A content
        // filter objects to a script, and a season is the fastest way to produce two refusals, so
        // reading this as the credential let one filtered show block the library ten minutes at a
        // time. It takes something unrelated to say anything about the key.
        assert!(
            matches!(refuse(&ep(2, 6)), Corroboration::SameSeries),
            "the next episode of the same series did not corroborate the show"
        );
        assert!(
            matches!(refuse("tt0111161:0:0"), Corroboration::Unrelated),
            "a film after a series did not implicate the credential"
        );
        // Series ids compared as ids, not as prefixes of the whole title string.
        assert_eq!(series_of(&ep(2, 6)), "tt1234567");
        assert_ne!(series_of(&ep(2, 6)), series_of("tt12345678:2:6"));

        // The real decision, not a copy of it spelled inline — the previous version of this test
        // re-typed the rule as a closure, which passes just as happily when the handler stops
        // agreeing with it.
        let scope = |key_certain: bool, spent: bool, title: &str| {
            refusal_scope(&series, "install-two", &llm, title, key_certain, spent)
        };
        // A refusal that arrives after a billed batch is the content, so it is the SERIES. The
        // provider accepted this key in this run; blocking the install on that let one filtered show
        // re-arm the install-wide marker every time the previous one lapsed.
        assert!(matches!(scope(false, true, &ep(3, 1)), RefusalScope::Series));
        // Every time, not only the first: this is a measurement, so it does not need corroborating.
        assert!(matches!(scope(false, true, &ep(3, 2)), RefusalScope::Series));
        // And it is a real throttle — leaving it at the per-title marker alone was the regression
        // this replaced, where each next episode re-bought a download, a slot and half a bill.
        assert_ne!(
            series_refused_key("install-two", &llm, "tt1234567"),
            series_refused_key("install-two", &llm, "tt7654321"),
            "one filtered series backed off an unrelated one"
        );
        assert_ne!(
            series_refused_key("install-two", &llm, "tt1234567"),
            series_refused_key("install-three", &llm, "tt1234567"),
            "one install's filtered series backed off another install"
        );
        // A dead key bills nothing, so it rests on corroboration — and a second EPISODE is not the
        // corroboration that implicates a key. This is the case `spent` cannot catch: a content
        // filter that objects to a show's dialogue objects to the glossary sample drawn from that
        // same dialogue, so the filtered run often bills nothing and arrives here rather than above.
        assert!(matches!(scope(false, false, &ep(4, 1)), RefusalScope::TitleOnly));
        assert!(matches!(scope(false, false, &ep(4, 2)), RefusalScope::Series));
        // An unrelated title is what implicates the credential.
        assert!(matches!(scope(false, false, "tt0068646:0:0"), RefusalScope::Install));
        // And an unambiguous status blocks at once, billed or not — a key revoked mid-run is dead.
        assert!(matches!(scope(true, true, &ep(5, 1)), RefusalScope::Install));

        // And the markers are distinct namespaces, so none can be read as another.
        assert_ne!(ambiguous_refusal_key("install-one", &llm), credential_refused_key("install-one", &llm));
        assert_ne!(
            series_refused_key("install-one", &llm, "tt1234567"),
            credential_refused_key("install-one", &llm)
        );
    }

    /// A scratch tag has to be a FILENAME, and a cache key is not one. The ladder's temp writer
    /// creates `work_dir` and nothing below it, so a single separator anywhere in the key put the
    /// file in a directory that does not exist and every alignment failed ENOENT — as a `syncfail:`
    /// marker and a provisional body, which is to say silently.
    #[test]
    fn a_scratch_tag_is_a_single_filename_component() {
        // The real default model of a real provider, not a hand-written string: this was live for
        // every OpenRouter install, and asserting a literal would not have noticed.
        let model = userconfig::Provider::OpenRouter.default_model();
        assert!(model.contains('/'), "the case this guards is gone; find what replaced it");
        let key = translate_body_key("tt0111161", None, None, "SV", &LlmConfig {
            provider: userconfig::Provider::OpenRouter,
            model: model.into(),
            api_key: "k".into(),
        });
        let tag = scratch_tag(&key, 0);
        assert!(!tag.contains('/'), "the tag names a subdirectory: {tag}");
        assert_eq!(std::path::Path::new(&tag).components().count(), 1, "{tag}");

        // Nothing else can escape either — a `lang` arrives percent-decoded, so it can carry any
        // byte the length bound allows.
        for hostile in ["..", "../../etc/passwd", "a\\b", "a\0b", "a.b", "sv-SE"] {
            let tag = scratch_tag(hostile, 1);
            assert!(
                tag.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'),
                "{hostile:?} survived as {tag:?}"
            );
            assert_eq!(std::path::Path::new(&tag).components().count(), 1, "{tag}");
        }

        // Bounded in BYTES, which is what NAME_MAX counts. Mapping to ASCII first is what makes the
        // char-wise take agree with it; a multibyte model name wrote up to four times the bound.
        let wide = format!("translate:tt1:0:0:SV:openai:{}", "é".repeat(200));
        assert!(scratch_tag(&wide, 999).len() <= 96, "{}", scratch_tag(&wide, 999).len());

        // And it is still unique per invocation, which is what stops one alignment reading another's
        // half-written scratch file and caching it for sixty days.
        assert_ne!(scratch_tag(&key, 0), scratch_tag(&key, 1));
    }

    /// A failure belongs to the install that had it. The translated BODY is the same bytes whoever
    /// asked for it and is deliberately shared; a failure is not. One install with a dead provider
    /// key would otherwise hand every other install on the same provider and model a 502, a
    /// `.status` of `failed`, and its progress bar — the cheapest cross-tenant lever in the service,
    /// out of something meant to protect one viewer's bill.
    #[test]
    fn a_failure_belongs_to_the_install_that_had_it() {
        let llm = LlmConfig {
            provider: userconfig::Provider::OpenAI,
            model: "gpt-4o-mini".into(),
            api_key: "k".into(),
        };
        let key = |config: &str| translate_fail_key(config, "tt0111161", None, None, "SV", &llm);

        assert_ne!(key("install-one"), key("install-two"), "two installs shared a failure marker");
        assert_eq!(key("install-one"), key("install-one"), "the key is not stable for one install");

        // A paid body must survive the source choice changing, and it does so structurally: the key
        // is built from the title and names no file id at all. The pin is the only pointer to a
        // source, so while the key named one, a source dying as a SECOND language was added
        // re-bought the FIRST at full price out of a cache that still held it.
        let body_of = translate_body_key("tt0111161", None, None, "SV", &llm);
        assert!(body_of.contains("tt0111161"), "the body key does not identify the title: {body_of}");
        assert_eq!(body_of.matches(':').count(), 6, "unexpected key shape: {body_of}");

        // The body key, by contrast, carries no install at all — that sharing is the point of it.
        let body = translate_body_key("tt0111161", None, None, "SV", &llm);
        assert!(!body.contains("install-one"));
        // And neither key may leak the config segment itself: it is a bearer secret, and these
        // become filenames.
        assert!(!key("install-one").contains("install-one"), "the config segment reached the key");
    }

    /// Only a failure that indicts the SOURCE may drop the pin, and this is the mapping that decides
    /// it. Blaming the source for a quota exhaustion or a rate limit re-picks and owes a metered
    /// credit for the replacement, at the moment the viewer has run out of them; blaming the service
    /// for a deleted upload leaves the pin resolving to nothing, and the ten-minute marker then
    /// cycles the same failure for the pin's whole 180-day life, for every install and language.
    #[test]
    fn only_a_dead_source_unpins() {
        use opensubtitles::DownloadError;

        let dead = classify_download(DownloadError::Gone("opensubtitles download 404".into()));
        assert!(matches!(dead, TranslationFailure::Source(_)), "a dead upload must drop the pin");

        for service in [
            DownloadError::Unavailable("opensubtitles download 429".into()),
            DownloadError::Unavailable("opensubtitles download 406".into()),
            DownloadError::Unavailable("download request failed: connection reset".into()),
            // A single suspect link fetch is not evidence about the file either. Only a repeat,
            // which `remember_failure` promotes to `Gone`, may drop a pin shared by every install.
            DownloadError::Suspect("subtitle link 404 Not Found".into()),
            DownloadError::Suspect("subtitle link returned no cues".into()),
        ] {
            let message = format!("{service:?}");
            assert!(
                matches!(classify_download(service), TranslationFailure::Model(_)),
                "a service failure dropped the pin: {message}"
            );
        }
    }

    /// A failed download must not be re-bought on the next request — the credit is charged on the
    /// API call, so a file that will not resolve costs one every time anything asks. And the two
    /// kinds of failure are remembered differently: the API naming an id is a fact about the file
    /// and shared, a credential's quota is not.
    #[test]
    fn a_failed_download_is_remembered_and_a_repeat_escalates() {
        use opensubtitles::DownloadError;

        let state = state("dl-fail");
        let http = reqwest::Client::new();
        let cfg = userconfig::decode(state.config_keyring.as_ref(), &config_segment()).unwrap();
        let client = os_client(&state, &http, &cfg);

        assert!(remembered_failure(&state, &client, 5).is_none(), "precondition: nothing remembered");

        // The API's verdict lands straight on the shared marker.
        remember_failure(&state, &client, 5, &DownloadError::Gone("404".into()));
        assert!(matches!(remembered_failure(&state, &client, 5), Some(DownloadError::Gone(_))));

        // One suspect link fetch earns a short backoff, so a burst of requests for the same file
        // costs one credit rather than one each. It reads back as `Suspect`, which refuses without
        // unpinning anything.
        remember_failure(&state, &client, 6, &DownloadError::Suspect("link 404".into()));
        assert!(matches!(remembered_failure(&state, &client, 6), Some(DownloadError::Suspect(_))));

        // A sibling from the SAME incident must not confirm it. Twenty picker URLs all fetch the
        // same reference, so two failures land milliseconds apart — one event, and the verdict they
        // would fabricate is the only one trusted to drop a pin shared by every install.
        remember_failure(&state, &client, 6, &DownloadError::Suspect("link 404".into()));
        assert!(
            state.cache.get(&dead_file_key(6)).is_none(),
            "a burst confirmed itself and fabricated a verdict"
        );

        // A failure on a LATER occasion does confirm it. Backdating the strike past the backoff
        // window stands in for waiting one out.
        let long_ago = (unix_seconds() - SYNC_RETRY_TTL.as_secs() - 1).to_string();
        state.cache.put_mem(suspect_strike_key(6), long_ago, DEAD_FILE_TTL);
        remember_failure(&state, &client, 6, &DownloadError::Suspect("link 404".into()));
        assert!(
            matches!(remembered_failure(&state, &client, 6), Some(DownloadError::Gone(_))),
            "a second occasion did not confirm the file is dead"
        );

        // "Too big to translate" is a different verdict from "will not download", and must not be
        // filed as one. This file fetches perfectly; the plain picker should keep handing it over,
        // and only the choice of a translation SOURCE skips it.
        assert_ne!(oversized_file_key(6), dead_file_key(6));
        assert_ne!(oversized_file_key(6), suspect_backoff_key(6));
        state.cache.put(oversized_file_key(9), "1".into(), SOURCE_PIN_TTL);
        assert!(
            remembered_failure(&state, &client, 9).is_none(),
            "an oversized file was read back as one that will not download"
        );

        // And a clean download clears the record, so a file that works nine times in ten is never
        // convicted by two bad afternoons a day apart.
        remember_failure(&state, &client, 8, &DownloadError::Suspect("blip".into()));
        forget_failure(&state, 8);
        assert!(
            remembered_failure(&state, &client, 8).is_none(),
            "a success left the file under suspicion"
        );

        // A credential's own trouble is remembered per credential, not for everyone: one install
        // exhausting its allowance must not deny the file to another.
        remember_failure(&state, &client, 7, &DownloadError::Unavailable("429".into()));
        assert!(matches!(remembered_failure(&state, &client, 7), Some(DownloadError::Unavailable(_))));
        let other = opensubtitles::Client { api_key: "a-different-install-key", ..client };
        assert!(
            remembered_failure(&state, &other, 7).is_none(),
            "one install's quota denied the file to another"
        );
        // And the file-scoped ones are shared, because they are facts about the file.
        assert!(matches!(remembered_failure(&state, &other, 5), Some(DownloadError::Gone(_))));
    }

    /// The pin is what keeps a drifting source pick from spending a metered credit each time it
    /// drifts. It has to be scoped to the title alone — every language of one film translating from
    /// the same source, so a film costs one download rather than one per language — and removable,
    /// because a pin naming a file that will not download would otherwise be honoured
    /// forever: the ten-minute marker expires, the same pin is read, the same failure follows, and
    /// the title is stuck for every install and language, since the pin is scoped to neither.
    #[test]
    fn the_source_pin_is_per_title_and_can_be_dropped() {
        let state = state("pin");
        let film = source_pin_key("tt0111161", None, None);

        // A film and one of its episodes are different pins; so are two films.
        assert_ne!(film, source_pin_key("tt0111161", Some(1), Some(2)));
        assert_ne!(film, source_pin_key("tt0068646", None, None));
        // It holds a file id, and lives outside every namespace a body is read from.
        assert!(!BODY_PREFIXES.iter().any(|p| film.starts_with(p)), "a pin sits in a body namespace");

        state.cache.put(film.clone(), "77".into(), CACHE_TTL);
        assert_eq!(state.cache.get(&film).and_then(|v| v.parse::<i64>().ok()), Some(77));

        // Removable in both tiers — a dead pin has to actually go, not be written back with a
        // zero lifetime, which would be a blocking disk write to say "forget this".
        state.cache.remove(&film);
        assert_eq!(state.cache.get(&film), None, "a dropped pin was still honoured");
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
