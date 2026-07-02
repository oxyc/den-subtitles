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

/// Pull `videoHash` out of the Stremio extra-args blob (already `.json`-stripped).
fn extra_hash(extra: &str) -> Option<String> {
    let decoded = percent_decode(extra);
    for pair in decoded.split('&') {
        if let Some(v) = pair.strip_prefix("videoHash=") {
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
    // to spare a live round-trip every time the app reopens a title.
    let hash = extra_hash(extra);
    let search_key = format!(
        "search:{imdb}:{}:{}:{}",
        season.unwrap_or(0),
        episode.unwrap_or(0),
        hash.as_deref().unwrap_or("")
    );
    let subs: Vec<opensubtitles::Subtitle> = if let Some(hit) = state
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
                if let Ok(json) = serde_json::to_string(&s) {
                    state.cache.put(search_key, json, SEARCH_TTL);
                }
                s
            }
            // Empty-200 is the correct Stremio shape for "nothing", but log the cause (our error
            // strings carry no key) so an upstream/quota failure isn't invisible.
            Err(e) => {
                eprintln!("subtitles: opensubtitles search failed for {imdb}: {e}");
                return httputil::json(StatusCode::OK, &json!({"subtitles": []}), "no-store");
            }
        }
    };

    let base = self_base(state, headers, config);
    let out: Vec<Value> = subs
        .iter()
        .map(|s| {
            json!({
                "id": format!("os-{}", s.file_id),
                "url": format!("{base}/subtitle/{}.srt", s.file_id),
                "lang": s.lang,
            })
        })
        .collect();
    httputil::json(StatusCode::OK, &json!({"subtitles": out}), "public, max-age=3600")
}

/// GET /<config>/subtitle/<file_id>.srt — download one OpenSubtitles file, cache, serve.
pub async fn handle_subtitle_file(state: &Arc<AppState>, config: &str, file_id: i64) -> Response<Body> {
    let Some(cfg) = userconfig::decode(config) else {
        return httputil::text(StatusCode::BAD_REQUEST, "bad_config");
    };
    let cache_key = format!("os:{file_id}");
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
    match client.download(file_id).await {
        Ok(body) => {
            state.cache.put(cache_key, body.clone(), CACHE_TTL);
            httputil::srt(body)
        }
        // An upstream/quota failure is a bad-gateway condition, not a missing resource.
        Err(e) => {
            eprintln!("subtitle: download of file {file_id} failed: {e}");
            httputil::text(StatusCode::BAD_GATEWAY, "upstream subtitle fetch failed")
        }
    }
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
