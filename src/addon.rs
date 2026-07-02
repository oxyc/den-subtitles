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
        let host = hdr("x-forwarded-host")
            .or_else(|| hdr("host"))
            .unwrap_or("localhost")
            .to_string();
        format!("{proto}://{host}")
    };
    format!("{root}/{config}")
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

    let client = opensubtitles::Client {
        http: &state.http,
        api_key: &cfg.opensubtitles_key,
        token: cfg.opensubtitles_token.as_deref(),
    };
    // Ask for everything; the app filters/selects by its own preferred-language rules.
    let subs = match client
        .search(&imdb, season, episode, "all", extra_hash(extra).as_deref())
        .await
    {
        Ok(s) => s,
        Err(_) => return httputil::json(StatusCode::OK, &json!({"subtitles": []}), "no-store"),
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
    let client = opensubtitles::Client {
        http: &state.http,
        api_key: &cfg.opensubtitles_key,
        token: cfg.opensubtitles_token.as_deref(),
    };
    match client.download(file_id).await {
        Ok(body) => {
            state.cache.put(cache_key, body.clone(), CACHE_TTL);
            httputil::srt(body)
        }
        Err(_) => httputil::text(StatusCode::NOT_FOUND, "download failed"),
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
        match produce_translation(state, &cfg, llm, &imdb, season, episode, lang, &cache_key).await {
            Ok(()) => {}
            Err(e) => return httputil::text(StatusCode::BAD_GATEWAY, &e),
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
    let client = opensubtitles::Client {
        http: &state.http,
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
    let translated = translate::translate(&state.http, llm, &cues, lang).await?;
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
