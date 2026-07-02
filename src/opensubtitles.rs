//! OpenSubtitles REST client (api.opensubtitles.com). The addon holds its own API-consumer key (an
//! addon-level env secret, not the user's), so downloads draw on one managed quota and we can cache
//! the fetched SRT under our own stable URL — dodging the per-IP anonymous download cap.
//!
//! The one thing that makes subtitles well-synced is passing the file's `moviehash`: OpenSubtitles
//! flags results authored against that exact encode with `moviehash_match`, and those are correct by
//! construction. The Den app computes the OSHash of the playing file and sends it as `videoHash`;
//! we forward it straight through and float the matches to the top.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One search hit, with the metadata needed to rank fit-to-stream and to show detail in the app
/// picker. (De)serializable so a whole search result caches as JSON and rebuilds into a response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subtitle {
    pub file_id: i64,
    pub lang: String,
    /// True when this subtitle was matched to the exact file by hash → already in sync.
    pub hash_match: bool,
    /// Download count — a tie-breaker when several subs share a language and none is a hash match.
    pub downloads: i64,
    /// The uploader's release string (e.g. "Fight.Club.1999.1080p.BluRay.x264-GROUP") — matched
    /// against the playing file's name to judge fit.
    pub release: String,
    pub hd: bool,
    /// Frame rate the sub was timed against (0 when unknown) — shown for context.
    pub fps: f64,
    pub from_trusted: bool,
    /// Machine/AI-translated subs are low quality — demoted to the bottom of the ranking.
    pub machine_translated: bool,
    pub ai_translated: bool,
    /// 0–10 community rating.
    pub ratings: f64,
}

pub struct Client<'a> {
    pub http: &'a reqwest::Client,
    pub api_key: &'a str,
    /// Optional service-account bearer (raises the download quota above anonymous).
    pub token: Option<&'a str>,
}

impl<'a> Client<'a> {
    /// Search by IMDb id (+ optional episode) and optional file hash. Results are ordered
    /// hash-matches-first, then by download count.
    pub async fn search(
        &self,
        imdb_id: &str,
        season: Option<i64>,
        episode: Option<i64>,
        languages: &str,
        moviehash: Option<&str>,
    ) -> Result<Vec<Subtitle>, String> {
        // The imdb id goes in as digits only (no "tt").
        let imdb_num = imdb_id.trim_start_matches("tt");
        let mut query: Vec<(String, String)> = vec![
            ("imdb_id".into(), imdb_num.into()),
            ("languages".into(), languages.to_string()),
        ];
        if let Some(s) = season {
            query.push(("season_number".into(), s.to_string()));
        }
        if let Some(e) = episode {
            query.push(("episode_number".into(), e.to_string()));
        }
        if let Some(h) = moviehash {
            query.push(("moviehash".into(), h.to_string()));
        }

        let resp = self
            .http
            .get("https://api.opensubtitles.com/api/v1/subtitles")
            .header("Api-Key", self.api_key)
            .header("User-Agent", "den-subtitles v0.1")
            .query(&query)
            .send()
            .await
            .map_err(|e| format!("search failed: {e}"))?;
        if !resp.status().is_success() {
            return Err(format!("opensubtitles search {}", resp.status()));
        }
        let v: Value = crate::fetch::capped_json(resp, crate::fetch::MAX_BODY).await?;
        // Returned unranked (and cached that way — the ranking is filename-specific, so the handler
        // ranks per request against the playing file).
        Ok(parse_search(&v))
    }

    /// Resolve a `file_id` to a temporary download link, then fetch the subtitle text.
    pub async fn download(&self, file_id: i64) -> Result<String, String> {
        let mut req = self
            .http
            .post("https://api.opensubtitles.com/api/v1/download")
            .header("Api-Key", self.api_key)
            .header("User-Agent", "den-subtitles v0.1")
            .json(&serde_json::json!({ "file_id": file_id }));
        if let Some(t) = self.token {
            req = req.bearer_auth(t);
        }
        let resp = req.send().await.map_err(|e| format!("download request failed: {e}"))?;
        if !resp.status().is_success() {
            return Err(format!("opensubtitles download {}", resp.status()));
        }
        let v: Value = crate::fetch::capped_json(resp, crate::fetch::MAX_BODY).await?;
        let link = v["link"].as_str().ok_or("no download link")?;
        // The link is OpenSubtitles-supplied and points at their CDN — cap the fetched body.
        let resp = self
            .http
            .get(link)
            .send()
            .await
            .map_err(|e| format!("fetch link failed: {e}"))?;
        crate::fetch::capped_text(resp, crate::fetch::MAX_BODY).await
    }
}

fn parse_search(v: &Value) -> Vec<Subtitle> {
    let Some(items) = v["data"].as_array() else { return Vec::new() };
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let attrs = &item["attributes"];
        // Each subtitle "entry" carries one or more files; we take the first (the SRT).
        let file = &attrs["files"][0];
        let Some(file_id) = file["file_id"].as_i64() else { continue };
        out.push(Subtitle {
            file_id,
            lang: attrs["language"].as_str().unwrap_or("").to_string(),
            hash_match: attrs["moviehash_match"].as_bool().unwrap_or(false),
            downloads: attrs["download_count"].as_i64().unwrap_or(0),
            release: attrs["release"].as_str().unwrap_or("").to_string(),
            hd: attrs["hd"].as_bool().unwrap_or(false),
            fps: attrs["fps"].as_f64().unwrap_or(0.0),
            from_trusted: attrs["from_trusted"].as_bool().unwrap_or(false),
            machine_translated: attrs["machine_translated"].as_bool().unwrap_or(false),
            ai_translated: attrs["ai_translated"].as_bool().unwrap_or(false),
            ratings: attrs["ratings"].as_f64().unwrap_or(0.0),
        });
    }
    out
}

/// Score a subtitle's fit to the playing file. A hash match is decisive; otherwise release/filename
/// overlap dominates (same encode ⇒ same timing), with trust/ratings/downloads as tie-breakers.
/// Machine/AI-translated subs are pushed below everything.
pub fn fit_score(s: &Subtitle, filename: Option<&str>) -> i64 {
    const HASH: i64 = 1_000_000;
    const JUNK: i64 = 2_000_000; // demote machine/AI below even a no-info sub
    let mut score = 0i64;
    if s.hash_match {
        score += HASH;
    }
    if let Some(f) = filename {
        score += release_fit(f, &s.release);
    }
    if s.from_trusted {
        score += 400;
    }
    score += (s.ratings.clamp(0.0, 10.0) * 100.0) as i64; // 0..1000
    score += (((s.downloads.max(0) as f64).ln_1p()) * 60.0) as i64; // ~0..700
    if s.machine_translated || s.ai_translated {
        score -= JUNK;
    }
    score
}

/// Order subtitles for the picker: grouped by language, best-fit first within each language. The
/// app's own dedupe-keep-first then naturally keeps the optimal sub per language.
pub fn rank(subs: &mut [Subtitle], filename: Option<&str>) {
    subs.sort_by(|a, b| {
        a.lang
            .cmp(&b.lang)
            .then_with(|| fit_score(b, filename).cmp(&fit_score(a, filename)))
    });
}

/// Best subtitle for a wanted language by fit score (hash match, then quality) — independent of any
/// prior ordering.
pub fn best_for<'s>(subs: &'s [Subtitle], lang: &str) -> Option<&'s Subtitle> {
    subs.iter()
        .filter(|s| s.lang.eq_ignore_ascii_case(lang))
        .max_by_key(|s| fit_score(s, None))
}

/// Significant release tokens shared between the file name and a sub's release string imply the same
/// encode (hence the same timing). Weight resolution/source/codec, and reward a matching group tag.
fn release_fit(filename: &str, release: &str) -> i64 {
    if release.is_empty() {
        return 0;
    }
    let f = tokenize(filename);
    let r = tokenize(release);
    let mut score = 0i64;
    for (tok, weight) in SIGNIFICANT {
        if f.iter().any(|t| t == tok) && r.iter().any(|t| t == tok) {
            score += weight;
        }
    }
    // The release group (the tag after the last '-') is the strongest same-encode signal.
    if let (Some(gf), Some(gr)) = (release_group(filename), release_group(release)) {
        if gf == gr {
            score += 3000;
        }
    }
    score
}

/// Resolution/source/codec tokens and their weights (higher = stronger same-encode evidence).
const SIGNIFICANT: &[(&str, i64)] = &[
    ("2160p", 800), ("1080p", 800), ("720p", 800), ("480p", 800),
    ("bluray", 600), ("blu", 600), ("bdrip", 600), ("brrip", 600), ("remux", 600),
    ("web", 500), ("webrip", 500), ("webdl", 500), ("hdtv", 500), ("dvdrip", 500), ("hdrip", 500),
    ("x264", 200), ("x265", 200), ("h264", 200), ("h265", 200), ("hevc", 200), ("avc", 200),
];

/// Lowercase alphanumeric tokens.
fn tokenize(s: &str) -> Vec<String> {
    s.to_ascii_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect()
}

/// The release group tag — the single token after the last '-'. Rejects a hyphen that's part of the
/// *title* (e.g. "Spider-Man.2002.1080p.BluRay.x264" → tail "Man.2002.1080p…" has dots → not a
/// group), which would otherwise force a bogus group match between unrelated encodes.
fn release_group(s: &str) -> Option<String> {
    // Strip a trailing file extension so "…-GROUP.mkv" → "…-GROUP".
    let stem = match s.rsplit_once('.') {
        Some((head, ext)) if (1..=4).contains(&ext.len()) && ext.chars().all(|c| c.is_ascii_alphanumeric()) => head,
        _ => s,
    };
    let tail = stem.rsplit_once('-')?.1;
    // A real group tag is one token — any '.' or space means the '-' was inside the title/metadata.
    if tail.is_empty() || tail.contains('.') || tail.contains(' ') {
        return None;
    }
    Some(tail.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_results() {
        let v = json!({"data": [
            {"attributes": {"language": "en", "moviehash_match": false, "download_count": 10,
                "release": "Fight.Club.1999.1080p.BluRay.x264-AMIABLE", "files": [{"file_id": 1}]}},
            {"attributes": {"language": "en", "moviehash_match": false, "download_count": 999,
                "release": "Fight.Club.720p.WEB", "ai_translated": true, "files": [{"file_id": 2}]}},
        ]});
        let subs = parse_search(&v);
        assert_eq!(subs.len(), 2);
        assert_eq!(subs[0].release, "Fight.Club.1999.1080p.BluRay.x264-AMIABLE");
    }

    #[test]
    fn ranks_hash_then_stream_fit_and_demotes_machine() {
        let hashed = sub(1, "en", true, 5, "whatever");
        let fits = sub(2, "en", false, 5, "Fight.Club.1999.1080p.BluRay.x264-AMIABLE");
        let popular_junk = sub_ai(3, "en", 99999, "Fight.Club.1999.1080p.BluRay.x264-AMIABLE");
        let filename = Some("Fight.Club.1999.1080p.BluRay.x264-AMIABLE.mkv");
        // hash match beats a perfect release match
        assert!(fit_score(&hashed, filename) > fit_score(&fits, filename));
        // a release/group match beats an unrelated sub
        assert!(fit_score(&fits, filename) > fit_score(&sub(9, "en", false, 5, "Random.CAM"), filename));
        // machine/AI is demoted below a plain sub despite huge downloads
        assert!(fit_score(&popular_junk, filename) < fit_score(&sub(9, "en", false, 0, ""), filename));
    }

    fn sub(id: i64, lang: &str, hash: bool, dl: i64, release: &str) -> Subtitle {
        Subtitle {
            file_id: id, lang: lang.into(), hash_match: hash, downloads: dl, release: release.into(),
            hd: false, fps: 0.0, from_trusted: false, machine_translated: false, ai_translated: false, ratings: 0.0,
        }
    }
    fn sub_ai(id: i64, lang: &str, dl: i64, release: &str) -> Subtitle {
        Subtitle { ai_translated: true, ..sub(id, lang, false, dl, release) }
    }

    #[test]
    fn release_group_ignores_hyphens_in_titles() {
        // A hyphen in the title must NOT be read as a release group.
        assert_eq!(release_group("Spider-Man.2002.1080p.BluRay.x264.mkv"), None);
        assert_eq!(release_group("Spider-Man.2002.480p.DVDRip"), None);
        // A real trailing group tag still resolves (with or without extension).
        assert_eq!(release_group("Fight.Club.1999.1080p.BluRay.x264-AMIABLE"), Some("amiable".into()));
        assert_eq!(release_group("Fight.Club.1999.1080p.BluRay.x264-AMIABLE.mkv"), Some("amiable".into()));
    }

    #[test]
    fn spiderman_no_false_group_match() {
        // The correct 1080p encode must outrank a wrong 480p one that shares the bogus "man" token.
        let filename = Some("Spider-Man.2002.1080p.BluRay.x264-AMIABLE.mkv");
        let correct = sub(1, "en", false, 5, "Spider-Man.2002.1080p.BluRay.x264-AMIABLE");
        let wrong = sub(2, "en", false, 5, "Spider-Man.2002.480p.DVDRip");
        assert!(fit_score(&correct, filename) > fit_score(&wrong, filename));
    }
}
