//! OpenSubtitles REST client (api.opensubtitles.com). The addon holds its own API-consumer key (an
//! addon-level env secret, not the user's), so downloads draw on one managed quota and we can cache
//! the fetched SRT under our own stable URL — dodging the per-IP anonymous download cap.
//!
//! The one thing that makes subtitles well-synced is passing the file's `moviehash`: OpenSubtitles
//! flags results authored against that exact encode with `moviehash_match`, and those are correct by
//! construction. The Den app computes the OSHash of the playing file and sends it as `videoHash`;
//! we forward it straight through and float the matches to the top.

use serde_json::Value;

/// One search hit, reduced to what selection + download need.
#[derive(Debug, Clone)]
pub struct Subtitle {
    pub file_id: i64,
    pub lang: String,
    /// True when this subtitle was matched to the exact file by hash → already in sync.
    pub hash_match: bool,
    /// Download count — the tie-breaker when several subs share a language and none is a hash match.
    pub downloads: i64,
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
        let v: Value = resp.json().await.map_err(|e| format!("bad json: {e}"))?;
        let mut subs = parse_search(&v);
        // Hash matches first (already in sync), then most-downloaded (community-vetted timing).
        subs.sort_by(|a, b| b.hash_match.cmp(&a.hash_match).then(b.downloads.cmp(&a.downloads)));
        Ok(subs)
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
        let v: Value = resp.json().await.map_err(|e| format!("bad json: {e}"))?;
        let link = v["link"].as_str().ok_or("no download link")?;
        let body = self
            .http
            .get(link)
            .send()
            .await
            .map_err(|e| format!("fetch link failed: {e}"))?
            .text()
            .await
            .map_err(|e| format!("read subtitle failed: {e}"))?;
        Ok(body)
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
        });
    }
    out
}

/// Pick the best subtitle for a wanted language: a hash match if any, else the most-downloaded.
/// (`search` already sorts that way, so this is just the first match in `lang`.)
pub fn best_for<'s>(subs: &'s [Subtitle], lang: &str) -> Option<&'s Subtitle> {
    subs.iter().find(|s| s.lang.eq_ignore_ascii_case(lang))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_and_orders_results() {
        let v = json!({"data": [
            {"attributes": {"language": "en", "moviehash_match": false, "download_count": 10,
                "release": "WEB", "files": [{"file_id": 1}]}},
            {"attributes": {"language": "sv", "moviehash_match": true, "download_count": 2,
                "release": "BluRay", "files": [{"file_id": 2}]}},
        ]});
        let mut subs = parse_search(&v);
        subs.sort_by(|a, b| b.hash_match.cmp(&a.hash_match).then(b.downloads.cmp(&a.downloads)));
        assert_eq!(subs[0].file_id, 2); // hash match first
        assert_eq!(best_for(&subs, "en").unwrap().file_id, 1);
    }
}
