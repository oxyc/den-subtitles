//! OpenSubtitles REST client (api.opensubtitles.com). The API-consumer key is the user's own BYOK key
//! (`UserConfig::opensubtitles_key`, entered at /configure and carried — sealed — in the install URL);
//! no OpenSubtitles credential lives in the addon environment. We cache the fetched SRT under our own
//! stable URL so a given file resolves fast and dodges the per-IP anonymous download cap.
//!
//! The one thing that makes subtitles well-synced is passing the file's `moviehash`: OpenSubtitles
//! flags results authored against that exact encode with `moviehash_match`, and those are correct by
//! construction. The Den app computes the OSHash of the playing file and sends it as `videoHash`;
//! we forward it straight through and float the matches to the top.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// OpenSubtitles asks every API consumer for `User-Agent: <app name> v<version>`, so this one keeps
/// that form rather than the `den-subtitles/<version>` the shared client sends everywhere else.
const USER_AGENT: &str = concat!("den-subtitles v", env!("CARGO_PKG_VERSION"));

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

/// Live API root. A field rather than a literal so the two-hop download flow — the API call, then
/// the CDN link it hands back — can be driven against a local server in tests.
pub const API: &str = "https://api.opensubtitles.com/api/v1";

pub struct Client<'a> {
    pub http: &'a reqwest::Client,
    pub api_key: &'a str,
    /// Optional service-account bearer (raises the download quota above anonymous).
    pub token: Option<&'a str>,
    pub api_base: &'a str,
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
        let mut query: Vec<(String, String)> =
            vec![("imdb_id".into(), imdb_num.into()), ("languages".into(), languages.to_string())];
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
            .get(format!("{}/subtitles", self.api_base))
            .header("Api-Key", self.api_key)
            .header("User-Agent", USER_AGENT)
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
    ///
    /// The error says whether the FILE is the problem or the service is. Callers that remember a
    /// file id — the pinned translation source — need to stop remembering it on the first and keep
    /// it on the second, and the two are otherwise indistinguishable from a message.
    pub async fn download(&self, file_id: i64) -> Result<String, DownloadError> {
        let mut req = self
            .http
            .post(format!("{}/download", self.api_base))
            .header("Api-Key", self.api_key)
            .header("User-Agent", USER_AGENT)
            // `sub_format` asked for rather than assumed. `srt::parse` is an SRT parser and nothing
            // downstream handles ASS or WebVTT, so the format was already load-bearing — it was just
            // whatever the endpoint happened to default to.
            .json(&serde_json::json!({ "file_id": file_id, "sub_format": "srt" }));
        if let Some(t) = self.token {
            req = req.bearer_auth(t);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| DownloadError::Unavailable(format!("download request failed: {e}")))?;
        if !resp.status().is_success() {
            let code = resp.status();
            return Err(DownloadError::from_status(code, format!("opensubtitles download {code}")));
        }
        let v: Value = crate::fetch::capped_json(resp, crate::fetch::MAX_BODY)
            .await
            .map_err(DownloadError::Unavailable)?;
        let link =
            v["link"].as_str().ok_or_else(|| DownloadError::Unavailable("no download link".to_string()))?;
        // The link is OpenSubtitles-supplied and points at their CDN — cap the fetched body.
        let resp = self
            .http
            .get(link)
            .send()
            .await
            // `without_url`: reqwest's Display prints the URL it failed on, and this one is the
            // one-shot download link — a bearer capability that would land in the container log.
            .map_err(|e| DownloadError::Unavailable(format!("fetch link failed: {}", e.without_url())))?;
        // The API call above is status-checked and this one was not, so a CDN 403/404/429 (an
        // expired or rate-limited link) returned its HTML error page AS the subtitle — cached under
        // the file id for 60 days and served `immutable`. One transient blip, one track that
        // silently shows nothing forever.
        if !resp.status().is_success() {
            // `Suspect`, not `from_status`: a 404 here is an expired one-shot capability URL far
            // more often than a missing file, and only the API's verdict is trusted to unpin.
            let code = resp.status();
            return Err(DownloadError::Suspect(format!("subtitle link {code}")));
        }
        let body = crate::fetch::capped_text(resp, crate::fetch::MAX_BODY)
            .await
            .map_err(DownloadError::Unavailable)?;
        // A 200 is not proof it is a subtitle: a CDN error or interstitial page is a 200 often
        // enough. Anything with no cue in it cannot be one.
        //
        // `Suspect`: an interstitial served as a 200 and a genuinely cue-less upload look identical
        // from here. The credit is already spent, so it is worth remembering either way — but not
        // worth unpinning on, which is what the caller does with a repeat.
        if !crate::srt::has_a_cue(&body) {
            return Err(DownloadError::Suspect("subtitle link returned no cues".to_string()));
        }
        Ok(body)
    }
}

/// Why a subtitle could not be fetched, split by whether the FILE or the SERVICE is at fault.
///
/// The distinction is what lets a remembered file id be forgotten at the right time. A 404 says the
/// upload is gone and whatever named it should stop; a 429 says the viewer is out of downloads for
/// today, and forgetting the file over that throws away a choice that was fine — so the replacement
/// costs another metered credit, out of the allowance that just ran out.
#[derive(Debug)]
pub enum DownloadError {
    /// The API says this id does not exist. Authoritative, true for everyone, and the only verdict
    /// trusted to drop a source pin shared by every install.
    Gone(String),
    /// The API handed us a link and what came back was not a subtitle — a CDN 404 on the one-shot
    /// URL, or an interstitial served as a 200. Usually transient, occasionally a bad upload, and
    /// not distinguishable from here. Retrying immediately will not help either way, so it is
    /// remembered; but a single one of these must not unpin anything.
    Suspect(String),
    /// Quota, rate limit, a revoked key, transport. About one credential, not about the file.
    Unavailable(String),
}

impl DownloadError {
    /// The API's own verdict on the id: 404 and 410 name the file. Everything else — 401/403 (key),
    /// 406/429 (quota), 5xx — is the service, and a file id must survive all of them.
    ///
    /// Only for the API call. A 404 on the CDN link that follows means an expired one-shot URL, not
    /// a missing file, so that site reports `Suspect`.
    fn from_status(status: reqwest::StatusCode, message: String) -> DownloadError {
        match status.as_u16() {
            404 | 410 => DownloadError::Gone(message),
            _ => DownloadError::Unavailable(message),
        }
    }

    pub fn message(self) -> String {
        match self {
            DownloadError::Gone(m) | DownloadError::Suspect(m) | DownloadError::Unavailable(m) => m,
        }
    }
}

fn parse_search(v: &Value) -> Vec<Subtitle> {
    let Some(items) = v["data"].as_array() else { return Vec::new() };
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let attrs = &item["attributes"];
        // A multi-file entry is a multi-CD release, and only one part of it can be served. Taking
        // the first looked fine everywhere it shows — full release string, hash match, quality
        // flags — and then stopped partway through the film, which is the failure that never
        // reports itself. Skip it and let a complete subtitle win instead.
        let files = attrs["files"].as_array().map(Vec::as_slice).unwrap_or_default();
        let [file] = files else { continue };
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
    let mut score = text_score(s);
    if s.hash_match {
        score += HASH;
    }
    if let Some(f) = filename {
        score += release_fit(f, &s.release);
    }
    score
}

/// How good this subtitle is as TEXT — everything in `fit_score` except the two terms that are
/// about which encode is playing (the hash match and the release-name overlap).
///
/// This is what you want when choosing something to TRANSLATE. The translated text is the same text
/// whatever the encode, and the timing is corrected afterwards by the sync ladder, so letting the
/// hash steer the pick would resolve every encode of one film to its own source — and the source
/// pin, which is what keeps a title's languages reading from one file, would be taken off each
/// encode by the next. Every flip is another metered download.
pub fn text_score(s: &Subtitle) -> i64 {
    const JUNK: i64 = 2_000_000; // demote machine/AI below even a no-info sub
    let mut score = 0i64;
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
        a.lang.cmp(&b.lang).then_with(|| fit_score(b, filename).cmp(&fit_score(a, filename)))
    });
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
    ("2160p", 800),
    ("1080p", 800),
    ("720p", 800),
    ("480p", 800),
    ("bluray", 600),
    ("blu", 600),
    ("bdrip", 600),
    ("brrip", 600),
    ("remux", 600),
    ("web", 500),
    ("webrip", 500),
    ("webdl", 500),
    ("hdtv", 500),
    ("dvdrip", 500),
    ("hdrip", 500),
    ("x264", 200),
    ("x265", 200),
    ("h264", 200),
    ("h265", 200),
    ("hevc", 200),
    ("avc", 200),
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
        Some((head, ext))
            if (1..=4).contains(&ext.len()) && ext.chars().all(|c| c.is_ascii_alphanumeric()) =>
        {
            head
        }
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

    /// A multi-file entry is a multi-CD release. Only one part can ever be served, and serving it
    /// looks completely normal in the picker — full release string, quality flags, hash match —
    /// then stops partway through the film. A complete subtitle is a better answer than a fragment
    /// that never says it is one.
    #[test]
    fn a_multi_cd_entry_is_not_offered_as_a_whole_subtitle() {
        let v = json!({"data": [
            {"attributes": {"language": "en", "download_count": 9999, "release": "Old.DVDRip.CD1-CD2",
                "files": [{"file_id": 10}, {"file_id": 11}]}},
            {"attributes": {"language": "en", "download_count": 5, "release": "Complete.1080p",
                "files": [{"file_id": 12}]}},
        ]});
        let subs = parse_search(&v);
        assert_eq!(subs.len(), 1, "a two-CD entry was offered as a subtitle");
        assert_eq!(subs[0].file_id, 12, "the complete subtitle should be the one that survives");

        // And an entry with no files at all is not one either.
        let empty = json!({"data": [{"attributes": {"language": "en", "files": []}}]});
        assert!(parse_search(&empty).is_empty());
    }

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
            file_id: id,
            lang: lang.into(),
            hash_match: hash,
            downloads: dl,
            release: release.into(),
            hd: false,
            fps: 0.0,
            from_trusted: false,
            machine_translated: false,
            ai_translated: false,
            ratings: 0.0,
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

#[cfg(test)]
mod download_tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// A one-shot server: answers the API call with a link back to itself, then answers that link
    /// with whatever the case under test wants the CDN to say.
    async fn upstream(cdn_status: &'static str, cdn_body: &'static str) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            for _ in 0..2 {
                let Ok((mut sock, _)) = listener.accept().await else { return };
                let mut buf = [0u8; 2048];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]).to_string();
                let (status, body) = if req.starts_with("POST") {
                    ("200 OK".to_string(), format!(r#"{{"link":"http://{addr}/cdn"}}"#))
                } else {
                    (cdn_status.to_string(), cdn_body.to_string())
                };
                let resp = format!(
                    "HTTP/1.1 {status}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = sock.write_all(resp.as_bytes()).await;
                let _ = sock.shutdown().await;
            }
        });
        format!("http://{addr}")
    }

    async fn download_from(status: &'static str, body: &'static str) -> Result<String, DownloadError> {
        let base = upstream(status, body).await;
        let http = reqwest::Client::new();
        Client { http: &http, api_key: "k", token: None, api_base: &base }.download(1).await
    }

    /// The error has to say whether the FILE or the SERVICE failed, because the pinned translation
    /// source is forgotten on one and kept on the other — and keeping it through a quota exhaustion
    /// is what stops the addon buying a replacement source, with a credit it does not have, for a
    /// source that was never the problem.
    #[tokio::test]
    async fn a_download_error_says_whose_fault_it_is() {
        // The CDN's verdicts are SUSPECT, never `Gone`. A 404 on a one-shot link means it expired,
        // and a cue-less 200 is usually an interstitial — both clear on their own, and only `Gone`
        // is allowed to drop the source pin shared by every install and language. A repeat promotes
        // them; one does not.
        let err = download_from("404 Not Found", "gone").await.expect_err("404 must fail");
        assert!(matches!(err, DownloadError::Suspect(_)), "an expired link unpinned the source: {err:?}");

        let err = download_from("200 OK", "<html>not a subtitle</html>").await.expect_err("must fail");
        assert!(matches!(err, DownloadError::Suspect(_)), "an interstitial unpinned the source: {err:?}");

        // Every other CDN status is the same kind of fact — the link did not work — and none of them
        // may unpin either. `download_from` drives the CDN leg only; the fake API always answers.
        for status in
            ["406 Not Acceptable", "429 Too Many Requests", "503 Service Unavailable", "403 Forbidden"]
        {
            let err = download_from(status, "nope").await.expect_err("must fail");
            assert!(matches!(err, DownloadError::Suspect(_)), "{status} from the CDN: {err:?}");
        }

        // The API's own verdict on the id is the one signal that names the FILE, and the only one
        // trusted to drop a shared pin. Everything else it can say is about the credential.
        assert!(matches!(
            DownloadError::from_status(reqwest::StatusCode::NOT_FOUND, "opensubtitles download 404".into()),
            DownloadError::Gone(_)
        ));
        assert!(matches!(
            DownloadError::from_status(reqwest::StatusCode::GONE, "opensubtitles download 410".into()),
            DownloadError::Gone(_)
        ));
        for code in [
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            reqwest::StatusCode::NOT_ACCEPTABLE,
            reqwest::StatusCode::UNAUTHORIZED,
            reqwest::StatusCode::SERVICE_UNAVAILABLE,
        ] {
            assert!(
                matches!(DownloadError::from_status(code, "x".into()), DownloadError::Unavailable(_)),
                "the API's {code} was blamed on the file"
            );
        }
    }

    /// The API call was status-checked and the CDN fetch that follows it was not, so an expired or
    /// rate-limited link returned its error page AS the subtitle — stored under the file id for 60
    /// days and served `immutable`. One blip, one track that silently shows nothing forever.
    #[tokio::test]
    async fn a_failed_link_fetch_is_not_a_subtitle() {
        let err = download_from("403 Forbidden", "<html><body>Forbidden</body></html>")
            .await
            .expect_err("a 403 from the CDN must not become the subtitle");
        let err = err.message();
        assert!(err.contains("403"), "unexpected error: {err}");
    }

    /// And a 200 is not proof either — a CDN interstitial is a 200 often enough.
    #[tokio::test]
    async fn a_link_body_with_no_cues_is_not_a_subtitle() {
        assert!(download_from("200 OK", "<html>just a page</html>").await.is_err());
        assert!(download_from("200 OK", "").await.is_err());
    }

    #[tokio::test]
    async fn a_real_subtitle_comes_back_intact() {
        let srt = "1\n00:00:01,000 --> 00:00:02,000\nhello\n";
        assert_eq!(download_from("200 OK", srt).await.unwrap(), srt);
    }
}
