//! What the process says on stderr. Kept to state changes and rate-limited failures, so an outage
//! costs the journal a line a minute rather than a line per request — and never an install's config
//! segment or a query string, both of which carry credentials.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Routes served at a fixed path. Every other path is `/<config>/…`, whose first segment is an
/// install's credentials (or a sealed blob of them).
const FIXED_ROUTES: [&str; 7] =
    ["/", "/health", "/metrics", "/manifest.json", "/configure", "/configure/", "/config-key"];

/// A request path fit for the log: the first segment of anything but a fixed route becomes
/// `<config>`. An unknown one-segment path is redacted too, since it may be a config segment sent
/// with no resource after it. Takes the path alone because the query is never logged: `?resync=`
/// carries a stream URL with the provider's token in it.
pub fn redact_path(path: &str) -> String {
    if FIXED_ROUTES.contains(&path) {
        return path.to_string();
    }
    let rest = path.trim_start_matches('/');
    match rest.find('/') {
        Some(i) => format!("/<config>{}", &rest[i..]),
        None => "/<config>".to_string(),
    }
}

/// The caller's `X-Request-Id`, fit for the log: only `[A-Za-z0-9_-]`, at most 32 characters. It is
/// what joins this line to the app's own log line for the same request. Anything else in the header is
/// dropped rather than escaped, so a caller cannot forge a second field or a second line; nothing left
/// means no id.
pub fn request_id(headers: &hyper::HeaderMap) -> Option<String> {
    let raw = headers.get("x-request-id")?.to_str().ok()?;
    let id: String =
        raw.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-').take(32).collect();
    (!id.is_empty()).then_some(id)
}

/// One request log line: `<METHOD> <redacted path> <status> <ms>ms[ rid=<id>]`.
pub fn request_line(method: &str, path: &str, status: u16, ms: u128, rid: Option<&str>) -> String {
    let line = format!("{method} {} {status} {ms}ms", redact_path(path));
    match rid {
        Some(rid) => format!("{line} rid={rid}"),
        None => line,
    }
}

/// One structured decision-log line, in the convention shared across den services: `event=<token>
/// outcome=<served|skipped|refused|degraded|fallback> reason=<text> [upstream=<name>] [dur_ms=<n>]`
/// plus identity fields, then `rid=<id>` last.
///
/// Identity fields — title id, season/episode, file ids, language, release names — go through `id`,
/// which drops them outright when `identity` (the `LOG_IDENTITY` env flag, read once in `Config`) is
/// off. Everything else on the line is unaffected: the flag exists to let a deploy that must not
/// keep that detail turn it off, not to make the detail that makes a bug reproducible the opt-in
/// case.
///
/// Nothing here decodes a config segment or a query string — the two places a credential would come
/// from — so a caller would have to go out of its way to end up logging one. It still has to not do
/// that: `Line` sanitizes a value for the LINE FORMAT (no stray spaces), not for secrecy.
pub struct Line {
    buf: String,
    identity: bool,
}

impl Line {
    pub fn new(event: &str, outcome: &str, identity: bool) -> Line {
        let mut buf = String::new();
        let _ = write!(buf, "event={event} outcome={outcome}");
        Line { buf, identity }
    }

    /// A short token saying why, e.g. `hash_match`, `sync_failed`, `upstream_unavailable`. Free text
    /// is fine too — `field` turns any whitespace in it to `_` so it still reads as one token.
    pub fn reason(self, reason: &str) -> Line {
        self.field("reason", reason)
    }

    /// The external system this decision depended on (`opensubtitles`, `translator`), when there is
    /// one.
    pub fn upstream(self, upstream: &str) -> Line {
        self.field("upstream", upstream)
    }

    pub fn dur_ms(self, ms: u128) -> Line {
        self.field("dur_ms", ms)
    }

    /// A value that is never identifying on its own — a count, a duration, a flag, a cache
    /// hit/miss — written whatever `identity` is.
    pub fn field(mut self, key: &str, value: impl std::fmt::Display) -> Line {
        let _ = write!(self.buf, " {key}={}", sanitized(&value.to_string()));
        self
    }

    /// An identity field — title id, season/episode, a file id, a language, a release name. Dropped
    /// outright when `identity` is off; everything else on the line still reaches the log.
    pub fn id(self, key: &str, value: impl std::fmt::Display) -> Line {
        if self.identity {
            self.field(key, value)
        } else {
            self
        }
    }

    /// The caller's request id, meant to go last — the join key to the app's own line for the same
    /// request. A no-op when there isn't one.
    pub fn rid(self, rid: Option<&str>) -> Line {
        match rid {
            Some(r) => self.field("rid", r),
            None => self,
        }
    }

    pub fn finish(self) -> String {
        self.buf
    }
}

/// A logged value is one token: this line format has no quoting, so a raw space would read as a
/// second field. Mangles rather than rejects — a release name or a filename is exactly the kind of
/// thing worth keeping, underscored, over losing outright.
fn sanitized(s: &str) -> String {
    s.chars().map(|c| if c.is_whitespace() { '_' } else { c }).collect()
}

/// How often one kind of recurring failure may reach the log.
const LOG_EVERY_SECS: u64 = 60;

/// Lets one line per minute through for one kind of recurring failure. A single timestamp compared on
/// each occurrence — no timer, nothing running between failures.
#[derive(Default)]
pub struct LogGate(AtomicU64);

impl LogGate {
    pub const fn new() -> LogGate {
        LogGate(AtomicU64::new(0))
    }

    /// Whether this occurrence should be logged.
    pub fn allow(&self) -> bool {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        self.allow_at(now)
    }

    fn allow_at(&self, now: u64) -> bool {
        let last = self.0.load(Ordering::Relaxed);
        if last != 0 && now.saturating_sub(last) < LOG_EVERY_SECS {
            return false;
        }
        // Two occurrences racing past the check: only the one that moves the timestamp logs.
        self.0.compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_config_segment_never_reaches_the_log() {
        assert_eq!(
            redact_path("/eyJvc0tleSI6InNlY3JldCJ9/subtitles/movie/tt1.json"),
            "/<config>/subtitles/movie/tt1.json"
        );
        assert_eq!(redact_path("/Ac3WWHz/subtitle/42.srt"), "/<config>/subtitle/42.srt");
        assert_eq!(
            redact_path("/Ac3WWHz/translate/movie/tt1/Swedish.json"),
            "/<config>/translate/movie/tt1/Swedish.json"
        );
        // A config segment with nothing after it, and a path with extra leading slashes.
        assert_eq!(redact_path("/eyJvc0tleSI6InNlY3JldCJ9"), "/<config>");
        assert_eq!(redact_path("//eyJvc0tleSI6InNlY3JldCJ9/manifest.json"), "/<config>/manifest.json");
    }

    #[test]
    fn fixed_routes_are_logged_as_they_are() {
        for path in FIXED_ROUTES {
            assert_eq!(redact_path(path), path);
        }
    }

    fn rid(value: &str) -> Option<String> {
        let mut headers = hyper::HeaderMap::new();
        headers.insert("x-request-id", hyper::header::HeaderValue::from_str(value).unwrap());
        request_id(&headers)
    }

    #[test]
    fn a_request_line_carries_the_caller_s_id() {
        let id = rid("a1B2-c3_d4");
        assert_eq!(
            request_line("GET", "/Ac3WWHz/subtitle/42.srt", 200, 7, id.as_deref()),
            "GET /<config>/subtitle/42.srt 200 7ms rid=a1B2-c3_d4"
        );
    }

    #[test]
    fn a_hostile_request_id_is_stripped_and_truncated() {
        // Spaces and `=` would forge a field; only the safe characters survive, in order.
        assert_eq!(rid("ab cd=ef"), Some("abcdef".to_string()));
        assert_eq!(rid(&"x".repeat(100)), Some("x".repeat(32)));
        assert_eq!(rid("!!!"), None, "nothing safe left means no id");
    }

    #[test]
    fn without_a_request_id_the_line_is_unchanged() {
        assert_eq!(request_id(&hyper::HeaderMap::new()), None);
        assert_eq!(request_line("GET", "/health", 200, 0, None), "GET /health 200 0ms");
    }

    #[test]
    fn a_gate_lets_one_line_through_a_minute() {
        let gate = LogGate::new();
        assert!(gate.allow_at(1_000), "the first occurrence logs");
        assert!(!gate.allow_at(1_001));
        assert!(!gate.allow_at(1_000 + LOG_EVERY_SECS - 1));
        assert!(gate.allow_at(1_000 + LOG_EVERY_SECS), "a minute later it logs again");
        assert!(!gate.allow_at(1_000 + LOG_EVERY_SECS + 1));
    }

    #[test]
    fn identity_fields_are_dropped_when_the_flag_is_off() {
        let on = Line::new("rank", "served", true).id("imdb", "tt123").field("candidates", 5).finish();
        assert_eq!(on, "event=rank outcome=served imdb=tt123 candidates=5");
        let off = Line::new("rank", "served", false).id("imdb", "tt123").field("candidates", 5).finish();
        assert_eq!(off, "event=rank outcome=served candidates=5", "identity=false must drop imdb alone");
    }

    #[test]
    fn a_value_with_whitespace_cannot_forge_a_second_field() {
        let line = Line::new("rank", "served", true).id("release", "Fight Club 1999").finish();
        assert_eq!(line, "event=rank outcome=served release=Fight_Club_1999");
    }

    #[test]
    fn rid_is_a_no_op_when_absent_and_a_field_when_present() {
        let with_rid = Line::new("serve", "served", true).rid(Some("abc123")).finish();
        assert_eq!(with_rid, "event=serve outcome=served rid=abc123");
        let without = Line::new("serve", "served", true).rid(None).finish();
        assert_eq!(without, "event=serve outcome=served");
    }

    #[test]
    fn reason_upstream_and_dur_ms_read_as_their_own_fields() {
        let line = Line::new("serve", "degraded", true)
            .reason("sync_failed")
            .upstream("opensubtitles")
            .dur_ms(42)
            .finish();
        assert_eq!(line, "event=serve outcome=degraded reason=sync_failed upstream=opensubtitles dur_ms=42");
    }
}
