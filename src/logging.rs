//! What the process says on stderr. Kept to state changes and rate-limited failures, so an outage
//! costs the journal a line a minute rather than a line per request — and never an install's config
//! segment or a query string, both of which carry credentials.

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

    #[test]
    fn a_gate_lets_one_line_through_a_minute() {
        let gate = LogGate::new();
        assert!(gate.allow_at(1_000), "the first occurrence logs");
        assert!(!gate.allow_at(1_001));
        assert!(!gate.allow_at(1_000 + LOG_EVERY_SECS - 1));
        assert!(gate.allow_at(1_000 + LOG_EVERY_SECS), "a minute later it logs again");
        assert!(!gate.allow_at(1_000 + LOG_EVERY_SECS + 1));
    }
}
