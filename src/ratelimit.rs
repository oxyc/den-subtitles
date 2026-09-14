//! Rate-limit pauses shared across requests, and the header parsing that feeds them.
//!
//! A service that says "slow down" is saying it about the CREDENTIAL, not about the one request that
//! happened to hear it. Remembering the refusal only where it landed — one title, one file, one batch
//! — left every other request on the same key asking a service that had already refused it, which is
//! how a rate limit turns into a longer one. A `Pauses` is that memory: one entry per hashed
//! credential, consulted before any call and cleared by the first answer.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::header::{HeaderMap, RETRY_AFTER};
use reqwest::StatusCode;
use tokio::time::Instant;

/// Longest wait honoured from a service's own headers, or reached by our backoff. A service naming
/// a day in `Retry-After` is more likely confused than serious, and an hour of silence is already
/// long enough for anyone to notice.
pub const MAX_PAUSE: Duration = Duration::from_secs(60 * 60);
/// Credentials tracked at once. An entry is a few bytes, but the keys come from install URLs, which
/// anyone can mint, so the map needs a ceiling like everything else keyed by them.
const MAX_KEYS: usize = 1024;
/// The first wait when a service refuses without saying how long; doubled per refusal in a row.
const BASE_BACKOFF: Duration = Duration::from_secs(2);

static JITTER_TICK: AtomicU64 = AtomicU64::new(0);

/// A hashed credential. Hashed so a key never sits in memory in a second place, and scoped so the
/// same string used for two services does not share one pause.
pub fn key(scope: &str, credential: &str) -> u64 {
    crate::httputil::stable_hash(format!("{scope}\n{credential}").as_bytes())
}

/// Is this refusal a rate limit? A 429 always; a 503 only when it names a wait, since a bare 503 is
/// the service being unwell rather than the credential being told to stop.
pub fn throttled(status: StatusCode, headers: &HeaderMap) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS
        || (status == StatusCode::SERVICE_UNAVAILABLE && headers.contains_key(RETRY_AFTER))
}

#[derive(Default)]
pub struct Pauses {
    inner: Mutex<HashMap<u64, Pause>>,
}

struct Pause {
    until: Instant,
    /// Unstated refusals in a row, for the exponential backoff. Reset by an answer.
    strikes: u32,
}

impl Pauses {
    /// How much longer this credential must wait, or `None` if it may call now.
    pub fn remaining(&self, key: u64) -> Option<Duration> {
        let g = self.inner.lock().unwrap();
        let left = g.get(&key)?.until.checked_duration_since(Instant::now())?;
        (!left.is_zero()).then_some(left)
    }

    /// The service refused on rate grounds. `stated` is its own `Retry-After`, if it gave one;
    /// otherwise the wait doubles per refusal in a row, with jitter so the requests it held do not
    /// all come back in the same instant. Returns the wait now in force.
    pub fn limited(&self, key: u64, stated: Option<Duration>) -> Duration {
        let now = Instant::now();
        let mut g = self.inner.lock().unwrap();
        make_room(&mut g, key, now);
        let p = g.entry(key).or_insert(Pause { until: now, strikes: 0 });
        match stated {
            Some(wait) => p.until = p.until.max(now + wait.min(MAX_PAUSE)),
            // Already paused: this is a request that was in flight when the first refusal landed.
            // Same incident, so it must not count as another strike and double the wait.
            None if p.until > now => {}
            None => {
                p.strikes = p.strikes.saturating_add(1);
                let base = BASE_BACKOFF.saturating_mul(1 << (p.strikes - 1).min(20));
                // Up to a quarter again, from a counter rather than the clock so it is testable.
                let spread = (JITTER_TICK.fetch_add(1, Ordering::Relaxed) * 373) % 250;
                let wait = (base + base.mul_f64(spread as f64 / 1000.0)).min(MAX_PAUSE);
                p.until = now + wait;
            }
        }
        p.until - now
    }

    /// Hold the credential for a known span — a window the service said is used up, or a daily quota.
    /// Not capped here: a quota's reset is a fact, not a guess, and the caller bounds it.
    pub fn hold(&self, key: u64, wait: Duration) {
        let now = Instant::now();
        let mut g = self.inner.lock().unwrap();
        make_room(&mut g, key, now);
        let p = g.entry(key).or_insert(Pause { until: now, strikes: 0 });
        p.until = p.until.max(now + wait);
    }

    /// The service answered. That ends the pause and the run of strikes behind it.
    pub fn answered(&self, key: u64) {
        self.inner.lock().unwrap().remove(&key);
    }
}

/// Keep the map under `MAX_KEYS` before inserting `key`. Lapsed pauses go first — losing one costs
/// only its strike count — and if every entry is live, the one closest to lapsing.
fn make_room(g: &mut HashMap<u64, Pause>, key: u64, now: Instant) {
    if g.len() < MAX_KEYS || g.contains_key(&key) {
        return;
    }
    g.retain(|_, p| p.until > now);
    if g.len() >= MAX_KEYS {
        if let Some(soonest) = g.iter().min_by_key(|(_, p)| p.until).map(|(k, _)| *k) {
            g.remove(&soonest);
        }
    }
}

/// A response's `Retry-After`, capped at `MAX_PAUSE`. `None` when there is none or it cannot be read,
/// which leaves the caller on its own backoff — the safe direction.
pub fn retry_after(headers: &HeaderMap, now: SystemTime) -> Option<Duration> {
    parse_retry_after(headers.get(RETRY_AFTER)?.to_str().ok()?, now)
}

/// Delay-seconds or an HTTP-date. Only the IMF-fixdate form of the date is read; RFC 9110's two
/// obsolete forms fall back to our own backoff rather than being guessed at. A date in the past is
/// "now".
pub fn parse_retry_after(value: &str, now: SystemTime) -> Option<Duration> {
    let value = value.trim();
    let wait = match value.parse::<u64>() {
        Ok(secs) => Duration::from_secs(secs),
        Err(_) => until(http_date(value)?, now),
    };
    Some(wait.min(MAX_PAUSE))
}

/// How long until the window reopens, when a successful response says none of it is left.
///
/// OpenSubtitles sends `X-RateLimit-Remaining` / `X-RateLimit-Reset` and the IETF draft's
/// `RateLimit-Remaining` / `RateLimit-Reset`. Reset is delta seconds in both; a value large enough to
/// be a unix time is read as one, since some services send that instead.
pub fn exhausted_for(headers: &HeaderMap, now: SystemTime) -> Option<Duration> {
    let num = |name: &str| headers.get(name)?.to_str().ok()?.trim().parse::<u64>().ok();
    let (remaining, reset) = match num("x-ratelimit-remaining") {
        Some(r) => (r, num("x-ratelimit-reset")),
        None => (num("ratelimit-remaining")?, num("ratelimit-reset")),
    };
    if remaining > 0 {
        return None;
    }
    let reset = reset?;
    let wait = match reset > 1_000_000_000 {
        true => until(reset as i64, now),
        false => Duration::from_secs(reset),
    };
    Some(wait.min(MAX_PAUSE))
}

/// Time left in the current UTC day — the bucket a daily quota counted by `unix_seconds / 86400`
/// resets on.
pub fn until_utc_midnight(now: SystemTime) -> Duration {
    let secs = now.duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    Duration::from_secs(86_400 - secs % 86_400)
}

/// From `now` until a unix time; zero if it has passed.
pub fn until(at_unix: i64, now: SystemTime) -> Duration {
    let now = now.duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    Duration::from_secs(at_unix.saturating_sub(now).max(0) as u64)
}

/// An ISO 8601 UTC timestamp as OpenSubtitles writes `reset_time_utc`: `2022-01-30T06:00:47.000Z`.
pub fn iso_utc(s: &str) -> Option<i64> {
    let (date, time) = s.trim().split_once('T')?;
    let time = time.strip_suffix('Z')?.split('.').next()?;
    let mut d = date.split('-');
    let (year, month, day) = (d.next()?.parse().ok()?, d.next()?.parse().ok()?, d.next()?.parse().ok()?);
    if d.next().is_some() {
        return None;
    }
    unix_time(year, month, day, time)
}

const MONTHS: [&str; 12] =
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/// IMF-fixdate: `Sun, 06 Nov 1994 08:49:37 GMT`.
fn http_date(s: &str) -> Option<i64> {
    let mut parts = s.split_ascii_whitespace();
    parts.next()?.strip_suffix(',')?;
    let day = parts.next()?.parse().ok()?;
    let mon = parts.next()?;
    let month = MONTHS.iter().position(|m| m.eq_ignore_ascii_case(mon))? as u32 + 1;
    let year = parts.next()?.parse().ok()?;
    let time = parts.next()?;
    if parts.next()? != "GMT" || parts.next().is_some() {
        return None;
    }
    unix_time(year, month, day, time)
}

fn unix_time(year: i64, month: u32, day: u32, hms: &str) -> Option<i64> {
    let mut t = hms.split(':');
    let (h, m, s): (u32, u32, u32) =
        (t.next()?.parse().ok()?, t.next()?.parse().ok()?, t.next()?.parse().ok()?);
    if t.next().is_some()
        || !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || h > 23
        || m > 59
        || s > 60
    {
        return None;
    }
    Some(days_from_civil(year, month, day) * 86_400 + i64::from(h * 3600 + m * 60 + s))
}

/// Days since 1970-01-01 in the proleptic Gregorian calendar (Howard Hinnant's algorithm).
fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = i64::from((month + 9) % 12);
    let doy = (153 * mp + 2) / 5 + i64::from(day) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(unix: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(unix)
    }

    #[test]
    fn retry_after_reads_seconds_dates_and_refuses_garbage() {
        // RFC 9110's own example date.
        let date = 784_111_777;
        assert_eq!(parse_retry_after("120", at(0)), Some(Duration::from_secs(120)));
        assert_eq!(parse_retry_after(" 7 ", at(0)), Some(Duration::from_secs(7)));
        assert_eq!(
            parse_retry_after("Sun, 06 Nov 1994 08:49:37 GMT", at(date - 90)),
            Some(Duration::from_secs(90))
        );
        // A date already past means "now", not a negative wait or a parse failure.
        assert_eq!(parse_retry_after("Sun, 06 Nov 1994 08:49:37 GMT", at(date + 5)), Some(Duration::ZERO));
        // Both forms are capped: a stated day is an hour.
        assert_eq!(parse_retry_after("86400", at(0)), Some(MAX_PAUSE));
        assert_eq!(parse_retry_after("Sun, 06 Nov 1994 08:49:37 GMT", at(0)), Some(MAX_PAUSE));
        for junk in [
            "",
            "soon",
            "-5",
            "1.5",
            "Sun, 06 Nov 1994 08:49:37",
            "Sun, 06 Foo 1994 08:49:37 GMT",
            "Sun, 32 Nov 1994 08:49:37 GMT",
            "Sunday, 06-Nov-94 08:49:37 GMT",
        ] {
            assert_eq!(parse_retry_after(junk, at(date)), None, "{junk:?}");
        }
    }

    #[test]
    fn iso_reset_times_parse_as_opensubtitles_writes_them() {
        assert_eq!(iso_utc("2022-01-30T06:00:47.000Z"), Some(1_643_522_447));
        assert_eq!(iso_utc("2022-01-30T06:00:47Z"), Some(1_643_522_447));
        assert_eq!(iso_utc("1970-01-01T00:00:00Z"), Some(0));
        for junk in ["", "2022-01-30 06:00:47", "2022-01-30T06:00:47", "2022-13-01T00:00:00Z", "tomorrow"] {
            assert_eq!(iso_utc(junk), None, "{junk:?}");
        }
    }

    #[test]
    fn an_exhausted_window_pauses_until_its_reset() {
        let mut h = HeaderMap::new();
        h.insert("x-ratelimit-remaining", "0".parse().unwrap());
        h.insert("x-ratelimit-reset", "9".parse().unwrap());
        assert_eq!(exhausted_for(&h, at(1_000)), Some(Duration::from_secs(9)));
        // A unix-time reset reads as one.
        h.insert("x-ratelimit-reset", "1700000030".parse().unwrap());
        assert_eq!(exhausted_for(&h, at(1_700_000_000)), Some(Duration::from_secs(30)));
        // Something left is no pause.
        h.insert("x-ratelimit-remaining", "3".parse().unwrap());
        assert_eq!(exhausted_for(&h, at(0)), None);
        // The draft's names, when the X- ones are absent.
        let mut d = HeaderMap::new();
        d.insert("ratelimit-remaining", "0".parse().unwrap());
        d.insert("ratelimit-reset", "4".parse().unwrap());
        assert_eq!(exhausted_for(&d, at(0)), Some(Duration::from_secs(4)));
    }

    #[test]
    fn midnight_is_the_end_of_the_utc_day() {
        assert_eq!(until_utc_midnight(at(86_400 * 3)), Duration::from_secs(86_400));
        assert_eq!(until_utc_midnight(at(86_400 * 3 + 86_399)), Duration::from_secs(1));
    }

    #[tokio::test(start_paused = true)]
    async fn an_answer_ends_a_pause_and_its_strikes() {
        let p = Pauses::default();
        let k = key("test", "credential");
        assert_eq!(p.remaining(k), None);

        // Unstated refusals double, and a refusal from a request already in flight does not.
        let first = p.limited(k, None);
        assert!(first >= BASE_BACKOFF && first <= BASE_BACKOFF * 5 / 4, "{first:?}");
        assert!(p.limited(k, None) <= first, "an in-flight sibling doubled the wait");
        tokio::time::advance(first).await;
        let second = p.limited(k, None);
        assert!(second >= BASE_BACKOFF * 2, "a second refusal did not back off further: {second:?}");

        // A stated wait is taken as stated, never shortening one already in force.
        p.limited(k, Some(Duration::from_secs(30)));
        assert!(p.remaining(k).unwrap() > Duration::from_secs(29));

        p.answered(k);
        assert_eq!(p.remaining(k), None, "an answer left the credential paused");
        let fresh = p.limited(k, None);
        assert!(fresh <= BASE_BACKOFF * 5 / 4, "an answer did not reset the backoff: {fresh:?}");
    }

    #[test]
    fn the_map_is_bounded() {
        let p = Pauses::default();
        for i in 0..(MAX_KEYS as u64 + 50) {
            p.hold(i, Duration::from_secs(60 + i));
        }
        let g = p.inner.lock().unwrap();
        assert_eq!(g.len(), MAX_KEYS);
        // The newest survive; the ones closest to lapsing made room.
        assert!(g.contains_key(&(MAX_KEYS as u64 + 49)));
        assert!(!g.contains_key(&0));
    }
}
