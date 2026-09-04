//! Small hyper response helpers (same shape as den-reel's httputil).

use bytes::Bytes;
use http_body_util::Full;
use hyper::header::{HeaderMap, HeaderValue, CACHE_CONTROL, CONTENT_TYPE, ETAG, IF_NONE_MATCH};
use hyper::{Response, StatusCode};
use serde::Serialize;

pub type Body = Full<Bytes>;

/// A strong, quoted ETag derived from the response body. A fast non-crypto hash (std
/// `DefaultHasher`) is plenty — an ETag only needs to change when the bytes change, not resist an
/// adversary. Length is folded in as a cheap extra guard against hash collisions.
fn etag_of(bytes: &[u8]) -> String {
    // FNV-1a-64: a FIXED, non-crypto hash — deterministic across restarts AND toolchain versions
    // (unlike std's DefaultHasher, whose algorithm std may change between compiler releases, which
    // would shift every ETag once). Length is folded in as a cheap extra guard against collisions.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3); // FNV prime
    }
    format!("\"{h:016x}-{:x}\"", bytes.len())
}

/// Whether a response is a cacheable success that should carry a validator (ETag): a 200 with a
/// caching directive that isn't `no-store`. Errors and `no-store` bodies never get an ETag.
fn cacheable(status: StatusCode, cache_control: &str) -> bool {
    status == StatusCode::OK && !cache_control.is_empty() && !cache_control.contains("no-store")
}

/// Text replies are always `no-store`: these are errors (400/404/502/503) and a transient upstream
/// failure must never be cached — a cached 502 would wedge a subtitle until the entry expired.
pub fn text(status: StatusCode, body: &str) -> Response<Body> {
    Response::builder()
        .status(status)
        .header(CONTENT_TYPE, "text/plain; charset=utf-8")
        .header(CACHE_CONTROL, "no-store")
        .body(Full::new(Bytes::from(body.to_owned())))
        .unwrap()
}

pub fn html(status: StatusCode, body: &'static str, cache_control: &str) -> Response<Body> {
    let mut b = Response::builder()
        .status(status)
        .header(CONTENT_TYPE, "text/html; charset=utf-8")
        .header(CACHE_CONTROL, HeaderValue::from_str(cache_control).unwrap());
    if cacheable(status, cache_control) {
        b = b.header(ETAG, etag_of(body.as_bytes()));
    }
    b.body(Full::new(Bytes::from_static(body.as_bytes()))).unwrap()
}

pub fn json<T: Serialize>(status: StatusCode, value: &T, cache_control: &str) -> Response<Body> {
    let bytes = serde_json::to_vec(value).unwrap_or_else(|_| b"{}".to_vec());
    let mut b = Response::builder()
        .status(status)
        .header(CONTENT_TYPE, "application/json");
    if !cache_control.is_empty() {
        b = b.header(CACHE_CONTROL, cache_control);
    }
    if cacheable(status, cache_control) {
        b = b.header(ETAG, etag_of(&bytes));
    }
    b.body(Full::new(Bytes::from(bytes))).unwrap()
}

/// A settled subtitle: this file_id/sync variant is byte-stable forever, so `immutable` and a year.
pub fn srt(body: String) -> Response<Body> {
    srt_cached(body, "public, max-age=31536000, immutable")
}

/// A subtitle we may yet serve differently — the requested alignment failed and this is the raw
/// fallback. `immutable` would pin the client to it for a year, so it must revalidate instead.
pub fn srt_provisional(body: String) -> Response<Body> {
    srt_cached(body, "public, max-age=60, must-revalidate")
}

fn srt_cached(body: String, cache_control: &str) -> Response<Body> {
    let etag = etag_of(body.as_bytes());
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "application/x-subrip; charset=utf-8")
        .header(CACHE_CONTROL, cache_control)
        .header(ETAG, etag)
        .body(Full::new(Bytes::from(body)))
        .unwrap()
}

/// Re-render a subtitle response as WebVTT.
///
/// Applied at the router, once, rather than threaded through the sync ladder as a format flag: every
/// path that can produce a subtitle — a cache hit, a fresh download, an alignment, a translation, the
/// provisional fallback — would otherwise need to carry it, and each is a place to get it wrong.
///
/// Anything that is not a subtitle body (an error, a 304) passes through untouched. The ETag is
/// recomputed because the bytes are genuinely different, and the caching directive is carried over
/// because whether the body is settled or provisional is not changed by re-rendering it.
pub async fn to_vtt(resp: Response<Body>) -> Response<Body> {
    use http_body_util::BodyExt;

    let is_subtitle = resp
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("application/x-subrip"));
    if resp.status() != StatusCode::OK || !is_subtitle {
        return resp;
    }
    let cache_control = resp
        .headers()
        .get(CACHE_CONTROL)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("no-store")
        .to_string();
    // `Full` is already in memory, so this await resolves immediately and cannot stall the thread.
    let bytes = match resp.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(_) => return text(StatusCode::INTERNAL_SERVER_ERROR, "subtitle body unavailable"),
    };
    let Ok(srt) = std::str::from_utf8(&bytes) else {
        return text(StatusCode::INTERNAL_SERVER_ERROR, "subtitle body was not utf-8");
    };
    let vtt = crate::srt::serialize_vtt(&crate::srt::parse(srt));
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "text/vtt; charset=utf-8")
        .header(CACHE_CONTROL, cache_control)
        .header(ETAG, etag_of(vtt.as_bytes()))
        .body(Full::new(Bytes::from(vtt)))
        .unwrap()
}

/// Honor a conditional GET: if the request's `If-None-Match` matches the response's `ETag`,
/// collapse to a `304 Not Modified` that keeps the `ETag` + `Cache-Control` headers and drops the
/// body. A no-op for responses without an ETag (errors, `no-store`) or a non-matching request.
pub fn apply_conditional(resp: Response<Body>, req_headers: &HeaderMap) -> Response<Body> {
    let Some(etag) = resp.headers().get(ETAG) else {
        return resp;
    };
    let matched = req_headers
        .get(IF_NONE_MATCH)
        .is_some_and(|inm| if_none_match_matches(inm, etag));
    if !matched {
        return resp;
    }
    let mut b = Response::builder().status(StatusCode::NOT_MODIFIED);
    let headers = b.headers_mut().expect("fresh builder has headers");
    if let Some(v) = resp.headers().get(ETAG) {
        headers.insert(ETAG, v.clone());
    }
    if let Some(v) = resp.headers().get(CACHE_CONTROL) {
        headers.insert(CACHE_CONTROL, v.clone());
    }
    b.body(Full::new(Bytes::new())).unwrap()
}

/// A HEAD response must not carry a body (RFC 9110 §9.3.2) — drop it, keeping every header (including
/// the `Content-Length` a GET would have returned). The router builds full-body responses regardless of
/// method and hyper does not auto-strip, so the HEAD path must do it explicitly.
pub fn strip_body(resp: Response<Body>) -> Response<Body> {
    let (parts, _body) = resp.into_parts();
    Response::from_parts(parts, Full::new(Bytes::new()))
}

/// RFC 9110 `If-None-Match`: `*` matches anything; otherwise any entry in the comma-separated list
/// that equals the ETag matches. Our ETags are strong, but we compare with the weak-validator
/// prefix (`W/`) stripped from both sides so a proxy that weakened it still gets its 304.
fn if_none_match_matches(inm: &HeaderValue, etag: &HeaderValue) -> bool {
    let (Ok(inm), Ok(etag)) = (inm.to_str(), etag.to_str()) else {
        return false;
    };
    let etag = etag.trim_start_matches("W/");
    inm.split(',').any(|candidate| {
        let candidate = candidate.trim();
        candidate == "*" || candidate.trim_start_matches("W/") == etag
    })
}

/// Minimal percent-decode (`%XX` + `+`→space) — enough for the extra-args a Stremio client sends.
pub fn percent_decode(s: &str) -> String {
    decode(s, true)
}

/// Percent-decoding for a PATH segment, where `+` is a literal plus. Form-decoding a path turns
/// "HDR10+.WEB-DL" into "HDR10 .WEB-DL", which is then what the release name is ranked on.
pub fn percent_decode_path(s: &str) -> String {
    decode(s, false)
}

fn decode(s: &str, plus_is_space: bool) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hi = hex(bytes[i + 1]);
                let lo = hex(bytes[i + 2]);
                if let (Some(h), Some(l)) = (hi, lo) {
                    out.push(h << 4 | l);
                    i += 3;
                    continue;
                }
                out.push(b'%');
                i += 1;
            }
            b'+' if plus_is_space => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}
