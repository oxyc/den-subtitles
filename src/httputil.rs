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
    format!("\"{:016x}-{:x}\"", stable_hash(bytes), bytes.len())
}

/// FNV-1a-64: a FIXED, non-crypto hash — deterministic across restarts AND toolchain versions.
///
/// Anything whose value outlives the process must use this rather than std's `DefaultHasher`, whose
/// algorithm std may change between compiler releases. A toolchain bump would otherwise silently
/// rename every hashed cache file, reset every install's daily allowance, and move every failure
/// marker — all at once, and invisibly.
pub fn stable_hash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3); // FNV prime
    }
    h
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
pub async fn to_vtt(resp: Response<Body>, req_headers: &HeaderMap) -> Response<Body> {
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

    // The VTT ETag is derived from the SRT one rather than from the converted bytes, so a
    // conditional request can be answered WITHOUT converting. Conversion is a full parse and
    // re-serialize of the whole document, and `apply_conditional` runs after the router — so every
    // `If-None-Match` hit was paying for a body it then threw away, on the one runtime thread.
    // Derived, not invented: the SRT ETag already changes whenever the bytes do.
    /// Bumped whenever `serialize_vtt` changes what it emits. The validator is derived from the SRT
    /// ETag, which does not move when the CONVERTER moves — so without this, changing the rendering
    /// (as escaping the payload did) leaves every revalidating client and proxy on the old output.
    const RENDERING: u32 = 2;

    let vtt_etag = resp
        .headers()
        .get(ETAG)
        .and_then(|v| v.to_str().ok())
        .map(|srt_etag| {
            let mut seed = srt_etag.as_bytes().to_vec();
            seed.extend_from_slice(&RENDERING.to_le_bytes());
            format!("\"{:016x}-vtt\"", stable_hash(&seed))
        });
    if let Some(etag) = &vtt_etag {
        if req_headers.get(IF_NONE_MATCH).is_some_and(|inm| if_none_match_matches(inm, &HeaderValue::from_str(etag).unwrap())) {
            return Response::builder()
                .status(StatusCode::NOT_MODIFIED)
                .header(CACHE_CONTROL, cache_control)
                .header(ETAG, etag)
                .body(Full::new(Bytes::new()))
                .unwrap();
        }
    }
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
        .header(ETAG, vtt_etag.unwrap_or_else(|| etag_of(vtt.as_bytes())))
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

#[cfg(test)]
mod tests {
    use super::*;

    const SRT: &str = "1\n00:00:01,000 --> 00:00:02,000\n<i>Hello</i> & goodbye\n";

    async fn body_of(resp: Response<Body>) -> String {
        use http_body_util::BodyExt;
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    #[tokio::test]
    async fn vtt_conversion_rewrites_the_body_and_the_validator() {
        let out = to_vtt(srt(SRT.to_string()), &HeaderMap::new()).await;
        assert_eq!(out.status(), StatusCode::OK);
        assert_eq!(out.headers().get(CONTENT_TYPE).unwrap(), "text/vtt; charset=utf-8");
        // The caching directive carries over: re-rendering does not change whether the body is
        // settled or provisional.
        assert!(out.headers().get(CACHE_CONTROL).unwrap().to_str().unwrap().contains("immutable"));
        // A `.vtt` and a `.srt` of the same subtitle are different bytes and must not share a
        // validator, or a client that fetched one gets a 304 for the other.
        let srt_etag = srt(SRT.to_string()).headers().get(ETAG).unwrap().clone();
        let vtt_etag = out.headers().get(ETAG).unwrap().clone();
        assert_ne!(&vtt_etag, &srt_etag);

        // And the validator must depend on the RENDERING VERSION, not only on the SRT bytes. It is
        // derived from the SRT ETag, which does not move when the converter moves — so without the
        // version folded in, changing what `serialize_vtt` emits (as escaping the payload did) leaves
        // every revalidating client on the old output. Asserting the derivation is not the bare hash
        // of the SRT ETag is what pins the fold; a test comparing it to the SRT ETag alone passes
        // with the fold deleted, since the `-vtt` suffix already differs.
        let unversioned = format!("\"{:016x}-vtt\"", stable_hash(srt_etag.to_str().unwrap().as_bytes()));
        assert_ne!(
            vtt_etag.to_str().unwrap(),
            unversioned.as_str(),
            "the VTT validator ignores the rendering version"
        );

        let body = body_of(out).await;
        assert!(body.starts_with("WEBVTT"));
        assert!(body.contains("<i>Hello</i> &amp; goodbye"), "payload not rendered as VTT: {body:?}");
    }

    /// A provisional body stays provisional through the conversion — `immutable` on a stand-in would
    /// pin a client to an unaligned subtitle for a year.
    #[tokio::test]
    async fn a_provisional_body_is_still_provisional_as_vtt() {
        let out = to_vtt(srt_provisional(SRT.to_string()), &HeaderMap::new()).await;
        let cc = out.headers().get(CACHE_CONTROL).unwrap().to_str().unwrap().to_string();
        assert!(!cc.contains("immutable"), "a stand-in became immutable: {cc}");
        assert!(cc.contains("must-revalidate"));
    }

    /// A conditional request is answered WITHOUT converting. Conversion is a full parse and
    /// re-serialize of the document, and `apply_conditional` runs after the router — so every 304
    /// was paying for a body it then threw away.
    #[tokio::test]
    async fn a_conditional_vtt_request_is_answered_without_converting() {
        let first = to_vtt(srt(SRT.to_string()), &HeaderMap::new()).await;
        let etag = first.headers().get(ETAG).unwrap().clone();

        let mut headers = HeaderMap::new();
        headers.insert(IF_NONE_MATCH, etag.clone());
        let second = to_vtt(srt(SRT.to_string()), &headers).await;

        assert_eq!(second.status(), StatusCode::NOT_MODIFIED);
        assert_eq!(second.headers().get(ETAG).unwrap(), &etag);
        assert!(second.headers().get(CACHE_CONTROL).is_some(), "a 304 keeps its caching directive");
        assert!(body_of(second).await.is_empty(), "a 304 has no body");
    }

    /// Anything that is not a subtitle body passes through untouched — an error, or a body some
    /// other part of the router already shaped.
    #[tokio::test]
    async fn a_non_subtitle_response_passes_through() {
        let err = to_vtt(text(StatusCode::BAD_GATEWAY, "upstream said no"), &HeaderMap::new()).await;
        assert_eq!(err.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(body_of(err).await, "upstream said no");

        let js = to_vtt(json(StatusCode::OK, &"x", "no-store"), &HeaderMap::new()).await;
        assert_eq!(js.headers().get(CONTENT_TYPE).unwrap(), "application/json");
    }
}
