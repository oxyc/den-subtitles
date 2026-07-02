//! Small hyper response helpers (same shape as den-reel's httputil).

use bytes::Bytes;
use http_body_util::Full;
use hyper::header::{HeaderValue, CACHE_CONTROL, CONTENT_TYPE};
use hyper::{Response, StatusCode};
use serde::Serialize;

pub type Body = Full<Bytes>;

pub fn text(status: StatusCode, body: &str) -> Response<Body> {
    Response::builder()
        .status(status)
        .header(CONTENT_TYPE, "text/plain; charset=utf-8")
        .body(Full::new(Bytes::from(body.to_owned())))
        .unwrap()
}

pub fn html(status: StatusCode, body: &'static str, cache_control: &str) -> Response<Body> {
    Response::builder()
        .status(status)
        .header(CONTENT_TYPE, "text/html; charset=utf-8")
        .header(CACHE_CONTROL, HeaderValue::from_str(cache_control).unwrap())
        .body(Full::new(Bytes::from_static(body.as_bytes())))
        .unwrap()
}

pub fn json<T: Serialize>(status: StatusCode, value: &T, cache_control: &str) -> Response<Body> {
    let bytes = serde_json::to_vec(value).unwrap_or_else(|_| b"{}".to_vec());
    let mut b = Response::builder()
        .status(status)
        .header(CONTENT_TYPE, "application/json");
    if !cache_control.is_empty() {
        b = b.header(CACHE_CONTROL, cache_control);
    }
    b.body(Full::new(Bytes::from(bytes))).unwrap()
}

/// A subtitle payload served as text (SRT). Cached hard — a given translated/synced file is stable.
pub fn srt(body: String) -> Response<Body> {
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "application/x-subrip; charset=utf-8")
        .header(CACHE_CONTROL, "public, max-age=604800")
        .body(Full::new(Bytes::from(body)))
        .unwrap()
}

/// Minimal percent-decode (`%XX` + `+`→space) — enough for the extra-args a Stremio client sends.
pub fn percent_decode(s: &str) -> String {
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
            b'+' => {
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
