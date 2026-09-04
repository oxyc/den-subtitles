//! Bounded body readers. Upstream bodies — OpenSubtitles JSON, the OpenSubtitles-supplied download
//! link (points wherever they say), and LLM responses — are read with a hard byte ceiling so a
//! hostile or runaway response can't OOM the container. We reject early on an oversized declared
//! `Content-Length` and, for chunked bodies with none, abort once the accumulated bytes cross the
//! cap.

use futures_util::StreamExt;
use serde::de::DeserializeOwned;

/// Generous ceiling: an SRT is tens of KB, an LLM JSON reply a few hundred; 12 MiB is far above any
/// legitimate body while still bounding memory per in-flight request.
pub const MAX_BODY: usize = 12 * 1024 * 1024;

/// Read a response body into memory, capped at `max` bytes.
pub async fn capped_bytes(resp: reqwest::Response, max: usize) -> Result<Vec<u8>, String> {
    if let Some(len) = resp.content_length() {
        if len as usize > max {
            return Err(format!("upstream body too large: {len} bytes"));
        }
    }
    let mut stream = resp.bytes_stream();
    let mut out: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("read body: {e}"))?;
        if out.len() + chunk.len() > max {
            return Err(format!("upstream body exceeded {max} bytes"));
        }
        out.extend_from_slice(&chunk);
    }
    Ok(out)
}

/// Capped body → lossy UTF-8 text.
///
/// Lossy is safe here because of a guarantee that lives outside this repo: the api.opensubtitles.com
/// `/download` endpoint converts on its side, so the file behind the temporary link it hands back is
/// UTF-8 whatever encoding the uploader used. Subtitle files in the wild are full of Windows-1251 and
/// ISO-8859-2, and this function would turn one of those into replacement characters silently —
/// `srt::has_a_cue` only inspects timing and index lines, which are ASCII either way, so a mojibake'd
/// file passes every check and caches for sixty days looking fine.
///
/// Written down because nothing else records it and no test can pin it. If a second path ever fetches
/// a subtitle from somewhere that makes no such promise, this is the line that has to change first.
pub async fn capped_text(resp: reqwest::Response, max: usize) -> Result<String, String> {
    let bytes = capped_bytes(resp, max).await?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Capped body → deserialized JSON.
pub async fn capped_json<T: DeserializeOwned>(resp: reqwest::Response, max: usize) -> Result<T, String> {
    let bytes = capped_bytes(resp, max).await?;
    serde_json::from_slice(&bytes).map_err(|e| format!("bad json: {e}"))
}
