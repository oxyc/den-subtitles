//! The translation harness. This is the part that makes "translate a whole 1200-cue film without
//! losing focus" actually work — the model never sees the full file, never sees timestamps, and is
//! held to a strict same-length contract per batch:
//!
//!   * cues are chunked into small batches (BATCH), so no single call carries the whole film;
//!   * each batch is sent as a JSON array of dialogue strings — timecodes/indices stay here;
//!   * the reply MUST be a JSON array of the same length; a mismatch splits the batch and retries
//!     (down to a single cue), so a merge/split/drop can never silently shift the rest of the film;
//!   * a rolling window of the last few (source → translation) pairs rides along as context, so
//!     names, tone and register stay consistent across batch boundaries (this is what beats a
//!     literal MT pass).
//!
//! Batches run sequentially: continuity (the rolling context) depends on the previous batch, and a
//! ~30-60s background job for a full film is well within the click-and-wait UX. Cheap models
//! (gpt-4o-mini / gemini-flash / haiku) clear the "good enough to follow the movie" bar here.

use std::time::Duration;

use serde_json::{json, Value};

use crate::srt::Cue;
use crate::userconfig::{LlmConfig, Provider};

/// Cues per model call. Small enough that a retry-on-mismatch reprocesses little work, large enough
/// to amortise the per-request latency and give the model intra-scene context.
const BATCH: usize = 40;
/// How many prior (source → translation) pairs to carry forward for cross-batch consistency.
const CONTEXT_WINDOW: usize = 6;
/// Per-batch upstream bound for an LLM/DeepL call (above the client default; a completion is slower
/// than a metadata fetch but must still be bounded).
const LLM_TIMEOUT: Duration = Duration::from_secs(120);
/// Ceiling on cues we'll translate for one title. A real film is ~1–3k cues; anything past this is a
/// pathological/hostile SRT that would run unbounded (cost, wall-clock), so we refuse it.
const MAX_CUES: usize = 6000;

/// Translate every cue's text into `target_lang` (a display name like "English"), preserving each
/// cue's index/timing. Returns the same cues with translated `text`, or an error string.
pub async fn translate(
    client: &reqwest::Client,
    llm: &LlmConfig,
    cues: &[Cue],
    target_lang: &str,
) -> Result<Vec<Cue>, String> {
    if cues.is_empty() {
        return Ok(Vec::new());
    }
    if cues.len() > MAX_CUES {
        return Err(format!("subtitle too large: {} cues (max {MAX_CUES})", cues.len()));
    }
    let mut out: Vec<Cue> = Vec::with_capacity(cues.len());
    // Rolling context: the tail of already-translated pairs, refreshed as we go.
    let mut context: Vec<(String, String)> = Vec::new();

    for batch in cues.chunks(BATCH) {
        let sources: Vec<String> = batch.iter().map(|c| c.text.clone()).collect();
        let translated = translate_batch(client, llm, &sources, target_lang, &context).await?;
        for (cue, text) in batch.iter().zip(translated) {
            context.push((cue.text.clone(), text.clone()));
            out.push(Cue { text, ..cue.clone() });
        }
        if context.len() > CONTEXT_WINDOW {
            context.drain(..context.len() - CONTEXT_WINDOW);
        }
    }
    Ok(out)
}

/// Translate one batch under the same-length contract. On a length mismatch, split and retry so a
/// single misbehaving batch degrades to smaller batches instead of corrupting the whole film.
async fn translate_batch(
    client: &reqwest::Client,
    llm: &LlmConfig,
    sources: &[String],
    target_lang: &str,
    context: &[(String, String)],
) -> Result<Vec<String>, String> {
    if sources.is_empty() {
        return Ok(Vec::new());
    }
    let result = match llm.provider {
        // DeepL is a 1:1 text-array MT endpoint, not a chat model — no prompt, no JSON contract.
        Provider::DeepL => deepl_translate(client, llm, sources, target_lang).await,
        _ => llm_translate(client, llm, sources, target_lang, context).await,
    };

    match result {
        Ok(v) if v.len() == sources.len() => Ok(v),
        Ok(_) | Err(_) if sources.len() > 1 => {
            // Split and retry each half. Context is best-effort continuity, not correctness, so we
            // don't thread the first half's output into the second here — keeps the split simple.
            let mid = sources.len() / 2;
            // Box the recursive futures — an async fn can't hold an unboxed future of itself.
            let mut left = Box::pin(translate_batch(client, llm, &sources[..mid], target_lang, context)).await?;
            let right = Box::pin(translate_batch(client, llm, &sources[mid..], target_lang, context)).await?;
            left.extend(right);
            Ok(left)
        }
        // A single cue that still won't come back cleanly: keep the source text rather than fail the
        // whole film (one untranslated line beats no subtitles).
        Ok(_) => Ok(sources.to_vec()),
        Err(e) => Err(e),
    }
}

/// Chat-model path (OpenAI / xAI / OpenRouter / Anthropic / Google). Sends a JSON array, parses a
/// JSON array back.
async fn llm_translate(
    client: &reqwest::Client,
    llm: &LlmConfig,
    sources: &[String],
    target_lang: &str,
    context: &[(String, String)],
) -> Result<Vec<String>, String> {
    let system = format!(
        "You are a professional subtitle translator. Translate each string in the user's JSON array \
         into {target_lang}. Return ONLY a JSON array of strings, the SAME length and order as the \
         input, one translation per input string. Preserve line breaks (\\n) inside a string. Keep \
         lines concise and idiomatic for on-screen reading. Never merge, split, reorder, add, or \
         drop entries. Output nothing but the JSON array."
    );
    let mut user = String::new();
    if !context.is_empty() {
        user.push_str(
            "Context — already translated earlier in this film, for consistency of names and tone. \
             Do NOT re-translate these; they are reference only:\n",
        );
        for (src, dst) in context {
            user.push_str(&format!("- {src:?} => {dst:?}\n"));
        }
        user.push('\n');
    }
    user.push_str("Translate this JSON array:\n");
    user.push_str(&serde_json::to_string(sources).map_err(|e| e.to_string())?);

    let text = call_chat(client, llm, &system, &user).await?;
    parse_json_array(&text).ok_or_else(|| "model did not return a JSON array".to_string())
}

/// Dispatch a single (system, user) chat turn to the configured provider and return the assistant
/// text. Bodies are built as `Value` so the three request shapes stay readable side by side.
async fn call_chat(
    client: &reqwest::Client,
    llm: &LlmConfig,
    system: &str,
    user: &str,
) -> Result<String, String> {
    let (url, body, auth) = match llm.provider {
        // OpenAI-compatible chat/completions: OpenAI, xAI, OpenRouter.
        Provider::OpenAI | Provider::Xai | Provider::OpenRouter => {
            let base = match llm.provider {
                Provider::OpenAI => "https://api.openai.com/v1",
                Provider::Xai => "https://api.x.ai/v1",
                _ => "https://openrouter.ai/api/v1",
            };
            (
                format!("{base}/chat/completions"),
                json!({
                    "model": llm.model,
                    "temperature": 0.2,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                }),
                Auth::Bearer,
            )
        }
        Provider::Anthropic => (
            "https://api.anthropic.com/v1/messages".to_string(),
            json!({
                "model": llm.model,
                "max_tokens": 8192,
                "temperature": 0.2,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }),
            Auth::AnthropicKey,
        ),
        Provider::Google => (
            format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
                llm.model
            ),
            json!({
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0.2},
            }),
            Auth::GoogleKey,
        ),
        Provider::DeepL => return Err("DeepL does not use the chat path".to_string()),
    };

    // Override the client's default timeout upward: an LLM completion is legitimately slower than an
    // OpenSubtitles call, but still must be bounded so a stuck provider can't hang the request.
    let mut req = client.post(&url).timeout(LLM_TIMEOUT).json(&body);
    req = match auth {
        Auth::Bearer => req.bearer_auth(&llm.api_key),
        Auth::AnthropicKey => req
            .header("x-api-key", &llm.api_key)
            .header("anthropic-version", "2023-06-01"),
        Auth::GoogleKey => req.header("x-goog-api-key", &llm.api_key),
    };

    let resp = req.send().await.map_err(|e| format!("request failed: {e}"))?;
    if !resp.status().is_success() {
        let code = resp.status();
        let body = crate::fetch::capped_text(resp, 64 * 1024).await.unwrap_or_default();
        return Err(format!("provider {code}: {}", truncate(&body, 200)));
    }
    let v: Value = crate::fetch::capped_json(resp, crate::fetch::MAX_BODY).await?;
    extract_text(llm.provider, &v).ok_or_else(|| "no text in provider response".to_string())
}

enum Auth {
    Bearer,
    AnthropicKey,
    GoogleKey,
}

/// Pull the assistant text out of each provider's response envelope.
fn extract_text(provider: Provider, v: &Value) -> Option<String> {
    match provider {
        Provider::OpenAI | Provider::Xai | Provider::OpenRouter => {
            v["choices"][0]["message"]["content"].as_str().map(str::to_string)
        }
        Provider::Anthropic => v["content"][0]["text"].as_str().map(str::to_string),
        Provider::Google => v["candidates"][0]["content"]["parts"][0]["text"].as_str().map(str::to_string),
        Provider::DeepL => None,
    }
}

/// DeepL `/v2/translate`: an array of texts in → an array of translations out, 1:1 by construction.
async fn deepl_translate(
    client: &reqwest::Client,
    llm: &LlmConfig,
    sources: &[String],
    target_lang: &str,
) -> Result<Vec<String>, String> {
    let body = json!({ "text": sources, "target_lang": deepl_code(target_lang) });
    let resp = client
        .post("https://api-free.deepl.com/v2/translate")
        .timeout(LLM_TIMEOUT)
        .header("Authorization", format!("DeepL-Auth-Key {}", llm.api_key))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("deepl {}", resp.status()));
    }
    let v: Value = crate::fetch::capped_json(resp, crate::fetch::MAX_BODY).await?;
    let arr = v["translations"].as_array().ok_or("no translations")?;
    Ok(arr.iter().filter_map(|t| t["text"].as_str().map(str::to_string)).collect())
}

/// DeepL wants an upper-case language code. Map the display names Den sends; fall back to the first
/// two letters upper-cased (covers the common `sv`/`no`/`da`/`en` → `SV`/`NB`/`DA`/`EN` cases).
fn deepl_code(lang: &str) -> String {
    match lang.to_ascii_lowercase().as_str() {
        "english" | "en" => "EN-US".to_string(),
        "swedish" | "sv" => "SV".to_string(),
        "norwegian" | "no" | "nb" => "NB".to_string(),
        "danish" | "da" => "DA".to_string(),
        "finnish" | "fi" => "FI".to_string(),
        "german" | "de" => "DE".to_string(),
        other => other.chars().take(2).collect::<String>().to_ascii_uppercase(),
    }
}

/// Extract a JSON string array from model output, tolerating markdown code fences and leading prose
/// by scanning for the first `[` … matching `]`.
fn parse_json_array(text: &str) -> Option<Vec<String>> {
    let start = text.find('[')?;
    let end = text.rfind(']')?;
    if end <= start {
        return None;
    }
    let arr: Vec<Value> = serde_json::from_str(&text[start..=end]).ok()?;
    Some(arr.into_iter().map(|v| v.as_str().unwrap_or_default().to_string()).collect())
}

fn truncate(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_fenced_json_array() {
        let out = parse_json_array("```json\n[\"Hello\", \"How are you?\"]\n```").unwrap();
        assert_eq!(out, vec!["Hello", "How are you?"]);
    }

    #[test]
    fn deepl_code_maps_names_and_falls_back() {
        assert_eq!(deepl_code("English"), "EN-US");
        assert_eq!(deepl_code("sv"), "SV");
        assert_eq!(deepl_code("pt"), "PT");
    }
}
