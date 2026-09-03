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

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
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
    run_translation(&Upstream { client, llm, target_lang }, cues).await
}

/// The harness proper, over any upstream. Split from `translate` so the same-length contract and the
/// unusable-output gate are testable without a provider.
async fn run_translation(upstream: &(dyn BatchCall + Sync), cues: &[Cue]) -> Result<Vec<Cue>, String> {
    let mut out: Vec<Cue> = Vec::with_capacity(cues.len());
    // Rolling context: the tail of already-translated pairs, refreshed as we go.
    let mut context: Vec<(String, String)> = Vec::new();

    // A wrong-length reply degrades to keeping the source text, which is right for one stray cue and
    // wrong for a film: a consistently misbehaving model hits that leaf for every cue and returns the
    // untranslated original, which then caches for 60 days as a successful translation.
    let budget = Budget {
        untranslated: AtomicUsize::new(0),
        calls: AtomicUsize::new(0),
        max_calls: call_budget(cues.len()),
    };

    for batch in cues.chunks(BATCH) {
        let sources: Vec<String> = batch.iter().map(|c| c.text.clone()).collect();
        let translated = translate_batch(upstream, &sources, &context, &budget).await?;
        for (cue, text) in batch.iter().zip(translated) {
            context.push((cue.text.clone(), text.clone()));
            out.push(Cue { text, ..cue.clone() });
        }
        if context.len() > CONTEXT_WINDOW {
            context.drain(..context.len() - CONTEXT_WINDOW);
        }
        // Bail early on a film that is clearly not being translated: a wrong-length model costs
        // 2n-1 calls a batch, so running to the end means thousands of paid calls to learn what the
        // opening showed. Only after a real sample, though — the first batch is title cards and
        // song lyrics, the harshest forty cues in the film.
        let kept = budget.untranslated.load(Ordering::Relaxed);
        if out.len() >= MIN_GATE_SAMPLE && hopeless(kept, out.len()) {
            return Err(format!("model returned unusable output for {kept} of {} cues", out.len()));
        }
    }
    // The verdict for the film as a whole, which is also the only gate a short track ever meets.
    let kept = budget.untranslated.load(Ordering::Relaxed);
    if unusable(kept, cues.len()) {
        return Err(format!("model returned unusable output for {kept} of {} cues", cues.len()));
    }
    debug_assert_eq!(out.len(), cues.len());
    Ok(out)
}

/// Cues that must be seen before the ratio is allowed to abort a run mid-film.
const MIN_GATE_SAMPLE: usize = 120;
/// Fallbacks below this never condemn a run, however small it is. A signs-only track is mostly
/// proper nouns and place names, and those legitimately come back unchanged.
const MIN_UNUSABLE: usize = 8;

/// Has too much come back unusable to call this a translation? A model wrong more than a quarter of
/// the time is not translating, and its output must not be cached as though it were.
fn unusable(kept: usize, seen: usize) -> bool {
    kept >= MIN_UNUSABLE && kept * 4 > seen
}

/// The mid-run bail is deliberately harsher than the verdict: it exists to stop paying for a run
/// that is clearly lost, and a film is only judged in full at the end.
fn hopeless(kept: usize, seen: usize) -> bool {
    kept >= MIN_UNUSABLE && kept * 2 > seen
}

/// Upstream calls a run may make. The happy path is one per batch; a wrong-length reply splits into
/// 2n-1, and the ratio gate only catches that when the fallbacks are frequent enough — at exactly a
/// quarter it fired never and cost 8,850 calls on one request. This bounds the bill regardless.
fn call_budget(cues: usize) -> usize {
    20 * cues.div_ceil(BATCH) + 40
}

/// What a run is allowed to spend, and what it has spent.
struct Budget {
    untranslated: AtomicUsize,
    calls: AtomicUsize,
    max_calls: usize,
}

/// One upstream call: a batch of source lines in, the same number of translated lines out (or an
/// error). Taken as a parameter so the contract logic below is testable without a provider.
trait BatchCall: Sync {
    fn call(
        &self,
        sources: &[String],
        context: &[(String, String)],
    ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, String>> + Send + '_>>;
}

/// The live call, dispatched by provider.
struct Upstream<'a> {
    client: &'a reqwest::Client,
    llm: &'a LlmConfig,
    target_lang: &'a str,
}

impl BatchCall for Upstream<'_> {
    fn call(
        &self,
        sources: &[String],
        context: &[(String, String)],
    ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, String>> + Send + '_>> {
        let sources = sources.to_vec();
        let context = context.to_vec();
        Box::pin(async move {
            match self.llm.provider {
                // DeepL is a 1:1 text-array MT endpoint, not a chat model — no prompt, no JSON contract.
                Provider::DeepL => deepl_translate(self.client, self.llm, &sources, self.target_lang).await,
                _ => llm_translate(self.client, self.llm, &sources, self.target_lang, &context).await,
            }
        })
    }
}

/// Translate one batch under the same-length contract. On a length mismatch, split and retry so a
/// single misbehaving batch degrades to smaller batches instead of corrupting the whole film.
async fn translate_batch(
    upstream: &(dyn BatchCall + Sync),
    sources: &[String],
    context: &[(String, String)],
    budget: &Budget,
) -> Result<Vec<String>, String> {
    if sources.is_empty() {
        return Ok(Vec::new());
    }
    if budget.calls.fetch_add(1, Ordering::Relaxed) >= budget.max_calls {
        return Err("translation exceeded its upstream call budget".to_string());
    }
    let result = upstream.call(sources, context).await;

    match result {
        Ok(v) if v.len() == sources.len() => {
            // The right length is not the same as a translation. An echo of the source, or blanks,
            // satisfies the count — and a cheap model does exactly that when the target language is
            // a typo. Counted rather than rejected: a batch that legitimately matches (names,
            // numbers, "OK") costs nothing, while a film-wide echo trips the gate.
            let dead = v
                .iter()
                .zip(sources.iter())
                .filter(|(t, s)| !s.trim().is_empty() && (t.trim().is_empty() || t.trim() == s.trim()))
                .count();
            budget.untranslated.fetch_add(dead, Ordering::Relaxed);
            Ok(v)
        }
        // Only a CONTRACT violation splits. An upstream error is not one — splitting on it spent six
        // more calls against a provider that had just said no, with no backoff, before failing anyway.
        Ok(_) if sources.len() > 1 => {
            // Split and retry each half. Context is best-effort continuity, not correctness, so we
            // don't thread the first half's output into the second here — keeps the split simple.
            let mid = sources.len() / 2;
            // Box the recursive futures — an async fn can't hold an unboxed future of itself.
            let mut left = Box::pin(translate_batch(upstream, &sources[..mid], context, budget)).await?;
            let right = Box::pin(translate_batch(upstream, &sources[mid..], context, budget)).await?;
            left.extend(right);
            Ok(left)
        }
        // A single cue that still won't come back cleanly: keep the source text rather than fail the
        // whole film (one untranslated line beats no subtitles). Counted, because a model that is
        // consistently wrong drives EVERY cue to this leaf and returns the source film verbatim.
        Ok(_) => {
            budget.untranslated.fetch_add(sources.len(), Ordering::Relaxed);
            Ok(sources.to_vec())
        }
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

    let resp = req.send().await.map_err(|e| format!("request failed: {}", e.without_url()))?;
    if !resp.status().is_success() {
        let code = resp.status();
        // The status only. This string is logged, and the body is the PROVIDER's text about a
        // request that carried the user's key — OpenAI's 401 quotes a masked form of it back, and a
        // self-hosted gateway is under no obligation to mask anything.
        return Err(format!("provider {code}"));
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
    // Every element must be a string. `as_str().unwrap_or_default()` blanked non-strings while keeping
    // the count, so `[{"text":"Hej"},…]` — a plausible reply shape — passed the length check and cached
    // a full track of empty dialogue.
    arr.into_iter().map(|v| v.as_str().map(str::to_string)).collect()
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

#[cfg(test)]
mod contract_tests {
    use super::*;
    use std::sync::Mutex;

    fn cues(n: usize) -> Vec<Cue> {
        (0..n)
            .map(|i| Cue { index: i as u32 + 1, start: i as u64 * 1000, end: i as u64 * 1000 + 900, text: format!("line {i}") })
            .collect()
    }

    /// A canned upstream. `reply` builds a response from the batch it was given, so a fake can be
    /// wrong in exactly the way a real model is wrong.
    struct Fake<F: Fn(&[String]) -> Result<Vec<String>, String> + Send + Sync> {
        reply: F,
        calls: Mutex<usize>,
    }

    impl<F: Fn(&[String]) -> Result<Vec<String>, String> + Send + Sync> BatchCall for Fake<F> {
        fn call(
            &self,
            sources: &[String],
            _context: &[(String, String)],
        ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, String>> + Send + '_>> {
            *self.calls.lock().unwrap() += 1;
            let r = (self.reply)(sources);
            Box::pin(async move { r })
        }
    }

    fn fake<F: Fn(&[String]) -> Result<Vec<String>, String> + Send + Sync>(reply: F) -> Fake<F> {
        Fake { reply, calls: Mutex::new(0) }
    }

    async fn run(upstream: &(dyn BatchCall + Sync), n: usize) -> Result<Vec<String>, String> {
        let src: Vec<String> = cues(n).iter().map(|c| c.text.clone()).collect();
        let budget = Budget {
            untranslated: AtomicUsize::new(0),
            calls: AtomicUsize::new(0),
            max_calls: call_budget(n),
        };
        translate_batch(upstream, &src, &[], &budget).await
    }

    /// A reply of the right LENGTH but the wrong SHAPE must not be accepted. `as_str()` on a non-string
    /// yielded "" while keeping the count, so this passed the length check and produced a subtitle
    /// track that loads, is selectable, and shows nothing.
    #[test]
    fn non_string_elements_are_not_a_valid_reply() {
        for body in [
            r#"[{"text":"a"},{"text":"b"}]"#,
            r#"["a", null]"#,
            r#"[["a"],["b"]]"#,
            r#"[1, 2]"#,
        ] {
            assert!(parse_json_array(body).is_none(), "accepted a non-string array: {body}");
        }
        // The valid shape still parses, fences and prose included.
        assert_eq!(parse_json_array("```json\n[\"a\",\"b\"]\n```").unwrap(), vec!["a", "b"]);
    }

    /// One stray cue keeps its source text rather than failing the film — the documented policy.
    #[tokio::test]
    async fn a_single_bad_cue_keeps_its_source_text() {
        // Any batch holding the bad cue comes back the wrong length, so the split walks down to that
        // cue alone; every batch without it answers correctly.
        let up = fake(|s: &[String]| {
            if s.iter().any(|t| t == "line 3") {
                Ok(vec!["junk".to_string(); s.len() + 1])
            } else {
                Ok(s.iter().map(|t| format!("T:{t}")).collect())
            }
        });
        let out = run(&up, 8).await.unwrap();
        assert_eq!(out.len(), 8);
        assert_eq!(out[3], "line 3", "the unusable cue should fall back to its source");
        assert_eq!(out[0], "T:line 0", "its neighbours must still be translated");
    }

    /// A model that is consistently wrong-length drives EVERY cue to that same leaf, and the result is
    /// the untranslated film returned as a success. `translate` must refuse it rather than cache it.
    /// A model wrong at every size drives the split to every leaf, so every cue keeps its source —
    /// the untranslated film. It must fail, and it must stop paying to find that out. Which of the
    /// two guards trips is not the point; that neither lets it through, and that the bill is
    /// bounded, is.
    #[tokio::test]
    async fn a_wholly_unusable_model_is_an_error_not_an_untranslated_film() {
        let up = fake(|s: &[String]| Ok(vec!["junk".to_string(); s.len() + 1]));
        assert!(run_translation(&up, &cues(60)).await.is_err(), "an untranslated film was returned");
        assert!(
            *up.calls.lock().unwrap() <= call_budget(60),
            "spent {} calls against a budget of {}",
            up.calls.lock().unwrap(),
            call_budget(60)
        );
    }

    /// The budget is the guard for the case the ratio cannot see: a model wrong on exactly a
    /// quarter of cues never trips `unusable` (`kept * 4 > seen` is false at exactly a quarter) and
    /// used to run to the end at ~60 calls a batch — 8,850 on one request the viewer is waiting on.
    #[tokio::test]
    async fn a_run_cannot_outspend_its_call_budget() {
        let up = fake(|s: &[String]| {
            // Wrong-length whenever the batch holds a cue whose number is divisible by 4.
            if s.iter().any(|t| t.trim_start_matches("line ").parse::<usize>().is_ok_and(|n| n % 4 == 0)) {
                Ok(vec!["junk".to_string(); s.len() + 1])
            } else {
                Ok(s.iter().map(|t| format!("T:{t}")).collect())
            }
        });
        let budget = call_budget(2000);
        let _ = run_translation(&up, &cues(2000)).await;
        assert!(
            *up.calls.lock().unwrap() <= budget,
            "spent {} calls against a budget of {budget}",
            up.calls.lock().unwrap()
        );
    }

    /// A short track must not be failed by one odd line. `kept * 4 > seen` alone fails a 3-cue
    /// forced-narrative track on a single fallback, which is precisely the case the fallback exists
    /// to serve.
    #[tokio::test]
    async fn one_bad_cue_never_fails_a_short_track() {
        let up = fake(|s: &[String]| {
            if s.len() == 1 && s[0] == "line 1" {
                Ok(vec!["junk".into(), "junk".into()])
            } else if s.iter().any(|t| t == "line 1") {
                Ok(vec!["junk".to_string(); s.len() + 1])
            } else {
                Ok(s.iter().map(|t| format!("T:{t}")).collect())
            }
        });
        let out = run_translation(&up, &cues(3)).await.expect("one odd line must not fail a 3-cue track");
        assert_eq!(out[1].text, "line 1", "the odd cue keeps its source");
        assert_eq!(out[0].text, "T:line 0");
    }

    /// The gate must not fire on a film that mostly translated — a handful of odd lines is the case
    /// the source-text fallback exists for.
    #[tokio::test]
    async fn a_few_bad_cues_do_not_fail_the_film() {
        let up = fake(|s: &[String]| {
            if s.len() == 1 && s[0].ends_with('7') {
                Ok(vec!["junk".into(), "junk".into()])
            } else if s.iter().any(|t| t.ends_with('7')) {
                Ok(vec!["junk".to_string(); s.len() + 1])
            } else {
                Ok(s.iter().map(|t| format!("T:{t}")).collect())
            }
        });
        let out = run_translation(&up, &cues(60)).await.expect("a mostly-translated film must succeed");
        assert_eq!(out.len(), 60);
        assert_eq!(out[0].text, "T:line 0");
        // Timing and index are never handed to the model, so they must survive untouched.
        assert_eq!((out[5].index, out[5].start, out[5].end), (6, 5000, 5900));
    }

    /// A reply of the right length that translated nothing. Both shapes satisfy the count and both
    /// produce a track that loads, is selectable, and shows nothing useful — the echo shows the
    /// original language, the blanks show nothing at all.
    #[tokio::test]
    async fn a_right_length_reply_that_translated_nothing_is_refused() {
        // An exact echo, and the four ways a model dresses one up. Comparing byte-exact caught only
        // the first: a trailing newline is the commonest shape of LLM string output, and
        // `srt::serialize` strips it, so the cached artifact was byte-identical to the source.
        for (name, decorate) in [
            ("exact", "" as &str),
            ("trailing newline", "\n"),
            ("trailing space", " "),
            ("trailing tab", "\t"),
        ] {
            let echo = fake(move |s: &[String]| Ok(s.iter().map(|t| format!("{t}{decorate}")).collect()));
            let err = run_translation(&echo, &cues(200)).await.unwrap_err();
            assert!(err.contains("unusable"), "an echo ({name}) was accepted: {err}");
        }
        // A leading space is the same trick from the other end.
        let echo = fake(|s: &[String]| Ok(s.iter().map(|t| format!(" {t}")).collect()));
        assert!(run_translation(&echo, &cues(200)).await.is_err(), "a leading-space echo was accepted");

        let blanks = fake(|s: &[String]| Ok(vec![String::new(); s.len()]));
        let err = run_translation(&blanks, &cues(200)).await.unwrap_err();
        assert!(err.contains("unusable"), "a reply of empty strings was accepted: {err}");
    }

    /// A signs-only or forced-narrative track is mostly proper nouns and place names, which
    /// legitimately come back unchanged. The ratio alone rejected four matches in a twelve-cue
    /// track — and a rejection costs the full LLM bill and caches nothing, so every retry re-pays.
    #[tokio::test]
    async fn a_short_track_of_mostly_names_still_translates() {
        for (total, matching) in [(8usize, 3usize), (12, 4), (20, 6)] {
            let up = fake(move |s: &[String]| {
                Ok(s.iter()
                    .map(|t| {
                        let n: usize = t.trim_start_matches("line ").parse().unwrap_or(999);
                        if n < matching { t.clone() } else { format!("T:{t}") }
                    })
                    .collect())
            });
            let out = run_translation(&up, &cues(total)).await;
            assert!(out.is_ok(), "{matching} names in a {total}-cue track was rejected: {out:?}");
        }
    }

    /// A handful of cues that legitimately come back unchanged — names, numbers, "OK" — must not
    /// fail a film, and they cluster in the opening titles where the early gate samples.
    #[tokio::test]
    async fn cues_that_legitimately_match_do_not_fail_a_film() {
        let up = fake(|s: &[String]| {
            Ok(s.iter()
                .map(|t| if t.ends_with('3') || t.ends_with('7') { t.clone() } else { format!("T:{t}") })
                .collect())
        });
        let out = run_translation(&up, &cues(600)).await.expect("a fifth of cues matching is normal");
        assert_eq!(out.len(), 600);
    }

    /// The opening forty cues are the harshest sample a film ever offers — title cards, a song
    /// lyric, a run of names — so judging the whole film on them rejects good translations. Eleven
    /// odd cues in the first batch of a 600-cue film is normal; aborting there is not.
    #[tokio::test]
    async fn a_rough_opening_does_not_abort_a_good_film() {
        let up = fake(|s: &[String]| {
            Ok(s.iter()
                .map(|t| {
                    let n: usize = t.trim_start_matches("line ").parse().unwrap_or(999);
                    if n < 11 { t.clone() } else { format!("T:{t}") }
                })
                .collect())
        });
        let out = run_translation(&up, &cues(600)).await.expect("a rough opening must not abort the film");
        assert_eq!(out.len(), 600);
        assert_eq!(out[300].text, "T:line 300");
    }

    /// An upstream error must not be split-retried — that spent six more calls against a provider
    /// that had just said no, with no backoff, and failed anyway.
    #[tokio::test]
    async fn an_upstream_error_is_not_split_retried() {
        let up = fake(|_: &[String]| Err("provider 429: rate limited".to_string()));
        assert!(run_translation(&up, &cues(40)).await.is_err());
        assert_eq!(*up.calls.lock().unwrap(), 1, "a 429 was split-retried instead of surfacing");
    }

    /// A transport/auth error is not a contract violation: it must surface, not degrade to the source.
    #[tokio::test]
    async fn an_upstream_error_still_fails() {
        let up = fake(|_: &[String]| Err("provider 401: bad key".to_string()));
        assert!(run(&up, 4).await.is_err(), "an upstream failure must not become an untranslated film");
    }
}
