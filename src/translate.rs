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

/// Where a finished batch is remembered between runs, so a film that dies at cue 1100 of 1200 does
/// not re-buy the 1100 that worked. A trait rather than the cache itself, so the harness tests can
/// drive the resume path without a disk tier.
pub trait BatchStore: Sync {
    fn get(&self, key: &str) -> Option<String>;
    fn put(&self, key: String, value: String);
}

/// Batches are remembered only long enough to rescue a retry, not for the life of the finished
/// translation. Once the whole film is cached under its own key these are redundant, and a film is
/// ~150 of them — keeping a second copy of every translation for two months is a poor trade for a
/// resume window. A day covers "it failed, I tried again this evening"; the ten-minute failure
/// backoff means the retries that matter arrive far sooner than that.
const BATCH_TTL: Duration = Duration::from_secs(60 * 60 * 24);

impl BatchStore for crate::cache::Cache {
    fn get(&self, key: &str) -> Option<String> {
        crate::cache::Cache::get(self, key)
    }
    fn put(&self, key: String, value: String) {
        crate::cache::Cache::put(self, key, value, BATCH_TTL);
    }
}

/// A store plus the prefix that scopes keys to one (provider, model, language). Without the prefix a
/// DeepL batch and a GPT batch of the same lines would be the same entry.
struct Resume<'a> {
    store: &'a dyn BatchStore,
    prefix: String,
}

/// Identity of one batch's translation: the exact source lines it covers.
fn batch_key(prefix: &str, sources: &[String]) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    // Length-delimited so ["ab","c"] cannot hash the same as ["a","bc"]. `str`'s own Hash impl
    // already delimits, but a batch key naming the wrong lines would serve one scene's dialogue over
    // another's, so it is spelled out rather than relied upon.
    sources.len().hash(&mut h);
    for s in sources {
        s.len().hash(&mut h);
        s.hash(&mut h);
    }
    format!("batch:{prefix}:{:016x}", h.finish())
}

/// How many of these translations came back unusable — blank, or an echo of the source. A blank
/// source has nothing to translate, so it is never counted against the model.
///
/// One function because the live path and the resume path must score a batch identically: a film
/// that was half-translated before a crash has to reach the same verdict on retry as it would have
/// reached in one run.
fn dead_count(translated: &[String], sources: &[String]) -> usize {
    translated
        .iter()
        .zip(sources.iter())
        .filter(|(t, s)| !s.trim().is_empty() && (t.trim().is_empty() || t.trim() == s.trim()))
        .count()
}

/// Translate every cue's text into `target_lang` (a display name like "English"), preserving each
/// cue's index/timing. Returns the same cues with translated `text`, or an error string.
pub async fn translate(
    client: &reqwest::Client,
    llm: &LlmConfig,
    cues: &[Cue],
    target_lang: &str,
    store: &dyn BatchStore,
) -> Result<Vec<Cue>, String> {
    if cues.is_empty() {
        return Ok(Vec::new());
    }
    if cues.len() > MAX_CUES {
        return Err(format!("subtitle too large: {} cues (max {MAX_CUES})", cues.len()));
    }
    let resume = Resume {
        store,
        prefix: format!("{}:{}:{}", llm.provider.tag(), llm.model, canonical_lang(target_lang)),
    };
    run_translation(&Upstream { client, llm, target_lang }, cues, RUN_DEADLINE, Some(&resume)).await
}

/// The harness proper, over any upstream. Split from `translate` so the same-length contract and the
/// unusable-output gate are testable without a provider.
async fn run_translation(
    upstream: &(dyn BatchCall + Sync),
    cues: &[Cue],
    deadline: Duration,
    resume: Option<&Resume<'_>>,
) -> Result<Vec<Cue>, String> {
    let mut out: Vec<Cue> = Vec::with_capacity(cues.len());
    // Rolling context: the tail of already-translated pairs, refreshed as we go.
    let mut context: Vec<(String, String)> = Vec::new();

    // A wrong-length reply degrades to keeping the source text, which is right for one stray cue and
    // wrong for a film: a consistently misbehaving model hits that leaf for every cue and returns the
    // untranslated original, which then caches for 60 days as a successful translation.
    let budget = Budget {
        untranslated: AtomicUsize::new(0),
        started: std::time::Instant::now(),
        deadline,
    };

    for batch in cues.chunks(BATCH) {
        // Calls were counted but never timed, and batches run in sequence at up to LLM_TIMEOUT
        // each — so the permitted ceiling was measured in hours, long after the viewer gave up,
        // still spending their key.
        let sources: Vec<String> = batch.iter().map(|c| c.text.clone()).collect();
        let translated = translate_batch(upstream, &sources, &context, &budget, resume).await?;
        for (cue, text) in batch.iter().zip(translated) {
            // Only a pair that actually CHANGED goes into the context. An unchanged one is either a
            // name that legitimately stays put or a cue the model failed on and we fell back to
            // source — and from here those are indistinguishable. The context is rendered into the
            // next prompt as "already translated, do not re-translate", so handing it a failure
            // presents that failure as precedent and invites the model to repeat it on the same
            // word later in the film. An identity pair teaches it almost nothing anyway.
            if text.trim() != cue.text.trim() {
                context.push((cue.text.clone(), text.clone()));
            }
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
        // Bail early only when the verdict is already decided — that is, when the dead count alone
        // exceeds the bar for the WHOLE film, so no amount of perfect translation in the cues still
        // to come could rescue it. `kept` only grows, so this can never refuse a film the final
        // verdict would have passed.
        //
        // Judging the sample against itself is what cannot be done here: three batches of a
        // name-dense opening is not evidence about the ninety batches after it, and doing that
        // refused films whose true ratio was nowhere near the bar.
        if unusable(kept, cues.len()) {
            return Err(format!("model returned unusable output for {kept} of {} cues", cues.len()));
        }
    }
    // No verdict after the loop: the check above runs after the last batch too, against the same
    // count and the same denominator, so a second one could never reach a different answer.
    debug_assert_eq!(out.len(), cues.len());
    Ok(out)
}

/// Wall-clock ceiling for one film. Well past a healthy run (~30 sequential calls) and well short
/// of what the call budget alone permits at LLM_TIMEOUT apiece.
const RUN_DEADLINE: Duration = Duration::from_secs(600);

/// Has too much come back unusable to call this a translation?
///
/// Purely a ratio, plus "nothing translated at all" as the unambiguous case at any size. The bar is
/// high — two thirds — because the opposite error is worse: a forced-narrative track is mostly place
/// names and proper nouns that legitimately come back unchanged, and refusing one costs the full LLM
/// bill, caches nothing, and makes every retry pay again. A third unchanged is a normal signs track.
///
/// It carried an absolute floor for a while, to stop the ratio failing short tracks back when the
/// bar was a quarter. Once the bar moved to two thirds the floor was doing nothing useful and quite
/// a lot of harm: a floor of eight let seven dead cues out of eight through — 87% untranslated,
/// served and cached for sixty days. Ratios do not need a size exemption; the old threshold did.
///
/// "Nothing translated at all" needs no clause of its own either: 3n > 2n holds for every n > 0.
fn unusable(kept: usize, seen: usize) -> bool {
    // A one-cue track is 0% or 100% dead by construction, so the ratio can only ever refuse it —
    // and if that cue is a sign reading MOSCOW, coming back unchanged is the correct translation.
    // Refusing costs the viewer the track entirely and repeats on every retry; accepting costs one
    // line that may already be right.
    seen > 1 && kept * 3 > seen * 2
}




/// What a run has produced, and how long it has had.
///
/// There is no call counter. The split is a binary tree with one leaf per cue, so a batch of n
/// costs at most 2n-1 calls and a run at most 2*cues - batches — a cap the recursion enforces by
/// its own shape. A counter on top of that could only ever fire below the structural maximum, and
/// sized below it, it becomes a stricter quality gate than the quality gate. Time is the bound that
/// is actually reachable, so time is the bound that is kept.
struct Budget {
    untranslated: AtomicUsize,
    started: std::time::Instant,
    deadline: Duration,
}

impl Budget {
    fn spent(&self) -> bool {
        self.started.elapsed() > self.deadline
    }
}

/// Why a call did not produce a usable batch.
///
/// The distinction decides everything below it: a CONTRACT failure is the model misbehaving, and
/// degrades — split the batch, and at one cue keep the source. An UPSTREAM failure is the provider
/// refusing, and splitting on that just spends more calls at a service that has already said no.
/// Wrong shape and wrong length are both the model misbehaving; treating only the length that way
/// made one chatty reply fatal to a whole film.
#[derive(Debug)]
enum CallError {
    Contract(String),
    Upstream(String),
}

/// Everything that reaches here as a bare string came from the transport or a provider status —
/// the model's own misbehaviour is constructed explicitly as `Contract`.
impl From<String> for CallError {
    fn from(m: String) -> Self {
        CallError::Upstream(m)
    }
}

impl CallError {
    fn into_message(self) -> String {
        match self {
            CallError::Contract(m) | CallError::Upstream(m) => m,
        }
    }
}

/// One upstream call: a batch of source lines in, the same number of translated lines out (or an
/// error). Taken as a parameter so the contract logic below is testable without a provider.
trait BatchCall: Sync {
    fn call(
        &self,
        sources: &[String],
        context: &[(String, String)],
    ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, CallError>> + Send + '_>>;
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
    ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, CallError>> + Send + '_>> {
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
    resume: Option<&Resume<'_>>,
) -> Result<Vec<String>, String> {
    if sources.is_empty() {
        return Ok(Vec::new());
    }
    // Already bought, in some earlier run of this film that did not reach the end. Checked before
    // the deadline below on purpose: replaying what is already paid for costs nothing, and refusing
    // it would make a resumed run fail in exactly the place the previous one did.
    let key = resume.map(|r| batch_key(&r.prefix, sources));
    if let (Some(r), Some(k)) = (resume, key.as_ref()) {
        if let Some(cached) = r.store.get(k).and_then(|s| serde_json::from_str::<Vec<String>>(&s).ok()) {
            // Length is re-checked rather than trusted: the same-length contract is the one thing
            // holding cue and timing together, and a stored entry is as untrusted as a model reply.
            if cached.len() == sources.len() {
                // Scored exactly as a live reply would be, so a resumed film reaches the same
                // verdict as one translated in a single run.
                budget.untranslated.fetch_add(dead_count(&cached, sources), Ordering::Relaxed);
                return Ok(cached);
            }
        }
    }
    // Out of money or out of time. The run did not finish, and saying so is the only honest answer:
    // a cue that was never sent is not evidence about translation quality, it is the absence of
    // evidence. Counting those as "unchanged" and letting the ratio judge them meant a merely SLOW
    // provider could translate the first third of a film, have the rest handed back untouched, land
    // at 66% — just under the bar — and be served and cached for sixty days as a translation.
    //
    // The ratio's two-thirds bar is sized for cues the model DID return unchanged: names, signs,
    // numbers. Skipped cues are a different fact and cannot share that counter.
    //
    // Checked here rather than only between batches because one wrong-length batch splits into up
    // to 2n-1 calls without the outer loop regaining control.
    if budget.spent() {
        return Err("translation ran out of time".to_string());
    }
    let result = upstream.call(sources, context).await;

    match result {
        Ok(v) if v.len() == sources.len() => {
            // A blanked line is shipped as its source, the same choice the wrong-length leaf makes:
            // an untranslated line still shows the original language, a blank one shows nothing.
            // Counting it and then serving the blank anyway was the gap — below the ratio, those
            // cues rendered empty in an otherwise successful film.
            let out: Vec<String> = v
                .into_iter()
                .zip(sources.iter())
                // A blank source has nothing to translate, so anything the model invented for it is
                // not a translation — it is a line the film does not contain. Both directions of
                // blankness resolve to the source.
                .map(|(t, s)| if t.trim().is_empty() || s.trim().is_empty() { s.clone() } else { t })
                .collect();
            // The right length is not the same as a translation. An echo of the source, or blanks,
            // satisfies the count — and a cheap model does exactly that when the target language is
            // a typo. Counted rather than rejected: a batch that legitimately matches (names,
            // numbers, "OK") costs nothing, while a film-wide echo trips the gate. Scored after the
            // substitution above, which is score-neutral — a blank became its source, and both count
            // as dead — and lets the resume path score the stored value the same way.
            budget.untranslated.fetch_add(dead_count(&out, sources), Ordering::Relaxed);
            // Remember it, so a run that dies later does not buy this batch again. Only this arm:
            // the degraded leaves below kept the source text because the model would not cooperate,
            // and storing that would turn one bad minute into a day of never retrying it.
            if let (Some(r), Some(k)) = (resume, key) {
                if let Ok(json) = serde_json::to_string(&out) {
                    r.store.put(k, json);
                }
            }
            Ok(out)
        }
        // A contract violation splits; an upstream refusal does not — splitting on that spent six
        // more calls against a provider that had just said no, with no backoff, before failing anyway.
        Ok(_) | Err(CallError::Contract(_)) if sources.len() > 1 => {
            // Split and retry each half. Context is best-effort continuity, not correctness, so we
            // don't thread the first half's output into the second here — keeps the split simple.
            let mid = sources.len() / 2;
            // Box the recursive futures — an async fn can't hold an unboxed future of itself.
            let mut left = Box::pin(translate_batch(upstream, &sources[..mid], context, budget, resume)).await?;
            let right = Box::pin(translate_batch(upstream, &sources[mid..], context, budget, resume)).await?;
            left.extend(right);
            Ok(left)
        }
        // A single cue that still won't come back cleanly: keep the source text rather than fail the
        // whole film (one untranslated line beats no subtitles). Counted, because a model that is
        // consistently wrong drives EVERY cue to this leaf and returns the source film verbatim.
        Ok(_) | Err(CallError::Contract(_)) => {
            budget.untranslated.fetch_add(sources.len(), Ordering::Relaxed);
            Ok(sources.to_vec())
        }
        Err(e) => Err(e.into_message()),
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
) -> Result<Vec<String>, CallError> {
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
    user.push_str(&serde_json::to_string(sources).map_err(|e| CallError::Upstream(e.to_string()))?);

    let text = call_chat_typed(client, llm, &system, &user).await?;
    parse_json_array(&text)
        .ok_or_else(|| CallError::Contract("model did not return a JSON array".to_string()))
}

/// Dispatch a single (system, user) chat turn to the configured provider and return the assistant
/// text. Bodies are built as `Value` so the three request shapes stay readable side by side.
async fn call_chat_typed(
    client: &reqwest::Client,
    llm: &LlmConfig,
    system: &str,
    user: &str,
) -> Result<String, CallError> {
    let (url, body, auth) = match llm.provider {
        // OpenAI-compatible chat/completions: OpenAI, xAI, OpenRouter.
        Provider::OpenAI | Provider::Xai | Provider::OpenRouter => {
            let base = chat_base(llm.provider);
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
        Provider::DeepL => return Err(CallError::Upstream("DeepL does not use the chat path".into())),
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
        return Err(CallError::Upstream(format!("provider {code}")));
    }
    let v = provider_json(resp).await?;
    // Contract, not Upstream: a 200 with no usable text is a safety filter or an empty candidate
    // list, and both are about THIS batch's dialogue — film dialogue trips content filters routinely.
    // As an Upstream it propagated, so one filtered batch killed the whole film instead of falling
    // back to source text for those cues.
    extract_text(llm.provider, &v)
        .ok_or_else(|| CallError::Contract("no text in provider response".to_string()))
}

enum Auth {
    Bearer,
    AnthropicKey,
    GoogleKey,
}

/// Pull the assistant text out of each provider's response envelope.
/// OpenAI-compatible chat root. A function rather than a literal so a test can drive the real
/// request/response path — the classification of an empty reply lives at that call site, and
/// mis-classifying it there is what killed whole films twice.
/// DeepL's root, a function for the same reason `chat_base` is: the reply handling below had no
/// test at all, and that is where these bugs live.
#[cfg(not(test))]
fn deepl_base() -> String {
    "https://api-free.deepl.com".to_string()
}

#[cfg(test)]
fn deepl_base() -> String {
    DEEPL_BASE.with(|b| b.borrow().clone()).unwrap_or_else(|| "http://127.0.0.1:1".to_string())
}

#[cfg(test)]
thread_local! {
    static DEEPL_BASE: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

#[cfg(not(test))]
fn chat_base(provider: Provider) -> String {
    match provider {
        Provider::OpenAI => "https://api.openai.com/v1".to_string(),
        Provider::Xai => "https://api.x.ai/v1".to_string(),
        _ => "https://openrouter.ai/api/v1".to_string(),
    }
}

#[cfg(test)]
thread_local! {
    static CHAT_BASE: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn chat_base(_provider: Provider) -> String {
    CHAT_BASE.with(|b| b.borrow().clone()).unwrap_or_else(|| "http://127.0.0.1:1".to_string())
}

/// A 200's body, decoded — with the failure split the way the caller needs it.
///
/// The body arriving intact and being unusable is the same kind of event as a reply with no text in
/// it: the provider answered, about THIS batch, and answered badly. So an unparseable body and one
/// that runs past the size cap are Contract — splitting asks for less at a time, which is the one
/// thing that can help. Only the stream itself failing is Upstream.
///
/// This was the whole of the previous fix's gap: the classification was applied one statement too
/// late, and `?` on the decode above it still resolved to Upstream through `From<String>`.
async fn provider_json(resp: reqwest::Response) -> Result<Value, CallError> {
    let bytes = crate::fetch::capped_bytes(resp, crate::fetch::MAX_BODY)
        .await
        .map_err(|e| match e.starts_with("read body:") {
            true => CallError::Upstream(e),
            false => CallError::Contract(e),
        })?;
    serde_json::from_slice(&bytes)
        .map_err(|e| CallError::Contract(format!("provider returned unparseable JSON: {e}")))
}

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
) -> Result<Vec<String>, CallError> {
    let Some(code) = deepl_code(target_lang) else {
        // Upstream, not Contract: splitting the batch cannot make DeepL learn the language.
        return Err(CallError::Upstream(format!("deepl has no code for {target_lang}")));
    };
    let body = json!({ "text": sources, "target_lang": code });
    let resp = client
        .post(format!("{}/v2/translate", deepl_base()))
        .timeout(LLM_TIMEOUT)
        .header("Authorization", format!("DeepL-Auth-Key {}", llm.api_key))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(CallError::Upstream(format!("deepl {}", resp.status())));
    }
    let v = provider_json(resp).await?;
    let arr = v["translations"]
        .as_array()
        .ok_or_else(|| CallError::Contract("deepl returned no translations".to_string()))?;
    // Invalidate, don't drop. `filter_map` silently shrank the reply, which the same-length check
    // upstream happens to catch — but only by coincidence: drops offset by spurious extra entries
    // would misalign every cue at the right length and pass. The chat path already collects into
    // an Option for exactly this reason.
    arr.iter()
        .map(|t| t["text"].as_str().map(str::to_string))
        .collect::<Option<Vec<String>>>()
        .ok_or_else(|| CallError::Contract("deepl returned a non-string translation".to_string()))
}

/// DeepL wants an upper-case language code. Map the display names Den sends; fall back to the first
/// two letters upper-cased (covers the common `sv`/`no`/`da`/`en` → `SV`/`NB`/`DA`/`EN` cases).
fn deepl_code(lang: &str) -> Option<&'static str> {
    // Named in full, because guessing is worse than refusing here. The old fallback took the first
    // two letters and uppercased them, which is not a language code — it is a coincidence. Estonian
    // became ES, which is DeepL's code for SPANISH, so a request for Estonian came back as fluent
    // Spanish: right shape, right length, not an echo, so nothing downstream could tell. Cached
    // sixty days and served immutable. Slovak became SL, which is Slovenian. Portuguese and Polish
    // both became PO, which is nothing at all.
    Some(match lang.to_ascii_lowercase().as_str() {
        "english" | "en" => "EN-US",
        "swedish" | "sv" => "SV",
        "norwegian" | "no" | "nb" => "NB",
        "danish" | "da" => "DA",
        "finnish" | "fi" => "FI",
        "german" | "de" => "DE",
        "spanish" | "es" => "ES",
        "portuguese" | "pt" => "PT-PT",
        "french" | "fr" => "FR",
        "italian" | "it" => "IT",
        "dutch" | "nl" => "NL",
        "polish" | "pl" => "PL",
        "russian" | "ru" => "RU",
        "turkish" | "tr" => "TR",
        "czech" | "cs" => "CS",
        "greek" | "el" => "EL",
        "japanese" | "ja" => "JA",
        "korean" | "ko" => "KO",
        "chinese" | "zh" => "ZH",
        "ukrainian" | "uk" => "UK",
        "indonesian" | "id" => "ID",
        "estonian" | "et" => "ET",
        "slovak" | "sk" => "SK",
        "slovenian" | "sl" => "SL",
        "romanian" | "ro" => "RO",
        "hungarian" | "hu" => "HU",
        "bulgarian" | "bg" => "BG",
        "latvian" | "lv" => "LV",
        "lithuanian" | "lt" => "LT",
        "arabic" | "ar" => "AR",
        _ => return None,
    })
}

/// The cache-key form of a target language.
///
/// Case, surrounding space and internal spacing are not different languages, but they were different
/// cache keys — and a translate cache key is a whole film's LLM bill charged to the viewer's own key.
/// "Swedish", "swedish" and " Swedish " bought the same film three times, and nothing about a repeat
/// request ever hit the cache.
///
/// Where `deepl_code` knows the language its code is the canonical form, so a request naming the code
/// ("sv") and one naming the language ("Swedish") meet as well. That table is the only name↔code
/// mapping in the tree and stays the only one — a second copy here is a place for the two to drift,
/// and the drift would be silent. Editing it moves cache keys, which costs a re-translation: fine for
/// a cache, worth knowing before you edit.
///
/// A language the table doesn't carry (Hebrew, Thai, Vietnamese — all fine for a chat model, none of
/// them DeepL languages) keeps its normalized name, which still merges the casing and spacing
/// variants. It is deliberately not a guess: `deepl_code` documents at length what guessing a code
/// from the first two letters cost, and this is the same table saying "I don't know this one".
pub fn canonical_lang(lang: &str) -> String {
    // split_whitespace collapses runs and trims both ends in one pass.
    let normalized = lang.split_whitespace().collect::<Vec<_>>().join(" ").to_ascii_lowercase();
    match deepl_code(&normalized) {
        Some(code) => code.to_string(),
        None => normalized,
    }
}

/// Extract a JSON string array from model output, tolerating markdown code fences and leading prose
/// by scanning for the first `[` … matching `]`.
fn parse_json_array(text: &str) -> Option<Vec<String>> {
    // Two linear passes, and they are not interchangeable.
    //
    // The first takes spans where the bracket depth returns to zero — the outermost array wins, so
    // a nested one can never answer with a fragment of itself.
    //
    // The second exists because an UNMATCHED `[` earlier in the text (a model echoing a cue like
    // "[MUSIC PLAYING" with no closing bracket) means depth never returns to zero, and the whole
    // reply was discarded even though a perfectly good array sat right after it. It pairs brackets
    // on a stack instead, so an unbalanced prefix costs nothing.
    //
    // Both walk each byte once. Retrying from every `[` — the obvious way to write this — rescans
    // the same interior per candidate, which was quadratic and could hold the one runtime thread.
    top_level_array(text).or_else(|| innermost_array(text)).flatten()
}

/// How many candidate spans either pass will try to decode. A real reply has one; prose with
/// brackets in it has a handful. Deeply nested junk has thousands, and each costs a parse down to
/// serde's recursion limit — ~3s for 800 KB, tens of seconds at the 12 MiB body cap, on the single
/// thread that serves every request.
const MAX_CANDIDATES: usize = 64;

/// Spans that open at depth 0 and close back to it, tried left to right.
fn top_level_array(text: &str) -> Option<Option<Vec<String>>> {
    let mut scan = Scan::new();
    let mut start = None;
    let mut tried = 0usize;
    for (i, &b) in text.as_bytes().iter().enumerate() {
        match (scan.step(b), start) {
            (Step::Open(1), _) => start = Some(i),
            (Step::Close(0), Some(s)) => {
                start = None;
                if let Some(reply) = reply_at(&text[s..=i]) {
                    return Some(reply);
                }
                tried += 1;
                if tried >= MAX_CANDIDATES {
                    return None;
                }
            }
            _ => {}
        }
    }
    None
}

/// Every balanced pair, innermost-first. For text whose depth never returns to zero — an unmatched
/// `[` earlier in the reply, which the pass above cannot see past.
fn innermost_array(text: &str) -> Option<Option<Vec<String>>> {
    let mut scan = Scan::new();
    let mut opens: Vec<usize> = Vec::new();
    let mut tried = 0usize;
    for (i, &b) in text.as_bytes().iter().enumerate() {
        match scan.step(b) {
            Step::Open(_) => opens.push(i),
            Step::Close(_) => {
                if let Some(s) = opens.pop() {
                    if let Some(reply) = reply_at(&text[s..=i]) {
                        return Some(reply);
                    }
                    tried += 1;
                    if tried >= MAX_CANDIDATES {
                        return None;
                    }
                }
            }
            _ => {}
        }
    }
    None
}

/// Is this span the reply?
///
/// `None` means it is not a JSON array at all, so the caller keeps looking. `Some` means it IS the
/// reply — and the inner `Option` says whether it is a valid one, because a well-formed array of
/// non-strings is a bad reply rather than a reason to carry on and answer with something nested
/// inside it.
fn reply_at(slice: &str) -> Option<Option<Vec<String>>> {
    let arr: Vec<Value> = serde_json::from_str(slice).ok()?;
    Some(arr.into_iter().map(|v| v.as_str().map(str::to_string)).collect())
}

enum Step {
    Open(usize),
    Close(usize),
    Other,
}

/// Bracket depth that ignores anything inside a JSON string, escapes included.
struct Scan {
    depth: usize,
    in_string: bool,
    escaped: bool,
}

impl Scan {
    fn new() -> Scan {
        Scan { depth: 0, in_string: false, escaped: false }
    }

    fn step(&mut self, b: u8) -> Step {
        if self.in_string {
            match b {
                _ if self.escaped => self.escaped = false,
                b'\\' => self.escaped = true,
                b'"' => self.in_string = false,
                _ => {}
            }
            return Step::Other;
        }
        match b {
            b'"' => {
                self.in_string = true;
                Step::Other
            }
            b'[' => {
                self.depth += 1;
                Step::Open(self.depth)
            }
            b']' if self.depth > 0 => {
                self.depth -= 1;
                Step::Close(self.depth)
            }
            _ => Step::Other,
        }
    }
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
        assert_eq!(deepl_code("English"), Some("EN-US"));
        assert_eq!(deepl_code("sv"), Some("SV"));
        assert_eq!(deepl_code("pt"), Some("PT-PT"));

        // The old fallback took the first two letters, which produced a DIFFERENT REAL LANGUAGE for
        // some inputs — fluent, right-length, undetectable downstream, cached for sixty days.
        assert_eq!(deepl_code("Estonian"), Some("ET"), "Estonian truncated to ES, which is Spanish");
        assert_eq!(deepl_code("Slovak"), Some("SK"), "Slovak truncated to SL, which is Slovenian");
        // And these truncated to codes DeepL does not have, so they simply always failed.
        for lang in ["Spanish", "Portuguese", "Chinese", "Turkish", "Dutch", "Polish", "Indonesian"] {
            assert!(deepl_code(lang).is_some(), "{lang} has no DeepL code");
        }
        // An unknown language is refused rather than guessed at.
        assert_eq!(deepl_code("Klingon"), None);
        assert_eq!(deepl_code("xx"), None);
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
    struct Fake<F: Fn(&[String]) -> Result<Vec<String>, CallError> + Send + Sync> {
        reply: F,
        calls: Mutex<usize>,
    }

    impl<F: Fn(&[String]) -> Result<Vec<String>, CallError> + Send + Sync> BatchCall for Fake<F> {
        fn call(
            &self,
            sources: &[String],
            _context: &[(String, String)],
        ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, CallError>> + Send + '_>> {
            *self.calls.lock().unwrap() += 1;
            let r = (self.reply)(sources);
            Box::pin(async move { r })
        }
    }

    fn fake<F: Fn(&[String]) -> Result<Vec<String>, CallError> + Send + Sync>(reply: F) -> Fake<F> {
        Fake { reply, calls: Mutex::new(0) }
    }

    /// A batch store in memory, so the resume path is testable without a disk tier.
    #[derive(Default)]
    struct MemStore(Mutex<std::collections::HashMap<String, String>>);

    impl BatchStore for MemStore {
        fn get(&self, key: &str) -> Option<String> {
            self.0.lock().unwrap().get(key).cloned()
        }
        fn put(&self, key: String, value: String) {
            self.0.lock().unwrap().insert(key, value);
        }
    }

    impl MemStore {
        fn len(&self) -> usize {
            self.0.lock().unwrap().len()
        }
    }

    /// A film that dies partway must not be re-bought from the start. The completed batches are
    /// remembered, so the retry pays only for what actually failed — before this, a run that died on
    /// the last batch threw away every translated cue before it and charged the viewer's key again
    /// from cue one.
    #[tokio::test]
    async fn a_failed_run_resumes_instead_of_re_buying_what_worked() {
        let store = MemStore::default();
        let resume = Resume { store: &store, prefix: "openai:m:SV".into() };
        let film = cues(120); // three batches of BATCH=40

        // First attempt: the provider refuses once the third batch comes round. An Upstream refusal
        // is not split-retried, so this is exactly one failed call after two good ones.
        let up = fake(|src: &[String]| {
            if src[0] == "line 80" {
                return Err(CallError::Upstream("provider 429".into()));
            }
            Ok(src.iter().map(|s| format!("SV {s}")).collect())
        });
        let first = run_translation(&up, &film, Duration::from_secs(600), Some(&resume)).await;
        assert!(first.is_err(), "the run should have failed on the third batch");
        assert_eq!(*up.calls.lock().unwrap(), 3, "two good batches and the refusal");
        assert_eq!(store.len(), 2, "the two paid batches should have been remembered");

        // Second attempt, provider healthy. Only the batch that failed may reach it.
        let up2 = fake(|src: &[String]| Ok(src.iter().map(|s| format!("SV {s}")).collect()));
        let done = run_translation(&up2, &film, Duration::from_secs(600), Some(&resume))
            .await
            .expect("the retry should finish");
        assert_eq!(*up2.calls.lock().unwrap(), 1, "the retry re-bought batches it already had");

        // And the film is whole, in order — a resumed run is not a half-translated one.
        assert_eq!(done.len(), 120);
        for (i, cue) in done.iter().enumerate() {
            assert_eq!(cue.text, format!("SV line {i}"), "cue {i} came back wrong");
            assert_eq!(cue.start, film[i].start, "cue {i} lost its timing");
        }
    }

    /// The degraded leaves must NOT be remembered. Keeping the source text is what the harness does
    /// when the model will not cooperate on a single cue; storing that would turn one bad minute into
    /// a day of serving it back without ever retrying.
    #[tokio::test]
    async fn a_kept_source_is_not_remembered_as_a_translation() {
        let store = MemStore::default();
        let resume = Resume { store: &store, prefix: "p".into() };
        // Always three lines, whatever it was asked for. A two-cue batch splits to single cues, and
        // a single cue still gets three back — so every leaf is a wrong-length reply and keeps its
        // source. (One line back would have MATCHED a split-to-one batch and translated it.)
        let up = fake(|_: &[String]| Ok(vec!["a".to_string(), "b".to_string(), "c".to_string()]));
        let out = run_translation(&up, &cues(2), Duration::from_secs(600), Some(&resume)).await;
        assert!(out.is_err(), "a film that translated nothing is not a translation");
        assert_eq!(store.len(), 0, "a kept-source leaf was remembered as if it were a translation");
    }

    /// A resumed run must reach the same verdict as one done in a single pass. The quality gate
    /// counts unusable cues, and a batch answered from the store has to be scored the same way the
    /// live reply was — otherwise a film that echoed its source would pass on retry simply because
    /// nothing recounted it.
    #[tokio::test]
    async fn a_resumed_run_scores_the_same_as_a_fresh_one() {
        let film = cues(80);
        // Echoes the source back: right length, no translation. Trips the ratio gate.
        let echo = |src: &[String]| Ok(src.to_vec());

        let fresh = run_translation(&fake(echo), &film, Duration::from_secs(600), None).await;
        assert!(fresh.is_err(), "an echo is not a translation");

        // Same film, same echo, but with everything served from the store the second time.
        let store = MemStore::default();
        let resume = Resume { store: &store, prefix: "p".into() };
        let _ = run_translation(&fake(echo), &film, Duration::from_secs(600), Some(&resume)).await;
        let replayed = run_translation(&fake(echo), &film, Duration::from_secs(600), Some(&resume)).await;
        assert!(replayed.is_err(), "a replayed echo passed the gate a fresh one failed");
    }

    /// The harness with a deadline long enough never to be the thing under test, and no batch store
    /// — these cases are about what the model does, so nothing may be answered from a previous run.
    async fn run_translation_t(up: &(dyn BatchCall + Sync), cues: &[Cue]) -> Result<Vec<Cue>, String> {
        run_translation(up, cues, Duration::from_secs(600), None).await
    }

    async fn run(upstream: &(dyn BatchCall + Sync), n: usize) -> Result<Vec<String>, String> {
        let src: Vec<String> = cues(n).iter().map(|c| c.text.clone()).collect();
        let budget = Budget {
            untranslated: AtomicUsize::new(0),
            started: std::time::Instant::now(),
            deadline: Duration::from_secs(600),
        };
        translate_batch(upstream, &src, &[], &budget, None).await
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

        // A bracket in the surrounding prose used to break the whole film: the scan ran from the
        // first `[` to the LAST `]`, so a trailing "[sic]" or a leading "I [will] translate:"
        // produced no array at all, which was then fatal rather than degrading.
        for chatty in [
            "[\"a\",\"b\"]\n\nNote: I preserved [sic] the tone.",
            "I [will] translate:\n[\"a\",\"b\"]",
            "Sure! Here is [the] result: [\"a\",\"b\"]",
        ] {
            assert_eq!(parse_json_array(chatty).as_deref(), Some(&["a".to_string(), "b".to_string()][..]),
                "chatty reply not recovered: {chatty:?}");
        }
    }

    /// The scanner runs over a reply body capped at 12 MiB, on the one runtime thread. Retrying
    /// from every `[` re-scanned the same interior for each one: 64k open brackets took 1.7s, and a
    /// megabyte of them would have held the whole service. A candidate that is not an array is
    /// skipped whole, so each byte is examined once.
    ///
    /// The bound is loose on purpose — it catches a return to quadratic, not milliseconds. Linear
    /// does a million brackets in about a millisecond.
    #[test]
    fn the_array_scan_stays_linear() {
        for input in ["[".repeat(400_000), "[x]".repeat(200_000)] {
            let started = std::time::Instant::now();
            assert!(parse_json_array(&input).is_none());
            let took = started.elapsed();
            assert!(took < Duration::from_secs(10), "scan took {took:?} — quadratic again?");
        }
    }

    /// Linear is not the same as cheap. A deeply nested reply is one span per level, and each costs
    /// a parse down to serde's recursion limit — measured at ~3s for 800 KB, tens of seconds at the
    /// 12 MiB body cap, on the single thread that serves every request. Only so many candidates are
    /// worth trying: a real reply has one, prose with brackets has a handful.
    #[test]
    fn a_deeply_nested_reply_does_not_stall_the_thread() {
        let depth = 400_000;
        let input = format!("{}x{}", "[".repeat(depth), "]".repeat(depth));
        let started = std::time::Instant::now();
        assert!(parse_json_array(&input).is_none());
        let took = started.elapsed();
        assert!(took < Duration::from_secs(5), "nested scan took {took:?}");
    }

    /// A candidate that closes but is not an array must be stepped over, not re-entered — and the
    /// real array after it still found. This is the case that keeps the skip honest: stepping one
    /// byte instead would also find it, just quadratically.
    #[test]
    fn a_non_array_candidate_is_skipped_and_the_real_one_found() {
        assert_eq!(parse_json_array(r#"[not json] then ["y"]"#).unwrap(), vec!["y"]);
        // But a WELL-FORMED array is the reply, whatever follows it — so a numeric one is an
        // invalid reply rather than a reason to keep looking. That is what stops a nested
        // array being answered with a fragment of itself.
        assert!(parse_json_array(r#"[1,2] then ["y"]"#).is_none());
        // Brackets inside strings are not structure.
        assert_eq!(parse_json_array(r#"["a[b","c]d"]"#).unwrap(), vec!["a[b", "c]d"]);
        // A `]` inside a string, reached through an escaped quote: still not structure.
        let escaped = "[\"a\\\"]b\"]";
        assert_eq!(parse_json_array(escaped).unwrap(), vec!["a\"]b"]);
        // Unbalanced: nothing to find, and it must say so rather than scan forever.
        assert!(parse_json_array("[[[[[[").is_none());

        // An UNMATCHED `[` before the array — a model echoing a cue like "[MUSIC PLAYING" with no
        // closing bracket — used to discard the whole reply, because depth never returned to zero
        // and the scan concluded nothing later could close either. It can: a later span pairs its
        // own brackets regardless of what came before it.
        for chatty in [
            r#"[ this is an unclosed bracket in my prose ["a","b"]"#,
            r#"note [unbalanced then ["a","b"]"#,
            r#"[MUSIC PLAYING becomes ["a","b"]"#,
        ] {
            assert_eq!(
                parse_json_array(chatty).as_deref(),
                Some(&["a".to_string(), "b".to_string()][..]),
                "an unmatched bracket discarded a valid array: {chatty:?}"
            );
        }
        // Outermost-wins is a guarantee of the first pass, and holds for any BALANCED reply — which
        // is every reply a working model sends.
        assert!(parse_json_array(r#"[["a"],["b"]]"#).is_none());
        // The fallback pass is best-effort by construction: it pairs brackets on a stack, so for
        // input the first pass cannot see past it answers innermost-first and may return a nested
        // fragment. That is a wrong length, which splits and ends at the source text — the same
        // place any other malformed reply ends up, and better than discarding the film.
        assert_eq!(parse_json_array(r#"[[["a"],["b"]]"#).as_deref(), Some(&["a".to_string()][..]));
    }

    /// A model that answers with prose instead of an array is breaking the SAME contract as one
    /// that answers with the wrong number of lines — so it must degrade the same way. Treating it
    /// as an upstream refusal made one chatty reply on batch six kill a whole film.
    #[tokio::test]
    async fn a_reply_that_is_not_an_array_degrades_instead_of_killing_the_film() {
        let up = fake(|s: &[String]| {
            if s.iter().any(|t| t == "line 3") {
                Err(CallError::Contract("model did not return a JSON array".into()))
            } else {
                Ok(s.iter().map(|t| format!("T:{t}")).collect())
            }
        });
        let out = run_translation_t(&up, &cues(8)).await.expect("one chatty reply must not kill the film");
        assert_eq!(out.len(), 8);
        assert_eq!(out[3].text, "line 3", "the unparseable cue should keep its source");
        assert_eq!(out[0].text, "T:line 0", "its neighbours must still be translated");
    }

    /// Calls were counted but never timed, and batches run in sequence at up to LLM_TIMEOUT each,
    /// so the permitted ceiling ran to hours — long after the viewer gave up, still spending their
    /// key. A run stops when its wall clock does, whatever the call count says.
    #[tokio::test]
    async fn a_run_stops_when_its_deadline_passes() {
        // A call that takes real time, the way a provider does.
        struct Slow(Mutex<usize>);
        impl BatchCall for Slow {
            fn call(
                &self,
                sources: &[String],
                _context: &[(String, String)],
            ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, CallError>> + Send + '_>> {
                *self.0.lock().unwrap() += 1;
                // Wrong length, so every batch splits — the deadline has to be enforced INSIDE the
                // recursion, not just between batches. A correct-length fake here never recurses,
                // and so only ever exercised the one check that was never broken.
                let out: Vec<String> = vec!["junk".to_string(); sources.len() + 1];
                Box::pin(async move {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    Ok(out)
                })
            }
        }
        // 400 cues is 10 batches at 20ms each; the deadline expires partway, far under the budget.
        let up = Slow(Mutex::new(0));
        let out = run_translation(&up, &cues(400), Duration::from_millis(50), None).await;
        // It stops CALLING — that is the deadline's whole job. One 40-cue batch splits into up to
        // 79 calls, so a deadline checked only between batches would let all of them run first.
        assert!(*up.0.lock().unwrap() < 20, "it kept calling past the deadline");
        // Whether the run then fails is the ratio's decision, not the clock's. This fake is
        // wrong-length throughout, so everything fell back to source and the ratio refuses it.
        assert!(out.is_err(), "a run that translated almost nothing must not be served");
    }

    /// A forced-narrative track is mostly place names and proper nouns that legitimately come back
    /// unchanged. Refusing it costs the full LLM bill, caches nothing, and makes every retry pay
    /// again — so the bar for "not a translation" has to sit well above a normal signs track.
    #[tokio::test]
    async fn a_forced_narrative_track_is_not_refused() {
        for total in [30usize, 40, 60, 100, 150] {
            let up = fake(move |s: &[String]| {
                Ok(s.iter()
                    .map(|t| {
                        let n: usize = t.trim_start_matches("line ").parse().unwrap_or(999);
                        // Three in ten unchanged, spread through the track.
                        if n % 10 < 3 { t.clone() } else { format!("T:{t}") }
                    })
                    .collect())
            });
            let out = run_translation_t(&up, &cues(total)).await;
            assert!(out.is_ok(), "a {total}-cue forced track at 30% names was refused: {out:?}");
        }
    }

    /// Nothing translated at all is refused at any size — the floor protects tracks that are partly
    /// unchanged, not ones that came back verbatim.
    #[tokio::test]
    async fn a_tiny_track_echoed_verbatim_is_still_refused() {
        for total in [3usize, 7, 8] {
            let echo = fake(|s: &[String]| Ok(s.to_vec()));
            let out = run_translation_t(&echo, &cues(total)).await;
            assert!(out.is_err(), "a {total}-cue track echoed verbatim was accepted");
        }
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
        assert!(run_translation_t(&up, &cues(60)).await.is_err(), "an untranslated film was returned");
        assert!(
            *up.calls.lock().unwrap() <= 2 * 60,
            "spent {} calls for 60 cues, above 2n",
            up.calls.lock().unwrap()
        );
    }

    /// A run already certain to fail stops early — but only once it is CERTAIN, meaning the dead
    /// count alone already exceeds the bar for the whole film. An echoing model reaches that two
    /// thirds of the way through, and the remaining batches are never paid for.
    #[tokio::test]
    async fn a_doomed_run_stops_before_the_end() {
        let up = fake(|s: &[String]| Ok(s.to_vec()));
        assert!(run_translation_t(&up, &cues(2000)).await.is_err(), "an echoing film must be refused");
        let spent = *up.calls.lock().unwrap();
        let all_batches = 2000_usize.div_ceil(BATCH);
        assert!(spent < all_batches, "spent {spent} calls of {all_batches} batches — it ran to the end");
    }

    /// And it must never stop early on a film that would have passed. A cold open of title cards,
    /// song lyrics and place names can run for several batches; judging those against themselves
    /// refused films whose true ratio was nowhere near the bar.
    #[tokio::test]
    async fn a_long_rough_opening_does_not_abort_a_good_film() {
        // The first 120 cues — three whole batches — come back unchanged; the remaining 1080 are
        // translated cleanly. 90 dead in 1200 is 7.5%, nowhere near two thirds.
        let up = fake(|s: &[String]| {
            Ok(s.iter()
                .map(|t| {
                    let n: usize = t.trim_start_matches("line ").parse().unwrap_or(9999);
                    if n < 90 { t.clone() } else { format!("T:{t}") }
                })
                .collect())
        });
        let out = run_translation_t(&up, &cues(1200)).await.expect("a rough opening must not abort it");
        assert_eq!(out.len(), 1200);
        assert_eq!(out[1000].text, "T:line 1000");
    }

    /// A provider that is merely SLOW, and otherwise perfect, must not produce a served film. It
    /// translates the opening, the clock runs out, and the rest is never sent — folding those
    /// never-attempted cues into the same counter as legitimate unchanged ones put a 66%
    /// source-language track just under the bar, served and cached for sixty days.
    #[tokio::test]
    async fn a_slow_but_correct_provider_does_not_produce_a_half_translated_film() {
        struct Slow(Mutex<usize>);
        impl BatchCall for Slow {
            fn call(
                &self,
                sources: &[String],
                _context: &[(String, String)],
            ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, CallError>> + Send + '_>> {
                *self.0.lock().unwrap() += 1;
                // Always correct — the only problem is that it is slow.
                let out: Vec<String> = sources.iter().map(|t| format!("T:{t}")).collect();
                Box::pin(async move {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    Ok(out)
                })
            }
        }
        let up = Slow(Mutex::new(0));
        // 2000 cues is 50 batches at 20ms. The clock is set to run out well past halfway, so the
        // skipped share lands UNDER the two-thirds bar — which is the whole point: at 80% skipped
        // the ratio would refuse it anyway and prove nothing. 13a measured the live case at 66.0%.
        let out = run_translation(&up, &cues(2000), Duration::from_millis(600), None).await;
        let called = *up.0.lock().unwrap();
        assert!(called < 50, "it kept calling past the deadline ({called} calls)");
        assert!(called > 10, "the deadline tripped too early to test the ratio ({called} calls)");
        assert!(out.is_err(), "a film that was mostly never sent must not be served: {out:?}");
    }

    /// The cost ceiling must not become a stricter quality gate than the quality gate. A film with
    /// scattered wrong-length replies — a model that declines to echo back the blank cues real SRTs
    /// carry — splits often enough to exhaust the budget at well under a tenth of cues dead, and
    /// that used to hard-fail a film the two-thirds bar is explicitly written to accept.
    #[tokio::test]
    async fn exhausting_the_budget_does_not_refuse_an_acceptable_film() {
        // One cue in ten comes back wrong-length, so every batch splits to isolate four of them.
        let up = fake(|s: &[String]| {
            if s.iter().any(|t| t.trim_start_matches("line ").parse::<usize>().is_ok_and(|n| n.is_multiple_of(10))) {
                Ok(vec!["junk".to_string(); s.len() + 1])
            } else {
                Ok(s.iter().map(|t| format!("T:{t}")).collect())
            }
        });
        let out = run_translation_t(&up, &cues(2000)).await;
        assert!(out.is_ok(), "the cost ceiling refused a film the ratio accepts: {out:?}");
        assert_eq!(out.unwrap().len(), 2000);
    }

    /// Cost is bounded by the shape of the recursion, not by a counter. The split is a binary tree
    /// with one leaf per cue, so a batch of n costs at most 2n-1 calls however badly the model
    /// behaves. Pinned here because a counter above that bound is dead code and a counter below it
    /// refuses films the ratio is meant to accept — both of which have shipped.
    #[tokio::test]
    async fn cost_is_bounded_by_the_split_itself() {
        // Exactly one batch, wrong-length at every size, so every node of the tree fails and the
        // ratio bail cannot cut the run short before the worst case is actually reached. Asserted
        // as equality: a bound the run never approaches would not notice the shape changing.
        let up = fake(|s: &[String]| Ok(vec!["junk".to_string(); s.len() + 1]));
        let _ = run_translation_t(&up, &cues(BATCH)).await;
        assert_eq!(*up.calls.lock().unwrap(), 2 * BATCH - 1, "the split is not a binary tree of cues");
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
        let out = run_translation_t(&up, &cues(3)).await.expect("one odd line must not fail a 3-cue track");
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
        let out = run_translation_t(&up, &cues(60)).await.expect("a mostly-translated film must succeed");
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
            let err = run_translation_t(&echo, &cues(200)).await.unwrap_err();
            assert!(err.contains("unusable"), "an echo ({name}) was accepted: {err}");
        }
        // A leading space is the same trick from the other end.
        let echo = fake(|s: &[String]| Ok(s.iter().map(|t| format!(" {t}")).collect()));
        assert!(run_translation_t(&echo, &cues(200)).await.is_err(), "a leading-space echo was accepted");

        let blanks = fake(|s: &[String]| Ok(vec![String::new(); s.len()]));
        let err = run_translation_t(&blanks, &cues(200)).await.unwrap_err();
        assert!(err.contains("unusable"), "a reply of empty strings was accepted: {err}");
    }

    /// A blank source cue has nothing to translate, so whatever the model returns for it is a line
    /// the film does not contain. The dead-count and the fallback both skipped blank sources — they
    /// were written for the mirror case — so an invented line shipped verbatim and cached.
    #[tokio::test]
    async fn an_invented_line_for_a_blank_cue_is_dropped() {
        let mut film = cues(6);
        film[2].text = String::new();
        film[4].text = "   ".into();
        let up = fake(|s: &[String]| {
            Ok(s.iter()
                .map(|t| if t.trim().is_empty() { "INVENTED".to_string() } else { format!("T:{t}") })
                .collect())
        });
        let out = run_translation_t(&up, &film).await.expect("five good cues is not an unusable film");
        assert_eq!(out[2].text, "", "a line was invented for an empty cue");
        assert_eq!(out[4].text, "   ", "a line was invented for a whitespace-only cue");
        assert_eq!(out[3].text, "T:line 3", "its neighbours still translate");
    }

    /// The rolling context is rendered into the next batch's prompt as "already translated, do not
    /// re-translate". A cue that fell back to its source is a FAILURE, and putting it there offers
    /// that failure to the model as precedent — inviting it to leave the same word alone next time.
    /// Nothing here distinguishes a fallback from a name that legitimately stays put, so neither
    /// goes in; an identity pair teaches the model almost nothing either way.
    #[tokio::test]
    async fn a_failed_cue_is_not_offered_as_precedent() {
        use std::sync::Mutex as StdMutex;
        struct Recorder(StdMutex<Vec<Vec<(String, String)>>>);
        impl BatchCall for Recorder {
            fn call(
                &self,
                sources: &[String],
                context: &[(String, String)],
            ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, CallError>> + Send + '_>> {
                self.0.lock().unwrap().push(context.to_vec());
                // "line 79" is the last cue of batch two, so its failure would otherwise still be
                // in the window when batch three is sent.
                let bad = sources.iter().any(|t| t == "line 79");
                let out: Vec<String> = if bad {
                    vec!["junk".to_string(); sources.len() + 1]
                } else {
                    sources.iter().map(|t| format!("T:{t}")).collect()
                };
                Box::pin(async move { Ok(out) })
            }
        }
        let up = Recorder(StdMutex::new(Vec::new()));
        let out = run_translation_t(&up, &cues(120)).await.expect("one bad cue must not fail the film");
        assert_eq!(out[79].text, "line 79", "the bad cue should have kept its source");

        let seen = up.0.lock().unwrap();
        let offered: Vec<(String, String)> = seen.iter().flatten().cloned().collect();
        let echoes: Vec<&(String, String)> = offered.iter().filter(|(s, d)| s == d).collect();
        assert!(echoes.is_empty(), "an unchanged pair was offered as an established translation: {echoes:?}");
        // And the context is still doing its job for cues that really were translated.
        assert!(
            offered.iter().any(|(s, d)| d == &format!("T:{s}")),
            "no real pairs reached the context at all"
        );
    }

    /// Blanks below the ratio pass the gate, and those cues used to be SHIPPED blank — rendering as
    /// nothing in a film that otherwise succeeded and caching that way for 60 days. A blanked line
    /// falls back to its source, the same choice the wrong-length leaf makes.
    #[tokio::test]
    async fn a_blanked_cue_falls_back_to_its_source() {
        // One in ten blank: under the 25% gate, so the run succeeds and the cues are served.
        let up = fake(|s: &[String]| {
            Ok(s.iter()
                .map(|t| {
                    let n: usize = t.trim_start_matches("line ").parse().unwrap_or(999);
                    if n.is_multiple_of(10) { String::new() } else { format!("T:{t}") }
                })
                .collect())
        });
        let out = run_translation_t(&up, &cues(200)).await.expect("a tenth blank is under the gate");
        assert_eq!(out[10].text, "line 10", "a blanked cue was shipped empty");
        assert_eq!(out[11].text, "T:line 11");
        assert!(out.iter().all(|c| !c.text.trim().is_empty()), "a cue would render as nothing");
    }

    /// The gate is a ratio at every size above one. An absolute floor exempted short tracks from it
    /// entirely: with a floor of eight, seven dead cues out of eight — 87% of the track still in the
    /// source language — passed and cached for sixty days as a successful translation.
    ///
    /// Asserted as behaviour rather than by re-deriving the formula, which would pass for any
    /// threshold as long as the copy stayed in sync with the code.
    #[test]
    fn no_size_is_exempt_from_the_ratio() {
        for seen in 2..=40usize {
            // Nothing translated, and one lone survivor, are both non-translations at every size.
            assert!(unusable(seen, seen), "a wholly untranslated {seen}-cue track was accepted");
            if seen >= 4 {
                assert!(unusable(seen - 1, seen), "{} of {seen} dead was accepted", seen - 1);
            }
            // A third unchanged is a normal signs track and must survive at every size.
            assert!(!unusable(seen / 3, seen), "{} names in {seen} cues was refused", seen / 3);
        }
        // The exact shapes the floor used to hide.
        for (kept, seen) in [(7usize, 8usize), (7, 9), (7, 10)] {
            assert!(unusable(kept, seen), "{kept} of {seen} dead must not be a translation");
        }
    }

    /// A one-cue track is 0% or 100% dead by construction, so a ratio can only ever refuse it — and
    /// a sign reading MOSCOW coming back unchanged IS the translation. Refusing costs the viewer the
    /// track entirely, permanently: the model is deterministic, so every retry reproduces it.
    #[test]
    fn a_single_cue_track_is_never_refused() {
        assert!(!unusable(1, 1));
        assert!(!unusable(0, 1));
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
            let out = run_translation_t(&up, &cues(total)).await;
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
        let out = run_translation_t(&up, &cues(600)).await.expect("a fifth of cues matching is normal");
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
        let out = run_translation_t(&up, &cues(600)).await.expect("a rough opening must not abort the film");
        assert_eq!(out.len(), 600);
        assert_eq!(out[300].text, "T:line 300");
    }

    /// An upstream error must not be split-retried — that spent six more calls against a provider
    /// that had just said no, with no backoff, and failed anyway.
    #[tokio::test]
    async fn an_upstream_error_is_not_split_retried() {
        let up = fake(|_: &[String]| Err(CallError::Upstream("provider 429: rate limited".into())));
        assert!(run_translation_t(&up, &cues(40)).await.is_err());
        assert_eq!(*up.calls.lock().unwrap(), 1, "a 429 was split-retried instead of surfacing");
    }

    /// A transport/auth error is not a contract violation: it must surface, not degrade to the source.
    #[tokio::test]
    async fn an_upstream_error_still_fails() {
        let up = fake(|_: &[String]| Err(CallError::Upstream("provider 401: bad key".into())));
        assert!(run(&up, 4).await.is_err(), "an upstream failure must not become an untranslated film");
    }
}

#[cfg(test)]
mod provider_reply_tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// A one-shot chat endpoint answering with a canned body.
    async fn provider_owned(status: &'static str, body: String) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = [0u8; 8192];
                let _ = sock.read(&mut buf).await;
                let resp = format!(
                    "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = sock.write_all(resp.as_bytes()).await;
                let _ = sock.shutdown().await;
            }
        });
        format!("http://{addr}")
    }

    async fn call(body: &'static str, status: &'static str) -> Result<Vec<String>, CallError> {
        call_owned(body.to_string(), status).await
    }

    async fn call_owned(body: String, status: &'static str) -> Result<Vec<String>, CallError> {
        let base = provider_owned(status, body).await;
        CHAT_BASE.with(|b| *b.borrow_mut() = Some(base));
        let http = reqwest::Client::new();
        let llm = LlmConfig {
            provider: Provider::OpenAI,
            api_key: "k".into(),
            model: "m".into(),
        };
        llm_translate(&http, &llm, &["a".to_string()], "Swedish", &[]).await
    }

    /// A 200 whose envelope carries no text is a safety filter or an empty candidate list, and both
    /// are about THIS batch's dialogue — film dialogue trips content filters routinely. Classified
    /// as Upstream it propagated and killed the whole film; it has to degrade like any other reply
    /// the model got wrong.
    #[tokio::test]
    async fn an_empty_reply_is_a_contract_violation_not_a_refusal() {
        let err = call(r#"{"choices":[]}"#, "200 OK").await.unwrap_err();
        assert!(
            matches!(err, CallError::Contract(_)),
            "an empty provider reply must degrade, not propagate: {err:?}"
        );
    }

    /// A 200 carrying a body that will not decode is the provider answering badly about THIS
    /// batch, not refusing to serve — an HTML error page from a gateway, or a stream a proxy cut
    /// short. The previous fix classified the reply-with-no-text case and left the decode one
    /// statement above it on the Upstream path, so this still killed whole films.
    #[tokio::test]
    async fn an_unparseable_body_is_a_contract_violation_not_a_refusal() {
        for body in ["<html><body>Bad Gateway</body></html>", "", "{\"choices\": [truncated"] {
            let err = call_owned(body.to_string(), "200 OK").await.unwrap_err();
            assert!(
                matches!(err, CallError::Contract(_)),
                "a 200 with body {body:?} must degrade, not propagate: {err:?}"
            );
        }
    }

    /// A reply that is text but not an array is the model misbehaving too.
    #[tokio::test]
    async fn prose_instead_of_an_array_is_a_contract_violation() {
        let body = r#"{"choices":[{"message":{"content":"I cannot help with that."}}]}"#;
        assert!(matches!(call(body, "200 OK").await.unwrap_err(), CallError::Contract(_)));
    }

    /// A body that runs past the cap is the model free-running on this batch's dialogue — asking
    /// for less at a time is exactly what can help, so it degrades rather than propagating.
    #[tokio::test]
    async fn an_oversized_body_is_a_contract_violation() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = [0u8; 8192];
                let _ = sock.read(&mut buf).await;
                // Declared over the cap: rejected before a byte of it is transferred.
                let head = format!(
                    "HTTP/1.1 200 OK\r\ncontent-length: {}\r\n\r\n",
                    crate::fetch::MAX_BODY + 1
                );
                let _ = sock.write_all(head.as_bytes()).await;
                let _ = sock.shutdown().await;
            }
        });
        CHAT_BASE.with(|b| *b.borrow_mut() = Some(format!("http://{addr}")));
        let http = reqwest::Client::new();
        let llm = LlmConfig { provider: Provider::OpenAI, api_key: "k".into(), model: "m".into() };
        let err = llm_translate(&http, &llm, &["a".to_string()], "Swedish", &[])
            .await
            .expect_err("an oversized body must fail");
        assert!(matches!(err, CallError::Contract(_)), "an oversized body must degrade: {err:?}");
    }

    /// A stream that dies mid-body IS upstream — the provider stopped talking, and asking for less
    /// at a time cannot help. This is the one half of a failed read that must not degrade.
    #[tokio::test]
    async fn a_dead_stream_is_an_upstream_failure() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = [0u8; 8192];
                let _ = sock.read(&mut buf).await;
                // Promise 500 bytes, send 10, hang up.
                let _ = sock
                    .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 500\r\n\r\n0123456789")
                    .await;
                let _ = sock.shutdown().await;
            }
        });
        CHAT_BASE.with(|b| *b.borrow_mut() = Some(format!("http://{addr}")));
        let http = reqwest::Client::new();
        let llm = LlmConfig { provider: Provider::OpenAI, api_key: "k".into(), model: "m".into() };
        let err = llm_translate(&http, &llm, &["a".to_string()], "Swedish", &[])
            .await
            .expect_err("a truncated stream must fail");
        assert!(matches!(err, CallError::Upstream(_)), "a dead stream must propagate: {err:?}");
    }

    /// But a provider REFUSING is upstream, and must not be split-retried.
    #[tokio::test]
    async fn a_provider_status_error_is_an_upstream_failure() {
        let err = call(r#"{"error":"nope"}"#, "429 Too Many Requests").await.unwrap_err();
        assert!(matches!(err, CallError::Upstream(_)), "a 429 must propagate: {err:?}");
    }

    /// And a good reply still comes back.
    #[tokio::test]
    async fn a_well_formed_reply_parses() {
        let body = r#"{"choices":[{"message":{"content":"[\"Hej\"]"}}]}"#;
        assert_eq!(call(body, "200 OK").await.unwrap(), vec!["Hej"]);
    }
}

#[cfg(test)]
mod deepl_tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn deepl(body: &'static str) -> Result<Vec<String>, CallError> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = [0u8; 8192];
                let _ = sock.read(&mut buf).await;
                let resp = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = sock.write_all(resp.as_bytes()).await;
                let _ = sock.shutdown().await;
            }
        });
        DEEPL_BASE.with(|b| *b.borrow_mut() = Some(format!("http://{addr}")));
        let http = reqwest::Client::new();
        let llm = LlmConfig { provider: Provider::DeepL, api_key: "k".into(), model: String::new() };
        deepl_translate(&http, &llm, &["a".to_string(), "b".to_string()], "Swedish").await
    }

    /// A malformed entry must invalidate the reply, not vanish from it. Dropping it shrank the
    /// array, which the same-length check upstream catches only by coincidence — a drop offset by
    /// a spurious extra entry would misalign every cue at exactly the right length.
    #[tokio::test]
    async fn a_non_string_translation_invalidates_the_reply() {
        let err = deepl(r#"{"translations":[{"text":"Hej"},{"text":null}]}"#)
            .await
            .expect_err("a null translation was accepted");
        assert!(matches!(err, CallError::Contract(_)), "{err:?}");

        // And the shape that would otherwise slip through: one dropped, one added, length intact.
        let err = deepl(r#"{"translations":[{"text":"Hej"},{"nope":"x"},{"text":"Da"}]}"#)
            .await
            .expect_err("a misaligned reply of the right length was accepted");
        assert!(matches!(err, CallError::Contract(_)), "{err:?}");
    }

    /// An unknown language never reaches the network: splitting cannot make DeepL learn it, so it
    /// fails hard rather than degrading to source text.
    #[tokio::test]
    async fn an_unknown_language_is_refused_before_calling() {
        let http = reqwest::Client::new();
        let llm = LlmConfig { provider: Provider::DeepL, api_key: "k".into(), model: String::new() };
        let err = deepl_translate(&http, &llm, &["a".to_string()], "Klingon")
            .await
            .expect_err("an unknown language was sent to DeepL");
        assert!(matches!(err, CallError::Upstream(_)), "must not split-retry: {err:?}");
    }

    #[tokio::test]
    async fn a_well_formed_reply_parses() {
        let out = deepl(r#"{"translations":[{"text":"Hej"},{"text":"Da"}]}"#).await.unwrap();
        assert_eq!(out, vec!["Hej", "Da"]);
    }
}
