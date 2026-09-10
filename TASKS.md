# Task list — audited improvements

Derived from a full read of `src/` plus an adversarial audit pass. Every claim below carries the
`file:line` it was verified against. **Verify the premise still holds before implementing an item; if
the code has moved and the premise is false, skip it and say so.**

Priorities, in order: **correctness of what already ships → resource usage (OpenSubtitles quota, the
user's LLM bill, CPU/subprocess spawns) → performance → new surface area.**

## Do not implement — invalidated by audit

- ~~Charset detection / `encoding_rs` in `fetch.rs`~~. The api.opensubtitles.com `/download` endpoint
  converts server-side; the file behind the temporary link is always UTF-8, so `from_utf8_lossy`
  (`fetch.rs:36`) never corrupts anything. Replaced by task 15.
- ~~Consensus clustering as a Tier-1 anchor~~. `sync.rs:104` nulls ffsubsync's stdout/stderr so no
  offset is reported, and ffsubsync applies a framerate *scale*, not a scalar shift. Pairwise over N
  candidates is N-1 Python spawns at `TIER1_BUDGET` = 20s each (`sync.rs:34`) on a
  `new_current_thread` runtime (`main.rs:229`). Not viable in a request path.
- ~~Add the video hash to the translate cache key~~. It would mint a fresh full-film LLM bill per
  encode — exactly what that key is shaped to avoid. See task 5 for the correct shape.

## Tier A — cheap, high value

### 1. Stop burning an OpenSubtitles credit per translation attempt
`produce_translation` calls `client.download(source.file_id)` directly (`addon.rs:566`) instead of
`subtitle_srt` (`addon.rs:384`), so the source SRT is neither read from nor written to the
`os:{file_id}` cache. The credit is spent on the API call itself, and dodging that quota is the
stated reason the proxy-and-cache design exists (`opensubtitles.rs:3`). Every retry after the 600s
backoff, and every second target language for the same film, pays again for a cached file.

**Fix:** swap to `subtitle_srt`.

### 2. Tier-1 picks the alphabetically-first hash match, not the best one
`tier1_reference` (`addon.rs:258`) is `find(|s| s.hash_match)`, called at `addon.rs:218` *after*
`rank` sorted by `lang` ascending (`opensubtitles.rs:194`). A hash-matched forced/signs-only track —
a handful of cues — therefore beats a full one as the ffsubsync reference, and the bad alignment
caches for 60 days.

**Fix:** `max_by_key(|s| (s.hash_match, fit_score(s, None)))` over the hash matches. No invariant
change.

### 3. The documented non-English fallback is dead code
`addon.rs:562` searches `languages = "en"`, so `.or_else(|| subs.first())` at `:564` can never be a
non-English fallback — contradicting the comment at `:561`. A title with no English sub returns
`"no source subtitle to translate"` even when a good Spanish source exists.

**Fix:** search `"all"` and let `best_for` prefer English; or fix the comment. Prefer the former.

### 4. `lang` is unnormalized free text in the translate cache key
`addon.rs:501` interpolates `lang` verbatim, bounded only by `MAX_LANG`. `"Swedish"`, `"swedish"`
and `"sv"` are three keys and three full films billed, while `deepl_code` (`translate.rs:542`)
already maps all three to `SV`.

**Fix:** canonicalize `lang` before it enters the key. Also shrinks the abuse surface in task 13.

## Tier B — structural

### 5. Translated tracks bypass the sync ladder entirely
`addon.rs:562` searches with `None` for the moviehash; `best_for` → `fit_score(s, None)`
(`opensubtitles.rs:203`) so neither hash nor filename affects the pick, and with no hash sent
`hash_match` is false on every result anyway. `main.rs:135` requires exactly 5 path segments, so
`videoHash`/`filename` cannot reach the handler. The cache key (`addon.rs:501`) carries no source id
and no hash. Net: a translated track is timed against an arbitrary encode and never aligned.

**Fix shape:** extend the route to carry the Stremio `<extra>` blob; thread hash+filename into the
search and rank with `opensubtitles::rank`. Key the *translation* by source content identity (source
`file_id`), then apply Tier-1/Tier-2 as a **separate, separately-cached layer** on top of the
translated body, reusing the raw/`:ref:`/`:resync:` namespacing at `addon.rs:459`. One translation,
many alignments.

### 6. No single-flight on expensive work
`AppState` (`state.rs:13`) has no in-flight map or semaphore; both handlers are read-cache-then-work
with no lock (`addon.rs:299`/`:332`, `addon.rs:511`/`:522`). Note the app's own `.json`-then-`.srt`
flow (`addon.rs:474`) is sequential, so the real exposure is multiple devices and manual retries
inside the up-to-600s window. `SYNC_SEQ` is **not** a mitigation — per `addon.rs:329` it exists to
make concurrent duplicates *safe*, not to prevent them.

**Fix:** a keyed in-flight map on `AppState` (`Mutex<HashMap<String, broadcast::Sender<…>>>`) checked
after the cache miss, covering both translation and the Tier-2 alass path.

### 7. Runs are all-or-nothing; the retry re-pays in full
`run_translation` accumulates into a local `out` (`translate.rs:63`); the only `cache.put` is after
full success (`addon.rs:572`). The `syncfail:` marker then refuses retries for 600s (`addon.rs:31`,
`:526`).

**Fix:** cache per batch, keyed on `hash(sources) + provider + model + lang`. Batches become
idempotent, a failed run resumes, and recurring lines are free across a series.

## Tier C — sequenced; do in this order or not at all

### 8. Retry the same batch on a retryable upstream error
`translate.rs:301` propagates `Upstream` at any batch size, `translate.rs:81` propagates it out of
the loop; the only `sleep` calls in the file are inside `#[cfg(test)]` fakes. One 429 mid-film kills
a ten-minute job.

**Fix:** retry the *same* batch 2–3 times honoring `Retry-After` with exponential backoff for
429/5xx only (401/400 must not retry), then surface `Upstream`. `RUN_DEADLINE` already bounds total
retry time. This complements — does not conflict with — the deliberate no-split decision at
`translate.rs:283`.

**Gotcha:** the test at `translate.rs:1339` asserts `calls == 1` exactly, not "no split". It must be
rewritten to assert the split never *widened* (every recorded call has `sources.len() == BATCH`), or
the invariant is silently deleted while the suite stays green.

### 9. Bounded concurrent batches
Sequential today (`translate.rs:76`). Requires task 8 first: 6000/40 = 150 batches, and unbounded
fan-out on one BYOK key is a guaranteed 429, which currently aborts the run.

**Two guards break and must be reworked, not dropped:**
- `unusable` early-bail (`translate.rs:97`) stays *sound* but stops being *early* — with all batches
  in flight the film is already paid for. Test `a_doomed_run_stops_before_the_end`
  (`translate.rs:1015`) fails.
- The deadline check is at batch *entry* (`translate.rs:253`); fan out 150 and all pass at t≈0. Test
  `a_run_stops_when_its_deadline_passes` (`translate.rs:914`) fails.

**Fix:** a bounded semaphore (4–6), and move both gates to fire per completion *window* rather than
per entry, so they still bound spend and wall-clock.

### 10. Glossary pass (only after 8 and 9)
One cheap pass over the file yielding proper nouns / terms of address / register, inlined into every
batch prompt, replacing the 6-pair rolling `CONTEXT_WINDOW` that cannot remember a name from reel one.

This *does* violate the stated "the model never sees the full file" principle (`translate.rs:2`) —
but precisely: the same-length contract is per-batch and cue-aligned, and a glossary output is a name
list, not a cue mapping, so the "LLM ate the SRT" failure class is not reintroduced. The real cost is
the focus argument plus roughly doubled input tokens. Update the module doc if implemented.

## Tier D — smaller

### 11. `/health` is blind to half the failures
`os_fails` is only touched in `handle_subtitles` (`addon.rs:193`, `:203`);
`produce_translation`'s OpenSubtitles failures (`addon.rs:562`) never reach `/health`
(`main.rs:45`, `:79`). Fix this before adding any cost/usage counters.

**Since fixed:** the counter moved into the shared search path, so a translation's failed search
reaches `/health` too — and it is also published on `/metrics` as
`subtitles_consecutive_failures{kind="opensubtitles"}`.

### 12. No progress signal on a request that can block 600s
`RUN_DEADLINE` (`translate.rs:122`) awaited inline at `addon.rs:522`; no 202/polling shape in
`main.rs:133`. This is *intentional* — `addon.rs:474` documents the app showing its own wait — so the
improvement is resumability plus a progress signal, not "the endpoint blocks unnoticed". Any proxy
idle timeout in between converts a successful translation into a client-side failure whose work is
discarded (task 7).

### 13. Per-config ceilings
Nothing is keyed to the config segment. `MAX_CUES` (`translate.rs:37`), `RUN_DEADLINE` and the 600s
backoff each cap one *run*; what is uncapped is **breadth** — distinct titles × languages, each a
fresh paid film against a URL that is a bearer secret.

### 14. Widen the Tier-1 anchor — SKIPPED, premise does not hold

**Not implemented.** The premise is that a release-group match makes a subtitle usable as a timing
anchor. It does not, and the reason is sharper than "it might be wrong".

An anchor is worth having for exactly one property: its timing is known-correct. A hash match has
that by construction — OpenSubtitles is asserting the sub was authored against this exact encode. A
release-group match asserts something else entirely: same *encode*, nothing about whether *this
upload* is correctly timed against it. Uploaders hand-shift subs routinely.

And the two facts pull apart in the case that matters. If a release-matched sub is correctly timed,
the target probably already agrees with it and the alignment is a no-op. If it is mis-timed, we have
just aligned a good subtitle onto a bad reference and made it worse — for the no-hash-match case,
which is the common one currently served correctly as-is.

The proposed shift guard bounds the damage rather than fixing the reasoning: it caps how wrong a bad
anchor can make things, while still preferring a guess over the honest "served as-is". That trades a
known-correct behaviour for a speculative one, in the direction of priority 1.

Tier 2 remains the answer for a no-hash-match sub, and it is one the user asks for explicitly.

### 14b. Original proposal, for the record
`tier1_reference` (`addon.rs:258`) accepts only `hash_match`. A release-group match is evidence about
the *encode*, not about this upload being correctly timed against it, and uploaders hand-shift subs
routinely; it is weighted 3000 for *ranking* (`opensubtitles.rs:224`), where being wrong costs a
worse pick, whereas as an *anchor* being wrong corrupts every other track for 60 days. If
implemented: require group + resolution + source token, and refuse an alignment whose implied shift
exceeds a few seconds. Document it as an intentional relaxation of the trusted-anchor invariant
stated at `addon.rs:254` and `sync.rs:5`.

### 15. Make the two undeclared assumptions explicit
The encoding item is a non-issue only because of an external guarantee nothing records. Add a
one-line comment at `fetch.rs:33` naming it, and send `"sub_format": "srt"` explicitly in the
download POST (`opensubtitles.rs:102`) so the SRT-only assumption in `srt::parse` is enforced rather
than incidental.

### 16. Nice-to-have

- **WebVTT output route — done.** Rendered at the router from the finished response, so no format
  flag runs through the sync ladder.
- **Series prefetch — SKIPPED.** It would start a full translation of an episode the viewer has not
  asked for, spending their own provider account on a guess. That is the exact cost task 13 was added
  to bound, and the two land in the same release: adding a daily ceiling and then quietly consuming
  it speculatively is incoherent. Worth revisiting only behind an explicit opt-in, and with prefetched
  runs charged to a separate allowance from the ones a viewer asked for.
- **`/login` to mint `osToken` — SKIPPED.** It means accepting an OpenSubtitles username and password,
  which is a materially larger credential surface than the current design has anywhere: every secret
  today is a token the user pastes once, and none is a reusable account password. The gain is saving
  one paste during setup. Wrong trade for this codebase, where the config segment is already a bearer
  secret handled with some care.
