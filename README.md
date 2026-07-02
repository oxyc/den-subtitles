# den-subtitles

A self-hosted **Stremio subtitles addon** for [Den](https://github.com/oxyc/den), in one small Rust
binary + the subtitle-sync toolchain. It does three things the public OpenSubtitles addon can't:

1. **Fetch** — OpenSubtitles, **hash-matched** and served from our own cache (dodges the per-IP
   download quota).
2. **Translate** — **BYOK** AI translation (default `gpt-4o-mini`) through a harness that survives a
   full film without losing cue↔timing sync.
3. **Sync** — an auto-sync ladder (hash → reference-align → `alass` audio VAD) so subtitles line up.

```
Den (Apple TV) ──/<config>/subtitles/movie/tt…/videoHash=…──►  addon   OpenSubtitles (hash-first)
               ◄──── { subtitles:[ {url:/subtitle/…} ] }──────┘         served from our cache
Den (app UX)   ──/<config>/translate/movie/tt…/English.json─►  addon   fetch EN → LLM harness → SRT
               ◄──── { url:/translate/…/English.srt }─────────┘         cached per hash+lang+model
Den (engine)   ──GET …/English.srt───────────────────────────►  addon   instant (cache warm) native track
```

It's the **den-reel shape** (an I/O proxy whose CPU work lives in subprocesses), not the den-scout
shape — so it's Rust: hyper + tokio + reqwest-rustls + `tokio::process`, with a slim-debian runtime
carrying `alass`/`ffsubsync`/`ffmpeg`.

## Why each piece

- **Hash matching is what makes subtitles synced.** A subtitle's timing is tied to a *specific
  encode*. The Den app computes the file's OSHash and sends it as `videoHash`; we forward it as
  `moviehash` and float `moviehash_match` results to the top — those are correct by construction.
- **Translation stays app-triggered.** Native OpenSubtitles tracks come back on the normal
  `subtitles` resource. Translation is a separate, app-driven endpoint (`/translate/…`) so the app
  can render a faded "English → Translate to X" track: on tap it warms the `.json` endpoint (showing
  its own wait), then hands the returned `.srt` URL to the engine as a **native track**.
- **The translation harness** (`src/translate.rs`) is the crown jewel. The model never sees the
  whole file or any timestamp; cues go out in small JSON-array batches under a strict same-length
  contract; a length mismatch splits the batch and retries down to a single cue; a rolling window of
  prior (source→translation) pairs keeps names/tone consistent across the film.

## Config

Per-install config is base64url-encoded in the addon URL (den-scout / Torrentio style), a bearer
secret the app stores in the Keychain. The **OpenSubtitles key** (subtitle source) is required; the
**LLM key** (translation) is **optional** — omit it for a fetch + auto-sync-only install with no AI.
Build it at `/configure`. Nothing credential-shaped lives in the environment; `.env.example` is just
infra (port, cache, origin).

Supported providers: OpenAI, Google, Anthropic, xAI, OpenRouter (chat) and DeepL (MT). Default model
is the cheap/fast/decent tier per provider; step up to a bigger model to re-translate a title that
reads badly (the cache is keyed by provider+model, so it just overwrites).

## Run

```sh
cp .env.example .env          # infra only — keys are entered at /configure
cargo run                     # local (needs alass/ffsubsync on PATH for the sync tiers)
# or
docker build -t den-subtitles . && docker run -p 8093:8093 --env-file .env den-subtitles
```

`cargo test` covers the SRT round-trip, config decode/validate, the JSON-array parse, the
OpenSubtitles result ordering, the Tier-1 reference selection, the `?resync=` SSRF guard, and the
sync subprocess orchestration (spawn → arg contract → read-back → cleanup, against fake binaries).

## Status

Working: manifest + `/configure`, OpenSubtitles hash-matched search, cached subtitle proxy, the full
BYOK translation harness (all providers), the cache, the Docker image, and the **sync ladder**
(`src/sync.rs`) — now wired into the request path:

- **Tier 1 (automatic).** When a search returns a hash-matched sub, every other sub is handed back
  with `?ref=<id>` and reference-aligned to that anchor on fetch (`ffsubsync`, no audio).
- **Tier 2 (user action).** The Den app's "Re-sync with audio" menu item (shown only for a
  non-hash-matched sub) calls the subtitle proxy with `?resync=<stream-url>`; the addon runs `alass`
  against the stream audio server-side and the app swaps in the re-synced track.

Known gap: Tier 1 needs a hash-matched anchor in the results. When the search returns *no* hash
match — the common out-of-sync case — there is no trusted reference, so Tier 1 stays off and the sub
is served as-is; only the Tier-2 resync closes it. See the ticket.

Deploy note: set `PUBLIC_BASE_URL` in production (see `.env.example`) so the `/subtitle` and
`/translate` URLs handed back to the app are pinned to the real origin rather than derived from a
client-controlled `Host` / `X-Forwarded-Host` header.
