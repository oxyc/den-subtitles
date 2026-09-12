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
Den (app UX)   ──/<config>/translate/movie/tt…/<extra>/Sw.json►  addon  fetch source → LLM → SRT → sync
               ◄──── { url:/translate/…/Swedish.srt }─────────┘         cached per title+lang+model
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
- **A translation goes through the sync ladder too.** It is an SRT carrying its source's timings, so
  it inherits whatever offset that source had. The translated *text* is cached against the **title**
  — one LLM bill however many encodes of the film you watch — and the per-encode alignment hangs off
  that entry the same way `?ref=` variants hang off a downloaded sub. Pass the `<extra>` blob for
  this: it is what makes a hash-matched anchor findable.

  Keyed by title rather than by the source file it was translated from, which is the correction of
  the obvious design: a per-title pin already guarantees one film translates from one source, so
  source-keying bought nothing, and it meant that if the pin ever moved — a source going dead while
  a second language was being added — every language already paid for became unreachable and was
  re-bought at full price from a cache that still held it. See `translate_body_key` in
  `src/addon.rs` for the full argument.
- **The translation harness** (`src/translate.rs`) is the crown jewel. The model never sees the
  whole file or any timestamp; cues go out in small JSON-array batches under a strict same-length
  contract; a length mismatch splits the batch and retries down to a single cue; a rolling window of
  prior (source→translation) pairs keeps names/tone consistent across the film.

## Status

Working: manifest + `/configure`, OpenSubtitles hash-matched search, cached subtitle proxy, the full
BYOK translation harness (all providers), the cache, the Docker image, and the **sync ladder**
(`src/sync.rs`) — now wired into the request path:

- **Tier 1 (automatic).** When a search returns a hash-matched sub, every other sub is handed back
  with `?ref=<id>` and reference-aligned to that anchor on fetch (`ffsubsync`, no audio).
- **Tier 2 (user action).** The Den app's "Re-sync with audio" menu item (shown only for a
  non-hash-matched sub) calls the subtitle proxy with `?resync=<stream-url>`; the addon runs `alass`
  against the stream audio server-side and the app swaps in the re-synced track. The stream URL has
  to be one of den-scout's play routes — `<origin>/<config>/play/<token>`, or the ticket form
  `<origin>/p/<ticket>` (one base64url segment) — at an origin listed in
  `SCOUT_ORIGINS`; any other target, or any target when that is unset, is ignored and the sub is
  served unaligned. alass never gets the URL itself: it reads a relay on 127.0.0.1 that follows
  scout's redirect to the debrid CDN, refuses a hop that is neither a listed origin nor a public
  address, pins the address it checked, and never passes a redirect on to ffmpeg.

Known gap: Tier 1 needs a hash-matched anchor in the results. When the search returns *no* hash
match — the common out-of-sync case — there is no trusted reference, so Tier 1 stays off and the sub
is served as-is; only the Tier-2 resync closes it. See the ticket.

## Routes

Every reply carries `Access-Control-Allow-Origin: *`, and `OPTIONS` on any path answers a 204 CORS
preflight. A path the router does not serve answers 404 `{"error":"not_found"}`. `<config>` is the
per-install config segment built at `/configure`.

The subtitles, subtitle and translate responses carry `Server-Timing`, naming where the time went —
`opensubtitles;dur=`, `download;dur=`, `sync;dur=` (when a sync tier ran), `translate;dur=`, or
`cache;desc=hit` — and `total;dur=`, in milliseconds. An answer that is a fallback carries
`X-Den-Degraded: <reason>`: `upstream_unavailable` for a subtitles list left empty because the
OpenSubtitles search failed (or a translation served unaligned because its anchor search failed), and
`sync_failed` for a subtitle served unaligned because its sync failed. A normal answer carries neither
reason.

- `GET /health` — liveness, always 200: `{"status":"ok"}`, or `{"status":"degraded",…}` with reason
  `upstream_unavailable` after three OpenSubtitles failures in a row.
- `GET /metrics` — Prometheus text for `Authorization: Bearer <METRICS_TOKEN>`; the unknown-path 404
  when the token is unset or wrong. It publishes what the addon already keeps — `subtitles_build_info`,
  the OpenSubtitles failure streak behind `/health`, the memory cache's bytes and entries, whether the
  disk tier is on and how many of its writes failed, and the sync jobs and translations running now —
  computed per scrape, with nothing per-install in the labels.
- `GET /`, `GET /configure` — the install page that builds (and, with `CONFIG_KEY` set, seals) the
  config segment.
- `GET /config-key` — `{"key":"<base64 X25519 public key>","epoch":<CONFIG_EPOCH>}` for `/configure`
  to seal to; 404 `{"error":"no_key","epoch":<CONFIG_EPOCH>}` when sealing is off.
- `GET /manifest.json` — the unconfigured manifest (`configurationRequired`), what a client sees
  before installing.
- `GET /<config>/manifest.json` — the configured manifest, carrying the install's id as the top-level
  `denInstallId` (absent for a link without one); 400 `{"error":"bad_config"}` for a segment
  that does not decode, or whose install is revoked (see Configuration). Every `/<config>/…` route
  gives a revoked install that same answer.
- `GET /<config>/subtitles/<type>/<id>[/<extra>].json` — the Stremio subtitles resource:
  hash-matched-first OpenSubtitles results, each `url` pointing at `/subtitle` below.
- `GET /<config>/subtitle/<file_id>.srt` (or `.vtt`) — one subtitle, proxied and cached; `.vtt` is
  the same document as WebVTT. `?ref=<file_id>` reference-aligns it to a hash-matched anchor (Tier 1);
  `?resync=<stream-url>` aligns it to the stream's audio with `alass` (Tier 2). Whatever encoding the
  upload is in (Windows-125x, ISO-8859-x, UTF-16), it is served as UTF-8; `lang=<code>`, set on the
  URLs the subtitles resource hands back for languages with their own legacy encodings, hints that
  detection.
- `GET /<config>/translate/<type>/<id>[/<extra>]/<lang>.json` — runs (or finds cached) the
  translation and answers `{"url":"…/<lang>.srt"}`.
- `GET /<config>/translate/<type>/<id>[/<extra>]/<lang>.srt` (or `.vtt`) — the translated subtitle,
  through the same sync ladder; `?resync=<stream-url>` applies here too.
- `GET /<config>/translate/<type>/<id>[/<extra>]/<lang>.status` — how far a running translation has
  got (`working`/`idle`/`done`/`failed`), answered without any upstream call.

## Configuration

Per-install config is base64url-encoded in the addon URL (den-scout / Torrentio style), a bearer
secret the app stores in the Keychain. The **OpenSubtitles key** (subtitle source) is required; the
**LLM key** (translation) is **optional** — omit it for a fetch + auto-sync-only install with no AI.
Build it at `/configure`.

Every link `/configure` builds also carries an install id (`iid`, 16 random bytes) and the config
epoch it was built in (`ep`, from `/config-key`), sealed along with the keys. `REVOKED_INSTALLS`
refuses one install by its id; raising `CONFIG_EPOCH` refuses every link stamped below it (a link
from before ids existed counts as epoch 0), and new links pick up the new epoch. The config is in
every `/subtitle` and `/translate` URL the addon hands out, so a revocation reaches those too. **Key
rotation is not revocation:** `CONFIG_KEYS_PREV` keeps links sealed to an old key opening, so
rotating `CONFIG_KEY` does nothing to a leaked link — revoke it instead. A refused install is logged
as `bad_config: install revoked (iid=<first 6 chars>…)` or `bad_config: install epoch too old`.

A link's id is shown in two places, spelled as `REVOKED_INSTALLS` takes it: `/configure` prints it
under the link it just built, and the install's manifest carries it as `denInstallId`, so a client
holding the link can show it (Den: Settings › Plugins). That is how to revoke one leaked link without
raising the epoch for all of them.

Supported providers: OpenAI, Google, Anthropic, xAI, OpenRouter (chat) and DeepL (MT). Default model
is the cheap/fast/decent tier per provider; step up to a bigger model to re-translate a title that
reads badly (the cache is keyed by provider+model, so it just overwrites).

DeepL API Free and Pro keys both work; the key's `:fx` suffix selects the Free endpoint. Paid
translations retain their original source file ID with the SRT, so replacing a title's source for a
new language cannot make an existing translation inherit the replacement's timing. Older cached
SRTs remain usable and are aligned conservatively when a hash-matched reference is available.
New translation cache entries store a JSON envelope containing both source and SRT. Upgrades read
existing plain-SRT entries; downgrades need a pre-upgrade cache snapshot because older binaries do
not understand the new envelopes.

No user credential lives in the environment. The environment is addon infrastructure only, all of
it optional (`.env.example` lists the same):

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8093` | HTTP listen port. |
| `CACHE_DIR` | `$TMPDIR/den-subtitles-cache` (image: `/cache`) | Disk cache tier and the sync scratch dir. |
| `CACHE_MAX_BYTES` | `268435456` (256 MiB) | Cache byte budget. |
| `PUBLIC_BASE_URL` | unset (derived from `Host`) | Fixed origin for the `/subtitle` and `/translate` URLs handed back to the app. |
| `CONFIG_KEY` | unset (sealing off) | Base64 X25519 private key `/configure` seals configs to; back it up. |
| `CONFIG_KEYS_PREV` | unset | Comma-separated prior keys, so a rotation keeps old installs working — which is also why rotation is not revocation. |
| `REVOKED_INSTALLS` | unset | Comma-separated install ids (`iid`, 22 characters) refused outright. A malformed entry is skipped with a warning. |
| `CONFIG_EPOCH` | `0` | Links stamped with an `ep` below this are refused; raise it to revoke every existing link without rotating `CONFIG_KEY`. `/config-key` hands it to `/configure` for new links. Not a non-negative integer → warned and enforced as `0`. |
| `METRICS_TOKEN` | unset (`/metrics` 404s) | Bearer token for `/metrics`. |
| `LOG_REQUESTS` | unset (off; `0` is off too) | One stderr line per request, `<METHOD> <path> <status> <ms>ms`, with the config segment as `<config>` and no query string. |
| `SCOUT_ORIGINS` | unset (Tier 2 off) | Comma-separated den-scout origins (`http://192.168.86.193:8080`) a `?resync=` target may be at — the origin of the stream URLs scout hands the app. |
| `SCOUT_ALIASES` | unset | `<public origin>=<LAN origin>` pairs (`https://d-play.oxy.fi=http://192.168.86.193:8080`): a resync target on scout's public name is fetched at its LAN address, so two services on one box never go through the WAN and the tunnel. Both sides must also be in `SCOUT_ORIGINS`. |
| `ALASS_PATH` | `alass` (image: `/usr/local/bin/alass`) | The `alass` binary for Tier-2 audio sync. |
| `FFSUBSYNC_PATH` | `ffsubsync` (image: `/usr/local/bin/ffsubsync`) | The `ffsubsync` binary for Tier-1 reference sync. |

On a trusted LAN the origin derived from the request's `Host` / `X-Forwarded-Host` header is fine —
leave `PUBLIC_BASE_URL` unset. Once the addon is reachable by untrusted clients (i.e. exposed
publicly), set it to the real origin: a client controls its own `Host` header, so an unset origin
lets a forged header steer those URLs at an attacker's server.

## Run

```sh
cp .env.example .env          # infra only — keys are entered at /configure
cargo run                     # local (needs alass/ffsubsync on PATH for the sync tiers)
# or
docker build -t den-subtitles . && docker run -p 8093:8093 --env-file .env den-subtitles

cargo test --locked           # what CI runs, with cargo fmt --check and clippy -D warnings
OPENSUBTITLES_KEY=… ./scripts/smoke.sh   # live smoke test against the real OpenSubtitles API
```

`cargo test` covers the SRT round-trip, subtitle encoding detection, config decode/validate, the JSON-array parse, the
OpenSubtitles result ordering, the Tier-1 reference selection, the `?resync=` target allowlist and
its relay (a redirect to an internal address is never followed), the
router's status codes and headers, and the sync subprocess orchestration (spawn → arg contract →
read-back → cleanup, against fake binaries).

## Deploy

The live deploy is Podman Quadlet on the homelab box, from the den repo's `deploy/` — its
`deploy/README.md` covers the stack mechanics. The unit, `den-subtitles.container`, publishes LAN host
port 8093, drops every capability, sets no-new-privileges, caps memory at 512 MiB, runs as uid 65532
(the image's non-root user), and bind-mounts the cache from `/var/lib/den/subtitles-cache` (owned by
that uid). Updates go through the health-gated `den-update` script, which proves a new image answers
`/health` and `/manifest.json` before pinning its digest; the image carries no HEALTHCHECK, so
nothing probes an idle box.

**Release images.** `docker-publish` builds on a `v*` tag, and again every Monday: the weekly run
rebuilds the newest `v*` tag (never `main`) with the base images re-pulled, no build cache and a fresh
`apt-get upgrade`, and publishes it as `:X.Y.Z-patch.<date>.<run>` and `:latest`. That is how a Debian
security fix — ffmpeg decodes untrusted media here — reaches the box between releases, through
`den-update`'s probe and rollback like any release. Trivy scans each image before `:latest` moves: a
CRITICAL with a fix available fails the run (on the weekly rebuild only in OS packages, the part a
rebuild can fix), and fixable HIGH and CRITICAL findings go to code scanning. A finding that does not
apply goes in `.trivyignore` with a reason. Every image carries SLSA provenance and an SBOM and is signed
keylessly with cosign; verify a digest with:

```sh
cosign verify \
  --certificate-identity-regexp '^https://github\.com/oxyc/den-subtitles/\.github/workflows/docker-publish\.yml@refs/(heads/main|tags/v[0-9]+\.[0-9]+\.[0-9]+)$' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  ghcr.io/oxyc/den-subtitles@sha256:<digest>
```
