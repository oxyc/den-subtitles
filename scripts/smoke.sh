#!/usr/bin/env bash
# Live smoke test of the OpenSubtitles path against the real API. Free (no LLM calls).
#
#   OPENSUBTITLES_KEY=your-key ./scripts/smoke.sh          # default: Shawshank (tt0111161)
#   OPENSUBTITLES_KEY=your-key ./scripts/smoke.sh tt0068646 # any imdb id (The Godfather)
#
# Builds the binary, starts it on a spare port with a temp cache, builds a config blob carrying
# your key (apiKey is a placeholder — translation is never invoked), then exercises manifest →
# subtitles search → one download → SRT sanity. The key stays in your shell env; it is never
# printed, written to disk, or committed.
set -euo pipefail

: "${OPENSUBTITLES_KEY:?set OPENSUBTITLES_KEY in your environment}"
IMDB="${1:-tt0111161}"
PORT="${PORT:-8099}"
CACHE="$(mktemp -d)"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

b64url() { python3 -c 'import base64,sys; print(base64.urlsafe_b64encode(sys.stdin.buffer.read()).decode().rstrip("="))'; }

echo "→ building…"
( cd "$ROOT" && cargo build --quiet )

echo "→ starting den-subtitles on :$PORT"
PORT="$PORT" CACHE_DIR="$CACHE" "$ROOT/target/debug/den-subtitles" &
SRV=$!
trap 'kill "$SRV" 2>/dev/null || true; rm -rf "$CACHE"' EXIT
for _ in $(seq 1 30); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 0.2; done

CFG=$(printf '{"provider":"openai","apiKey":"smoke-placeholder","osKey":"%s"}' "$OPENSUBTITLES_KEY" | b64url)
BASE="http://127.0.0.1:$PORT/$CFG"

echo "→ configured manifest:"
curl -s "$BASE/manifest.json" | python3 -m json.tool | sed 's/^/    /'

echo "→ subtitles search for $IMDB (hash-matches sort first):"
SUBS=$(curl -s "$BASE/subtitles/movie/$IMDB.json")
echo "$SUBS" | python3 -c '
import json,sys
d=json.load(sys.stdin); subs=d.get("subtitles",[])
print(f"    {len(subs)} results")
for s in subs[:8]: print(f"    - {s[\"lang\"]:5} {s[\"id\"]:12} {s[\"url\"]}")
'

FIRST=$(echo "$SUBS" | python3 -c 'import json,sys; s=json.load(sys.stdin)["subtitles"]; print(s[0]["url"] if s else "")')
if [ -n "$FIRST" ]; then
  echo "→ downloading first subtitle (proxied + cached), first 6 lines:"
  curl -s "$FIRST" | head -n 6 | sed 's/^/    /'
else
  echo "    (no subtitles returned — try another imdb id)"
fi
echo "✓ done"
