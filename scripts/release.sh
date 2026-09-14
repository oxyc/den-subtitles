#!/usr/bin/env bash
# Cut a release, and refuse to cut a broken one.
#
#   scripts/release.sh 0.37.0
#
# Tagging by hand is how several releases in this fleet built nothing, and the failure hides well: CI gates on
# a formatter, a tag that fails it produces NO image, and `den-update` then reports "already at <digest>" —
# which means "the registry is unchanged" and reads as "nothing to deploy". The tag exists, the deploy command
# exits 0, and the box keeps serving the old binary.
#
# den-update cannot tell the difference, because a release that never happened and a release that broke leave
# the registry in exactly the same state. So the check belongs here: this checks first and tags second, then
# waits for the build rather than assuming it.
#
# One file, copied verbatim into every addon repo. It works out the toolchain and the repo slug itself, so
# there is nothing per-repo to keep in sync — and a script that differed per repo is how one of them would
# quietly stop checking.
set -euo pipefail

version="${1:-}"
[ -n "$version" ] || { echo "usage: scripts/release.sh <version>   e.g. 0.37.0" >&2; exit 1; }
case "$version" in
    v*) echo "error: give the version without the leading v ($version -> ${version#v})" >&2; exit 1 ;;
    *.*.*) ;;
    *) echo "error: '$version' is not a semver version" >&2; exit 1 ;;
esac

cd "$(dirname "$0")/.."
# From the remote, so nothing here names a repo.
slug="$(git remote get-url origin | sed -e 's#.*[:/]\([^/]*/[^/]*\)$#\1#' -e 's/\.git$//')"

git diff --quiet || { echo "error: working tree has uncommitted changes" >&2; exit 1; }
git fetch --quiet origin
branch="$(git symbolic-ref --quiet --short HEAD || echo HEAD)"
main="$(git symbolic-ref --quiet --short refs/remotes/origin/HEAD | sed 's#^origin/##' || echo main)"
# These repos are worked on by several sessions at once; releasing on top of a stale main is how a push gets
# rejected halfway through, with a tag already made.
git merge-base --is-ancestor "origin/$main" HEAD \
    || { echo "error: origin/$main has moved — rebase onto it before releasing" >&2; exit 1; }
git rev-parse "v$version" >/dev/null 2>&1 && { echo "error: tag v$version already exists" >&2; exit 1; }

# FORMAT THE WAY CI DOES. `cargo fmt` is unavailable on at least one machine here (no rustfmt component, no
# rustup) and fails with "no such command", which reads as "not applicable" — that reading is what cost the
# releases. nix runs the real thing against the repo's own config.
run_rustfmt() {
    if cargo fmt --all --check 2>/dev/null; then return 0; fi
    command -v nix >/dev/null 2>&1 \
        || { echo "error: no rustfmt, and no nix to borrow one" >&2; return 1; }
    # shellcheck disable=SC2046
    nix run nixpkgs#rustfmt -- --edition 2021 --check $(git ls-files '*.rs')
}

if [ -f Cargo.toml ]; then
    echo "==> formatting";  run_rustfmt
    echo "==> tests";       cargo test --quiet
    version_file="Cargo.toml"
    current="$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -1)"
elif [ -f go.mod ]; then
    echo "==> formatting"
    unformatted="$(gofmt -l .)"
    [ -z "$unformatted" ] || { echo "error: needs gofmt:"; echo "$unformatted"; exit 1; }
    echo "==> vet";   go vet ./...
    echo "==> tests"; go test ./...
    # Go carries its version in a constant rather than a manifest.
    version_file="$(git grep -l 'manifestVersion = "' -- '*.go' | head -1)"
    current="$(sed -n 's/.*manifestVersion = "\(.*\)".*/\1/p' "$version_file" | head -1)"
else
    echo "error: no Cargo.toml and no go.mod — teach this script the toolchain" >&2
    exit 1
fi

[ -n "$current" ] || { echo "error: could not read the current version" >&2; exit 1; }
[ "$current" != "$version" ] || { echo "error: already at $version" >&2; exit 1; }

echo "==> version $current -> $version"
if [ "$version_file" = "Cargo.toml" ]; then
    sed -i.bak "0,/^version = \"$current\"/s//version = \"$version\"/" Cargo.toml && rm -f Cargo.toml.bak
    cargo build --quiet   # refresh the lockfile's own entry
    git add Cargo.toml Cargo.lock
else
    sed -i.bak "s/manifestVersion = \"$current\"/manifestVersion = \"$version\"/" "$version_file"
    rm -f "$version_file.bak"
    git add "$version_file"
fi
git commit --quiet -m "${slug#*/} $version"

echo "==> pushing"
git push --quiet origin "HEAD:$main"
git tag -a "v$version" -m "${slug#*/} $version"
git push --quiet origin "v$version"

# WAIT FOR THE IMAGE. Without this the script ends exactly where the old mistake began: a pushed tag, and no
# idea whether anything was built from it.
command -v gh >/dev/null 2>&1 || {
    echo "note: no gh — check it yourself: https://github.com/$slug/actions"; exit 0; }
echo "==> waiting for docker-publish"
status=""; conclusion=""
for _ in $(seq 1 80); do
    read -r status conclusion <<<"$(gh run list --repo "$slug" --limit 1 --json status,conclusion \
        --jq '.[0] | "\(.status) \(.conclusion // "-")"')"
    [ "$status" = "completed" ] && break
    sleep 15
done
if [ "$status" != "completed" ]; then
    echo "note: still building after 20 minutes — gh run list --repo $slug"; exit 0
fi
if [ "$conclusion" != "success" ]; then
    echo "error: the build for v$version FAILED ($conclusion). No image exists, so there is nothing to" >&2
    echo "       deploy — den-update will report the box is already at the previous digest, and that is" >&2
    echo "       THIS, not a no-op. Logs: gh run view --repo $slug --log-failed" >&2
    exit 1
fi
echo "v$version built. Deploy with:"
echo "  ssh root@pve 'incus exec den -- env TUF_ROOT=/var/lib/den/sigstore den-update ${slug#*/}'"
