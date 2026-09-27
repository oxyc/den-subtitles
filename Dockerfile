# den-subtitles — Rust binary + the subtitle-sync toolchain (alass, ffmpeg), one image.
#
# Unlike den-scout (pure net/http → distroless-static), this addon shells out to sync binaries, so
# the runtime is debian-slim carrying them — the den-reel shape. alass is a static Rust binary we
# build in a side stage; ffmpeg supplies its audio decode.

# ---- build the Rust binary -------------------------------------------------
FROM rust:1-trixie AS build
WORKDIR /src
# Cache deps: build manifests + a dummy main first so a code-only change re-links only our crate.
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs && cargo build --release --locked && rm -rf src
COPY src ./src
RUN touch src/main.rs && cargo build --release --locked

# ---- build alass (static Rust CLI) ----------------------------------------
FROM rust:1-trixie AS alass
# Pinned: unpinned, every rebuild shipped whatever the registry served that day,
# and nothing tells you when an aligner's behaviour changed. These are the versions the image ran when
# they were pinned; bump them deliberately.
ARG ALASS_VERSION=2.0.0
RUN cargo install alass-cli --version ${ALASS_VERSION} --root /alass --locked

# ---- parity corpus (CI-only; never copied into runtime) -------------------
FROM debian:trixie-slim AS corpus
ARG FFSUBSYNC_VERSION=0.5.1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip \
    && pip3 install --no-cache-dir --break-system-packages ffsubsync==${FFSUBSYNC_VERSION} \
    && apt-get purge -y python3-pip && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
COPY --from=alass /alass/bin/alass-cli /usr/local/bin/alass
COPY --from=build /src/target/release/den-subtitles /usr/local/bin/den-subtitles
COPY scripts/tier1-corpus.py /usr/local/bin/tier1-corpus
ENTRYPOINT ["python3", "/usr/local/bin/tier1-corpus"]

# ---- runtime --------------------------------------------------------------
FROM debian:trixie-slim
# ffmpeg supplies alass's audio decode; ca-certs supports outbound TLS. `upgrade` first: the slim
# base is refreshed only every few weeks, and ffmpeg decodes untrusted media, so every build (the weekly
# patch rebuild included) takes the current Debian security fixes rather than the base's.
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
      ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=alass /alass/bin/alass-cli /usr/local/bin/alass
COPY --from=build /src/target/release/den-subtitles /usr/local/bin/den-subtitles

# Non-root, with the uid every den addon image uses (distroless's `nonroot`, 65532), so the box chowns
# one uid for every writable host dir. /cache is created owned by it, so a fresh volume mounted there
# is writable.
RUN useradd --system --uid 65532 --user-group --create-home --home-dir /home/nonroot nonroot \
    && mkdir -p /cache && chown nonroot:nonroot /cache

WORKDIR /app
ENV PORT=8093 \
    CACHE_DIR=/cache \
    ALASS_PATH=/usr/local/bin/alass
VOLUME ["/cache"]
EXPOSE 8093

# No HEALTHCHECK, deliberately: a periodic probe keeps an idle box awake. Health is checked when it
# matters — by the deploy (den/deploy/den-update.sh), against /health and /manifest.json over HTTP.
USER 65532:65532
ENTRYPOINT ["den-subtitles"]
