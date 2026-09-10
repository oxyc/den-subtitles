# den-subtitles — Rust binary + the subtitle-sync toolchain (alass, ffsubsync, ffmpeg), one image.
#
# Unlike den-scout (pure net/http → distroless-static), this addon shells out to sync binaries, so
# the runtime is debian-slim carrying them — the den-reel shape. alass is a static Rust binary we
# build in a side stage; ffsubsync is a pip install; ffmpeg supplies the audio decode both use.

# ---- build the Rust binary -------------------------------------------------
FROM rust:1-bookworm AS build
WORKDIR /src
# Cache deps: build manifests + a dummy main first so a code-only change re-links only our crate.
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs && cargo build --release --locked && rm -rf src
COPY src ./src
RUN touch src/main.rs && cargo build --release --locked

# ---- build alass (static Rust CLI) ----------------------------------------
FROM rust:1-bookworm AS alass
# Pinned, like ffsubsync below: unpinned, every rebuild shipped whatever the registry served that day,
# and nothing tells you when an aligner's behaviour changed. These are the versions the image ran when
# they were pinned; bump them deliberately.
ARG ALASS_VERSION=2.0.0
RUN cargo install alass-cli --version ${ALASS_VERSION} --root /alass --locked

# ---- runtime --------------------------------------------------------------
FROM debian:bookworm-slim
# ffmpeg (audio decode for alass/ffsubsync) + python for ffsubsync + ca-certs for outbound TLS.
ARG FFSUBSYNC_VERSION=0.5.1
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg python3 python3-pip ca-certificates \
    && pip3 install --no-cache-dir --break-system-packages ffsubsync==${FFSUBSYNC_VERSION} \
    && apt-get purge -y python3-pip && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY --from=alass /alass/bin/alass-cli /usr/local/bin/alass
COPY --from=build /src/target/release/den-subtitles /usr/local/bin/den-subtitles

WORKDIR /app
ENV PORT=8093 \
    CACHE_DIR=/cache \
    ALASS_PATH=/usr/local/bin/alass \
    FFSUBSYNC_PATH=/usr/local/bin/ffsubsync
VOLUME ["/cache"]
EXPOSE 8093

# The binary self-checks (no curl on the health path). start-period covers cold start.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s CMD ["den-subtitles", "healthcheck"]
CMD ["den-subtitles"]
