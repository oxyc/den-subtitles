//! Auto-sync ladder. Out-of-sync subtitles are the #1 complaint; we fix them cheapest-first:
//!
//!   * Tier 0 — hash match: nothing to do here. A subtitle that OpenSubtitles returned with
//!     `moviehash_match` was authored against this exact encode, so it is already in sync. Selection
//!     (see `opensubtitles.rs`) floats those to the top; the sync tiers below only run for the rest.
//!   * Tier 1 — reference alignment (fast, no audio): align the target subtitle against a subtitle
//!     we trust to be in sync (e.g. an English hash-match). Sub-second, runs on every result.
//!   * Tier 2 — audio VAD (robust, opt-in): align against the actual audio with `alass`
//!     (splits-aware — handles ad breaks / different cuts). Costs a stream fetch, so it's a per-title
//!     user action, not automatic.
//!
//! Each tier shells out to a binary (`ffsubsync`, `alass`) exactly like reel drives yt-dlp/ffmpeg —
//! the CPU work lives in the subprocess, not this runtime.
//!
//! Wired into the subtitle proxy: a `?ref=<file_id>` on `/subtitle/…` runs Tier 1 against that
//! (hash-matched) reference; a `?resync=<stream-url>` runs Tier 2 against the audio.

use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::timeout;

/// Binary paths + work dir, from env. Defaults assume the binaries are on PATH (the container
/// installs them).
pub struct SyncTools {
    pub ffsubsync: String,
    pub alass: String,
    pub work_dir: PathBuf,
}

const TIER1_BUDGET: Duration = Duration::from_secs(20);
const TIER2_BUDGET: Duration = Duration::from_secs(90);

impl SyncTools {
    /// Tier 1: shift `target_srt` to line up with `reference_srt` (both SRT text). No audio needed,
    /// so this is safe to run on every result when a trusted reference exists.
    pub async fn sync_to_reference(
        &self,
        target_srt: &str,
        reference_srt: &str,
        tag: &str,
    ) -> Result<String, String> {
        let target = self.write_temp(tag, "target.srt", target_srt.as_bytes()).await?;
        let reference = self.write_temp(tag, "reference.srt", reference_srt.as_bytes()).await?;
        let out = self.temp_path(tag, "synced.srt");
        // ffsubsync <reference> -i <unsynced> -o <out>. Reference-mode skips audio extraction.
        let status = self
            .run(
                &self.ffsubsync,
                &[
                    reference.to_string_lossy().as_ref(),
                    "-i",
                    target.to_string_lossy().as_ref(),
                    "-o",
                    out.to_string_lossy().as_ref(),
                ],
                TIER1_BUDGET,
            )
            .await?;
        self.finish(status, &out, [&target, &reference]).await
    }

    /// Tier 2: align `target_srt` against the media at `media_url` (a stream the addon can reach)
    /// using alass. `alass` pulls/decodes the audio itself via ffmpeg, so we hand it the URL.
    pub async fn sync_to_audio(
        &self,
        target_srt: &str,
        media_url: &str,
        tag: &str,
    ) -> Result<String, String> {
        let target = self.write_temp(tag, "target.srt", target_srt.as_bytes()).await?;
        let out = self.temp_path(tag, "synced.srt");
        // alass <reference-media> <incorrect-subs> <output>. It runs cropdetect-free VAD + a
        // split-aware DP alignment, correcting constant offset AND mid-file drift.
        let status = self
            .run(
                &self.alass,
                &[
                    media_url,
                    target.to_string_lossy().as_ref(),
                    out.to_string_lossy().as_ref(),
                ],
                TIER2_BUDGET,
            )
            .await?;
        self.finish(status, &out, [&target]).await
    }

    async fn run(&self, bin: &str, args: &[&str], budget: Duration) -> Result<std::process::ExitStatus, String> {
        let child = Command::new(bin)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true) // a timed-out/cancelled request kills the subprocess with the task
            .spawn()
            .map_err(|e| format!("spawn {bin}: {e}"))?;
        match timeout(budget, wait(child)).await {
            Ok(Ok(status)) => Ok(status),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(format!("{bin} timed out")),
        }
    }

    /// Read back the synced file on success, clean up inputs, and return the SRT.
    async fn finish<const N: usize>(
        &self,
        status: std::process::ExitStatus,
        out: &PathBuf,
        inputs: [&PathBuf; N],
    ) -> Result<String, String> {
        let result = if status.success() {
            tokio::fs::read_to_string(out).await.map_err(|e| format!("read synced: {e}"))
        } else {
            Err(format!("sync exited {status}"))
        };
        for p in inputs {
            let _ = tokio::fs::remove_file(p).await;
        }
        let _ = tokio::fs::remove_file(out).await;
        result
    }

    fn temp_path(&self, tag: &str, name: &str) -> PathBuf {
        self.work_dir.join(format!("{tag}-{name}"))
    }

    async fn write_temp(&self, tag: &str, name: &str, bytes: &[u8]) -> Result<PathBuf, String> {
        tokio::fs::create_dir_all(&self.work_dir)
            .await
            .map_err(|e| format!("mkdir work: {e}"))?;
        let path = self.temp_path(tag, name);
        let mut f = tokio::fs::File::create(&path).await.map_err(|e| format!("create temp: {e}"))?;
        f.write_all(bytes).await.map_err(|e| format!("write temp: {e}"))?;
        Ok(path)
    }
}

async fn wait(mut child: tokio::process::Child) -> Result<std::process::ExitStatus, String> {
    child.wait().await.map_err(|e| format!("wait: {e}"))
}
