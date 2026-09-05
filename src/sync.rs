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
        // If the SECOND write fails, `finish` — the only cleanup — is never reached, and the first
        // file stays forever: the sync work dir has no sweep. Disk pressure is exactly when the
        // second write fails, so the leak compounds the condition that caused it.
        let reference = match self.write_temp(tag, "reference.srt", reference_srt.as_bytes()).await {
            Ok(p) => p,
            Err(e) => {
                let _ = tokio::fs::remove_file(&target).await;
                return Err(e);
            }
        };
        let out = self.temp_path(tag, "synced.srt");
        // ffsubsync <reference> -i <unsynced> -o <out>. Reference-mode skips audio extraction.
        let run_result = self
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
            .await;
        self.finish(run_result, &out, [&target, &reference]).await
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
        let run_result = self
            .run(
                &self.alass,
                &[
                    media_url,
                    target.to_string_lossy().as_ref(),
                    out.to_string_lossy().as_ref(),
                ],
                TIER2_BUDGET,
            )
            .await;
        self.finish(run_result, &out, [&target]).await
    }

    /// Reclaim scratch files no run could still be using.
    ///
    /// `finish` is the only cleanup and it is skipped entirely when the future is dropped — a client
    /// disconnecting inside the Tier-1 or Tier-2 budget, a deploy, an OOM. Nothing else walks this
    /// directory: the cache sweep covers `store` only. So each cancelled alignment left its inputs
    /// behind for good, on a volume that is meant to be bounded.
    ///
    /// Age, not ownership: a file older than the longest a run may take cannot belong to a live one.
    pub fn sweep_scratch(&self) {
        /// Comfortably past `TIER2_BUDGET`, the longest any run is allowed to hold a scratch file.
        const ABANDONED: Duration = Duration::from_secs(30 * 60);

        let Ok(entries) = std::fs::read_dir(&self.work_dir) else { return };
        for entry in entries.flatten() {
            let Ok(meta) = entry.metadata() else { continue };
            if !meta.is_file() {
                continue;
            }
            // An undatable file is left alone rather than guessed at — the same call `Cache::sweep`
            // makes, and for the same reason: deleting on that signal kills live work.
            if meta.modified().ok().and_then(|m| m.elapsed().ok()).is_some_and(|age| age > ABANDONED) {
                let _ = std::fs::remove_file(entry.path());
            }
        }
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

    /// Read back the synced file on success, clean up scratch files either way, and return the SRT.
    /// Takes the raw `run` result (not just an `ExitStatus`) so a spawn/timeout failure — which
    /// never produced an exit status — still cleans up the temp inputs we already wrote.
    async fn finish<const N: usize>(
        &self,
        run_result: Result<std::process::ExitStatus, String>,
        out: &PathBuf,
        inputs: [&PathBuf; N],
    ) -> Result<String, String> {
        let result = match run_result {
            Ok(status) if status.success() => match read_capped(out).await {
                // Exit 0 is the binary's opinion, not a result. ffsubsync and alass both exit 0
                // having written nothing usable when handed a target they can't parse, and that
                // empty string was cached for 60 days as the finished alignment — worse than the
                // raw sub, which at least plays.
                Ok(body) if !crate::srt::has_a_cue(&body) => {
                    Err("sync produced no cues".to_string())
                }
                Ok(body) => Ok(body),
                Err(e) => Err(format!("read synced: {e}")),
            },
            Ok(status) => Err(format!("sync exited {status}")),
            Err(e) => Err(e),
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
        // A failed write leaves the file `create` already made; nothing else ever removes it.
        if let Err(e) = write_all_and_flush(&mut f, bytes).await {
            drop(f);
            let _ = tokio::fs::remove_file(&path).await;
            return Err(format!("write temp: {e}"));
        }
        Ok(path)
    }
}

async fn wait(mut child: tokio::process::Child) -> Result<std::process::ExitStatus, String> {
    child.wait().await.map_err(|e| format!("wait: {e}"))
}

// Exercise the process orchestration (spawn → arg contract → read back → cleanup) against fake
// shell-script binaries, so no real `ffsubsync`/`alass` is needed. Unix-only: they rely on `/bin/sh`
// and a `chmod +x` script, which is what CI (ubuntu) and the dev machine (macOS) both have.
#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn work_dir(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("den-subs-synctest-{tag}"))
    }

    // Write an executable `#!/bin/sh` script into `dir` and return its path.
    async fn fake_bin(dir: &std::path::Path, name: &str, body: &str) -> String {
        tokio::fs::create_dir_all(dir).await.unwrap();
        let path = dir.join(name);
        tokio::fs::write(&path, format!("#!/bin/sh\n{body}\n")).await.unwrap();
        let mut perms = tokio::fs::metadata(&path).await.unwrap().permissions();
        perms.set_mode(0o755);
        tokio::fs::set_permissions(&path, perms).await.unwrap();
        path.to_string_lossy().into_owned()
    }

    fn tools(dir: &std::path::Path, ffsubsync: String) -> SyncTools {
        SyncTools { ffsubsync, alass: "alass-unused".into(), work_dir: dir.to_path_buf() }
    }

    /// A real cue, because the tiers now have to return something that parses as a subtitle.
    const SUB: &str = "1\n00:00:01,000 --> 00:00:02,000\nhello\n";
    const REF: &str = "1\n00:00:03,000 --> 00:00:04,000\nreference\n";

    #[tokio::test]
    async fn tier1_success_returns_synced_output_and_removes_temps() {
        let dir = work_dir("t1-ok");
        // ffsubsync's contract is `<reference> -i <target> -o <out>`; positional $3=target, $5=out.
        // The fake "aligns" by copying the target through, so we can assert the round-trip.
        let bin = fake_bin(&dir, "fake-ffsubsync", r#"cp "$3" "$5""#).await;
        let out = tools(&dir, bin).sync_to_reference(SUB, REF, "tag-ok").await;
        assert_eq!(out.unwrap(), SUB);
        // Inputs and the output scratch file are all cleaned up on success.
        assert!(!dir.join("tag-ok-target.srt").exists());
        assert!(!dir.join("tag-ok-reference.srt").exists());
        assert!(!dir.join("tag-ok-synced.srt").exists());
        tokio::fs::remove_dir_all(&dir).await.ok();
    }

    #[tokio::test]
    async fn tier1_nonzero_exit_errors_and_removes_input_temps() {
        let dir = work_dir("t1-fail");
        let bin = fake_bin(&dir, "fake-ffsubsync", "exit 1").await;
        let out = tools(&dir, bin).sync_to_reference(SUB, REF, "tag-fail").await;
        assert!(out.is_err());
        // A failed alignment must not leave its inputs behind for the next caller to trip over.
        assert!(!dir.join("tag-fail-target.srt").exists());
        assert!(!dir.join("tag-fail-reference.srt").exists());
        tokio::fs::remove_dir_all(&dir).await.ok();
    }

    /// Exit 0 is the binary's opinion, not a result. Both tiers exit 0 having written nothing
    /// usable when the target won't parse, and that empty output was cached for 60 days as the
    /// finished alignment — an empty subtitle track, where the raw sub would at least have played.
    #[tokio::test]
    async fn a_zero_exit_with_no_cues_is_not_an_alignment() {
        for (name, body) in [("empty", r#": > "$5""#), ("html", r#"echo '<html>nope</html>' > "$5""#)] {
            let dir = work_dir(&format!("t1-junk-{name}"));
            let bin = fake_bin(&dir, "fake-ffsubsync", body).await;
            let out = tools(&dir, bin).sync_to_reference(SUB, REF, "tag-junk").await;
            assert!(out.is_err(), "{name} output was accepted as an alignment: {out:?}");
            assert!(!dir.join("tag-junk-target.srt").exists(), "{name} leaked its inputs");
            tokio::fs::remove_dir_all(&dir).await.ok();
        }
    }

    /// The second write failing skips `finish`, the only cleanup — and the sync work dir has no
    /// sweep, so the first file stays forever. Disk pressure is exactly when a second write fails,
    /// so the leak feeds the condition that caused it.
    /// The tier binary's output goes straight into the cache, and a runaway alignment has no other
    /// ceiling on it — every other body this addon ingests is capped at MAX_BODY.
    #[tokio::test]
    async fn an_oversized_alignment_is_refused() {
        let dir = work_dir("t1-huge");
        // Write one byte past the cap to $5 and exit 0.
        // A REAL cue first, then filler past the cap — so it clears the "has any cues" gate and the
        // size cap is the only thing that can refuse it.
        let script = format!(
            "printf '1\\n00:00:01,000 --> 00:00:02,000\\nhi\\n' > \"$5\"; head -c {} /dev/zero | tr '\\0' 'a' >> \"$5\"",
            crate::fetch::MAX_BODY
        );
        let bin = fake_bin(&dir, "fake-ffsubsync", &script).await;
        let out = tools(&dir, bin).sync_to_reference(SUB, REF, "tag-huge").await;
        let err = out.expect_err("an oversized alignment was accepted");
        assert!(err.contains("too large"), "refused for the wrong reason: {err}");
        assert!(!dir.join("tag-huge-target.srt").exists(), "inputs leaked");
        tokio::fs::remove_dir_all(&dir).await.ok();
    }

    #[tokio::test]
    async fn a_failed_second_write_does_not_leak_the_first() {
        let dir = work_dir("t1-write-fail");
        tokio::fs::create_dir_all(&dir).await.unwrap();
        // A directory where the reference file needs to be: `File::create` on it fails.
        tokio::fs::create_dir_all(dir.join("tag-wf-reference.srt")).await.unwrap();

        let out = tools(&dir, "unused".into()).sync_to_reference(SUB, REF, "tag-wf").await;
        assert!(out.is_err(), "the write should have failed");
        assert!(!dir.join("tag-wf-target.srt").exists(), "the first temp was left behind");
        tokio::fs::remove_dir_all(&dir).await.ok();
    }

    #[tokio::test]
    async fn missing_binary_errors_and_still_removes_input_temps() {
        let dir = work_dir("t1-nobin");
        let out = tools(&dir, "/nonexistent/xyzzy-ffsubsync".into())
            .sync_to_reference(SUB, REF, "tag-nobin")
            .await;
        assert!(out.is_err());
        // A spawn failure produced no exit status, but the temp inputs written before the spawn must
        // not leak — every request under a misconfigured binary path would otherwise pile them up.
        assert!(!dir.join("tag-nobin-target.srt").exists());
        assert!(!dir.join("tag-nobin-reference.srt").exists());
        tokio::fs::remove_dir_all(&dir).await.ok();
    }

    /// A cancelled alignment skips `finish`, the only cleanup, and nothing else walks this
    /// directory — the cache sweep covers `store` only. Each one leaked its inputs permanently.
    #[test]
    fn the_scratch_sweep_reclaims_what_a_cancelled_run_left() {
        let dir = work_dir("scratch-sweep");
        std::fs::create_dir_all(&dir).unwrap();
        let tools = SyncTools { ffsubsync: "x".into(), alass: "y".into(), work_dir: dir.clone() };

        let abandoned = dir.join("old-tag-target.srt");
        let in_flight = dir.join("live-tag-target.srt");
        std::fs::write(&abandoned, "x").unwrap();
        std::fs::write(&in_flight, "x").unwrap();
        // Backdate one past the longest a run may hold a file.
        let long_ago = std::time::SystemTime::now() - Duration::from_secs(60 * 60);
        let f = std::fs::File::options().write(true).open(&abandoned).unwrap();
        f.set_modified(long_ago).unwrap();
        drop(f);

        tools.sweep_scratch();
        assert!(!abandoned.exists(), "an abandoned scratch file was left behind");
        assert!(in_flight.exists(), "the sweep deleted a file a live run could still be using");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn tier2_audio_success_returns_synced_output() {
        let dir = work_dir("t2-ok");
        // alass's contract is `<media> <target> <out>`; positional $2=target, $3=out.
        let bin = fake_bin(&dir, "fake-alass", r#"cp "$2" "$3""#).await;
        let mut t = tools(&dir, "ffsubsync-unused".into());
        t.alass = bin;
        let out = t.sync_to_audio(SUB, "http://192.168.1.9/s.mkv", "tag-t2").await;
        assert_eq!(out.unwrap(), SUB);
        assert!(!dir.join("tag-t2-target.srt").exists());
        assert!(!dir.join("tag-t2-synced.srt").exists());
        tokio::fs::remove_dir_all(&dir).await.ok();
    }
}

/// Read a tier binary's output, bounded like every other body this addon ingests. It goes straight
/// into the cache, and a runaway alignment has no other ceiling on it.
async fn read_capped(path: &PathBuf) -> Result<String, String> {
    use tokio::io::AsyncReadExt;
    let file = tokio::fs::File::open(path).await.map_err(|e| format!("read synced: {e}"))?;
    let mut buf = Vec::new();
    file.take(crate::fetch::MAX_BODY as u64 + 1)
        .read_to_end(&mut buf)
        .await
        .map_err(|e| format!("read synced: {e}"))?;
    if buf.len() > crate::fetch::MAX_BODY {
        return Err("sync output too large".to_string());
    }
    String::from_utf8(buf).map_err(|_| "sync output is not utf-8".to_string())
}

/// tokio's File buffers: `write_all` returns Ok and stashes the real error for the next write or
/// flush, so without the flush an ENOSPC handed the tier binary an empty file and called it a
/// success.
async fn write_all_and_flush(f: &mut tokio::fs::File, bytes: &[u8]) -> std::io::Result<()> {
    f.write_all(bytes).await?;
    f.flush().await
}
