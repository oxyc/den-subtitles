//! Auto-sync ladder. Out-of-sync subtitles are the #1 complaint; we fix them cheapest-first:
//!
//!   * Tier 0 — hash match: nothing to do here. A subtitle that OpenSubtitles returned with
//!     `moviehash_match` was authored against this exact encode, so it is already in sync. Selection
//!     (see `opensubtitles.rs`) floats those to the top; the sync tiers below only run for the rest.
//!   * Tier 1 — reference alignment (fast, no audio): align the target subtitle against a subtitle
//!     we trust to be in sync (e.g. an English hash-match). Sub-second, runs on every result.
//!   * Tier 2 — partial audio VAD (opt-in): extract an 80-second batch, align its two independent
//!     40-second probes, and stop when they agree; a distant 160-second fallback brings difficult
//!     tracks to four minutes of total evidence.
//!     Costs a stream fetch, so it's a per-title user action, not automatic.
//!
//! Each tier shells out to `alass` exactly like reel drives yt-dlp/ffmpeg —
//! the CPU work lives in the subprocess, not this runtime.
//!
//! Wired into the subtitle proxy: a `?ref=<file_id>` on `/subtitle/…` runs Tier 1 against that
//! (hash-matched) reference; a `?resync=<stream-url>` runs Tier 2 against the audio.

use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::{timeout, Instant};

/// Binary paths + work dir, from env. Defaults assume the binaries are on PATH (the container
/// installs them).
pub struct SyncTools {
    pub alass: String,
    pub ffmpeg: String,
    pub work_dir: PathBuf,
}

/// A Tier-2 result carries both the shifted subtitle and the shift alass measured. The body already
/// contains this correction; callers expose the number as metadata so clients can display it without
/// applying it a second time.
pub struct AudioSync {
    pub body: String,
    pub shift_ms: i64,
}

const TIER1_BUDGET: Duration = Duration::from_secs(20);
// Stay below the Apple client's 105-second held-answer timeout, including relay setup and response.
const TIER2_TOTAL_BUDGET: Duration = Duration::from_secs(90);
const TIER2_FAST_EXTRACT_BUDGET: Duration = Duration::from_secs(28);
const TIER2_FALLBACK_EXTRACT_BUDGET: Duration = Duration::from_secs(52);
const TIER2_ALIGN_BUDGET: Duration = Duration::from_secs(5);
// A contiguous four-minute window meant reading roughly 500 MiB from a 6 GiB, 43-minute Blu-ray.
// The fast batch is one third of the four-minute ceiling. Only disagreement fetches the remaining
// two thirds from a distant part of the episode.
const TIER2_FAST_PROBE_SECONDS: u64 = 40;
const TIER2_FALLBACK_PROBE_SECONDS: u64 = 80;
const TIER2_MIN_PROBE_CUES: usize = 5;
const TIER2_MAX_SHIFT_MS: i64 = 120_000;
const TIER2_SAMPLE_LEAD_MS: u64 = 30_000;
const TIER2_SHIFT_AGREEMENT_MS: i64 = 250;

struct AudioBatch {
    start_ms: u64,
    seconds: u64,
    probes: [Vec<crate::srt::Cue>; 2],
}

impl SyncTools {
    /// Tier 1: shift `target_srt` to line up with `reference_srt` (both SRT text). No audio needed,
    /// so this is safe to run on every result when a trusted reference exists.
    pub async fn sync_to_reference(
        &self,
        target_srt: &str,
        reference_srt: &str,
        tag: &str,
    ) -> Result<String, String> {
        // OpenSubtitles files are valid enough for our deliberately tolerant parser, but can still
        // carry exporter prose, padded separators, missing indices, or a BOM that alass rejects.
        // Canonicalize at the process boundary; this preserves cue text/timing while making the
        // aligner's accepted input exactly the same set the service itself accepts.
        let target_srt = canonical_srt(target_srt, "target")?;
        let reference_srt = canonical_srt(reference_srt, "reference")?;
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
        // `--no-split` is the constant-offset mode: Tier 1's reference is already trusted to match
        // this exact encode, so split detection buys no correctness and triples peak memory. Keep it
        // before the positional arguments so the CLI contract is unambiguous in tests and logs.
        let run_result = self
            .run(
                &self.alass,
                &[
                    "--no-split",
                    reference.to_string_lossy().as_ref(),
                    target.to_string_lossy().as_ref(),
                    out.to_string_lossy().as_ref(),
                ],
                TIER1_BUDGET,
            )
            .await;
        self.finish(run_result, &out, [&target, &reference]).await
    }

    /// Tier 2: extract a short audio window from `media_url`, align the matching target cues to it,
    /// and apply the validated constant offset to the whole subtitle. The request path passes a
    /// `resync::Relay` on loopback, never the caller's URL.
    pub async fn sync_to_audio(
        &self,
        target_srt: &str,
        media_url: &str,
        tag: &str,
    ) -> Result<AudioSync, String> {
        let all = crate::srt::parse(target_srt);
        if all.is_empty() {
            return Err("target subtitle has no cues".into());
        }
        let batches = partial_audio_batches(&all)?;
        let started = Instant::now();
        let mut shifts = Vec::new();
        let mut last_error = None;

        for (batch_index, batch) in batches.iter().enumerate() {
            let audio_name = format!("batch-{batch_index}.wav");
            let audio = self.temp_path(tag, &audio_name);
            let sample_start_seconds = format!("{:.3}", batch.start_ms as f64 / 1000.0);
            let batch_seconds = batch.seconds.to_string();
            let extract_cap =
                if batch_index == 0 { TIER2_FAST_EXTRACT_BUDGET } else { TIER2_FALLBACK_EXTRACT_BUDGET };
            let extract_budget = match remaining_stage_budget(started, extract_cap) {
                Ok(budget) => budget,
                Err(e) => {
                    remove_files([&audio]).await;
                    return Err(e);
                }
            };
            let extract = self
                .run(
                    &self.ffmpeg,
                    &[
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-nostdin",
                        "-ss",
                        &sample_start_seconds,
                        "-i",
                        media_url,
                        "-t",
                        &batch_seconds,
                        "-map",
                        "0:a:0",
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        "8000",
                        "-c:a",
                        "pcm_s16le",
                        "-y",
                        audio.to_string_lossy().as_ref(),
                    ],
                    extract_budget,
                )
                .await;
            if let Err(e) = require_success(extract, "partial audio extraction") {
                remove_files([&audio]).await;
                last_error = Some(e);
                continue;
            }

            for (probe_index, probe) in batch.probes.iter().enumerate() {
                let target_name = format!("batch-{batch_index}-probe-{probe_index}-target.srt");
                let out_name = format!("batch-{batch_index}-probe-{probe_index}-synced.srt");
                let target =
                    match self.write_temp(tag, &target_name, crate::srt::serialize(probe).as_bytes()).await {
                        Ok(target) => target,
                        Err(e) => {
                            remove_files([&audio]).await;
                            return Err(e);
                        }
                    };
                let out = self.temp_path(tag, &out_name);
                let align_budget = match remaining_stage_budget(started, TIER2_ALIGN_BUDGET) {
                    Ok(budget) => budget,
                    Err(e) => {
                        remove_files([&target, &audio, &out]).await;
                        return Err(e);
                    }
                };
                let align = self
                    .run(
                        &self.alass,
                        &[
                            "--no-split",
                            "--disable-fps-guessing",
                            audio.to_string_lossy().as_ref(),
                            target.to_string_lossy().as_ref(),
                            out.to_string_lossy().as_ref(),
                        ],
                        align_budget,
                    )
                    .await;
                if let Err(e) = require_success(align, "partial audio alignment") {
                    remove_files([&target, &out]).await;
                    last_error = Some(e);
                    continue;
                }
                let measured = read_capped(&out).await.and_then(|body| {
                    let aligned = crate::srt::parse(&body);
                    partial_audio_shift(probe, &aligned)
                });
                remove_files([&target, &out]).await;
                match measured {
                    Ok(shift) => {
                        shifts.push(shift);
                        if let Some(shift_ms) = consensus_shift(&shifts) {
                            remove_files([&audio]).await;
                            return Ok(AudioSync { body: shift_cues(&all, shift_ms), shift_ms });
                        }
                    }
                    Err(e) => last_error = Some(e),
                }
            }
            remove_files([&audio]).await;
        }

        let detail = match last_error {
            Some(error) => format!("{error}; measured offsets: {shifts:?}"),
            None => format!("measured offsets did not agree: {shifts:?}"),
        };
        Err(format!("partial audio sync could not confirm one offset across two windows: {detail}"))
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
        /// Comfortably past the Tier-2 extraction + alignment budgets, the longest a run may hold a
        /// file.
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

    async fn run(
        &self,
        bin: &str,
        args: &[&str],
        budget: Duration,
    ) -> Result<std::process::ExitStatus, String> {
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
                // Exit 0 is the binary's opinion, not a result. alass can exit 0
                // having written nothing usable when handed a target they can't parse, and that
                // empty string was cached for 60 days as the finished alignment — worse than the
                // raw sub, which at least plays.
                Ok(body) if !crate::srt::has_a_cue(&body) => Err("sync produced no cues".to_string()),
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
        tokio::fs::create_dir_all(&self.work_dir).await.map_err(|e| format!("mkdir work: {e}"))?;
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

fn require_success(run: Result<std::process::ExitStatus, String>, step: &str) -> Result<(), String> {
    match run {
        Ok(status) if status.success() => Ok(()),
        Ok(status) => Err(format!("{step} exited {status}")),
        Err(e) => Err(e),
    }
}

fn remaining_stage_budget(started: Instant, stage_cap: Duration) -> Result<Duration, String> {
    let remaining = TIER2_TOTAL_BUDGET.saturating_sub(started.elapsed());
    if remaining.is_zero() {
        Err("partial audio sync timed out".into())
    } else {
        Ok(stage_cap.min(remaining))
    }
}

fn partial_audio_batches(cues: &[crate::srt::Cue]) -> Result<Vec<AudioBatch>, String> {
    let mut ordered = cues.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|cue| cue.start);
    let fast_probe_ms = TIER2_FAST_PROBE_SECONDS * 1000;
    let fallback_probe_ms = TIER2_FALLBACK_PROBE_SECONDS * 1000;
    let mut fast = audio_batch_candidates(&ordered, fast_probe_ms);
    fast.sort_by_key(|(start, left, _, right)| (std::cmp::Reverse(right - left), *start));
    let Some(first) = fast.into_iter().next() else {
        return Err("no 80-second batch has enough cues in both 40-second probes".into());
    };

    let first_end = first.0.saturating_add(fast_probe_ms * 2);
    let fallback = audio_batch_candidates(&ordered, fallback_probe_ms)
        .into_iter()
        .filter(|(start, _, _, _)| {
            let end = start.saturating_add(fallback_probe_ms * 2);
            end <= first.0 || first_end <= *start
        })
        .max_by_key(|(start, left, _, right)| {
            (start.abs_diff(first.0) / (fallback_probe_ms * 2), right - left, std::cmp::Reverse(*start))
        });

    let mut selected = vec![(first, fast_probe_ms)];
    if let Some(fallback) = fallback {
        selected.push((fallback, fallback_probe_ms));
    }
    Ok(selected
        .into_iter()
        .map(|((start_ms, left, middle, right), probe_ms)| AudioBatch {
            start_ms,
            seconds: probe_ms * 2 / 1000,
            probes: [
                normalize_probe(&ordered[left..middle], start_ms),
                normalize_probe(&ordered[middle..right], start_ms),
            ],
        })
        .collect())
}

fn audio_batch_candidates(ordered: &[&crate::srt::Cue], probe_ms: u64) -> Vec<(u64, usize, usize, usize)> {
    let batch_ms = probe_ms * 2;
    // Find dialogue-dense candidates whose two non-overlapping halves both carry enough evidence.
    // Sort once outside, then slide all three bounds forward: O(n) for each of the two batch sizes.
    let starts =
        std::iter::once(0).chain(ordered.iter().map(|cue| cue.start.saturating_sub(TIER2_SAMPLE_LEAD_MS)));
    let (mut left, mut middle, mut right) = (0, 0, 0);
    let mut candidates = Vec::new();
    for start in starts {
        while left < ordered.len() && ordered[left].start < start {
            left += 1;
        }
        middle = middle.max(left);
        let split = start.saturating_add(probe_ms);
        while middle < ordered.len() && ordered[middle].start < split {
            middle += 1;
        }
        right = right.max(middle);
        let end = start.saturating_add(batch_ms);
        while right < ordered.len() && ordered[right].start < end {
            right += 1;
        }
        if middle - left >= TIER2_MIN_PROBE_CUES && right - middle >= TIER2_MIN_PROBE_CUES {
            candidates.push((start, left, middle, right));
        }
    }
    candidates
}

fn normalize_probe(cues: &[&crate::srt::Cue], start_ms: u64) -> Vec<crate::srt::Cue> {
    cues.iter()
        .enumerate()
        .map(|(index, cue)| {
            let mut cue = (*cue).clone();
            cue.index = index as u32 + 1;
            cue.start -= start_ms;
            cue.end = cue.end.saturating_sub(start_ms);
            cue
        })
        .collect()
}

fn consensus_shift(shifts: &[i64]) -> Option<i64> {
    let mut best = Vec::new();
    for &candidate in shifts {
        let mut cluster = shifts
            .iter()
            .copied()
            .filter(|shift| (*shift - candidate).abs() <= TIER2_SHIFT_AGREEMENT_MS)
            .collect::<Vec<_>>();
        if cluster.len() > best.len() {
            cluster.sort_unstable();
            best = cluster;
        }
    }
    (best.len() >= 2).then(|| {
        let middle = best.len() / 2;
        if best.len() % 2 == 0 {
            (best[middle - 1] + best[middle]) / 2
        } else {
            best[middle]
        }
    })
}

fn partial_audio_shift(original: &[crate::srt::Cue], aligned: &[crate::srt::Cue]) -> Result<i64, String> {
    let by_index: std::collections::HashMap<_, _> = aligned.iter().map(|cue| (cue.index, cue)).collect();
    let mut deltas: Vec<i64> = original
        .iter()
        .filter_map(|cue| {
            let shifted = by_index.get(&cue.index)?;
            // alass clamps a shifted cue at zero. That pair no longer carries the actual offset.
            (shifted.start > 0).then_some(shifted.start as i64 - cue.start as i64)
        })
        .collect();
    if deltas.len() < 3 {
        return Err("partial audio sync produced too few comparable cues".into());
    }
    deltas.sort_unstable();
    let shift = deltas[deltas.len() / 2];
    if shift.abs() > TIER2_MAX_SHIFT_MS {
        return Err(format!("partial audio sync proposed implausible shift {shift}ms"));
    }
    let agreeing = deltas.iter().filter(|delta| (**delta - shift).abs() <= 100).count();
    if agreeing * 4 < deltas.len() * 3 {
        return Err("partial audio sync did not find one stable offset".into());
    }
    Ok(shift)
}

fn shift_cues(cues: &[crate::srt::Cue], shift: i64) -> String {
    let shifted: Vec<_> = cues
        .iter()
        .cloned()
        .map(|mut cue| {
            cue.start = shift_timestamp(cue.start, shift);
            cue.end = shift_timestamp(cue.end, shift);
            cue
        })
        .collect();
    crate::srt::serialize(&shifted)
}

fn shift_timestamp(value: u64, shift: i64) -> u64 {
    if shift < 0 {
        value.saturating_sub(shift.unsigned_abs())
    } else {
        value.saturating_add(shift as u64)
    }
}

async fn remove_files<const N: usize>(paths: [&PathBuf; N]) {
    for path in paths {
        let _ = tokio::fs::remove_file(path).await;
    }
}

fn canonical_srt(body: &str, name: &str) -> Result<String, String> {
    let cues = crate::srt::parse(body);
    if cues.is_empty() {
        return Err(format!("{name} subtitle has no cues"));
    }
    Ok(crate::srt::serialize(&cues))
}

async fn wait(mut child: tokio::process::Child) -> Result<std::process::ExitStatus, String> {
    child.wait().await.map_err(|e| format!("wait: {e}"))
}

// Exercise the process orchestration (spawn → arg contract → read back → cleanup) against fake
// shell-script binaries, so no real `alass` is needed. Unix-only: they rely on `/bin/sh`
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

    fn tools(dir: &std::path::Path, alass: String) -> SyncTools {
        SyncTools { alass, ffmpeg: "ffmpeg".into(), work_dir: dir.to_path_buf() }
    }

    /// A real cue, because the tiers now have to return something that parses as a subtitle.
    const SUB: &str = "1\n00:00:01,000 --> 00:00:02,000\nhello\n";
    const REF: &str = "1\n00:00:03,000 --> 00:00:04,000\nreference\n";

    #[tokio::test]
    async fn tier1_success_returns_synced_output_and_removes_temps() {
        let dir = work_dir("t1-ok");
        // Tier 1's contract is `--no-split <reference> <target> <out>`.
        // The fake "aligns" by copying the target through, so we can assert the round-trip.
        let bin = fake_bin(&dir, "fake-alass", r#"test "$1" = --no-split && cp "$3" "$4""#).await;
        let out = tools(&dir, bin).sync_to_reference(SUB, REF, "tag-ok").await;
        assert_eq!(out.unwrap(), SUB);
        // Inputs and the output scratch file are all cleaned up on success.
        assert!(!dir.join("tag-ok-target.srt").exists());
        assert!(!dir.join("tag-ok-reference.srt").exists());
        assert!(!dir.join("tag-ok-synced.srt").exists());
        tokio::fs::remove_dir_all(&dir).await.ok();
    }

    #[tokio::test]
    async fn tier1_canonicalizes_srt_that_the_service_accepts_but_alass_does_not() {
        let dir = work_dir("t1-canonical");
        let bin = fake_bin(
            &dir,
            "fake-alass",
            r#"test "$1" = --no-split; ! grep -q 'exported by editor' "$2"; cp "$3" "$4""#,
        )
        .await;
        let malformed = "\u{feff}exported by editor\r\n\r\n1\r\n00:00:01,000 --> 00:00:02,000\r\nhello";
        let out = tools(&dir, bin).sync_to_reference(malformed, malformed, "tag-canonical").await.unwrap();
        assert_eq!(crate::srt::parse(&out), crate::srt::parse(SUB));
        tokio::fs::remove_dir_all(&dir).await.ok();
    }

    #[tokio::test]
    async fn tier1_nonzero_exit_errors_and_removes_input_temps() {
        let dir = work_dir("t1-fail");
        let bin = fake_bin(&dir, "fake-alass", "exit 1").await;
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
            let body = body.replace("$5", "$4");
            let bin = fake_bin(&dir, "fake-alass", &body).await;
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
    /// ceiling on it — every subtitle body this addon ingests has a dedicated size cap.
    #[tokio::test]
    async fn an_oversized_alignment_is_refused() {
        let dir = work_dir("t1-huge");
        // Write one byte past the cap to $5 and exit 0.
        // A REAL cue first, then filler past the cap — so it clears the "has any cues" gate and the
        // size cap is the only thing that can refuse it.
        let script = format!(
            "printf '1\\n00:00:01,000 --> 00:00:02,000\\nhi\\n' > \"$5\"; head -c {} /dev/zero | tr '\\0' 'a' >> \"$5\"",
            crate::fetch::MAX_SUBTITLE_BODY
        );
        let script = script.replace("$5", "$4");
        let bin = fake_bin(&dir, "fake-alass", &script).await;
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
        let out =
            tools(&dir, "/nonexistent/xyzzy-alass".into()).sync_to_reference(SUB, REF, "tag-nobin").await;
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
        let tools = SyncTools { alass: "y".into(), ffmpeg: "y".into(), work_dir: dir.clone() };

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
    async fn tier2_audio_accepts_two_agreeing_probes_from_one_batch() {
        let dir = work_dir("t2-ok");
        let full = (0..36)
            .map(|index| {
                let batch = index / 12;
                let half = (index % 12) / 6;
                let within = index % 6;
                let start = batch as u64 * 600_000 + 20_000 + half as u64 * 45_000 + within as u64 * 4_000;
                crate::srt::Cue { index: index + 1, start, end: start + 1_000, text: format!("cue {index}") }
            })
            .collect::<Vec<_>>();
        // Copying each normalized target reports a zero shift. Both probes in the first
        // batch agree, so the distant fallback batch must never be extracted.
        let alass = fake_bin(&dir, "fake-alass", "cp \"$4\" \"$5\"").await;
        let calls = dir.join("ffmpeg-calls");
        // ffmpeg's destination is its last argument.
        let ffmpeg = fake_bin(
            &dir,
            "fake-ffmpeg",
            &format!(
                "test \"$5\" = -ss; printf '%s\\n' \"$6\" >> '{}'; for last; do :; done; printf RIFF > \"$last\"",
                calls.to_string_lossy()
            ),
        )
        .await;
        let t = SyncTools { alass, ffmpeg, work_dir: dir.clone() };
        let out = t
            .sync_to_audio(&crate::srt::serialize(&full), "http://192.168.1.9/s.mkv", "tag-t2")
            .await
            .unwrap();
        assert_eq!(out.shift_ms, 0);
        let shifted = crate::srt::parse(&out.body);
        assert_eq!(shifted, full);
        assert_eq!(tokio::fs::read_to_string(&calls).await.unwrap().lines().count(), 1);
        assert!(!dir.join("tag-t2-batch-0-probe-0-target.srt").exists());
        assert!(!dir.join("tag-t2-batch-0.wav").exists());
        assert!(!dir.join("tag-t2-batch-0-probe-0-synced.srt").exists());
        tokio::fs::remove_dir_all(&dir).await.ok();
    }

    #[tokio::test]
    async fn tier2_audio_uses_the_distant_fallback_after_disagreement() {
        let dir = work_dir("t2-fallback");
        let full = (0..108)
            .map(|index| {
                let region = index / 36;
                let within = index % 36;
                let start = 600_000 + region as u64 * 600_000 + 10_000 + within as u64 * 4_000;
                crate::srt::Cue { index: index + 1, start, end: start + 1_000, text: format!("cue {index}") }
            })
            .collect::<Vec<_>>();
        let batches = partial_audio_batches(&full).unwrap();
        assert_eq!(batches.len(), 2, "the fixture needs a distant fallback batch");

        // First probe says 0, second says +1s, so the fast batch cannot be trusted. The first
        // distant probe says 0 again, producing a two-window consensus across different regions.
        let mut disagree = batches[0].probes[1].clone();
        for cue in &mut disagree {
            cue.start += 1_000;
            cue.end += 1_000;
        }
        let disagree_path = dir.join("disagree.srt");
        tokio::fs::create_dir_all(&dir).await.unwrap();
        tokio::fs::write(&disagree_path, crate::srt::serialize(&disagree)).await.unwrap();
        let align_calls = dir.join("align-calls");
        let alass = fake_bin(
            &dir,
            "fake-alass",
            &format!(
                "n=$(cat '{}' 2>/dev/null || echo 0); n=$((n + 1)); echo \"$n\" > '{}'; if test \"$n\" = 2; then cp '{}' \"$5\"; else cp \"$4\" \"$5\"; fi",
                align_calls.to_string_lossy(),
                align_calls.to_string_lossy(),
                disagree_path.to_string_lossy()
            ),
        )
        .await;
        let extract_calls = dir.join("extract-calls");
        let ffmpeg = fake_bin(
            &dir,
            "fake-ffmpeg",
            &format!(
                "printf x >> '{}'; for last; do :; done; printf RIFF > \"$last\"",
                extract_calls.to_string_lossy()
            ),
        )
        .await;
        let t = SyncTools { alass, ffmpeg, work_dir: dir.clone() };
        let out = t
            .sync_to_audio(&crate::srt::serialize(&full), "http://192.168.1.9/s.mkv", "tag-fallback")
            .await
            .unwrap();
        assert_eq!(out.shift_ms, 0);
        assert_eq!(tokio::fs::read_to_string(&extract_calls).await.unwrap(), "xx");
        assert_eq!(tokio::fs::read_to_string(&align_calls).await.unwrap().trim(), "3");
        tokio::fs::remove_dir_all(&dir).await.ok();
    }

    #[test]
    fn partial_audio_shift_rejects_an_implausible_or_unstable_answer() {
        let cues = (1..=5)
            .map(|index| crate::srt::Cue {
                index,
                start: index as u64 * 20_000,
                end: index as u64 * 20_000 + 1_000,
                text: "x".into(),
            })
            .collect::<Vec<_>>();
        let mut huge = cues.clone();
        for cue in &mut huge {
            cue.start += 180_000;
        }
        assert!(partial_audio_shift(&cues, &huge).unwrap_err().contains("implausible"));

        let mut unstable = cues.clone();
        for (i, cue) in unstable.iter_mut().enumerate() {
            cue.start = (cue.start as i64 + i as i64 * 1_000) as u64;
        }
        assert!(partial_audio_shift(&cues, &unstable).unwrap_err().contains("stable"));
    }

    #[test]
    fn partial_audio_batches_have_two_independent_probes_and_distant_fallback() {
        let cues = (0..108)
            .map(|index| {
                let region = index / 36;
                let within = index % 36;
                let start = 600_000 + region as u64 * 600_000 + 10_000 + within as u64 * 4_000;
                crate::srt::Cue { index: 7, start, end: start + 1_000, text: "x".into() }
            })
            .collect::<Vec<_>>();
        let batches = partial_audio_batches(&cues).unwrap();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].seconds, 80);
        assert_eq!(batches[1].seconds, 160);
        assert!(batches[0].start_ms + 80_000 <= batches[1].start_ms);
        assert!(batches[0].probes.iter().all(|probe| probe.len() >= TIER2_MIN_PROBE_CUES));
        assert!(batches[1].probes.iter().all(|probe| probe.len() >= TIER2_MIN_PROBE_CUES));
        assert!(batches[0].probes[0].iter().all(|cue| cue.start < 40_000));
        assert!(batches[0].probes[1].iter().all(|cue| cue.start >= 40_000));
        assert!(batches[1].probes[0].iter().all(|cue| cue.start < 80_000));
        assert!(batches[1].probes[1].iter().all(|cue| cue.start >= 80_000));
    }

    #[test]
    fn progressive_consensus_requires_two_close_offsets() {
        assert_eq!(consensus_shift(&[-5_880]), None);
        assert_eq!(consensus_shift(&[-5_880, -5_760]), Some(-5_820));
        assert_eq!(consensus_shift(&[-5_880, -4_000]), None);
        assert_eq!(consensus_shift(&[-5_880, -4_000, -5_800]), Some(-5_840));
    }
}

/// Read a tier binary's output, bounded like every other body this addon ingests. It goes straight
/// into the cache, and a runaway alignment has no other ceiling on it.
async fn read_capped(path: &PathBuf) -> Result<String, String> {
    use tokio::io::AsyncReadExt;
    let file = tokio::fs::File::open(path).await.map_err(|e| format!("read synced: {e}"))?;
    let mut buf = Vec::new();
    file.take(crate::fetch::MAX_SUBTITLE_BODY as u64 + 1)
        .read_to_end(&mut buf)
        .await
        .map_err(|e| format!("read synced: {e}"))?;
    if buf.len() > crate::fetch::MAX_SUBTITLE_BODY {
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
