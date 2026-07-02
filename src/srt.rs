//! SRT parse/serialize. The translation harness only ever touches `text` — indices and timestamps
//! are preserved verbatim, so the model can neither renumber a cue nor corrupt a timecode (the
//! classic "LLM ate the SRT" failure mode is structurally impossible when timing never leaves here).

/// One subtitle cue. `start`/`end` are milliseconds from the start of the media.
#[derive(Debug, Clone, PartialEq)]
pub struct Cue {
    pub index: u32,
    pub start: u64,
    pub end: u64,
    /// The dialogue, `\n`-joined if the cue spans multiple display lines.
    pub text: String,
}

/// Parse an SRT document. Tolerant of CRLF, a UTF-8 BOM, and blank runs; a malformed block is
/// skipped rather than aborting the whole file (a single bad cue shouldn't lose the movie).
pub fn parse(input: &str) -> Vec<Cue> {
    let input = input.strip_prefix('\u{feff}').unwrap_or(input);
    let mut cues = Vec::new();
    // Blocks are separated by a blank line. Normalise CRLF first so the split is uniform.
    let normalized = input.replace("\r\n", "\n").replace('\r', "\n");
    for block in normalized.split("\n\n") {
        let block = block.trim_matches('\n');
        if block.is_empty() {
            continue;
        }
        let mut lines = block.lines();
        let Some(first) = lines.next() else { continue };
        // First line is the index; some files omit it and lead with the timecode — tolerate both.
        let (index, timing_line) = match first.trim().parse::<u32>() {
            Ok(n) => (n, lines.next().unwrap_or("")),
            Err(_) => (cues.len() as u32 + 1, first),
        };
        let Some((start, end)) = parse_timing(timing_line) else { continue };
        let text = lines.collect::<Vec<_>>().join("\n");
        cues.push(Cue { index, start, end, text });
    }
    cues
}

/// Serialize cues back to a well-formed SRT (LF newlines, blank-line separated, trailing newline).
pub fn serialize(cues: &[Cue]) -> String {
    let mut out = String::new();
    for (i, c) in cues.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(&c.index.to_string());
        out.push('\n');
        out.push_str(&format_ts(c.start));
        out.push_str(" --> ");
        out.push_str(&format_ts(c.end));
        out.push('\n');
        out.push_str(&c.text);
        out.push('\n');
    }
    out
}

/// `HH:MM:SS,mmm --> HH:MM:SS,mmm` → (start_ms, end_ms). Also accepts a `.` millisecond separator
/// (VTT-style) since some sources mix them.
fn parse_timing(line: &str) -> Option<(u64, u64)> {
    let (a, b) = line.split_once("-->")?;
    Some((parse_ts(a.trim())?, parse_ts(b.trim())?))
}

fn parse_ts(s: &str) -> Option<u64> {
    // HH:MM:SS,mmm — split off millis on either separator first.
    let (hms, ms) = s.split_once([',', '.'])?;
    let mut parts = hms.split(':');
    let h: u64 = parts.next()?.parse().ok()?;
    let m: u64 = parts.next()?.parse().ok()?;
    let sec: u64 = parts.next()?.parse().ok()?;
    // Pad/truncate the millisecond field to exactly 3 digits before parsing (e.g. "5" → 500).
    let ms: u64 = format!("{:0<3}", &ms[..ms.len().min(3)]).parse().ok()?;
    Some(((h * 60 + m) * 60 + sec) * 1000 + ms)
}

fn format_ts(ms: u64) -> String {
    let (h, rem) = (ms / 3_600_000, ms % 3_600_000);
    let (m, rem) = (rem / 60_000, rem % 60_000);
    let (s, milli) = (rem / 1000, rem % 1000);
    format!("{h:02}:{m:02}:{s:02},{milli:03}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_a_basic_cue() {
        let src = "1\n00:00:01,000 --> 00:00:04,000\nHej, hur mår du?\n";
        let cues = parse(src);
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0], Cue { index: 1, start: 1000, end: 4000, text: "Hej, hur mår du?".into() });
        assert_eq!(serialize(&cues), src);
    }

    #[test]
    fn keeps_multiline_text_and_skips_garbage() {
        let src = "1\n00:00:01,000 --> 00:00:02,000\nline one\nline two\n\nGARBAGE\n\n2\n00:00:03,000 --> 00:00:04,000\nnext\n";
        let cues = parse(src);
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0].text, "line one\nline two");
        assert_eq!(cues[1].index, 2);
    }
}
