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
    // Normalise CRLF first so a line break is one character everywhere below.
    let normalized = input.replace("\r\n", "\n").replace('\r', "\n");
    // A separator is a line with nothing but whitespace on it — not the literal "\n\n".
    //
    // Real OpenSubtitles files carry separator lines holding a space or a tab, and splitting on
    // "\n\n" does not see those: the next cue stays inside the previous block, where the index and
    // timing lines are already taken, so its timecode is joined onto the previous cue's dialogue.
    // The result is not a dropped line but a wrong one — cue 2 vanishes, its words display at cue
    // 1's timestamp, and "00:00:05,000 --> 00:00:06,000" renders on screen as a line of dialogue
    // and is sent to the translator as text. One stray space silently eats the rest of the block.
    let mut block: Vec<&str> = Vec::new();
    for line in normalized.split('\n') {
        if line.trim().is_empty() {
            push_cue(&mut cues, &block);
            block.clear();
        } else {
            block.push(line);
        }
    }
    push_cue(&mut cues, &block);
    cues
}

/// Turn one block's lines into a cue, skipping anything malformed — a single bad cue shouldn't
/// lose the movie.
fn push_cue(cues: &mut Vec<Cue>, block: &[&str]) {
    let mut lines = block.iter().copied();
    let Some(first) = lines.next() else { return };
    // First line is the index; some files omit it and lead with the timecode — tolerate both.
    let (index, timing_line) = match first.trim().parse::<u32>() {
        Ok(n) => (n, lines.next().unwrap_or("")),
        Err(_) => (cues.len() as u32 + 1, first),
    };
    let Some((start, end)) = parse_timing(timing_line) else { return };
    let text = lines.collect::<Vec<_>>().join("\n");
    cues.push(Cue { index, start, end, text });
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
    // char-based (not byte-slice): the ms field is arbitrary downloaded text, and a multibyte char
    // straddling byte index 3 would panic a byte slice.
    let ms3: String = ms.chars().take(3).collect();
    let ms: u64 = format!("{ms3:0<3}").parse().ok()?;
    // Saturating math: a malformed HH:MM:SS from hostile input mustn't overflow-panic in debug.
    let total = h
        .saturating_mul(60)
        .saturating_add(m)
        .saturating_mul(60)
        .saturating_add(sec)
        .saturating_mul(1000)
        .saturating_add(ms);
    Some(total)
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
    fn does_not_panic_on_multibyte_millisecond_field() {
        // A malformed timecode whose millisecond field carries a multibyte char at byte 3 must not
        // panic a byte slice; the cue is simply skipped (unparseable ms).
        let src = "1\n00:00:01,ab€ --> 00:00:02,000\nhi\n";
        assert!(parse(src).is_empty());
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

#[cfg(test)]
mod separator_tests {
    use super::*;

    /// A separator line carrying a space or a tab is still a separator. Splitting on the literal
    /// "\n\n" missed those and swallowed the following cue whole — so this asserts the cue count,
    /// the recovered cue's OWN timing, and that no timecode leaked into anyone's dialogue. The
    /// count alone would pass on a parser that kept two cues but glued the text to the wrong one.
    #[test]
    fn a_whitespace_only_line_separates_cues() {
        for sep in ["   ", "\t", " \t ", ""] {
            let input = format!(
                "1\n00:00:01,000 --> 00:00:02,000\nfirst\n{sep}\n2\n00:00:05,000 --> 00:00:06,000\nsecond\n"
            );
            let cues = parse(&input);
            assert_eq!(cues.len(), 2, "separator {sep:?} did not separate");
            assert_eq!(cues[0].text, "first", "separator {sep:?}");
            assert_eq!(cues[1].text, "second", "separator {sep:?}");
            // The second cue keeps its own timing rather than inheriting the first's.
            assert_eq!((cues[1].start, cues[1].end), (5000, 6000), "separator {sep:?}");
            for c in &cues {
                assert!(!c.text.contains("-->"), "a timecode became dialogue: {:?}", c.text);
            }
        }
    }

    /// The same file with CRLF line endings, since normalisation runs before the split.
    #[test]
    fn a_whitespace_only_line_separates_cues_with_crlf() {
        let input = "1\r\n00:00:01,000 --> 00:00:02,000\r\nfirst\r\n \r\n2\r\n00:00:05,000 --> 00:00:06,000\r\nsecond\r\n";
        let cues = parse(input);
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[1].start, 5000);
    }

    /// Runs of blank lines are not empty cues, and a file that ends without a trailing newline
    /// still yields its last cue — the final block is flushed after the loop, not by a separator.
    #[test]
    fn blank_runs_and_a_missing_trailing_newline() {
        let cues = parse("1\n00:00:01,000 --> 00:00:02,000\nfirst\n\n \n\n2\n00:00:05,000 --> 00:00:06,000\nlast");
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[1].text, "last");
    }
}
