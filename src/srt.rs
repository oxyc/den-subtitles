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
    // An empty line always separates cues. A line holding only whitespace separates only when a cue
    // header follows it.
    //
    // Real OpenSubtitles files carry separator lines with a space or a tab on them, and splitting on
    // "\n\n" missed those: the next cue stayed inside the previous block, where the index and timing
    // lines were already taken, so its timecode was joined onto the previous cue's dialogue — cue 2
    // gone, its words shown at cue 1's timestamp, its timecode rendered as dialogue. But treating
    // every blank-ish line as a separator is the same bug mirrored: a padded line INSIDE a cue would
    // truncate it and drop the dialogue after it. Only the lookahead tells the two apart.
    let lines: Vec<&str> = normalized.split('\n').collect();
    let mut block: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if !lines[i].trim().is_empty() {
            block.push(lines[i]);
            i += 1;
            continue;
        }
        // Take the whole blank run in one step. Deciding per line meant rescanning the rest of the
        // run for every line in it — quadratic, and since this parser now gates every downloaded
        // subtitle, a file of padded lines was minutes to hours of CPU on a runtime that has one
        // thread for every connection.
        let end = i + lines[i..].iter().take_while(|l| l.trim().is_empty()).count();
        // A truly empty line always separates. A run of only whitespace separates when a cue
        // header follows it; otherwise it is padding inside the dialogue and stays in the text.
        if lines[i..end].iter().any(|l| l.is_empty()) || starts_a_cue(&lines[end..]) {
            push_cue(&mut cues, &block);
            block.clear();
        } else {
            block.extend_from_slice(&lines[i..end]);
        }
        i = end;
    }
    push_cue(&mut cues, &block);
    cues
}

/// Does a cue begin here — an index followed by a timing line, or a bare timing line? Called with
/// the run of blank lines already skipped, so it looks at two lines and returns.
fn starts_a_cue(rest: &[&str]) -> bool {
    let Some(first) = rest.first() else { return true }; // trailing padding ends the file
    parse_timing(first).is_some()
        || (first.trim().parse::<u32>().is_ok() && rest.get(1).is_some_and(|l| parse_timing(l).is_some()))
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
        // Blank lines are dropped, not written: one inside a cue's text ENDS that cue for every
        // reader downstream, silently losing the rest of it. A translation model's output reaches
        // here verbatim, so this is the only place that can guarantee a parseable document.
        let mut first = true;
        for line in c.text.lines().filter(|l| !l.trim().is_empty()) {
            if !first {
                out.push('\n');
            }
            out.push_str(line);
            first = false;
        }
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

    /// A cue whose text carries a blank line — which a translation model can return, since its
    /// output reaches serialize verbatim — must not end the cue early. It used to: everything after
    /// the blank line was lost, and the truncated document was cached for 60 days.
    #[test]
    fn a_blank_line_inside_a_cue_does_not_truncate_the_document() {
        let cues = vec![
            Cue { index: 1, start: 1000, end: 2000, text: "Hej.\n\nHur mår du?".into() },
            Cue { index: 2, start: 3000, end: 4000, text: "Bra.".into() },
        ];
        let round_tripped = parse(&serialize(&cues));
        assert_eq!(round_tripped.len(), 2, "a cue was lost");
        assert_eq!(round_tripped[0].text, "Hej.\nHur mår du?", "dialogue after the blank line went missing");
        assert_eq!(round_tripped[1].text, "Bra.");
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
    /// This parser gates every downloaded subtitle body (up to MAX_BODY = 12 MiB of third-party CDN
    /// content), it has no await in it, and the runtime has one thread — so its cost is the whole
    /// server's cost. Deciding the separator question per line rescanned the rest of the blank run
    /// each time: 40k padded lines took 1.6s, 160k took 27s, and a megabyte never finished.
    ///
    /// The bound is loose on purpose — it is here to catch a return to quadratic, not to police
    /// milliseconds. Linear does this in single-digit ms; quadratic needs minutes.
    #[test]
    fn a_long_run_of_padded_lines_stays_linear() {
        let mut input = String::from("1\n00:00:01,000 --> 00:00:02,000\nhi\n");
        for _ in 0..200_000 {
            input.push_str(" \n");
        }
        input.push_str("not-a-cue-header\n");
        let started = std::time::Instant::now();
        let cues = parse(&input);
        let took = started.elapsed();
        assert_eq!(cues.len(), 1);
        assert!(took < std::time::Duration::from_secs(10), "parse took {took:?} — quadratic again?");
    }


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

    /// The converse of the rule above, and the case the first version of this fix regressed: a
    /// padded line INSIDE a cue is dialogue, not a separator. Treating it as one truncated the cue
    /// and dropped everything after it — cue A rendered blank, its second line gone.
    #[test]
    fn a_padded_line_inside_a_cue_keeps_the_dialogue_after_it() {
        let cues = parse("1\n00:00:01,000 --> 00:00:04,000\n \n- Hello there.\n\n2\n00:00:05,000 --> 00:00:06,000\nsecond\n");
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0].text, " \n- Hello there.", "the dialogue after the padded line was dropped");
        assert_eq!(cues[1].text, "second");

        let cues = parse("1\n00:00:01,000 --> 00:00:04,000\nHello.\n \nGoodbye.\n\n2\n00:00:05,000 --> 00:00:06,000\nsecond\n");
        assert_eq!(cues[0].text, "Hello.\n \nGoodbye.");
        assert_eq!(cues.len(), 2);
    }

    /// An EMPTY line always separates, whatever follows — that is the format, and it is how a
    /// malformed block gets skipped rather than folded into its neighbour's dialogue.
    #[test]
    fn an_empty_line_separates_even_before_garbage() {
        let cues = parse("1\n00:00:01,000 --> 00:00:02,000\nline one\n\nGARBAGE\n\n2\n00:00:03,000 --> 00:00:04,000\nnext\n");
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0].text, "line one", "garbage was folded into the cue text");
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
