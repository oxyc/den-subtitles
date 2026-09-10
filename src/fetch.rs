//! Bounded body readers. Upstream bodies — OpenSubtitles JSON, the OpenSubtitles-supplied download
//! link (points wherever they say), and LLM responses — are read with a hard byte ceiling so a
//! hostile or runaway response can't OOM the container. We reject early on an oversized declared
//! `Content-Length` and, for chunked bodies with none, abort once the accumulated bytes cross the
//! cap. A downloaded subtitle is then decoded to UTF-8 from whatever encoding it arrived in.

use chardetng::{EncodingDetector, Iso2022JpDetection, Utf8Detection};
use encoding_rs::{Encoding, UTF_8};
use futures_util::StreamExt;
use serde::de::DeserializeOwned;

/// Generous ceiling: an SRT is tens of KB, an LLM JSON reply a few hundred; 12 MiB is far above any
/// legitimate body while still bounding memory per in-flight request.
pub const MAX_BODY: usize = 12 * 1024 * 1024;

/// Read a response body into memory, capped at `max` bytes.
pub async fn capped_bytes(resp: reqwest::Response, max: usize) -> Result<Vec<u8>, String> {
    if let Some(len) = resp.content_length() {
        if len as usize > max {
            return Err(format!("upstream body too large: {len} bytes"));
        }
    }
    let mut stream = resp.bytes_stream();
    let mut out: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("read body: {e}"))?;
        if out.len() + chunk.len() > max {
            return Err(format!("upstream body exceeded {max} bytes"));
        }
        out.extend_from_slice(&chunk);
    }
    Ok(out)
}

/// A downloaded subtitle as UTF-8, whatever encoding the uploader's editor saved it in, plus the
/// encoding it was converted from when that was not UTF-8. `lang` is the subtitle's declared
/// language, used only as a detection hint.
///
/// The api.opensubtitles.com `/download` endpoint is meant to convert to UTF-8 on its side, and does
/// not always. Subtitle files in the wild are full of Windows-1250/1251, ISO-8859-x and UTF-16, and
/// the player drops any cue whose text is not valid UTF-8 — after which the app switches the track
/// off, so the viewer sees nothing at all. Decoding lossily was no better: every Cyrillic letter
/// became U+FFFD, and since `srt::has_a_cue` only reads the ASCII timing and index lines, the garbage
/// passed every check and cached for sixty days.
///
/// In order: a BOM names the encoding outright (UTF-8 or UTF-16 LE/BE); valid UTF-8 is taken as it
/// is; otherwise chardetng guesses, the way Firefox does for an unlabeled page.
pub fn subtitle_text(bytes: Vec<u8>, lang: Option<&str>) -> (String, Option<&'static Encoding>) {
    if let Some((encoding, bom)) = Encoding::for_bom(&bytes) {
        let (text, _) = encoding.decode_without_bom_handling(&bytes[bom..]);
        return (text.into_owned(), (encoding != UTF_8).then_some(encoding));
    }
    let bytes = match String::from_utf8(bytes) {
        Ok(text) => return (text, None),
        Err(e) => e.into_bytes(),
    };
    if is_damaged_utf8(&bytes) {
        return (String::from_utf8_lossy(&bytes).into_owned(), None);
    }
    let mut detector = EncodingDetector::new(Iso2022JpDetection::Deny);
    detector.feed(&bytes, true);
    // UTF-8 is already ruled out above, so the guess is only ever between legacy encodings.
    let encoding = detector.guess(lang.and_then(encoding_tld).map(str::as_bytes), Utf8Detection::Deny);
    let (text, _) = encoding.decode_without_bom_handling(&bytes);
    (text.into_owned(), Some(encoding))
}

/// UTF-8 with a few broken bytes in it — two files concatenated, one bad cue — rather than a legacy
/// encoding. Legacy 8-bit text almost never forms a well-formed multi-byte UTF-8 sequence by
/// accident, so a body with more well-formed non-ASCII characters than broken spots is UTF-8 that
/// took damage. Handing it to the detector instead would decode the whole file as Windows-1252 and
/// turn every good character into mojibake to account for a few bad ones.
fn is_damaged_utf8(bytes: &[u8]) -> bool {
    let (mut good, mut bad) = (0usize, 0usize);
    for chunk in bytes.utf8_chunks() {
        good += chunk.valid().chars().filter(|c| !c.is_ascii()).count();
        bad += usize::from(!chunk.invalid().is_empty());
    }
    good > bad
}

/// The country-code TLD chardetng should assume for a subtitle in `lang` (an OpenSubtitles language
/// code such as `ru`, `pt-BR` or `zh-TW`), or `None` when the language says nothing useful.
///
/// chardetng takes its hint as the TLD a page was served from, because that is what narrows the
/// candidates on the web: `.ru` means the Cyrillic encodings, `.cz` the Central European ones. A
/// subtitle's language is the same fact from a different place. Without it, a short Latin-2 file can
/// lose to Windows-1252 and a Cyrillic one to a neighbouring Cyrillic encoding. Languages whose
/// legacy encoding is Windows-1252 map to nothing, which is chardetng's default anyway.
pub fn encoding_tld(lang: &str) -> Option<&'static str> {
    let lang = lang.to_ascii_lowercase();
    let tld = match lang.as_str() {
        "zh-tw" | "zh-hk" => "tw",
        _ => match lang.split(['-', '_']).next().unwrap_or("") {
            "ru" => "ru",
            "uk" => "ua",
            "be" => "by",
            "bg" => "bg",
            "sr" => "rs",
            "mk" => "mk",
            "kk" => "kz",
            "pl" => "pl",
            "cs" => "cz",
            "sk" => "sk",
            "hu" => "hu",
            "ro" => "ro",
            "hr" => "hr",
            "sl" => "si",
            "bs" => "ba",
            "lt" => "lt",
            "lv" => "lv",
            "el" => "gr",
            "tr" => "tr",
            "is" => "is",
            "ar" => "sa",
            "fa" => "ir",
            "he" => "il",
            // `ze` is OpenSubtitles' code for Chinese/English bilingual files.
            "zh" | "ze" => "cn",
            "ja" => "jp",
            "ko" => "kr",
            "th" => "th",
            "vi" => "vn",
            _ => return None,
        },
    };
    Some(tld)
}

/// Capped body → deserialized JSON.
pub async fn capped_json<T: DeserializeOwned>(resp: reqwest::Response, max: usize) -> Result<T, String> {
    let bytes = capped_bytes(resp, max).await?;
    serde_json::from_slice(&bytes).map_err(|e| format!("bad json: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A subtitle-shaped body around `dialogue`, so detection sees what it sees in production: ASCII
    /// timing lines with the non-ASCII text between them.
    fn srt(dialogue: &str) -> String {
        format!("1\r\n00:00:01,000 --> 00:00:04,000\r\n{dialogue}\r\n\r\n2\r\n00:00:05,000 --> 00:00:08,000\r\n{dialogue}\r\n")
    }

    fn encode(encoding: &'static Encoding, text: &str) -> Vec<u8> {
        let (bytes, _, unmappable) = encoding.encode(text);
        assert!(!unmappable, "the fixture does not fit {}", encoding.name());
        bytes.into_owned()
    }

    #[test]
    fn plain_utf8_is_untouched() {
        let body = srt("Hej, hur mår du? Привет!");
        assert_eq!(subtitle_text(body.clone().into_bytes(), Some("sv")), (body, None));
    }

    /// A UTF-8 BOM ahead of the first cue index is not dialogue; it is stripped, and nothing was
    /// converted.
    #[test]
    fn a_utf8_bom_is_stripped() {
        let body = srt("Привет, как дела?");
        let mut bytes = vec![0xEF, 0xBB, 0xBF];
        bytes.extend_from_slice(body.as_bytes());
        assert_eq!(subtitle_text(bytes, None), (body, None));
    }

    #[test]
    fn utf16_with_a_bom_is_decoded() {
        let body = srt("Привет, как дела?");
        let mut le = vec![0xFF, 0xFE];
        le.extend(body.encode_utf16().flat_map(u16::to_le_bytes));
        assert_eq!(subtitle_text(le, None), (body.clone(), Some(encoding_rs::UTF_16LE)));

        let mut be = vec![0xFE, 0xFF];
        be.extend(body.encode_utf16().flat_map(u16::to_be_bytes));
        assert_eq!(subtitle_text(be, None), (body, Some(encoding_rs::UTF_16BE)));
    }

    /// Lossy decoding turned every one of these letters into U+FFFD, and the player dropped the cue.
    #[test]
    fn windows_1251_cyrillic_is_decoded() {
        let body = srt("Привет, как дела? Всё хорошо, спасибо.");
        let bytes = encode(encoding_rs::WINDOWS_1251, &body);
        for lang in [Some("ru"), None] {
            assert_eq!(
                subtitle_text(bytes.clone(), lang),
                (body.clone(), Some(encoding_rs::WINDOWS_1251)),
                "hint {lang:?}"
            );
        }
    }

    /// `ś` and `ź` sit at 0x9C and 0x9F in Windows-1250, which are control codes in ISO-8859-2, so
    /// only the right guess reproduces the text.
    #[test]
    fn windows_1250_polish_is_decoded() {
        let body = srt("Zażółć gęślą jaźń.");
        let bytes = encode(encoding_rs::WINDOWS_1250, &body);
        for lang in [Some("pl"), Some("cs"), None] {
            let (text, converted) = subtitle_text(bytes.clone(), lang);
            assert_eq!(text, body, "hint {lang:?}");
            assert!(converted.is_some(), "a conversion went unreported for hint {lang:?}");
        }
    }

    /// A UTF-8 file with one broken byte stays UTF-8. Sending it to the detector would decode every
    /// good character as Windows-1252 to account for the one bad one.
    #[test]
    fn utf8_with_a_stray_byte_stays_utf8() {
        let mut bytes = srt("Привет, как дела?").into_bytes();
        bytes.push(0xFF);
        let (text, converted) = subtitle_text(bytes, None);
        assert!(text.contains("Привет, как дела?"), "good UTF-8 was re-decoded: {text:?}");
        assert_eq!(converted, None);
    }

    #[test]
    fn a_language_maps_to_the_tld_its_encodings_live_under() {
        assert_eq!(encoding_tld("ru"), Some("ru"));
        assert_eq!(encoding_tld("cs"), Some("cz"));
        assert_eq!(encoding_tld("el"), Some("gr"));
        assert_eq!(encoding_tld("pt-BR"), None, "a Windows-1252 language needs no hint");
        assert_eq!(encoding_tld("zh-TW"), Some("tw"));
        assert_eq!(encoding_tld("zh-CN"), Some("cn"));
        assert_eq!(encoding_tld("EN"), None);
        // Whatever arrives on the query string, only a TLD from the table ever reaches chardetng,
        // which panics on anything upper-case, dotted or non-ASCII.
        assert_eq!(encoding_tld("ru.evil/Ü"), None);
        assert_eq!(encoding_tld(""), None);
    }
}
