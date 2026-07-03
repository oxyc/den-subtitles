//! Per-install config, base64url-encoded into the addon URL path (Torrentio/den-scout style). It
//! carries the user's BYOK LLM credential, so it is a **bearer secret**: the Den app builds it at
//! `/configure`, stores it in the Keychain, and never logs it. We validate + clamp the untrusted
//! blob before use and never echo the key back.

use base64::Engine;
use serde::Deserialize;

/// LLM providers we know how to call. The wire value is the lowercase tag in the config JSON.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    OpenAI,
    Anthropic,
    Google,
    Xai,
    OpenRouter,
    DeepL,
}

impl Provider {
    fn parse(s: &str) -> Option<Provider> {
        Some(match s {
            "openai" => Provider::OpenAI,
            "anthropic" => Provider::Anthropic,
            "google" => Provider::Google,
            "xai" => Provider::Xai,
            "openrouter" => Provider::OpenRouter,
            "deepl" => Provider::DeepL,
            _ => return None,
        })
    }

    /// Sensible default model when the user leaves the field blank — the cheap/fast/decent tier.
    pub fn default_model(self) -> &'static str {
        match self {
            Provider::OpenAI => "gpt-4o-mini",
            Provider::Anthropic => "claude-haiku-4-5",
            Provider::Google => "gemini-2.0-flash",
            Provider::Xai => "grok-2",
            Provider::OpenRouter => "openai/gpt-4o-mini",
            Provider::DeepL => "", // DeepL has no model knob
        }
    }
}

/// The translation credential. Optional on a config — a subtitles-only install (fetch + auto-sync,
/// no AI) simply omits it.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub provider: Provider,
    pub model: String,
    pub api_key: String,
}

/// A validated install config. Everything is BYOK and rides in the addon URL — the app stores the
/// blob in the Keychain. The OpenSubtitles key (subtitle source) is required; the LLM credential
/// (translation) is optional.
#[derive(Debug, Clone)]
pub struct UserConfig {
    /// OpenSubtitles API-consumer key — the subtitle source. Required.
    pub opensubtitles_key: String,
    /// Optional OpenSubtitles service-account bearer to lift the download quota above anonymous.
    pub opensubtitles_token: Option<String>,
    /// Translation credential. `None` → subtitles-only (fetch + sync, no AI translation).
    pub llm: Option<LlmConfig>,
    /// Auto-sync non-hash results through Tier-1 reference alignment (`?ref=`). On by default; when
    /// off, the subtitle proxy still serves and caches, just without the automatic alignment.
    pub auto_sync: bool,
}

/// Untrusted wire shape before validation.
#[derive(Deserialize)]
struct RawConfig {
    #[serde(default)]
    provider: String,
    #[serde(default)]
    model: String,
    #[serde(rename = "apiKey", default)]
    api_key: String,
    #[serde(rename = "osKey", default)]
    opensubtitles_key: String,
    #[serde(rename = "osToken")]
    opensubtitles_token: Option<String>,
    #[serde(rename = "autoSync")]
    auto_sync: Option<bool>,
}

/// Decode the config path segment into a validated config, or `None` (→ 400). The decoded bytes are
/// either a SEALED blob (first byte == `SEALED_VERSION` → decrypt with the keyring) or a legacy plaintext
/// JSON config (first byte `{`). Sealed with no keyring, or a decrypt failure, fails CLOSED — never a
/// partial/empty config. Mirrors den-scout (den-scout/docs/SEALED-CONFIG.md).
pub fn decode(keyring: Option<&crate::seal::Keyring>, blob: &str) -> Option<UserConfig> {
    let data = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(blob).ok()?;
    let data = if data.first() == Some(&crate::seal::SEALED_VERSION) {
        keyring?.open(&data[1..])? // sealed but no key, or decrypt fail → None
    } else {
        data // legacy plaintext
    };
    let raw: RawConfig = serde_json::from_slice(&data).ok()?;
    validate(raw)
}

fn validate(raw: RawConfig) -> Option<UserConfig> {
    // The OpenSubtitles key is the subtitle source — required, bounded.
    if raw.opensubtitles_key.is_empty() || raw.opensubtitles_key.len() > 128 {
        return None;
    }
    let opensubtitles_token = raw
        .opensubtitles_token
        .filter(|t| !t.is_empty() && t.len() <= 512);

    // The LLM credential is optional. It's absent iff neither a provider nor a key was given
    // (subtitles-only). If either is present, both must be valid — a half-filled AI section (e.g. a
    // key with no provider) is a 400 rather than a silent no-translate surprise.
    let llm = if raw.provider.is_empty() && raw.api_key.is_empty() {
        None
    } else {
        let provider = Provider::parse(&raw.provider)?;
        if raw.api_key.is_empty() || raw.api_key.len() > 512 {
            return None;
        }
        let model = if raw.model.trim().is_empty() {
            provider.default_model().to_string()
        } else if raw.model.len() > 128 {
            return None;
        } else {
            raw.model.trim().to_string()
        };
        Some(LlmConfig { provider, model, api_key: raw.api_key })
    };

    Some(UserConfig {
        opensubtitles_key: raw.opensubtitles_key,
        opensubtitles_token,
        llm,
        auto_sync: raw.auto_sync.unwrap_or(true),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(json: &str) -> String {
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(json)
    }

    #[test]
    fn decodes_and_defaults_the_model() {
        let blob = encode(r#"{"provider":"openai","apiKey":"sk-test","osKey":"os-test"}"#);
        let cfg = decode(None, &blob).unwrap();
        let llm = cfg.llm.unwrap();
        assert_eq!(llm.provider, Provider::OpenAI);
        assert_eq!(llm.model, "gpt-4o-mini");
        assert_eq!(cfg.opensubtitles_key, "os-test");
        assert!(cfg.auto_sync);
    }

    #[test]
    fn llm_is_optional_when_only_opensubtitles_is_given() {
        let cfg = decode(None, &encode(r#"{"osKey":"os-test"}"#)).unwrap();
        assert!(cfg.llm.is_none());
        assert_eq!(cfg.opensubtitles_key, "os-test");
    }

    #[test]
    fn rejects_bad_input() {
        // OpenSubtitles key is always required.
        assert!(decode(None, &encode(r#"{"provider":"openai","apiKey":"x"}"#)).is_none());
        // A half-filled AI section (provider, no key) is rejected, not silently dropped.
        assert!(decode(None, &encode(r#"{"provider":"openai","osKey":"o"}"#)).is_none());
        // Unknown provider.
        assert!(decode(None, &encode(r#"{"provider":"acme","apiKey":"x","osKey":"o"}"#)).is_none());
    }

    #[test]
    fn decodes_a_sealed_segment_minted_by_the_browser() {
        // A fixed segment MINTED by the /configure browser bundle (tweetnacl + blake2b crypto_box_seal),
        // sealing {osKey, provider, apiKey} to the vector key — the JS→Rust interop gate + full decode
        // path. Regenerate with scratch/sealsrc/entry.js's denSeal() if the wire format ever changes.
        const VEC_PRIV: &str = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=";
        const JS_SEG: &str = "Ac3WWHzRZKV9OjdSgIPaNFFhaE9UY0vwxgSO6F5Ghug1nyjlKUodEQmhhlPhX-j1KffJnpj58HPhlpePWcbnuX9GL9rGMsdki1hGXSzRG94ON_aYocvFkl9bSU2QZa8o3waeHHm9wmjLQg";
        let kr = crate::seal::Keyring::from_env(VEC_PRIV, "").unwrap().unwrap();

        let cfg = decode(Some(&kr), JS_SEG).expect("sealed segment decodes");
        assert_eq!(cfg.opensubtitles_key, "os-js-ok");
        assert_eq!(cfg.llm.unwrap().provider, Provider::OpenAI);

        // Fail CLOSED: a sealed segment with no keyring configured.
        assert!(decode(None, JS_SEG).is_none());
        // Back-compat: legacy plaintext still decodes with a keyring present.
        assert!(decode(Some(&kr), &encode(r#"{"osKey":"os-legacy"}"#)).is_some());
    }
}
