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

/// A validated install config. Everything is BYOK and rides in the addon URL — the app stores the
/// blob in the Keychain. Both credentials are the user's: the OpenSubtitles consumer key (the
/// subtitle source) and the LLM key (translation).
#[derive(Debug, Clone)]
pub struct UserConfig {
    pub provider: Provider,
    pub model: String,
    pub api_key: String,
    /// OpenSubtitles API-consumer key — the subtitle source. Required.
    pub opensubtitles_key: String,
    /// Optional OpenSubtitles service-account bearer to lift the download quota above anonymous.
    pub opensubtitles_token: Option<String>,
    /// Auto-sync every result through the cheap tiers (hash + reference). On by default.
    /// Consumed once the sync ladder is wired into the request path (next increment).
    #[allow(dead_code)]
    pub auto_sync: bool,
}

/// Untrusted wire shape before validation.
#[derive(Deserialize)]
struct RawConfig {
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

/// Decode the base64url blob into a validated config, or `None` (→ 400). Mirrors den-scout's
/// decode → strict-whitelist → clamp seam so an opaque `configId` can slot in later.
pub fn decode(blob: &str) -> Option<UserConfig> {
    let data = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(blob).ok()?;
    let raw: RawConfig = serde_json::from_slice(&data).ok()?;
    validate(raw)
}

fn validate(raw: RawConfig) -> Option<UserConfig> {
    let provider = Provider::parse(&raw.provider)?;
    // DeepL uses the key with no model; every LLM provider needs a key. An empty key is a 400 rather
    // than a runtime auth failure the user can't see.
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
    // The OpenSubtitles key is the subtitle source — required, bounded.
    if raw.opensubtitles_key.is_empty() || raw.opensubtitles_key.len() > 128 {
        return None;
    }
    let opensubtitles_token = raw
        .opensubtitles_token
        .filter(|t| !t.is_empty() && t.len() <= 512);
    Some(UserConfig {
        provider,
        model,
        api_key: raw.api_key,
        opensubtitles_key: raw.opensubtitles_key,
        opensubtitles_token,
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
        let cfg = decode(&blob).unwrap();
        assert_eq!(cfg.provider, Provider::OpenAI);
        assert_eq!(cfg.model, "gpt-4o-mini");
        assert_eq!(cfg.opensubtitles_key, "os-test");
        assert!(cfg.auto_sync);
    }

    #[test]
    fn rejects_unknown_provider_and_missing_keys() {
        // Unknown provider.
        assert!(decode(&encode(r#"{"provider":"acme","apiKey":"x","osKey":"o"}"#)).is_none());
        // Missing LLM key.
        assert!(decode(&encode(r#"{"provider":"openai","osKey":"o"}"#)).is_none());
        // Missing OpenSubtitles key.
        assert!(decode(&encode(r#"{"provider":"openai","apiKey":"x"}"#)).is_none());
    }
}
