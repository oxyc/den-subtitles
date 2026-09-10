//! Per-install config, base64url-encoded into the addon URL path (Torrentio/den-scout style). It
//! carries the user's BYOK LLM credential, so it is a **bearer secret**: the Den app builds it at
//! `/configure`, stores it in the Keychain, and never logs it. We validate + clamp the untrusted
//! blob before use and never echo the key back.

use std::collections::HashSet;

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

    /// Stable short name for this provider. Goes into cache keys, so it is deliberately not the
    /// Debug spelling: renaming a variant should not silently move every key that names it.
    pub fn tag(self) -> &'static str {
        match self {
            Provider::OpenAI => "openai",
            Provider::Anthropic => "anthropic",
            Provider::Google => "google",
            Provider::Xai => "xai",
            Provider::OpenRouter => "openrouter",
            Provider::DeepL => "deepl",
        }
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
    /// Install id (`iid`), minted by /configure for each link it builds: what `REVOKED_INSTALLS`
    /// names. `None` on a link built before ids existed, which only `CONFIG_EPOCH` can revoke.
    pub iid: Option<String>,
    /// The config epoch (`ep`) the link was stamped with; absent reads as 0.
    pub ep: u64,
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
    iid: Option<String>,
    ep: Option<u64>,
}

/// An install id as /configure mints it: 16 random bytes, base64url, unpadded — 22 characters. The
/// engine refuses padding and non-zero trailing bits, so each id has exactly one spelling and a
/// `REVOKED_INSTALLS` entry cannot be dodged by re-encoding the same bytes.
fn is_install_id(s: &str) -> bool {
    s.len() == 22 && base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(s).is_ok_and(|b| b.len() == 16)
}

/// Which installs are refused (issue #8 R3). Naming an install id kills one leaked link; raising the
/// epoch kills every link stamped before it. Neither touches the sealing key — and rotating that key
/// would not do this, since `CONFIG_KEYS_PREV` keeps links sealed to the old key opening on purpose.
#[derive(Debug, Default)]
pub struct Revocation {
    revoked: HashSet<String>,
    epoch: u64,
}

impl Revocation {
    /// From `REVOKED_INSTALLS` (comma-separated install ids) and `CONFIG_EPOCH` (default 0). A
    /// malformed entry is skipped with a warning: no config that decodes can carry it, so it could
    /// never match. An unparseable epoch is said loudly and enforced as 0, and the startup line's
    /// `epoch=` shows the value actually in force.
    pub fn from_env(revoked: &str, epoch: Option<&str>) -> Revocation {
        let mut ids = HashSet::new();
        for entry in revoked.split(',').map(str::trim).filter(|e| !e.is_empty()) {
            if is_install_id(entry) {
                ids.insert(entry.to_string());
            } else {
                eprintln!("warning: skipping a REVOKED_INSTALLS entry that is not a 22-character install id");
            }
        }
        let epoch = match epoch.map(str::trim).filter(|e| !e.is_empty()) {
            None => 0,
            Some(raw) => raw.parse().unwrap_or_else(|_| {
                eprintln!("warning: CONFIG_EPOCH={raw:?} is not a non-negative integer — enforcing epoch 0");
                0
            }),
        };
        Revocation { revoked: ids, epoch }
    }

    pub fn revoked_count(&self) -> usize {
        self.revoked.len()
    }

    /// The oldest epoch still admitted — what /configure stamps into a new link.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Whether `cfg`'s install is still admitted. The id is checked first, as the more specific answer.
    fn check(&self, cfg: &UserConfig) -> Result<(), Rejected> {
        if let Some(iid) = cfg.iid.as_deref().filter(|iid| self.revoked.contains(*iid)) {
            return Err(Rejected::Revoked { iid_prefix: iid.chars().take(6).collect() });
        }
        if cfg.ep < self.epoch {
            return Err(Rejected::EpochTooOld { ep: cfg.ep, epoch: self.epoch });
        }
        Ok(())
    }
}

/// Why a config segment was not accepted. Every variant gets the same answer on the wire; the
/// difference is for the log only, and nothing here is a credential: a revoked id is kept to its
/// first six characters, and an admitted install's id never gets this far.
#[derive(Debug, PartialEq, Eq)]
pub enum Rejected {
    Undecodable,
    Revoked { iid_prefix: String },
    EpochTooOld { ep: u64, epoch: u64 },
}

impl std::fmt::Display for Rejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Rejected::Undecodable => f.write_str("config does not decode"),
            Rejected::Revoked { iid_prefix } => write!(f, "install revoked (iid={iid_prefix}…)"),
            Rejected::EpochTooOld { ep, epoch } => {
                write!(f, "install epoch too old (ep={ep} < CONFIG_EPOCH={epoch})")
            }
        }
    }
}

/// `decode`, then the revocation check.
pub fn decode_checked(
    keyring: Option<&crate::seal::Keyring>,
    revocation: &Revocation,
    blob: &str,
) -> Result<UserConfig, Rejected> {
    let cfg = decode(keyring, blob).ok_or(Rejected::Undecodable)?;
    revocation.check(&cfg)?;
    Ok(cfg)
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
    // A malformed install id is a malformed config: admitting it would leave an install that no
    // `REVOKED_INSTALLS` entry can ever name.
    if raw.iid.as_deref().is_some_and(|iid| !is_install_id(iid)) {
        return None;
    }
    // The OpenSubtitles key is the subtitle source — required, bounded.
    if raw.opensubtitles_key.is_empty() || raw.opensubtitles_key.len() > 128 {
        return None;
    }
    let opensubtitles_token = raw.opensubtitles_token.filter(|t| !t.is_empty() && t.len() <= 512);

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
        } else if raw.model.trim().len() > 128 {
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
        iid: raw.iid,
        ep: raw.ep.unwrap_or(0),
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

    /// The model is stored trimmed and its emptiness is judged trimmed, so its length must be too.
    /// Measuring the untrimmed string rejected the WHOLE config — every route 400s — over padding
    /// that was about to be thrown away.
    #[test]
    fn a_padded_model_name_is_judged_on_what_is_kept() {
        let padded = format!("{}gpt-4o-mini{}", " ".repeat(80), " ".repeat(80));
        let blob = encode(&format!(
            r#"{{"provider":"openai","apiKey":"sk-test","osKey":"os-test","model":"{padded}"}}"#
        ));
        let cfg = decode(None, &blob).expect("padding is not a reason to reject a config");
        assert_eq!(cfg.llm.unwrap().model, "gpt-4o-mini");

        // A genuinely over-long model is still refused.
        let long = "m".repeat(129);
        let blob = encode(&format!(
            r#"{{"provider":"openai","apiKey":"sk-test","osKey":"os-test","model":"{long}"}}"#
        ));
        assert!(decode(None, &blob).is_none(), "an over-long model name must be refused");
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

    /// Bytes 0..16 as an install id, and a second one that uses both url-safe characters.
    const IID: &str = "AAECAwQFBgcICQoLDA0ODw";
    const OTHER_IID: &str = "_-_-_-_-_-_-_-_-_-_-_w";

    #[test]
    fn install_ids_are_validated_strictly() {
        for ok in [IID, OTHER_IID] {
            assert!(is_install_id(ok), "refused {ok}");
        }
        let refused = [
            "",
            "AAECAwQFBgcICQoLDA0OD",    // 21 characters
            "AAECAwQFBgcICQoLDA0ODwA",  // 23 characters
            "AAECAwQFBgcICQoLDA0ODw==", // padded
            "AAECAwQFBgcICQoLDA0ODx",   // the same bytes with a non-zero trailing bit
            "AAECAwQFBgcICQoLDA0O+w",   // standard base64, not url-safe
            "AAECAwQFBgcICQoLDA0O/w",
            "AAECAwQFBgcICQoLDA0O w",
        ];
        for bad in refused {
            assert!(!is_install_id(bad), "accepted {bad:?}");
        }
    }

    #[test]
    fn a_config_carries_its_install_id_and_epoch() {
        let cfg = decode(None, &encode(&format!(r#"{{"osKey":"o","iid":"{IID}","ep":3}}"#))).unwrap();
        assert_eq!(cfg.iid.as_deref(), Some(IID));
        assert_eq!(cfg.ep, 3);
        // Absent: no id, epoch 0.
        let cfg = decode(None, &encode(r#"{"osKey":"o"}"#)).unwrap();
        assert!(cfg.iid.is_none());
        assert_eq!(cfg.ep, 0);
        // A malformed id or epoch makes the whole config malformed.
        for bad in [
            r#"{"osKey":"o","iid":"short"}"#,
            r#"{"osKey":"o","iid":"AAECAwQFBgcICQoLDA0ODw=="}"#,
            r#"{"osKey":"o","iid":7}"#,
            r#"{"osKey":"o","ep":-1}"#,
            r#"{"osKey":"o","ep":1.5}"#,
            r#"{"osKey":"o","ep":"1"}"#,
        ] {
            assert!(decode(None, &encode(bad)).is_none(), "accepted {bad}");
        }
    }

    #[test]
    fn a_listed_install_and_an_old_epoch_are_refused() {
        let revocation = Revocation::from_env(&format!(" {IID} ,, not-an-id"), Some("2"));
        assert_eq!(revocation.revoked_count(), 1, "the malformed entry is skipped");
        assert_eq!(revocation.epoch(), 2);
        let check = |json: &str| decode_checked(None, &revocation, &encode(json));

        let listed = format!(r#"{{"osKey":"o","iid":"{IID}","ep":5}}"#);
        assert_eq!(check(&listed).unwrap_err(), Rejected::Revoked { iid_prefix: "AAECAw".into() });
        assert!(check(&format!(r#"{{"osKey":"o","iid":"{OTHER_IID}","ep":5}}"#)).is_ok());

        assert_eq!(check(r#"{"osKey":"o","ep":1}"#).unwrap_err(), Rejected::EpochTooOld { ep: 1, epoch: 2 });
        assert_eq!(check(r#"{"osKey":"o"}"#).unwrap_err(), Rejected::EpochTooOld { ep: 0, epoch: 2 });
        assert!(check(r#"{"osKey":"o","ep":2}"#).is_ok());

        // Nothing configured: a link with no id and no epoch is admitted, as every link before ids was.
        assert!(decode_checked(None, &Revocation::default(), &encode(r#"{"osKey":"o"}"#)).is_ok());
        assert_eq!(decode_checked(None, &revocation, "!!").unwrap_err(), Rejected::Undecodable);
    }

    #[test]
    fn a_refusal_names_no_more_than_the_revoked_id_s_prefix() {
        assert_eq!(
            Rejected::Revoked { iid_prefix: "AAECAw".into() }.to_string(),
            "install revoked (iid=AAECAw…)"
        );
        assert_eq!(
            Rejected::EpochTooOld { ep: 1, epoch: 2 }.to_string(),
            "install epoch too old (ep=1 < CONFIG_EPOCH=2)"
        );
    }

    #[test]
    fn an_unparseable_epoch_enforces_zero() {
        assert_eq!(Revocation::from_env("", None).epoch(), 0);
        assert_eq!(Revocation::from_env("", Some("two")).epoch(), 0);
        assert_eq!(Revocation::from_env("", Some("-1")).epoch(), 0);
        assert_eq!(Revocation::from_env("", Some(" 7 ")).epoch(), 7);
    }
}
