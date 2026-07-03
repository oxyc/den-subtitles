//! Sealed config-in-URL — the den-subtitles half of den-scout/docs/SEALED-CONFIG.md. The config path
//! segment may carry a config SEALED to the addon's X25519 key (libsodium `crypto_box_seal`) instead of
//! plaintext base64, so the BYOK OpenSubtitles + LLM keys are never plaintext in the URL and never stored
//! server-side — decrypted per request, then dropped.
//!
//! Wire format is byte-identical to the Go addon (den-scout) and the browser bundle: the decoded segment
//! is `version_byte(0x01) ‖ crypto_box_seal(config_json)`. Decrypt-only (no rng dep).

use base64::Engine;
use crypto_box::SecretKey;

/// First byte of a DECODED config segment marking a sealed blob. Legacy plaintext decodes to JSON whose
/// first byte is `{` (0x7b), so the two never collide.
pub const SEALED_VERSION: u8 = 0x01;

/// The addon's recipient keys: current first, then prior keys (tried in order on open) so the keypair can
/// be rotated without breaking installs whose URL was sealed to an older key.
pub struct Keyring {
    keys: Vec<SecretKey>,
}

impl Keyring {
    /// Build from a base64 current private key + comma-separated prior keys. Empty current → `None`
    /// (sealed URLs disabled; legacy plaintext still works). A malformed key is an `Err`.
    pub fn from_env(current: &str, prev: &str) -> Result<Option<Keyring>, String> {
        let current = current.trim();
        if current.is_empty() {
            return Ok(None);
        }
        let mut keys = Vec::new();
        let mut add = |s: &str| -> Result<(), String> {
            let s = s.trim();
            if s.is_empty() {
                return Ok(());
            }
            let raw = decode_key(s).ok_or("bad base64 config key")?;
            let arr: [u8; 32] = raw
                .try_into()
                .map_err(|_| "config key must be a 32-byte X25519 private key".to_string())?;
            keys.push(SecretKey::from(arr));
            Ok(())
        };
        add(current).map_err(|e| format!("current: {e}"))?; // a bad CURRENT key is a real misconfig
        // A malformed PRIOR key is skipped, not fatal — a typo in one rotation entry must not disable the
        // whole ring (which would silently take sealing offline for the good current key too).
        for p in prev.split(',') {
            if let Err(e) = add(p) {
                eprintln!("den-subtitles: skipping a malformed SUBS_CONFIG_KEYS_PREV entry: {e}");
            }
        }
        Ok(Some(Keyring { keys }))
    }

    /// Open a `crypto_box_seal` ciphertext with the keyring. `None` on any failure (fail closed — the
    /// caller then serves a 400, never a partial/empty config).
    pub fn open(&self, sealed: &[u8]) -> Option<Vec<u8>> {
        for sk in &self.keys {
            if let Ok(pt) = sk.unseal(sealed) {
                return Some(pt);
            }
        }
        None
    }

    /// The current recipient public key (std base64), served at `/config-key` so a client can seal to it.
    pub fn current_pub_b64(&self) -> String {
        match self.keys.first() {
            Some(sk) => base64::engine::general_purpose::STANDARD.encode(sk.public_key().as_bytes()),
            None => String::new(),
        }
    }
}

/// Accept std/url base64 with or without padding for the key material.
fn decode_key(s: &str) -> Option<Vec<u8>> {
    use base64::engine::general_purpose as gp;
    gp::STANDARD
        .decode(s)
        .ok()
        .or_else(|| gp::STANDARD_NO_PAD.decode(s).ok())
        .or_else(|| gp::URL_SAFE.decode(s).ok())
        .or_else(|| gp::URL_SAFE_NO_PAD.decode(s).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    // The SAME fixed libsodium crypto_box_seal vector the Go addon tests (PyNaCl SealedBox). The interop
    // GATE: this ciphertext, produced by a real libsodium binding, must open in Rust too.
    const VEC_PRIV_B64: &str = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=";
    const VEC_PUB_B64: &str = "j0DFrbaPJWJK5bIU6nZ6bslNgp09e14a0bpvPiE4KF8=";
    const VEC_CT_B64: &str = "YVIKGV1+YCwzCoPC0WNrcle9bYR0iWhBDAsy2ylVhGmDWneqpfb/Oug5izEfwx2Q9j8UDKM2XF/u+Q9Sg1jqQeMb5RNWpLZk+81tVixiI5qFpc/zNGAwfTJSMVj+B48nCbgqRk0rqQzqniVKm1d85g==";
    const VEC_PLAIN: &str = r#"{"debrid":[{"service":"realdebrid","token":"SEALED-VECTOR-OK"}]}"#;

    #[test]
    fn opens_libsodium_vector() {
        let kr = Keyring::from_env(VEC_PRIV_B64, "").unwrap().unwrap();
        assert_eq!(kr.current_pub_b64(), VEC_PUB_B64, "derived pub must match libsodium");
        let ct = base64::engine::general_purpose::STANDARD.decode(VEC_CT_B64).unwrap();
        let pt = kr.open(&ct).expect("open libsodium vector");
        assert_eq!(String::from_utf8(pt).unwrap(), VEC_PLAIN);
    }

    #[test]
    fn fails_closed() {
        let kr = Keyring::from_env(VEC_PRIV_B64, "").unwrap().unwrap();
        let mut ct = base64::engine::general_purpose::STANDARD.decode(VEC_CT_B64).unwrap();
        let n = ct.len() - 1;
        ct[n] ^= 0xff; // tamper
        assert!(kr.open(&ct).is_none(), "tampered ciphertext must not open");
        assert!(kr.open(b"short").is_none());
        // Wrong recipient key.
        let other = Keyring::from_env(&base64::engine::general_purpose::STANDARD.encode([0u8; 32]), "")
            .unwrap()
            .unwrap();
        let good = base64::engine::general_purpose::STANDARD.decode(VEC_CT_B64).unwrap();
        assert!(other.open(&good).is_none());
    }

    #[test]
    fn empty_current_disables() {
        assert!(Keyring::from_env("", "").unwrap().is_none());
        assert!(Keyring::from_env("not-base64-@@@", "").is_err() || Keyring::from_env("AAAA", "").is_err());
    }

    #[test]
    fn a_malformed_prior_key_is_skipped_not_fatal() {
        // A typo in one rotation entry must not disable the whole ring — the current key still opens.
        let kr = Keyring::from_env(VEC_PRIV_B64, "not-base64-@@@,also!!bad")
            .unwrap()
            .expect("current key valid → ring built despite bad prior entries");
        let ct = base64::engine::general_purpose::STANDARD.decode(VEC_CT_B64).unwrap();
        assert_eq!(String::from_utf8(kr.open(&ct).unwrap()).unwrap(), VEC_PLAIN);
    }
}
