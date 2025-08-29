use crate::register_if;
use chacha20poly1305::KeyInit;
use hmac::Mac;
use lyra_core::pretty::format_value;
use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

#[cfg(feature = "tools")]
use crate::tool_spec;
#[cfg(feature = "tools")]
use crate::tools::add_specs;
#[cfg(feature = "tools")]
use crate::{schema_bool, schema_str};
#[cfg(feature = "tools")]
use std::collections::HashMap;

pub fn register_crypto(ev: &mut Evaluator) {
    // Randomness
    ev.register("RandomBytes", random_bytes as NativeFn, Attributes::empty());
    ev.register("RandomHex", random_hex as NativeFn, Attributes::empty());
    // Hashing
    ev.register("Hash", hash_fn as NativeFn, Attributes::LISTABLE);
    // AEAD (ChaCha20-Poly1305)
    ev.register("AeadKeyGen", aead_key_gen as NativeFn, Attributes::empty());
    ev.register("AeadEncrypt", aead_encrypt as NativeFn, Attributes::empty());
    ev.register("AeadDecrypt", aead_decrypt as NativeFn, Attributes::empty());
    // Signatures (Ed25519)
    ev.register("KeypairGenerate", keypair_generate as NativeFn, Attributes::empty());
    ev.register("Sign", sign_fn as NativeFn, Attributes::empty());
    ev.register("Verify", verify_fn as NativeFn, Attributes::empty());
    // HMAC
    ev.register("Hmac", hmac_fn as NativeFn, Attributes::empty());
    ev.register("HmacVerify", hmac_verify as NativeFn, Attributes::empty());
    // HKDF + Argon2id
    ev.register("Hkdf", hkdf_fn as NativeFn, Attributes::empty());
    ev.register("PasswordHash", password_hash_fn as NativeFn, Attributes::empty());
    ev.register("PasswordVerify", password_verify_fn as NativeFn, Attributes::empty());
    // JOSE (JWT)
    ev.register("JwtSign", jwt_sign as NativeFn, Attributes::empty());
    ev.register("JwtVerify", jwt_verify as NativeFn, Attributes::empty());
    // UUIDs
    ev.register("UuidV4", uuid_v4 as NativeFn, Attributes::empty());
    ev.register("UuidV7", uuid_v7 as NativeFn, Attributes::empty());

    #[cfg(feature = "tools")]
    add_specs(vec![
        tool_spec!("RandomBytes", summary: "Generate cryptographically secure random bytes", params: ["len","opts"], tags: ["crypto","random"], output_schema: schema_str!()),
        tool_spec!("RandomHex", summary: "Generate random hex string of n bytes", params: ["len"], tags: ["crypto","random"], output_schema: schema_str!()),
        tool_spec!("Hash", summary: "Compute digest (BLAKE3/SHA-256)", params: ["input","opts"], tags: ["crypto","hash"], output_schema: schema_str!()),
        tool_spec!("AeadKeyGen", summary: "Generate AEAD key (ChaCha20-Poly1305)", params: ["opts"], tags: ["crypto","aead"], output_schema: lyra_core::value::Value::Assoc(HashMap::from([(String::from("type"), lyra_core::value::Value::String(String::from("object")))]))),
        tool_spec!("AeadEncrypt", summary: "Encrypt with AEAD (ChaCha20-Poly1305)", params: ["plaintext","key","opts"], tags: ["crypto","aead"], output_schema: lyra_core::value::Value::Assoc(HashMap::from([(String::from("type"), lyra_core::value::Value::String(String::from("object")))]))),
        tool_spec!("AeadDecrypt", summary: "Decrypt AEAD envelope", params: ["envelope","key","opts"], tags: ["crypto","aead"], output_schema: schema_str!()),
        tool_spec!("KeypairGenerate", summary: "Generate signing keypair (Ed25519)", params: ["opts"], tags: ["crypto","sign"], output_schema: lyra_core::value::Value::Assoc(HashMap::from([(String::from("type"), lyra_core::value::Value::String(String::from("object")))]))),
        tool_spec!("Sign", summary: "Sign message (Ed25519)", params: ["message","secretKey","opts"], tags: ["crypto","sign"], output_schema: schema_str!()),
        tool_spec!("Verify", summary: "Verify signature (Ed25519)", params: ["message","signature","publicKey","opts"], tags: ["crypto","sign"], output_schema: schema_bool!()),
        tool_spec!("Hmac", summary: "HMAC (SHA-256 or SHA-512)", params: ["message","key","opts"], tags: ["crypto","mac"], output_schema: schema_str!()),
        tool_spec!("HmacVerify", summary: "Verify HMAC signature", params: ["message","key","signature","opts"], tags: ["crypto","mac"], output_schema: schema_bool!()),
        tool_spec!("Hkdf", summary: "HKDF (SHA-256 or SHA-512)", params: ["ikm","opts"], tags: ["crypto","kdf"]),
        tool_spec!("PasswordHash", summary: "Password hash with Argon2id (PHC string)", params: ["password","opts"], tags: ["crypto","kdf"]),
        tool_spec!("PasswordVerify", summary: "Verify Argon2id password hash", params: ["password","hash"], tags: ["crypto","kdf"], output_schema: schema_bool!()),
        tool_spec!("JwtSign", summary: "Sign JWT (HS256 or EdDSA)", params: ["claims","key","opts"], tags: ["crypto","jwt"]),
        tool_spec!("JwtVerify", summary: "Verify JWT and return claims", params: ["jwt","keys","opts"], tags: ["crypto","jwt"], output_schema: lyra_core::value::Value::Assoc(HashMap::from([(String::from("type"), lyra_core::value::Value::String(String::from("object")))]))),
    ]);
}

pub fn register_crypto_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    register_if(ev, pred, "RandomBytes", random_bytes as NativeFn, Attributes::empty());
    register_if(ev, pred, "RandomHex", random_hex as NativeFn, Attributes::empty());
    register_if(ev, pred, "Hash", hash_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "AeadKeyGen", aead_key_gen as NativeFn, Attributes::empty());
    register_if(ev, pred, "AeadEncrypt", aead_encrypt as NativeFn, Attributes::empty());
    register_if(ev, pred, "AeadDecrypt", aead_decrypt as NativeFn, Attributes::empty());
    register_if(ev, pred, "KeypairGenerate", keypair_generate as NativeFn, Attributes::empty());
    register_if(ev, pred, "Sign", sign_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Verify", verify_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Hmac", hmac_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "HmacVerify", hmac_verify as NativeFn, Attributes::empty());
    register_if(ev, pred, "Hkdf", hkdf_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "PasswordHash", password_hash_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "PasswordVerify", password_verify_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "JwtSign", jwt_sign as NativeFn, Attributes::empty());
    register_if(ev, pred, "JwtVerify", jwt_verify as NativeFn, Attributes::empty());
    register_if(ev, pred, "UuidV4", uuid_v4 as NativeFn, Attributes::empty());
    register_if(ev, pred, "UuidV7", uuid_v7 as NativeFn, Attributes::empty());
}

fn failure(tag: &str, msg: &str) -> Value {
    Value::Assoc(
        vec![
            ("message".to_string(), Value::String(msg.to_string())),
            ("tag".to_string(), Value::String(tag.to_string())),
        ]
        .into_iter()
        .collect(),
    )
}

fn as_string(ev: &mut Evaluator, v: Value) -> String {
    match ev.eval(v) {
        Value::String(s) | Value::Symbol(s) => s,
        other => format_value(&other),
    }
}

fn get_assoc(
    ev: &mut Evaluator,
    v: Option<Value>,
) -> Option<std::collections::HashMap<String, Value>> {
    v.and_then(|x| match ev.eval(x) {
        Value::Assoc(m) => Some(m),
        _ => None,
    })
}

fn base64url_encode(data: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(data)
}

fn base64url_decode(s: &str) -> Result<Vec<u8>, String> {
    use base64::Engine as _;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(s).map_err(|e| e.to_string())
}

fn hex_encode(data: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(data.len() * 2);
    for b in data {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

fn hex_decode(s: &str) -> Result<Vec<u8>, String> {
    if s.len() % 2 != 0 {
        return Err("hex length must be even".into());
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    for i in (0..s.len()).step_by(2) {
        let h = from_hex(bytes[i])?;
        let l = from_hex(bytes[i + 1])?;
        out.push((h << 4) | l);
    }
    Ok(out)
}

fn from_hex(c: u8) -> Result<u8, String> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => Err("invalid hex".into()),
    }
}

fn encode_bytes(data: &[u8], enc: &str) -> Value {
    match enc {
        "hex" => Value::String(hex_encode(data)),
        "base64url" | "base64" => Value::String(base64url_encode(data)),
        _ => Value::String(base64url_encode(data)),
    }
}

fn decode_bytes(s: &str, enc: &str) -> Result<Vec<u8>, String> {
    match enc {
        "hex" => hex_decode(s),
        "base64url" | "base64" | "b64" => base64url_decode(s),
        _ => base64url_decode(s),
    }
}

fn read_key_material(
    ev: &mut Evaluator,
    v: Value,
    enc_opt: Option<String>,
) -> Result<Vec<u8>, String> {
    let vv = ev.eval(v);
    match vv {
        Value::Assoc(m) => {
            if let Some(Value::String(k)) = m.get("Key") {
                let enc = m
                    .get("Encoding")
                    .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
                    .or(enc_opt)
                    .unwrap_or_else(|| "utf8".into());
                if enc.eq_ignore_ascii_case("utf8") {
                    Ok(k.as_bytes().to_vec())
                } else {
                    decode_bytes(k, &enc)
                }
            } else {
                Err("Missing Key".into())
            }
        }
        Value::String(s) | Value::Symbol(s) => {
            // Default treat as UTF-8 unless encoding specified
            let enc = enc_opt.unwrap_or_else(|| "utf8".into());
            if enc.eq_ignore_ascii_case("utf8") {
                Ok(s.into_bytes())
            } else {
                decode_bytes(&s, &enc)
            }
        }
        _ => Err("Invalid key format".into()),
    }
}

fn read_bytes_with_encoding(ev: &mut Evaluator, v: Value, enc: &str) -> Result<Vec<u8>, String> {
    match ev.eval(v) {
        Value::String(s) | Value::Symbol(s) => {
            if enc.eq_ignore_ascii_case("utf8") {
                Ok(s.into_bytes())
            } else {
                decode_bytes(&s, enc)
            }
        }
        _ => Err("Expected string".into()),
    }
}

// Randomness
fn random_bytes(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("RandomBytes".into())), args };
    }
    let n = match &args[0] {
        Value::Integer(i) if *i >= 0 => *i as usize,
        _ => return Value::Expr { head: Box::new(Value::Symbol("RandomBytes".into())), args },
    };
    let enc = get_assoc(_ev, args.get(1).cloned())
        .and_then(|m| {
            m.get("Encoding").and_then(|v| {
                if let Value::String(s) = v {
                    Some(s.clone())
                } else {
                    None
                }
            })
        })
        .unwrap_or_else(|| "base64url".into());
    let mut buf = vec![0u8; n];
    use rand::RngCore;
    let mut rng = rand::rngs::OsRng;
    rng.fill_bytes(&mut buf);
    encode_bytes(&buf, &enc)
}

fn random_hex(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("RandomHex".into())), args };
    }
    let n = match &args[0] {
        Value::Integer(i) if *i >= 0 => *i as usize,
        _ => return Value::Expr { head: Box::new(Value::Symbol("RandomHex".into())), args },
    };
    let mut buf = vec![0u8; n];
    use rand::RngCore;
    let mut rng = rand::rngs::OsRng;
    rng.fill_bytes(&mut buf);
    Value::String(hex_encode(&buf))
}

// Hashing
fn hash_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("Hash".into())), args };
    }
    let input = as_string(ev, args[0].clone());
    let opts = get_assoc(ev, args.get(1).cloned()).unwrap_or_default();
    let alg = opts
        .get("Alg")
        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
        .unwrap_or_else(|| "BLAKE3".into());
    let enc = opts
        .get("Encoding")
        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
        .unwrap_or_else(|| "hex".into());
    let bytes = input.as_bytes();
    let out = if alg.eq_ignore_ascii_case("BLAKE3") {
        blake3::hash(bytes).as_bytes().to_vec()
    } else if alg.eq_ignore_ascii_case("SHA-256") {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(bytes);
        h.finalize().to_vec()
    } else {
        return failure("Crypto::hash", &format!("Unsupported alg: {}", alg));
    };
    encode_bytes(&out, &enc)
}

// AEAD (ChaCha20-Poly1305)
fn aead_key_gen(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let opts = get_assoc(ev, args.get(0).cloned()).unwrap_or_default();
    let alg = opts
        .get("alg").or_else(|| opts.get("Alg"))
        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
        .unwrap_or_else(|| "ChaCha20-Poly1305".into());
    if alg != "ChaCha20-Poly1305" {
        return failure("Crypto::aead", &format!("Unsupported AEAD alg: {}", alg));
    }
    let mut key = [0u8; 32];
    use rand::RngCore;
    let mut rng = rand::rngs::OsRng;
    rng.fill_bytes(&mut key);
    Value::Assoc(
        vec![
            ("Alg".into(), Value::String(alg)),
            ("Key".into(), Value::String(base64url_encode(&key))),
        ]
        .into_iter()
        .collect(),
    )
}

fn read_key_bytes(ev: &mut Evaluator, v: Value) -> Result<(String, Vec<u8>), String> {
    let vv = ev.eval(v);
    match vv {
        Value::Assoc(m) => {
            let alg = m
                .get("alg").or_else(|| m.get("Alg"))
                .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
                .ok_or_else(|| "Missing Alg".to_string())?;
            let k = m
                .get("key").or_else(|| m.get("Key"))
                .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
                .ok_or_else(|| "Missing Key".to_string())?;
            let kb = base64url_decode(&k)?;
            Ok((alg, kb))
        }
        Value::String(s) | Value::Symbol(s) => {
            Ok(("ChaCha20-Poly1305".into(), base64url_decode(&s)?))
        }
        _ => Err("Invalid key format".into()),
    }
}

fn aead_encrypt(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("AeadEncrypt".into())), args };
    }
    let plaintext = as_string(ev, args[0].clone());
    let (alg, key) = match read_key_bytes(ev, args[1].clone()) {
        Ok(v) => v,
        Err(e) => return failure("Crypto::aead", &format!("Key: {}", e)),
    };
    if alg != "ChaCha20-Poly1305" {
        return failure("Crypto::aead", &format!("Unsupported AEAD alg: {}", alg));
    }
    let opts = get_assoc(ev, args.get(2).cloned()).unwrap_or_default();
    let aad = opts
        .get("aad").or_else(|| opts.get("AAD"))
        .and_then(|v| if let Value::String(s) = v { Some(s.as_bytes().to_vec()) } else { None })
        .unwrap_or_default();
    let nonce_bytes = if let Some(Value::String(nonce_s)) = opts.get("nonce").or_else(|| opts.get("Nonce")) {
        match base64url_decode(nonce_s) {
            Ok(b) => b,
            Err(e) => return failure("Crypto::aead", &format!("Nonce: {}", e)),
        }
    } else {
        let mut n = [0u8; 12];
        use rand::RngCore;
        let mut rng = rand::rngs::OsRng;
        rng.fill_bytes(&mut n);
        n.to_vec()
    };
    if nonce_bytes.len() != 12 {
        return failure("Crypto::aead", "Nonce must be 12 bytes (base64url)");
    }
    let cipher = match chacha20poly1305::ChaCha20Poly1305::new_from_slice(&key) {
        Ok(c) => c,
        Err(_) => return failure("Crypto::aead", "Invalid key length"),
    };
    use chacha20poly1305::aead::{Aead, Payload};
    let nonce = chacha20poly1305::Nonce::from_slice(&nonce_bytes);
    let ciphertext = match cipher.encrypt(nonce, Payload { msg: plaintext.as_bytes(), aad: &aad }) {
        Ok(ct) => ct,
        Err(_) => return failure("Crypto::aead", "Encryption failed"),
    };
    Value::Assoc(
        vec![
            ("Alg".into(), Value::String(alg)),
            ("Nonce".into(), Value::String(base64url_encode(&nonce_bytes))),
            ("Ciphertext".into(), Value::String(base64url_encode(&ciphertext))),
        ]
        .into_iter()
        .collect(),
    )
}

fn aead_decrypt(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("AeadDecrypt".into())), args };
    }
    let (alg, key) = match read_key_bytes(ev, args[1].clone()) {
        Ok(v) => v,
        Err(e) => return failure("Crypto::aead", &format!("Key: {}", e)),
    };
    if alg != "ChaCha20-Poly1305" {
        return failure("Crypto::aead", &format!("Unsupported AEAD alg: {}", alg));
    }
    let (nonce_b64, ct_b64, aad_bytes) = match ev.eval(args[0].clone()) {
        Value::Assoc(m) => {
            let n = m
                .get("Nonce")
                .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
                .unwrap_or_else(|| String::new());
            let c = m
                .get("Ciphertext")
                .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
                .unwrap_or_else(|| String::new());
            let aad = m
                .get("AAD")
                .and_then(
                    |v| if let Value::String(s) = v { Some(s.as_bytes().to_vec()) } else { None },
                )
                .unwrap_or_default();
            (n, c, aad)
        }
        Value::String(s) | Value::Symbol(s) => {
            // Expect opts to carry nonce and optional AAD
            let opts = get_assoc(ev, args.get(2).cloned()).unwrap_or_default();
            let n = opts
                .get("Nonce")
                .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
                .unwrap_or_else(|| String::new());
            let aad = opts
                .get("AAD")
                .and_then(
                    |v| if let Value::String(s) = v { Some(s.as_bytes().to_vec()) } else { None },
                )
                .unwrap_or_default();
            (n, s, aad)
        }
        _ => return failure("Crypto::aead", "Invalid envelope/ciphertext"),
    };
    let nonce_bytes = match base64url_decode(&nonce_b64) {
        Ok(b) => b,
        Err(e) => return failure("Crypto::aead", &format!("Nonce: {}", e)),
    };
    if nonce_bytes.len() != 12 {
        return failure("Crypto::aead", "Nonce must be 12 bytes (base64url)");
    }
    let ct = match base64url_decode(&ct_b64) {
        Ok(b) => b,
        Err(e) => return failure("Crypto::aead", &format!("Ciphertext: {}", e)),
    };
    let cipher = match chacha20poly1305::ChaCha20Poly1305::new_from_slice(&key) {
        Ok(c) => c,
        Err(_) => return failure("Crypto::aead", "Invalid key length"),
    };
    use chacha20poly1305::aead::{Aead, Payload};
    let nonce = chacha20poly1305::Nonce::from_slice(&nonce_bytes);
    match cipher.decrypt(nonce, Payload { msg: &ct, aad: &aad_bytes }) {
        Ok(pt) => match String::from_utf8(pt.clone()) {
            Ok(s) => Value::String(s),
            Err(_) => Value::Expr {
                head: Box::new(Value::Symbol("Binary".into())),
                args: vec![Value::String(base64url_encode(&pt))],
            },
        },
        Err(_) => failure("Crypto::aead", "Decryption failed"),
    }
}

// Signatures (Ed25519)
fn keypair_generate(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let opts = get_assoc(ev, args.get(0).cloned()).unwrap_or_default();
    let alg = opts
        .get("Alg")
        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
        .unwrap_or_else(|| "Ed25519".into());
    if alg != "Ed25519" {
        return failure("Crypto::sign", &format!("Unsupported signing alg: {}", alg));
    }
    use rand::rngs::OsRng;
    let mut csprng = OsRng;
    let sk = ed25519_dalek::SigningKey::generate(&mut csprng);
    let pk: ed25519_dalek::VerifyingKey = sk.verifying_key();
    Value::Assoc(
        vec![
            ("Alg".into(), Value::String(alg)),
            ("Public".into(), Value::String(base64url_encode(pk.as_bytes()))),
            ("Secret".into(), Value::String(base64url_encode(&sk.to_bytes()))),
        ]
        .into_iter()
        .collect(),
    )
}

fn read_signing_key(ev: &mut Evaluator, v: Value) -> Result<ed25519_dalek::SigningKey, String> {
    match ev.eval(v) {
        Value::Assoc(m) => {
            let alg = m
                .get("Alg")
                .and_then(|v| if let Value::String(s) = v { Some(s) } else { None })
                .ok_or_else(|| "Missing Alg".to_string())?;
            if alg != "Ed25519" {
                return Err("Unsupported signing alg".into());
            }
            let s = m
                .get("Secret")
                .and_then(|v| if let Value::String(s) = v { Some(s) } else { None })
                .ok_or_else(|| "Missing Secret".to_string())?;
            let b = base64url_decode(s)?;
            let arr: [u8; 32] = b.try_into().map_err(|_| "Bad secret length".to_string())?;
            Ok(ed25519_dalek::SigningKey::from_bytes(&arr))
        }
        Value::String(s) | Value::Symbol(s) => {
            let b = base64url_decode(&s)?;
            let arr: [u8; 32] = b.try_into().map_err(|_| "Bad secret length".to_string())?;
            Ok(ed25519_dalek::SigningKey::from_bytes(&arr))
        }
        _ => Err("Invalid secret key format".into()),
    }
}

fn read_verify_key(ev: &mut Evaluator, v: Value) -> Result<ed25519_dalek::VerifyingKey, String> {
    match ev.eval(v) {
        Value::Assoc(m) => {
            let alg = m
                .get("Alg")
                .and_then(|v| if let Value::String(s) = v { Some(s) } else { None })
                .ok_or_else(|| "Missing Alg".to_string())?;
            if alg != "Ed25519" {
                return Err("Unsupported signing alg".into());
            }
            let p = m
                .get("Public")
                .and_then(|v| if let Value::String(s) = v { Some(s) } else { None })
                .ok_or_else(|| "Missing Public".to_string())?;
            let b = base64url_decode(p)?;
            ed25519_dalek::VerifyingKey::from_bytes(&b.try_into().map_err(|_| "Bad public length")?)
                .map_err(|_| "Invalid public key".into())
        }
        Value::String(s) | Value::Symbol(s) => {
            let b = base64url_decode(&s)?;
            ed25519_dalek::VerifyingKey::from_bytes(&b.try_into().map_err(|_| "Bad public length")?)
                .map_err(|_| "Invalid public key".into())
        }
        _ => Err("Invalid public key format".into()),
    }
}

fn sign_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("Sign".into())), args };
    }
    let msg = as_string(ev, args[0].clone());
    let sk = match read_signing_key(ev, args[1].clone()) {
        Ok(k) => k,
        Err(e) => return failure("Crypto::sign", &e),
    };
    use ed25519_dalek::Signer;
    let sig = sk.sign(msg.as_bytes());
    Value::String(base64url_encode(&sig.to_bytes()))
}

fn verify_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 3 {
        return Value::Expr { head: Box::new(Value::Symbol("Verify".into())), args };
    }
    let msg = as_string(ev, args[0].clone());
    let sig_b64 = as_string(ev, args[1].clone());
    let pk = match read_verify_key(ev, args[2].clone()) {
        Ok(k) => k,
        Err(e) => return failure("Crypto::sign", &e),
    };
    let sig_bytes = match base64url_decode(&sig_b64) {
        Ok(b) => b,
        Err(e) => return failure("Crypto::sign", &format!("Signature: {}", e)),
    };
    let sig = match ed25519_dalek::Signature::from_slice(&sig_bytes) {
        Ok(s) => s,
        Err(_) => return failure("Crypto::sign", "Invalid signature bytes"),
    };
    use ed25519_dalek::Verifier;
    match pk.verify(msg.as_bytes(), &sig) {
        Ok(_) => Value::Boolean(true),
        Err(_) => Value::Boolean(false),
    }
}

// HMAC (SHA-256 / SHA-512)
fn hmac_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("Hmac".into())), args };
    }
    let msg = as_string(ev, args[0].clone());
    let opts = get_assoc(ev, args.get(2).cloned()).unwrap_or_default();
    let alg = opts
        .get("Alg")
        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
        .unwrap_or_else(|| "SHA-256".into());
    let out_enc = opts
        .get("Encoding")
        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
        .unwrap_or_else(|| "hex".into());
    let key_enc = opts.get("KeyEncoding").and_then(|v| {
        if let Value::String(s) = v {
            Some(s.clone())
        } else {
            None
        }
    });
    let key_bytes = match read_key_material(ev, args[1].clone(), key_enc) {
        Ok(b) => b,
        Err(e) => return failure("Crypto::hmac", &e),
    };
    let mac = if alg.eq_ignore_ascii_case("SHA-256") {
        use hmac::Hmac;
        use sha2::Sha256;
        let mut m = <Hmac<Sha256> as hmac::digest::KeyInit>::new_from_slice(&key_bytes)
            .map_err(|_| "Bad key")
            .unwrap();
        m.update(msg.as_bytes());
        m.finalize().into_bytes().to_vec()
    } else if alg.eq_ignore_ascii_case("SHA-512") {
        use hmac::Hmac;
        use sha2::Sha512;
        let mut m = <Hmac<Sha512> as hmac::digest::KeyInit>::new_from_slice(&key_bytes)
            .map_err(|_| "Bad key")
            .unwrap();
        m.update(msg.as_bytes());
        m.finalize().into_bytes().to_vec()
    } else {
        return failure("Crypto::hmac", &format!("Unsupported alg: {}", alg));
    };
    encode_bytes(&mac, &out_enc)
}

fn hmac_verify(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 3 {
        return Value::Expr { head: Box::new(Value::Symbol("HmacVerify".into())), args };
    }
    let msg = as_string(ev, args[0].clone());
    let opts = get_assoc(ev, args.get(3).cloned()).unwrap_or_default();
    let alg = opts
        .get("Alg")
        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
        .unwrap_or_else(|| "SHA-256".into());
    let key_enc = opts.get("KeyEncoding").and_then(|v| {
        if let Value::String(s) = v {
            Some(s.clone())
        } else {
            None
        }
    });
    let sig_enc = opts
        .get("InputEncoding")
        .or_else(|| opts.get("Encoding"))
        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
        .unwrap_or_else(|| "hex".into());
    let key_bytes = match read_key_material(ev, args[1].clone(), key_enc) {
        Ok(b) => b,
        Err(e) => return failure("Crypto::hmac", &e),
    };
    let sig_str = as_string(ev, args[2].clone());
    let sig = match decode_bytes(&sig_str, &sig_enc) {
        Ok(b) => b,
        Err(_) => return Value::Boolean(false),
    };
    let mac = if alg.eq_ignore_ascii_case("SHA-256") {
        use hmac::Hmac;
        use sha2::Sha256;
        let mut m = <Hmac<Sha256> as hmac::digest::KeyInit>::new_from_slice(&key_bytes)
            .map_err(|_| "Bad key")
            .unwrap();
        m.update(msg.as_bytes());
        m.finalize().into_bytes().to_vec()
    } else if alg.eq_ignore_ascii_case("SHA-512") {
        use hmac::Hmac;
        use sha2::Sha512;
        let mut m = <Hmac<Sha512> as hmac::digest::KeyInit>::new_from_slice(&key_bytes)
            .map_err(|_| "Bad key")
            .unwrap();
        m.update(msg.as_bytes());
        m.finalize().into_bytes().to_vec()
    } else {
        return Value::Boolean(false);
    };
    use subtle::ConstantTimeEq;
    Value::Boolean(mac.ct_eq(&sig).unwrap_u8() == 1)
}

// UUIDs
fn uuid_v4(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if !args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("UuidV4".into())), args };
    }
    let id = uuid::Uuid::new_v4();
    Value::String(id.to_string())
}

fn uuid_v7(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if !args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("UuidV7".into())), args };
    }
    let id = uuid::Uuid::now_v7();
    Value::String(id.to_string())
}

// HKDF (SHA-256 / SHA-512)
fn hkdf_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("Hkdf".into())), args };
    }
    let opts = get_assoc(ev, args.get(1).cloned()).unwrap_or_default();
    let hash = opts
        .get("Hash")
        .or_else(|| opts.get("Alg"))
        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
        .unwrap_or_else(|| "SHA-256".into());
    let out_enc = opts
        .get("Encoding")
        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
        .unwrap_or_else(|| "hex".into());
    let in_enc = opts
        .get("InputEncoding")
        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
        .unwrap_or_else(|| "utf8".into());
    let salt_enc = opts
        .get("SaltEncoding")
        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
        .unwrap_or_else(|| "hex".into());
    let info_enc = opts
        .get("InfoEncoding")
        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
        .unwrap_or_else(|| "hex".into());
    let length = opts
        .get("Length")
        .and_then(|v| if let Value::Integer(i) = v { Some(*i as usize) } else { None })
        .unwrap_or(32);
    let ikm = match read_bytes_with_encoding(ev, args[0].clone(), &in_enc) {
        Ok(b) => b,
        Err(e) => return failure("Crypto::hkdf", &format!("IKM: {}", e)),
    };
    let salt = if let Some(v) = opts.get("Salt").cloned() {
        match read_bytes_with_encoding(ev, v, &salt_enc) {
            Ok(b) => b,
            Err(e) => return failure("Crypto::hkdf", &format!("Salt: {}", e)),
        }
    } else {
        Vec::new()
    };
    let info = if let Some(v) = opts.get("Info").cloned() {
        match read_bytes_with_encoding(ev, v, &info_enc) {
            Ok(b) => b,
            Err(e) => return failure("Crypto::hkdf", &format!("Info: {}", e)),
        }
    } else {
        Vec::new()
    };
    if hash.eq_ignore_ascii_case("SHA-256") {
        use hkdf::Hkdf;
        use sha2::Sha256;
        let hk = Hkdf::<Sha256>::new(Some(&salt), &ikm);
        let mut okm = vec![0u8; length];
        if hk.expand(&info, &mut okm).is_err() {
            return failure("Crypto::hkdf", "Expand failed");
        }
        encode_bytes(&okm, &out_enc)
    } else if hash.eq_ignore_ascii_case("SHA-512") {
        use hkdf::Hkdf;
        use sha2::Sha512;
        let hk = Hkdf::<Sha512>::new(Some(&salt), &ikm);
        let mut okm = vec![0u8; length];
        if hk.expand(&info, &mut okm).is_err() {
            return failure("Crypto::hkdf", "Expand failed");
        }
        encode_bytes(&okm, &out_enc)
    } else {
        failure("Crypto::hkdf", &format!("Unsupported hash: {}", hash))
    }
}

// Argon2id password hashing (PHC string)
fn password_hash_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("PasswordHash".into())), args };
    }
    let password = as_string(ev, args[0].clone());
    let opts = get_assoc(ev, args.get(1).cloned()).unwrap_or_default();
    let mem_mib = opts
        .get("MemoryMiB")
        .and_then(|v| if let Value::Integer(i) = v { Some(*i as u32) } else { None })
        .unwrap_or(64);
    let iterations = opts
        .get("Iterations")
        .and_then(|v| if let Value::Integer(i) = v { Some(*i as u32) } else { None })
        .unwrap_or(3);
    let parallelism = opts
        .get("Parallelism")
        .and_then(|v| if let Value::Integer(i) = v { Some(*i as u32) } else { None })
        .unwrap_or(1);
    let salt_opt =
        opts.get("Salt")
            .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None });
    let salt_enc = opts
        .get("SaltEncoding")
        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
        .unwrap_or_else(|| "base64url".into());
    let salt_bytes = if let Some(salt) = salt_opt {
        match decode_bytes(&salt, &salt_enc) {
            Ok(b) => b,
            Err(e) => return failure("Crypto::argon", &format!("Salt: {}", e)),
        }
    } else {
        let mut s = [0u8; 16];
        use rand::RngCore;
        let mut rng = rand::rngs::OsRng;
        rng.fill_bytes(&mut s);
        s.to_vec()
    };
    use argon2::{
        password_hash::{PasswordHasher, SaltString},
        Algorithm, Argon2, Params, Version,
    };
    let params = match Params::new(mem_mib * 1024, iterations, parallelism, None) {
        Ok(p) => p,
        Err(_) => return failure("Crypto::argon", "Invalid params"),
    };
    let alg = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let salt = SaltString::encode_b64(&salt_bytes).map_err(|_| ()).unwrap();
    match alg.hash_password(password.as_bytes(), &salt) {
        Ok(ph) => Value::String(ph.to_string()),
        Err(_) => failure("Crypto::argon", "Hash failed"),
    }
}

fn password_verify_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("PasswordVerify".into())), args };
    }
    let password = as_string(ev, args[0].clone());
    let hash = as_string(ev, args[1].clone());
    use argon2::{
        password_hash::{PasswordHash, PasswordVerifier},
        Argon2,
    };
    match PasswordHash::new(&hash) {
        Ok(parsed) => {
            Value::Boolean(Argon2::default().verify_password(password.as_bytes(), &parsed).is_ok())
        }
        Err(_) => Value::Boolean(false),
    }
}

// Helpers: JSON conversion
fn value_to_json(ev: &mut Evaluator, v: Value) -> Result<serde_json::Value, String> {
    let vv = ev.eval(v);
    serde_json::to_value(&vv).map_err(|e| e.to_string())
}

fn json_to_value(j: &serde_json::Value) -> Value {
    match j {
        serde_json::Value::Null => Value::Symbol("Null".into()),
        serde_json::Value::Bool(b) => Value::Boolean(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Integer(i)
            } else if let Some(f) = n.as_f64() {
                Value::Real(f)
            } else {
                Value::String(n.to_string())
            }
        }
        serde_json::Value::String(s) => Value::String(s.clone()),
        serde_json::Value::Array(a) => Value::List(a.iter().map(json_to_value).collect()),
        serde_json::Value::Object(m) => {
            Value::Assoc(m.iter().map(|(k, v)| (k.clone(), json_to_value(v))).collect())
        }
    }
}

fn now_unix() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

// JWT Sign (HS256, EdDSA)
fn jwt_sign(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("JwtSign".into())), args };
    }
    let claims_json = match value_to_json(ev, args[0].clone()) {
        Ok(j) => j,
        Err(e) => return failure("Crypto::jwt", &format!("Claims: {}", e)),
    };
    let opts = get_assoc(ev, args.get(2).cloned()).unwrap_or_default();
    let alg = opts
        .get("alg").or_else(|| opts.get("Alg"))
        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
        .unwrap_or_else(|| "HS256".into());
    let kid = opts
        .get("kid").or_else(|| opts.get("Kid"))
        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None });
    let mut header = serde_json::json!({"alg": alg, "typ": "JWT"});
    if let Some(k) = kid {
        header["kid"] = serde_json::Value::String(k);
    }
    // Optionally add exp from ExpIn seconds
    if let Some(Value::Integer(secs)) = opts.get("expIn").or_else(|| opts.get("ExpIn")) {
        if *secs > 0 {
            if let serde_json::Value::Object(mut m) = claims_json.clone() {
                let _ = m.insert(
                    "exp".into(),
                    serde_json::Value::Number(serde_json::Number::from(now_unix() + *secs)),
                );
            }
        }
    }
    let header_b64 = base64url_encode(serde_json::to_string(&header).unwrap().as_bytes());
    let payload_b64 = base64url_encode(serde_json::to_vec(&claims_json).unwrap().as_slice());
    let signing_input = format!("{}.{}", header_b64, payload_b64);
    let sig = if alg.eq_ignore_ascii_case("HS256") {
        let key_enc = opts.get("keyEncoding").or_else(|| opts.get("KeyEncoding")).and_then(|v| {
            if let Value::String(s) = v {
                Some(s.clone())
            } else {
                None
            }
        });
        let key_bytes = match read_key_material(ev, args[1].clone(), key_enc) {
            Ok(b) => b,
            Err(e) => return failure("Crypto::jwt", &e),
        };
        use hmac::{Hmac, Mac};
        use sha2::Sha256;
        let mut m = <Hmac<Sha256> as hmac::digest::KeyInit>::new_from_slice(&key_bytes)
            .map_err(|_| "Bad key")
            .unwrap();
        m.update(signing_input.as_bytes());
        m.finalize().into_bytes().to_vec()
    } else if alg.eq_ignore_ascii_case("EdDSA") {
        let sk = match read_signing_key(ev, args[1].clone()) {
            Ok(k) => k,
            Err(e) => return failure("Crypto::jwt", &e),
        };
        use ed25519_dalek::Signer;
        sk.sign(signing_input.as_bytes()).to_bytes().to_vec()
    } else {
        return failure("Crypto::jwt", &format!("Unsupported alg: {}", alg));
    };
    Value::String(format!("{}.{}", signing_input, base64url_encode(&sig)))
}

// JWT Verify
fn jwt_verify(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("JwtVerify".into())), args };
    }
    let jwt = as_string(ev, args[0].clone());
    let opts = get_assoc(ev, args.get(2).cloned()).unwrap_or_default();
    let validate_time = opts
        .get("validateTime").or_else(|| opts.get("ValidateTime"))
        .and_then(|v| if let Value::Boolean(b) = v { Some(*b) } else { None })
        .unwrap_or(true);
    let audience = opts.get("audience").or_else(|| opts.get("Audience")).and_then(|v| {
        if let Value::String(s) = v {
            Some(s.clone())
        } else {
            None
        }
    });
    let issuer =
        opts.get("issuer").or_else(|| opts.get("Issuer"))
            .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None });
    let parts: Vec<&str> = jwt.split('.').collect();
    if parts.len() != 3 {
        return Value::Assoc(vec![("valid".into(), Value::Boolean(false))].into_iter().collect());
    }
    let header_bytes = match base64url_decode(parts[0]) {
        Ok(b) => b,
        Err(_) => {
            return Value::Assoc(
                vec![("valid".into(), Value::Boolean(false))].into_iter().collect(),
            )
        }
    };
    let payload_bytes = match base64url_decode(parts[1]) {
        Ok(b) => b,
        Err(_) => {
            return Value::Assoc(
                vec![("valid".into(), Value::Boolean(false))].into_iter().collect(),
            )
        }
    };
    let sig_bytes = match base64url_decode(parts[2]) {
        Ok(b) => b,
        Err(_) => {
            return Value::Assoc(
                vec![("valid".into(), Value::Boolean(false))].into_iter().collect(),
            )
        }
    };
    let header_json: serde_json::Value = match serde_json::from_slice(&header_bytes) {
        Ok(j) => j,
        Err(_) => {
            return Value::Assoc(
                vec![("valid".into(), Value::Boolean(false))].into_iter().collect(),
            )
        }
    };
    let alg = header_json.get("alg").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let signing_input = format!("{}.{}", parts[0], parts[1]);
    // keys can be single or list
    let key_list: Vec<Value> = match ev.eval(args[1].clone()) {
        Value::List(xs) => xs,
        v => vec![v],
    };
    let mut valid = false;
    if alg.eq_ignore_ascii_case("HS256") {
        for k in key_list.iter() {
            let key_bytes = match read_key_material(ev, k.clone(), None) {
                Ok(b) => b,
                Err(_) => continue,
            };
            use hmac::{Hmac, Mac};
            use sha2::Sha256;
            use subtle::ConstantTimeEq;
            let mut m = <Hmac<Sha256> as hmac::digest::KeyInit>::new_from_slice(&key_bytes)
                .map_err(|_| "Bad key")
                .unwrap();
            m.update(signing_input.as_bytes());
            let tag = m.finalize().into_bytes();
            if tag.ct_eq(&sig_bytes).unwrap_u8() == 1 {
                valid = true;
                break;
            }
        }
    } else if alg.eq_ignore_ascii_case("EdDSA") {
        for k in key_list.iter() {
            if let Ok(pk) = read_verify_key(ev, k.clone()) {
                use ed25519_dalek::{Signature, Verifier};
                if let Ok(sig) = Signature::from_slice(&sig_bytes) {
                    if pk.verify(signing_input.as_bytes(), &sig).is_ok() {
                        valid = true;
                        break;
                    }
                }
            }
        }
    } else {
        return Value::Assoc(vec![("valid".into(), Value::Boolean(false))].into_iter().collect());
    }
    if !valid {
        return Value::Assoc(vec![("valid".into(), Value::Boolean(false))].into_iter().collect());
    }
    // Time/claims validation
    let claims_json: serde_json::Value = match serde_json::from_slice(&payload_bytes) {
        Ok(j) => j,
        Err(_) => serde_json::Value::Null,
    };
    if validate_time {
        let now = now_unix();
        if let Some(exp) = claims_json.get("exp").and_then(|v| v.as_i64()) {
            if now > exp {
                return Value::Assoc(
                    vec![("valid".into(), Value::Boolean(false))].into_iter().collect(),
                );
            }
        }
        if let Some(nbf) = claims_json.get("nbf").and_then(|v| v.as_i64()) {
            if now < nbf {
                return Value::Assoc(
                    vec![("valid".into(), Value::Boolean(false))].into_iter().collect(),
                );
            }
        }
        if let Some(iat) = claims_json.get("iat").and_then(|v| v.as_i64()) {
            if now + 300 < iat {
                return Value::Assoc(
                    vec![("valid".into(), Value::Boolean(false))].into_iter().collect(),
                );
            }
        }
    }
    if let Some(aud) = audience {
        if claims_json.get("aud").and_then(|v| v.as_str()) != Some(aud.as_str()) {
            return Value::Assoc(
                vec![("valid".into(), Value::Boolean(false))].into_iter().collect(),
            );
        }
    }
    if let Some(iss) = issuer {
        if claims_json.get("iss").and_then(|v| v.as_str()) != Some(iss.as_str()) {
            return Value::Assoc(
                vec![("valid".into(), Value::Boolean(false))].into_iter().collect(),
            );
        }
    }
    Value::Assoc(
        vec![
            ("valid".into(), Value::Boolean(true)),
            ("header".into(), json_to_value(&header_json)),
            ("claims".into(), json_to_value(&claims_json)),
        ]
        .into_iter()
        .collect(),
    )
}
