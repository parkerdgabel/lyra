#![cfg(feature = "crypto")]

use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;

#[test]
fn crypto_random_and_hash() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let rb = ev.eval(Value::expr(Value::Symbol("RandomBytes".into()), vec![Value::Integer(16)]));
    assert!(matches!(rb, Value::String(ref s) if !s.is_empty()));
    let rh = ev.eval(Value::expr(Value::Symbol("RandomHex".into()), vec![Value::Integer(8)]));
    assert!(matches!(rh, Value::String(ref s) if s.len()==16));
    let h = ev.eval(Value::expr(Value::Symbol("Hash".into()), vec![Value::String("abc".into())]));
    assert!(matches!(h, Value::String(ref s) if s.len()>=32));
}

#[test]
fn crypto_aead_roundtrip() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let key = ev.eval(Value::expr(Value::Symbol("AeadKeyGen".into()), vec![]));
    let pt = Value::String("hello world".into());
    let env =
        ev.eval(Value::expr(Value::Symbol("AeadEncrypt".into()), vec![pt.clone(), key.clone()]));
    let dec = ev.eval(Value::expr(Value::Symbol("AeadDecrypt".into()), vec![env, key]));
    assert_eq!(dec, pt);
}

#[test]
fn crypto_sign_verify() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let kp = ev.eval(Value::expr(Value::Symbol("KeypairGenerate".into()), vec![]));
    let msg = Value::String("test message".into());
    let sig = ev.eval(Value::expr(Value::Symbol("Sign".into()), vec![msg.clone(), kp.clone()]));
    let valid = ev.eval(Value::expr(
        Value::Symbol("Verify".into()),
        vec![msg.clone(), sig.clone(), kp.clone()],
    ));
    assert_eq!(valid, Value::Boolean(true));
    let invalid = ev.eval(Value::expr(
        Value::Symbol("Verify".into()),
        vec![Value::String("tampered".into()), sig, kp],
    ));
    assert_eq!(invalid, Value::Boolean(false));
}

#[test]
fn crypto_hmac() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // Test vector: HMAC-SHA256("The quick brown fox jumps over the lazy dog", key="key")
    let msg = Value::String("The quick brown fox jumps over the lazy dog".into());
    let key = Value::String("key".into());
    let mac = ev.eval(Value::expr(
        Value::Symbol("Hmac".into()),
        vec![
            msg.clone(),
            key.clone(),
            Value::Assoc(
                [
                    ("Alg".into(), Value::String("SHA-256".into())),
                    ("Encoding".into(), Value::String("hex".into())),
                ]
                .into_iter()
                .collect(),
            ),
        ],
    ));
    let mac_hex = if let Value::String(s) = mac { s } else { panic!("Hmac output not string") };
    assert_eq!(mac_hex, "f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8");

    // Verify good and bad
    let ok = ev.eval(Value::expr(
        Value::Symbol("HmacVerify".into()),
        vec![
            msg.clone(),
            Value::String("key".into()),
            Value::String(mac_hex.clone()),
            Value::Assoc(
                [
                    ("Alg".into(), Value::String("SHA-256".into())),
                    ("InputEncoding".into(), Value::String("hex".into())),
                ]
                .into_iter()
                .collect(),
            ),
        ],
    ));
    assert_eq!(ok, Value::Boolean(true));
    let bad = ev.eval(Value::expr(
        Value::Symbol("HmacVerify".into()),
        vec![
            msg,
            Value::String("key".into()),
            Value::String("00".into()),
            Value::Assoc(
                [
                    ("Alg".into(), Value::String("SHA-256".into())),
                    ("InputEncoding".into(), Value::String("hex".into())),
                ]
                .into_iter()
                .collect(),
            ),
        ],
    ));
    assert_eq!(bad, Value::Boolean(false));
}

#[test]
fn crypto_hkdf_and_argon() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // RFC 5869 Test Case 1 (SHA-256)
    // ikm = 0x0b repeated 22, salt = 0x000102...0x0c, info = 0xf0f1...0xf9, L=42
    let ikm = Value::String("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b".into());
    let salt = Value::String("000102030405060708090a0b0c".into());
    let info = Value::String("f0f1f2f3f4f5f6f7f8f9".into());
    let okm = ev.eval(Value::expr(
        Value::Symbol("Hkdf".into()),
        vec![
            ikm,
            Value::Assoc(
                [
                    ("Hash".into(), Value::String("SHA-256".into())),
                    ("InputEncoding".into(), Value::String("hex".into())),
                    ("Salt".into(), salt),
                    ("SaltEncoding".into(), Value::String("hex".into())),
                    ("Info".into(), info),
                    ("InfoEncoding".into(), Value::String("hex".into())),
                    ("Length".into(), Value::Integer(42)),
                    ("Encoding".into(), Value::String("hex".into())),
                ]
                .into_iter()
                .collect(),
            ),
        ],
    ));
    let okm_hex = if let Value::String(s) = okm { s } else { panic!("Hkdf output not string") };
    assert_eq!(
        okm_hex,
        "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"
    );

    // Argon2id password hash/verify roundtrip
    let ph = ev.eval(Value::expr(
        Value::Symbol("PasswordHash".into()),
        vec![
            Value::String("s3cret".into()),
            Value::Assoc(
                [
                    ("MemoryMiB".into(), Value::Integer(32)),
                    ("Iterations".into(), Value::Integer(2)),
                    ("Parallelism".into(), Value::Integer(1)),
                ]
                .into_iter()
                .collect(),
            ),
        ],
    ));
    let phs = if let Value::String(s) = ph { s } else { panic!("PasswordHash output not string") };
    let ok = ev.eval(Value::expr(
        Value::Symbol("PasswordVerify".into()),
        vec![Value::String("s3cret".into()), Value::String(phs.clone())],
    ));
    assert_eq!(ok, Value::Boolean(true));
    let bad = ev.eval(Value::expr(
        Value::Symbol("PasswordVerify".into()),
        vec![Value::String("wrong".into()), Value::String(phs)],
    ));
    assert_eq!(bad, Value::Boolean(false));
}

#[test]
fn crypto_jwt_hs256_and_eddsa() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // HS256 roundtrip
    let claims = Value::Assoc(
        [
            ("sub".into(), Value::String("1234567890".into())),
            ("name".into(), Value::String("John Doe".into())),
            ("admin".into(), Value::Boolean(true)),
        ]
        .into_iter()
        .collect(),
    );
    let jwt = ev.eval(Value::expr(
        Value::Symbol("JwtSign".into()),
        vec![
            claims.clone(),
            Value::String("secret".into()),
            Value::Assoc([("Alg".into(), Value::String("HS256".into()))].into_iter().collect()),
        ],
    ));
    let jwt_s = if let Value::String(s) = jwt { s } else { panic!("JwtSign output not string") };
    let ver = ev.eval(Value::expr(
        Value::Symbol("JwtVerify".into()),
        vec![
            Value::String(jwt_s.clone()),
            Value::String("secret".into()),
            Value::Assoc([("ValidateTime".into(), Value::Boolean(false))].into_iter().collect()),
        ],
    ));
    if let Value::Assoc(m) = ver {
        assert_eq!(m.get("valid"), Some(&Value::Boolean(true)));
    } else {
        panic!("JwtVerify result");
    }
    // EdDSA roundtrip
    let kp = ev.eval(Value::expr(Value::Symbol("KeypairGenerate".into()), vec![]));
    let jwt2 = ev.eval(Value::expr(
        Value::Symbol("JwtSign".into()),
        vec![
            claims,
            kp.clone(),
            Value::Assoc([("Alg".into(), Value::String("EdDSA".into()))].into_iter().collect()),
        ],
    ));
    let jwt2s = if let Value::String(s) = jwt2 { s } else { panic!("JwtSign output not string") };
    let ver2 = ev.eval(Value::expr(
        Value::Symbol("JwtVerify".into()),
        vec![
            Value::String(jwt2s),
            kp,
            Value::Assoc([("ValidateTime".into(), Value::Boolean(false))].into_iter().collect()),
        ],
    ));
    if let Value::Assoc(m) = ver2 {
        assert_eq!(m.get("valid"), Some(&Value::Boolean(true)));
    } else {
        panic!("JwtVerify result")
    }
}
