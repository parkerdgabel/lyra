# CRYPTO

| Function | Usage | Summary |
|---|---|---|
| `AeadDecrypt` | `AeadDecrypt[envelope, key, opts]` | Decrypt AEAD envelope |
| `AeadEncrypt` | `AeadEncrypt[plaintext, key, opts]` | Encrypt with AEAD (ChaCha20-Poly1305) |
| `AeadKeyGen` | `AeadKeyGen[opts]` | Generate AEAD key (ChaCha20-Poly1305) |
| `Hash` | `Hash[input, opts]` | Compute digest (BLAKE3/SHA-256) |
| `Hkdf` | `Hkdf[ikm, opts]` | HKDF (SHA-256 or SHA-512) |
| `Hmac` | `Hmac[message, key, opts]` | HMAC (SHA-256 or SHA-512) |
| `HmacVerify` | `HmacVerify[message, key, signature, opts]` | Verify HMAC signature |
| `JwtSign` | `JwtSign[claims, key, opts]` | Sign JWT (HS256 or EdDSA) |
| `JwtVerify` | `JwtVerify[jwt, keys, opts]` | Verify JWT and return claims |
| `KeypairGenerate` | `KeypairGenerate[opts]` | Generate signing keypair (Ed25519) |
| `PasswordHash` | `PasswordHash[password, opts]` | Password hash with Argon2id (PHC string) |
| `PasswordVerify` | `PasswordVerify[password, hash]` | Verify Argon2id password hash |
| `RandomBytes` | `RandomBytes[len, opts]` | Generate cryptographically secure random bytes |
| `RandomHex` | `RandomHex[len]` | Generate random hex string of n bytes |
| `Sign` | `Sign[message, secretKey, opts]` | Sign message (Ed25519) |
| `Verify` | `Verify[message, signature, publicKey, opts]` | Verify signature (Ed25519) |
