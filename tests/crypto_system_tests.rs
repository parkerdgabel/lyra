//! Comprehensive tests for the Lyra Cryptography & Security System
//! 
//! This test file defines the expected behavior for all cryptographic functions
//! following TDD principles. Tests are organized by function category.

use lyra::vm::Value;
use lyra::stdlib::{StandardLibrary, crypto};

#[cfg(test)]
mod crypto_tests {
    use super::*;

    fn create_stdlib() -> StandardLibrary {
        StandardLibrary::new()
    }

    fn execute_crypto_function(func_name: &str, args: &[Value]) -> Result<Value, String> {
        match func_name {
            "Hash" => crypto::hash(args).map_err(|e| format!("{:?}", e)),
            "HMAC" => crypto::hmac_fn(args).map_err(|e| format!("{:?}", e)),
            "VerifyChecksum" => crypto::verify_checksum(args).map_err(|e| format!("{:?}", e)),
            "HashPassword" => crypto::hash_password(args).map_err(|e| format!("{:?}", e)),
            "AESGenerateKey" => crypto::aes_generate_key(args).map_err(|e| format!("{:?}", e)),
            "AESEncrypt" => crypto::aes_encrypt(args).map_err(|e| format!("{:?}", e)),
            "AESDecrypt" => crypto::aes_decrypt(args).map_err(|e| format!("{:?}", e)),
            "RandomBytes" => crypto::random_bytes(args).map_err(|e| format!("{:?}", e)),
            "RandomString" => crypto::random_string(args).map_err(|e| format!("{:?}", e)),
            "RandomUUID" => crypto::random_uuid(args).map_err(|e| format!("{:?}", e)),
            "RandomInteger" => crypto::random_integer(args).map_err(|e| format!("{:?}", e)),
            "RandomPassword" => crypto::random_password(args).map_err(|e| format!("{:?}", e)),
            "KeyDerive" => crypto::key_derive(args).map_err(|e| format!("{:?}", e)),
            "SecureCompare" => crypto::secure_compare(args).map_err(|e| format!("{:?}", e)),
            "ConstantTimeEquals" => crypto::constant_time_equals(args).map_err(|e| format!("{:?}", e)),
            "Base32Encode" => crypto::base32_encode(args).map_err(|e| format!("{:?}", e)),
            "Base32Decode" => crypto::base32_decode(args).map_err(|e| format!("{:?}", e)),
            "HexEncode" => crypto::hex_encode(args).map_err(|e| format!("{:?}", e)),
            "HexDecode" => crypto::hex_decode(args).map_err(|e| format!("{:?}", e)),
            _ => Err(format!("Unknown function: {}", func_name)),
        }
    }

    // ============================================================================
    // 1. CRYPTOGRAPHIC HASHING TESTS
    // ============================================================================

    #[test]
    fn test_hash_sha256() {
        let _stdlib = create_stdlib();
        
        // Test basic SHA256 hashing
        let result = execute_crypto_function("Hash", &[
            Value::String("Hello World".to_string()),
            Value::String("SHA256".to_string())
        ]);
        
        assert!(result.is_ok());
        if let Ok(Value::String(hash)) = result {
            assert_eq!(hash.len(), 64); // SHA256 produces 32 bytes = 64 hex chars
            assert_eq!(hash, "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e");
        } else {
            panic!("Expected string hash result");
        }
    }

    #[test]
    fn test_hash_multiple_algorithms() {
        let mut vm = create_test_vm();
        let test_data = "test data";

        // Test MD5
        let md5_result = execute_function(&mut vm, "Hash", vec![
            Value::String(test_data.to_string()),
            Value::String("MD5".to_string())
        ]);
        assert!(md5_result.is_ok());
        if let Ok(Value::String(hash)) = md5_result {
            assert_eq!(hash.len(), 32); // MD5 produces 16 bytes = 32 hex chars
        }

        // Test SHA1
        let sha1_result = execute_function(&mut vm, "Hash", vec![
            Value::String(test_data.to_string()),
            Value::String("SHA1".to_string())
        ]);
        assert!(sha1_result.is_ok());
        if let Ok(Value::String(hash)) = sha1_result {
            assert_eq!(hash.len(), 40); // SHA1 produces 20 bytes = 40 hex chars
        }

        // Test SHA3-256
        let sha3_result = execute_function(&mut vm, "Hash", vec![
            Value::String(test_data.to_string()),
            Value::String("SHA3-256".to_string())
        ]);
        assert!(sha3_result.is_ok());
        if let Ok(Value::String(hash)) = sha3_result {
            assert_eq!(hash.len(), 64); // SHA3-256 produces 32 bytes = 64 hex chars
        }

        // Test BLAKE3
        let blake3_result = execute_function(&mut vm, "Hash", vec![
            Value::String(test_data.to_string()),
            Value::String("BLAKE3".to_string())
        ]);
        assert!(blake3_result.is_ok());
        if let Ok(Value::String(hash)) = blake3_result {
            assert_eq!(hash.len(), 64); // BLAKE3 produces 32 bytes = 64 hex chars
        }
    }

    #[test]
    fn test_hmac_operations() {
        let mut vm = create_test_vm();
        
        let result = execute_function(&mut vm, "HMAC", vec![
            Value::String("message".to_string()),
            Value::String("secret_key".to_string()),
            Value::String("SHA256".to_string())
        ]);
        
        assert!(result.is_ok());
        if let Ok(Value::String(hmac)) = result {
            assert_eq!(hmac.len(), 64); // HMAC-SHA256 produces 32 bytes = 64 hex chars
            // HMAC should be deterministic for same input
            let result2 = execute_function(&mut vm, "HMAC", vec![
                Value::String("message".to_string()),
                Value::String("secret_key".to_string()),
                Value::String("SHA256".to_string())
            ]).unwrap();
            assert_eq!(result.unwrap(), result2);
        }
    }

    #[test]
    fn test_verify_checksum() {
        let mut vm = create_test_vm();
        let test_data = "test data";
        
        // First get the hash
        let hash_result = execute_function(&mut vm, "Hash", vec![
            Value::String(test_data.to_string()),
            Value::String("SHA256".to_string())
        ]).unwrap();
        
        let hash = match hash_result {
            Value::String(h) => h,
            _ => panic!("Expected hash string"),
        };
        
        // Verify the checksum matches
        let verify_result = execute_function(&mut vm, "VerifyChecksum", vec![
            Value::String(test_data.to_string()),
            Value::String(hash.clone()),
            Value::String("SHA256".to_string())
        ]);
        
        assert!(verify_result.is_ok());
        assert_eq!(verify_result.unwrap(), Value::Boolean(true));
        
        // Verify incorrect checksum fails
        let verify_fail_result = execute_function(&mut vm, "VerifyChecksum", vec![
            Value::String("different data".to_string()),
            Value::String(hash),
            Value::String("SHA256".to_string())
        ]);
        
        assert!(verify_fail_result.is_ok());
        assert_eq!(verify_fail_result.unwrap(), Value::Boolean(false));
    }

    #[test]
    fn test_password_hashing() {
        let mut vm = create_test_vm();
        
        let result = execute_function(&mut vm, "HashPassword", vec![
            Value::String("user_password_123".to_string()),
            Value::String("random_salt".to_string())
        ]);
        
        assert!(result.is_ok());
        if let Ok(Value::String(hashed)) = result {
            // Bcrypt hash should start with $2b$ or similar
            assert!(hashed.starts_with("$2"));
            assert!(hashed.len() >= 60); // Bcrypt hashes are typically 60 chars
        }
    }

    // ============================================================================
    // 2. SYMMETRIC ENCRYPTION TESTS
    // ============================================================================

    #[test]
    fn test_aes_key_generation() {
        let mut vm = create_test_vm();
        
        // Test 256-bit key generation
        let result = execute_function(&mut vm, "AESGenerateKey", vec![
            Value::Integer(256)
        ]);
        
        assert!(result.is_ok());
        // Key should be a Foreign object (CryptoKey)
        assert!(matches!(result.unwrap(), Value::LyObj(_)));
    }

    #[test]
    fn test_aes_encrypt_decrypt() {
        let mut vm = create_test_vm();
        let plaintext = "This is secret data";
        
        // Generate a key
        let key = execute_function(&mut vm, "AESGenerateKey", vec![
            Value::Integer(256)
        ]).unwrap();
        
        // Encrypt the data
        let encrypted = execute_function(&mut vm, "AESEncrypt", vec![
            Value::String(plaintext.to_string()),
            key.clone(),
            Value::String("GCM".to_string())
        ]);
        
        assert!(encrypted.is_ok());
        let encrypted_data = encrypted.unwrap();
        
        // Decrypt the data
        let decrypted = execute_function(&mut vm, "AESDecrypt", vec![
            encrypted_data,
            key,
            Value::String("GCM".to_string())
        ]);
        
        assert!(decrypted.is_ok());
        assert_eq!(decrypted.unwrap(), Value::String(plaintext.to_string()));
    }

    #[test]
    fn test_chacha20_operations() {
        let mut vm = create_test_vm();
        let plaintext = "ChaCha20 test data";
        
        // Generate key and nonce
        let key = execute_function(&mut vm, "RandomBytes", vec![Value::Integer(32)]).unwrap();
        let nonce = execute_function(&mut vm, "RandomBytes", vec![Value::Integer(12)]).unwrap();
        
        // Encrypt
        let encrypted = execute_function(&mut vm, "ChaCha20Encrypt", vec![
            Value::String(plaintext.to_string()),
            key.clone(),
            nonce.clone()
        ]);
        
        assert!(encrypted.is_ok());
        
        // Decrypt
        let decrypted = execute_function(&mut vm, "ChaCha20Decrypt", vec![
            encrypted.unwrap(),
            key,
            nonce
        ]);
        
        assert!(decrypted.is_ok());
        assert_eq!(decrypted.unwrap(), Value::String(plaintext.to_string()));
    }

    // ============================================================================
    // 3. ASYMMETRIC ENCRYPTION TESTS  
    // ============================================================================

    #[test]
    fn test_rsa_key_generation() {
        let mut vm = create_test_vm();
        
        let result = execute_function(&mut vm, "RSAGenerateKeys", vec![
            Value::Integer(2048)
        ]);
        
        assert!(result.is_ok());
        // Should return a dictionary with public and private keys
        if let Ok(Value::List(keys)) = result {
            assert_eq!(keys.len(), 2); // Should have public and private key
        }
    }

    #[test]
    fn test_rsa_encrypt_decrypt() {
        let mut vm = create_test_vm();
        let message = "RSA test message";
        
        // Generate key pair
        let keypair = execute_function(&mut vm, "RSAGenerateKeys", vec![
            Value::Integer(2048)
        ]).unwrap();
        
        let (public_key, private_key) = match keypair {
            Value::List(keys) => (keys[0].clone(), keys[1].clone()),
            _ => panic!("Expected key pair"),
        };
        
        // Encrypt with public key
        let encrypted = execute_function(&mut vm, "RSAEncrypt", vec![
            Value::String(message.to_string()),
            public_key
        ]);
        
        assert!(encrypted.is_ok());
        
        // Decrypt with private key
        let decrypted = execute_function(&mut vm, "RSADecrypt", vec![
            encrypted.unwrap(),
            private_key
        ]);
        
        assert!(decrypted.is_ok());
        assert_eq!(decrypted.unwrap(), Value::String(message.to_string()));
    }

    #[test]
    fn test_ecdsa_signatures() {
        let mut vm = create_test_vm();
        let message = "Message to sign";
        
        // Generate ECDSA key pair
        let keypair = execute_function(&mut vm, "ECDSAGenerateKeys", vec![
            Value::String("secp256r1".to_string())
        ]).unwrap();
        
        let (public_key, private_key) = match keypair {
            Value::List(keys) => (keys[0].clone(), keys[1].clone()),
            _ => panic!("Expected ECDSA key pair"),
        };
        
        // Sign the message
        let signature = execute_function(&mut vm, "ECDSASign", vec![
            Value::String(message.to_string()),
            private_key
        ]);
        
        assert!(signature.is_ok());
        
        // Verify the signature
        let verification = execute_function(&mut vm, "ECDSAVerify", vec![
            Value::String(message.to_string()),
            signature.unwrap(),
            public_key
        ]);
        
        assert!(verification.is_ok());
        assert_eq!(verification.unwrap(), Value::Boolean(true));
    }

    // ============================================================================
    // 4. RANDOM GENERATION TESTS
    // ============================================================================

    #[test]
    fn test_random_bytes() {
        let mut vm = create_test_vm();
        
        let result = execute_function(&mut vm, "RandomBytes", vec![Value::Integer(32)]);
        assert!(result.is_ok());
        
        // Should return binary data (represented as hex string or bytes)
        if let Ok(value) = result {
            assert!(matches!(value, Value::String(_) | Value::LyObj(_)));
        }
    }

    #[test]
    fn test_random_string() {
        let mut vm = create_test_vm();
        
        let result = execute_function(&mut vm, "RandomString", vec![
            Value::Integer(16),
            Value::String("alphanumeric".to_string())
        ]);
        
        assert!(result.is_ok());
        if let Ok(Value::String(random_str)) = result {
            assert_eq!(random_str.len(), 16);
            // Should only contain alphanumeric characters
            assert!(random_str.chars().all(|c| c.is_alphanumeric()));
        }
    }

    #[test]
    fn test_random_uuid() {
        let mut vm = create_test_vm();
        
        let result = execute_function(&mut vm, "RandomUUID", vec![]);
        assert!(result.is_ok());
        
        if let Ok(Value::String(uuid)) = result {
            // UUID v4 format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
            assert_eq!(uuid.len(), 36);
            assert_eq!(uuid.chars().nth(8).unwrap(), '-');
            assert_eq!(uuid.chars().nth(13).unwrap(), '-');
            assert_eq!(uuid.chars().nth(14).unwrap(), '4'); // Version 4
            assert_eq!(uuid.chars().nth(18).unwrap(), '-');
            assert_eq!(uuid.chars().nth(23).unwrap(), '-');
        }
    }

    #[test]
    fn test_random_integer() {
        let mut vm = create_test_vm();
        
        let result = execute_function(&mut vm, "RandomInteger", vec![
            Value::Integer(1),
            Value::Integer(100)
        ]);
        
        assert!(result.is_ok());
        if let Ok(Value::Integer(rand_int)) = result {
            assert!(rand_int >= 1 && rand_int <= 100);
        }
    }

    #[test]
    fn test_random_password() {
        let mut vm = create_test_vm();
        
        let result = execute_function(&mut vm, "RandomPassword", vec![
            Value::Integer(20),
            Value::String("strong".to_string())
        ]);
        
        assert!(result.is_ok());
        if let Ok(Value::String(password)) = result {
            assert_eq!(password.len(), 20);
            // Strong password should contain mix of chars
            let has_upper = password.chars().any(|c| c.is_uppercase());
            let has_lower = password.chars().any(|c| c.is_lowercase());
            let has_digit = password.chars().any(|c| c.is_digit(10));
            let has_special = password.chars().any(|c| "!@#$%^&*()_+-=[]{}|;':\",./<>?".contains(c));
            
            assert!(has_upper && has_lower && has_digit && has_special);
        }
    }

    // ============================================================================
    // 5. KEY MANAGEMENT TESTS
    // ============================================================================

    #[test]
    fn test_key_derivation() {
        let mut vm = create_test_vm();
        
        let result = execute_function(&mut vm, "KeyDerive", vec![
            Value::String("password123".to_string()),
            Value::String("salt_value".to_string()),
            Value::String("PBKDF2".to_string())
        ]);
        
        assert!(result.is_ok());
        // Should return derived key as binary data
        assert!(matches!(result.unwrap(), Value::String(_) | Value::LyObj(_)));
    }

    #[test]
    fn test_secure_compare() {
        let mut vm = create_test_vm();
        
        // Test equal strings
        let equal_result = execute_function(&mut vm, "SecureCompare", vec![
            Value::String("secret123".to_string()),
            Value::String("secret123".to_string())
        ]);
        
        assert!(equal_result.is_ok());
        assert_eq!(equal_result.unwrap(), Value::Boolean(true));
        
        // Test different strings
        let different_result = execute_function(&mut vm, "SecureCompare", vec![
            Value::String("secret123".to_string()),
            Value::String("secret456".to_string())
        ]);
        
        assert!(different_result.is_ok());
        assert_eq!(different_result.unwrap(), Value::Boolean(false));
    }

    #[test]
    fn test_constant_time_equals() {
        let mut vm = create_test_vm();
        
        let result = execute_function(&mut vm, "ConstantTimeEquals", vec![
            Value::String("test_data".to_string()),
            Value::String("test_data".to_string())
        ]);
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Boolean(true));
    }

    // ============================================================================
    // 6. ENCODING & UTILITY TESTS
    // ============================================================================

    #[test]
    fn test_base32_operations() {
        let mut vm = create_test_vm();
        let test_data = "Hello, World!";
        
        // Encode
        let encoded = execute_function(&mut vm, "Base32Encode", vec![
            Value::String(test_data.to_string())
        ]);
        
        assert!(encoded.is_ok());
        if let Ok(Value::String(encoded_str)) = encoded {
            // Decode
            let decoded = execute_function(&mut vm, "Base32Decode", vec![
                Value::String(encoded_str)
            ]);
            
            assert!(decoded.is_ok());
            assert_eq!(decoded.unwrap(), Value::String(test_data.to_string()));
        }
    }

    #[test]
    fn test_hex_operations() {
        let mut vm = create_test_vm();
        let test_data = "Hello, World!";
        
        // Encode to hex
        let encoded = execute_function(&mut vm, "HexEncode", vec![
            Value::String(test_data.to_string())
        ]);
        
        assert!(encoded.is_ok());
        if let Ok(Value::String(hex_str)) = encoded {
            assert_eq!(hex_str, "48656c6c6f2c20576f726c6421");
            
            // Decode from hex
            let decoded = execute_function(&mut vm, "HexDecode", vec![
                Value::String(hex_str)
            ]);
            
            assert!(decoded.is_ok());
            assert_eq!(decoded.unwrap(), Value::String(test_data.to_string()));
        }
    }

    // ============================================================================
    // 7. ERROR HANDLING TESTS
    // ============================================================================

    #[test]
    fn test_invalid_hash_algorithm() {
        let mut vm = create_test_vm();
        
        let result = execute_function(&mut vm, "Hash", vec![
            Value::String("test".to_string()),
            Value::String("INVALID_ALGORITHM".to_string())
        ]);
        
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unsupported hash algorithm"));
    }

    #[test]
    fn test_invalid_aes_key_size() {
        let mut vm = create_test_vm();
        
        let result = execute_function(&mut vm, "AESGenerateKey", vec![
            Value::Integer(123) // Invalid key size
        ]);
        
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid AES key size"));
    }

    #[test]
    fn test_rsa_key_size_validation() {
        let mut vm = create_test_vm();
        
        let result = execute_function(&mut vm, "RSAGenerateKeys", vec![
            Value::Integer(512) // Too small for security
        ]);
        
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("RSA key size too small"));
    }

    // ============================================================================
    // 8. INTEGRATION TESTS
    // ============================================================================

    #[test]
    fn test_crypto_workflow() {
        let mut vm = create_test_vm();
        
        // 1. Generate random password
        let password = execute_function(&mut vm, "RandomPassword", vec![
            Value::Integer(16),
            Value::String("strong".to_string())
        ]).unwrap();
        
        // 2. Hash the password
        let salt = execute_function(&mut vm, "RandomBytes", vec![Value::Integer(16)]).unwrap();
        let hashed_password = execute_function(&mut vm, "HashPassword", vec![
            password.clone(),
            salt
        ]);
        assert!(hashed_password.is_ok());
        
        // 3. Generate AES key
        let aes_key = execute_function(&mut vm, "AESGenerateKey", vec![
            Value::Integer(256)
        ]).unwrap();
        
        // 4. Encrypt sensitive data
        let sensitive_data = "This is very sensitive information";
        let encrypted = execute_function(&mut vm, "AESEncrypt", vec![
            Value::String(sensitive_data.to_string()),
            aes_key.clone(),
            Value::String("GCM".to_string())
        ]);
        assert!(encrypted.is_ok());
        
        // 5. Generate RSA keys for key exchange
        let rsa_keys = execute_function(&mut vm, "RSAGenerateKeys", vec![
            Value::Integer(2048)
        ]);
        assert!(rsa_keys.is_ok());
        
        // 6. Generate UUID for session
        let session_id = execute_function(&mut vm, "RandomUUID", vec![]);
        assert!(session_id.is_ok());
        
        // All operations should complete successfully
        assert!(true);
    }

    #[test]
    fn test_file_checksum_verification() {
        let mut vm = create_test_vm();
        
        // Create temporary file content
        let file_content = "This is file content for checksum testing";
        
        // Calculate checksum using Hash function (simulating file read)
        let checksum = execute_function(&mut vm, "Hash", vec![
            Value::String(file_content.to_string()),
            Value::String("SHA256".to_string())
        ]).unwrap();
        
        // Verify checksum
        let verification = execute_function(&mut vm, "VerifyChecksum", vec![
            Value::String(file_content.to_string()),
            checksum,
            Value::String("SHA256".to_string())
        ]);
        
        assert!(verification.is_ok());
        assert_eq!(verification.unwrap(), Value::Boolean(true));
    }
}