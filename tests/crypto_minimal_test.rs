//! Minimal crypto test to verify basic functionality

use lyra::vm::Value;
use lyra::stdlib::crypto;

#[cfg(test)]
mod crypto_minimal_tests {
    use super::*;

    #[test]
    fn test_hash_sha256_basic() {
        let result = crypto::hash(&[
            Value::String("Hello World".to_string()),
            Value::String("SHA256".to_string())
        ]);
        
        assert!(result.is_ok());
        if let Ok(Value::String(hash)) = result {
            assert_eq!(hash.len(), 64); // SHA256 produces 32 bytes = 64 hex chars
            // Known SHA256 hash of "Hello World"
            assert_eq!(hash, "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e");
        } else {
            panic!("Expected string hash result");
        }
    }

    #[test]
    fn test_hmac_sha256() {
        let result = crypto::hmac_fn(&[
            Value::String("message".to_string()),
            Value::String("secret_key".to_string()),
            Value::String("SHA256".to_string())
        ]);
        
        assert!(result.is_ok());
        if let Ok(Value::String(hmac)) = result {
            assert_eq!(hmac.len(), 64); // HMAC-SHA256 produces 32 bytes = 64 hex chars
            // HMAC should be deterministic for same input
            let result2 = crypto::hmac_fn(&[
                Value::String("message".to_string()),
                Value::String("secret_key".to_string()),
                Value::String("SHA256".to_string())
            ]).unwrap();
            assert_eq!(Value::String(hmac), result2);
        }
    }

    #[test]
    fn test_verify_checksum() {
        let test_data = "test data";
        
        // First get the hash
        let hash_result = crypto::hash(&[
            Value::String(test_data.to_string()),
            Value::String("SHA256".to_string())
        ]).unwrap();
        
        let hash = match hash_result {
            Value::String(h) => h,
            _ => panic!("Expected hash string"),
        };
        
        // Verify the checksum matches
        let verify_result = crypto::verify_checksum(&[
            Value::String(test_data.to_string()),
            Value::String(hash.clone()),
            Value::String("SHA256".to_string())
        ]);
        
        assert!(verify_result.is_ok());
        assert_eq!(verify_result.unwrap(), Value::Boolean(true));
        
        // Verify incorrect checksum fails
        let verify_fail_result = crypto::verify_checksum(&[
            Value::String("different data".to_string()),
            Value::String(hash),
            Value::String("SHA256".to_string())
        ]);
        
        assert!(verify_fail_result.is_ok());
        assert_eq!(verify_fail_result.unwrap(), Value::Boolean(false));
    }

    #[test]
    fn test_random_bytes() {
        let result = crypto::random_bytes(&[Value::Integer(32)]);
        assert!(result.is_ok());
        
        if let Ok(Value::String(hex_bytes)) = result {
            assert_eq!(hex_bytes.len(), 64); // 32 bytes = 64 hex chars
            // Should be different each time
            let result2 = crypto::random_bytes(&[Value::Integer(32)]).unwrap();
            if let Value::String(hex_bytes2) = result2 {
                assert_ne!(hex_bytes, hex_bytes2);
            }
        }
    }

    #[test]
    fn test_random_string() {
        let result = crypto::random_string(&[
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
        let result = crypto::random_uuid(&[]);
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
        let result = crypto::random_integer(&[
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
        let result = crypto::random_password(&[
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

    #[test]
    fn test_hex_encode_decode() {
        let test_data = "Hello, World!";
        
        // Encode to hex
        let encoded = crypto::hex_encode(&[Value::String(test_data.to_string())]);
        assert!(encoded.is_ok());
        
        if let Ok(Value::String(hex_str)) = encoded {
            assert_eq!(hex_str, "48656c6c6f2c20576f726c6421");
            
            // Decode from hex
            let decoded = crypto::hex_decode(&[Value::String(hex_str)]);
            assert!(decoded.is_ok());
            assert_eq!(decoded.unwrap(), Value::String(test_data.to_string()));
        }
    }

    #[test]
    fn test_base32_encode_decode() {
        let test_data = "Hello, World!";
        
        // Encode
        let encoded = crypto::base32_encode(&[Value::String(test_data.to_string())]);
        assert!(encoded.is_ok());
        
        if let Ok(Value::String(encoded_str)) = encoded {
            // Decode
            let decoded = crypto::base32_decode(&[Value::String(encoded_str)]);
            assert!(decoded.is_ok());
            assert_eq!(decoded.unwrap(), Value::String(test_data.to_string()));
        }
    }

    #[test]
    fn test_secure_compare() {
        // Test equal strings
        let equal_result = crypto::secure_compare(&[
            Value::String("secret123".to_string()),
            Value::String("secret123".to_string())
        ]);
        
        assert!(equal_result.is_ok());
        assert_eq!(equal_result.unwrap(), Value::Boolean(true));
        
        // Test different strings
        let different_result = crypto::secure_compare(&[
            Value::String("secret123".to_string()),
            Value::String("secret456".to_string())
        ]);
        
        assert!(different_result.is_ok());
        assert_eq!(different_result.unwrap(), Value::Boolean(false));
    }

    #[test]
    fn test_key_derive_pbkdf2() {
        let result = crypto::key_derive(&[
            Value::String("password123".to_string()),
            Value::String("salt_value".to_string()),
            Value::String("PBKDF2".to_string())
        ]);
        
        assert!(result.is_ok());
        if let Ok(Value::String(derived_key)) = result {
            assert_eq!(derived_key.len(), 64); // 32 bytes = 64 hex chars
            // Should be deterministic for same inputs
            let result2 = crypto::key_derive(&[
                Value::String("password123".to_string()),
                Value::String("salt_value".to_string()),
                Value::String("PBKDF2".to_string())
            ]).unwrap();
            assert_eq!(Value::String(derived_key), result2);
        }
    }

    #[test]
    fn test_aes_key_generation() {
        let result = crypto::aes_generate_key(&[Value::Integer(256)]);
        assert!(result.is_ok());
        
        // Key should be a Foreign object (CryptoKey)
        assert!(matches!(result.unwrap(), Value::LyObj(_)));
    }

    #[test]  
    fn test_aes_encrypt_decrypt() {
        let plaintext = "This is secret data";
        
        // Generate a key
        let key = crypto::aes_generate_key(&[Value::Integer(256)]).unwrap();
        
        // Encrypt the data
        let encrypted = crypto::aes_encrypt(&[
            Value::String(plaintext.to_string()),
            key.clone(),
            Value::String("GCM".to_string())
        ]);
        
        assert!(encrypted.is_ok());
        let encrypted_data = encrypted.unwrap();
        
        // Decrypt the data
        let decrypted = crypto::aes_decrypt(&[
            encrypted_data,
            key,
            Value::String("GCM".to_string())
        ]);
        
        assert!(decrypted.is_ok());
        assert_eq!(decrypted.unwrap(), Value::String(plaintext.to_string()));
    }

    #[test]
    fn test_hash_password() {
        let result = crypto::hash_password(&[
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

    // Error handling tests
    #[test]
    fn test_invalid_hash_algorithm() {
        let result = crypto::hash(&[
            Value::String("test".to_string()),
            Value::String("INVALID_ALGORITHM".to_string())
        ]);
        
        assert!(result.is_err());
        let error_msg = format!("{:?}", result.unwrap_err());
        assert!(error_msg.contains("Unsupported hash algorithm"));
    }

    #[test]
    fn test_invalid_aes_key_size() {
        let result = crypto::aes_generate_key(&[Value::Integer(123)]); // Invalid key size
        
        assert!(result.is_err());
        let error_msg = format!("{:?}", result.unwrap_err());
        assert!(error_msg.contains("Invalid AES key size"));
    }
}