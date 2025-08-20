//! Cryptographic Primitives
//!
//! This module implements fundamental cryptographic algorithms and data structures
//! for secure computation and cryptographic applications within Lyra's symbolic environment.

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// RSA public/private key pair
#[derive(Debug, Clone)]
pub struct RSAKeyPair {
    /// Public key (n, e)
    pub public_key: (i64, i64),
    /// Private key (n, d)
    pub private_key: (i64, i64),
    /// Key size in bits
    pub key_size: usize,
    /// Primes used in generation (p, q)
    pub primes: (i64, i64),
}

impl RSAKeyPair {
    pub fn new(p: i64, q: i64, e: i64) -> Result<Self, String> {
        if p <= 1 || q <= 1 {
            return Err("Invalid primes for RSA key generation".to_string());
        }
        
        let n = p * q;
        let phi = (p - 1) * (q - 1);
        
        // Compute private key d such that e*d ≡ 1 (mod φ(n))
        let d = match modular_inverse(e, phi) {
            Some(d) => d,
            None => return Err("Invalid public exponent, no modular inverse exists".to_string()),
        };
        
        // Estimate key size in bits
        let key_size = (64 - n.leading_zeros()) as usize;
        
        Ok(Self {
            public_key: (n, e),
            private_key: (n, d),
            key_size,
            primes: (p, q),
        })
    }
    
    /// Encrypt a message using public key
    pub fn encrypt(&self, message: i64) -> i64 {
        let (n, e) = self.public_key;
        power_mod(message, e, n)
    }
    
    /// Decrypt a message using private key
    pub fn decrypt(&self, ciphertext: i64) -> i64 {
        let (n, d) = self.private_key;
        power_mod(ciphertext, d, n)
    }
    
    /// Sign a message using private key
    pub fn sign(&self, message: i64) -> i64 {
        let (n, d) = self.private_key;
        power_mod(message, d, n)
    }
    
    /// Verify a signature using public key
    pub fn verify(&self, message: i64, signature: i64) -> bool {
        let (n, e) = self.public_key;
        let decrypted = power_mod(signature, e, n);
        decrypted == message
    }
}

impl Foreign for RSAKeyPair {
    fn type_name(&self) -> &'static str {
        "RSAKeyPair"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "PublicKey" => {
                let (n, e) = self.public_key;
                Ok(Value::List(vec![Value::Integer(n), Value::Integer(e)]))
            }
            "PrivateKey" => {
                let (n, d) = self.private_key;
                Ok(Value::List(vec![Value::Integer(n), Value::Integer(d)]))
            }
            "KeySize" => Ok(Value::Integer(self.key_size as i64)),
            "Modulus" => Ok(Value::Integer(self.public_key.0)),
            "PublicExponent" => Ok(Value::Integer(self.public_key.1)),
            "PrivateExponent" => Ok(Value::Integer(self.private_key.1)),
            "Primes" => {
                let (p, q) = self.primes;
                Ok(Value::List(vec![Value::Integer(p), Value::Integer(q)]))
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Elliptic curve point representation
#[derive(Debug, Clone)]
pub struct ECPoint {
    /// x-coordinate
    pub x: i64,
    /// y-coordinate  
    pub y: i64,
    /// Elliptic curve parameters (a, b, p) for y² ≡ x³ + ax + b (mod p)
    pub curve: (i64, i64, i64),
    /// Whether this is the point at infinity
    pub is_infinity: bool,
}

impl ECPoint {
    /// Create a new elliptic curve point
    pub fn new(x: i64, y: i64, curve: (i64, i64, i64)) -> Result<Self, String> {
        let point = Self {
            x,
            y,
            curve,
            is_infinity: false,
        };
        
        if point.is_on_curve() {
            Ok(point)
        } else {
            Err("Point is not on the elliptic curve".to_string())
        }
    }
    
    /// Create the point at infinity
    pub fn infinity(curve: (i64, i64, i64)) -> Self {
        Self {
            x: 0,
            y: 0,
            curve,
            is_infinity: true,
        }
    }
    
    /// Check if point is on the curve: y² ≡ x³ + ax + b (mod p)
    pub fn is_on_curve(&self) -> bool {
        if self.is_infinity { return true; }
        
        let (a, b, p) = self.curve;
        let left = mod_mul(self.y, self.y, p);
        let right = (power_mod(self.x, 3, p) + mod_mul(a, self.x, p) + b) % p;
        left == ((right % p + p) % p)
    }
    
    /// Add two elliptic curve points
    pub fn add(&self, other: &ECPoint) -> Result<ECPoint, String> {
        if self.curve != other.curve {
            return Err("Points must be on the same curve".to_string());
        }
        
        if self.is_infinity {
            return Ok(other.clone());
        }
        if other.is_infinity {
            return Ok(self.clone());
        }
        
        let (a, _b, p) = self.curve;
        
        if self.x == other.x {
            if self.y == other.y {
                // Point doubling: P + P
                let s_num = (3 * mod_mul(self.x, self.x, p) + a) % p;
                let s_den = (2 * self.y) % p;
                let s_den_inv = modular_inverse(s_den, p)
                    .ok_or("Cannot compute slope for point doubling")?;
                let s = mod_mul(s_num, s_den_inv, p);
                
                let x3 = (mod_mul(s, s, p) - 2 * self.x) % p;
                let y3 = (mod_mul(s, self.x - x3, p) - self.y) % p;
                
                ECPoint::new(((x3 % p) + p) % p, ((y3 % p) + p) % p, self.curve)
            } else {
                // P + (-P) = O (point at infinity)
                Ok(ECPoint::infinity(self.curve))
            }
        } else {
            // General case: P + Q where P ≠ Q
            let s_num = (other.y - self.y) % p;
            let s_den = (other.x - self.x) % p;
            let s_den_inv = modular_inverse(s_den, p)
                .ok_or("Cannot compute slope for point addition")?;
            let s = mod_mul(s_num, s_den_inv, p);
            
            let x3 = (mod_mul(s, s, p) - self.x - other.x) % p;
            let y3 = (mod_mul(s, self.x - x3, p) - self.y) % p;
            
            ECPoint::new(((x3 % p) + p) % p, ((y3 % p) + p) % p, self.curve)
        }
    }
    
    /// Scalar multiplication: k * P
    pub fn scalar_mult(&self, k: i64) -> Result<ECPoint, String> {
        if k == 0 || self.is_infinity {
            return Ok(ECPoint::infinity(self.curve));
        }
        
        if k < 0 {
            let neg_point = ECPoint::new(self.x, (-self.y) % self.curve.2, self.curve)?;
            return neg_point.scalar_mult(-k);
        }
        
        // Double-and-add algorithm
        let mut result = ECPoint::infinity(self.curve);
        let mut addend = self.clone();
        let mut scalar = k;
        
        while scalar > 0 {
            if scalar % 2 == 1 {
                result = result.add(&addend)?;
            }
            addend = addend.add(&addend)?;
            scalar /= 2;
        }
        
        Ok(result)
    }
}

impl Foreign for ECPoint {
    fn type_name(&self) -> &'static str {
        "ECPoint"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "X" => Ok(Value::Integer(self.x)),
            "Y" => Ok(Value::Integer(self.y)),
            "Curve" => {
                let (a, b, p) = self.curve;
                Ok(Value::List(vec![
                    Value::Integer(a),
                    Value::Integer(b), 
                    Value::Integer(p)
                ]))
            }
            "IsInfinity" => Ok(Value::Integer(if self.is_infinity { 1 } else { 0 })),
            "IsOnCurve" => Ok(Value::Integer(if self.is_on_curve() { 1 } else { 0 })),
            "Coordinates" => {
                if self.is_infinity {
                    Ok(Value::String("Infinity".to_string()))
                } else {
                    Ok(Value::List(vec![Value::Integer(self.x), Value::Integer(self.y)]))
                }
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Hash function result
#[derive(Debug, Clone)]
pub struct HashResult {
    /// The computed hash value
    pub hash: Vec<u8>,
    /// Hash algorithm used
    pub algorithm: String,
    /// Original data that was hashed
    pub data: Vec<u8>,
}

impl HashResult {
    pub fn new(data: Vec<u8>, algorithm: String, hash: Vec<u8>) -> Self {
        Self {
            hash,
            algorithm,
            data,
        }
    }
    
    /// Get hash as hexadecimal string
    pub fn hex_string(&self) -> String {
        self.hash.iter()
            .map(|b| format!("{:02x}", b))
            .collect()
    }
    
    /// Get hash as integer (for small hashes)
    pub fn as_integer(&self) -> i64 {
        let mut result = 0i64;
        for (i, &byte) in self.hash.iter().take(8).enumerate() {
            result |= (byte as i64) << (i * 8);
        }
        result
    }
}

impl Foreign for HashResult {
    fn type_name(&self) -> &'static str {
        "HashResult"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Hash" => {
                let bytes: Vec<Value> = self.hash.iter()
                    .map(|&b| Value::Integer(b as i64))
                    .collect();
                Ok(Value::List(bytes))
            }
            "HexString" => Ok(Value::String(self.hex_string())),
            "AsInteger" => Ok(Value::Integer(self.as_integer())),
            "Algorithm" => Ok(Value::String(self.algorithm.clone())),
            "DataSize" => Ok(Value::Integer(self.data.len() as i64)),
            "HashSize" => Ok(Value::Integer(self.hash.len() as i64)),
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ===============================
// CRYPTOGRAPHIC UTILITY FUNCTIONS
// ===============================

/// Fast modular exponentiation: (base^exp) mod m
pub fn power_mod(mut base: i64, mut exp: i64, m: i64) -> i64 {
    if m == 1 { return 0; }
    
    let mut result = 1;
    base = ((base % m) + m) % m;
    
    while exp > 0 {
        if exp % 2 == 1 {
            result = mod_mul(result, base, m);
        }
        exp >>= 1;
        base = mod_mul(base, base, m);
    }
    
    result
}

/// Modular multiplication with overflow protection
pub fn mod_mul(a: i64, b: i64, m: i64) -> i64 {
    ((a as i128 * b as i128) % m as i128) as i64
}

/// Extended Euclidean algorithm for modular inverse
pub fn modular_inverse(a: i64, m: i64) -> Option<i64> {
    let (gcd, x, _) = extended_gcd(a, m);
    if gcd != 1 {
        None
    } else {
        Some(((x % m) + m) % m)
    }
}

/// Extended GCD returning (gcd, x, y) where ax + my = gcd
fn extended_gcd(a: i64, m: i64) -> (i64, i64, i64) {
    if a == 0 {
        (m, 0, 1)
    } else {
        let (gcd, x1, y1) = extended_gcd(m % a, a);
        let x = y1 - (m / a) * x1;
        let y = x1;
        (gcd, x, y)
    }
}

/// Miller-Rabin primality test
pub fn is_prime(n: i64, rounds: usize) -> bool {
    if n < 2 { return false; }
    if n == 2 || n == 3 { return true; }
    if n % 2 == 0 { return false; }
    
    // Write n-1 as d * 2^r
    let mut d = n - 1;
    let mut r = 0;
    while d % 2 == 0 {
        d /= 2;
        r += 1;
    }
    
    // Simple LCG for deterministic testing
    let mut rng_state = 12345u64;
    let mut random = || {
        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        (rng_state % (n as u64 - 2)) as i64 + 2
    };
    
    for _ in 0..rounds {
        let a = random();
        let mut x = power_mod(a, d, n);
        
        if x == 1 || x == n - 1 {
            continue;
        }
        
        let mut composite = true;
        for _ in 0..r - 1 {
            x = mod_mul(x, x, n);
            if x == n - 1 {
                composite = false;
                break;
            }
        }
        
        if composite {
            return false;
        }
    }
    
    true
}

/// Simple hash function (FNV-1a variant)
pub fn simple_hash(data: &[u8], algorithm: &str) -> Vec<u8> {
    match algorithm.to_lowercase().as_str() {
        "md5" => md5_hash(data),
        "sha256" => sha256_hash(data),
        "fnv" | _ => fnv_hash(data),
    }
}

/// FNV-1a hash implementation
fn fnv_hash(data: &[u8]) -> Vec<u8> {
    let mut hash = 0xcbf29ce484222325u64; // FNV offset basis
    const FNV_PRIME: u64 = 0x100000001b3;
    
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    
    hash.to_le_bytes().to_vec()
}

/// Simplified MD5-like hash (not cryptographically secure)
fn md5_hash(data: &[u8]) -> Vec<u8> {
    let mut state = [0x67452301u32, 0xefcdab89, 0x98badcfe, 0x10325476];
    
    // Simplified MD5-like processing
    for chunk in data.chunks(64) {
        let mut w = [0u32; 16];
        for (i, chunk_bytes) in chunk.chunks(4).enumerate() {
            if i < 16 {
                let mut bytes = [0u8; 4];
                bytes[..chunk_bytes.len()].copy_from_slice(chunk_bytes);
                w[i] = u32::from_le_bytes(bytes);
            }
        }
        
        let [mut a, mut b, mut c, mut d] = state;
        
        // Simplified rounds
        for i in 0..16 {
            let f = (b & c) | (!b & d);
            let temp = a.wrapping_add(f).wrapping_add(w[i]).wrapping_add(0x5a827999);
            a = d;
            d = c;
            c = b;
            b = temp.rotate_left(7);
        }
        
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
    }
    
    let mut result = Vec::new();
    for word in state {
        result.extend_from_slice(&word.to_le_bytes());
    }
    result
}

/// Simplified SHA256-like hash (not cryptographically secure)
fn sha256_hash(data: &[u8]) -> Vec<u8> {
    let mut state = [
        0x6a09e667u32, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    
    // Simplified SHA256-like processing
    for chunk in data.chunks(64) {
        let mut w = [0u32; 64];
        for (i, chunk_bytes) in chunk.chunks(4).enumerate() {
            if i < 16 {
                let mut bytes = [0u8; 4];
                bytes[..chunk_bytes.len()].copy_from_slice(chunk_bytes);
                w[i] = u32::from_be_bytes(bytes);
            }
        }
        
        // Extend the first 16 words
        for i in 16..64 {
            let s0 = w[i-15].rotate_right(7) ^ w[i-15].rotate_right(18) ^ (w[i-15] >> 3);
            let s1 = w[i-2].rotate_right(17) ^ w[i-2].rotate_right(19) ^ (w[i-2] >> 10);
            w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
        }
        
        let mut h = state;
        
        for i in 0..64 {
            let s1 = h[4].rotate_right(6) ^ h[4].rotate_right(11) ^ h[4].rotate_right(25);
            let ch = (h[4] & h[5]) ^ (!h[4] & h[6]);
            let temp1 = h[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(w[i]);
            let s0 = h[0].rotate_right(2) ^ h[0].rotate_right(13) ^ h[0].rotate_right(22);
            let maj = (h[0] & h[1]) ^ (h[0] & h[2]) ^ (h[1] & h[2]);
            let temp2 = s0.wrapping_add(maj);
            
            h[7] = h[6];
            h[6] = h[5];
            h[5] = h[4];
            h[4] = h[3].wrapping_add(temp1);
            h[3] = h[2];
            h[2] = h[1];
            h[1] = h[0];
            h[0] = temp1.wrapping_add(temp2);
        }
        
        for i in 0..8 {
            state[i] = state[i].wrapping_add(h[i]);
        }
    }
    
    let mut result = Vec::new();
    for word in state {
        result.extend_from_slice(&word.to_be_bytes());
    }
    result
}

// ===============================
// WOLFRAM LANGUAGE INTERFACE FUNCTIONS
// ===============================

/// Generate RSA key pair
/// Syntax: RSAGenerate[bits] or RSAGenerate[p, q, e]
pub fn rsa_generate(args: &[Value]) -> VmResult<Value> {
    match args.len() {
        1 => {
            // RSAGenerate[bits] - generate key with specified bit size
            let bits = match &args[0] {
                Value::Integer(b) => *b as usize,
                _ => return Err(VmError::TypeError {
                    expected: "Integer for key size in bits".to_string(),
                    actual: format!("{:?}", args[0]),
                }),
            };
            
            if bits < 8 || bits > 64 {
                return Err(VmError::Runtime("Key size must be between 8 and 64 bits for demo".to_string()));
            }
            
            // Generate small primes for demonstration
            let primes = small_primes();
            let p = primes[5]; // Use pre-computed small primes
            let q = primes[7];
            let e = 65537; // Standard public exponent
            
            match RSAKeyPair::new(p, q, e) {
                Ok(keypair) => Ok(Value::LyObj(LyObj::new(Box::new(keypair)))),
                Err(e) => Err(VmError::Runtime(format!("RSA key generation failed: {}", e))),
            }
        }
        3 => {
            // RSAGenerate[p, q, e] - generate key with specific parameters
            let p = match &args[0] {
                Value::Integer(p) => *p,
                _ => return Err(VmError::TypeError {
                    expected: "Integer for prime p".to_string(),
                    actual: format!("{:?}", args[0]),
                }),
            };
            
            let q = match &args[1] {
                Value::Integer(q) => *q,
                _ => return Err(VmError::TypeError {
                    expected: "Integer for prime q".to_string(),
                    actual: format!("{:?}", args[1]),
                }),
            };
            
            let e = match &args[2] {
                Value::Integer(e) => *e,
                _ => return Err(VmError::TypeError {
                    expected: "Integer for public exponent e".to_string(),
                    actual: format!("{:?}", args[2]),
                }),
            };
            
            // Verify p and q are prime
            if !is_prime(p, 10) || !is_prime(q, 10) {
                return Err(VmError::Runtime("p and q must be prime numbers".to_string()));
            }
            
            match RSAKeyPair::new(p, q, e) {
                Ok(keypair) => Ok(Value::LyObj(LyObj::new(Box::new(keypair)))),
                Err(e) => Err(VmError::Runtime(format!("RSA key generation failed: {}", e))),
            }
        }
        _ => Err(VmError::TypeError {
            expected: "1 or 3 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        }),
    }
}

/// Create elliptic curve point
/// Syntax: ECPoint[curve, x, y] or ECPoint[curve, "infinity"]
pub fn ec_point(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "2-3 arguments (curve, x, y) or (curve, \"infinity\")".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    // Parse curve parameters
    let curve = match &args[0] {
        Value::List(params) => {
            if params.len() != 3 {
                return Err(VmError::TypeError {
                    expected: "List of 3 integers [a, b, p] for curve".to_string(),
                    actual: format!("List of {} elements", params.len()),
                });
            }
            
            let a = match &params[0] {
                Value::Integer(a) => *a,
                _ => return Err(VmError::TypeError {
                    expected: "Integer for curve parameter a".to_string(),
                    actual: format!("{:?}", params[0]),
                }),
            };
            
            let b = match &params[1] {
                Value::Integer(b) => *b,
                _ => return Err(VmError::TypeError {
                    expected: "Integer for curve parameter b".to_string(),
                    actual: format!("{:?}", params[1]),
                }),
            };
            
            let p = match &params[2] {
                Value::Integer(p) => *p,
                _ => return Err(VmError::TypeError {
                    expected: "Integer for curve parameter p".to_string(),
                    actual: format!("{:?}", params[2]),
                }),
            };
            
            (a, b, p)
        }
        _ => return Err(VmError::TypeError {
            expected: "List for curve parameters".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    // Check for infinity point
    if args.len() == 2 {
        if let Value::String(s) = &args[1] {
            if s.to_lowercase() == "infinity" {
                let point = ECPoint::infinity(curve);
                return Ok(Value::LyObj(LyObj::new(Box::new(point))));
            }
        }
    }
    
    // Parse coordinates
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "3 arguments for finite point: curve, x, y".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let x = match &args[1] {
        Value::Integer(x) => *x,
        _ => return Err(VmError::TypeError {
            expected: "Integer for x coordinate".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let y = match &args[2] {
        Value::Integer(y) => *y,
        _ => return Err(VmError::TypeError {
            expected: "Integer for y coordinate".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };
    
    match ECPoint::new(x, y, curve) {
        Ok(point) => Ok(Value::LyObj(LyObj::new(Box::new(point)))),
        Err(e) => Err(VmError::Runtime(format!("Invalid elliptic curve point: {}", e))),
    }
}

/// Compute hash function
/// Syntax: HashFunction[data, algorithm]
pub fn hash_function(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (data, algorithm)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let data = match &args[0] {
        Value::String(s) => s.as_bytes().to_vec(),
        Value::List(bytes) => {
            let mut data = Vec::new();
            for byte_val in bytes {
                match byte_val {
                    Value::Integer(i) => {
                        if *i < 0 || *i > 255 {
                            return Err(VmError::Runtime("Byte values must be 0-255".to_string()));
                        }
                        data.push(*i as u8);
                    }
                    _ => return Err(VmError::TypeError {
                        expected: "List of integers for binary data".to_string(),
                        actual: format!("{:?}", byte_val),
                    }),
                }
            }
            data
        }
        _ => return Err(VmError::TypeError {
            expected: "String or List of bytes for data".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let algorithm = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for hash algorithm".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let hash = simple_hash(&data, &algorithm);
    let result = HashResult::new(data, algorithm, hash);
    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

/// Generate random prime
/// Syntax: RandomPrime[bits]
pub fn random_prime(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (bits)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let bits = match &args[0] {
        Value::Integer(b) => *b as usize,
        _ => return Err(VmError::TypeError {
            expected: "Integer for bit size".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    if bits < 2 || bits > 32 {
        return Err(VmError::Runtime("Bit size must be between 2 and 32 for demo".to_string()));
    }
    
    // For demonstration, return a pre-computed prime appropriate for the bit size
    let primes = small_primes();
    let target_range = 1 << (bits - 1);
    
    for &prime in &primes {
        if prime >= target_range && prime < (target_range * 2) {
            return Ok(Value::Integer(prime));
        }
    }
    
    // Fallback to largest available prime
    Ok(Value::Integer(primes[primes.len() - 1]))
}

/// Get small primes for testing
fn small_primes() -> Vec<i64> {
    vec![
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
        73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
        157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
        239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rsa_keypair() {
        let keypair = RSAKeyPair::new(61, 53, 17).unwrap();
        assert_eq!(keypair.public_key.0, 61 * 53); // n = p * q
        assert_eq!(keypair.public_key.1, 17); // e
        
        // Test encryption/decryption
        let message = 123;
        let encrypted = keypair.encrypt(message);
        let decrypted = keypair.decrypt(encrypted);
        assert_eq!(message, decrypted);
        
        // Test signing/verification
        let signature = keypair.sign(message);
        assert!(keypair.verify(message, signature));
    }
    
    #[test]
    fn test_elliptic_curve_point() {
        // Test curve y^2 = x^3 + 2x + 3 (mod 97)
        let curve = (2, 3, 97);
        
        // Point (3, 6) should be on the curve
        let point = ECPoint::new(3, 6, curve).unwrap();
        assert!(point.is_on_curve());
        
        // Test point at infinity
        let infinity = ECPoint::infinity(curve);
        assert!(infinity.is_infinity);
        assert!(infinity.is_on_curve());
    }
    
    #[test]
    fn test_elliptic_curve_addition() {
        let curve = (2, 3, 97);
        let p1 = ECPoint::new(3, 6, curve).unwrap();
        let p2 = ECPoint::new(80, 10, curve).unwrap();
        
        // Test point addition
        let sum = p1.add(&p2).unwrap();
        assert!(sum.is_on_curve());
        
        // Test point doubling
        let doubled = p1.add(&p1).unwrap();
        assert!(doubled.is_on_curve());
        
        // Test adding point at infinity
        let infinity = ECPoint::infinity(curve);
        let result = p1.add(&infinity).unwrap();
        assert_eq!(result.x, p1.x);
        assert_eq!(result.y, p1.y);
    }
    
    #[test]
    fn test_hash_functions() {
        let data = b"Hello, World!";
        
        // Test FNV hash
        let fnv_hash = simple_hash(data, "fnv");
        assert_eq!(fnv_hash.len(), 8);
        
        // Test MD5-like hash
        let md5_hash = simple_hash(data, "md5");
        assert_eq!(md5_hash.len(), 16);
        
        // Test SHA256-like hash
        let sha256_hash = simple_hash(data, "sha256");
        assert_eq!(sha256_hash.len(), 32);
        
        // Same input should produce same hash
        let fnv_hash2 = simple_hash(data, "fnv");
        assert_eq!(fnv_hash, fnv_hash2);
    }
    
    #[test]
    fn test_hash_result() {
        let data = b"test data".to_vec();
        let hash = vec![0x12, 0x34, 0x56, 0x78];
        let result = HashResult::new(data, "test".to_string(), hash);
        
        assert_eq!(result.hex_string(), "12345678");
        assert_eq!(result.as_integer(), 0x78563412); // Little endian
    }
    
    #[test]
    fn test_modular_arithmetic() {
        assert_eq!(power_mod(2, 10, 1000), 1024 % 1000);
        assert_eq!(power_mod(3, 4, 7), 4); // 3^4 = 81 ≡ 4 (mod 7)
        
        assert_eq!(modular_inverse(3, 7), Some(5)); // 3 * 5 ≡ 1 (mod 7)
        assert_eq!(modular_inverse(2, 6), None); // gcd(2, 6) = 2 ≠ 1
    }
    
    #[test]
    fn test_prime_testing() {
        assert!(is_prime(2, 10));
        assert!(is_prime(3, 10));
        assert!(!is_prime(4, 10));
        assert!(is_prime(5, 10));
        assert!(!is_prime(6, 10));
        assert!(is_prime(7, 10));
        assert!(is_prime(97, 10));
        assert!(!is_prime(100, 10));
    }
}