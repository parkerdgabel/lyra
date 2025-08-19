//! Compression and optimization for serialized data

use super::{SerializationError, SerializationResult};

/// Compression algorithms supported by Lyra
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionAlgorithm {
    None,
    Lz4,
    Zstd,
}

/// Compression settings and parameters
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    pub algorithm: CompressionAlgorithm,
    pub level: i32,
    pub dictionary: Option<Vec<u8>>,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            algorithm: CompressionAlgorithm::Lz4,
            level: 1,
            dictionary: None,
        }
    }
}

/// Compress data using the specified algorithm
pub fn compress(data: &[u8], config: &CompressionConfig) -> SerializationResult<Vec<u8>> {
    match config.algorithm {
        CompressionAlgorithm::None => Ok(data.to_vec()),
        CompressionAlgorithm::Lz4 => {
            lz4_flex::compress(data).map_err(|e| {
                SerializationError::CompressionError(format!("LZ4 compression failed: {}", e))
            })
        }
        CompressionAlgorithm::Zstd => {
            zstd::bulk::compress(data, config.level).map_err(|e| {
                SerializationError::CompressionError(format!("Zstd compression failed: {}", e))
            })
        }
    }
}

/// Decompress data using the specified algorithm
pub fn decompress(data: &[u8], algorithm: CompressionAlgorithm) -> SerializationResult<Vec<u8>> {
    match algorithm {
        CompressionAlgorithm::None => Ok(data.to_vec()),
        CompressionAlgorithm::Lz4 => {
            lz4_flex::decompress_size_prepended(data).map_err(|e| {
                SerializationError::CompressionError(format!("LZ4 decompression failed: {}", e))
            })
        }
        CompressionAlgorithm::Zstd => {
            zstd::bulk::decompress(data, 1024 * 1024 * 100) // 100MB max
                .map_err(|e| {
                    SerializationError::CompressionError(format!("Zstd decompression failed: {}", e))
                })
        }
    }
}

/// Compressed data container with metadata
#[derive(Debug, Clone)]
pub struct CompressedData {
    pub algorithm: CompressionAlgorithm,
    pub original_size: usize,
    pub compressed_data: Vec<u8>,
}

impl CompressedData {
    pub fn new(data: &[u8], config: &CompressionConfig) -> SerializationResult<Self> {
        let compressed_data = compress(data, config)?;
        Ok(Self {
            algorithm: config.algorithm,
            original_size: data.len(),
            compressed_data,
        })
    }
    
    pub fn decompress(&self) -> SerializationResult<Vec<u8>> {
        decompress(&self.compressed_data, self.algorithm)
    }
    
    pub fn compression_ratio(&self) -> f64 {
        if self.original_size == 0 {
            1.0
        } else {
            self.compressed_data.len() as f64 / self.original_size as f64
        }
    }
}

/// Smart compression that chooses the best algorithm based on data characteristics
pub struct SmartCompressor {
    configs: Vec<CompressionConfig>,
}

impl SmartCompressor {
    pub fn new() -> Self {
        Self {
            configs: vec![
                CompressionConfig {
                    algorithm: CompressionAlgorithm::None,
                    level: 0,
                    dictionary: None,
                },
                CompressionConfig {
                    algorithm: CompressionAlgorithm::Lz4,
                    level: 1,
                    dictionary: None,
                },
                CompressionConfig {
                    algorithm: CompressionAlgorithm::Zstd,
                    level: 3,
                    dictionary: None,
                },
            ]
        }
    }
    
    pub fn compress_adaptive(&self, data: &[u8]) -> SerializationResult<CompressedData> {
        // For small data, don't compress
        if data.len() < 1024 {
            return CompressedData::new(data, &self.configs[0]);
        }
        
        // For symbolic data (mostly text), use Zstd
        if Self::is_mostly_text(data) {
            return CompressedData::new(data, &self.configs[2]);
        }
        
        // For general data, use LZ4 for speed
        CompressedData::new(data, &self.configs[1])
    }
    
    fn is_mostly_text(data: &[u8]) -> bool {
        if data.is_empty() {
            return false;
        }
        
        let ascii_count = data.iter().filter(|&&b| b.is_ascii_graphic() || b.is_ascii_whitespace()).count();
        ascii_count as f64 / data.len() as f64 > 0.8
    }
}

impl Default for SmartCompressor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_none() {
        let data = b"Hello, World!";
        let config = CompressionConfig {
            algorithm: CompressionAlgorithm::None,
            level: 0,
            dictionary: None,
        };
        
        let compressed = compress(data, &config).unwrap();
        assert_eq!(compressed, data);
        
        let decompressed = decompress(&compressed, CompressionAlgorithm::None).unwrap();
        assert_eq!(decompressed, data);
    }
    
    #[test]
    fn test_compression_lz4() {
        let data = b"Hello, World! This is a test string for compression.";
        let config = CompressionConfig {
            algorithm: CompressionAlgorithm::Lz4,
            level: 1,
            dictionary: None,
        };
        
        let compressed = compress(data, &config).unwrap();
        let decompressed = decompress(&compressed, CompressionAlgorithm::Lz4).unwrap();
        assert_eq!(decompressed, data);
    }
    
    #[test]
    fn test_compression_zstd() {
        let data = b"Hello, World! This is a test string for compression with Zstd.";
        let config = CompressionConfig {
            algorithm: CompressionAlgorithm::Zstd,
            level: 3,
            dictionary: None,
        };
        
        let compressed = compress(data, &config).unwrap();
        let decompressed = decompress(&compressed, CompressionAlgorithm::Zstd).unwrap();
        assert_eq!(decompressed, data);
    }
    
    #[test]
    fn test_compressed_data() {
        let data = b"This is test data for the CompressedData structure.";
        let config = CompressionConfig::default();
        
        let compressed_data = CompressedData::new(data, &config).unwrap();
        assert_eq!(compressed_data.original_size, data.len());
        
        let decompressed = compressed_data.decompress().unwrap();
        assert_eq!(decompressed, data);
        
        let ratio = compressed_data.compression_ratio();
        assert!(ratio > 0.0);
    }
    
    #[test]
    fn test_smart_compressor() {
        let compressor = SmartCompressor::new();
        
        // Test with text data
        let text_data = b"This is a text string with many repeated patterns and words";
        let compressed = compressor.compress_adaptive(text_data).unwrap();
        let decompressed = compressed.decompress().unwrap();
        assert_eq!(decompressed, text_data);
        
        // Test with small data
        let small_data = b"small";
        let compressed = compressor.compress_adaptive(small_data).unwrap();
        assert_eq!(compressed.algorithm, CompressionAlgorithm::None);
    }
}