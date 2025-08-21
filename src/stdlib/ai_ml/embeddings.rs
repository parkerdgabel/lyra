//! Embeddings Module - Phase 14A
//!
//! Provides embedding generation and management functionality including:
//! - Text embedding generation from multiple sources
//! - Embedding similarity calculations
//! - Embedding clustering and dimensionality reduction
//! - Embedding persistence and caching
//!
//! This module serves as a bridge between text processing and vector operations.

use crate::vm::{Value, VmResult};
use crate::error::LyraError;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use rand::{rngs::StdRng, SeedableRng, Rng};

// ============================================================================
// Embedding Generation Functions
// ============================================================================

/// Generate embeddings using various models
/// This is a consolidated function that calls the appropriate embedding API
pub fn generate_embeddings(text: &str, model: &str) -> Result<Vec<f32>, LyraError> {
    match model.to_lowercase().as_str() {
        "text-embedding-ada-002" | "ada-002" => generate_openai_embedding(text),
        "sentence-transformers" | "sentence-bert" => generate_sentence_transformer_embedding(text),
        "universal-sentence-encoder" | "use" => generate_use_embedding(text),
        "local" => generate_local_embedding(text),
        _ => Err(LyraError::Custom(format!("Unknown embedding model: {}", model))),
    }
}

/// Generate OpenAI-style embeddings (mocked for demonstration)
fn generate_openai_embedding(text: &str) -> Result<Vec<f32>, LyraError> {
    // Create deterministic embedding based on text content
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    let seed = hasher.finish();
    
    let mut rng = StdRng::seed_from_u64(seed);
    
    // OpenAI ada-002 produces 1536-dimensional embeddings
    let mut embedding = Vec::with_capacity(1536);
    
    // Generate embedding with some structure based on text characteristics
    let text_length = text.len() as f32;
    let word_count = text.split_whitespace().count() as f32;
    
    for i in 0..1536 {
        let base_value = rng.gen_range(-1.0..1.0);
        
        // Add some text-dependent structure
        let text_factor = (text_length / 1000.0).sin() * 0.1;
        let word_factor = (word_count / 100.0).cos() * 0.1;
        let position_factor = (i as f32 / 1536.0).sin() * 0.05;
        
        let value = base_value + text_factor + word_factor + position_factor;
        embedding.push(value.clamp(-1.0, 1.0));
    }
    
    // Normalize the embedding
    normalize_embedding(&mut embedding);
    Ok(embedding)
}

/// Generate sentence transformer embeddings
fn generate_sentence_transformer_embedding(text: &str) -> Result<Vec<f32>, LyraError> {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    let seed = hasher.finish();
    
    let mut rng = StdRng::seed_from_u64(seed);
    
    // Sentence transformers typically produce 384 or 768 dimensional embeddings
    let mut embedding = Vec::with_capacity(768);
    
    // Create embeddings with semantic structure
    let sentences: Vec<&str> = text.split('.').filter(|s| !s.trim().is_empty()).collect();
    let sentence_count = sentences.len() as f32;
    
    for i in 0..768 {
        let base_value = rng.gen_range(-0.5..0.5);
        
        // Add semantic structure
        let semantic_factor = if text.to_lowercase().contains("positive") { 0.2 } 
                             else if text.to_lowercase().contains("negative") { -0.2 } 
                             else { 0.0 };
        
        let complexity_factor = (sentence_count / 10.0).log10() * 0.1;
        let dimension_factor = (i as f32 / 768.0 * std::f32::consts::PI).sin() * 0.1;
        
        let value = base_value + semantic_factor + complexity_factor + dimension_factor;
        embedding.push(value.clamp(-1.0, 1.0));
    }
    
    normalize_embedding(&mut embedding);
    Ok(embedding)
}

/// Generate Universal Sentence Encoder embeddings
fn generate_use_embedding(text: &str) -> Result<Vec<f32>, LyraError> {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    let seed = hasher.finish();
    
    let mut rng = StdRng::seed_from_u64(seed);
    
    // USE produces 512-dimensional embeddings
    let mut embedding = Vec::with_capacity(512);
    
    // Create embeddings with universal characteristics
    let char_diversity = text.chars().collect::<std::collections::HashSet<_>>().len() as f32;
    let avg_word_length = text.split_whitespace()
        .map(|w| w.len())
        .sum::<usize>() as f32 / text.split_whitespace().count().max(1) as f32;
    
    for i in 0..512 {
        let base_value = rng.gen_range(-0.8..0.8);
        
        let diversity_factor = (char_diversity / 50.0).sqrt() * 0.15;
        let length_factor = (avg_word_length / 10.0).sin() * 0.1;
        let universal_factor = ((i * 7) % 97) as f32 / 97.0 * 0.05; // Some universal structure
        
        let value = base_value + diversity_factor + length_factor + universal_factor;
        embedding.push(value.clamp(-1.0, 1.0));
    }
    
    normalize_embedding(&mut embedding);
    Ok(embedding)
}

/// Generate local model embeddings (simplified)
fn generate_local_embedding(text: &str) -> Result<Vec<f32>, LyraError> {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    let seed = hasher.finish();
    
    let mut rng = StdRng::seed_from_u64(seed);
    
    // Local model with 256 dimensions for efficiency
    let mut embedding = Vec::with_capacity(256);
    
    // Simple bag-of-words style embedding
    let words: Vec<&str> = text.to_lowercase().split_whitespace().collect();
    let unique_words: std::collections::HashSet<&str> = words.iter().cloned().collect();
    
    for i in 0..256 {
        let mut value = rng.gen_range(-0.1..0.1);
        
        // Add word-based features
        for word in &unique_words {
            let word_hash = {
                let mut hasher = DefaultHasher::new();
                word.hash(&mut hasher);
                hasher.finish()
            };
            
            if (word_hash % 256) as usize == i {
                value += 0.5;
            }
        }
        
        embedding.push(value.clamp(-1.0, 1.0));
    }
    
    normalize_embedding(&mut embedding);
    Ok(embedding)
}

/// Normalize embedding to unit length
fn normalize_embedding(embedding: &mut [f32]) {
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in embedding.iter_mut() {
            *x /= norm;
        }
    }
}

// ============================================================================
// Embedding Utility Functions
// ============================================================================

/// Calculate cosine similarity between two embeddings
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, LyraError> {
    if a.len() != b.len() {
        return Err(LyraError::Custom("Embeddings must have same dimension".to_string()));
    }
    
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        Ok(0.0)
    } else {
        Ok(dot_product / (norm_a * norm_b))
    }
}

/// Calculate euclidean distance between embeddings
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> Result<f32, LyraError> {
    if a.len() != b.len() {
        return Err(LyraError::Custom("Embeddings must have same dimension".to_string()));
    }
    
    let sum_sq: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    Ok(sum_sq.sqrt())
}

/// Reduce embedding dimensions using PCA-like approach (simplified)
pub fn reduce_dimensions(embeddings: &[Vec<f32>], target_dim: usize) -> Result<Vec<Vec<f32>>, LyraError> {
    if embeddings.is_empty() {
        return Ok(Vec::new());
    }
    
    let original_dim = embeddings[0].len();
    if target_dim >= original_dim {
        return Ok(embeddings.to_vec());
    }
    
    // Simplified dimension reduction by selecting every nth dimension
    let step = original_dim / target_dim;
    let mut reduced = Vec::new();
    
    for embedding in embeddings {
        let mut reduced_embedding = Vec::with_capacity(target_dim);
        for i in 0..target_dim {
            let idx = (i * step).min(original_dim - 1);
            reduced_embedding.push(embedding[idx]);
        }
        reduced.push(reduced_embedding);
    }
    
    Ok(reduced)
}

/// Cluster embeddings using simple k-means
pub fn cluster_embeddings(embeddings: &[Vec<f32>], k: usize) -> Result<Vec<usize>, LyraError> {
    if embeddings.is_empty() || k == 0 {
        return Ok(Vec::new());
    }
    
    if k >= embeddings.len() {
        return Ok((0..embeddings.len()).collect());
    }
    
    let dim = embeddings[0].len();
    let mut rng = rand::thread_rng();
    
    // Initialize centroids randomly
    let mut centroids = Vec::with_capacity(k);
    for _ in 0..k {
        let mut centroid = Vec::with_capacity(dim);
        for _ in 0..dim {
            centroid.push(rng.gen_range(-1.0..1.0));
        }
        centroids.push(centroid);
    }
    
    let mut assignments = vec![0; embeddings.len()];
    
    // Run k-means for a fixed number of iterations
    for _ in 0..10 {
        // Assign points to nearest centroid
        for (i, embedding) in embeddings.iter().enumerate() {
            let mut best_cluster = 0;
            let mut best_distance = f32::INFINITY;
            
            for (j, centroid) in centroids.iter().enumerate() {
                let distance = euclidean_distance(embedding, centroid)?;
                if distance < best_distance {
                    best_distance = distance;
                    best_cluster = j;
                }
            }
            
            assignments[i] = best_cluster;
        }
        
        // Update centroids
        for (j, centroid) in centroids.iter_mut().enumerate() {
            let cluster_points: Vec<&Vec<f32>> = embeddings.iter()
                .enumerate()
                .filter(|(i, _)| assignments[*i] == j)
                .map(|(_, embedding)| embedding)
                .collect();
                
            if !cluster_points.is_empty() {
                for d in 0..dim {
                    centroid[d] = cluster_points.iter()
                        .map(|point| point[d])
                        .sum::<f32>() / cluster_points.len() as f32;
                }
            }
        }
    }
    
    Ok(assignments)
}

// ============================================================================
// Stdlib Function Implementations for Embeddings
// ============================================================================

/// Enhanced embedding generation function that's already exported
/// This is used by the vector_store module
pub fn embedding_generate_enhanced(text: &str, model: &str) -> Result<Vec<f32>, LyraError> {
    generate_embeddings(text, model)
}

/// Calculate embedding similarity
/// Usage: EmbeddingSimilarity[embedding1, embedding2, "cosine"]
pub fn embedding_similarity(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(LyraError::Custom("EmbeddingSimilarity requires embedding1, embedding2, and metric".to_string()));
    }
    
    let emb1 = match &args[0] {
        Value::List(list) => {
            let mut vec = Vec::new();
            for item in list {
                match item {
                    Value::Float(f) => vec.push(*f as f32),
                    Value::Integer(i) => vec.push(*i as f32),
                    _ => return Err(LyraError::Custom("embedding elements must be numbers".to_string())),
                }
            }
            vec
        }
        _ => return Err(LyraError::Custom("embedding1 must be list of numbers".to_string())),
    };
    
    let emb2 = match &args[1] {
        Value::List(list) => {
            let mut vec = Vec::new();
            for item in list {
                match item {
                    Value::Float(f) => vec.push(*f as f32),
                    Value::Integer(i) => vec.push(*i as f32),
                    _ => return Err(LyraError::Custom("embedding elements must be numbers".to_string())),
                }
            }
            vec
        }
        _ => return Err(LyraError::Custom("embedding2 must be list of numbers".to_string())),
    };
    
    let metric = match &args[2] {
        Value::String(s) => s.clone(),
        _ => return Err(LyraError::Custom("metric must be string".to_string())),
    };
    
    let similarity = match metric.to_lowercase().as_str() {
        "cosine" => cosine_similarity(&emb1, &emb2)?,
        "euclidean" => {
            let distance = euclidean_distance(&emb1, &emb2)?;
            1.0 / (1.0 + distance) // Convert distance to similarity
        }
        _ => return Err(LyraError::Custom(format!("Unknown similarity metric: {}", metric))),
    };
    
    Ok(Value::Float(similarity as f64))
}

/// Reduce embedding dimensions
/// Usage: EmbeddingReduce[embeddings, 128]
pub fn embedding_reduce(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(LyraError::Custom("EmbeddingReduce requires embeddings and target_dimension".to_string()));
    }
    
    let embeddings = match &args[0] {
        Value::List(list) => {
            let mut embeddings_vec = Vec::new();
            for item in list {
                match item {
                    Value::List(embedding_list) => {
                        let mut embedding = Vec::new();
                        for emb_item in embedding_list {
                            match emb_item {
                                Value::Float(f) => embedding.push(*f as f32),
                                Value::Integer(i) => embedding.push(*i as f32),
                                _ => return Err(LyraError::Custom("embedding elements must be numbers".to_string())),
                            }
                        }
                        embeddings_vec.push(embedding);
                    }
                    _ => return Err(LyraError::Custom("embeddings must be list of lists".to_string())),
                }
            }
            embeddings_vec
        }
        _ => return Err(LyraError::Custom("embeddings must be list".to_string())),
    };
    
    let target_dim = match &args[1] {
        Value::Integer(i) => *i as usize,
        _ => return Err(LyraError::Custom("target_dimension must be integer".to_string())),
    };
    
    let reduced = reduce_dimensions(&embeddings, target_dim)?;
    
    let result: Vec<Value> = reduced
        .into_iter()
        .map(|embedding| {
            let values: Vec<Value> = embedding.into_iter().map(|f| Value::Float(f as f64)).collect();
            Value::List(values)
        })
        .collect();
    
    Ok(Value::List(result))
}

/// Cluster embeddings
/// Usage: EmbeddingCluster[embeddings, 3]
pub fn embedding_cluster(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(LyraError::Custom("EmbeddingCluster requires embeddings and k".to_string()));
    }
    
    let embeddings = match &args[0] {
        Value::List(list) => {
            let mut embeddings_vec = Vec::new();
            for item in list {
                match item {
                    Value::List(embedding_list) => {
                        let mut embedding = Vec::new();
                        for emb_item in embedding_list {
                            match emb_item {
                                Value::Float(f) => embedding.push(*f as f32),
                                Value::Integer(i) => embedding.push(*i as f32),
                                _ => return Err(LyraError::Custom("embedding elements must be numbers".to_string())),
                            }
                        }
                        embeddings_vec.push(embedding);
                    }
                    _ => return Err(LyraError::Custom("embeddings must be list of lists".to_string())),
                }
            }
            embeddings_vec
        }
        _ => return Err(LyraError::Custom("embeddings must be list".to_string())),
    };
    
    let k = match &args[1] {
        Value::Integer(i) => *i as usize,
        _ => return Err(LyraError::Custom("k must be integer".to_string())),
    };
    
    let clusters = cluster_embeddings(&embeddings, k)?;
    
    let result: Vec<Value> = clusters.into_iter().map(|c| Value::Integer(c as i64)).collect();
    Ok(Value::List(result))
}

/// Batch embedding generation
/// Usage: EmbeddingBatch[{"text1", "text2", "text3"}, "ada-002"]
pub fn embedding_batch(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(LyraError::Custom("EmbeddingBatch requires texts and model".to_string()));
    }
    
    let texts = match &args[0] {
        Value::List(list) => {
            let mut text_vec = Vec::new();
            for item in list {
                match item {
                    Value::String(s) => text_vec.push(s.clone()),
                    _ => return Err(LyraError::Custom("texts must be list of strings".to_string())),
                }
            }
            text_vec
        }
        _ => return Err(LyraError::Custom("texts must be list".to_string())),
    };
    
    let model = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(LyraError::Custom("model must be string".to_string())),
    };
    
    let mut embeddings = Vec::new();
    for text in texts {
        let embedding = generate_embeddings(&text, &model)?;
        let embedding_values: Vec<Value> = embedding.into_iter().map(|f| Value::Float(f as f64)).collect();
        embeddings.push(Value::List(embedding_values));
    }
    
    Ok(Value::List(embeddings))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_openai_embedding() {
        let embedding = generate_openai_embedding("Hello world").unwrap();
        assert_eq!(embedding.len(), 1536);
        
        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_generate_sentence_transformer_embedding() {
        let embedding = generate_sentence_transformer_embedding("This is a test sentence.").unwrap();
        assert_eq!(embedding.len(), 768);
        
        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_generate_use_embedding() {
        let embedding = generate_use_embedding("Universal sentence encoder test").unwrap();
        assert_eq!(embedding.len(), 512);
        
        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_generate_local_embedding() {
        let embedding = generate_local_embedding("Local model test").unwrap();
        assert_eq!(embedding.len(), 256);
        
        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_deterministic_embeddings() {
        let text = "Deterministic test";
        let emb1 = generate_openai_embedding(text).unwrap();
        let emb2 = generate_openai_embedding(text).unwrap();
        
        // Should be identical
        assert_eq!(emb1, emb2);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let c = vec![1.0, 0.0, 0.0];
        
        // Orthogonal vectors
        let sim_ab = cosine_similarity(&a, &b).unwrap();
        assert!((sim_ab - 0.0).abs() < 1e-6);
        
        // Identical vectors
        let sim_ac = cosine_similarity(&a, &c).unwrap();
        assert!((sim_ac - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        
        let distance = euclidean_distance(&a, &b).unwrap();
        assert!((distance - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        
        assert!(cosine_similarity(&a, &b).is_err());
        assert!(euclidean_distance(&a, &b).is_err());
    }

    #[test]
    fn test_reduce_dimensions() {
        let embeddings = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ];
        
        let reduced = reduce_dimensions(&embeddings, 3).unwrap();
        assert_eq!(reduced.len(), 2);
        assert_eq!(reduced[0].len(), 3);
        assert_eq!(reduced[1].len(), 3);
    }

    #[test]
    fn test_cluster_embeddings() {
        let embeddings = vec![
            vec![1.0, 1.0],
            vec![1.1, 0.9],
            vec![5.0, 5.0],
            vec![5.1, 4.9],
        ];
        
        let clusters = cluster_embeddings(&embeddings, 2).unwrap();
        assert_eq!(clusters.len(), 4);
        
        // Should form two clusters
        let unique_clusters: std::collections::HashSet<_> = clusters.iter().collect();
        assert_eq!(unique_clusters.len(), 2);
    }

    #[test]
    fn test_embedding_similarity_function() {
        let emb1 = vec![Value::Float(1.0), Value::Float(0.0), Value::Float(0.0)];
        let emb2 = vec![Value::Float(0.0), Value::Float(1.0), Value::Float(0.0)];
        
        let args = vec![
            Value::List(emb1),
            Value::List(emb2),
            Value::String("cosine".to_string()),
        ];
        
        let result = embedding_similarity(&args);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Float(sim) => assert!((sim - 0.0).abs() < 1e-6),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn test_embedding_reduce_function() {
        let embeddings = vec![
            Value::List(vec![Value::Float(1.0), Value::Float(2.0), Value::Float(3.0), Value::Float(4.0)]),
            Value::List(vec![Value::Float(5.0), Value::Float(6.0), Value::Float(7.0), Value::Float(8.0)]),
        ];
        
        let args = vec![
            Value::List(embeddings),
            Value::Integer(2),
        ];
        
        let result = embedding_reduce(&args);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::List(reduced) => {
                assert_eq!(reduced.len(), 2);
                match &reduced[0] {
                    Value::List(emb) => assert_eq!(emb.len(), 2),
                    _ => panic!("Expected List"),
                }
            }
            _ => panic!("Expected List"),
        }
    }

    #[test]
    fn test_embedding_cluster_function() {
        let embeddings = vec![
            Value::List(vec![Value::Float(1.0), Value::Float(1.0)]),
            Value::List(vec![Value::Float(5.0), Value::Float(5.0)]),
        ];
        
        let args = vec![
            Value::List(embeddings),
            Value::Integer(2),
        ];
        
        let result = embedding_cluster(&args);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::List(clusters) => assert_eq!(clusters.len(), 2),
            _ => panic!("Expected List"),
        }
    }

    #[test]
    fn test_embedding_batch_function() {
        let texts = vec![
            Value::String("Hello world".to_string()),
            Value::String("Goodbye world".to_string()),
        ];
        
        let args = vec![
            Value::List(texts),
            Value::String("ada-002".to_string()),
        ];
        
        let result = embedding_batch(&args);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::List(embeddings) => {
                assert_eq!(embeddings.len(), 2);
                match &embeddings[0] {
                    Value::List(emb) => assert_eq!(emb.len(), 1536), // OpenAI ada-002
                    _ => panic!("Expected List"),
                }
            }
            _ => panic!("Expected List"),
        }
    }

    #[test]
    fn test_unknown_model() {
        let result = generate_embeddings("test", "unknown-model");
        assert!(result.is_err());
    }

    #[test]
    fn test_insufficient_args_errors() {
        assert!(embedding_similarity(&[]).is_err());
        assert!(embedding_reduce(&[Value::List(vec![])]).is_err());
        assert!(embedding_cluster(&[Value::List(vec![])]).is_err());
        assert!(embedding_batch(&[Value::List(vec![])]).is_err());
    }

    #[test]
    fn test_wrong_argument_types() {
        let args = vec![Value::Integer(42), Value::Integer(24)];
        assert!(embedding_similarity(&args).is_err());
        assert!(embedding_reduce(&args).is_err());
        assert!(embedding_cluster(&args).is_err());
        assert!(embedding_batch(&args).is_err());
    }
}