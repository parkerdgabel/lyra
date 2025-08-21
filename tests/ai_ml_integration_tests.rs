//! Phase 14A: AI/ML Integration Tests
//!
//! Comprehensive integration tests for all AI/ML functionality including:
//! - LLM client operations with mock API responses
//! - Vector database operations and similarity search
//! - RAG pipeline end-to-end workflows
//! - Model management and inference
//! - Embedding generation and clustering
//!
//! All tests use mocked external APIs for deterministic behavior.

use lyra::vm::{Value, VirtualMachine};
use lyra::stdlib::StandardLibrary;
use lyra::error::LyraError;
use std::collections::HashMap;

// ============================================================================
// Test Utilities
// ============================================================================

fn create_test_vm() -> VirtualMachine {
    let stdlib = StandardLibrary::new();
    VirtualMachine::new_with_stdlib(stdlib)
}

fn assert_value_type(value: &Value, expected_type: &str) {
    match (value, expected_type) {
        (Value::String(_), "String") => (),
        (Value::Integer(_), "Integer") => (),
        (Value::Float(_), "Float") => (),
        (Value::List(_), "List") => (),
        (Value::Dict(_), "Dict") => (),
        (Value::LyObj(_), "LyObj") => (),
        _ => panic!("Expected {} but got {:?}", expected_type, value),
    }
}

fn extract_string_value(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        _ => panic!("Expected String value"),
    }
}

fn extract_list_value(value: &Value) -> &Vec<Value> {
    match value {
        Value::List(list) => list,
        _ => panic!("Expected List value"),
    }
}

fn extract_dict_value(value: &Value) -> &HashMap<String, Value> {
    match value {
        Value::Dict(dict) => dict,
        _ => panic!("Expected Dict value"),
    }
}

// ============================================================================
// LLM Integration Tests
// ============================================================================

#[test]
fn test_openai_chat_client_creation() {
    let vm = create_test_vm();
    
    let args = vec![
        Value::String("gpt-4".to_string()),
        Value::String("test-api-key".to_string()),
    ];
    
    let result = vm.call_stdlib_function("OpenAIChat", &args);
    assert!(result.is_ok());
    assert_value_type(&result.unwrap(), "LyObj");
}

#[test]
fn test_anthropic_chat_client_creation() {
    let vm = create_test_vm();
    
    let args = vec![
        Value::String("claude-3-sonnet-20240229".to_string()),
        Value::String("test-api-key".to_string()),
    ];
    
    let result = vm.call_stdlib_function("AnthropicChat", &args);
    assert!(result.is_ok());
    assert_value_type(&result.unwrap(), "LyObj");
}

#[test]
fn test_local_llm_client_creation() {
    let vm = create_test_vm();
    
    let args = vec![
        Value::String("llama2".to_string()),
        Value::String("http://localhost:11434".to_string()),
    ];
    
    let result = vm.call_stdlib_function("LocalLLM", &args);
    assert!(result.is_ok());
    assert_value_type(&result.unwrap(), "LyObj");
}

#[test]
fn test_token_count_function() {
    let vm = create_test_vm();
    
    let args = vec![
        Value::String("Hello world this is a test message".to_string()),
        Value::String("gpt-4".to_string()),
    ];
    
    let result = vm.call_stdlib_function("TokenCount", &args);
    assert!(result.is_ok());
    
    match result.unwrap() {
        Value::Integer(count) => assert!(count > 0),
        _ => panic!("Expected Integer"),
    }
}

#[test]
fn test_chat_completion_function() {
    let vm = create_test_vm();
    
    let args = vec![
        Value::String("What is artificial intelligence?".to_string()),
        Value::String("gpt-4".to_string()),
    ];
    
    let result = vm.call_stdlib_function("ChatCompletion", &args);
    assert!(result.is_ok());
    
    let response = extract_string_value(&result.unwrap());
    assert!(response.contains("artificial intelligence") || response.contains("AI"));
}

#[test]
fn test_llm_embedding_function() {
    let vm = create_test_vm();
    
    let args = vec![
        Value::String("Generate embedding for this text".to_string()),
        Value::String("text-embedding-ada-002".to_string()),
    ];
    
    let result = vm.call_stdlib_function("LLMEmbedding", &args);
    assert!(result.is_ok());
    
    let embedding = extract_list_value(&result.unwrap());
    assert_eq!(embedding.len(), 1536); // OpenAI ada-002 dimension
}

#[test]
fn test_llm_batch_processing() {
    let vm = create_test_vm();
    
    // First create a client
    let client_args = vec![
        Value::String("gpt-4".to_string()),
        Value::String("test-key".to_string()),
    ];
    let client = vm.call_stdlib_function("OpenAIChat", &client_args).unwrap();
    
    // Then test batch processing
    let prompts = vec![
        Value::String("What is AI?".to_string()),
        Value::String("What is ML?".to_string()),
        Value::String("What is NLP?".to_string()),
    ];
    
    let batch_args = vec![
        client,
        Value::List(prompts),
    ];
    
    let result = vm.call_stdlib_function("LLMBatch", &batch_args);
    assert!(result.is_ok());
    
    let responses = extract_list_value(&result.unwrap());
    assert_eq!(responses.len(), 3);
}

// ============================================================================
// Vector Database Tests
// ============================================================================

#[test]
fn test_vector_store_creation() {
    let vm = create_test_vm();
    
    let args = vec![
        Value::String("test_embeddings".to_string()),
        Value::Integer(1536),
        Value::String("cosine".to_string()),
    ];
    
    let result = vm.call_stdlib_function("VectorStore", &args);
    assert!(result.is_ok());
    assert_value_type(&result.unwrap(), "LyObj");
}

#[test]
fn test_embedding_generation() {
    let vm = create_test_vm();
    
    let args = vec![
        Value::String("This is a test document for embedding generation".to_string()),
        Value::String("text-embedding-ada-002".to_string()),
    ];
    
    let result = vm.call_stdlib_function("EmbeddingGenerate", &args);
    assert!(result.is_ok());
    
    let embedding = extract_list_value(&result.unwrap());
    assert_eq!(embedding.len(), 1536);
    
    // Verify embedding values are floats in reasonable range
    for value in embedding {
        match value {
            Value::Float(f) => assert!(f.abs() <= 1.0),
            _ => panic!("Expected Float in embedding"),
        }
    }
}

#[test]
fn test_vector_insert_and_search() {
    let vm = create_test_vm();
    
    // Create vector store
    let store_args = vec![
        Value::String("test_store".to_string()),
        Value::Integer(3),
        Value::String("cosine".to_string()),
    ];
    let store = vm.call_stdlib_function("VectorStore", &store_args).unwrap();
    
    // Insert vectors
    let vector1 = vec![Value::Float(1.0), Value::Float(0.0), Value::Float(0.0)];
    let vector2 = vec![Value::Float(0.0), Value::Float(1.0), Value::Float(0.0)];
    let vector3 = vec![Value::Float(0.5), Value::Float(0.5), Value::Float(0.0)];
    
    let insert_args1 = vec![
        store.clone(),
        Value::String("vec1".to_string()),
        Value::List(vector1),
    ];
    vm.call_stdlib_function("VectorInsert", &insert_args1).unwrap();
    
    let insert_args2 = vec![
        store.clone(),
        Value::String("vec2".to_string()),
        Value::List(vector2),
    ];
    vm.call_stdlib_function("VectorInsert", &insert_args2).unwrap();
    
    let insert_args3 = vec![
        store.clone(),
        Value::String("vec3".to_string()),
        Value::List(vector3),
    ];
    vm.call_stdlib_function("VectorInsert", &insert_args3).unwrap();
    
    // Search for similar vectors
    let query_vector = vec![Value::Float(1.0), Value::Float(0.1), Value::Float(0.0)];
    let search_args = vec![
        store,
        Value::List(query_vector),
        Value::Integer(2),
        Value::Dict(HashMap::new()), // No filter
    ];
    
    let result = vm.call_stdlib_function("VectorSearch", &search_args);
    assert!(result.is_ok());
    
    let search_results = extract_list_value(&result.unwrap());
    assert_eq!(search_results.len(), 2);
    
    // Verify results have required fields
    for result_item in search_results {
        let result_dict = extract_dict_value(result_item);
        assert!(result_dict.contains_key("id"));
        assert!(result_dict.contains_key("score"));
        assert!(result_dict.contains_key("distance"));
    }
}

#[test]
fn test_vector_similarity_calculation() {
    let vm = create_test_vm();
    
    let vec1 = vec![Value::Float(1.0), Value::Float(0.0), Value::Float(0.0)];
    let vec2 = vec![Value::Float(0.0), Value::Float(1.0), Value::Float(0.0)];
    
    let args = vec![
        Value::List(vec1),
        Value::List(vec2),
        Value::String("cosine".to_string()),
    ];
    
    let result = vm.call_stdlib_function("VectorSimilarity", &args);
    assert!(result.is_ok());
    
    match result.unwrap() {
        Value::Float(similarity) => {
            assert!((similarity - 0.0).abs() < 1e-6); // Orthogonal vectors
        }
        _ => panic!("Expected Float"),
    }
}

#[test]
fn test_vector_clustering() {
    let vm = create_test_vm();
    
    // Create vector store with test data
    let store_args = vec![
        Value::String("cluster_test".to_string()),
        Value::Integer(2),
        Value::String("euclidean".to_string()),
    ];
    let store = vm.call_stdlib_function("VectorStore", &store_args).unwrap();
    
    // Insert test vectors forming two clusters
    let vectors = vec![
        (vec![Value::Float(1.0), Value::Float(1.0)], "cluster1_1"),
        (vec![Value::Float(1.1), Value::Float(0.9)], "cluster1_2"),
        (vec![Value::Float(5.0), Value::Float(5.0)], "cluster2_1"),
        (vec![Value::Float(5.1), Value::Float(4.9)], "cluster2_2"),
    ];
    
    for (vector, id) in vectors {
        let insert_args = vec![
            store.clone(),
            Value::String(id.to_string()),
            Value::List(vector),
        ];
        vm.call_stdlib_function("VectorInsert", &insert_args).unwrap();
    }
    
    // Perform clustering
    let mut cluster_params = HashMap::new();
    cluster_params.insert("k".to_string(), Value::Integer(2));
    
    let cluster_args = vec![
        store,
        Value::String("kmeans".to_string()),
        Value::Dict(cluster_params),
    ];
    
    let result = vm.call_stdlib_function("VectorCluster", &cluster_args);
    assert!(result.is_ok());
    
    let clusters = extract_list_value(&result.unwrap());
    assert_eq!(clusters.len(), 2); // Should find 2 clusters
}

// ============================================================================
// RAG Pipeline Tests
// ============================================================================

#[test]
fn test_document_chunking() {
    let vm = create_test_vm();
    
    let document = "This is the first sentence. This is the second sentence. \
                   This is the third sentence. This is the fourth sentence.";
    
    let args = vec![
        Value::String(document.to_string()),
        Value::String("sentence".to_string()),
        Value::Integer(100),
        Value::Integer(20),
    ];
    
    let result = vm.call_stdlib_function("DocumentChunk", &args);
    assert!(result.is_ok());
    
    let chunks = extract_list_value(&result.unwrap());
    assert!(!chunks.is_empty());
    
    // Verify chunk structure
    for chunk in chunks {
        let chunk_dict = extract_dict_value(chunk);
        assert!(chunk_dict.contains_key("id"));
        assert!(chunk_dict.contains_key("content"));
        assert!(chunk_dict.contains_key("start"));
        assert!(chunk_dict.contains_key("end"));
    }
}

#[test]
fn test_chunk_embedding_generation() {
    let vm = create_test_vm();
    
    let chunks = vec![
        {
            let mut chunk = HashMap::new();
            chunk.insert("content".to_string(), Value::String("First chunk content".to_string()));
            Value::Dict(chunk)
        },
        {
            let mut chunk = HashMap::new();
            chunk.insert("content".to_string(), Value::String("Second chunk content".to_string()));
            Value::Dict(chunk)
        },
    ];
    
    let args = vec![
        Value::List(chunks),
        Value::String("text-embedding-ada-002".to_string()),
    ];
    
    let result = vm.call_stdlib_function("ChunkEmbedding", &args);
    assert!(result.is_ok());
    
    let embeddings = extract_list_value(&result.unwrap());
    assert_eq!(embeddings.len(), 2);
    
    // Verify each embedding has correct dimension
    for embedding in embeddings {
        let embedding_vec = extract_list_value(embedding);
        assert_eq!(embedding_vec.len(), 1536);
    }
}

#[test]
fn test_context_retrieval() {
    let vm = create_test_vm();
    
    // Create a mock vector store
    let store_args = vec![
        Value::String("rag_store".to_string()),
        Value::Integer(1536),
        Value::String("cosine".to_string()),
    ];
    let store = vm.call_stdlib_function("VectorStore", &store_args).unwrap();
    
    let args = vec![
        store,
        Value::String("What is machine learning?".to_string()),
        Value::Integer(3),
        Value::Dict(HashMap::new()),
    ];
    
    let result = vm.call_stdlib_function("ContextRetrieval", &args);
    assert!(result.is_ok());
    
    let contexts = extract_list_value(&result.unwrap());
    assert_eq!(contexts.len(), 3);
    
    // Verify context structure
    for context in contexts {
        let context_dict = extract_dict_value(context);
        assert!(context_dict.contains_key("id"));
        assert!(context_dict.contains_key("content"));
        assert!(context_dict.contains_key("score"));
    }
}

#[test]
fn test_rag_query_processing() {
    let vm = create_test_vm();
    
    let contexts = vec![
        {
            let mut context = HashMap::new();
            context.insert("content".to_string(), Value::String("Machine learning is a subset of AI".to_string()));
            context.insert("score".to_string(), Value::Float(0.95));
            Value::Dict(context)
        },
        {
            let mut context = HashMap::new();
            context.insert("content".to_string(), Value::String("Deep learning uses neural networks".to_string()));
            context.insert("score".to_string(), Value::Float(0.87));
            Value::Dict(context)
        },
    ];
    
    let template = "Context: {context}\nQuestion: {question}\nAnswer:";
    
    let args = vec![
        Value::String("What is machine learning?".to_string()),
        Value::List(contexts),
        Value::String("gpt-4".to_string()),
        Value::String(template.to_string()),
    ];
    
    let result = vm.call_stdlib_function("RAGQuery", &args);
    assert!(result.is_ok());
    
    let answer = extract_string_value(&result.unwrap());
    assert!(answer.contains("machine learning") || answer.contains("What is machine learning?"));
}

#[test]
fn test_context_ranking() {
    let vm = create_test_vm();
    
    let contexts = vec![
        {
            let mut context = HashMap::new();
            context.insert("content".to_string(), Value::String("Lower relevance content".to_string()));
            context.insert("score".to_string(), Value::Float(0.6));
            Value::Dict(context)
        },
        {
            let mut context = HashMap::new();
            context.insert("content".to_string(), Value::String("Higher relevance content".to_string()));
            context.insert("score".to_string(), Value::Float(0.9));
            Value::Dict(context)
        },
        {
            let mut context = HashMap::new();
            context.insert("content".to_string(), Value::String("Medium relevance content".to_string()));
            context.insert("score".to_string(), Value::Float(0.75));
            Value::Dict(context)
        },
    ];
    
    let args = vec![
        Value::List(contexts),
        Value::String("test query".to_string()),
        Value::String("similarity".to_string()),
    ];
    
    let result = vm.call_stdlib_function("ContextRank", &args);
    assert!(result.is_ok());
    
    let ranked_contexts = extract_list_value(&result.unwrap());
    assert_eq!(ranked_contexts.len(), 3);
    
    // Verify contexts are sorted by score (descending)
    let scores: Vec<f64> = ranked_contexts.iter()
        .map(|context| {
            let context_dict = extract_dict_value(context);
            match context_dict.get("score") {
                Some(Value::Float(score)) => *score,
                _ => 0.0,
            }
        })
        .collect();
    
    assert!(scores[0] >= scores[1]);
    assert!(scores[1] >= scores[2]);
}

#[test]
fn test_document_indexing() {
    let vm = create_test_vm();
    
    let documents = vec![
        Value::String("First document about artificial intelligence".to_string()),
        Value::String("Second document about machine learning algorithms".to_string()),
        Value::String("Third document about deep learning neural networks".to_string()),
    ];
    
    let args = vec![
        Value::List(documents),
        Value::String("recursive".to_string()),
        Value::String("text-embedding-ada-002".to_string()),
    ];
    
    let result = vm.call_stdlib_function("DocumentIndex", &args);
    assert!(result.is_ok());
    
    let index_result = extract_dict_value(&result.unwrap());
    assert!(index_result.contains_key("indexed_documents"));
    assert!(index_result.contains_key("total_chunks"));
    assert!(index_result.contains_key("status"));
    
    match index_result.get("indexed_documents") {
        Some(Value::Integer(count)) => assert_eq!(*count, 3),
        _ => panic!("Expected indexed_documents count"),
    }
}

#[test]
fn test_context_window_management() {
    let vm = create_test_vm();
    
    let contexts = vec![
        {
            let mut context = HashMap::new();
            context.insert("content".to_string(), Value::String("Short content".to_string()));
            Value::Dict(context)
        },
        {
            let mut context = HashMap::new();
            context.insert("content".to_string(), Value::String("This is much longer content that might exceed the context window limit".to_string()));
            Value::Dict(context)
        },
        {
            let mut context = HashMap::new();
            context.insert("content".to_string(), Value::String("Another piece of content".to_string()));
            Value::Dict(context)
        },
    ];
    
    let args = vec![
        Value::List(contexts),
        Value::Integer(50), // Small context window
        Value::String("truncate".to_string()),
    ];
    
    let result = vm.call_stdlib_function("ContextWindow", &args);
    assert!(result.is_ok());
    
    let windowed_contexts = extract_list_value(&result.unwrap());
    assert!(!windowed_contexts.is_empty());
    
    // Verify total content length is within limit
    let total_length: usize = windowed_contexts.iter()
        .map(|context| {
            let context_dict = extract_dict_value(context);
            match context_dict.get("content") {
                Some(Value::String(content)) => content.len(),
                _ => 0,
            }
        })
        .sum();
    
    assert!(total_length <= 50);
}

// ============================================================================
// Model Management Tests
// ============================================================================

#[test]
fn test_model_loading_error_handling() {
    let vm = create_test_vm();
    
    // Try to load non-existent model file
    let args = vec![
        Value::String("/nonexistent/path/model.onnx".to_string()),
        Value::String("onnx".to_string()),
        Value::String("cpu".to_string()),
    ];
    
    let result = vm.call_stdlib_function("ModelLoad", &args);
    assert!(result.is_err()); // Should fail because file doesn't exist
}

#[test]
fn test_model_inference_argument_validation() {
    let vm = create_test_vm();
    
    // Test with invalid model object
    let args = vec![
        Value::String("not_a_model".to_string()),
        Value::List(vec![Value::Float(1.0), Value::Float(2.0)]),
    ];
    
    let result = vm.call_stdlib_function("ModelInference", &args);
    assert!(result.is_err());
}

#[test]
fn test_model_fine_tuning_mock() {
    let vm = create_test_vm();
    
    // Create mock training data
    let training_data = vec![
        {
            let mut sample = HashMap::new();
            sample.insert("input".to_string(), Value::List(vec![Value::Float(1.0), Value::Float(2.0)]));
            sample.insert("output".to_string(), Value::List(vec![Value::Float(0.5)]));
            Value::Dict(sample)
        }
    ];
    
    let mut parameters = HashMap::new();
    parameters.insert("learning_rate".to_string(), Value::Float(0.001));
    parameters.insert("epochs".to_string(), Value::Integer(5));
    parameters.insert("batch_size".to_string(), Value::Integer(16));
    
    let args = vec![
        Value::String("mock_model".to_string()), // Invalid model for testing
        Value::List(training_data),
        Value::Dict(parameters),
    ];
    
    let result = vm.call_stdlib_function("ModelFineTune", &args);
    assert!(result.is_err()); // Should fail with invalid model
}

#[test]
fn test_model_validation_mock() {
    let vm = create_test_vm();
    
    let test_data = vec![
        {
            let mut sample = HashMap::new();
            sample.insert("input".to_string(), Value::List(vec![Value::Float(1.0)]));
            sample.insert("expected".to_string(), Value::List(vec![Value::Float(0.8)]));
            Value::Dict(sample)
        }
    ];
    
    let metrics = vec![
        Value::String("accuracy".to_string()),
        Value::String("precision".to_string()),
        Value::String("recall".to_string()),
    ];
    
    let args = vec![
        Value::String("mock_model".to_string()),
        Value::List(test_data),
        Value::List(metrics),
    ];
    
    let result = vm.call_stdlib_function("ModelValidate", &args);
    assert!(result.is_err()); // Should fail with invalid model
}

// ============================================================================
// Embedding Functionality Tests
// ============================================================================

#[test]
fn test_embedding_similarity_calculations() {
    let vm = create_test_vm();
    
    // Test identical embeddings
    let embedding = vec![Value::Float(1.0), Value::Float(0.0), Value::Float(0.0)];
    
    let args = vec![
        Value::List(embedding.clone()),
        Value::List(embedding),
        Value::String("cosine".to_string()),
    ];
    
    let result = vm.call_stdlib_function("EmbeddingSimilarity", &args);
    assert!(result.is_ok());
    
    match result.unwrap() {
        Value::Float(similarity) => assert!((similarity - 1.0).abs() < 1e-6),
        _ => panic!("Expected Float"),
    }
}

#[test]
fn test_embedding_batch_processing() {
    let vm = create_test_vm();
    
    let texts = vec![
        Value::String("First text for embedding".to_string()),
        Value::String("Second text for embedding".to_string()),
        Value::String("Third text for embedding".to_string()),
    ];
    
    let args = vec![
        Value::List(texts),
        Value::String("sentence-transformers".to_string()),
    ];
    
    let result = vm.call_stdlib_function("EmbeddingBatch", &args);
    assert!(result.is_ok());
    
    let embeddings = extract_list_value(&result.unwrap());
    assert_eq!(embeddings.len(), 3);
    
    // Verify each embedding
    for embedding in embeddings {
        let embedding_vec = extract_list_value(embedding);
        assert_eq!(embedding_vec.len(), 768); // Sentence transformer dimension
    }
}

#[test]
fn test_embedding_dimension_reduction() {
    let vm = create_test_vm();
    
    let embeddings = vec![
        Value::List((0..1536).map(|i| Value::Float(i as f64 * 0.001)).collect()),
        Value::List((0..1536).map(|i| Value::Float((i + 100) as f64 * 0.001)).collect()),
        Value::List((0..1536).map(|i| Value::Float((i + 200) as f64 * 0.001)).collect()),
    ];
    
    let args = vec![
        Value::List(embeddings),
        Value::Integer(256),
    ];
    
    let result = vm.call_stdlib_function("EmbeddingReduce", &args);
    assert!(result.is_ok());
    
    let reduced_embeddings = extract_list_value(&result.unwrap());
    assert_eq!(reduced_embeddings.len(), 3);
    
    for embedding in reduced_embeddings {
        let embedding_vec = extract_list_value(embedding);
        assert_eq!(embedding_vec.len(), 256);
    }
}

#[test]
fn test_embedding_clustering() {
    let vm = create_test_vm();
    
    // Create embeddings that should form distinct clusters
    let embeddings = vec![
        Value::List(vec![Value::Float(1.0), Value::Float(1.0)]),
        Value::List(vec![Value::Float(1.1), Value::Float(0.9)]),
        Value::List(vec![Value::Float(5.0), Value::Float(5.0)]),
        Value::List(vec![Value::Float(5.1), Value::Float(4.9)]),
    ];
    
    let args = vec![
        Value::List(embeddings),
        Value::Integer(2),
    ];
    
    let result = vm.call_stdlib_function("EmbeddingCluster", &args);
    assert!(result.is_ok());
    
    let cluster_assignments = extract_list_value(&result.unwrap());
    assert_eq!(cluster_assignments.len(), 4);
    
    // Verify cluster assignments are valid integers
    for assignment in cluster_assignments {
        match assignment {
            Value::Integer(cluster_id) => assert!(*cluster_id >= 0 && *cluster_id < 2),
            _ => panic!("Expected Integer cluster assignment"),
        }
    }
}

// ============================================================================
// Error Handling and Edge Cases
// ============================================================================

#[test]
fn test_insufficient_arguments_errors() {
    let vm = create_test_vm();
    
    // Test various functions with insufficient arguments
    assert!(vm.call_stdlib_function("OpenAIChat", &[]).is_err());
    assert!(vm.call_stdlib_function("VectorStore", &[Value::String("test".to_string())]).is_err());
    assert!(vm.call_stdlib_function("DocumentChunk", &[]).is_err());
    assert!(vm.call_stdlib_function("EmbeddingGenerate", &[Value::String("text".to_string())]).is_err());
}

#[test]
fn test_wrong_argument_types() {
    let vm = create_test_vm();
    
    // Test functions with wrong argument types
    let wrong_args = vec![Value::Integer(42), Value::Integer(24)];
    
    assert!(vm.call_stdlib_function("OpenAIChat", &wrong_args).is_err());
    assert!(vm.call_stdlib_function("VectorStore", &wrong_args).is_err());
    assert!(vm.call_stdlib_function("EmbeddingGenerate", &wrong_args).is_err());
}

#[test]
fn test_empty_input_handling() {
    let vm = create_test_vm();
    
    // Test with empty strings
    let args = vec![
        Value::String("".to_string()),
        Value::String("text-embedding-ada-002".to_string()),
    ];
    
    let result = vm.call_stdlib_function("EmbeddingGenerate", &args);
    assert!(result.is_ok()); // Should handle empty strings gracefully
}

#[test]
fn test_large_input_handling() {
    let vm = create_test_vm();
    
    // Test with very large text input
    let large_text = "word ".repeat(10000);
    let args = vec![
        Value::String(large_text),
        Value::String("sentence".to_string()),
        Value::Integer(1000),
        Value::Integer(100),
    ];
    
    let result = vm.call_stdlib_function("DocumentChunk", &args);
    assert!(result.is_ok());
    
    let chunks = extract_list_value(&result.unwrap());
    assert!(!chunks.is_empty());
}

// ============================================================================
// Performance and Stress Tests
// ============================================================================

#[test]
fn test_concurrent_vector_operations() {
    use std::thread;
    use std::sync::Arc;
    
    let vm = Arc::new(create_test_vm());
    
    // Create vector store
    let store_args = vec![
        Value::String("concurrent_test".to_string()),
        Value::Integer(10),
        Value::String("cosine".to_string()),
    ];
    let store = vm.call_stdlib_function("VectorStore", &store_args).unwrap();
    let store = Arc::new(store);
    
    // Spawn multiple threads to perform concurrent operations
    let mut handles = vec![];
    
    for i in 0..5 {
        let vm_clone = vm.clone();
        let store_clone = store.clone();
        
        let handle = thread::spawn(move || {
            let vector = vec![
                Value::Float(i as f64),
                Value::Float((i + 1) as f64),
                Value::Float((i + 2) as f64),
            ];
            
            let insert_args = vec![
                (*store_clone).clone(),
                Value::String(format!("vec_{}", i)),
                Value::List(vector),
            ];
            
            vm_clone.call_stdlib_function("VectorInsert", &insert_args)
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        let result = handle.join().unwrap();
        assert!(result.is_ok());
    }
}

#[test]
fn test_memory_usage_with_large_embeddings() {
    let vm = create_test_vm();
    
    // Generate multiple large embeddings
    let mut large_embeddings = Vec::new();
    for i in 0..100 {
        let text = format!("This is test document number {} with some content", i);
        let args = vec![
            Value::String(text),
            Value::String("text-embedding-ada-002".to_string()),
        ];
        
        let result = vm.call_stdlib_function("EmbeddingGenerate", &args);
        assert!(result.is_ok());
        large_embeddings.push(result.unwrap());
    }
    
    assert_eq!(large_embeddings.len(), 100);
    
    // Test clustering with large dataset
    let cluster_args = vec![
        Value::List(large_embeddings),
        Value::Integer(5),
    ];
    
    let result = vm.call_stdlib_function("EmbeddingCluster", &cluster_args);
    assert!(result.is_ok());
}

// ============================================================================
// Integration Workflow Tests
// ============================================================================

#[test]
fn test_complete_rag_workflow() {
    let vm = create_test_vm();
    
    // Step 1: Create vector store
    let store_args = vec![
        Value::String("rag_workflow_test".to_string()),
        Value::Integer(1536),
        Value::String("cosine".to_string()),
    ];
    let store = vm.call_stdlib_function("VectorStore", &store_args).unwrap();
    
    // Step 2: Chunk documents
    let document = "Artificial intelligence is a field of computer science. \
                   Machine learning is a subset of AI. Deep learning uses neural networks. \
                   Natural language processing deals with text understanding.";
    
    let chunk_args = vec![
        Value::String(document.to_string()),
        Value::String("sentence".to_string()),
        Value::Integer(200),
        Value::Integer(50),
    ];
    let chunks = vm.call_stdlib_function("DocumentChunk", &chunk_args).unwrap();
    
    // Step 3: Generate embeddings for chunks
    let embedding_args = vec![
        chunks,
        Value::String("text-embedding-ada-002".to_string()),
    ];
    let embeddings = vm.call_stdlib_function("ChunkEmbedding", &embedding_args).unwrap();
    
    // Verify workflow completed successfully
    assert_value_type(&store, "LyObj");
    assert_value_type(&embeddings, "List");
    
    let embedding_list = extract_list_value(&embeddings);
    assert!(!embedding_list.is_empty());
}

#[test]
fn test_llm_to_vector_integration() {
    let vm = create_test_vm();
    
    // Step 1: Create LLM client
    let client_args = vec![
        Value::String("gpt-4".to_string()),
        Value::String("test-key".to_string()),
    ];
    let llm_client = vm.call_stdlib_function("OpenAIChat", &client_args).unwrap();
    
    // Step 2: Generate embeddings
    let embedding_args = vec![
        Value::String("Test query for embedding".to_string()),
        Value::String("text-embedding-ada-002".to_string()),
    ];
    let query_embedding = vm.call_stdlib_function("EmbeddingGenerate", &embedding_args).unwrap();
    
    // Step 3: Create vector store and perform search
    let store_args = vec![
        Value::String("integration_test".to_string()),
        Value::Integer(1536),
        Value::String("cosine".to_string()),
    ];
    let store = vm.call_stdlib_function("VectorStore", &store_args).unwrap();
    
    // Verify all components were created successfully
    assert_value_type(&llm_client, "LyObj");
    assert_value_type(&query_embedding, "List");
    assert_value_type(&store, "LyObj");
}

#[test]
fn test_end_to_end_ai_ml_pipeline() {
    let vm = create_test_vm();
    
    // This test verifies that all major AI/ML components can work together
    
    // 1. Document processing
    let documents = vec![
        Value::String("AI and machine learning overview".to_string()),
        Value::String("Deep learning neural networks".to_string()),
        Value::String("Natural language processing techniques".to_string()),
    ];
    
    let index_args = vec![
        Value::List(documents),
        Value::String("sentence".to_string()),
        Value::String("text-embedding-ada-002".to_string()),
    ];
    let index_result = vm.call_stdlib_function("DocumentIndex", &index_args).unwrap();
    
    // 2. Vector similarity testing
    let vec1 = vec![Value::Float(1.0), Value::Float(0.0), Value::Float(0.0)];
    let vec2 = vec![Value::Float(0.9), Value::Float(0.1), Value::Float(0.0)];
    
    let similarity_args = vec![
        Value::List(vec1),
        Value::List(vec2),
        Value::String("cosine".to_string()),
    ];
    let similarity = vm.call_stdlib_function("VectorSimilarity", &similarity_args).unwrap();
    
    // 3. LLM query processing
    let chat_args = vec![
        Value::String("What is artificial intelligence?".to_string()),
        Value::String("gpt-4".to_string()),
    ];
    let chat_response = vm.call_stdlib_function("ChatCompletion", &chat_args).unwrap();
    
    // Verify all pipeline stages completed successfully
    assert_value_type(&index_result, "Dict");
    assert_value_type(&similarity, "Float");
    assert_value_type(&chat_response, "String");
    
    // Verify reasonable outputs
    match similarity {
        Value::Float(sim) => assert!(sim > 0.8), // Should be high similarity
        _ => panic!("Expected Float similarity"),
    }
    
    let response_text = extract_string_value(&chat_response);
    assert!(!response_text.is_empty());
}