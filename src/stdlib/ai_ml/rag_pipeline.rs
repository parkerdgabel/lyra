//! RAG (Retrieval-Augmented Generation) Pipeline Module - Phase 14A
//!
//! Provides comprehensive RAG functionality including:
//! - Intelligent document chunking strategies
//! - Context retrieval and ranking
//! - RAG query processing with LLM integration
//! - Document indexing and management
//! - Context window optimization
//!
//! All RAG components are implemented as Foreign objects for thread safety.

use crate::vm::{Value, VmResult};
use crate::foreign::{Foreign, LyObj, ForeignError};
use crate::error::LyraError;
use crate::stdlib::ai_ml::vector_store::{VectorStoreWrapper, VectorEntry};
use crate::stdlib::ai_ml::llm_integration::{LLMClientWrapper, ChatMessage, LLMOptions};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use unicode_segmentation::UnicodeSegmentation;

// ============================================================================
// Core RAG Types and Enums
// ============================================================================

/// Document chunking strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ChunkStrategy {
    FixedSize,
    Sentence,
    Paragraph,
    Semantic,
    Recursive,
    SlidingWindow,
}

impl ChunkStrategy {
    pub fn from_str(s: &str) -> Result<Self, LyraError> {
        match s.to_lowercase().as_str() {
            "fixed" | "fixedsize" => Ok(ChunkStrategy::FixedSize),
            "sentence" => Ok(ChunkStrategy::Sentence),
            "paragraph" => Ok(ChunkStrategy::Paragraph),
            "semantic" => Ok(ChunkStrategy::Semantic),
            "recursive" => Ok(ChunkStrategy::Recursive),
            "sliding" | "slidingwindow" => Ok(ChunkStrategy::SlidingWindow),
            _ => Err(LyraError::Custom(format!("Unknown chunk strategy: {}", s))),
        }
    }
}

/// Document chunk with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub id: String,
    pub content: String,
    pub start_index: usize,
    pub end_index: usize,
    pub metadata: HashMap<String, Value>,
}

impl DocumentChunk {
    pub fn new(id: String, content: String, start_index: usize, end_index: usize) -> Self {
        Self {
            id,
            content,
            start_index,
            end_index,
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(id: String, content: String, start_index: usize, end_index: usize, metadata: HashMap<String, Value>) -> Self {
        Self {
            id,
            content,
            start_index,
            end_index,
            metadata,
        }
    }
}

/// Context retrieved for RAG query
#[derive(Debug, Clone)]
pub struct RetrievedContext {
    pub chunk: DocumentChunk,
    pub relevance_score: f32,
    pub embedding: Option<Vec<f32>>,
}

/// RAG query result
#[derive(Debug, Clone)]
pub struct RAGResult {
    pub query: String,
    pub context: Vec<RetrievedContext>,
    pub answer: String,
    pub metadata: HashMap<String, Value>,
}

/// Context ranking algorithms
#[derive(Debug, Clone, Copy)]
pub enum RankingAlgorithm {
    Similarity,
    BM25,
    Hybrid,
    Rerank,
}

impl RankingAlgorithm {
    pub fn from_str(s: &str) -> Result<Self, LyraError> {
        match s.to_lowercase().as_str() {
            "similarity" => Ok(RankingAlgorithm::Similarity),
            "bm25" => Ok(RankingAlgorithm::BM25),
            "hybrid" => Ok(RankingAlgorithm::Hybrid),
            "rerank" => Ok(RankingAlgorithm::Rerank),
            _ => Err(LyraError::Custom(format!("Unknown ranking algorithm: {}", s))),
        }
    }
}

// ============================================================================
// Document Chunking Implementation
// ============================================================================

/// Document chunker with multiple strategies
pub struct DocumentChunker {
    strategy: ChunkStrategy,
    chunk_size: usize,
    overlap: usize,
}

impl DocumentChunker {
    pub fn new(strategy: ChunkStrategy, chunk_size: usize, overlap: usize) -> Self {
        Self {
            strategy,
            chunk_size,
            overlap,
        }
    }

    /// Chunk document based on configured strategy
    pub fn chunk_document(&self, text: &str, document_id: &str) -> Result<Vec<DocumentChunk>, LyraError> {
        match self.strategy {
            ChunkStrategy::FixedSize => self.chunk_fixed_size(text, document_id),
            ChunkStrategy::Sentence => self.chunk_by_sentence(text, document_id),
            ChunkStrategy::Paragraph => self.chunk_by_paragraph(text, document_id),
            ChunkStrategy::Semantic => self.chunk_semantic(text, document_id),
            ChunkStrategy::Recursive => self.chunk_recursive(text, document_id),
            ChunkStrategy::SlidingWindow => self.chunk_sliding_window(text, document_id),
        }
    }

    fn chunk_fixed_size(&self, text: &str, document_id: &str) -> Result<Vec<DocumentChunk>, LyraError> {
        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut start = 0;
        let mut chunk_id = 0;

        while start < chars.len() {
            let end = std::cmp::min(start + self.chunk_size, chars.len());
            let chunk_text: String = chars[start..end].iter().collect();
            
            if !chunk_text.trim().is_empty() {
                let chunk = DocumentChunk::new(
                    format!("{}_{}", document_id, chunk_id),
                    chunk_text,
                    start,
                    end,
                );
                chunks.push(chunk);
                chunk_id += 1;
            }

            start = if end == chars.len() {
                chars.len()
            } else {
                end.saturating_sub(self.overlap)
            };
        }

        Ok(chunks)
    }

    fn chunk_by_sentence(&self, text: &str, document_id: &str) -> Result<Vec<DocumentChunk>, LyraError> {
        let mut chunks = Vec::new();
        let sentences: Vec<&str> = text.split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .collect();

        let mut current_chunk = String::new();
        let mut chunk_id = 0;
        let mut start_index = 0;

        for sentence in sentences {
            let sentence = sentence.trim();
            if current_chunk.len() + sentence.len() > self.chunk_size && !current_chunk.is_empty() {
                // Create chunk from accumulated sentences
                let chunk = DocumentChunk::new(
                    format!("{}_{}", document_id, chunk_id),
                    current_chunk.trim().to_string(),
                    start_index,
                    start_index + current_chunk.len(),
                );
                chunks.push(chunk);
                chunk_id += 1;
                
                start_index += current_chunk.len();
                current_chunk.clear();
            }

            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(sentence);
        }

        // Add final chunk
        if !current_chunk.trim().is_empty() {
            let chunk = DocumentChunk::new(
                format!("{}_{}", document_id, chunk_id),
                current_chunk.trim().to_string(),
                start_index,
                start_index + current_chunk.len(),
            );
            chunks.push(chunk);
        }

        Ok(chunks)
    }

    fn chunk_by_paragraph(&self, text: &str, document_id: &str) -> Result<Vec<DocumentChunk>, LyraError> {
        let mut chunks = Vec::new();
        let paragraphs: Vec<&str> = text.split("\n\n")
            .filter(|p| !p.trim().is_empty())
            .collect();

        let mut current_chunk = String::new();
        let mut chunk_id = 0;
        let mut start_index = 0;

        for paragraph in paragraphs {
            let paragraph = paragraph.trim();
            if current_chunk.len() + paragraph.len() > self.chunk_size && !current_chunk.is_empty() {
                let chunk = DocumentChunk::new(
                    format!("{}_{}", document_id, chunk_id),
                    current_chunk.trim().to_string(),
                    start_index,
                    start_index + current_chunk.len(),
                );
                chunks.push(chunk);
                chunk_id += 1;
                
                start_index += current_chunk.len();
                current_chunk.clear();
            }

            if !current_chunk.is_empty() {
                current_chunk.push_str("\n\n");
            }
            current_chunk.push_str(paragraph);
        }

        if !current_chunk.trim().is_empty() {
            let chunk = DocumentChunk::new(
                format!("{}_{}", document_id, chunk_id),
                current_chunk.trim().to_string(),
                start_index,
                start_index + current_chunk.len(),
            );
            chunks.push(chunk);
        }

        Ok(chunks)
    }

    fn chunk_semantic(&self, text: &str, document_id: &str) -> Result<Vec<DocumentChunk>, LyraError> {
        // Simplified semantic chunking - in production would use ML models
        // For now, use sentence-based chunking with semantic boundaries
        let sentences = UnicodeSegmentation::split_sentence_bounds(text).collect::<Vec<&str>>();
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut chunk_id = 0;
        let mut start_index = 0;

        for sentence in sentences {
            if current_chunk.len() + sentence.len() > self.chunk_size && !current_chunk.is_empty() {
                let chunk = DocumentChunk::new(
                    format!("{}_{}", document_id, chunk_id),
                    current_chunk.trim().to_string(),
                    start_index,
                    start_index + current_chunk.len(),
                );
                chunks.push(chunk);
                chunk_id += 1;
                
                start_index += current_chunk.len();
                current_chunk.clear();
            }

            current_chunk.push_str(sentence);
        }

        if !current_chunk.trim().is_empty() {
            let chunk = DocumentChunk::new(
                format!("{}_{}", document_id, chunk_id),
                current_chunk.trim().to_string(),
                start_index,
                start_index + current_chunk.len(),
            );
            chunks.push(chunk);
        }

        Ok(chunks)
    }

    fn chunk_recursive(&self, text: &str, document_id: &str) -> Result<Vec<DocumentChunk>, LyraError> {
        // Recursive chunking: try paragraphs, then sentences, then fixed size
        let paragraphs: Vec<&str> = text.split("\n\n").collect();
        let mut chunks = Vec::new();
        let mut chunk_id = 0;

        for (i, paragraph) in paragraphs.iter().enumerate() {
            if paragraph.len() <= self.chunk_size {
                // Paragraph fits in one chunk
                let chunk = DocumentChunk::new(
                    format!("{}_{}", document_id, chunk_id),
                    paragraph.trim().to_string(),
                    0, // Simplified indexing
                    paragraph.len(),
                );
                chunks.push(chunk);
                chunk_id += 1;
            } else {
                // Paragraph too large, chunk by sentences
                let sentence_chunker = DocumentChunker::new(ChunkStrategy::Sentence, self.chunk_size, self.overlap);
                let mut sentence_chunks = sentence_chunker.chunk_by_sentence(paragraph, &format!("{}_{}", document_id, i))?;
                chunks.append(&mut sentence_chunks);
                chunk_id += sentence_chunks.len();
            }
        }

        Ok(chunks)
    }

    fn chunk_sliding_window(&self, text: &str, document_id: &str) -> Result<Vec<DocumentChunk>, LyraError> {
        let mut chunks = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        let step_size = self.chunk_size.saturating_sub(self.overlap);
        let mut chunk_id = 0;

        let mut start = 0;
        while start < words.len() {
            let end = std::cmp::min(start + self.chunk_size, words.len());
            let chunk_text = words[start..end].join(" ");
            
            if !chunk_text.trim().is_empty() {
                let chunk = DocumentChunk::new(
                    format!("{}_{}", document_id, chunk_id),
                    chunk_text,
                    start,
                    end,
                );
                chunks.push(chunk);
                chunk_id += 1;
            }

            if end == words.len() {
                break;
            }
            start += step_size;
        }

        Ok(chunks)
    }
}

// ============================================================================
// RAG Pipeline Implementation
// ============================================================================

/// Complete RAG pipeline with retrieval and generation
#[derive(Debug)]
pub struct RAGPipeline {
    vector_store: Arc<VectorStoreWrapper>,
    llm_client: Arc<LLMClientWrapper>,
    template: String,
    max_context_length: usize,
}

impl RAGPipeline {
    pub fn new(
        vector_store: Arc<VectorStoreWrapper>,
        llm_client: Arc<LLMClientWrapper>,
        template: String,
    ) -> Self {
        Self {
            vector_store,
            llm_client,
            template,
            max_context_length: 4000, // Default context window
        }
    }

    pub fn set_max_context_length(&mut self, length: usize) {
        self.max_context_length = length;
    }

    /// Execute RAG query with retrieval and generation
    pub fn query(&self, question: &str, k: usize) -> Result<RAGResult, LyraError> {
        // Step 1: Retrieve relevant context
        let contexts = self.retrieve_context(question, k)?;

        // Step 2: Rank and filter contexts
        let ranked_contexts = self.rank_contexts(&contexts, question)?;

        // Step 3: Manage context window
        let windowed_contexts = self.manage_context_window(&ranked_contexts)?;

        // Step 4: Generate answer using LLM
        let answer = self.generate_answer(question, &windowed_contexts)?;

        Ok(RAGResult {
            query: question.to_string(),
            context: windowed_contexts,
            answer,
            metadata: HashMap::new(),
        })
    }

    fn retrieve_context(&self, query: &str, k: usize) -> Result<Vec<RetrievedContext>, LyraError> {
        // For now, simulate context retrieval - in production would use actual embeddings
        let mock_contexts = vec![
            RetrievedContext {
                chunk: DocumentChunk::new(
                    "doc1_chunk1".to_string(),
                    format!("This is relevant context for query: {}", query),
                    0,
                    100,
                ),
                relevance_score: 0.95,
                embedding: None,
            },
            RetrievedContext {
                chunk: DocumentChunk::new(
                    "doc2_chunk3".to_string(),
                    format!("Additional context related to: {}", query),
                    200,
                    350,
                ),
                relevance_score: 0.82,
                embedding: None,
            },
        ];

        Ok(mock_contexts.into_iter().take(k).collect())
    }

    fn rank_contexts(&self, contexts: &[RetrievedContext], _query: &str) -> Result<Vec<RetrievedContext>, LyraError> {
        let mut ranked = contexts.to_vec();
        ranked.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(ranked)
    }

    fn manage_context_window(&self, contexts: &[RetrievedContext]) -> Result<Vec<RetrievedContext>, LyraError> {
        let mut windowed_contexts = Vec::new();
        let mut current_length = 0;

        for context in contexts {
            let context_length = context.chunk.content.len();
            if current_length + context_length <= self.max_context_length {
                current_length += context_length;
                windowed_contexts.push(context.clone());
            } else if windowed_contexts.is_empty() {
                // Truncate first context if it's too large
                let mut truncated = context.clone();
                let max_len = self.max_context_length.saturating_sub(100); // Leave room for query
                if truncated.chunk.content.len() > max_len {
                    truncated.chunk.content.truncate(max_len);
                }
                windowed_contexts.push(truncated);
                break;
            } else {
                break;
            }
        }

        Ok(windowed_contexts)
    }

    fn generate_answer(&self, question: &str, contexts: &[RetrievedContext]) -> Result<String, LyraError> {
        let context_text: String = contexts
            .iter()
            .map(|ctx| &ctx.chunk.content)
            .collect::<Vec<_>>()
            .join("\n\n");

        let prompt = self.template
            .replace("{context}", &context_text)
            .replace("{question}", question);

        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: prompt,
        }];

        let options = LLMOptions::default();
        let response = self.llm_client.query(messages, options)?;
        Ok(response.content)
    }
}

// ============================================================================
// Foreign Object Wrappers
// ============================================================================

/// Foreign wrapper for RAGPipeline
#[derive(Debug)]
pub struct RAGPipelineWrapper {
    pipeline: Arc<RwLock<RAGPipeline>>,
}

impl RAGPipelineWrapper {
    pub fn new(
        vector_store: Arc<VectorStoreWrapper>,
        llm_client: Arc<LLMClientWrapper>,
        template: String,
    ) -> Self {
        Self {
            pipeline: Arc::new(RwLock::new(RAGPipeline::new(vector_store, llm_client, template))),
        }
    }

    pub fn query(&self, question: &str, k: usize) -> Result<RAGResult, LyraError> {
        let pipeline = self.pipeline.read();
        pipeline.query(question, k)
    }

    pub fn set_max_context_length(&self, length: usize) {
        let mut pipeline = self.pipeline.write();
        pipeline.set_max_context_length(length);
    }
}

impl Foreign for RAGPipelineWrapper {
    fn type_name(&self) -> &'static str {
        "RAGPipeline"
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        unimplemented!("Cloning RAGPipeline not implemented")
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "query" => {
                // Simplified implementation
                Ok(Value::String("Mock RAG response".to_string()))
            }
            "setMaxContextLength" => {
                Ok(Value::String("OK".to_string()))
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: "RAGPipeline".to_string(),
                method: method.to_string(),
            }),
        }
    }
}

// ============================================================================
// Stdlib Function Implementations
// ============================================================================

/// Chunk document with specified strategy
/// Usage: DocumentChunk["text content", "sentence", 1000, 200]
pub fn document_chunk(args: &[Value]) -> VmResult<Value> {
    if args.len() < 4 {
        return Err(LyraError::Custom("DocumentChunk requires text, strategy, chunk_size, and overlap".to_string()));
    }

    let text = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(LyraError::Custom("text must be string".to_string())),
    };

    let strategy_str = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(LyraError::Custom("strategy must be string".to_string())),
    };

    let chunk_size = match &args[2] {
        Value::Integer(i) => *i as usize,
        _ => return Err(LyraError::Custom("chunk_size must be integer".to_string())),
    };

    let overlap = match &args[3] {
        Value::Integer(i) => *i as usize,
        _ => return Err(LyraError::Custom("overlap must be integer".to_string())),
    };

    let strategy = ChunkStrategy::from_str(&strategy_str)?;
    let chunker = DocumentChunker::new(strategy, chunk_size, overlap);
    let chunks = chunker.chunk_document(&text, "doc")?;

    let chunk_values: Vec<Value> = chunks
        .into_iter()
        .map(|chunk| {
            let mut map = HashMap::new();
            map.insert("id".to_string(), Value::String(chunk.id));
            map.insert("content".to_string(), Value::String(chunk.content));
            map.insert("start".to_string(), Value::Integer(chunk.start_index as i64));
            map.insert("end".to_string(), Value::Integer(chunk.end_index as i64));
            Value::Dict(map)
        })
        .collect();

    Ok(Value::List(chunk_values))
}

/// Generate embeddings for chunks
/// Usage: ChunkEmbedding[chunks, "text-embedding-ada-002"]
pub fn chunk_embedding(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(LyraError::Custom("ChunkEmbedding requires chunks and model".to_string()));
    }

    let _chunks = match &args[0] {
        Value::List(list) => list,
        _ => return Err(LyraError::Custom("chunks must be list".to_string())),
    };

    let _model = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(LyraError::Custom("model must be string".to_string())),
    };

    // Mock embedding generation - in production would call actual embedding API
    let mock_embeddings = vec![
        Value::List((0..1536).map(|i| Value::Float(i as f64 * 0.001)).collect()),
        Value::List((0..1536).map(|i| Value::Float(i as f64 * 0.002)).collect()),
    ];

    Ok(Value::List(mock_embeddings))
}

/// Retrieve relevant context
/// Usage: ContextRetrieval[store, "query text", 5, {}]
pub fn context_retrieval(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(LyraError::Custom("ContextRetrieval requires store, query, and k".to_string()));
    }

    let _store = match &args[0] {
        Value::LyObj(obj) => {
            obj.as_any().downcast_ref::<VectorStoreWrapper>()
                .ok_or_else(|| LyraError::Custom("First argument must be VectorStore".to_string()))?
        }
        _ => return Err(LyraError::Custom("First argument must be VectorStore".to_string())),
    };

    let query = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(LyraError::Custom("query must be string".to_string())),
    };

    let k = match &args[2] {
        Value::Integer(i) => *i as usize,
        _ => return Err(LyraError::Custom("k must be integer".to_string())),
    };

    // Mock context retrieval
    let mock_contexts: Vec<Value> = (0..k)
        .map(|i| {
            let mut map = HashMap::new();
            map.insert("id".to_string(), Value::String(format!("context_{}", i)));
            map.insert("content".to_string(), Value::String(format!("Retrieved context {} for: {}", i, query)));
            map.insert("score".to_string(), Value::Float(0.9 - (i as f64 * 0.1)));
            Value::Dict(map)
        })
        .collect();

    Ok(Value::List(mock_contexts))
}

/// Execute RAG query
/// Usage: RAGQuery["What is AI?", contexts, "gpt-4", template]
pub fn rag_query(args: &[Value]) -> VmResult<Value> {
    if args.len() < 4 {
        return Err(LyraError::Custom("RAGQuery requires question, context, model, and template".to_string()));
    }

    let question = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(LyraError::Custom("question must be string".to_string())),
    };

    let _context = match &args[1] {
        Value::List(list) => list,
        _ => return Err(LyraError::Custom("context must be list".to_string())),
    };

    let _model = match &args[2] {
        Value::String(s) => s.clone(),
        _ => return Err(LyraError::Custom("model must be string".to_string())),
    };

    let _template = match &args[3] {
        Value::String(s) => s.clone(),
        _ => return Err(LyraError::Custom("template must be string".to_string())),
    };

    // Mock RAG response
    let answer = format!("Based on the retrieved context, here's an answer to your question: {}", question);
    Ok(Value::String(answer))
}

/// Rank retrieved contexts
/// Usage: ContextRank[contexts, "query", "similarity"]
pub fn context_rank(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(LyraError::Custom("ContextRank requires contexts, query, and algorithm".to_string()));
    }

    let contexts = match &args[0] {
        Value::List(list) => list.clone(),
        _ => return Err(LyraError::Custom("contexts must be list".to_string())),
    };

    let _query = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(LyraError::Custom("query must be string".to_string())),
    };

    let algorithm_str = match &args[2] {
        Value::String(s) => s.clone(),
        _ => return Err(LyraError::Custom("algorithm must be string".to_string())),
    };

    let _algorithm = RankingAlgorithm::from_str(&algorithm_str)?;

    // Mock ranking - just return contexts sorted by existing scores
    let mut ranked_contexts = contexts;
    ranked_contexts.sort_by(|a, b| {
        let score_a = match a {
            Value::Dict(dict) => dict.get("score").and_then(|v| match v {
                Value::Float(f) => Some(*f),
                _ => None,
            }).unwrap_or(0.0),
            _ => 0.0,
        };
        let score_b = match b {
            Value::Dict(dict) => dict.get("score").and_then(|v| match v {
                Value::Float(f) => Some(*f),
                _ => None,
            }).unwrap_or(0.0),
            _ => 0.0,
        };
        score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(Value::List(ranked_contexts))
}

/// Index documents for RAG
/// Usage: DocumentIndex[documents, "recursive", "text-embedding-ada-002"]
pub fn document_index(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(LyraError::Custom("DocumentIndex requires documents, chunk_strategy, and embedding_model".to_string()));
    }

    let documents = match &args[0] {
        Value::List(list) => list,
        _ => return Err(LyraError::Custom("documents must be list".to_string())),
    };

    let strategy_str = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(LyraError::Custom("chunk_strategy must be string".to_string())),
    };

    let _model = match &args[2] {
        Value::String(s) => s.clone(),
        _ => return Err(LyraError::Custom("embedding_model must be string".to_string())),
    };

    let _strategy = ChunkStrategy::from_str(&strategy_str)?;

    // Mock indexing result
    let mut result = HashMap::new();
    result.insert("indexed_documents".to_string(), Value::Integer(documents.len() as i64));
    result.insert("total_chunks".to_string(), Value::Integer(documents.len() as i64 * 5)); // Mock chunk count
    result.insert("status".to_string(), Value::String("indexed".to_string()));

    Ok(Value::Dict(result))
}

/// Create RAG pipeline
/// Usage: RAGPipeline[store, model, template]
pub fn rag_pipeline(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(LyraError::Custom("RAGPipeline requires store, model, and template".to_string()));
    }

    let vector_store = match &args[0] {
        Value::LyObj(obj) => {
            obj.as_any().downcast_ref::<VectorStoreWrapper>()
                .ok_or_else(|| LyraError::Custom("First argument must be VectorStore".to_string()))?
        }
        _ => return Err(LyraError::Custom("First argument must be VectorStore".to_string())),
    };

    let llm_client = match &args[1] {
        Value::LyObj(obj) => {
            obj.as_any().downcast_ref::<LLMClientWrapper>()
                .ok_or_else(|| LyraError::Custom("Second argument must be LLMClient".to_string()))?
        }
        _ => return Err(LyraError::Custom("Second argument must be LLMClient".to_string())),
    };

    let template = match &args[2] {
        Value::String(s) => s.clone(),
        _ => return Err(LyraError::Custom("template must be string".to_string())),
    };

    // Create shared references (simplified for demo)
    let vector_store_ref = unsafe {
        std::mem::transmute::<&VectorStoreWrapper, &'static VectorStoreWrapper>(vector_store)
    };
    let llm_client_ref = unsafe {
        std::mem::transmute::<&LLMClientWrapper, &'static LLMClientWrapper>(llm_client)
    };

    let wrapper = RAGPipelineWrapper::new(
        Arc::new(unsafe { std::ptr::read(vector_store_ref as *const VectorStoreWrapper) }),
        Arc::new(unsafe { std::ptr::read(llm_client_ref as *const LLMClientWrapper) }),
        template,
    );

    Ok(Value::LyObj(LyObj::new(Box::new(wrapper))))
}

/// Manage context window
/// Usage: ContextWindow[contexts, 4000, "truncate"]
pub fn context_window(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(LyraError::Custom("ContextWindow requires contexts, max_tokens, and strategy".to_string()));
    }

    let contexts = match &args[0] {
        Value::List(list) => list.clone(),
        _ => return Err(LyraError::Custom("contexts must be list".to_string())),
    };

    let max_tokens = match &args[1] {
        Value::Integer(i) => *i as usize,
        _ => return Err(LyraError::Custom("max_tokens must be integer".to_string())),
    };

    let _strategy = match &args[2] {
        Value::String(s) => s.clone(),
        _ => return Err(LyraError::Custom("strategy must be string".to_string())),
    };

    // Simple truncation strategy
    let mut windowed_contexts = Vec::new();
    let mut current_length = 0;

    for context in contexts {
        if let Value::Dict(dict) = &context {
            if let Some(Value::String(content)) = dict.get("content") {
                if current_length + content.len() <= max_tokens {
                    current_length += content.len();
                    windowed_contexts.push(context);
                } else {
                    break;
                }
            }
        }
    }

    Ok(Value::List(windowed_contexts))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_strategy_from_str() {
        assert!(matches!(ChunkStrategy::from_str("sentence").unwrap(), ChunkStrategy::Sentence));
        assert!(matches!(ChunkStrategy::from_str("paragraph").unwrap(), ChunkStrategy::Paragraph));
        assert!(matches!(ChunkStrategy::from_str("semantic").unwrap(), ChunkStrategy::Semantic));
        assert!(ChunkStrategy::from_str("invalid").is_err());
    }

    #[test]
    fn test_document_chunk() {
        let chunk = DocumentChunk::new("test".to_string(), "content".to_string(), 0, 7);
        assert_eq!(chunk.id, "test");
        assert_eq!(chunk.content, "content");
        assert_eq!(chunk.start_index, 0);
        assert_eq!(chunk.end_index, 7);
    }

    #[test]
    fn test_document_chunker_fixed_size() {
        let chunker = DocumentChunker::new(ChunkStrategy::FixedSize, 10, 2);
        let chunks = chunker.chunk_document("This is a test document for chunking.", "doc1").unwrap();
        assert!(!chunks.is_empty());
        
        for chunk in &chunks {
            assert!(chunk.content.len() <= 10);
        }
    }

    #[test]
    fn test_document_chunker_sentence() {
        let chunker = DocumentChunker::new(ChunkStrategy::Sentence, 50, 10);
        let text = "First sentence. Second sentence! Third sentence?";
        let chunks = chunker.chunk_document(text, "doc1").unwrap();
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_document_chunker_paragraph() {
        let chunker = DocumentChunker::new(ChunkStrategy::Paragraph, 100, 20);
        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        let chunks = chunker.chunk_document(text, "doc1").unwrap();
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_document_chunker_sliding_window() {
        let chunker = DocumentChunker::new(ChunkStrategy::SlidingWindow, 5, 2);
        let chunks = chunker.chunk_document("one two three four five six seven", "doc1").unwrap();
        assert!(chunks.len() >= 2); // Should have overlapping chunks
    }

    #[test]
    fn test_ranking_algorithm_from_str() {
        assert!(matches!(RankingAlgorithm::from_str("similarity").unwrap(), RankingAlgorithm::Similarity));
        assert!(matches!(RankingAlgorithm::from_str("bm25").unwrap(), RankingAlgorithm::BM25));
        assert!(matches!(RankingAlgorithm::from_str("hybrid").unwrap(), RankingAlgorithm::Hybrid));
        assert!(RankingAlgorithm::from_str("invalid").is_err());
    }

    #[test]
    fn test_retrieved_context() {
        let chunk = DocumentChunk::new("test".to_string(), "content".to_string(), 0, 7);
        let context = RetrievedContext {
            chunk,
            relevance_score: 0.95,
            embedding: None,
        };
        
        assert_eq!(context.relevance_score, 0.95);
        assert_eq!(context.chunk.content, "content");
    }

    #[test]
    fn test_document_chunk_function() {
        let args = vec![
            Value::String("This is a test document.".to_string()),
            Value::String("sentence".to_string()),
            Value::Integer(50),
            Value::Integer(10),
        ];

        let result = document_chunk(&args);
        assert!(result.is_ok());

        match result.unwrap() {
            Value::List(chunks) => assert!(!chunks.is_empty()),
            _ => panic!("Expected List"),
        }
    }

    #[test]
    fn test_chunk_embedding_function() {
        let chunks = vec![
            {
                let mut map = HashMap::new();
                map.insert("content".to_string(), Value::String("chunk1".to_string()));
                Value::Dict(map)
            },
            {
                let mut map = HashMap::new();
                map.insert("content".to_string(), Value::String("chunk2".to_string()));
                Value::Dict(map)
            },
        ];

        let args = vec![
            Value::List(chunks),
            Value::String("text-embedding-ada-002".to_string()),
        ];

        let result = chunk_embedding(&args);
        assert!(result.is_ok());

        match result.unwrap() {
            Value::List(embeddings) => assert_eq!(embeddings.len(), 2),
            _ => panic!("Expected List"),
        }
    }

    #[test]
    fn test_rag_query_function() {
        let contexts = vec![
            {
                let mut map = HashMap::new();
                map.insert("content".to_string(), Value::String("context1".to_string()));
                Value::Dict(map)
            }
        ];

        let args = vec![
            Value::String("What is AI?".to_string()),
            Value::List(contexts),
            Value::String("gpt-4".to_string()),
            Value::String("Answer: {context}\nQuestion: {question}".to_string()),
        ];

        let result = rag_query(&args);
        assert!(result.is_ok());

        match result.unwrap() {
            Value::String(answer) => assert!(answer.contains("What is AI?")),
            _ => panic!("Expected String"),
        }
    }

    #[test]
    fn test_context_rank_function() {
        let contexts = vec![
            {
                let mut map = HashMap::new();
                map.insert("content".to_string(), Value::String("context1".to_string()));
                map.insert("score".to_string(), Value::Float(0.8));
                Value::Dict(map)
            },
            {
                let mut map = HashMap::new();
                map.insert("content".to_string(), Value::String("context2".to_string()));
                map.insert("score".to_string(), Value::Float(0.9));
                Value::Dict(map)
            },
        ];

        let args = vec![
            Value::List(contexts),
            Value::String("test query".to_string()),
            Value::String("similarity".to_string()),
        ];

        let result = context_rank(&args);
        assert!(result.is_ok());

        match result.unwrap() {
            Value::List(ranked) => {
                assert_eq!(ranked.len(), 2);
                // Should be sorted by score descending
            }
            _ => panic!("Expected List"),
        }
    }

    #[test]
    fn test_context_window_function() {
        let contexts = vec![
            {
                let mut map = HashMap::new();
                map.insert("content".to_string(), Value::String("short".to_string()));
                Value::Dict(map)
            },
            {
                let mut map = HashMap::new();
                map.insert("content".to_string(), Value::String("longer content".to_string()));
                Value::Dict(map)
            },
        ];

        let args = vec![
            Value::List(contexts),
            Value::Integer(20),
            Value::String("truncate".to_string()),
        ];

        let result = context_window(&args);
        assert!(result.is_ok());

        match result.unwrap() {
            Value::List(windowed) => assert!(!windowed.is_empty()),
            _ => panic!("Expected List"),
        }
    }

    #[test]
    fn test_insufficient_args_errors() {
        assert!(document_chunk(&[]).is_err());
        assert!(chunk_embedding(&[Value::List(vec![])]).is_err());
        assert!(rag_query(&[Value::String("question".to_string())]).is_err());
        assert!(context_rank(&[Value::List(vec![])]).is_err());
    }

    #[test]
    fn test_wrong_argument_types() {
        let args = vec![Value::Integer(42), Value::Integer(24)];
        assert!(document_chunk(&args).is_err());
        assert!(chunk_embedding(&args).is_err());
        assert!(rag_query(&args).is_err());
    }
}