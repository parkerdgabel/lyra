//! Phase 14A: LLM & Vector Database Integration
//!
//! This module provides comprehensive AI/ML integration for the Lyra symbolic computation engine,
//! focusing on Large Language Models (LLMs) and Vector Database operations for production use.
//!
//! ## Core Features
//! - LLM Client integration (OpenAI, Anthropic, Local models)
//! - Vector database operations with similarity search
//! - RAG (Retrieval-Augmented Generation) pipeline
//! - Model management and inference
//! - Embedding generation and clustering
//!
//! ## Architecture
//! All complex AI/ML types are implemented as Foreign objects to maintain VM simplicity
//! following the Foreign Object pattern. This ensures thread safety and clean separation
//! between VM core and AI/ML functionality.

pub mod llm_integration;
pub mod vector_store;
pub mod rag_pipeline;
pub mod model_management;
pub mod embeddings;
pub mod mlops;

// Re-export all public functions for stdlib registration
pub use llm_integration::*;
pub use vector_store::*; 
pub use rag_pipeline::*;
pub use model_management::*;
pub use embeddings::*;
pub use mlops::*;