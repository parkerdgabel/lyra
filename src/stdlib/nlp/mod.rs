//! Natural Language Processing Module for Lyra
//!
//! Comprehensive NLP toolkit implementing text processing, analysis, and modeling algorithms.
//! All complex text types are implemented as Foreign objects following the LyObj pattern.

use crate::vm::{Value, VmResult};
use std::collections::HashMap;

pub mod text;
pub mod analysis;
pub mod sentiment;

// Export main NLP functions
pub use text::*;
pub use analysis::*;
pub use sentiment::*;

/// Central registry for all NLP functions
pub fn register_nlp_functions() -> HashMap<String, fn(&[Value]) -> VmResult<Value>> {
    let mut functions = HashMap::new();
    
    // Text Processing Functions
    functions.insert("TokenizeText".to_string(), text::tokenize_text as fn(&[Value]) -> VmResult<Value>);
    functions.insert("StemText".to_string(), text::stem_text as fn(&[Value]) -> VmResult<Value>);
    functions.insert("GenerateNGrams".to_string(), text::generate_ngrams as fn(&[Value]) -> VmResult<Value>);
    functions.insert("TFIDFVectorize".to_string(), text::tfidf_vectorize as fn(&[Value]) -> VmResult<Value>);
    functions.insert("WordFrequency".to_string(), text::word_frequency as fn(&[Value]) -> VmResult<Value>);
    functions.insert("TextSimilarity".to_string(), text::text_similarity as fn(&[Value]) -> VmResult<Value>);
    functions.insert("NormalizeText".to_string(), text::normalize_text as fn(&[Value]) -> VmResult<Value>);
    functions.insert("RemoveStopWords".to_string(), text::remove_stop_words as fn(&[Value]) -> VmResult<Value>);
    
    // Text Analysis Functions  
    functions.insert("SentimentAnalysis".to_string(), analysis::sentiment_analysis as fn(&[Value]) -> VmResult<Value>);
    functions.insert("NamedEntityRecognition".to_string(), analysis::named_entity_recognition as fn(&[Value]) -> VmResult<Value>);
    functions.insert("POSTagging".to_string(), analysis::pos_tagging as fn(&[Value]) -> VmResult<Value>);
    functions.insert("LanguageDetection".to_string(), analysis::language_detection as fn(&[Value]) -> VmResult<Value>);
    functions.insert("TextClassification".to_string(), analysis::text_classification as fn(&[Value]) -> VmResult<Value>);
    functions.insert("KeywordExtraction".to_string(), analysis::keyword_extraction as fn(&[Value]) -> VmResult<Value>);
    
    // Language Modeling Functions
    functions.insert("LanguageModel".to_string(), analysis::language_model as fn(&[Value]) -> VmResult<Value>);
    functions.insert("TextSummarization".to_string(), analysis::text_summarization as fn(&[Value]) -> VmResult<Value>);
    functions.insert("SpellCheck".to_string(), analysis::spell_check as fn(&[Value]) -> VmResult<Value>);
    functions.insert("SpellCorrect".to_string(), analysis::spell_correct as fn(&[Value]) -> VmResult<Value>);
    
    // Sentiment Analysis Functions
    functions.insert("RuleBased SentimentAnalysis".to_string(), sentiment::rule_based_sentiment as fn(&[Value]) -> VmResult<Value>);
    functions.insert("StatisticalSentiment".to_string(), sentiment::statistical_sentiment as fn(&[Value]) -> VmResult<Value>);
    functions.insert("EmotionDetection".to_string(), sentiment::emotion_detection as fn(&[Value]) -> VmResult<Value>);
    
    functions
}