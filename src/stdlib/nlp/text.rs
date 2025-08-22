//! Text processing algorithms and Foreign object implementations
//! 
//! Implements core text processing functionality including tokenization, stemming,
//! n-gram generation, TF-IDF vectorization, and text similarity metrics.

use crate::vm::{Value, VmResult, VmError};
use crate::foreign::{LyObj, Foreign, ForeignError};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::any::Any;
use regex::Regex;

/// Text document Foreign object for holding processed text data
#[derive(Debug, Clone)]
pub struct TextDocument {
    pub original_text: String,
    pub tokens: Vec<String>,
    pub normalized_text: String,
    pub language: Option<String>,
}

impl Foreign for TextDocument {
    fn type_name(&self) -> &'static str {
        "TextDocument"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "tokens" => Ok(Value::List(self.tokens.iter().map(|t| Value::String(t.clone())).collect())),
            "originalText" => Ok(Value::String(self.original_text.clone())),
            "normalizedText" => Ok(Value::String(self.normalized_text.clone())),
            "language" => Ok(Value::String(self.language.clone().unwrap_or_else(|| "unknown".to_string()))),
            _ => Err(ForeignError::UnknownMethod { 
                type_name: self.type_name().to_string(), 
                method: method.to_string() 
            })
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl fmt::Display for TextDocument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TextDocument[tokens: {}, language: {:?}]", 
               self.tokens.len(), self.language)
    }
}

/// Tokenized text Foreign object
#[derive(Debug, Clone)]
pub struct TokenizedText {
    pub tokens: Vec<String>,
    pub token_positions: Vec<(usize, usize)>, // Start and end positions in original text
    pub tokenization_method: String,
}

impl Foreign for TokenizedText {
    fn type_name(&self) -> &'static str {
        "TokenizedText"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "tokens" => Ok(Value::List(self.tokens.iter().map(|t| Value::String(t.clone())).collect())),
            "method" => Ok(Value::String(self.tokenization_method.clone())),
            "length" => Ok(Value::Integer(self.tokens.len() as i64)),
            _ => Err(ForeignError::UnknownMethod { 
                type_name: self.type_name().to_string(), 
                method: method.to_string() 
            })
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl fmt::Display for TokenizedText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TokenizedText[tokens: {}, method: {}]", 
               self.tokens.len(), self.tokenization_method)
    }
}

/// TF-IDF model Foreign object
#[derive(Debug, Clone)]
pub struct TFIDFModel {
    pub vocabulary: HashMap<String, usize>,
    pub idf_scores: HashMap<String, f64>,
    pub document_count: usize,
    pub feature_vectors: Vec<HashMap<String, f64>>,
}

impl Foreign for TFIDFModel {
    fn type_name(&self) -> &'static str {
        "TFIDFModel"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "vocabularySize" => Ok(Value::Integer(self.vocabulary.len() as i64)),
            "documentCount" => Ok(Value::Integer(self.document_count as i64)),
            "transform" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 1, 
                        actual: args.len() 
                    });
                }
                match &args[0] {
                    Value::String(text) => {
                        let vector = self.transform(text);
                        let vector_pairs: Vec<Value> = vector.into_iter()
                            .map(|(term, score)| Value::List(vec![
                                Value::String(term), 
                                Value::Real(score)
                            ]))
                            .collect();
                        Ok(Value::List(vector_pairs))
                    }
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "string".to_string(),
                        actual: format!("{:?}", args[0])
                    })
                }
            }
            _ => Err(ForeignError::UnknownMethod { 
                type_name: self.type_name().to_string(), 
                method: method.to_string() 
            })
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl fmt::Display for TFIDFModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TFIDFModel[vocabulary: {}, documents: {}]", 
               self.vocabulary.len(), self.document_count)
    }
}

impl TFIDFModel {
    pub fn new() -> Self {
        TFIDFModel {
            vocabulary: HashMap::new(),
            idf_scores: HashMap::new(),
            document_count: 0,
            feature_vectors: Vec::new(),
        }
    }
    
    pub fn fit(&mut self, documents: &[String]) -> Result<(), String> {
        self.document_count = documents.len();
        
        // Build vocabulary
        for doc in documents {
            let tokens = simple_tokenize(doc);
            let unique_tokens: HashSet<String> = tokens.into_iter().collect();
            
            for token in unique_tokens {
                *self.vocabulary.entry(token).or_insert(0) += 1;
            }
        }
        
        // Calculate IDF scores
        for (term, doc_frequency) in &self.vocabulary {
            let idf = (self.document_count as f64 / *doc_frequency as f64).ln();
            self.idf_scores.insert(term.clone(), idf);
        }
        
        // Calculate TF-IDF vectors for each document
        for doc in documents {
            let tokens = simple_tokenize(doc);
            let mut term_frequencies = HashMap::new();
            
            // Calculate term frequencies
            for token in &tokens {
                *term_frequencies.entry(token.clone()).or_insert(0) += 1;
            }
            
            // Calculate TF-IDF scores
            let mut tfidf_vector = HashMap::new();
            for (term, tf) in term_frequencies {
                if let Some(idf) = self.idf_scores.get(&term) {
                    let tfidf_score = (tf as f64) * idf;
                    tfidf_vector.insert(term, tfidf_score);
                }
            }
            
            self.feature_vectors.push(tfidf_vector);
        }
        
        Ok(())
    }
    
    pub fn transform(&self, text: &str) -> HashMap<String, f64> {
        let tokens = simple_tokenize(text);
        let mut term_frequencies = HashMap::new();
        
        // Calculate term frequencies
        for token in &tokens {
            *term_frequencies.entry(token.clone()).or_insert(0) += 1;
        }
        
        // Calculate TF-IDF scores
        let mut tfidf_vector = HashMap::new();
        for (term, tf) in term_frequencies {
            if let Some(idf) = self.idf_scores.get(&term) {
                let tfidf_score = (tf as f64) * idf;
                tfidf_vector.insert(term, tfidf_score);
            }
        }
        
        tfidf_vector
    }
}

/// Word frequency map Foreign object
#[derive(Debug, Clone)]
pub struct WordFrequencyMap {
    pub frequencies: HashMap<String, usize>,
    pub total_words: usize,
}

impl Foreign for WordFrequencyMap {
    fn type_name(&self) -> &'static str {
        "WordFrequencyMap"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "size" => Ok(Value::Integer(self.frequencies.len() as i64)),
            "totalWords" => Ok(Value::Integer(self.total_words as i64)),
            "get" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity { 
                        method: method.to_string(), 
                        expected: 1, 
                        actual: args.len() 
                    });
                }
                match &args[0] {
                    Value::String(word) => {
                        let frequency = self.frequencies.get(word).unwrap_or(&0);
                        Ok(Value::Integer(*frequency as i64))
                    }
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "string".to_string(),
                        actual: format!("{:?}", args[0])
                    })
                }
            }
            "mostCommon" => {
                let mut word_freq: Vec<(&String, &usize)> = self.frequencies.iter().collect();
                word_freq.sort_by(|a, b| b.1.cmp(a.1));
                let top_words: Vec<Value> = word_freq.into_iter()
                    .take(10) // Top 10 most common words
                    .map(|(word, freq)| Value::List(vec![
                        Value::String(word.clone()), 
                        Value::Integer(*freq as i64)
                    ]))
                    .collect();
                Ok(Value::List(top_words))
            }
            _ => Err(ForeignError::UnknownMethod { 
                type_name: self.type_name().to_string(), 
                method: method.to_string() 
            })
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl fmt::Display for WordFrequencyMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WordFrequencyMap[unique_words: {}, total: {}]", 
               self.frequencies.len(), self.total_words)
    }
}

// Helper functions for text processing

fn simple_tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|s| s.to_lowercase())
        .collect()
}

fn regex_tokenize(text: &str, pattern: &str) -> Result<Vec<String>, String> {
    let regex = Regex::new(pattern)
        .map_err(|e| format!("Invalid regex pattern: {}", e))?;
    
    Ok(regex.split(text)
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().to_lowercase())
        .collect())
}

fn word_boundary_tokenize(text: &str) -> Vec<String> {
    let regex = Regex::new(r"\b\w+\b").unwrap();
    regex.find_iter(text)
        .map(|mat| mat.as_str().to_lowercase())
        .collect()
}

/// Porter stemming algorithm implementation (simplified)
fn porter_stem(word: &str) -> String {
    let word = word.to_lowercase();
    let mut stem = word.clone();
    
    // Step 1a: plurals and -ed, -ing
    if stem.ends_with("sses") {
        stem = stem[..stem.len()-2].to_string();
    } else if stem.ends_with("ies") {
        stem = stem[..stem.len()-2].to_string();
    } else if stem.ends_with("ss") {
        // Keep as is
    } else if stem.ends_with("s") && stem.len() > 1 {
        stem = stem[..stem.len()-1].to_string();
    }
    
    // Step 1b: -ed, -ing
    if stem.ends_with("eed") && stem.len() > 3 {
        stem = stem[..stem.len()-1].to_string();
    } else if stem.ends_with("ed") && stem.len() > 2 {
        let temp = stem[..stem.len()-2].to_string();
        if temp.chars().any(|c| "aeiou".contains(c)) {
            stem = temp;
        }
    } else if stem.ends_with("ing") && stem.len() > 3 {
        let temp = stem[..stem.len()-3].to_string();
        if temp.chars().any(|c| "aeiou".contains(c)) {
            stem = temp;
        }
    }
    
    // Step 2: -ational, -tional, etc.
    if stem.ends_with("ational") {
        stem = stem[..stem.len()-7].to_string() + "ate";
    } else if stem.ends_with("tional") {
        stem = stem[..stem.len()-6].to_string() + "tion";
    } else if stem.ends_with("ator") {
        stem = stem[..stem.len()-4].to_string() + "ate";
    } else if stem.ends_with("alism") {
        stem = stem[..stem.len()-5].to_string() + "al";
    }
    
    stem
}

/// Calculate cosine similarity between two TF-IDF vectors
fn cosine_similarity(vec1: &HashMap<String, f64>, vec2: &HashMap<String, f64>) -> f64 {
    let mut dot_product = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;
    
    // Get all unique terms
    let all_terms: HashSet<&String> = vec1.keys().chain(vec2.keys()).collect();
    
    for term in all_terms {
        let val1 = vec1.get(term).unwrap_or(&0.0);
        let val2 = vec2.get(term).unwrap_or(&0.0);
        
        dot_product += val1 * val2;
        norm1 += val1 * val1;
        norm2 += val2 * val2;
    }
    
    if norm1 == 0.0 || norm2 == 0.0 {
        0.0
    } else {
        dot_product / (norm1.sqrt() * norm2.sqrt())
    }
}

/// Calculate Jaccard similarity between two token sets
fn jaccard_similarity(tokens1: &[String], tokens2: &[String]) -> f64 {
    let set1: HashSet<&String> = tokens1.iter().collect();
    let set2: HashSet<&String> = tokens2.iter().collect();
    
    let intersection: HashSet<&String> = set1.intersection(&set2).cloned().collect();
    let union: HashSet<&String> = set1.union(&set2).cloned().collect();
    
    if union.is_empty() {
        0.0
    } else {
        intersection.len() as f64 / union.len() as f64
    }
}

// Exported function implementations

/// Tokenize text using different methods
pub fn tokenize_text(args: &[Value]) -> VmResult<Value> {
    if args.is_empty() {
        return Err(VmError::Runtime(format!("Wrong arity: expected {}, got {}", 1, 0 )));
    }
    
    let text = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError { expected: "string".to_string(), actual: format!("{:? }", args[0]) 
        }),
    };
    
    let method = if args.len() > 1 {
        match &args[1] {
            Value::String(s) => s.as_str(),
            _ => "default",
        }
    } else {
        "default"
    };
    
    let tokens = match method {
        "whitespace" => text.split_whitespace().map(|s| s.to_string()).collect(),
        "word_boundary" => word_boundary_tokenize(&text),
        "regex" => {
            if args.len() < 3 {
                return Err(VmError::Runtime(format!("Wrong arity: expected {}, got {}", 3, args.len() )));
            }
            let pattern = match &args[2] {
                Value::String(p) => p,
                _ => return Err(VmError::TypeError { expected: "string".to_string(), actual: format!("{:? }", args[2]) 
                }),
            };
            regex_tokenize(&text, pattern)
                .map_err(|e| VmError::Runtime(e ))?
        },
        _ => word_boundary_tokenize(&text), // default method
    };
    
    let token_values: Vec<Value> = tokens.iter()
        .map(|token| Value::String(token.clone()))
        .collect();
    
    Ok(Value::List(token_values))
}

/// Apply Porter stemming to a list of tokens
pub fn stem_text(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!("Wrong arity: expected {}, got {}", 1, args.len() )));
    }
    
    let tokens = match &args[0] {
        Value::List(tokens) => tokens,
        _ => return Err(VmError::TypeError { expected: "list".to_string(), actual: format!("{:? }", args[0]) 
        }),
    };
    
    let stemmed_tokens: Result<Vec<Value>, VmError> = tokens.iter()
        .map(|token| match token {
            Value::String(word) => Ok(Value::String(porter_stem(word))),
            _ => Err(VmError::TypeError { expected: "string".to_string(), actual: format!("{:? }", token) 
            }),
        })
        .collect();
    
    Ok(Value::List(stemmed_tokens?))
}

/// Generate n-grams from a list of tokens
pub fn generate_ngrams(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!("Wrong arity: expected {}, got {}", 2, args.len() )));
    }
    
    let tokens = match &args[0] {
        Value::List(tokens) => tokens,
        _ => return Err(VmError::TypeError { expected: "list".to_string(), actual: format!("{:? }", args[0]) 
        }),
    };
    
    let n = match &args[1] {
        Value::Integer(n) => *n as usize,
        _ => return Err(VmError::TypeError { expected: "integer".to_string(), actual: format!("{:? }", args[1]) 
        }),
    };
    
    if n == 0 || tokens.len() < n {
        return Ok(Value::List(vec![]));
    }
    
    let mut ngrams = Vec::new();
    
    for i in 0..=tokens.len() - n {
        let ngram: Vec<Value> = tokens[i..i+n].to_vec();
        ngrams.push(Value::List(ngram));
    }
    
    Ok(Value::List(ngrams))
}

/// Create TF-IDF model from a collection of documents
pub fn tfidf_vectorize(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!("Wrong arity: expected {}, got {}", 1, args.len() )));
    }
    
    let documents = match &args[0] {
        Value::List(docs) => docs,
        _ => return Err(VmError::TypeError { expected: "list".to_string(), actual: format!("{:? }", args[0]) 
        }),
    };
    
    let doc_strings: Result<Vec<String>, VmError> = documents.iter()
        .map(|doc| match doc {
            Value::String(text) => Ok(text.clone()),
            _ => Err(VmError::TypeError { expected: "string".to_string(), actual: format!("{:? }", doc) 
            }),
        })
        .collect();
    
    let doc_strings = doc_strings?;
    let mut tfidf_model = TFIDFModel::new();
    
    tfidf_model.fit(&doc_strings)
        .map_err(|e| VmError::Runtime(e ))?;
    
    Ok(Value::LyObj(LyObj::new(Box::new(tfidf_model))))
}

/// Calculate word frequencies from a list of tokens
pub fn word_frequency(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!("Wrong arity: expected {}, got {}", 1, args.len() )));
    }
    
    let tokens = match &args[0] {
        Value::List(tokens) => tokens,
        _ => return Err(VmError::TypeError { expected: "list".to_string(), actual: format!("{:? }", args[0]) 
        }),
    };
    
    let mut frequencies = HashMap::new();
    let mut total_words = 0;
    
    for token in tokens {
        match token {
            Value::String(word) => {
                *frequencies.entry(word.clone()).or_insert(0) += 1;
                total_words += 1;
            }
            _ => return Err(VmError::TypeError { expected: "string".to_string(), actual: format!("{:? }", token) 
            }),
        }
    }
    
    let freq_map = WordFrequencyMap {
        frequencies,
        total_words,
    };
    
    Ok(Value::LyObj(LyObj::new(Box::new(freq_map))))
}

/// Calculate text similarity using various metrics
pub fn text_similarity(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::Runtime(format!("Wrong arity: expected {}, got {}", 2, args.len() )));
    }
    
    let text1 = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError { expected: "string".to_string(), actual: format!("{:? }", args[0]) 
        }),
    };
    
    let text2 = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError { expected: "string".to_string(), actual: format!("{:? }", args[1]) 
        }),
    };
    
    let metric = if args.len() > 2 {
        match &args[2] {
            Value::String(s) => s.as_str(),
            _ => "cosine",
        }
    } else {
        "cosine"
    };
    
    let similarity = match metric {
        "jaccard" => {
            let tokens1 = simple_tokenize(&text1);
            let tokens2 = simple_tokenize(&text2);
            jaccard_similarity(&tokens1, &tokens2)
        }
        "cosine" => {
            // Create simple TF-IDF vectors for the two texts
            let mut tfidf_model = TFIDFModel::new();
            let docs = vec![text1.clone(), text2.clone()];
            tfidf_model.fit(&docs)
                .map_err(|e| VmError::Runtime(e ))?;
            
            let vec1 = tfidf_model.transform(&text1);
            let vec2 = tfidf_model.transform(&text2);
            cosine_similarity(&vec1, &vec2)
        }
        _ => return Err(VmError::Runtime(format!("Unknown similarity metric: {}", metric))),
    };
    
    Ok(Value::Real(similarity))
}

/// Normalize text by lowercasing and removing punctuation
pub fn normalize_text(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!("Wrong arity: expected {}, got {}", 1, args.len() )));
    }
    
    let text = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError { expected: "string".to_string(), actual: format!("{:? }", args[0]) 
        }),
    };
    
    // Convert to lowercase and remove punctuation except spaces
    let normalized: String = text.chars()
        .map(|c| if c.is_alphabetic() || c.is_whitespace() { c.to_lowercase().next().unwrap_or(c) } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ");
    
    Ok(Value::String(normalized))
}

/// Remove common stop words from text
pub fn remove_stop_words(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!("Wrong arity: expected {}, got {}", 1, args.len() )));
    }
    
    let tokens = match &args[0] {
        Value::List(tokens) => tokens,
        _ => return Err(VmError::TypeError { expected: "list".to_string(), actual: format!("{:? }", args[0]) 
        }),
    };
    
    // Common English stop words
    let stop_words: HashSet<&str> = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "her", "its", "our", "their", "this", "that", "these", "those"].iter().collect();
    
    let filtered_tokens: Vec<Value> = tokens.iter()
        .filter_map(|token| match token {
            Value::String(word) => {
                let lower_word = word.to_lowercase();
                if !stop_words.contains(lower_word.as_str()) {
                    Some(token.clone())
                } else {
                    None
                }
            }
            _ => Some(token.clone()), // Keep non-string tokens as-is
        })
        .collect();
    
    Ok(Value::List(filtered_tokens))
}