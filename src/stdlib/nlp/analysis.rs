//! Text analysis algorithms including sentiment analysis, NER, POS tagging, and language modeling
//!
//! Implements advanced NLP analysis functionality using statistical models and rule-based approaches

use crate::vm::{Value, VmResult, VmError};
use crate::stdlib::common::assoc;
use std::collections::{HashMap, HashSet};
use std::fmt;
use regex::Regex;

// Removed legacy Foreign result wrappers; returning Associations instead

//

/// Named entity Foreign object
//

/// POS tag result Foreign object
//

/// Language model Foreign object
#[derive(Debug, Clone)]
pub struct LanguageModel {
    pub n: usize,  // n-gram size
    pub ngram_counts: HashMap<Vec<String>, usize>,
    pub vocabulary: HashSet<String>,
    pub total_ngrams: usize,
}

// No Foreign impl; internal helper only

impl fmt::Display for LanguageModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LanguageModel[n: {}, vocab_size: {}, ngrams: {}]", 
               self.n, self.vocabulary.len(), self.total_ngrams)
    }
}

impl LanguageModel {
    pub fn new(n: usize) -> Self {
        LanguageModel {
            n,
            ngram_counts: HashMap::new(),
            vocabulary: HashSet::new(),
            total_ngrams: 0,
        }
    }
    
    pub fn train(&mut self, documents: &[String]) -> Result<(), String> {
        for doc in documents {
            let tokens = simple_tokenize(doc);
            
            // Add tokens to vocabulary
            for token in &tokens {
                self.vocabulary.insert(token.clone());
            }
            
            // Generate n-grams and count them
            if tokens.len() >= self.n {
                for i in 0..=tokens.len() - self.n {
                    let ngram = tokens[i..i + self.n].to_vec();
                    *self.ngram_counts.entry(ngram).or_insert(0) += 1;
                    self.total_ngrams += 1;
                }
            }
        }
        
        Ok(())
    }
    
    pub fn probability(&self, ngram: &[String]) -> f64 {
        if ngram.len() != self.n {
            return 0.0;
        }
        
        let count = self.ngram_counts.get(ngram).unwrap_or(&0);
        if self.total_ngrams == 0 {
            0.0
        } else {
            *count as f64 / self.total_ngrams as f64
        }
    }
}

/// Text classification result Foreign object
//

/// Spell check result Foreign object
#[derive(Debug, Clone)]
pub struct SpellCheckResult {
    pub word: String,
    pub is_correct: bool,
    pub suggestions: Vec<String>,
    pub position: usize,
}

//

impl fmt::Display for SpellCheckResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SpellCheckResult[word: \"{}\", correct: {}, suggestions: {}]", 
               self.word, self.is_correct, self.suggestions.len())
    }
}

// Helper functions

fn simple_tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|s| s.to_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

fn normalize_word(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_alphabetic())
        .collect::<String>()
        .to_lowercase()
}

// Sentiment analysis using rule-based approach
fn rule_based_sentiment_score(text: &str) -> (f64, f64) {
    let positive_words: HashSet<&str> = ["good", "great", "excellent", "amazing", "wonderful", 
        "fantastic", "awesome", "brilliant", "outstanding", "superb", "love", "like", "enjoy", 
        "happy", "pleased", "satisfied", "perfect", "best", "incredible", "marvelous"].iter().cloned().collect();
    
    let negative_words: HashSet<&str> = ["bad", "terrible", "awful", "horrible", "disgusting", 
        "hate", "dislike", "worst", "disappointing", "poor", "sad", "angry", "frustrated", 
        "annoying", "pathetic", "useless", "broken", "fail", "failure", "wrong"].iter().cloned().collect();
    
    let intensifiers: HashMap<&str, f64> = [
        ("very", 1.5), ("extremely", 2.0), ("incredibly", 2.0), ("absolutely", 1.8),
        ("quite", 1.2), ("really", 1.3), ("so", 1.4), ("too", 1.3)
    ].iter().cloned().collect();
    
    let tokens = simple_tokenize(text);
    let mut positive_score = 0.0;
    let mut negative_score = 0.0;
    let mut current_intensifier = 1.0;
    
    for token in &tokens {
        let word = normalize_word(token);
        
        // Check for intensifiers
        if let Some(&intensity) = intensifiers.get(word.as_str()) {
            current_intensifier = intensity;
            continue;
        }
        
        // Check for sentiment words
        if positive_words.contains(word.as_str()) {
            positive_score += 1.0 * current_intensifier;
        } else if negative_words.contains(word.as_str()) {
            negative_score += 1.0 * current_intensifier;
        }
        
        // Reset intensifier
        current_intensifier = 1.0;
    }
    
    let polarity = if positive_score + negative_score > 0.0 {
        (positive_score - negative_score) / (positive_score + negative_score)
    } else {
        0.0
    };
    
    let confidence = (positive_score + negative_score) / (tokens.len() as f64).max(1.0);
    
    (polarity, confidence.min(1.0))
}

// Simple pattern-based named entity recognition
fn extract_named_entities(text: &str) -> Vec<(String, String, usize, usize, f64)> {
    let mut entities: Vec<(String, String, usize, usize, f64)> = Vec::new();
    
    // Person names (simple pattern: capitalized words)
    let person_regex = Regex::new(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b").unwrap();
    for mat in person_regex.find_iter(text) {
        entities.push((mat.as_str().to_string(), "PERSON".to_string(), mat.start(), mat.end(), 0.7));
    }
    
    // Organizations (simple pattern: words containing Corp, Inc, Ltd, etc.)
    let org_regex = Regex::new(r"\b[A-Z][a-zA-Z\s]*(Corp|Inc|Ltd|LLC|Company|Corporation)\b").unwrap();
    for mat in org_regex.find_iter(text) {
        entities.push((mat.as_str().to_string(), "ORGANIZATION".to_string(), mat.start(), mat.end(), 0.8));
    }
    
    // Locations (simple pattern: common location indicators)
    let location_keywords = ["Street", "Avenue", "Road", "Drive", "Lane", "City", "State", "Country"];
    for keyword in location_keywords {
        let pattern = format!(r"\b[A-Z][a-zA-Z\s]*{}\b", keyword);
        if let Ok(regex) = Regex::new(&pattern) {
            for mat in regex.find_iter(text) {
                entities.push((mat.as_str().to_string(), "LOCATION".to_string(), mat.start(), mat.end(), 0.6));
            }
        }
    }
    
    entities
}

// Simple POS tagging using pattern matching
fn simple_pos_tag(tokens: &[String]) -> Vec<(String, String, usize)> {
    let mut tags: Vec<(String, String, usize)> = Vec::new();
    
    let articles = ["the", "a", "an"];
    let prepositions = ["in", "on", "at", "by", "for", "with", "from", "to", "of"];
    let pronouns = ["i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"];
    let conjunctions = ["and", "or", "but", "so", "yet", "for", "nor"];
    
    for (i, token) in tokens.iter().enumerate() {
        let word = token.to_lowercase();
        let tag = if articles.contains(&word.as_str()) {
            "DET"
        } else if prepositions.contains(&word.as_str()) {
            "ADP"
        } else if pronouns.contains(&word.as_str()) {
            "PRON"
        } else if conjunctions.contains(&word.as_str()) {
            "CCONJ"
        } else if word.ends_with("ly") {
            "ADV"
        } else if word.ends_with("ed") || word.ends_with("ing") {
            "VERB"
        } else if word.chars().next().unwrap_or(' ').is_uppercase() {
            "NOUN"
        } else if word.ends_with("s") && word.len() > 2 {
            "NOUN"
        } else {
            "NOUN" // Default to noun
        };
        
        tags.push((token.clone(), tag.to_string(), i));
    }
    
    tags
}

// Simple language detection based on character patterns
fn detect_language(text: &str) -> String {
    let char_counts: HashMap<char, usize> = text.chars()
        .filter(|c| c.is_alphabetic())
        .fold(HashMap::new(), |mut acc, c| {
            *acc.entry(c.to_lowercase().next().unwrap_or(c)).or_insert(0) += 1;
            acc
        });
    
    let total_chars = char_counts.values().sum::<usize>() as f64;
    
    if total_chars == 0.0 {
        return "unknown".to_string();
    }
    
    // Simple heuristics based on character frequency
    let e_freq = *char_counts.get(&'e').unwrap_or(&0) as f64 / total_chars;
    let a_freq = *char_counts.get(&'a').unwrap_or(&0) as f64 / total_chars;
    let o_freq = *char_counts.get(&'o').unwrap_or(&0) as f64 / total_chars;
    let i_freq = *char_counts.get(&'i').unwrap_or(&0) as f64 / total_chars;
    
    // Check for Spanish characteristics
    if text.contains("ñ") || text.contains("¿") || text.contains("¡") {
        return "es".to_string();
    }
    
    // Check for French characteristics
    if text.contains("ç") || text.contains("è") || text.contains("é") || text.contains("à") {
        return "fr".to_string();
    }
    
    // Check for German characteristics
    if text.contains("ß") || text.contains("ä") || text.contains("ö") || text.contains("ü") {
        return "de".to_string();
    }
    
    // Check for Italian characteristics (high frequency of vowels)
    if a_freq > 0.11 && o_freq > 0.09 && i_freq > 0.10 {
        return "it".to_string();
    }
    
    // Default to English
    "en".to_string()
}

// Extractive text summarization
fn extractive_summarize(text: &str, num_sentences: usize) -> String {
    let sentences: Vec<&str> = text.split('.')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    
    if sentences.len() <= num_sentences {
        return text.to_string();
    }
    
    // Score sentences by word frequency
    let all_words: Vec<String> = sentences.iter()
        .flat_map(|sentence| simple_tokenize(sentence))
        .collect();
    
    let mut word_freq = HashMap::new();
    for word in &all_words {
        *word_freq.entry(word.clone()).or_insert(0) += 1;
    }
    
    // Score each sentence
    let mut sentence_scores: Vec<(usize, f64)> = sentences.iter()
        .enumerate()
        .map(|(i, sentence)| {
            let words = simple_tokenize(sentence);
            let score: f64 = words.iter()
                .map(|word| *word_freq.get(word).unwrap_or(&0) as f64)
                .sum::<f64>() / words.len().max(1) as f64;
            (i, score)
        })
        .collect();
    
    // Sort by score and take top sentences
    sentence_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut selected_indices: Vec<usize> = sentence_scores.iter()
        .take(num_sentences)
        .map(|(i, _)| *i)
        .collect();
    
    // Sort selected sentences by original order
    selected_indices.sort();
    
    let summary: Vec<String> = selected_indices.iter()
        .map(|&i| sentences[i].to_string())
        .collect();
    
    summary.join(". ") + "."
}

// Simple spell checker using edit distance
fn spell_check_word(word: &str, dictionary: &HashSet<String>) -> SpellCheckResult {
    let normalized = normalize_word(word);
    
    if dictionary.contains(&normalized) {
        return SpellCheckResult {
            word: word.to_string(),
            is_correct: true,
            suggestions: vec![],
            position: 0,
        };
    }
    
    // Find suggestions using edit distance
    let mut suggestions = Vec::new();
    for dict_word in dictionary {
        if edit_distance(&normalized, dict_word) <= 2 {
            suggestions.push(dict_word.clone());
        }
    }
    
    // Sort suggestions by edit distance
    suggestions.sort_by_key(|s| edit_distance(&normalized, s));
    suggestions.truncate(5); // Keep top 5 suggestions
    
    SpellCheckResult {
        word: word.to_string(),
        is_correct: false,
        suggestions,
        position: 0,
    }
}

fn edit_distance(s1: &str, s2: &str) -> usize {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();
    let m = s1_chars.len();
    let n = s2_chars.len();
    
    let mut dp = vec![vec![0; n + 1]; m + 1];
    
    // Initialize base cases
    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }
    
    // Fill the DP table
    for i in 1..=m {
        for j in 1..=n {
            if s1_chars[i-1] == s2_chars[j-1] {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = 1 + dp[i-1][j].min(dp[i][j-1]).min(dp[i-1][j-1]);
            }
        }
    }
    
    dp[m][n]
}

// Default dictionary for spell checking
fn get_basic_dictionary() -> HashSet<String> {
    let words = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"];
    
    words.iter().map(|s| s.to_string()).collect()
}

// Exported function implementations

/// Perform sentiment analysis on text
pub fn sentiment_analysis(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!("Wrong arity: expected {}, got {}", 1, args.len() )));
    }
    
    let text = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError { expected: "string".to_string(), actual: format!("{:? }", args[0]) 
        }),
    };
    
    let (polarity, confidence) = rule_based_sentiment_score(&text);
    
    let label = if polarity > 0.1 {
        "positive"
    } else if polarity < -0.1 {
        "negative"
    } else {
        "neutral"
    };
    
    let mut scores = HashMap::new();
    scores.insert("polarity".to_string(), Value::Real(polarity));
    scores.insert("confidence".to_string(), Value::Real(confidence));
    Ok(assoc(vec![
        ("label", Value::String(label.to_string())),
        ("polarity", Value::Real(polarity)),
        ("confidence", Value::Real(confidence)),
        ("scores", Value::Object(scores)),
    ]))
}

/// Extract named entities from text
pub fn named_entity_recognition(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!("Wrong arity: expected {}, got {}", 1, args.len() )));
    }
    
    let text = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError { expected: "string".to_string(), actual: format!("{:? }", args[0]) 
        }),
    };
    
    let entities = extract_named_entities(&text);
    let entity_values: Vec<Value> = entities.into_iter()
        .map(|(text, etype, start, end, conf)| assoc(vec![
            ("text", Value::String(text)),
            ("type", Value::String(etype)),
            ("start", Value::Integer(start as i64)),
            ("end", Value::Integer(end as i64)),
            ("confidence", Value::Real(conf)),
        ]))
        .collect();
    Ok(Value::List(entity_values))
}

/// Perform POS tagging on text
pub fn pos_tagging(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!("Wrong arity: expected {}, got {}", 1, args.len() )));
    }
    
    let text = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError { expected: "string".to_string(), actual: format!("{:? }", args[0]) 
        }),
    };
    
    let tokens = simple_tokenize(&text);
    let pos_tags = simple_pos_tag(&tokens);
    
    let tag_values: Vec<Value> = pos_tags.into_iter()
        .map(|(word, tag, pos)| assoc(vec![
            ("word", Value::String(word)),
            ("tag", Value::String(tag)),
            ("position", Value::Integer(pos as i64)),
        ]))
        .collect();
    Ok(Value::List(tag_values))
}

/// Detect the language of text
pub fn language_detection(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!("Wrong arity: expected {}, got {}", 1, args.len() )));
    }
    
    let text = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError { expected: "string".to_string(), actual: format!("{:? }", args[0]) 
        }),
    };
    
    let language = detect_language(&text);
    Ok(Value::String(language))
}

/// Classify text using a specified model
pub fn text_classification(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!("Wrong arity: expected {}, got {}", 2, args.len() )));
    }
    
    let text = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError { expected: "string".to_string(), actual: format!("{:? }", args[0]) 
        }),
    };
    
    let model_type = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError { expected: "string".to_string(), actual: format!("{:? }", args[1]) 
        }),
    };
    
    // For now, implement a simple sentiment-based classification
    let (polarity, confidence) = rule_based_sentiment_score(&text);
    
    let (predicted_class, class_confidence) = match model_type.as_str() {
        "movie_sentiment" => {
            if polarity > 0.1 {
                ("positive".to_string(), confidence)
            } else if polarity < -0.1 {
                ("negative".to_string(), confidence)
            } else {
                ("neutral".to_string(), confidence)
            }
        }
        _ => ("unknown".to_string(), 0.0),
    };
    
    let mut probs = HashMap::new();
    probs.insert("positive".to_string(), Value::Real(if polarity > 0.0 { polarity } else { 0.0 }));
    probs.insert("negative".to_string(), Value::Real(if polarity < 0.0 { -polarity } else { 0.0 }));
    probs.insert("neutral".to_string(), Value::Real(1.0 - polarity.abs()));
    Ok(assoc(vec![
        ("class", Value::String(predicted_class)),
        ("confidence", Value::Real(class_confidence)),
        ("probabilities", Value::Object(probs)),
        ("model", Value::String(model_type)),
    ]))
}

/// Extract keywords from text
pub fn keyword_extraction(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!("Wrong arity: expected {}, got {}", 1, args.len() )));
    }
    
    let text = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError { expected: "string".to_string(), actual: format!("{:? }", args[0]) 
        }),
    };
    
    let tokens = simple_tokenize(&text);
    
    // Simple keyword extraction: words longer than 4 characters, not common words
    let stop_words: HashSet<&str> = ["this", "that", "with", "from", "they", "have", "been", "will", "their", "said", "each", "which", "them", "than", "many", "some", "what", "would", "there", "more"].iter().cloned().collect();
    
    let mut word_freq = HashMap::new();
    for token in &tokens {
        if token.len() > 4 && !stop_words.contains(token.as_str()) {
            *word_freq.entry(token.clone()).or_insert(0) += 1;
        }
    }
    
    // Sort by frequency and take top keywords
    let mut keywords: Vec<(String, usize)> = word_freq.into_iter().collect();
    keywords.sort_by(|a, b| b.1.cmp(&a.1));
    keywords.truncate(10);
    
    let keyword_values: Vec<Value> = keywords.into_iter()
        .map(|(word, _freq)| Value::String(word))
        .collect();
    
    Ok(Value::List(keyword_values))
}

/// Create a language model from training data
pub fn language_model(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!("Wrong arity: expected {}, got {}", 2, args.len() )));
    }
    
    let documents = match &args[0] {
        Value::List(docs) => docs,
        _ => return Err(VmError::TypeError { expected: "list".to_string(), actual: format!("{:? }", args[0]) 
        }),
    };
    
    let n = match &args[1] {
        Value::Integer(n) => *n as usize,
        _ => return Err(VmError::TypeError { expected: "integer".to_string(), actual: format!("{:? }", args[1]) 
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
    let mut model = LanguageModel::new(n);
    model.train(&doc_strings)
        .map_err(|e| VmError::Runtime(e ))?;
    Ok(assoc(vec![
        ("n", Value::Integer(n as i64)),
        ("vocabSize", Value::Integer(model.vocabulary.len() as i64)),
        ("totalNGrams", Value::Integer(model.total_ngrams as i64)),
    ]))
}

/// Summarize text using extractive summarization
pub fn text_summarization(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!("Wrong arity: expected {}, got {}", 2, args.len() )));
    }
    
    let text = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError { expected: "string".to_string(), actual: format!("{:? }", args[0]) 
        }),
    };
    
    let num_sentences = match &args[1] {
        Value::Integer(n) => *n as usize,
        _ => return Err(VmError::TypeError { expected: "integer".to_string(), actual: format!("{:? }", args[1]) 
        }),
    };
    
    let summary = extractive_summarize(&text, num_sentences);
    Ok(Value::String(summary))
}

/// Check spelling of text
pub fn spell_check(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!("Wrong arity: expected {}, got {}", 1, args.len() )));
    }
    
    let text = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError { expected: "string".to_string(), actual: format!("{:? }", args[0]) 
        }),
    };
    
    let dictionary = get_basic_dictionary();
    let tokens = simple_tokenize(&text);
    
    let mut errors = Vec::new();
    for (i, token) in tokens.iter().enumerate() {
        let mut r = spell_check_word(token, &dictionary);
        r.position = i;
        if !r.is_correct {
            errors.push(assoc(vec![
                ("word", Value::String(r.word)),
                ("isCorrect", Value::Boolean(r.is_correct)),
                ("position", Value::Integer(r.position as i64)),
                ("suggestions", Value::List(r.suggestions.into_iter().map(Value::String).collect())),
            ]));
        }
    }
    Ok(Value::List(errors))
}

/// Correct spelling in text
pub fn spell_correct(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!("Wrong arity: expected {}, got {}", 1, args.len() )));
    }
    
    let text = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError { expected: "string".to_string(), actual: format!("{:? }", args[0]) 
        }),
    };
    
    let dictionary = get_basic_dictionary();
    let words: Vec<&str> = text.split_whitespace().collect();
    
    let mut corrected_words = Vec::new();
    for word in words {
        let check_result = spell_check_word(word, &dictionary);
        if check_result.is_correct || check_result.suggestions.is_empty() {
            corrected_words.push(word.to_string());
        } else {
            corrected_words.push(check_result.suggestions[0].clone());
        }
    }
    
    Ok(Value::String(corrected_words.join(" ")))
}
