//! Specialized sentiment analysis algorithms
//! 
//! Implements rule-based and statistical sentiment analysis approaches,
//! including emotion detection and advanced sentiment modeling.

use crate::vm::{Value, VmResult, VmError};
use crate::foreign::{LyObj, Foreign, ForeignError};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::any::Any;

/// Rule-based sentiment result with detailed rule explanations
#[derive(Debug, Clone)]
pub struct RuleBasedSentimentResult {
    pub polarity: f64,
    pub confidence: f64,
    pub label: String,
    pub triggered_rules: Vec<String>,
    pub word_scores: HashMap<String, f64>,
    pub sentence_scores: Vec<f64>,
}

impl Foreign for RuleBasedSentimentResult {
    fn type_name(&self) -> &'static str {
        "RuleBasedSentimentResult"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        Err(ForeignError::UnknownMethod { 
            type_name: self.type_name().to_string(), 
            method: method.to_string() 
        })
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl fmt::Display for RuleBasedSentimentResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RuleBasedSentimentResult[label: {}, polarity: {:.3}, rules: {}]", 
               self.label, self.polarity, self.triggered_rules.len())
    }
}

/// Statistical sentiment model Foreign object
#[derive(Debug, Clone)]
pub struct StatisticalSentimentModel {
    pub word_probabilities: HashMap<String, HashMap<String, f64>>, // word -> {class -> probability}
    pub class_priors: HashMap<String, f64>,
    pub vocabulary: HashSet<String>,
    pub classes: Vec<String>,
}

impl Foreign for StatisticalSentimentModel {
    fn type_name(&self) -> &'static str {
        "StatisticalSentimentModel"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        Err(ForeignError::UnknownMethod { 
            type_name: self.type_name().to_string(), 
            method: method.to_string() 
        })
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl fmt::Display for StatisticalSentimentModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StatisticalSentimentModel[vocab: {}, classes: {}]", 
               self.vocabulary.len(), self.classes.len())
    }
}

impl StatisticalSentimentModel {
    pub fn new() -> Self {
        StatisticalSentimentModel {
            word_probabilities: HashMap::new(),
            class_priors: HashMap::new(),
            vocabulary: HashSet::new(),
            classes: Vec::new(),
        }
    }
    
    pub fn train(&mut self, training_data: &[(String, String)]) -> Result<(), String> {
        if training_data.is_empty() {
            return Err("Training data cannot be empty".to_string());
        }
        
        // Collect all classes and build vocabulary
        let mut class_counts = HashMap::new();
        let mut word_class_counts: HashMap<String, HashMap<String, usize>> = HashMap::new();
        let mut class_word_totals: HashMap<String, usize> = HashMap::new();
        
        for (text, class) in training_data {
            *class_counts.entry(class.clone()).or_insert(0) += 1;
            if !self.classes.contains(class) {
                self.classes.push(class.clone());
            }
            
            let tokens = simple_tokenize(text);
            for token in tokens {
                self.vocabulary.insert(token.clone());
                *word_class_counts
                    .entry(token)
                    .or_insert_with(HashMap::new)
                    .entry(class.clone())
                    .or_insert(0) += 1;
                *class_word_totals.entry(class.clone()).or_insert(0) += 1;
            }
        }
        
        // Calculate class priors
        let total_docs = training_data.len() as f64;
        for (class, count) in class_counts {
            self.class_priors.insert(class, count as f64 / total_docs);
        }
        
        // Calculate word probabilities using Laplace smoothing
        let vocab_size = self.vocabulary.len();
        for word in &self.vocabulary {
            let mut word_probs = HashMap::new();
            
            for class in &self.classes {
                let word_count = word_class_counts
                    .get(word)
                    .and_then(|wc| wc.get(class))
                    .unwrap_or(&0);
                let total_words_in_class = class_word_totals.get(class).unwrap_or(&0);
                
                // Laplace smoothing
                let probability = (*word_count as f64 + 1.0) / 
                                 (*total_words_in_class as f64 + vocab_size as f64);
                word_probs.insert(class.clone(), probability);
            }
            
            self.word_probabilities.insert(word.clone(), word_probs);
        }
        
        Ok(())
    }
    
    pub fn predict(&self, text: &str) -> HashMap<String, f64> {
        let tokens = simple_tokenize(text);
        let mut class_scores = HashMap::new();
        
        for class in &self.classes {
            let prior = self.class_priors.get(class).unwrap_or(&0.0);
            let mut log_likelihood = prior.ln();
            
            for token in &tokens {
                if let Some(word_probs) = self.word_probabilities.get(token) {
                    if let Some(prob) = word_probs.get(class) {
                        log_likelihood += prob.ln();
                    }
                }
            }
            
            class_scores.insert(class.clone(), log_likelihood.exp());
        }
        
        // Normalize probabilities
        let total_score: f64 = class_scores.values().sum();
        if total_score > 0.0 {
            for score in class_scores.values_mut() {
                *score /= total_score;
            }
        }
        
        class_scores
    }
}

/// Emotion detection result Foreign object
#[derive(Debug, Clone)]
pub struct EmotionResult {
    pub primary_emotion: String,
    pub confidence: f64,
    pub emotion_scores: HashMap<String, f64>,
}

impl Foreign for EmotionResult {
    fn type_name(&self) -> &'static str {
        "EmotionResult"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        Err(ForeignError::UnknownMethod { 
            type_name: self.type_name().to_string(), 
            method: method.to_string() 
        })
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl fmt::Display for EmotionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EmotionResult[primary: {}, confidence: {:.3}]", 
               self.primary_emotion, self.confidence)
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

// Advanced rule-based sentiment analysis
fn advanced_rule_based_sentiment(text: &str) -> RuleBasedSentimentResult {
    let mut triggered_rules = Vec::new();
    let mut word_scores = HashMap::new();
    let mut sentence_scores = Vec::new();
    
    // Extended sentiment lexicons
    let positive_words: HashMap<&str, f64> = [
        ("excellent", 2.0), ("amazing", 2.0), ("wonderful", 2.0), ("fantastic", 2.0),
        ("outstanding", 1.8), ("superb", 1.8), ("brilliant", 1.8), ("marvelous", 1.8),
        ("good", 1.5), ("great", 1.5), ("nice", 1.2), ("pleasant", 1.2),
        ("love", 2.2), ("adore", 2.0), ("like", 1.0), ("enjoy", 1.3),
        ("happy", 1.5), ("pleased", 1.3), ("satisfied", 1.2), ("delighted", 1.8),
        ("perfect", 2.0), ("best", 1.8), ("awesome", 1.8), ("incredible", 2.0)
    ].iter().cloned().collect();
    
    let negative_words: HashMap<&str, f64> = [
        ("terrible", -2.0), ("awful", -2.0), ("horrible", -2.2), ("disgusting", -2.5),
        ("bad", -1.5), ("poor", -1.3), ("disappointing", -1.5), ("pathetic", -2.0),
        ("hate", -2.2), ("despise", -2.5), ("dislike", -1.0), ("detest", -2.3),
        ("sad", -1.5), ("angry", -1.8), ("frustrated", -1.6), ("annoying", -1.4),
        ("worst", -2.0), ("useless", -1.8), ("broken", -1.2), ("fail", -1.5),
        ("wrong", -1.0), ("mistake", -1.2), ("problem", -1.1), ("issue", -1.0)
    ].iter().cloned().collect();
    
    let intensifiers: HashMap<&str, f64> = [
        ("very", 1.5), ("extremely", 2.2), ("incredibly", 2.0), ("absolutely", 2.0),
        ("quite", 1.3), ("really", 1.4), ("so", 1.5), ("too", 1.3),
        ("totally", 1.8), ("completely", 1.9), ("utterly", 2.1), ("highly", 1.6)
    ].iter().cloned().collect();
    
    let negators = ["not", "no", "never", "nothing", "nowhere", "neither", "nobody", "none"];
    
    // Split into sentences and analyze each
    let sentences: Vec<&str> = text.split(&['.', '!', '?'][..])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    
    for sentence in &sentences {
        let tokens = simple_tokenize(sentence);
        let mut sentence_score = 0.0;
        let mut current_intensifier = 1.0;
        let mut negation_active = false;
        let mut negation_window = 0;
        
        for (i, token) in tokens.iter().enumerate() {
            let word = normalize_word(token);
            
            // Handle negation
            if negators.contains(&word.as_str()) {
                negation_active = true;
                negation_window = 3; // Negation affects next 3 words
                triggered_rules.push(format!("Negation detected: '{}'", word));
                continue;
            }
            
            // Handle intensifiers
            if let Some(&intensity) = intensifiers.get(word.as_str()) {
                current_intensifier *= intensity;
                triggered_rules.push(format!("Intensifier: '{}' (factor: {:.1})", word, intensity));
                continue;
            }
            
            // Score sentiment words
            let mut word_sentiment = 0.0;
            
            if let Some(&pos_score) = positive_words.get(word.as_str()) {
                word_sentiment = pos_score * current_intensifier;
                triggered_rules.push(format!("Positive word: '{}' (score: {:.1})", word, word_sentiment));
            } else if let Some(&neg_score) = negative_words.get(word.as_str()) {
                word_sentiment = neg_score * current_intensifier;
                triggered_rules.push(format!("Negative word: '{}' (score: {:.1})", word, word_sentiment));
            }
            
            // Apply negation
            if negation_active && negation_window > 0 {
                word_sentiment *= -0.8; // Flip and reduce intensity
                triggered_rules.push(format!("Negation applied to '{}'", word));
                negation_window -= 1;
                if negation_window == 0 {
                    negation_active = false;
                }
            }
            
            word_scores.insert(format!("{}_{}", word, i), word_sentiment);
            sentence_score += word_sentiment;
            current_intensifier = 1.0; // Reset intensifier after use
        }
        
        sentence_scores.push(sentence_score);
    }
    
    // Calculate overall sentiment
    let total_score: f64 = sentence_scores.iter().sum();
    let avg_score = total_score / sentence_scores.len().max(1) as f64;
    
    // Normalize polarity to [-1, 1]
    let polarity = avg_score.tanh(); // Use tanh for smooth normalization
    
    // Calculate confidence based on score magnitude and consistency
    let score_magnitude = total_score.abs();
    let score_variance = if sentence_scores.len() > 1 {
        let mean = avg_score;
        let variance: f64 = sentence_scores.iter()
            .map(|s| (s - mean).powi(2))
            .sum::<f64>() / sentence_scores.len() as f64;
        variance.sqrt()
    } else {
        0.0
    };
    
    let confidence = (score_magnitude / (score_variance + 1.0)).min(1.0);
    
    let label = if polarity > 0.15 {
        "positive"
    } else if polarity < -0.15 {
        "negative"
    } else {
        "neutral"
    };
    
    RuleBasedSentimentResult {
        polarity,
        confidence,
        label: label.to_string(),
        triggered_rules,
        word_scores,
        sentence_scores,
    }
}

// Emotion detection using keyword-based approach
fn detect_emotions(text: &str) -> EmotionResult {
    let emotion_lexicons: HashMap<&str, Vec<&str>> = [
        ("joy", vec!["happy", "joy", "delighted", "cheerful", "elated", "excited", "thrilled", "glad", "pleased"]),
        ("anger", vec!["angry", "furious", "rage", "mad", "irritated", "annoyed", "frustrated", "outraged"]),
        ("fear", vec!["afraid", "scared", "terrified", "anxious", "worried", "nervous", "panic", "frightened"]),
        ("sadness", vec!["sad", "depressed", "melancholy", "gloomy", "miserable", "sorrowful", "mournful"]),
        ("disgust", vec!["disgusted", "revolted", "repulsed", "nauseated", "sick", "appalled"]),
        ("surprise", vec!["surprised", "amazed", "astonished", "shocked", "stunned", "bewildered"]),
        ("trust", vec!["trust", "confident", "secure", "safe", "reliable", "dependable"]),
        ("anticipation", vec!["excited", "eager", "hopeful", "optimistic", "expectant", "anticipating"])
    ].iter().cloned().collect();
    
    let tokens = simple_tokenize(text);
    let mut emotion_scores = HashMap::new();
    
    // Initialize emotion scores
    for emotion in emotion_lexicons.keys() {
        emotion_scores.insert(emotion.to_string(), 0.0);
    }
    
    // Score emotions based on keyword presence
    for token in &tokens {
        let word = normalize_word(token);
        for (emotion, keywords) in &emotion_lexicons {
            if keywords.contains(&word.as_str()) {
                *emotion_scores.entry(emotion.to_string()).or_insert(0.0) += 1.0;
            }
        }
    }
    
    // Find primary emotion and normalize scores
    let total_score: f64 = emotion_scores.values().sum();
    let primary_emotion = if total_score > 0.0 {
        // Normalize scores to probabilities
        for score in emotion_scores.values_mut() {
            *score /= total_score;
        }
        
        emotion_scores.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(emotion, _)| emotion.clone())
            .unwrap_or_else(|| "neutral".to_string())
    } else {
        "neutral".to_string()
    };
    
    let confidence = emotion_scores.get(&primary_emotion).copied().unwrap_or(0.0);
    
    EmotionResult {
        primary_emotion,
        confidence,
        emotion_scores,
    }
}

// Exported function implementations

/// Rule-based sentiment analysis with detailed explanations
pub fn rule_based_sentiment(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!("Wrong arity: expected {}, got {}", 1, args.len() )));
    }
    
    let text = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError { expected: "string".to_string(), actual: format!("{:? }", args[0]) 
        }),
    };
    
    let result = advanced_rule_based_sentiment(&text);
    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

/// Statistical sentiment analysis using Naive Bayes
pub fn statistical_sentiment(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!("Wrong arity: expected {}, got {}", 2, args.len() )));
    }
    
    let text = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError { expected: "string".to_string(), actual: format!("{:? }", args[0]) 
        }),
    };
    
    let training_data = match &args[1] {
        Value::List(data) => data,
        _ => return Err(VmError::TypeError { expected: "list".to_string(), actual: format!("{:? }", args[1]) 
        }),
    };
    
    // Parse training data
    let mut parsed_data = Vec::new();
    for item in training_data {
        match item {
            Value::List(pair) => {
                if pair.len() == 2 {
                    if let (Value::String(text), Value::String(label)) = (&pair[0], &pair[1]) {
                        parsed_data.push((text.clone(), label.clone()));
                    }
                }
            }
            _ => return Err(VmError::TypeError {
                expected: "list of [text, label] pairs".to_string(),
                actual: format!("{:?}", item)
            }),
        }
    }
    
    if parsed_data.is_empty() {
        return Err(VmError::Runtime("Training data cannot be empty".to_string()
        ));
    }
    
    // Train model
    let mut model = StatisticalSentimentModel::new();
    model.train(&parsed_data)
        .map_err(|e| VmError::Runtime(e ))?;
    
    // Predict sentiment for input text
    let predictions = model.predict(&text);
    
    // Create result similar to rule-based sentiment
    let (predicted_class, confidence) = predictions.iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(class, conf)| (class.clone(), *conf))
        .unwrap_or_else(|| ("neutral".to_string(), 0.0));
    
    // Convert to polarity scale
    let polarity = match predicted_class.as_str() {
        "positive" => confidence,
        "negative" => -confidence,
        _ => 0.0,
    };
    
    let result = RuleBasedSentimentResult {
        polarity,
        confidence,
        label: predicted_class,
        triggered_rules: vec!["Statistical model prediction".to_string()],
        word_scores: HashMap::new(),
        sentence_scores: vec![polarity],
    };
    
    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

/// Emotion detection in text
pub fn emotion_detection(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!("Wrong arity: expected {}, got {}", 1, args.len() )));
    }
    
    let text = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError { expected: "string".to_string(), actual: format!("{:? }", args[0]) 
        }),
    };
    
    let result = detect_emotions(&text);
    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}