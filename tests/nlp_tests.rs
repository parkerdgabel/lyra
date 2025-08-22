//! Comprehensive tests for Natural Language Processing module
//! Following TDD principles - tests written before implementation

use lyra::vm::{Value, VirtualMachine, VmError};
use lyra::stdlib::StandardLibrary;

#[cfg(test)]
mod text_processing_tests {
    use super::*;
    
    #[test]
    fn test_tokenize_text_basic() {
        let mut vm = VirtualMachine::new();
        let stdlib = StandardLibrary::new();
        
        // Test basic tokenization
        let result = stdlib.get_function("TokenizeText")
            .unwrap()(&[Value::String("Hello world! How are you?".to_string())]);
        
        match result {
            Ok(Value::List(tokens)) => {
                assert_eq!(tokens.len(), 6); // "Hello", "world", "!", "How", "are", "you", "?"
                if let Value::String(first_token) = &tokens[0] {
                    assert_eq!(first_token, "Hello");
                }
            }
            _ => panic!("Expected tokenized list"),
        }
    }
    
    #[test]
    fn test_tokenize_text_whitespace() {
        let stdlib = StandardLibrary::new();
        
        // Test whitespace tokenization
        let result = stdlib.get_function("TokenizeText")
            .unwrap()(&[
                Value::String("The quick brown fox".to_string()),
                Value::String("whitespace".to_string())
            ]);
        
        match result {
            Ok(Value::List(tokens)) => {
                assert_eq!(tokens.len(), 4);
                if let Value::String(token) = &tokens[1] {
                    assert_eq!(token, "quick");
                }
            }
            _ => panic!("Expected tokenized list"),
        }
    }
    
    #[test]
    fn test_tokenize_text_regex() {
        let stdlib = StandardLibrary::new();
        
        // Test regex tokenization
        let result = stdlib.get_function("TokenizeText")
            .unwrap()(&[
                Value::String("word1,word2;word3:word4".to_string()),
                Value::String("regex".to_string()),
                Value::String(r"[,;:]+".to_string())
            ]);
        
        match result {
            Ok(Value::List(tokens)) => {
                assert_eq!(tokens.len(), 4);
                if let Value::String(token) = &tokens[2] {
                    assert_eq!(token, "word3");
                }
            }
            _ => panic!("Expected tokenized list"),
        }
    }
    
    #[test]
    fn test_stem_text_porter() {
        let stdlib = StandardLibrary::new();
        
        // Test Porter stemming algorithm
        let result = stdlib.get_function("StemText")
            .unwrap()(&[Value::List(vec![
                Value::String("running".to_string()),
                Value::String("flies".to_string()),
                Value::String("dogs".to_string()),
                Value::String("fairly".to_string())
            ])]);
        
        match result {
            Ok(Value::List(stems)) => {
                assert_eq!(stems.len(), 4);
                if let Value::String(stem) = &stems[0] {
                    assert_eq!(stem, "run");
                }
                if let Value::String(stem) = &stems[1] {
                    assert_eq!(stem, "fli");
                }
            }
            _ => panic!("Expected stemmed list"),
        }
    }
    
    #[test]
    fn test_generate_ngrams() {
        let stdlib = StandardLibrary::new();
        
        // Test bigram generation
        let result = stdlib.get_function("GenerateNGrams")
            .unwrap()(&[
                Value::List(vec![
                    Value::String("the".to_string()),
                    Value::String("quick".to_string()),
                    Value::String("brown".to_string()),
                    Value::String("fox".to_string())
                ]),
                Value::Integer(2) // bigrams
            ]);
        
        match result {
            Ok(Value::List(ngrams)) => {
                assert_eq!(ngrams.len(), 3); // 4 words -> 3 bigrams
                if let Value::List(first_ngram) = &ngrams[0] {
                    assert_eq!(first_ngram.len(), 2);
                    if let Value::String(word) = &first_ngram[0] {
                        assert_eq!(word, "the");
                    }
                    if let Value::String(word) = &first_ngram[1] {
                        assert_eq!(word, "quick");
                    }
                }
            }
            _ => panic!("Expected n-grams list"),
        }
    }
    
    #[test]
    fn test_generate_trigrams() {
        let stdlib = StandardLibrary::new();
        
        // Test trigram generation
        let result = stdlib.get_function("GenerateNGrams")
            .unwrap()(&[
                Value::List(vec![
                    Value::String("the".to_string()),
                    Value::String("quick".to_string()),
                    Value::String("brown".to_string()),
                    Value::String("fox".to_string()),
                    Value::String("jumps".to_string())
                ]),
                Value::Integer(3) // trigrams
            ]);
        
        match result {
            Ok(Value::List(ngrams)) => {
                assert_eq!(ngrams.len(), 3); // 5 words -> 3 trigrams
                if let Value::List(first_ngram) = &ngrams[0] {
                    assert_eq!(first_ngram.len(), 3);
                }
            }
            _ => panic!("Expected n-grams list"),
        }
    }
    
    #[test]
    fn test_tfidf_vectorize() {
        let stdlib = StandardLibrary::new();
        
        // Test TF-IDF vectorization
        let documents = vec![
            Value::String("the quick brown fox".to_string()),
            Value::String("the lazy dog sleeps".to_string()),
            Value::String("quick brown animals run".to_string())
        ];
        
        let result = stdlib.get_function("TFIDFVectorize")
            .unwrap()(&[Value::List(documents)]);
        
        match result {
            Ok(Value::LyObj(_)) => {
                // Should return TFIDFModel Foreign object
                // Will verify detailed functionality in implementation tests
            }
            _ => panic!("Expected TF-IDF model object"),
        }
    }
    
    #[test]
    fn test_word_frequency() {
        let stdlib = StandardLibrary::new();
        
        let result = stdlib.get_function("WordFrequency")
            .unwrap()(&[Value::List(vec![
                Value::String("the".to_string()),
                Value::String("quick".to_string()),
                Value::String("the".to_string()),
                Value::String("fox".to_string()),
                Value::String("quick".to_string()),
                Value::String("the".to_string())
            ])]);
        
        match result {
            Ok(Value::LyObj(_)) => {
                // Should return word frequency map as Foreign object
            }
            _ => panic!("Expected word frequency map"),
        }
    }
    
    #[test]
    fn test_normalize_text() {
        let stdlib = StandardLibrary::new();
        
        let result = stdlib.get_function("NormalizeText")
            .unwrap()(&[Value::String("Hello, WORLD!!! How are YOU???".to_string())]);
        
        match result {
            Ok(Value::String(normalized)) => {
                // Should be lowercased and punctuation removed
                assert!(normalized.contains("hello"));
                assert!(normalized.contains("world"));
                assert!(!normalized.contains("!"));
            }
            _ => panic!("Expected normalized string"),
        }
    }
}

#[cfg(test)]
mod text_analysis_tests {
    use super::*;
    
    #[test]
    fn test_sentiment_analysis_positive() {
        let stdlib = StandardLibrary::new();
        
        let result = stdlib.get_function("SentimentAnalysis")
            .unwrap()(&[Value::String("I love this amazing product! It's fantastic!".to_string())]);
        
        match result {
            Ok(Value::LyObj(_)) => {
                // Should return sentiment score object with positive sentiment
            }
            _ => panic!("Expected sentiment analysis result"),
        }
    }
    
    #[test]
    fn test_sentiment_analysis_negative() {
        let stdlib = StandardLibrary::new();
        
        let result = stdlib.get_function("SentimentAnalysis")
            .unwrap()(&[Value::String("I hate this terrible product. It's awful!".to_string())]);
        
        match result {
            Ok(Value::LyObj(_)) => {
                // Should return sentiment score object with negative sentiment
            }
            _ => panic!("Expected sentiment analysis result"),
        }
    }
    
    #[test]
    fn test_named_entity_recognition() {
        let stdlib = StandardLibrary::new();
        
        let result = stdlib.get_function("NamedEntityRecognition")
            .unwrap()(&[Value::String("John Smith works at Microsoft in Seattle.".to_string())]);
        
        match result {
            Ok(Value::List(entities)) => {
                // Should extract entities: "John Smith" (PERSON), "Microsoft" (ORG), "Seattle" (LOC)
                assert!(entities.len() >= 2);
            }
            _ => panic!("Expected named entity list"),
        }
    }
    
    #[test]
    fn test_pos_tagging() {
        let stdlib = StandardLibrary::new();
        
        let result = stdlib.get_function("POSTagging")
            .unwrap()(&[Value::String("The quick brown fox jumps".to_string())]);
        
        match result {
            Ok(Value::List(tagged_words)) => {
                assert_eq!(tagged_words.len(), 5);
                // Each item should be a list [word, tag]
            }
            _ => panic!("Expected POS tagged list"),
        }
    }
    
    #[test]
    fn test_language_detection() {
        let stdlib = StandardLibrary::new();
        
        let result = stdlib.get_function("LanguageDetection")
            .unwrap()(&[Value::String("Hello world, how are you doing today?".to_string())]);
        
        match result {
            Ok(Value::String(language)) => {
                assert_eq!(language, "en");
            }
            _ => panic!("Expected language code"),
        }
    }
    
    #[test]
    fn test_language_detection_spanish() {
        let stdlib = StandardLibrary::new();
        
        let result = stdlib.get_function("LanguageDetection")
            .unwrap()(&[Value::String("Hola mundo, ¿cómo estás hoy?".to_string())]);
        
        match result {
            Ok(Value::String(language)) => {
                assert_eq!(language, "es");
            }
            _ => panic!("Expected Spanish language code"),
        }
    }
    
    #[test]
    fn test_text_classification() {
        let stdlib = StandardLibrary::new();
        
        // Test with pre-trained model (placeholder for now)
        let result = stdlib.get_function("TextClassification")
            .unwrap()(&[
                Value::String("This movie is absolutely amazing! Best film ever!".to_string()),
                Value::String("movie_sentiment".to_string()) // model type
            ]);
        
        match result {
            Ok(Value::LyObj(_)) => {
                // Should return classification result with confidence scores
            }
            _ => panic!("Expected classification result"),
        }
    }
    
    #[test]
    fn test_keyword_extraction() {
        let stdlib = StandardLibrary::new();
        
        let result = stdlib.get_function("KeywordExtraction")
            .unwrap()(&[Value::String("Machine learning and artificial intelligence are transforming the technology industry with neural networks and deep learning algorithms.".to_string())]);
        
        match result {
            Ok(Value::List(keywords)) => {
                assert!(keywords.len() >= 3);
                // Should extract keywords like "machine learning", "artificial intelligence", etc.
            }
            _ => panic!("Expected keyword list"),
        }
    }
}

#[cfg(test)]
mod language_modeling_tests {
    use super::*;
    
    #[test]
    fn test_language_model_creation() {
        let stdlib = StandardLibrary::new();
        
        let training_data = vec![
            Value::String("the quick brown fox".to_string()),
            Value::String("the lazy dog sleeps".to_string()),
            Value::String("brown foxes are quick".to_string())
        ];
        
        let result = stdlib.get_function("LanguageModel")
            .unwrap()(&[
                Value::List(training_data),
                Value::Integer(2) // bigram model
            ]);
        
        match result {
            Ok(Value::LyObj(_)) => {
                // Should return trained language model
            }
            _ => panic!("Expected language model"),
        }
    }
    
    #[test]
    fn test_text_similarity_cosine() {
        let stdlib = StandardLibrary::new();
        
        let result = stdlib.get_function("TextSimilarity")
            .unwrap()(&[
                Value::String("The quick brown fox jumps over the lazy dog".to_string()),
                Value::String("A fast brown fox leaps over a sleepy dog".to_string()),
                Value::String("cosine".to_string()) // similarity metric
            ]);
        
        match result {
            Ok(Value::Real(similarity)) => {
                assert!(similarity >= 0.0 && similarity <= 1.0);
                assert!(similarity > 0.5); // Should be quite similar
            }
            _ => panic!("Expected similarity score"),
        }
    }
    
    #[test]
    fn test_text_similarity_jaccard() {
        let stdlib = StandardLibrary::new();
        
        let result = stdlib.get_function("TextSimilarity")
            .unwrap()(&[
                Value::String("cat dog bird".to_string()),
                Value::String("cat dog fish".to_string()),
                Value::String("jaccard".to_string()) // similarity metric
            ]);
        
        match result {
            Ok(Value::Real(similarity)) => {
                assert!(similarity >= 0.0 && similarity <= 1.0);
                // Should be 2/4 = 0.5 (2 common words, 4 total unique words)
                assert!((similarity - 0.5).abs() < 0.01);
            }
            _ => panic!("Expected Jaccard similarity score"),
        }
    }
    
    #[test]
    fn test_text_summarization() {
        let stdlib = StandardLibrary::new();
        
        let long_text = "Natural language processing is a field of artificial intelligence. \
                        It focuses on the interaction between computers and human language. \
                        NLP techniques are used in many applications today. \
                        Text analysis is a key component of NLP systems. \
                        Machine learning algorithms power modern NLP tools.";
        
        let result = stdlib.get_function("TextSummarization")
            .unwrap()(&[
                Value::String(long_text.to_string()),
                Value::Integer(2) // number of sentences
            ]);
        
        match result {
            Ok(Value::String(summary)) => {
                // Summary should be shorter than original
                assert!(summary.len() < long_text.len());
                // Should contain key concepts
                assert!(summary.to_lowercase().contains("nlp") || 
                        summary.to_lowercase().contains("natural language") ||
                        summary.to_lowercase().contains("processing"));
            }
            _ => panic!("Expected text summary"),
        }
    }
    
    #[test]
    fn test_spell_check() {
        let stdlib = StandardLibrary::new();
        
        let result = stdlib.get_function("SpellCheck")
            .unwrap()(&[Value::String("Ths is a tset with som mispellings".to_string())]);
        
        match result {
            Ok(Value::List(errors)) => {
                assert!(errors.len() >= 3); // Should find multiple misspellings
            }
            _ => panic!("Expected spell check errors list"),
        }
    }
    
    #[test]
    fn test_spell_correct() {
        let stdlib = StandardLibrary::new();
        
        let result = stdlib.get_function("SpellCorrect")
            .unwrap()(&[Value::String("Ths is a tset".to_string())]);
        
        match result {
            Ok(Value::String(corrected)) => {
                // Should correct obvious misspellings
                assert!(corrected.contains("This"));
                assert!(corrected.contains("test"));
            }
            _ => panic!("Expected corrected text"),
        }
    }
}

#[cfg(test)]
mod advanced_nlp_tests {
    use super::*;
    
    #[test]
    fn test_rule_based_sentiment() {
        let stdlib = StandardLibrary::new();
        
        let result = stdlib.get_function("RuleBasedSentiment")
            .unwrap()(&[Value::String("I absolutely love this fantastic product!".to_string())]);
        
        match result {
            Ok(Value::LyObj(_)) => {
                // Should return detailed sentiment analysis with rule explanations
            }
            _ => panic!("Expected rule-based sentiment result"),
        }
    }
    
    #[test]
    fn test_statistical_sentiment() {
        let stdlib = StandardLibrary::new();
        
        // Need to create a training corpus for statistical model
        let training_data = vec![
            Value::List(vec![Value::String("I love this".to_string()), Value::String("positive".to_string())]),
            Value::List(vec![Value::String("I hate this".to_string()), Value::String("negative".to_string())]),
            Value::List(vec![Value::String("This is okay".to_string()), Value::String("neutral".to_string())]),
        ];
        
        let result = stdlib.get_function("StatisticalSentiment")
            .unwrap()(&[
                Value::String("This product is amazing".to_string()),
                Value::List(training_data)
            ]);
        
        match result {
            Ok(Value::LyObj(_)) => {
                // Should return statistical sentiment classification
            }
            _ => panic!("Expected statistical sentiment result"),
        }
    }
    
    #[test]
    fn test_emotion_detection() {
        let stdlib = StandardLibrary::new();
        
        let result = stdlib.get_function("EmotionDetection")
            .unwrap()(&[Value::String("I'm so excited and happy about this news!".to_string())]);
        
        match result {
            Ok(Value::LyObj(_)) => {
                // Should return emotion scores (joy, anger, fear, sadness, etc.)
            }
            _ => panic!("Expected emotion detection result"),
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_full_nlp_pipeline() {
        let stdlib = StandardLibrary::new();
        
        let text = "I really love this amazing product! It's the best purchase I've ever made. \
                   The quality is outstanding and the customer service was excellent. \
                   I would definitely recommend this to everyone!";
        
        // 1. Tokenize
        let tokens_result = stdlib.get_function("TokenizeText")
            .unwrap()(&[Value::String(text.to_string())]);
        
        assert!(tokens_result.is_ok());
        
        // 2. Sentiment analysis
        let sentiment_result = stdlib.get_function("SentimentAnalysis")
            .unwrap()(&[Value::String(text.to_string())]);
        
        assert!(sentiment_result.is_ok());
        
        // 3. Extract keywords
        let keywords_result = stdlib.get_function("KeywordExtraction")
            .unwrap()(&[Value::String(text.to_string())]);
        
        assert!(keywords_result.is_ok());
        
        // 4. Language detection
        let language_result = stdlib.get_function("LanguageDetection")
            .unwrap()(&[Value::String(text.to_string())]);
        
        assert!(language_result.is_ok());
    }
    
    #[test]
    fn test_multilingual_support() {
        let stdlib = StandardLibrary::new();
        
        let texts = vec![
            ("Hello world", "en"),
            ("Hola mundo", "es"),
            ("Bonjour le monde", "fr"),
            ("Hallo Welt", "de"),
            ("Ciao mondo", "it")
        ];
        
        for (text, expected_lang) in texts {
            let result = stdlib.get_function("LanguageDetection")
                .unwrap()(&[Value::String(text.to_string())]);
            
            match result {
                Ok(Value::String(detected_lang)) => {
                    assert_eq!(detected_lang, expected_lang);
                }
                _ => panic!("Failed to detect language for: {}", text),
            }
        }
    }
    
    #[test]
    fn test_performance_large_text() {
        let stdlib = StandardLibrary::new();
        
        // Create large text document
        let large_text = "This is a test sentence. ".repeat(1000);
        
        let start_time = std::time::Instant::now();
        
        let result = stdlib.get_function("TokenizeText")
            .unwrap()(&[Value::String(large_text)]);
        
        let duration = start_time.elapsed();
        
        assert!(result.is_ok());
        assert!(duration.as_millis() < 1000); // Should process within 1 second
    }
}