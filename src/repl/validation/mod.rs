//! Smart input validation for Lyra REPL
//!
//! This module provides comprehensive input validation capabilities including
//! bracket/quote balance checking, syntax validation, and multiline expression detection.

use crate::repl::config::{ReplConfig, ValidationConfig};
use rustyline::validate::ValidationResult;
use rustyline::Result as RustylineResult;

pub mod bracket_matcher;
pub mod multiline;
pub mod syntax_checker;

use bracket_matcher::BracketMatcher;
use multiline::MultilineDetector;
use syntax_checker::SyntaxChecker;

/// Smart input validator for Lyra expressions
pub struct LyraInputValidator {
    /// Bracket and quote balance checker
    bracket_matcher: BracketMatcher,
    /// Multiline expression detector
    multiline_detector: MultilineDetector,
    /// Lightweight syntax checker
    syntax_checker: SyntaxChecker,
    /// Validation configuration
    config: ValidationConfig,
}


/// Result of input validation
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationStatus {
    /// Input is valid and complete
    Valid,
    /// Input is valid but incomplete (needs multiline continuation)
    Incomplete { 
        reason: String,
        suggested_continuation: Option<String>,
    },
    /// Input has syntax errors
    Invalid { 
        error: String,
        suggestion: Option<String>,
    },
}

impl LyraInputValidator {
    /// Create a new input validator
    pub fn new(config: &ReplConfig) -> Self {
        Self {
            bracket_matcher: BracketMatcher::new(),
            multiline_detector: MultilineDetector::new(),
            syntax_checker: SyntaxChecker::new(),
            config: config.validation.clone(),
        }
    }
    
    /// Update the validator configuration
    pub fn update_config(&mut self, config: &ReplConfig) {
        self.config = config.validation.clone();
    }
    
    /// Validate input text and determine if continuation is needed
    pub fn validate_input(&self, input: &str) -> ValidationStatus {
        // Quick check for empty input
        if input.trim().is_empty() {
            return ValidationStatus::Valid;
        }
        
        // Check bracket/quote balance if enabled
        if self.config.enable_bracket_matching {
            match self.bracket_matcher.check_balance(input) {
                Ok(balance_result) => {
                    if !balance_result.is_balanced {
                        return ValidationStatus::Incomplete {
                            reason: balance_result.error_message,
                            suggested_continuation: balance_result.suggested_fix,
                        };
                    }
                }
                Err(error) => {
                    return ValidationStatus::Invalid {
                        error: format!("Bracket matching error: {}", error),
                        suggestion: Some("Check for mismatched brackets or quotes".to_string()),
                    };
                }
            }
        }
        
        // Check for multiline expressions if enabled
        if self.config.enable_multiline_detection {
            match self.multiline_detector.detect_multiline(input) {
                Ok(multiline_result) => {
                    if multiline_result.needs_continuation {
                        return ValidationStatus::Incomplete {
                            reason: multiline_result.reason,
                            suggested_continuation: multiline_result.suggested_continuation,
                        };
                    }
                }
                Err(error) => {
                    // Multiline detection errors are usually not fatal
                    if self.config.show_validation_hints {
                        eprintln!("Multiline detection warning: {}", error);
                    }
                }
            }
        }
        
        // Check syntax if enabled
        if self.config.enable_syntax_checking {
            match self.syntax_checker.check_syntax(input) {
                Ok(syntax_result) => {
                    if !syntax_result.is_valid {
                        return ValidationStatus::Invalid {
                            error: syntax_result.error_message,
                            suggestion: syntax_result.suggestion,
                        };
                    }
                }
                Err(error) => {
                    return ValidationStatus::Invalid {
                        error: format!("Syntax error: {}", error),
                        suggestion: Some("Check expression syntax".to_string()),
                    };
                }
            }
        }
        
        ValidationStatus::Valid
    }
    
    /// Convert validation status to rustyline ValidationResult
    pub fn to_rustyline_result(&self, status: ValidationStatus) -> RustylineResult<ValidationResult> {
        match status {
            ValidationStatus::Valid => {
                Ok(ValidationResult::Valid(None))
            }
            ValidationStatus::Incomplete { reason, .. } => {
                // Signal that input is incomplete and needs continuation
                Ok(ValidationResult::Incomplete)
            }
            ValidationStatus::Invalid { error, suggestion } => {
                let message = if let Some(hint) = suggestion {
                    format!("{}: {}", error, hint)
                } else {
                    error
                };
                
                if self.config.show_validation_hints {
                    Ok(ValidationResult::Invalid(Some(message)))
                } else {
                    Ok(ValidationResult::Invalid(None))
                }
            }
        }
    }
    
    /// Get validation statistics for debugging
    pub fn get_stats(&self) -> ValidationStats {
        ValidationStats {
            bracket_matching_enabled: self.config.enable_bracket_matching,
            syntax_checking_enabled: self.config.enable_syntax_checking,
            multiline_detection_enabled: self.config.enable_multiline_detection,
            validation_delay_ms: self.config.validation_delay_ms,
        }
    }
}

/// Statistics about the validator's configuration and performance
#[derive(Debug, Clone)]
pub struct ValidationStats {
    pub bracket_matching_enabled: bool,
    pub syntax_checking_enabled: bool,
    pub multiline_detection_enabled: bool,
    pub validation_delay_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_creation() {
        let config = ReplConfig::default();
        let validator = LyraInputValidator::new(&config);
        
        let stats = validator.get_stats();
        assert!(stats.bracket_matching_enabled);
        assert!(stats.syntax_checking_enabled);
        assert!(stats.multiline_detection_enabled);
    }
    
    #[test]
    fn test_empty_input_validation() {
        let config = ReplConfig::default();
        let validator = LyraInputValidator::new(&config);
        
        let result = validator.validate_input("");
        assert_eq!(result, ValidationStatus::Valid);
        
        let result = validator.validate_input("   ");
        assert_eq!(result, ValidationStatus::Valid);
    }
    
    #[test]
    fn test_simple_valid_input() {
        let config = ReplConfig::default();
        let validator = LyraInputValidator::new(&config);
        
        let result = validator.validate_input("42");
        assert_eq!(result, ValidationStatus::Valid);
        
        let result = validator.validate_input("Sin[Pi/2]");
        assert_eq!(result, ValidationStatus::Valid);
    }
    
    #[test]
    fn test_rustyline_conversion() {
        let config = ReplConfig::default();
        let validator = LyraInputValidator::new(&config);
        
        // Test valid conversion
        let status = ValidationStatus::Valid;
        let result = validator.to_rustyline_result(status).unwrap();
        matches!(result, ValidationResult::Valid(None));
        
        // Test incomplete conversion
        let status = ValidationStatus::Incomplete {
            reason: "Unclosed bracket".to_string(),
            suggested_continuation: None,
        };
        let result = validator.to_rustyline_result(status).unwrap();
        matches!(result, ValidationResult::Incomplete);
    }
    
    #[test]
    fn test_config_update() {
        let config = ReplConfig::default();
        let mut validator = LyraInputValidator::new(&config);
        
        // Update config
        let new_config = ReplConfig::default();
        validator.update_config(&new_config);
        
        // Should not panic and should work normally
        let result = validator.validate_input("Sin[x]");
        assert_eq!(result, ValidationStatus::Valid);
    }
}