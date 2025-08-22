//! Enhanced Error Handler for REPL
//!
//! This module provides user-friendly error messages with helpful suggestions,
//! "Did you mean?" alternatives, and recovery guidance for the REPL environment.

use crate::error::LyraError;
use crate::error_enhancement::{ErrorEnhancer, EnhancedError, ErrorCategory};
use crate::repl::ReplError;
use std::fmt;

/// Enhanced error handler for REPL interactions
pub struct ReplErrorHandler {
    error_enhancer: ErrorEnhancer,
    show_verbose_errors: bool,
    show_suggestions: bool,
    show_examples: bool,
}

/// Enhanced REPL error message with formatting
#[derive(Debug, Clone)]
pub struct ReplEnhancedError {
    pub error_type: String,
    pub friendly_message: String,
    pub formatted_message: String,
    pub suggestions: Vec<String>,
    pub did_you_mean: Vec<String>,
    pub code_examples: Vec<String>, 
    pub recovery_steps: Vec<String>,
    pub help_commands: Vec<String>,
}

impl ReplErrorHandler {
    /// Create a new enhanced error handler
    pub fn new() -> Self {
        Self {
            error_enhancer: ErrorEnhancer::new(),
            show_verbose_errors: true,
            show_suggestions: true,
            show_examples: true,
        }
    }
    
    /// Configure verbosity settings
    pub fn configure(&mut self, verbose: bool, suggestions: bool, examples: bool) {
        self.show_verbose_errors = verbose;
        self.show_suggestions = suggestions;
        self.show_examples = examples;
    }
    
    /// Handle and format any REPL error with enhancements
    pub fn handle_error(&self, error: &ReplError, input: &str) -> ReplEnhancedError {
        match error {
            ReplError::ParseError { message } => {
                let lyra_error = LyraError::Parse { 
                    message: message.clone(), 
                    position: 0 
                };
                self.format_enhanced_error(&lyra_error, input)
            },
            ReplError::CompilationError { message } => {
                let lyra_error = LyraError::Compilation { 
                    message: message.clone() 
                };
                self.format_enhanced_error(&lyra_error, input)
            },
            ReplError::RuntimeError { message } => {
                let lyra_error = LyraError::Runtime { 
                    message: message.clone() 
                };
                self.format_enhanced_error(&lyra_error, input)
            },
            _ => self.format_generic_error(error, input),
        }
    }
    
    /// Format enhanced error with user-friendly presentation
    fn format_enhanced_error(&self, lyra_error: &LyraError, input: &str) -> ReplEnhancedError {
        let enhanced = self.error_enhancer.enhance_error(lyra_error, input);
        
        let error_type = match enhanced.error_category {
            ErrorCategory::UnknownFunction => "Unknown Function".to_string(),
            ErrorCategory::BracketMismatch => "Syntax Error".to_string(),
            ErrorCategory::ArgumentCount => "Wrong Number of Arguments".to_string(),
            ErrorCategory::ArgumentType => "Type Mismatch".to_string(),
            ErrorCategory::DivisionByZero => "Division by Zero".to_string(),
            ErrorCategory::IndexOutOfBounds => "Index Out of Bounds".to_string(),
            ErrorCategory::ParseError => "Syntax Error".to_string(),
            ErrorCategory::FileNotFound => "File Not Found".to_string(),
            ErrorCategory::PatternSyntax => "Pattern Error".to_string(),
            ErrorCategory::RuntimeEvaluation => "Runtime Error".to_string(),
        };
        
        let formatted_message = self.format_error_message(&enhanced);
        let suggestions = self.format_suggestions(&enhanced);
        let examples = self.format_code_examples(&enhanced);
        let help_commands = self.suggest_help_commands(&enhanced.error_category);
        
        ReplEnhancedError {
            error_type,
            friendly_message: enhanced.friendly_message,
            formatted_message,
            suggestions,
            did_you_mean: enhanced.did_you_mean,
            code_examples: examples,
            recovery_steps: enhanced.recovery_steps,
            help_commands,
        }
    }
    
    /// Format generic error for non-Lyra errors
    fn format_generic_error(&self, error: &ReplError, _input: &str) -> ReplEnhancedError {
        let error_type = match error {
            ReplError::InvalidMetaCommand { .. } => "Invalid Command".to_string(),
            ReplError::Config(_) => "Configuration Error".to_string(),
            ReplError::History(_) => "History Error".to_string(),
            ReplError::Other { .. } => "General Error".to_string(),
            _ => "Error".to_string(),
        };
        
        let friendly_message = match error {
            ReplError::InvalidMetaCommand { command } => {
                format!("I don't recognize the command '{}'.", command)
            },
            _ => "An error occurred.".to_string(),
        };
        
        let suggestions = match error {
            ReplError::InvalidMetaCommand { .. } => {
                vec![
                    "Use '%help' to see all available commands".to_string(),
                    "Check the command spelling and try again".to_string(),
                ]
            },
            _ => vec!["Try the operation again".to_string()],
        };
        
        let help_commands = match error {
            ReplError::InvalidMetaCommand { .. } => {
                vec!["%help".to_string(), "%commands".to_string()]
            },
            _ => vec!["%help".to_string()],
        };
        
        ReplEnhancedError {
            error_type,
            friendly_message: friendly_message.clone(),
            formatted_message: self.format_simple_error(&friendly_message),
            suggestions,
            did_you_mean: vec![],
            code_examples: vec![],
            recovery_steps: vec![],
            help_commands,
        }
    }
    
    /// Format the main error message with styling
    fn format_error_message(&self, enhanced: &EnhancedError) -> String {
        let mut output = String::new();
        
        // Error header with icon
        let icon = match enhanced.error_category {
            ErrorCategory::UnknownFunction => "🔍",
            ErrorCategory::BracketMismatch => "⚠️",
            ErrorCategory::ArgumentCount | ErrorCategory::ArgumentType => "🔧",
            ErrorCategory::DivisionByZero => "➗",
            ErrorCategory::IndexOutOfBounds => "📋",
            ErrorCategory::ParseError => "📝",
            ErrorCategory::FileNotFound => "📁",
            ErrorCategory::PatternSyntax => "🔗",
            ErrorCategory::RuntimeEvaluation => "⚡",
        };
        
        output.push_str(&format!("{} {}\n", icon, enhanced.friendly_message));
        
        // Position information if available
        if let Some(ref pos) = enhanced.position_info {
            output.push_str(&format!("   at line {}, column {}\n", pos.line, pos.column));
            output.push_str(&format!("   {}\n", pos.context));
            output.push_str(&format!("   {}\n", pos.pointer));
        }
        
        output
    }
    
    /// Format suggestions section
    fn format_suggestions(&self, enhanced: &EnhancedError) -> Vec<String> {
        if !self.show_suggestions || enhanced.suggestions.is_empty() {
            return vec![];
        }
        
        let mut formatted = vec!["💡 Suggestions:".to_string()];
        
        for (i, suggestion) in enhanced.suggestions.iter().enumerate() {
            let confidence_indicator = if suggestion.confidence > 0.8 {
                "⭐"
            } else if suggestion.confidence > 0.6 {
                "✨"
            } else {
                "💭"
            };
            
            formatted.push(format!(
                "   {} {}: {}", 
                confidence_indicator, 
                i + 1, 
                suggestion.description
            ));
            
            if let Some(ref code) = suggestion.fix_code {
                formatted.push(format!("      Try: {}", code));
            }
        }
        
        // Add "Did you mean?" suggestions
        if !enhanced.did_you_mean.is_empty() {
            formatted.push("".to_string());
            formatted.push("🤔 Did you mean:".to_string());
            for alternative in &enhanced.did_you_mean {
                formatted.push(format!("   • {}", alternative));
            }
        }
        
        formatted
    }
    
    /// Format code examples
    fn format_code_examples(&self, enhanced: &EnhancedError) -> Vec<String> {
        if !self.show_examples || enhanced.code_examples.is_empty() {
            return vec![];
        }
        
        let mut formatted = vec!["📚 Examples:".to_string()];
        
        for example in &enhanced.code_examples {
            formatted.push(format!("   {}", example.title));
            formatted.push(format!("   ✅ {}", example.correct_code));
            formatted.push(format!("      {}", example.explanation));
            
            if let Some(ref mistake) = example.common_mistake {
                formatted.push(format!("   ❌ {}", mistake));
            }
            formatted.push("".to_string());
        }
        
        formatted
    }
    
    /// Suggest relevant help commands
    fn suggest_help_commands(&self, error_category: &ErrorCategory) -> Vec<String> {
        match error_category {
            ErrorCategory::UnknownFunction => {
                vec![
                    "%functions".to_string(),
                    "%search <term>".to_string(),
                    "%help <function>".to_string(),
                ]
            },
            ErrorCategory::BracketMismatch | ErrorCategory::ParseError => {
                vec![
                    "%syntax".to_string(),
                    "%examples".to_string(),
                ]
            },
            ErrorCategory::ArgumentCount | ErrorCategory::ArgumentType => {
                vec![
                    "%help <function>".to_string(),
                    "%signatures".to_string(),
                ]
            },
            ErrorCategory::DivisionByZero | ErrorCategory::RuntimeEvaluation => {
                vec![
                    "%debug".to_string(),
                    "%validate".to_string(),
                ]
            },
            _ => vec!["%help".to_string()],
        }
    }
    
    /// Format simple error message
    fn format_simple_error(&self, message: &str) -> String {
        format!("❌ {}", message)
    }
    
    /// Create contextual tips based on error category
    pub fn get_contextual_tips(&self, error_category: &ErrorCategory) -> Vec<String> {
        match error_category {
            ErrorCategory::UnknownFunction => vec![
                "💡 Function names are case-sensitive (use 'Sin', not 'sin')".to_string(),
                "💡 Use square brackets for function calls: Sin[x]".to_string(),
                "💡 Type '%functions' to see all available functions".to_string(),
            ],
            ErrorCategory::BracketMismatch => vec![
                "💡 Check that all brackets [ ] are properly matched".to_string(),
                "💡 Use [ ] for functions, { } for lists, ( ) for grouping".to_string(),
                "💡 Count opening and closing brackets carefully".to_string(),
            ],
            ErrorCategory::ArgumentCount => vec![
                "💡 Check the function signature with '%help <function>'".to_string(),
                "💡 Some functions have optional arguments".to_string(),
                "💡 Use '%examples <function>' to see usage examples".to_string(),
            ],
            ErrorCategory::DivisionByZero => vec![
                "💡 Add a condition to check for zero before dividing".to_string(),
                "💡 Use If[divisor != 0, expression, alternative]".to_string(),
                "💡 Consider using Missing for undefined results".to_string(),
            ],
            _ => vec![
                "💡 Use '%help' to explore available commands".to_string(),
                "💡 Try '%examples' to see syntax examples".to_string(),
            ],
        }
    }
}

impl fmt::Display for ReplEnhancedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.formatted_message)?;
        
        // Add suggestions
        for suggestion in &self.suggestions {
            writeln!(f, "{}", suggestion)?;
        }
        
        // Add examples
        for example in &self.code_examples {
            writeln!(f, "{}", example)?;
        }
        
        // Add recovery steps
        if !self.recovery_steps.is_empty() {
            writeln!(f, "\n🔧 Recovery Steps:")?;
            for step in &self.recovery_steps {
                writeln!(f, "   • {}", step)?;
            }
        }
        
        // Add help commands
        if !self.help_commands.is_empty() {
            writeln!(f, "\n🆘 Get Help:")?;
            for cmd in &self.help_commands {
                writeln!(f, "   • {}", cmd)?;
            }
        }
        
        Ok(())
    }
}

impl Default for ReplErrorHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unknown_function_error() {
        let handler = ReplErrorHandler::new();
        let error = ReplError::ParseError {
            message: "Unknown function: sine".to_string(),
        };
        
        let enhanced = handler.handle_error(&error, "sine[0]");
        assert_eq!(enhanced.error_type, "Syntax Error");
        assert!(!enhanced.suggestions.is_empty());
    }
    
    #[test]
    fn test_invalid_command_error() {
        let handler = ReplErrorHandler::new();
        let error = ReplError::InvalidMetaCommand {
            command: "%invalid".to_string(),
        };
        
        let enhanced = handler.handle_error(&error, "%invalid");
        assert_eq!(enhanced.error_type, "Invalid Command");
        assert!(enhanced.help_commands.contains(&"%help".to_string()));
    }
    
    #[test]
    fn test_contextual_tips() {
        let handler = ReplErrorHandler::new();
        let tips = handler.get_contextual_tips(&ErrorCategory::UnknownFunction);
        assert!(!tips.is_empty());
        assert!(tips.iter().any(|tip| tip.contains("case-sensitive")));
    }
}