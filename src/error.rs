use thiserror::Error;
use crate::vm::VmError;
use crate::foreign::ForeignError;
use crate::compiler::CompilerError;

pub type Result<T> = std::result::Result<T, Error>;
pub type LyraResult<T> = std::result::Result<T, LyraError>;

/// Helper function to convert position to line and column numbers
pub fn position_to_line_col(source: &str, position: usize) -> (usize, usize) {
    let mut line = 1;
    let mut col = 1;

    for (i, ch) in source.chars().enumerate() {
        if i >= position {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }

    (line, col)
}

/// Enhanced error display with source context
pub fn format_error_with_context(error: &Error, source: &str) -> String {
    match error {
        Error::Parse { message, position } | Error::Lexer { message, position } => {
            let (line, col) = position_to_line_col(source, *position);
            let lines: Vec<&str> = source.lines().collect();

            let mut result = format!("Error at line {}, column {}: {}\n", line, col, message);

            // Show the problematic line if available
            if line > 0 && line <= lines.len() {
                let error_line = lines[line - 1];
                result.push_str(&format!("  {} | {}\n", line, error_line));

                // Add pointer to the exact position
                let spaces = " ".repeat(line.to_string().len() + 3 + col.saturating_sub(1));
                result.push_str(&format!("{}^\n", spaces));
            }

            // Add helpful suggestions
            result.push_str(&suggest_fix(message));

            result
        }
        _ => error.to_string(),
    }
}

/// Provide helpful suggestions based on error message  
fn suggest_fix(message: &str) -> String {
    let suggestions = suggest_fix_for_message(message);
    if !suggestions.is_empty() {
        format!("\nSuggestions:\n{}\n", suggestions.join("\n"))
    } else {
        String::new()
    }
}

/// Get a list of suggestions for an error message
fn suggest_fix_for_message(message: &str) -> Vec<String> {
    let mut suggestions = Vec::new();

    if message.contains("Unexpected token") {
        suggestions.push("• Check for missing brackets [ ] or parentheses ( )".to_string());
        suggestions.push("• Verify function call syntax: FunctionName[arg1, arg2]".to_string());
        suggestions.push("• Make sure list syntax uses braces: {item1, item2}".to_string());
    }

    if message.contains("Expected") && message.contains("bracket") {
        suggestions.push("• Function calls require square brackets: Sin[x] not Sin(x)".to_string());
        suggestions.push("• Lists use curly braces: {1, 2, 3}".to_string());
    }

    if message.contains("parameter") {
        suggestions.push("• Arrow function parameters should be symbols: x -> x^2".to_string());
        suggestions.push("• Multiple parameters: (x, y) -> x + y".to_string());
    }

    if message.contains("Unknown") || message.contains("symbol") {
        suggestions
            .push("• Available functions: Sin, Cos, Tan, Exp, Log, Sqrt, Length, Head, Tail".to_string());
        suggestions.push("• Use the REPL command 'functions' to see all available functions".to_string());
    }

    suggestions
}

/// Unified error type for the Lyra language system
/// 
/// This error type encompasses all possible errors that can occur in the Lyra system,
/// providing proper error chaining and context preservation.
#[derive(Error, Debug)]
pub enum LyraError {
    #[error("Parse error: {message} at position {position}")]
    Parse { message: String, position: usize },

    #[error("Lexer error: {message} at position {position}")]
    Lexer { message: String, position: usize },

    #[error("Runtime error: {message}")]
    Runtime { message: String },

    #[error("Compilation error: {message}")]
    Compilation { message: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Unknown symbol: {symbol}")]
    UnknownSymbol { symbol: String },

    #[error("Type error: expected {expected}, got {actual}")]
    Type { expected: String, actual: String },
    
    #[error("VM error: {0}")]
    Vm(#[from] VmError),
    
    #[error("Foreign object error: {0}")]
    Foreign(#[from] ForeignError),
    
    #[error("Compiler error: {0}")]
    Compiler(#[from] CompilerError),
    
    #[error("{message}: {source}")]
    WithContext { message: String, source: Box<LyraError> },
}

impl LyraError {
    /// Add context to an error
    pub fn with_context(self, context: &str) -> Self {
        LyraError::WithContext {
            message: context.to_string(),
            source: Box::new(self),
        }
    }
    
    /// Check if this is a runtime error
    pub fn is_runtime_error(&self) -> bool {
        matches!(self, LyraError::Runtime { .. } | LyraError::Vm(VmError::Runtime(_)))
    }
    
    /// Check if this is a parse error
    pub fn is_parse_error(&self) -> bool {
        matches!(self, LyraError::Parse { .. } | LyraError::Lexer { .. })
    }
    
    /// Check if this is a type error
    pub fn is_type_error(&self) -> bool {
        matches!(self, LyraError::Type { .. } | LyraError::Vm(VmError::TypeError { .. }))
    }
    
    /// Get recovery suggestions for this error
    pub fn recovery_suggestions(&self) -> Vec<String> {
        match self {
            LyraError::Parse { message, .. } | LyraError::Lexer { message, .. } => {
                suggest_fix_for_message(message)
            }
            LyraError::Type { expected, actual } => {
                vec![
                    format!("Convert {} to {} before using", actual, expected),
                    "Check the function signature and argument types".to_string(),
                ]
            }
            LyraError::UnknownSymbol { symbol } => {
                vec![
                    format!("Define the symbol '{}' before using it", symbol),
                    "Check spelling and case sensitivity".to_string(),
                    "Available functions: Sin, Cos, Tan, Exp, Log, Sqrt, Length, Head, Tail".to_string(),
                ]
            }
            LyraError::Vm(vm_error) => {
                match vm_error {
                    VmError::TypeError { expected, actual } => {
                        vec![
                            format!("Convert {} to {} before using", actual, expected),
                            "Check the function signature and argument types".to_string(),
                        ]
                    }
                    VmError::IndexError { index, length } => {
                        vec![
                            format!("Use an index between 0 and {}", length.saturating_sub(1)),
                            format!("Index {} is out of bounds", index),
                        ]
                    }
                    VmError::DivisionByZero => {
                        vec![
                            "Check that the denominator is not zero".to_string(),
                            "Add a condition to handle zero values".to_string(),
                        ]
                    }
                    _ => vec!["Check the operation and its arguments".to_string()],
                }
            }
            LyraError::Foreign(foreign_error) => {
                match foreign_error {
                    ForeignError::UnknownMethod { type_name, method } => {
                        vec![
                            format!("Available methods for {}: Use Help[object] to see all methods", type_name),
                            format!("Method '{}' does not exist for type '{}'", method, type_name),
                        ]
                    }
                    ForeignError::InvalidArity { method, expected, actual } => {
                        vec![
                            format!("Method '{}' expects {} arguments, got {}", method, expected, actual),
                            "Check the method signature".to_string(),
                        ]
                    }
                    _ => vec!["Check the foreign object operation".to_string()],
                }
            }
            _ => vec![],
        }
    }
}

/// Legacy error type for backward compatibility
#[derive(Error, Debug)]
pub enum Error {
    #[error("Parse error: {message} at position {position}")]
    Parse { message: String, position: usize },

    #[error("Lexer error: {message} at position {position}")]
    Lexer { message: String, position: usize },

    #[error("Runtime error: {message}")]
    Runtime { message: String },

    #[error("Compilation error: {message}")]
    Compilation { message: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Unknown symbol: {symbol}")]
    UnknownSymbol { symbol: String },

    #[error("Type error: expected {expected}, got {actual}")]
    Type { expected: String, actual: String },

    #[error("Security violation: {0}")]
    SecurityViolation(String),
}

impl From<crate::compiler::CompilerError> for Error {
    fn from(err: crate::compiler::CompilerError) -> Self {
        Error::Compilation {
            message: err.to_string(),
        }
    }
}

impl From<Error> for LyraError {
    fn from(err: Error) -> Self {
        match err {
            Error::Parse { message, position } => LyraError::Parse { message, position },
            Error::Lexer { message, position } => LyraError::Lexer { message, position },
            Error::Runtime { message } => LyraError::Runtime { message },
            Error::Compilation { message } => LyraError::Compilation { message },
            Error::Io(io_err) => LyraError::Io(io_err),
            Error::UnknownSymbol { symbol } => LyraError::UnknownSymbol { symbol },
            Error::Type { expected, actual } => LyraError::Type { expected, actual },
            Error::SecurityViolation(msg) => LyraError::Runtime { message: msg },
        }
    }
}
