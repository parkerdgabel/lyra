use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

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
    let mut suggestions = Vec::new();

    if message.contains("Unexpected token") {
        suggestions.push("• Check for missing brackets [ ] or parentheses ( )");
        suggestions.push("• Verify function call syntax: FunctionName[arg1, arg2]");
        suggestions.push("• Make sure list syntax uses braces: {item1, item2}");
    }

    if message.contains("Expected") && message.contains("bracket") {
        suggestions.push("• Function calls require square brackets: Sin[x] not Sin(x)");
        suggestions.push("• Lists use curly braces: {1, 2, 3}");
    }

    if message.contains("parameter") {
        suggestions.push("• Arrow function parameters should be symbols: x -> x^2");
        suggestions.push("• Multiple parameters: (x, y) -> x + y");
    }

    if message.contains("Unknown") || message.contains("symbol") {
        suggestions
            .push("• Available functions: Sin, Cos, Tan, Exp, Log, Sqrt, Length, Head, Tail");
        suggestions.push("• Use the REPL command 'functions' to see all available functions");
    }

    if !suggestions.is_empty() {
        format!("\nSuggestions:\n{}\n", suggestions.join("\n"))
    } else {
        String::new()
    }
}

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
}

impl From<crate::compiler::CompilerError> for Error {
    fn from(err: crate::compiler::CompilerError) -> Self {
        Error::Compilation {
            message: err.to_string(),
        }
    }
}
