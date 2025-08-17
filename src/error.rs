use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

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