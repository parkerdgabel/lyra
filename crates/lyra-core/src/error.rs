use thiserror::Error;

pub type Result<T> = std::result::Result<T, LyraError>;

#[derive(Error, Debug)]
pub enum LyraError {
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Type error: {0}")]
    Type(String),
    #[error("Runtime error: {0}")]
    Runtime(String),
}
