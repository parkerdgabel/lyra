pub mod ast;
pub mod bytecode;
pub mod compiler;
// TODO: Concurrency module needs additional fixes for match expressions - postponed
// pub mod concurrency;
pub mod error;
pub mod foreign;
pub mod format;
pub mod lexer;
pub mod linker;
pub mod memory;
pub mod modules;
pub mod parser;
// TODO: Tree-shaking has complex dependencies - postponed
// pub mod tree_shaking;
pub mod pattern_matcher;
pub mod pure_function;
pub mod repl;
pub mod rules_engine;
pub mod security;
pub mod stdlib;
pub mod types;
pub mod vm;
pub mod vm_components;
pub mod compiler_components;
pub mod unified_errors;
pub mod common_utils;

pub use error::{Error, Result};
pub use unified_errors::{LyraUnifiedError, LyraResult};
