pub mod ast;
pub mod bytecode;
pub mod compiler;
// TODO: Concurrency module needs additional fixes for match expressions - postponed
// pub mod concurrency;
pub mod error;
pub mod foreign;
pub mod format;
pub mod io;
pub mod lexer;
pub mod linker;
pub mod memory;
pub mod modules;
pub mod parser;
// TODO: Tree-shaking has complex dependencies - postponed
// pub mod tree_shaking;
pub mod pattern_matcher;
pub mod repl;
pub mod rules_engine;
// TODO: Serialization module has compilation issues - temporarily disabled
// pub mod serialization;
pub mod stdlib;
// TODO: Types module has compilation issues - temporarily disabled
// pub mod types;
pub mod vm;

pub use error::{Error, Result};
