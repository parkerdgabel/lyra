pub mod ast;
pub mod bytecode;
pub mod compiler;
// TODO: Concurrency has thread safety issues - needs architectural review
// pub mod concurrency;
pub mod error;
pub mod foreign;
pub mod format;
pub mod lexer;
pub mod linker;
pub mod memory;
// TODO: Modules system has integration issues - postponed  
// pub mod modules;
pub mod parser;
// TODO: Tree-shaking has complex dependencies - postponed
// pub mod tree_shaking;
pub mod pattern_matcher;
pub mod repl;
pub mod rules_engine;
// TODO: Serialization has complex trait dependencies - postponed
// pub mod serialization;
pub mod stdlib;
// TODO: Types module has integration issues - postponed 
// pub mod types;
pub mod vm;

pub use error::{Error, Result};
