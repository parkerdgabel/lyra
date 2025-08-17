//! Foreign data types for Lyra standard library
//!
//! This module contains thread-safe (Send + Sync) implementations of data structures
//! that can be used as Foreign objects in the Lyra VM. These provide efficient
//! interop between the VM and standard library functions.

pub mod table;

pub use table::{ForeignTable, ForeignSeries};