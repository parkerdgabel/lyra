//! Foreign data types for Lyra standard library
//!
//! This module contains thread-safe (Send + Sync) implementations of data structures
//! that can be used as Foreign objects in the Lyra VM. These provide efficient
//! interop between the VM and standard library functions.

pub mod dataset;
pub mod schema;
pub mod series;
pub mod table;
pub mod tensor;

pub use dataset::ForeignDataset;
pub use schema::{ForeignSchema, SchemaType};
pub use series::{ForeignSeries, SeriesType};
pub use table::ForeignTable;
pub use tensor::ForeignTensor;