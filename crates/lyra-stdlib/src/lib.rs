//! Lyra Standard Library registration helpers.

use lyra_runtime::Evaluator;

pub mod math {}
pub mod logic {}
pub mod list {}
pub mod string {}
pub mod assoc {}
pub mod concurrency {}
pub mod explain {}
pub mod schema {}
pub mod testing {}

pub fn register_all(_ev: &mut Evaluator) {
    // Will register all stdlib domains.
}

pub fn register_with(_ev: &mut Evaluator, _groups: &[&str]) {
    // Will register selected domains.
}
