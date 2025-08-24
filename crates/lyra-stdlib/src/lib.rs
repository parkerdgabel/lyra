//! Lyra Standard Library registration helpers.

use lyra_runtime::Evaluator;

#[cfg(feature = "math")] pub mod math;
#[cfg(feature = "logic")] pub mod logic;
#[cfg(feature = "tools")] pub mod tools;
#[cfg(feature = "list")] pub mod list;
#[cfg(feature = "string")] pub mod string;
#[cfg(feature = "assoc")] pub mod assoc;
#[cfg(feature = "concurrency")] pub mod concurrency;
#[cfg(feature = "schema")] pub mod schema;
#[cfg(feature = "explain")] pub mod explain;
#[cfg(feature = "testing")] pub mod testing;

pub fn register_all(ev: &mut Evaluator) {
    // Core forms from the runtime (assignment, replacement, threading)
    #[cfg(feature = "core")] lyra_runtime::eval::register_core(ev);
    // Introspection helpers for tool discovery
    lyra_runtime::eval::register_introspection(ev);
    #[cfg(feature = "string")] string::register_string(ev);
    #[cfg(feature = "math")] math::register_math(ev);
    #[cfg(feature = "list")] list::register_list(ev);
    #[cfg(feature = "tools")] tools::register_tools(ev);
    #[cfg(feature = "assoc")] assoc::register_assoc(ev);
    #[cfg(feature = "logic")] logic::register_logic(ev);
    #[cfg(feature = "concurrency")] concurrency::register_concurrency(ev);
    #[cfg(feature = "schema")] schema::register_schema(ev);
    #[cfg(feature = "explain")] explain::register_explain(ev);
    #[cfg(feature = "testing")] testing::register_testing(ev);
}

pub fn register_with(ev: &mut Evaluator, groups: &[&str]) {
    for g in groups {
        match *g {
            "string" => { #[cfg(feature = "string")] string::register_string(ev) }
            "math" => { #[cfg(feature = "math")] math::register_math(ev) }
            "list" => { #[cfg(feature = "list")] list::register_list(ev) }
            "tools" => { #[cfg(feature = "tools")] tools::register_tools(ev) }
            "assoc" => { #[cfg(feature = "assoc")] assoc::register_assoc(ev) }
            "logic" => { #[cfg(feature = "logic")] logic::register_logic(ev) }
            "concurrency" => { #[cfg(feature = "concurrency")] concurrency::register_concurrency(ev) }
            "schema" => { #[cfg(feature = "schema")] schema::register_schema(ev) }
            "explain" => { #[cfg(feature = "explain")] explain::register_explain(ev) }
            _ => {}
        }
    }
}
