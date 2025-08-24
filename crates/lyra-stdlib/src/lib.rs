//! Lyra Standard Library registration helpers.

use lyra_runtime::Evaluator;

pub mod math;
pub mod logic;
pub mod list;
pub mod string;
pub mod assoc;
pub mod concurrency {}
pub mod explain {}
pub mod schema {}
pub mod testing {}

pub fn register_all(ev: &mut Evaluator) {
    string::register_string(ev);
    math::register_math(ev);
    list::register_list(ev);
    assoc::register_assoc(ev);
    logic::register_logic(ev);
}

pub fn register_with(ev: &mut Evaluator, groups: &[&str]) {
    for g in groups {
        match *g {
            "string" => string::register_string(ev),
            "math" => math::register_math(ev),
            "list" => list::register_list(ev),
            "assoc" => assoc::register_assoc(ev),
            "logic" => logic::register_logic(ev),
            _ => {}
        }
    }
}
