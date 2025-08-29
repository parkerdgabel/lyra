#![cfg(feature = "math")]

use lyra_core::pretty::format_value;
use lyra_parser::Parser;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;

fn eval_one(src: &str) -> String {
    let mut p = Parser::from_source(src);
    let vals = p.parse_all().expect("parse");
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    format_value(&ev.eval(vals.into_iter().last().unwrap()))
}

#[test]
fn permutations_basic() {
    assert_eq!(eval_one("Permutations[{1,2}]"), "{{1, 2}, {2, 1}}");
    assert_eq!(eval_one("Length[Permutations[{1,2,3}]]"), "6");
    assert_eq!(
        eval_one("Permutations[3, 2]"),
        "{{1, 2}, {1, 3}, {2, 1}, {2, 3}, {3, 1}, {3, 2}}"
    );
}

#[test]
fn combinations_basic() {
    assert_eq!(eval_one("Combinations[{1,2,3}, 2]"), "{{1, 2}, {1, 3}, {2, 3}}");
    assert_eq!(eval_one("Length[Combinations[4, 2]]"), "6");
}

#[test]
fn counts() {
    assert_eq!(eval_one("PermutationsCount[5]"), "120");
    assert_eq!(eval_one("PermutationsCount[5,2]"), "20");
    assert_eq!(eval_one("CombinationsCount[5,2]"), "10");
}
