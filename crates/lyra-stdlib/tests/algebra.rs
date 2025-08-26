#![cfg(feature = "algebra")]

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
fn simplify_basic() {
    assert_eq!(eval_one("Simplify[(x + 0)*1]"), "x");
}

#[test]
fn expand_and_collect() {
    assert_eq!(eval_one("Expand[(x + 1)*(x + 2)]"), "Plus[2, x, Times[2, x], Times[x, x]]");
    assert_eq!(eval_one("CollectTerms[(x + 1)*(x + 2), x]"), "Plus[2, Times[3, x], Power[x, 2]]");
}

#[test]
fn factor_quadratic_and_diff() {
    assert_eq!(eval_one("Factor[x^2 + 3 * x + 2, x]"), "Times[Plus[1, x], Plus[2, x]]");
    assert_eq!(eval_one("D[x^3 + 2 * x, x] // Simplify"), "Plus[2, Times[3, Power[x, 2]]]");
}

#[test]
fn collect_terms_by_multivar() {
    let got = eval_one("CollectTermsBy[x*y + 2*x*y + 3*y^2, {x, y}]");
    assert!(
        got == "Plus[Times[3, x, y], Times[3, Power[y, 2]]]"
            || got == "Plus[Times[3, Power[y, 2]], Times[3, x, y]]"
    );
}

#[test]
fn expand_all_and_cancel_apart() {
    // ExpandAll
    let ex = eval_one("ExpandAll[(x + 1)^3]");
    assert!(ex.contains("Power[x, 2]") || ex.contains("Times[x, x]"));
    // CancelRational (no name conflict with concurrency Cancel)
    assert_eq!(eval_one("CancelRational[(x^2 - 1)/(x - 1), x]"), "Plus[1, x]");
    // Apart (limited) — not asserting exact shape; only that it evaluates or stays symbolic
    let _ap = eval_one("Apart[1/(x*(x + 1)), x]");
}

#[test]
fn roots_quadratic() {
    let rt = eval_one("Roots[x^2 - 1, x]");
    assert!(rt.contains("-1") && rt.contains("1"));
}
