#![cfg(feature = "math")]

use lyra_runtime::Evaluator;
use lyra_core::pretty::format_value;
use lyra_parser::Parser;
use lyra_stdlib as stdlib;

fn eval_one(src: &str) -> String {
    let mut p = Parser::from_source(src);
    let vals = p.parse_all().expect("parse");
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    format_value(&ev.eval(vals.into_iter().last().unwrap()))
}

#[test]
fn inverse_trig_and_angles() {
    assert_eq!(eval_one("ASin[0]"), "0.0");
    assert_eq!(eval_one("ACos[1]"), "0.0");
    assert_eq!(eval_one("ATan[0]"), "0.0");
    // ATan2 sanity via degrees
    assert_eq!(eval_one("Round[ToDegrees[ATan2[1, 0]]]"), "90");
}

#[test]
fn nthroot_and_factorials() {
    assert_eq!(eval_one("NthRoot[27, 3]"), "3");
    assert_eq!(eval_one("Factorial[5]"), "120");
    assert_eq!(eval_one("Binomial[5, 2]"), "10");
}

#[test]
fn median_variance_stddev() {
    assert_eq!(eval_one("Median[{3,1,2}]"), "2");
    assert_eq!(eval_one("Median[{1,2,3,4}]"), "5/2");
    assert_eq!(eval_one("Variance[{1,2,3,4}]"), "1.25");
    // Round[StdDev^2*100] == 125
    assert_eq!(eval_one("Round[StandardDeviation[{1,2,3,4}]^2 * 100]"), "125");
}

