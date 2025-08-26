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
fn trig_and_constants() {
    assert_eq!(eval_one("Sin[Pi/2] + Cos[0]"), "2.0");
    assert_eq!(eval_one("Tan[0]"), "0.0");
}

#[test]
fn rounding_and_division() {
    assert_eq!(eval_one("Equal[Floor[2.8], 2]"), "True");
    assert_eq!(eval_one("Equal[Ceiling[-2.2], -2]"), "True");
    assert_eq!(eval_one("Equal[Round[2.5], 2]"), "True");
    assert_eq!(eval_one("Equal[Trunc[-2.9], -2]"), "True");
    assert_eq!(eval_one("Equal[Mod[-7, 3], 2]"), "True");
    assert_eq!(eval_one("Equal[Quotient[-7, 3], -3]"), "True");
    assert_eq!(eval_one("Equal[Remainder[-7, 3], 2]"), "True");
    assert_eq!(eval_one("DivMod[7, 3]"), "{2, 1}");
}

#[test]
fn exp_log_sqrt() {
    assert_eq!(eval_one("Sqrt[9]"), "3");
    assert_eq!(eval_one("Log[E]"), "1.0");
    assert_eq!(eval_one("Log[8, 2]"), "3.0");
    assert_eq!(eval_one("Exp[0]"), "1.0");
}

#[test]
fn totals_and_means() {
    assert_eq!(eval_one("Equal[Total[{1,2,3,4}], 10]"), "True");
    assert_eq!(eval_one("Equal[Mean[{1,2,3,4}], 5/2]"), "True");
}

#[test]
fn gcd_lcm_clip_signum() {
    assert_eq!(eval_one("Equal[GCD[12, 18], 6]"), "True");
    assert_eq!(eval_one("Equal[LCM[6, 15], 30]"), "True");
    assert_eq!(eval_one("Clip[-1, {0, 10}]"), "0.0");
    assert_eq!(eval_one("Clip[11, {0, 10}]"), "10.0");
    assert_eq!(eval_one("Equal[Signum[-3], -1]"), "True");
    assert_eq!(eval_one("Equal[Signum[0], 0]"), "True");
    assert_eq!(eval_one("Equal[Signum[3.5], 1]"), "True");
}
