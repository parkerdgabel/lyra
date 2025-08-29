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
fn prime_factors_and_phi_mu() {
    assert_eq!(eval_one("PrimeFactors[84]"), "{2, 2, 3, 7}");
    assert_eq!(eval_one("EulerPhi[36]"), "12");
    assert_eq!(eval_one("MobiusMu[36]"), "0");
    assert_eq!(eval_one("MobiusMu[35]"), "1");
    assert_eq!(eval_one("MobiusMu[30]"), "-1");
}

#[test]
fn powermod_basic_and_inverse() {
    assert_eq!(eval_one("PowerMod[2, 10, 1000]"), "24");
    assert_eq!(eval_one("PowerMod[3, -1, 11]"), "4");
}

