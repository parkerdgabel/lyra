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
fn gcd_inverse_crt() {
    assert_eq!(eval_one("ExtendedGCD[240, 46]"), "{2, -9, 47}");
    assert_eq!(eval_one("ModInverse[3, 11]"), "4");
    assert_eq!(eval_one("ChineseRemainder[{2,3,2}, {3,5,7}]"), "23");
}

#[test]
fn primality_and_next() {
    assert_eq!(eval_one("PrimeQ[17]"), "True");
    assert_eq!(eval_one("PrimeQ[1]"), "False");
    assert_eq!(eval_one("NextPrime[10]"), "11");
}

#[test]
fn divides_coprime_factor() {
    assert_eq!(eval_one("DividesQ[3, 12]"), "True");
    assert_eq!(eval_one("DividesQ[5, 12]"), "False");
    assert_eq!(eval_one("CoprimeQ[12, 35]"), "True");
    assert_eq!(eval_one("CoprimeQ[12, 18]"), "False");
    assert_eq!(eval_one("FactorInteger[84]"), "{{2, 2}, {3, 1}, {7, 1}}");
}

