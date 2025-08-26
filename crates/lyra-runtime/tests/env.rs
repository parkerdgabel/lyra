use lyra_core::pretty::format_value;
use lyra_parser::Parser;
use lyra_runtime::{set_default_registrar, Evaluator};
use lyra_stdlib as stdlib;

fn eval_last(src: &str) -> String {
    let mut p = Parser::from_source(src);
    let vals = p.parse_all().expect("parse");
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    format_value(&ev.eval(vals.into_iter().last().unwrap()))
}

#[test]
fn unset_removes_binding() {
    // After Unset, symbol should evaluate to itself (not a bound value)
    let out = eval_last("Set[x, 42]; Unset[x]; x");
    assert_eq!(out, "x");
}
