use lyra_core::value::Value;
use lyra_runtime::{eval::Evaluator, set_default_registrar};
use lyra_stdlib as stdlib;

fn sym(s: &str) -> Value { Value::Symbol(s.into()) }

#[test]
fn rational_arithmetic_plus_times() {
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    // Plus[1/2, 1/3] = 5/6
    let a = Value::Rational { num: 1, den: 2 };
    let b = Value::Rational { num: 1, den: 3 };
    let expr = Value::Expr { head: Box::new(sym("Plus")), args: vec![a, b] };
    let out = ev.eval(expr);
    assert_eq!(out, Value::Rational { num: 5, den: 6 });

    // Times[2, 1/3] = 2/3
    let t = Value::Expr { head: Box::new(sym("Times")), args: vec![ Value::Integer(2), Value::Rational { num: 1, den: 3 } ] };
    let out2 = ev.eval(t);
    assert_eq!(out2, Value::Rational { num: 2, den: 3 });
}

#[test]
fn complex_addition() {
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    let z1 = Value::Complex { re: Box::new(Value::Integer(1)), im: Box::new(Value::Integer(2)) };
    let z2 = Value::Complex { re: Box::new(Value::Integer(3)), im: Box::new(Value::Integer(-1)) };
    let expr = Value::Expr { head: Box::new(sym("Plus")), args: vec![z1, z2] };
    let out = ev.eval(expr);
    assert_eq!(out, Value::Complex { re: Box::new(Value::Integer(4)), im: Box::new(Value::Integer(1)) });
}

