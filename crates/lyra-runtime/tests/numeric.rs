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

#[test]
fn bigreal_arithmetic() {
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    let a = Value::BigReal("1.5".into());
    let b = Value::Integer(2);
    let expr = Value::Expr { head: Box::new(sym("Plus")), args: vec![a, b] };
    let out = ev.eval(expr);
    assert_eq!(out, Value::BigReal("3.5".into()));
    let c = Value::BigReal("2.5".into());
    let d = Value::BigReal("2".into());
    let expr2 = Value::Expr { head: Box::new(sym("Times")), args: vec![c, d] };
    let out2 = ev.eval(expr2);
    assert_eq!(out2, Value::BigReal("5".into()));
}

#[test]
fn rational_minus_divide_power() {
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    // Minus[1/2, 1/3] = 1/6
    let expr = Value::Expr { head: Box::new(sym("Minus")), args: vec![ Value::Rational { num: 1, den: 2 }, Value::Rational { num: 1, den: 3 } ] };
    let out = ev.eval(expr);
    assert_eq!(out, Value::Rational { num: 1, den: 6 });
    // Divide[1/2, 1/3] = 3/2
    let expr2 = Value::Expr { head: Box::new(sym("Divide")), args: vec![ Value::Rational { num: 1, den: 2 }, Value::Rational { num: 1, den: 3 } ] };
    let out2 = ev.eval(expr2);
    assert_eq!(out2, Value::Rational { num: 3, den: 2 });
    // Power[1/2, 2] = 1/4
    let expr3 = Value::Expr { head: Box::new(sym("Power")), args: vec![ Value::Rational { num: 1, den: 2 }, Value::Integer(2) ] };
    let out3 = ev.eval(expr3);
    assert_eq!(out3, Value::Rational { num: 1, den: 4 });
}

#[test]
fn complex_minus_and_scalar_division() {
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    // (1+2i) - (3-1i) = -2+3i
    let z1 = Value::Complex { re: Box::new(Value::Integer(1)), im: Box::new(Value::Integer(2)) };
    let z2 = Value::Complex { re: Box::new(Value::Integer(3)), im: Box::new(Value::Integer(-1)) };
    let expr = Value::Expr { head: Box::new(sym("Minus")), args: vec![z1, z2] };
    let out = ev.eval(expr);
    assert_eq!(out, Value::Complex { re: Box::new(Value::Integer(-2)), im: Box::new(Value::Integer(3)) });
    // (1+2i)/2 = 1/2 + 1i
    let z3 = Value::Complex { re: Box::new(Value::Integer(1)), im: Box::new(Value::Integer(2)) };
    let expr2 = Value::Expr { head: Box::new(sym("Divide")), args: vec![ z3, Value::Integer(2) ] };
    let out2 = ev.eval(expr2);
    assert_eq!(out2, Value::Complex { re: Box::new(Value::Rational { num: 1, den: 2 }), im: Box::new(Value::Integer(1)) });
}

#[test]
fn complex_division_and_abs() {
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    // (1+2i)/(3-1i) = (1/10) + (7/10)i
    let z1 = Value::Complex { re: Box::new(Value::Integer(1)), im: Box::new(Value::Integer(2)) };
    let z2 = Value::Complex { re: Box::new(Value::Integer(3)), im: Box::new(Value::Integer(-1)) };
    let expr = Value::Expr { head: Box::new(sym("Divide")), args: vec![ z1, z2 ] };
    let out = ev.eval(expr);
    assert_eq!(out, Value::Complex { re: Box::new(Value::Rational { num: 1, den: 10 }), im: Box::new(Value::Rational { num: 7, den: 10 }) });
    // Abs[3+4i] = 5
    let z3 = Value::Complex { re: Box::new(Value::Integer(3)), im: Box::new(Value::Integer(4)) };
    let aexpr = Value::Expr { head: Box::new(sym("Abs")), args: vec![ z3 ] };
    let aout = ev.eval(aexpr);
    assert_eq!(aout, Value::Integer(5));
}
