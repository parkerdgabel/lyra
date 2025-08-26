use lyra_runtime::Evaluator;
use lyra_runtime::eval::set_default_registrar;
use lyra_core::pretty::format_value;
use lyra_core::value::Value;
use lyra_stdlib as stdlib;
use std::sync::Once;

static INIT: Once = Once::new();
fn ensure_default() { INIT.call_once(|| set_default_registrar(stdlib::register_all)); }

fn eval_v(ev: &mut Evaluator, v: lyra_core::value::Value) -> String { format_value(&ev.eval(v)) }

#[test]
fn try_basic_and_default() {
    ensure_default();
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let s1 = eval_v(&mut ev, Value::expr(Value::Symbol("Try".into()), vec![
        Value::expr(Value::Symbol("Plus".into()), vec![Value::Integer(1), Value::Integer(2)])
    ]));
    assert!(s1.contains("\"ok\" -> True") && s1.contains("\"value\" -> 3"));
    let s2 = eval_v(&mut ev, Value::expr(Value::Symbol("Try".into()), vec![
        Value::expr(Value::Symbol("Fail".into()), vec![Value::String("oops".into())])
    ]));
    assert!(s2.contains("\"ok\" -> False") && s2.contains("\"message\" -> \"Failure\""));
    let s3 = eval_v(&mut ev, Value::expr(Value::Symbol("Try".into()), vec![
        Value::expr(Value::Symbol("Fail".into()), vec![Value::String("oops".into())]),
        Value::Integer(42)
    ]));
    assert_eq!(s3, "42");
}

#[test]
fn onfailure_and_catch() {
    ensure_default();
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let s1 = eval_v(&mut ev, Value::expr(Value::Symbol("OnFailure".into()), vec![
        Value::expr(Value::Symbol("Fail".into()), vec![Value::String("oops".into())]),
        Value::String("handled".into())
    ]));
    assert_eq!(s1, "\"handled\"");
    let handlers = Value::assoc(vec![
        ("x", Value::String("hx".into())),
        ("_", Value::String("def".into())),
    ]);
    let s2 = eval_v(&mut ev, Value::expr(Value::Symbol("Catch".into()), vec![
        Value::expr(Value::Symbol("Fail".into()), vec![Value::String("x".into())]),
        handlers
    ]));
    assert_eq!(s2, "\"hx\"");
    let handlers2 = Value::assoc(vec![ ("x", Value::String("hx".into())), ("_", Value::String("def".into())) ]);
    let s3 = eval_v(&mut ev, Value::expr(Value::Symbol("Catch".into()), vec![
        Value::expr(Value::Symbol("Fail".into()), vec![Value::String("y".into())]),
        handlers2
    ]));
    assert_eq!(s3, "\"def\"");
}

#[test]
fn throw_and_finally() {
    ensure_default();
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // Throw builds a Failure shape
    let thrown = Value::expr(Value::Symbol("Throw".into()), vec![
        Value::String("oops".into()),
        Value::assoc(vec![("code", Value::Integer(1))])
    ]);
    let handlers = Value::assoc(vec![("oops", Value::String("thrown".into()))]);
    let s1 = eval_v(&mut ev, Value::expr(Value::Symbol("Catch".into()), vec![thrown, handlers]));
    assert_eq!(s1, "\"thrown\"");
    // Finally runs cleanup and returns original result
    let s2 = eval_v(&mut ev, Value::expr(Value::Symbol("Finally".into()), vec![
        Value::expr(Value::Symbol("Fail".into()), vec![Value::String("z".into())]),
        Value::expr(Value::Symbol("Set".into()), vec![Value::Symbol("touched".into()), Value::Boolean(true)])
    ]));
    assert!(s2.contains("\"message\" -> \"Failure\""));
    let s3 = eval_v(&mut ev, Value::Symbol("touched".into()));
    assert_eq!(s3, "True");
}

#[test]
fn catch_rules_list_and_assert_throws() {
    ensure_default();
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // List of Rule[...] handlers
    let rules = Value::list(vec![
        Value::expr(Value::Symbol("Rule".into()), vec![Value::String("a".into()), Value::String("ha".into())]),
        Value::expr(Value::Symbol("Rule".into()), vec![Value::String("_".into()), Value::String("def".into())]),
    ]);
    let out = eval_v(&mut ev, Value::expr(Value::Symbol("Catch".into()), vec![
        Value::expr(Value::Symbol("Fail".into()), vec![Value::String("a".into())]),
        rules,
    ]));
    assert_eq!(out, "\"ha\"");

    // AssertThrows tests are in testing_basic (feature=testing)
}
