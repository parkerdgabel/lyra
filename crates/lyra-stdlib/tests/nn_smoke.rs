#![cfg(feature = "nn")]

use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;

// Targeted test 1: construct a NetChain and check properties
#[test]
fn net_chain_properties() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);

    let net = ev.eval(Value::expr(
        Value::Symbol("Sequential".into()),
        vec![Value::List(vec![Value::Symbol("Dense".into())])],
    ));

    let props = ev.eval(Value::expr(
        Value::Symbol("Property".into()),
        vec![net.clone(), Value::String("Properties".into())],
    ));
    if let Value::List(xs) = props {
        assert!(!xs.is_empty());
    } else {
        panic!("expected list of properties");
    }

    let kind = ev.eval(Value::expr(
        Value::Symbol("Property".into()),
        vec![net.clone(), Value::String("Kind".into())],
    ));
    assert!(matches!(kind, Value::String(_)));

    let summary = ev.eval(Value::expr(Value::Symbol("Summary".into()), vec![net]));
    if let Value::Assoc(m) = summary {
        assert!(m.contains_key("LayerCount"));
    } else {
        panic!("expected assoc summary");
    }
}

// Targeted test 2: NetApply returns identity for MVP
#[test]
fn net_apply_identity() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);

    let net = ev.eval(Value::expr(
        Value::Symbol("Sequential".into()),
        vec![Value::List(vec![Value::expr(Value::Symbol("Relu".into()), vec![])])],
    ));

    // Direct NetApply call (no training)
    let out = ev.eval(Value::expr(Value::Symbol("Predict".into()), vec![net, Value::Integer(42)]));
    assert!(matches!(out, Value::Integer(42)));
}
