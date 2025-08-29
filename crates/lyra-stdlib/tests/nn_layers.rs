#![cfg(feature = "nn")]

use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;

fn is_layer(v: &Value) -> bool {
    if let Value::Assoc(m) = v {
        if let Some(Value::String(t)) = m.get("__type") {
            return t == "Layer";
        }
    }
    false
}

#[test]
fn construct_basic_layers() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);

    let lin = ev.eval(Value::expr(Value::Symbol("Dense".into()), vec![Value::Integer(8)]));
    assert!(is_layer(&lin));

    let relu = ev.eval(Value::expr(Value::Symbol("Relu".into()), vec![]));
    assert!(is_layer(&relu));

    let drop = ev.eval(Value::expr(Value::Symbol("Dropout".into()), vec![Value::Real(0.25)]));
    assert!(is_layer(&drop));

    let flat = ev.eval(Value::expr(Value::Symbol("Flatten".into()), vec![]));
    assert!(is_layer(&flat));

    let sm = ev.eval(Value::expr(Value::Symbol("Softmax".into()), vec![]));
    assert!(is_layer(&sm));
}

#[test]
fn net_chain_with_layers() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let net = ev.eval(Value::expr(
        Value::Symbol("Sequential".into()),
        vec![Value::List(vec![
            Value::expr(Value::Symbol("Dense".into()), vec![Value::Integer(16)]),
            Value::expr(Value::Symbol("Relu".into()), vec![]),
            Value::expr(Value::Symbol("Dropout".into()), vec![Value::Real(0.1)]),
        ])],
    ));

    // Ensure NetProperty["Layers"] returns a list
    let layers = ev.eval(Value::expr(
        Value::Symbol("Property".into()),
        vec![net.clone(), Value::String("Layers".into())],
    ));
    if let Value::List(xs) = layers {
        assert!(!xs.is_empty());
    } else {
        panic!("expected list of layers");
    }

    // LayerSummaries include key params
    let ls = ev.eval(Value::expr(
        Value::Symbol("Property".into()),
        vec![net.clone(), Value::String("LayerSummaries".into())],
    ));
    if let Value::List(xs) = ls {
        assert_eq!(xs.len(), 3);
        if let Value::Assoc(m0) = &xs[0] {
            assert!(m0.contains_key("Output"));
            assert!(m0.contains_key("Bias"));
        } else {
            panic!("summary assoc expected")
        }
        if let Value::Assoc(m1) = &xs[1] {
            assert!(m1.contains_key("Type"));
        } else {
            panic!("summary assoc expected")
        }
        if let Value::Assoc(m2) = &xs[2] {
            let _ = m2;
        }
    } else {
        panic!("expected list of summaries");
    }
}

#[test]
fn linear_bias_false_zero_input() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let net = ev.eval(Value::expr(
        Value::Symbol("Sequential".into()),
        vec![Value::List(vec![Value::expr(
            Value::Symbol("Dense".into()),
            vec![
                Value::Integer(3),
                Value::Assoc([("Bias".into(), Value::Boolean(false))].into_iter().collect()),
            ],
        )])],
    ));
    // zero input => zero output when no bias
    let x = Value::List(vec![
        Value::Integer(0),
        Value::Integer(0),
        Value::Integer(0),
        Value::Integer(0),
    ]);
    let y = ev.eval(Value::expr(Value::Symbol("Predict".into()), vec![net, x]));
    if let Value::List(xs) = y {
        for v in xs {
            match v {
                Value::Real(r) => assert!(r.abs() < 1e-12),
                Value::Integer(n) => assert_eq!(n, 0),
                _ => panic!("expected num"),
            }
        }
    } else {
        panic!("expected list")
    }
}

#[test]
fn activation_unknown_defaults_relu() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let net = ev.eval(Value::expr(
        Value::Symbol("Sequential".into()),
        vec![Value::List(vec![Value::expr(
            Value::Symbol("__ActivationLayer".into()),
            vec![Value::String("Bogus".into())],
        )])],
    ));
    let x = Value::List(vec![Value::Real(-2.0), Value::Real(3.0)]);
    let y = ev.eval(Value::expr(Value::Symbol("Predict".into()), vec![net, x]));
    if let Value::List(xs) = y {
        let a = match &xs[0] {
            Value::Real(r) => *r,
            Value::Integer(n) => *n as f64,
            _ => 0.0,
        };
        let b = match &xs[1] {
            Value::Real(r) => *r,
            Value::Integer(n) => *n as f64,
            _ => 0.0,
        };
        assert!(a >= 0.0);
        assert!(b >= 3.0);
    } else {
        panic!("expected list")
    }
}

#[test]
fn net_chain_linear_softmax_forward() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let net = ev.eval(Value::expr(
        Value::Symbol("Sequential".into()),
        vec![Value::List(vec![
            Value::expr(Value::Symbol("Dense".into()), vec![Value::Integer(2)]),
            Value::expr(Value::Symbol("Softmax".into()), vec![]),
        ])],
    ));

    let x = Value::List(vec![Value::Real(1.0), Value::Real(-1.0), Value::Real(0.5)]);
    let y = ev.eval(Value::expr(Value::Symbol("Predict".into()), vec![net, x]));
    if let Value::List(xs) = y {
        assert_eq!(xs.len(), 2);
        let mut sum = 0.0f64;
        for v in xs {
            match v {
                Value::Real(r) => {
                    assert!(r >= 0.0 && r <= 1.0);
                    sum += r;
                }
                Value::Integer(n) => {
                    let r = n as f64;
                    assert!(r >= 0.0 && r <= 1.0);
                    sum += r;
                }
                _ => panic!("expected numeric prob"),
            }
        }
        assert!((sum - 1.0).abs() < 1e-6);
    } else {
        panic!("expected list output")
    }
}
