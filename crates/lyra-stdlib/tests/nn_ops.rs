#![cfg(feature = "nn")]

use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;

#[test]
fn pooling_avg_forward() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);

    let net = ev.eval(Value::expr(
        Value::Symbol("Sequential".into()),
        vec![Value::List(vec![Value::expr(
            Value::Symbol("Pooling".into()),
            vec![
                Value::String("Avg".into()),
                Value::Integer(2),
                Value::Assoc([("Stride".into(), Value::Integer(2))].into_iter().collect()),
            ],
        )])],
    ));

    let x = Value::List(vec![
        Value::Integer(1),
        Value::Integer(2),
        Value::Integer(3),
        Value::Integer(5),
    ]);
    let y = ev.eval(Value::expr(Value::Symbol("Predict".into()), vec![net, x]));
    if let Value::List(xs) = y {
        assert_eq!(xs.len(), 2);
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
        assert!((a - 1.5).abs() < 1e-9);
        assert!((b - 4.0).abs() < 1e-9);
    } else {
        panic!("expected list output")
    }
}

#[test]
fn embedding_returns_dim() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let net = ev.eval(Value::expr(
        Value::Symbol("Sequential".into()),
        vec![Value::List(vec![Value::expr(
            Value::Symbol("Embedding".into()),
            vec![Value::Integer(100), Value::Integer(7)],
        )])],
    ));

    let y = ev.eval(Value::expr(Value::Symbol("Predict".into()), vec![net, Value::Integer(3)]));
    if let Value::List(xs) = y {
        assert_eq!(xs.len(), 7);
    } else {
        panic!("expected list output")
    }
}

#[test]
fn conv1d_softmax_probs() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let net = ev.eval(Value::expr(
        Value::Symbol("Sequential".into()),
        vec![Value::List(vec![
            Value::expr(
                Value::Symbol("Convolution1D".into()),
                vec![Value::Integer(3), Value::Integer(2)],
            ),
            Value::expr(Value::Symbol("Softmax".into()), vec![]),
        ])],
    ));

    let x =
        Value::List(vec![Value::Real(0.1), Value::Real(-0.2), Value::Real(0.3), Value::Real(0.0)]);
    let y = ev.eval(Value::expr(Value::Symbol("Predict".into()), vec![net, x]));
    if let Value::List(xs) = y {
        assert!(!xs.is_empty());
        let mut sum = 0.0;
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
                _ => panic!("expected numeric"),
            }
        }
        assert!((sum - 1.0).abs() < 1e-6);
    } else {
        panic!("expected list output")
    }
}
