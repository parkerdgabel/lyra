#![cfg(feature = "nn")]

use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;

#[test]
fn separable_init_and_forward() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // Minimal shape with 1x1 kernel
    let net = ev.eval(Value::expr(
        Value::Symbol("Sequential".into()),
        vec![Value::List(vec![Value::expr(
            Value::Symbol("SeparableConv2D".into()),
            vec![Value::Assoc([
                ("Output".into(), Value::Integer(1)),
                ("KernelSize".into(), Value::Integer(1)),
                ("Stride".into(), Value::Integer(1)),
                ("Padding".into(), Value::Integer(0)),
                ("InputChannels".into(), Value::Integer(1)),
                ("Height".into(), Value::Integer(1)),
                ("Width".into(), Value::Integer(1)),
            ].into_iter().collect())],
        )])],
    ));

    let net2 = ev.eval(Value::expr(Value::Symbol("Initialize".into()), vec![net]));
    // Predict with a 1x1 tensor; expect a 1x1x1 tensor back
    let y = ev.eval(Value::expr(
        Value::Symbol("Predict".into()),
        vec![net2, Value::PackedArray { shape: vec![1,1,1], data: vec![2.0] }],
    ));
    match y {
        Value::PackedArray { shape, data } => {
            assert_eq!(shape, vec![1,1,1]);
            assert_eq!(data.len(), 1);
            assert!(data[0].is_finite());
        }
        _ => panic!("expected Tensor output"),
    }
}

#[test]
fn residual_wrap_activation() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let net = ev.eval(Value::expr(
        Value::Symbol("Sequential".into()),
        vec![Value::List(vec![Value::expr(
            Value::Symbol("Residual".into()),
            vec![Value::List(vec![Value::expr(Value::Symbol("Relu".into()), vec![])])],
        )])],
    ));
    let x = Value::List(vec![Value::Integer(2)]);
    let y = ev.eval(Value::expr(Value::Symbol("Predict".into()), vec![net, x]));
    match y {
        Value::List(xs) => {
            assert_eq!(xs.len(), 1);
            // y = Relu(2) + 2 = 4
            let v = match xs[0] { Value::Integer(n) => n as f64, Value::Real(r) => r, _ => 0.0 };
            assert!((v - 4.0).abs() < 1e-9);
        }
        _ => panic!("expected list output"),
    }
}

