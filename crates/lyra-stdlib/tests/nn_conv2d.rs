#![cfg(feature = "nn")]

use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;

#[test]
fn conv2d_small_forward() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);

    // Convolution2D with 1 out channel, 1 in channel, 2x2 kernel of ones, no padding/stride
    let layer = Value::expr(
        Value::Symbol("Convolution2D".into()),
        vec![Value::Assoc([
            ("Output".into(), Value::Integer(1)),
            ("KernelSize".into(), Value::List(vec![Value::Integer(2), Value::Integer(2)])),
            ("Stride".into(), Value::Integer(1)),
            ("Padding".into(), Value::Integer(1)),
            ("InputChannels".into(), Value::Integer(1)),
            ("Height".into(), Value::Integer(2)),
            ("Width".into(), Value::Integer(2)),
            // W[out][in][kh][kw]
            (
                "W".into(),
                Value::List(vec![Value::List(vec![Value::List(vec![
                    Value::List(vec![Value::Integer(1), Value::Integer(1)]),
                    Value::List(vec![Value::Integer(1), Value::Integer(1)]),
                ])])]),
            ),
            ("b".into(), Value::List(vec![Value::Integer(0)])),
        ]
        .into_iter()
        .collect())],
    );

    let net = ev.eval(Value::expr(
        Value::Symbol("Sequential".into()),
        vec![Value::List(vec![layer])],
    ));

    // Input: 1x2x2 flattened
    let x = Value::List(vec![
        Value::Integer(1),
        Value::Integer(2),
        Value::Integer(3),
        Value::Integer(4),
    ]);

    let y = ev.eval(Value::expr(Value::Symbol("Predict".into()), vec![net, x]));
    eprintln!("conv2d output: {}", lyra_core::pretty::format_value(&y));
    match y {
        Value::List(xs) => {
            assert_eq!(xs.len(), 9);
            let v = match &xs[4] { Value::Real(r) => *r, Value::Integer(n) => *n as f64, _ => 0.0 };
            assert!((v - 10.0).abs() < 1e-9);
        }
        Value::PackedArray { shape, data } => {
            assert_eq!(shape.len(), 3);
            assert_eq!(shape[1] * shape[2], 9);
            assert!((data[4] - 10.0).abs() < 1e-9);
        }
        _ => panic!("expected tensor or list output"),
    }
}

#[test]
fn pool2d_small_forward() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);

    let layer = Value::expr(
        Value::Symbol("Pooling2D".into()),
        vec![
            Value::String("Max".into()),
            Value::Integer(2),
            Value::Assoc([
                ("InputChannels".into(), Value::Integer(1)),
                ("Height".into(), Value::Integer(2)),
                ("Width".into(), Value::Integer(2)),
            ]
            .into_iter()
            .collect()),
        ],
    );
    let net = ev.eval(Value::expr(
        Value::Symbol("Sequential".into()),
        vec![Value::List(vec![layer])],
    ));
    let x = Value::List(vec![
        Value::Integer(1),
        Value::Integer(2),
        Value::Integer(3),
        Value::Integer(4),
    ]);
    let y = ev.eval(Value::expr(Value::Symbol("Predict".into()), vec![net, x]));
    eprintln!("pool2d output: {}", lyra_core::pretty::format_value(&y));
    match y {
        Value::List(xs) => {
            assert_eq!(xs.len(), 1);
            let v = match &xs[0] { Value::Real(r) => *r, Value::Integer(n) => *n as f64, _ => 0.0 };
            assert!((v - 4.0).abs() < 1e-9);
        }
        Value::PackedArray { shape: _, data } => {
            assert_eq!(data.len(), 1);
            assert!((data[0] - 4.0).abs() < 1e-9);
        }
        _ => panic!("expected tensor or list output"),
    }
}

#[test]
fn pool2d_avg_forward() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);

    let layer = Value::expr(
        Value::Symbol("Pooling2D".into()),
        vec![
            Value::String("Avg".into()),
            Value::Integer(2),
            Value::Assoc([
                ("InputChannels".into(), Value::Integer(1)),
                ("Height".into(), Value::Integer(2)),
                ("Width".into(), Value::Integer(2)),
            ]
            .into_iter()
            .collect()),
        ],
    );
    let net = ev.eval(Value::expr(
        Value::Symbol("Sequential".into()),
        vec![Value::List(vec![layer])],
    ));
    let x = Value::List(vec![
        Value::Integer(1),
        Value::Integer(2),
        Value::Integer(3),
        Value::Integer(4),
    ]);
    let y = ev.eval(Value::expr(Value::Symbol("Predict".into()), vec![net, x]));
    match y {
        Value::List(xs) => {
            assert_eq!(xs.len(), 1);
            let v = match &xs[0] { Value::Real(r) => *r, Value::Integer(n) => *n as f64, _ => 0.0 };
            assert!((v - 2.5).abs() < 1e-9);
        }
        Value::PackedArray { shape: _, data } => {
            assert_eq!(data.len(), 1);
            assert!((data[0] - 2.5).abs() < 1e-9);
        }
        _ => panic!("expected tensor or list output"),
    }
}
