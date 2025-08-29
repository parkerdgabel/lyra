#![cfg(feature = "nn")]

use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;

#[test]
fn upsample2d_nearest_forward() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);

    // Upsample by 2 using Nearest, with known input dims 1x2x2
    let layer = Value::expr(
        Value::Symbol("Upsample2D".into()),
        vec![Value::Assoc([
            ("Scale".into(), Value::Integer(2)),
            ("Mode".into(), Value::String("Nearest".into())),
            ("InputChannels".into(), Value::Integer(1)),
            ("Height".into(), Value::Integer(2)),
            ("Width".into(), Value::Integer(2)),
        ].into_iter().collect())]
    );
    let net = ev.eval(Value::expr(Value::Symbol("Sequential".into()), vec![Value::List(vec![layer])]));

    // Input: 1x2x2 flattened: [[1,2],[3,4]]
    let x = Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3), Value::Integer(4)]);
    let y = ev.eval(Value::expr(Value::Symbol("Predict".into()), vec![net, x]));

    match y {
        Value::PackedArray { shape, data } => {
            assert_eq!(shape, vec![1,4,4]);
            // Nearest upsample repeats rows/cols
            assert!((data[0] - 1.0).abs() < 1e-9);
            assert!((data[1] - 1.0).abs() < 1e-9);
            assert!((data[2] - 2.0).abs() < 1e-9);
            assert!((data[3] - 2.0).abs() < 1e-9);
            assert!((data[4] - 1.0).abs() < 1e-9);
            assert!((data[5] - 1.0).abs() < 1e-9);
            assert!((data[6] - 2.0).abs() < 1e-9);
            assert!((data[7] - 2.0).abs() < 1e-9);
        }
        Value::List(xs) => {
            assert_eq!(xs.len(), 16);
        }
        other => panic!("unexpected: {}", lyra_core::pretty::format_value(&other)),
    }
}

#[test]
fn upsample2d_bilinear_forward() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);

    // Bilinear upsample by 2, input dims 1x2x2 with values [[1,2],[3,4]]
    let layer = Value::expr(
        Value::Symbol("Upsample2D".into()),
        vec![Value::Assoc([
            ("Scale".into(), Value::Integer(2)),
            ("Mode".into(), Value::String("Bilinear".into())),
            ("InputChannels".into(), Value::Integer(1)),
            ("Height".into(), Value::Integer(2)),
            ("Width".into(), Value::Integer(2)),
        ].into_iter().collect())]
    );
    let net = ev.eval(Value::expr(Value::Symbol("Sequential".into()), vec![Value::List(vec![layer])]));

    let x = Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3), Value::Integer(4)]);
    let y = ev.eval(Value::expr(Value::Symbol("Predict".into()), vec![net, x]));
    let (shape, data) = match y {
        Value::PackedArray { shape, data } => (shape, data),
        Value::List(xs) => (vec![1,4,4], xs.into_iter().map(|v| match v { Value::Integer(n)=>n as f64, Value::Real(r)=>r, _=>0.0 }).collect()),
        other => panic!("unexpected: {}", lyra_core::pretty::format_value(&other)),
    };
    assert_eq!(shape, vec![1,4,4]);
    // Spot-check a few values
    // (0,1) ~ 1.5
    assert!((data[0*16 + 0*4 + 1] - 1.5).abs() < 1e-9);
    // (1,0) ~ 2.0
    assert!((data[0*16 + 1*4 + 0] - 2.0).abs() < 1e-9);
    // (1,1) ~ 2.5
    assert!((data[0*16 + 1*4 + 1] - 2.5).abs() < 1e-9);
    // (3,3) ~ 4.0 (clamped edge)
    assert!((data[0*16 + 3*4 + 3] - 4.0).abs() < 1e-9);
}
