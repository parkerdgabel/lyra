#![cfg(feature = "nn")]

use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;

#[test]
fn convtranspose2d_small_forward() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);

    // ConvTranspose2D with 1 out ch, 1 in ch, 2x2 kernel of ones, stride 2
    let layer = Value::expr(
        Value::Symbol("ConvTranspose2D".into()),
        vec![Value::Assoc([
            ("Output".into(), Value::Integer(1)),
            ("KernelSize".into(), Value::List(vec![Value::Integer(2), Value::Integer(2)])),
            ("Stride".into(), Value::Integer(2)),
            ("Padding".into(), Value::Integer(0)),
            ("InputChannels".into(), Value::Integer(1)),
            ("Height".into(), Value::Integer(1)),
            ("Width".into(), Value::Integer(1)),
            // W[out][in][kh][kw] = ones
            ("W".into(), Value::List(vec![Value::List(vec![Value::List(vec![
                Value::List(vec![Value::Integer(1), Value::Integer(1)]),
                Value::List(vec![Value::Integer(1), Value::Integer(1)]),
            ])])])),
            ("b".into(), Value::List(vec![Value::Integer(0)])),
        ].into_iter().collect())]
    );

    let net = ev.eval(Value::expr(Value::Symbol("Sequential".into()), vec![Value::List(vec![layer])]));
    // Input single pixel value 3.0 (1x1x1)
    let x = Value::List(vec![Value::Integer(3)]);
    let y = ev.eval(Value::expr(Value::Symbol("Predict".into()), vec![net, x]));
    match y {
        Value::PackedArray { shape, data } => {
            assert_eq!(shape, vec![1,2,2]);
            for v in data { assert!((v - 3.0).abs() < 1e-9); }
        }
        Value::List(xs) => {
            assert_eq!(xs.len(), 4);
        }
        other => panic!("unexpected: {}", lyra_core::pretty::format_value(&other)),
    }
}

