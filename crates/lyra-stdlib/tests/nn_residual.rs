#![cfg(feature = "nn")]

use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;

#[test]
fn residual_wrapper_activation_adds_skip() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);

    // Residual wrapper with inner Relu: y = x + Relu(x)
    // For x = {-1, 2}, Relu(x) = {0,2} so y = {-1,4}
    let res = Value::expr(
        Value::Symbol("Residual".into()),
        vec![Value::List(vec![
            Value::expr(Value::Symbol("Relu".into()), vec![]),
        ])],
    );
    let net = ev.eval(Value::expr(Value::Symbol("Sequential".into()), vec![Value::List(vec![res])]));
    let x = Value::List(vec![Value::Integer(-1), Value::Integer(2)]);
    let y = ev.eval(Value::expr(Value::Symbol("Predict".into()), vec![net, x]));
    match y {
        Value::List(xs) => {
            assert_eq!(xs.len(), 2);
            let a = match &xs[0] { Value::Integer(n) => *n as f64, Value::Real(r) => *r, _ => 0.0 };
            let b = match &xs[1] { Value::Integer(n) => *n as f64, Value::Real(r) => *r, _ => 0.0 };
            assert!((a + 1.0).abs() < 1e-9); // -1
            assert!((b - 4.0).abs() < 1e-9); // 4
        }
        Value::PackedArray { shape, data } => {
            assert_eq!(shape, vec![2]);
            assert!((data[0] + 1.0).abs() < 1e-9);
            assert!((data[1] - 4.0).abs() < 1e-9);
        }
        other => panic!("unexpected: {}", lyra_core::pretty::format_value(&other)),
    }
}

