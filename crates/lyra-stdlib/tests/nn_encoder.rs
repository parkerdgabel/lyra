#![cfg(feature = "nn")]

use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;

#[test]
fn transformer_encoder_identity_weights() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // Seq=2, Dim=2, Hidden=2, Heads=1; set weights to identity to test residual paths
    let eye2 = Value::List(vec![
        Value::List(vec![Value::Integer(1), Value::Integer(0)]),
        Value::List(vec![Value::Integer(0), Value::Integer(1)]),
    ]);
    let zeros2 = Value::List(vec![Value::Integer(0), Value::Integer(0)]);
    let layer = Value::expr(Value::Symbol("TransformerEncoder".into()), vec![Value::Assoc([
        ("SeqLen".into(), Value::Integer(2)),
        ("ModelDim".into(), Value::Integer(2)),
        ("HiddenDim".into(), Value::Integer(2)),
        ("NumHeads".into(), Value::Integer(1)),
        // MHA
        ("Wq".into(), eye2.clone()), ("Wk".into(), eye2.clone()), ("Wv".into(), eye2.clone()), ("Wo".into(), eye2.clone()),
        ("bq".into(), zeros2.clone()), ("bk".into(), zeros2.clone()), ("bv".into(), zeros2.clone()), ("bo".into(), zeros2.clone()),
        // FFN (two identities)
        ("W1".into(), eye2.clone()), ("b1".into(), zeros2.clone()),
        ("W2".into(), eye2.clone()), ("b2".into(), zeros2.clone()),
        ("Activation".into(), Value::String("Gelu".into())),
    ].into_iter().collect())]);
    let net = ev.eval(Value::expr(Value::Symbol("Sequential".into()), vec![Value::List(vec![layer])]));
    // Input tokens [[1,0],[0,1]]; with identity setup and residuals, output remains stable and normalized
    let x = Value::List(vec![Value::Integer(1), Value::Integer(0), Value::Integer(0), Value::Integer(1)]);
    let y = ev.eval(Value::expr(Value::Symbol("Predict".into()), vec![net, x]));
    match y {
        Value::PackedArray { shape, data } => { assert_eq!(shape, vec![2,2]); assert_eq!(data.len(), 4); },
        Value::List(xs) => { assert_eq!(xs.len(), 4); },
        other => panic!("unexpected: {}", lyra_core::pretty::format_value(&other)),
    }
}

