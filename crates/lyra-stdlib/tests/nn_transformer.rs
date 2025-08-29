#![cfg(feature = "nn")]

use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;

#[test]
fn mha_identity_weights_average() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // 2 tokens, dim=2, heads=1; Wq=Wk=Wv=I, Wo=I
    let eye2 = Value::List(vec![
        Value::List(vec![Value::Integer(1), Value::Integer(0)]),
        Value::List(vec![Value::Integer(0), Value::Integer(1)]),
    ]);
    let layer = Value::expr(
        Value::Symbol("MultiHeadAttention".into()),
        vec![Value::Assoc([
            ("SeqLen".into(), Value::Integer(2)),
            ("ModelDim".into(), Value::Integer(2)),
            ("NumHeads".into(), Value::Integer(1)),
            ("Wq".into(), eye2.clone()),
            ("Wk".into(), eye2.clone()),
            ("Wv".into(), eye2.clone()),
            ("Wo".into(), eye2.clone()),
            ("bq".into(), Value::List(vec![Value::Integer(0), Value::Integer(0)])),
            ("bk".into(), Value::List(vec![Value::Integer(0), Value::Integer(0)])),
            ("bv".into(), Value::List(vec![Value::Integer(0), Value::Integer(0)])),
            ("bo".into(), Value::List(vec![Value::Integer(0), Value::Integer(0)])),
        ].into_iter().collect())]
    );
    let net = ev.eval(Value::expr(Value::Symbol("Sequential".into()), vec![Value::List(vec![layer])]));
    // x = [[1,0],[0,1]] flattened
    let x = Value::List(vec![Value::Integer(1), Value::Integer(0), Value::Integer(0), Value::Integer(1)]);
    let y = ev.eval(Value::expr(Value::Symbol("Predict".into()), vec![net, x]));
    let data = match y { Value::PackedArray { data, .. } => data, Value::List(xs) => xs.into_iter().map(|v| match v { Value::Integer(n)=>n as f64, Value::Real(r)=>r, _=>0.0 }).collect(), other => panic!("unexpected: {}", lyra_core::pretty::format_value(&other)) };
    // With symmetrical inputs and identity projections, each token attends equally -> average
    // Expected out: for each token, 0.5*([1,0]+[0,1]) = [0.5,0.5]
    assert!((data[0] - 0.5).abs() < 1e-9);
    assert!((data[1] - 0.5).abs() < 1e-9);
    assert!((data[2] - 0.5).abs() < 1e-9);
    assert!((data[3] - 0.5).abs() < 1e-9);
}

#[test]
fn positional_encoding_adds_signal() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // PE with SeqLen=2, Dim=4 on zero input -> returns sinusoid values directly
    let pe = Value::expr(
        Value::Symbol("PositionalEncoding".into()),
        vec![Value::Assoc([
            ("SeqLen".into(), Value::Integer(2)),
            ("ModelDim".into(), Value::Integer(4)),
        ].into_iter().collect())]
    );
    let net = ev.eval(Value::expr(Value::Symbol("Sequential".into()), vec![Value::List(vec![pe])]));
    let x = Value::List(vec![Value::Integer(0), Value::Integer(0), Value::Integer(0), Value::Integer(0),
                              Value::Integer(0), Value::Integer(0), Value::Integer(0), Value::Integer(0)]);
    let y = ev.eval(Value::expr(Value::Symbol("Predict".into()), vec![net, x]));
    let data = match y { Value::PackedArray { data, .. } => data, Value::List(xs) => xs.into_iter().map(|v| match v { Value::Integer(n)=>n as f64, Value::Real(r)=>r, _=>0.0 }).collect(), other => panic!("unexpected: {}", lyra_core::pretty::format_value(&other)) };
    // Check first token (pos=0): sin(0)=0, cos(0)=1
    assert!(data[0].abs() < 1e-12);
    assert!((data[1] - 1.0).abs() < 1e-12);
}

#[test]
fn mha_causal_mask_limits_future() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // 2 tokens, dim=2, identity projections; causal mask
    let eye2 = Value::List(vec![
        Value::List(vec![Value::Integer(1), Value::Integer(0)]),
        Value::List(vec![Value::Integer(0), Value::Integer(1)]),
    ]);
    let layer = Value::expr(
        Value::Symbol("MultiHeadAttention".into()),
        vec![Value::Assoc([
            ("SeqLen".into(), Value::Integer(2)),
            ("ModelDim".into(), Value::Integer(2)),
            ("NumHeads".into(), Value::Integer(1)),
            ("Mask".into(), Value::String("Causal".into())),
            ("Wq".into(), eye2.clone()), ("Wk".into(), eye2.clone()), ("Wv".into(), eye2.clone()), ("Wo".into(), eye2.clone()),
            ("bq".into(), Value::List(vec![Value::Integer(0), Value::Integer(0)])),
            ("bk".into(), Value::List(vec![Value::Integer(0), Value::Integer(0)])),
            ("bv".into(), Value::List(vec![Value::Integer(0), Value::Integer(0)])),
            ("bo".into(), Value::List(vec![Value::Integer(0), Value::Integer(0)])),
        ].into_iter().collect())]
    );
    let net = ev.eval(Value::expr(Value::Symbol("Sequential".into()), vec![Value::List(vec![layer])]));
    // x = [[1,0],[0,1]] flattened
    let x = Value::List(vec![Value::Integer(1), Value::Integer(0), Value::Integer(0), Value::Integer(1)]);
    let y = ev.eval(Value::expr(Value::Symbol("Predict".into()), vec![net, x]));
    let data = match y { Value::PackedArray { data, .. } => data, Value::List(xs) => xs.into_iter().map(|v| match v { Value::Integer(n)=>n as f64, Value::Real(r)=>r, _=>0.0 }).collect(), other => panic!("unexpected: {}", lyra_core::pretty::format_value(&other)) };
    // pos0 should attend only to itself -> [1,0]; pos1 attends to both -> average [0.5,0.5]
    assert!((data[0] - 1.0).abs() < 1e-9);
    assert!((data[1] - 0.0).abs() < 1e-9);
    assert!((data[2] - 0.5).abs() < 1e-9);
    assert!((data[3] - 0.5).abs() < 1e-9);
}
