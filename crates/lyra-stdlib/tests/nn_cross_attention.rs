#![cfg(feature = "nn")]

use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;

fn eye2() -> Value {
    Value::List(vec![
        Value::List(vec![Value::Integer(1), Value::Integer(0)]),
        Value::List(vec![Value::Integer(0), Value::Integer(1)]),
    ])
}

fn zeros2() -> Value { Value::List(vec![Value::Integer(0), Value::Integer(0)]) }

fn zeros_mat2() -> Value {
    Value::List(vec![
        Value::List(vec![Value::Integer(0), Value::Integer(0)]),
        Value::List(vec![Value::Integer(0), Value::Integer(0)]),
    ])
}

#[test]
fn cross_attention_uniform_when_qk_zero() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // Seq=2, Dim=2, Heads=1; set Wq=Wk=0 so attention logits are zeros -> uniform
    // Wv=I, Wo=I, Memory has two one-hot tokens -> output is average [0.5,0.5] per query
    let mem = Value::List(vec![
        Value::List(vec![Value::Integer(1), Value::Integer(0)]),
        Value::List(vec![Value::Integer(0), Value::Integer(1)]),
    ]);
    let layer = Value::expr(Value::Symbol("CrossAttention".into()), vec![Value::Assoc([
        ("SeqLen".into(), Value::Integer(2)),
        ("ModelDim".into(), Value::Integer(2)),
        ("NumHeads".into(), Value::Integer(1)),
        ("Wq".into(), zeros_mat2()), ("Wk".into(), zeros_mat2()), ("Wv".into(), eye2()), ("Wo".into(), eye2()),
        ("bq".into(), zeros2()), ("bk".into(), zeros2()), ("bv".into(), zeros2()), ("bo".into(), zeros2()),
        ("Memory".into(), mem),
    ].into_iter().collect())]);
    let net = ev.eval(Value::expr(Value::Symbol("Sequential".into()), vec![Value::List(vec![layer])]));
    // Input 2x2, arbitrary
    let x = Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3), Value::Integer(4)]);
    let y = ev.eval(Value::expr(Value::Symbol("Predict".into()), vec![net, x]));
    let data = match y {
        Value::PackedArray { data, .. } => data,
        Value::List(xs) => xs.into_iter().map(|v| match v { Value::Integer(n)=>n as f64, Value::Real(r)=>r, _=>0.0 }).collect(),
        other => panic!("unexpected: {}", lyra_core::pretty::format_value(&other)),
    };
    assert_eq!(data.len(), 4);
    for &v in &data { assert!((v - 0.5).abs() < 1e-9); }
}

#[test]
fn cross_attention_memory_mask_vector() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // With MemoryMask vector [1,0], only first memory token is allowed -> output equals first memory
    let mem = Value::List(vec![
        Value::List(vec![Value::Integer(1), Value::Integer(0)]),
        Value::List(vec![Value::Integer(0), Value::Integer(1)]),
    ]);
    let layer = Value::expr(Value::Symbol("CrossAttention".into()), vec![Value::Assoc([
        ("SeqLen".into(), Value::Integer(2)),
        ("ModelDim".into(), Value::Integer(2)),
        ("NumHeads".into(), Value::Integer(1)),
        ("Wq".into(), zeros_mat2()), ("Wk".into(), zeros_mat2()), ("Wv".into(), eye2()), ("Wo".into(), eye2()),
        ("bq".into(), zeros2()), ("bk".into(), zeros2()), ("bv".into(), zeros2()), ("bo".into(), zeros2()),
        ("Memory".into(), mem),
        ("MemoryMask".into(), Value::List(vec![Value::Integer(1), Value::Integer(0)])),
    ].into_iter().collect())]);
    let net = ev.eval(Value::expr(Value::Symbol("Sequential".into()), vec![Value::List(vec![layer])]));
    let x = Value::List(vec![Value::Integer(0), Value::Integer(0), Value::Integer(0), Value::Integer(0)]);
    let y = ev.eval(Value::expr(Value::Symbol("Predict".into()), vec![net, x]));
    let data = match y {
        Value::PackedArray { data, .. } => data,
        Value::List(xs) => xs.into_iter().map(|v| match v { Value::Integer(n)=>n as f64, Value::Real(r)=>r, _=>0.0 }).collect(),
        other => panic!("unexpected: {}", lyra_core::pretty::format_value(&other)),
    };
    assert_eq!(data, vec![1.0, 0.0, 1.0, 0.0]);
}

