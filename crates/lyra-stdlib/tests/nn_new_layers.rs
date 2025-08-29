#![cfg(feature = "nn")]

use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;

#[test]
fn rmsnorm_normalizes_rms_per_token() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // seq=2, dim=2
    let layer = Value::expr(Value::Symbol("RMSNorm".into()), vec![Value::Assoc([
        ("SeqLen".into(), Value::Integer(2)),
        ("ModelDim".into(), Value::Integer(2)),
    ].into_iter().collect())]);
    let net = ev.eval(Value::expr(Value::Symbol("Sequential".into()), vec![Value::List(vec![layer])]));
    let x = Value::List(vec![Value::Integer(3), Value::Integer(4), Value::Integer(1), Value::Integer(1)]);
    let y = ev.eval(Value::expr(Value::Symbol("Predict".into()), vec![net, x]));
    let data = match y { Value::PackedArray { data, .. } => data, Value::List(xs) => xs.into_iter().map(|v| match v { Value::Integer(n)=>n as f64, Value::Real(r)=>r, _=>0.0 }).collect(), _ => vec![] };
    assert_eq!(data.len(), 4);
    // RMS per row should be ~1
    let rms0 = ((data[0]*data[0] + data[1]*data[1])/2.0).sqrt();
    let rms1 = ((data[2]*data[2] + data[3]*data[3])/2.0).sqrt();
    assert!((rms0 - 1.0).abs() < 1e-6);
    assert!((rms1 - 1.0).abs() < 1e-6);
}

#[test]
fn positional_embedding_adds_learned_offsets() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let pe = Value::expr(Value::Symbol("PositionalEmbedding".into()), vec![Value::Assoc([
        ("SeqLen".into(), Value::Integer(2)),
        ("ModelDim".into(), Value::Integer(2)),
        ("P".into(), Value::List(vec![
            Value::List(vec![Value::Integer(1), Value::Integer(2)]),
            Value::List(vec![Value::Integer(3), Value::Integer(4)]),
        ])),
    ].into_iter().collect())]);
    let net = ev.eval(Value::expr(Value::Symbol("Sequential".into()), vec![Value::List(vec![pe])]));
    let x = Value::List(vec![Value::Integer(0), Value::Integer(0), Value::Integer(0), Value::Integer(0)]);
    let y = ev.eval(Value::expr(Value::Symbol("Predict".into()), vec![net, x]));
    let data = match y { Value::PackedArray { data, .. } => data, Value::List(xs) => xs.into_iter().map(|v| match v { Value::Integer(n)=>n as f64, Value::Real(r)=>r, _=>0.0 }).collect(), _ => vec![] };
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn patch_embedding2d_produces_tokens() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // 1x2x2 image, patch 2x2 -> 1 token, dim=2; weights all ones for easy check
    let w = Value::List(vec![
        Value::List(vec![Value::List(vec![Value::List(vec![Value::Integer(1), Value::Integer(1)]), Value::List(vec![Value::Integer(1), Value::Integer(1)])])]),
        Value::List(vec![Value::List(vec![Value::List(vec![Value::Integer(1), Value::Integer(1)]), Value::List(vec![Value::Integer(1), Value::Integer(1)])])]),
    ]);
    let b = Value::List(vec![Value::Integer(0), Value::Integer(0)]);
    let layer = Value::expr(Value::Symbol("PatchEmbedding2D".into()), vec![Value::Assoc([
        ("PatchSize".into(), Value::List(vec![Value::Integer(2), Value::Integer(2)])),
        ("ModelDim".into(), Value::Integer(2)),
        ("InputChannels".into(), Value::Integer(1)),
        ("Height".into(), Value::Integer(2)),
        ("Width".into(), Value::Integer(2)),
        ("W".into(), w),
        ("b".into(), b),
    ].into_iter().collect())]);
    let net = ev.eval(Value::expr(Value::Symbol("Sequential".into()), vec![Value::List(vec![layer])]));
    // Input CHW: C=1, H=2, W=2 -> [1,2,3,4]
    let x = Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3), Value::Integer(4)]);
    let y = ev.eval(Value::expr(Value::Symbol("Predict".into()), vec![net, x]));
    let data = match y { Value::PackedArray { data, .. } => data, Value::List(xs) => xs.into_iter().map(|v| match v { Value::Integer(n)=>n as f64, Value::Real(r)=>r, _=>0.0 }).collect(), _ => vec![] };
    assert_eq!(data.len(), 2);
    assert!((data[0] - 10.0).abs() < 1e-9);
    assert!((data[1] - 10.0).abs() < 1e-9);
}

