#![cfg(feature = "nn")]

use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;

#[test]
fn transformer_decoder_builds_and_runs_with_memory() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // Decoder with Seq=2, Dim=4, Heads=2, simple fixed memory 2x4
    let mem = Value::List(vec![
        Value::List(vec![Value::Integer(1), Value::Integer(0), Value::Integer(0), Value::Integer(0)]),
        Value::List(vec![Value::Integer(0), Value::Integer(1), Value::Integer(0), Value::Integer(0)]),
    ]);
    let layer = Value::expr(Value::Symbol("TransformerDecoder".into()), vec![Value::Assoc([
        ("SeqLen".into(), Value::Integer(2)),
        ("ModelDim".into(), Value::Integer(4)),
        ("NumHeads".into(), Value::Integer(2)),
        ("HiddenDim".into(), Value::Integer(8)),
        ("Memory".into(), mem),
        ("Causal".into(), Value::Boolean(true)),
    ].into_iter().collect())]);
    let net = ev.eval(Value::expr(Value::Symbol("Sequential".into()), vec![Value::List(vec![layer])]));
    // Input 2x4 flattened (zeros)
    let x = Value::List(vec![Value::Integer(0), Value::Integer(0), Value::Integer(0), Value::Integer(0),
                              Value::Integer(0), Value::Integer(0), Value::Integer(0), Value::Integer(0)]);
    let y = ev.eval(Value::expr(Value::Symbol("Predict".into()), vec![net, x]));
    match y {
        Value::PackedArray { shape, data } => { assert_eq!(shape, vec![2,4]); assert_eq!(data.len(), 8); },
        Value::List(xs) => { assert_eq!(xs.len(), 8); },
        other => panic!("unexpected: {}", lyra_core::pretty::format_value(&other)),
    }
}

