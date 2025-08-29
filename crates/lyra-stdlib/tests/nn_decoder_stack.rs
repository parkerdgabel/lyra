#![cfg(feature = "nn")]

use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;

#[test]
fn transformer_decoder_stack_builds_and_runs() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let mem = Value::List(vec![
        Value::List(vec![Value::Integer(1), Value::Integer(0), Value::Integer(0), Value::Integer(0)]),
        Value::List(vec![Value::Integer(0), Value::Integer(1), Value::Integer(0), Value::Integer(0)]),
    ]);
    let net = ev.eval(Value::expr(
        Value::Symbol("TransformerDecoderStack".into()),
        vec![Value::Assoc([
            ("Layers".into(), Value::Integer(2)),
            ("SeqLen".into(), Value::Integer(2)),
            ("ModelDim".into(), Value::Integer(4)),
            ("NumHeads".into(), Value::Integer(2)),
            ("HiddenDim".into(), Value::Integer(8)),
            ("Causal".into(), Value::Boolean(true)),
            ("Memory".into(), mem),
        ].into_iter().collect())]
    ));
    let x = Value::List(vec![Value::Integer(0), Value::Integer(0), Value::Integer(0), Value::Integer(0),
                              Value::Integer(0), Value::Integer(0), Value::Integer(0), Value::Integer(0)]);
    let y = ev.eval(Value::expr(Value::Symbol("Predict".into()), vec![net, x]));
    match y {
        Value::PackedArray { shape, data } => { assert_eq!(shape, vec![2,4]); assert_eq!(data.len(), 8); },
        Value::List(xs) => { assert_eq!(xs.len(), 8); },
        other => panic!("unexpected: {}", lyra_core::pretty::format_value(&other)),
    }
}

#[test]
fn transformer_decoder_stack_propagates_causal_mask() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let mem = Value::List(vec![
        Value::List(vec![Value::Integer(1), Value::Integer(0)]),
        Value::List(vec![Value::Integer(0), Value::Integer(1)]),
    ]);
    let net = ev.eval(Value::expr(
        Value::Symbol("TransformerDecoderStack".into()),
        vec![Value::Assoc([
            ("Layers".into(), Value::Integer(3)),
            ("SeqLen".into(), Value::Integer(2)),
            ("ModelDim".into(), Value::Integer(2)),
            ("NumHeads".into(), Value::Integer(1)),
            ("HiddenDim".into(), Value::Integer(4)),
            ("Causal".into(), Value::Boolean(true)),
            ("Memory".into(), mem),
        ].into_iter().collect())]
    ));
    let layers = ev.eval(Value::expr(Value::Symbol("NetProperty".into()), vec![net.clone(), Value::String("Layers".into())]));
    let mut count = 0;
    if let Value::List(ls) = layers {
        for l in ls {
            if let Value::Assoc(m) = l {
                if let Some(Value::Assoc(p)) = m.get("Params") {
                    if let Some(Value::Boolean(true)) = p.get("Causal") { count += 1; }
                }
            }
        }
    }
    assert_eq!(count, 3);
}

