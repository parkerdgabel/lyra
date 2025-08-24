use lyra_core::value::Value;
use lyra_runtime::{eval::Evaluator, set_default_registrar};
use lyra_stdlib as stdlib;

fn sym(s: &str) -> Value { Value::Symbol(s.into()) }

#[test]
fn packed_roundtrip_and_shape() {
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    let m = Value::List(vec![
        Value::List(vec![Value::Integer(1), Value::Integer(2)]),
        Value::List(vec![Value::Integer(3), Value::Integer(4)]),
    ]);
    let pack = Value::Expr { head: Box::new(sym("PackedArray")), args: vec![m.clone()] };
    let p = ev.eval(pack);
    // shape is {2,2}
    let shape = ev.eval(Value::Expr { head: Box::new(sym("PackedShape")), args: vec![p.clone()] });
    assert_eq!(shape, Value::List(vec![Value::Integer(2), Value::Integer(2)]));
    // roundtrip back to list
    let l = ev.eval(Value::Expr { head: Box::new(sym("PackedToList")), args: vec![p] });
    assert_eq!(l, Value::List(vec![
        Value::List(vec![Value::Real(1.0), Value::Real(2.0)]),
        Value::List(vec![Value::Real(3.0), Value::Real(4.0)]),
    ]));
}

