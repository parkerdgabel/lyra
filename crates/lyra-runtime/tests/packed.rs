use lyra_core::value::Value;
use lyra_runtime::{eval::Evaluator, set_default_registrar};
use lyra_stdlib as stdlib;

fn sym(s: &str) -> Value {
    Value::Symbol(s.into())
}

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
    assert_eq!(
        l,
        Value::List(vec![
            Value::List(vec![Value::Real(1.0), Value::Real(2.0)]),
            Value::List(vec![Value::Real(3.0), Value::Real(4.0)]),
        ])
    );
}

#[test]
fn packed_elementwise_and_scalar_ops() {
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    let a = Value::List(vec![
        Value::List(vec![Value::Integer(1), Value::Integer(2)]),
        Value::List(vec![Value::Integer(3), Value::Integer(4)]),
    ]);
    let b = Value::List(vec![
        Value::List(vec![Value::Integer(5), Value::Integer(6)]),
        Value::List(vec![Value::Integer(7), Value::Integer(8)]),
    ]);
    let pa = ev.eval(Value::Expr { head: Box::new(sym("PackedArray")), args: vec![a] });
    let pb = ev.eval(Value::Expr { head: Box::new(sym("PackedArray")), args: vec![b] });
    // Elementwise addition
    let sum =
        ev.eval(Value::Expr { head: Box::new(sym("Plus")), args: vec![pa.clone(), pb.clone()] });
    let suml = ev.eval(Value::Expr { head: Box::new(sym("PackedToList")), args: vec![sum] });
    assert_eq!(
        suml,
        Value::List(vec![
            Value::List(vec![Value::Real(6.0), Value::Real(8.0)]),
            Value::List(vec![Value::Real(10.0), Value::Real(12.0)]),
        ])
    );
    // Scalar multiply
    let prod =
        ev.eval(Value::Expr { head: Box::new(sym("Times")), args: vec![pa, Value::Integer(10)] });
    let prodl = ev.eval(Value::Expr { head: Box::new(sym("PackedToList")), args: vec![prod] });
    assert_eq!(
        prodl,
        Value::List(vec![
            Value::List(vec![Value::Real(10.0), Value::Real(20.0)]),
            Value::List(vec![Value::Real(30.0), Value::Real(40.0)]),
        ])
    );
}

#[test]
fn packed_map_elementwise() {
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    let a = Value::List(vec![
        Value::List(vec![Value::Integer(1), Value::Integer(2)]),
        Value::List(vec![Value::Integer(3), Value::Integer(4)]),
    ]);
    let pa = ev.eval(Value::Expr { head: Box::new(sym("PackedArray")), args: vec![a] });
    // Map[(x)=>x+1, pa]
    let f = Value::PureFunction {
        params: Some(vec!["x".into()]),
        body: Box::new(Value::Expr {
            head: Box::new(sym("Plus")),
            args: vec![Value::Symbol("x".into()), Value::Integer(1)],
        }),
    };
    let mapped = ev.eval(Value::Expr { head: Box::new(sym("Map")), args: vec![f, pa] });
    let out = ev.eval(Value::Expr { head: Box::new(sym("PackedToList")), args: vec![mapped] });
    assert_eq!(
        out,
        Value::List(vec![
            Value::List(vec![Value::Real(2.0), Value::Real(3.0)]),
            Value::List(vec![Value::Real(4.0), Value::Real(5.0)]),
        ])
    );
}

#[test]
fn packed_broadcast_minus_divide() {
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    // A shape {2,1}
    let a = Value::List(vec![
        Value::List(vec![Value::Integer(1)]),
        Value::List(vec![Value::Integer(2)]),
    ]);
    // B shape {1,2}
    let b = Value::List(vec![Value::List(vec![Value::Integer(10), Value::Integer(20)])]);
    let pa = ev.eval(Value::Expr { head: Box::new(sym("PackedArray")), args: vec![a] });
    let pb = ev.eval(Value::Expr { head: Box::new(sym("PackedArray")), args: vec![b] });
    let diff =
        ev.eval(Value::Expr { head: Box::new(sym("Minus")), args: vec![pb.clone(), pa.clone()] });
    let diff_l = ev.eval(Value::Expr { head: Box::new(sym("PackedToList")), args: vec![diff] });
    assert_eq!(
        diff_l,
        Value::List(vec![
            Value::List(vec![Value::Real(9.0), Value::Real(19.0)]),
            Value::List(vec![Value::Real(8.0), Value::Real(18.0)]),
        ])
    );
    let quot = ev.eval(Value::Expr { head: Box::new(sym("Divide")), args: vec![pb, pa] });
    let quot_l = ev.eval(Value::Expr { head: Box::new(sym("PackedToList")), args: vec![quot] });
    assert_eq!(
        quot_l,
        Value::List(vec![
            Value::List(vec![Value::Real(10.0), Value::Real(20.0)]),
            Value::List(vec![Value::Real(5.0), Value::Real(10.0)]),
        ])
    );
}
