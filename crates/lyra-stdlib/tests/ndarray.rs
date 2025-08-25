use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;

#[test]
fn ndarray_shape_reshape_transpose() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // Create 2x3
    let a = ev.eval(Value::expr(Value::Symbol("NDArray".into()), vec![
        Value::List(vec![
            Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)]),
            Value::List(vec![Value::Integer(4), Value::Integer(5), Value::Integer(6)]),
        ])
    ]));
    let shp = ev.eval(Value::expr(Value::Symbol("NDShape".into()), vec![a.clone()]));
    assert_eq!(format!("{}", lyra_core::pretty::format_value(&shp)), "{2, 3}");
    let b = ev.eval(Value::expr(Value::Symbol("NDReshape".into()), vec![a.clone(), Value::List(vec![Value::Integer(3), Value::Integer(2)])]));
    let shp2 = ev.eval(Value::expr(Value::Symbol("NDShape".into()), vec![b.clone()]));
    assert_eq!(format!("{}", lyra_core::pretty::format_value(&shp2)), "{3, 2}");
    let t = ev.eval(Value::expr(Value::Symbol("NDTranspose".into()), vec![b.clone()]));
    let shp3 = ev.eval(Value::expr(Value::Symbol("NDShape".into()), vec![t.clone()]));
    assert_eq!(format!("{}", lyra_core::pretty::format_value(&shp3)), "{2, 3}");
}

#[test]
fn ndarray_sum_mean_argmax_matmul() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let a = ev.eval(Value::expr(Value::Symbol("NDArray".into()), vec![
        Value::List(vec![
            Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)]),
            Value::List(vec![Value::Integer(4), Value::Integer(5), Value::Integer(6)]),
        ])
    ]));
    let s0 = ev.eval(Value::expr(Value::Symbol("NDSum".into()), vec![a.clone()]));
    assert_eq!(s0, Value::Real(21.0));
    let s1 = ev.eval(Value::expr(Value::Symbol("NDSum".into()), vec![a.clone(), Value::Integer(0)]));
    match s1 { Value::PackedArray { shape, data } => { assert_eq!(shape, vec![3]); assert_eq!(data, vec![5.0,7.0,9.0]); }, other => panic!("unexpected: {}", lyra_core::pretty::format_value(&other)) }
    let m1 = ev.eval(Value::expr(Value::Symbol("NDMean".into()), vec![a.clone(), Value::Integer(1)]));
    match m1 { Value::PackedArray { shape, data } => { assert_eq!(shape, vec![2]); assert_eq!(data, vec![2.0,5.0]); }, other => panic!("unexpected mean: {}", lyra_core::pretty::format_value(&other)) }
    // MatMul (2x3) * (3x2)
    let b = ev.eval(Value::expr(Value::Symbol("NDArray".into()), vec![
        Value::List(vec![
            Value::List(vec![Value::Integer(1), Value::Integer(2)]),
            Value::List(vec![Value::Integer(3), Value::Integer(4)]),
            Value::List(vec![Value::Integer(5), Value::Integer(6)]),
        ])
    ]));
    let mm = ev.eval(Value::expr(Value::Symbol("NDMatMul".into()), vec![a.clone(), b.clone()]));
    match mm { Value::PackedArray { shape, data } => { assert_eq!(shape, vec![2,2]); assert_eq!(data, vec![22.0, 28.0, 49.0, 64.0]); }, other => panic!("unexpected matmul: {}", lyra_core::pretty::format_value(&other)) }
}
