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
    // NDPermuteDims explicit permutation
    let p = ev.eval(Value::expr(Value::Symbol("NDPermuteDims".into()), vec![b.clone(), Value::List(vec![Value::Integer(1), Value::Integer(0)])]));
    let shp4 = ev.eval(Value::expr(Value::Symbol("NDShape".into()), vec![p.clone()]));
    assert_eq!(format!("{}", lyra_core::pretty::format_value(&shp4)), "{2, 3}");
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
    // NDSlice along axis 1: take last 2 columns (2..3) from 2x3
    let sl = ev.eval(Value::expr(Value::Symbol("NDSlice".into()), vec![a.clone(), Value::Integer(1), Value::Integer(2), Value::Integer(2)]));
    match sl { Value::PackedArray { shape, data } => { assert_eq!(shape, vec![2,2]); assert_eq!(data, vec![2.0,3.0,5.0,6.0]); }, other => panic!("unexpected slice: {}", lyra_core::pretty::format_value(&other)) }
    // 1D slice form
    let v = ev.eval(Value::expr(Value::Symbol("NDArray".into()), vec![ Value::List(vec![Value::Integer(10), Value::Integer(20), Value::Integer(30), Value::Integer(40)]) ]));
    let sl1 = ev.eval(Value::expr(Value::Symbol("NDSlice".into()), vec![v.clone(), Value::List(vec![Value::Integer(2), Value::Integer(2)])]));
    match sl1 { Value::PackedArray { shape, data } => { assert_eq!(shape, vec![2]); assert_eq!(data, vec![20.0, 30.0]); }, other => panic!("unexpected 1D slice: {}", lyra_core::pretty::format_value(&other)) }
    // Multi-axis spec: {row index=2, col range {2,1}} => 1D [5]
    let spec = Value::List(vec![Value::Integer(2), Value::List(vec![Value::Integer(2), Value::Integer(1)])]);
    let sl2 = ev.eval(Value::expr(Value::Symbol("NDSlice".into()), vec![a.clone(), spec]));
    match sl2 { Value::PackedArray { shape, data } => { assert_eq!(shape, vec![1]); assert_eq!(data, vec![5.0]); }, other => panic!("unexpected multi-axis scalar-ish slice: {}", lyra_core::pretty::format_value(&other)) }
    // With All symbol on rows, same as axis slice above
    let spec_all = Value::List(vec![Value::Symbol("All".into()), Value::List(vec![Value::Integer(2), Value::Integer(2)])]);
    let sl3 = ev.eval(Value::expr(Value::Symbol("NDSlice".into()), vec![a.clone(), spec_all]));
    match sl3 { Value::PackedArray { shape, data } => { assert_eq!(shape, vec![2,2]); assert_eq!(data, vec![2.0,3.0,5.0,6.0]); }, other => panic!("unexpected multi-axis slice: {}", lyra_core::pretty::format_value(&other)) }
}
