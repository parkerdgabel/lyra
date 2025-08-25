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
    // NDMap: double elements
    let times2 = Value::PureFunction { params: None, body: Box::new(Value::Expr { head: Box::new(Value::Symbol("Times".into())), args: vec![Value::Slot(None), Value::Integer(2)] }) };
    let m = ev.eval(Value::expr(Value::Symbol("NDMap".into()), vec![a.clone(), times2]));
    match m { Value::PackedArray { shape, data } => { assert_eq!(shape, vec![2,3]); assert_eq!(data, vec![2.0,4.0,6.0,8.0,10.0,12.0]); }, other => panic!("unexpected map: {}", lyra_core::pretty::format_value(&other)) }
    // NDReduce overall with Plus
    let r = ev.eval(Value::expr(Value::Symbol("NDReduce".into()), vec![a.clone(), Value::Symbol("Plus".into())]));
    assert_eq!(r, Value::Real(21.0));
    // NDReduce across axis 1 with Max
    let r2 = ev.eval(Value::expr(Value::Symbol("NDReduce".into()), vec![a.clone(), Value::Symbol("Max".into()), Value::Integer(1)]));
    match r2 { Value::PackedArray { shape, data } => { assert_eq!(shape, vec![2]); assert_eq!(data, vec![3.0,6.0]); }, other => panic!("unexpected reduce axis: {}", lyra_core::pretty::format_value(&other)) }
    // NDAdd broadcasting row vector
    let row = ev.eval(Value::expr(Value::Symbol("NDArray".into()), vec![ Value::List(vec![Value::Integer(10), Value::Integer(20), Value::Integer(30)]) ]));
    let addres = ev.eval(Value::expr(Value::Symbol("NDAdd".into()), vec![a.clone(), row.clone()]));
    match addres { Value::PackedArray { shape, data } => { assert_eq!(shape, vec![2,3]); assert_eq!(data, vec![11.0,22.0,33.0,14.0,25.0,36.0]); }, other => panic!("unexpected NDAdd: {}", lyra_core::pretty::format_value(&other)) }
    // NDMul broadcasting 2x1 * 1x3
    let col = ev.eval(Value::expr(Value::Symbol("NDArray".into()), vec![ Value::List(vec![Value::List(vec![Value::Integer(2)]), Value::List(vec![Value::Integer(3)]) ]) ]));
    let mulres = ev.eval(Value::expr(Value::Symbol("NDMul".into()), vec![col.clone(), row.clone()]));
    match mulres { Value::PackedArray { shape, data } => { assert_eq!(shape, vec![2,3]); assert_eq!(data, vec![20.0,40.0,60.0,30.0,60.0,90.0]); }, other => panic!("unexpected NDMul: {}", lyra_core::pretty::format_value(&other)) }
    // NDEltwise with a custom function: (x+y)/2
    let halfsum = Value::PureFunction { params: None, body: Box::new(Value::Expr { head: Box::new(Value::Symbol("Divide".into())), args: vec![ Value::Expr { head: Box::new(Value::Symbol("Plus".into())), args: vec![Value::Slot(Some(1)), Value::Slot(Some(2))] }, Value::Integer(2) ] }) };
    let ew = ev.eval(Value::expr(Value::Symbol("NDEltwise".into()), vec![halfsum, a.clone(), a.clone()]));
    match ew { Value::PackedArray { shape, data } => { assert_eq!(shape, vec![2,3]); assert_eq!(data, vec![1.0,2.0,3.0,4.0,5.0,6.0]); }, other => panic!("unexpected NDEltwise: {}", lyra_core::pretty::format_value(&other)) }
    // NDPow
    let p2 = ev.eval(Value::expr(Value::Symbol("NDPow".into()), vec![a.clone(), Value::Integer(2)]));
    match &p2 { Value::PackedArray { shape, data } => { assert_eq!(shape, &vec![2,3]); assert_eq!(data, &vec![1.0,4.0,9.0,16.0,25.0,36.0]); }, other => panic!("unexpected NDPow: {}", lyra_core::pretty::format_value(&other)) }
    // pow with negative exponent (scalar)
    let inv = ev.eval(Value::expr(Value::Symbol("NDPow".into()), vec![a.clone(), Value::Integer(-1)]));
    match inv { Value::PackedArray { shape, data } => { assert_eq!(shape, vec![2,3]); assert!((data[0] - 1.0).abs() < 1e-9 && (data[1] - 0.5).abs() < 1e-9); }, _ => panic!("unexpected NDPow negative") }
    // NDClip and NDRelu
    let mixed = ev.eval(Value::expr(Value::Symbol("NDArray".into()), vec![ Value::List(vec![Value::Integer(-1), Value::Integer(0), Value::Integer(2), Value::Integer(5)]) ]));
    let clipped = ev.eval(Value::expr(Value::Symbol("NDClip".into()), vec![mixed.clone(), Value::Integer(0), Value::Integer(3)]));
    match clipped { Value::PackedArray { shape, data } => { assert_eq!(shape, vec![4]); assert_eq!(data, vec![0.0,0.0,2.0,3.0]); }, other => panic!("unexpected NDClip: {}", lyra_core::pretty::format_value(&other)) }
    let relu = ev.eval(Value::expr(Value::Symbol("NDRelu".into()), vec![mixed.clone()]));
    match relu { Value::PackedArray { shape, data } => { assert_eq!(shape, vec![4]); assert_eq!(data, vec![0.0,0.0,2.0,5.0]); }, other => panic!("unexpected NDRelu: {}", lyra_core::pretty::format_value(&other)) }
    // NDExp/NDLog/NDSqrt simple sanity
    let ones = ev.eval(Value::expr(Value::Symbol("NDArray".into()), vec![ Value::List(vec![Value::Integer(1), Value::Integer(1)]) ]));
    let _ex = ev.eval(Value::expr(Value::Symbol("NDExp".into()), vec![ones.clone()]));
    let lg = ev.eval(Value::expr(Value::Symbol("NDLog".into()), vec![ones.clone()]));
    match lg { Value::PackedArray { data, .. } => assert!(data.iter().all(|x| x.abs() < 1e-9)), _ => panic!() }
    let sq = ev.eval(Value::expr(Value::Symbol("NDSqrt".into()), vec![p2.clone()]));
    match sq { Value::PackedArray { data, .. } => assert!(data.iter().zip(vec![1.0,2.0,3.0,4.0,5.0,6.0]).all(|(a,b)| (a-b).abs()<1e-9)), _ => panic!() }
}
