use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_runtime::attrs::Attributes;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

fn as_packed(ev: &mut Evaluator, v: Value) -> Option<(Vec<usize>, Vec<f64>)> {
    match ev.eval(v) {
        Value::PackedArray { shape, data } => Some((shape, data)),
        other => match ev.eval(Value::Expr { head: Box::new(Value::Symbol("PackedArray".into())), args: vec![other] }) {
            Value::PackedArray { shape, data } => Some((shape, data)),
            _ => None,
        }
    }
}

fn pack(shape: Vec<usize>, data: Vec<f64>) -> Value { Value::PackedArray { shape, data } }

fn total_elems(shape: &[usize]) -> usize { shape.iter().product::<usize>() }

fn strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    let mut st = vec![0; n];
    let mut acc = 1usize;
    for i in (0..n).rev() { st[i] = acc; acc *= shape[i]; }
    st
}

fn idx_of(idx: &[usize], st: &[usize]) -> usize { idx.iter().zip(st.iter()).map(|(i, s)| i * s).sum() }

fn ndarray(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // NDArray[x, opts?] -> PackedArray[x]
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("NDArray".into())), args } }
    match ev.eval(Value::Expr { head: Box::new(Value::Symbol("PackedArray".into())), args: vec![args[0].clone()] }) {
        Value::PackedArray { shape, data } => pack(shape, data),
        other => other,
    }
}

fn nd_shape(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("NDShape".into())), args } }
    match as_packed(ev, args[0].clone()) { Some((shape, _)) => Value::List(shape.into_iter().map(|d| Value::Integer(d as i64)).collect()), _ => Value::Expr { head: Box::new(Value::Symbol("NDShape".into())), args } }
}

fn nd_type(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("NDType".into())), args } }
    match as_packed(ev, args[0].clone()) { Some(_) => Value::String("Float64".into()), _ => Value::Expr { head: Box::new(Value::Symbol("NDType".into())), args } }
}

fn nd_as_type(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Only Float64 supported; passthrough
    if args.len()<1 { return Value::Expr { head: Box::new(Value::Symbol("NDAsType".into())), args } }
    ev.eval(args[0].clone())
}

fn nd_reshape(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // NDReshape[a, {dims...}] supports one -1 to infer
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("NDReshape".into())), args } }
    let (shape, data) = match as_packed(ev, args[0].clone()) { Some(x)=>x, None=> return Value::Expr { head: Box::new(Value::Symbol("NDReshape".into())), args } };
    let new_shape_v = ev.eval(args[1].clone());
    let dims: Vec<i64> = match &new_shape_v { Value::List(xs) => xs.iter().filter_map(|v| if let Value::Integer(n)=v { Some(*n) } else { None }).collect(), _ => Vec::new() };
    if dims.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("NDReshape".into())), args: vec![pack(shape, data), new_shape_v] } }
    let mut out_shape: Vec<usize> = Vec::with_capacity(dims.len());
    let mut infer_at: Option<usize> = None;
    for (i, d) in dims.iter().enumerate() {
        if *d == -1 { if infer_at.is_some() { return Value::Expr { head: Box::new(Value::Symbol("NDReshape".into())), args } } infer_at = Some(i); out_shape.push(1); }
        else if *d > 0 { out_shape.push(*d as usize); } else { return Value::Expr { head: Box::new(Value::Symbol("NDReshape".into())), args } }
    }
    let total_old = data.len();
    let known: usize = out_shape.iter().product();
    if let Some(pos) = infer_at { if known == 0 || total_old % known != 0 { return Value::Expr { head: Box::new(Value::Symbol("NDReshape".into())), args } } else { out_shape[pos] = total_old / known; } }
    if total_old != out_shape.iter().product::<usize>() { return Value::Expr { head: Box::new(Value::Symbol("NDReshape".into())), args } }
    pack(out_shape, data)
}

fn nd_transpose(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // NDTranspose[a] (reverse axes) or NDTranspose[a, {perm...}]
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("NDTranspose".into())), args } }
    let (shape, data) = match as_packed(ev, args[0].clone()) { Some(x)=>x, None=> return Value::Expr { head: Box::new(Value::Symbol("NDTranspose".into())), args } };
    let ndim = shape.len();
    let perm: Vec<usize> = if args.len()>=2 {
        match ev.eval(args[1].clone()) { Value::List(xs) => xs.into_iter().filter_map(|v| if let Value::Integer(n)=v { Some(n as usize) } else { None }).collect(), _ => Vec::new() }
    } else { (0..ndim).rev().collect() };
    if perm.len()!=ndim || {
        let mut s = perm.clone(); s.sort_unstable(); s != (0..ndim).collect::<Vec<_>>()
    } { return Value::Expr { head: Box::new(Value::Symbol("NDTranspose".into())), args } }
    let in_strides = strides(&shape);
    let out_shape: Vec<usize> = perm.iter().map(|&i| shape[i]).collect();
    let out_strides = strides(&out_shape);
    // map output index to input index
    let total = total_elems(&out_shape);
    let mut out = vec![0f64; total];
    for linear in 0..total {
        // decode linear into coords in out
        let mut rem = linear;
        let mut coords = vec![0usize; out_shape.len()];
        for i in 0..out_shape.len() {
            let stride_tail = out_shape[i+1..].iter().product::<usize>().max(1);
            coords[i] = rem / stride_tail; rem %= stride_tail;
        }
        let mut in_coords = vec![0usize; ndim];
        for (o,i_ax) in coords.iter().zip(perm.iter()) { in_coords[*i_ax] = *o; }
        let off_in = idx_of(&in_coords, &in_strides);
        out[linear] = data[off_in];
    }
    pack(out_shape, out)
}

fn nd_concat(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // NDConcat[{a,b,...}, axis]
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("NDConcat".into())), args } }
    let list = ev.eval(args[0].clone());
    let axis = match ev.eval(args[1].clone()) { Value::Integer(n) => n as isize, _ => -1 };
    let items: Vec<(Vec<usize>, Vec<f64>)> = match list { Value::List(xs) => xs.into_iter().filter_map(|v| as_packed(ev, v)).collect(), _ => Vec::new() };
    if items.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("NDConcat".into())), args } }
    let ndim = items[0].0.len();
    let ax = if axis < 0 { 0 } else { axis as usize };
    if ax >= ndim { return Value::Expr { head: Box::new(Value::Symbol("NDConcat".into())), args } }
    for (sh, _) in &items { if sh.len()!=ndim { return Value::Expr { head: Box::new(Value::Symbol("NDConcat".into())), args } } }
    let mut out_shape = items[0].0.clone();
    let mut cat_dim = 0usize;
    for (sh, _) in &items { for i in 0..ndim { if i==ax { cat_dim += sh[i]; } else if sh[i] != out_shape[i] { return Value::Expr { head: Box::new(Value::Symbol("NDConcat".into())), args } } } }
    out_shape[ax] = cat_dim;
    // simple: iterate over blocks along axis
    let mut out = Vec::with_capacity(total_elems(&out_shape));
    for (sh, data) in items.into_iter() {
        // The data is already row-major; appending along axis requires regrouping if axis != last.
        if ax == ndim - 1 {
            out.extend(data);
        } else {
            // stride over slices along axis and append slices
            let st = strides(&sh);
            let outer: usize = sh[..ax].iter().product();
            let inner: usize = sh[ax+1..].iter().product();
            let len = sh[ax];
            for o in 0..outer {
                for k in 0..len {
                    let base = o * st[0] + k * st[ax];
                    for i in 0..inner { out.push(data[base + i]); }
                }
            }
        }
    }
    pack(out_shape, out)
}

fn nd_sum(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // NDSum[a, axis?]
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("NDSum".into())), args } }
    let (shape, data) = match as_packed(ev, args[0].clone()) { Some(x)=>x, None=> return Value::Expr { head: Box::new(Value::Symbol("NDSum".into())), args } };
    if args.len()==1 { return Value::Real(data.iter().copied().sum()); }
    let axis = match ev.eval(args[1].clone()) { Value::Integer(n) => n as isize, _ => -1 };
    if axis < 0 || (axis as usize) >= shape.len() { return Value::Expr { head: Box::new(Value::Symbol("NDSum".into())), args } }
    let ax = axis as usize;
    let out_shape: Vec<usize> = shape.iter().enumerate().filter_map(|(i,&d)| if i==ax { None } else { Some(d) }).collect();
    let st = strides(&shape);
    let outer: usize = shape[..ax].iter().product();
    let inner: usize = shape[ax+1..].iter().product();
    let len = shape[ax];
    let mut out = vec![0f64; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            let mut acc = 0f64;
            for k in 0..len { let idx = o*st[0] + k*st[ax] + i; acc += data[idx]; }
            out[o*inner + i] = acc;
        }
    }
    if out_shape.is_empty() { Value::Real(out[0]) } else { pack(out_shape, out) }
}

fn nd_mean(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("NDMean".into())), args } }
    let (shape, data) = match as_packed(ev, args[0].clone()) { Some(x)=>x, None=> return Value::Expr { head: Box::new(Value::Symbol("NDMean".into())), args } };
    if args.len()==1 { return Value::Real(data.iter().copied().sum::<f64>() / (data.len() as f64)); }
    let axis = match ev.eval(args[1].clone()) { Value::Integer(n) => n as isize, _ => -1 };
    if axis < 0 || (axis as usize) >= shape.len() { return Value::Expr { head: Box::new(Value::Symbol("NDMean".into())), args } }
    let ax = axis as usize;
    let sum = nd_sum(ev, vec![pack(shape.clone(), data.clone()), Value::Integer(ax as i64)]);
    match sum {
        Value::PackedArray { shape: osh, data: od } => pack(osh, od.into_iter().map(|x| x / (shape[ax] as f64)).collect()),
        Value::Real(x) => Value::Real(x / (shape[ax] as f64)),
        other => other,
    }
}

fn nd_argmax(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // NDArgMax[a, axis?] returns indices along axis (or linear index when scalar)
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("NDArgMax".into())), args } }
    let (shape, data) = match as_packed(ev, args[0].clone()) { Some(x)=>x, None=> return Value::Expr { head: Box::new(Value::Symbol("NDArgMax".into())), args } };
    if shape.is_empty() { return Value::Integer(1); }
    if args.len()==1 {
        let mut best = 0usize; let mut bestv = data[0];
        for (i,&x) in data.iter().enumerate() { if x > bestv { best=i; bestv=x; } }
        return Value::Integer((best as i64) + 1);
    }
    let axis = match ev.eval(args[1].clone()) { Value::Integer(n) => n as isize, _ => -1 };
    if axis < 0 || (axis as usize) >= shape.len() { return Value::Expr { head: Box::new(Value::Symbol("NDArgMax".into())), args } }
    let ax = axis as usize;
    let out_shape: Vec<usize> = shape.iter().enumerate().filter_map(|(i,&d)| if i==ax { None } else { Some(d) }).collect();
    let st = strides(&shape);
    let outer: usize = shape[..ax].iter().product();
    let inner: usize = shape[ax+1..].iter().product();
    let len = shape[ax];
    let mut out_idx = vec![0f64; outer*inner];
    for o in 0..outer { for i in 0..inner {
        let mut best = 0usize; let mut bestv = f64::NEG_INFINITY;
        for k in 0..len { let idx = o*st[0] + k*st[ax] + i; let x = data[idx]; if x > bestv { bestv = x; best = k; } }
        out_idx[o*inner + i] = (best as f64) + 1.0; // 1-based
    } }
    if out_shape.is_empty() { Value::Integer(out_idx[0] as i64) } else { pack(out_shape, out_idx) }
}

fn nd_matmul(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("NDMatMul".into())), args } }
    let (a_sh, a_dat) = match as_packed(ev, args[0].clone()) { Some(x)=>x, None=> return Value::Expr { head: Box::new(Value::Symbol("NDMatMul".into())), args } };
    let (b_sh, b_dat) = match as_packed(ev, args[1].clone()) { Some(x)=>x, None=> return Value::Expr { head: Box::new(Value::Symbol("NDMatMul".into())), args } };
    if a_sh.len()!=2 || b_sh.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("NDMatMul".into())), args } }
    let (m, k) = (a_sh[0], a_sh[1]);
    let (k2, n) = (b_sh[0], b_sh[1]);
    if k != k2 { return Value::Expr { head: Box::new(Value::Symbol("NDMatMul".into())), args } }
    let mut out = vec![0f64; m*n];
    for i in 0..m { for j in 0..n { let mut acc = 0f64; for t in 0..k { acc += a_dat[i*k + t] * b_dat[t*n + j]; } out[i*n + j] = acc; } }
    pack(vec![m, n], out)
}

fn nd_slice(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Forms:
    // - NDSlice[a, axis, start, len]  (axis 0-based; start 1-based; len>=0)
    // - NDSlice[a, {start, len}]      (1D only; start 1-based)
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("NDSlice".into())), args } }
    let (shape, data) = match as_packed(ev, args[0].clone()) { Some(x)=>x, None=> return Value::Expr { head: Box::new(Value::Symbol("NDSlice".into())), args } };
    let ndim = shape.len();
    if args.len()==2 {
        // 1D form: list {start,len}
        match ev.eval(args[1].clone()) {
            Value::List(xs) if xs.len()==2 && ndim==1 => {
                let start = match xs[0] { Value::Integer(n)=>n, _=> return Value::Expr { head: Box::new(Value::Symbol("NDSlice".into())), args } };
                let len = match xs[1] { Value::Integer(n)=>n, _=> return Value::Expr { head: Box::new(Value::Symbol("NDSlice".into())), args } };
                if start < 1 || len < 0 { return Value::Expr { head: Box::new(Value::Symbol("NDSlice".into())), args } }
                let s0 = (start as usize).saturating_sub(1);
                let l = len as usize;
                if s0 + l > shape[0] { return Value::Expr { head: Box::new(Value::Symbol("NDSlice".into())), args } }
                let out = data[s0..s0+l].to_vec();
                return pack(vec![l], out);
            }
            _ => {}
        }
    }
    if args.len()!=4 { return Value::Expr { head: Box::new(Value::Symbol("NDSlice".into())), args } }
    let axis = match ev.eval(args[1].clone()) { Value::Integer(n)=> n as isize, _=> return Value::Expr { head: Box::new(Value::Symbol("NDSlice".into())), args } };
    let start = match ev.eval(args[2].clone()) { Value::Integer(n)=> n, _=> return Value::Expr { head: Box::new(Value::Symbol("NDSlice".into())), args } };
    let len = match ev.eval(args[3].clone()) { Value::Integer(n)=> n, _=> return Value::Expr { head: Box::new(Value::Symbol("NDSlice".into())), args } };
    if axis < 0 || (axis as usize) >= ndim || start < 1 || len < 0 { return Value::Expr { head: Box::new(Value::Symbol("NDSlice".into())), args } }
    let ax = axis as usize;
    let s0 = (start as usize).saturating_sub(1);
    let l = len as usize;
    if s0 + l > shape[ax] { return Value::Expr { head: Box::new(Value::Symbol("NDSlice".into())), args } }
    let st = strides(&shape);
    let outer: usize = shape[..ax].iter().product();
    let inner: usize = shape[ax+1..].iter().product();
    let mut out_shape = shape.clone(); out_shape[ax] = l;
    let mut out = Vec::with_capacity(total_elems(&out_shape));
    for o in 0..outer { for k in 0..l { let base = o*st[0] + (s0+k)*st[ax]; for i in 0..inner { out.push(data[base + i]); } } }
    pack(out_shape, out)
}

fn nd_permute_dims(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Alias to NDTranspose with explicit permutation
    match args.as_slice() {
        [a, perm] => nd_transpose(ev, vec![a.clone(), perm.clone()]),
        _ => Value::Expr { head: Box::new(Value::Symbol("NDPermuteDims".into())), args },
    }
}

pub fn register_ndarray(ev: &mut Evaluator) {
    ev.register("NDArray", ndarray as NativeFn, Attributes::HOLD_ALL);
    ev.register("NDShape", nd_shape as NativeFn, Attributes::empty());
    ev.register("NDReshape", nd_reshape as NativeFn, Attributes::empty());
    ev.register("NDTranspose", nd_transpose as NativeFn, Attributes::empty());
    ev.register("NDConcat", nd_concat as NativeFn, Attributes::empty());
    ev.register("NDSum", nd_sum as NativeFn, Attributes::empty());
    ev.register("NDMean", nd_mean as NativeFn, Attributes::empty());
    ev.register("NDArgMax", nd_argmax as NativeFn, Attributes::empty());
    ev.register("NDMatMul", nd_matmul as NativeFn, Attributes::empty());
    ev.register("NDType", nd_type as NativeFn, Attributes::empty());
    ev.register("NDAsType", nd_as_type as NativeFn, Attributes::empty());
    ev.register("NDSlice", nd_slice as NativeFn, Attributes::empty());
    ev.register("NDPermuteDims", nd_permute_dims as NativeFn, Attributes::empty());
}
