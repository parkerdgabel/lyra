use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

fn as_packed(ev: &mut Evaluator, v: Value) -> Option<(Vec<usize>, Vec<f64>)> {
    match ev.eval(v) {
        Value::PackedArray { shape, data } => Some((shape, data)),
        other => match ev.eval(Value::Expr {
            head: Box::new(Value::Symbol("PackedArray".into())),
            args: vec![other],
        }) {
            Value::PackedArray { shape, data } => Some((shape, data)),
            _ => None,
        },
    }
}

fn pack(shape: Vec<usize>, data: Vec<f64>) -> Value {
    Value::PackedArray { shape, data }
}

fn total_elems(shape: &[usize]) -> usize {
    shape.iter().product::<usize>()
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    let mut st = vec![0; n];
    let mut acc = 1usize;
    for i in (0..n).rev() {
        st[i] = acc;
        acc *= shape[i];
    }
    st
}

fn idx_of(idx: &[usize], st: &[usize]) -> usize {
    idx.iter().zip(st.iter()).map(|(i, s)| i * s).sum()
}

fn ndarray(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // NDArray[x, opts?] -> PackedArray[x]
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("NDArray".into())), args };
    }
    match ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("PackedArray".into())),
        args: vec![args[0].clone()],
    }) {
        Value::PackedArray { shape, data } => pack(shape, data),
        other => other,
    }
}

fn nd_shape(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("NDShape".into())), args };
    }
    match as_packed(ev, args[0].clone()) {
        Some((shape, _)) => {
            Value::List(shape.into_iter().map(|d| Value::Integer(d as i64)).collect())
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("NDShape".into())), args },
    }
}

fn nd_type(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("NDType".into())), args };
    }
    match as_packed(ev, args[0].clone()) {
        Some(_) => Value::String("Float64".into()),
        _ => Value::Expr { head: Box::new(Value::Symbol("NDType".into())), args },
    }
}

fn nd_as_type(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Only Float64 supported; passthrough
    if args.len() < 1 {
        return Value::Expr { head: Box::new(Value::Symbol("NDAsType".into())), args };
    }
    ev.eval(args[0].clone())
}

fn nd_reshape(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // NDReshape[a, {dims...}] supports one -1 to infer
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("NDReshape".into())), args };
    }
    let (shape, data) = match as_packed(ev, args[0].clone()) {
        Some(x) => x,
        None => return Value::Expr { head: Box::new(Value::Symbol("NDReshape".into())), args },
    };
    let new_shape_v = ev.eval(args[1].clone());
    let dims: Vec<i64> = match &new_shape_v {
        Value::List(xs) => xs
            .iter()
            .filter_map(|v| if let Value::Integer(n) = v { Some(*n) } else { None })
            .collect(),
        _ => Vec::new(),
    };
    if dims.is_empty() {
        return Value::Expr {
            head: Box::new(Value::Symbol("NDReshape".into())),
            args: vec![pack(shape, data), new_shape_v],
        };
    }
    let mut out_shape: Vec<usize> = Vec::with_capacity(dims.len());
    let mut infer_at: Option<usize> = None;
    for (i, d) in dims.iter().enumerate() {
        if *d == -1 {
            if infer_at.is_some() {
                return Value::Expr { head: Box::new(Value::Symbol("NDReshape".into())), args };
            }
            infer_at = Some(i);
            out_shape.push(1);
        } else if *d > 0 {
            out_shape.push(*d as usize);
        } else {
            return Value::Expr { head: Box::new(Value::Symbol("NDReshape".into())), args };
        }
    }
    let total_old = data.len();
    let known: usize = out_shape.iter().product();
    if let Some(pos) = infer_at {
        if known == 0 || total_old % known != 0 {
            return Value::Expr { head: Box::new(Value::Symbol("NDReshape".into())), args };
        } else {
            out_shape[pos] = total_old / known;
        }
    }
    if total_old != out_shape.iter().product::<usize>() {
        return Value::Expr { head: Box::new(Value::Symbol("NDReshape".into())), args };
    }
    pack(out_shape, data)
}

fn nd_transpose(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // NDTranspose[a] (reverse axes) or NDTranspose[a, {perm...}]
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("NDTranspose".into())), args };
    }
    let (shape, data) = match as_packed(ev, args[0].clone()) {
        Some(x) => x,
        None => return Value::Expr { head: Box::new(Value::Symbol("NDTranspose".into())), args },
    };
    let ndim = shape.len();
    let perm: Vec<usize> = if args.len() >= 2 {
        match ev.eval(args[1].clone()) {
            Value::List(xs) => xs
                .into_iter()
                .filter_map(|v| if let Value::Integer(n) = v { Some(n as usize) } else { None })
                .collect(),
            _ => Vec::new(),
        }
    } else {
        (0..ndim).rev().collect()
    };
    if perm.len() != ndim || {
        let mut s = perm.clone();
        s.sort_unstable();
        s != (0..ndim).collect::<Vec<_>>()
    } {
        return Value::Expr { head: Box::new(Value::Symbol("NDTranspose".into())), args };
    }
    let in_strides = strides(&shape);
    let out_shape: Vec<usize> = perm.iter().map(|&i| shape[i]).collect();
    let _out_strides = strides(&out_shape);
    // map output index to input index
    let total = total_elems(&out_shape);
    let mut out = vec![0f64; total];
    for linear in 0..total {
        // decode linear into coords in out
        let mut rem = linear;
        let mut coords = vec![0usize; out_shape.len()];
        for i in 0..out_shape.len() {
            let stride_tail = out_shape[i + 1..].iter().product::<usize>().max(1);
            coords[i] = rem / stride_tail;
            rem %= stride_tail;
        }
        let mut in_coords = vec![0usize; ndim];
        for (o, i_ax) in coords.iter().zip(perm.iter()) {
            in_coords[*i_ax] = *o;
        }
        let off_in = idx_of(&in_coords, &in_strides);
        out[linear] = data[off_in];
    }
    pack(out_shape, out)
}

fn nd_concat(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // NDConcat[{a,b,...}, axis]
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("NDConcat".into())), args };
    }
    let list = ev.eval(args[0].clone());
    let axis = match ev.eval(args[1].clone()) {
        Value::Integer(n) => n as isize,
        _ => -1,
    };
    let items: Vec<(Vec<usize>, Vec<f64>)> = match list {
        Value::List(xs) => xs.into_iter().filter_map(|v| as_packed(ev, v)).collect(),
        _ => Vec::new(),
    };
    if items.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("NDConcat".into())), args };
    }
    let ndim = items[0].0.len();
    let ax = if axis < 0 { 0 } else { axis as usize };
    if ax >= ndim {
        return Value::Expr { head: Box::new(Value::Symbol("NDConcat".into())), args };
    }
    for (sh, _) in &items {
        if sh.len() != ndim {
            return Value::Expr { head: Box::new(Value::Symbol("NDConcat".into())), args };
        }
    }
    let mut out_shape = items[0].0.clone();
    let mut cat_dim = 0usize;
    for (sh, _) in &items {
        for i in 0..ndim {
            if i == ax {
                cat_dim += sh[i];
            } else if sh[i] != out_shape[i] {
                return Value::Expr { head: Box::new(Value::Symbol("NDConcat".into())), args };
            }
        }
    }
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
            let inner: usize = sh[ax + 1..].iter().product();
            let len = sh[ax];
            for o in 0..outer {
                for k in 0..len {
                    let base = o * st[0] + k * st[ax];
                    for i in 0..inner {
                        out.push(data[base + i]);
                    }
                }
            }
        }
    }
    pack(out_shape, out)
}

fn nd_sum(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // NDSum[a, axis?]
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("NDSum".into())), args };
    }
    let (shape, data) = match as_packed(ev, args[0].clone()) {
        Some(x) => x,
        None => return Value::Expr { head: Box::new(Value::Symbol("NDSum".into())), args },
    };
    if args.len() == 1 {
        return Value::Real(data.iter().copied().sum());
    }
    let axis = match ev.eval(args[1].clone()) {
        Value::Integer(n) => n as isize,
        _ => -1,
    };
    if axis < 0 || (axis as usize) >= shape.len() {
        return Value::Expr { head: Box::new(Value::Symbol("NDSum".into())), args };
    }
    let ax = axis as usize;
    let out_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| if i == ax { None } else { Some(d) })
        .collect();
    let st = strides(&shape);
    let outer: usize = shape[..ax].iter().product();
    let inner: usize = shape[ax + 1..].iter().product();
    let len = shape[ax];
    let mut out = vec![0f64; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            let mut acc = 0f64;
            for k in 0..len {
                let idx = o * st[0] + k * st[ax] + i;
                acc += data[idx];
            }
            out[o * inner + i] = acc;
        }
    }
    if out_shape.is_empty() {
        Value::Real(out[0])
    } else {
        pack(out_shape, out)
    }
}

fn nd_mean(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("NDMean".into())), args };
    }
    let (shape, data) = match as_packed(ev, args[0].clone()) {
        Some(x) => x,
        None => return Value::Expr { head: Box::new(Value::Symbol("NDMean".into())), args },
    };
    if args.len() == 1 {
        return Value::Real(data.iter().copied().sum::<f64>() / (data.len() as f64));
    }
    let axis = match ev.eval(args[1].clone()) {
        Value::Integer(n) => n as isize,
        _ => -1,
    };
    if axis < 0 || (axis as usize) >= shape.len() {
        return Value::Expr { head: Box::new(Value::Symbol("NDMean".into())), args };
    }
    let ax = axis as usize;
    let sum = nd_sum(ev, vec![pack(shape.clone(), data.clone()), Value::Integer(ax as i64)]);
    match sum {
        Value::PackedArray { shape: osh, data: od } => {
            pack(osh, od.into_iter().map(|x| x / (shape[ax] as f64)).collect())
        }
        Value::Real(x) => Value::Real(x / (shape[ax] as f64)),
        other => other,
    }
}

fn nd_argmax(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // NDArgMax[a, axis?] returns indices along axis (or linear index when scalar)
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("NDArgMax".into())), args };
    }
    let (shape, data) = match as_packed(ev, args[0].clone()) {
        Some(x) => x,
        None => return Value::Expr { head: Box::new(Value::Symbol("NDArgMax".into())), args },
    };
    if shape.is_empty() {
        return Value::Integer(1);
    }
    if args.len() == 1 {
        let mut best = 0usize;
        let mut bestv = data[0];
        for (i, &x) in data.iter().enumerate() {
            if x > bestv {
                best = i;
                bestv = x;
            }
        }
        return Value::Integer((best as i64) + 1);
    }
    let axis = match ev.eval(args[1].clone()) {
        Value::Integer(n) => n as isize,
        _ => -1,
    };
    if axis < 0 || (axis as usize) >= shape.len() {
        return Value::Expr { head: Box::new(Value::Symbol("NDArgMax".into())), args };
    }
    let ax = axis as usize;
    let out_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| if i == ax { None } else { Some(d) })
        .collect();
    let st = strides(&shape);
    let outer: usize = shape[..ax].iter().product();
    let inner: usize = shape[ax + 1..].iter().product();
    let len = shape[ax];
    let mut out_idx = vec![0f64; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            let mut best = 0usize;
            let mut bestv = f64::NEG_INFINITY;
            for k in 0..len {
                let idx = o * st[0] + k * st[ax] + i;
                let x = data[idx];
                if x > bestv {
                    bestv = x;
                    best = k;
                }
            }
            out_idx[o * inner + i] = (best as f64) + 1.0; // 1-based
        }
    }
    if out_shape.is_empty() {
        Value::Integer(out_idx[0] as i64)
    } else {
        pack(out_shape, out_idx)
    }
}

fn nd_matmul(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("NDMatMul".into())), args };
    }
    let (a_sh, a_dat) = match as_packed(ev, args[0].clone()) {
        Some(x) => x,
        None => return Value::Expr { head: Box::new(Value::Symbol("NDMatMul".into())), args },
    };
    let (b_sh, b_dat) = match as_packed(ev, args[1].clone()) {
        Some(x) => x,
        None => return Value::Expr { head: Box::new(Value::Symbol("NDMatMul".into())), args },
    };
    if a_sh.len() != 2 || b_sh.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("NDMatMul".into())), args };
    }
    let (m, k) = (a_sh[0], a_sh[1]);
    let (k2, n) = (b_sh[0], b_sh[1]);
    if k != k2 {
        return Value::Expr { head: Box::new(Value::Symbol("NDMatMul".into())), args };
    }
    let mut out = vec![0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f64;
            for t in 0..k {
                acc += a_dat[i * k + t] * b_dat[t * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    pack(vec![m, n], out)
}

fn map_unary(shape: Vec<usize>, data: Vec<f64>, f: impl Fn(f64) -> f64) -> Value {
    let out: Vec<f64> = data.into_iter().map(|x| f(x)).collect();
    pack(shape, out)
}

fn nd_pow(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // NDPow[a, b] elementwise power with broadcasting (uses powf)
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("NDPow".into())), args };
    }
    let a = ev.eval(args[0].clone());
    let b = ev.eval(args[1].clone());
    match (&a, &b) {
        (
            Value::PackedArray { shape: sa, data: da },
            Value::PackedArray { shape: sb, data: db },
        ) => broadcast_binop_arrays(sa, da, sb, db, |x, y| x.powf(y)).unwrap_or(Value::Expr {
            head: Box::new(Value::Symbol("NDPow".into())),
            args: vec![a, b],
        }),
        (Value::PackedArray { shape, data }, Value::Integer(k)) => {
            let kk = *k;
            if kk >= 0 {
                return pack(shape.clone(), data.iter().map(|x| x.powi(kk as i32)).collect());
            }
            let p = (-kk) as i32;
            return pack(shape.clone(), data.iter().map(|x| (1.0_f64) / (x.powi(p))).collect());
        }
        (Value::PackedArray { shape, data }, other) => {
            if let Some(s) = num_to_f64(other) {
                return pack(shape.clone(), data.iter().map(|x| x.powf(s)).collect());
            }
            Value::Expr { head: Box::new(Value::Symbol("NDPow".into())), args: vec![a, b] }
        }
        (Value::Integer(base_k), Value::PackedArray { shape, data }) => {
            let bk = *base_k as f64;
            return pack(shape.clone(), data.iter().map(|x| bk.powf(*x)).collect());
        }
        (other, Value::PackedArray { shape, data }) => {
            if let Some(s) = num_to_f64(other) {
                return pack(shape.clone(), data.iter().map(|x| s.powf(*x)).collect());
            }
            Value::Expr { head: Box::new(Value::Symbol("NDPow".into())), args: vec![a, b] }
        }
        _ => {
            // scalar^scalar
            match (&a, &b) {
                (Value::Integer(x), Value::Integer(y)) => {
                    if *y >= 0 {
                        Value::Integer(x.pow(*y as u32))
                    } else {
                        Value::Real((*x as f64).powf(*y as f64))
                    }
                }
                _ => match (num_to_f64(&a), num_to_f64(&b)) {
                    (Some(x), Some(y)) => Value::Real(x.powf(y)),
                    _ => Value::Expr {
                        head: Box::new(Value::Symbol("NDPow".into())),
                        args: vec![a, b],
                    },
                },
            }
        }
    }
}

fn nd_clip(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // NDClip[a, min, max] scalar bounds
    if args.len() != 3 {
        return Value::Expr { head: Box::new(Value::Symbol("NDClip".into())), args };
    }
    let (shape, data) = match as_packed(ev, args[0].clone()) {
        Some(x) => x,
        None => return Value::Expr { head: Box::new(Value::Symbol("NDClip".into())), args },
    };
    let mn = match ev.eval(args[1].clone()) {
        v => match num_to_f64(&v) {
            Some(x) => x,
            None => return Value::Expr { head: Box::new(Value::Symbol("NDClip".into())), args },
        },
    };
    let mx = match ev.eval(args[2].clone()) {
        v => match num_to_f64(&v) {
            Some(x) => x,
            None => return Value::Expr { head: Box::new(Value::Symbol("NDClip".into())), args },
        },
    };
    map_unary(shape, data, |x| x.max(mn).min(mx))
}

fn nd_relu(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("NDRelu".into())), args };
    }
    let (shape, data) = match as_packed(ev, args[0].clone()) {
        Some(x) => x,
        None => return Value::Expr { head: Box::new(Value::Symbol("NDRelu".into())), args },
    };
    map_unary(shape, data, |x| if x < 0.0 { 0.0 } else { x })
}

fn nd_unary(ev: &mut Evaluator, args: Vec<Value>, f: fn(f64) -> f64, name: &str) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol(name.into())), args };
    }
    let (shape, data) = match as_packed(ev, args[0].clone()) {
        Some(x) => x,
        None => return Value::Expr { head: Box::new(Value::Symbol(name.into())), args },
    };
    map_unary(shape, data, f)
}

fn nd_exp(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    nd_unary(ev, args, |x| x.exp(), "NDExp")
}
fn nd_sqrt(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    nd_unary(ev, args, |x| x.sqrt(), "NDSqrt")
}
fn nd_log(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    nd_unary(ev, args, |x| x.ln(), "NDLog")
}
fn nd_sin(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    nd_unary(ev, args, |x| x.sin(), "NDSin")
}
fn nd_cos(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    nd_unary(ev, args, |x| x.cos(), "NDCos")
}
fn nd_tanh(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    nd_unary(ev, args, |x| x.tanh(), "NDTanh")
}

fn num_to_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Integer(n) => Some(*n as f64),
        Value::Real(x) => Some(*x),
        Value::Rational { num, den } => {
            if *den != 0 {
                Some((*num as f64) / (*den as f64))
            } else {
                None
            }
        }
        Value::BigReal(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

fn broadcast_shapes(s1: &[usize], s2: &[usize]) -> Option<Vec<usize>> {
    let ndim = std::cmp::max(s1.len(), s2.len());
    let mut sh1 = vec![1; ndim];
    let mut sh2 = vec![1; ndim];
    sh1[ndim - s1.len()..].clone_from_slice(&s1);
    sh2[ndim - s2.len()..].clone_from_slice(&s2);
    let mut out = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let (a, b) = (sh1[i], sh2[i]);
        if a == b {
            out.push(a);
        } else if a == 1 {
            out.push(b);
        } else if b == 1 {
            out.push(a);
        } else {
            return None;
        }
    }
    Some(out)
}

fn broadcast_binop_arrays(
    a_sh: &[usize],
    a_dat: &[f64],
    b_sh: &[usize],
    b_dat: &[f64],
    op: fn(f64, f64) -> f64,
) -> Option<Value> {
    let out_shape = broadcast_shapes(a_sh, b_sh)?;
    let ndim = out_shape.len();
    let mut sh1 = vec![1; ndim];
    let mut sh2 = vec![1; ndim];
    sh1[ndim - a_sh.len()..].clone_from_slice(a_sh);
    sh2[ndim - b_sh.len()..].clone_from_slice(b_sh);
    let st1 = strides(&sh1);
    let st2 = strides(&sh2);
    let out_total: usize = out_shape.iter().product();
    // tails for decoding linear -> coords
    let mut tails: Vec<usize> = vec![1; ndim];
    for i in 0..ndim {
        tails[i] = out_shape[i + 1..].iter().product::<usize>().max(1);
    }
    let mut out = Vec::with_capacity(out_total);
    for lin in 0..out_total {
        let mut rem = lin;
        let mut off1 = 0usize;
        let mut off2 = 0usize;
        for i in 0..ndim {
            let c = rem / tails[i];
            rem %= tails[i];
            let c1 = if sh1[i] == 1 { 0 } else { c };
            let c2 = if sh2[i] == 1 { 0 } else { c };
            off1 += c1 * st1[i];
            off2 += c2 * st2[i];
        }
        out.push(op(a_dat[off1], b_dat[off2]));
    }
    Some(pack(out_shape, out))
}

fn nd_add(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("NDAdd".into())), args };
    }
    let a = ev.eval(args[0].clone());
    let b = ev.eval(args[1].clone());
    match (&a, &b) {
        (
            Value::PackedArray { shape: sa, data: da },
            Value::PackedArray { shape: sb, data: db },
        ) => broadcast_binop_arrays(sa, da, sb, db, |x, y| x + y).unwrap_or(Value::Expr {
            head: Box::new(Value::Symbol("NDAdd".into())),
            args: vec![a, b],
        }),
        (Value::PackedArray { shape, data }, other)
        | (other, Value::PackedArray { shape, data }) => {
            if let Some(s) = num_to_f64(other) {
                return pack(shape.clone(), data.iter().map(|x| x + s).collect());
            }
            Value::Expr { head: Box::new(Value::Symbol("NDAdd".into())), args: vec![a, b] }
        }
        _ => match (num_to_f64(&a), num_to_f64(&b)) {
            (Some(x), Some(y)) => Value::Real(x + y),
            _ => Value::Expr { head: Box::new(Value::Symbol("NDAdd".into())), args: vec![a, b] },
        },
    }
}

fn nd_sub(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("NDSub".into())), args };
    }
    let a = ev.eval(args[0].clone());
    let b = ev.eval(args[1].clone());
    match (&a, &b) {
        (
            Value::PackedArray { shape: sa, data: da },
            Value::PackedArray { shape: sb, data: db },
        ) => broadcast_binop_arrays(sa, da, sb, db, |x, y| x - y).unwrap_or(Value::Expr {
            head: Box::new(Value::Symbol("NDSub".into())),
            args: vec![a, b],
        }),
        (Value::PackedArray { shape, data }, other) => {
            if let Some(s) = num_to_f64(other) {
                return pack(shape.clone(), data.iter().map(|x| x - s).collect());
            }
            Value::Expr { head: Box::new(Value::Symbol("NDSub".into())), args: vec![a, b] }
        }
        (other, Value::PackedArray { shape, data }) => {
            if let Some(s) = num_to_f64(other) {
                return pack(shape.clone(), data.iter().map(|x| s - *x).collect());
            }
            Value::Expr { head: Box::new(Value::Symbol("NDSub".into())), args: vec![a, b] }
        }
        _ => match (num_to_f64(&a), num_to_f64(&b)) {
            (Some(x), Some(y)) => Value::Real(x - y),
            _ => Value::Expr { head: Box::new(Value::Symbol("NDSub".into())), args: vec![a, b] },
        },
    }
}

fn nd_mul(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("NDMul".into())), args };
    }
    let a = ev.eval(args[0].clone());
    let b = ev.eval(args[1].clone());
    match (&a, &b) {
        (
            Value::PackedArray { shape: sa, data: da },
            Value::PackedArray { shape: sb, data: db },
        ) => broadcast_binop_arrays(sa, da, sb, db, |x, y| x * y).unwrap_or(Value::Expr {
            head: Box::new(Value::Symbol("NDMul".into())),
            args: vec![a, b],
        }),
        (Value::PackedArray { shape, data }, other)
        | (other, Value::PackedArray { shape, data }) => {
            if let Some(s) = num_to_f64(other) {
                return pack(shape.clone(), data.iter().map(|x| x * s).collect());
            }
            Value::Expr { head: Box::new(Value::Symbol("NDMul".into())), args: vec![a, b] }
        }
        _ => match (num_to_f64(&a), num_to_f64(&b)) {
            (Some(x), Some(y)) => Value::Real(x * y),
            _ => Value::Expr { head: Box::new(Value::Symbol("NDMul".into())), args: vec![a, b] },
        },
    }
}

fn nd_div(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("NDDiv".into())), args };
    }
    let a = ev.eval(args[0].clone());
    let b = ev.eval(args[1].clone());
    match (&a, &b) {
        (
            Value::PackedArray { shape: sa, data: da },
            Value::PackedArray { shape: sb, data: db },
        ) => broadcast_binop_arrays(sa, da, sb, db, |x, y| x / y).unwrap_or(Value::Expr {
            head: Box::new(Value::Symbol("NDDiv".into())),
            args: vec![a, b],
        }),
        (Value::PackedArray { shape, data }, other) => {
            if let Some(s) = num_to_f64(other) {
                return pack(shape.clone(), data.iter().map(|x| x / s).collect());
            }
            Value::Expr { head: Box::new(Value::Symbol("NDDiv".into())), args: vec![a, b] }
        }
        (other, Value::PackedArray { shape, data }) => {
            if let Some(s) = num_to_f64(other) {
                return pack(shape.clone(), data.iter().map(|x| s / *x).collect());
            }
            Value::Expr { head: Box::new(Value::Symbol("NDDiv".into())), args: vec![a, b] }
        }
        _ => match (num_to_f64(&a), num_to_f64(&b)) {
            (Some(x), Some(y)) => Value::Real(x / y),
            _ => Value::Expr { head: Box::new(Value::Symbol("NDDiv".into())), args: vec![a, b] },
        },
    }
}

fn nd_eltwise(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // NDEltwise[f, a, b]
    if args.len() != 3 {
        return Value::Expr { head: Box::new(Value::Symbol("NDEltwise".into())), args };
    }
    let f = args[0].clone();
    let a = ev.eval(args[1].clone());
    let b = ev.eval(args[2].clone());
    match (&a, &b) {
        (
            Value::PackedArray { shape: sa, data: da },
            Value::PackedArray { shape: sb, data: db },
        ) => {
            let out_shape = match broadcast_shapes(sa, sb) {
                Some(s) => s,
                None => {
                    return Value::Expr {
                        head: Box::new(Value::Symbol("NDEltwise".into())),
                        args: vec![f, a, b],
                    }
                }
            };
            let ndim = out_shape.len();
            let mut sh1 = vec![1; ndim];
            let mut sh2 = vec![1; ndim];
            sh1[ndim - sa.len()..].clone_from_slice(sa);
            sh2[ndim - sb.len()..].clone_from_slice(sb);
            let st1 = strides(&sh1);
            let st2 = strides(&sh2);
            let mut tails = vec![1usize; ndim];
            for i in 0..ndim {
                tails[i] = out_shape[i + 1..].iter().product::<usize>().max(1);
            }
            let total: usize = out_shape.iter().product();
            let mut out = Vec::with_capacity(total);
            for lin in 0..total {
                let mut rem = lin;
                let mut off1 = 0usize;
                let mut off2 = 0usize;
                for i in 0..ndim {
                    let c = rem / tails[i];
                    rem %= tails[i];
                    let c1 = if sh1[i] == 1 { 0 } else { c };
                    let c2 = if sh2[i] == 1 { 0 } else { c };
                    off1 += c1 * st1[i];
                    off2 += c2 * st2[i];
                }
                if let Some(y) = call_binary_to_f64(ev, &f, da[off1], db[off2]) {
                    out.push(y);
                } else {
                    return Value::Expr {
                        head: Box::new(Value::Symbol("NDEltwise".into())),
                        args: vec![f, a, b],
                    };
                }
            }
            pack(out_shape, out)
        }
        (Value::PackedArray { shape, data }, other)
        | (other, Value::PackedArray { shape, data }) => {
            if let Some(s) = num_to_f64(other) {
                let mut out = Vec::with_capacity(data.len());
                for &x in data {
                    if let Some(y) = call_binary_to_f64(ev, &f, x, s) {
                        out.push(y);
                    } else {
                        return Value::Expr {
                            head: Box::new(Value::Symbol("NDEltwise".into())),
                            args: vec![f, a, b],
                        };
                    }
                }
                pack(shape.clone(), out)
            } else {
                Value::Expr {
                    head: Box::new(Value::Symbol("NDEltwise".into())),
                    args: vec![f, a, b],
                }
            }
        }
        _ => match (num_to_f64(&a), num_to_f64(&b)) {
            (Some(x), Some(y)) => {
                call_binary_to_f64(ev, &f, x, y).map(Value::Real).unwrap_or(Value::Expr {
                    head: Box::new(Value::Symbol("NDEltwise".into())),
                    args: vec![f, a, b],
                })
            }
            _ => Value::Expr {
                head: Box::new(Value::Symbol("NDEltwise".into())),
                args: vec![f, a, b],
            },
        },
    }
}

fn nd_slice(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Forms:
    // - NDSlice[a, axis, start, len]  (axis 0-based; start 1-based; len>=0)
    // - NDSlice[a, {start, len}]      (1D only; start 1-based)
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("NDSlice".into())), args };
    }
    let (shape, data) = match as_packed(ev, args[0].clone()) {
        Some(x) => x,
        None => return Value::Expr { head: Box::new(Value::Symbol("NDSlice".into())), args },
    };
    let ndim = shape.len();
    if args.len() == 2 {
        let spec_v = ev.eval(args[1].clone());
        // 1D form: list {start,len}
        if let Value::List(xs) = &spec_v {
            if xs.len() == 2 && ndim == 1 {
                let start = match xs[0] {
                    Value::Integer(n) => n,
                    _ => {
                        return Value::Expr {
                            head: Box::new(Value::Symbol("NDSlice".into())),
                            args,
                        }
                    }
                };
                let len = match xs[1] {
                    Value::Integer(n) => n,
                    _ => {
                        return Value::Expr {
                            head: Box::new(Value::Symbol("NDSlice".into())),
                            args,
                        }
                    }
                };
                if start < 1 || len < 0 {
                    return Value::Expr { head: Box::new(Value::Symbol("NDSlice".into())), args };
                }
                let s0 = (start as usize).saturating_sub(1);
                let l = len as usize;
                if s0 + l > shape[0] {
                    return Value::Expr { head: Box::new(Value::Symbol("NDSlice".into())), args };
                }
                let out = data[s0..s0 + l].to_vec();
                return pack(vec![l], out);
            }
        }
        // Multi-axis spec: list of per-axis specs
        if let Value::List(specs) = spec_v {
            if specs.len() != ndim {
                return Value::Expr { head: Box::new(Value::Symbol("NDSlice".into())), args };
            }
            enum AxSpec {
                All,
                Index(usize),
                Range(usize, usize),
            }
            let mut parsed: Vec<AxSpec> = Vec::with_capacity(ndim);
            for (i, sp) in specs.iter().enumerate() {
                match sp {
                    Value::Symbol(s) if s == "All" => parsed.push(AxSpec::All),
                    Value::Integer(n) => {
                        if *n < 1 || (*n as usize) > shape[i] {
                            return Value::Expr {
                                head: Box::new(Value::Symbol("NDSlice".into())),
                                args,
                            };
                        }
                        parsed.push(AxSpec::Index((*n as usize) - 1));
                    }
                    Value::List(xs) if xs.len() == 2 => {
                        let start = match xs[0] {
                            Value::Integer(n) => n,
                            _ => {
                                return Value::Expr {
                                    head: Box::new(Value::Symbol("NDSlice".into())),
                                    args,
                                }
                            }
                        };
                        let len = match xs[1] {
                            Value::Integer(n) => n,
                            _ => {
                                return Value::Expr {
                                    head: Box::new(Value::Symbol("NDSlice".into())),
                                    args,
                                }
                            }
                        };
                        if start < 1 || len < 0 {
                            return Value::Expr {
                                head: Box::new(Value::Symbol("NDSlice".into())),
                                args,
                            };
                        }
                        let s0 = (start as usize).saturating_sub(1);
                        let l = len as usize;
                        if s0 + l > shape[i] {
                            return Value::Expr {
                                head: Box::new(Value::Symbol("NDSlice".into())),
                                args,
                            };
                        }
                        parsed.push(AxSpec::Range(s0, l));
                    }
                    _ => {
                        return Value::Expr {
                            head: Box::new(Value::Symbol("NDSlice".into())),
                            args,
                        }
                    }
                }
            }
            // Compute output shape
            let mut out_shape: Vec<usize> = Vec::new();
            for sp in &parsed {
                match sp {
                    AxSpec::All => out_shape.push(0),
                    AxSpec::Range(_, l) => out_shape.push(*l),
                    AxSpec::Index(_) => {}
                }
            }
            // Fill 'All' dims from input shape
            let mut out_dims: Vec<usize> = Vec::new();
            for (i, sp) in parsed.iter().enumerate() {
                match sp {
                    AxSpec::All => out_dims.push(shape[i]),
                    AxSpec::Range(_, l) => out_dims.push(*l),
                    AxSpec::Index(_) => {}
                }
            }
            // out_dims is the actual out shape
            let out_shape = out_dims;
            let out_total: usize =
                if out_shape.is_empty() { 1 } else { out_shape.iter().product() };
            // Precompute tail products for decoding out coords
            let mut tails: Vec<usize> = vec![1; out_shape.len()];
            for i in 0..out_shape.len() {
                tails[i] = out_shape[i + 1..].iter().product::<usize>().max(1);
            }
            let in_strides = strides(&shape);
            let mut out_data: Vec<f64> = Vec::with_capacity(out_total);
            for lin in 0..out_total {
                // build out coords
                let mut rem = lin;
                let mut out_coords = vec![0usize; out_shape.len()];
                for i in 0..out_shape.len() {
                    let denom = tails[i];
                    let c = rem / denom;
                    rem %= denom;
                    out_coords[i] = c;
                }
                // map to in coords
                let mut in_coords = vec![0usize; ndim];
                let mut oi = 0usize;
                for (ax, sp) in parsed.iter().enumerate() {
                    match sp {
                        AxSpec::All => {
                            in_coords[ax] = out_coords[oi];
                            oi += 1;
                        }
                        AxSpec::Range(s0, _l) => {
                            in_coords[ax] = s0 + out_coords[oi];
                            oi += 1;
                        }
                        AxSpec::Index(idx) => {
                            in_coords[ax] = *idx;
                        }
                    }
                }
                let off = idx_of(&in_coords, &in_strides);
                out_data.push(data[off]);
            }
            return if out_shape.is_empty() {
                Value::Real(out_data[0])
            } else {
                pack(out_shape, out_data)
            };
        }
    }
    if args.len() != 4 {
        return Value::Expr { head: Box::new(Value::Symbol("NDSlice".into())), args };
    }
    let axis = match ev.eval(args[1].clone()) {
        Value::Integer(n) => n as isize,
        _ => return Value::Expr { head: Box::new(Value::Symbol("NDSlice".into())), args },
    };
    let start = match ev.eval(args[2].clone()) {
        Value::Integer(n) => n,
        _ => return Value::Expr { head: Box::new(Value::Symbol("NDSlice".into())), args },
    };
    let len = match ev.eval(args[3].clone()) {
        Value::Integer(n) => n,
        _ => return Value::Expr { head: Box::new(Value::Symbol("NDSlice".into())), args },
    };
    if axis < 0 || (axis as usize) >= ndim || start < 1 || len < 0 {
        return Value::Expr { head: Box::new(Value::Symbol("NDSlice".into())), args };
    }
    let ax = axis as usize;
    let s0 = (start as usize).saturating_sub(1);
    let l = len as usize;
    if s0 + l > shape[ax] {
        return Value::Expr { head: Box::new(Value::Symbol("NDSlice".into())), args };
    }
    let st = strides(&shape);
    let outer: usize = shape[..ax].iter().product();
    let inner: usize = shape[ax + 1..].iter().product();
    let mut out_shape = shape.clone();
    out_shape[ax] = l;
    let mut out = Vec::with_capacity(total_elems(&out_shape));
    for o in 0..outer {
        for k in 0..l {
            let base = o * st[0] + (s0 + k) * st[ax];
            for i in 0..inner {
                out.push(data[base + i]);
            }
        }
    }
    pack(out_shape, out)
}

fn nd_permute_dims(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Alias to NDTranspose with explicit permutation
    match args.as_slice() {
        [a, perm] => nd_transpose(ev, vec![a.clone(), perm.clone()]),
        _ => Value::Expr { head: Box::new(Value::Symbol("NDPermuteDims".into())), args },
    }
}

use crate::register_if;

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
    ev.register("NDMap", nd_map as NativeFn, Attributes::HOLD_ALL);
    ev.register("NDReduce", nd_reduce as NativeFn, Attributes::HOLD_ALL);
    ev.register("NDAdd", nd_add as NativeFn, Attributes::empty());
    ev.register("NDSub", nd_sub as NativeFn, Attributes::empty());
    ev.register("NDMul", nd_mul as NativeFn, Attributes::empty());
    ev.register("NDDiv", nd_div as NativeFn, Attributes::empty());
    ev.register("NDEltwise", nd_eltwise as NativeFn, Attributes::HOLD_ALL);
    ev.register("NDPow", nd_pow as NativeFn, Attributes::empty());
    ev.register("NDClip", nd_clip as NativeFn, Attributes::empty());
    ev.register("NDRelu", nd_relu as NativeFn, Attributes::empty());
    ev.register("NDExp", nd_exp as NativeFn, Attributes::empty());
    ev.register("NDSqrt", nd_sqrt as NativeFn, Attributes::empty());
    ev.register("NDLog", nd_log as NativeFn, Attributes::empty());
    ev.register("NDSin", nd_sin as NativeFn, Attributes::empty());
    ev.register("NDCos", nd_cos as NativeFn, Attributes::empty());
    ev.register("NDTanh", nd_tanh as NativeFn, Attributes::empty());
}

pub fn register_ndarray_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    register_if(ev, pred, "NDArray", ndarray as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "NDShape", nd_shape as NativeFn, Attributes::empty());
    register_if(ev, pred, "NDReshape", nd_reshape as NativeFn, Attributes::empty());
    register_if(ev, pred, "NDTranspose", nd_transpose as NativeFn, Attributes::empty());
    register_if(ev, pred, "NDConcat", nd_concat as NativeFn, Attributes::empty());
    register_if(ev, pred, "NDSum", nd_sum as NativeFn, Attributes::empty());
    register_if(ev, pred, "NDMean", nd_mean as NativeFn, Attributes::empty());
    register_if(ev, pred, "NDArgMax", nd_argmax as NativeFn, Attributes::empty());
    register_if(ev, pred, "NDMatMul", nd_matmul as NativeFn, Attributes::empty());
    register_if(ev, pred, "NDType", nd_type as NativeFn, Attributes::empty());
    register_if(ev, pred, "NDAsType", nd_as_type as NativeFn, Attributes::empty());
    register_if(ev, pred, "NDSlice", nd_slice as NativeFn, Attributes::empty());
    register_if(ev, pred, "NDPermuteDims", nd_permute_dims as NativeFn, Attributes::empty());
    register_if(ev, pred, "NDMap", nd_map as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "NDReduce", nd_reduce as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "NDAdd", nd_add as NativeFn, Attributes::empty());
    register_if(ev, pred, "NDSub", nd_sub as NativeFn, Attributes::empty());
    register_if(ev, pred, "NDMul", nd_mul as NativeFn, Attributes::empty());
    register_if(ev, pred, "NDDiv", nd_div as NativeFn, Attributes::empty());
    register_if(ev, pred, "NDEltwise", nd_eltwise as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "NDPow", nd_pow as NativeFn, Attributes::empty());
    register_if(ev, pred, "NDClip", nd_clip as NativeFn, Attributes::empty());
    register_if(ev, pred, "NDRelu", nd_relu as NativeFn, Attributes::empty());
    register_if(ev, pred, "NDExp", nd_exp as NativeFn, Attributes::empty());
    register_if(ev, pred, "NDSqrt", nd_sqrt as NativeFn, Attributes::empty());
    register_if(ev, pred, "NDLog", nd_log as NativeFn, Attributes::empty());
    register_if(ev, pred, "NDSin", nd_sin as NativeFn, Attributes::empty());
    register_if(ev, pred, "NDCos", nd_cos as NativeFn, Attributes::empty());
    register_if(ev, pred, "NDTanh", nd_tanh as NativeFn, Attributes::empty());
}

fn call_unary_to_f64(ev: &mut Evaluator, f: &Value, x: f64) -> Option<f64> {
    let res = ev.eval(Value::Expr { head: Box::new(f.clone()), args: vec![Value::Real(x)] });
    match res {
        Value::Integer(n) => Some(n as f64),
        Value::Real(v) => Some(v),
        Value::Rational { num, den } => {
            if den != 0 {
                Some((num as f64) / (den as f64))
            } else {
                None
            }
        }
        Value::BigReal(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

fn call_binary_to_f64(ev: &mut Evaluator, f: &Value, a: f64, b: f64) -> Option<f64> {
    let res = ev.eval(Value::Expr {
        head: Box::new(f.clone()),
        args: vec![Value::Real(a), Value::Real(b)],
    });
    match res {
        Value::Integer(n) => Some(n as f64),
        Value::Real(v) => Some(v),
        Value::Rational { num, den } => {
            if den != 0 {
                Some((num as f64) / (den as f64))
            } else {
                None
            }
        }
        Value::BigReal(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

fn nd_map(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // NDMap[a, f]
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("NDMap".into())), args };
    }
    let (shape, data) = match as_packed(ev, args[0].clone()) {
        Some(x) => x,
        None => return Value::Expr { head: Box::new(Value::Symbol("NDMap".into())), args },
    };
    let f = args[1].clone();
    let mut out: Vec<f64> = Vec::with_capacity(data.len());
    for x in data {
        if let Some(y) = call_unary_to_f64(ev, &f, x) {
            out.push(y);
        } else {
            return Value::Expr {
                head: Box::new(Value::Symbol("NDMap".into())),
                args: vec![pack(shape, out), f],
            };
        }
    }
    pack(shape, out)
}

fn nd_reduce(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // NDReduce[a, f] or NDReduce[a, f, axis]
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("NDReduce".into())), args };
    }
    let (shape, data) = match as_packed(ev, args[0].clone()) {
        Some(x) => x,
        None => return Value::Expr { head: Box::new(Value::Symbol("NDReduce".into())), args },
    };
    let f = args[1].clone();
    if args.len() == 2 {
        if data.is_empty() {
            return Value::Expr { head: Box::new(Value::Symbol("NDReduce".into())), args };
        }
        let mut acc = data[0];
        for &x in &data[1..] {
            if let Some(y) = call_binary_to_f64(ev, &f, acc, x) {
                acc = y;
            } else {
                return Value::Expr { head: Box::new(Value::Symbol("NDReduce".into())), args };
            }
        }
        return Value::Real(acc);
    }
    let axis = match ev.eval(args[2].clone()) {
        Value::Integer(n) => n as isize,
        _ => -1,
    };
    if axis < 0 || (axis as usize) >= shape.len() {
        return Value::Expr { head: Box::new(Value::Symbol("NDReduce".into())), args };
    }
    let ax = axis as usize;
    let st = strides(&shape);
    let outer: usize = shape[..ax].iter().product();
    let inner: usize = shape[ax + 1..].iter().product();
    let len = shape[ax];
    let mut out = vec![0f64; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            let base0 = o * st[0] + 0 * st[ax] + i;
            let mut acc = data[base0];
            for k in 1..len {
                let idx = o * st[0] + k * st[ax] + i;
                let x = data[idx];
                if let Some(y) = call_binary_to_f64(ev, &f, acc, x) {
                    acc = y;
                } else {
                    return Value::Expr { head: Box::new(Value::Symbol("NDReduce".into())), args };
                }
            }
            out[o * inner + i] = acc;
        }
    }
    let out_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| if i == ax { None } else { Some(d) })
        .collect();
    if out_shape.is_empty() {
        Value::Real(out[0])
    } else {
        pack(out_shape, out)
    }
}
