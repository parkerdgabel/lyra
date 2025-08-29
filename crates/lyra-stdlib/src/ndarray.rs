use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

fn as_packed(ev: &mut Evaluator, v: Value) -> Option<(Vec<usize>, Vec<f64>)> {
    match ev.eval(v) {
        Value::PackedArray { shape, data } => Some((shape, data)),
        other => match ev.eval(Value::Expr {
            head: Box::new(Value::Symbol("__PackedArray".into())),
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
        head: Box::new(Value::Symbol("__PackedArray".into())),
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

// -------- Tier1 Linalg & FFT stubs (return unevaluated until implemented) --------
fn uneval(head: &str, args: Vec<Value>) -> Value {
    Value::Expr { head: Box::new(Value::Symbol(head.into())), args }
}

fn read_matrix(ev: &mut Evaluator, v: Value) -> Option<(usize, usize, Vec<f64>)> {
    match ev.eval(v) {
        Value::PackedArray { shape, data } => {
            if shape.len() == 2 { Some((shape[0], shape[1], data)) } else { None }
        }
        Value::List(rows) => {
            if rows.is_empty() { return None; }
            let mut out: Vec<f64> = Vec::new();
            let mut ncols: Option<usize> = None;
            for r in rows.iter() {
                if let Value::List(cols) = ev.eval(r.clone()) {
                    if ncols.is_none() { ncols = Some(cols.len()); }
                    if Some(cols.len()) != ncols { return None; }
                    for c in cols.iter() {
                        if let Some(x) = num_to_f64(&ev.eval(c.clone())) { out.push(x); } else { return None; }
                    }
                } else { return None; }
            }
            Some((rows.len(), ncols.unwrap_or(0), out))
        }
        _ => None,
    }
}

fn read_vector(ev: &mut Evaluator, v: Value) -> Option<Vec<f64>> {
    match ev.eval(v) {
        Value::PackedArray { shape, data } => {
            match shape.as_slice() {
                [n] => Some(data),
                [n, 1] => Some(data),
                _ => None,
            }
        }
        Value::List(xs) => {
            let mut out = Vec::with_capacity(xs.len());
            for it in xs {
                if let Some(x) = num_to_f64(&ev.eval(it)) { out.push(x); } else { return None; }
            }
            Some(out)
        }
        _ => None,
    }
}

fn determinant(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return uneval("Determinant", args); }
    let (n, m, mut a) = match read_matrix(ev, args[0].clone()) { Some(t) => t, None => return uneval("Determinant", args) };
    if n != m { return uneval("Determinant", args); }
    let mut sign = 1.0f64;
    let mut det = 1.0f64;
    let eps = 1e-12;
    for k in 0..n {
        // pivot
        let mut pivot = k; let mut maxv = a[k*m + k].abs();
        for i in (k+1)..n { let v = a[i*m + k].abs(); if v > maxv { maxv = v; pivot = i; } }
        if maxv < eps { return Value::Real(0.0); }
        if pivot != k { for j in 0..m { a.swap(k*m + j, pivot*m + j); } sign *= -1.0; }
        let akk = a[k*m + k]; det *= akk;
        if k+1 < n {
            for i in (k+1)..n {
                let f = a[i*m + k] / akk;
                for j in (k+1)..m { a[i*m + j] -= f * a[k*m + j]; }
            }
        }
    }
    Value::Real(sign * det)
}

fn inverse(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return uneval("Inverse", args); }
    let (n, m, a0) = match read_matrix(ev, args[0].clone()) { Some(t) => t, None => return uneval("Inverse", args) };
    if n != m { return uneval("Inverse", args); }
    let n = n; let m = m;
    let mut a = a0.clone();
    // augmented identity
    let mut b: Vec<f64> = vec![0.0; n*n];
    for i in 0..n { b[i*n + i] = 1.0; }
    let eps = 1e-12;
    for k in 0..n {
        // pivot
        let mut pivot = k; let mut maxv = a[k*m + k].abs();
        for i in (k+1)..n { let v = a[i*m + k].abs(); if v > maxv { maxv = v; pivot = i; } }
        if maxv < eps { return uneval("Inverse", args); }
        if pivot != k { for j in 0..m { a.swap(k*m + j, pivot*m + j); }
                        for j in 0..n { b.swap(k*n + j, pivot*n + j); } }
        // normalize row k
        let akk = a[k*m + k];
        for j in 0..m { a[k*m + j] /= akk; }
        for j in 0..n { b[k*n + j] /= akk; }
        // eliminate others
        for i in 0..n {
            if i == k { continue; }
            let f = a[i*m + k];
            for j in 0..m { a[i*m + j] -= f * a[k*m + j]; }
            for j in 0..n { b[i*n + j] -= f * b[k*n + j]; }
        }
    }
    pack(vec![n, n], b)
}

fn linear_solve(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return uneval("LinearSolve", args); }
    let (n, m, _a) = match read_matrix(ev, args[0].clone()) { Some(t) => t, None => return uneval("LinearSolve", args) };
    if n != m { return uneval("LinearSolve", args); }
    // Support vector or matrix RHS
    if let Some(mut bvec) = read_vector(ev, args[1].clone()) {
        if bvec.len() != n { return uneval("LinearSolve", args); }
        // Try Cholesky first (SPD); fall back to LU
        if let Some((nn, l)) = cholesky_decomp(ev, args[0].clone()) {
            if nn != n { return uneval("LinearSolve", args); }
            // Solve L y = b; then L^T x = y
            forward_subst_lower(n, &l, &mut bvec);
            // form U=L^T on the fly in back substitution
            // Using the same routine requires explicit upper; implement inline
            for ii in 0..n { let i = n-1-ii; let mut s=bvec[i]; for j in (i+1)..n { s -= l[j*n + i]*bvec[j]; } bvec[i] = s / l[i*n + i]; }
            return pack(vec![n], bvec);
        }
        if let Some((nn, lu, piv)) = lu_decomp(ev, args[0].clone()) {
            if nn != n { return uneval("LinearSolve", args); }
            lu_solve(n, &lu, &piv, &mut bvec);
            return pack(vec![n], bvec);
        }
        return uneval("LinearSolve", args);
    }
    if let Some((rb_m, rb_n, mut bmat)) = read_matrix(ev, args[1].clone()) {
        if rb_m != n { return uneval("LinearSolve", args); }
        // Solve for each RHS column
        if let Some((nn, l)) = cholesky_decomp(ev, args[0].clone()) {
            if nn != n { return uneval("LinearSolve", args); }
            for j in 0..rb_n {
                let mut col = vec![0.0; n];
                for i in 0..n { col[i] = bmat[i*rb_n + j]; }
                forward_subst_lower(n, &l, &mut col);
                for ii in 0..n { let i = n-1-ii; let mut s=col[i]; for jj in (i+1)..n { s -= l[jj*n + i]*col[jj]; } col[i] = s / l[i*n + i]; }
                for i in 0..n { bmat[i*rb_n + j] = col[i]; }
            }
            return pack(vec![n, rb_n], bmat);
        }
        if let Some((nn, lu, piv)) = lu_decomp(ev, args[0].clone()) {
            if nn != n { return uneval("LinearSolve", args); }
            for j in 0..rb_n {
                let mut col = vec![0.0; n];
                for i in 0..n { col[i] = bmat[i*rb_n + j]; }
                lu_solve(n, &lu, &piv, &mut col);
                for i in 0..n { bmat[i*rb_n + j] = col[i]; }
            }
            return pack(vec![n, rb_n], bmat);
        }
        return uneval("LinearSolve", args);
    }
    uneval("LinearSolve", args)
}
fn svd(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // One-pass SVD via eigen-decomposition of A^T A; returns reduced form <|U,S,V|>
    if args.len() != 1 { return uneval("SVD", args); }
    let (m, n, a) = match read_matrix(ev, args[0].clone()) { Some(t) => t, None => return uneval("SVD", args) };
    // Build AtA (n x n)
    let mut ata = vec![0.0; n*n];
    for i in 0..n { for j in 0..n { let mut s=0.0; for r in 0..m { s += a[r*n + i]*a[r*n + j]; } ata[i*n + j] = s; } }
    // Eigen decomposition of AtA (symmetric)
    let evres = eigen(ev, vec![pack(vec![n, n], ata.clone())]);
    let (evals, vdata) = match evres {
        Value::Assoc(mut map) => {
            let vvals = match map.remove("Eigenvalues") { Some(Value::PackedArray { data, .. }) => data, _ => return uneval("SVD", args) };
            let vvecs = match map.remove("Eigenvectors") { Some(Value::PackedArray { data, .. }) => data, _ => return uneval("SVD", args) };
            (vvals, vvecs)
        }
        _ => return uneval("SVD", args),
    };
    // Singular values are sqrt of eigenvalues (clamped to >=0). Sort descending.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        let si = if evals[i] > 0.0 { evals[i].sqrt() } else { 0.0 };
        let sj = if evals[j] > 0.0 { evals[j].sqrt() } else { 0.0 };
        sj.partial_cmp(&si).unwrap_or(std::cmp::Ordering::Equal)
    });
    let k = std::cmp::min(m, n);
    // Build reduced V (n x k) and S (k)
    let mut Vred = vec![0.0; n*k];
    let mut S = vec![0.0; k];
    for j in 0..k {
        let idx = order[j];
        S[j] = if evals[idx] > 0.0 { evals[idx].sqrt() } else { 0.0 };
        for i in 0..n { Vred[i*k + j] = vdata[i*n + idx]; }
    }
    // Build U as A * V / S
    let mut U = vec![0.0; m*k];
    for j in 0..k {
        let sj = S[j];
        if sj > 1e-15 {
            for i in 0..m {
                let mut s = 0.0; for t in 0..n { s += a[i*n + t] * Vred[t*k + j]; }
                U[i*k + j] = s / sj;
            }
        } else {
            for i in 0..m { U[i*k + j] = 0.0; }
        }
    }
    let mut assoc = std::collections::HashMap::new();
    assoc.insert("U".into(), pack(vec![m, k], U));
    assoc.insert("S".into(), pack(vec![k], S));
    assoc.insert("V".into(), pack(vec![n, k], Vred));
    Value::Assoc(assoc)
}
fn qr(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Householder QR: returns <|Q, R|> where A = Q R
    if args.len() != 1 { return uneval("QR", args); }
    let (m, n, mut a) = match read_matrix(ev, args[0].clone()) { Some(t) => t, None => return uneval("QR", args) };
    let m = m; let n = n;
    let kmax = std::cmp::min(m, n);
    // Initialize Q = I_m
    let mut q = vec![0.0; m*m];
    for i in 0..m { q[i*m + i] = 1.0; }
    let eps = 1e-12f64;
    for k in 0..kmax {
        // x = a[k..m, k]
        let mut norm_x = 0.0;
        for i in k..m { let v = a[i*n + k]; norm_x += v*v; }
        let norm_x = norm_x.sqrt();
        if norm_x <= eps { continue; }
        let x0 = a[k*n + k];
        let sign = if x0 >= 0.0 { 1.0 } else { -1.0 };
        // v = x + sign*||x||*e1
        let mut v = vec![0.0; m - k];
        for i in 0..(m - k) { v[i] = a[(k + i)*n + k]; }
        v[0] += sign * norm_x;
        // tau = 2/(v^T v)
        let mut beta = 0.0; for &vi in &v { beta += vi*vi; }
        if beta <= eps { continue; }
        let tau = 2.0 / beta;
        // Apply H = I - tau*v*v^T to A: A[k.., k..] -= tau*v*(v^T*A[k.., k..])
        for j in k..n {
            let mut s = 0.0; for i in 0..(m - k) { s += v[i] * a[(k + i)*n + j]; }
            s *= tau;
            for i in 0..(m - k) { a[(k + i)*n + j] -= s * v[i]; }
        }
        // Apply H to Q on the left: Q[:, k..] <- Q[:, k..] - (Q[:, k..]*v)*tau*v^T
        for j in 0..m { // operate on columns of Q (since Q is m x m, left-apply H)
            let mut s = 0.0; for i in 0..(m - k) { s += v[i] * q[(k + i)*m + j]; }
            s *= tau;
            for i in 0..(m - k) { q[(k + i)*m + j] -= s * v[i]; }
        }
    }
    // R is current A; zero tiny values below diagonal for clarity
    for i in 0..m { for j in 0..std::cmp::min(i, n) { if i>j && a[i*n + j].abs() < 1e-14 { a[i*n + j] = 0.0; } } }
    // Reduced form option: QR[A, <|"Reduced"->True|>] or QR[A, "Reduced"]
    let mut reduced = false;
    if args.len() >= 2 {
        match ev.eval(args[1].clone()) {
            Value::Assoc(m) => {
                if let Some(Value::Boolean(b)) = m.get("Reduced") { reduced = *b; }
            }
            Value::String(s) if s.eq_ignore_ascii_case("Reduced") => reduced = true,
            Value::Symbol(s) if s == "Reduced" => reduced = true,
            _ => {}
        }
    }
    let (qv, rv) = if reduced {
        let k = std::cmp::min(m, n);
        // Q_reduced: first k columns of Q (m x m) -> (m x k)
        let mut qred = vec![0.0; m*k];
        for i in 0..m { for j in 0..k { qred[i*k + j] = q[i*m + j]; } }
        // R_reduced: first k rows of R (m x n) -> (k x n)
        let mut rred = vec![0.0; k*n];
        for i in 0..k { for j in 0..n { rred[i*n + j] = a[i*n + j]; } }
        (pack(vec![m, k], qred), pack(vec![k, n], rred))
    } else {
        (pack(vec![m, m], q), pack(vec![m, n], a))
    };
    let mut assoc = std::collections::HashMap::new();
    assoc.insert("Q".into(), qv);
    assoc.insert("R".into(), rv);
    Value::Assoc(assoc)
}
fn lu(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return uneval("LU", args); }
    match lu_decomp(ev, args[0].clone()) {
        Some((n, lu, piv)) => {
            // Extract L and U from combined LU matrix
            let mut L = vec![0.0; n*n];
            let mut U = vec![0.0; n*n];
            for i in 0..n {
                for j in 0..n {
                    if i > j { L[i*n + j] = lu[i*n + j]; U[i*n + j] = 0.0; }
                    else if i == j { L[i*n + j] = 1.0; U[i*n + j] = lu[i*n + j]; }
                    else { L[i*n + j] = 0.0; U[i*n + j] = lu[i*n + j]; }
                }
            }
            let lval = pack(vec![n, n], L);
            let uval = pack(vec![n, n], U);
            let pivv = Value::List(piv.into_iter().map(|i| Value::Integer((i as i64)+1)).collect());
            let mut assoc = std::collections::HashMap::new();
            assoc.insert("L".into(), lval);
            assoc.insert("U".into(), uval);
            assoc.insert("Pivots".into(), pivv);
            Value::Assoc(assoc)
        }
        None => uneval("LU", args),
    }
}

fn cholesky(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return uneval("Cholesky", args); }
    match cholesky_decomp(ev, args[0].clone()) {
        Some((n, l)) => pack(vec![n, n], l),
        None => uneval("Cholesky", args),
    }
}
fn eigen(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Symmetric eigen-decomposition via Jacobi rotations: returns <|Eigenvalues, Eigenvectors|>
    if args.len() != 1 { return uneval("EigenDecomposition", args); }
    let (n, m, a0) = match read_matrix(ev, args[0].clone()) { Some(t) => t, None => return uneval("EigenDecomposition", args) };
    if n != m { return uneval("EigenDecomposition", args); }
    let n = n;
    let mut a = a0.clone();
    // Check symmetry
    let mut symmetric = true;
    'outer: for i in 0..n { for j in (i+1)..n { if (a[i*m + j] - a[j*m + i]).abs() > 1e-9 { symmetric = false; break 'outer; } } }
    if !symmetric {
        // General (non-symmetric) case: QR iterations for eigenvalues, then inverse iteration for eigenvectors (best-effort, real only)
        let n = n;
        let orig = pack(vec![n, n], a.clone());
        let mut Ak = a.clone();
        let mut Qacc = vec![0.0; n*n]; for i in 0..n { Qacc[i*n + i] = 1.0; }
        let max_iter = 200usize;
        for _ in 0..max_iter {
            // QR factorization of Ak
            let qr_assoc = qr(ev, vec![pack(vec![n, n], Ak.clone())]);
            let (Q, R) = match qr_assoc {
                Value::Assoc(mut m) => {
                    let qv = match m.remove("Q") { Some(Value::PackedArray { shape, data }) if shape==vec![n,n] => data, _ => break };
                    let rv = match m.remove("R") { Some(Value::PackedArray { shape, data }) if shape==vec![n,n] => data, _ => break };
                    (qv, rv)
                }
                _ => break,
            };
            // Ak = R * Q
            let mut newA = vec![0.0; n*n];
            for i in 0..n { for j in 0..n { let mut s=0.0; for k in 0..n { s += R[i*n + k]*Q[k*n + j]; } newA[i*n + j] = s; } }
            Ak = newA;
            // Qacc = Qacc * Q
            let mut newQacc = vec![0.0; n*n];
            for i in 0..n { for j in 0..n { let mut s=0.0; for k in 0..n { s += Qacc[i*n + k]*Q[k*n + j]; } newQacc[i*n + j] = s; } }
            Qacc = newQacc;
            // Check convergence: off-diagonal norm
            let mut off = 0.0; for i in 0..n { for j in 0..n { if i!=j { off += Ak[i*n + j].abs(); } } }
            if off < 1e-10 { break; }
        }
        // Extract eigenvalues (real approximations)
        let mut evals = vec![0.0; n]; for i in 0..n { evals[i] = Ak[i*n + i]; }
        // Inverse iteration for eigenvectors
        let mut evecs = vec![0.0; n*n];
        for idx in 0..n {
            let lam = evals[idx];
            // Build (A - lam I)
            let mut shifted = a.clone();
            for i in 0..n { shifted[i*n + i] -= lam; }
            let mut v = vec![0.0; n];
            for i in 0..n { v[i] = if i==idx% n { 1.0 } else { 0.1 }; }
            let mut vv = pack(vec![n], v.clone());
            let mut mat = pack(vec![n, n], shifted.clone());
            // Perform a few inverse iterations
            let iters = 8;
            let mut ok = true;
            for _ in 0..iters {
                let sol = linear_solve(ev, vec![mat.clone(), vv.clone()]);
                let vec_out = match sol { Value::PackedArray { shape, data } if shape==vec![n] => data, _ => { ok=false; break } };
                // normalize
                let mut norm = 0.0; for x in &vec_out { norm += x*x; } norm = norm.sqrt().max(1e-18);
                let mut vn = vec![0.0; n]; for i in 0..n { vn[i] = vec_out[i] / norm; }
                vv = pack(vec![n], vn.clone());
            }
            let vec_final = match vv { Value::PackedArray { data, .. } => data, _ => vec![0.0; n] };
            // residual to check 
            let mut res = 0.0; for i in 0..n { let mut s=0.0; for j in 0..n { s += a[i*n + j]*vec_final[j]; } let r = s - lam*vec_final[i]; res += r*r; }
            if ok && res.sqrt() < 1e-6 {
                for i in 0..n { evecs[i*n + idx] = vec_final[i]; }
            } else {
                // fall back to Schur vector (column of Qacc)
                for i in 0..n { evecs[i*n + idx] = Qacc[i*n + idx]; }
            }
        }
        let mut assoc = std::collections::HashMap::new();
        assoc.insert("Eigenvalues".into(), pack(vec![n], evals));
        assoc.insert("Eigenvectors".into(), pack(vec![n, n], evecs));
        return Value::Assoc(assoc);
    }
    // Initialize eigenvectors as identity
    let mut v = vec![0.0; n*n]; for i in 0..n { v[i*n + i] = 1.0; }
    let max_sweeps = 50usize;
    let tol = 1e-12f64;
    for _sweep in 0..max_sweeps {
        let mut changed = false;
        for p in 0..n {
            for q in (p+1)..n {
                let apq = a[p*m + q];
                if apq.abs() > tol {
                    let app = a[p*m + p]; let aqq = a[q*m + q];
                    let phi = 0.5 * (2.0*apq).atan2(aqq - app);
                    let c = phi.cos(); let s = phi.sin();
                    // Update rows/cols p and q
                    for k in 0..n {
                        if k != p && k != q {
                            let aik = a[p*m + k]; let akq = a[q*m + k];
                            let new_aik = c*aik - s*akq;
                            let new_akq = s*aik + c*akq;
                            a[p*m + k] = new_aik; a[k*m + p] = new_aik;
                            a[q*m + k] = new_akq; a[k*m + q] = new_akq;
                        }
                    }
                    let app_new = c*c*app - 2.0*s*c*apq + s*s*aqq;
                    let aqq_new = s*s*app + 2.0*s*c*apq + c*c*aqq;
                    a[p*m + p] = app_new;
                    a[q*m + q] = aqq_new;
                    a[p*m + q] = 0.0; a[q*m + p] = 0.0;
                    // Update V
                    for k in 0..n {
                        let vkp = v[k*n + p]; let vkq = v[k*n + q];
                        v[k*n + p] = c*vkp - s*vkq;
                        v[k*n + q] = s*vkp + c*vkq;
                    }
                    changed = true;
                }
            }
        }
        if !changed { break; }
    }
    // Eigenvalues are diagonal of A; eigenvectors are columns of V
    let mut evals = vec![0.0; n]; for i in 0..n { evals[i] = a[i*m + i]; }
    let mut assoc = std::collections::HashMap::new();
    assoc.insert("Eigenvalues".into(), pack(vec![n], evals));
    assoc.insert("Eigenvectors".into(), pack(vec![n, n], v));
    Value::Assoc(assoc)
}
fn parse_complex_list(ev: &mut Evaluator, v: Value) -> Option<Vec<(f64, f64)>> {
    match ev.eval(v) {
        Value::List(xs) => {
            let mut out = Vec::with_capacity(xs.len());
            for it in xs {
                let itv = ev.eval(it);
                match itv {
                    Value::Integer(n) => out.push((n as f64, 0.0)),
                    Value::Real(x) => out.push((x, 0.0)),
                    Value::Rational { num, den } => {
                        if den != 0 { out.push(((num as f64)/(den as f64), 0.0)); } else { return None; }
                    }
                    Value::Complex { re, im } => {
                        let rr = num_to_f64(&ev.eval(*re))?;
                        let ii = num_to_f64(&ev.eval(*im))?;
                        out.push((rr, ii));
                    }
                    _ => return None,
                }
            }
            Some(out)
        }
        _ => None,
    }
}

fn dft(x: &[(f64, f64)], inverse: bool) -> Vec<(f64, f64)> {
    let n = x.len();
    let mut out = vec![(0.0, 0.0); n];
    if n == 0 { return out; }
    let s = if inverse { 1.0 } else { -1.0 };
    for k in 0..n {
        let mut rk = 0.0; let mut ik = 0.0;
        for (n_idx, &(re, im)) in x.iter().enumerate() {
            let theta = 2.0 * std::f64::consts::PI * (k as f64) * (n_idx as f64) / (n as f64);
            let cs = theta.cos(); let sn = theta.sin();
            // multiply (re + i im) * (cos + i s sin)
            rk += re*cs - im*(s*sn);
            ik += re*(s*sn) + im*cs;
        }
        if inverse { rk /= n as f64; ik /= n as f64; }
        out[k] = (rk, ik);
    }
    out
}

fn fft(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // FFT[x, n?] -> list of Complex
    if args.is_empty() { return uneval("FFT", args); }
    let mut x = match parse_complex_list(ev, args[0].clone()) { Some(v) => v, None => return uneval("FFT", args) };
    // Optional N (pad/truncate)
    if args.len() >= 2 {
        if let Some(nn) = num_to_f64(&ev.eval(args[1].clone())).map(|z| z as usize) {
            if nn > 0 {
                x.resize(nn, (0.0, 0.0));
                x.truncate(nn);
            }
        }
    }
    let y = dft(&x, false);
    let vals: Vec<Value> = y.into_iter().map(|(re, im)| Value::Complex { re: Box::new(Value::Real(re)), im: Box::new(Value::Real(im)) }).collect();
    Value::List(vals)
}

fn ifft(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 && args.len() != 2 { return uneval("IFFT", args); }
    let x = match parse_complex_list(ev, args[0].clone()) { Some(v) => v, None => return uneval("IFFT", args) };
    let y = dft(&x, true);
    // If imaginary parts are ~0, return reals; else complex
    let mut max_im: f64 = 0.0; for &(_, im) in &y { let a = im.abs(); if a > max_im { max_im = a; } }
    if max_im < 1e-12 {
        Value::List(y.into_iter().map(|(re, _)| Value::Real(re)).collect())
    } else {
        Value::List(y.into_iter().map(|(re, im)| Value::Complex { re: Box::new(Value::Real(re)), im: Box::new(Value::Real(im)) }).collect())
    }
}

fn convolve(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Convolve[a, b, mode?] where a,b are lists (real). mode: "Full" (default) | "Same" | "Valid"
    if args.len() < 2 { return uneval("Convolve", args); }
    let av = match ev.eval(args[0].clone()) { Value::List(xs) => xs.into_iter().filter_map(|v| num_to_f64(&ev.eval(v))).collect::<Vec<_>>(), _ => Vec::new() };
    let bv = match ev.eval(args[1].clone()) { Value::List(xs) => xs.into_iter().filter_map(|v| num_to_f64(&ev.eval(v))).collect::<Vec<_>>(), _ => Vec::new() };
    if av.is_empty() || bv.is_empty() { return uneval("Convolve", args); }
    let mode = if args.len() >= 3 {
        match ev.eval(args[2].clone()) {
            Value::String(s) => s,
            Value::Symbol(s) => s,
            _ => "Full".into(),
        }
    } else { "Full".into() };
    let n = av.len(); let m = bv.len();
    let mut out_full = vec![0.0; n + m - 1];
    for i in 0..n {
        for j in 0..m { out_full[i + j] += av[i] * bv[j]; }
    }
    let out = if mode.eq_ignore_ascii_case("Full") {
        out_full
    } else if mode.eq_ignore_ascii_case("Same") {
        // center crop to length n
        if n + m - 1 <= n { out_full }
        else {
            let start = (m - 1) / 2;
            out_full[start..start + n].to_vec()
        }
    } else if mode.eq_ignore_ascii_case("Valid") {
        if n >= m {
            out_full[m - 1..n].to_vec()
        } else { Vec::new() }
    } else { out_full };
    Value::List(out.into_iter().map(Value::Real).collect())
}

fn stft(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // STFT[x, size, hop?] or STFT[x, <|Size->n, Hop->h, Window->name|>]
    if args.is_empty() { return uneval("STFT", args); }
    let xs = match ev.eval(args[0].clone()) { Value::List(vs) => vs.into_iter().filter_map(|v| num_to_f64(&ev.eval(v))).collect::<Vec<_>>(), _ => Vec::new() };
    if xs.is_empty() { return uneval("STFT", args); }
    let mut size: usize = 0; let mut hop: usize = 0; let mut win_name: String = "Hann".into();
    if args.len() >= 2 {
        match ev.eval(args[1].clone()) {
            Value::Integer(n) if n > 0 => { size = n as usize; },
            Value::Assoc(m) => {
                if let Some(Value::Integer(n)) = m.get("Size") { if *n > 0 { size = *n as usize; } }
                if let Some(Value::Integer(h)) = m.get("Hop") { if *h > 0 { hop = *h as usize; } }
                if let Some(Value::String(s)) = m.get("Window") { win_name = s.clone(); }
                if let Some(Value::Symbol(s)) = m.get("Window") { win_name = s.clone(); }
            }
            _ => {}
        }
    }
    if args.len() >= 3 {
        if let Value::Integer(h) = ev.eval(args[2].clone()) { if h > 0 { hop = h as usize; } }
    }
    if size == 0 { size = 1024; }
    if hop == 0 { hop = size / 2; }
    // make window
    let wv = match window_weights(ev, vec![Value::String(win_name), Value::Integer(size as i64)]) { v @ Value::List(_) => v, _ => window_weights(ev, vec![Value::String("Hann".into()), Value::Integer(size as i64)]) };
    let w: Vec<f64> = match wv { Value::List(xs) => xs.into_iter().filter_map(|v| num_to_f64(&ev.eval(v))).collect(), _ => (0..size).map(|_|1.0).collect() };
    // frames
    let mut frames: Vec<Value> = Vec::new();
    let mut pos = 0usize;
    while pos + size <= xs.len() {
        let mut seg: Vec<(f64, f64)> = Vec::with_capacity(size);
        for i in 0..size { seg.push((xs[pos + i] * w[i], 0.0)); }
        let spec = dft(&seg, false);
        let half = size / 2 + 1;
        let frame: Vec<Value> = spec.into_iter().take(half).map(|(re, im)| Value::Complex { re: Box::new(Value::Real(re)), im: Box::new(Value::Real(im)) }).collect();
        frames.push(Value::List(frame));
        pos += hop;
    }
    Value::List(frames)
}

fn filter_fir(_ev: &mut Evaluator, args: Vec<Value>) -> Value { uneval("FilterFIR", args) }
fn filter_iir(_ev: &mut Evaluator, args: Vec<Value>) -> Value { uneval("FilterIIR", args) }

fn window_weights(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Window[type, n, opts?]
    if args.len() < 2 { return uneval("Window", args); }
    let t = ev.eval(args[0].clone());
    let n = match ev.eval(args[1].clone()) { Value::Integer(k) if k>0 => k as usize, _ => return uneval("Window", args) };
    let name = match t { Value::String(s) => s, Value::Symbol(s) => s, _ => return uneval("Window", args) };
    let pi = std::f64::consts::PI;
    let out: Vec<f64> = match name.to_lowercase().as_str() {
        "hann" | "hanning" => (0..n).map(|i| 0.5 - 0.5*((2.0*pi*(i as f64))/((n as f64))).cos()).collect(),
        "hamming" => (0..n).map(|i| 0.54 - 0.46*((2.0*pi*(i as f64))/((n as f64))).cos()).collect(),
        "blackman" => (0..n).map(|i| {
            let a0=0.42; let a1=0.5; let a2=0.08; let x=(2.0*pi*(i as f64))/((n as f64));
            a0 - a1*x.cos() + a2*(2.0*x).cos()
        }).collect(),
        _ => (0..n).map(|_| 1.0).collect(),
    };
    Value::List(out.into_iter().map(Value::Real).collect())
}

// ----- Core linalg utilities: Dot, Transpose, Trace, Norm, plus better LinearSolve -----

fn dot(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return uneval("Dot", args); }
    // Try vector dot product first
    if let (Some(a), Some(b)) = (read_vector(ev, args[0].clone()), read_vector(ev, args[1].clone())) {
        if a.len() != b.len() { return uneval("Dot", args); }
        let s: f64 = a.iter().zip(b.iter()).map(|(x,y)| x*y).sum();
        return Value::Real(s);
    }
    // Try matrix*vector or vector*matrix or matrix*matrix via explicit loops
    if let Some((m, k, a)) = read_matrix(ev, args[0].clone()) {
        if let Some(bv) = read_vector(ev, args[1].clone()) {
            if k != bv.len() { return uneval("Dot", args); }
            let mut out = vec![0.0; m];
            for i in 0..m { let mut acc=0.0; for j in 0..k { acc += a[i*k + j]*bv[j]; } out[i]=acc; }
            return pack(vec![m], out);
        }
        if let Some((k2, n, b)) = read_matrix(ev, args[1].clone()) {
            if k != k2 { return uneval("Dot", args); }
            let mut out = vec![0.0; m*n];
            for i in 0..m { for j in 0..n { let mut acc=0.0; for t in 0..k { acc += a[i*k + t]*b[t*n + j]; } out[i*n + j]=acc; } }
            return pack(vec![m, n], out);
        }
    }
    if let Some(av) = read_vector(ev, args[0].clone()) {
        if let Some((m, n, b)) = read_matrix(ev, args[1].clone()) {
            if av.len() != m { return uneval("Dot", args); }
            let mut out = vec![0.0; n];
            for j in 0..n { let mut acc=0.0; for i in 0..m { acc += av[i]*b[i*n + j]; } out[j]=acc; }
            return pack(vec![n], out);
        }
    }
    uneval("Dot", args)
}

fn transpose_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [a] => {
            if let Some((m, n, a)) = read_matrix(ev, a.clone()) {
                let mut out = vec![0.0; m*n];
                for i in 0..m { for j in 0..n { out[j*m + i] = a[i*n + j]; } }
                return pack(vec![n, m], out);
            }
            // Vectors: transpose is a no-op
            if let Some(v) = read_vector(ev, a.clone()) { return pack(vec![v.len()], v); }
            uneval("Transpose", args)
        }
        [a, perm] => {
            // Delegate to NDTranspose for explicit permutations
            nd_transpose(ev, vec![a.clone(), perm.clone()])
        }
        _ => uneval("Transpose", args),
    }
}

fn trace_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return uneval("Trace", args); }
    let (m, n, a) = match read_matrix(ev, args[0].clone()) { Some(t) => t, None => return uneval("Trace", args) };
    if m != n { return uneval("Trace", args); }
    let mut s = 0.0; for i in 0..n { s += a[i*n + i]; }
    Value::Real(s)
}

fn norm_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return uneval("Norm", args); }
    // Vector norms
    if let Some(v) = read_vector(ev, args[0].clone()) {
        // p defaults to 2
        let p_opt = if args.len() >= 2 { Some(ev.eval(args[1].clone())) } else { None };
        match p_opt {
            Some(Value::Symbol(s)) if s == "Infinity" => {
                let m = v.into_iter().map(|x| x.abs()).fold(0.0, f64::max);
                return Value::Real(m);
            }
            Some(Value::Integer(p)) if p == 1 => {
                let s: f64 = v.into_iter().map(|x| x.abs()).sum();
                return Value::Real(s);
            }
            Some(Value::Integer(p)) if p == 2 => {
                let s: f64 = v.iter().map(|x| x*x).sum();
                return Value::Real(s.sqrt());
            }
            Some(Value::Real(p)) => {
                let p = p;
                if p > 0.0 {
                    let s: f64 = v.iter().map(|x| x.abs().powf(p)).sum();
                    return Value::Real(s.powf(1.0/p));
                }
            }
            Some(Value::Integer(p)) if p > 0 => {
                let p = p as i64 as f64;
                let s: f64 = v.iter().map(|x| x.abs().powf(p)).sum();
                return Value::Real(s.powf(1.0/p));
            }
            _ => {
                let s: f64 = v.iter().map(|x| x*x).sum();
                return Value::Real(s.sqrt());
            }
        }
    }
    // Matrix norms
    if let Some((m, n, a)) = read_matrix(ev, args[0].clone()) {
        // Specific p-norms when requested explicitly
        if args.len() >= 2 {
            match ev.eval(args[1].clone()) {
                // Spectral norm (largest singular value)
                Value::Integer(p) if p == 2 => {
                    let sv = svd(ev, vec![pack(vec![m, n], a.clone())]);
                    if let Value::Assoc(mut map) = sv {
                        if let Some(Value::PackedArray { data: s, .. }) = map.remove("S") {
                            let maxs = s.into_iter().fold(0.0f64, |acc, v| if v>acc { v } else { acc });
                            return Value::Real(maxs);
                        }
                    }
                    return uneval("Norm", args);
                }
                Value::Real(p) if (p - 2.0).abs() < 1e-12 => {
                    let sv = svd(ev, vec![pack(vec![m, n], a.clone())]);
                    if let Value::Assoc(mut map) = sv {
                        if let Some(Value::PackedArray { data: s, .. }) = map.remove("S") {
                            let maxs = s.into_iter().fold(0.0f64, |acc, v| if v>acc { v } else { acc });
                            return Value::Real(maxs);
                        }
                    }
                    return uneval("Norm", args);
                }
                // Frobenius alias
                Value::Symbol(s) if s == "Frobenius" => {
                    let s: f64 = a.iter().map(|x| x*x).sum();
                    return Value::Real(s.sqrt());
                }
                Value::String(s) if s.eq_ignore_ascii_case("Frobenius") => {
                    let s: f64 = a.iter().map(|x| x*x).sum();
                    return Value::Real(s.sqrt());
                }
                // 1-norm: maximum column sum
                Value::Integer(p) if p == 1 => {
                    let mut best = 0.0f64;
                    for j in 0..n { let mut s=0.0; for i in 0..m { s += a[i*n + j].abs(); } if s>best { best = s; } }
                    return Value::Real(best);
                }
                Value::Real(p) if (p - 1.0).abs() < 1e-12 => {
                    let mut best = 0.0f64;
                    for j in 0..n { let mut s=0.0; for i in 0..m { s += a[i*n + j].abs(); } if s>best { best = s; } }
                    return Value::Real(best);
                }
                // Infinity-norm: maximum row sum
                Value::Symbol(s) if s == "Infinity" => {
                    let mut best = 0.0f64;
                    for i in 0..m { let mut s=0.0; for j in 0..n { s += a[i*n + j].abs(); } if s>best { best = s; } }
                    return Value::Real(best);
                }
                Value::String(s) if s.eq_ignore_ascii_case("Infinity") => {
                    let mut best = 0.0f64;
                    for i in 0..m { let mut s=0.0; for j in 0..n { s += a[i*n + j].abs(); } if s>best { best = s; } }
                    return Value::Real(best);
                }
                _ => {}
            }
        }
        // Default Frobenius norm
        let s: f64 = a.iter().map(|x| x*x).sum();
        return Value::Real(s.sqrt());
    }
    uneval("Norm", args)
}

fn pseudoinverse_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() || args.len() > 2 { return uneval("PseudoInverse", args); }
    let (m, n, a) = match read_matrix(ev, args[0].clone()) { Some(t) => t, None => return uneval("PseudoInverse", args) };
    let sv = svd(ev, vec![pack(vec![m, n], a.clone())]);
    let (U, S, V) = match sv {
        Value::Assoc(mut map) => {
            let u = match map.remove("U") { Some(Value::PackedArray { shape, data }) => (shape, data), _ => return uneval("PseudoInverse", args) };
            let s = match map.remove("S") { Some(Value::PackedArray { shape, data }) => (shape, data), _ => return uneval("PseudoInverse", args) };
            let v = match map.remove("V") { Some(Value::PackedArray { shape, data }) => (shape, data), _ => return uneval("PseudoInverse", args) };
            (u, s, v)
        }
        _ => return uneval("PseudoInverse", args),
    };
    let (ush, udat) = U; let (ssh, svals) = S; let (vsh, vdat) = V;
    // Shapes: U m×k, S k, V n×k
    if ush.len()!=2 || vsh.len()!=2 || ssh.len()!=1 { return uneval("PseudoInverse", args); }
    let k = ssh[0];
    if ush[0] != m || ush[1] != k || vsh[0] != n || vsh[1] != k || svals.len()!=k { return uneval("PseudoInverse", args); }
    // Build S^+ (k x k) diagonal with tolerance
    let mut Sinv = vec![0.0; k*k];
    let maxs = svals.iter().fold(0.0f64, |acc, &v| if v>acc { v } else { acc });
    let mut tol = 1e-12 * (m.max(n) as f64) * maxs;
    if args.len() == 2 {
        let opt = ev.eval(args[1].clone());
        match opt {
            Value::Assoc(map) => {
                if let Some(v) = map.get("Tolerance") {
                    if let Some(t) = num_to_f64(v) { tol = t; }
                }
            }
            // Also accept numeric shorthand: PseudoInverse[A, tol]
            other => {
                if let Some(t) = num_to_f64(&other) { tol = t; }
            }
        }
    }
    for i in 0..k { let s = svals[i]; if s > tol { Sinv[i*k + i] = 1.0 / s; } }
    // Compute V * S^+ (n x k)
    let mut VS = vec![0.0; n*k];
    for i in 0..n { for j in 0..k { let mut s=0.0; for t in 0..k { s += vdat[i*k + t] * Sinv[t*k + j]; } VS[i*k + j] = s; } }
    // Compute (V * S^+) * U^T => (n x m)
    let mut Aplus = vec![0.0; n*m];
    for i in 0..n { for j in 0..m { let mut s=0.0; for t in 0..k { s += VS[i*k + t] * udat[j*k + t]; } Aplus[i*m + j] = s; } }
    pack(vec![n, m], Aplus)
}

fn matrix_norm_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // MatrixNorm[A, p]; supports p=2 (spectral), p=1 (max column sum), Infinity (max row sum), and "Frobenius"
    if args.len() != 2 { return uneval("MatrixNorm", args); }
    let (m, n, a) = match read_matrix(ev, args[0].clone()) { Some(t) => t, None => return uneval("MatrixNorm", args) };
    match ev.eval(args[1].clone()) {
        Value::Integer(p) if p == 2 => {
            let sv = svd(ev, vec![pack(vec![m, n], a.clone())]);
            if let Value::Assoc(mut map) = sv {
                if let Some(Value::PackedArray { data: s, .. }) = map.remove("S") {
                    let maxs = s.into_iter().fold(0.0f64, |acc, v| if v>acc { v } else { acc });
                    return Value::Real(maxs);
                }
            }
            uneval("MatrixNorm", args)
        }
        Value::Integer(p) if p == 1 => {
            let mut best = 0.0f64;
            for j in 0..n { let mut s=0.0; for i in 0..m { s += a[i*n + j].abs(); } if s>best { best = s; } }
            Value::Real(best)
        }
        Value::Symbol(s) if s == "Frobenius" => {
            let s: f64 = a.iter().map(|x| x*x).sum();
            Value::Real(s.sqrt())
        }
        Value::String(s) if s.eq_ignore_ascii_case("Frobenius") => {
            let s: f64 = a.iter().map(|x| x*x).sum();
            Value::Real(s.sqrt())
        }
        Value::Symbol(s) if s == "Infinity" => {
            let mut best = 0.0f64;
            for i in 0..m { let mut s=0.0; for j in 0..n { s += a[i*n + j].abs(); } if s>best { best = s; } }
            Value::Real(best)
        }
        Value::String(s) if s.eq_ignore_ascii_case("Infinity") => {
            let mut best = 0.0f64;
            for i in 0..m { let mut s=0.0; for j in 0..n { s += a[i*n + j].abs(); } if s>best { best = s; } }
            Value::Real(best)
        }
        Value::Real(p) if (p - 2.0).abs() < 1e-12 => {
            let sv = svd(ev, vec![pack(vec![m, n], a.clone())]);
            if let Value::Assoc(mut map) = sv {
                if let Some(Value::PackedArray { data: s, .. }) = map.remove("S") {
                    let maxs = s.into_iter().fold(0.0f64, |acc, v| if v>acc { v } else { acc });
                    return Value::Real(maxs);
                }
            }
            uneval("MatrixNorm", args)
        }
        Value::Real(p) if (p - 1.0).abs() < 1e-12 => {
            let mut best = 0.0f64;
            for j in 0..n { let mut s=0.0; for i in 0..m { s += a[i*n + j].abs(); } if s>best { best = s; } }
            Value::Real(best)
        }
        _ => uneval("MatrixNorm", args),
    }
}

fn diagonal_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return uneval("Diagonal", args); }
    if let Some((m, n, a)) = read_matrix(ev, args[0].clone()) {
        let k = std::cmp::min(m, n);
        let mut d = vec![0.0; k];
        for i in 0..k { d[i] = a[i*n + i]; }
        return pack(vec![k], d);
    }
    uneval("Diagonal", args)
}

fn diag_matrix_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return uneval("DiagMatrix", args); }
    if let Some(v) = read_vector(ev, args[0].clone()) {
        let n = v.len();
        let mut a = vec![0.0; n*n];
        for i in 0..n { a[i*n + i] = v[i]; }
        return pack(vec![n, n], a);
    }
    uneval("DiagMatrix", args)
}

fn rank_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return uneval("Rank", args); }
    let (m, n, _a) = match read_matrix(ev, args[0].clone()) { Some(t) => t, None => return uneval("Rank", args) };
    // Use QR (reduced) and count non-negligible diagonal of R
    let qr_res = qr(ev, vec![args[0].clone(), Value::Symbol("Reduced".into())]);
    let r = match qr_res { Value::Assoc(mut map) => map.remove("R"), _ => None };
    if let Some(Value::PackedArray { shape, data }) = r {
        if shape.len() == 2 {
            let k = std::cmp::min(m, n);
            // Tolerance: eps * max(m,n) * maxdiag
            let mut maxdiag = 0.0f64;
            for i in 0..k { let v = data[i*shape[1] + i].abs(); if v > maxdiag { maxdiag = v; } }
            let tol = 1e-10 * (std::cmp::max(m, n) as f64) * f64::max(maxdiag, 1.0);
            let mut rnk = 0i64;
            for i in 0..k { if data[i*shape[1] + i].abs() > tol { rnk += 1; } }
            return Value::Integer(rnk);
        }
    }
    uneval("Rank", args)
}

fn condition_number_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return uneval("ConditionNumber", args); }
    let (m, n, a) = match read_matrix(ev, args[0].clone()) { Some(t) => t, None => return uneval("ConditionNumber", args) };
    // Build AtA (n x n)
    let mut ata = vec![0.0; n*n];
    for i in 0..n { for j in 0..n { let mut s=0.0; for r in 0..m { s += a[r*n + i]*a[r*n + j]; } ata[i*n + j] = s; } }
    // Power iteration for largest eigenvalue of AtA (sigma_max^2)
    let mut x = vec![1.0; n];
    let mut lam_max = 0.0;
    for _ in 0..100 {
        // y = AtA * x
        let mut y = vec![0.0; n];
        for i in 0..n { let mut s=0.0; for j in 0..n { s += ata[i*n + j] * x[j]; } y[i]=s; }
        let norm = y.iter().map(|v| v*v).sum::<f64>().sqrt();
        if norm == 0.0 { break; }
        for i in 0..n { x[i] = y[i]/norm; }
        lam_max = 0.0; for i in 0..n { lam_max += x[i]*y[i]; }
    }
    // Inverse power iteration for smallest eigenvalue of AtA (sigma_min^2)
    // Factor AtA once (prefer Cholesky)
    let ata_val = pack(vec![n, n], ata.clone());
    let (use_chol, l_mat, lu_mat, piv) = match cholesky_decomp(ev, ata_val.clone()) {
        Some((_nn, l)) => (true, Some(l), None, None),
        None => match lu_decomp(ev, ata_val) { Some((_nn, lu, piv)) => (false, None, Some(lu), Some(piv)), None => (false, None, None, None) }
    };
    let mut x2 = vec![1.0; n];
    let mut lam_min = 0.0;
    if use_chol || lu_mat.is_some() {
        for _ in 0..100 {
            // Solve (AtA) y = x2
            let mut y = x2.clone();
            if use_chol {
                let l = l_mat.as_ref().unwrap();
                // forward then back with L^T
                forward_subst_lower(n, l, &mut y);
                for ii in 0..n { let i = n-1-ii; let mut s=y[i]; for j in (i+1)..n { s -= l[j*n + i]*y[j]; } y[i] = s / l[i*n + i]; }
            } else if let (Some(lu), Some(piv)) = (lu_mat.as_ref(), piv.as_ref()) {
                let mut tmp = y.clone();
                lu_solve(n, lu, piv, &mut tmp); y = tmp;
            }
            let norm = y.iter().map(|v| v*v).sum::<f64>().sqrt();
            if norm == 0.0 { break; }
            for i in 0..n { x2[i] = y[i]/norm; }
            lam_min = 0.0; for i in 0..n { lam_min += x2[i]*y[i]; }
        }
    } else {
        // Singular or zero matrix
        return Value::Symbol("Infinity".into());
    }
    let sig_max = lam_max.abs().sqrt();
    let sig_min = lam_min.abs().sqrt();
    if sig_min <= 1e-15 { return Value::Symbol("Infinity".into()); }
    Value::Real(sig_max / sig_min)
}

// ---- Decompositions for LinearSolve ----
fn cholesky_decomp(ev: &mut Evaluator, v: Value) -> Option<(usize, Vec<f64>)> {
    let (n, m, a0) = read_matrix(ev, v)?; if n != m { return None; }
    let mut l = vec![0.0; n*n];
    // Copy A since we only read it
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a0[i*m + j];
            for k in 0..j { sum -= l[i*n + k]*l[j*n + k]; }
            if i == j {
                if sum <= 0.0 { return None; }
                l[i*n + j] = sum.sqrt();
            } else {
                l[i*n + j] = sum / l[j*n + j];
            }
        }
    }
    Some((n, l))
}

fn forward_subst_lower(n: usize, l: &[f64], b: &mut [f64]) {
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i { s -= l[i*n + j] * b[j]; }
        b[i] = s / l[i*n + i];
    }
}

fn back_subst_upper(n: usize, u: &[f64], b: &mut [f64]) {
    for ii in 0..n { let i = n-1-ii; let mut s=b[i]; for j in (i+1)..n { s -= u[i*n + j]*b[j]; } b[i] = s / u[i*n + i]; }
}

fn lu_decomp(ev: &mut Evaluator, v: Value) -> Option<(usize, Vec<f64>, Vec<usize>)> {
    let (n, m, mut a) = read_matrix(ev, v)?; if n != m { return None; }
    let mut piv: Vec<usize> = (0..n).collect();
    for k in 0..n {
        // pivot selection
        let mut p = k; let mut maxv = a[k*m + k].abs();
        for i in (k+1)..n { let v = a[i*m + k].abs(); if v > maxv { maxv = v; p = i; } }
        if maxv == 0.0 { return None; }
        if p != k { for j in 0..n { a.swap(k*m + j, p*m + j); } piv.swap(k, p); }
        // elimination
        for i in (k+1)..n {
            a[i*m + k] /= a[k*m + k];
            let lik = a[i*m + k];
            for j in (k+1)..n { a[i*m + j] -= lik * a[k*m + j]; }
        }
    }
    Some((n, a, piv))
}

fn lu_solve(n: usize, lu: &[f64], piv: &[usize], b: &mut [f64]) {
    // Apply permutation to b
    let mut pb = vec![0.0; n];
    for i in 0..n { pb[i] = b[piv[i]]; }
    // Forward solve Ly = Pb
    for i in 0..n {
        let mut s = pb[i];
        for j in 0..i { s -= lu[i*n + j] * pb[j]; }
        pb[i] = s; // since diag(L)=1
    }
    // Back solve Ux = y
    for ii in 0..n { let i = n-1-ii; let mut s = pb[i]; for j in (i+1)..n { s -= lu[i*n + j]*pb[j]; } pb[i] = s / lu[i*n + i]; }
    b.copy_from_slice(&pb);
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

    // Tier1: Linalg + FFT heads (stubs)
    ev.register("Determinant", determinant as NativeFn, Attributes::empty());
    ev.register("Inverse", inverse as NativeFn, Attributes::empty());
    ev.register("LinearSolve", linear_solve as NativeFn, Attributes::empty());
    ev.register("SVD", svd as NativeFn, Attributes::empty());
    ev.register("QR", qr as NativeFn, Attributes::empty());
    ev.register("LU", lu as NativeFn, Attributes::empty());
    ev.register("Cholesky", cholesky as NativeFn, Attributes::empty());
    ev.register("EigenDecomposition", eigen as NativeFn, Attributes::empty());
    ev.register("Dot", dot as NativeFn, Attributes::empty());
    ev.register("Transpose", transpose_fn as NativeFn, Attributes::empty());
    ev.register("Trace", trace_fn as NativeFn, Attributes::empty());
    ev.register("Norm", norm_fn as NativeFn, Attributes::empty());
    ev.register("PseudoInverse", pseudoinverse_fn as NativeFn, Attributes::empty());
    ev.register("MatrixNorm", matrix_norm_fn as NativeFn, Attributes::empty());
    ev.register("FFT", fft as NativeFn, Attributes::empty());
    ev.register("IFFT", ifft as NativeFn, Attributes::empty());
    ev.register("Convolve", convolve as NativeFn, Attributes::empty());
    ev.register("STFT", stft as NativeFn, Attributes::empty());
    ev.register("FilterFIR", filter_fir as NativeFn, Attributes::empty());
    ev.register("FilterIIR", filter_iir as NativeFn, Attributes::empty());
    ev.register("Window", window_weights as NativeFn, Attributes::empty());
    ev.register("Diagonal", diagonal_fn as NativeFn, Attributes::empty());
    ev.register("DiagMatrix", diag_matrix_fn as NativeFn, Attributes::empty());
    ev.register("Rank", rank_fn as NativeFn, Attributes::empty());
    ev.register("ConditionNumber", condition_number_fn as NativeFn, Attributes::empty());

    // Canonical tensor entry points
    ev.register("Tensor", tensor_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Shape", shape_fn as NativeFn, Attributes::empty());
    ev.register("Reshape", reshape_generic as NativeFn, Attributes::empty());
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

    // Tier1: Linalg + FFT heads (stubs)
    register_if(ev, pred, "Determinant", determinant as NativeFn, Attributes::empty());
    register_if(ev, pred, "Inverse", inverse as NativeFn, Attributes::empty());
    register_if(ev, pred, "LinearSolve", linear_solve as NativeFn, Attributes::empty());
    register_if(ev, pred, "SVD", svd as NativeFn, Attributes::empty());
    register_if(ev, pred, "QR", qr as NativeFn, Attributes::empty());
    register_if(ev, pred, "LU", lu as NativeFn, Attributes::empty());
    register_if(ev, pred, "Cholesky", cholesky as NativeFn, Attributes::empty());
    register_if(ev, pred, "EigenDecomposition", eigen as NativeFn, Attributes::empty());
    register_if(ev, pred, "Dot", dot as NativeFn, Attributes::empty());
    register_if(ev, pred, "Transpose", transpose_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Trace", trace_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Norm", norm_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "PseudoInverse", pseudoinverse_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "MatrixNorm", matrix_norm_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "FFT", fft as NativeFn, Attributes::empty());
    register_if(ev, pred, "IFFT", ifft as NativeFn, Attributes::empty());
    register_if(ev, pred, "Convolve", convolve as NativeFn, Attributes::empty());
    register_if(ev, pred, "STFT", stft as NativeFn, Attributes::empty());
    register_if(ev, pred, "FilterFIR", filter_fir as NativeFn, Attributes::empty());
    register_if(ev, pred, "FilterIIR", filter_iir as NativeFn, Attributes::empty());
    register_if(ev, pred, "Window", window_weights as NativeFn, Attributes::empty());
    register_if(ev, pred, "Diagonal", diagonal_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "DiagMatrix", diag_matrix_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Rank", rank_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "ConditionNumber", condition_number_fn as NativeFn, Attributes::empty());

    // Generic entry points (canonical names) with tensor dispatch
    register_if(ev, pred, "Tensor", tensor_fn as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "Shape", shape_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Reshape", reshape_generic as NativeFn, Attributes::empty());
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

// ---- Generic wrappers ----

fn tensor_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Tensor[x, opts?] -> PackedArray[x]; alias of NDArray
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("Tensor".into())), args };
    }
    ndarray(ev, args)
}

fn shape_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Shape[tensor]
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("Shape".into())), args };
    }
    nd_shape(ev, args)
}

fn reshape_generic(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Reshape[tensor, new_shape]
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("Reshape".into())), args };
    }
    // Only attempt for PackedArray; otherwise, pass through unevaluated
    match as_packed(ev, args[0].clone()) {
        Some(_) => nd_reshape(ev, args),
        None => Value::Expr { head: Box::new(Value::Symbol("Reshape".into())), args },
    }
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
