use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

#[derive(Clone, Default)]
struct NetState {
    kind: String,                 // "Chain" | "Graph"
    layers: Vec<Value>,           // raw layer specs (opaque Values)
    graph: Option<Value>,         // edges/ports (opaque)
    opts: HashMap<String, Value>, // construction options
    encoder: Option<Value>,
    decoder: Option<Value>,
    initialized: bool,
    trained_epochs: usize,
    method: String,               // optimizer (e.g., Adam)
    batch_size: usize,
}

static NN_REG: OnceLock<Mutex<HashMap<i64, NetState>>> = OnceLock::new();
static NEXT_ID: OnceLock<std::sync::atomic::AtomicI64> = OnceLock::new();

fn reg() -> &'static Mutex<HashMap<i64, NetState>> { NN_REG.get_or_init(|| Mutex::new(HashMap::new())) }
fn next_id() -> i64 { let a = NEXT_ID.get_or_init(|| std::sync::atomic::AtomicI64::new(1)); a.fetch_add(1, std::sync::atomic::Ordering::Relaxed) }

fn net_handle(id: i64) -> Value { Value::assoc(vec![("__type", Value::String("Net".into())), ("id", Value::Integer(id))]) }
fn get_net_id(v: &Value) -> Option<i64> {
    if let Value::Assoc(m) = v { if matches!(m.get("__type"), Some(Value::String(s)) if s=="Net") { if let Some(Value::Integer(id)) = m.get("id") { return Some(*id); } } }
    None
}

pub fn register_nn(ev: &mut Evaluator) {
    ev.register("NetChain", net_chain as NativeFn, Attributes::empty());
    ev.register("NetGraph", net_graph as NativeFn, Attributes::empty());
    ev.register("NetInitialize", net_initialize as NativeFn, Attributes::empty());
    ev.register("NetTrain", net_train as NativeFn, Attributes::empty());
    ev.register("NetApply", net_apply as NativeFn, Attributes::empty());
    ev.register("NetProperty", net_property as NativeFn, Attributes::empty());
    ev.register("NetSummary", net_summary as NativeFn, Attributes::empty());
    ev.register("NetEncoder", net_encoder as NativeFn, Attributes::empty());
    ev.register("NetDecoder", net_decoder as NativeFn, Attributes::empty());
    // Layers (constructors return opaque layer specs)
    ev.register("LinearLayer", linear_layer as NativeFn, Attributes::empty());
    ev.register("ActivationLayer", activation_layer as NativeFn, Attributes::empty());
    ev.register("DropoutLayer", dropout_layer as NativeFn, Attributes::empty());
    ev.register("FlattenLayer", flatten_layer as NativeFn, Attributes::empty());
    ev.register("SoftmaxLayer", softmax_layer as NativeFn, Attributes::empty());
    ev.register("ConvolutionLayer", convolution_layer as NativeFn, Attributes::empty());
    ev.register("PoolingLayer", pooling_layer as NativeFn, Attributes::empty());
    ev.register("BatchNormLayer", batchnorm_layer as NativeFn, Attributes::empty());
    ev.register("ReshapeLayer", reshape_layer as NativeFn, Attributes::empty());
    ev.register("TransposeLayer", transpose_layer as NativeFn, Attributes::empty());
    ev.register("ConcatLayer", concat_layer as NativeFn, Attributes::empty());
    ev.register("AddLayer", add_layer as NativeFn, Attributes::empty());
    ev.register("MulLayer", mul_layer as NativeFn, Attributes::empty());
    ev.register("EmbeddingLayer", embedding_layer as NativeFn, Attributes::empty());
    ev.register("LayerNormLayer", layernorm_layer as NativeFn, Attributes::empty());
}

// --------- helpers ---------

fn parse_opts(ev: &mut Evaluator, args: &[Value]) -> (Vec<Value>, HashMap<String, Value>) {
    if args.is_empty() { return (vec![], HashMap::new()); }
    let mut pos: Vec<Value> = Vec::new();
    let mut opts: HashMap<String, Value> = HashMap::new();
    for (i, a) in args.iter().enumerate() {
        if i == args.len() - 1 {
            let evd = ev.eval(a.clone());
            if let Value::Assoc(m) = evd { opts = m; continue; }
        }
        pos.push(a.clone());
    }
    (pos, opts)
}

fn get_option<'a>(opts: &'a HashMap<String, Value>, key: &str) -> Option<&'a Value> { opts.get(key).or_else(|| opts.get(&key.to_string())) }

fn layer_spec(kind: &str, params: HashMap<String, Value>) -> Value {
    Value::assoc(vec![
        ("__type", Value::String("Layer".into())),
        ("LayerType", Value::String(kind.into())),
        ("Params", Value::Assoc(params)),
    ])
}

fn to_vec_f64(v: &Value) -> Option<Vec<f64>> {
    match v {
        Value::Real(x) => Some(vec![*x]),
        Value::Integer(n) => Some(vec![*n as f64]),
        Value::List(xs) => {
            let mut out = Vec::with_capacity(xs.len());
            for x in xs {
                match x { Value::Real(r)=> out.push(*r), Value::Integer(n)=> out.push(*n as f64), _=> return None }
            }
            Some(out)
        }
        _ => None,
    }
}

fn flatten_to_vec_f64(v: &Value, out: &mut Vec<f64>) -> bool {
    match v {
        Value::Real(x) => { out.push(*x); true }
        Value::Integer(n) => { out.push(*n as f64); true }
        Value::List(xs) => { for x in xs { if !flatten_to_vec_f64(x, out) { return false; } } true }
        _ => false,
    }
}

fn from_vec_like(input_like: &Value, xs: &[f64]) -> Value {
    let is_scalar = matches!(input_like, Value::Real(_) | Value::Integer(_));
    if is_scalar && xs.len()==1 {
        let x = xs[0];
        if x.fract()==0.0 { Value::Integer(x as i64) } else { Value::Real(x) }
    } else {
        Value::List(xs.iter().map(|x| Value::Real(*x)).collect())
    }
}

fn lcg(seed: &mut u64) -> f64 {
    // Deterministic pseudo-random in [-0.1, 0.1]
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let v = ((*seed >> 33) as f64) / ((1u64<<31) as f64);
    (v - 1.0) * 0.1
}

fn assoc_get<'a>(m: &'a HashMap<String, Value>, k: &str) -> Option<&'a Value> { m.get(k).or_else(|| m.get(&k.to_string())) }
fn as_i64(v: &Value) -> Option<i64> { match v { Value::Integer(n)=>Some(*n), Value::Real(x)=>Some(*x as i64), _=>None } }
fn as_bool(v: &Value) -> Option<bool> { match v { Value::Boolean(b)=>Some(*b), _=>None } }

fn ensure_linear_params(id: i64, layer_idx: usize, layer: &Value, in_dim: usize) -> (Value, usize) {
    // Returns (updated_layer, out_dim)
    if let Value::Assoc(m) = layer {
        if let Some(Value::Assoc(params)) = m.get("Params") {
            let out_dim = assoc_get(params, "Output").and_then(|v| match v { Value::Integer(n)=>Some((*n).max(0) as usize), Value::Real(x)=>Some((*x as i64).max(0) as usize), _=>None }).unwrap_or(in_dim.max(1));
            let has_w = assoc_get(params, "W").is_some();
            let has_b = assoc_get(params, "b").is_some();
            if has_w && has_b { return (layer.clone(), out_dim); }
            // build new params with W,b
            let mut p2 = params.clone();
            let mut seed = (id as u64) ^ ((layer_idx as u64) << 13) ^ 0x9E3779B97F4A7C15;
            let mut rows: Vec<Value> = Vec::with_capacity(out_dim);
            for _ in 0..out_dim {
                let mut row: Vec<Value> = Vec::with_capacity(in_dim.max(1));
                for _ in 0..in_dim.max(1) { row.push(Value::Real(lcg(&mut seed))); }
                rows.push(Value::List(row));
            }
            let mut b: Vec<Value> = Vec::with_capacity(out_dim);
            for _ in 0..out_dim { b.push(Value::Real(0.0)); }
            p2.insert("W".into(), Value::List(rows));
            p2.insert("b".into(), Value::List(b));
            let mut m2 = m.clone();
            m2.insert("Params".into(), Value::Assoc(p2));
            return (Value::Assoc(m2), out_dim);
        }
    }
    (layer.clone(), in_dim)
}

fn apply_linear(params: &HashMap<String, Value>, x: &[f64]) -> Option<Vec<f64>> {
    let w = assoc_get(params, "W")?;
    let b = assoc_get(params, "b");
    let (w_rows, in_dim) = match w {
        Value::List(rows) => {
            let r = rows.len();
            let c = rows.get(0).and_then(|v| if let Value::List(xs) = v { Some(xs.len()) } else { None }).unwrap_or(0);
            (r, c)
        }
        _ => return None,
    };
    if in_dim==0 { return None; }
    let mut y = vec![0.0f64; w_rows];
    match w {
        Value::List(rows) => {
            for (i, row) in rows.iter().enumerate() {
                let mut acc = 0.0f64;
                if let Value::List(xs) = row {
                    for (j, v) in xs.iter().enumerate().take(x.len()) {
                        let wij = match v { Value::Real(r)=>*r, Value::Integer(n)=>*n as f64, _=>0.0 };
                        acc += wij * x.get(j).cloned().unwrap_or(0.0);
                    }
                }
                y[i] = acc;
            }
        }
        _ => {}
    }
    let use_bias = assoc_get(params, "Bias").and_then(|v| as_bool(v)).unwrap_or(true);
    if use_bias { if let Some(Value::List(bs)) = b { for (i, v) in bs.iter().enumerate().take(y.len()) { let bi = match v { Value::Real(r)=>*r, Value::Integer(n)=>*n as f64, _=>0.0 }; y[i] += bi; } } }
    Some(y)
}

fn apply_activation(kind: &str, x: &mut [f64]) {
    match kind.to_lowercase().as_str() {
        "relu" => { for v in x.iter_mut() { if *v < 0.0 { *v = 0.0; } } }
        "sigmoid" => { for v in x.iter_mut() { *v = 1.0/(1.0+(-*v).exp()); } }
        "tanh" => { for v in x.iter_mut() { *v = v.tanh(); } }
        "gelu" => {
            for v in x.iter_mut() {
                let u = *v;
                // tanh approximation
                let k = (0.79788456*(u + 0.044715*u*u*u)).tanh();
                *v = 0.5*u*(1.0 + k);
            }
        }
        "softmax" => {
            // in-place: compute stable softmax
            let maxv = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0; for v in x.iter_mut() { *v = (*v - maxv).exp(); sum += *v; }
            if sum != 0.0 { for v in x.iter_mut() { *v /= sum; } }
        }
        _ => {}
    }
}

fn ensure_conv1d_params(id: i64, layer_idx: usize, layer: &Value, _in_dim: usize) -> (Value, usize, usize) {
    // returns (updated_layer, out_channels, kernel_size)
    if let Value::Assoc(m) = layer {
        if let Some(Value::Assoc(params)) = m.get("Params") {
            let out_ch = assoc_get(params, "Output").and_then(|v| as_i64(v)).unwrap_or(1).max(1) as usize;
            let k = assoc_get(params, "KernelSize").and_then(|v| as_i64(v)).unwrap_or(3).max(1) as usize;
            let has_w = assoc_get(params, "W").is_some();
            let has_b = assoc_get(params, "b").is_some();
            if has_w && has_b { return (layer.clone(), out_ch, k); }
            let mut p2 = params.clone();
            let mut seed = (id as u64) ^ ((layer_idx as u64) << 9) ^ 0x517CC1B7;
            // one kernel per out channel
            let mut kernels: Vec<Value> = Vec::with_capacity(out_ch);
            for _ in 0..out_ch {
                let mut ker: Vec<Value> = Vec::with_capacity(k);
                for _ in 0..k { ker.push(Value::Real(lcg(&mut seed))); }
                kernels.push(Value::List(ker));
            }
            let mut b: Vec<Value> = Vec::with_capacity(out_ch);
            for _ in 0..out_ch { b.push(Value::Real(0.0)); }
            p2.insert("W".into(), Value::List(kernels));
            p2.insert("b".into(), Value::List(b));
            let mut m2 = m.clone(); m2.insert("Params".into(), Value::Assoc(p2));
            return (Value::Assoc(m2), out_ch, k);
        }
    }
    (layer.clone(), 1, 3)
}

fn apply_conv1d(params: &HashMap<String, Value>, x: &[f64]) -> Option<Vec<f64>> {
    let w = assoc_get(params, "W")?; let b = assoc_get(params, "b")?;
    let out_ch = match w { Value::List(rows) => rows.len(), _=>0 };
    let k = match w { Value::List(rows) => rows.get(0).and_then(|v| if let Value::List(xs)=v { Some(xs.len()) } else { None }).unwrap_or(0), _=>0 };
    let stride = assoc_get(params, "Stride").and_then(|v| as_i64(v)).unwrap_or(1).max(1) as usize;
    let pad = assoc_get(params, "Padding").and_then(|v| as_i64(v)).unwrap_or(0).max(0) as usize;
    let n = x.len();
    let out_len = if n + 2*pad >= k { ((n + 2*pad - k)/stride) + 1 } else { 0 };
    let mut y = vec![0.0f64; out_len * out_ch];
    if out_len == 0 { return Some(y); }
    for oc in 0..out_ch {
        // bias
        let b_oc = if let Value::List(bs) = b { bs.get(oc).and_then(|v| match v { Value::Real(r)=>Some(*r), Value::Integer(n)=>Some(*n as f64), _=>None }).unwrap_or(0.0) } else { 0.0 };
        // kernel
        let ker: Vec<f64> = match w { Value::List(rows) => {
            if let Some(Value::List(xs)) = rows.get(oc) { xs.iter().map(|v| match v { Value::Real(r)=>*r, Value::Integer(n)=>*n as f64, _=>0.0 }).collect() } else { vec![0.0; k] }
        } _ => vec![0.0; k] };
        for i in 0..out_len {
            let mut acc = 0.0f64;
            for j in 0..k {
                let xi = i*stride + j;
                let xv = if xi < pad { 0.0 } else { let ii = xi - pad; if ii < n { x[ii] } else { 0.0 } };
                acc += ker[j] * xv;
            }
            y[oc*out_len + i] = acc + b_oc;
        }
    }
    Some(y)
}

fn apply_pool1d(params: &HashMap<String, Value>, x: &[f64]) -> Option<Vec<f64>> {
    let size = assoc_get(params, "Size").and_then(|v| as_i64(v)).unwrap_or(2).max(1) as usize;
    let stride = assoc_get(params, "Stride").and_then(|v| as_i64(v)).unwrap_or(size as i64).max(1) as usize;
    let kind = assoc_get(params, "PoolType").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None }).unwrap_or("Max".into());
    let n = x.len(); if n < size { return Some(vec![]); }
    let out_len = ((n - size)/stride) + 1;
    let mut y = vec![0.0f64; out_len];
    for i in 0..out_len {
        let start = i*stride; let end = start + size;
        let window = &x[start..end];
        y[i] = if kind.eq_ignore_ascii_case("avg") || kind.eq_ignore_ascii_case("average") {
            window.iter().sum::<f64>() / (size as f64)
        } else {
            window.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        };
    }
    Some(y)
}

fn apply_batchnorm(params: &HashMap<String, Value>, x: &mut [f64]) {
    let eps = assoc_get(params, "Epsilon").and_then(|v| match v { Value::Real(r)=>Some(*r), Value::Integer(n)=>Some(*n as f64), _=>None }).unwrap_or(1e-5);
    let gamma = assoc_get(params, "Gamma");
    let beta = assoc_get(params, "Beta");
    let mean = if x.is_empty() { 0.0 } else { x.iter().sum::<f64>()/(x.len() as f64) };
    let var = if x.is_empty() { 1.0 } else { let m=mean; let mut s=0.0; for v in x.iter() { let d = *v - m; s += d*d; } s/(x.len() as f64) };
    let inv = 1.0/ (var + eps).sqrt();
    for i in 0..x.len() {
        let mut y = (x[i]-mean) * inv;
        let g = match gamma { Some(Value::List(gs)) => gs.get(i).and_then(|v| match v { Value::Real(r)=>Some(*r), Value::Integer(n)=>Some(*n as f64), _=>None }).unwrap_or(1.0), Some(Value::Real(r))=>*r, Some(Value::Integer(n))=>*n as f64, _=>1.0 };
        let be = match beta { Some(Value::List(bs)) => bs.get(i).and_then(|v| match v { Value::Real(r)=>Some(*r), Value::Integer(n)=>Some(*n as f64), _=>None }).unwrap_or(0.0), Some(Value::Real(r))=>*r, Some(Value::Integer(n))=>*n as f64, _=>0.0 };
        x[i] = y*g + be;
    }
}

fn apply_layernorm(params: &HashMap<String, Value>, x: &mut [f64]) { apply_batchnorm(params, x) }

fn build_shape_nested(xs: &[f64], shape: &[usize]) -> Value {
    if shape.is_empty() { return Value::List(vec![]); }
    fn rec(vs: &[f64], shape: &[usize], idx: &mut usize) -> Value {
        if shape.len() == 1 {
            let n = shape[0];
            let mut out: Vec<Value> = Vec::with_capacity(n);
            for _ in 0..n { let x = vs.get(*idx).cloned().unwrap_or(0.0); *idx += 1; out.push(Value::Real(x)); }
            Value::List(out)
        } else {
            let n = shape[0];
            let mut out: Vec<Value> = Vec::with_capacity(n);
            for _ in 0..n { out.push(rec(vs, &shape[1..], idx)); }
            Value::List(out)
        }
    }
    let mut i = 0usize;
    rec(xs, shape, &mut i)
}

// --------- constructors ---------

fn net_chain(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let (pos, opts) = parse_opts(ev, &args);
    let layers = if pos.is_empty() { vec![] } else { match ev.eval(pos[0].clone()) { Value::List(vs)=>vs, v=>vec![v] } };
    let id = next_id();
    reg().lock().unwrap().insert(id, NetState { kind: "Chain".into(), layers, graph: None, opts, encoder: None, decoder: None, initialized: false, trained_epochs: 0, method: "Adam".into(), batch_size: 32 });
    net_handle(id)
}

fn net_graph(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // NetGraph[assoc, edges, opts]
    let (_pos, opts) = parse_opts(ev, &args);
    let id = next_id();
    let graph = if args.len()>=2 { Some(ev.eval(args[1].clone())) } else { None };
    let layers = if args.len()>=1 { match ev.eval(args[0].clone()) { Value::Assoc(m)=> m.into_iter().map(|(k,v)| Value::expr(Value::symbol("Rule"), vec![Value::String(k), v])).collect(), _=>vec![] } } else { vec![] };
    reg().lock().unwrap().insert(id, NetState { kind: "Graph".into(), layers, graph, opts, encoder: None, decoder: None, initialized: false, trained_epochs: 0, method: "Adam".into(), batch_size: 32 });
    net_handle(id)
}

fn net_initialize(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("NetInitialize".into())), args } }
    let net = args[0].clone();
    if let Some(id) = get_net_id(&net) { if let Some(mut st) = reg().lock().unwrap().remove(&id) { st.initialized = true; reg().lock().unwrap().insert(id, st); }
        return net;
    }
    net
}

// --------- train/apply ---------

fn net_train(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("NetTrain".into())), args } }
    let (pos, opts) = parse_opts(ev, &args);
    let net = ev.eval(pos.get(0).cloned().unwrap_or(Value::Symbol("Null".into())));
    let _data = pos.get(1).cloned().unwrap_or(Value::List(vec![]));
    let epochs = get_option(&opts, "Epochs").and_then(|v| match v { Value::Integer(n)=>Some(*n as usize), Value::Real(x)=>Some(*x as usize), _=>None }).unwrap_or(1);
    let method = get_option(&opts, "Method").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None }).unwrap_or("Adam".into());
    let batch = get_option(&opts, "BatchSize").and_then(|v| match v { Value::Integer(n)=>Some(*n as usize), Value::Real(x)=>Some(*x as usize), _=>None }).unwrap_or(32);
    if let Some(id) = get_net_id(&net) { if let Some(mut st) = reg().lock().unwrap().remove(&id) { st.trained_epochs += epochs; st.method = method; st.batch_size = batch; reg().lock().unwrap().insert(id, st); } }
    // Return callable model: PureFunction[x |-> NetApply[net, x, opts]]
    let body = Value::expr(Value::symbol("NetApply"), vec![net.clone(), Value::slot(None), Value::slot(Some(2))]);
    Value::PureFunction { params: None, body: Box::new(body) }
}

fn net_apply(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("NetApply".into())), args } }
    if args.len() < 2 { return Value::Symbol("Null".into()); }
    let net_v = ev.eval(args[0].clone());
    let x_in = ev.eval(args[1].clone());
    let opts = if args.len() >= 3 { ev.eval(args[2].clone()) } else { Value::Assoc(HashMap::new()) };
    let mut x_vec: Vec<f64> = match to_vec_f64(&x_in) {
        Some(v) => v,
        None => { let mut out = Vec::new(); if flatten_to_vec_f64(&x_in, &mut out) { out } else { return x_in; } }
    };
    let id = match get_net_id(&net_v) { Some(id)=>id, None=> return x_in };
    let mut state = match reg().lock().unwrap().remove(&id) { Some(s)=>s, None=> return x_in };
    // Only NetChain forward is supported for MVP
    if state.kind == "Chain" {
        // Iterate layers; lazy init for Linear when needed
        let mut new_layers: Vec<Value> = Vec::with_capacity(state.layers.len());
        let mut in_dim = x_vec.len();
        let mut pending_shape: Option<Vec<usize>> = None;
        for (idx, layer) in state.layers.iter().enumerate() {
            let mut current = layer.clone();
            if let Value::Assoc(m) = &layer {
                let ltype = m.get("LayerType").and_then(|v| match v { Value::String(s)=>Some(s.as_str().to_string()), Value::Symbol(s)=>Some(s.clone()), _=>None }).unwrap_or_default();
                let params = m.get("Params").and_then(|v| if let Value::Assoc(p)=v { Some(p.clone()) } else { None }).unwrap_or_default();
                match ltype.as_str() {
                    "Linear" => {
                        let (updated, out_dim) = ensure_linear_params(id, idx, layer, in_dim);
                        current = updated;
                        // apply
                        if let Value::Assoc(m2) = &current { if let Some(Value::Assoc(p2)) = m2.get("Params") { if let Some(y) = apply_linear(p2, &x_vec) { x_vec = y; in_dim = x_vec.len(); } } }
                    }
                    "Convolution" => {
                        let (updated, out_ch, _k) = ensure_conv1d_params(id, idx, layer, in_dim);
                        current = updated;
                        if let Value::Assoc(m2) = &current { if let Some(Value::Assoc(p2)) = m2.get("Params") { if let Some(y) = apply_conv1d(p2, &x_vec) { x_vec = y; in_dim = x_vec.len(); } } }
                    }
                    "Pooling" => {
                        if let Some(y) = apply_pool1d(&params, &x_vec) { x_vec = y; in_dim = x_vec.len(); }
                    }
                    "BatchNorm" => { apply_batchnorm(&params, &mut x_vec); }
                    "LayerNorm" => { apply_layernorm(&params, &mut x_vec); }
                    "Activation" => {
                        let kind = params.get("Type").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None }).unwrap_or("ReLU".into());
                        apply_activation(&kind, &mut x_vec);
                    }
                    "Flatten" => {
                        // no-op: we already flattened input when needed
                    }
                    "Dropout" => {
                        // inference-time no-op
                    }
                    "Reshape" => {
                        // store shape for final output reconstruction
                        if let Some(Value::List(xs)) = params.get("Shape") {
                            let mut shp: Vec<usize> = Vec::new();
                            for v in xs { if let Some(n)=as_i64(v) { shp.push(n.max(0) as usize); } }
                            if !shp.is_empty() { pending_shape = Some(shp); }
                        }
                    }
                    "Embedding" => {
                        // Only support as first layer on integer input
                        if idx == 0 {
                            let token = match &x_in { Value::Integer(n)=>Some(*n), _=>None };
                            if let Some(tok) = token { if let Some(dim) = as_i64(params.get("Dim").unwrap_or(&Value::Integer(16))) { let d = dim.max(1) as usize; let mut seed = (id as u64) ^ 0xDEADBEEF ^ ((tok as u64) << 17); let mut out = vec![0.0f64; d]; for i in 0..d { out[i] = lcg(&mut seed); } x_vec = out; in_dim = x_vec.len(); } }
                        }
                    }
                    _ => {}
                }
            }
            new_layers.push(current);
        }
        state.layers = new_layers;
        // put back updated state
        reg().lock().unwrap().insert(id, state);
        if let Some(shape) = pending_shape { return build_shape_nested(&x_vec, &shape); }
        return from_vec_like(&x_in, &x_vec);
    }
    // Unknown net kind: return input
    reg().lock().unwrap().insert(id, state);
    x_in
}

// --------- properties/summary ---------

fn net_property(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("NetProperty".into())), args } }
    let net_v = ev.eval(args[0].clone());
    let prop = ev.eval(args[1].clone());
    let id = match get_net_id(&net_v) { Some(id)=>id, None=> return Value::Expr { head: Box::new(Value::Symbol("NetProperty".into())), args } };
    let st = match reg().lock().unwrap().get(&id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("NetProperty".into())), args } };
    match prop {
        Value::String(s) | Value::Symbol(s) => {
            match s.as_str() {
                "Properties" => Value::List(vec![
                    "Kind","Layers","Graph","Initialized","Method","Epochs","BatchSize","Encoder","Decoder","LayerSummaries"
                ].into_iter().map(|k| Value::String(k.into())).collect()),
                "Kind" => Value::String(st.kind),
                "Layers" => Value::List(st.layers),
                "Graph" => st.graph.unwrap_or(Value::Symbol("Null".into())),
                "Initialized" => Value::Boolean(st.initialized),
                "Method" => Value::String(st.method),
                "Epochs" => Value::Integer(st.trained_epochs as i64),
                "BatchSize" => Value::Integer(st.batch_size as i64),
                "Encoder" => st.encoder.unwrap_or(Value::Symbol("Automatic".into())),
                "Decoder" => st.decoder.unwrap_or(Value::Symbol("Automatic".into())),
                "LayerSummaries" => {
                    let mut out: Vec<Value> = Vec::with_capacity(st.layers.len());
                    for l in &st.layers { out.push(layer_summary(l)); }
                    Value::List(out)
                }
                _ => Value::Symbol("Null".into()),
            }
        }
        _ => Value::Symbol("Null".into()),
    }
}

fn net_summary(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("NetSummary".into())), args } }
    let net_v = ev.eval(args[0].clone());
    let id = match get_net_id(&net_v) { Some(id)=>id, None=> return Value::Expr { head: Box::new(Value::Symbol("NetSummary".into())), args } };
    let st = match reg().lock().unwrap().get(&id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("NetSummary".into())), args } };
    Value::Assoc(HashMap::from([
        ("Kind".into(), Value::String(st.kind)),
        ("LayerCount".into(), Value::Integer(st.layers.len() as i64)),
        ("Initialized".into(), Value::Boolean(st.initialized)),
        ("Method".into(), Value::String(st.method)),
        ("Epochs".into(), Value::Integer(st.trained_epochs as i64)),
    ]))
}

fn layer_summary(layer: &Value) -> Value {
    if let Value::Assoc(m) = layer {
        let ltype = m.get("LayerType").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None }).unwrap_or("Unknown".into());
        let params = m.get("Params").and_then(|v| if let Value::Assoc(p)=v { Some(p.clone()) } else { None }).unwrap_or_default();
        let mut out: HashMap<String, Value> = HashMap::new();
        out.insert("LayerType".into(), Value::String(ltype.clone()));
        match ltype.as_str() {
            "Linear" => {
                if let Some(v)=params.get("Output") { out.insert("Output".into(), v.clone()); }
                if let Some(v)=params.get("Bias") { out.insert("Bias".into(), v.clone()); }
            }
            "Activation" => { if let Some(v)=params.get("Type") { out.insert("Type".into(), v.clone()); } }
            "Convolution" => {
                for k in ["Output","KernelSize","Stride","Padding"] { if let Some(v)=params.get(k) { out.insert(k.into(), v.clone()); } }
            }
            "Pooling" => { for k in ["PoolType","Size","Stride"] { if let Some(v)=params.get(k) { out.insert(k.into(), v.clone()); } } }
            "BatchNorm"|"LayerNorm" => { if let Some(v)=params.get("Epsilon") { out.insert("Epsilon".into(), v.clone()); } }
            "Embedding" => { for k in ["Vocab","Dim"] { if let Some(v)=params.get(k) { out.insert(k.into(), v.clone()); } } }
            "Reshape" => { if let Some(v)=params.get("Shape") { out.insert("Shape".into(), v.clone()); } }
            _ => {}
        }
        return Value::Assoc(out);
    }
    Value::assoc(vec![("LayerType", Value::String("Unknown".into()))])
}

// --------- encoders/decoders (stubs) ---------

fn net_encoder(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // NetEncoder[spec|auto]
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("NetEncoder".into())), args } }
    match &args[0] { Value::String(_)|Value::Symbol(_)|Value::Assoc(_)|Value::List(_) => args[0].clone(), _ => Value::Symbol("Automatic".into()) }
}

fn net_decoder(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // NetDecoder[spec|auto]
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("NetDecoder".into())), args } }
    match &args[0] { Value::String(_)|Value::Symbol(_)|Value::Assoc(_)|Value::List(_) => args[0].clone(), _ => Value::Symbol("Automatic".into()) }
}

// --------- layer constructors (descriptive only) ---------

fn linear_layer(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let (pos, opts) = parse_opts(ev, &args);
    // Accept LinearLayer[out] or LinearLayer[<|"Output"->n, ...|>]
    let mut params: HashMap<String, Value> = HashMap::new();
    if let Some(Value::Assoc(m)) = args.last().and_then(|v| match v { Value::Assoc(_) => Some(ev.eval(v.clone())), _ => None }) { params = m; }
    if let Some(first) = pos.get(0) {
        match ev.eval(first.clone()) {
            Value::Integer(n) => { params.entry("Output".into()).or_insert(Value::Integer(n)); },
            Value::Real(x) => { params.entry("Output".into()).or_insert(Value::Integer(x as i64)); },
            _ => {}
        }
    }
    for (k,v) in opts { params.insert(k, v); }
    if !params.contains_key("Bias") { params.insert("Bias".into(), Value::Boolean(true)); }
    if !params.contains_key("Output") { params.insert("Output".into(), Value::Integer(0)); }
    layer_spec("Linear", params)
}

fn activation_layer(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let (pos, opts) = parse_opts(ev, &args);
    let mut params: HashMap<String, Value> = HashMap::new();
    if let Some(Value::Assoc(m)) = args.last().and_then(|v| match v { Value::Assoc(_) => Some(ev.eval(v.clone())), _ => None }) { params = m; }
    if let Some(first) = pos.get(0) {
        match ev.eval(first.clone()) { Value::String(s)|Value::Symbol(s) => { params.entry("Type".into()).or_insert(Value::String(s)); }, _ => {} }
    }
    for (k,v) in opts { params.insert(k, v); }
    // normalize/validate type
    let at = params.get("Type").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None }).unwrap_or("ReLU".into());
    let at_lc = at.to_lowercase();
    let valid = ["relu","sigmoid","tanh","gelu","softmax"];
    let at_norm = if valid.contains(&at_lc.as_str()) { match at_lc.as_str() { "relu"=>"ReLU", "sigmoid"=>"Sigmoid", "tanh"=>"Tanh", "gelu"=>"GELU", "softmax"=>"Softmax", _=>"ReLU" }.to_string() } else { "ReLU".into() };
    params.insert("Type".into(), Value::String(at_norm));
    layer_spec("Activation", params)
}

fn dropout_layer(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let (pos, opts) = parse_opts(ev, &args);
    let mut params: HashMap<String, Value> = HashMap::new();
    if let Some(first) = pos.get(0) { match ev.eval(first.clone()) { Value::Real(x)=>{ params.insert("Rate".into(), Value::Real(x)); }, Value::Integer(n)=>{ params.insert("Rate".into(), Value::Real(n as f64)); }, _=>{} } }
    for (k,v) in opts { params.insert(k, v); }
    if !params.contains_key("Rate") { params.insert("Rate".into(), Value::Real(0.5)); }
    layer_spec("Dropout", params)
}

fn flatten_layer(_ev: &mut Evaluator, _args: Vec<Value>) -> Value {
    layer_spec("Flatten", HashMap::new())
}

fn softmax_layer(_ev: &mut Evaluator, _args: Vec<Value>) -> Value { layer_spec("Activation", HashMap::from([("Type".into(), Value::String("Softmax".into()))])) }

fn convolution_layer(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // 1D conv constructor: ConvolutionLayer[outChannels, kernelSize, opts]
    let (pos, opts) = parse_opts(ev, &args);
    let mut params: HashMap<String, Value> = HashMap::new();
    if let Some(Value::Assoc(m)) = args.last().and_then(|v| match v { Value::Assoc(_) => Some(ev.eval(v.clone())), _ => None }) { params = m; }
    if let Some(first) = pos.get(0) { if let Some(n) = as_i64(&ev.eval(first.clone())) { params.entry("Output".into()).or_insert(Value::Integer(n)); } }
    if let Some(second) = pos.get(1) { if let Some(n) = as_i64(&ev.eval(second.clone())) { params.entry("KernelSize".into()).or_insert(Value::Integer(n)); } }
    for (k,v) in opts { params.insert(k, v); }
    if !params.contains_key("Output") { params.insert("Output".into(), Value::Integer(1)); }
    if !params.contains_key("KernelSize") { params.insert("KernelSize".into(), Value::Integer(3)); }
    if !params.contains_key("Stride") { params.insert("Stride".into(), Value::Integer(1)); }
    if !params.contains_key("Padding") { params.insert("Padding".into(), Value::Integer(0)); }
    layer_spec("Convolution", params)
}

fn pooling_layer(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // 1D pooling: PoolingLayer["Max"|"Avg", size, opts]
    let (pos, opts) = parse_opts(ev, &args);
    let mut params: HashMap<String, Value> = HashMap::new();
    if let Some(kind) = pos.get(0) { if let Value::String(s)|Value::Symbol(s) = ev.eval(kind.clone()) { params.insert("PoolType".into(), Value::String(s)); } }
    if let Some(sz) = pos.get(1) { if let Some(n) = as_i64(&ev.eval(sz.clone())) { params.insert("Size".into(), Value::Integer(n)); } }
    for (k,v) in opts { params.insert(k, v); }
    if !params.contains_key("PoolType") { params.insert("PoolType".into(), Value::String("Max".into())); }
    if !params.contains_key("Size") { params.insert("Size".into(), Value::Integer(2)); }
    if !params.contains_key("Stride") { params.insert("Stride".into(), Value::Integer(2)); }
    layer_spec("Pooling", params)
}

fn batchnorm_layer(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let (_pos, opts) = parse_opts(ev, &args);
    let mut params = opts;
    if !params.contains_key("Epsilon") { params.insert("Epsilon".into(), Value::Real(1e-5)); }
    layer_spec("BatchNorm", params)
}

fn reshape_layer(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let (pos, opts) = parse_opts(ev, &args);
    let mut params = opts;
    if let Some(shape) = pos.get(0) { params.insert("Shape".into(), ev.eval(shape.clone())); }
    layer_spec("Reshape", params)
}

fn transpose_layer(ev: &mut Evaluator, args: Vec<Value>) -> Value { let (_pos, opts) = parse_opts(ev, &args); layer_spec("Transpose", opts) }
fn concat_layer(ev: &mut Evaluator, args: Vec<Value>) -> Value { let (_pos, opts) = parse_opts(ev, &args); layer_spec("Concat", opts) }
fn add_layer(_ev: &mut Evaluator, _args: Vec<Value>) -> Value { layer_spec("Add", HashMap::new()) }
fn mul_layer(_ev: &mut Evaluator, _args: Vec<Value>) -> Value { layer_spec("Mul", HashMap::new()) }

fn embedding_layer(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // EmbeddingLayer[vocab, dim]
    let (pos, opts) = parse_opts(ev, &args);
    let mut params = opts;
    if let Some(v) = pos.get(0) { if let Some(n) = as_i64(&ev.eval(v.clone())) { params.insert("Vocab".into(), Value::Integer(n)); } }
    if let Some(v) = pos.get(1) { if let Some(n) = as_i64(&ev.eval(v.clone())) { params.insert("Dim".into(), Value::Integer(n)); } }
    if !params.contains_key("Vocab") { params.insert("Vocab".into(), Value::Integer(1000)); }
    if !params.contains_key("Dim") { params.insert("Dim".into(), Value::Integer(16)); }
    layer_spec("Embedding", params)
}

fn layernorm_layer(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let (_pos, opts) = parse_opts(ev, &args);
    let mut params = opts;
    if !params.contains_key("Epsilon") { params.insert("Epsilon".into(), Value::Real(1e-5)); }
    layer_spec("LayerNorm", params)
}
