use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
#[cfg(feature = "tools")] use crate::tools::add_specs;
#[cfg(feature = "tools")] use crate::tool_spec;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

// Simple in-memory model registry (now with basic linear/logistic models)
#[derive(Clone)]
struct ModelState {
    task: String,                 // "classification" | "regression" | "clustering" | ...
    method: String,               // e.g., "Baseline"
    // Classification
    classes: Option<Vec<Value>>,  // distinct labels
    class_counts: Option<HashMap<String, i64>>, // key: pretty(value)
    majority: Option<Value>,
    // Regression
    mean: Option<f64>,
    var: Option<f64>,
    // Misc
    train_size: usize,
    // Preprocessing pipeline (impute/encode/standardize)
    preproc: Option<Preproc>,
    // Linear/Logistic parameters
    // Regression
    lin_w: Option<Vec<f64>>,   // weights
    lin_b: Option<f64>,        // bias
    resid_std: Option<f64>,    // residual std for intervals
    // Classification (one-vs-rest for multi-class)
    cls_w: Option<Vec<Vec<f64>>>, // per-class weights
    cls_b: Option<Vec<f64>>,      // per-class biases
}

#[derive(Clone)]
struct Preproc {
    feature_cols: Vec<String>,
    numeric_cols: Vec<String>,
    categorical_cols: Vec<String>,
    num_stats: HashMap<String, (f64, f64)>, // mean, std
    cat_vocab: HashMap<String, Vec<String>>, // col -> categories
    dim: usize,
}

static ML_REG: OnceLock<Mutex<HashMap<i64, ModelState>>> = OnceLock::new();
static NEXT_ID: OnceLock<std::sync::atomic::AtomicI64> = OnceLock::new();

fn ml_reg() -> &'static Mutex<HashMap<i64, ModelState>> { ML_REG.get_or_init(|| Mutex::new(HashMap::new())) }
fn next_id() -> i64 { let a = NEXT_ID.get_or_init(|| std::sync::atomic::AtomicI64::new(1)); a.fetch_add(1, std::sync::atomic::Ordering::Relaxed) }

fn model_handle(id: i64) -> Value { Value::assoc(vec![("__type", Value::String("MLModel".into())), ("id", Value::Integer(id))]) }
fn get_model_id(v: &Value) -> Option<i64> {
    if let Value::Assoc(m) = v { if matches!(m.get("__type"), Some(Value::String(s)) if s=="MLModel") { if let Some(Value::Integer(id)) = m.get("id") { return Some(*id); } } }
    None
}

pub fn register_ml(ev: &mut Evaluator) {
    ev.register("Classify", classify as NativeFn, Attributes::empty());
    ev.register("Predict", predict as NativeFn, Attributes::empty());
    ev.register("Cluster", cluster as NativeFn, Attributes::empty());
    ev.register("FeatureExtract", feature_extract as NativeFn, Attributes::empty());
    ev.register("DimensionReduce", dimension_reduce as NativeFn, Attributes::empty());
    // Function-object application and properties
    ev.register("MLApply", ml_apply as NativeFn, Attributes::empty());
    ev.register("MLProperty", ml_property as NativeFn, Attributes::empty());
    // Measurements
    ev.register("ClassifyMeasurements", classify_measurements as NativeFn, Attributes::empty());
    ev.register("PredictMeasurements", predict_measurements as NativeFn, Attributes::empty());
    // CV / Tuning
    ev.register("MLCrossValidate", ml_cross_validate as NativeFn, Attributes::empty());
    ev.register("MLTune", ml_tune as NativeFn, Attributes::empty());

    #[cfg(feature = "tools")]
    add_specs(vec![
        tool_spec!("Classify", summary: "Train a classifier (baseline/logistic)", params: ["data","opts"], tags: ["ml","classification"]),
        tool_spec!("Predict", summary: "Train a regressor (baseline/linear)", params: ["data","opts"], tags: ["ml","regression"]),
        tool_spec!("Cluster", summary: "Cluster points (prototype)", params: ["data","opts"], tags: ["ml","clustering"]),
        tool_spec!("FeatureExtract", summary: "Learn preprocessing (impute/encode/standardize)", params: ["data","opts"], tags: ["ml","preprocess"]),
        tool_spec!("DimensionReduce", summary: "Reduce dimensionality (PCA-like)", params: ["data","opts"], tags: ["ml","preprocess"]),
        tool_spec!("MLApply", summary: "Apply a trained model to input", params: ["model","x","opts"], tags: ["ml","inference"], examples: [
            Value::Assoc(HashMap::from([
                ("args".into(), Value::Assoc(HashMap::from([
                    ("model".into(), Value::Assoc(HashMap::from([(String::from("__type"), Value::String(String::from("MLModel")))]))),
                    ("x".into(), Value::Assoc(HashMap::from([(String::from("x"), Value::Integer(1))])))
                ])))
            ]))
        ]),
        tool_spec!("MLProperty", summary: "Inspect trained model properties", params: ["model","prop"], tags: ["ml","introspect"]),
        tool_spec!("ClassifyMeasurements", summary: "Evaluate classifier metrics", params: ["model","data","opts"], tags: ["ml","metrics"]),
        tool_spec!("PredictMeasurements", summary: "Evaluate regressor metrics", params: ["model","data","opts"], tags: ["ml","metrics"]),
        tool_spec!("MLCrossValidate", summary: "Cross-validate with simple split", params: ["data","opts"], tags: ["ml","cv"]),
        tool_spec!("MLTune", summary: "Parameter sweep with basic scoring", params: ["data","opts"], tags: ["ml","tune"]),
    ]);
}

// ---------- Utilities ----------

fn to_list(ev: &mut Evaluator, v: Value) -> Vec<Value> { match ev.eval(v) { Value::List(xs)=>xs, other=>vec![other] } }

fn is_assoc(v: &Value) -> bool { matches!(v, Value::Assoc(_)) }

fn get_option<'a>(opts: &'a HashMap<String, Value>, key: &str) -> Option<&'a Value> { opts.get(key).or_else(|| opts.get(&key.to_string())) }

fn parse_opts(ev: &mut Evaluator, args: &[Value]) -> (Vec<Value>, HashMap<String, Value>) {
    if args.is_empty() { return (vec![], HashMap::new()); }
    let mut pos: Vec<Value> = Vec::new();
    let mut opts: HashMap<String, Value> = HashMap::new();
    for (i, a) in args.iter().enumerate() {
        if i == args.len() - 1 && matches!(ev.eval(a.clone()), Value::Assoc(_)) {
            if let Value::Assoc(m) = ev.eval(a.clone()) { opts = m; }
        } else { pos.push(a.clone()); }
    }
    (pos, opts)
}

fn collect_dataset_rows(ev: &mut Evaluator, ds: &Value) -> Option<Vec<HashMap<String, Value>>> {
    let rows_v = ev.eval(Value::expr(Value::symbol("Collect"), vec![ds.clone()]));
    match rows_v {
        Value::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items { if let Value::Assoc(m) = it { out.push(m); } }
            Some(out)
        }
        _ => None,
    }
}

fn pretty_key(v: &Value) -> String { lyra_core::pretty::format_value(v) }

fn as_f64(v: &Value) -> Option<f64> { match v { Value::Integer(n)=>Some(*n as f64), Value::Real(x)=>Some(*x), _=>None } }

fn z_for_conf(level: f64) -> f64 {
    // quick approximations for common levels
    if (level-0.90).abs() < 1e-6 { 1.6449 }
    else if (level-0.95).abs() < 1e-6 { 1.96 }
    else if (level-0.99).abs() < 1e-6 { 2.575 }
    else { 1.96 }
}

// Detect schema and build preprocessing from rows of Assoc or list-of-numeric
fn build_preproc_from_pairs(pairs: &[(Value, Value)], features_opt: Option<Vec<String>>, target: Option<String>) -> Preproc {
    // Gather keys and types
    // If x is List -> positional features f0,f1,... treated numeric
    let mut feature_cols: Vec<String> = Vec::new();
    let mut numeric_cols: Vec<String> = Vec::new();
    let mut categorical_cols: Vec<String> = Vec::new();
    let mut num_sums: HashMap<String, f64> = HashMap::new();
    let mut num_cnts: HashMap<String, f64> = HashMap::new();
    let mut cat_counts: HashMap<String, HashMap<String, i64>> = HashMap::new();
    for (x, _) in pairs.iter() {
        match x {
            Value::Assoc(m) => {
                for (k, v) in m {
                    if Some(k)==target.as_ref() { continue; }
                    if let Some(fs) = &features_opt { if !fs.contains(k) { continue; } }
                    if !feature_cols.contains(k) { feature_cols.push(k.clone()); }
                    match v { Value::Integer(n) => { *num_sums.entry(k.clone()).or_insert(0.0) += *n as f64; *num_cnts.entry(k.clone()).or_insert(0.0) += 1.0; },
                               Value::Real(x) => { *num_sums.entry(k.clone()).or_insert(0.0) += *x; *num_cnts.entry(k.clone()).or_insert(0.0) += 1.0; },
                               other => { let key = pretty_key(other); let row = cat_counts.entry(k.clone()).or_insert_with(HashMap::new); *row.entry(key).or_insert(0) += 1; } }
                }
            }
            Value::List(vs) => {
                for (i, v) in vs.iter().enumerate() {
                    let k = format!("f{}", i);
                    if !feature_cols.contains(&k) { feature_cols.push(k.clone()); }
                    match v { Value::Integer(n) => { *num_sums.entry(k.clone()).or_insert(0.0) += *n as f64; *num_cnts.entry(k.clone()).or_insert(0.0) += 1.0; },
                               Value::Real(x) => { *num_sums.entry(k.clone()).or_insert(0.0) += *x; *num_cnts.entry(k.clone()).or_insert(0.0) += 1.0; },
                               other => { let key = pretty_key(other); let row = cat_counts.entry(k.clone()).or_insert_with(HashMap::new); *row.entry(key).or_insert(0) += 1; } }
                }
            }
            _ => {}
        }
    }
    // Separate numeric/categorical
    for k in &feature_cols { if num_cnts.contains_key(k) { numeric_cols.push(k.clone()); } else { categorical_cols.push(k.clone()); } }
    // Stats
    let mut num_stats: HashMap<String,(f64,f64)> = HashMap::new();
    for k in &numeric_cols { let mean = num_sums.get(k).cloned().unwrap_or(0.0)/num_cnts.get(k).cloned().unwrap_or(1.0); num_stats.insert(k.clone(), (mean, 1.0)); }
    let mut cat_vocab: HashMap<String, Vec<String>> = HashMap::new();
    for (k, row) in cat_counts { let mut cats: Vec<(String,i64)> = row.into_iter().collect(); cats.sort_by_key(|(_,c)| -c); let vocab: Vec<String> = cats.into_iter().map(|(s,_)| s).take(100).collect(); cat_vocab.insert(k, vocab); }
    // Dim: numeric (1 each) + sum(one-hot sizes)
    let mut dim = numeric_cols.len();
    for v in cat_vocab.values() { dim += v.len(); }
    Preproc { feature_cols, numeric_cols, categorical_cols, num_stats, cat_vocab, dim }
}

fn featurize(pre: &Preproc, x: &Value) -> Vec<f64> {
    let mut out = vec![0.0f64; pre.dim];
    let mut idx = 0usize;
    // Numeric
    for c in &pre.numeric_cols {
        let val = match x { Value::Assoc(m) => m.get(c).and_then(|v| as_f64(v)).unwrap_or(pre.num_stats.get(c).map(|(m,_)| *m).unwrap_or(0.0)),
                            Value::List(vs) => { let i = c.strip_prefix('f').and_then(|s| s.parse::<usize>().ok()).unwrap_or(0); vs.get(i).and_then(|v| as_f64(v)).unwrap_or(pre.num_stats.get(c).map(|(m,_)| *m).unwrap_or(0.0)) },
                            _ => pre.num_stats.get(c).map(|(m,_)| *m).unwrap_or(0.0) };
        let (mean, std) = pre.num_stats.get(c).cloned().unwrap_or((0.0,1.0));
        out[idx] = if std>0.0 { (val-mean)/std } else { val-mean }; idx+=1;
    }
    // Categorical
    for c in &pre.categorical_cols {
        let vocab = pre.cat_vocab.get(c).cloned().unwrap_or_else(|| vec![]);
        let val_key = match x { Value::Assoc(m) => m.get(c).map(|v| pretty_key(v)).unwrap_or_else(|| "__MISSING__".into()),
                                Value::List(vs) => { let i = c.strip_prefix('f').and_then(|s| s.parse::<usize>().ok()).unwrap_or(0); vs.get(i).map(|v| pretty_key(v)).unwrap_or_else(|| "__MISSING__".into()) },
                                _ => "__MISSING__".into() };
        for (j, cat) in vocab.iter().enumerate() { out[idx+j] = if *cat == val_key { 1.0 } else { 0.0 }; }
        idx += vocab.len();
    }
    out
}

fn dot(w: &Vec<f64>, x: &Vec<f64>) -> f64 { w.iter().zip(x.iter()).map(|(a,b)| a*b).sum() }

// ---------- Trainers ----------

fn classify(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Classify".into())), args } }
    let (pos, opts) = parse_opts(ev, &args);
    // Supported inputs: list of Rule[x,y], or Dataset with Target -> col, or (X, y) NDArrays
    let mut labels: Vec<Value> = Vec::new();
    if pos.len() == 1 {
        let train = ev.eval(pos[0].clone());
        match train {
            Value::List(items) => {
                for it in items {
                    if let Value::Expr { head, args } = it { if matches!(*head, Value::Symbol(ref s) if s=="Rule") && args.len()==2 { labels.push(args[1].clone()); } }
                }
            }
            Value::Assoc(_) => {
                let target = get_option(&opts, "Target").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None });
                if let Some(tcol) = target { if let Some(rows) = collect_dataset_rows(ev, &train) { for r in rows { if let Some(y)=r.get(&tcol) { labels.push(y.clone()); } } } }
            }
            _ => {}
        }
    } else if pos.len() == 2 {
        // Assume (X, y) where y is vector/1D NDArray
        let yv = ev.eval(pos[1].clone());
        match yv {
            Value::PackedArray { shape, data } => {
                if shape.len()==1 { for v in data { labels.push(Value::Real(v)); } }
            }
            Value::List(vs) => { labels.extend(vs); }
            other => labels.push(other),
        }
    }
    if labels.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Classify".into())), args }; }
    // Select method
    let method = get_option(&opts, "Method").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None }).unwrap_or("Automatic".into());
    // Build training pairs (x,y)
    let mut pairs: Vec<(Value, Value)> = Vec::new();
    if pos.len()==1 {
        let train = ev.eval(pos[0].clone());
        match train {
            Value::List(items) => { for it in items { if let Value::Expr { head, args } = it { if matches!(*head, Value::Symbol(ref s) if s=="Rule") && args.len()==2 { pairs.push((args[0].clone(), args[1].clone())); } } } }
            Value::Assoc(_) => {
                let target = get_option(&opts, "Target").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None });
                if let Some(tcol) = target { if let Some(rows)=collect_dataset_rows(ev, &train) { for m in rows { let y=m.get(&tcol).cloned().unwrap_or(Value::Symbol("Null".into())); pairs.push((Value::Assoc(m), y)); } } }
            }
            _=>{}
        }
    }
    // Unique classes
    let mut class_values: Vec<Value> = Vec::new();
    for (_, y) in &pairs { let k = pretty_key(y); if !class_values.iter().any(|v| pretty_key(v)==k) { class_values.push(y.clone()); } }
    let use_logistic = method.eq_ignore_ascii_case("logistic") || (method.eq_ignore_ascii_case("automatic") && class_values.len() > 1 && class_values.len() <= 10 && !pairs.is_empty());
    if use_logistic {
        let pre = build_preproc_from_pairs(&pairs, None, None);
        let epochs = get_option(&opts, "Epochs").and_then(|v| as_f64(v)).unwrap_or(50.0) as usize;
        let lr = get_option(&opts, "LearningRate").and_then(|v| as_f64(v)).unwrap_or(0.1);
        let l2 = get_option(&opts, "L2").and_then(|v| as_f64(v)).unwrap_or(0.0);
        let mut w_all: Vec<Vec<f64>> = Vec::new(); let mut b_all: Vec<f64> = Vec::new();
        for c in &class_values {
            let mut w = vec![0.0; pre.dim]; let mut b = 0.0;
            for _ in 0..epochs {
                let mut grad_w = vec![0.0f64; pre.dim]; let mut grad_b = 0.0f64;
                for (x, yv) in pairs.iter() {
                    let xvec = featurize(&pre, x);
                    let y = if pretty_key(yv) == pretty_key(c) { 1.0 } else { 0.0 };
                    let z = dot(&w, &xvec) + b; let p = 1.0/(1.0+(-z).exp()); let err = p - y;
                    for i in 0..pre.dim { grad_w[i] += err * xvec[i]; }
                    grad_b += err;
                }
                let n = pairs.len() as f64;
                for i in 0..pre.dim { w[i] -= lr * ((grad_w[i]/n) + l2*w[i]); }
                b -= lr * (grad_b/n);
            }
            w_all.push(w); b_all.push(b);
        }
        let id = next_id();
        ml_reg().lock().unwrap().insert(id, ModelState { task: "classification".into(), method: "Logistic".into(), classes: Some(class_values), class_counts: None, majority: None, mean: None, var: None, train_size: pairs.len(), preproc: Some(pre), lin_w: None, lin_b: None, resid_std: None, cls_w: Some(w_all), cls_b: Some(b_all) });
        let body = Value::expr(Value::symbol("MLApply"), vec![model_handle(id), Value::slot(None), Value::slot(Some(2))]);
        return Value::PureFunction { params: None, body: Box::new(body) };
    }
    // Majority baseline
    let mut counts: HashMap<String, i64> = HashMap::new();
    for (_, y) in &pairs { let k = pretty_key(y); *counts.entry(k).or_insert(0) += 1; }
    let mut classes: Vec<Value> = Vec::new();
    let mut majority: Option<(Value, i64)> = None;
    for (_, y) in pairs { let k = pretty_key(&y); if !classes.iter().any(|c| pretty_key(c)==k) { classes.push(y.clone()); }
        let c = *counts.get(&k).unwrap_or(&0); if majority.as_ref().map(|(_,m)| c>*m).unwrap_or(true) { majority = Some((y, c)); }
    }
    let id = next_id();
    let maj_y = majority.as_ref().map(|(y, _)| y.clone());
    let maj_n = majority.map(|(_, m)| m as usize).unwrap_or(0);
    ml_reg().lock().unwrap().insert(id, ModelState { task: "classification".into(), method: "Baseline".into(), classes: Some(classes), class_counts: Some(counts), majority: maj_y, mean: None, var: None, train_size: maj_n, preproc: None, lin_w: None, lin_b: None, resid_std: None, cls_w: None, cls_b: None });
    let body = Value::expr(Value::symbol("MLApply"), vec![model_handle(id), Value::slot(None), Value::slot(Some(2))]);
    Value::PureFunction { params: None, body: Box::new(body) }
}

fn predict(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Predict".into())), args } }
    let (pos, _opts) = parse_opts(ev, &args);
    // Supported: list of Rule[x,y], Dataset + Target, or (X, y)
    let mut ys: Vec<f64> = Vec::new();
    if pos.len() == 1 {
        let train = ev.eval(pos[0].clone());
        match train {
            Value::List(items) => {
                for it in items { if let Value::Expr { head, args } = it { if matches!(*head, Value::Symbol(ref s) if s=="Rule") && args.len()==2 {
                    match ev.eval(args[1].clone()) { Value::Integer(n)=>ys.push(n as f64), Value::Real(x)=>ys.push(x), _=>{} }
                } } }
            }
            Value::Assoc(_) => {
                // Dataset with Target column
                // Try to find Target in options from args
                let (_p2, opts2) = parse_opts(ev, &args);
                let target = get_option(&opts2, "Target").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None });
                if let Some(tcol) = target {
                    if let Some(rows) = collect_dataset_rows(ev, &train) {
                        for r in rows {
                            if let Some(v) = r.get(&tcol) {
                                match v { Value::Integer(n)=>ys.push(*n as f64), Value::Real(x)=>ys.push(*x), _=>{} }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    } else if pos.len() == 2 {
        let yv = ev.eval(pos[1].clone());
        match yv {
            Value::PackedArray { shape, data } => { if shape.len()==1 { ys.extend(data); } }
            Value::List(vs) => { for v in vs { match v { Value::Integer(n)=>ys.push(n as f64), Value::Real(x)=>ys.push(x), _=>{} } }
            }
            _ => {}
        }
    }
    if ys.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Predict".into())), args }; }
    // Attempt linear training when possible
    let (pos2, opts2) = parse_opts(ev, &args);
    let method = get_option(&opts2, "Method").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None }).unwrap_or("Automatic".into());
    let mut pairs: Vec<(Value, f64)> = Vec::new();
    if pos2.len()==1 {
        let train = ev.eval(pos2[0].clone());
        match train {
            Value::List(items) => { for it in items { if let Value::Expr { head, args } = it { if matches!(*head, Value::Symbol(ref s) if s=="Rule") && args.len()==2 { let y=as_f64(&ev.eval(args[1].clone())); if let Some(yf)=y { pairs.push((args[0].clone(), yf)); } } } } }
            Value::Assoc(_) => { let target = get_option(&opts2, "Target").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None }); if let Some(tcol)=target { if let Some(rows)=collect_dataset_rows(ev, &train) { for m in rows { if let Some(yv)=m.get(&tcol).and_then(|v| as_f64(v)) { pairs.push((Value::Assoc(m), yv)); } } } } }
            _=>{}
        }
    }
    if !pairs.is_empty() && (method.eq_ignore_ascii_case("linear") || method.eq_ignore_ascii_case("automatic")) {
        let pre = build_preproc_from_pairs(&pairs.iter().map(|(x,y)| (x.clone(), Value::Real(*y))).collect::<Vec<_>>(), None, None);
        let epochs = get_option(&opts2, "Epochs").and_then(|v| as_f64(v)).unwrap_or(50.0) as usize;
        let lr = get_option(&opts2, "LearningRate").and_then(|v| as_f64(v)).unwrap_or(0.1);
        let l2 = get_option(&opts2, "L2").and_then(|v| as_f64(v)).unwrap_or(0.0);
        let mut w = vec![0.0f64; pre.dim]; let mut b = 0.0f64;
        if pre.dim == 0 { // intercept-only: set to mean directly
            b = pairs.iter().map(|(_,y)| *y).sum::<f64>() / (pairs.len() as f64);
        } else {
            for _ in 0..epochs {
                let mut grad_w = vec![0.0f64; pre.dim]; let mut grad_b = 0.0f64;
                for (x, y) in pairs.iter() {
                    let xv = featurize(&pre, x);
                    let yhat = dot(&w, &xv) + b; let err = yhat - *y;
                    for i in 0..pre.dim { grad_w[i] += err * xv[i]; }
                    grad_b += err;
                }
                let n = pairs.len() as f64; for i in 0..pre.dim { w[i] -= lr * ((grad_w[i]/n) + l2*w[i]); } b -= lr * (grad_b/n);
            }
        }
        let mut se = 0.0f64; let mut n = 0usize; for (x,y) in pairs.iter() { let xv=featurize(&pre, x); let yhat=dot(&w,&xv)+b; let e=yhat-*y; se+=e*e; n+=1; }
        let resid_std = if n>1 { (se/(n as f64)).sqrt() } else { 0.0 };
        let id = next_id();
        ml_reg().lock().unwrap().insert(id, ModelState { task: "regression".into(), method: "Linear".into(), classes: None, class_counts: None, majority: None, mean: None, var: None, train_size: pairs.len(), preproc: Some(pre), lin_w: Some(w), lin_b: Some(b), resid_std: Some(resid_std), cls_w: None, cls_b: None });
        let body = Value::expr(Value::symbol("MLApply"), vec![model_handle(id), Value::slot(None), Value::slot(Some(2))]);
        return Value::PureFunction { params: None, body: Box::new(body) };
    }
    // Baseline
    let n = ys.len() as f64; let mean = ys.iter().sum::<f64>()/n; let var = ys.iter().map(|x| (x-mean)*(x-mean)).sum::<f64>()/n.max(1.0);
    let id = next_id();
    ml_reg().lock().unwrap().insert(id, ModelState { task: "regression".into(), method: "Baseline".into(), classes: None, class_counts: None, majority: None, mean: Some(mean), var: Some(var), train_size: ys.len(), preproc: None, lin_w: None, lin_b: None, resid_std: Some(var.sqrt()), cls_w: None, cls_b: None });
    let body = Value::expr(Value::symbol("MLApply"), vec![model_handle(id), Value::slot(None), Value::slot(Some(2))]);
    Value::PureFunction { params: None, body: Box::new(body) }
}

fn cluster(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Cluster".into())), args } }
    let id = next_id();
    ml_reg().lock().unwrap().insert(id, ModelState { task: "clustering".into(), method: "Baseline".into(), classes: None, class_counts: None, majority: None, mean: None, var: None, train_size: 0, preproc: None, lin_w: None, lin_b: None, resid_std: None, cls_w: None, cls_b: None });
    let body = Value::expr(Value::symbol("MLApply"), vec![model_handle(id), Value::slot(None), Value::slot(Some(2))]);
    Value::PureFunction { params: None, body: Box::new(body) }
}

fn feature_extract(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("FeatureExtract".into())), args } }
    // Identity transformer stub
    let id = next_id();
    ml_reg().lock().unwrap().insert(id, ModelState { task: "transform".into(), method: "Identity".into(), classes: None, class_counts: None, majority: None, mean: None, var: None, train_size: 0, preproc: None, lin_w: None, lin_b: None, resid_std: None, cls_w: None, cls_b: None });
    let body = Value::expr(Value::symbol("MLApply"), vec![model_handle(id), Value::slot(None), Value::slot(Some(2))]);
    Value::PureFunction { params: None, body: Box::new(body) }
}

fn dimension_reduce(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("DimensionReduce".into())), args } }
    let (_pos, _opts) = parse_opts(ev, &args);
    let id = next_id();
    ml_reg().lock().unwrap().insert(id, ModelState { task: "transform".into(), method: "PCA-Stub".into(), classes: None, class_counts: None, majority: None, mean: None, var: None, train_size: 0, preproc: None, lin_w: None, lin_b: None, resid_std: None, cls_w: None, cls_b: None });
    let body = Value::expr(Value::symbol("MLApply"), vec![model_handle(id), Value::slot(None), Value::slot(Some(2))]);
    Value::PureFunction { params: None, body: Box::new(body) }
}

// ---------- Apply and Properties ----------

fn ml_apply(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // MLApply[model, x, opts?]
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("MLApply".into())), args } }
    let model_v = ev.eval(args[0].clone());
    let x = args[1].clone();
    let opts = if args.len()>=3 { ev.eval(args[2].clone()) } else { Value::Assoc(HashMap::new()) };
    let id = match get_model_id(&model_v) { Some(id)=>id, None=> return Value::Expr { head: Box::new(Value::Symbol("MLApply".into())), args: vec![model_v, x] } };
    let reg = ml_reg().lock().unwrap();
    let st = match reg.get(&id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("MLApply".into())), args } };
    let opts_map = if let Value::Assoc(m) = opts { m } else { HashMap::new() };
    // List/batch support: if x is a list, map
    if let Value::List(xs) = ev.eval(x.clone()) {
        let mut out: Vec<Value> = Vec::with_capacity(xs.len());
        for xi in xs { out.push(ml_apply(ev, vec![model_v.clone(), xi, Value::Assoc(opts_map.clone())])); }
        return Value::List(out);
    }
    match st.task.as_str() {
        "classification" => {
            let output = get_option(&opts_map, "Output").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None }).unwrap_or("Label".into());
            // If trained logistic, compute real scores
            if st.method.eq_ignore_ascii_case("Logistic") {
                let pre = match &st.preproc { Some(p)=>p.clone(), None=> return Value::Symbol("Null".into()) };
                let cls_w = match &st.cls_w { Some(w)=>w.clone(), None=> vec![] };
                let cls_b = match &st.cls_b { Some(b)=>b.clone(), None=> vec![] };
                let classes = st.classes.clone().unwrap_or(vec![]);
                let xv = featurize(&pre, &x);
                let mut scores: Vec<f64> = Vec::new();
                for (w, b) in cls_w.iter().zip(cls_b.iter()) { scores.push(1.0/(1.0 + (-(dot(w, &xv) + *b)).exp())); }
                // normalize to sum=1 if multi-class
                let sum: f64 = scores.iter().sum();
                let probs_norm: Vec<f64> = if classes.len()>1 && sum>0.0 { scores.iter().map(|s| s/sum).collect() } else { scores };
                if output.eq_ignore_ascii_case("Probabilities") {
                    let mut probs: HashMap<String, Value> = HashMap::new();
                    for (c, p) in classes.iter().zip(probs_norm.iter()) { probs.insert(pretty_key(c), Value::Real(*p)); }
                    return Value::Assoc(probs);
                }
                if output.eq_ignore_ascii_case("TopK") {
                    let k = get_option(&opts_map, "K").and_then(|v| as_f64(v)).unwrap_or(3.0) as usize;
                    let mut pairs: Vec<(String, f64)> = classes.iter().zip(probs_norm.iter()).map(|(c,p)| (pretty_key(c), *p)).collect();
                    pairs.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    let out: Vec<Value> = pairs.into_iter().take(k).map(|(c,p)| Value::assoc(vec![("class", Value::String(c)), ("prob", Value::Real(p))])).collect();
                    return Value::List(out);
                }
                if output.eq_ignore_ascii_case("DecisionValue") {
                    if probs_norm.len()==1 { return Value::Real(probs_norm[0]); }
                    let mut assoc: HashMap<String, Value> = HashMap::new();
                    for (c,p) in classes.iter().zip(probs_norm.iter()) { assoc.insert(pretty_key(c), Value::Real(*p)); }
                    return Value::Assoc(assoc);
                }
                // Default label: argmax probability
                if let Some((idx,_)) = probs_norm.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)) { return classes.get(idx).cloned().unwrap_or(Value::Symbol("Null".into())); }
                return Value::Symbol("Null".into());
            }
            // Baseline model fallback
            if output.eq_ignore_ascii_case("Probabilities") {
                let mut probs: HashMap<String, Value> = HashMap::new();
                if let (Some(classes), Some(counts)) = (st.classes.clone(), st.class_counts.clone()) {
                    let total: f64 = counts.values().map(|c| *c as f64).sum();
                    for c in classes { let k = pretty_key(&c); let p = counts.get(&k).map(|v| (*v as f64)/total).unwrap_or(0.0); probs.insert(k, Value::Real(p)); }
                }
                Value::Assoc(probs)
            } else { st.majority.clone().unwrap_or(Value::Symbol("Null".into())) }
        }
        "regression" => {
            let output = get_option(&opts_map, "Output").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None }).unwrap_or("Value".into());
            if st.method.eq_ignore_ascii_case("Linear") {
                let pre = match &st.preproc { Some(p)=>p.clone(), None=> return Value::Symbol("Null".into()) };
                let w = st.lin_w.clone().unwrap_or(vec![]); let b = st.lin_b.unwrap_or(0.0);
                let xv = featurize(&pre, &x);
                let yhat = dot(&w, &xv) + b;
                if output.eq_ignore_ascii_case("PredictionInterval") {
                    let cl = get_option(&opts_map, "ConfidenceLevel").and_then(|v| as_f64(v)).unwrap_or(0.9);
                    let z = z_for_conf(cl);
                    let sigma = st.resid_std.unwrap_or(0.0);
                    return Value::assoc(vec![
                        ("mean", Value::Real(yhat)),
                        ("lower", Value::Real(yhat - z*sigma)),
                        ("upper", Value::Real(yhat + z*sigma)),
                    ]);
                }
                return Value::Real(yhat);
            }
            // Baseline mean
            Value::Real(st.mean.unwrap_or(0.0))
        }
        "clustering" => Value::Integer(1),
        "transform" => x,
        _ => Value::Expr { head: Box::new(Value::Symbol("MLApply".into())), args },
    }
}

fn ml_property(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // MLProperty[model, prop, extra?]
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("MLProperty".into())), args } }
    let arg0 = ev.eval(args[0].clone());
    let prop = ev.eval(args[1].clone());
    // Accept either a model handle or a function object wrapping MLApply[model, ...]
    let model_v = if get_model_id(&arg0).is_some() { arg0.clone() } else {
        match &arg0 {
            Value::PureFunction { body, .. } => {
                match &**body {
                    Value::Expr { head, args } if matches!(&**head, Value::Symbol(s) if s=="MLApply") && !args.is_empty() => args[0].clone(),
                    _ => arg0.clone(),
                }
            }
            _ => arg0.clone(),
        }
    };
    let id = match get_model_id(&model_v) { Some(id)=>id, None=> return Value::Expr { head: Box::new(Value::Symbol("MLProperty".into())), args } };
    let reg = ml_reg().lock().unwrap();
    let st = match reg.get(&id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("MLProperty".into())), args } };
    match prop {
        Value::String(s)|Value::Symbol(s) => {
            let key = s.to_lowercase();
            match key.as_str() {
                "properties" => Value::List(vec!["Method","Task","TrainingSize","Classes"].into_iter().map(|s| Value::String(s.into())).collect()),
                "method" => Value::String(st.method),
                "task" => Value::String(st.task),
                "trainingsize" => Value::Integer(st.train_size as i64),
                "classes" => {
                    if let Some(cs) = st.classes { Value::List(cs) } else { Value::List(vec![]) }
                }
                _ => Value::Symbol("Null".into()),
            }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("MLProperty".into())), args },
    }
}

// ---------- Measurements ----------

fn classify_measurements(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // ClassifyMeasurements[clf, test]
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("ClassifyMeasurements".into())), args } }
    let clf = ev.eval(args[0].clone());
    let test = ev.eval(args[1].clone());
    // Build predictions and compare
    let mut y_true: Vec<Value> = Vec::new();
    let mut y_pred: Vec<Value> = Vec::new();
    match test {
        Value::List(items) => {
            for it in items {
                if let Value::Expr { head, args } = it { if matches!(*head, Value::Symbol(ref s) if s=="Rule") && args.len()==2 {
                    let xt = args[0].clone();
                    let yt = args[1].clone();
                    let yhat = ev.eval(Value::expr(clf.clone(), vec![xt]));
                    y_true.push(yt); y_pred.push(yhat);
                } }
            }
        }
        Value::Assoc(_) => {
            // Dataset with Target? Try Target->col in second arg of this call is not supported; expect pairs form for v1
        }
        _ => {}
    }
    // Accuracy
    let mut correct = 0i64;
    let mut cm: HashMap<String, HashMap<String, i64>> = HashMap::new();
    for (t, p) in y_true.iter().zip(y_pred.iter()) {
        if t==p { correct += 1; }
        let tk = pretty_key(t); let pk = pretty_key(p);
        let row = cm.entry(tk).or_insert_with(HashMap::new);
        *row.entry(pk).or_insert(0) += 1;
    }
    let acc = if y_true.is_empty() { 0.0 } else { (correct as f64)/(y_true.len() as f64) };
    let cm_rows: Vec<Value> = cm.into_iter().map(|(tk, row)| {
        let mut inner: HashMap<String, Value> = HashMap::new();
        for (pk, c) in row { inner.insert(pk, Value::Integer(c)); }
        Value::assoc(vec![("true", Value::String(tk)), ("counts", Value::Assoc(inner))])
    }).collect();
    Value::Assoc(HashMap::from([
        ("Accuracy".into(), Value::Real(acc)),
        ("ConfusionMatrix".into(), Value::List(cm_rows)),
    ]))
}

fn predict_measurements(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // PredictMeasurements[pred, test]
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("PredictMeasurements".into())), args } }
    let pred_fn = ev.eval(args[0].clone());
    let test = ev.eval(args[1].clone());
    let mut y_true: Vec<f64> = Vec::new();
    let mut y_pred: Vec<f64> = Vec::new();
    match test {
        Value::List(items) => {
            for it in items { if let Value::Expr { head, args } = it { if matches!(*head, Value::Symbol(ref s) if s=="Rule") && args.len()==2 {
                let xt = args[0].clone(); let yt = args[1].clone();
                let yhat = ev.eval(Value::expr(pred_fn.clone(), vec![xt]));
                if let Some(ytf) = match yt { Value::Integer(n)=>Some(n as f64), Value::Real(x)=>Some(x), _=>None } { y_true.push(ytf); }
                if let Some(yhf) = match yhat { Value::Integer(n)=>Some(n as f64), Value::Real(x)=>Some(x), _=>None } { y_pred.push(yhf); }
            } } }
        }
        _ => {}
    }
    let n = y_true.len().min(y_pred.len());
    let mut se = 0.0f64;
    for i in 0..n { let d = y_true[i]-y_pred[i]; se += d*d; }
    let rmse = if n==0 { 0.0 } else { (se/(n as f64)).sqrt() };
    Value::Assoc(HashMap::from([("RMSE".into(), Value::Real(rmse))]))
}

// ---------- Cross Validation & Tuning ----------

fn kfold_indices(n: usize, k: usize, seed: u64) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut idx: Vec<usize> = (0..n).collect();
    // simple shuffle using LCG
    let mut s = seed.max(1);
    for i in 0..n { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); let j = (s as usize) % n; idx.swap(i, j); }
    let fold_size = (n + k - 1) / k;
    let mut folds = Vec::new();
    for f in 0..k {
        let start = f*fold_size; let end = ((f+1)*fold_size).min(n);
        let valid: Vec<usize> = idx[start..end].to_vec();
        let mut train = Vec::new(); train.extend_from_slice(&idx[..start]); train.extend_from_slice(&idx[end..]);
        folds.push((train, valid));
    }
    folds
}

fn ml_cross_validate(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // MLCrossValidate[trainerOrSpec, train, opts]
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("MLCrossValidate".into())), args } }
    let (_pos, opts) = parse_opts(ev, &args);
    let k = get_option(&opts, "K").and_then(|v| as_f64(v)).unwrap_or(5.0) as usize;
    let epochs_opt = get_option(&opts, "Epochs").and_then(|v| as_f64(v)).unwrap_or(30.0) as usize;
    let metric = get_option(&opts, "Metric").and_then(|v| if let Value::String(s)|Value::Symbol(s) = v { Some(s.clone()) } else { None }).unwrap_or("auto".into());
    // Expect training pairs list in args[1]
    let train = ev.eval(args[1].clone());
    let mut pairs_cls: Vec<(Value, Value)> = Vec::new();
    let mut pairs_reg: Vec<(Value, f64)> = Vec::new();
    match train {
        Value::List(items) => {
            for it in items { if let Value::Expr { head, args } = it { if matches!(*head, Value::Symbol(ref s) if s=="Rule") && args.len()==2 {
                let yv = ev.eval(args[1].clone());
                if let Some(yf) = as_f64(&yv) { pairs_reg.push((args[0].clone(), yf)); } else { pairs_cls.push((args[0].clone(), yv)); }
            } } }
        }
        _ => { return Value::Expr { head: Box::new(Value::Symbol("MLCrossValidate".into())), args } }
    }
    let n = pairs_cls.len().max(pairs_reg.len());
    if n==0 { return Value::Expr { head: Box::new(Value::Symbol("MLCrossValidate".into())), args } }
    let folds = kfold_indices(n, k.max(2), 1337);
    let mut scores: Vec<f64> = Vec::new();
    for (train_idx, valid_idx) in folds {
        if !pairs_reg.is_empty() {
            let tr: Vec<Value> = train_idx.iter().map(|&i| Value::expr(Value::symbol("Rule"), vec![pairs_reg[i].0.clone(), Value::Real(pairs_reg[i].1)])).collect();
            let model = ev.eval(Value::expr(Value::symbol("Predict"), vec![Value::List(tr.clone()), Value::assoc(vec![("Method", Value::String("Linear".into())), ("Epochs", Value::Integer(epochs_opt as i64))]) ]));
            let valid: Vec<Value> = valid_idx.iter().map(|&i| Value::expr(Value::symbol("Rule"), vec![pairs_reg[i].0.clone(), Value::Real(pairs_reg[i].1)])).collect();
            let meas = predict_measurements(ev, vec![model, Value::List(valid)]);
            if let Value::Assoc(m) = meas { if let Some(Value::Real(rmse)) = m.get("RMSE") { scores.push(*rmse); } }
        } else {
            let tr: Vec<Value> = train_idx.iter().map(|&i| Value::expr(Value::symbol("Rule"), vec![pairs_cls[i].0.clone(), pairs_cls[i].1.clone()])).collect();
            let model = ev.eval(Value::expr(Value::symbol("Classify"), vec![Value::List(tr.clone()), Value::assoc(vec![("Method", Value::String("Logistic".into())), ("Epochs", Value::Integer(epochs_opt as i64))]) ]));
            let valid: Vec<Value> = valid_idx.iter().map(|&i| Value::expr(Value::symbol("Rule"), vec![pairs_cls[i].0.clone(), pairs_cls[i].1.clone()])).collect();
            let meas = classify_measurements(ev, vec![model, Value::List(valid)]);
            if let Value::Assoc(m) = meas { if let Some(Value::Real(acc)) = m.get("Accuracy") { scores.push(*acc); } }
        }
    }
    let avg = if scores.is_empty() { 0.0 } else { scores.iter().sum::<f64>()/(scores.len() as f64) };
    Value::assoc(vec![("score", Value::Real(avg)), ("metric", Value::String(metric))])
}

fn ml_tune(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // MLTune[train, opts]: supports L2 grid for Linear/Logistic
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("MLTune".into())), args } }
    let (pos, opts) = parse_opts(ev, &args);
    if pos.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("MLTune".into())), args } }
    let train = ev.eval(pos[0].clone());
    let search = get_option(&opts, "SearchSpace").cloned().unwrap_or(Value::Assoc(HashMap::new()));
    let l2_vals: Vec<f64> = match search { Value::Assoc(m) => m.get("L2").and_then(|v| match v { Value::List(xs)=>Some(xs.iter().filter_map(|x| as_f64(x)).collect()), _=>None }).unwrap_or(vec![0.0, 1e-2, 1e-1]), _=> vec![0.0] };
    let k = get_option(&opts, "K").and_then(|v| as_f64(v)).unwrap_or(3.0) as usize;
    let epochs_opt = get_option(&opts, "Epochs").and_then(|v| as_f64(v)).unwrap_or(30.0) as usize;
    let mut best_score = std::f64::INFINITY; let mut best_model = Value::Symbol("Null".into()); let mut report: Vec<Value> = Vec::new();
    match train {
        Value::List(items) => {
            // Decide task by target type in first item
            let mut is_reg = false; if let Some(Value::Expr { head, args }) = items.first() { if matches!(&**head, Value::Symbol(s) if s=="Rule") && args.len()==2 { is_reg = as_f64(&ev.eval(args[1].clone())).is_some(); } }
            for l2 in l2_vals {
                if is_reg {
                    let model = ev.eval(Value::expr(Value::symbol("Predict"), vec![Value::List(items.clone()), Value::assoc(vec![("Method", Value::String("Linear".into())), ("L2", Value::Real(l2)), ("Epochs", Value::Integer(epochs_opt as i64))]) ]));
                    let cv = ml_cross_validate(ev, vec![Value::Symbol("Predict".into()), Value::List(items.clone()), Value::assoc(vec![("K", Value::Integer(k as i64)), ("Epochs", Value::Integer(epochs_opt as i64))])]);
                    let mut score = std::f64::INFINITY; if let Value::Assoc(m) = cv { if let Some(Value::Real(s)) = m.get("score") { score = *s; } }
                    report.push(Value::assoc(vec![("L2", Value::Real(l2)), ("score", Value::Real(score))]));
                    if score < best_score { best_score = score; best_model = model; }
                } else {
                    let model = ev.eval(Value::expr(Value::symbol("Classify"), vec![Value::List(items.clone()), Value::assoc(vec![("Method", Value::String("Logistic".into())), ("L2", Value::Real(l2)), ("Epochs", Value::Integer(epochs_opt as i64))]) ]));
                    let cv = ml_cross_validate(ev, vec![Value::Symbol("Classify".into()), Value::List(items.clone()), Value::assoc(vec![("K", Value::Integer(k as i64)), ("Epochs", Value::Integer(epochs_opt as i64))])]);
                    let mut score = 0.0; if let Value::Assoc(m) = cv { if let Some(Value::Real(s)) = m.get("score") { score = *s; } }
                    report.push(Value::assoc(vec![("L2", Value::Real(l2)), ("score", Value::Real(score))]));
                    // Higher is better for accuracy; invert to compare with best_score initialized as +inf
                    let cmp = -score; if cmp < best_score { best_score = cmp; best_model = model; }
                }
            }
        }
        _ => {}
    }
    Value::assoc(vec![("model", best_model), ("report", Value::List(report))])
}
