use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_runtime::attrs::Attributes;
use std::collections::HashMap;
use std::sync::{OnceLock, Mutex};

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

#[derive(Clone)]
struct Item { id: String, vec: Vec<f64>, meta: Value }

#[derive(Clone)]
struct Store { name: String, dims: usize, items: Vec<Item> }

static STORES: OnceLock<Mutex<HashMap<String, Store>>> = OnceLock::new();
fn stores() -> &'static Mutex<HashMap<String, Store>> { STORES.get_or_init(|| Mutex::new(HashMap::new())) }

pub fn register_vector(ev: &mut Evaluator) {
    ev.register("VSNew", vs_new as NativeFn, Attributes::empty());
    ev.register("VSUpsert", vs_upsert as NativeFn, Attributes::LISTABLE);
    ev.register("VSQuery", vs_query as NativeFn, Attributes::empty());
    ev.register("VSDelete", vs_delete as NativeFn, Attributes::LISTABLE);
    ev.register("VSCount", vs_count as NativeFn, Attributes::empty());
    ev.register("VSReset", vs_reset as NativeFn, Attributes::empty());
}

pub fn register_vector_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str)->bool) {
    super::register_if(ev, pred, "VSNew", vs_new as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "VSUpsert", vs_upsert as NativeFn, Attributes::LISTABLE);
    super::register_if(ev, pred, "VSQuery", vs_query as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "VSDelete", vs_delete as NativeFn, Attributes::LISTABLE);
    super::register_if(ev, pred, "VSCount", vs_count as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "VSReset", vs_reset as NativeFn, Attributes::empty());
}

fn assoc_str(m: &HashMap<String, Value>, k: &str) -> Option<String> { m.get(k).and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None }) }

fn vs_new(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // VSNew[<|Name->"default", Dims->n|>]
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("VSNew".into())), args } }
    let mut name = "default".to_string();
    let mut dims: usize = 0;
    if let Value::Assoc(m) = &args[0] {
        if let Some(s) = assoc_str(m, "Name") { name = s; }
        if let Some(Value::Integer(n)) = m.get("Dims") { dims = *n as usize; }
    }
    if dims == 0 { dims = 3; }
    let mut guard = stores().lock().unwrap();
    guard.insert(name.clone(), Store { name: name.clone(), dims, items: Vec::new() });
    Value::Assoc(HashMap::from([(String::from("__type"), Value::String(String::from("VectorStore"))), (String::from("Name"), Value::String(name))]))
}

fn vs_count(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let name = store_name_arg(args.get(0));
    let guard = stores().lock().unwrap();
    if let Some(s) = guard.get(&name) { Value::Integer(s.items.len() as i64) } else { Value::Integer(0) }
}

fn vs_reset(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let name = store_name_arg(args.get(0));
    let mut guard = stores().lock().unwrap();
    guard.remove(&name);
    Value::Boolean(true)
}

fn store_name_arg(arg: Option<&Value>) -> String {
    match arg { Some(Value::Assoc(m)) => assoc_str(m, "Name").unwrap_or_else(|| "default".into()), Some(Value::String(s))|Some(Value::Symbol(s)) => s.clone(), _ => "default".into() }
}

fn as_vecf(v: &Value) -> Option<Vec<f64>> { match v { Value::List(items) => Some(items.iter().filter_map(|x| match x { Value::Real(r)=>Some(*r), Value::Integer(i)=>Some(*i as f64), _=>None }).collect()), _=>None } }

pub fn vs_upsert(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // VSUpsert[store, {<|id, vector, meta?|>, ...}]
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("VSUpsert".into())), args } }
    let name = store_name_arg(args.get(0));
    let items_v = args[1].clone();
    let mut guard = stores().lock().unwrap();
    let st = guard.entry(name.clone()).or_insert(Store { name: name.clone(), dims: 3, items: Vec::new() });
    match items_v {
        Value::List(rows) => {
            for r in rows {
                if let Value::Assoc(m) = r {
                    if let Some(id) = assoc_str(&m, "id") {
                        if let Some(vecv) = m.get("vector").and_then(as_vecf) {
                            // replace if exists
                            if let Some(pos) = st.items.iter().position(|it| it.id==id) { st.items.remove(pos); }
                            st.dims = vecv.len();
                            let meta = m.get("meta").cloned().unwrap_or(Value::Assoc(HashMap::new()));
                            st.items.push(Item { id, vec: vecv, meta });
                        }
                    }
                }
            }
            Value::Integer(st.items.len() as i64)
        }
        _ => Value::Integer(st.items.len() as i64)
    }
}

fn cosine(a: &[f64], b: &[f64]) -> f64 { let dot: f64 = a.iter().zip(b.iter()).map(|(x,y)| x*y).sum(); let na: f64 = a.iter().map(|x| x*x).sum::<f64>().sqrt(); let nb: f64 = b.iter().map(|x| x*x).sum::<f64>().sqrt(); if na==0.0 || nb==0.0 { 0.0 } else { dot/(na*nb) } }

fn vectorize_from_value(v: &Value, dims: usize) -> Vec<f64> {
    match v {
        Value::String(s) => hash_embed(s, dims),
        Value::List(items) => items.iter().filter_map(|x| match x { Value::Real(r)=>Some(*r), Value::Integer(i)=>Some(*i as f64), _=>None }).collect(),
        _ => vec![0.0; dims]
    }
}

fn hash_embed(s: &str, dims: usize) -> Vec<f64> {
    // Simple bag-of-chars hash to fixed dims
    let mut v = vec![0.0; dims];
    for (i, ch) in s.chars().enumerate() { let idx = (ch as usize + i) % dims; v[idx] += 1.0; }
    v
}

pub fn vs_query(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // VSQuery[store, vector|text, <|K->n, Filter->assoc|>]
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("VSQuery".into())), args } }
    let name = store_name_arg(args.get(0));
    let query = args[1].clone();
    let mut k: usize = 5;
    let mut filter: Option<HashMap<String, Value>> = None;
    if let Some(Value::Assoc(m)) = args.get(2) {
        if let Some(Value::Integer(n)) = m.get("K") { if *n>0 { k=*n as usize; } }
        if let Some(Value::Assoc(f)) = m.get("Filter") { filter = Some(f.clone()); }
    }
    let guard = stores().lock().unwrap();
    if let Some(st) = guard.get(&name) {
        let qv = vectorize_from_value(&query, st.dims);
        let mut scored: Vec<(f64, &Item)> = st.items.iter().filter(|it| match &filter { None=>true, Some(f)=> f.iter().all(|(k,v)| match it.meta { Value::Assoc(ref mm) => mm.get(k)==Some(v), _=>false }) }).map(|it| (cosine(&qv, &it.vec), it)).collect();
        scored.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let out: Vec<Value> = scored.into_iter().take(k).map(|(score, it)| Value::Assoc(HashMap::from([
            ("id".into(), Value::String(it.id.clone())),
            ("score".into(), Value::Real(score)),
            ("meta".into(), it.meta.clone()),
        ]))).collect();
        Value::List(out)
    } else { Value::List(vec![]) }
}

fn vs_delete(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // VSDelete[store, {ids...}]
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("VSDelete".into())), args } }
    let name = store_name_arg(args.get(0));
    let mut guard = stores().lock().unwrap();
    if let Some(st) = guard.get_mut(&name) {
        if let Value::List(ids) = &args[1] {
            for idv in ids {
                if let Value::String(id) | Value::Symbol(id) = idv { if let Some(pos) = st.items.iter().position(|it| &it.id==id) { st.items.remove(pos); } }
            }
        }
    }
    Value::Boolean(true)
}
