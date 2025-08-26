use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
use std::collections::HashMap;
use std::sync::{OnceLock, Mutex};
#[cfg(feature = "db_sqlite")] use rusqlite::{Connection, params};

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

pub fn register_vector_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    super::register_if(ev, pred, "VSNew", vs_new as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "VSUpsert", vs_upsert as NativeFn, Attributes::LISTABLE);
    super::register_if(ev, pred, "VSQuery", vs_query as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "VSDelete", vs_delete as NativeFn, Attributes::LISTABLE);
    super::register_if(ev, pred, "VSCount", vs_count as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "VSReset", vs_reset as NativeFn, Attributes::empty());
}

fn assoc_str(m: &HashMap<String, Value>, k: &str) -> Option<String> {
    m.get(k).and_then(|v| match v { Value::String(s) | Value::Symbol(s) => Some(s.clone()), _ => None })
}

fn store_name_arg(arg: Option<&Value>) -> String {
    match arg {
        Some(Value::Assoc(m)) => assoc_str(m, "Name").unwrap_or_else(|| "default".into()),
        Some(Value::String(s)) | Some(Value::Symbol(s)) => s.clone(),
        _ => "default".into(),
    }
}

fn as_vecf(v: &Value) -> Option<Vec<f64>> {
    match v {
        Value::List(items) => Some(
            items
                .iter()
                .filter_map(|x| match x { Value::Real(r) => Some(*r), Value::Integer(i) => Some(*i as f64), _ => None })
                .collect(),
        ),
        _ => None,
    }
}

fn hash_embed(s: &str, dims: usize) -> Vec<f64> {
    let mut v = vec![0.0; dims];
    for (i, ch) in s.chars().enumerate() {
        let idx = (ch as usize + i) % dims;
        v[idx] += 1.0;
    }
    v
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

fn vs_new(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // VSNew[<|Name->"default", Dims->n|>] or VSNew["sqlite://path.db"]
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("VSNew".into())), args };
    }
    let mut name = "default".to_string();
    let mut dims: usize = 3;
    match &args[0] {
        Value::Assoc(m) => {
            if let Some(s) = assoc_str(m, "Name") { name = s; }
            if let Some(Value::Integer(n)) = m.get("Dims") { dims = (*n).max(1) as usize; }
        }
        Value::String(s) | Value::Symbol(s) => name = s.clone(),
        _ => {}
    }

    if name.starts_with("sqlite://") {
        #[cfg(feature = "db_sqlite")]
        {
            let path = name.trim_start_matches("sqlite://");
            if let Ok(conn) = Connection::open(path) {
                let _ = conn.execute(
                    "CREATE TABLE IF NOT EXISTS vs_items(id TEXT PRIMARY KEY, dim INTEGER, vec BLOB, meta TEXT, text TEXT)",
                    [],
                );
                let _ = conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS vs_fts USING fts5(id, text)", []);
            }
        }
        return Value::Assoc(HashMap::from([
            (String::from("__type"), Value::String(String::from("VectorStore"))),
            (String::from("Name"), Value::String(name)),
        ]));
    }

    let mut guard = stores().lock().unwrap();
    guard.insert(name.clone(), Store { name: name.clone(), dims, items: Vec::new() });
    Value::Assoc(HashMap::from([
        (String::from("__type"), Value::String(String::from("VectorStore"))),
        (String::from("Name"), Value::String(name)),
    ]))
}

fn vs_count(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let name = store_name_arg(args.get(0));
    if name.starts_with("sqlite://") {
        #[cfg(feature = "db_sqlite")]
        {
            let path = name.trim_start_matches("sqlite://");
            if let Ok(conn) = Connection::open(path) {
                if let Ok(cnt) = conn.query_row("SELECT COUNT(*) FROM vs_items", [], |r| r.get::<_, i64>(0)) {
                    return Value::Integer(cnt);
                }
            }
        }
        return Value::Integer(0);
    }
    let guard = stores().lock().unwrap();
    if let Some(s) = guard.get(&name) { Value::Integer(s.items.len() as i64) } else { Value::Integer(0) }
}

fn vs_reset(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let name = store_name_arg(args.get(0));
    if name.starts_with("sqlite://") {
        #[cfg(feature = "db_sqlite")]
        {
            let path = name.trim_start_matches("sqlite://");
            if let Ok(conn) = Connection::open(path) {
                let _ = conn.execute("DELETE FROM vs_items", []);
                let _ = conn.execute("DELETE FROM vs_fts", []);
            }
        }
        return Value::Boolean(true);
    }
    let mut guard = stores().lock().unwrap();
    guard.remove(&name);
    Value::Boolean(true)
}

pub fn vs_upsert(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // VSUpsert[store, {<|id, vector, meta?|>, ...}]
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("VSUpsert".into())), args };
    }
    let name = store_name_arg(args.get(0));
    let items_v = args[1].clone();

    if name.starts_with("sqlite://") {
        #[cfg(feature = "db_sqlite")]
        {
            let path = name.trim_start_matches("sqlite://");
            if let Ok(conn) = Connection::open(path) {
                if let Value::List(rows) = items_v {
                    for r in rows {
                        if let Value::Assoc(m) = r {
                            if let Some(id) = assoc_str(&m, "id") {
                                if let Some(vecv) = m.get("vector").and_then(as_vecf) {
                                    let dim = vecv.len() as i64;
                                    let mut blob = Vec::with_capacity(vecv.len() * 8);
                                    for f in vecv { blob.extend_from_slice(&f.to_le_bytes()); }
                                    let meta_s = m
                                        .get("meta")
                                        .map(|x| lyra_core::pretty::format_value(x))
                                        .unwrap_or_else(|| "<||>".into());
                                    let text_s = if let Some(Value::Assoc(mm)) = m.get("meta") {
                                        if let Some(Value::String(t)) = mm.get("text") { t.clone() } else { String::new() }
                                    } else { String::new() };
                                    let _ = conn.execute(
                                        "REPLACE INTO vs_items(id, dim, vec, meta, text) VALUES (?1, ?2, ?3, ?4, ?5)",
                                        params![id, dim, blob, meta_s, text_s],
                                    );
                                    let _ = conn.execute(
                                        "REPLACE INTO vs_fts(id, text) VALUES (?1, ?2)",
                                        params![id, text_s],
                                    );
                                }
                            }
                        }
                    }
                }
                if let Value::Integer(n) = vs_count(_ev, vec![Value::String(name.clone())]) { return Value::Integer(n); }
            }
        }
        return Value::Integer(0);
    }

    let mut guard = stores().lock().unwrap();
    let st = guard
        .entry(name.clone())
        .or_insert(Store { name: name.clone(), dims: 3, items: Vec::new() });
    match items_v {
        Value::List(rows) => {
            for r in rows {
                if let Value::Assoc(m) = r {
                    if let Some(id) = assoc_str(&m, "id") {
                        if let Some(vecv) = m.get("vector").and_then(as_vecf) {
                            if let Some(pos) = st.items.iter().position(|it| it.id == id) { st.items.remove(pos); }
                            st.dims = vecv.len();
                            let meta = m.get("meta").cloned().unwrap_or(Value::Assoc(HashMap::new()));
                            st.items.push(Item { id, vec: vecv, meta });
                        }
                    }
                }
            }
            Value::Integer(st.items.len() as i64)
        }
        _ => Value::Integer(st.items.len() as i64),
    }
}

pub fn vs_query(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // VSQuery[store, vector|text, <|K->n, Filter->assoc, Hybrid->..., Alpha->...|>]
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("VSQuery".into())), args };
    }
    let name = store_name_arg(args.get(0));
    let query = args[1].clone();
    let mut k: usize = 5;
    let mut filter: Option<HashMap<String, Value>> = None;
    let mut hybrid: bool = false;
    let mut alpha: f64 = 0.5;
    if let Some(Value::Assoc(m)) = args.get(2) {
        if let Some(Value::Integer(n)) = m.get("K") { if *n > 0 { k = *n as usize; } }
        if let Some(Value::Assoc(f)) = m.get("Filter") { filter = Some(f.clone()); }
        if let Some(Value::Boolean(b)) = m.get("Hybrid") { hybrid = *b; }
        if let Some(Value::String(s)) = m.get("Hybrid") { let ls = s.to_lowercase(); if ls=="true"||ls=="on"||ls=="1" { hybrid = true; } }
        if let Some(Value::Real(a)) = m.get("Alpha") { alpha = *a; }
    }

    if name.starts_with("sqlite://") {
        #[cfg(feature = "db_sqlite")]
        {
            let path = name.trim_start_matches("sqlite://");
            if let Ok(conn) = Connection::open(path) {
                let mut t_candidates: Vec<(String, f64)> = Vec::new();
                if let Value::String(qs) | Value::Symbol(qs) = &query {
                    if let Ok(mut st) = conn.prepare("SELECT id, bm25(vs_fts) as rank FROM vs_fts WHERE vs_fts MATCH ?1 ORDER BY rank LIMIT 100") {
                        if let Ok(rows) = st.query_map([qs], |r| Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?))) {
                            for row in rows.flatten() { let norm = 1.0 / (1.0 + row.1.max(0.0)); t_candidates.push((row.0, norm)); }
                        }
                    }
                }
                let qv = match &query {
                    Value::List(_) => as_vecf(&query).unwrap_or_else(|| vec![]),
                    Value::String(s) | Value::Symbol(s) => {
                        let dim: usize = conn
                            .query_row("SELECT COALESCE(MAX(dim),3) FROM vs_items", [], |r| r.get::<_, i64>(0))
                            .unwrap_or(3) as usize;
                        hash_embed(s, dim)
                    }
                    _ => vec![],
                };
                let mut scored: Vec<(f64, String, Value)> = Vec::new();
                if let Ok(mut st) = conn.prepare("SELECT id, vec, meta FROM vs_items") {
                    if let Ok(rows) = st.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?, r.get::<_, String>(2)?))) {
                        for row in rows.flatten() {
                            let (id, blob, meta_s) = row;
                            let mut vecf: Vec<f64> = Vec::new();
                            for chunk in blob.chunks_exact(8) { let mut arr = [0u8; 8]; arr.copy_from_slice(chunk); vecf.push(f64::from_le_bytes(arr)); }
                            let vscore = if qv.is_empty() { 0.0 } else { cosine(&qv, &vecf) };
                            let tscore = t_candidates.iter().find(|(cid, _)| cid == &id).map(|(_, s)| *s).unwrap_or(0.0);
                            let score = if hybrid { alpha * tscore + (1.0 - alpha) * vscore } else if !qv.is_empty() { vscore } else { tscore };
                            let meta_v = Value::String(meta_s);
                            scored.push((score, id, meta_v));
                        }
                    }
                }
                scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                let out: Vec<Value> = scored
                    .into_iter()
                    .take(k)
                    .map(|(score, id, meta)| Value::Assoc(HashMap::from([
                        ("id".into(), Value::String(id)),
                        ("score".into(), Value::Real(score)),
                        ("meta".into(), meta),
                    ])))
                    .collect();
                return Value::List(out);
            }
        }
        return Value::List(vec![]);
    }

    let guard = stores().lock().unwrap();
    if let Some(st) = guard.get(&name) {
        let qv = vectorize_from_value(&query, st.dims);
        let mut scored: Vec<(f64, &Item)> = st
            .items
            .iter()
            .filter(|it| match &filter {
                None => true,
                Some(f) => f.iter().all(|(k, v)| match it.meta {
                    Value::Assoc(ref mm) => mm.get(k) == Some(v),
                    _ => false,
                }),
            })
            .map(|it| (cosine(&qv, &it.vec), it))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let out: Vec<Value> = scored
            .into_iter()
            .take(k)
            .map(|(score, it)| Value::Assoc(HashMap::from([
                ("id".into(), Value::String(it.id.clone())),
                ("score".into(), Value::Real(score)),
                ("meta".into(), it.meta.clone()),
            ])))
            .collect();
        Value::List(out)
    } else {
        Value::List(vec![])
    }
}

fn vectorize_from_value(v: &Value, dims: usize) -> Vec<f64> {
    match v {
        Value::String(s) => hash_embed(s, dims),
        Value::List(items) => items
            .iter()
            .filter_map(|x| match x { Value::Real(r) => Some(*r), Value::Integer(i) => Some(*i as f64), _ => None })
            .collect(),
        _ => vec![0.0; dims],
    }
}

fn vs_delete(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // VSDelete[store, {ids...}]
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("VSDelete".into())), args };
    }
    let name = store_name_arg(args.get(0));
    if name.starts_with("sqlite://") {
        #[cfg(feature = "db_sqlite")]
        {
            let path = name.trim_start_matches("sqlite://");
            if let Ok(conn) = Connection::open(path) {
                if let Value::List(ids) = &args[1] {
                    for idv in ids {
                        if let Value::String(id) | Value::Symbol(id) = idv {
                            let _ = conn.execute("DELETE FROM vs_items WHERE id=?1", params![id]);
                            let _ = conn.execute("DELETE FROM vs_fts WHERE id=?1", params![id]);
                        }
                    }
                }
            }
        }
        return Value::Boolean(true);
    }
    let mut guard = stores().lock().unwrap();
    if let Some(st) = guard.get_mut(&name) {
        if let Value::List(ids) = &args[1] {
            for idv in ids {
                if let Value::String(id) | Value::Symbol(id) = idv {
                    if let Some(pos) = st.items.iter().position(|it| &it.id == id) { st.items.remove(pos); }
                }
            }
        }
    }
    Value::Boolean(true)
}
