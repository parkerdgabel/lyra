use crate::register_if;
#[cfg(feature = "tools")]
use crate::tool_spec;
#[cfg(feature = "tools")]
use crate::tools::add_specs;
use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

// Simple in-memory graph engine (dynamic adjacency lists)
#[derive(Clone)]
struct Node {
    id: String,
    label: Option<String>,
    weight: Option<f64>,
    attrs: HashMap<String, Value>,
}

#[derive(Clone)]
struct Edge {
    id: i64,
    src: String,
    dst: String,
    key: Option<String>,
    weight: Option<f64>,
    attrs: HashMap<String, Value>,
}

#[derive(Clone)]
struct GraphState {
    directed: bool,
    multigraph: bool,
    allow_self_loops: bool,
    nodes: HashMap<String, Node>,
    out_adj: HashMap<String, Vec<i64>>, // node id -> edge ids (out)
    in_adj: HashMap<String, Vec<i64>>,  // node id -> edge ids (in)
    edges: HashMap<i64, Edge>,          // eid -> edge
    edge_index: HashMap<(String, String, Option<String>), HashSet<i64>>, // (src,dst,key) -> eids
    next_edge_id: i64,
}

static G_REG: OnceLock<Mutex<HashMap<i64, GraphState>>> = OnceLock::new();
static NEXT_G_ID: OnceLock<std::sync::atomic::AtomicI64> = OnceLock::new();
fn g_reg() -> &'static Mutex<HashMap<i64, GraphState>> {
    G_REG.get_or_init(|| Mutex::new(HashMap::new()))
}
fn next_g_id() -> i64 {
    let a = NEXT_G_ID.get_or_init(|| std::sync::atomic::AtomicI64::new(1));
    a.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

fn graph_handle(id: i64) -> Value {
    Value::Assoc(HashMap::from([
        ("__type".to_string(), Value::String("Graph".into())),
        ("id".to_string(), Value::Integer(id)),
    ]))
}

fn get_graph(v: &Value) -> Option<i64> {
    if let Value::Assoc(m) = v {
        if matches!(m.get("__type"), Some(Value::String(s)) if s=="Graph") {
            if let Some(Value::Integer(id)) = m.get("id") {
                return Some(*id);
            }
        }
    }
    None
}

// ---------- Helpers ----------
fn key_of(v: &Value) -> String {
    match v {
        Value::String(s) | Value::Symbol(s) => s.clone(),
        Value::Integer(n) => n.to_string(),
        Value::Real(f) => f.to_string(),
        _ => lyra_core::pretty::format_value(v),
    }
}

fn parse_node(ev: &mut Evaluator, v: Value) -> Option<Node> {
    match ev.eval(v) {
        Value::String(s) | Value::Symbol(s) => {
            Some(Node { id: s, label: None, weight: None, attrs: HashMap::new() })
        }
        Value::Integer(n) => {
            Some(Node { id: n.to_string(), label: None, weight: None, attrs: HashMap::new() })
        }
        Value::Assoc(m) => {
            let id =
                m.get("id").map(key_of).or_else(|| m.get("Id").map(key_of)).unwrap_or_default();
            if id.is_empty() {
                return None;
            }
            let label = m.get("label").or_else(|| m.get("Label")).and_then(|v| match v {
                Value::String(s) | Value::Symbol(s) => Some(s.clone()),
                _ => None,
            });
            let weight = m.get("weight").or_else(|| m.get("Weight")).and_then(|v| match v {
                Value::Integer(i) => Some(*i as f64),
                Value::Real(f) => Some(*f),
                _ => None,
            });
            let attrs = m
                .get("attrs")
                .or_else(|| m.get("Attrs"))
                .and_then(|v| match v {
                    Value::Assoc(mm) => Some(mm.clone()),
                    _ => None,
                })
                .unwrap_or_default();
            Some(Node { id, label, weight, attrs })
        }
        _ => None,
    }
}

fn parse_edge(ev: &mut Evaluator, v: Value) -> Option<(String, String, Edge)> {
    let vv = ev.eval(v);
    match vv {
        Value::Assoc(m) => {
            let src = m.get("src").or_else(|| m.get("Src")).map(key_of)?;
            let dst = m.get("dst").or_else(|| m.get("Dst")).map(key_of)?;
            let key = m.get("key").or_else(|| m.get("Key")).and_then(|v| match v {
                Value::String(s) | Value::Symbol(s) => Some(s.clone()),
                _ => None,
            });
            let weight = m.get("weight").or_else(|| m.get("Weight")).and_then(|v| match v {
                Value::Integer(i) => Some(*i as f64),
                Value::Real(f) => Some(*f),
                _ => None,
            });
            let attrs = m
                .get("attrs")
                .or_else(|| m.get("Attrs"))
                .and_then(|v| match v {
                    Value::Assoc(mm) => Some(mm.clone()),
                    _ => None,
                })
                .unwrap_or_default();
            Some((src.clone(), dst.clone(), Edge { id: 0, src, dst, key, weight, attrs }))
        }
        _ => None,
    }
}

// ---------- API ----------
fn graph_create(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Graph[opts: Assoc]
    let (directed, multigraph, allow_self_loops) = match args.get(0).cloned() {
        Some(v) => match ev.eval(v) {
            Value::Assoc(m) => {
                let d = m
                    .get("directed")
                    .or_else(|| m.get("Directed"))
                    .and_then(|v| match v {
                        Value::Boolean(b) => Some(*b),
                        _ => None,
                    })
                    .unwrap_or(true);
                let mg = m
                    .get("multigraph")
                    .or_else(|| m.get("Multi"))
                    .and_then(|v| match v {
                        Value::Boolean(b) => Some(*b),
                        _ => None,
                    })
                    .unwrap_or(false);
                let sl = m
                    .get("allow_self_loops")
                    .or_else(|| m.get("SelfLoops"))
                    .and_then(|v| match v {
                        Value::Boolean(b) => Some(*b),
                        _ => None,
                    })
                    .unwrap_or(true);
                (d, mg, sl)
            }
            _ => (true, false, true),
        },
        None => (true, false, true),
    };
    let id = next_g_id();
    let st = GraphState {
        directed,
        multigraph,
        allow_self_loops,
        nodes: HashMap::new(),
        out_adj: HashMap::new(),
        in_adj: HashMap::new(),
        edges: HashMap::new(),
        edge_index: HashMap::new(),
        next_edge_id: 1,
    };
    g_reg().lock().unwrap().insert(id, st);
    graph_handle(id)
}

fn drop_graph(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("DropGraph".into())), args };
    }
    match get_graph(&args[0]) {
        Some(id) => {
            g_reg().lock().unwrap().remove(&id);
            Value::Boolean(true)
        }
        None => Value::Expr { head: Box::new(Value::Symbol("DropGraph".into())), args },
    }
}

fn graph_info(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("GraphInfo".into())), args };
    }
    match get_graph(&args[0]) {
        Some(id) => {
            let reg = g_reg().lock().unwrap();
            if let Some(st) = reg.get(&id) {
                Value::Assoc(HashMap::from([
                    ("id".into(), Value::Integer(id)),
                    ("directed".into(), Value::Boolean(st.directed)),
                    ("multigraph".into(), Value::Boolean(st.multigraph)),
                    ("nodes".into(), Value::Integer(st.nodes.len() as i64)),
                    ("edges".into(), Value::Integer(st.edges.len() as i64)),
                ]))
            } else {
                Value::Assoc(HashMap::new())
            }
        }
        None => Value::Expr { head: Box::new(Value::Symbol("GraphInfo".into())), args },
    }
}

fn add_nodes(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("AddNodes".into())), args };
    }
    let gid = match get_graph(&args[0]) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("AddNodes".into())), args },
    };
    let mut added = 0i64;
    let mut reg = g_reg().lock().unwrap();
    let st = reg.get_mut(&gid).unwrap();
    match ev.eval(args[1].clone()) {
        Value::List(items) => {
            for it in items {
                if let Some(n) = parse_node(ev, it) {
                    if !st.nodes.contains_key(&n.id) {
                        st.out_adj.entry(n.id.clone()).or_default();
                        st.in_adj.entry(n.id.clone()).or_default();
                        st.nodes.insert(n.id.clone(), n);
                        added += 1;
                    }
                }
            }
        }
        Value::Assoc(_) => {
            if let Some(n) = parse_node(ev, args[1].clone()) {
                if !st.nodes.contains_key(&n.id) {
                    st.out_adj.entry(n.id.clone()).or_default();
                    st.in_adj.entry(n.id.clone()).or_default();
                    st.nodes.insert(n.id.clone(), n);
                    added += 1;
                }
            }
        }
        _ => {}
    }
    Value::Integer(added)
}

fn add_edges(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("AddEdges".into())), args };
    }
    let gid = match get_graph(&args[0]) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("AddEdges".into())), args },
    };
    let mut added = 0i64;
    let mut reg = g_reg().lock().unwrap();
    let st = reg.get_mut(&gid).unwrap();
    let mut add_one = |mut e: Edge| {
        if !st.allow_self_loops && e.src == e.dst {
            return;
        }
        if !st.multigraph {
            if let Some(ids) = st.edge_index.get(&(e.src.clone(), e.dst.clone(), e.key.clone())) {
                if !ids.is_empty() {
                    return;
                }
            }
        }
        // Ensure nodes exist
        if !st.nodes.contains_key(&e.src) {
            st.nodes.insert(
                e.src.clone(),
                Node { id: e.src.clone(), label: None, weight: None, attrs: HashMap::new() },
            );
            st.out_adj.entry(e.src.clone()).or_default();
            st.in_adj.entry(e.src.clone()).or_default();
        }
        if !st.nodes.contains_key(&e.dst) {
            st.nodes.insert(
                e.dst.clone(),
                Node { id: e.dst.clone(), label: None, weight: None, attrs: HashMap::new() },
            );
            st.out_adj.entry(e.dst.clone()).or_default();
            st.in_adj.entry(e.dst.clone()).or_default();
        }
        let eid = st.next_edge_id;
        st.next_edge_id += 1;
        e.id = eid;
        st.out_adj.entry(e.src.clone()).or_default().push(eid);
        st.in_adj.entry(e.dst.clone()).or_default().push(eid);
        if !st.directed {
            // For undirected, also mirror adjacency for neighbor queries; keep single edge record
            st.out_adj.entry(e.dst.clone()).or_default().push(eid);
            st.in_adj.entry(e.src.clone()).or_default().push(eid);
        }
        st.edge_index.entry((e.src.clone(), e.dst.clone(), e.key.clone())).or_default().insert(eid);
        st.edges.insert(eid, e);
        added += 1;
    };
    match ev.eval(args[1].clone()) {
        Value::List(items) => {
            for it in items {
                if let Some((_s, _d, e)) = parse_edge(ev, it) {
                    add_one(e);
                }
            }
        }
        Value::Assoc(_) => {
            if let Some((_s, _d, e)) = parse_edge(ev, args[1].clone()) {
                add_one(e);
            }
        }
        _ => {}
    }
    Value::Integer(added)
}

fn upsert_nodes(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("UpsertNodes".into())), args };
    }
    let gid = match get_graph(&args[0]) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("UpsertNodes".into())), args },
    };
    let mut upserted = 0i64;
    let mut reg = g_reg().lock().unwrap();
    let st = reg.get_mut(&gid).unwrap();
    let upsert_one = |st: &mut GraphState, n: Node| {
        if let Some(existing) = st.nodes.get_mut(&n.id) {
            if n.label.is_some() {
                existing.label = n.label;
            }
            if n.weight.is_some() {
                existing.weight = n.weight;
            }
            for (k, v) in n.attrs.into_iter() {
                existing.attrs.insert(k, v);
            }
        } else {
            st.out_adj.entry(n.id.clone()).or_default();
            st.in_adj.entry(n.id.clone()).or_default();
            st.nodes.insert(n.id.clone(), n);
        }
    };
    match ev.eval(args[1].clone()) {
        Value::List(items) => {
            for it in items {
                if let Some(n) = parse_node(ev, it) {
                    upsert_one(st, n);
                    upserted += 1;
                }
            }
        }
        Value::Assoc(_) => {
            if let Some(n) = parse_node(ev, args[1].clone()) {
                upsert_one(st, n);
                upserted += 1;
            }
        }
        other => {
            if let Some(n) = parse_node(ev, other) {
                upsert_one(st, n);
                upserted += 1;
            }
        }
    }
    Value::Integer(upserted)
}

fn upsert_edges(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("UpsertEdges".into())), args };
    }
    let gid = match get_graph(&args[0]) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("UpsertEdges".into())), args },
    };
    let mut upserted = 0i64;
    let mut reg = g_reg().lock().unwrap();
    let st = reg.get_mut(&gid).unwrap();
    let mut upsert_one = |e: Edge| {
        // If simple graph or non-multi upsert, update existing (src,dst,key) if present; else insert
        if let Some(ids) = st.edge_index.get(&(e.src.clone(), e.dst.clone(), e.key.clone())) {
            if let Some(eid) = ids.iter().next().cloned() {
                if let Some(ex) = st.edges.get_mut(&eid) {
                    ex.weight = e.weight.or(ex.weight);
                    for (k, v) in e.attrs.into_iter() {
                        ex.attrs.insert(k, v);
                    }
                    upserted += 1;
                    return;
                }
            }
        }
        // Insert new
        let mut e2 = e;
        if !st.nodes.contains_key(&e2.src) {
            st.nodes.insert(
                e2.src.clone(),
                Node { id: e2.src.clone(), label: None, weight: None, attrs: HashMap::new() },
            );
            st.out_adj.entry(e2.src.clone()).or_default();
            st.in_adj.entry(e2.src.clone()).or_default();
        }
        if !st.nodes.contains_key(&e2.dst) {
            st.nodes.insert(
                e2.dst.clone(),
                Node { id: e2.dst.clone(), label: None, weight: None, attrs: HashMap::new() },
            );
            st.out_adj.entry(e2.dst.clone()).or_default();
            st.in_adj.entry(e2.dst.clone()).or_default();
        }
        let eid = st.next_edge_id;
        st.next_edge_id += 1;
        e2.id = eid;
        st.out_adj.entry(e2.src.clone()).or_default().push(eid);
        st.in_adj.entry(e2.dst.clone()).or_default().push(eid);
        if !st.directed {
            st.out_adj.entry(e2.dst.clone()).or_default().push(eid);
            st.in_adj.entry(e2.src.clone()).or_default().push(eid);
        }
        st.edge_index
            .entry((e2.src.clone(), e2.dst.clone(), e2.key.clone()))
            .or_default()
            .insert(eid);
        st.edges.insert(eid, e2);
        upserted += 1;
    };
    match ev.eval(args[1].clone()) {
        Value::List(items) => {
            for it in items {
                if let Some((_s, _d, e)) = parse_edge(ev, it) {
                    upsert_one(e);
                }
            }
        }
        Value::Assoc(_) => {
            if let Some((_s, _d, e)) = parse_edge(ev, args[1].clone()) {
                upsert_one(e);
            }
        }
        _ => {}
    }
    Value::Integer(upserted)
}

fn remove_nodes(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("RemoveNodes".into())), args };
    }
    let gid = match get_graph(&args[0]) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("RemoveNodes".into())), args },
    };
    // Optional opts: <|Cascade->True|>
    let cascade = if args.len() >= 3 {
        match ev.eval(args[2].clone()) {
            Value::Assoc(m) => m
                .get("Cascade")
                .or_else(|| m.get("cascade"))
                .and_then(|v| match v {
                    Value::Boolean(b) => Some(*b),
                    _ => None,
                })
                .unwrap_or(true),
            _ => true,
        }
    } else {
        true
    };
    let mut removed = 0i64;
    let mut reg = g_reg().lock().unwrap();
    let st = reg.get_mut(&gid).unwrap();
    let mut to_remove: Vec<String> = Vec::new();
    match ev.eval(args[1].clone()) {
        Value::List(items) => {
            for it in items {
                let id = key_of(&ev.eval(it));
                if !id.is_empty() {
                    to_remove.push(id);
                }
            }
        }
        other => {
            let id = key_of(&ev.eval(other));
            if !id.is_empty() {
                to_remove.push(id);
            }
        }
    }
    // Collect incident edges if cascade
    let mut edges_to_remove: HashSet<i64> = HashSet::new();
    for nid in &to_remove {
        if cascade {
            if let Some(out) = st.out_adj.get(nid) {
                for eid in out {
                    edges_to_remove.insert(*eid);
                }
            }
            if let Some(inn) = st.in_adj.get(nid) {
                for eid in inn {
                    edges_to_remove.insert(*eid);
                }
            }
        }
    }
    // Remove edges first
    for eid in edges_to_remove {
        if let Some(e) = st.edges.remove(&eid) {
            if let Some(v) = st.out_adj.get_mut(&e.src) {
                v.retain(|x| *x != eid);
            }
            if let Some(v) = st.in_adj.get_mut(&e.dst) {
                v.retain(|x| *x != eid);
            }
            if !st.directed {
                if let Some(v) = st.out_adj.get_mut(&e.dst) {
                    v.retain(|x| *x != eid);
                }
                if let Some(v) = st.in_adj.get_mut(&e.src) {
                    v.retain(|x| *x != eid);
                }
            }
            if let Some(idxset) =
                st.edge_index.get_mut(&(e.src.clone(), e.dst.clone(), e.key.clone()))
            {
                idxset.remove(&eid);
            }
        }
    }
    // Remove nodes
    for nid in to_remove {
        if st.nodes.remove(&nid).is_some() {
            st.out_adj.remove(&nid);
            st.in_adj.remove(&nid);
            removed += 1;
        }
    }
    Value::Integer(removed)
}

fn remove_edges(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("RemoveEdges".into())), args };
    }
    let gid = match get_graph(&args[0]) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("RemoveEdges".into())), args },
    };
    let mut removed = 0i64;
    let mut reg = g_reg().lock().unwrap();
    let st = reg.get_mut(&gid).unwrap();
    let mut eids: Vec<i64> = Vec::new();
    let push_by_assoc = |st: &GraphState, m: &HashMap<String, Value>, out: &mut Vec<i64>| {
        let src = m.get("src").or_else(|| m.get("Src")).map(key_of);
        let dst = m.get("dst").or_else(|| m.get("Dst")).map(key_of);
        let key = m.get("key").or_else(|| m.get("Key")).and_then(|v| match v {
            Value::String(s) | Value::Symbol(s) => Some(s.clone()),
            _ => None,
        });
        if let (Some(s), Some(d)) = (src, dst) {
            if let Some(set) = st.edge_index.get(&(s, d, key)) {
                out.extend(set.iter().cloned());
            }
        }
    };
    match ev.eval(args[1].clone()) {
        Value::Integer(n) => eids.push(n),
        Value::List(items) => {
            for it in items {
                let vv = ev.eval(it);
                match vv {
                    Value::Integer(n) => eids.push(n),
                    Value::Assoc(m) => push_by_assoc(st, &m, &mut eids),
                    _ => {}
                }
            }
        }
        Value::Assoc(m) => push_by_assoc(st, &m, &mut eids),
        _ => {}
    }
    // Dedup
    let mut seen = HashSet::new();
    eids.retain(|eid| seen.insert(*eid));
    for eid in eids {
        if let Some(e) = st.edges.remove(&eid) {
            if let Some(v) = st.out_adj.get_mut(&e.src) {
                v.retain(|x| *x != eid);
            }
            if let Some(v) = st.in_adj.get_mut(&e.dst) {
                v.retain(|x| *x != eid);
            }
            if !st.directed {
                if let Some(v) = st.out_adj.get_mut(&e.dst) {
                    v.retain(|x| *x != eid);
                }
                if let Some(v) = st.in_adj.get_mut(&e.src) {
                    v.retain(|x| *x != eid);
                }
            }
            if let Some(idxset) =
                st.edge_index.get_mut(&(e.src.clone(), e.dst.clone(), e.key.clone()))
            {
                idxset.remove(&eid);
            }
            removed += 1;
        }
    }
    Value::Integer(removed)
}

fn list_nodes(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("ListNodes".into())), args };
    }
    let gid = match get_graph(&args[0]) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("ListNodes".into())), args },
    };
    let reg = g_reg().lock().unwrap();
    let st = reg.get(&gid).unwrap();
    let mut rows: Vec<Value> = Vec::with_capacity(st.nodes.len());
    for n in st.nodes.values() {
        rows.push(Value::Assoc(HashMap::from([
            ("id".into(), Value::String(n.id.clone())),
            (
                "label".into(),
                n.label.clone().map(Value::String).unwrap_or(Value::Symbol("Null".into())),
            ),
            ("weight".into(), n.weight.map(Value::Real).unwrap_or(Value::Symbol("Null".into()))),
            ("attrs".into(), Value::Assoc(n.attrs.clone())),
        ])));
    }
    Value::Expr {
        head: Box::new(Value::Symbol("DatasetFromRows".into())),
        args: vec![Value::List(rows)],
    }
}

fn list_edges(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("ListEdges".into())), args };
    }
    let gid = match get_graph(&args[0]) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("ListEdges".into())), args },
    };
    let reg = g_reg().lock().unwrap();
    let st = reg.get(&gid).unwrap();
    let mut rows: Vec<Value> = Vec::with_capacity(st.edges.len());
    for e in st.edges.values() {
        rows.push(Value::Assoc(HashMap::from([
            ("id".into(), Value::Integer(e.id)),
            ("src".into(), Value::String(e.src.clone())),
            ("dst".into(), Value::String(e.dst.clone())),
            (
                "key".into(),
                e.key.clone().map(Value::String).unwrap_or(Value::Symbol("Null".into())),
            ),
            ("weight".into(), e.weight.map(Value::Real).unwrap_or(Value::Symbol("Null".into()))),
            ("attrs".into(), Value::Assoc(e.attrs.clone())),
        ])));
    }
    Value::Expr {
        head: Box::new(Value::Symbol("DatasetFromRows".into())),
        args: vec![Value::List(rows)],
    }
}

fn has_node(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("HasNode".into())), args };
    }
    let gid = match get_graph(&args[0]) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("HasNode".into())), args },
    };
    let id = key_of(&ev.eval(args[1].clone()));
    let reg = g_reg().lock().unwrap();
    let st = reg.get(&gid).unwrap();
    Value::Boolean(st.nodes.contains_key(&id))
}

fn has_edge(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 3 {
        return Value::Expr { head: Box::new(Value::Symbol("HasEdge".into())), args };
    }
    let gid = match get_graph(&args[0]) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("HasEdge".into())), args },
    };
    let src = key_of(&ev.eval(args[1].clone()));
    let dst = key_of(&ev.eval(args[2].clone()));
    let key = args.get(3).map(|v| ev.eval(v.clone())).and_then(|v| match v {
        Value::String(s) | Value::Symbol(s) => Some(s),
        _ => None,
    });
    let reg = g_reg().lock().unwrap();
    let st = reg.get(&gid).unwrap();
    Value::Boolean(st.edge_index.get(&(src, dst, key)).map(|ids| !ids.is_empty()).unwrap_or(false))
}

fn neighbors(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("Neighbors".into())), args };
    }
    let gid = match get_graph(&args[0]) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("Neighbors".into())), args },
    };
    let nid = key_of(&ev.eval(args[1].clone()));
    let dir = args
        .get(2)
        .map(|v| ev.eval(v.clone()))
        .and_then(|v| match v {
            Value::String(s) | Value::Symbol(s) => Some(s.to_lowercase()),
            _ => None,
        })
        .unwrap_or_else(|| "out".into());
    let reg = g_reg().lock().unwrap();
    let st = reg.get(&gid).unwrap();
    let mut out: Vec<Value> = Vec::new();
    let mut push_from_edges = |edges: &Vec<i64>, is_out: bool| {
        for eid in edges {
            if let Some(e) = st.edges.get(eid) {
                let other = if is_out { &e.dst } else { &e.src };
                out.push(Value::String(other.clone()));
            }
        }
    };
    if let Some(v) = st.out_adj.get(&nid) {
        if dir == "out" || dir == "all" {
            push_from_edges(v, true);
        }
    }
    if let Some(v) = st.in_adj.get(&nid) {
        if dir == "in" || dir == "all" {
            push_from_edges(v, false);
        }
    }
    Value::List(out)
}

fn incident_edges(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("IncidentEdges".into())), args };
    }
    let gid = match get_graph(&args[0]) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("IncidentEdges".into())), args },
    };
    let nid = key_of(&ev.eval(args[1].clone()));
    let dir = args
        .get(2)
        .map(|v| ev.eval(v.clone()))
        .and_then(|v| match v {
            Value::String(s) | Value::Symbol(s) => Some(s.to_lowercase()),
            _ => None,
        })
        .unwrap_or_else(|| "all".into());
    let reg = g_reg().lock().unwrap();
    let st = reg.get(&gid).unwrap();
    let mut rows: Vec<Value> = Vec::new();
    let mut collect = |eids: &Vec<i64>| {
        for eid in eids {
            if let Some(e) = st.edges.get(eid) {
                rows.push(Value::Assoc(HashMap::from([
                    ("id".into(), Value::Integer(e.id)),
                    ("src".into(), Value::String(e.src.clone())),
                    ("dst".into(), Value::String(e.dst.clone())),
                    (
                        "key".into(),
                        e.key.clone().map(Value::String).unwrap_or(Value::Symbol("Null".into())),
                    ),
                    (
                        "weight".into(),
                        e.weight.map(Value::Real).unwrap_or(Value::Symbol("Null".into())),
                    ),
                    ("attrs".into(), Value::Assoc(e.attrs.clone())),
                ])));
            }
        }
    };
    if dir == "out" || dir == "all" {
        if let Some(v) = st.out_adj.get(&nid) {
            collect(v);
        }
    }
    if dir == "in" || dir == "all" {
        if let Some(v) = st.in_adj.get(&nid) {
            collect(v);
        }
    }
    Value::Expr {
        head: Box::new(Value::Symbol("DatasetFromRows".into())),
        args: vec![Value::List(rows)],
    }
}

fn subgraph(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Subgraph[g, nodes]
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("Subgraph".into())), args };
    }
    let gid = match get_graph(&args[0]) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("Subgraph".into())), args },
    };
    let mut keep: HashSet<String> = HashSet::new();
    match ev.eval(args[1].clone()) {
        Value::List(items) => {
            for it in items {
                let id = key_of(&ev.eval(it));
                if !id.is_empty() {
                    keep.insert(id);
                }
            }
        }
        other => {
            let id = key_of(&ev.eval(other));
            if !id.is_empty() {
                keep.insert(id);
            }
        }
    }
    let reg = g_reg().lock().unwrap();
    let st = reg.get(&gid).unwrap().clone();
    drop(reg);
    // New graph
    let new_id = next_g_id();
    let mut ng = GraphState {
        directed: st.directed,
        multigraph: st.multigraph,
        allow_self_loops: st.allow_self_loops,
        nodes: HashMap::new(),
        out_adj: HashMap::new(),
        in_adj: HashMap::new(),
        edges: HashMap::new(),
        edge_index: HashMap::new(),
        next_edge_id: 1,
    };
    for (id, n) in st.nodes.iter() {
        if keep.contains(id) {
            ng.nodes.insert(id.clone(), n.clone());
            ng.out_adj.entry(id.clone()).or_default();
            ng.in_adj.entry(id.clone()).or_default();
        }
    }
    for e in st.edges.values() {
        if keep.contains(&e.src) && keep.contains(&e.dst) {
            let mut e2 = e.clone();
            e2.id = ng.next_edge_id;
            ng.next_edge_id += 1;
            ng.out_adj.entry(e2.src.clone()).or_default().push(e2.id);
            ng.in_adj.entry(e2.dst.clone()).or_default().push(e2.id);
            if !ng.directed {
                ng.out_adj.entry(e2.dst.clone()).or_default().push(e2.id);
                ng.in_adj.entry(e2.src.clone()).or_default().push(e2.id);
            }
            ng.edge_index
                .entry((e2.src.clone(), e2.dst.clone(), e2.key.clone()))
                .or_default()
                .insert(e2.id);
            ng.edges.insert(e2.id, e2);
        }
    }
    g_reg().lock().unwrap().insert(new_id, ng);
    graph_handle(new_id)
}

fn hash_with_seed(s: &str, seed: i64) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    seed.hash(&mut h);
    s.hash(&mut h);
    h.finish()
}

fn sample_nodes(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // SampleNodes[g, k, opts? <|Seed->i64|>]
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("SampleNodes".into())), args };
    }
    let gid = match get_graph(&args[0]) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("SampleNodes".into())), args },
    };
    let k = match ev.eval(args[1].clone()) {
        Value::Integer(n) if n > 0 => n as usize,
        other => {
            return Value::Expr {
                head: Box::new(Value::Symbol("SampleNodes".into())),
                args: vec![args[0].clone(), other],
            }
        }
    };
    let seed = if args.len() >= 3 {
        match ev.eval(args[2].clone()) {
            Value::Assoc(m) => m
                .get("Seed")
                .and_then(|v| if let Value::Integer(n) = v { Some(*n) } else { None })
                .unwrap_or(0),
            _ => 0,
        }
    } else {
        0
    };
    let reg = g_reg().lock().unwrap();
    let st = reg.get(&gid).unwrap();
    let mut ids: Vec<&String> = st.nodes.keys().collect();
    ids.sort_by_key(|id| hash_with_seed(id, seed));
    let take = k.min(ids.len());
    Value::List(ids.into_iter().take(take).map(|s| Value::String(s.clone())).collect())
}

fn sample_edges(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // SampleEdges[g, k, opts? <|Seed->i64|>]
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("SampleEdges".into())), args };
    }
    let gid = match get_graph(&args[0]) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("SampleEdges".into())), args },
    };
    let k = match ev.eval(args[1].clone()) {
        Value::Integer(n) if n > 0 => n as usize,
        other => {
            return Value::Expr {
                head: Box::new(Value::Symbol("SampleEdges".into())),
                args: vec![args[0].clone(), other],
            }
        }
    };
    let seed = if args.len() >= 3 {
        match ev.eval(args[2].clone()) {
            Value::Assoc(m) => m
                .get("Seed")
                .and_then(|v| if let Value::Integer(n) = v { Some(*n) } else { None })
                .unwrap_or(0),
            _ => 0,
        }
    } else {
        0
    };
    let reg = g_reg().lock().unwrap();
    let st = reg.get(&gid).unwrap();
    let mut items: Vec<&Edge> = st.edges.values().collect();
    items.sort_by_key(|e| {
        hash_with_seed(
            &format!("{}:{}:{}:{}", e.src, e.dst, e.key.clone().unwrap_or_default(), e.id),
            seed,
        )
    });
    let take = k.min(items.len());
    let mut rows: Vec<Value> = Vec::with_capacity(take);
    for e in items.into_iter().take(take) {
        rows.push(Value::Assoc(HashMap::from([
            ("id".into(), Value::Integer(e.id)),
            ("src".into(), Value::String(e.src.clone())),
            ("dst".into(), Value::String(e.dst.clone())),
            (
                "key".into(),
                e.key.clone().map(Value::String).unwrap_or(Value::Symbol("Null".into())),
            ),
            ("weight".into(), e.weight.map(Value::Real).unwrap_or(Value::Symbol("Null".into()))),
            ("attrs".into(), Value::Assoc(e.attrs.clone())),
        ])));
    }
    Value::Expr {
        head: Box::new(Value::Symbol("DatasetFromRows".into())),
        args: vec![Value::List(rows)],
    }
}

// ---------- Algorithms ----------
fn bfs(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("BFS".into())), args };
    }
    let gid = match get_graph(&args[0]) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("BFS".into())), args },
    };
    let start = key_of(&ev.eval(args[1].clone()));
    let mut dir = "out".to_string();
    let mut max_depth: Option<i64> = None;
    if args.len() >= 3 {
        if let Value::Assoc(m) = ev.eval(args[2].clone()) {
            if let Some(Value::String(s)) | Some(Value::Symbol(s)) =
                m.get("Direction").or_else(|| m.get("dir")).or_else(|| m.get("Dir"))
            {
                dir = s.to_lowercase();
            }
            if let Some(Value::Integer(d)) = m.get("MaxDepth").or_else(|| m.get("Depth")) {
                max_depth = Some(*d);
            }
        }
    }
    let reg = g_reg().lock().unwrap();
    let st = match reg.get(&gid) {
        Some(s) => s.clone(),
        None => return Value::List(Vec::new()),
    };
    drop(reg);
    use std::collections::{HashSet, VecDeque};
    let mut q = VecDeque::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut order: Vec<Value> = Vec::new();
    let mut parent: HashMap<String, String> = HashMap::new();
    q.push_back((start.clone(), 0i64));
    seen.insert(start.clone());
    while let Some((u, d)) = q.pop_front() {
        order.push(Value::String(u.clone()));
        if let Some(md) = max_depth {
            if d >= md {
                continue;
            }
        }
        let mut push_neighbors = |edges: &Vec<i64>, is_out: bool| {
            for eid in edges {
                if let Some(e) = st.edges.get(eid) {
                    let v = if is_out { &e.dst } else { &e.src };
                    if !seen.contains(v) {
                        seen.insert(v.clone());
                        parent.insert(v.clone(), u.clone());
                        q.push_back((v.clone(), d + 1));
                    }
                }
            }
        };
        if dir == "out" || dir == "all" {
            if let Some(es) = st.out_adj.get(&u) {
                push_neighbors(es, true);
            }
        }
        if dir == "in" || dir == "all" {
            if let Some(es) = st.in_adj.get(&u) {
                push_neighbors(es, false);
            }
        }
    }
    Value::Assoc(HashMap::from([
        ("order".into(), Value::List(order)),
        (
            "parent".into(),
            Value::Assoc(parent.into_iter().map(|(k, v)| (k, Value::String(v))).collect()),
        ),
    ]))
}

fn dfs(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("DFS".into())), args };
    }
    let gid = match get_graph(&args[0]) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("DFS".into())), args },
    };
    let start = key_of(&ev.eval(args[1].clone()));
    let mut dir = "out".to_string();
    if args.len() >= 3 {
        if let Value::Assoc(m) = ev.eval(args[2].clone()) {
            if let Some(Value::String(s)) | Some(Value::Symbol(s)) =
                m.get("Direction").or_else(|| m.get("dir")).or_else(|| m.get("Dir"))
            {
                dir = s.to_lowercase();
            }
        }
    }
    let reg = g_reg().lock().unwrap();
    let st = match reg.get(&gid) {
        Some(s) => s.clone(),
        None => return Value::List(Vec::new()),
    };
    drop(reg);
    use std::collections::HashSet;
    let mut seen: HashSet<String> = HashSet::new();
    let mut order: Vec<Value> = Vec::new();
    let mut stack: Vec<String> = vec![start.clone()];
    while let Some(u) = stack.pop() {
        if !seen.insert(u.clone()) {
            continue;
        }
        order.push(Value::String(u.clone()));
        let mut push_neighbors = |edges: &Vec<i64>, is_out: bool| {
            for eid in edges {
                if let Some(e) = st.edges.get(eid) {
                    let v = if is_out { &e.dst } else { &e.src };
                    if !seen.contains(v) {
                        stack.push(v.clone());
                    }
                }
            }
        };
        if dir == "out" || dir == "all" {
            if let Some(es) = st.out_adj.get(&u) {
                push_neighbors(es, true);
            }
        }
        if dir == "in" || dir == "all" {
            if let Some(es) = st.in_adj.get(&u) {
                push_neighbors(es, false);
            }
        }
    }
    Value::List(order)
}

fn shortest_paths(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("ShortestPaths".into())), args };
    }
    let gid = match get_graph(&args[0]) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("ShortestPaths".into())), args },
    };
    let src = key_of(&ev.eval(args[1].clone()));
    let mut weighted = false;
    let mut dir = "out".to_string();
    if args.len() >= 3 {
        if let Value::Assoc(m) = ev.eval(args[2].clone()) {
            if let Some(Value::Boolean(b)) = m.get("Weighted") {
                weighted = *b;
            }
            if let Some(Value::String(s)) | Some(Value::Symbol(s)) =
                m.get("Direction").or_else(|| m.get("dir")).or_else(|| m.get("Dir"))
            {
                dir = s.to_lowercase();
            }
        }
    }
    let reg = g_reg().lock().unwrap();
    let st = match reg.get(&gid) {
        Some(s) => s.clone(),
        None => return Value::Assoc(HashMap::new()),
    };
    drop(reg);
    if !weighted {
        // BFS layers for unweighted distance
        use std::collections::{HashSet, VecDeque};
        let mut dist: HashMap<String, f64> = HashMap::new();
        let mut parent: HashMap<String, String> = HashMap::new();
        let mut q = VecDeque::new();
        let mut seen: HashSet<String> = HashSet::new();
        seen.insert(src.clone());
        dist.insert(src.clone(), 0.0);
        q.push_back(src.clone());
        while let Some(u) = q.pop_front() {
            let du = *dist.get(&u).unwrap_or(&0.0);
            let mut push_neighbors = |edges: &Vec<i64>, is_out: bool| {
                for eid in edges {
                    if let Some(e) = st.edges.get(eid) {
                        let v = if is_out { &e.dst } else { &e.src };
                        if !seen.contains(v) {
                            seen.insert(v.clone());
                            dist.insert(v.clone(), du + 1.0);
                            parent.insert(v.clone(), u.clone());
                            q.push_back(v.clone());
                        }
                    }
                }
            };
            if dir == "out" || dir == "all" {
                if let Some(es) = st.out_adj.get(&u) {
                    push_neighbors(es, true);
                }
            }
            if dir == "in" || dir == "all" {
                if let Some(es) = st.in_adj.get(&u) {
                    push_neighbors(es, false);
                }
            }
        }
        return Value::Assoc(HashMap::from([
            (
                "dist".into(),
                Value::Assoc(dist.into_iter().map(|(k, v)| (k, Value::Real(v))).collect()),
            ),
            (
                "parent".into(),
                Value::Assoc(parent.into_iter().map(|(k, v)| (k, Value::String(v))).collect()),
            ),
        ]));
    }
    // Dijkstra
    use std::cmp::Ordering;
    use std::collections::{BinaryHeap, HashSet};
    #[derive(Clone)]
    struct State {
        cost: f64,
        node: String,
    }
    impl PartialEq for State {
        fn eq(&self, other: &Self) -> bool {
            self.cost == other.cost && self.node == other.node
        }
    }
    impl Eq for State {}
    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            other.cost.partial_cmp(&self.cost)
        }
    }
    impl Ord for State {
        fn cmp(&self, other: &Self) -> Ordering {
            self.partial_cmp(other).unwrap_or(Ordering::Equal)
        }
    }
    let mut dist: HashMap<String, f64> = HashMap::new();
    let mut parent: HashMap<String, String> = HashMap::new();
    let mut heap = BinaryHeap::new();
    dist.insert(src.clone(), 0.0);
    heap.push(State { cost: 0.0, node: src.clone() });
    let mut settled: HashSet<String> = HashSet::new();
    while let Some(State { cost, node }) = heap.pop() {
        if let Some(&dold) = dist.get(&node) {
            if cost > dold + f64::EPSILON {
                continue;
            }
        }
        if !settled.insert(node.clone()) {
            continue;
        }
        let mut relax_edges = |edges: &Vec<i64>, is_out: bool| {
            for eid in edges {
                if let Some(e) = st.edges.get(eid) {
                    let v = if is_out { &e.dst } else { &e.src };
                    let w = e.weight.unwrap_or(1.0).max(0.0);
                    let nd = cost + w;
                    let cur = dist.get(v).copied().unwrap_or(f64::INFINITY);
                    if nd + f64::EPSILON < cur {
                        dist.insert(v.clone(), nd);
                        parent.insert(v.clone(), node.clone());
                        heap.push(State { cost: nd, node: v.clone() });
                    }
                }
            }
        };
        if dir == "out" || dir == "all" {
            if let Some(es) = st.out_adj.get(&node) {
                relax_edges(es, true);
            }
        }
        if dir == "in" || dir == "all" {
            if let Some(es) = st.in_adj.get(&node) {
                relax_edges(es, false);
            }
        }
    }
    Value::Assoc(HashMap::from([
        ("dist".into(), Value::Assoc(dist.into_iter().map(|(k, v)| (k, Value::Real(v))).collect())),
        (
            "parent".into(),
            Value::Assoc(parent.into_iter().map(|(k, v)| (k, Value::String(v))).collect()),
        ),
    ]))
}

fn connected_components(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 1 {
        return Value::Expr { head: Box::new(Value::Symbol("ConnectedComponents".into())), args };
    }
    let gid = match get_graph(&args[0]) {
        Some(id) => id,
        None => {
            return Value::Expr {
                head: Box::new(Value::Symbol("ConnectedComponents".into())),
                args,
            }
        }
    };
    let reg = g_reg().lock().unwrap();
    let st = match reg.get(&gid) {
        Some(s) => s.clone(),
        None => return Value::Assoc(HashMap::new()),
    };
    drop(reg);
    use std::collections::{HashMap as Map, HashSet, VecDeque};
    let mut comp_id = 0i64;
    let mut comp_map: Map<String, i64> = Map::new();
    let mut seen: HashSet<String> = HashSet::new();
    for nid in st.nodes.keys() {
        if seen.contains(nid) {
            continue;
        }
        comp_id += 1;
        let mut q = VecDeque::new();
        q.push_back(nid.clone());
        seen.insert(nid.clone());
        comp_map.insert(nid.clone(), comp_id);
        while let Some(u) = q.pop_front() {
            let mut push = |edges: &Vec<i64>, is_out: bool| {
                for eid in edges {
                    if let Some(e) = st.edges.get(eid) {
                        let v = if is_out { &e.dst } else { &e.src };
                        if !seen.contains(v) {
                            seen.insert(v.clone());
                            comp_map.insert(v.clone(), comp_id);
                            q.push_back(v.clone());
                        }
                    }
                }
            };
            // Weakly connect: traverse both in/out
            if let Some(es) = st.out_adj.get(&u) {
                push(es, true);
            }
            if let Some(es) = st.in_adj.get(&u) {
                push(es, false);
            }
        }
    }
    Value::Assoc(comp_map.into_iter().map(|(k, v)| (k, Value::Integer(v))).collect())
}

fn strongly_connected_components(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 1 {
        return Value::Expr {
            head: Box::new(Value::Symbol("StronglyConnectedComponents".into())),
            args,
        };
    }
    let gid = match get_graph(&args[0]) {
        Some(id) => id,
        None => {
            return Value::Expr {
                head: Box::new(Value::Symbol("StronglyConnectedComponents".into())),
                args,
            }
        }
    };
    let reg = g_reg().lock().unwrap();
    let st = match reg.get(&gid) {
        Some(s) => s.clone(),
        None => return Value::List(Vec::new()),
    };
    drop(reg);
    // Kosaraju: order by finish time in DFS on original, then DFS on transpose
    use std::collections::HashSet;
    let mut seen: HashSet<String> = HashSet::new();
    let mut order: Vec<String> = Vec::new();
    fn dfs1(
        u: &str,
        st: &GraphState,
        seen: &mut std::collections::HashSet<String>,
        order: &mut Vec<String>,
    ) {
        if !seen.insert(u.to_string()) {
            return;
        }
        if let Some(es) = st.out_adj.get(u) {
            for eid in es {
                if let Some(e) = st.edges.get(eid) {
                    dfs1(&e.dst, st, seen, order);
                }
            }
        }
        order.push(u.to_string());
    }
    for nid in st.nodes.keys() {
        if !seen.contains(nid) {
            dfs1(nid, &st, &mut seen, &mut order);
        }
    }
    // Transpose DFS
    seen.clear();
    let mut comps: Vec<Vec<Value>> = Vec::new();
    fn dfs2(
        u: &str,
        st: &GraphState,
        seen: &mut std::collections::HashSet<String>,
        acc: &mut Vec<Value>,
    ) {
        if !seen.insert(u.to_string()) {
            return;
        }
        acc.push(Value::String(u.to_string()));
        if let Some(es) = st.in_adj.get(u) {
            for eid in es {
                if let Some(e) = st.edges.get(eid) {
                    dfs2(&e.src, st, seen, acc);
                }
            }
        }
    }
    while let Some(u) = order.pop() {
        if !seen.contains(&u) {
            let mut acc: Vec<Value> = Vec::new();
            dfs2(&u, &st, &mut seen, &mut acc);
            comps.push(acc);
        }
    }
    Value::List(comps.into_iter().map(Value::List).collect())
}

fn topological_sort(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 1 {
        return Value::Expr { head: Box::new(Value::Symbol("TopologicalSort".into())), args };
    }
    let gid = match get_graph(&args[0]) {
        Some(id) => id,
        None => {
            return Value::Expr { head: Box::new(Value::Symbol("TopologicalSort".into())), args }
        }
    };
    let reg = g_reg().lock().unwrap();
    let st = match reg.get(&gid) {
        Some(s) => s.clone(),
        None => return Value::List(Vec::new()),
    };
    drop(reg);
    // Kahn's algorithm
    use std::collections::{HashMap as Map, VecDeque};
    let mut indeg: Map<String, i64> = Map::new();
    for nid in st.nodes.keys() {
        indeg.insert(nid.clone(), 0);
    }
    for e in st.edges.values() {
        *indeg.entry(e.dst.clone()).or_insert(0) += 1;
    }
    let mut q = VecDeque::new();
    for (n, d) in indeg.iter() {
        if *d == 0 {
            q.push_back(n.clone());
        }
    }
    let mut out: Vec<Value> = Vec::new();
    let mut left = st.edges.len();
    while let Some(u) = q.pop_front() {
        out.push(Value::String(u.clone()));
        if let Some(es) = st.out_adj.get(&u) {
            for eid in es {
                if let Some(e) = st.edges.get(eid) {
                    if let Some(v) = indeg.get_mut(&e.dst) {
                        *v -= 1;
                        if *v == 0 {
                            q.push_back(e.dst.clone());
                        }
                        left = left.saturating_sub(1);
                    }
                }
            }
        }
    }
    if left > 0 {
        Value::Symbol("Null".into())
    } else {
        Value::List(out)
    }
}

// PageRank via power iteration
fn pagerank(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("PageRank".into())), args };
    }
    let gid = match get_graph(&args[0]) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("PageRank".into())), args },
    };
    let mut damping = 0.85f64;
    let mut tol = 1e-6f64;
    let mut max_iter = 100usize;
    let mut personalized: Option<HashMap<String, f64>> = None;
    let mut dir = "out".to_string();
    if args.len() >= 2 {
        if let Value::Assoc(m) = ev.eval(args[1].clone()) {
            if let Some(Value::Real(a)) = m.get("Damping") {
                damping = *a;
            } else if let Some(Value::Integer(a)) = m.get("Damping") {
                damping = *a as f64;
            }
            if let Some(Value::Real(t)) = m.get("Tol") {
                tol = *t;
            } else if let Some(Value::Integer(t)) = m.get("Tol") {
                tol = (*t as f64).max(1e-12);
            }
            if let Some(Value::Integer(mi)) = m.get("MaxIter") {
                max_iter = (*mi).max(1) as usize;
            }
            if let Some(Value::Assoc(v)) = m.get("Personalization") {
                let mut p = HashMap::new();
                for (k, val) in v {
                    if let Value::Real(f) = val {
                        p.insert(k.clone(), *f);
                    } else if let Value::Integer(i) = val {
                        p.insert(k.clone(), *i as f64);
                    }
                }
                personalized = Some(p);
            }
            if let Some(Value::String(s)) | Some(Value::Symbol(s)) =
                m.get("Direction").or_else(|| m.get("dir")).or_else(|| m.get("Dir"))
            {
                dir = s.to_lowercase();
            }
        }
    }
    let reg = g_reg().lock().unwrap();
    let st = match reg.get(&gid) {
        Some(s) => s.clone(),
        None => return Value::Assoc(HashMap::new()),
    };
    drop(reg);
    let n = st.nodes.len().max(1);
    let nodes: Vec<String> = st.nodes.keys().cloned().collect();
    let mut idx: HashMap<String, usize> = HashMap::new();
    for (i, k) in nodes.iter().enumerate() {
        idx.insert(k.clone(), i);
    }
    let mut outdeg: Vec<usize> = vec![0; n];
    for e in st.edges.values() {
        if let (Some(&i), Some(&j)) = (idx.get(&e.src), idx.get(&e.dst)) {
            if dir == "out" || dir == "all" {
                outdeg[i] += 1;
            } else {
                outdeg[j] += 1;
            }
        }
    }
    let mut r = vec![1.0 / (n as f64); n];
    let mut v = vec![0.0; n];
    let teleport: Vec<f64> = if let Some(p) = personalized {
        let mut sum = 0.0;
        let mut t = vec![0.0; n];
        for (k, val) in p {
            if let Some(&i) = idx.get(&k) {
                t[i] = val.max(0.0);
                sum += t[i];
            }
        }
        if sum > 0.0 {
            for x in t.iter_mut() {
                *x /= sum;
            }
            t
        } else {
            vec![1.0 / (n as f64); n]
        }
    } else {
        vec![1.0 / (n as f64); n]
    };
    for _ in 0..max_iter {
        for i in 0..n {
            v[i] = 0.0;
        }
        for e in st.edges.values() {
            let (u, w) =
                if dir == "out" || dir == "all" { (&e.src, &e.dst) } else { (&e.dst, &e.src) };
            let ui = *idx.get(u).unwrap();
            let wi = *idx.get(w).unwrap();
            let deg = outdeg[ui].max(1) as f64;
            v[wi] += r[ui] / deg;
        }
        let mut diff = 0.0;
        // Distribute dangling mass uniformly
        let mut dsum = 0.0;
        for i in 0..n {
            if outdeg[i] == 0 {
                dsum += r[i];
            }
        }
        let dshare = damping * dsum / (n as f64);
        for i in 0..n {
            let newv = damping * v[i] + dshare + (1.0 - damping) * teleport[i];
            diff += (newv - r[i]).abs();
            r[i] = newv;
        }
        if diff < tol {
            break;
        }
    }
    Value::Assoc(nodes.into_iter().enumerate().map(|(i, k)| (k, Value::Real(r[i]))).collect())
}

fn degree_centrality(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("DegreeCentrality".into())), args };
    }
    let gid = match get_graph(&args[0]) {
        Some(id) => id,
        None => {
            return Value::Expr { head: Box::new(Value::Symbol("DegreeCentrality".into())), args }
        }
    };
    let mut mode = "out".to_string();
    let mut normalize = true;
    if args.len() >= 2 {
        if let Value::Assoc(m) = ev.eval(args[1].clone()) {
            if let Some(Value::String(s)) | Some(Value::Symbol(s)) =
                m.get("Mode").or_else(|| m.get("mode"))
            {
                mode = s.to_lowercase();
            }
            if let Some(Value::Boolean(b)) = m.get("Normalize") {
                normalize = *b;
            }
        }
    }
    let reg = g_reg().lock().unwrap();
    let st = reg.get(&gid).unwrap().clone();
    drop(reg);
    let n = st.nodes.len().max(1) as f64;
    let mut deg: HashMap<String, f64> = st.nodes.keys().map(|k| (k.clone(), 0.0)).collect();
    match mode.as_str() {
        "in" => {
            for e in st.edges.values() {
                *deg.get_mut(&e.dst).unwrap() += 1.0;
            }
        }
        "all" => {
            for e in st.edges.values() {
                *deg.get_mut(&e.src).unwrap() += 1.0;
                *deg.get_mut(&e.dst).unwrap() += 1.0;
            }
        }
        _ => {
            for e in st.edges.values() {
                *deg.get_mut(&e.src).unwrap() += 1.0;
            }
        }
    }
    if normalize {
        let z = if mode == "all" { 2.0 * (n - 1.0) } else { n - 1.0 };
        if z > 0.0 {
            for v in deg.values_mut() {
                *v /= z;
            }
        }
    }
    Value::Assoc(deg.into_iter().map(|(k, v)| (k, Value::Real(v))).collect())
}

fn closeness_centrality(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("ClosenessCentrality".into())), args };
    }
    let gid = match get_graph(&args[0]) {
        Some(id) => id,
        None => {
            return Value::Expr {
                head: Box::new(Value::Symbol("ClosenessCentrality".into())),
                args,
            }
        }
    };
    let mut mode = "out".to_string();
    let mut normalize = true;
    if args.len() >= 2 {
        if let Value::Assoc(m) = ev.eval(args[1].clone()) {
            if let Some(Value::String(s)) | Some(Value::Symbol(s)) =
                m.get("Mode").or_else(|| m.get("mode"))
            {
                mode = s.to_lowercase();
            }
            if let Some(Value::Boolean(b)) = m.get("Normalize") {
                normalize = *b;
            }
        }
    }
    let reg = g_reg().lock().unwrap();
    let st = reg.get(&gid).unwrap().clone();
    drop(reg);
    use std::collections::{HashSet, VecDeque};
    let mut close: HashMap<String, f64> = HashMap::new();
    for start in st.nodes.keys() {
        let mut dist: HashMap<String, i64> = HashMap::new();
        let mut q = VecDeque::new();
        let mut seen: HashSet<String> = HashSet::new();
        seen.insert(start.clone());
        dist.insert(start.clone(), 0);
        q.push_back(start.clone());
        while let Some(u) = q.pop_front() {
            let du = *dist.get(&u).unwrap();
            let mut push_neighbors = |edges: &Vec<i64>, is_out: bool| {
                for eid in edges {
                    if let Some(e) = st.edges.get(eid) {
                        let v = if is_out { &e.dst } else { &e.src };
                        if !seen.contains(v) {
                            seen.insert(v.clone());
                            dist.insert(v.clone(), du + 1);
                            q.push_back(v.clone());
                        }
                    }
                }
            };
            if mode == "out" || mode == "all" {
                if let Some(es) = st.out_adj.get(&u) {
                    push_neighbors(es, true);
                }
            }
            if mode == "in" || mode == "all" {
                if let Some(es) = st.in_adj.get(&u) {
                    push_neighbors(es, false);
                }
            }
        }
        let mut sum = 0.0;
        for (k, d) in dist.iter() {
            if k != start {
                sum += *d as f64;
            }
        }
        let score = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        close.insert(start.clone(), score);
    }
    if normalize {
        let m = (st.nodes.len() as f64) - 1.0;
        if m > 0.0 {
            for v in close.values_mut() {
                *v *= m;
            }
        }
    }
    Value::Assoc(close.into_iter().map(|(k, v)| (k, Value::Real(v))).collect())
}

fn local_clustering(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("LocalClustering".into())), args };
    }
    let gid = match get_graph(&args[0]) {
        Some(id) => id,
        None => {
            return Value::Expr { head: Box::new(Value::Symbol("LocalClustering".into())), args }
        }
    };
    let reg = g_reg().lock().unwrap();
    let st = reg.get(&gid).unwrap().clone();
    drop(reg);
    let mut neigh: HashMap<String, HashSet<String>> = HashMap::new();
    for n in st.nodes.keys() {
        neigh.insert(n.clone(), HashSet::new());
    }
    for e in st.edges.values() {
        neigh.get_mut(&e.src).unwrap().insert(e.dst.clone());
        neigh.get_mut(&e.dst).unwrap().insert(e.src.clone());
    }
    let mut cc: HashMap<String, f64> = HashMap::new();
    for (u, ns) in neigh.iter() {
        let k = ns.len();
        if k < 2 {
            cc.insert(u.clone(), 0.0);
            continue;
        }
        let mut tri = 0usize;
        let ns_vec: Vec<&String> = ns.iter().collect();
        for i in 0..ns_vec.len() {
            for j in (i + 1)..ns_vec.len() {
                if neigh.get(ns_vec[i]).unwrap().contains(ns_vec[j]) {
                    tri += 1;
                }
            }
        }
        let denom = (k * (k - 1)) / 2;
        let coef = if denom > 0 { (tri as f64) / (denom as f64) } else { 0.0 };
        cc.insert(u.clone(), coef);
    }
    Value::Assoc(cc.into_iter().map(|(k, v)| (k, Value::Real(v))).collect())
}

fn global_clustering(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let lc = local_clustering(ev, args);
    match lc {
        Value::Assoc(m) => {
            let mut s = 0.0;
            let mut c = 0.0;
            for v in m.values() {
                if let Value::Real(f) = v {
                    s += *f;
                    c += 1.0;
                }
            }
            if c > 0.0 {
                Value::Real(s / c)
            } else {
                Value::Real(0.0)
            }
        }
        _ => lc,
    }
}

fn kcore_decomposition(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("KCoreDecomposition".into())), args };
    }
    let gid = match get_graph(&args[0]) {
        Some(id) => id,
        None => {
            return Value::Expr { head: Box::new(Value::Symbol("KCoreDecomposition".into())), args }
        }
    };
    let reg = g_reg().lock().unwrap();
    let st = reg.get(&gid).unwrap().clone();
    drop(reg);
    use std::collections::VecDeque;
    let mut deg: HashMap<String, i64> = HashMap::new();
    for n in st.nodes.keys() {
        let d = st.out_adj.get(n).map(|v| v.len()).unwrap_or(0)
            + st.in_adj.get(n).map(|v| v.len()).unwrap_or(0);
        deg.insert(n.clone(), d as i64);
    }
    let mut core: HashMap<String, i64> = HashMap::new();
    let mut q: VecDeque<String> = st.nodes.keys().cloned().collect();
    let mut k = 0i64;
    while let Some(u) = q.pop_front() {
        let du = *deg.get(&u).unwrap_or(&0);
        if du < k {
            core.insert(u.clone(), k);
            continue;
        }
        if du <= k {
            core.insert(u.clone(), du);
        }
        for eid in st.out_adj.get(&u).unwrap_or(&Vec::new()) {
            if let Some(e) = st.edges.get(eid) {
                if let Some(v) = deg.get_mut(&e.dst) {
                    if *v > 0 {
                        *v -= 1;
                    }
                }
            }
        }
        for eid in st.in_adj.get(&u).unwrap_or(&Vec::new()) {
            if let Some(e) = st.edges.get(eid) {
                if let Some(v) = deg.get_mut(&e.src) {
                    if *v > 0 {
                        *v -= 1;
                    }
                }
            }
        }
        if du > k {
            k = du;
        }
    }
    Value::Assoc(core.into_iter().map(|(k, v)| (k, Value::Integer(v))).collect())
}

fn kcore(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("KCore".into())), args };
    }
    let gid = match get_graph(&args[0]) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("KCore".into())), args },
    };
    let k = match ev.eval(args[1].clone()) {
        Value::Integer(n) => n.max(0),
        other => {
            return Value::Expr {
                head: Box::new(Value::Symbol("KCore".into())),
                args: vec![args[0].clone(), other],
            }
        }
    };
    let reg = g_reg().lock().unwrap();
    let st = reg.get(&gid).unwrap().clone();
    drop(reg);
    let mut alive: HashSet<String> = st.nodes.keys().cloned().collect();
    let mut changed = true;
    while changed {
        changed = false;
        let cur: Vec<String> = alive.iter().cloned().collect();
        for u in cur {
            let mut d = 0i64;
            if let Some(es) = st.out_adj.get(&u) {
                for eid in es {
                    if let Some(e) = st.edges.get(eid) {
                        if alive.contains(&e.dst) {
                            d += 1;
                        }
                    }
                }
            }
            if let Some(es) = st.in_adj.get(&u) {
                for eid in es {
                    if let Some(e) = st.edges.get(eid) {
                        if alive.contains(&e.src) {
                            d += 1;
                        }
                    }
                }
            }
            if d < k {
                alive.remove(&u);
                changed = true;
            }
        }
    }
    let new_id = next_g_id();
    let mut ng = GraphState {
        directed: st.directed,
        multigraph: st.multigraph,
        allow_self_loops: st.allow_self_loops,
        nodes: HashMap::new(),
        out_adj: HashMap::new(),
        in_adj: HashMap::new(),
        edges: HashMap::new(),
        edge_index: HashMap::new(),
        next_edge_id: 1,
    };
    for u in alive.iter() {
        ng.nodes.insert(u.clone(), st.nodes.get(u).unwrap().clone());
        ng.out_adj.entry(u.clone()).or_default();
        ng.in_adj.entry(u.clone()).or_default();
    }
    for e in st.edges.values() {
        if alive.contains(&e.src) && alive.contains(&e.dst) {
            let mut e2 = e.clone();
            e2.id = ng.next_edge_id;
            ng.next_edge_id += 1;
            ng.out_adj.get_mut(&e2.src).unwrap().push(e2.id);
            ng.in_adj.get_mut(&e2.dst).unwrap().push(e2.id);
            if !ng.directed {
                ng.out_adj.get_mut(&e2.dst).unwrap().push(e2.id);
                ng.in_adj.get_mut(&e2.src).unwrap().push(e2.id);
            }
            ng.edge_index
                .entry((e2.src.clone(), e2.dst.clone(), e2.key.clone()))
                .or_default()
                .insert(e2.id);
            ng.edges.insert(e2.id, e2);
        }
    }
    g_reg().lock().unwrap().insert(new_id, ng);
    graph_handle(new_id)
}

fn mst(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("MinimumSpanningTree".into())), args };
    }
    let gid = match get_graph(&args[0]) {
        Some(id) => id,
        None => {
            return Value::Expr {
                head: Box::new(Value::Symbol("MinimumSpanningTree".into())),
                args,
            }
        }
    };
    let reg = g_reg().lock().unwrap();
    let st = reg.get(&gid).unwrap().clone();
    drop(reg);
    if st.directed {
        return Value::Expr { head: Box::new(Value::Symbol("MinimumSpanningTree".into())), args };
    }
    let mut edges: Vec<&Edge> = st.edges.values().collect();
    edges.sort_by(|a, b| {
        a.weight
            .unwrap_or(1.0)
            .partial_cmp(&b.weight.unwrap_or(1.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let nodes: Vec<String> = st.nodes.keys().cloned().collect();
    let mut idx: HashMap<String, usize> = HashMap::new();
    for (i, n) in nodes.iter().enumerate() {
        idx.insert(n.clone(), i);
    }
    let mut parent: Vec<usize> = (0..nodes.len()).collect();
    let mut rank = vec![0usize; nodes.len()];
    fn find(p: &mut [usize], x: usize) -> usize {
        if p[x] != x {
            let px = p[x];
            p[x] = find(p, px);
        }
        p[x]
    }
    fn union(p: &mut [usize], r: &mut [usize], a: usize, b: usize) {
        let mut x = find(p, a);
        let mut y = find(p, b);
        if x == y {
            return;
        }
        if r[x] < r[y] {
            std::mem::swap(&mut x, &mut y);
        }
        p[y] = x;
        if r[x] == r[y] {
            r[x] += 1;
        }
    }
    let mut chosen: Vec<&Edge> = Vec::new();
    for e in edges {
        let u = idx[&e.src];
        let v = idx[&e.dst];
        let fu = find(&mut parent, u);
        let fv = find(&mut parent, v);
        if fu != fv {
            union(&mut parent, &mut rank, fu, fv);
            chosen.push(e);
            if chosen.len() + 1 >= nodes.len() {
                break;
            }
        }
    }
    let rows: Vec<Value> = chosen
        .into_iter()
        .map(|e| {
            Value::Assoc(HashMap::from([
                ("id".into(), Value::Integer(e.id)),
                ("src".into(), Value::String(e.src.clone())),
                ("dst".into(), Value::String(e.dst.clone())),
                (
                    "weight".into(),
                    e.weight.map(Value::Real).unwrap_or(Value::Symbol("Null".into())),
                ),
            ]))
        })
        .collect();
    Value::Expr {
        head: Box::new(Value::Symbol("DatasetFromRows".into())),
        args: vec![Value::List(rows)],
    }
}

fn max_flow(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 3 {
        return Value::Expr { head: Box::new(Value::Symbol("MaxFlow".into())), args };
    }
    let gid = match get_graph(&args[0]) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("MaxFlow".into())), args },
    };
    let s = key_of(&ev.eval(args[1].clone()));
    let t = key_of(&ev.eval(args[2].clone()));
    // Options: <|MaxAugmentations->N, MaxBFSSteps->M|>
    let mut max_aug = None::<usize>;
    let mut max_bfs = None::<usize>;
    if args.len() >= 4 {
        if let Value::Assoc(m) = ev.eval(args[3].clone()) {
            if let Some(Value::Integer(n)) = m.get("MaxAugmentations") {
                max_aug = Some((*n).max(0) as usize);
            }
            if let Some(Value::Integer(n)) = m.get("MaxBFSSteps") {
                max_bfs = Some((*n).max(0) as usize);
            }
        }
    }
    let reg = g_reg().lock().unwrap();
    let st = reg.get(&gid).unwrap().clone();
    drop(reg);
    use std::collections::HashMap as Map;
    let mut cap: Map<(String, String), f64> = Map::new();
    let mut adj: Map<String, Vec<String>> = Map::new();
    for e in st.edges.values() {
        let w = e.weight.unwrap_or(1.0).max(0.0);
        *cap.entry((e.src.clone(), e.dst.clone())).or_insert(0.0) += w;
        adj.entry(e.src.clone()).or_default().push(e.dst.clone());
        adj.entry(e.dst.clone()).or_default();
        adj.entry(e.dst.clone()).or_default().push(e.src.clone());
        cap.entry((e.dst.clone(), e.src.clone())).or_insert(0.0);
    }
    let mut flow = 0.0;
    let mut iters: usize = 0;
    let aug_cap = max_aug.unwrap_or_else(|| st.nodes.len().saturating_mul(st.edges.len()).max(1));
    loop {
        use std::collections::HashSet;
        use std::collections::VecDeque;
        let mut q = VecDeque::new();
        let mut parent: Map<String, Option<String>> = Map::new();
        let mut visited: HashSet<String> = HashSet::new();
        for n in st.nodes.keys() {
            parent.insert(n.clone(), None);
        }
        q.push_back(s.clone());
        visited.insert(s.clone());
        let mut bfs_steps: usize = 0;
        while let Some(u) = q.pop_front() {
            if u == t {
                break;
            }
            if let Some(ns) = adj.get(&u) {
                for v in ns {
                    if !visited.contains(v)
                        && parent[v].is_none()
                        && cap.get(&(u.clone(), v.clone())).copied().unwrap_or(0.0) > 1e-12
                    {
                        parent.insert(v.clone(), Some(u.clone()));
                        visited.insert(v.clone());
                        q.push_back(v.clone());
                    }
                }
            }
            bfs_steps += 1;
            if let Some(maxb) = max_bfs {
                if bfs_steps >= maxb {
                    break;
                }
            }
        }
        if parent[&t].is_none() {
            break;
        }
        let mut add = f64::INFINITY;
        let mut v = t.clone();
        while let Some(Some(u)) = parent.get(&v) {
            let c = cap.get(&(u.clone(), v.clone())).copied().unwrap_or(0.0);
            add = add.min(c);
            v = u.clone();
        }
        v = t.clone();
        while let Some(Some(u)) = parent.get(&v) {
            let key_uv = (u.clone(), v.clone());
            let key_vu = (v.clone(), u.clone());
            let cu = cap.get_mut(&key_uv).unwrap();
            *cu -= add;
            let cv = cap.get_mut(&key_vu).unwrap();
            *cv += add;
            v = u.clone();
        }
        flow += add;
        iters += 1;
        if iters >= aug_cap {
            break;
        }
    }
    Value::Real(flow)
}

/// Register graph algorithms and utilities: construction, traversals,
/// shortest paths, centrality, connectivity, layout, and formats.
pub fn register_graphs(ev: &mut Evaluator) {
    ev.register("Graph", graph_create as NativeFn, Attributes::empty());
    ev.register("DropGraph", drop_graph as NativeFn, Attributes::empty());
    ev.register("GraphInfo", graph_info as NativeFn, Attributes::empty());

    ev.register("AddNodes", add_nodes as NativeFn, Attributes::empty());
    ev.register("UpsertNodes", upsert_nodes as NativeFn, Attributes::empty());
    ev.register("AddEdges", add_edges as NativeFn, Attributes::empty());
    ev.register("UpsertEdges", upsert_edges as NativeFn, Attributes::empty());
    ev.register("RemoveNodes", remove_nodes as NativeFn, Attributes::empty());
    ev.register("RemoveEdges", remove_edges as NativeFn, Attributes::empty());
    ev.register("ListNodes", list_nodes as NativeFn, Attributes::empty());
    ev.register("ListEdges", list_edges as NativeFn, Attributes::empty());

    ev.register("HasNode", has_node as NativeFn, Attributes::empty());
    ev.register("HasEdge", has_edge as NativeFn, Attributes::empty());
    ev.register("Neighbors", neighbors as NativeFn, Attributes::empty());

    ev.register("IncidentEdges", incident_edges as NativeFn, Attributes::empty());
    ev.register("Subgraph", subgraph as NativeFn, Attributes::empty());
    ev.register("SampleNodes", sample_nodes as NativeFn, Attributes::empty());
    ev.register("SampleEdges", sample_edges as NativeFn, Attributes::empty());

    ev.register("BFS", bfs as NativeFn, Attributes::empty());
    ev.register("DFS", dfs as NativeFn, Attributes::empty());
    ev.register("ShortestPaths", shortest_paths as NativeFn, Attributes::empty());
    ev.register("ConnectedComponents", connected_components as NativeFn, Attributes::empty());
    ev.register(
        "StronglyConnectedComponents",
        strongly_connected_components as NativeFn,
        Attributes::empty(),
    );
    ev.register("TopologicalSort", topological_sort as NativeFn, Attributes::empty());
    ev.register("PageRank", pagerank as NativeFn, Attributes::empty());
    ev.register("DegreeCentrality", degree_centrality as NativeFn, Attributes::empty());
    ev.register("ClosenessCentrality", closeness_centrality as NativeFn, Attributes::empty());
    ev.register("LocalClustering", local_clustering as NativeFn, Attributes::empty());
    ev.register("GlobalClustering", global_clustering as NativeFn, Attributes::empty());
    ev.register("KCoreDecomposition", kcore_decomposition as NativeFn, Attributes::empty());
    ev.register("KCore", kcore as NativeFn, Attributes::empty());
    ev.register("MinimumSpanningTree", mst as NativeFn, Attributes::empty());
    ev.register("MaxFlow", max_flow as NativeFn, Attributes::empty());

    #[cfg(feature = "tools")]
    {
        use lyra_core::value::Value;
        // removed unused import
        add_specs(vec![
            tool_spec!(
                "Graph",
                summary: "Create a graph handle",
                params: ["opts?"],
                tags: ["graph","graphs","lifecycle"],
                examples: [
                    Value::String("g := Graph[]".into()),
                    Value::String("Graph[<|Directed->True|>]".into()),
                ]
            ),
            tool_spec!(
                "GraphInfo",
                summary: "Summary and counts for graph",
                params: ["graph"],
                tags: ["graph","introspect"],
                examples: [ Value::String("GraphInfo[g]".into()) ]
            ),
            tool_spec!(
                "AddNodes",
                summary: "Add nodes to graph",
                params: ["graph","nodes"],
                tags: ["graph","mutate"],
                examples: [ Value::String("AddNodes[g, {\"a\", \"b\"}]".into()) ]
            ),
            tool_spec!(
                "AddEdges",
                summary: "Add edges to graph",
                params: ["graph","edges"],
                tags: ["graph","mutate"],
                examples: [ Value::String("AddEdges[g, {<|Src->\"a\",Dst->\"b\"|>}]".into()) ]
            ),
            tool_spec!(
                "RemoveNodes",
                summary: "Remove nodes from graph",
                params: ["graph","nodes"],
                tags: ["graph","remove"],
                examples: [ Value::String("RemoveNodes[g, {\"a\"}]".into()) ]
            ),
            tool_spec!(
                "RemoveEdges",
                summary: "Remove edges from graph",
                params: ["graph","edges"],
                tags: ["graph","remove"],
                examples: [ Value::String("RemoveEdges[g, {<|Src->\"a\",Dst->\"b\"|>}]".into()) ]
            ),
            tool_spec!(
                "ListNodes",
                summary: "List node ids",
                params: ["graph"],
                tags: ["graph","list"]
            ),
            tool_spec!(
                "ListEdges",
                summary: "List edges as associations",
                params: ["graph"],
                tags: ["graph","list"]
            ),
            tool_spec!(
                "Neighbors",
                summary: "Neighbor nodes for a given node",
                params: ["graph","node","opts?"],
                tags: ["graph","query"],
                examples: [ Value::String("Neighbors[g, \"a\"]".into()) ]
            ),
            tool_spec!(
                "BFS",
                summary: "Breadth-first traversal",
                params: ["graph","source","opts?"],
                tags: ["graph","traversal"]
            ),
            tool_spec!(
                "DFS",
                summary: "Depth-first traversal",
                params: ["graph","source","opts?"],
                tags: ["graph","traversal"]
            ),
            tool_spec!(
                "ShortestPaths",
                summary: "Shortest path distances",
                params: ["graph","source","opts?"],
                tags: ["graph","paths"]
            ),
            tool_spec!(
                "PageRank",
                summary: "PageRank centrality",
                params: ["graph","opts?"],
                tags: ["graph","centrality"]
            ),
            tool_spec!(
                "KCore",
                summary: "k-core subgraph nodes",
                params: ["graph","k"],
                tags: ["graph","decomposition"]
            ),
            tool_spec!(
                "MinimumSpanningTree",
                summary: "Minimum spanning tree",
                params: ["graph","opts?"],
                tags: ["graph","mst"]
            ),
            tool_spec!(
                "MaxFlow",
                summary: "Maximum flow value",
                params: ["graph","source","sink","opts?"],
                tags: ["graph","flow"]
            ),
        ]);
    }
}

/// Conditionally register graph algorithms/utilities based on `pred`.
pub fn register_graphs_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    register_if(ev, pred, "Graph", graph_create as NativeFn, Attributes::empty());
    register_if(ev, pred, "DropGraph", drop_graph as NativeFn, Attributes::empty());
    register_if(ev, pred, "GraphInfo", graph_info as NativeFn, Attributes::empty());
    register_if(ev, pred, "AddNodes", add_nodes as NativeFn, Attributes::empty());
    register_if(ev, pred, "UpsertNodes", upsert_nodes as NativeFn, Attributes::empty());
    register_if(ev, pred, "AddEdges", add_edges as NativeFn, Attributes::empty());
    register_if(ev, pred, "UpsertEdges", upsert_edges as NativeFn, Attributes::empty());
    register_if(ev, pred, "RemoveNodes", remove_nodes as NativeFn, Attributes::empty());
    register_if(ev, pred, "RemoveEdges", remove_edges as NativeFn, Attributes::empty());
    register_if(ev, pred, "ListNodes", list_nodes as NativeFn, Attributes::empty());
    register_if(ev, pred, "ListEdges", list_edges as NativeFn, Attributes::empty());
    register_if(ev, pred, "HasNode", has_node as NativeFn, Attributes::empty());
    register_if(ev, pred, "HasEdge", has_edge as NativeFn, Attributes::empty());
    register_if(ev, pred, "Neighbors", neighbors as NativeFn, Attributes::empty());
    register_if(ev, pred, "IncidentEdges", incident_edges as NativeFn, Attributes::empty());
    register_if(ev, pred, "Subgraph", subgraph as NativeFn, Attributes::empty());
    register_if(ev, pred, "SampleNodes", sample_nodes as NativeFn, Attributes::empty());
    register_if(ev, pred, "SampleEdges", sample_edges as NativeFn, Attributes::empty());
    register_if(ev, pred, "BFS", bfs as NativeFn, Attributes::empty());
    register_if(ev, pred, "DFS", dfs as NativeFn, Attributes::empty());
    register_if(ev, pred, "ShortestPaths", shortest_paths as NativeFn, Attributes::empty());
    register_if(
        ev,
        pred,
        "ConnectedComponents",
        connected_components as NativeFn,
        Attributes::empty(),
    );
    register_if(
        ev,
        pred,
        "StronglyConnectedComponents",
        strongly_connected_components as NativeFn,
        Attributes::empty(),
    );
    register_if(ev, pred, "TopologicalSort", topological_sort as NativeFn, Attributes::empty());
    register_if(ev, pred, "PageRank", pagerank as NativeFn, Attributes::empty());
    register_if(ev, pred, "DegreeCentrality", degree_centrality as NativeFn, Attributes::empty());
    register_if(
        ev,
        pred,
        "ClosenessCentrality",
        closeness_centrality as NativeFn,
        Attributes::empty(),
    );
    register_if(ev, pred, "LocalClustering", local_clustering as NativeFn, Attributes::empty());
    register_if(ev, pred, "GlobalClustering", global_clustering as NativeFn, Attributes::empty());
    register_if(
        ev,
        pred,
        "KCoreDecomposition",
        kcore_decomposition as NativeFn,
        Attributes::empty(),
    );
    register_if(ev, pred, "KCore", kcore as NativeFn, Attributes::empty());
    register_if(ev, pred, "MinimumSpanningTree", mst as NativeFn, Attributes::empty());
    register_if(ev, pred, "MaxFlow", max_flow as NativeFn, Attributes::empty());
}
