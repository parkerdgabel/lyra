use crate::register_if;
#[cfg(feature = "tools")]
use crate::tool_spec;
#[cfg(feature = "tools")]
use crate::tools::add_specs;
use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

#[derive(Clone)]
struct FrameState {
    cols: Vec<String>,
    data: HashMap<String, Vec<Value>>, // column -> values (all vectors same length)
}

static FRAME_REG: OnceLock<Mutex<HashMap<i64, FrameState>>> = OnceLock::new();
static NEXT_FRAME_ID: OnceLock<std::sync::atomic::AtomicI64> = OnceLock::new();

fn frame_reg() -> &'static Mutex<HashMap<i64, FrameState>> {
    FRAME_REG.get_or_init(|| Mutex::new(HashMap::new()))
}
fn next_frame_id() -> i64 {
    let a = NEXT_FRAME_ID.get_or_init(|| std::sync::atomic::AtomicI64::new(1));
    a.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

fn frame_handle(id: i64) -> Value {
    Value::Assoc(HashMap::from([
        ("__type".to_string(), Value::String("Frame".into())),
        ("id".to_string(), Value::Integer(id)),
    ]))
}

fn get_frame(v: &Value) -> Option<i64> {
    if let Value::Assoc(m) = v {
        if matches!(m.get("__type"), Some(Value::String(s)) if s=="Frame") {
            if let Some(Value::Integer(id)) = m.get("id") {
                return Some(*id);
            }
        }
    }
    None
}

pub(crate) fn frame_count(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("Count".into())), args };
    }
    let id = match get_frame(&ev.eval(args[0].clone())) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("Count".into())), args },
    };
    let reg = frame_reg().lock().unwrap();
    if let Some(st) = reg.get(&id) {
        if let Some(first_col) = st.cols.first() {
            if let Some(vec) = st.data.get(first_col) { return Value::Integer(vec.len() as i64); }
        }
        return Value::Integer(0);
    }
    Value::Integer(0)
}

pub(crate) fn frame_info(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("Info".into())), args };
    }
    let subj = ev.eval(args[0].clone());
    let id = match get_frame(&subj) { Some(id) => id, None => return Value::Expr { head: Box::new(Value::Symbol("Info".into())), args: vec![subj] } };
    let reg = frame_reg().lock().unwrap();
    if let Some(st) = reg.get(&id) {
        let rows = if let Some(c0) = st.cols.first() { st.data.get(c0).map(|v| v.len()).unwrap_or(0) } else { 0 } as i64;
        return Value::Assoc(HashMap::from([
            ("Type".into(), Value::String("Frame".into())),
            ("Rows".into(), Value::Integer(rows)),
            ("Columns".into(), Value::List(st.cols.iter().cloned().map(Value::String).collect())),
        ]));
    }
    Value::Expr { head: Box::new(Value::Symbol("Info".into())), args: vec![subj] }
}

fn frame_from_rows(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("FrameFromRows".into())), args };
    }
    let rows_v = ev.eval(args[0].clone());
    let rows = match rows_v {
        Value::List(vs) => vs,
        _ => return Value::Expr { head: Box::new(Value::Symbol("FrameFromRows".into())), args: vec![rows_v] },
    };
    // Collect union of columns (sorted for determinism)
    use std::collections::BTreeSet;
    let mut cols_set: BTreeSet<String> = BTreeSet::new();
    let mut assoc_rows: Vec<HashMap<String, Value>> = Vec::with_capacity(rows.len());
    for r in rows.into_iter() {
        if let Value::Assoc(m) = r {
            for k in m.keys() { cols_set.insert(k.clone()); }
            assoc_rows.push(m);
        } else {
            // If non-assoc found, leave unevaluated
            return Value::Expr { head: Box::new(Value::Symbol("FrameFromRows".into())), args: vec![Value::List(vec![Value::Assoc(HashMap::new())])] };
        }
    }
    let cols: Vec<String> = cols_set.into_iter().collect();
    let n = assoc_rows.len();
    let mut data: HashMap<String, Vec<Value>> = HashMap::new();
    for c in &cols {
        data.insert(c.clone(), vec![Value::Symbol("Null".into()); n]);
    }
    for (i, r) in assoc_rows.into_iter().enumerate() {
        for c in &cols {
            if let Some(v) = r.get(c) { if let Some(col) = data.get_mut(c) { col[i] = v.clone(); } }
        }
    }
    let id = next_frame_id();
    frame_reg().lock().unwrap().insert(id, FrameState { cols: cols.clone(), data });
    frame_handle(id)
}

fn collect_frame(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("Collect".into())), args };
    }
    let id = match get_frame(&ev.eval(args[0].clone())) { Some(id) => id, None => return Value::Expr { head: Box::new(Value::Symbol("Collect".into())), args } };
    let reg = frame_reg().lock().unwrap();
    let st = match reg.get(&id) { Some(s) => s.clone(), None => return Value::Expr { head: Box::new(Value::Symbol("Collect".into())), args } };
    let n = if let Some(c0) = st.cols.first() { st.data.get(c0).map(|v| v.len()).unwrap_or(0) } else { 0 };
    let mut rows: Vec<Value> = Vec::with_capacity(n);
    for i in 0..n {
        let mut m = HashMap::new();
        for c in &st.cols {
            if let Some(vec) = st.data.get(c) { m.insert(c.clone(), vec[i].clone()); }
        }
        rows.push(Value::Assoc(m));
    }
    Value::List(rows)
}

fn columns_frame(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("Columns".into())), args };
    }
    let id = match get_frame(&ev.eval(args[0].clone())) { Some(id) => id, None => return Value::Expr { head: Box::new(Value::Symbol("Columns".into())), args } };
    let reg = frame_reg().lock().unwrap();
    if let Some(st) = reg.get(&id) {
        return Value::List(st.cols.iter().cloned().map(Value::String).collect());
    }
    Value::List(vec![])
}

fn frame_select(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("FrameSelect".into())), args };
    }
    let subj = ev.eval(args[0].clone());
    let id = match get_frame(&subj) { Some(id) => id, None => return Value::Expr { head: Box::new(Value::Symbol("FrameSelect".into())), args: vec![subj, ev.eval(args[1].clone())] } };
    let spec_v = ev.eval(args[1].clone());
    let reg = frame_reg().lock().unwrap();
    let st = match reg.get(&id) { Some(s) => s.clone(), None => return Value::Expr { head: Box::new(Value::Symbol("FrameSelect".into())), args: vec![subj, spec_v] } };
    let n = if let Some(c0) = st.cols.first() { st.data.get(c0).map(|v| v.len()).unwrap_or(0) } else { 0 };
    match spec_v {
        Value::List(cols_v) => {
            let want: Vec<String> = cols_v.into_iter().filter_map(|v| match v { Value::String(s) | Value::Symbol(s) => Some(s), _ => None }).collect();
            // Build new state with only wanted columns (in given order)
            let mut data: HashMap<String, Vec<Value>> = HashMap::new();
            for c in &want {
                if let Some(vec) = st.data.get(c) { data.insert(c.clone(), vec.clone()); }
            }
            let id2 = next_frame_id();
            frame_reg().lock().unwrap().insert(id2, FrameState { cols: want, data });
            frame_handle(id2)
        }
        Value::Assoc(mapping) => {
            // Compute new columns via expressions evaluated per-row with row assoc
            let mut new_cols: Vec<String> = Vec::new();
            let mut new_data: HashMap<String, Vec<Value>> = HashMap::new();
            // Prebuild row cache per index
            let mut row_cache: Vec<Value> = Vec::with_capacity(n);
            for i in 0..n {
                let mut m = HashMap::new();
                for c in &st.cols {
                    if let Some(vec) = st.data.get(c) { m.insert(c.clone(), vec[i].clone()); }
                }
                row_cache.push(Value::Assoc(m));
            }
            for (new_name, expr) in mapping.into_iter() {
                new_cols.push(new_name.clone());
                let mut col_vals: Vec<Value> = Vec::with_capacity(n);
                for i in 0..n {
                    let v = ev.eval(Value::Expr { head: Box::new(expr.clone()), args: vec![row_cache[i].clone()] });
                    col_vals.push(v);
                }
                new_data.insert(new_name, col_vals);
            }
            let id2 = next_frame_id();
            frame_reg().lock().unwrap().insert(id2, FrameState { cols: new_cols, data: new_data });
            frame_handle(id2)
        }
        other => Value::Expr { head: Box::new(Value::Symbol("FrameSelect".into())), args: vec![frame_handle(id), other] },
    }
}

fn frame_filter(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("FrameFilter".into())), args };
    }
    let pred = ev.eval(args[1].clone());
    let subj = ev.eval(args[0].clone());
    let id = match get_frame(&subj) { Some(id) => id, None => return Value::Expr { head: Box::new(Value::Symbol("FrameFilter".into())), args: vec![subj, pred] } };
    let reg = frame_reg().lock().unwrap();
    let st = match reg.get(&id) { Some(s) => s.clone(), None => return Value::Expr { head: Box::new(Value::Symbol("FrameFilter".into())), args: vec![frame_handle(id), pred] } };
    let n = if let Some(c0) = st.cols.first() { st.data.get(c0).map(|v| v.len()).unwrap_or(0) } else { 0 };
    // Build row cache
    let mut row_cache: Vec<Value> = Vec::with_capacity(n);
    for i in 0..n {
        let mut m = HashMap::new();
        for c in &st.cols {
            if let Some(vec) = st.data.get(c) { m.insert(c.clone(), vec[i].clone()); }
        }
        row_cache.push(Value::Assoc(m));
    }
    // Build mask
    let mut keep: Vec<bool> = Vec::with_capacity(n);
    for i in 0..n {
        let v = ev.eval(Value::Expr { head: Box::new(pred.clone()), args: vec![row_cache[i].clone()] });
        let b = matches!(v, Value::Boolean(true));
        keep.push(b);
    }
    // Apply mask across columns
    let mut data: HashMap<String, Vec<Value>> = HashMap::new();
    for c in &st.cols {
        if let Some(vec) = st.data.get(c) {
            let mut out: Vec<Value> = Vec::new();
            for (i, v) in vec.iter().enumerate() {
                if keep[i] { out.push(v.clone()); }
            }
            data.insert(c.clone(), out);
        }
    }
    let id2 = next_frame_id();
    frame_reg().lock().unwrap().insert(id2, FrameState { cols: st.cols.clone(), data });
    frame_handle(id2)
}

// Describe helpers reused from dataset semantics (simplified)
#[derive(Copy, Clone)]
enum ColType { Null, Boolean, Integer, Real, String }

fn value_to_coltype(v: &Value) -> ColType {
    match v {
        Value::Boolean(_) => ColType::Boolean,
        Value::Integer(_) => ColType::Integer,
        Value::Real(_) | Value::Rational { .. } | Value::BigReal(_) => ColType::Real,
        Value::String(_) => ColType::String,
        Value::Symbol(s) if s == "Null" => ColType::Null,
        Value::Symbol(_) => ColType::String,
        _ => ColType::String,
    }
}
fn unify(a: ColType, b: ColType) -> ColType {
    use ColType::*;
    match (a, b) {
        (Null, x) => x,
        (x, Null) => x,
        (String, _) | (_, String) => String,
        (Real, _) | (_, Real) => Real,
        (Integer, Boolean) | (Boolean, Integer) => Integer,
        (Integer, Integer) => Integer,
        (Boolean, Boolean) => Boolean,
    }
}

fn describe_frame(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("Describe".into())), args };
    }
    let subj = ev.eval(args[0].clone());
    let id = match get_frame(&subj) { Some(id) => id, None => return Value::Expr { head: Box::new(Value::Symbol("Describe".into())), args: vec![subj] } };
    let reg = frame_reg().lock().unwrap();
    let st = match reg.get(&id) { Some(s) => s.clone(), None => return Value::Expr { head: Box::new(Value::Symbol("Describe".into())), args: vec![frame_handle(id)] } };
    let n = if let Some(c0) = st.cols.first() { st.data.get(c0).map(|v| v.len()).unwrap_or(0) } else { 0 } as i64;
    let mut cols_sum: HashMap<String, Value> = HashMap::new();
    for c in &st.cols {
        let vec = st.data.get(c).cloned().unwrap_or_default();
        let mut non_null = 0i64; let mut nulls = 0i64; let mut ty = ColType::Null; let mut example: Option<Value> = None;
        use std::collections::HashSet; let mut uniq: HashSet<String> = HashSet::new();
        for v in vec.iter() {
            if matches!(v, Value::Symbol(s) if s=="Null") { nulls += 1; continue; }
            non_null += 1; if example.is_none() { example = Some(v.clone()); }
            ty = unify(ty, value_to_coltype(v));
            uniq.insert(lyra_core::pretty::format_value(v));
        }
        cols_sum.insert(c.clone(), Value::Assoc(HashMap::from([
            ("type".into(), Value::String(match ty { ColType::Null=>"Null", ColType::Boolean=>"Boolean", ColType::Integer=>"Integer", ColType::Real=>"Real", ColType::String=>"String" }.into())),
            ("nonNull".into(), Value::Integer(non_null)),
            ("nulls".into(), Value::Integer(nulls)),
            ("unique".into(), Value::Integer(uniq.len() as i64)),
            ("example".into(), example.unwrap_or(Value::Symbol("Null".into()))),
        ])));
    }
    Value::Assoc(HashMap::from([
        ("count".into(), Value::Integer(n)),
        ("columns".into(), Value::Assoc(cols_sum)),
    ]))
}

/// Register frame APIs for in-memory tabular data: construction, selection,
/// transformation, joins/grouping/aggregation, and display helpers.
pub fn register_frame(ev: &mut Evaluator) {
    ev.register("FrameFromRows", frame_from_rows as NativeFn, Attributes::empty());
    ev.register("FrameCollect", collect_frame as NativeFn, Attributes::empty());
    ev.register("FrameColumns", columns_frame as NativeFn, Attributes::empty());
    ev.register("FrameSelect", frame_select as NativeFn, Attributes::empty());
    ev.register("FrameFilter", frame_filter as NativeFn, Attributes::HOLD_ALL);
    ev.register("FrameDescribe", describe_frame as NativeFn, Attributes::empty());
    // Tier1: GroupBy/Aggregate (stubs)
    ev.register("GroupBy", frame_group_by as NativeFn, Attributes::empty());
    ev.register("Aggregate", frame_aggregate as NativeFn, Attributes::empty());
    ev.register("FrameJoin", frame_join as NativeFn, Attributes::empty());
    ev.register("FrameUnion", frame_union as NativeFn, Attributes::empty());
    ev.register("FrameHead", frame_head as NativeFn, Attributes::empty());
    ev.register("FrameTail", frame_tail as NativeFn, Attributes::empty());
    ev.register("FrameOffset", frame_offset as NativeFn, Attributes::empty());
    ev.register("FrameSort", frame_sort as NativeFn, Attributes::empty());
    ev.register("FrameDistinct", frame_distinct as NativeFn, Attributes::empty());
    ev.register("FrameWriteCSV", frame_write_csv as NativeFn, Attributes::empty());
    ev.register("FrameWriteJSONLines", frame_write_jsonl as NativeFn, Attributes::empty());

    #[cfg(feature = "tools")]
    add_specs(vec![
        tool_spec!("FrameFromRows", summary: "Create a Frame from assoc rows", params: ["rows"], tags: ["frame","create"], examples: [Value::String("f := FrameFromRows[{<|a->1|>,<|a->2|>}]".into())]),
        tool_spec!("FrameCollect", summary: "Materialize Frame to list of rows", params: ["frame"], tags: ["frame","io"], examples: [Value::String("FrameCollect[f]  ==> {<|...|>,...}".into())]),
        tool_spec!("FrameColumns", summary: "List column names for a Frame", params: ["frame"], tags: ["frame","schema"], examples: [Value::String("FrameColumns[f]".into())]),
        tool_spec!("FrameSelect", summary: "Select/compute columns in Frame", params: ["frame","spec"], tags: ["frame","transform","select"], examples: [Value::String("FrameSelect[f, {\"a\"}]".into())]),
        tool_spec!("FrameFilter", summary: "Filter rows in a Frame", params: ["frame","pred"], tags: ["frame","transform","filter"], examples: [Value::String("FrameFilter[f, #a>1 &]".into())]),
        tool_spec!("FrameDescribe", summary: "Quick stats by columns", params: ["frame","opts?"], tags: ["frame","stats"], examples: [Value::String("FrameDescribe[f]".into())]),
        tool_spec!("GroupBy", summary: "Group rows by column(s)", params: ["frame","cols"], tags: ["frame","group"], examples: [Value::String("GroupBy[f, {\"a\"}]".into())]),
        tool_spec!("Aggregate", summary: "Aggregate grouped data (stub)", params: ["group","spec"], tags: ["frame","group","aggregate"], examples: [Value::String("Aggregate[g, <|Count->True|>]".into())]),
        tool_spec!("FrameJoin", summary: "Join two Frames by keys", params: ["left","right","on?","opts?"], tags: ["frame","join"], examples: [Value::String("FrameJoin[f1, f2, {\"id\"}]".into())]),
        tool_spec!("FrameUnion", summary: "Union Frames by columns (schema union)", params: ["frames…"], tags: ["frame","set"], examples: [Value::String("FrameUnion[f1, f2]".into())]),
        tool_spec!("FrameHead", summary: "Take first n rows from Frame", params: ["frame","n?"], tags: ["frame","inspect"], examples: [Value::String("FrameHead[f, 5]".into())]),
        tool_spec!("FrameTail", summary: "Take last n rows from Frame", params: ["frame","n?"], tags: ["frame","inspect"], examples: [Value::String("FrameTail[f, 5]".into())]),
        tool_spec!("FrameOffset", summary: "Skip first n rows of Frame", params: ["frame","n"], tags: ["frame","transform"], examples: [Value::String("FrameOffset[f, 10]".into())]),
        tool_spec!("FrameSort", summary: "Sort Frame by columns", params: ["frame","by"], tags: ["frame","sort"], examples: [Value::String("FrameSort[f, {\"a\"}]".into())]),
        tool_spec!("FrameDistinct", summary: "Distinct rows in Frame (optional columns)", params: ["frame","cols?"], tags: ["frame","distinct"], examples: [Value::String("FrameDistinct[f, {\"a\"}]".into())]),
        tool_spec!("FrameWriteCSV", summary: "Write Frame to CSV file", params: ["path","frame","opts?"], tags: ["frame","io","csv"], examples: [Value::String("FrameWriteCSV[\"out.csv\", f]".into())]),
        tool_spec!("FrameWriteJSONLines", summary: "Write Frame rows as JSON Lines", params: ["path","frame","opts?"], tags: ["frame","io","json"], examples: [Value::String("FrameWriteJSONLines[\"out.jsonl\", f]".into())]),
    ]);
}

/// Conditionally register frame APIs based on `pred`.
pub fn register_frame_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    register_if(ev, pred, "FrameFromRows", frame_from_rows as NativeFn, Attributes::empty());
    register_if(ev, pred, "FrameCollect", collect_frame as NativeFn, Attributes::empty());
    register_if(ev, pred, "FrameColumns", columns_frame as NativeFn, Attributes::empty());
    register_if(ev, pred, "FrameSelect", frame_select as NativeFn, Attributes::empty());
    register_if(ev, pred, "FrameFilter", frame_filter as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "FrameDescribe", describe_frame as NativeFn, Attributes::empty());
    register_if(ev, pred, "GroupBy", frame_group_by as NativeFn, Attributes::empty());
    register_if(ev, pred, "Aggregate", frame_aggregate as NativeFn, Attributes::empty());
    register_if(ev, pred, "FrameJoin", frame_join as NativeFn, Attributes::empty());
    register_if(ev, pred, "FrameUnion", frame_union as NativeFn, Attributes::empty());
    register_if(ev, pred, "FrameHead", frame_head as NativeFn, Attributes::empty());
    register_if(ev, pred, "FrameTail", frame_tail as NativeFn, Attributes::empty());
    register_if(ev, pred, "FrameOffset", frame_offset as NativeFn, Attributes::empty());
    register_if(ev, pred, "FrameSort", frame_sort as NativeFn, Attributes::empty());
    register_if(ev, pred, "FrameDistinct", frame_distinct as NativeFn, Attributes::empty());
    register_if(ev, pred, "FrameWriteCSV", frame_write_csv as NativeFn, Attributes::empty());
    register_if(ev, pred, "FrameWriteJSONLines", frame_write_jsonl as NativeFn, Attributes::empty());
}

fn parse_on_columns(v: &Value) -> Vec<String> {
    match v {
        Value::List(vs) => vs.iter().filter_map(|x| match x { Value::String(s) | Value::Symbol(s) => Some(s.clone()), _ => None }).collect(),
        Value::String(s) | Value::Symbol(s) => vec![s.clone()],
        _ => vec![],
    }
}

// -------- Tier1 GroupBy/Aggregate stubs --------
fn frame_group_by(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // GroupBy[frame, cols]
    Value::Expr { head: Box::new(Value::Symbol("GroupBy".into())), args }
}

fn frame_aggregate(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Aggregate[group, <|Agg->...|>]
    Value::Expr { head: Box::new(Value::Symbol("Aggregate".into())), args }
}

fn frame_join(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // FrameJoin[left, right, on?, opts?]
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("FrameJoin".into())), args }; }
    let left = ev.eval(args[0].clone());
    let right = ev.eval(args[1].clone());
    let on = if args.len() >= 3 { parse_on_columns(&ev.eval(args[2].clone())) } else { vec![] };
    let (how, suffix_right) = if args.len() >= 4 {
        if let Value::Assoc(m) = ev.eval(args[3].clone()) {
            let how = match m.get("How") { Some(Value::String(s))|Some(Value::Symbol(s)) => s.to_lowercase(), _ => "inner".into() };
            let suf = match m.get("SuffixRight") { Some(Value::String(s))|Some(Value::Symbol(s)) => s.clone(), _ => String::from("_r") };
            (how, suf)
        } else { (String::from("inner"), String::from("_r")) }
    } else { (String::from("inner"), String::from("_r")) };
    let (lid, rid) = match (get_frame(&left), get_frame(&right)) { (Some(l), Some(r)) => (l, r), _ => return Value::Expr { head: Box::new(Value::Symbol("FrameJoin".into())), args: vec![left, right] } };
    let reg = frame_reg().lock().unwrap();
    let l = match reg.get(&lid) { Some(s) => s.clone(), None => return Value::Expr { head: Box::new(Value::Symbol("FrameJoin".into())), args: vec![left, right] } };
    let r = match reg.get(&rid) { Some(s) => s.clone(), None => return Value::Expr { head: Box::new(Value::Symbol("FrameJoin".into())), args: vec![left, right] } };

    // Prepare output columns: left cols then right cols (renamed for conflicts)
    let mut out_cols: Vec<String> = l.cols.clone();
    let mut right_map: HashMap<String, String> = HashMap::new();
    for rc in &r.cols {
        let mut name = rc.clone();
        if out_cols.contains(&name) || on.contains(rc) {
            name.push_str(&suffix_right);
            while out_cols.contains(&name) { name.push_str(&suffix_right); }
        }
        right_map.insert(rc.clone(), name.clone());
        out_cols.push(name);
    }

    // Helper: key for a row
    let key_for = |state: &FrameState, i: usize| -> String {
        if on.is_empty() { return String::from(""); }
        let parts: Vec<String> = on.iter().map(|c| {
            state.data.get(c).and_then(|v| v.get(i)).map(|v| lyra_core::pretty::format_value(v)).unwrap_or_default()
        }).collect();
        parts.join("\u{1f}")
    };

    // Build index for right
    let rlen = if let Some(c0) = r.cols.first() { r.data.get(c0).map(|v| v.len()).unwrap_or(0) } else { 0 };
    let mut rindex: HashMap<String, Vec<usize>> = HashMap::new();
    for j in 0..rlen { let k = key_for(&r, j); rindex.entry(k).or_default().push(j); }

    // Prepare output column vectors
    let mut data: HashMap<String, Vec<Value>> = HashMap::new();
    for c in &out_cols { data.insert(c.clone(), Vec::new()); }

    // Iterate left rows and match
    let llen = if let Some(c0) = l.cols.first() { l.data.get(c0).map(|v| v.len()).unwrap_or(0) } else { 0 };
    for i in 0..llen {
        let key = key_for(&l, i);
        let matches = rindex.get(&key).cloned().unwrap_or_default();
        if !matches.is_empty() {
            for j in matches {
                // push left cols
                for lc in &l.cols {
                    if let Some(vec) = l.data.get(lc) { data.get_mut(lc).unwrap().push(vec[i].clone()); }
                }
                // push right cols (renamed)
                for rc in &r.cols {
                    let outc = right_map.get(rc).unwrap();
                    if let Some(vec) = r.data.get(rc) { data.get_mut(outc).unwrap().push(vec[j].clone()); }
                }
            }
        } else if how == "left" {
            // push left row with Nulls for right
            for lc in &l.cols {
                if let Some(vec) = l.data.get(lc) { data.get_mut(lc).unwrap().push(vec[i].clone()); }
            }
            for rc in &r.cols {
                let outc = right_map.get(rc).unwrap();
                data.get_mut(outc).unwrap().push(Value::Symbol("Null".into()));
            }
        }
    }

    let id2 = next_frame_id();
    frame_reg().lock().unwrap().insert(id2, FrameState { cols: out_cols, data });
    frame_handle(id2)
}

fn frame_union(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Accept either a list of frames or variadic frames
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("FrameUnion".into())), args }; }
    let frames: Vec<Value> = if args.len() == 1 {
        let a0 = ev.eval(args[0].clone());
        match a0 { Value::List(vs) => vs, other => vec![other] }
    } else {
        args.into_iter().map(|a| ev.eval(a)).collect()
    };
    let ids: Vec<i64> = frames.into_iter().filter_map(|v| get_frame(&v)).collect();
    if ids.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("FrameUnion".into())), args: vec![] }; }
    let reg = frame_reg().lock().unwrap();
    let states: Vec<FrameState> = ids.iter().filter_map(|id| reg.get(id).cloned()).collect();
    if states.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("FrameUnion".into())), args: vec![] }; }
    // Ordered schema union: first frame cols, then new cols in discovery order
    let mut out_cols: Vec<String> = Vec::new();
    for st in &states {
        for c in &st.cols { if !out_cols.contains(c) { out_cols.push(c.clone()); } }
    }
    let mut data: HashMap<String, Vec<Value>> = HashMap::new();
    for c in &out_cols { data.insert(c.clone(), Vec::new()); }
    for st in &states {
        let m = if let Some(c0) = st.cols.first() { st.data.get(c0).map(|v| v.len()).unwrap_or(0) } else { 0 };
        for c in &out_cols {
            if let Some(vec) = st.data.get(c) {
                data.get_mut(c).unwrap().extend(vec.iter().cloned());
            } else {
                let v = data.get_mut(c).unwrap();
                v.extend((0..m).map(|_| Value::Symbol("Null".into())));
            }
        }
    }
    let id2 = next_frame_id();
    frame_reg().lock().unwrap().insert(id2, FrameState { cols: out_cols, data });
    frame_handle(id2)
}

fn frame_head(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("FrameHead".into())), args }; }
    let subj = ev.eval(args[0].clone());
    let n = if args.len() >= 2 { match ev.eval(args[1].clone()) { Value::Integer(k)=>k, other=> return Value::Expr{ head: Box::new(Value::Symbol("FrameHead".into())), args: vec![subj, other] } } } else { 10 };
    let id = match get_frame(&subj) { Some(id)=>id, None=> return Value::Expr { head: Box::new(Value::Symbol("FrameHead".into())), args: vec![subj] } };
    let reg = frame_reg().lock().unwrap();
    let st = match reg.get(&id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("FrameHead".into())), args: vec![frame_handle(id)] } };
    let take = n.max(0) as usize;
    let mut data: HashMap<String, Vec<Value>> = HashMap::new();
    for c in &st.cols {
        if let Some(vec) = st.data.get(c) {
            let mut out = vec.clone();
            if out.len() > take { out.truncate(take); }
            data.insert(c.clone(), out);
        }
    }
    let id2 = next_frame_id();
    frame_reg().lock().unwrap().insert(id2, FrameState { cols: st.cols.clone(), data });
    frame_handle(id2)
}

fn frame_tail(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("FrameTail".into())), args }; }
    let subj = ev.eval(args[0].clone());
    let n = if args.len() >= 2 { match ev.eval(args[1].clone()) { Value::Integer(k)=>k, other=> return Value::Expr{ head: Box::new(Value::Symbol("FrameTail".into())), args: vec![subj, other] } } } else { 10 };
    let id = match get_frame(&subj) { Some(id)=>id, None=> return Value::Expr { head: Box::new(Value::Symbol("FrameTail".into())), args: vec![subj] } };
    let reg = frame_reg().lock().unwrap();
    let st = match reg.get(&id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("FrameTail".into())), args: vec![frame_handle(id)] } };
    let k = n.max(0) as usize;
    // Determine length from first col
    let len = if let Some(c0) = st.cols.first() { st.data.get(c0).map(|v| v.len()).unwrap_or(0) } else { 0 };
    let start = len.saturating_sub(k);
    let mut data: HashMap<String, Vec<Value>> = HashMap::new();
    for c in &st.cols {
        if let Some(vec) = st.data.get(c) {
            let out: Vec<Value> = vec.iter().skip(start).cloned().collect();
            data.insert(c.clone(), out);
        }
    }
    let id2 = next_frame_id();
    frame_reg().lock().unwrap().insert(id2, FrameState { cols: st.cols.clone(), data });
    frame_handle(id2)
}

fn frame_offset(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("FrameOffset".into())), args }; }
    let subj = ev.eval(args[0].clone());
    let n = match ev.eval(args[1].clone()) { Value::Integer(k)=>k, other=> return Value::Expr{ head: Box::new(Value::Symbol("FrameOffset".into())), args: vec![subj, other] } };
    let id = match get_frame(&subj) { Some(id)=>id, None=> return Value::Expr { head: Box::new(Value::Symbol("FrameOffset".into())), args: vec![subj, Value::Integer(n)] } };
    let reg = frame_reg().lock().unwrap();
    let st = match reg.get(&id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("FrameOffset".into())), args: vec![frame_handle(id), Value::Integer(n)] } };
    let k = n.max(0) as usize;
    let mut data: HashMap<String, Vec<Value>> = HashMap::new();
    for c in &st.cols {
        if let Some(vec) = st.data.get(c) {
            let out: Vec<Value> = vec.iter().skip(k).cloned().collect();
            data.insert(c.clone(), out);
        }
    }
    let id2 = next_frame_id();
    frame_reg().lock().unwrap().insert(id2, FrameState { cols: st.cols.clone(), data });
    frame_handle(id2)
}

fn parse_sort_spec(spec: &Value) -> Option<Vec<(String, bool)>> {
    match spec {
        Value::List(vs) => {
            let cols: Vec<(String, bool)> = vs.iter().filter_map(|v| match v { Value::String(s)|Value::Symbol(s)=> Some((s.clone(), true)), _=> None }).collect();
            if cols.is_empty() { None } else { Some(cols) }
        }
        Value::Assoc(m) => {
            let mut out: Vec<(String, bool)> = Vec::new();
            for (col, dir) in m.iter() {
                let asc = match dir { Value::String(s)|Value::Symbol(s)=> s.to_lowercase() != "desc", _=> true };
                out.push((col.clone(), asc));
            }
            if out.is_empty() { None } else { Some(out) }
        }
        Value::String(s) | Value::Symbol(s) => Some(vec![(s.clone(), true)]),
        _ => None,
    }
}

fn frame_sort(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("FrameSort".into())), args }; }
    let subj = ev.eval(args[0].clone());
    let spec = ev.eval(args[1].clone());
    let id = match get_frame(&subj) { Some(id)=>id, None=> return Value::Expr { head: Box::new(Value::Symbol("FrameSort".into())), args: vec![subj, spec] } };
    let reg = frame_reg().lock().unwrap();
    let st = match reg.get(&id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("FrameSort".into())), args: vec![frame_handle(id), spec] } };
    let by = match parse_sort_spec(&spec) { Some(b)=>b, None=> return Value::Expr { head: Box::new(Value::Symbol("FrameSort".into())), args: vec![frame_handle(id), spec] } };
    // Build row indices and sort by keys
    let n = if let Some(c0) = st.cols.first() { st.data.get(c0).map(|v| v.len()).unwrap_or(0) } else { 0 };
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| {
        for (col, asc) in by.iter() {
            let va = st.data.get(col).and_then(|v| v.get(i)).cloned().unwrap_or(Value::Symbol("Null".into()));
            let vb = st.data.get(col).and_then(|v| v.get(j)).cloned().unwrap_or(Value::Symbol("Null".into()));
            let ka = lyra_runtime::eval::value_order_key(&va);
            let kb = lyra_runtime::eval::value_order_key(&vb);
            let ord = ka.cmp(&kb);
            if ord != std::cmp::Ordering::Equal { return if *asc { ord } else { ord.reverse() }; }
        }
        std::cmp::Ordering::Equal
    });
    let mut data: HashMap<String, Vec<Value>> = HashMap::new();
    for c in &st.cols {
        if let Some(vec) = st.data.get(c) {
            let mut out: Vec<Value> = Vec::with_capacity(n);
            for &i in &idx { out.push(vec[i].clone()); }
            data.insert(c.clone(), out);
        }
    }
    let id2 = next_frame_id();
    frame_reg().lock().unwrap().insert(id2, FrameState { cols: st.cols.clone(), data });
    frame_handle(id2)
}

fn frame_distinct(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("FrameDistinct".into())), args }; }
    let subj = ev.eval(args[0].clone());
    let id = match get_frame(&subj) { Some(id)=>id, None=> return Value::Expr { head: Box::new(Value::Symbol("FrameDistinct".into())), args: vec![subj] } };
    let cols: Option<Vec<String>> = if args.len() >= 2 {
        match ev.eval(args[1].clone()) {
            Value::List(vs) => Some(vs.into_iter().filter_map(|v| match v { Value::String(s)|Value::Symbol(s)=> Some(s), _=> None }).collect()),
            _ => None,
        }
    } else { None };
    let reg = frame_reg().lock().unwrap();
    let st = match reg.get(&id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("FrameDistinct".into())), args: vec![frame_handle(id)] } };
    let n = if let Some(c0) = st.cols.first() { st.data.get(c0).map(|v| v.len()).unwrap_or(0) } else { 0 };
    use std::collections::HashSet; let mut seen: HashSet<String> = HashSet::new();
    let mut keep_idx: Vec<usize> = Vec::new();
    for i in 0..n {
        let key = if let Some(cols) = &cols {
            cols.iter().map(|c| st.data.get(c).and_then(|v| v.get(i)).map(|v| lyra_core::pretty::format_value(v)).unwrap_or_default()).collect::<Vec<_>>().join("\u{1f}")
        } else {
            // full row key
            let mut m: HashMap<String, Value> = HashMap::new();
            for c in &st.cols { if let Some(vec) = st.data.get(c) { m.insert(c.clone(), vec[i].clone()); } }
            lyra_core::pretty::format_value(&Value::Assoc(m))
        };
        if seen.insert(key) { keep_idx.push(i); }
    }
    let mut data: HashMap<String, Vec<Value>> = HashMap::new();
    for c in &st.cols {
        if let Some(vec) = st.data.get(c) {
            let mut out: Vec<Value> = Vec::with_capacity(keep_idx.len());
            for &i in &keep_idx { out.push(vec[i].clone()); }
            data.insert(c.clone(), out);
        }
    }
    let id2 = next_frame_id();
    frame_reg().lock().unwrap().insert(id2, FrameState { cols: st.cols.clone(), data });
    frame_handle(id2)
}

fn frame_write_csv(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("FrameWriteCSV".into())), args }; }
    let path = ev.eval(args[0].clone());
    let frame = ev.eval(args[1].clone());
    let opts = if args.len()>=3 { ev.eval(args[2].clone()) } else { Value::Assoc(HashMap::new()) };
    let rows = ev.eval(Value::expr(Value::symbol("FrameCollect"), vec![frame]));
    ev.eval(Value::expr(Value::symbol("WriteCSV"), vec![path, rows, opts]))
}

fn frame_write_jsonl(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("FrameWriteJSONLines".into())), args }; }
    let path = ev.eval(args[0].clone());
    let frame = ev.eval(args[1].clone());
    let rows = ev.eval(Value::expr(Value::symbol("FrameCollect"), vec![frame]));
    let mut lines: Vec<Value> = Vec::new();
    if let Value::List(rs) = rows { for r in rs { lines.push(ev.eval(Value::expr(Value::symbol("ToJson"), vec![r, Value::Assoc(HashMap::new())]))); } }
    let text = Value::String(lines.into_iter().map(|v| match v { Value::String(s)=>s, other=> lyra_core::pretty::format_value(&other)}).collect::<Vec<_>>().join("\n"));
    ev.eval(Value::expr(Value::symbol("WriteFile"), vec![path, text]))
}
