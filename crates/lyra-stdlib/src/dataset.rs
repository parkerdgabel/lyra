use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_runtime::attrs::Attributes;
use std::collections::HashMap;
use std::sync::{OnceLock, Mutex};

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

// Minimal logical plan representation
#[derive(Clone)]
enum Plan {
    FromRows(Vec<Value>),                 // Vec<Assoc>
    Select { input: Box<Plan>, cols: Vec<String> },
    SelectRename { input: Box<Plan>, mapping: HashMap<String, String> }, // new -> old
    Filter { input: Box<Plan>, pred: Value }, // pred[row] -> Boolean
    Limit { input: Box<Plan>, n: i64 },
    WithColumns { input: Box<Plan>, defs: HashMap<String, Value> }, // defs: col -> expr[row]
    GroupBy { input: Box<Plan>, keys: Vec<String> },
    Agg { input: Box<Plan>, aggs: HashMap<String, Value> }, // aggs: outCol -> AggExpr
    Join { left: Box<Plan>, right: Box<Plan>, on: Vec<String>, how: String },
    Sort { input: Box<Plan>, by: Vec<(String, bool)> }, // (col, asc)
    Distinct { input: Box<Plan>, cols: Option<Vec<String>> },
}

#[derive(Clone)]
struct DatasetState { plan: Plan }

static DS_REG: OnceLock<Mutex<HashMap<i64, DatasetState>>> = OnceLock::new();
static NEXT_DS_ID: OnceLock<std::sync::atomic::AtomicI64> = OnceLock::new();

fn ds_reg() -> &'static Mutex<HashMap<i64, DatasetState>> { DS_REG.get_or_init(|| Mutex::new(HashMap::new())) }
fn next_ds_id() -> i64 { let a = NEXT_DS_ID.get_or_init(|| std::sync::atomic::AtomicI64::new(1)); a.fetch_add(1, std::sync::atomic::Ordering::Relaxed) }

fn ds_handle(id: i64) -> Value {
    Value::Assoc(HashMap::from([
        ("__type".to_string(), Value::String("Dataset".into())),
        ("id".to_string(), Value::Integer(id)),
    ]))
}

fn get_ds(v: &Value) -> Option<i64> {
    if let Value::Assoc(m) = v {
        if matches!(m.get("__type"), Some(Value::String(s)) if s=="Dataset") {
            if let Some(Value::Integer(id)) = m.get("id") { return Some(*id); }
        }
    }
    None
}

fn eval_plan(ev: &mut Evaluator, p: &Plan) -> Vec<Value> {
    match p {
        Plan::FromRows(rows) => rows.clone(),
        Plan::Select { input, cols } => {
            let rows = eval_plan(ev, input);
            rows.into_iter().map(|r| match r {
                Value::Assoc(m) => {
                    let mut out = HashMap::new();
                    for c in cols { if let Some(v) = m.get(c) { out.insert(c.clone(), v.clone()); } }
                    Value::Assoc(out)
                }
                other => other,
            }).collect()
        }
        Plan::SelectRename { input, mapping } => {
            let rows = eval_plan(ev, input);
            rows.into_iter().map(|r| match r {
                Value::Assoc(m) => {
                    let mut out = HashMap::new();
                    for (newk, oldk) in mapping { if let Some(v) = m.get(oldk) { out.insert(newk.clone(), v.clone()); } }
                    Value::Assoc(out)
                }
                other => other,
            }).collect()
        }
        Plan::Filter { input, pred } => {
            let rows = eval_plan(ev, input);
            rows.into_iter().filter_map(|r| {
                let keep = match ev.eval(Value::Expr { head: Box::new(pred.clone()), args: vec![r.clone()] }) {
                    Value::Boolean(b) => b,
                    _ => false,
                };
                if keep { Some(r) } else { None }
            }).collect()
        }
        Plan::Limit { input, n } => {
            let mut rows = eval_plan(ev, input);
            let take = (*n).max(0) as usize;
            if rows.len() > take { rows.truncate(take); }
            rows
        }
        Plan::WithColumns { input, defs } => {
            let rows = eval_plan(ev, input);
            rows.into_iter().map(|r| match r {
                Value::Assoc(mut m) => {
                    for (k, expr) in defs.iter() {
                        let val = ev.eval(Value::Expr { head: Box::new(expr.clone()), args: vec![Value::Assoc(m.clone())] });
                        m.insert(k.clone(), val);
                    }
                    Value::Assoc(m)
                }
                other => other,
            }).collect()
        }
        Plan::GroupBy { input, keys } => {
            // No materialization; represent as rows annotated with group key (for Explain only)
            // For evaluation, just pass-through rows; aggregation happens in Agg
            eval_plan(ev, input)
        }
        Plan::Agg { input, aggs } => {
            // If the input is a GroupBy plan, extract its keys; otherwise aggregate over all rows
            fn extract_group_keys(p: &Plan) -> Option<Vec<String>> {
                match p { Plan::GroupBy { keys, .. } => Some(keys.clone()), _ => None }
            }
            let (base_plan, keys) = match &**input {
                Plan::GroupBy { input: inner, keys } => (&**inner, keys.clone()),
                _ => (&**input, Vec::new()),
            };
            let rows = eval_plan(ev, base_plan);
            use std::collections::HashMap as Map;
            let mut groups: Map<String, Vec<Value>> = Map::new();
            let mut key_vals: Map<String, HashMap<String, Value>> = Map::new();
            for r in rows.into_iter() {
                if let Value::Assoc(m) = &r {
                    let kk: Vec<String> = keys.iter().map(|k| m.get(k).map(|v| lyra_core::pretty::format_value(v)).unwrap_or_default()).collect();
                    let key = kk.join("\u{1f}");
                    key_vals.entry(key.clone()).or_default();
                    groups.entry(key).or_default().push(r);
                }
            }
            let mut out_rows: Vec<Value> = Vec::new();
            for (k, rows) in groups.into_iter() {
                let first = rows.get(0).cloned().unwrap_or(Value::Assoc(HashMap::new()));
                let mut out = match first {
                    Value::Assoc(m) => {
                        let mut mm = HashMap::new();
                        for col in &keys { if let Some(v) = m.get(col) { mm.insert(col.clone(), v.clone()); } }
                        mm
                    }
                    _ => HashMap::new(),
                };
                // compute aggregates
                for (out_col, spec) in aggs.iter() {
                    let val = eval_agg_spec(ev, spec, &rows);
                    out.insert(out_col.clone(), val);
                }
                out_rows.push(Value::Assoc(out));
            }
            out_rows
        }
        Plan::Join { left, right, on, how } => {
            let lrows = eval_plan(ev, left);
            let rrows = eval_plan(ev, right);
            use std::collections::HashMap as Map;
            let mut rmap: Map<String, Vec<HashMap<String, Value>>> = Map::new();
            for r in rrows.into_iter() {
                if let Value::Assoc(m) = r { let key = join_key(&m, on); rmap.entry(key).or_default().push(m); }
            }
            let mut out: Vec<Value> = Vec::new();
            for lr in lrows.into_iter() {
                if let Value::Assoc(lm) = lr {
                    let key = join_key(&lm, on);
                    if let Some(rrs) = rmap.get(&key) {
                        for rm in rrs {
                            out.push(Value::Assoc(merge_rows(&lm, rm, on)));
                        }
                    } else if how=="left" {
                        out.push(Value::Assoc(lm));
                    }
                }
            }
            out
        }
        Plan::Sort { input, by } => {
            let mut rows = eval_plan(ev, input);
            rows.sort_by(|a, b| {
                let (ma, mb) = match (a, b) { (Value::Assoc(ma), Value::Assoc(mb)) => (ma, mb), _ => return std::cmp::Ordering::Equal };
                for (col, asc) in by.iter() {
                    let va = ma.get(col).cloned().unwrap_or(Value::Symbol("Null".into()));
                    let vb = mb.get(col).cloned().unwrap_or(Value::Symbol("Null".into()));
                    let ka = lyra_runtime::eval::value_order_key(&va);
                    let kb = lyra_runtime::eval::value_order_key(&vb);
                    let ord = ka.cmp(&kb);
                    if ord != std::cmp::Ordering::Equal {
                        return if *asc { ord } else { ord.reverse() };
                    }
                }
                std::cmp::Ordering::Equal
            });
            rows
        }
        Plan::Distinct { input, cols } => {
            use std::collections::HashSet;
            let rows = eval_plan(ev, input);
            let mut seen: HashSet<String> = HashSet::new();
            let mut out: Vec<Value> = Vec::new();
            for r in rows.into_iter() {
                if let Value::Assoc(m) = &r {
                    let key = if let Some(cols) = cols {
                        cols.iter().map(|c| m.get(c).map(|v| lyra_core::pretty::format_value(v)).unwrap_or_default()).collect::<Vec<_>>().join("\u{1f}")
                    } else {
                        lyra_core::pretty::format_value(&r)
                    };
                    if seen.insert(key) { out.push(r); }
                } else {
                    let key = lyra_core::pretty::format_value(&r);
                    if seen.insert(key) { out.push(r); }
                }
            }
            out
        }
    }
}

fn join_key(m: &HashMap<String, Value>, cols: &[String]) -> String {
    cols.iter().map(|c| m.get(c).map(|v| lyra_core::pretty::format_value(v)).unwrap_or_default()).collect::<Vec<_>>().join("\u{1f}")
}

fn merge_rows(left: &HashMap<String, Value>, right: &HashMap<String, Value>, on: &[String]) -> HashMap<String, Value> {
    let mut out = left.clone();
    for (k, v) in right.iter() {
        if on.contains(k) {
            // keep left join key
            continue;
        }
        if out.contains_key(k) {
            out.insert(format!("{}_right", k), v.clone());
        } else {
            out.insert(k.clone(), v.clone());
        }
    }
    out
}

fn eval_agg_spec(ev: &mut Evaluator, spec: &Value, rows: &Vec<Value>) -> Value {
    match spec {
        Value::Expr { head, args } => {
            match &**head {
                Value::Symbol(s) if s=="Count" => Value::Integer(rows.len() as i64),
                Value::Symbol(s) if s=="Sum" => {
                    let col = match args.get(0) {
                        Some(Value::String(c))|Some(Value::Symbol(c)) => c.clone(),
                        Some(expr) => {
                            // sum of expression over rows: expr[row]
                            let mut acc = 0.0f64;
                            for r in rows {
                                let x = ev.eval(Value::Expr { head: Box::new(expr.clone()), args: vec![r.clone()] });
                                match x { Value::Integer(n)=>acc+=n as f64, Value::Real(f)=>acc+=f, _=>{} }
                            }
                            return Value::Real(acc);
                        }
                        None => String::new(),
                    };
                    let mut acc = 0.0f64;
                    for r in rows {
                        if let Value::Assoc(m) = r { if let Some(v)=m.get(&col) { match v { Value::Integer(n)=>acc+=*n as f64, Value::Real(f)=>acc+=*f, _=>{} } } }
                    }
                    Value::Real(acc)
                }
                _ => Value::Symbol("Null".into()),
            }
        }
        _ => Value::Symbol("Null".into()),
    }
}

fn dataset_from_rows(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("DatasetFromRows".into())), args } }
    let rows_v = ev.eval(args[0].clone());
    let rows = match rows_v {
        Value::List(vs) => vs,
        other => return Value::Expr { head: Box::new(Value::Symbol("DatasetFromRows".into())), args: vec![other] },
    };
    let id = next_ds_id();
    ds_reg().lock().unwrap().insert(id, DatasetState { plan: Plan::FromRows(rows) });
    ds_handle(id)
}

fn collect_ds(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Collect".into())), args } }
    let id = match get_ds(&ev.eval(args[0].clone())) { Some(id)=>id, None => return Value::Expr { head: Box::new(Value::Symbol("Collect".into())), args } };
    let reg = ds_reg().lock().unwrap();
    let st = match reg.get(&id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("Collect".into())), args } };
    Value::List(eval_plan(ev, &st.plan))
}

fn select_cols(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("SelectCols".into())), args } }
    let id = match get_ds(&ev.eval(args[0].clone())) { Some(id)=>id, None => return Value::Expr { head: Box::new(Value::Symbol("SelectCols".into())), args } };
    let cols_v = ev.eval(args[1].clone());
    let cols: Vec<String> = match cols_v {
        Value::List(vs) => vs.into_iter().filter_map(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s), _=>None }).collect(),
        _ => return Value::Expr { head: Box::new(Value::Symbol("SelectCols".into())), args },
    };
    let mut reg = ds_reg().lock().unwrap();
    let st = match reg.get(&id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("SelectCols".into())), args } };
    let new_id = next_ds_id();
    reg.insert(new_id, DatasetState { plan: Plan::Select { input: Box::new(st.plan), cols } });
    ds_handle(new_id)
}

fn filter_rows(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("FilterRows".into())), args } }
    let pred = args[0].clone();
    let id = match get_ds(&ev.eval(args[1].clone())) { Some(id)=>id, None => return Value::Expr { head: Box::new(Value::Symbol("FilterRows".into())), args } };
    let mut reg = ds_reg().lock().unwrap();
    let st = match reg.get(&id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("FilterRows".into())), args } };
    let new_id = next_ds_id();
    reg.insert(new_id, DatasetState { plan: Plan::Filter { input: Box::new(st.plan), pred } });
    ds_handle(new_id)
}

fn limit_rows(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("LimitRows".into())), args } }
    let id = match get_ds(&ev.eval(args[0].clone())) { Some(id)=>id, None => return Value::Expr { head: Box::new(Value::Symbol("LimitRows".into())), args } };
    let n = match ev.eval(args[1].clone()) { Value::Integer(k)=>k, other=> return Value::Expr { head: Box::new(Value::Symbol("LimitRows".into())), args: vec![ev.eval(args[0].clone()), other] } };
    let mut reg = ds_reg().lock().unwrap();
    let st = match reg.get(&id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("LimitRows".into())), args } };
    let new_id = next_ds_id();
    reg.insert(new_id, DatasetState { plan: Plan::Limit { input: Box::new(st.plan), n } });
    ds_handle(new_id)
}

fn count_ds(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Count".into())), args } }
    let id = match get_ds(&ev.eval(args[0].clone())) { Some(id)=>id, None => return Value::Expr { head: Box::new(Value::Symbol("Count".into())), args } };
    let reg = ds_reg().lock().unwrap();
    let st = match reg.get(&id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("Count".into())), args } };
    Value::Integer(eval_plan(ev, &st.plan).len() as i64)
}

fn columns_ds(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Columns".into())), args } }
    let id = match get_ds(&ev.eval(args[0].clone())) { Some(id)=>id, None => return Value::Expr { head: Box::new(Value::Symbol("Columns".into())), args } };
    let reg = ds_reg().lock().unwrap();
    let st = match reg.get(&id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("Columns".into())), args } };
    let rows = eval_plan(ev, &st.plan);
    if let Some(Value::Assoc(m)) = rows.first() {
        Value::List(m.keys().cloned().map(Value::String).collect())
    } else { Value::List(vec![]) }
}

fn dataset_schema(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("DatasetSchema".into())), args } }
    let id = match get_ds(&ev.eval(args[0].clone())) { Some(id)=>id, None => return Value::Expr { head: Box::new(Value::Symbol("DatasetSchema".into())), args } };
    let reg = ds_reg().lock().unwrap();
    let st = match reg.get(&id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("DatasetSchema".into())), args } };
    let rows = eval_plan(ev, &st.plan);
    let cols = if let Some(Value::Assoc(m)) = rows.first() { m.keys().cloned().collect::<Vec<_>>() } else { Vec::new() };
    Value::Assoc(HashMap::from([
        ("name".to_string(), Value::String("Dataset/v1".into())),
        ("columns".to_string(), Value::List(cols.into_iter().map(Value::String).collect())),
    ]))
}

fn explain_ds(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("ExplainDataset".into())), args } }
    let v = &args[0];
    let id = match get_ds(v) { Some(id)=>id, None => return Value::Expr { head: Box::new(Value::Symbol("ExplainDataset".into())), args } };
    let reg = ds_reg().lock().unwrap();
    let st = match reg.get(&id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("ExplainDataset".into())), args } };
    fn pp(plan: &Plan, indent: usize, out: &mut String) {
        let pad = " ".repeat(indent);
        match plan {
            Plan::FromRows(rows) => { out.push_str(&format!("{}FromRows(rows={})\n", pad, rows.len())); }
            Plan::Select { input, cols } => { out.push_str(&format!("{}Select {:?}\n", pad, cols)); pp(input, indent+2, out); }
            Plan::Filter { input, .. } => { out.push_str(&format!("{}Filter [pred]\n", pad)); pp(input, indent+2, out); }
            Plan::Limit { input, n } => { out.push_str(&format!("{}Limit {}\n", pad, n)); pp(input, indent+2, out); }
            Plan::SelectRename { input, mapping } => { out.push_str(&format!("{}SelectRename {:?}\n", pad, mapping.keys().collect::<Vec<_>>())); pp(input, indent+2, out); }
            Plan::WithColumns { input, defs } => { out.push_str(&format!("{}WithColumns keys={:?}\n", pad, defs.keys().collect::<Vec<_>>())); pp(input, indent+2, out); }
            Plan::GroupBy { input, keys } => { out.push_str(&format!("{}GroupBy {:?}\n", pad, keys)); pp(input, indent+2, out); }
            Plan::Agg { input, aggs } => { out.push_str(&format!("{}Agg {:?}\n", pad, aggs.keys().collect::<Vec<_>>())); pp(input, indent+2, out); }
            Plan::Join { left, right, on, how } => { out.push_str(&format!("{}Join how={} on={:?}\n", pad, how, on)); pp(left, indent+2, out); pp(right, indent+2, out); }
            Plan::Sort { input, by } => { out.push_str(&format!("{}Sort {:?}\n", pad, by)); pp(input, indent+2, out); }
            Plan::Distinct { input, cols } => { out.push_str(&format!("{}Distinct {:?}\n", pad, cols)); pp(input, indent+2, out); }
        }
    }
    let mut s = String::new();
    pp(&st.plan, 0, &mut s);
    Value::String(s)
}

fn show_ds(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let (ds_v, n) = match args.as_slice() {
        [ds] => (ds.clone(), 20i64),
        [ds, Value::Integer(k)] => (ds.clone(), *k),
        _ => return Value::Expr { head: Box::new(Value::Symbol("ShowDataset".into())), args },
    };
    let id = match get_ds(&ev.eval(ds_v)) { Some(id)=>id, None => return Value::Expr { head: Box::new(Value::Symbol("ShowDataset".into())), args } };
    let reg = ds_reg().lock().unwrap();
    let st = match reg.get(&id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("ShowDataset".into())), args } };
    let mut rows = eval_plan(ev, &st.plan);
    let take = n.max(0) as usize;
    if rows.len() > take { rows.truncate(take); }
    // render basic table
    let cols: Vec<String> = if let Some(Value::Assoc(m)) = rows.first() { m.keys().cloned().collect() } else { vec![] };
    let mut widths: HashMap<String, usize> = cols.iter().map(|c| (c.clone(), c.len())).collect();
    for r in &rows {
        if let Value::Assoc(m) = r {
            for c in &cols {
                let s = m.get(c).map(|v| lyra_core::pretty::format_value(v)).unwrap_or_default();
                let w = widths.get_mut(c).unwrap();
                *w = (*w).max(s.len());
            }
        }
    }
    let mut out = String::new();
    // header
    for (i, c) in cols.iter().enumerate() {
        if i>0 { out.push_str(" | "); }
        let w = *widths.get(c).unwrap_or(&c.len());
        out.push_str(&format!("{:<w$}", c, w=w));
    }
    if !cols.is_empty() { out.push('\n'); out.push_str(&cols.iter().enumerate().map(|(i,c)| {
        let w = *widths.get(c).unwrap();
        let sep = "-".repeat(w);
        if i==0 { sep } else { format!("-+-{}", sep) }
    }).collect::<Vec<_>>().join("")); out.push('\n'); }
    for r in rows {
        if let Value::Assoc(m) = r {
            for (i, c) in cols.iter().enumerate() {
                if i>0 { out.push_str(" | "); }
                let s = m.get(c).map(|v| lyra_core::pretty::format_value(v)).unwrap_or_default();
                let w = *widths.get(c).unwrap();
                out.push_str(&format!("{:<w$}", s, w=w));
            }
            out.push('\n');
        }
    }
    Value::String(out)
}

fn read_csv_dataset(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Delegate to existing ReadCSV to get rows, then wrap
    let call = Value::Expr { head: Box::new(Value::Symbol("ReadCSV".into())), args };
    let rows = ev.eval(call);
    dataset_from_rows(ev, vec![rows])
}

fn read_jsonl_dataset(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Read lines; parse each JSON into a row (Assoc). Non-object JSON rows are skipped.
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("ReadJsonLinesDataset".into())), args } }
    let call = Value::Expr { head: Box::new(Value::Symbol("ReadLines".into())), args: vec![args[0].clone()] };
    let lines_v = ev.eval(call);
    let mut rows: Vec<Value> = Vec::new();
    if let Value::List(lines) = lines_v {
        for l in lines {
            let j = ev.eval(Value::Expr { head: Box::new(Value::Symbol("FromJson".into())), args: vec![l] });
            if let Value::Assoc(_) = j { rows.push(j); }
        }
    }
    dataset_from_rows(ev, vec![Value::List(rows)])
}

fn with_columns(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("WithColumns".into())), args } }
    let id = match get_ds(&ev.eval(args[0].clone())) { Some(id)=>id, None => return Value::Expr { head: Box::new(Value::Symbol("WithColumns".into())), args } };
    let defs_v = ev.eval(args[1].clone());
    let defs: HashMap<String, Value> = match defs_v {
        Value::Assoc(m) => m,
        _ => return Value::Expr { head: Box::new(Value::Symbol("WithColumns".into())), args },
    };
    let mut reg = ds_reg().lock().unwrap();
    let st = match reg.get(&id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("WithColumns".into())), args } };
    let new_id = next_ds_id();
    reg.insert(new_id, DatasetState { plan: Plan::WithColumns { input: Box::new(st.plan), defs } });
    ds_handle(new_id)
}

fn group_by(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("GroupBy".into())), args } }
    let id = match get_ds(&ev.eval(args[0].clone())) { Some(id)=>id, None => return Value::Expr { head: Box::new(Value::Symbol("GroupBy".into())), args } };
    let keys_v = ev.eval(args[1].clone());
    let keys: Vec<String> = match keys_v { Value::List(vs) => vs.into_iter().filter_map(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s), _=>None }).collect(), _ => return Value::Expr { head: Box::new(Value::Symbol("GroupBy".into())), args }, };
    let mut reg = ds_reg().lock().unwrap();
    let st = match reg.get(&id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("GroupBy".into())), args } };
    let new_id = next_ds_id();
    reg.insert(new_id, DatasetState { plan: Plan::GroupBy { input: Box::new(st.plan), keys } });
    ds_handle(new_id)
}

fn agg(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("Agg".into())), args } }
    let id = match get_ds(&ev.eval(args[0].clone())) { Some(id)=>id, None => return Value::Expr { head: Box::new(Value::Symbol("Agg".into())), args } };
    let aggs_v = ev.eval(args[1].clone());
    let aggs: HashMap<String, Value> = match aggs_v { Value::Assoc(m) => m, _ => return Value::Expr { head: Box::new(Value::Symbol("Agg".into())), args }, };
    let mut reg = ds_reg().lock().unwrap();
    let st = match reg.get(&id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("Agg".into())), args } };
    let new_id = next_ds_id();
    reg.insert(new_id, DatasetState { plan: Plan::Agg { input: Box::new(st.plan), aggs } });
    ds_handle(new_id)
}

fn join_ds(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Join[left, right, onList, opts? where How->"inner"|"left"]
    if args.len()<3 { return Value::Expr { head: Box::new(Value::Symbol("Join".into())), args } }
    let left_id = match get_ds(&ev.eval(args[0].clone())) { Some(id)=>id, None => return Value::Expr { head: Box::new(Value::Symbol("Join".into())), args } };
    let right_id = match get_ds(&ev.eval(args[1].clone())) { Some(id)=>id, None => return Value::Expr { head: Box::new(Value::Symbol("Join".into())), args } };
    let on_v = ev.eval(args[2].clone());
    let on: Vec<String> = match on_v { Value::List(vs) => vs.into_iter().filter_map(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s), _=>None }).collect(), _ => return Value::Expr { head: Box::new(Value::Symbol("Join".into())), args }, };
    let how = if let Some(Value::Assoc(m)) = args.get(3).map(|x| ev.eval(x.clone())) { if let Some(Value::String(s))|Some(Value::Symbol(s)) = m.get("How") { s.clone() } else { "inner".into() } } else { "inner".into() };
    let mut reg = ds_reg().lock().unwrap();
    let l = match reg.get(&left_id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("Join".into())), args } };
    let r = match reg.get(&right_id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("Join".into())), args } };
    let new_id = next_ds_id();
    reg.insert(new_id, DatasetState { plan: Plan::Join { left: Box::new(l.plan), right: Box::new(r.plan), on, how } });
    ds_handle(new_id)
}

fn parse_sort_spec(spec: &Value) -> Option<Vec<(String, bool)>> {
    match spec {
        Value::List(vs) => {
            let cols: Vec<(String, bool)> = vs.iter().filter_map(|v| match v { Value::String(s)|Value::Symbol(s)=>Some((s.clone(), true)), _=>None }).collect();
            if cols.is_empty() { None } else { Some(cols) }
        }
        Value::Assoc(m) => {
            let mut out: Vec<(String, bool)> = Vec::new();
            for (col, dir) in m.iter() {
                let asc = match dir { Value::String(s)|Value::Symbol(s) => s.to_lowercase() != "desc", _ => true };
                out.push((col.clone(), asc));
            }
            if out.is_empty() { None } else { Some(out) }
        }
        Value::String(s) | Value::Symbol(s) => Some(vec![(s.clone(), true)]),
        _ => None,
    }
}

fn sort_ds(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("Sort".into())), args } }
    let id = match get_ds(&ev.eval(args[0].clone())) { Some(id)=>id, None => return Value::Expr { head: Box::new(Value::Symbol("Sort".into())), args } };
    let spec = ev.eval(args[1].clone());
    let by = match parse_sort_spec(&spec) { Some(b)=>b, None => return Value::Expr { head: Box::new(Value::Symbol("Sort".into())), args } };
    let mut reg = ds_reg().lock().unwrap();
    let st = match reg.get(&id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("Sort".into())), args } };
    let new_id = next_ds_id();
    reg.insert(new_id, DatasetState { plan: Plan::Sort { input: Box::new(st.plan), by } });
    ds_handle(new_id)
}

fn distinct_ds(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Distinct[ds] or Distinct[ds, cols]
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Distinct".into())), args } }
    let id = match get_ds(&ev.eval(args[0].clone())) { Some(id)=>id, None => return Value::Expr { head: Box::new(Value::Symbol("Distinct".into())), args } };
    let cols: Option<Vec<String>> = if args.len()>=2 {
        match ev.eval(args[1].clone()) {
            Value::List(vs) => Some(vs.into_iter().filter_map(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s), _=>None }).collect()),
            _ => None,
        }
    } else { None };
    let mut reg = ds_reg().lock().unwrap();
    let st = match reg.get(&id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("Distinct".into())), args } };
    let new_id = next_ds_id();
    reg.insert(new_id, DatasetState { plan: Plan::Distinct { input: Box::new(st.plan), cols } });
    ds_handle(new_id)
}

fn rename_cols(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("RenameCols".into())), args } }
    // Delegate to Select with mapping for consistency
    select_general(ev, args)
}

pub fn register_dataset(ev: &mut Evaluator) {
    ev.register("DatasetFromRows", dataset_from_rows as NativeFn, Attributes::empty());
    ev.register("Collect", collect_ds as NativeFn, Attributes::empty());
    ev.register("SelectCols", select_cols as NativeFn, Attributes::empty());
    ev.register("Select", select_general as NativeFn, Attributes::empty());
    ev.register("FilterRows", filter_rows as NativeFn, Attributes::HOLD_ALL);
    ev.register("LimitRows", limit_rows as NativeFn, Attributes::empty());
    ev.register("WithColumns", with_columns as NativeFn, Attributes::HOLD_ALL);
    ev.register("Count", count_ds as NativeFn, Attributes::empty());
    ev.register("Columns", columns_ds as NativeFn, Attributes::empty());
    ev.register("DatasetSchema", dataset_schema as NativeFn, Attributes::empty());
    ev.register("ExplainDataset", explain_ds as NativeFn, Attributes::empty());
    ev.register("ReadCSVDataset", read_csv_dataset as NativeFn, Attributes::empty());
    ev.register("ShowDataset", show_ds as NativeFn, Attributes::empty());
    ev.register("ReadJsonLinesDataset", read_jsonl_dataset as NativeFn, Attributes::empty());
    ev.register("GroupBy", group_by as NativeFn, Attributes::empty());
    ev.register("Agg", agg as NativeFn, Attributes::empty());
    ev.register("Join", join_ds as NativeFn, Attributes::empty());
    ev.register("Sort", sort_ds as NativeFn, Attributes::empty());
    ev.register("Distinct", distinct_ds as NativeFn, Attributes::empty());
    ev.register("RenameCols", rename_cols as NativeFn, Attributes::empty());
}

fn select_general(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("Select".into())), args } }
    let subj = ev.eval(args[0].clone());
    let spec_v = ev.eval(args[1].clone());
    // Dataset projection
    if let Some(ds_id) = get_ds(&subj) {
        match &spec_v {
            Value::List(vs) => {
                let cols: Vec<String> = vs.iter().filter_map(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None }).collect();
                if cols.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Select".into())), args: vec![subj, spec_v] } }
                return select_cols(ev, vec![ds_handle(ds_id), Value::List(cols.into_iter().map(Value::String).collect())]);
            }
            Value::Assoc(map) => {
                let mut mapping: HashMap<String, String> = HashMap::new();
                for (newk, src) in map.iter() {
                    if let Value::String(s)|Value::Symbol(s) = src { mapping.insert(newk.clone(), s.clone()); }
                }
                if mapping.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Select".into())), args: vec![subj, spec_v] } }
                // Build SelectRename plan node
                let mut reg = ds_reg().lock().unwrap();
                let st = match reg.get(&ds_id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("Select".into())), args: vec![subj, spec_v] } };
                let new_id = next_ds_id();
                reg.insert(new_id, DatasetState { plan: Plan::SelectRename { input: Box::new(st.plan), mapping } });
                return ds_handle(new_id);
            }
            _ => {}
        }
    }
    // Assoc projection/rename
    match (&subj, &spec_v) {
        (Value::Assoc(m), Value::List(keys)) => {
            let mut out = HashMap::new();
            for k in keys {
                if let Value::String(s)|Value::Symbol(s) = k { if let Some(v)=m.get(s) { out.insert(s.clone(), v.clone()); } }
            }
            return Value::Assoc(out);
        }
        (Value::Assoc(m), Value::Assoc(map)) => {
            // rename: newName -> oldName
            let mut out = HashMap::new();
            for (newk, src) in map {
                if let Value::String(srcs)|Value::Symbol(srcs) = src { if let Some(v)=m.get(srcs) { out.insert(newk.clone(), v.clone()); } }
            }
            return Value::Assoc(out);
        }
        _ => {}
    }
    // List of Assoc: project per row
    if let Value::List(items) = subj.clone() {
        match &spec_v {
            Value::List(keys) => {
                if items.iter().all(|it| matches!(it, Value::Assoc(_))) && keys.iter().all(|k| matches!(k, Value::String(_)|Value::Symbol(_))) {
                    let mut out_rows: Vec<Value> = Vec::with_capacity(items.len());
                    for it in items.into_iter() {
                        if let Value::Assoc(m) = it {
                            let mut row = HashMap::new();
                            for k in keys {
                                if let Value::String(s)|Value::Symbol(s) = k { if let Some(v)=m.get(s) { row.insert(s.clone(), v.clone()); } }
                            }
                            out_rows.push(Value::Assoc(row));
                        }
                    }
                    return Value::List(out_rows);
                }
                // List indexing: keys are integers (1-based)
                if keys.iter().all(|k| matches!(k, Value::Integer(_))) && items.iter().all(|it| !matches!(it, Value::Assoc(_))) {
                    let idxs: Vec<usize> = keys.iter().filter_map(|k| if let Value::Integer(i)=k { Some((*i).max(1) as usize - 1) } else { None }).collect();
                    let out: Vec<Value> = idxs.into_iter().filter_map(|u| items.get(u).cloned()).collect();
                    return Value::List(out);
                }
            }
            Value::Assoc(rename) => {
                if items.iter().all(|it| matches!(it, Value::Assoc(_))) {
                    let mut out_rows: Vec<Value> = Vec::with_capacity(items.len());
                    for it in items.into_iter() {
                        if let Value::Assoc(m) = it {
                            let mut row = HashMap::new();
                            for (newk, src) in rename.iter() {
                                if let Value::String(srcs)|Value::Symbol(srcs) = src { if let Some(v)=m.get(srcs) { row.insert(newk.clone(), v.clone()); } }
                            }
                            out_rows.push(Value::Assoc(row));
                        }
                    }
                    return Value::List(out_rows);
                }
            }
            _ => {}
        }
    }
    // Fallback: not supported form
    Value::Expr { head: Box::new(Value::Symbol("Select".into())), args: vec![subj, spec_v] }
}
