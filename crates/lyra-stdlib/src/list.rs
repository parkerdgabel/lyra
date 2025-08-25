use lyra_core::value::Value;
use lyra_runtime::{Evaluator};
use lyra_runtime::attrs::Attributes;
#[cfg(feature = "tools")] use crate::tools::add_specs;
#[cfg(feature = "tools")] use crate::tool_spec;
#[cfg(feature = "tools")] use std::collections::HashMap;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

pub fn register_list(ev: &mut Evaluator) {
    ev.register("Length", length as NativeFn, Attributes::empty());
    ev.register("Range", range as NativeFn, Attributes::empty());
    ev.register("Join", join as NativeFn, Attributes::empty());
    ev.register("Reverse", reverse as NativeFn, Attributes::empty());
    ev.register("Total", total as NativeFn, Attributes::empty());
    ev.register("Flatten", flatten as NativeFn, Attributes::empty());
    ev.register("Partition", partition as NativeFn, Attributes::empty());
    ev.register("Transpose", transpose as NativeFn, Attributes::empty());
    ev.register("Map", map_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Filter", filter as NativeFn, Attributes::HOLD_ALL);
    ev.register("Reject", reject as NativeFn, Attributes::HOLD_ALL);
    ev.register("Any", any_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("All", all_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Find", find_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Position", position_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Take", take_fn as NativeFn, Attributes::empty());
    ev.register("Drop", drop_fn as NativeFn, Attributes::empty());
    ev.register("TakeWhile", take_while_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("DropWhile", drop_while_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Zip", zip_fn as NativeFn, Attributes::empty());
    ev.register("Unzip", unzip_fn as NativeFn, Attributes::empty());
    ev.register("Sort", sort_fn as NativeFn, Attributes::empty());
    ev.register("Unique", unique_fn as NativeFn, Attributes::empty());
    ev.register("Tally", tally_fn as NativeFn, Attributes::empty());
    ev.register("CountBy", count_by_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Reduce", reduce_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Scan", scan_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("MapIndexed", map_indexed_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Slice", slice_list_fn as NativeFn, Attributes::empty());
    ev.register("PackedArray", packed_array as NativeFn, Attributes::HOLD_ALL);
    ev.register("PackedToList", packed_to_list as NativeFn, Attributes::HOLD_ALL);
    ev.register("PackedShape", packed_shape as NativeFn, Attributes::empty());
    ev.register("Part", part as NativeFn, Attributes::empty());

    #[cfg(feature = "tools")]
    add_specs(vec![
        tool_spec!("Length", summary: "Length of list or string", params: ["x"], tags: ["list","string"], output_schema: Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("integer")))])), examples: [
            Value::Assoc(HashMap::from([
                ("args".to_string(), Value::Assoc(HashMap::from([("x".to_string(), Value::List(vec![Value::Integer(1), Value::Integer(2)]))]))),
                ("result".to_string(), Value::Integer(2)),
            ]))
        ]),
        tool_spec!("Join", summary: "Concatenate two lists", params: ["a","b"], tags: ["list"], input_schema: Value::Assoc(HashMap::from([
            ("type".to_string(), Value::String("object".into())),
            ("properties".to_string(), Value::Assoc(HashMap::from([
                ("a".to_string(), Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("array")))]))),
                ("b".to_string(), Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("array")))]))),
            ]))),
            ("required".to_string(), Value::List(vec![Value::String("a".into()), Value::String("b".into())])),
        ])), output_schema: Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("array")))])), examples: [
            Value::Assoc(HashMap::from([
                ("args".to_string(), Value::Assoc(HashMap::from([
                    ("a".to_string(), Value::List(vec![Value::Integer(1)])),
                    ("b".to_string(), Value::List(vec![Value::Integer(2)])),
                ]))),
                ("result".to_string(), Value::List(vec![Value::Integer(1), Value::Integer(2)])),
            ]))
        ]),
        tool_spec!("Total", summary: "Sum elements in a list", params: ["list"], tags: ["list","math"], input_schema: Value::Assoc(HashMap::from([
            ("type".to_string(), Value::String("object".into())),
            ("properties".to_string(), Value::Assoc(HashMap::from([(String::from("list"), Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("array")))])))]))),
            ("required".to_string(), Value::List(vec![Value::String("list".into())])),
        ])), output_schema: Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("number")))]))),
        tool_spec!("Part", summary: "Index into list/assoc", params: ["subject","index"], tags: ["list","assoc"]),
        tool_spec!("Range", summary: "Create a numeric range", params: ["a","b"], tags: ["list","math"], input_schema: Value::Assoc(HashMap::from([
            ("type".to_string(), Value::String("object".into())),
            ("properties".to_string(), Value::Assoc(HashMap::from([
                ("a".to_string(), Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("integer")))]))),
                ("b".to_string(), Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("integer")))]))),
            ]))),
        ]))),
        tool_spec!("Flatten", summary: "Flatten list by a level", params: ["list","levels"], tags: ["list"], input_schema: Value::Assoc(HashMap::from([
            ("type".to_string(), Value::String("object".into())),
            ("properties".to_string(), Value::Assoc(HashMap::from([
                ("list".to_string(), Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("array")))]))),
                ("levels".to_string(), Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("integer")))]))),
            ]))),
        ]))),
        tool_spec!("Partition", summary: "Partition list into fixed-size chunks", params: ["list","n","step"], tags: ["list"], input_schema: Value::Assoc(HashMap::from([
            ("type".to_string(), Value::String("object".into())),
            ("properties".to_string(), Value::Assoc(HashMap::from([
                ("list".to_string(), Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("array")))]))),
                ("n".to_string(), Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("integer")))]))),
                ("step".to_string(), Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("integer")))]))),
            ]))),
            ("required".to_string(), Value::List(vec![Value::String("list".into()), Value::String("n".into())])),
        ]))),
        tool_spec!("Transpose", summary: "Transpose a list of lists", params: ["rows"], tags: ["list"], input_schema: Value::Assoc(HashMap::from([
            ("type".to_string(), Value::String("object".into())),
            ("properties".to_string(), Value::Assoc(HashMap::from([(String::from("rows"), Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("array")))])))]))),
            ("required".to_string(), Value::List(vec![Value::String("rows".into())])),
        ]))),
        tool_spec!("Filter", summary: "Keep elements where pred[x] is True", params: ["pred","list"], tags: ["list","filter"]),
        tool_spec!("Reject", summary: "Drop elements where pred[x] is True", params: ["pred","list"], tags: ["list","filter"]),
        tool_spec!("Any", summary: "True if any matches (optionally using pred)", params: ["list","pred?"], tags: ["list","logic"]),
        tool_spec!("All", summary: "True if all match (optionally using pred)", params: ["list","pred?"], tags: ["list","logic"]),
        tool_spec!("Find", summary: "First element where pred[x] is True", params: ["pred","list"], tags: ["list","search"]),
        tool_spec!("Position", summary: "1-based index of first match", params: ["pred","list"], tags: ["list","search"], output_schema: Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("integer")))]))),
        tool_spec!("Take", summary: "Take first n (or last n if negative)", params: ["list","n"], tags: ["list","slice"]),
        tool_spec!("Drop", summary: "Drop first n (or last n if negative)", params: ["list","n"], tags: ["list","slice"]),
        tool_spec!("TakeWhile", summary: "Take while pred[x] stays True", params: ["pred","list"], tags: ["list","slice"]),
        tool_spec!("DropWhile", summary: "Drop while pred[x] stays True", params: ["pred","list"], tags: ["list","slice"]),
        tool_spec!("Zip", summary: "Zip two lists into pairs", params: ["a","b"], tags: ["list","zip"]),
        tool_spec!("Unzip", summary: "Unzip list of pairs into two lists", params: ["pairs"], tags: ["list","zip"]),
        tool_spec!("Sort", summary: "Sort a list by value", params: ["list"], tags: ["list","sort"]),
        tool_spec!("Unique", summary: "Stable deduplicate a list", params: ["list"], tags: ["list","set"]),
        tool_spec!("Tally", summary: "Counts by value (assoc)", params: ["list"], tags: ["list","aggregate"]),
        tool_spec!("CountBy", summary: "Counts by key function (assoc)", params: ["f","list"], tags: ["list","aggregate"]),
        tool_spec!("Reduce", summary: "Fold list with function", params: ["f","init?","list"], tags: ["list","fold"]),
        tool_spec!("Scan", summary: "Prefix scan with function", params: ["f","init?","list"], tags: ["list","fold"]),
        tool_spec!("MapIndexed", summary: "Map with index (1-based)", params: ["f","list"], tags: ["list","map"]),
        tool_spec!("Slice", summary: "Slice list by start and length", params: ["list","start","len?"], tags: ["list","slice"]),
    ]);
}

fn length(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Length".into())), args } }
    match ev.eval(args[0].clone()) {
        Value::List(v) => Value::Integer(v.len() as i64),
        Value::String(s) => Value::Integer(s.chars().count() as i64),
        _ => Value::Integer(0),
    }
}

fn range(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] => Value::List((1..=*n).map(Value::Integer).collect()),
        [Value::Integer(a), Value::Integer(b)] => {
            let (start, end) = (*a, *b);
            let mut out = Vec::new();
            if start <= end { for i in start..=end { out.push(Value::Integer(i)); } }
            else { for i in (end..=start).rev() { out.push(Value::Integer(i)); } }
            Value::List(out)
        }
        [a, b] => { let a1 = ev.eval(a.clone()); let b1 = ev.eval(b.clone()); range(ev, vec![a1, b1]) },
        _ => Value::Expr { head: Box::new(Value::Symbol("Range".into())), args },
    }
}

fn join(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::List(a), Value::List(b)] => {
            let mut out = a.clone();
            out.extend(b.clone());
            Value::List(out)
        }
        [a, b] => { let a1 = ev.eval(a.clone()); let b1 = ev.eval(b.clone()); join(ev, vec![a1, b1]) },
        _ => Value::Expr { head: Box::new(Value::Symbol("Join".into())), args },
    }
}

fn reverse(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Reverse".into())), args } }
    match ev.eval(args[0].clone()) {
        Value::List(mut v) => { v.reverse(); Value::List(v) }
        other => other,
    }
}

fn total(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v] => { let vv = ev.eval(v.clone()); sum_list(ev, vv) },
        [v, Value::Symbol(s)] if s == "Infinity" => { let vv = ev.eval(v.clone()); sum_all(ev, vv) },
        [v, Value::Integer(n)] => {
            let mut val = ev.eval(v.clone());
            let mut lvl = *n;
            while lvl > 0 { val = sum_list(ev, val); lvl -= 1; }
            val
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("Total".into())), args },
    }
}

fn sum_list(ev: &mut Evaluator, v: Value) -> Value {
    match v {
        Value::PackedArray { data, .. } => {
            let sum: f64 = data.iter().copied().sum();
            Value::Real(sum)
        }
        Value::List(items) => {
            let mut acc_i: Option<i64> = Some(0);
            let mut acc_f: Option<f64> = Some(0.0);
            for it in items { match ev.eval(it) {
                Value::Integer(n) => { if let Some(i)=acc_i { acc_i=Some(i+n); } else if let Some(f)=acc_f { acc_f=Some(f+n as f64);} }
                Value::Real(x) => { acc_i=None; if let Some(f)=acc_f { acc_f=Some(f+x);} }
                other => { let inner = total(ev, vec![other]); match inner { Value::Integer(n)=>{ if let Some(i)=acc_i { acc_i=Some(i+n);} else if let Some(f)=acc_f { acc_f=Some(f+n as f64);} } Value::Real(x)=>{ acc_i=None; if let Some(f)=acc_f { acc_f=Some(f+x);} } _=>{} } }
            }}
            if let Some(i)=acc_i { Value::Integer(i) } else { Value::Real(acc_f.unwrap_or(0.0)) }
        }
        other => other,
    }
}

fn sum_all(ev: &mut Evaluator, v: Value) -> Value {
    match v {
        Value::PackedArray { data, .. } => {
            let sum: f64 = data.iter().copied().sum();
            Value::Real(sum)
        }
        Value::List(items) => {
            let mut acc = Value::Integer(0);
            for it in items { let itv = ev.eval(it); let s = sum_all(ev, itv); acc = add_values(acc, s); }
            acc
        }
        Value::Integer(_) | Value::Real(_) => v,
        other => other,
    }
}

fn add_values(a: Value, b: Value) -> Value {
    match (a, b) {
        (Value::Integer(x), Value::Integer(y)) => Value::Integer(x + y),
        (Value::Integer(x), Value::Real(y)) => Value::Real((x as f64) + y),
        (Value::Real(x), Value::Integer(y)) => Value::Real(x + (y as f64)),
        (Value::Real(x), Value::Real(y)) => Value::Real(x + y),
        (x, y) => Value::Expr { head: Box::new(Value::Symbol("Plus".into())), args: vec![x, y] },
    }
}

// Filter[pred, list]
fn filter(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("Filter".into())), args } }
    let pred = args[0].clone();
    match ev.eval(args[1].clone()) {
        Value::List(items) => {
            let mut out = Vec::new();
            for it in items {
                let ok = matches!(ev.eval(Value::Expr { head: Box::new(pred.clone()), args: vec![it.clone()] }), Value::Boolean(true));
                if ok { out.push(ev.eval(it)); }
            }
            Value::List(out)
        }
        other => other,
    }
}

// Reject[pred, list]
fn reject(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("Reject".into())), args } }
    let pred = args[0].clone();
    match ev.eval(args[1].clone()) {
        Value::List(items) => {
            let mut out = Vec::new();
            for it in items {
                let ok = !matches!(ev.eval(Value::Expr { head: Box::new(pred.clone()), args: vec![it.clone()] }), Value::Boolean(true));
                if ok { out.push(ev.eval(it)); }
            }
            Value::List(out)
        }
        other => other,
    }
}

// Any[list] or Any[pred, list]
fn any_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [list] => match ev.eval(list.clone()) {
            Value::List(items) => Value::Boolean(items.into_iter().any(|it| matches!(ev.eval(it), Value::Boolean(true)))),
            other => other,
        },
        [pred, list] => match ev.eval(list.clone()) {
            Value::List(items) => {
                let res = items.into_iter().any(|it| matches!(ev.eval(Value::Expr { head: Box::new(pred.clone()), args: vec![it] }), Value::Boolean(true)));
                Value::Boolean(res)
            }
            other => other,
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("Any".into())), args },
    }
}

// All[list] or All[pred, list]
fn all_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [list] => match ev.eval(list.clone()) {
            Value::List(items) => Value::Boolean(items.into_iter().all(|it| matches!(ev.eval(it), Value::Boolean(true)))),
            other => other,
        },
        [pred, list] => match ev.eval(list.clone()) {
            Value::List(items) => {
                let res = items.into_iter().all(|it| matches!(ev.eval(Value::Expr { head: Box::new(pred.clone()), args: vec![it] }), Value::Boolean(true)));
                Value::Boolean(res)
            }
            other => other,
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("All".into())), args },
    }
}

// Find[pred, list]
fn find_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("Find".into())), args } }
    let pred = args[0].clone();
    match ev.eval(args[1].clone()) {
        Value::List(items) => {
            for it in items { if matches!(ev.eval(Value::Expr { head: Box::new(pred.clone()), args: vec![it.clone()] }), Value::Boolean(true)) { return ev.eval(it); } }
            Value::Symbol("Null".into())
        }
        other => other,
    }
}

// Position[pred, list]: 1-based index, -1 if not found
fn position_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("Position".into())), args } }
    let pred = args[0].clone();
    match ev.eval(args[1].clone()) {
        Value::List(items) => {
            for (i, it) in items.into_iter().enumerate() {
                if matches!(ev.eval(Value::Expr { head: Box::new(pred.clone()), args: vec![it] }), Value::Boolean(true)) { return Value::Integer((i as i64) + 1); }
            }
            Value::Integer(-1)
        }
        other => other,
    }
}

// Take[list, n] (n<0 => last |n|)
fn take_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("Take".into())), args } }
    let n = match ev.eval(args[1].clone()) { Value::Integer(k) => k, other => { return Value::Expr { head: Box::new(Value::Symbol("Take".into())), args: vec![ev.eval(args[0].clone()), other] } } };
    match ev.eval(args[0].clone()) {
        Value::List(items) => {
            let len = items.len() as i64;
            let k = if n >= 0 { n.min(len).max(0) } else { (-n).min(len).max(0) } as usize;
            let slice = if n >= 0 { items.into_iter().take(k).collect() } else { items.into_iter().rev().take(k).collect::<Vec<_>>().into_iter().rev().collect() };
            Value::List(slice)
        }
        other => other,
    }
}

// Drop[list, n] (n<0 => drop last |n|)
fn drop_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("Drop".into())), args } }
    let n = match ev.eval(args[1].clone()) { Value::Integer(k) => k, other => { return Value::Expr { head: Box::new(Value::Symbol("Drop".into())), args: vec![ev.eval(args[0].clone()), other] } } };
    match ev.eval(args[0].clone()) {
        Value::List(items) => {
            let len = items.len() as i64;
            let k = if n >= 0 { n.min(len).max(0) } else { (-n).min(len).max(0) } as usize;
            let slice: Vec<Value> = if n >= 0 { items.into_iter().skip(k).collect() } else { items.into_iter().take((len as usize).saturating_sub(k)).collect() };
            Value::List(slice)
        }
        other => other,
    }
}

// Sort[list] using value_order_key on evaluated items
fn sort_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::Expr { head: Box::new(Value::Symbol("Sort".into())), args } }
    match ev.eval(args[0].clone()) {
        Value::List(items) => {
            let mut evald: Vec<Value> = items.into_iter().map(|x| ev.eval(x)).collect();
            evald.sort_by(|a, b| lyra_runtime::eval::value_order_key(a).cmp(&lyra_runtime::eval::value_order_key(b)));
            Value::List(evald)
        }
        other => other,
    }
}

// SortBy[f, list]

// Unique[list] — stable dedupe using value_order_key of evaluated items
fn unique_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::Expr { head: Box::new(Value::Symbol("Unique".into())), args } }
    match ev.eval(args[0].clone()) {
        Value::List(items) => {
            use std::collections::HashSet;
            let mut seen: HashSet<String> = HashSet::new();
            let mut out = Vec::new();
            for it in items { let v = ev.eval(it); let k = lyra_runtime::eval::value_order_key(&v); if seen.insert(k) { out.push(v); } }
            Value::List(out)
        }
        other => other,
    }
}

// Tally[list] — assoc of formatted value -> count
fn tally_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::Expr { head: Box::new(Value::Symbol("Tally".into())), args } }
    match ev.eval(args[0].clone()) {
        Value::List(items) => {
            use std::collections::HashMap;
            let mut counts: HashMap<String, i64> = HashMap::new();
            for it in items { let v = ev.eval(it); let k = lyra_core::pretty::format_value(&v); *counts.entry(k).or_insert(0) += 1; }
            Value::Assoc(counts.into_iter().map(|(k,c)|(k, Value::Integer(c))).collect())
        }
        other => other,
    }
}

// CountBy[f, list] — assoc of derived key -> count
fn count_by_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("CountBy".into())), args } }
    match ev.eval(args[1].clone()) {
        Value::List(items) => {
            use std::collections::HashMap;
            let mut counts: HashMap<String, i64> = HashMap::new();
            for it in items {
                let k_v = ev.eval(Value::Expr { head: Box::new(args[0].clone()), args: vec![it] });
                let k = match k_v { Value::String(s)=>s, Value::Symbol(s)=>s, other=> lyra_core::pretty::format_value(&other) };
                *counts.entry(k).or_insert(0) += 1;
            }
            Value::Assoc(counts.into_iter().map(|(k,c)|(k, Value::Integer(c))).collect())
        }
        other => other,
    }
}

// TakeWhile[pred, list]
fn take_while_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("TakeWhile".into())), args } }
    let pred = args[0].clone();
    match ev.eval(args[1].clone()) {
        Value::List(items) => {
            let mut out = Vec::new();
            for it in items {
                if matches!(ev.eval(Value::Expr { head: Box::new(pred.clone()), args: vec![it.clone()] }), Value::Boolean(true)) {
                    out.push(ev.eval(it));
                } else { break; }
            }
            Value::List(out)
        }
        other => other,
    }
}

// DropWhile[pred, list]
fn drop_while_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("DropWhile".into())), args } }
    let pred = args[0].clone();
    match ev.eval(args[1].clone()) {
        Value::List(items) => {
            let mut skipping = true;
            let mut out = Vec::new();
            for it in items {
                if skipping {
                    if matches!(ev.eval(Value::Expr { head: Box::new(pred.clone()), args: vec![it.clone()] }), Value::Boolean(true)) {
                        continue;
                    } else { skipping = false; out.push(ev.eval(it)); }
                } else { out.push(ev.eval(it)); }
            }
            Value::List(out)
        }
        other => other,
    }
}

// Zip[a, b]
fn zip_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("Zip".into())), args } }
    let a = ev.eval(args[0].clone());
    let b = ev.eval(args[1].clone());
    match (a, b) {
        (Value::List(la), Value::List(lb)) => {
            let n = std::cmp::min(la.len(), lb.len());
            let mut out = Vec::with_capacity(n);
            for i in 0..n { out.push(Value::List(vec![la[i].clone(), lb[i].clone()])); }
            Value::List(out)
        }
        (aa, bb) => Value::Expr { head: Box::new(Value::Symbol("Zip".into())), args: vec![aa, bb] }
    }
}

// Unzip[pairs]
fn unzip_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::Expr { head: Box::new(Value::Symbol("Unzip".into())), args } }
    match ev.eval(args[0].clone()) {
        Value::List(items) => {
            let mut a = Vec::new();
            let mut b = Vec::new();
            for it in items {
                match ev.eval(it) {
                    Value::List(mut pair) if pair.len() == 2 => {
                        b.push(pair.pop().unwrap());
                        a.push(pair.pop().unwrap());
                    }
                    other => { a.push(other); b.push(Value::Symbol("Null".into())); }
                }
            }
            Value::List(vec![Value::List(a), Value::List(b)])
        }
        other => other,
    }
}

// Reduce[f, list] or Reduce[f, init, list]
fn reduce_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [f, list] => match ev.eval(list.clone()) {
            Value::List(items) => {
                let mut iter = items.into_iter();
                let first = match iter.next() { Some(x) => ev.eval(x), None => return Value::Symbol("Null".into()) };
                let mut acc = first;
                for it in iter {
                    let v_it = ev.eval(it);
                    let expr = Value::Expr { head: Box::new(f.clone()), args: vec![acc, v_it] };
                    acc = ev.eval(expr);
                }
                acc
            }
            other => other,
        },
        [f, init, list] => match ev.eval(list.clone()) {
            Value::List(items) => {
                let mut acc = ev.eval(init.clone());
                for it in items {
                    let v_it = ev.eval(it);
                    let expr = Value::Expr { head: Box::new(f.clone()), args: vec![acc, v_it] };
                    acc = ev.eval(expr);
                }
                acc
            }
            other => other,
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("Reduce".into())), args },
    }
}

// Scan[f, list] or Scan[f, init, list]
fn scan_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [f, list] => match ev.eval(list.clone()) {
            Value::List(items) => {
                let mut out = Vec::new();
                let mut iter = items.into_iter();
                if let Some(x) = iter.next() {
                    let mut acc = ev.eval(x);
                    out.push(acc.clone());
                    for it in iter {
                        let v_it = ev.eval(it);
                        let expr = Value::Expr { head: Box::new(f.clone()), args: vec![acc, v_it] };
                        acc = ev.eval(expr);
                        out.push(acc.clone());
                    }
                }
                Value::List(out)
            }
            other => other,
        },
        [f, init, list] => match ev.eval(list.clone()) {
            Value::List(items) => {
                let mut out = Vec::new();
                let mut acc = ev.eval(init.clone());
                for it in items {
                    let v_it = ev.eval(it);
                    let expr = Value::Expr { head: Box::new(f.clone()), args: vec![acc, v_it] };
                    acc = ev.eval(expr);
                    out.push(acc.clone());
                }
                Value::List(out)
            }
            other => other,
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("Scan".into())), args },
    }
}

// MapIndexed[f, list] — 1-based index
fn map_indexed_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("MapIndexed".into())), args } }
    match ev.eval(args[1].clone()) {
        Value::List(items) => {
            let out: Vec<Value> = items.into_iter().enumerate().map(|(i, it)| {
                let v_it = ev.eval(it);
                let call = Value::Expr { head: Box::new(args[0].clone()), args: vec![Value::Integer((i as i64)+1), v_it] };
                ev.eval(call)
            }).collect();
            Value::List(out)
        }
        other => other,
    }
}

// Slice[list, start, len?] — negative start counts from end
fn slice_list_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [list, start] => match (ev.eval(list.clone()), ev.eval(start.clone())) {
            (Value::List(items), Value::Integer(s)) => slice_apply(items, s, None),
            (a, b) => Value::Expr { head: Box::new(Value::Symbol("Slice".into())), args: vec![a, b] }
        },
        [list, start, len] => match (ev.eval(list.clone()), ev.eval(start.clone()), ev.eval(len.clone())) {
            (Value::List(items), Value::Integer(s), Value::Integer(l)) => slice_apply(items, s, Some(l)),
            (a, b, c) => Value::Expr { head: Box::new(Value::Symbol("Slice".into())), args: vec![a, b, c] }
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("Slice".into())), args },
    }
}

fn slice_apply(items: Vec<Value>, start: i64, len: Option<i64>) -> Value {
    let n = items.len() as i64;
    let mut s = if start >= 0 { start } else { n + start };
    if s < 0 { s = 0; }
    if s > n { s = n; }
    let e = if let Some(l) = len { if l <= 0 { s } else { (s + l).min(n) } } else { n };
    let s_usize = s as usize; let e_usize = e as usize;
    Value::List(items.into_iter().skip(s_usize).take(e_usize.saturating_sub(s_usize)).collect())
}

fn flatten(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v] => flatten_once(ev.eval(v.clone())),
        [v, Value::Integer(n)] => {
            let mut res = ev.eval(v.clone());
            for _ in 0..*n { res = flatten_once(res); }
            res
        }
        [v, Value::Symbol(s)] if s=="Infinity" => {
            let mut res = ev.eval(v.clone());
            loop { let next = flatten_once(res.clone()); if next==res { break; } res = next; }
            res
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("Flatten".into())), args },
    }
}

fn flatten_once(v: Value) -> Value {
    match v {
        Value::List(items) => Value::List(items.into_iter().flat_map(|x| match x { Value::List(inner)=>inner, other=>vec![other] }).collect()),
        other => other,
    }
}

// Partition[list, n], Partition[list, n, step]
fn partition(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v, Value::Integer(n)] => do_partition(ev.eval(v.clone()), *n as usize, *n as usize),
        [v, Value::Integer(n), Value::Integer(step)] => do_partition(ev.eval(v.clone()), *n as usize, *step as usize),
        _ => Value::Expr { head: Box::new(Value::Symbol("Partition".into())), args },
    }
}

fn do_partition(v: Value, n: usize, step: usize) -> Value {
    match v {
        Value::List(items) => {
            let mut out: Vec<Value> = Vec::new();
            let mut i = 0usize;
            while i < items.len() {
                let end = (i + n).min(items.len());
                out.push(Value::List(items[i..end].to_vec()));
                i += step;
            }
            Value::List(out)
        }
        other => other,
    }
}

fn transpose(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Transpose".into())), args } }
    match ev.eval(args[0].clone()) {
        Value::List(rows) => {
            if rows.is_empty() { return Value::List(vec![]) }
            let cols = match &rows[0] { Value::List(v)=> v.len(), _=>0 };
            let mut out: Vec<Value> = Vec::with_capacity(cols);
            for c in 0..cols {
                let mut col: Vec<Value> = Vec::with_capacity(rows.len());
                for r in &rows { if let Value::List(rv) = r { col.push(rv.get(c).cloned().unwrap_or(Value::Symbol("Null".into()))); } }
                out.push(Value::List(col));
            }
            Value::List(out)
        }
        other => other,
    }
}

fn apply_fn(ev: &mut Evaluator, f: &Value, arg: Value) -> Value {
    match f {
        Value::PureFunction { .. } | Value::Symbol(_) | Value::Expr { .. } => {
            let call = Value::Expr { head: Box::new(f.clone()), args: vec![arg] };
            ev.eval(call)
        }
        other => Value::Expr { head: Box::new(other.clone()), args: vec![arg] },
    }
}

fn map_packed(ev: &mut Evaluator, f: &Value, shape: Vec<usize>, data: Vec<f64>) -> Value {
    let mut out: Vec<f64> = Vec::with_capacity(data.len());
    for x in data.into_iter() {
        let y = apply_fn(ev, f, Value::Real(x));
        match y {
            Value::Integer(n) => out.push(n as f64),
            Value::Real(r) => out.push(r),
            Value::Rational { num, den } if den != 0 => out.push((num as f64)/(den as f64)),
            Value::BigReal(s) => if let Ok(r)=s.parse::<f64>() { out.push(r) } else { return Value::Expr { head: Box::new(Value::Symbol("Map".into())), args: vec![f.clone(), Value::PackedArray { shape, data: out }] } },
            other => {
                // Fallback: map over unpacked list
                let list = packed_to_list(ev, vec![Value::PackedArray { shape: shape.clone(), data: out }]);
                return map_list(ev, f, list);
            }
        }
    }
    Value::PackedArray { shape, data: out }
}

fn map_list(ev: &mut Evaluator, f: &Value, v: Value) -> Value {
    match v {
        Value::List(items) => Value::List(items.into_iter().map(|it| apply_fn(ev, f, it)).collect()),
        other => apply_fn(ev, f, other),
    }
}

fn map_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("Map".into())), args } }
    let f = args[0].clone();
    let subj = ev.eval(args[1].clone());
    match subj {
        Value::PackedArray { shape, data } => map_packed(ev, &f, shape, data),
        other => map_list(ev, &f, other),
    }
}

fn infer_shape(v: &Value) -> Option<Vec<usize>> {
    match v {
        Value::List(items) => {
            if items.is_empty() { return Some(vec![0]); }
            let mut shape = infer_shape(&items[0])?;
            for it in items.iter().skip(1) {
                let shp = infer_shape(it)?;
                if shp != shape { return None; }
            }
            let mut out = vec![items.len()];
            out.extend(shape.drain(..));
            Some(out)
        }
        Value::Integer(_) | Value::Real(_) | Value::Rational{..} | Value::BigReal(_) => Some(vec![]),
        _ => None,
    }
}

fn flatten_numeric_rowmajor(v: &Value, out: &mut Vec<f64>) -> bool {
    match v {
        Value::List(items) => { for it in items { if !flatten_numeric_rowmajor(it, out) { return false; } } true }
        Value::Integer(n) => { out.push(*n as f64); true }
        Value::Real(x) => { out.push(*x); true }
        Value::Rational { num, den } => { if *den==0 { return false } out.push((*num as f64)/(*den as f64)); true }
        Value::BigReal(s) => { if let Ok(x)=s.parse::<f64>() { out.push(x); true } else { false } }
        _ => false,
    }
}

fn build_list_from_flat(shape: &[usize], data: &[f64], idx: &mut usize) -> Value {
    if shape.is_empty() { let x = data[*idx]; *idx += 1; Value::Real(x) }
    else {
        let dim = shape[0];
        let mut items: Vec<Value> = Vec::with_capacity(dim);
        for _ in 0..dim { items.push(build_list_from_flat(&shape[1..], data, idx)); }
        Value::List(items)
    }
}

fn packed_array(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("PackedArray".into())), args } }
    let v = ev.eval(args[0].clone());
    if let Some(shape) = infer_shape(&v) {
        let mut flat: Vec<f64> = Vec::new();
        if !flatten_numeric_rowmajor(&v, &mut flat) { return Value::Expr { head: Box::new(Value::Symbol("PackedArray".into())), args: vec![v] } }
        return Value::PackedArray { shape, data: flat };
    }
    Value::Expr { head: Box::new(Value::Symbol("PackedArray".into())), args: vec![v] }
}

fn packed_to_list(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("PackedToList".into())), args } }
    match ev.eval(args[0].clone()) {
        Value::PackedArray { shape, data } => { let mut idx=0usize; build_list_from_flat(&shape, &data, &mut idx) }
        other => Value::Expr { head: Box::new(Value::Symbol("PackedToList".into())), args: vec![other] },
    }
}

fn packed_shape(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("PackedShape".into())), args } }
    match ev.eval(args[0].clone()) {
        Value::PackedArray { shape, .. } => Value::List(shape.into_iter().map(|d| Value::Integer(d as i64)).collect()),
        other => Value::Expr { head: Box::new(Value::Symbol("PackedShape".into())), args: vec![other] },
    }
}

// Part[list, i] and Part[assoc, "k"]
fn part(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v, Value::Integer(i)] => match ev.eval(v.clone()) {
            Value::List(items) => {
                let idx = *i;
                if idx <= 0 { return Value::Symbol("Null".into()); }
                let u = (idx as usize).saturating_sub(1);
                items.get(u).cloned().unwrap_or(Value::Symbol("Null".into()))
            }
            other => other,
        },
        [v, Value::String(k)] => match ev.eval(v.clone()) {
            Value::Assoc(m) => m.get(k).cloned().unwrap_or(Value::Symbol("Null".into())),
            other => other,
        },
        [a, b] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            part(ev, vec![aa, bb])
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("Part".into())), args },
    }
}
