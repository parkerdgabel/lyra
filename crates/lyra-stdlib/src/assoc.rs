use lyra_core::value::Value;
use lyra_runtime::{Evaluator};
use lyra_runtime::attrs::Attributes;
use std::collections::HashMap;
#[cfg(feature = "tools")] use crate::tools::add_specs;
#[cfg(feature = "tools")] use crate::tool_spec;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

pub fn register_assoc(ev: &mut Evaluator) {
    ev.register("Keys", keys as NativeFn, Attributes::empty());
    ev.register("Values", values as NativeFn, Attributes::empty());
    ev.register("Lookup", lookup as NativeFn, Attributes::empty());
    ev.register("AssociationMap", association_map as NativeFn, Attributes::empty());
    ev.register("AssociationMapKeys", association_map_keys as NativeFn, Attributes::empty());
    ev.register("AssociationMapKV", association_map_kv as NativeFn, Attributes::empty());
    ev.register("AssociationMapPairs", association_map_pairs as NativeFn, Attributes::empty());
    ev.register("Merge", merge as NativeFn, Attributes::empty());
    ev.register("KeySort", key_sort as NativeFn, Attributes::empty());
    ev.register("SortBy", sort_by as NativeFn, Attributes::HOLD_ALL);
    ev.register("GroupBy", group_by as NativeFn, Attributes::empty());
    // Immutable assoc utilities
    ev.register("AssocGet", assoc_get as NativeFn, Attributes::empty());
    ev.register("AssocContainsKeyQ", assoc_contains_key_q as NativeFn, Attributes::empty());
    ev.register("AssocSet", assoc_set as NativeFn, Attributes::empty());
    ev.register("AssocDelete", assoc_delete as NativeFn, Attributes::empty());
    ev.register("AssocSelect", assoc_select as NativeFn, Attributes::empty());
    ev.register("AssocDrop", assoc_drop as NativeFn, Attributes::empty());
    ev.register("AssocInvert", assoc_invert as NativeFn, Attributes::empty());
    ev.register("AssocRenameKeys", assoc_rename_keys as NativeFn, Attributes::empty());

    #[cfg(feature = "tools")]
    add_specs(vec![
        tool_spec!("Keys", summary: "List keys of an association", params: ["assoc"], tags: ["assoc"], output_schema: Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("array")))]))),
        tool_spec!("Values", summary: "List values of an association", params: ["assoc"], tags: ["assoc"], output_schema: Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("array")))]))),
        tool_spec!("Lookup", summary: "Lookup value from association", params: ["assoc","key","default"], tags: ["assoc"]),
        tool_spec!("Merge", summary: "Merge associations with optional combiner", params: ["args"], tags: ["assoc"], examples: [
            Value::Assoc(HashMap::from([
                ("args".to_string(), Value::Assoc(HashMap::from([(
                    "args".to_string(), Value::List(vec![
                        Value::Assoc(HashMap::from([(String::from("a"), Value::Integer(1))])),
                        Value::Assoc(HashMap::from([(String::from("a"), Value::Integer(2))])),
                    ]))
                ]))),
                ("result".to_string(), Value::Assoc(HashMap::from([(String::from("a"), Value::Integer(2))]))),
            ]))
        ]),
        tool_spec!("KeySort", summary: "Sort association by key", params: ["assoc"], tags: ["assoc"]),
        tool_spec!("SortBy", summary: "Sort association by derived key", params: ["keyFn","assoc"], tags: ["assoc"]),
        tool_spec!("GroupBy", summary: "Group list items by a key function", params: ["list","keyFn"], tags: ["list","assoc"]),
        tool_spec!("AssociationMap", summary: "Map values in an association", params: ["fn","assoc"], tags: ["assoc"]),
        tool_spec!("AssociationMapKeys", summary: "Map keys in an association", params: ["fn","assoc"], tags: ["assoc"]),
        tool_spec!("AssociationMapKV", summary: "Map key/value pairs in an association", params: ["fn","assoc"], tags: ["assoc"]),
        tool_spec!("AssociationMapPairs", summary: "Map to key/value pairs or assoc", params: ["fn","assoc"], tags: ["assoc"]),
    ]);
}

fn keys(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Keys".into())), args } }
    match ev.eval(args[0].clone()) {
        Value::Assoc(m) => Value::List(m.keys().cloned().map(Value::String).collect()),
        other => other,
    }
}

fn values(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Values".into())), args } }
    match ev.eval(args[0].clone()) {
        Value::Assoc(m) => Value::List(m.values().cloned().collect()),
        other => other,
    }
}

fn lookup(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [assoc, Value::String(k)] => match ev.eval(assoc.clone()) { Value::Assoc(m) => m.get(k).cloned().unwrap_or(Value::Symbol("Null".into())), other => other },
        [assoc, Value::String(k), default] => match ev.eval(assoc.clone()) { Value::Assoc(m) => m.get(k).cloned().unwrap_or(ev.eval(default.clone())), other => other },
        _ => Value::Expr { head: Box::new(Value::Symbol("Lookup".into())), args },
    }
}

fn association_map(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("AssociationMap".into())), args } }
    match ev.eval(args[1].clone()) {
        Value::Assoc(m) => {
            let mm = m.into_iter().map(|(k,v)| (k, ev.eval(Value::Expr { head: Box::new(args[0].clone()), args: vec![v] }))).collect();
            Value::Assoc(mm)
        }
        other => other,
    }
}

fn association_map_keys(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("AssociationMapKeys".into())), args } }
    match ev.eval(args[1].clone()) {
        Value::Assoc(m) => {
            let mut out = HashMap::new();
            for (k, v) in m.into_iter() {
                let newk_v = ev.eval(Value::Expr { head: Box::new(args[0].clone()), args: vec![Value::String(k.clone())] });
                let newk = match newk_v { Value::String(s)=>s, Value::Symbol(s)=>s, other=> lyra_core::pretty::format_value(&other) };
                out.insert(newk, ev.eval(v));
            }
            Value::Assoc(out)
        }
        other => other,
    }
}

fn association_map_kv(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("AssociationMapKV".into())), args } }
    match ev.eval(args[1].clone()) {
        Value::Assoc(m) => {
            let mut out = HashMap::new();
            for (k, v) in m.into_iter() {
                let newv = ev.eval(Value::Expr { head: Box::new(args[0].clone()), args: vec![Value::String(k.clone()), v] });
                out.insert(k, newv);
            }
            Value::Assoc(out)
        }
        other => other,
    }
}

fn association_map_pairs(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("AssociationMapPairs".into())), args } }
    match ev.eval(args[1].clone()) {
        Value::Assoc(m) => {
            let mut out = HashMap::new();
            for (k, v) in m.into_iter() {
                let res = ev.eval(Value::Expr { head: Box::new(args[0].clone()), args: vec![Value::String(k.clone()), v] });
                match res {
                    Value::List(mut pair) if pair.len()==2 => {
                        let newk = match pair.remove(0) { Value::String(s)=>s, Value::Symbol(s)=>s, other=> lyra_core::pretty::format_value(&other) };
                        let newv = pair.remove(0);
                        out.insert(newk, newv);
                    }
                    Value::Assoc(am) => {
                        let nk = am.get("key").cloned().unwrap_or(Value::String(k.clone()));
                        let nv = am.get("value").cloned().unwrap_or(Value::Symbol("Null".into()));
                        let newk = match nk { Value::String(s)=>s, Value::Symbol(s)=>s, other=> lyra_core::pretty::format_value(&other) };
                        out.insert(newk, nv);
                    }
                    other => { out.insert(k, other); }
                }
            }
            Value::Assoc(out)
        }
        other => other,
    }
}

fn merge(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Assoc(HashMap::new()) }
    // forms: Merge[listOfAssocs], Merge[a1, a2, ...], Merge[listOrArgs, combiner]
    let (vals, combiner) = if args.len()>=2 {
        if let Value::Expr { .. } | Value::Symbol(_) | Value::PureFunction { .. } = args[args.len()-1] {
            (args[..args.len()-1].to_vec(), Some(args[args.len()-1].clone()))
        } else { (args, None) }
    } else { (args, None) };
    let list: Vec<Value> = if vals.len()==1 { match ev.eval(vals[0].clone()) { Value::List(v)=>v, x=>vec![x] } } else { vals.into_iter().map(|a| ev.eval(a)).collect() };
    let mut groups: HashMap<String, Vec<Value>> = HashMap::new();
    for v in list {
        if let Value::Assoc(m) = v { for (k,val) in m { groups.entry(k).or_default().push(val); } }
    }
    let out = if let Some(f) = combiner {
        groups.into_iter().map(|(k, xs)| {
            let res = if xs.len()==1 { xs[0].clone() } else { ev.eval(Value::Expr { head: Box::new(f.clone()), args: vec![Value::List(xs)] }) };
            (k, res)
        }).collect()
    } else {
        groups.into_iter().map(|(k, mut xs)| (k, xs.pop().unwrap())).collect()
    };
    Value::Assoc(out)
}

fn key_sort(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("KeySort".into())), args } }
    match ev.eval(args[0].clone()) {
        Value::Assoc(m) => {
            let mut keys: Vec<String> = m.keys().cloned().collect();
            keys.sort();
            let mut out = HashMap::new();
            for k in keys { out.insert(k.clone(), m.get(&k).cloned().unwrap()); }
            Value::Assoc(out)
        }
        other => other,
    }
}

fn sort_by(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("SortBy".into())), args } }
    let f = args[0].clone();
    let subject = ev.eval(args[1].clone());
    match subject {
        Value::Assoc(m) => {
            let mut items: Vec<(String, Value)> = m.into_iter().collect();
            items.sort_by(|(ka, va), (kb, vb)| {
                let va_key = ev.eval(Value::Expr { head: Box::new(f.clone()), args: vec![Value::String(ka.clone()), va.clone()] });
                let vb_key = ev.eval(Value::Expr { head: Box::new(f.clone()), args: vec![Value::String(kb.clone()), vb.clone()] });
                lyra_runtime::eval::value_order_key(&va_key).cmp(&lyra_runtime::eval::value_order_key(&vb_key))
            });
            let mut out = HashMap::new();
            for (k,v) in items { out.insert(k,v); }
            Value::Assoc(out)
        }
        Value::List(items) => {
            let mut tuples: Vec<(String, Value)> = items.into_iter().map(|it| {
                let k = ev.eval(Value::Expr { head: Box::new(f.clone()), args: vec![it.clone()] });
                (lyra_runtime::eval::value_order_key(&k), ev.eval(it))
            }).collect();
            tuples.sort_by(|(ka, _), (kb, _)| ka.cmp(kb));
            Value::List(tuples.into_iter().map(|(_, v)| v).collect())
        }
        other => other,
    }
}

fn assoc_get(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [a, k] => match ev.eval(a.clone()) { Value::Assoc(m) => m.get(&match k { Value::String(s)|Value::Symbol(s)=>s.clone(), other=> lyra_core::pretty::format_value(&ev.eval(other.clone())) }).cloned().unwrap_or(Value::Symbol("Null".into())), other => other },
        [a, k, default] => match ev.eval(a.clone()) { Value::Assoc(m) => m.get(&match k { Value::String(s)|Value::Symbol(s)=>s.clone(), other=> lyra_core::pretty::format_value(&ev.eval(other.clone())) }).cloned().unwrap_or(ev.eval(default.clone())), other => other },
        _ => Value::Expr { head: Box::new(Value::Symbol("AssocGet".into())), args },
    }
}

fn assoc_contains_key_q(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("AssocContainsKeyQ".into())), args } }
    match ev.eval(args[0].clone()) { Value::Assoc(m) => {
        let k = match &args[1] { Value::String(s)|Value::Symbol(s)=>s.clone(), other=> lyra_core::pretty::format_value(&ev.eval(other.clone())) };
        Value::Boolean(m.contains_key(&k))
    } other => other }
}

fn assoc_set(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=3 { return Value::Expr { head: Box::new(Value::Symbol("AssocSet".into())), args } }
    match ev.eval(args[0].clone()) { Value::Assoc(mut m) => { let k = match &args[1] { Value::String(s)|Value::Symbol(s)=>s.clone(), other=> lyra_core::pretty::format_value(&ev.eval(other.clone())) }; let v = ev.eval(args[2].clone()); m.insert(k, v); Value::Assoc(m) } other => other }
}

fn assoc_delete(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("AssocDelete".into())), args } }
    match ev.eval(args[0].clone()) { Value::Assoc(mut m) => {
        match ev.eval(args[1].clone()) {
            Value::List(keys) => { for k in keys { if let Value::String(s)|Value::Symbol(s) = k { m.remove(&s); } }
            }
            Value::String(s) | Value::Symbol(s) => { m.remove(&s); }
            other => { let k = lyra_core::pretty::format_value(&other); m.remove(&k); }
        }
        Value::Assoc(m)
    } other => other }
}

fn assoc_select(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("AssocSelect".into())), args } }
    match (ev.eval(args[0].clone()), ev.eval(args[1].clone())) { (Value::Assoc(m), Value::List(keys)) => {
        let mut out = std::collections::HashMap::new();
        for k in keys { if let Value::String(s)|Value::Symbol(s) = k { if let Some(v)=m.get(&s) { out.insert(s.clone(), v.clone()); } } }
        Value::Assoc(out)
    } (other, _) => other }
}

fn assoc_drop(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("AssocDrop".into())), args } }
    match (ev.eval(args[0].clone()), ev.eval(args[1].clone())) { (Value::Assoc(mut m), Value::List(keys)) => {
        for k in keys { if let Value::String(s)|Value::Symbol(s) = k { m.remove(&s); } }
        Value::Assoc(m)
    } (other, _) => other }
}

fn assoc_invert(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("AssocInvert".into())), args } }
    match ev.eval(args[0].clone()) { Value::Assoc(m) => {
        let mut out = std::collections::HashMap::new();
        for (k,v) in m { let ks = k; let vs = lyra_core::pretty::format_value(&v); out.insert(vs, Value::String(ks)); }
        Value::Assoc(out)
    } other => other }
}

fn assoc_rename_keys(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("AssocRenameKeys".into())), args } }
    match (ev.eval(args[0].clone()), ev.eval(args[1].clone())) { (Value::Assoc(m), Value::Assoc(mapping)) => {
        let mut out = std::collections::HashMap::new();
        for (k,v) in m { let newk = mapping.get(&k).and_then(|vv| match vv { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None }).unwrap_or(k); out.insert(newk, v); }
        Value::Assoc(out)
    } (other, _) => other }
}


fn group_by(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("GroupBy".into())), args } }
    match ev.eval(args[0].clone()) {
        Value::List(items) => {
            let mut groups: HashMap<String, Vec<Value>> = HashMap::new();
            for it in items {
                let key_v = ev.eval(Value::Expr { head: Box::new(args[1].clone()), args: vec![it.clone()] });
                let key = match key_v { Value::String(s)=>s, Value::Symbol(s)=>s, other=> format!("{}", lyra_core::pretty::format_value(&other)) };
                groups.entry(key).or_default().push(ev.eval(it));
            }
            Value::Assoc(groups.into_iter().map(|(k,vs)|(k, Value::List(vs))).collect())
        }
        other => other,
    }
}
