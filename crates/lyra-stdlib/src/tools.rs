use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_runtime::attrs::Attributes;
use std::collections::HashMap;
use std::sync::{OnceLock, Mutex};

// Minimal in-memory tool registry storing self-describing specs as Assoc Values
static TOOL_REG: OnceLock<Mutex<HashMap<String, Value>>> = OnceLock::new();

fn tool_reg() -> &'static Mutex<HashMap<String, Value>> {
    TOOL_REG.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn register_tools(ev: &mut Evaluator) {
    ev.register("ToolsRegister", tools_register as NativeFn, Attributes::LISTABLE);
    ev.register("ToolsUnregister", tools_unregister as NativeFn, Attributes::empty());
    ev.register("ToolsList", tools_list as NativeFn, Attributes::empty());
    ev.register("ToolsDescribe", tools_describe as NativeFn, Attributes::empty());
    ev.register("ToolsSearch", tools_search as NativeFn, Attributes::empty());
    ev.register("ToolsInvoke", tools_invoke as NativeFn, Attributes::HOLD_ALL);
}

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

fn get_str(m: &HashMap<String, Value>, k: &str) -> Option<String> {
    m.get(k).and_then(|v| match v { Value::String(s)=>Some(s.clone()), Value::Symbol(s)=>Some(s.clone()), _=>None })
}

fn value_to_string(v: &Value) -> Option<String> {
    match v { Value::String(s)=>Some(s.clone()), Value::Symbol(s)=>Some(s.clone()), _=>None }
}

// ToolsRegister[spec] or ToolsRegister[{spec1, spec2, ...}]
fn tools_register(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("ToolsRegister".into())), args } }
    let mut reg = tool_reg().lock().unwrap();
    for a in args.into_iter() {
        match a {
            Value::Assoc(m) => {
                if let Some(id) = get_str(&m, "id").or_else(|| get_str(&m, "name")) {
                    reg.insert(id, Value::Assoc(m));
                }
            }
            Value::List(items) => {
                for it in items.into_iter() {
                    if let Value::Assoc(m) = it {
                        if let Some(id) = get_str(&m, "id").or_else(|| get_str(&m, "name")) {
                            reg.insert(id, Value::Assoc(m));
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Value::Boolean(true)
}

// ToolsUnregister[id]
fn tools_unregister(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("ToolsUnregister".into())), args } }
    let id = match &args[0] { Value::String(s)|Value::Symbol(s)=>s.clone(), _=>return Value::Boolean(false) };
    let mut reg = tool_reg().lock().unwrap();
    Value::Boolean(reg.remove(&id).is_some())
}

fn builtin_cards(ev: &mut Evaluator) -> Vec<Value> {
    // Fallback discovery using only builtin names and attributes
    // We cannot access builtins directly; expose minimal card for known exported functions via DescribeBuiltins (if present)
    // If DescribeBuiltins is not present, return empty list.
    let head = Value::Symbol("DescribeBuiltins".to_string());
    let expr = Value::Expr { head: Box::new(head), args: vec![] };
    let desc = ev.eval(expr);
    match desc {
        Value::List(items) => items,
        _ => vec![],
    }
}

// ToolsList[] -> list of cards (registered specs first, then fallback builtins not overridden)
fn tools_list(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if !args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("ToolsList".into())), args } }
    let reg = tool_reg().lock().unwrap();
    let mut out: Vec<Value> = Vec::new();
    for (id, spec) in reg.iter() {
        let mut m = match spec { Value::Assoc(m)=>m.clone(), _=>HashMap::new() };
        m.entry("id".to_string()).or_insert(Value::String(id.clone()));
        out.push(Value::Assoc(m));
    }
    // Add fallback builtin cards if available and not already present
    let mut seen: std::collections::HashSet<String> = out.iter().filter_map(|v| if let Value::Assoc(m)=v { get_str(m, "id") } else { None }).collect();
    for card in builtin_cards(ev) {
        if let Value::Assoc(m) = &card {
            if let Some(id) = get_str(m, "id").or_else(|| get_str(m, "name")) {
                if !seen.contains(&id) { out.push(card); seen.insert(id); }
            }
        }
    }
    Value::List(out)
}

// ToolsDescribe[id] -> spec or fallback card
fn tools_describe(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("ToolsDescribe".into())), args } }
    let key = match &args[0] { Value::String(s)|Value::Symbol(s)=>s.clone(), _=>return Value::Assoc(HashMap::new()) };
    let reg = tool_reg().lock().unwrap();
    if let Some(spec) = reg.get(&key) { return spec.clone(); }
    // try fallback cards by name/id
    for card in builtin_cards(ev) {
        if let Value::Assoc(m) = &card {
            let id = get_str(m, "id").or_else(|| get_str(m, "name"));
            if id.as_deref()==Some(&key) { return card; }
        }
    }
    Value::Assoc(HashMap::new())
}

// ToolsSearch[query, topK?] -> compact cards with naive scoring
fn tools_search(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("ToolsSearch".into())), args } }
    let q = match &args[0] { Value::String(s)|Value::Symbol(s)=>s.to_lowercase(), _=>String::new() };
    let topk = if args.len()>=2 { if let Value::Integer(k) = args[1].clone() { k.max(1) as usize } else { 10 } } else { 10 };
    let mut pool: Vec<Value> = match tools_list(ev, vec![]) { Value::List(vs)=>vs, _=>vec![] };
    let mut scored: Vec<(i64, Value)> = Vec::new();
    for v in pool.drain(..) {
        let mut score: i64 = 0;
        if let Value::Assoc(m) = &v {
            if let Some(s) = get_str(m, "name").or_else(|| get_str(m, "id")) { if s.to_lowercase().contains(&q) { score += 5; } }
            if let Some(s) = get_str(m, "summary") { if s.to_lowercase().contains(&q) { score += 3; } }
            if let Some(Value::List(tags)) = m.get("tags") {
                for t in tags { if let Some(ts) = value_to_string(t) { if ts.to_lowercase().contains(&q) { score += 2; } } }
            }
        }
        if score>0 || q.is_empty() { scored.push((score, v)); }
    }
    scored.sort_by(|a,b| b.0.cmp(&a.0));
    Value::List(scored.into_iter().take(topk).map(|(_s, v)| v).collect())
}

// ToolsInvoke[idOrName, argsAssoc?] -> evaluate head with positional mapping if params provided in spec
fn tools_invoke(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("ToolsInvoke".into())), args } }
    let key = match &args[0] { Value::String(s)|Value::Symbol(s)=>s.clone(), other=> match ev.eval(other.clone()) { Value::String(s)|Value::Symbol(s)=>s, _=>String::new() } };
    let provided = if args.len()>=2 { ev.eval(args[1].clone()) } else { Value::Assoc(HashMap::new()) };
    let (name, params): (String, Vec<String>) = {
        let reg = tool_reg().lock().unwrap();
        if let Some(Value::Assoc(m)) = reg.get(&key) {
            let nm = get_str(m, "impl").or_else(|| get_str(m, "name")).unwrap_or(key.clone());
            let ps: Vec<String> = match m.get("params") {
                Some(Value::List(vs)) => vs.iter().filter_map(|x| value_to_string(x)).collect(),
                _ => vec![],
            };
            (nm, ps)
        } else { (key.clone(), vec![]) }
    };
    // If provided args is an Assoc and params exist, map by order; else if provided is a List, use as-is.
    let call_args: Vec<Value> = match &provided {
        Value::Assoc(m) if !params.is_empty() => {
            let mut out: Vec<Value> = Vec::new();
            for p in params.iter() {
                out.push(m.get(p).cloned().unwrap_or(Value::Symbol("Null".into())));
            }
            out
        }
        Value::List(vs) => vs.clone(),
        Value::Assoc(m) => {
            // If no params, but single key "args" -> use list under it
            if let Some(Value::List(vs)) = m.get("args") { vs.clone() } else { vec![Value::Assoc(m.clone())] }
        }
        other => vec![other.clone()],
    };
    let expr = Value::Expr { head: Box::new(Value::Symbol(name)), args: call_args };
    ev.eval(expr)
}
