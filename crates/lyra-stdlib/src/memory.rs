use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

static SESS: OnceLock<Mutex<HashMap<String, Vec<Value>>>> = OnceLock::new();
fn sess() -> &'static Mutex<HashMap<String, Vec<Value>>> {
    SESS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn register_memory(ev: &mut Evaluator) {
    ev.register("Remember", remember as NativeFn, Attributes::LISTABLE);
    ev.register("Recall", recall as NativeFn, Attributes::empty());
    ev.register("SessionClear", session_clear as NativeFn, Attributes::empty());
}

pub fn register_memory_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    super::register_if(ev, pred, "Remember", remember as NativeFn, Attributes::LISTABLE);
    super::register_if(ev, pred, "Recall", recall as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "SessionClear", session_clear as NativeFn, Attributes::empty());
}

fn key_of(v: &Value) -> Option<String> {
    match v {
        Value::String(s) | Value::Symbol(s) => Some(s.clone()),
        _ => None,
    }
}

// Remember[session_id, data]
fn remember(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("Remember".into())), args };
    }
    let sid = match key_of(&args[0]) {
        Some(s) => s,
        None => return Value::Boolean(false),
    };
    let data = args[1].clone();
    let mut m = sess().lock().unwrap();
    m.entry(sid).or_insert_with(Vec::new).push(data);
    Value::Boolean(true)
}

// Recall[session_id, query?, K->n]
fn recall(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("Recall".into())), args };
    }
    let sid = match key_of(&args[0]) {
        Some(s) => s,
        None => return Value::List(vec![]),
    };
    let mut query: Option<String> = None;
    let mut k: usize = 5;
    for a in &args[1..] {
        match a {
            Value::String(s) | Value::Symbol(s) => {
                query = Some(s.clone());
            }
            Value::Assoc(m) => {
                if let Some(Value::Integer(n)) = m.get("K") {
                    if *n > 0 {
                        k = *n as usize;
                    }
                }
            }
            _ => {}
        }
    }
    let m = sess().lock().unwrap();
    let items = m.get(&sid).cloned().unwrap_or_default();
    let mut out: Vec<Value> = items;
    if let Some(q) = query {
        out = out
            .into_iter()
            .filter(|v| match v {
                Value::String(s) => s.contains(&q),
                Value::Assoc(am) => {
                    am.values().any(|vv| matches!(vv, Value::String(ss) if ss.contains(&q)))
                }
                _ => false,
            })
            .collect();
    }
    if out.len() > k {
        out = out.into_iter().rev().take(k).rev().collect();
    }
    Value::List(out)
}

fn session_clear(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("SessionClear".into())), args };
    }
    let sid = match key_of(&args[0]) {
        Some(s) => s,
        None => return Value::Boolean(false),
    };
    let mut m = sess().lock().unwrap();
    m.remove(&sid);
    Value::Boolean(true)
}
