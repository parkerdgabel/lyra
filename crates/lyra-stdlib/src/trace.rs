use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

#[derive(Clone)]
struct SpanRec {
    id: i64,
    name: String,
    attrs: HashMap<String, Value>,
    start: i64,
    end: Option<i64>,
}

static SPANS: OnceLock<Mutex<Vec<SpanRec>>> = OnceLock::new();
static STACK: OnceLock<Mutex<Vec<i64>>> = OnceLock::new();
static NEXT_ID: OnceLock<std::sync::atomic::AtomicI64> = OnceLock::new();

fn spans() -> &'static Mutex<Vec<SpanRec>> {
    SPANS.get_or_init(|| Mutex::new(Vec::new()))
}
fn stack() -> &'static Mutex<Vec<i64>> {
    STACK.get_or_init(|| Mutex::new(Vec::new()))
}
fn next_id() -> i64 {
    let a = NEXT_ID.get_or_init(|| std::sync::atomic::AtomicI64::new(1));
    a.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}
fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

pub fn register_trace(ev: &mut Evaluator) {
    ev.register("Span", span as NativeFn, Attributes::empty());
    ev.register("SpanEnd", span_end as NativeFn, Attributes::empty());
    ev.register("TraceGet", trace_get as NativeFn, Attributes::empty());
    ev.register("TraceExport", trace_export as NativeFn, Attributes::empty());
}

pub fn register_trace_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    super::register_if(ev, pred, "Span", span as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "SpanEnd", span_end as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "TraceGet", trace_get as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "TraceExport", trace_export as NativeFn, Attributes::empty());
}

fn span(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Span["name", Attrs-><|...|>]
    let name = args
        .get(0)
        .and_then(|v| match v {
            Value::String(s) | Value::Symbol(s) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_else(|| "span".into());
    let mut attrs: HashMap<String, Value> = HashMap::new();
    for a in &args {
        if let Value::Assoc(m) = a {
            if let Some(Value::Assoc(am)) = m.get("Attrs") {
                attrs = am.clone();
            }
        }
    }
    let id = next_id();
    let rec = SpanRec { id, name, attrs, start: now_ms(), end: None };
    spans().lock().unwrap().push(rec);
    stack().lock().unwrap().push(id);
    Value::Integer(id)
}

fn span_end(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // SpanEnd[] or SpanEnd[id]
    let id_opt = if args.is_empty() {
        stack().lock().unwrap().pop()
    } else {
        args.get(0).and_then(|v| if let Value::Integer(i) = v { Some(*i) } else { None })
    };
    if let Some(id) = id_opt {
        let mut guard = spans().lock().unwrap();
        for s in guard.iter_mut().rev() {
            if s.id == id && s.end.is_none() {
                s.end = Some(now_ms());
                break;
            }
        }
        Value::Boolean(true)
    } else {
        Value::Boolean(false)
    }
}

fn trace_get(_ev: &mut Evaluator, _args: Vec<Value>) -> Value {
    // Return list of assoc spans
    let list: Vec<Value> = spans()
        .lock()
        .unwrap()
        .iter()
        .map(|s| {
            Value::Assoc(HashMap::from([
                ("Id".into(), Value::Integer(s.id)),
                ("Name".into(), Value::String(s.name.clone())),
                ("Attrs".into(), Value::Assoc(s.attrs.clone())),
                ("Start".into(), Value::Integer(s.start)),
                ("End".into(), s.end.map(Value::Integer).unwrap_or(Value::Symbol("Null".into()))),
            ]))
        })
        .collect();
    Value::List(list)
}

fn trace_export(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // TraceExport["json", Path->"..."]
    let mut fmt = "json".to_string();
    let mut path: Option<String> = None;
    for a in &args {
        match a {
            Value::String(s) | Value::Symbol(s) => fmt = s.clone(),
            Value::Assoc(m) => {
                if let Some(Value::String(p)) = m.get("Path") {
                    path = Some(p.clone());
                }
            }
            _ => {}
        }
    }
    let spans_list = trace_get(&mut Evaluator::new(), vec![]);
    if fmt == "json" {
        if let Some(p) = path {
            if let Value::List(items) = spans_list {
                let json = serde_json::to_string(&items).unwrap_or("[]".into());
                let _ = std::fs::write(p, json);
                return Value::Boolean(true);
            }
        }
    }
    Value::Boolean(false)
}
