use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
use std::sync::{Mutex, OnceLock};

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

#[derive(Default)]
struct Mx {
    tool_calls: i64,
    model_calls: i64,
    tokens_in: i64,
    tokens_out: i64,
    cost_usd: f64,
}

static MX: OnceLock<Mutex<Mx>> = OnceLock::new();
fn mx() -> &'static Mutex<Mx> {
    MX.get_or_init(|| Mutex::new(Mx::default()))
}

pub fn register_metrics(ev: &mut Evaluator) {
    ev.register("Metrics", metrics as NativeFn, Attributes::empty());
    ev.register("CostAdd", cost_add as NativeFn, Attributes::empty());
    ev.register("CostSoFar", cost_so_far as NativeFn, Attributes::empty());
    ev.register("MetricsReset", metrics_reset as NativeFn, Attributes::empty());
}

pub fn register_metrics_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    super::register_if(ev, pred, "Metrics", metrics as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "CostAdd", cost_add as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "CostSoFar", cost_so_far as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "MetricsReset", metrics_reset as NativeFn, Attributes::empty());
}

fn metrics(_ev: &mut Evaluator, _args: Vec<Value>) -> Value {
    let m = mx().lock().unwrap();
    Value::Assoc(std::collections::HashMap::from([
        ("ToolCalls".into(), Value::Integer(m.tool_calls)),
        ("ModelCalls".into(), Value::Integer(m.model_calls)),
        ("TokensIn".into(), Value::Integer(m.tokens_in)),
        ("TokensOut".into(), Value::Integer(m.tokens_out)),
        ("CostUSD".into(), Value::Real(m.cost_usd)),
    ]))
}

fn cost_add(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let mut delta = 0.0;
    if let Some(Value::Real(r)) = args.get(0) {
        delta = *r;
    }
    if let Some(Value::Integer(i)) = args.get(0) {
        delta = *i as f64;
    }
    let mut m = mx().lock().unwrap();
    m.cost_usd += delta;
    Value::Real(m.cost_usd)
}

fn cost_so_far(_ev: &mut Evaluator, _args: Vec<Value>) -> Value {
    Value::Real(mx().lock().unwrap().cost_usd)
}

fn metrics_reset(_ev: &mut Evaluator, _args: Vec<Value>) -> Value {
    *mx().lock().unwrap() = Mx::default();
    Value::Boolean(true)
}
