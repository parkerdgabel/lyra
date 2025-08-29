use crate::attrs::Attributes;
use crate::eval::{Evaluator, NativeFn};
use lyra_core::schema::schema_of;
use lyra_core::value::Value;

pub fn register_schema(ev: &mut Evaluator) {
    ev.register("Schema", schema_fn as NativeFn, Attributes::empty());
}

pub fn register_explain(ev: &mut Evaluator) {
    ev.register("Explain", explain_fn as NativeFn, Attributes::HOLD_ALL);
}

fn schema_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::Expr { head: Box::new(Value::Symbol("Schema".into())), args }; }
    let v = ev.eval(args[0].clone());
    match v { Value::Assoc(_) | Value::List(_) | Value::Expr { .. } | Value::Integer(_) | Value::Real(_) | Value::BigReal(_)
            | Value::Rational { .. } | Value::Complex { .. } | Value::PackedArray { .. } | Value::String(_) | Value::Symbol(_)
            | Value::Boolean(_) | Value::Slot(_) | Value::PureFunction { .. } => schema_of(&v) }
}

fn explain_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::Expr { head: Box::new(Value::Symbol("Explain".into())), args }; }
    let expr = args[0].clone();
    let env_snapshot = ev.env.clone();
    let mut ev2 = Evaluator::with_env(env_snapshot);
    ev2.trace_enabled = true;
    let _ = ev2.eval(expr);
    let steps = Value::List(ev2.trace_steps);
    Value::Assoc(vec![
        ("steps".to_string(), steps),
        ("algorithm".to_string(), Value::String("stub".into())),
        ("provider".to_string(), Value::String("cpu".into())),
        ("estCost".to_string(), Value::Assoc(Default::default())),
    ].into_iter().collect())
}

