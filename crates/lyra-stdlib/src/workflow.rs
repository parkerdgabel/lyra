use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_runtime::attrs::Attributes;
use std::collections::HashMap;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

pub fn register_workflow(ev: &mut Evaluator) {
    ev.register("Workflow", workflow as NativeFn, Attributes::HOLD_ALL);
}

pub fn register_workflow_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str)->bool) {
    super::register_if(ev, pred, "Workflow", workflow as NativeFn, Attributes::HOLD_ALL);
}

// Minimal sequential workflow runner: Workflow[{step1, step2, ...}] or Workflow[{<|"name"->..., "run"->expr|>, ...}]
fn workflow(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Workflow".into())), args } }
    let steps = match &args[0] { Value::List(items) => items.clone(), _ => return Value::List(vec![]) };
    let mut results: Vec<Value> = Vec::new();
    for step in steps {
        let (name, run_expr) = match &step {
            Value::Assoc(m) => {
                let nm = m.get("name").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None }).unwrap_or_else(|| "step".into());
                let run = m.get("run").cloned().unwrap_or(Value::Symbol("Null".into()));
                (nm, run)
            }
            other => ("step".into(), other.clone())
        };
        // Optional trace span per step
        let _ = ev.eval(Value::Expr { head: Box::new(Value::Symbol("Span".into())), args: vec![
            Value::String("workflow:step".into()),
            Value::Assoc(HashMap::from([(String::from("Name"), Value::String(name))]))
        ]});
        let res = ev.eval(run_expr);
        let _ = ev.eval(Value::Expr { head: Box::new(Value::Symbol("SpanEnd".into())), args: vec![] });
        results.push(res);
    }
    Value::List(results)
}
