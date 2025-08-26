use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_runtime::attrs::Attributes;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

pub fn register_policy(ev: &mut Evaluator) {
    ev.register("WithPolicy", with_policy as NativeFn, Attributes::HOLD_ALL);
}

pub fn register_policy_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str)->bool) {
    super::register_if(ev, pred, "WithPolicy", with_policy as NativeFn, Attributes::HOLD_ALL);
}

// WithPolicy[<|Capabilities->{...}|>, expr]
fn with_policy(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("WithPolicy".into())), args } }
    let mut caps: Option<Vec<Value>> = None;
    if let Value::Assoc(m) = &args[0] {
        if let Some(v) = m.get("Capabilities") { match v { Value::List(items) => { caps = Some(items.clone()); }, Value::String(_)|Value::Symbol(_) => { caps = Some(vec![v.clone()]); }, _=>{} } }
    }
    // Save old caps via ToolsGetCapabilities[]
    let old_caps = ev.eval(Value::Expr { head: Box::new(Value::Symbol("ToolsGetCapabilities".into())), args: vec![] });
    if let Some(list) = caps { let _ = ev.eval(Value::Expr { head: Box::new(Value::Symbol("ToolsSetCapabilities".into())), args: vec![ Value::List(list) ] }); }
    let result = ev.eval(args[1].clone());
    // Restore
    match old_caps { Value::List(list) => { let _ = ev.eval(Value::Expr { head: Box::new(Value::Symbol("ToolsSetCapabilities".into())), args: vec![ Value::List(list) ] }); }, _ => { let _ = ev.eval(Value::Expr { head: Box::new(Value::Symbol("ToolsSetCapabilities".into())), args: vec![ Value::List(vec![]) ] }); } }
    result
}

