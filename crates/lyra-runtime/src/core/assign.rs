use crate::attrs::Attributes;
use crate::eval::Evaluator;
use lyra_core::value::Value;

/// Register assignment and scoping forms: `Set`, `Unset`, `SetDelayed`, `With`.
pub fn register_assign(ev: &mut Evaluator) {
    ev.register("Set", set_fn as crate::eval::NativeFn, Attributes::HOLD_ALL);
    ev.register("Unset", unset_fn as crate::eval::NativeFn, Attributes::HOLD_ALL);
    ev.register("SetDelayed", set_delayed_fn as crate::eval::NativeFn, Attributes::HOLD_ALL);
    ev.register("With", with_fn as crate::eval::NativeFn, Attributes::HOLD_ALL);
}

fn set_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("Set".into())), args }; }
    match &args[0] {
        Value::Symbol(name) => {
            let v = ev.eval(args[1].clone());
            ev.env.insert(name.clone(), v.clone());
            v
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("Set".into())), args },
    }
}

fn unset_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::Expr { head: Box::new(Value::Symbol("Unset".into())), args }; }
    match &args[0] {
        Value::Symbol(name) => { ev.env.remove(name); Value::Symbol("Null".into()) }
        _ => Value::Expr { head: Box::new(Value::Symbol("Unset".into())), args },
    }
}

fn with_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("With".into())), args }; }
    let assoc = ev.eval(args[0].clone());
    let body = args[1].clone();
    let mut saved: Vec<(String, Option<Value>)> = Vec::new();
    if let Value::Assoc(m) = assoc {
        for (k, v) in m {
            let old = ev.env.get(&k).cloned();
            let newv = ev.eval(v);
            saved.push((k.clone(), old));
            ev.env.insert(k, newv);
        }
        let result = ev.eval(body);
        for (k, ov) in saved.into_iter() {
            if let Some(val) = ov { ev.env.insert(k, val); } else { ev.env.remove(&k); }
        }
        result
    } else {
        Value::Expr { head: Box::new(Value::Symbol("With".into())), args }
    }
}

fn set_delayed_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("SetDelayed".into())), args }; }
    let lhs = args[0].clone();
    let rhs = args[1].clone();
    match lhs.clone() {
        Value::Symbol(s) => {
            let rule = Value::Expr { head: Box::new(Value::Symbol("Rule".into())), args: vec![Value::Symbol(s.clone()), rhs] };
            return ev.eval(Value::Expr { head: Box::new(Value::Symbol("SetOwnValues".into())), args: vec![Value::Symbol(s), Value::List(vec![rule])] });
        }
        Value::Expr { head, .. } => {
            match *head {
                Value::Symbol(ref s) => {
                    let rule = Value::Expr { head: Box::new(Value::Symbol("Rule".into())), args: vec![lhs, rhs] };
                    return ev.eval(Value::Expr { head: Box::new(Value::Symbol("SetDownValues".into())), args: vec![Value::Symbol(s.clone()), Value::List(vec![rule])]});
                }
                Value::Expr { head: inner_head, .. } => {
                    if let Value::Symbol(s) = *inner_head {
                        let rule = Value::Expr { head: Box::new(Value::Symbol("Rule".into())), args: vec![lhs, rhs] };
                        return ev.eval(Value::Expr { head: Box::new(Value::Symbol("SetSubValues".into())), args: vec![Value::Symbol(s), Value::List(vec![rule])]});
                    }
                    Value::Expr { head: Box::new(Value::Symbol("SetDelayed".into())), args }
                }
                _ => Value::Expr { head: Box::new(Value::Symbol("SetDelayed".into())), args },
            }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("SetDelayed".into())), args },
    }
}
