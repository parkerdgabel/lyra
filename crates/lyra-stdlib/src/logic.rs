use lyra_core::value::Value;
use lyra_runtime::{Evaluator};
use lyra_runtime::attrs::Attributes;
use crate::register_if;
#[cfg(feature = "tools")] use crate::tools::{add_specs, schema_object_value};
#[cfg(feature = "tools")] use crate::tool_spec;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

pub fn register_logic(ev: &mut Evaluator) {
    ev.register("If", if_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("When", when_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Unless", unless_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Switch", switch_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Equal", equal as NativeFn, Attributes::LISTABLE);
    ev.register("Less", less as NativeFn, Attributes::LISTABLE);
    ev.register("LessEqual", less_equal as NativeFn, Attributes::LISTABLE);
    ev.register("Greater", greater as NativeFn, Attributes::LISTABLE);
    ev.register("GreaterEqual", greater_equal as NativeFn, Attributes::LISTABLE);
    ev.register("And", and_fn as NativeFn, Attributes::empty());
    ev.register("Or", or_fn as NativeFn, Attributes::empty());
    ev.register("Not", not_fn as NativeFn, Attributes::empty());
    ev.register("EvenQ", even_q as NativeFn, Attributes::LISTABLE);
    ev.register("OddQ", odd_q as NativeFn, Attributes::LISTABLE);

    #[cfg(feature = "tools")]
    add_specs(vec![
        tool_spec!("Equal", summary: "Test equality across arguments", params: ["args"], tags: ["logic"], input_schema: schema_object_value(vec![ (String::from("args"), Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("array")))]))) ], vec![]), output_schema: Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("boolean")))]))),
        tool_spec!("Less", summary: "Strictly increasing sequence", params: ["args"], tags: ["logic"], input_schema: schema_object_value(vec![ (String::from("args"), Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("array")))]))) ], vec![]), output_schema: Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("boolean")))]))),
        tool_spec!("LessEqual", summary: "Non-decreasing sequence", params: ["args"], tags: ["logic"], input_schema: schema_object_value(vec![ (String::from("args"), Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("array")))]))) ], vec![]), output_schema: Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("boolean")))]))),
        tool_spec!("Greater", summary: "Strictly decreasing sequence", params: ["args"], tags: ["logic"], input_schema: schema_object_value(vec![ (String::from("args"), Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("array")))]))) ], vec![]), output_schema: Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("boolean")))]))),
        tool_spec!("GreaterEqual", summary: "Non-increasing sequence", params: ["args"], tags: ["logic"], input_schema: schema_object_value(vec![ (String::from("args"), Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("array")))]))) ], vec![]), output_schema: Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("boolean")))]))),
        tool_spec!("And", summary: "Logical AND (short-circuit)", params: ["args"], tags: ["logic"], input_schema: schema_object_value(vec![ (String::from("args"), Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("array")))]))) ], vec![]), output_schema: Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("boolean")))]))),
        tool_spec!("Or", summary: "Logical OR (short-circuit)", params: ["args"], tags: ["logic"], input_schema: schema_object_value(vec![ (String::from("args"), Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("array")))]))) ], vec![]), output_schema: Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("boolean")))]))),
        tool_spec!("Not", summary: "Logical NOT", params: ["x"], tags: ["logic"], input_schema: schema_object_value(vec![ (String::from("x"), Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("boolean")))]))) ], vec![String::from("x")]), output_schema: Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("boolean")))]))),
        tool_spec!("EvenQ", summary: "Is integer even?", params: ["n"], tags: ["logic","math"], input_schema: schema_object_value(vec![ (String::from("n"), Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("integer")))]))) ], vec![String::from("n")]), output_schema: Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("boolean")))]))),
        tool_spec!("OddQ", summary: "Is integer odd?", params: ["n"], tags: ["logic","math"], input_schema: schema_object_value(vec![ (String::from("n"), Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("integer")))]))) ], vec![String::from("n")]), output_schema: Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("boolean")))]))),
    ]);
}

pub fn register_logic_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str)->bool) {
    register_if(ev, pred, "If", if_fn as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "When", when_fn as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "Unless", unless_fn as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "Switch", switch_fn as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "Equal", equal as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Less", less as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "LessEqual", less_equal as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Greater", greater as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "GreaterEqual", greater_equal as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "And", and_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Or", or_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Not", not_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "EvenQ", even_q as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "OddQ", odd_q as NativeFn, Attributes::LISTABLE);
}

fn equal(_ev: &mut Evaluator, args: Vec<Value>) -> Value { Value::Boolean(args.windows(2).all(|w| w[0]==w[1])) }
fn less(_ev: &mut Evaluator, args: Vec<Value>) -> Value { Value::Boolean(args.windows(2).all(|w| value_lt(&w[0], &w[1]))) }
fn less_equal(_ev: &mut Evaluator, args: Vec<Value>) -> Value { Value::Boolean(args.windows(2).all(|w| !value_gt(&w[0], &w[1]))) }
fn greater(_ev: &mut Evaluator, args: Vec<Value>) -> Value { Value::Boolean(args.windows(2).all(|w| value_gt(&w[0], &w[1]))) }
fn greater_equal(_ev: &mut Evaluator, args: Vec<Value>) -> Value { Value::Boolean(args.windows(2).all(|w| !value_lt(&w[0], &w[1]))) }

fn value_lt(a: &Value, b: &Value) -> bool { lyra_runtime::eval::value_order_key(a) < lyra_runtime::eval::value_order_key(b) }
fn value_gt(a: &Value, b: &Value) -> bool { lyra_runtime::eval::value_order_key(a) > lyra_runtime::eval::value_order_key(b) }

fn and_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    for a in args { if let Value::Boolean(false) = ev.eval(a) { return Value::Boolean(false); } }
    Value::Boolean(true)
}
fn or_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    for a in args { if let Value::Boolean(true) = ev.eval(a) { return Value::Boolean(true); } }
    Value::Boolean(false)
}
fn not_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [a] => match ev.eval(a.clone()) { Value::Boolean(b)=>Value::Boolean(!b), v=> Value::Expr { head: Box::new(Value::Symbol("Not".into())), args: vec![v] } },
        _ => Value::Expr { head: Box::new(Value::Symbol("Not".into())), args },
    }
}

fn even_q(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] => Value::Boolean(n % 2 == 0),
        [other] => Value::Boolean(matches!(other, Value::List(_)) == false && false),
        _ => Value::Expr { head: Box::new(Value::Symbol("EvenQ".into())), args },
    }
}

fn odd_q(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] => Value::Boolean(n % 2 != 0),
        [other] => Value::Boolean(matches!(other, Value::List(_)) == false && false),
        _ => Value::Expr { head: Box::new(Value::Symbol("OddQ".into())), args },
    }
}

// -------- Control flow --------
fn if_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [cond, then_] => match ev.eval(cond.clone()) {
            Value::Boolean(true) => ev.eval(then_.clone()),
            Value::Boolean(false) => Value::Symbol("Null".into()),
            other => Value::Expr { head: Box::new(Value::Symbol("If".into())), args: vec![other, then_.clone()] },
        },
        [cond, then_, else_] => match ev.eval(cond.clone()) {
            Value::Boolean(true) => ev.eval(then_.clone()),
            Value::Boolean(false) => ev.eval(else_.clone()),
            other => Value::Expr { head: Box::new(Value::Symbol("If".into())), args: vec![other, then_.clone(), else_.clone()] },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("If".into())), args },
    }
}

fn when_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [cond, body] => match ev.eval(cond.clone()) { Value::Boolean(true) => ev.eval(body.clone()), Value::Boolean(false)=>Value::Symbol("Null".into()), other=> Value::Expr { head: Box::new(Value::Symbol("When".into())), args: vec![other, body.clone()] } },
        [cond, body, else_] => match ev.eval(cond.clone()) { Value::Boolean(true) => ev.eval(body.clone()), Value::Boolean(false)=>ev.eval(else_.clone()), other=> Value::Expr { head: Box::new(Value::Symbol("When".into())), args: vec![other, body.clone(), else_.clone()] } },
        _ => Value::Expr { head: Box::new(Value::Symbol("When".into())), args },
    }
}

fn unless_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [cond, body] => match ev.eval(cond.clone()) { Value::Boolean(false) => ev.eval(body.clone()), Value::Boolean(true)=>Value::Symbol("Null".into()), other=> Value::Expr { head: Box::new(Value::Symbol("Unless".into())), args: vec![other, body.clone()] } },
        [cond, body, else_] => match ev.eval(cond.clone()) { Value::Boolean(false) => ev.eval(body.clone()), Value::Boolean(true)=>ev.eval(else_.clone()), other=> Value::Expr { head: Box::new(Value::Symbol("Unless".into())), args: vec![other, body.clone(), else_.clone()] } },
        _ => Value::Expr { head: Box::new(Value::Symbol("Unless".into())), args },
    }
}

fn switch_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Switch".into())), args } }
    let subj = ev.eval(args[0].clone());
    let (rules_v, default_v) = match args.as_slice() {
        [_, rules] => (ev.eval(rules.clone()), None),
        [_, rules, default_] => (ev.eval(rules.clone()), Some(default_.clone())),
        _ => (Value::Symbol("Null".into()), None),
    };
    // rules may be Assoc or List of pairs
    // Matching semantics:
    // - If rule key is a function/expr, apply to subj; if True, return evaluated value
    // - Else compare equality key == subj
    if let Value::Assoc(m) = rules_v {
        // try keys in sorted order for determinism
        let mut keys: Vec<String> = m.keys().cloned().collect();
        keys.sort();
        for k in keys {
            if k == "_" { continue; }
            let key_v = Value::String(k.clone());
            if value_matches(ev, &key_v, &subj) { return ev.eval(m.get(&k).cloned().unwrap()); }
        }
        if let Some(v) = m.get("_") { return ev.eval(v.clone()); }
        if let Some(d) = default_v { return ev.eval(d); }
        return Value::Symbol("Null".into());
    }
    if let Value::List(items) = rules_v {
        for it in items {
            if let Value::List(mut pair) = it { if pair.len()==2 { let rhs = pair.remove(1); let lhs = pair.remove(0); if value_matches(ev, &lhs, &subj) { return ev.eval(rhs); } } }
        }
        if let Some(d) = default_v { return ev.eval(d); }
        return Value::Symbol("Null".into());
    }
    Value::Expr { head: Box::new(Value::Symbol("Switch".into())), args: vec![subj, rules_v] }
}

fn value_matches(ev: &mut Evaluator, test: &Value, subj: &Value) -> bool {
    // Predicate form: test[subj] => True
    if matches!(test, Value::Expr { .. } | Value::PureFunction { .. } | Value::Symbol(_)) && !matches!(test, Value::String(_)) {
        let call = Value::Expr { head: Box::new(test.clone()), args: vec![subj.clone()] };
        return matches!(ev.eval(call), Value::Boolean(true));
    }
    // Literal equality
    test == subj
}
