use lyra_core::value::Value;
use lyra_runtime::{Evaluator};
use lyra_runtime::attrs::Attributes;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

pub fn register_logic(ev: &mut Evaluator) {
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
