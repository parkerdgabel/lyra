use lyra_core::value::Value;
use lyra_runtime::{Evaluator};
use lyra_runtime::attrs::Attributes;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

pub fn register_list(ev: &mut Evaluator) {
    ev.register("Length", length as NativeFn, Attributes::empty());
    ev.register("Range", range as NativeFn, Attributes::empty());
    ev.register("Join", join as NativeFn, Attributes::empty());
    ev.register("Reverse", reverse as NativeFn, Attributes::empty());
    ev.register("Flatten", flatten as NativeFn, Attributes::empty());
    ev.register("Partition", partition as NativeFn, Attributes::empty());
    ev.register("Transpose", transpose as NativeFn, Attributes::empty());
}

fn length(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Length".into())), args } }
    match ev.eval(args[0].clone()) {
        Value::List(v) => Value::Integer(v.len() as i64),
        Value::String(s) => Value::Integer(s.chars().count() as i64),
        _ => Value::Integer(0),
    }
}

fn range(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] => Value::List((1..=*n).map(Value::Integer).collect()),
        [Value::Integer(a), Value::Integer(b)] => {
            let (start, end) = (*a, *b);
            let mut out = Vec::new();
            if start <= end { for i in start..=end { out.push(Value::Integer(i)); } }
            else { for i in (end..=start).rev() { out.push(Value::Integer(i)); } }
            Value::List(out)
        }
        [a, b] => { let a1 = ev.eval(a.clone()); let b1 = ev.eval(b.clone()); range(ev, vec![a1, b1]) },
        _ => Value::Expr { head: Box::new(Value::Symbol("Range".into())), args },
    }
}

fn join(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::List(a), Value::List(b)] => {
            let mut out = a.clone();
            out.extend(b.clone());
            Value::List(out)
        }
        [a, b] => { let a1 = ev.eval(a.clone()); let b1 = ev.eval(b.clone()); join(ev, vec![a1, b1]) },
        _ => Value::Expr { head: Box::new(Value::Symbol("Join".into())), args },
    }
}

fn reverse(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Reverse".into())), args } }
    match ev.eval(args[0].clone()) {
        Value::List(mut v) => { v.reverse(); Value::List(v) }
        other => other,
    }
}

fn flatten(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v] => flatten_once(ev.eval(v.clone())),
        [v, Value::Integer(n)] => {
            let mut res = ev.eval(v.clone());
            for _ in 0..*n { res = flatten_once(res); }
            res
        }
        [v, Value::Symbol(s)] if s=="Infinity" => {
            let mut res = ev.eval(v.clone());
            loop { let next = flatten_once(res.clone()); if next==res { break; } res = next; }
            res
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("Flatten".into())), args },
    }
}

fn flatten_once(v: Value) -> Value {
    match v {
        Value::List(items) => Value::List(items.into_iter().flat_map(|x| match x { Value::List(inner)=>inner, other=>vec![other] }).collect()),
        other => other,
    }
}

// Partition[list, n], Partition[list, n, step]
fn partition(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v, Value::Integer(n)] => do_partition(ev.eval(v.clone()), *n as usize, *n as usize),
        [v, Value::Integer(n), Value::Integer(step)] => do_partition(ev.eval(v.clone()), *n as usize, *step as usize),
        _ => Value::Expr { head: Box::new(Value::Symbol("Partition".into())), args },
    }
}

fn do_partition(v: Value, n: usize, step: usize) -> Value {
    match v {
        Value::List(items) => {
            let mut out: Vec<Value> = Vec::new();
            let mut i = 0usize;
            while i < items.len() {
                let end = (i + n).min(items.len());
                out.push(Value::List(items[i..end].to_vec()));
                i += step;
            }
            Value::List(out)
        }
        other => other,
    }
}

fn transpose(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Transpose".into())), args } }
    match ev.eval(args[0].clone()) {
        Value::List(rows) => {
            if rows.is_empty() { return Value::List(vec![]) }
            let cols = match &rows[0] { Value::List(v)=> v.len(), _=>0 };
            let mut out: Vec<Value> = Vec::with_capacity(cols);
            for c in 0..cols {
                let mut col: Vec<Value> = Vec::with_capacity(rows.len());
                for r in &rows { if let Value::List(rv) = r { col.push(rv.get(c).cloned().unwrap_or(Value::Symbol("Null".into()))); } }
                out.push(Value::List(col));
            }
            Value::List(out)
        }
        other => other,
    }
}

