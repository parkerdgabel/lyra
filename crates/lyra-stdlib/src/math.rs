use lyra_runtime::{Evaluator};
use lyra_runtime::attrs::Attributes;
use lyra_core::value::Value;
#[cfg(feature = "tools")] use crate::tools::add_specs;
#[cfg(feature = "tools")] use std::collections::HashMap;

pub fn register_math(ev: &mut Evaluator) {
    ev.register("Plus", plus as NativeFn, Attributes::LISTABLE | Attributes::FLAT | Attributes::ORDERLESS | Attributes::ONE_IDENTITY);
    ev.register("Times", times as NativeFn, Attributes::LISTABLE | Attributes::FLAT | Attributes::ORDERLESS | Attributes::ONE_IDENTITY);
    ev.register("Minus", minus as NativeFn, Attributes::LISTABLE);
    ev.register("Divide", divide as NativeFn, Attributes::LISTABLE);
    ev.register("Power", power as NativeFn, Attributes::empty());
    ev.register("Abs", abs_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Min", min_fn as NativeFn, Attributes::FLAT | Attributes::ORDERLESS);
    ev.register("Max", max_fn as NativeFn, Attributes::FLAT | Attributes::ORDERLESS);

    #[cfg(feature = "tools")]
    add_specs(vec![
        // Plus
        Value::Assoc(HashMap::from([
            ("id".to_string(), Value::String("Plus".into())),
            ("name".to_string(), Value::String("Plus".into())),
            ("impl".to_string(), Value::String("Plus".into())),
            ("summary".to_string(), Value::String("Sum numbers (variadic)".into())),
            ("tags".to_string(), Value::List(vec![Value::String("math".into()), Value::String("sum".into())])),
            ("params".to_string(), Value::List(vec![Value::String("args".into())])),
            ("input_schema".to_string(), Value::Assoc(HashMap::from([
                ("type".to_string(), Value::String("object".into())),
                ("properties".to_string(), Value::Assoc(HashMap::from([
                    ("args".to_string(), Value::Assoc(HashMap::from([
                        ("type".to_string(), Value::String("array".into())),
                        ("items".to_string(), Value::Assoc(HashMap::from([
                            ("type".to_string(), Value::String("number".into())),
                        ]))),
                    ]))),
                ]))),
            ]))),
            ("output_schema".to_string(), Value::Assoc(HashMap::from([("type".to_string(), Value::String("number".into()))]))),
            ("examples".to_string(), Value::List(vec![
                Value::Assoc(HashMap::from([
                    ("args".to_string(), Value::Assoc(HashMap::from([("args".to_string(), Value::List(vec![Value::Integer(1), Value::Integer(2)]))]))),
                    ("result".to_string(), Value::Integer(3)),
                ])),
            ])),
        ])),

        // Times
        Value::Assoc(HashMap::from([
            ("id".to_string(), Value::String("Times".into())),
            ("name".to_string(), Value::String("Times".into())),
            ("impl".to_string(), Value::String("Times".into())),
            ("summary".to_string(), Value::String("Multiply numbers (variadic)".into())),
            ("tags".to_string(), Value::List(vec![Value::String("math".into()), Value::String("product".into())])),
            ("params".to_string(), Value::List(vec![Value::String("args".into())])),
            ("input_schema".to_string(), Value::Assoc(HashMap::from([
                ("type".to_string(), Value::String("object".into())),
                ("properties".to_string(), Value::Assoc(HashMap::from([
                    ("args".to_string(), Value::Assoc(HashMap::from([
                        ("type".to_string(), Value::String("array".into())),
                        ("items".to_string(), Value::Assoc(HashMap::from([("type".to_string(), Value::String("number".into()))]))),
                    ]))),
                ]))),
            ]))),
            ("output_schema".to_string(), Value::Assoc(HashMap::from([("type".to_string(), Value::String("number".into()))]))),
            ("examples".to_string(), Value::List(vec![Value::Assoc(HashMap::from([
                ("args".to_string(), Value::Assoc(HashMap::from([("args".to_string(), Value::List(vec![Value::Integer(2), Value::Integer(3)]))]))),
                ("result".to_string(), Value::Integer(6)),
            ]))])),
        ])),

        // Abs
        Value::Assoc(HashMap::from([
            ("id".to_string(), Value::String("Abs".into())),
            ("name".to_string(), Value::String("Abs".into())),
            ("impl".to_string(), Value::String("Abs".into())),
            ("summary".to_string(), Value::String("Absolute value".into())),
            ("tags".to_string(), Value::List(vec![Value::String("math".into())])),
            ("params".to_string(), Value::List(vec![Value::String("x".into())])),
            ("input_schema".to_string(), Value::Assoc(HashMap::from([
                ("type".to_string(), Value::String("object".into())),
                ("properties".to_string(), Value::Assoc(HashMap::from([("x".to_string(), Value::Assoc(HashMap::from([("type".to_string(), Value::String("number".into()))])))]))),
                ("required".to_string(), Value::List(vec![Value::String("x".into())])),
            ]))),
            ("output_schema".to_string(), Value::Assoc(HashMap::from([("type".to_string(), Value::String("number".into()))]))),
            ("examples".to_string(), Value::List(vec![Value::Assoc(HashMap::from([
                ("args".to_string(), Value::Assoc(HashMap::from([("x".to_string(), Value::Integer(-2))]))),
                ("result".to_string(), Value::Integer(2)),
            ]))])),
        ])),
    ]);
}

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

fn plus(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let mut acc_i: Option<i64> = Some(0);
    let mut acc_f: Option<f64> = Some(0.0);
    for a in args {
        match a {
            Value::Integer(n) => { if let Some(i)=acc_i { acc_i=Some(i+n); } else if let Some(f)=acc_f { acc_f=Some(f + n as f64); } },
            Value::Real(x) => { acc_i=None; if let Some(f)=acc_f { acc_f=Some(f + x); } },
            other => return Value::Expr { head: Box::new(Value::Symbol("Plus".into())), args: vec![other] },
        }
    }
    if let Some(i)=acc_i { Value::Integer(i) } else { Value::Real(acc_f.unwrap_or(0.0)) }
}

fn times(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let mut acc_i: Option<i64> = Some(1);
    let mut acc_f: Option<f64> = Some(1.0);
    for a in args {
        match a {
            Value::Integer(n) => { if let Some(i)=acc_i { acc_i=Some(i*n); } else if let Some(f)=acc_f { acc_f=Some(f * n as f64); } },
            Value::Real(x) => { acc_i=None; if let Some(f)=acc_f { acc_f=Some(f * x); } },
            other => return Value::Expr { head: Box::new(Value::Symbol("Times".into())), args: vec![other] },
        }
    }
    if let Some(i)=acc_i { Value::Integer(i) } else { Value::Real(acc_f.unwrap_or(1.0)) }
}

fn minus(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(a), Value::Integer(b)] => Value::Integer(a - b),
        [Value::Real(a), Value::Real(b)] => Value::Real(a - b),
        [Value::Integer(a), Value::Real(b)] => Value::Real((*a as f64) - b),
        [Value::Real(a), Value::Integer(b)] => Value::Real(a - (*b as f64)),
        [Value::Integer(a)] => Value::Integer(-a),
        [Value::Real(a)] => Value::Real(-a),
        other => Value::Expr { head: Box::new(Value::Symbol("Minus".into())), args: other.to_vec() },
    }
}

fn divide(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(a), Value::Integer(b)] => Value::Real((*a as f64) / (*b as f64)),
        [Value::Real(a), Value::Real(b)] => Value::Real(a / b),
        [Value::Integer(a), Value::Real(b)] => Value::Real((*a as f64) / b),
        [Value::Real(a), Value::Integer(b)] => Value::Real(a / (*b as f64)),
        other => Value::Expr { head: Box::new(Value::Symbol("Divide".into())), args: other.to_vec() },
    }
}

fn power(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(a), Value::Integer(b)] => Value::Real((*a as f64).powf(*b as f64)),
        [Value::Real(a), Value::Real(b)] => Value::Real(a.powf(*b)),
        [Value::Real(a), Value::Integer(b)] => Value::Real(a.powf(*b as f64)),
        [Value::Integer(a), Value::Real(b)] => Value::Real((*a as f64).powf(*b)),
        other => Value::Expr { head: Box::new(Value::Symbol("Power".into())), args: other.to_vec() },
    }
}

fn abs_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] => Value::Integer(n.abs()),
        [Value::Real(x)] => Value::Real(x.abs()),
        other => Value::Expr { head: Box::new(Value::Symbol("Abs".into())), args: other.to_vec() },
    }
}

fn min_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()==1 {
        match ev.eval(args[0].clone()) {
            Value::List(items) => return min_over_iter(items.into_iter()),
            other => return Value::Expr { head: Box::new(Value::Symbol("Min".into())), args: vec![other] },
        }
    }
    let evald: Vec<Value> = args.into_iter().map(|a| ev.eval(a)).collect();
    min_over_iter(evald.into_iter())
}

fn max_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()==1 {
        match ev.eval(args[0].clone()) {
            Value::List(items) => return max_over_iter(items.into_iter()),
            other => return Value::Expr { head: Box::new(Value::Symbol("Max".into())), args: vec![other] },
        }
    }
    let evald: Vec<Value> = args.into_iter().map(|a| ev.eval(a)).collect();
    max_over_iter(evald.into_iter())
}

fn min_over_iter<I: Iterator<Item=Value>>(iter: I) -> Value {
    let mut have = false;
    let mut use_real = false;
    let mut cur_i: i64 = 0;
    let mut cur_f: f64 = 0.0;
    for v in iter {
        match v {
            Value::Integer(n) => {
                if !have { have=true; cur_i=n; cur_f=n as f64; }
                else {
                    if use_real { if (n as f64) < cur_f { cur_f = n as f64; } }
                    else if n < cur_i { cur_i = n; }
                }
            }
            Value::Real(x) => {
                if !have { have=true; cur_f=x; cur_i=x as i64; use_real=true; }
                else { use_real=true; if x < cur_f { cur_f = x; } }
            }
            other => return Value::Expr { head: Box::new(Value::Symbol("Min".into())), args: vec![other] },
        }
    }
    if !have { Value::Expr { head: Box::new(Value::Symbol("Min".into())), args: vec![] } }
    else if use_real { Value::Real(cur_f) } else { Value::Integer(cur_i) }
}

fn max_over_iter<I: Iterator<Item=Value>>(iter: I) -> Value {
    let mut have = false;
    let mut use_real = false;
    let mut cur_i: i64 = 0;
    let mut cur_f: f64 = 0.0;
    for v in iter {
        match v {
            Value::Integer(n) => {
                if !have { have=true; cur_i=n; cur_f=n as f64; }
                else {
                    if use_real { if (n as f64) > cur_f { cur_f = n as f64; } }
                    else if n > cur_i { cur_i = n; }
                }
            }
            Value::Real(x) => {
                if !have { have=true; cur_f=x; cur_i=x as i64; use_real=true; }
                else { use_real=true; if x > cur_f { cur_f = x; } }
            }
            other => return Value::Expr { head: Box::new(Value::Symbol("Max".into())), args: vec![other] },
        }
    }
    if !have { Value::Expr { head: Box::new(Value::Symbol("Max".into())), args: vec![] } }
    else if use_real { Value::Real(cur_f) } else { Value::Integer(cur_i) }
}
