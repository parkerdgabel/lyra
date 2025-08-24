use lyra_runtime::{Evaluator};
use lyra_runtime::attrs::Attributes;
use lyra_core::value::Value;
#[cfg(feature = "tools")] use crate::tools::add_specs;
#[cfg(feature = "tools")] use crate::tool_spec;
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
        tool_spec!("Plus", summary: "Sum numbers (variadic)", params: ["args"], tags: ["math","sum"], input_schema: Value::Assoc(HashMap::from([
            ("type".to_string(), Value::String("object".into())),
            ("properties".to_string(), Value::Assoc(HashMap::from([
                ("args".to_string(), Value::Assoc(HashMap::from([
                    ("type".to_string(), Value::String("array".into())),
                    ("items".to_string(), Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("number")))]))),
                ]))),
            ]))),
        ])), output_schema: Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("number")))])), examples: [
            Value::Assoc(HashMap::from([
                ("args".to_string(), Value::Assoc(HashMap::from([("args".to_string(), Value::List(vec![Value::Integer(1), Value::Integer(2)]))]))),
                ("result".to_string(), Value::Integer(3)),
            ]))
        ]),
        tool_spec!("Times", summary: "Multiply numbers (variadic)", params: ["args"], tags: ["math","product"], input_schema: Value::Assoc(HashMap::from([
            ("type".to_string(), Value::String("object".into())),
            ("properties".to_string(), Value::Assoc(HashMap::from([
                ("args".to_string(), Value::Assoc(HashMap::from([
                    ("type".to_string(), Value::String("array".into())),
                    ("items".to_string(), Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("number")))]))),
                ]))),
            ]))),
        ])), output_schema: Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("number")))])), examples: [
            Value::Assoc(HashMap::from([
                ("args".to_string(), Value::Assoc(HashMap::from([("args".to_string(), Value::List(vec![Value::Integer(2), Value::Integer(3)]))]))),
                ("result".to_string(), Value::Integer(6)),
            ]))
        ]),
        tool_spec!("Abs", summary: "Absolute value", params: ["x"], tags: ["math"], input_schema: Value::Assoc(HashMap::from([
            ("type".to_string(), Value::String("object".into())),
            ("properties".to_string(), Value::Assoc(HashMap::from([(String::from("x"), Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("number")))])))]))),
            ("required".to_string(), Value::List(vec![Value::String("x".into())])),
        ])), output_schema: Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("number")))])), examples: [
            Value::Assoc(HashMap::from([
                ("args".to_string(), Value::Assoc(HashMap::from([("x".to_string(), Value::Integer(-2))]))),
                ("result".to_string(), Value::Integer(2)),
            ]))
        ]),
        tool_spec!("Min", summary: "Minimum of values or list", params: ["args"], tags: ["math"], input_schema: Value::Assoc(HashMap::from([
            ("type".to_string(), Value::String("object".into())),
            ("properties".to_string(), Value::Assoc(HashMap::from([
                ("args".to_string(), Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("array")))]))),
            ]))),
        ]))),
        tool_spec!("Max", summary: "Maximum of values or list", params: ["args"], tags: ["math"], input_schema: Value::Assoc(HashMap::from([
            ("type".to_string(), Value::String("object".into())),
            ("properties".to_string(), Value::Assoc(HashMap::from([
                ("args".to_string(), Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("array")))]))),
            ]))),
        ]))),
    ]);
}

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 { let t = b; b = a % b; a = t; }
    a.abs()
}

fn reduce_rat(num: i64, den: i64) -> (i64, i64) {
    if den == 0 { return (num, den); }
    let mut n = num;
    let mut d = den;
    if d < 0 { n = -n; d = -d; }
    let g = gcd(n, d);
    (n / g, d / g)
}

fn rat_value(num: i64, den: i64) -> Value {
    let (n, d) = reduce_rat(num, den);
    if d == 1 { Value::Integer(n) } else { Value::Rational { num: n, den: d } }
}

fn add_numeric(a: Value, b: Value) -> Option<Value> {
    match (a, b) {
        (Value::Integer(x), Value::Integer(y)) => Some(Value::Integer(x + y)),
        (Value::Real(x), Value::Real(y)) => Some(Value::Real(x + y)),
        (Value::Integer(x), Value::Real(y)) => Some(Value::Real((x as f64) + y)),
        (Value::Real(x), Value::Integer(y)) => Some(Value::Real(x + (y as f64))),
        (Value::Rational { num: n1, den: d1 }, Value::Rational { num: n2, den: d2 }) => Some(rat_value(n1 * d2 + n2 * d1, d1 * d2)),
        (Value::Integer(x), Value::Rational { num, den }) | (Value::Rational { num, den }, Value::Integer(x)) => {
            Some(rat_value(num + x * den, den))
        }
        (Value::Real(x), Value::Rational { num, den }) | (Value::Rational { num, den }, Value::Real(x)) => {
            Some(Value::Real(x + (num as f64)/(den as f64)))
        }
        (Value::Complex { re: ar, im: ai }, Value::Complex { re: br, im: bi }) => {
            let rr = add_numeric((*ar).clone(), (*br).clone())?;
            let ri = add_numeric((*ai).clone(), (*bi).clone())?;
            Some(Value::Complex { re: Box::new(rr), im: Box::new(ri) })
        }
        (Value::Complex { re, im }, other) | (other, Value::Complex { re, im }) => {
            let rr = add_numeric((*re).clone(), other)?;
            Some(Value::Complex { re: Box::new(rr), im })
        }
        _ => None,
    }
}

fn mul_numeric(a: Value, b: Value) -> Option<Value> {
    match (a, b) {
        (Value::Integer(x), Value::Integer(y)) => Some(Value::Integer(x * y)),
        (Value::Real(x), Value::Real(y)) => Some(Value::Real(x * y)),
        (Value::Integer(x), Value::Real(y)) => Some(Value::Real((x as f64) * y)),
        (Value::Real(x), Value::Integer(y)) => Some(Value::Real(x * (y as f64))),
        (Value::Rational { num: n1, den: d1 }, Value::Rational { num: n2, den: d2 }) => Some(rat_value(n1 * n2, d1 * d2)),
        (Value::Integer(x), Value::Rational { num, den }) | (Value::Rational { num, den }, Value::Integer(x)) => {
            Some(rat_value(num * x, den))
        }
        (Value::Real(x), Value::Rational { num, den }) | (Value::Rational { num, den }, Value::Real(x)) => {
            Some(Value::Real(x * (num as f64)/(den as f64)))
        }
        (Value::Complex { re: ar, im: ai }, Value::Complex { re: br, im: bi }) => {
            // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            let ac = mul_numeric((*ar).clone(), (*br).clone())?;
            let bd = mul_numeric((*ai).clone(), (*bi).clone())?;
            let ad = mul_numeric((*ar).clone(), (*bi).clone())?;
            let bc = mul_numeric((*ai).clone(), (*br).clone())?;
            let real = add_numeric(ac, Value::Expr { head: Box::new(Value::Symbol("Minus".into())), args: vec![bd] })?; // fallback unevaluated minus
            let imag = add_numeric(ad, bc)?;
            Some(Value::Complex { re: Box::new(real), im: Box::new(imag) })
        }
        (Value::Complex { re, im }, other) | (other, Value::Complex { re, im }) => {
            let ar = (*re).clone(); let ai = (*im).clone();
            let br = other.clone(); let bi = Value::Integer(0);
            let real = add_numeric(mul_numeric(ar.clone(), br.clone())?, Value::Expr { head: Box::new(Value::Symbol("Minus".into())), args: vec![mul_numeric(ai.clone(), bi.clone())?] })?;
            let imag = add_numeric(mul_numeric(ar, bi)?, mul_numeric(ai, br)?)?;
            Some(Value::Complex { re: Box::new(real), im: Box::new(imag) })
        }
        _ => None,
    }
}

fn sub_numeric(a: Value, b: Value) -> Option<Value> {
    match (a, b) {
        (Value::Integer(x), Value::Integer(y)) => Some(Value::Integer(x - y)),
        (Value::Real(x), Value::Real(y)) => Some(Value::Real(x - y)),
        (Value::Integer(x), Value::Real(y)) => Some(Value::Real((x as f64) - y)),
        (Value::Real(x), Value::Integer(y)) => Some(Value::Real(x - (y as f64))),
        (Value::Rational { num: n1, den: d1 }, Value::Rational { num: n2, den: d2 }) => Some(rat_value(n1 * d2 - n2 * d1, d1 * d2)),
        (Value::Integer(x), Value::Rational { num, den }) => {
            Some(rat_value(x * den - num, den))
        }
        (Value::Rational { num, den }, Value::Integer(x)) => {
            Some(rat_value(num - x * den, den))
        }
        (Value::Real(x), Value::Rational { num, den }) => Some(Value::Real(x - (num as f64)/(den as f64))),
        (Value::Rational { num, den }, Value::Real(x)) => Some(Value::Real((num as f64)/(den as f64) - x)),
        (Value::Complex { re: ar, im: ai }, Value::Complex { re: br, im: bi }) => {
            let rr = sub_numeric((*ar).clone(), (*br).clone())?;
            let ri = sub_numeric((*ai).clone(), (*bi).clone())?;
            Some(Value::Complex { re: Box::new(rr), im: Box::new(ri) })
        }
        (Value::Complex { re, im }, other) => {
            let rr = sub_numeric((*re).clone(), other)?;
            Some(Value::Complex { re: Box::new(rr), im })
        }
        (other, Value::Complex { re, im }) => {
            let rr = sub_numeric(other, (*re).clone())?;
            Some(Value::Complex { re: Box::new(Value::Expr { head: Box::new(Value::Symbol("Minus".into())), args: vec![rr] }), im: Box::new(Value::Expr { head: Box::new(Value::Symbol("Minus".into())), args: vec![*im] }) })
        }
        _ => None,
    }
}

fn div_numeric(a: Value, b: Value) -> Option<Value> {
    match (a, b) {
        (Value::Integer(x), Value::Integer(y)) => { if y == 0 { None } else { Some(rat_value(x, y)) } }
        (Value::Real(x), Value::Real(y)) => Some(Value::Real(x / y)),
        (Value::Integer(x), Value::Real(y)) => Some(Value::Real((x as f64) / y)),
        (Value::Real(x), Value::Integer(y)) => Some(Value::Real(x / (y as f64))),
        (Value::Rational { num: n1, den: d1 }, Value::Rational { num: n2, den: d2 }) => { if n2 == 0 { None } else { Some(rat_value(n1 * d2, d1 * n2)) } }
        (Value::Integer(x), Value::Rational { num, den }) => {
            if num == 0 { return None; }
            Some(rat_value(x * den, num))
        }
        (Value::Rational { num, den }, Value::Integer(x)) => {
            if x == 0 { return None; }
            Some(rat_value(num, den * x))
        }
        (Value::Real(x), Value::Rational { num, den }) => { if num == 0 { None } else { Some(Value::Real(x / ((num as f64)/(den as f64)))) } }
        (Value::Rational { num, den }, Value::Real(x)) => Some(Value::Real(((num as f64)/(den as f64)) / x)),
        (Value::Complex { re: ar, im: ai }, Value::Integer(y)) => {
            if y == 0 { return None; }
            let denom = Value::Integer(y);
            let rr = div_numeric((*ar).clone(), denom.clone())?;
            let ri = div_numeric((*ai).clone(), denom)?;
            Some(Value::Complex { re: Box::new(rr), im: Box::new(ri) })
        }
        (Value::Complex { re: ar, im: ai }, Value::Real(y)) => {
            let denom = Value::Real(y);
            let rr = div_numeric((*ar).clone(), denom.clone())?;
            let ri = div_numeric((*ai).clone(), denom)?;
            Some(Value::Complex { re: Box::new(rr), im: Box::new(ri) })
        }
        _ => None,
    }
}

fn pow_numeric(base: Value, exp: Value) -> Option<Value> {
    match (base.clone(), exp) {
        (Value::Integer(a), Value::Integer(e)) => {
            if e >= 0 { Some(Value::Integer(a.pow(e as u32))) } else { let (rn, rd) = reduce_rat(1, a.pow((-e) as u32)); Some(Value::Rational { num: rn, den: rd }) }
        }
        (Value::Real(a), Value::Integer(e)) => Some(Value::Real(a.powi(e as i32))),
        (Value::Rational { num, den }, Value::Integer(e)) => {
            if e >= 0 { Some(rat_value(num.pow(e as u32), den.pow(e as u32))) }
            else { let p = (-e) as u32; Some(rat_value(den.pow(p), num.pow(p))) }
        }
        (Value::Complex { .. }, Value::Integer(e)) => {
            let mut k = e.abs();
            if k == 0 { return Some(Value::Integer(1)); }
            // repeated multiplication (simple)
            let mut acc: Option<Value> = None;
            let mut base_opt: Option<Value> = None;
            if let (Value::Complex{..}, _) = (base.clone(), e) { base_opt = Some(base.clone()); }
            while k > 0 {
                if acc.is_none() { acc = base_opt.clone(); k -= 1; continue; }
                acc = Some(mul_numeric(acc.unwrap(), base_opt.clone().unwrap())?);
                k -= 1;
            }
            if e < 0 { div_numeric(Value::Integer(1), acc.unwrap()) } else { acc }
        }
        _ => None,
    }
}

fn plus(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let mut it = args.into_iter();
    if let Some(first) = it.next() {
        let mut acc = first;
        for a in it {
            match add_numeric(acc, a) { Some(v) => acc = v, None => return Value::Expr { head: Box::new(Value::Symbol("Plus".into())), args: vec![] } }
        }
        acc
    } else { Value::Integer(0) }
}

fn times(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let mut it = args.into_iter();
    if let Some(first) = it.next() {
        let mut acc = first;
        for a in it {
            match mul_numeric(acc, a) { Some(v) => acc = v, None => return Value::Expr { head: Box::new(Value::Symbol("Times".into())), args: vec![] } }
        }
        acc
    } else { Value::Integer(1) }
}

fn minus(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [a, b] => sub_numeric(a.clone(), b.clone()).unwrap_or(Value::Expr { head: Box::new(Value::Symbol("Minus".into())), args: vec![a.clone(), b.clone()] }),
        [a] => sub_numeric(Value::Integer(0), a.clone()).unwrap_or(Value::Expr { head: Box::new(Value::Symbol("Minus".into())), args: vec![a.clone()] }),
        other => Value::Expr { head: Box::new(Value::Symbol("Minus".into())), args: other.to_vec() },
    }
}

fn divide(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [a, b] => div_numeric(a.clone(), b.clone()).unwrap_or(Value::Expr { head: Box::new(Value::Symbol("Divide".into())), args: vec![a.clone(), b.clone()] }),
        other => Value::Expr { head: Box::new(Value::Symbol("Divide".into())), args: other.to_vec() },
    }
}

fn power(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [a, b] => pow_numeric(a.clone(), b.clone()).unwrap_or(Value::Expr { head: Box::new(Value::Symbol("Power".into())), args: vec![a.clone(), b.clone()] }),
        other => Value::Expr { head: Box::new(Value::Symbol("Power".into())), args: other.to_vec() },
    }
}

fn abs_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] => Value::Integer(n.abs()),
        [Value::Real(x)] => Value::Real(x.abs()),
        [Value::Rational { num, den }] => {
            let (n, d) = reduce_rat(num.abs(), den.abs());
            Value::Rational { num: n, den: d }
        }
        [Value::Complex { .. }] => {
            // magnitude if parts are Real/Integer
            Value::Expr { head: Box::new(Value::Symbol("Abs".into())), args: args.to_vec() }
        }
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
