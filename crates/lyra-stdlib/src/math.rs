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

fn format_float(v: f64) -> String {
    // Trim trailing zeros and decimal point
    let s = format!("{:.12}", v);
    let s = s.trim_end_matches('0').trim_end_matches('.').to_string();
    if s.is_empty() { "0".into() } else { s }
}

#[cfg(feature = "big-real-rug")]
fn bigreal_binop(ax: &str, by: &str, op: fn(Float, Float) -> Float) -> Option<String> {
    let a = Float::with_val(128, ax).ok()?;
    let b = Float::with_val(128, by).ok()?;
    let r = op(a, b);
    Some(r.to_string_radix(10, None))
}

#[cfg(not(feature = "big-real-rug"))]
fn bigreal_binop(ax: &str, by: &str, op: fn(f64, f64) -> f64) -> Option<String> {
    let a = ax.parse::<f64>().ok()?;
    let b = by.parse::<f64>().ok()?;
    Some(format_float(op(a, b)))
}

fn add_numeric(a: Value, b: Value) -> Option<Value> {
    match (a, b) {
        (Value::Integer(x), Value::Integer(y)) => Some(Value::Integer(x + y)),
        (Value::Real(x), Value::Real(y)) => Some(Value::Real(x + y)),
        (Value::Integer(x), Value::Real(y)) => Some(Value::Real((x as f64) + y)),
        (Value::Real(x), Value::Integer(y)) => Some(Value::Real(x + (y as f64))),
        (Value::BigReal(ax), Value::BigReal(by)) => {
            let s = bigreal_binop(&ax, &by, |a,b| a+b)?;
            Some(Value::BigReal(s))
        }
        (Value::BigReal(ax), other) | (other, Value::BigReal(ax)) => {
            let xf = ax.parse::<f64>().ok()?;
            let yf = match other { Value::Integer(n)=>n as f64, Value::Real(r)=>r, Value::Rational{num,den} => (num as f64)/(den as f64), _=>return None };
            Some(Value::BigReal(format_float(xf + yf)))
        }
        (Value::PackedArray { shape: s1, data: d1 }, Value::PackedArray { shape: s2, data: d2 }) => {
            if s1 == s2 && d1.len() == d2.len() {
                let data: Vec<f64> = d1.iter().zip(d2.iter()).map(|(x,y)| x + y).collect();
                Some(Value::PackedArray { shape: s1, data })
            } else { None }
        }
        (Value::PackedArray { shape, data }, other) | (other, Value::PackedArray { shape, data }) => {
            fn to_f64_scalar(v: &Value) -> Option<f64> {
                match v {
                    Value::Integer(n) => Some(*n as f64),
                    Value::Real(x) => Some(*x),
                    Value::Rational { num, den } => if *den != 0 { Some((*num as f64)/(*den as f64)) } else { None },
                    Value::BigReal(s) => s.parse::<f64>().ok(),
                    _ => None,
                }
            }
            if let Some(s) = to_f64_scalar(&other) {
                let mut out = data.clone();
                for x in &mut out { *x += s; }
                Some(Value::PackedArray { shape, data: out })
            } else { None }
        }
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
        (Value::BigReal(ax), Value::BigReal(by)) => {
            let s = bigreal_binop(&ax, &by, |a,b| a*b)?;
            Some(Value::BigReal(s))
        }
        (Value::BigReal(ax), other) | (other, Value::BigReal(ax)) => {
            let xf = ax.parse::<f64>().ok()?;
            let yf = match other { Value::Integer(n)=>n as f64, Value::Real(r)=>r, Value::Rational{num,den} => (num as f64)/(den as f64), _=>return None };
            Some(Value::BigReal(format_float(xf * yf)))
        }
        (Value::PackedArray { shape: s1, data: d1 }, Value::PackedArray { shape: s2, data: d2 }) => {
            if s1 == s2 && d1.len() == d2.len() {
                let data: Vec<f64> = d1.iter().zip(d2.iter()).map(|(x,y)| x * y).collect();
                Some(Value::PackedArray { shape: s1, data })
            } else { None }
        }
        (Value::PackedArray { shape, data }, other) | (other, Value::PackedArray { shape, data }) => {
            fn to_f64_scalar(v: &Value) -> Option<f64> {
                match v {
                    Value::Integer(n) => Some(*n as f64),
                    Value::Real(x) => Some(*x),
                    Value::Rational { num, den } => if *den != 0 { Some((*num as f64)/(*den as f64)) } else { None },
                    Value::BigReal(s) => s.parse::<f64>().ok(),
                    _ => None,
                }
            }
            if let Some(s) = to_f64_scalar(&other) {
                let mut out = data.clone();
                for x in &mut out { *x *= s; }
                Some(Value::PackedArray { shape, data: out })
            } else { None }
        }
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
        (Value::BigReal(ax), Value::BigReal(by)) => {
            let s = bigreal_binop(&ax, &by, |a,b| a-b)?;
            Some(Value::BigReal(s))
        }
        (Value::BigReal(ax), other) => {
            let xf = ax.parse::<f64>().ok()?;
            let yf = match other { Value::Integer(n)=>n as f64, Value::Real(r)=>r, Value::Rational{num,den} => (num as f64)/(den as f64), _=>return None };
            Some(Value::BigReal(format_float(xf - yf)))
        }
        (other, Value::BigReal(by)) => {
            let xf = match other { Value::Integer(n)=>n as f64, Value::Real(r)=>r, Value::Rational{num,den} => (num as f64)/(den as f64), _=>return None };
            let yf = by.parse::<f64>().ok()?;
            Some(Value::BigReal(format_float(xf - yf)))
        }
        (Value::PackedArray { shape: s1, data: d1 }, Value::PackedArray { shape: s2, data: d2 }) => {
            // broadcast-aware elementwise subtraction
            broadcast_elementwise(&s1, &d1, &s2, &d2, |x,y| x - y)
        }
        (Value::PackedArray { shape, data }, other) => {
            if let Some(s) = to_f64_scalar(&other) {
                let mut out = data.clone();
                for x in &mut out { *x -= s; }
                Some(Value::PackedArray { shape, data: out })
            } else { None }
        }
        (other, Value::PackedArray { shape, data }) => {
            if let Some(s) = to_f64_scalar(&other) {
                let mut out = data.clone();
                for x in &mut out { *x = s - *x; }
                Some(Value::PackedArray { shape, data: out })
            } else { None }
        }
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
        (Value::PackedArray { shape: s1, data: d1 }, Value::PackedArray { shape: s2, data: d2 }) => {
            broadcast_elementwise(&s1, &d1, &s2, &d2, |x,y| x / y)
        }
        (Value::PackedArray { shape, data }, other) => {
            if let Some(s) = to_f64_scalar(&other) {
                let mut out = data.clone();
                for x in &mut out { *x /= s; }
                Some(Value::PackedArray { shape, data: out })
            } else { None }
        }
        (other, Value::PackedArray { shape, data }) => {
            if let Some(s) = to_f64_scalar(&other) {
                let mut out = data.clone();
                for x in &mut out { *x = s / *x; }
                Some(Value::PackedArray { shape, data: out })
            } else { None }
        }
        (Value::Integer(x), Value::Integer(y)) => { if y == 0 { None } else { Some(rat_value(x, y)) } }
        (Value::Real(x), Value::Real(y)) => Some(Value::Real(x / y)),
        (Value::Integer(x), Value::Real(y)) => Some(Value::Real((x as f64) / y)),
        (Value::Real(x), Value::Integer(y)) => Some(Value::Real(x / (y as f64))),
        (Value::BigReal(ax), Value::BigReal(by)) => {
            let s = bigreal_binop(&ax, &by, |a,b| a/b)?;
            Some(Value::BigReal(s))
        }
        (Value::BigReal(ax), other) => {
            let xf = ax.parse::<f64>().ok()?;
            let yf = match other { Value::Integer(n)=>n as f64, Value::Real(r)=>r, Value::Rational{num,den} => (num as f64)/(den as f64), _=>return None };
            Some(Value::BigReal(format_float(xf / yf)))
        }
        (other, Value::BigReal(by)) => {
            let xf = match other { Value::Integer(n)=>n as f64, Value::Real(r)=>r, Value::Rational{num,den} => (num as f64)/(den as f64), _=>return None };
            let yf = by.parse::<f64>().ok()?;
            Some(Value::BigReal(format_float(xf / yf)))
        }
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
        (Value::Complex { re: ar, im: ai }, Value::Complex { re: br, im: bi }) => {
            // (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c^2 + d^2)
            let c2 = mul_numeric((*br).clone(), (*br).clone())?;
            let d2 = mul_numeric((*bi).clone(), (*bi).clone())?;
            let denom = add_numeric(c2, d2)?;
            // real numerator: ac + bd
            let ac = mul_numeric((*ar).clone(), (*br).clone())?;
            let bd = mul_numeric((*ai).clone(), (*bi).clone())?;
            let num_r = add_numeric(ac, bd)?;
            // imag numerator: bc - ad
            let bc = mul_numeric((*ai).clone(), (*br).clone())?;
            let ad = mul_numeric((*ar).clone(), (*bi).clone())?;
            let num_i = sub_numeric(bc, ad)?;
            let rr = div_numeric(num_r, denom.clone())?;
            let ri = div_numeric(num_i, denom)?;
            Some(Value::Complex { re: Box::new(rr), im: Box::new(ri) })
        }
        _ => None,
    }
}

fn to_f64_scalar(v: &Value) -> Option<f64> {
    match v {
        Value::Integer(n) => Some(*n as f64),
        Value::Real(x) => Some(*x),
        Value::Rational { num, den } => if *den != 0 { Some((*num as f64)/(*den as f64)) } else { None },
        Value::BigReal(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

fn broadcast_elementwise(s1: &Vec<usize>, d1: &Vec<f64>, s2: &Vec<usize>, d2: &Vec<f64>, op: fn(f64,f64)->f64) -> Option<Value> {
    let ndim = std::cmp::max(s1.len(), s2.len());
    let mut sh1 = vec![1; ndim];
    let mut sh2 = vec![1; ndim];
    sh1[ndim - s1.len()..].clone_from_slice(&s1);
    sh2[ndim - s2.len()..].clone_from_slice(&s2);
    let mut out_shape: Vec<usize> = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let a = sh1[i]; let b = sh2[i];
        if a == b { out_shape.push(a); }
        else if a == 1 { out_shape.push(b); }
        else if b == 1 { out_shape.push(a); }
        else { return None; }
    }
    let total: usize = out_shape.iter().product();
    let strides = |shape: &Vec<usize>| -> Vec<usize> {
        let mut st = vec![0; ndim];
        let mut acc = 1usize;
        for i in (0..ndim).rev() { st[i] = acc; acc *= shape[i]; }
        st
    };
    let st1 = strides(&sh1); let st2 = strides(&sh2);
    let mut out = Vec::with_capacity(total);
    for idx in 0..total {
        // convert idx to multi-index
        let mut rem = idx;
        let mut off1 = 0usize; let mut off2 = 0usize;
        for i in 0..ndim {
            let _dim = out_shape[i];
            let coord = rem / (out_shape[i+1..].iter().product::<usize>().max(1));
            rem %= out_shape[i+1..].iter().product::<usize>().max(1);
            let c1 = if sh1[i]==1 { 0 } else { coord };
            let c2 = if sh2[i]==1 { 0 } else { coord };
            off1 += c1 * st1[i]; off2 += c2 * st2[i];
        }
        out.push(op(d1[off1], d2[off2]));
    }
    Some(Value::PackedArray { shape: out_shape, data: out })
}

fn pow_numeric(base: Value, exp: Value) -> Option<Value> {
    match (base.clone(), exp) {
        (Value::Integer(a), Value::Integer(e)) => {
            if e >= 0 { Some(Value::Integer(a.pow(e as u32))) } else { let (rn, rd) = reduce_rat(1, a.pow((-e) as u32)); Some(Value::Rational { num: rn, den: rd }) }
        }
        (Value::Real(a), Value::Integer(e)) => Some(Value::Real(a.powi(e as i32))),
        (Value::BigReal(ax), Value::Integer(e)) => {
            let xf = ax.parse::<f64>().ok()?;
            Some(Value::BigReal(format_float(xf.powi(e as i32))))
        }
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
    if args.is_empty() { return Value::Integer(0); }
    let mut acc = args[0].clone();
    for i in 1..args.len() {
        let a = args[i].clone();
        match add_numeric(acc.clone(), a.clone()) {
            Some(v) => acc = v,
            None => {
                // Fall back to symbolic form preserving remaining args (with folded prefix)
                let mut rest = Vec::with_capacity(args.len() - i + 1);
                rest.push(acc);
                rest.push(a);
                rest.extend_from_slice(&args[i+1..]);
                return Value::Expr { head: Box::new(Value::Symbol("Plus".into())), args: rest };
            }
        }
    }
    acc
}

fn times(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Integer(1); }
    let mut acc = args[0].clone();
    for i in 1..args.len() {
        let a = args[i].clone();
        match mul_numeric(acc.clone(), a.clone()) {
            Some(v) => acc = v,
            None => {
                // Fall back to symbolic form preserving remaining args (with folded prefix)
                let mut rest = Vec::with_capacity(args.len() - i + 1);
                rest.push(acc);
                rest.push(a);
                rest.extend_from_slice(&args[i+1..]);
                return Value::Expr { head: Box::new(Value::Symbol("Times".into())), args: rest };
            }
        }
    }
    acc
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
        [Value::Complex { re, im }] => {
            // If parts are integers, try perfect square; else compute f64 sqrt(a^2+b^2)
            match (&**re, &**im) {
                (Value::Integer(a), Value::Integer(b)) => {
                    let aa = (*a).saturating_mul(*a);
                    let bb = (*b).saturating_mul(*b);
                    let sum = aa.saturating_add(bb);
                    let rt = (sum as f64).sqrt().round() as i64;
                    if rt.saturating_mul(rt) == sum { Value::Integer(rt) }
                    else { Value::Real((sum as f64).sqrt()) }
                }
                _ => {
                    fn to_f64(v: &Value) -> Option<f64> {
                        match v {
                            Value::Integer(n) => Some(*n as f64),
                            Value::Real(x) => Some(*x),
                            Value::Rational { num, den } => if *den != 0 { Some((*num as f64)/(*den as f64)) } else { None },
                            _ => None,
                        }
                    }
                    if let (Some(ar), Some(ai)) = (to_f64(&re), to_f64(&im)) {
                        Value::Real(((ar*ar)+(ai*ai)).sqrt())
                    } else {
                        Value::Expr { head: Box::new(Value::Symbol("Abs".into())), args: args.to_vec() }
                    }
                }
            }
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
