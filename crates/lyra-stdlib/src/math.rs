use crate::register_if;
#[cfg(feature = "tools")]
use crate::tool_spec;
#[cfg(feature = "tools")]
use crate::tools::add_specs;
use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
#[cfg(feature = "big-real-rug")]
use rug::Float;
#[cfg(feature = "tools")]
use std::collections::HashMap;

pub fn register_math(ev: &mut Evaluator) {
    ev.register(
        "Plus",
        plus as NativeFn,
        Attributes::LISTABLE | Attributes::FLAT | Attributes::ORDERLESS | Attributes::ONE_IDENTITY,
    );
    ev.register(
        "Times",
        times as NativeFn,
        Attributes::LISTABLE | Attributes::FLAT | Attributes::ORDERLESS | Attributes::ONE_IDENTITY,
    );
    ev.register("Minus", minus as NativeFn, Attributes::LISTABLE);
    ev.register("Divide", divide as NativeFn, Attributes::LISTABLE);
    ev.register("Power", power as NativeFn, Attributes::empty());
    ev.register("Abs", abs_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Min", min_fn as NativeFn, Attributes::FLAT | Attributes::ORDERLESS);
    ev.register("Max", max_fn as NativeFn, Attributes::FLAT | Attributes::ORDERLESS);

    // New math functions
    ev.register("Floor", floor_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Ceiling", ceiling_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Round", round_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Trunc", trunc_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Mod", mod_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Quotient", quotient_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Remainder", remainder_fn as NativeFn, Attributes::LISTABLE);
    ev.register("DivMod", divmod_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Sqrt", sqrt_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Exp", exp_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Log", log_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Sin", sin_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Cos", cos_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Tan", tan_fn as NativeFn, Attributes::LISTABLE);
    ev.register("ASin", asin_fn as NativeFn, Attributes::LISTABLE);
    ev.register("ACos", acos_fn as NativeFn, Attributes::LISTABLE);
    ev.register("ATan", atan_fn as NativeFn, Attributes::LISTABLE);
    ev.register("ATan2", atan2_fn as NativeFn, Attributes::empty());
    ev.register("NthRoot", nthroot_fn as NativeFn, Attributes::empty());
    ev.register("Total", total_fn as NativeFn, Attributes::empty());
    ev.register("Mean", mean_fn as NativeFn, Attributes::empty());
    ev.register("Median", median_fn as NativeFn, Attributes::empty());
    ev.register("Variance", variance_fn as NativeFn, Attributes::empty());
    ev.register("StandardDeviation", stddev_fn as NativeFn, Attributes::empty());
    ev.register("GCD", gcd_fn as NativeFn, Attributes::FLAT | Attributes::ORDERLESS);
    ev.register("LCM", lcm_fn as NativeFn, Attributes::FLAT | Attributes::ORDERLESS);
    ev.register("Factorial", factorial_fn as NativeFn, Attributes::empty());
    ev.register("Binomial", binomial_fn as NativeFn, Attributes::empty());
    ev.register("ToDegrees", to_degrees_fn as NativeFn, Attributes::LISTABLE);
    ev.register("ToRadians", to_radians_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Clip", clip_fn as NativeFn, Attributes::empty());
    ev.register("Signum", signum_fn as NativeFn, Attributes::LISTABLE);

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

pub fn register_math_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    register_if(
        ev,
        pred,
        "Plus",
        plus as NativeFn,
        Attributes::LISTABLE | Attributes::FLAT | Attributes::ORDERLESS | Attributes::ONE_IDENTITY,
    );
    register_if(
        ev,
        pred,
        "Times",
        times as NativeFn,
        Attributes::LISTABLE | Attributes::FLAT | Attributes::ORDERLESS | Attributes::ONE_IDENTITY,
    );
    register_if(ev, pred, "Minus", minus as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Divide", divide as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Power", power as NativeFn, Attributes::empty());
    register_if(ev, pred, "Abs", abs_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Min", min_fn as NativeFn, Attributes::FLAT | Attributes::ORDERLESS);
    register_if(ev, pred, "Max", max_fn as NativeFn, Attributes::FLAT | Attributes::ORDERLESS);
    register_if(ev, pred, "Floor", floor_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Ceiling", ceiling_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Round", round_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Trunc", trunc_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Mod", mod_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Quotient", quotient_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Remainder", remainder_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "DivMod", divmod_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Sqrt", sqrt_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Exp", exp_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Log", log_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Sin", sin_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Cos", cos_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Tan", tan_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "ASin", asin_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "ACos", acos_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "ATan", atan_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "ATan2", atan2_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "NthRoot", nthroot_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Total", total_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Mean", mean_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Median", median_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Variance", variance_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "StandardDeviation", stddev_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "GCD", gcd_fn as NativeFn, Attributes::FLAT | Attributes::ORDERLESS);
    register_if(ev, pred, "LCM", lcm_fn as NativeFn, Attributes::FLAT | Attributes::ORDERLESS);
    register_if(ev, pred, "Factorial", factorial_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Binomial", binomial_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "ToDegrees", to_degrees_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "ToRadians", to_radians_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Clip", clip_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Signum", signum_fn as NativeFn, Attributes::LISTABLE);
}

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.abs()
}

fn reduce_rat(num: i64, den: i64) -> (i64, i64) {
    if den == 0 {
        return (num, den);
    }
    let mut n = num;
    let mut d = den;
    if d < 0 {
        n = -n;
        d = -d;
    }
    let g = gcd(n, d);
    (n / g, d / g)
}

fn rat_value(num: i64, den: i64) -> Value {
    let (n, d) = reduce_rat(num, den);
    if d == 1 {
        Value::Integer(n)
    } else {
        Value::Rational { num: n, den: d }
    }
}

fn format_float(v: f64) -> String {
    // Trim trailing zeros and decimal point
    let s = format!("{:.12}", v);
    let s = s.trim_end_matches('0').trim_end_matches('.').to_string();
    if s.is_empty() {
        "0".into()
    } else {
        s
    }
}

#[cfg(feature = "big-real-rug")]
fn bigreal_binop(ax: &str, by: &str, op: fn(Float, Float) -> Float) -> Option<String> {
    let af = ax.parse::<f64>().ok()?;
    let bf = by.parse::<f64>().ok()?;
    let r = op(Float::with_val(128, af), Float::with_val(128, bf));
    Some(r.to_string())
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
            let s = bigreal_binop(&ax, &by, |a, b| a + b)?;
            Some(Value::BigReal(s))
        }
        (Value::BigReal(ax), other) | (other, Value::BigReal(ax)) => {
            let xf = ax.parse::<f64>().ok()?;
            let yf = match other {
                Value::Integer(n) => n as f64,
                Value::Real(r) => r,
                Value::Rational { num, den } => (num as f64) / (den as f64),
                _ => return None,
            };
            Some(Value::BigReal(format_float(xf + yf)))
        }
        (
            Value::PackedArray { shape: s1, data: d1 },
            Value::PackedArray { shape: s2, data: d2 },
        ) => {
            if s1 == s2 && d1.len() == d2.len() {
                let data: Vec<f64> = d1.iter().zip(d2.iter()).map(|(x, y)| x + y).collect();
                Some(Value::PackedArray { shape: s1, data })
            } else {
                None
            }
        }
        (Value::PackedArray { shape, data }, other)
        | (other, Value::PackedArray { shape, data }) => {
            fn to_f64_scalar(v: &Value) -> Option<f64> {
                match v {
                    Value::Integer(n) => Some(*n as f64),
                    Value::Real(x) => Some(*x),
                    Value::Rational { num, den } => {
                        if *den != 0 {
                            Some((*num as f64) / (*den as f64))
                        } else {
                            None
                        }
                    }
                    Value::BigReal(s) => s.parse::<f64>().ok(),
                    _ => None,
                }
            }
            if let Some(s) = to_f64_scalar(&other) {
                let mut out = data.clone();
                for x in &mut out {
                    *x += s;
                }
                Some(Value::PackedArray { shape, data: out })
            } else {
                None
            }
        }
        (Value::Rational { num: n1, den: d1 }, Value::Rational { num: n2, den: d2 }) => {
            Some(rat_value(n1 * d2 + n2 * d1, d1 * d2))
        }
        (Value::Integer(x), Value::Rational { num, den })
        | (Value::Rational { num, den }, Value::Integer(x)) => Some(rat_value(num + x * den, den)),
        (Value::Real(x), Value::Rational { num, den })
        | (Value::Rational { num, den }, Value::Real(x)) => {
            Some(Value::Real(x + (num as f64) / (den as f64)))
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
            let s = bigreal_binop(&ax, &by, |a, b| a * b)?;
            Some(Value::BigReal(s))
        }
        (Value::BigReal(ax), other) | (other, Value::BigReal(ax)) => {
            let xf = ax.parse::<f64>().ok()?;
            let yf = match other {
                Value::Integer(n) => n as f64,
                Value::Real(r) => r,
                Value::Rational { num, den } => (num as f64) / (den as f64),
                _ => return None,
            };
            Some(Value::BigReal(format_float(xf * yf)))
        }
        (
            Value::PackedArray { shape: s1, data: d1 },
            Value::PackedArray { shape: s2, data: d2 },
        ) => {
            if s1 == s2 && d1.len() == d2.len() {
                let data: Vec<f64> = d1.iter().zip(d2.iter()).map(|(x, y)| x * y).collect();
                Some(Value::PackedArray { shape: s1, data })
            } else {
                None
            }
        }
        (Value::PackedArray { shape, data }, other)
        | (other, Value::PackedArray { shape, data }) => {
            fn to_f64_scalar(v: &Value) -> Option<f64> {
                match v {
                    Value::Integer(n) => Some(*n as f64),
                    Value::Real(x) => Some(*x),
                    Value::Rational { num, den } => {
                        if *den != 0 {
                            Some((*num as f64) / (*den as f64))
                        } else {
                            None
                        }
                    }
                    Value::BigReal(s) => s.parse::<f64>().ok(),
                    _ => None,
                }
            }
            if let Some(s) = to_f64_scalar(&other) {
                let mut out = data.clone();
                for x in &mut out {
                    *x *= s;
                }
                Some(Value::PackedArray { shape, data: out })
            } else {
                None
            }
        }
        (Value::Rational { num: n1, den: d1 }, Value::Rational { num: n2, den: d2 }) => {
            Some(rat_value(n1 * n2, d1 * d2))
        }
        (Value::Integer(x), Value::Rational { num, den })
        | (Value::Rational { num, den }, Value::Integer(x)) => Some(rat_value(num * x, den)),
        (Value::Real(x), Value::Rational { num, den })
        | (Value::Rational { num, den }, Value::Real(x)) => {
            Some(Value::Real(x * (num as f64) / (den as f64)))
        }
        (Value::Complex { re: ar, im: ai }, Value::Complex { re: br, im: bi }) => {
            // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            let ac = mul_numeric((*ar).clone(), (*br).clone())?;
            let bd = mul_numeric((*ai).clone(), (*bi).clone())?;
            let ad = mul_numeric((*ar).clone(), (*bi).clone())?;
            let bc = mul_numeric((*ai).clone(), (*br).clone())?;
            let real = add_numeric(
                ac,
                Value::Expr { head: Box::new(Value::Symbol("Minus".into())), args: vec![bd] },
            )?; // fallback unevaluated minus
            let imag = add_numeric(ad, bc)?;
            Some(Value::Complex { re: Box::new(real), im: Box::new(imag) })
        }
        (Value::Complex { re, im }, other) | (other, Value::Complex { re, im }) => {
            let ar = (*re).clone();
            let ai = (*im).clone();
            let br = other.clone();
            let bi = Value::Integer(0);
            let real = add_numeric(
                mul_numeric(ar.clone(), br.clone())?,
                Value::Expr {
                    head: Box::new(Value::Symbol("Minus".into())),
                    args: vec![mul_numeric(ai.clone(), bi.clone())?],
                },
            )?;
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
            let s = bigreal_binop(&ax, &by, |a, b| a - b)?;
            Some(Value::BigReal(s))
        }
        (Value::BigReal(ax), other) => {
            let xf = ax.parse::<f64>().ok()?;
            let yf = match other {
                Value::Integer(n) => n as f64,
                Value::Real(r) => r,
                Value::Rational { num, den } => (num as f64) / (den as f64),
                _ => return None,
            };
            Some(Value::BigReal(format_float(xf - yf)))
        }
        (other, Value::BigReal(by)) => {
            let xf = match other {
                Value::Integer(n) => n as f64,
                Value::Real(r) => r,
                Value::Rational { num, den } => (num as f64) / (den as f64),
                _ => return None,
            };
            let yf = by.parse::<f64>().ok()?;
            Some(Value::BigReal(format_float(xf - yf)))
        }
        (
            Value::PackedArray { shape: s1, data: d1 },
            Value::PackedArray { shape: s2, data: d2 },
        ) => {
            // broadcast-aware elementwise subtraction
            broadcast_elementwise(&s1, &d1, &s2, &d2, |x, y| x - y)
        }
        (Value::PackedArray { shape, data }, other) => {
            if let Some(s) = to_f64_scalar(&other) {
                let mut out = data.clone();
                for x in &mut out {
                    *x -= s;
                }
                Some(Value::PackedArray { shape, data: out })
            } else {
                None
            }
        }
        (other, Value::PackedArray { shape, data }) => {
            if let Some(s) = to_f64_scalar(&other) {
                let mut out = data.clone();
                for x in &mut out {
                    *x = s - *x;
                }
                Some(Value::PackedArray { shape, data: out })
            } else {
                None
            }
        }
        (Value::Rational { num: n1, den: d1 }, Value::Rational { num: n2, den: d2 }) => {
            Some(rat_value(n1 * d2 - n2 * d1, d1 * d2))
        }
        (Value::Integer(x), Value::Rational { num, den }) => Some(rat_value(x * den - num, den)),
        (Value::Rational { num, den }, Value::Integer(x)) => Some(rat_value(num - x * den, den)),
        (Value::Real(x), Value::Rational { num, den }) => {
            Some(Value::Real(x - (num as f64) / (den as f64)))
        }
        (Value::Rational { num, den }, Value::Real(x)) => {
            Some(Value::Real((num as f64) / (den as f64) - x))
        }
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
            Some(Value::Complex {
                re: Box::new(Value::Expr {
                    head: Box::new(Value::Symbol("Minus".into())),
                    args: vec![rr],
                }),
                im: Box::new(Value::Expr {
                    head: Box::new(Value::Symbol("Minus".into())),
                    args: vec![*im],
                }),
            })
        }
        _ => None,
    }
}

fn div_numeric(a: Value, b: Value) -> Option<Value> {
    match (a, b) {
        (
            Value::PackedArray { shape: s1, data: d1 },
            Value::PackedArray { shape: s2, data: d2 },
        ) => broadcast_elementwise(&s1, &d1, &s2, &d2, |x, y| x / y),
        (Value::PackedArray { shape, data }, other) => {
            if let Some(s) = to_f64_scalar(&other) {
                let mut out = data.clone();
                for x in &mut out {
                    *x /= s;
                }
                Some(Value::PackedArray { shape, data: out })
            } else {
                None
            }
        }
        (other, Value::PackedArray { shape, data }) => {
            if let Some(s) = to_f64_scalar(&other) {
                let mut out = data.clone();
                for x in &mut out {
                    *x = s / *x;
                }
                Some(Value::PackedArray { shape, data: out })
            } else {
                None
            }
        }
        (Value::Integer(x), Value::Integer(y)) => {
            if y == 0 {
                None
            } else {
                Some(rat_value(x, y))
            }
        }
        (Value::Real(x), Value::Real(y)) => Some(Value::Real(x / y)),
        (Value::Integer(x), Value::Real(y)) => Some(Value::Real((x as f64) / y)),
        (Value::Real(x), Value::Integer(y)) => Some(Value::Real(x / (y as f64))),
        (Value::BigReal(ax), Value::BigReal(by)) => {
            let s = bigreal_binop(&ax, &by, |a, b| a / b)?;
            Some(Value::BigReal(s))
        }
        (Value::BigReal(ax), other) => {
            let xf = ax.parse::<f64>().ok()?;
            let yf = match other {
                Value::Integer(n) => n as f64,
                Value::Real(r) => r,
                Value::Rational { num, den } => (num as f64) / (den as f64),
                _ => return None,
            };
            Some(Value::BigReal(format_float(xf / yf)))
        }
        (other, Value::BigReal(by)) => {
            let xf = match other {
                Value::Integer(n) => n as f64,
                Value::Real(r) => r,
                Value::Rational { num, den } => (num as f64) / (den as f64),
                _ => return None,
            };
            let yf = by.parse::<f64>().ok()?;
            Some(Value::BigReal(format_float(xf / yf)))
        }
        (Value::Rational { num: n1, den: d1 }, Value::Rational { num: n2, den: d2 }) => {
            if n2 == 0 {
                None
            } else {
                Some(rat_value(n1 * d2, d1 * n2))
            }
        }
        (Value::Integer(x), Value::Rational { num, den }) => {
            if num == 0 {
                return None;
            }
            Some(rat_value(x * den, num))
        }
        (Value::Rational { num, den }, Value::Integer(x)) => {
            if x == 0 {
                return None;
            }
            Some(rat_value(num, den * x))
        }
        (Value::Real(x), Value::Rational { num, den }) => {
            if num == 0 {
                None
            } else {
                Some(Value::Real(x / ((num as f64) / (den as f64))))
            }
        }
        (Value::Rational { num, den }, Value::Real(x)) => {
            Some(Value::Real(((num as f64) / (den as f64)) / x))
        }
        (Value::Complex { re: ar, im: ai }, Value::Integer(y)) => {
            if y == 0 {
                return None;
            }
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
        Value::Rational { num, den } => {
            if *den != 0 {
                Some((*num as f64) / (*den as f64))
            } else {
                None
            }
        }
        Value::BigReal(s) => s.parse::<f64>().ok(),
        Value::Symbol(s) => match s.as_str() {
            "Pi" => Some(std::f64::consts::PI),
            "E" => Some(std::f64::consts::E),
            "Tau" => Some(std::f64::consts::TAU),
            "Degree" => Some(std::f64::consts::PI / 180.0),
            _ => None,
        },
        Value::Expr { head, args } => {
            if let Value::Symbol(name) = &**head {
                match (name.as_str(), args.as_slice()) {
                    ("Plus", list) => {
                        let mut acc = 0.0;
                        for a in list {
                            acc += to_f64_scalar(a)?;
                        }
                        Some(acc)
                    }
                    ("Times", list) => {
                        let mut acc = 1.0;
                        for a in list {
                            acc *= to_f64_scalar(a)?;
                        }
                        Some(acc)
                    }
                    ("Minus", [a]) => Some(-to_f64_scalar(a)?),
                    ("Minus", [a, b]) => Some(to_f64_scalar(a)? - to_f64_scalar(b)?),
                    ("Divide", [a, b]) => Some(to_f64_scalar(a)? / to_f64_scalar(b)?),
                    ("Power", [a, b]) => Some(to_f64_scalar(a)?.powf(to_f64_scalar(b)?)),
                    _ => None,
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

fn broadcast_elementwise(
    s1: &Vec<usize>,
    d1: &Vec<f64>,
    s2: &Vec<usize>,
    d2: &Vec<f64>,
    op: fn(f64, f64) -> f64,
) -> Option<Value> {
    let ndim = std::cmp::max(s1.len(), s2.len());
    let mut sh1 = vec![1; ndim];
    let mut sh2 = vec![1; ndim];
    sh1[ndim - s1.len()..].clone_from_slice(&s1);
    sh2[ndim - s2.len()..].clone_from_slice(&s2);
    let mut out_shape: Vec<usize> = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let a = sh1[i];
        let b = sh2[i];
        if a == b {
            out_shape.push(a);
        } else if a == 1 {
            out_shape.push(b);
        } else if b == 1 {
            out_shape.push(a);
        } else {
            return None;
        }
    }
    let total: usize = out_shape.iter().product();
    let strides = |shape: &Vec<usize>| -> Vec<usize> {
        let mut st = vec![0; ndim];
        let mut acc = 1usize;
        for i in (0..ndim).rev() {
            st[i] = acc;
            acc *= shape[i];
        }
        st
    };
    let st1 = strides(&sh1);
    let st2 = strides(&sh2);
    let mut out = Vec::with_capacity(total);
    for idx in 0..total {
        // convert idx to multi-index
        let mut rem = idx;
        let mut off1 = 0usize;
        let mut off2 = 0usize;
        for i in 0..ndim {
            let _dim = out_shape[i];
            let coord = rem / (out_shape[i + 1..].iter().product::<usize>().max(1));
            rem %= out_shape[i + 1..].iter().product::<usize>().max(1);
            let c1 = if sh1[i] == 1 { 0 } else { coord };
            let c2 = if sh2[i] == 1 { 0 } else { coord };
            off1 += c1 * st1[i];
            off2 += c2 * st2[i];
        }
        out.push(op(d1[off1], d2[off2]));
    }
    Some(Value::PackedArray { shape: out_shape, data: out })
}

fn pow_numeric(base: Value, exp: Value) -> Option<Value> {
    match (base.clone(), exp) {
        (Value::Integer(a), Value::Integer(e)) => {
            if e >= 0 {
                Some(Value::Integer(a.pow(e as u32)))
            } else {
                let (rn, rd) = reduce_rat(1, a.pow((-e) as u32));
                Some(Value::Rational { num: rn, den: rd })
            }
        }
        (Value::Real(a), Value::Integer(e)) => Some(Value::Real(a.powi(e as i32))),
        (Value::BigReal(ax), Value::Integer(e)) => {
            let xf = ax.parse::<f64>().ok()?;
            Some(Value::BigReal(format_float(xf.powi(e as i32))))
        }
        (Value::Rational { num, den }, Value::Integer(e)) => {
            if e >= 0 {
                Some(rat_value(num.pow(e as u32), den.pow(e as u32)))
            } else {
                let p = (-e) as u32;
                Some(rat_value(den.pow(p), num.pow(p)))
            }
        }
        (Value::Complex { .. }, Value::Integer(e)) => {
            let mut k = e.abs();
            if k == 0 {
                return Some(Value::Integer(1));
            }
            // repeated multiplication (simple)
            let mut acc: Option<Value> = None;
            let mut base_opt: Option<Value> = None;
            if let (Value::Complex { .. }, _) = (base.clone(), e) {
                base_opt = Some(base.clone());
            }
            while k > 0 {
                if acc.is_none() {
                    acc = base_opt.clone();
                    k -= 1;
                    continue;
                }
                acc = Some(mul_numeric(acc.unwrap(), base_opt.clone().unwrap())?);
                k -= 1;
            }
            if e < 0 {
                div_numeric(Value::Integer(1), acc.unwrap())
            } else {
                acc
            }
        }
        _ => None,
    }
}

fn plus(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Integer(0);
    }
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
                rest.extend_from_slice(&args[i + 1..]);
                return Value::Expr { head: Box::new(Value::Symbol("Plus".into())), args: rest };
            }
        }
    }
    acc
}

fn times(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Integer(1);
    }
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
                rest.extend_from_slice(&args[i + 1..]);
                return Value::Expr { head: Box::new(Value::Symbol("Times".into())), args: rest };
            }
        }
    }
    acc
}

fn minus(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [a, b] => sub_numeric(a.clone(), b.clone()).unwrap_or(Value::Expr {
            head: Box::new(Value::Symbol("Minus".into())),
            args: vec![a.clone(), b.clone()],
        }),
        [a] => sub_numeric(Value::Integer(0), a.clone()).unwrap_or(Value::Expr {
            head: Box::new(Value::Symbol("Minus".into())),
            args: vec![a.clone()],
        }),
        other => {
            Value::Expr { head: Box::new(Value::Symbol("Minus".into())), args: other.to_vec() }
        }
    }
}

fn divide(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [a, b] => div_numeric(a.clone(), b.clone()).unwrap_or(Value::Expr {
            head: Box::new(Value::Symbol("Divide".into())),
            args: vec![a.clone(), b.clone()],
        }),
        other => {
            Value::Expr { head: Box::new(Value::Symbol("Divide".into())), args: other.to_vec() }
        }
    }
}

fn power(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [a, b] => pow_numeric(a.clone(), b.clone()).unwrap_or(Value::Expr {
            head: Box::new(Value::Symbol("Power".into())),
            args: vec![a.clone(), b.clone()],
        }),
        other => {
            Value::Expr { head: Box::new(Value::Symbol("Power".into())), args: other.to_vec() }
        }
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
                    if rt.saturating_mul(rt) == sum {
                        Value::Integer(rt)
                    } else {
                        Value::Real((sum as f64).sqrt())
                    }
                }
                _ => {
                    fn to_f64(v: &Value) -> Option<f64> {
                        match v {
                            Value::Integer(n) => Some(*n as f64),
                            Value::Real(x) => Some(*x),
                            Value::Rational { num, den } => {
                                if *den != 0 {
                                    Some((*num as f64) / (*den as f64))
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        }
                    }
                    if let (Some(ar), Some(ai)) = (to_f64(&re), to_f64(&im)) {
                        Value::Real(((ar * ar) + (ai * ai)).sqrt())
                    } else {
                        Value::Expr {
                            head: Box::new(Value::Symbol("Abs".into())),
                            args: args.to_vec(),
                        }
                    }
                }
            }
        }
        other => Value::Expr { head: Box::new(Value::Symbol("Abs".into())), args: other.to_vec() },
    }
}

fn min_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() == 1 {
        match ev.eval(args[0].clone()) {
            Value::List(items) => return min_over_iter(items.into_iter()),
            other => {
                return Value::Expr {
                    head: Box::new(Value::Symbol("Min".into())),
                    args: vec![other],
                }
            }
        }
    }
    let evald: Vec<Value> = args.into_iter().map(|a| ev.eval(a)).collect();
    min_over_iter(evald.into_iter())
}

fn max_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() == 1 {
        match ev.eval(args[0].clone()) {
            Value::List(items) => return max_over_iter(items.into_iter()),
            other => {
                return Value::Expr {
                    head: Box::new(Value::Symbol("Max".into())),
                    args: vec![other],
                }
            }
        }
    }
    let evald: Vec<Value> = args.into_iter().map(|a| ev.eval(a)).collect();
    max_over_iter(evald.into_iter())
}

fn min_over_iter<I: Iterator<Item = Value>>(iter: I) -> Value {
    let mut have = false;
    let mut use_real = false;
    let mut cur_i: i64 = 0;
    let mut cur_f: f64 = 0.0;
    for v in iter {
        match v {
            Value::Integer(n) => {
                if !have {
                    have = true;
                    cur_i = n;
                    cur_f = n as f64;
                } else {
                    if use_real {
                        if (n as f64) < cur_f {
                            cur_f = n as f64;
                        }
                    } else if n < cur_i {
                        cur_i = n;
                    }
                }
            }
            Value::Real(x) => {
                if !have {
                    have = true;
                    cur_f = x;
                    cur_i = x as i64;
                    use_real = true;
                } else {
                    use_real = true;
                    if x < cur_f {
                        cur_f = x;
                    }
                }
            }
            other => {
                return Value::Expr {
                    head: Box::new(Value::Symbol("Min".into())),
                    args: vec![other],
                }
            }
        }
    }
    if !have {
        Value::Expr { head: Box::new(Value::Symbol("Min".into())), args: vec![] }
    } else if use_real {
        Value::Real(cur_f)
    } else {
        Value::Integer(cur_i)
    }
}

fn max_over_iter<I: Iterator<Item = Value>>(iter: I) -> Value {
    let mut have = false;
    let mut use_real = false;
    let mut cur_i: i64 = 0;
    let mut cur_f: f64 = 0.0;
    for v in iter {
        match v {
            Value::Integer(n) => {
                if !have {
                    have = true;
                    cur_i = n;
                    cur_f = n as f64;
                } else {
                    if use_real {
                        if (n as f64) > cur_f {
                            cur_f = n as f64;
                        }
                    } else if n > cur_i {
                        cur_i = n;
                    }
                }
            }
            Value::Real(x) => {
                if !have {
                    have = true;
                    cur_f = x;
                    cur_i = x as i64;
                    use_real = true;
                } else {
                    use_real = true;
                    if x > cur_f {
                        cur_f = x;
                    }
                }
            }
            other => {
                return Value::Expr {
                    head: Box::new(Value::Symbol("Max".into())),
                    args: vec![other],
                }
            }
        }
    }
    if !have {
        Value::Expr { head: Box::new(Value::Symbol("Max".into())), args: vec![] }
    } else if use_real {
        Value::Real(cur_f)
    } else {
        Value::Integer(cur_i)
    }
}

// ---------- New math ops implementations ----------

fn floor_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] => Value::Integer(*n),
        [v] => match to_f64_scalar(v) {
            Some(x) => Value::Integer(x.floor() as i64),
            None => map_unary_packed("Floor", v.clone(), |x| x.floor()),
        },
        other => {
            Value::Expr { head: Box::new(Value::Symbol("Floor".into())), args: other.to_vec() }
        }
    }
}

fn ceiling_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] => Value::Integer(*n),
        [v] => match to_f64_scalar(v) {
            Some(x) => Value::Integer(x.ceil() as i64),
            None => map_unary_packed("Ceiling", v.clone(), |x| x.ceil()),
        },
        other => {
            Value::Expr { head: Box::new(Value::Symbol("Ceiling".into())), args: other.to_vec() }
        }
    }
}

fn round_half_to_even(x: f64) -> f64 {
    let f = x.floor();
    let frac = x - f;
    if frac < 0.5 {
        f
    } else if frac > 0.5 {
        f + 1.0
    } else {
        // exactly half; choose even
        if (f as i64) % 2 == 0 {
            f
        } else {
            f + 1.0
        }
    }
}

fn round_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] => Value::Integer(*n),
        [v] => match to_f64_scalar(v) {
            Some(x) => Value::Integer(round_half_to_even(x) as i64),
            None => map_unary_packed("Round", v.clone(), |x| round_half_to_even(x)),
        },
        other => {
            Value::Expr { head: Box::new(Value::Symbol("Round".into())), args: other.to_vec() }
        }
    }
}

fn trunc_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] => Value::Integer(*n),
        [v] => match to_f64_scalar(v) {
            Some(x) => Value::Integer(x.trunc() as i64),
            None => map_unary_packed("Trunc", v.clone(), |x| x.trunc()),
        },
        other => {
            Value::Expr { head: Box::new(Value::Symbol("Trunc".into())), args: other.to_vec() }
        }
    }
}

fn floor_div(a: i64, b: i64) -> i64 {
    let mut q = a / b;
    let r = a % b;
    if (r != 0) && ((r > 0) != (b > 0)) {
        q -= 1;
    }
    q
}

fn mod_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(a), Value::Integer(b)] => {
            if *b == 0 {
                Value::Expr {
                    head: Box::new(Value::Symbol("Mod".into())),
                    args: vec![Value::Integer(*a), Value::Integer(*b)],
                }
            } else {
                Value::Integer(a.rem_euclid(*b))
            }
        }
        [a, b] => match (to_f64_scalar(a), to_f64_scalar(b)) {
            (Some(x), Some(y)) if y != 0.0 => Value::Real(x.rem_euclid(y)),
            _ => Value::Expr {
                head: Box::new(Value::Symbol("Mod".into())),
                args: vec![a.clone(), b.clone()],
            },
        },
        other => Value::Expr { head: Box::new(Value::Symbol("Mod".into())), args: other.to_vec() },
    }
}

fn quotient_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(a), Value::Integer(b)] => {
            if *b == 0 {
                Value::Expr {
                    head: Box::new(Value::Symbol("Quotient".into())),
                    args: vec![Value::Integer(*a), Value::Integer(*b)],
                }
            } else {
                Value::Integer(floor_div(*a, *b))
            }
        }
        [a, b] => match (to_f64_scalar(a), to_f64_scalar(b)) {
            (Some(x), Some(y)) if y != 0.0 => Value::Integer((x / y).floor() as i64),
            _ => Value::Expr {
                head: Box::new(Value::Symbol("Quotient".into())),
                args: vec![a.clone(), b.clone()],
            },
        },
        other => {
            Value::Expr { head: Box::new(Value::Symbol("Quotient".into())), args: other.to_vec() }
        }
    }
}

fn remainder_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(a), Value::Integer(b)] => {
            if *b == 0 {
                Value::Expr {
                    head: Box::new(Value::Symbol("Remainder".into())),
                    args: vec![Value::Integer(*a), Value::Integer(*b)],
                }
            } else {
                let q = floor_div(*a, *b);
                Value::Integer(a - b * q)
            }
        }
        [a, b] => match (to_f64_scalar(a), to_f64_scalar(b)) {
            (Some(x), Some(y)) if y != 0.0 => {
                let q = (x / y).floor();
                Value::Real(x - y * q)
            }
            _ => Value::Expr {
                head: Box::new(Value::Symbol("Remainder".into())),
                args: vec![a.clone(), b.clone()],
            },
        },
        other => {
            Value::Expr { head: Box::new(Value::Symbol("Remainder".into())), args: other.to_vec() }
        }
    }
}

fn divmod_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(a), Value::Integer(b)] => {
            if *b == 0 {
                Value::Expr {
                    head: Box::new(Value::Symbol("DivMod".into())),
                    args: vec![Value::Integer(*a), Value::Integer(*b)],
                }
            } else {
                let q = floor_div(*a, *b);
                let r = a - b * q;
                Value::List(vec![Value::Integer(q), Value::Integer(r)])
            }
        }
        [a, b] => match (to_f64_scalar(a), to_f64_scalar(b)) {
            (Some(x), Some(y)) if y != 0.0 => {
                let q = (x / y).floor();
                let r = x - y * q;
                Value::List(vec![Value::Integer(q as i64), Value::Real(r)])
            }
            _ => Value::Expr {
                head: Box::new(Value::Symbol("DivMod".into())),
                args: vec![a.clone(), b.clone()],
            },
        },
        other => {
            Value::Expr { head: Box::new(Value::Symbol("DivMod".into())), args: other.to_vec() }
        }
    }
}

fn sqrt_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    fn isqrt(n: i64) -> Option<i64> {
        if n < 0 {
            return None;
        }
        let r = (n as f64).sqrt() as i64;
        if r * r == n {
            Some(r)
        } else if (r + 1) * (r + 1) == n {
            Some(r + 1)
        } else {
            None
        }
    }
    match args.as_slice() {
        [Value::Integer(n)] => {
            if *n < 0 {
                Value::Expr {
                    head: Box::new(Value::Symbol("Sqrt".into())),
                    args: vec![Value::Integer(*n)],
                }
            } else if let Some(r) = isqrt(*n) {
                Value::Integer(r)
            } else {
                Value::Real((*n as f64).sqrt())
            }
        }
        [Value::Rational { num, den }] => {
            if *num < 0 {
                Value::Expr {
                    head: Box::new(Value::Symbol("Sqrt".into())),
                    args: vec![Value::Rational { num: *num, den: *den }],
                }
            } else {
                let nr = isqrt(*num);
                let dr = isqrt(*den);
                match (nr, dr) {
                    (Some(a), Some(b)) => rat_value(a, b),
                    _ => Value::Real(((*num as f64) / (*den as f64)).sqrt()),
                }
            }
        }
        [v] => match to_f64_scalar(v) {
            Some(x) if x >= 0.0 => Value::Real(x.sqrt()),
            _ => map_unary_packed("Sqrt", v.clone(), |x| x.sqrt()),
        },
        other => Value::Expr { head: Box::new(Value::Symbol("Sqrt".into())), args: other.to_vec() },
    }
}

fn exp_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v] => match to_f64_scalar(v) {
            Some(x) => Value::Real(x.exp()),
            _ => map_unary_packed("Exp", v.clone(), |x| x.exp()),
        },
        other => Value::Expr { head: Box::new(Value::Symbol("Exp".into())), args: other.to_vec() },
    }
}

fn log_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v] => match to_f64_scalar(v) {
            Some(x) if x > 0.0 => Value::Real(x.ln()),
            _ => Value::Expr { head: Box::new(Value::Symbol("Log".into())), args: vec![v.clone()] },
        },
        [v, base] => match (to_f64_scalar(v), to_f64_scalar(base)) {
            (Some(x), Some(b)) if x > 0.0 && b > 0.0 && b != 1.0 => Value::Real(x.log(b)),
            _ => Value::Expr {
                head: Box::new(Value::Symbol("Log".into())),
                args: vec![v.clone(), base.clone()],
            },
        },
        other => Value::Expr { head: Box::new(Value::Symbol("Log".into())), args: other.to_vec() },
    }
}

fn sin_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v] => match to_f64_scalar(v) {
            Some(x) => Value::Real(x.sin()),
            _ => map_unary_packed("Sin", v.clone(), |x| x.sin()),
        },
        other => Value::Expr { head: Box::new(Value::Symbol("Sin".into())), args: other.to_vec() },
    }
}

fn cos_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v] => match to_f64_scalar(v) {
            Some(x) => Value::Real(x.cos()),
            _ => map_unary_packed("Cos", v.clone(), |x| x.cos()),
        },
        other => Value::Expr { head: Box::new(Value::Symbol("Cos".into())), args: other.to_vec() },
    }
}

fn tan_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v] => match to_f64_scalar(v) {
            Some(x) => Value::Real(x.tan()),
            _ => map_unary_packed("Tan", v.clone(), |x| x.tan()),
        },
        other => Value::Expr { head: Box::new(Value::Symbol("Tan".into())), args: other.to_vec() },
    }
}

fn asin_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v] => match to_f64_scalar(v) {
            Some(x) => Value::Real(x.asin()),
            _ => map_unary_packed("ASin", v.clone(), |x| x.asin()),
        },
        other => Value::Expr { head: Box::new(Value::Symbol("ASin".into())), args: other.to_vec() },
    }
}

fn acos_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v] => match to_f64_scalar(v) {
            Some(x) => Value::Real(x.acos()),
            _ => map_unary_packed("ACos", v.clone(), |x| x.acos()),
        },
        other => Value::Expr { head: Box::new(Value::Symbol("ACos".into())), args: other.to_vec() },
    }
}

fn atan_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v] => match to_f64_scalar(v) {
            Some(x) => Value::Real(x.atan()),
            _ => map_unary_packed("ATan", v.clone(), |x| x.atan()),
        },
        other => Value::Expr { head: Box::new(Value::Symbol("ATan".into())), args: other.to_vec() },
    }
}

fn atan2_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [y, x] => match (to_f64_scalar(y), to_f64_scalar(x)) {
            (Some(a), Some(b)) => Value::Real(a.atan2(b)),
            _ => Value::Expr {
                head: Box::new(Value::Symbol("ATan2".into())),
                args: vec![y.clone(), x.clone()],
            },
        },
        other => {
            Value::Expr { head: Box::new(Value::Symbol("ATan2".into())), args: other.to_vec() }
        }
    }
}

fn nthroot_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(x), Value::Integer(n)] if *n > 0 => {
            let n_u = *n as u32;
            if *x == 0 {
                return Value::Integer(0);
            }
            if *x > 0 {
                // exact check for perfect power
                let xr = (*x as f64).powf(1.0 / (*n as f64));
                let r = xr.round() as i64;
                let mut acc: i128 = 1;
                for _ in 0..n_u {
                    acc = acc.saturating_mul(r as i128);
                }
                if acc == *x as i128 {
                    Value::Integer(r)
                } else {
                    Value::Real((*x as f64).powf(1.0 / (*n as f64)))
                }
            } else {
                if n % 2 != 0 {
                    // odd root of negative is negative root of abs
                    let xr = (-*x as f64).powf(1.0 / (*n as f64));
                    Value::Real(-xr)
                } else {
                    Value::Expr {
                        head: Box::new(Value::Symbol("NthRoot".into())),
                        args: vec![Value::Integer(*x), Value::Integer(*n)],
                    }
                }
            }
        }
        [a, b] => match (to_f64_scalar(a), to_f64_scalar(b)) {
            (Some(x), Some(n)) if n != 0.0 => Value::Real(x.powf(1.0 / n)),
            _ => Value::Expr {
                head: Box::new(Value::Symbol("NthRoot".into())),
                args: vec![a.clone(), b.clone()],
            },
        },
        other => {
            Value::Expr { head: Box::new(Value::Symbol("NthRoot".into())), args: other.to_vec() }
        }
    }
}

fn total_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() == 1 {
        match ev.eval(args[0].clone()) {
            Value::List(items) => {
                if items.is_empty() {
                    return Value::Integer(0);
                }
                let mut acc = items[0].clone();
                for a in items.into_iter().skip(1) {
                    match add_numeric(acc.clone(), a.clone()) {
                        Some(v) => acc = v,
                        None => {
                            return Value::Expr {
                                head: Box::new(Value::Symbol("Total".into())),
                                args: vec![Value::List(vec![])],
                            }
                        }
                    }
                }
                acc
            }
            other => other,
        }
    } else if !args.is_empty() {
        plus(ev, args)
    } else {
        Value::Integer(0)
    }
}

fn mean_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let (items, n) = if args.len() == 1 {
        match ev.eval(args[0].clone()) {
            Value::List(xs) => {
                let n = xs.len();
                (xs, n)
            }
            other => (vec![other], 1),
        }
    } else {
        let n = args.len();
        let items = args.into_iter().map(|a| ev.eval(a)).collect::<Vec<_>>();
        (items, n)
    };
    if n == 0 {
        return Value::Expr {
            head: Box::new(Value::Symbol("Mean".into())),
            args: vec![Value::List(vec![])],
        };
    }
    let sum = total_fn(ev, vec![Value::List(items)]);
    div_numeric(sum, Value::Integer(n as i64))
        .unwrap_or(Value::Expr { head: Box::new(Value::Symbol("Mean".into())), args: vec![] })
}

fn median_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let items = match args.as_slice() {
        [v] => match ev.eval(v.clone()) {
            Value::List(xs) => xs,
            other => vec![other],
        },
        _ => args.into_iter().map(|a| ev.eval(a)).collect(),
    };
    if items.is_empty() {
        return Value::Expr {
            head: Box::new(Value::Symbol("Median".into())),
            args: vec![Value::List(vec![])],
        };
    }
    let mut nums: Vec<(bool, f64, Value)> = Vec::with_capacity(items.len());
    for v in items {
        if let Some(x) = to_f64_scalar(&v) {
            nums.push((matches!(v, Value::Integer(_) | Value::Rational { .. }), x, v));
        } else {
            return Value::Expr { head: Box::new(Value::Symbol("Median".into())), args: vec![] };
        }
    }
    nums.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let n = nums.len();
    if n % 2 == 1 {
        nums[n / 2].2.clone()
    } else {
        // average of two middles; try exact if both are exact ints/rats
        let a = &nums[n / 2 - 1].2;
        let b = &nums[n / 2].2;
        match add_numeric(a.clone(), b.clone()).and_then(|s| div_numeric(s, Value::Integer(2))) {
            Some(v) => v,
            None => Value::Real((nums[n / 2 - 1].1 + nums[n / 2].1) / 2.0),
        }
    }
}

fn variance_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let items = match args.as_slice() {
        [v] => match ev.eval(v.clone()) {
            Value::List(xs) => xs,
            other => vec![other],
        },
        _ => args.into_iter().map(|a| ev.eval(a)).collect(),
    };
    if items.is_empty() {
        return Value::Expr {
            head: Box::new(Value::Symbol("Variance".into())),
            args: vec![Value::List(vec![])],
        };
    }
    let mut vals: Vec<f64> = Vec::with_capacity(items.len());
    for v in items {
        if let Some(x) = to_f64_scalar(&v) {
            vals.push(x);
        } else {
            return Value::Expr { head: Box::new(Value::Symbol("Variance".into())), args: vec![] };
        }
    }
    let n = vals.len() as f64;
    let mean = vals.iter().copied().sum::<f64>() / n;
    let var = vals
        .iter()
        .map(|x| {
            let d = *x - mean;
            d * d
        })
        .sum::<f64>()
        / n; // population variance
    Value::Real(var)
}

fn stddev_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match variance_fn(ev, args) {
        Value::Real(v) => Value::Real(v.sqrt()),
        other => other,
    }
}

fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
    if a == 0 {
        return b.abs();
    }
    if b == 0 {
        return a.abs();
    }
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}
fn lcm_i64(a: i64, b: i64) -> i64 {
    if a == 0 || b == 0 {
        0
    } else {
        (a / gcd_i64(a, b)).saturating_mul(b).abs()
    }
}

fn gcd_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Integer(0);
    }
    let mut acc: Option<i64> = None;
    for v in args {
        match v {
            Value::Integer(n) => {
                acc = Some(match acc {
                    None => n.abs(),
                    Some(a) => gcd_i64(a, n),
                });
            }
            _ => return Value::Expr { head: Box::new(Value::Symbol("GCD".into())), args: vec![] },
        }
    }
    Value::Integer(acc.unwrap_or(0))
}

fn lcm_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Integer(1);
    }
    let mut acc: Option<i64> = None;
    for v in args {
        match v {
            Value::Integer(n) => {
                acc = Some(match acc {
                    None => n.abs(),
                    Some(a) => lcm_i64(a, n),
                });
            }
            _ => return Value::Expr { head: Box::new(Value::Symbol("LCM".into())), args: vec![] },
        }
    }
    Value::Integer(acc.unwrap_or(1))
}

fn factorial_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] if *n >= 0 => {
            let mut acc: i128 = 1;
            for k in 2..=(*n as i128) {
                acc = acc.saturating_mul(k);
                if acc > i64::MAX as i128 {
                    return Value::Expr {
                        head: Box::new(Value::Symbol("Factorial".into())),
                        args: vec![Value::Integer(*n)],
                    };
                }
            }
            Value::Integer(acc as i64)
        }
        [v] => {
            Value::Expr { head: Box::new(Value::Symbol("Factorial".into())), args: vec![v.clone()] }
        }
        other => {
            Value::Expr { head: Box::new(Value::Symbol("Factorial".into())), args: other.to_vec() }
        }
    }
}

fn binomial_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n), Value::Integer(k)] if *n >= 0 && *k >= 0 && *k <= *n => {
            let n0 = *n;
            let k0 = *k;
            let k = std::cmp::min(k0 as i128, (n0 as i128) - (k0 as i128));
            let mut num: i128 = 1;
            let mut den: i128 = 1;
            for i in 1..=k {
                num = num.saturating_mul((*n as i128) - k + i);
                den = den.saturating_mul(i);
                let g = gcd_i128(num, den);
                num /= g;
                den /= g;
                if num > i64::MAX as i128 {
                    return Value::Expr {
                        head: Box::new(Value::Symbol("Binomial".into())),
                        args: vec![Value::Integer(n0), Value::Integer(k0)],
                    };
                }
            }
            if den != 1 {
                let g = gcd_i128(num, den);
                return rat_value((num / g) as i64, (den / g) as i64);
            }
            Value::Integer(num as i64)
        }
        [a, b] => Value::Expr {
            head: Box::new(Value::Symbol("Binomial".into())),
            args: vec![a.clone(), b.clone()],
        },
        other => {
            Value::Expr { head: Box::new(Value::Symbol("Binomial".into())), args: other.to_vec() }
        }
    }
}

fn gcd_i128(mut a: i128, mut b: i128) -> i128 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

fn to_degrees_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v] => match to_f64_scalar(v) {
            Some(x) => Value::Real(x * 180.0 / std::f64::consts::PI),
            _ => map_unary_packed("ToDegrees", v.clone(), |x| x * 180.0 / std::f64::consts::PI),
        },
        other => {
            Value::Expr { head: Box::new(Value::Symbol("ToDegrees".into())), args: other.to_vec() }
        }
    }
}

fn to_radians_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v] => match to_f64_scalar(v) {
            Some(x) => Value::Real(x * std::f64::consts::PI / 180.0),
            _ => map_unary_packed("ToRadians", v.clone(), |x| x * std::f64::consts::PI / 180.0),
        },
        other => {
            Value::Expr { head: Box::new(Value::Symbol("ToRadians".into())), args: other.to_vec() }
        }
    }
}

fn clip_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [x, Value::List(bounds)] if bounds.len() == 2 => {
            let min_v = to_f64_scalar(&bounds[0]);
            let max_v = to_f64_scalar(&bounds[1]);
            match (to_f64_scalar(x), min_v, max_v) {
                (Some(v), Some(lo), Some(hi)) => Value::Real(v.max(lo).min(hi)),
                _ => Value::Expr {
                    head: Box::new(Value::Symbol("Clip".into())),
                    args: vec![x.clone(), Value::List(bounds.clone())],
                },
            }
        }
        [v] => map_unary_packed("Clip", v.clone(), |x| x),
        other => Value::Expr { head: Box::new(Value::Symbol("Clip".into())), args: other.to_vec() },
    }
}

fn signum_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] => Value::Integer(n.signum()),
        [Value::Rational { num, .. }] => Value::Integer(num.signum()),
        [v] => match to_f64_scalar(v) {
            Some(x) => Value::Integer(if x > 0.0 {
                1
            } else if x < 0.0 {
                -1
            } else {
                0
            }),
            None => map_unary_packed("Signum", v.clone(), |x| {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }),
        },
        other => {
            Value::Expr { head: Box::new(Value::Symbol("Signum".into())), args: other.to_vec() }
        }
    }
}

fn map_unary_packed(head: &str, v: Value, f: fn(f64) -> f64) -> Value {
    match v {
        Value::PackedArray { shape, data } => {
            let out: Vec<f64> = data.into_iter().map(|x| f(x)).collect();
            Value::PackedArray { shape, data: out }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol(head.into())), args: vec![v] },
    }
}
