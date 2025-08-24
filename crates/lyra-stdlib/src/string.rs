use lyra_runtime::{Evaluator};
use lyra_runtime::attrs::Attributes;
use lyra_core::value::Value;

pub fn register_string(ev: &mut Evaluator) {
    ev.register("StringLength", string_length as NativeFn, Attributes::LISTABLE);
    ev.register("ToUpper", to_upper as NativeFn, Attributes::LISTABLE);
    ev.register("ToLower", to_lower as NativeFn, Attributes::LISTABLE);
    ev.register("StringJoin", string_join as NativeFn, Attributes::empty());
    ev.register("StringTrim", string_trim as NativeFn, Attributes::LISTABLE);
    ev.register("StringContains", string_contains as NativeFn, Attributes::empty());
    ev.register("StringSplit", string_split as NativeFn, Attributes::empty());
    ev.register("StartsWith", starts_with as NativeFn, Attributes::LISTABLE);
    ev.register("EndsWith", ends_with as NativeFn, Attributes::LISTABLE);
    ev.register("StringReplace", string_replace as NativeFn, Attributes::empty());
    ev.register("StringReverse", string_reverse as NativeFn, Attributes::LISTABLE);
    ev.register("StringPadLeft", string_pad_left as NativeFn, Attributes::LISTABLE);
    ev.register("StringPadRight", string_pad_right as NativeFn, Attributes::LISTABLE);
}

// Bring NativeFn alias into scope by mirroring lyra-runtime's type
type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

fn string_length(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s)] => Value::Integer(s.chars().count() as i64),
        [other] => match ev.eval(other.clone()) { Value::String(s)=>Value::Integer(s.chars().count() as i64), v=> Value::Expr { head: Box::new(Value::Symbol("StringLength".into())), args: vec![v] } },
        _ => Value::Expr { head: Box::new(Value::Symbol("StringLength".into())), args },
    }
}

fn to_upper(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s)] => Value::String(s.to_uppercase()),
        [other] => match ev.eval(other.clone()) { Value::String(s)=>Value::String(s.to_uppercase()), v=> Value::Expr { head: Box::new(Value::Symbol("ToUpper".into())), args: vec![v] } },
        _ => Value::Expr { head: Box::new(Value::Symbol("ToUpper".into())), args },
    }
}

fn to_lower(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s)] => Value::String(s.to_lowercase()),
        [other] => match ev.eval(other.clone()) { Value::String(s)=>Value::String(s.to_lowercase()), v=> Value::Expr { head: Box::new(Value::Symbol("ToLower".into())), args: vec![v] } },
        _ => Value::Expr { head: Box::new(Value::Symbol("ToLower".into())), args },
    }
}

fn string_join(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::List(parts)] => {
            let mut out = String::new();
            for p in parts { match ev.eval(p.clone()) { Value::String(s)=> out.push_str(&s), v => out.push_str(&lyra_core::pretty::format_value(&v)) } }
            Value::String(out)
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("StringJoin".into())), args },
    }
}

fn string_trim(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s)] => Value::String(s.trim().to_string()),
        [other] => match ev.eval(other.clone()) { Value::String(s)=>Value::String(s.trim().to_string()), v=> Value::Expr { head: Box::new(Value::Symbol("StringTrim".into())), args: vec![v] } },
        _ => Value::Expr { head: Box::new(Value::Symbol("StringTrim".into())), args },
    }
}

fn string_contains(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s), Value::String(sub)] => Value::Boolean(s.contains(sub)),
        [a, b] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            match (aa, bb) {
                (Value::String(s), Value::String(sub)) => Value::Boolean(s.contains(&sub)),
                (aa, bb) => Value::Expr { head: Box::new(Value::Symbol("StringContains".into())), args: vec![aa, bb] }
            }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("StringContains".into())), args },
    }
}

fn string_split(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s)] => {
            let parts: Vec<Value> = s.split_whitespace().map(|p| Value::String(p.to_string())).collect();
            Value::List(parts)
        }
        [Value::String(s), Value::String(d)] => {
            if d.is_empty() {
                Value::List(s.chars().map(|c| Value::String(c.to_string())).collect())
            } else {
                Value::List(s.split(d).map(|p| Value::String(p.to_string())).collect())
            }
        }
        [a] => match ev.eval(a.clone()) { Value::String(s)=> Value::List(s.split_whitespace().map(|p| Value::String(p.to_string())).collect()), v=> Value::Expr { head: Box::new(Value::Symbol("StringSplit".into())), args: vec![v] } },
        [a, b] => {
            let aa = ev.eval(a.clone()); let bb = ev.eval(b.clone());
            match (aa, bb) {
                (Value::String(s), Value::String(d)) => {
                    if d.is_empty() { Value::List(s.chars().map(|c| Value::String(c.to_string())).collect()) }
                    else { Value::List(s.split(&d).map(|p| Value::String(p.to_string())).collect()) }
                }
                (aa, bb) => Value::Expr { head: Box::new(Value::Symbol("StringSplit".into())), args: vec![aa, bb] }
            }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("StringSplit".into())), args },
    }
}

fn starts_with(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s), Value::String(p)] => Value::Boolean(s.starts_with(p)),
        [a, b] => { let aa = ev.eval(a.clone()); let bb = ev.eval(b.clone()); match (aa, bb) { (Value::String(s), Value::String(p)) => Value::Boolean(s.starts_with(&p)), (aa, bb) => Value::Expr { head: Box::new(Value::Symbol("StartsWith".into())), args: vec![aa, bb] } } }
        _ => Value::Expr { head: Box::new(Value::Symbol("StartsWith".into())), args },
    }
}

fn ends_with(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s), Value::String(p)] => Value::Boolean(s.ends_with(p)),
        [a, b] => { let aa = ev.eval(a.clone()); let bb = ev.eval(b.clone()); match (aa, bb) { (Value::String(s), Value::String(p)) => Value::Boolean(s.ends_with(&p)), (aa, bb) => Value::Expr { head: Box::new(Value::Symbol("EndsWith".into())), args: vec![aa, bb] } } }
        _ => Value::Expr { head: Box::new(Value::Symbol("EndsWith".into())), args },
    }
}

fn string_replace(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s), Value::String(from), Value::String(to)] => Value::String(s.replace(from, to)),
        [a, b, c] => {
            let aa = ev.eval(a.clone()); let bb = ev.eval(b.clone()); let cc = ev.eval(c.clone());
            match (aa, bb, cc) {
                (Value::String(s), Value::String(from), Value::String(to)) => Value::String(s.replace(&from, &to)),
                (aa, bb, cc) => Value::Expr { head: Box::new(Value::Symbol("StringReplace".into())), args: vec![aa, bb, cc] }
            }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("StringReplace".into())), args },
    }
}

fn string_reverse(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s)] => Value::String(s.chars().rev().collect()),
        [other] => match ev.eval(other.clone()) { Value::String(s)=> Value::String(s.chars().rev().collect()), v => Value::Expr { head: Box::new(Value::Symbol("StringReverse".into())), args: vec![v] } },
        _ => Value::Expr { head: Box::new(Value::Symbol("StringReverse".into())), args },
    }
}

fn get_first_char(s: &str) -> char { s.chars().next().unwrap_or(' ') }

fn string_pad_left(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s), Value::Integer(n)] => {
            let width = *n as usize;
            let len = s.chars().count();
            if len >= width { return Value::String(s.clone()); }
            let padc = ' ';
            let pads = std::iter::repeat(padc).take(width - len).collect::<String>();
            Value::String(format!("{}{}", pads, s))
        }
        [Value::String(s), Value::Integer(n), Value::String(p)] => {
            let width = *n as usize;
            let len = s.chars().count();
            if len >= width { return Value::String(s.clone()); }
            let padc = get_first_char(p);
            let pads = std::iter::repeat(padc).take(width - len).collect::<String>();
            Value::String(format!("{}{}", pads, s))
        }
        [a, b] => {
            let aa = ev.eval(a.clone()); let bb = ev.eval(b.clone());
            string_pad_left(ev, vec![aa, bb])
        }
        [a, b, c] => {
            let aa = ev.eval(a.clone()); let bb = ev.eval(b.clone()); let cc = ev.eval(c.clone());
            string_pad_left(ev, vec![aa, bb, cc])
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("StringPadLeft".into())), args },
    }
}

fn string_pad_right(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s), Value::Integer(n)] => {
            let width = *n as usize;
            let len = s.chars().count();
            if len >= width { return Value::String(s.clone()); }
            let padc = ' ';
            let pads = std::iter::repeat(padc).take(width - len).collect::<String>();
            Value::String(format!("{}{}", s, pads))
        }
        [Value::String(s), Value::Integer(n), Value::String(p)] => {
            let width = *n as usize;
            let len = s.chars().count();
            if len >= width { return Value::String(s.clone()); }
            let padc = get_first_char(p);
            let pads = std::iter::repeat(padc).take(width - len).collect::<String>();
            Value::String(format!("{}{}", s, pads))
        }
        [a, b] => {
            let aa = ev.eval(a.clone()); let bb = ev.eval(b.clone());
            string_pad_right(ev, vec![aa, bb])
        }
        [a, b, c] => {
            let aa = ev.eval(a.clone()); let bb = ev.eval(b.clone()); let cc = ev.eval(c.clone());
            string_pad_right(ev, vec![aa, bb, cc])
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("StringPadRight".into())), args },
    }
}

