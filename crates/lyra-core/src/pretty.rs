use crate::value::Value;

pub fn format_value(v: &Value) -> String {
    match v {
        Value::Integer(n) => n.to_string(),
        Value::Real(f) => {
            if f.fract() == 0.0 { format!("{:.1}", f) } else { f.to_string() }
        }
        Value::BigReal(s) => s.clone(),
        Value::Rational { num, den } => format!("{}/{}", num, den),
        Value::Complex { re, im } => format!("Complex[{}, {}]", format_value(re), format_value(im)),
        Value::PackedArray { shape, .. } => {
            let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            format!("PackedArray[{{{}}}]", dims.join(", "))
        }
        Value::String(s) => format!("\"{}\"", s),
        Value::Symbol(s) => s.clone(),
        Value::Boolean(b) => if *b { "True".into() } else { "False".into() },
        Value::List(items) => {
            let inner: Vec<String> = items.iter().map(format_value).collect();
            format!("{{{}}}", inner.join(", "))
        }
        Value::Assoc(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let parts: Vec<String> = keys.into_iter().map(|k| format!("\"{}\" -> {}", k, format_value(map.get(k).unwrap()))).collect();
            format!("<|{}|>", parts.join(", "))
        }
        Value::Expr { head, args } => {
            let h = format_value(head);
            let a: Vec<String> = args.iter().map(format_value).collect();
            format!("{}[{}]", h, a.join(", "))
        }
        Value::Slot(n) => match n { Some(k) => format!("#{}", k), None => "#".into() },
        Value::PureFunction { params, body } => {
            if let Some(ps) = params {
                let inside = ps.join(", ");
                format!("({}) => {}", inside, format_value(body))
            } else {
                format!("{}&", format_value(body))
            }
        }
    }
}
