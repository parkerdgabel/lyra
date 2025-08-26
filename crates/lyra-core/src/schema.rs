use crate::value::Value;
use std::collections::HashSet;

pub fn schema_of(v: &Value) -> Value {
    match v {
        Value::Assoc(m) => {
            let keys: HashSet<&str> = m.keys().map(|s| s.as_str()).collect();
            let name = if ["frequencies", "magnitudes", "method"].iter().all(|k| keys.contains(k)) {
                "SpectralResult/v1"
            } else if keys.contains("filterType") && keys.contains("filteredSignal") {
                "FilterResult/v1"
            } else {
                "Association/v1"
            };
            Value::assoc(vec![
                ("name", Value::String(name.to_string())),
                ("keys", Value::List(m.keys().cloned().map(Value::String).collect())),
            ])
        }
        Value::Integer(_) => Value::String("Integer/v1".into()),
        Value::Real(_) => Value::String("Real/v1".into()),
        Value::BigReal(_) => Value::String("BigReal/v1".into()),
        Value::Rational { .. } => Value::String("Rational/v1".into()),
        Value::Complex { .. } => Value::String("Complex/v1".into()),
        Value::PackedArray { .. } => Value::String("PackedArray/v1".into()),
        Value::String(_) => Value::String("String/v1".into()),
        Value::Symbol(_) => Value::String("Symbol/v1".into()),
        Value::Boolean(_) => Value::String("Boolean/v1".into()),
        Value::List(_) => Value::String("List/v1".into()),
        Value::Expr { .. } => Value::String("Expr/v1".into()),
        Value::Slot(_) => Value::String("Slot/v1".into()),
        Value::PureFunction { .. } => Value::String("PureFunction/v1".into()),
    }
}
