use crate::eval::Evaluator;
use lyra_core::value::Value;

pub(crate) fn env_keys(ev: &Evaluator) -> Vec<String> {
    ev.env.keys().cloned().collect()
}

pub(crate) fn set_env(ev: &mut Evaluator, key: &str, v: Value) {
    ev.env.insert(key.to_string(), v);
}

pub(crate) fn get_env(ev: &Evaluator, key: &str) -> Option<Value> {
    ev.env.get(key).cloned()
}

pub(crate) fn unset_env(ev: &mut Evaluator, key: &str) {
    ev.env.remove(key);
}

pub(crate) fn set_current_span(ev: &mut Evaluator, span: Option<(usize, usize)>) {
    ev.current_span = span;
}

pub(crate) fn make_error(ev: &Evaluator, message: &str, tag: &str) -> Value {
    let mut m = std::collections::HashMap::new();
    m.insert("error".to_string(), Value::Boolean(true));
    m.insert("message".to_string(), Value::String(message.into()));
    m.insert("tag".to_string(), Value::String(tag.into()));
    if let Some((s, e)) = ev.current_span {
        let mut span = std::collections::HashMap::new();
        span.insert("start".to_string(), Value::Integer(s as i64));
        span.insert("end".to_string(), Value::Integer(e as i64));
        m.insert("span".to_string(), Value::Assoc(span));
    }
    Value::Assoc(m)
}

