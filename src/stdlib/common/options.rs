//! Options extraction helpers for stdlib functions
//!
//! All options are passed as `Value::Object(HashMap<String, Value>)`.
//! These utilities provide consistent, typed access with defaults
//! and clear error messages.

use std::collections::HashMap;

use crate::vm::{VmError, VmResult, Value};

pub fn expect_object<'a>(value: &'a Value, ctx: &str) -> VmResult<&'a HashMap<String, Value>> {
    match value {
        Value::Object(map) => Ok(map),
        other => Err(VmError::TypeError {
            expected: format!("options object for {}", ctx),
            actual: format!("{:?}", other),
        }),
    }
}

pub fn get_bool(opts: &HashMap<String, Value>, key: &str, default: bool) -> VmResult<bool> {
    match opts.get(key) {
        None => Ok(default),
        Some(Value::Boolean(b)) => Ok(*b),
        Some(Value::Integer(i)) => Ok(*i != 0),
        Some(v) => Err(VmError::TypeError {
            expected: format!("Boolean for option '{}" , key),
            actual: format!("{:?}", v),
        }),
    }
}

pub fn get_int(opts: &HashMap<String, Value>, key: &str, default: i64) -> VmResult<i64> {
    match opts.get(key) {
        None => Ok(default),
        Some(Value::Integer(i)) => Ok(*i),
        Some(v) => Err(VmError::TypeError { expected: format!("Integer for option '{}'", key), actual: format!("{:?}", v) }),
    }
}

pub fn get_real(opts: &HashMap<String, Value>, key: &str, default: f64) -> VmResult<f64> {
    match opts.get(key) {
        None => Ok(default),
        Some(Value::Integer(i)) => Ok(*i as f64),
        Some(Value::Real(r)) => Ok(*r),
        Some(v) => Err(VmError::TypeError { expected: format!("Number for option '{}'", key), actual: format!("{:?}", v) }),
    }
}

pub fn get_string<'a>(opts: &'a HashMap<String, Value>, key: &str, default: &'a str) -> VmResult<String> {
    match opts.get(key) {
        None => Ok(default.to_string()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(v) => Err(VmError::TypeError { expected: format!("String for option '{}'", key), actual: format!("{:?}", v) }),
    }
}

pub fn get_enum(opts: &HashMap<String, Value>, key: &str, allowed: &[&str], default: &str) -> VmResult<String> {
    let val = get_string(opts, key, default)?;
    if allowed.iter().any(|a| a.eq_ignore_ascii_case(&val)) {
        Ok(val)
    } else {
        Err(VmError::TypeError { expected: format!("one of {:?} for option '{}'", allowed, key), actual: val })
    }
}

pub fn require_keys(opts: &HashMap<String, Value>, allowed: &[&str]) -> VmResult<()> {
    for k in opts.keys() {
        if !allowed.iter().any(|a| a.eq(k)) {
            return Err(VmError::TypeError { expected: format!("options among {:?}", allowed), actual: k.clone() });
        }
    }
    Ok(())
}

