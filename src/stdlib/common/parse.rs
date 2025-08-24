//! Typed parsing helpers for stdlib wrappers
//!
//! These utilities convert Lyra `Value` into strongly typed Rust values with
//! consistent error messages. Prefer using these helpers over ad-hoc matching.

use std::collections::HashMap;

use crate::vm::{VmError, VmResult, Value};

/// FromLyra converts a `Value` reference into a strongly-typed Rust value.
pub trait FromLyra: Sized {
    fn from_value(v: &Value) -> VmResult<Self>;
}

impl FromLyra for Value {
    fn from_value(v: &Value) -> VmResult<Self> { Ok(v.clone()) }
}

impl FromLyra for bool {
    fn from_value(v: &Value) -> VmResult<Self> {
        match v {
            Value::Boolean(b) => Ok(*b),
            Value::Integer(i) => Ok(*i != 0),
            other => Err(VmError::TypeError { expected: "Boolean".to_string(), actual: format!("{:?}", other) })
        }
    }
}

impl FromLyra for i64 {
    fn from_value(v: &Value) -> VmResult<Self> {
        match v {
            Value::Integer(i) => Ok(*i),
            other => Err(VmError::TypeError { expected: "Integer".to_string(), actual: format!("{:?}", other) })
        }
    }
}

impl FromLyra for usize {
    fn from_value(v: &Value) -> VmResult<Self> {
        match v {
            Value::Integer(i) if *i >= 0 => Ok(*i as usize),
            other => Err(VmError::TypeError { expected: "Non-negative Integer (usize)".to_string(), actual: format!("{:?}", other) })
        }
    }
}

impl FromLyra for f64 {
    fn from_value(v: &Value) -> VmResult<Self> {
        match v {
            Value::Real(r) => Ok(*r),
            Value::Integer(i) => Ok(*i as f64),
            other => Err(VmError::TypeError { expected: "Number".to_string(), actual: format!("{:?}", other) })
        }
    }
}

impl FromLyra for String {
    fn from_value(v: &Value) -> VmResult<Self> {
        match v {
            Value::String(s) | Value::Symbol(s) => Ok(s.clone()),
            other => Err(VmError::TypeError { expected: "String or Symbol".to_string(), actual: format!("{:?}", other) })
        }
    }
}

impl<T: FromLyra> FromLyra for Vec<T> {
    fn from_value(v: &Value) -> VmResult<Self> {
        match v {
            Value::List(items) => items.iter().map(T::from_value).collect(),
            other => Err(VmError::TypeError { expected: "List".to_string(), actual: format!("{:?}", other) })
        }
    }
}

impl FromLyra for HashMap<String, Value> {
    fn from_value(v: &Value) -> VmResult<Self> {
        match v {
            Value::Object(m) => Ok(m.clone()),
            other => Err(VmError::TypeError { expected: "Association (Object)".to_string(), actual: format!("{:?}", other) })
        }
    }
}

/// Parse a positional argument at index into type `T`.
pub fn arg<T: FromLyra>(args: &[Value], index: usize, ctx: &str) -> VmResult<T> {
    args.get(index)
        .ok_or_else(|| VmError::TypeError { expected: format!("{}: at least {} arguments", ctx, index + 1), actual: format!("{} args", args.len()) })
        .and_then(T::from_value)
}

/// Parse an optional trailing options association (last arg) if present.
pub fn trailing_options(args: &[Value]) -> Option<&HashMap<String, Value>> {
    args.last().and_then(|v| if let Value::Object(m) = v { Some(m) } else { None })
}

/// Parse a list argument at index into `Vec<T>`.
pub fn arg_list<T: FromLyra>(args: &[Value], index: usize, ctx: &str) -> VmResult<Vec<T>> {
    let v = args.get(index).ok_or_else(|| VmError::TypeError { expected: format!("{}: list at position {}", ctx, index + 1), actual: format!("{} args", args.len()) })?;
    <Vec<T>>::from_value(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_scalars_and_lists() {
        let v = Value::Integer(3);
        assert_eq!(i64::from_value(&v).unwrap(), 3);
        assert_eq!(usize::from_value(&v).unwrap(), 3usize);
        assert_eq!(f64::from_value(&v).unwrap(), 3.0);

        let v = Value::List(vec![Value::Integer(1), Value::Integer(2)]);
        let out: Vec<i64> = Vec::<i64>::from_value(&v).unwrap();
        assert_eq!(out, vec![1, 2]);
    }
}

