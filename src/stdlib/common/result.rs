//! Helpers for building Associations and common result shapes

use std::collections::HashMap;
use crate::vm::Value;

// Macro to build Associations with identifier keys in lowerCamelCase.
// Usage: assoc! { key1: expr1, keyTwo: expr2 }
// This helps avoid snake_case keys slipping into public outputs.
#[macro_export]
macro_rules! assoc {
    ( $( $key:ident : $val:expr ),* $(,)? ) => {{
        let mut _m: std::collections::HashMap<String, crate::vm::Value> =
            std::collections::HashMap::new();
        $(
            // Enforce identifier keys; convert to string
            _m.insert(stringify!($key).to_string(), $val);
        )*
        crate::vm::Value::Object(_m)
    }};
}

/// Build a Value::Object (Association) from key-value pairs.
pub fn assoc<K: Into<String>>(pairs: Vec<(K, Value)>) -> Value {
    let mut m: HashMap<String, Value> = HashMap::with_capacity(pairs.len());
    for (k, v) in pairs { m.insert(k.into(), v); }
    Value::Object(m)
}

/// Convenience to build a ConfidenceInterval association
pub fn confidence_interval(lower: f64, upper: f64) -> Value {
    assoc(vec![
        ("lower", Value::Real(lower)),
        ("upper", Value::Real(upper)),
    ])
}

/// Build a standardized derivative/differentiation result association
pub fn derivative_result(value: f64, error_estimate: f64, step: f64, method: &str, order: usize) -> Value {
    assoc! {
        value: Value::Real(value),
        errorEstimate: Value::Real(error_estimate),
        step: Value::Real(step),
        method: Value::String(method.to_string()),
        order: Value::Integer(order as i64)
    }
}

/// Build a standardized root-finding result association
pub fn root_result(root: f64, iterations: usize, function_value: f64, converged: bool, error_estimate: f64, method: &str) -> Value {
    assoc! {
        root: Value::Real(root),
        iterations: Value::Integer(iterations as i64),
        functionValue: Value::Real(function_value),
        converged: Value::Boolean(converged),
        errorEstimate: Value::Real(error_estimate),
        method: Value::String(method.to_string())
    }
}

/// Build a standardized numerical integration result association
pub fn integration_result(value: f64, error_estimate: f64, evaluations: usize, method: &str, converged: bool) -> Value {
    assoc! {
        value: Value::Real(value),
        errorEstimate: Value::Real(error_estimate),
        evaluations: Value::Integer(evaluations as i64),
        method: Value::String(method.to_string()),
        converged: Value::Boolean(converged)
    }
}

/// Build a standardized spectral analysis result association
pub fn spectral_result(frequencies: Vec<f64>, magnitudes: Vec<f64>, phases: Vec<f64>, sample_rate: f64, method: &str) -> Value {
    assoc! {
        frequencies: Value::List(frequencies.into_iter().map(Value::Real).collect()),
        magnitudes: Value::List(magnitudes.into_iter().map(Value::Real).collect()),
        phases: Value::List(phases.into_iter().map(Value::Real).collect()),
        sampleRate: Value::Real(sample_rate),
        method: Value::String(method.to_string())
    }
}

/// Build a standardized filter result association
pub fn filter_result(filter_type: &str, parameters: Vec<Value>, filtered_signal: Vec<f64>, success: bool, message: &str) -> Value {
    assoc! {
        filterType: Value::String(filter_type.to_string()),
        parameters: Value::List(parameters),
        success: Value::Boolean(success),
        message: Value::String(message.to_string()),
        filteredSignal: Value::List(filtered_signal.into_iter().map(Value::Real).collect())
    }
}
