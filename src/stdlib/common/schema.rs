//! Schema registry and Schema[] function for Associations
use std::collections::HashSet;
use crate::vm::{Value, VmResult, VmError};

/// Best-effort schema detection by key signature
fn detect_schema(obj: &std::collections::HashMap<String, Value>) -> String {
    let keys: HashSet<&str> = obj.keys().map(|s| s.as_str()).collect();

    // Spectral: frequencies+magnitudes+method (+ optional phases, sampleRate)
    if ["frequencies","magnitudes","method"].iter().all(|k| keys.contains(k)) {
        return "SpectralResult/v1".to_string();
    }
    // Filter: filterType + filteredSignal (+ parameters, success, message)
    if keys.contains("filterType") && keys.contains("filteredSignal") {
        return "FilterResult/v1".to_string();
    }
    // Numerical results
    if keys.contains("root") && keys.contains("iterations") && keys.contains("method") {
        return "RootResult/v1".to_string();
    }
    if keys.contains("evaluations") && keys.contains("value") && keys.contains("method") {
        return "IntegrationResult/v1".to_string();
    }
    if keys.contains("value") && keys.contains("errorEstimate") && keys.contains("order") {
        return "DerivativeResult/v1".to_string();
    }
    // ML evaluation
    if keys.contains("foldScores") && keys.contains("meanScore") {
        return "CrossValidationResult/v1".to_string();
    }
    if keys.contains("accuracy") && keys.contains("f1Score") {
        return "ClassificationReport/v1".to_string();
    }
    if keys.contains("meanSquaredError") && keys.contains("rSquared") {
        return "RegressionReport/v1".to_string();
    }
    // Graph
    if keys.contains("scores") && keys.contains("iterations") {
        return "PageRankResult/v1".to_string();
    }
    if keys.contains("communities") && keys.contains("modularity") {
        return "CommunityResult/v1".to_string();
    }
    // Vector/RAG
    if keys.contains("clusters") && keys.contains("algorithm") {
        return "VectorClusterResult/v1".to_string();
    }
    if keys.contains("question") && keys.contains("answer") {
        return "RAGQueryResult/v1".to_string();
    }
    // Number theory
    if keys.contains("hexString") && keys.contains("algorithm") {
        return "HashResult/v1".to_string();
    }
    // Default association
    "Association/v1".to_string()
}

/// Schema[expr] → Association describing schema name/version and keys
pub fn schema_function(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError { expected: "1 argument".into(), actual: format!("{}", args.len()) });
    }
    match &args[0] {
        Value::Object(m) => {
            let name = detect_schema(m);
            let mut out = std::collections::HashMap::new();
            out.insert("name".to_string(), Value::String(name));
            let key_list: Vec<Value> = m.keys().cloned().map(Value::String).collect();
            out.insert("keys".to_string(), Value::List(key_list));
            Ok(Value::Object(out))
        }
        other => {
            let mut out = std::collections::HashMap::new();
            out.insert("name".to_string(), Value::String(match other {
                Value::Integer(_) => "Integer/v1",
                Value::Real(_) => "Real/v1",
                Value::String(_) => "String/v1",
                Value::Symbol(_) => "Symbol/v1",
                Value::List(_) => "List/v1",
                Value::Boolean(_) => "Boolean/v1",
                Value::Missing => "Missing/v1",
                Value::LyObj(o) => o.type_name(),
                _ => "Value/v1",
            }.to_string()));
            Ok(Value::Object(out))
        }
    }
}

