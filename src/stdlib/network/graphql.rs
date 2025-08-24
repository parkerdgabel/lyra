//! Minimal GraphQL client (Phase A)
//!
//! Implements GraphQLClient and GraphQLQuery using existing HTTP client.

use std::any::Any;
use std::collections::HashMap;

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::stdlib::network::core::{HttpMethod, NetworkRequest, NetworkResponse};
use crate::stdlib::common::options as opt;
use crate::stdlib::network::http::HttpClient;
use crate::vm::{Value, VmError, VmResult};
use serde_json::json;
// duplicate import removed

#[derive(Debug, Clone)]
pub struct GraphQLClient {
    pub endpoint: String,
    pub headers: HashMap<String, String>,
    pub bearer_token: Option<String>,
}

impl GraphQLClient {
    pub fn new(endpoint: String) -> Self {
        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        Self { endpoint, headers, bearer_token: None }
    }
}

impl Foreign for GraphQLClient {
    fn type_name(&self) -> &'static str { "GraphQLClient" }
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Endpoint" => Ok(Value::String(self.endpoint.clone())),
            _ => Err(ForeignError::UnknownMethod { method: method.to_string(), type_name: self.type_name().to_string() })
        }
    }
    fn clone_boxed(&self) -> Box<dyn Foreign> { Box::new(self.clone()) }
    fn as_any(&self) -> &dyn Any { self }
}

pub fn graphql_client(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 { return Err(VmError::TypeError { expected: "GraphQLClient[endpoint, opts?]".to_string(), actual: format!("{} args", args.len()) }); }
    let endpoint = match &args[0] { Value::String(s) => s.clone(), v => return Err(VmError::TypeError { expected: "String endpoint".to_string(), actual: format!("{:?}", v) }) };
    let mut client = GraphQLClient::new(endpoint);
    if args.len() == 2 {
        let opts = opt::expect_object(&args[1], "GraphQLClient")?;
        // headers option
        if let Some(hv) = opts.get("headers") {
            if let Value::Object(hm) = hv {
                for (k, v) in hm.iter() { if let Value::String(s) = v { client.headers.insert(k.clone(), s.clone()); } }
            }
        }
        if let Some(Value::String(tok)) = opts.get("bearerToken") { client.bearer_token = Some(tok.clone()); }
    }
    Ok(Value::LyObj(LyObj::new(Box::new(client))))
}

fn value_to_json(val: &Value) -> serde_json::Value {
    match val {
        Value::Missing => serde_json::Value::Null,
        Value::Boolean(b) => serde_json::Value::Bool(*b),
        Value::Integer(i) => serde_json::Value::Number((*i).into()),
        Value::Real(f) => serde_json::Number::from_f64(*f).map(serde_json::Value::Number).unwrap_or(serde_json::Value::Null),
        Value::String(s) => serde_json::Value::String(s.clone()),
        Value::List(items) => serde_json::Value::Array(items.iter().map(value_to_json).collect()),
        Value::Object(map) => {
            let mut m = serde_json::Map::new();
            for (k, v) in map { m.insert(k.clone(), value_to_json(v)); }
            serde_json::Value::Object(m)
        }
        other => serde_json::Value::String(format!("{:?}", other)),
    }
}

fn make_body(query: &str, vars: Option<&Value>) -> String {
    let variables = match vars { Some(v) => value_to_json(v), None => json!({}) };
    let body = json!({
        "query": query,
        "variables": variables,
    });
    serde_json::to_string(&body).unwrap_or_else(|_| "{}".to_string())
}

pub fn graphql_query(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 4 { return Err(VmError::TypeError { expected: "GraphQLQuery[client|endpoint, query, variables?, opts?]".to_string(), actual: format!("{} args", args.len()) }); }
    // Resolve endpoint
    let endpoint = match &args[0] {
        Value::String(s) => s.clone(),
        Value::LyObj(obj) => {
            if let Some(c) = obj.downcast_ref::<GraphQLClient>() { c.endpoint.clone() } else { return Err(VmError::TypeError { expected: "GraphQLClient or endpoint string".to_string(), actual: obj.type_name().to_string() }) }
        }
        v => return Err(VmError::TypeError { expected: "GraphQLClient or endpoint string".to_string(), actual: format!("{:?}", v) })
    };
    let query = match &args[1] { Value::String(s) => s.clone(), v => return Err(VmError::TypeError { expected: "String query".to_string(), actual: format!("{:?}", v) }) };
    // variables arg (optional)
    let mut var_arg: Option<&Value> = None;
    let mut opts_arg: Option<&Value> = None;
    if args.len() >= 3 {
        if let Value::Object(_) = &args[2] { var_arg = Some(&args[2]); }
    }
    if args.len() == 4 {
        opts_arg = match &args[3] { Value::Object(_) => Some(&args[3]), v => return Err(VmError::TypeError { expected: "opts as Association".to_string(), actual: format!("{:?}", v) }) };
    }

    let body = make_body(&query, var_arg);
    let mut req = NetworkRequest::new(endpoint.clone(), HttpMethod::POST)
        .with_header("Content-Type".to_string(), "application/json".to_string())
        .with_body(body.into_bytes());
    // Merge headers from client and opts
    if let Value::LyObj(obj) = &args[0] {
        if let Some(c) = obj.downcast_ref::<GraphQLClient>() {
            for (k, v) in c.headers.iter() { req = req.with_header(k.clone(), v.clone()); }
            if let Some(tok) = &c.bearer_token { req = req.with_header("Authorization".into(), format!("Bearer {}", tok)); }
        }
    }
    if let Some(Value::Object(m)) = opts_arg {
        if let Some(Value::Object(hm)) = m.get("headers") {
            for (k, v) in hm.iter() { if let Value::String(s) = v { req = req.with_header(k.clone(), s.clone()); } }
        }
        if let Some(Value::String(tok)) = m.get("bearerToken") { req = req.with_header("Authorization".into(), format!("Bearer {}", tok)); }
    }
    let client = HttpClient::new();
    match client.execute(&req) {
        Ok(NetworkResponse { body, .. }) => {
            let s = String::from_utf8(body).map_err(|e| VmError::Runtime(format!("Invalid UTF-8 response: {}", e)))?;
            // Default parse JSON unless opts.parse == false
            let parse = if let Some(Value::Object(m)) = opts_arg { if let Some(Value::Boolean(b)) = m.get("parse") { *b } else { true } } else { true };
            if parse {
                // Use existing JSONParse
                match crate::stdlib::data_processing::json_parse(&[Value::String(s.clone())]) {
                    Ok(v) => {
                        // If response has errors and not disabled, raise
                        let mut error_on = true;
                        if let Some(Value::Object(m)) = opts_arg { if let Some(Value::Boolean(b)) = m.get("errorOnGraphQLErrors") { error_on = *b; } }
                        if error_on {
                            if let Value::Object(map) = &v { if map.get("errors").is_some() { return Err(VmError::Runtime("GraphQL errors present in response".into())); } }
                        }
                        Ok(v)
                    }
                    Err(e) => Err(e),
                }
            } else {
                Ok(Value::String(s))
            }
        }
        Err(e) => Err(VmError::Runtime(format!("GraphQL request failed: {}", e)))
    }
}

pub fn graphql_introspect(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 { return Err(VmError::TypeError { expected: "GraphQLIntrospect[client|endpoint]".to_string(), actual: format!("{} args", args.len()) }); }
    let endpoint = match &args[0] { Value::String(s) => s.clone(), Value::LyObj(obj) => { if let Some(c) = obj.downcast_ref::<GraphQLClient>() { c.endpoint.clone() } else { return Err(VmError::TypeError { expected: "GraphQLClient or endpoint".to_string(), actual: obj.type_name().to_string() }) } }, v => return Err(VmError::TypeError { expected: "GraphQLClient or endpoint".to_string(), actual: format!("{:?}", v) }) };
    let query = "query IntrospectionQuery { __schema { queryType { name } mutationType { name } subscriptionType { name } types { kind name } directives { name } } }";
    graphql_query(&[Value::String(endpoint), Value::String(query.to_string())])
}

#[derive(Debug, Clone)]
pub struct GraphQLResponse {
    pub data: Value,
    pub errors: Value,
    pub extensions: Value,
}

impl GraphQLResponse {
    pub fn new(root: Value) -> Self {
        let (mut data, mut errors, mut extensions) = (Value::Missing, Value::Missing, Value::Missing);
        if let Value::Object(map) = &root {
            if let Some(v) = map.get("data") { data = v.clone(); }
            if let Some(v) = map.get("errors") { errors = v.clone(); }
            if let Some(v) = map.get("extensions") { extensions = v.clone(); }
        }
        GraphQLResponse { data, errors, extensions }
    }
}

impl crate::foreign::Foreign for GraphQLResponse {
    fn type_name(&self) -> &'static str { "GraphQLResponse" }
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, crate::foreign::ForeignError> {
        match method {
            "Data" => Ok(self.data.clone()),
            "Errors" => Ok(self.errors.clone()),
            "Extensions" => Ok(self.extensions.clone()),
            "HasErrors" => {
                let has = match &self.errors {
                    Value::Missing => false,
                    Value::List(v) => !v.is_empty(),
                    _ => true,
                };
                Ok(Value::Boolean(has))
            }
            _ => Err(crate::foreign::ForeignError::UnknownMethod { method: method.to_string(), type_name: self.type_name().to_string() })
        }
    }
    fn clone_boxed(&self) -> Box<dyn crate::foreign::Foreign> { Box::new(self.clone()) }
    fn as_any(&self) -> &dyn Any { self }
}

/// GraphQLQueryResponse - like GraphQLQuery but returns a structured response object
pub fn graphql_query_response(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 4 { return Err(VmError::TypeError { expected: "GraphQLQueryResponse[client|endpoint, query, variables?, opts?]".into(), actual: format!("{} args", args.len()) }); }
    // We reuse GraphQLQuery to perform the call with parse=true
    let mut q_args: Vec<Value> = Vec::new();
    q_args.push(args[0].clone());
    q_args.push(args[1].clone());
    if args.len() >= 3 { q_args.push(args[2].clone()); }
    // Ensure parse=true in opts
    let opts = if args.len() == 4 {
        match &args[3] {
            Value::Object(m) => {
                let mut m2 = m.clone();
                m2.insert("parse".to_string(), Value::Boolean(true));
                Value::Object(m2)
            }
            v => return Err(VmError::TypeError { expected: "opts as Association".into(), actual: format!("{:?}", v) })
        }
    } else { Value::Object(std::collections::HashMap::from([("parse".to_string(), Value::Boolean(true))])) };
    if q_args.len() == 2 { q_args.push(Value::Object(std::collections::HashMap::new())); }
    q_args.push(opts);
    let root = graphql_query(&q_args)?;
    let resp = GraphQLResponse::new(root);
    Ok(Value::LyObj(LyObj::new(Box::new(resp))))
}

#[cfg(test)]
mod tests {
    use super::{make_body, GraphQLResponse};
    use crate::vm::Value;
    use std::collections::HashMap;

    fn assoc(pairs: Vec<(&str, Value)>) -> Value {
        let mut m = HashMap::new();
        for (k, v) in pairs { m.insert(k.to_string(), v); }
        Value::Object(m)
    }

    #[test]
    fn test_make_body_and_json_helpers() {
        let q = "{ user(id: \"1\"){ name } }";
        let vars = assoc(vec![("limit", Value::Integer(10))]);
        let body = make_body(q, Some(&vars));
        assert!(body.contains("\\\"query\\\""));
        assert!(body.contains("\\\"variables\\\""));
        assert!(body.contains("limit"));
        // Body is valid JSON containing our query and variables
        assert!(serde_json::from_str::<serde_json::Value>(&body).is_ok());
    }

    #[test]
    fn test_graphql_response_foreign() {
        // Build a synthetic GraphQL JSON-like Value
        let mut root = HashMap::new();
        root.insert("data".to_string(), Value::Object(HashMap::from([
            ("hello".to_string(), Value::String("world".to_string()))
        ])));
        root.insert("errors".to_string(), Value::List(vec![
            Value::Object(HashMap::from([("message".to_string(), Value::String("oops".to_string()))]))
        ]));
        let v = Value::Object(root);

        let resp = GraphQLResponse::new(v);
        // Methods
        assert_eq!(resp.call_method("HasErrors", &[]).unwrap(), Value::Boolean(true));
        let data = resp.call_method("Data", &[]).unwrap();
        if let Value::Object(m) = data { assert!(m.contains_key("hello")); } else { panic!("expected object"); }
        let errors = resp.call_method("Errors", &[]).unwrap();
        if let Value::List(list) = errors { assert!(!list.is_empty()); } else { panic!("expected list"); }
    }
}
