//! Configuration utilities
//!
//! Phase A implements object merge/get/set and environment overlay.
//! Loading/validation are minimal and may delegate to existing Import.

use std::collections::HashMap;

use crate::stdlib::common::options as opt;
use crate::stdlib::io;
use crate::vm::{Value, VmError, VmResult};
use std::fs;
use std::path::Path;

fn as_object(value: &Value) -> VmResult<&HashMap<String, Value>> {
    match value {
        Value::Object(map) => Ok(map),
        v => Err(VmError::TypeError { expected: "Association (Object)".to_string(), actual: format!("{:?}", v) }),
    }
}

fn deep_merge(a: &HashMap<String, Value>, b: &HashMap<String, Value>) -> HashMap<String, Value> {
    let mut out = a.clone();
    for (k, v) in b.iter() {
        match (out.get(k), v) {
            (Some(Value::Object(a_sub)), Value::Object(b_sub)) => {
                out.insert(k.clone(), Value::Object(deep_merge(a_sub, b_sub)));
            }
            _ => { out.insert(k.clone(), v.clone()); }
        }
    }
    out
}

fn get_path<'a>(mut cur: &'a Value, path: &str) -> Option<&'a Value> {
    if path.is_empty() { return Some(cur); }
    for part in path.split('.') {
        match cur {
            Value::Object(map) => {
                cur = map.get(part)?;
            }
            Value::List(list) => {
                if let Ok(idx) = part.parse::<usize>() {
                    cur = list.get(idx)?;
                } else { return None; }
            }
            _ => return None,
        }
    }
    Some(cur)
}

fn set_path(mut root: HashMap<String, Value>, path: &str, value: Value) -> HashMap<String, Value> {
    fn set_rec(map: &mut HashMap<String, Value>, parts: &[&str], value: Value) {
        if parts.is_empty() { return; }
        if parts.len() == 1 {
            map.insert(parts[0].to_string(), value);
            return;
        }
        let head = parts[0];
        let tail = &parts[1..];
        let entry = map.entry(head.to_string()).or_insert_with(|| Value::Object(HashMap::new()));
        if !matches!(entry, Value::Object(_)) {
            *entry = Value::Object(HashMap::new());
        }
        if let Value::Object(inner) = entry {
            set_rec(inner, tail, value);
        }
    }
    let parts: Vec<&str> = path.split('.').collect();
    set_rec(&mut root, &parts, value);
    root
}

pub fn config_merge(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError { expected: "ConfigMerge[assoc1, assoc2, opts?]".to_string(), actual: format!("{} args", args.len()) });
    }
    let a = as_object(&args[0])?;
    let b = as_object(&args[1])?;
    let _deep = if args.len() == 3 {
        let opts = opt::expect_object(&args[2], "ConfigMerge")?;
        opt::get_bool(opts, "deep", true)?
    } else { true };
    Ok(Value::Object(deep_merge(a, b)))
}

pub fn config_get(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 { return Err(VmError::TypeError { expected: "ConfigGet[assoc, keyPath, default?]".to_string(), actual: format!("{} args", args.len()) }); }
    let map = as_object(&args[0])?;
    let path = match &args[1] { Value::String(s) => s.clone(), v => return Err(VmError::TypeError { expected: "String path".to_string(), actual: format!("{:?}", v) }) };
    match get_path(&Value::Object(map.clone()), &path) {
        Some(v) => Ok(v.clone()),
        None => Ok(args.get(2).cloned().unwrap_or(Value::Missing)),
    }
}

pub fn config_set(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 { return Err(VmError::TypeError { expected: "ConfigSet[assoc, keyPath, value]".to_string(), actual: format!("{} args", args.len()) }); }
    let map = as_object(&args[0])?;
    let path = match &args[1] { Value::String(s) => s.clone(), v => return Err(VmError::TypeError { expected: "String path".to_string(), actual: format!("{:?}", v) }) };
    Ok(Value::Object(set_path(map.clone(), &path, args[2].clone())))
}

pub fn config_overlay_env(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 { return Err(VmError::TypeError { expected: "ConfigOverlayEnv[assoc, opts?]".to_string(), actual: format!("{} args", args.len()) }); }
    let base = as_object(&args[0])?;
    let (prefix, sep) = if args.len() == 2 {
        let opts = opt::expect_object(&args[1], "ConfigOverlayEnv")?;
        (opt::get_string(opts, "prefix", "LYRA_")?, opt::get_string(opts, "separator", "__")?)
    } else { ("LYRA_".to_string(), "__".to_string()) };

    let mut overlay: HashMap<String, Value> = HashMap::new();
    for (k, v) in std::env::vars() {
        if !k.starts_with(&prefix) { continue; }
        let key_path = k[prefix.len()..].replace(&sep, ".").to_lowercase();
        // Insert as string values by default
        // In later phases we can parse numbers/bools
        let parts: Vec<&str> = key_path.split('.').collect();
        let mut cur = overlay;
        // build nested map using set_path helper
        overlay = set_path(cur, &key_path, Value::String(v));
    }
    Ok(Value::Object(deep_merge(base, &overlay)))
}

pub fn config_load(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 { return Err(VmError::TypeError { expected: "ConfigLoad[pathOrString, opts?]".to_string(), actual: format!("{} args", args.len()) }); }
    // Minimal: if given a path as String and format json/text/auto, delegate to Import
    let input = match &args[0] { Value::String(s) => s.clone(), v => return Err(VmError::TypeError { expected: "String path".to_string(), actual: format!("{:?}", v) }) };
    let mut format = if args.len() == 2 {
        let opts = opt::expect_object(&args[1], "ConfigLoad")?;
        opt::get_string(opts, "format", "auto")?
    } else { "auto".to_string() };
    let lower = format.to_lowercase();
    // Auto-detect by extension when possible
    if lower == "auto" {
        if input.ends_with(".yaml") || input.ends_with(".yml") { format = "yaml".to_string(); }
        else if input.ends_with(".toml") { format = "toml".to_string(); }
        else if input.ends_with(".json") { format = "json".to_string(); }
        else { format = "text".to_string(); }
    }
    match format.as_str() {
        "json" | "text" => io::import(&[Value::String(input)]),
        "yaml" => {
            let content = fs::read_to_string(Path::new(&input)).map_err(|e| VmError::Runtime(format!("Read failed: {}", e)))?;
            crate::stdlib::util_serialization::yaml_parse(&[Value::String(content)])
        }
        "toml" => {
            let content = fs::read_to_string(Path::new(&input)).map_err(|e| VmError::Runtime(format!("Read failed: {}", e)))?;
            crate::stdlib::util_serialization::toml_parse(&[Value::String(content)])
        }
        other => Err(VmError::TypeError { expected: "format in {auto,json,text,toml,yaml}".to_string(), actual: other.to_string() })
    }
}

pub fn config_validate(args: &[Value]) -> VmResult<Value> {
    // Phase A: stub
    Err(VmError::Runtime("ConfigValidate is not implemented in Phase A".to_string()))
}

pub fn register_config_functions() -> std::collections::HashMap<String, fn(&[Value]) -> VmResult<Value>> {
    let mut f = std::collections::HashMap::new();
    f.insert("ConfigLoad".to_string(), config_load as fn(&[Value]) -> VmResult<Value>);
    f.insert("ConfigMerge".to_string(), config_merge as fn(&[Value]) -> VmResult<Value>);
    f.insert("ConfigOverlayEnv".to_string(), config_overlay_env as fn(&[Value]) -> VmResult<Value>);
    f.insert("ConfigGet".to_string(), config_get as fn(&[Value]) -> VmResult<Value>);
    f.insert("ConfigSet".to_string(), config_set as fn(&[Value]) -> VmResult<Value>);
    f.insert("ConfigValidate".to_string(), config_validate as fn(&[Value]) -> VmResult<Value>);
    f
}
