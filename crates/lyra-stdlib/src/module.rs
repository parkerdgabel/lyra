use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

pub fn register_module(ev: &mut Evaluator) {
    // When package support is enabled, prefer package::Using for package path semantics.
    // Expose file-based module loader under a different name to avoid conflicts.
    #[cfg(not(feature = "package"))]
    {
        ev.register("Using", using as NativeFn, Attributes::empty());
    }
    #[cfg(feature = "package")]
    {
        ev.register("UsingFile", using as NativeFn, Attributes::empty());
    }
    ev.register("Exported", exported as NativeFn, Attributes::empty());
    ev.register("ModuleInfo", module_info as NativeFn, Attributes::empty());
    ev.register("ResolveRelative", resolve_relative as NativeFn, Attributes::empty());
}

fn str_of(ev: &mut Evaluator, v: Value) -> String {
    match ev.eval(v) {
        Value::String(s) | Value::Symbol(s) => s,
        other => lyra_core::pretty::format_value(&other),
    }
}

fn current_dir(ev: &Evaluator) -> PathBuf {
    if let Some(Value::String(s)) = ev_env_get(ev, "CurrentDir") {
        PathBuf::from(s)
    } else {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    }
}

fn ev_env_get(_ev: &Evaluator, _key: &str) -> Option<Value> {
    // Evaluator has env_keys but not direct getter; hack by formatting? We'll mirror keys and use a hidden helper: We can't access env directly; so rely on ModuleInfo previously set.
    None
}

fn find_project_root(start: &Path) -> Option<PathBuf> {
    let mut p = Some(start);
    while let Some(cur) = p {
        let cand = cur.join("lyra.project");
        if cand.exists() {
            return Some(cur.to_path_buf());
        }
        p = cur.parent();
    }
    None
}

fn resolve_module_path(ev: &mut Evaluator, target: &str) -> Option<PathBuf> {
    // absolute or relative path with extension
    let is_pathish = target.starts_with("./")
        || target.starts_with("../")
        || target.starts_with('/')
        || target.ends_with(".lyra");
    if is_pathish {
        let base = current_dir(ev);
        let p =
            if Path::new(target).is_absolute() { PathBuf::from(target) } else { base.join(target) };
        return Some(p);
    }
    // try project manifest mapping
    let base = current_dir(ev);
    let root = find_project_root(&base);
    if let Some(root) = root {
        let manifest = root.join("lyra.project");
        if let Ok(s) = fs::read_to_string(&manifest) {
            // Try TOML first
            if let Ok(doc) = s.parse::<toml::Value>() {
                if let Some(mods) = doc.get("modules").and_then(|m| m.as_table()) {
                    if let Some(t) = mods.get(target).and_then(|v| v.as_str()) {
                        return Some(root.join(t));
                    }
                }
            }
        }
        // fallback to root/modules/<name>.lyra
        let alt = root.join("modules").join(format!("{}.lyra", target));
        if alt.exists() {
            return Some(alt);
        }
    }
    // fallback to name.lyra in current dir
    let local = current_dir(ev).join(format!("{}.lyra", target));
    if local.exists() {
        return Some(local);
    }
    None
}

fn using(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("Using".into())), args };
    }
    let target = str_of(ev, args[0].clone());
    let _opts = if args.len() > 1 {
        if let Value::Assoc(m) = ev.eval(args[1].clone()) {
            m
        } else {
            HashMap::new()
        }
    } else {
        HashMap::new()
    };
    let path = match resolve_module_path(ev, &target) {
        Some(p) => p,
        None => {
            return lyra_core::value::Value::Assoc(HashMap::from([
                (String::from("message"), Value::String(format!("Module not found: {}", target))),
                (String::from("tag"), Value::String(String::from("Module::using"))),
            ]))
        }
    };
    let abs = if path.is_absolute() { path } else { fs::canonicalize(&path).unwrap_or(path) };
    // read file
    let content = match fs::read_to_string(&abs) {
        Ok(s) => s,
        Err(e) => return failure("Module::using", &e.to_string()),
    };
    // capture keys before
    let before: std::collections::HashSet<String> = ev.env_keys().into_iter().collect();
    // set env vars
    let cur_dir = abs.parent().unwrap_or_else(|| Path::new(".")).to_path_buf();
    let proj_root = find_project_root(&cur_dir).unwrap_or(cur_dir.clone());
    ev.set_env("CurrentFile", Value::String(abs.to_string_lossy().to_string()));
    ev.set_env("CurrentDir", Value::String(cur_dir.to_string_lossy().to_string()));
    ev.set_env("ProjectRoot", Value::String(proj_root.to_string_lossy().to_string()));
    // parse and eval
    let mut p = lyra_parser::Parser::from_source(&content);
    match p.parse_all() {
        Ok(exprs) => {
            for e in exprs {
                let _ = ev.eval(e);
            }
        }
        Err(e) => {
            return failure("Module::using", &format!("parse: {:?}", e));
        }
    }
    // exports: declared or inferred
    let after: std::collections::HashSet<String> = ev.env_keys().into_iter().collect();
    let mut new_syms: Vec<String> = after.difference(&before).cloned().collect();
    new_syms.sort();
    // explicit Exported list
    let mut exports: Vec<(String, Value)> = Vec::new();
    if let Some(Value::List(xs)) = ev.get_env("__exports_declared") {
        let mut names: Vec<String> = Vec::new();
        for v in xs {
            if let Value::String(s) | Value::Symbol(s) = v {
                names.push(s.clone());
            }
        }
        if !names.is_empty() {
            new_syms = names;
        }
        ev.unset_env("__exports_declared");
    }
    for name in &new_syms {
        exports.push((name.clone(), Value::Symbol(name.clone())));
    }
    let module_map = Value::Assoc(HashMap::from([
        (String::from("Exports"), Value::Assoc(exports.into_iter().collect())),
        (String::from("Path"), Value::String(abs.to_string_lossy().to_string())),
        (String::from("Name"), Value::String(target.clone())),
    ]));
    module_map
}

fn exported(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("Exported".into())), args };
    }
    match ev.eval(args[0].clone()) {
        Value::List(xs) => {
            ev.set_env("__exports_declared", Value::List(xs));
            Value::Boolean(true)
        }
        other => {
            Value::Expr { head: Box::new(Value::Symbol("Exported".into())), args: vec![other] }
        }
    }
}

fn module_info(ev: &mut Evaluator, _args: Vec<Value>) -> Value {
    let mut m = HashMap::new();
    if let Some(v) = ev.get_env("CurrentFile") {
        m.insert("Path".into(), v);
    }
    if let Some(v) = ev.get_env("CurrentDir") {
        m.insert("Dir".into(), v);
    }
    if let Some(v) = ev.get_env("ProjectRoot") {
        m.insert("ProjectRoot".into(), v);
    }
    Value::Assoc(m)
}

fn resolve_relative(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("ResolveRelative".into())), args };
    }
    let p = str_of(ev, args[0].clone());
    let base = if let Some(Value::String(s)) = ev.get_env("CurrentDir") {
        PathBuf::from(s)
    } else {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    };
    let out = base.join(p);
    Value::String(out.to_string_lossy().to_string())
}

fn failure(tag: &str, msg: &str) -> Value {
    Value::Assoc(HashMap::from([
        (String::from("message"), Value::String(msg.to_string())),
        (String::from("tag"), Value::String(tag.to_string())),
    ]))
}
