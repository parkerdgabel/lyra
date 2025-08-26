use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_runtime::attrs::Attributes;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

pub fn register_project(ev: &mut Evaluator) {
    ev.register("ProjectDiscover", project_discover as NativeFn, Attributes::empty());
    ev.register("ProjectRoot", project_root as NativeFn, Attributes::empty());
    ev.register("ProjectLoad", project_load as NativeFn, Attributes::empty());
    ev.register("ProjectInfo", project_info as NativeFn, Attributes::empty());
}

fn failure(tag: &str, msg: &str) -> Value {
    Value::Assoc(HashMap::from([
        (String::from("message"), Value::String(msg.to_string())),
        (String::from("tag"), Value::String(tag.to_string())),
    ]))
}

fn as_string(v: &Value) -> Option<String> { match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None } }

fn discover_from(start: &Path) -> Option<PathBuf> {
    let mut p = Some(start);
    while let Some(cur) = p {
        let cand = cur.join("project.lyra");
        if cand.exists() { return Some(cur.to_path_buf()); }
        p = cur.parent();
    }
    None
}

fn current_dir(ev: &Evaluator) -> PathBuf {
    if let Some(Value::String(s)) = ev.get_env("CurrentDir") { PathBuf::from(s) } else { std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")) }
}

fn project_discover(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let base = if args.len()==1 { if let Some(s)=as_string(&ev.eval(args[0].clone())) { PathBuf::from(s) } else { current_dir(ev) } } else { current_dir(ev) };
    discover_from(&base).map(|p| Value::String(p.to_string_lossy().to_string())).unwrap_or(Value::Symbol("Null".into()))
}

fn project_root(ev: &mut Evaluator, _args: Vec<Value>) -> Value {
    if let Some(Value::String(s)) = ev.get_env("ProjectRoot") { return Value::String(s); }
    let base = current_dir(ev);
    discover_from(&base).map(|p| Value::String(p.to_string_lossy().to_string())).unwrap_or(Value::Symbol("Null".into()))
}

fn eval_manifest_at(path: &Path) -> Result<Value, String> {
    // Restricted evaluation context: only minimal helpers registered (module ResolveRelative)
    let content = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut ev2 = Evaluator::new();
    // seed env vars
    let dir = path.parent().unwrap_or_else(|| Path::new(".")).to_path_buf();
    ev2.set_env("CurrentFile", Value::String(path.to_string_lossy().to_string()));
    ev2.set_env("CurrentDir", Value::String(dir.to_string_lossy().to_string()));
    ev2.set_env("ProjectRoot", Value::String(dir.to_string_lossy().to_string()));
    // register minimal module helpers (ResolveRelative)
    crate::module::register_module(&mut ev2);
    let mut p = lyra_parser::Parser::from_source(&content);
    let exprs = p.parse_all().map_err(|e| format!("parse: {:?}", e))?;
    let mut last = Value::Symbol("Null".into());
    for e in exprs { last = ev2.eval(e); }
    Ok(last)
}

fn normalize_manifest(root: &Path, v: Value) -> Result<Value, String> {
    // Expect assoc with keys Project, Modules, Scripts, Env, Config, Workspace, Deps
    let mut m = match v { Value::Assoc(m)=>m, _=> return Err("project.lyra must evaluate to an association".into()) };
    // Normalize Modules to absolute paths
    if let Some(Value::Assoc(mods)) = m.get_mut("Modules") {
        let keys: Vec<String> = mods.keys().cloned().collect();
        for k in keys {
            if let Some(val) = mods.get(&k) {
                if let Some(s) = as_string(val) {
                    let p = PathBuf::from(s);
                    let abs = if p.is_absolute() { p } else { root.join(p) };
                    mods.insert(k, Value::String(abs.to_string_lossy().to_string()));
                }
            }
        }
    }
    Ok(Value::Assoc(m))
}

fn project_load(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let root = if args.len()==1 { as_string(&ev.eval(args[0].clone())).map(PathBuf::from) } else { None }
        .or_else(|| if let Some(Value::String(s)) = ev.get_env("ProjectRoot") { Some(PathBuf::from(s)) } else { None })
        .or_else(|| discover_from(&current_dir(ev)));
    let root = match root { Some(r)=>r, None=> return failure("Project::load", "No project found") };
    let manifest = root.join("project.lyra");
    match eval_manifest_at(&manifest).and_then(|v| normalize_manifest(&root, v)) {
        Ok(v) => v,
        Err(e) => failure("Project::load", &e)
    }
}

fn project_info(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let m = project_load(ev, args);
    match m {
        Value::Assoc(mut a) => {
            let root = if let Some(Value::String(r)) = ev.get_env("ProjectRoot") { r } else { current_dir(ev).to_string_lossy().to_string() };
            Value::Assoc(HashMap::from([
                (String::from("name"), a.get("Project").and_then(|p| match p { Value::Assoc(mm)=> mm.get("Name").cloned(), _=>None }).unwrap_or(Value::String(String::new()))),
                (String::from("version"), a.get("Project").and_then(|p| match p { Value::Assoc(mm)=> mm.get("Version").cloned(), _=>None }).unwrap_or(Value::String(String::new()))),
                (String::from("root"), Value::String(root)),
                (String::from("manifestPath"), Value::String(PathBuf::from(current_dir(ev)).join("project.lyra").to_string_lossy().to_string())),
                (String::from("modules"), a.remove("Modules").unwrap_or(Value::Assoc(HashMap::new()))),
                (String::from("scripts"), a.remove("Scripts").unwrap_or(Value::Assoc(HashMap::new()))),
                (String::from("env"), a.remove("Env").unwrap_or(Value::Assoc(HashMap::new()))),
                (String::from("config"), a.remove("Config").unwrap_or(Value::Assoc(HashMap::new()))),
            ]))
        }
        other => other
    }
}

