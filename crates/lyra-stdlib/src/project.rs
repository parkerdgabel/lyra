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
    ev.register("ProjectValidate", project_validate as NativeFn, Attributes::empty());
    ev.register("ProjectInit", project_init as NativeFn, Attributes::empty());
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

// ---------------- Validation & Init ----------------

#[derive(Default)]
struct Diag {
    errors: Vec<(String, String, String)>,   // (path, code, message)
    warnings: Vec<(String, String, String)>, // (path, code, message)
}

fn push_err(d: &mut Diag, path: &str, code: &str, msg: &str) { d.errors.push((path.to_string(), code.to_string(), msg.to_string())); }
fn push_warn(d: &mut Diag, path: &str, code: &str, msg: &str) { d.warnings.push((path.to_string(), code.to_string(), msg.to_string())); }

fn validate_manifest(root: &Path, v: &Value) -> Value {
    use std::collections::HashMap;
    let mut d = Diag::default();
    let top = match v { Value::Assoc(m)=>m.clone(), other=>{
        push_err(&mut d, "", "type.top", &format!("project.lyra must evaluate to an association, got {:?}", other));
        return Value::Assoc(HashMap::from([
            ("ok".into(), Value::Boolean(false)),
            ("errors".into(), Value::List(d.errors.into_iter().map(|(p,c,m)| Value::Assoc(HashMap::from([(String::from("path"),Value::String(p)),(String::from("code"),Value::String(c)),(String::from("message"),Value::String(m))]))).collect())),
            ("warnings".into(), Value::List(Vec::new())),
        ]));
    }};
    // Allowed keys
    let allowed: std::collections::HashSet<&str> = ["Project","Modules","Scripts","Env","Config","Workspace","Deps"].into_iter().collect();
    for k in top.keys() {
        if !allowed.contains(k.as_str()) {
            push_warn(&mut d, &format!("{}", k), "key.unknown", &format!("Unknown top-level key '{}'; allowed: Project, Modules, Scripts, Env, Config, Workspace, Deps", k));
        }
    }
    // Project
    match top.get("Project") { Some(Value::Assoc(pm)) => {
        match pm.get("Name") { Some(Value::String(_))|Some(Value::Symbol(_)) => {}, Some(other) => push_err(&mut d, "Project.Name", "type.string", &format!("expected string/symbol, got {:?}", other)), None => push_err(&mut d, "Project.Name", "missing", "Project.Name is required") }
        match pm.get("Version") { Some(Value::String(_))|Some(Value::Symbol(_)) => {}, Some(other) => push_err(&mut d, "Project.Version", "type.string", &format!("expected string/symbol, got {:?}", other)), None => push_err(&mut d, "Project.Version", "missing", "Project.Version is required") }
        if let Some(other) = pm.get("Description") { if !matches!(other, Value::String(_) | Value::Symbol(_)) { push_warn(&mut d, "Project.Description", "type.string", &format!("expected string/symbol, got {:?}", other)); } }
    }, Some(other) => push_err(&mut d, "Project", "type.assoc", &format!("expected association, got {:?}", other)), None => push_err(&mut d, "Project", "missing", "Top-level Project assoc is required") }
    // Modules
    match top.get("Modules") { Some(Value::Assoc(mm)) => {
        for (name, pathv) in mm.iter() {
            let pname = match name.as_str() { s => s };
            match pathv {
                Value::String(s)|Value::Symbol(s) => {
                    let p = PathBuf::from(s);
                    if !p.exists() { push_err(&mut d, &format!("Modules.{}", pname), "path.missing", &format!("module path does not exist: {}", s)); }
                    else if p.is_dir() { push_warn(&mut d, &format!("Modules.{}", pname), "path.is_dir", &format!("expected file, got directory: {}", s)); }
                }
                other => push_err(&mut d, &format!("Modules.{}", pname), "type.string", &format!("expected string path, got {:?}", other)),
            }
        }
    }, Some(other) => push_err(&mut d, "Modules", "type.assoc", &format!("expected association, got {:?}", other)), None => {} }
    // Scripts
    match top.get("Scripts") { Some(Value::Assoc(sm)) => {
        for (sname, val) in sm.iter() {
            if let Value::Assoc(am) = val {
                // Validate Module reference if present
                if let Some(Value::String(mn))|Some(Value::Symbol(mn)) = am.get("Module") {
                    if let Some(Value::Assoc(mm)) = top.get("Modules") { if !mm.contains_key(mn) { push_err(&mut d, &format!("Scripts.{}", sname), "script.unknownModule", &format!("references unknown module '{}'", mn)); } }
                }
            } else if !matches!(val, Value::String(_)|Value::Symbol(_)|Value::List(_)) {
                push_warn(&mut d, &format!("Scripts.{}", sname), "type.unsupported", &format!("script value should be assoc/string/list; got {:?}", val));
            }
        }
    }, Some(other) => push_err(&mut d, "Scripts", "type.assoc", &format!("expected association, got {:?}", other)), None => {} }
    // Env
    match top.get("Env") { Some(Value::Assoc(em)) => {
        for (k, v) in em.iter() { if !matches!(v, Value::String(_)|Value::Symbol(_)|Value::Integer(_)|Value::Real(_)) { push_warn(&mut d, &format!("Env.{}", k), "type.string", &format!("env var should be string/number, got {:?}", v)); } }
    }, Some(other) => push_err(&mut d, "Env", "type.assoc", &format!("expected association, got {:?}", other)), None => {} }
    // Config: best-effort assoc
    if let Some(other) = top.get("Config") { if !matches!(other, Value::Assoc(_)) { push_warn(&mut d, "Config", "type.assoc", &format!("expected association, got {:?}", other)); } }
    // Workspace (optional): Members list if present
    if let Some(Value::Assoc(ws)) = top.get("Workspace") {
        if let Some(m) = ws.get("Members").or_else(|| ws.get("members")).or_else(|| ws.get("members")) {
            if let Value::List(vs) = m { for v in vs { if !matches!(v, Value::String(_)|Value::Symbol(_)) { push_err(&mut d, "Workspace.Members", "type.string", &format!("expected string entries, got {:?}", v)); } } }
            else { push_err(&mut d, "Workspace.Members", "type.list", &format!("expected list of strings, got {:?}", m)); }
        }
    } else if let Some(other) = top.get("Workspace") { if !matches!(other, Value::Symbol(_)) { push_warn(&mut d, "Workspace", "type.assocOrNull", &format!("expected assoc or Null, got {:?}", other)); } }

    let mut errors_v: Vec<Value> = Vec::new();
    for (p,c,m) in d.errors.into_iter() { errors_v.push(Value::Assoc(HashMap::from([(String::from("path"),Value::String(p)),(String::from("code"),Value::String(c)),(String::from("message"),Value::String(m))]))); }
    let mut warns_v: Vec<Value> = Vec::new();
    for (p,c,m) in d.warnings.into_iter() { warns_v.push(Value::Assoc(HashMap::from([(String::from("path"),Value::String(p)),(String::from("code"),Value::String(c)),(String::from("message"),Value::String(m))]))); }
    Value::Assoc(HashMap::from([
        (String::from("ok"), Value::Boolean(errors_v.is_empty())),
        (String::from("errors"), Value::List(errors_v)),
        (String::from("warnings"), Value::List(warns_v)),
    ]))
}

fn project_validate(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // ProjectValidate[] or ProjectValidate[<path>]
    let root = if args.len()==1 { as_string(&ev.eval(args[0].clone())).map(PathBuf::from) } else { None }
        .or_else(|| if let Some(Value::String(s)) = ev.get_env("ProjectRoot") { Some(PathBuf::from(s)) } else { None })
        .or_else(|| discover_from(&current_dir(ev)));
    let root = match root { Some(r)=>r, None=> return failure("Project::validate", "No project found") };
    let manifest = root.join("project.lyra");
    match eval_manifest_at(&manifest).and_then(|v| normalize_manifest(&root, v)) {
        Ok(v) => validate_manifest(&root, &v),
        Err(e) => Value::Assoc(std::iter::IntoIterator::into_iter([
            (String::from("ok"), Value::Boolean(false)),
            (String::from("errors"), Value::List(vec![Value::Assoc(std::iter::IntoIterator::into_iter([
                (String::from("path"), Value::String(String::from("project.lyra"))),
                (String::from("code"), Value::String(String::from("parse"))),
                (String::from("message"), Value::String(e))
            ]).collect())])),
            (String::from("warnings"), Value::List(vec![])),
        ]).collect()),
    }
}

fn project_init(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    use std::collections::HashMap;
    // ProjectInit[] | ProjectInit["dir"] | ProjectInit[<|Name->.., Dir->..|>]
    let mut name_opt: Option<String> = None;
    let mut dir_opt: Option<PathBuf> = None;
    if args.len()==1 {
        match ev.eval(args[0].clone()) {
            Value::String(s)|Value::Symbol(s) => { dir_opt = Some(PathBuf::from(s)); }
            Value::Assoc(m) => {
                if let Some(Value::String(s))|Some(Value::Symbol(s)) = m.get("Name").or_else(|| m.get("name")) { name_opt = Some(s.clone()); }
                if let Some(Value::String(s))|Some(Value::Symbol(s)) = m.get("Dir").or_else(|| m.get("dir")) { dir_opt = Some(PathBuf::from(s)); }
            }
            _ => {}
        }
    }
    let cwd = current_dir(ev);
    let dir = dir_opt.unwrap_or_else(|| cwd.clone());
    let name = name_opt.unwrap_or_else(|| dir.file_name().and_then(|s| s.to_str()).unwrap_or("lyra-project").to_string());
    // Create structure
    if let Err(e) = std::fs::create_dir_all(dir.join("src")) { return failure("Project::init", &format!("create src dir: {}", e)); }
    let manifest_path = dir.join("project.lyra");
    if manifest_path.exists() { return failure("Project::init", "project.lyra already exists"); }
    let template = format!(
        "Exported[{{}}];\n\n<|\n  \"Project\" -> <| \"Name\"->\"{}\", \"Version\"->\"0.1.0\", \"Description\"->\"\" |>,\n  \"Modules\" -> <| \"main\" -> ResolveRelative[\"src/main.lyra\"] |>,\n  \"Scripts\" -> <| \"run\" -> <| \"Module\"->\"main\", \"Entry\"->\"Main\" |> |>,\n  \"Env\" -> <||>,\n  \"Config\" -> <||>,\n  \"Workspace\" -> Null,\n  \"Deps\" -> <||>\n|>\n",
        name
    );
    if let Err(e) = std::fs::write(&manifest_path, template) { return failure("Project::init", &format!("write project.lyra: {}", e)); }
    let main_src = "Main[] := (Puts[\"Hello from Lyra!\"]; )\n";
    if let Err(e) = std::fs::write(dir.join("src/main.lyra"), main_src) { return failure("Project::init", &format!("write src/main.lyra: {}", e)); }
    // Optional .gitignore if not present
    let gi = dir.join(".gitignore");
    if !gi.exists() {
        let _ = std::fs::write(gi, "target\nbuild\n.DS_Store\n");
    }
    Value::Assoc(HashMap::from([
        (String::from("ok"), Value::Boolean(true)),
        (String::from("root"), Value::String(dir.to_string_lossy().to_string())),
        (String::from("manifest"), Value::String(manifest_path.to_string_lossy().to_string())),
        (String::from("created"), Value::List(vec![Value::String(String::from("project.lyra")), Value::String(String::from("src/main.lyra"))])),
    ]))
}
