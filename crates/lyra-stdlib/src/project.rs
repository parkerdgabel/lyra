use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

pub fn register_project(ev: &mut Evaluator) {
    ev.register("ProjectDiscover", project_discover as NativeFn, Attributes::empty());
    ev.register("ProjectRoot", project_root as NativeFn, Attributes::empty());
    ev.register("ProjectLoad", project_load as NativeFn, Attributes::empty());
    ev.register("ProjectInfo", project_info as NativeFn, Attributes::empty());
    ev.register("ProjectValidate", project_validate as NativeFn, Attributes::empty());
    ev.register("ProjectInit", project_init as NativeFn, Attributes::empty());
    // New helpers for discoverability and execution
    ev.register("ProjectSchema", project_schema as NativeFn, Attributes::empty());
    ev.register("ProjectModules", project_modules as NativeFn, Attributes::empty());
    ev.register("ProjectFiles", project_files as NativeFn, Attributes::empty());
    ev.register("ProjectScripts", project_scripts as NativeFn, Attributes::empty());
    ev.register("ProjectRun", project_run as NativeFn, Attributes::HOLD_ALL);
    ev.register("ProjectGraph", project_graph as NativeFn, Attributes::empty());
    ev.register("ProjectWatch", project_watch as NativeFn, Attributes::HOLD_ALL);
}

fn failure(tag: &str, msg: &str) -> Value {
    Value::Assoc(HashMap::from([
        (String::from("message"), Value::String(msg.to_string())),
        (String::from("tag"), Value::String(tag.to_string())),
    ]))
}

fn as_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) | Value::Symbol(s) => Some(s.clone()),
        _ => None,
    }
}

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
    if let Some(Value::String(s)) = ev.get_env("CurrentDir") {
        PathBuf::from(s)
    } else {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    }
}

fn project_discover(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let base = if args.len() == 1 {
        if let Some(s) = as_string(&ev.eval(args[0].clone())) {
            PathBuf::from(s)
        } else {
            current_dir(ev)
        }
    } else {
        current_dir(ev)
    };
    discover_from(&base)
        .map(|p| Value::String(p.to_string_lossy().to_string()))
        .unwrap_or(Value::Symbol("Null".into()))
}

fn project_root(ev: &mut Evaluator, _args: Vec<Value>) -> Value {
    if let Some(Value::String(s)) = ev.get_env("ProjectRoot") {
        return Value::String(s);
    }
    let base = current_dir(ev);
    discover_from(&base)
        .map(|p| Value::String(p.to_string_lossy().to_string()))
        .unwrap_or(Value::Symbol("Null".into()))
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
    for e in exprs {
        last = ev2.eval(e);
    }
    Ok(last)
}

fn normalize_manifest(root: &Path, v: Value) -> Result<Value, String> {
    // Expect assoc with keys Project, Modules, Scripts, Env, Config, Workspace, Deps
    let mut m = match v {
        Value::Assoc(m) => m,
        _ => return Err("project.lyra must evaluate to an association".into()),
    };
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
    let root = if args.len() >= 1 {
        as_string(&ev.eval(args[0].clone())).map(PathBuf::from)
    } else {
        None
    }
    .or_else(|| {
        if let Some(Value::String(s)) = ev.get_env("ProjectRoot") {
            Some(PathBuf::from(s))
        } else {
            None
        }
    })
    .or_else(|| discover_from(&current_dir(ev)));
    let root = match root {
        Some(r) => r,
        None => return failure("Project::load", "No project found"),
    };
    let manifest = root.join("project.lyra");
    if !manifest.exists() { return failure("Project::load", "project.lyra not found"); }
    let prof_opt: Option<String> = if args.len() >= 2 { match ev.eval(args[1].clone()) { Value::Assoc(m) => m.get("Profile").and_then(as_string), _ => None } } else { None };
    match eval_manifest_at(&manifest).and_then(|v| normalize_manifest(&root, v)) {
        Ok(v) => {
            if let Some(p) = prof_opt { if let Value::Assoc(m) = v { return Value::Assoc(manifest_overlay_profile(m, &p)); } }
            v
        }
        Err(e) => failure("Project::load", &e),
    }
}

fn project_info(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let m = project_load(ev, args);
    match m {
        Value::Assoc(mut a) => {
            let root = if let Some(Value::String(r)) = ev.get_env("ProjectRoot") {
                r
            } else {
                current_dir(ev).to_string_lossy().to_string()
            };
            Value::Assoc(HashMap::from([
                (
                    String::from("name"),
                    a.get("Project")
                        .and_then(|p| match p {
                            Value::Assoc(mm) => mm.get("Name").cloned(),
                            _ => None,
                        })
                        .unwrap_or(Value::String(String::new())),
                ),
                (
                    String::from("version"),
                    a.get("Project")
                        .and_then(|p| match p {
                            Value::Assoc(mm) => mm.get("Version").cloned(),
                            _ => None,
                        })
                        .unwrap_or(Value::String(String::new())),
                ),
                (String::from("root"), Value::String(root)),
                (
                    String::from("manifestPath"),
                    Value::String(
                        PathBuf::from(current_dir(ev))
                            .join("project.lyra")
                            .to_string_lossy()
                            .to_string(),
                    ),
                ),
                (
                    String::from("modules"),
                    a.remove("Modules").unwrap_or(Value::Assoc(HashMap::new())),
                ),
                (
                    String::from("scripts"),
                    a.remove("Scripts").unwrap_or(Value::Assoc(HashMap::new())),
                ),
                (String::from("env"), a.remove("Env").unwrap_or(Value::Assoc(HashMap::new()))),
                (
                    String::from("config"),
                    a.remove("Config").unwrap_or(Value::Assoc(HashMap::new())),
                ),
            ]))
        }
        other => other,
    }
}

// ---------- Schema & helpers ----------

fn project_schema(_ev: &mut Evaluator, _args: Vec<Value>) -> Value {
    let mut props: HashMap<String, Value> = HashMap::new();
    // Minimal, human-readable schema (informal)
    props.insert(
        "Project".into(),
        Value::Assoc(HashMap::from([
            ("type".into(), Value::String("object".into())),
            ("properties".into(), Value::Assoc(HashMap::from([
                ("Name".into(), Value::Assoc(HashMap::from([("type".into(), Value::String("string".into()))]))),
                ("Version".into(), Value::Assoc(HashMap::from([("type".into(), Value::String("string".into()))]))),
                ("Description".into(), Value::Assoc(HashMap::from([("type".into(), Value::String("string".into()))]))),
            ]))),
            ("required".into(), Value::List(vec![Value::String("Name".into()), Value::String("Version".into())])),
        ])),
    );
    props.insert(
        "Modules".into(),
        Value::Assoc(HashMap::from([("type".into(), Value::String("object".into()))])),
    );
    props.insert(
        "Scripts".into(),
        Value::Assoc(HashMap::from([("type".into(), Value::String("object".into()))])),
    );
    props.insert(
        "Env".into(),
        Value::Assoc(HashMap::from([("type".into(), Value::String("object".into()))])),
    );
    props.insert(
        "Config".into(),
        Value::Assoc(HashMap::from([("type".into(), Value::String("object".into()))])),
    );
    props.insert(
        "Profiles".into(),
        Value::Assoc(HashMap::from([("type".into(), Value::String("object".into()))])),
    );
    props.insert(
        "Deps".into(),
        Value::Assoc(HashMap::from([("type".into(), Value::String("object".into()))])),
    );
    Value::Assoc(HashMap::from([
        ("type".into(), Value::String("object".into())),
        ("properties".into(), Value::Assoc(props)),
        (
            "required".into(),
            Value::List(vec![Value::String("Project".into())]),
        ),
    ]))
}

fn manifest_overlay_profile(mut top: HashMap<String, Value>, profile: &str) -> HashMap<String, Value> {
    let layer_assoc: Option<HashMap<String, Value>> = top
        .get("Profiles")
        .and_then(|v| if let Value::Assoc(p) = v { p.get(profile) } else { None })
        .and_then(|v| if let Value::Assoc(m) = v { Some(m.clone()) } else { None });
    if let Some(layer) = layer_assoc {
        for k in ["Env", "Config", "Scripts"].iter() {
            if let Some(Value::Assoc(ov)) = layer.get(*k) {
                let base = top.entry((*k).into()).or_insert(Value::Assoc(HashMap::new()));
                if let Value::Assoc(bm) = base {
                    for (kk, vv) in ov { bm.insert(kk.clone(), vv.clone()); }
                }
            }
        }
    }
    top
}

fn project_modules(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let profile = args.get(1).and_then(|v| match ev.eval(v.clone()) { Value::Assoc(m) => m.get("Profile").cloned(), _ => None });
    let rootarg = args.get(0).cloned();
    let loaded = project_load(ev, rootarg.into_iter().collect());
    match loaded {
        Value::Assoc(m) => {
            let mut top = m.clone();
            if let Some(Value::String(p)) = profile { top = manifest_overlay_profile(top, &p); }
            match top.get("Modules") {
                Some(Value::Assoc(mm)) => Value::Assoc(mm.clone()),
                _ => Value::Assoc(HashMap::new()),
            }
        }
        _ => Value::Assoc(HashMap::new()),
    }
}

fn project_files(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match project_modules(ev, args) {
        Value::Assoc(mm) => {
            let mut xs: Vec<Value> = Vec::new();
            for (_k, v) in mm { if let Value::String(s) = v { xs.push(Value::String(s)); } }
            Value::List(xs)
        }
        _ => Value::List(vec![]),
    }
}

fn normalize_script_entry(root: &Path, name: &str, v: &Value) -> (String, Value) {
    // Accept only Assoc for now; fill defaults.
    if let Value::Assoc(m) = v {
        let kind = m.get("Kind").and_then(as_string).unwrap_or_else(|| "Lyra".into());
        let mut out: HashMap<String, Value> = HashMap::new();
        out.insert("Kind".into(), Value::String(kind.clone()));
        match kind.as_str() {
            "Lyra" => {
                if let Some(Value::String(modn)) | Some(Value::Symbol(modn)) = m.get("Module").or_else(|| m.get("module")) { out.insert("Module".into(), Value::String(modn.clone())); }
                if let Some(Value::String(entry)) | Some(Value::Symbol(entry)) = m.get("Entry").or_else(|| m.get("entry")) { out.insert("Entry".into(), Value::String(entry.clone())); }
            }
            "Shell" => {
                if let Some(Value::String(cmd)) | Some(Value::Symbol(cmd)) = m.get("Cmd").or_else(|| m.get("cmd")) { out.insert("Cmd".into(), Value::String(cmd.clone())); }
            }
            _ => {}
        }
        // Common fields
        if let Some(Value::List(a)) = m.get("Args").or_else(|| m.get("args")) { out.insert("Args".into(), Value::List(a.clone())); } else { out.insert("Args".into(), Value::List(vec![])); }
        if let Some(Value::Assoc(env)) = m.get("Env").or_else(|| m.get("env")) { out.insert("Env".into(), Value::Assoc(env.clone())); } else { out.insert("Env".into(), Value::Assoc(HashMap::new())); }
        let cwd = m.get("Cwd").or_else(|| m.get("cwd")).and_then(as_string).unwrap_or_else(|| root.to_string_lossy().to_string());
        out.insert("Cwd".into(), Value::String(cwd));
        if let Some(Value::List(xs)) = m.get("Before").or_else(|| m.get("before")) { out.insert("Before".into(), Value::List(xs.clone())); } else { out.insert("Before".into(), Value::List(vec![])); }
        if let Some(Value::List(xs)) = m.get("After").or_else(|| m.get("after")) { out.insert("After".into(), Value::List(xs.clone())); } else { out.insert("After".into(), Value::List(vec![])); }
        return (name.into(), Value::Assoc(out));
    }
    (name.into(), Value::Assoc(HashMap::new()))
}

fn project_scripts(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // ProjectScripts[root?, opts?{Profile}]
    let profile = args.get(1).and_then(|v| match ev.eval(v.clone()) { Value::Assoc(m) => m.get("Profile").cloned(), _ => None });
    let rootarg = args.get(0).cloned();
    let loaded = project_load(ev, rootarg.into_iter().collect());
    match loaded {
        Value::Assoc(mut top) => {
            // root path for Cwd default
            let root_path = if let Some(Value::String(s)) = ev.get_env("ProjectRoot") { PathBuf::from(s) } else { current_dir(ev) };
            // Apply profile overlay if requested
            if let Some(Value::String(p)) = profile { top = manifest_overlay_profile(top.clone(), &p); }
            if let Some(Value::Assoc(sm)) = top.get("Scripts") {
                let mut out: HashMap<String, Value> = HashMap::new();
                for (k, v) in sm.iter() {
                    let (kk, vv) = normalize_script_entry(&root_path, k, v);
                    out.insert(kk, vv);
                }
                Value::Assoc(out)
            } else {
                Value::Assoc(HashMap::new())
            }
        }
        _ => Value::Assoc(HashMap::new()),
    }
}

fn project_run(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // ProjectRun[name, args?, opts?]
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("ProjectRun".into())), args }; }
    let name = match ev.eval(args[0].clone()) { Value::String(s) | Value::Symbol(s) => s, _ => return failure("Project::run", "Script name must be a string") };
    let user_args: Vec<Value> = match args.get(1).map(|v| ev.eval(v.clone())) { Some(Value::List(vs)) => vs, Some(v) => vec![v], _ => vec![] };
    let (profile_opt, root_opt, env_overlay): (Option<String>, Option<String>, HashMap<String, Value>) = match args.get(2).map(|v| ev.eval(v.clone())) {
        Some(Value::Assoc(m)) => {
            let prof = m.get("Profile").and_then(as_string);
            let root = m.get("Root").and_then(as_string);
            let e = m.get("Env").and_then(|v| if let Value::Assoc(a)=v { Some(a.clone()) } else { None }).unwrap_or_default();
            (prof, root, e)
        }
        _ => (None, None, HashMap::new()),
    };
    // Load scripts
    let mut script_args: Vec<Value> = Vec::new();
    if let Some(r) = root_opt { script_args.push(Value::String(r)); }
    if let Some(p) = &profile_opt { script_args.push(Value::Assoc(HashMap::from([(String::from("Profile"), Value::String(p.clone()))]))); }
    let scripts_v = project_scripts(ev, script_args);
    let (root_dir, _) = {
        let base = current_dir(ev);
        let root = discover_from(&base).unwrap_or(base);
        (root, ())
    };
    match scripts_v {
        Value::Assoc(map) => {
            if let Some(Value::Assoc(s)) = map.get(&name) {
                let kind = s.get("Kind").and_then(as_string).unwrap_or_else(|| "Lyra".into());
                let cwd = s.get("Cwd").and_then(as_string).unwrap_or_else(|| root_dir.to_string_lossy().to_string());
                // Apply Env overlay (temp) for Lyra; pass through to Shell
                let mut old_env: HashMap<String, Option<Value>> = HashMap::new();
                for (k, v) in env_overlay.iter() { old_env.insert(k.clone(), ev.get_env(k)); ev.set_env(k, v.clone()); }
                let t0 = std::time::Instant::now();
                let result = match kind.as_str() {
                    "Shell" => {
                        let cmd = s.get("Cmd").and_then(as_string).unwrap_or_else(|| String::from(""));
                        let base_args: Vec<Value> = match s.get("Args") { Some(Value::List(vs)) => vs.clone(), _ => vec![] };
                        let mut argv: Vec<Value> = base_args; argv.extend(user_args.clone());
                        let mut opts = HashMap::new();
                        opts.insert("Cwd".into(), Value::String(cwd.clone()));
                        if !env_overlay.is_empty() { opts.insert("Env".into(), Value::Assoc(env_overlay.clone())); }
                        ev.eval(Value::Expr { head: Box::new(Value::Symbol("Run".into())), args: vec![Value::String(cmd), Value::List(argv), Value::Assoc(opts)] })
                    }
                    _ => {
                        // Lyra script: Using[modulePathOrName]; Entry[args…]
                        let modname = s.get("Module").and_then(as_string).unwrap_or_else(|| String::from("main"));
                        // Try to resolve module path via Modules map
                        let modules = project_modules(ev, Vec::new());
                        let mod_path = match modules {
                            Value::Assoc(mm) => mm.get(&modname).and_then(|v| as_string(v)).unwrap_or(modname.clone()),
                            _ => modname.clone(),
                        };
                        // Load module (pathish is OK per module::Using)
                        let _ = ev.eval(Value::Expr { head: Box::new(Value::Symbol("Using".into())), args: vec![Value::String(mod_path)] });
                        let entry = s.get("Entry").and_then(as_string).unwrap_or_else(|| String::from("Main"));
                        ev.eval(Value::Expr { head: Box::new(Value::Symbol(entry)), args: user_args.clone() })
                    }
                };
                let elapsed = t0.elapsed().as_millis() as i64;
                // Restore env overlay
                for (k, prev) in old_env.into_iter() { match prev { Some(v) => ev.set_env(&k, v), None => ev.unset_env(&k) } }
                Value::Assoc(HashMap::from([
                    ("ok".into(), Value::Boolean(true)),
                    ("status".into(), Value::Integer(0)),
                    ("durationMs".into(), Value::Integer(elapsed)),
                    ("value".into(), result),
                    ("script".into(), Value::String(name)),
                    ("cwd".into(), Value::String(cwd)),
                ]))
            } else {
                failure("Project::run", &format!("Unknown script: {}", name))
            }
        }
        _ => failure("Project::run", "No scripts"),
    }
}

fn project_graph(ev: &mut Evaluator, _args: Vec<Value>) -> Value {
    // Best-effort: Modules + naive Using["name"] scan for imports
    let modules = project_modules(ev, Vec::new());
    let mut imports: HashMap<String, Vec<String>> = HashMap::new();
    let mut files: Vec<String> = Vec::new();
    match &modules {
        Value::Assoc(mm) => {
            for (mname, vp) in mm.iter() {
                if let Value::String(path) = vp {
                    files.push(path.clone());
                    if let Ok(src) = std::fs::read_to_string(path) {
                        let mut deps: Vec<String> = Vec::new();
                        for cap in src.match_indices("Using[") { // simple heuristic
                            let rest = &src[cap.0+6..];
                            // try string literal first
                            if let Some(iq) = rest.find('"') { let s2 = &rest[iq+1..]; if let Some(jq) = s2.find('"') { let name = &s2[..jq]; if !name.is_empty() { deps.push(name.into()); continue; } } }
                            // try symbol letters
                            let sym: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c=='_' || *c=='$').collect();
                            if !sym.is_empty() { deps.push(sym); }
                        }
                        deps.sort(); deps.dedup();
                        imports.insert(mname.clone(), deps);
                    }
                }
            }
        }
        _ => {}
    }
    // reverse map
    let mut rev: HashMap<String, Vec<String>> = HashMap::new();
    for (m, ds) in imports.iter() { for d in ds { rev.entry(d.clone()).or_default().push(m.clone()); } }
    for (_k, v) in rev.iter_mut() { v.sort(); v.dedup(); }
    Value::Assoc(HashMap::from([
        ("Modules".into(), modules),
        ("Imports".into(), Value::Assoc(imports.into_iter().map(|(k, vs)| (k, Value::List(vs.into_iter().map(Value::String).collect()))).collect())),
        ("Reverse".into(), Value::Assoc(rev.into_iter().map(|(k, vs)| (k, Value::List(vs.into_iter().map(Value::String).collect()))).collect())),
        ("Files".into(), Value::List(files.into_iter().map(Value::String).collect())),
    ]))
}

fn project_watch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // ProjectWatch[name, opts?]
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("ProjectWatch".into())), args }; }
    let name = match ev.eval(args[0].clone()) { Value::String(s) | Value::Symbol(s) => s, _ => return failure("Project::watch", "Script name must be a string") };
    let (profile_opt, root_opt, debounce_ms): (Option<String>, Option<String>, Option<i64>) = match args.get(1).map(|v| ev.eval(v.clone())) {
        Some(Value::Assoc(m)) => {
            let prof = m.get("Profile").and_then(as_string);
            let root = m.get("Root").and_then(as_string);
            let db = m.get("DebounceMs").and_then(|v| if let Value::Integer(n)=v { Some(*n) } else { None });
            (prof, root, db)
        }
        _ => (None, None, None),
    };
    let base = root_opt.map(PathBuf::from).unwrap_or_else(|| discover_from(&current_dir(ev)).unwrap_or(current_dir(ev)));
    let root = base;
    // Build handler: () => ProjectRun[name, {}, <|Profile, Root|>]
    let mut run_args: Vec<Value> = vec![Value::String(name.clone()), Value::List(vec![])];
    let mut optm: HashMap<String, Value> = HashMap::new();
    optm.insert("Root".into(), Value::String(root.to_string_lossy().to_string()));
    if let Some(p) = profile_opt.clone() { optm.insert("Profile".into(), Value::String(p)); }
    run_args.push(Value::Assoc(optm));
    let handler = Value::pure_function(None, Value::expr(Value::symbol("ProjectRun"), run_args));
    // Call WatchDirectory on root
    let mut wopts: HashMap<String, Value> = HashMap::new();
    wopts.insert("Recursive".into(), Value::Boolean(true));
    if let Some(n) = debounce_ms { if n > 0 { wopts.insert("DebounceMs".into(), Value::Integer(n)); } }
    let token = ev.eval(Value::expr(Value::symbol("WatchDirectory"), vec![Value::String(root.to_string_lossy().to_string()), handler, Value::Assoc(wopts)]));
    // Extract id if possible
    let (id, active) = match &token { Value::Assoc(m) => (m.get("id").and_then(|v| if let Value::Integer(n)=v{Some(*n)}else{None}).unwrap_or(0), true), _ => (0, false) };
    Value::Assoc(HashMap::from([
        ("__type".into(), Value::String("ProjectWatch".into())),
        ("id".into(), Value::Integer(id)),
        ("active".into(), Value::Boolean(active)),
        ("name".into(), Value::String(name)),
        ("root".into(), Value::String(root.to_string_lossy().to_string())),
        ("token".into(), token),
    ]))
}

// ---------------- Validation & Init ----------------

#[derive(Default)]
struct Diag {
    errors: Vec<(String, String, String)>, // (path, code, message)
    warnings: Vec<(String, String, String)>, // (path, code, message)
}

fn push_err(d: &mut Diag, path: &str, code: &str, msg: &str) {
    d.errors.push((path.to_string(), code.to_string(), msg.to_string()));
}
fn push_warn(d: &mut Diag, path: &str, code: &str, msg: &str) {
    d.warnings.push((path.to_string(), code.to_string(), msg.to_string()));
}

fn validate_manifest(_root: &Path, v: &Value) -> Value {
    use std::collections::HashMap;
    let mut d = Diag::default();
    let top = match v {
        Value::Assoc(m) => m.clone(),
        other => {
            push_err(
                &mut d,
                "",
                "type.top",
                &format!("project.lyra must evaluate to an association, got {:?}", other),
            );
            return Value::Assoc(HashMap::from([
                ("ok".into(), Value::Boolean(false)),
                (
                    "errors".into(),
                    Value::List(
                        d.errors
                            .into_iter()
                            .map(|(p, c, m)| {
                                Value::Assoc(HashMap::from([
                                    (String::from("path"), Value::String(p)),
                                    (String::from("code"), Value::String(c)),
                                    (String::from("message"), Value::String(m)),
                                ]))
                            })
                            .collect(),
                    ),
                ),
                ("warnings".into(), Value::List(Vec::new())),
            ]));
        }
    };
    // Allowed keys
    let allowed: std::collections::HashSet<&str> =
        ["Project", "Modules", "Scripts", "Env", "Config", "Workspace", "Deps"]
            .into_iter()
            .collect();
    for k in top.keys() {
        if !allowed.contains(k.as_str()) {
            push_warn(&mut d, &format!("{}", k), "key.unknown", &format!("Unknown top-level key '{}'; allowed: Project, Modules, Scripts, Env, Config, Workspace, Deps", k));
        }
    }
    // Project
    match top.get("Project") {
        Some(Value::Assoc(pm)) => {
            match pm.get("Name") {
                Some(Value::String(_)) | Some(Value::Symbol(_)) => {}
                Some(other) => push_err(
                    &mut d,
                    "Project.Name",
                    "type.string",
                    &format!("expected string/symbol, got {:?}", other),
                ),
                None => push_err(&mut d, "Project.Name", "missing", "Project.Name is required"),
            }
            match pm.get("Version") {
                Some(Value::String(_)) | Some(Value::Symbol(_)) => {}
                Some(other) => push_err(
                    &mut d,
                    "Project.Version",
                    "type.string",
                    &format!("expected string/symbol, got {:?}", other),
                ),
                None => {
                    push_err(&mut d, "Project.Version", "missing", "Project.Version is required")
                }
            }
            if let Some(other) = pm.get("Description") {
                if !matches!(other, Value::String(_) | Value::Symbol(_)) {
                    push_warn(
                        &mut d,
                        "Project.Description",
                        "type.string",
                        &format!("expected string/symbol, got {:?}", other),
                    );
                }
            }
        }
        Some(other) => push_err(
            &mut d,
            "Project",
            "type.assoc",
            &format!("expected association, got {:?}", other),
        ),
        None => push_err(&mut d, "Project", "missing", "Top-level Project assoc is required"),
    }
    // Modules
    match top.get("Modules") {
        Some(Value::Assoc(mm)) => {
            for (name, pathv) in mm.iter() {
                let pname = match name.as_str() {
                    s => s,
                };
                match pathv {
                    Value::String(s) | Value::Symbol(s) => {
                        let p = PathBuf::from(s);
                        if !p.exists() {
                            push_err(
                                &mut d,
                                &format!("Modules.{}", pname),
                                "path.missing",
                                &format!("module path does not exist: {}", s),
                            );
                        } else if p.is_dir() {
                            push_warn(
                                &mut d,
                                &format!("Modules.{}", pname),
                                "path.is_dir",
                                &format!("expected file, got directory: {}", s),
                            );
                        }
                    }
                    other => push_err(
                        &mut d,
                        &format!("Modules.{}", pname),
                        "type.string",
                        &format!("expected string path, got {:?}", other),
                    ),
                }
            }
        }
        Some(other) => push_err(
            &mut d,
            "Modules",
            "type.assoc",
            &format!("expected association, got {:?}", other),
        ),
        None => {}
    }
    // Scripts
    match top.get("Scripts") {
        Some(Value::Assoc(sm)) => {
            for (sname, val) in sm.iter() {
                if let Value::Assoc(am) = val {
                    // Validate Module reference if present
                    if let Some(Value::String(mn)) | Some(Value::Symbol(mn)) = am.get("Module") {
                        if let Some(Value::Assoc(mm)) = top.get("Modules") {
                            if !mm.contains_key(mn) {
                                push_err(
                                    &mut d,
                                    &format!("Scripts.{}", sname),
                                    "script.unknownModule",
                                    &format!("references unknown module '{}'", mn),
                                );
                            }
                        }
                    }
                } else if !matches!(val, Value::String(_) | Value::Symbol(_) | Value::List(_)) {
                    push_warn(
                        &mut d,
                        &format!("Scripts.{}", sname),
                        "type.unsupported",
                        &format!("script value should be assoc/string/list; got {:?}", val),
                    );
                }
            }
        }
        Some(other) => push_err(
            &mut d,
            "Scripts",
            "type.assoc",
            &format!("expected association, got {:?}", other),
        ),
        None => {}
    }
    // Env
    match top.get("Env") {
        Some(Value::Assoc(em)) => {
            for (k, v) in em.iter() {
                if !matches!(
                    v,
                    Value::String(_) | Value::Symbol(_) | Value::Integer(_) | Value::Real(_)
                ) {
                    push_warn(
                        &mut d,
                        &format!("Env.{}", k),
                        "type.string",
                        &format!("env var should be string/number, got {:?}", v),
                    );
                }
            }
        }
        Some(other) => {
            push_err(&mut d, "Env", "type.assoc", &format!("expected association, got {:?}", other))
        }
        None => {}
    }
    // Config: best-effort assoc
    if let Some(other) = top.get("Config") {
        if !matches!(other, Value::Assoc(_)) {
            push_warn(
                &mut d,
                "Config",
                "type.assoc",
                &format!("expected association, got {:?}", other),
            );
        }
    }
    // Workspace (optional): Members list if present
    if let Some(Value::Assoc(ws)) = top.get("Workspace") {
        if let Some(m) =
            ws.get("Members").or_else(|| ws.get("members")).or_else(|| ws.get("members"))
        {
            if let Value::List(vs) = m {
                for v in vs {
                    if !matches!(v, Value::String(_) | Value::Symbol(_)) {
                        push_err(
                            &mut d,
                            "Workspace.Members",
                            "type.string",
                            &format!("expected string entries, got {:?}", v),
                        );
                    }
                }
            } else {
                push_err(
                    &mut d,
                    "Workspace.Members",
                    "type.list",
                    &format!("expected list of strings, got {:?}", m),
                );
            }
        }
    } else if let Some(other) = top.get("Workspace") {
        if !matches!(other, Value::Symbol(_)) {
            push_warn(
                &mut d,
                "Workspace",
                "type.assocOrNull",
                &format!("expected assoc or Null, got {:?}", other),
            );
        }
    }

    let mut errors_v: Vec<Value> = Vec::new();
    for (p, c, m) in d.errors.into_iter() {
        errors_v.push(Value::Assoc(HashMap::from([
            (String::from("path"), Value::String(p)),
            (String::from("code"), Value::String(c)),
            (String::from("message"), Value::String(m)),
        ])));
    }
    let mut warns_v: Vec<Value> = Vec::new();
    for (p, c, m) in d.warnings.into_iter() {
        warns_v.push(Value::Assoc(HashMap::from([
            (String::from("path"), Value::String(p)),
            (String::from("code"), Value::String(c)),
            (String::from("message"), Value::String(m)),
        ])));
    }
    Value::Assoc(HashMap::from([
        (String::from("ok"), Value::Boolean(errors_v.is_empty())),
        (String::from("errors"), Value::List(errors_v)),
        (String::from("warnings"), Value::List(warns_v)),
    ]))
}

fn project_validate(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // ProjectValidate[] or ProjectValidate[<path>]
    let root = if args.len() == 1 {
        as_string(&ev.eval(args[0].clone())).map(PathBuf::from)
    } else {
        None
    }
    .or_else(|| {
        if let Some(Value::String(s)) = ev.get_env("ProjectRoot") {
            Some(PathBuf::from(s))
        } else {
            None
        }
    })
    .or_else(|| discover_from(&current_dir(ev)));
    let root = match root {
        Some(r) => r,
        None => return failure("Project::validate", "No project found"),
    };
    let manifest = root.join("project.lyra");
    match eval_manifest_at(&manifest).and_then(|v| normalize_manifest(&root, v)) {
        Ok(v) => validate_manifest(&root, &v),
        Err(e) => Value::Assoc(
            std::iter::IntoIterator::into_iter([
                (String::from("ok"), Value::Boolean(false)),
                (
                    String::from("errors"),
                    Value::List(vec![Value::Assoc(
                        std::iter::IntoIterator::into_iter([
                            (String::from("path"), Value::String(String::from("project.lyra"))),
                            (String::from("code"), Value::String(String::from("parse"))),
                            (String::from("message"), Value::String(e)),
                        ])
                        .collect(),
                    )]),
                ),
                (String::from("warnings"), Value::List(vec![])),
            ])
            .collect(),
        ),
    }
}

fn project_init(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    use std::collections::HashMap;
    // ProjectInit[] | ProjectInit["dir"] | ProjectInit[<|Name->.., Dir->..|>]
    let mut name_opt: Option<String> = None;
    let mut dir_opt: Option<PathBuf> = None;
    if args.len() == 1 {
        match ev.eval(args[0].clone()) {
            Value::String(s) | Value::Symbol(s) => {
                dir_opt = Some(PathBuf::from(s));
            }
            Value::Assoc(m) => {
                if let Some(Value::String(s)) | Some(Value::Symbol(s)) =
                    m.get("Name").or_else(|| m.get("name"))
                {
                    name_opt = Some(s.clone());
                }
                if let Some(Value::String(s)) | Some(Value::Symbol(s)) =
                    m.get("Dir").or_else(|| m.get("dir"))
                {
                    dir_opt = Some(PathBuf::from(s));
                }
            }
            _ => {}
        }
    }
    let cwd = current_dir(ev);
    let dir = dir_opt.unwrap_or_else(|| cwd.clone());
    let name = name_opt.unwrap_or_else(|| {
        dir.file_name().and_then(|s| s.to_str()).unwrap_or("lyra-project").to_string()
    });
    // Create structure
    if let Err(e) = std::fs::create_dir_all(dir.join("src")) {
        return failure("Project::init", &format!("create src dir: {}", e));
    }
    let manifest_path = dir.join("project.lyra");
    if manifest_path.exists() {
        return failure("Project::init", "project.lyra already exists");
    }
    let template = format!(
        "Exported[{{}}];\n\n<|\n  \"Project\" -> <| \"Name\"->\"{}\", \"Version\"->\"0.1.0\", \"Description\"->\"\" |>,\n  \"Modules\" -> <| \"main\" -> ResolveRelative[\"src/main.lyra\"] |>,\n  \"Scripts\" -> <| \"run\" -> <| \"Module\"->\"main\", \"Entry\"->\"Main\" |> |>,\n  \"Env\" -> <||>,\n  \"Config\" -> <||>,\n  \"Workspace\" -> Null,\n  \"Deps\" -> <||>\n|>\n",
        name
    );
    if let Err(e) = std::fs::write(&manifest_path, template) {
        return failure("Project::init", &format!("write project.lyra: {}", e));
    }
    let main_src = "Main[] := (Puts[\"Hello from Lyra!\"]; )\n";
    if let Err(e) = std::fs::write(dir.join("src/main.lyra"), main_src) {
        return failure("Project::init", &format!("write src/main.lyra: {}", e));
    }
    // Optional .gitignore if not present
    let gi = dir.join(".gitignore");
    if !gi.exists() {
        let _ = std::fs::write(gi, "target\nbuild\n.DS_Store\n");
    }
    Value::Assoc(HashMap::from([
        (String::from("ok"), Value::Boolean(true)),
        (String::from("root"), Value::String(dir.to_string_lossy().to_string())),
        (String::from("manifest"), Value::String(manifest_path.to_string_lossy().to_string())),
        (
            String::from("created"),
            Value::List(vec![
                Value::String(String::from("project.lyra")),
                Value::String(String::from("src/main.lyra")),
            ]),
        ),
    ]))
}
