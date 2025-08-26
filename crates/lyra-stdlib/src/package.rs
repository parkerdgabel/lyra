use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
use crate::register_if;
use std::fs;
use std::path::{Path, PathBuf};

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

pub fn register_package(ev: &mut Evaluator) {
    ev.register("Using", using_fn as NativeFn, Attributes::HOLD_FIRST);
    ev.register("Unuse", unuse_fn as NativeFn, Attributes::empty());
    ev.register("ReloadPackage", reload_package_fn as NativeFn, Attributes::empty());
    ev.register("WithPackage", with_package_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("BeginModule", begin_module_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("EndModule", end_module_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Export", export_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Private", private_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("CurrentModule", current_module_fn as NativeFn, Attributes::empty());
    ev.register("ModulePath", module_path_fn as NativeFn, Attributes::empty());
    ev.register("SetModulePath", set_module_path_fn as NativeFn, Attributes::empty());
    ev.register("PackageInfo", package_info_fn as NativeFn, Attributes::empty());
    ev.register("PackageVersion", package_version_fn as NativeFn, Attributes::empty());
    ev.register("PackagePath", package_path_fn as NativeFn, Attributes::empty());
    ev.register("ListInstalledPackages", list_installed_packages_fn as NativeFn, Attributes::empty());
    ev.register("NewPackage", new_package_fn as NativeFn, Attributes::empty());
    ev.register("NewModule", new_module_fn as NativeFn, Attributes::empty());
    ev.register("ImportedSymbols", imported_symbols_fn as NativeFn, Attributes::empty());
    ev.register("LoadedPackages", loaded_packages_fn as NativeFn, Attributes::empty());
    ev.register("RegisterExports", register_exports_fn as NativeFn, Attributes::empty());
    ev.register("PackageExports", package_exports_fn as NativeFn, Attributes::empty());

    // Dev/PM stubs until lyra-pm is wired in
    ev.register("BuildPackage", build_package_fn as NativeFn, Attributes::empty());
    ev.register("TestPackage", test_package_fn as NativeFn, Attributes::empty());
    ev.register("LintPackage", lint_package_fn as NativeFn, Attributes::empty());
    ev.register("PackPackage", pack_package_fn as NativeFn, Attributes::empty());
    ev.register("GenerateSBOM", generate_sbom_fn as NativeFn, Attributes::empty());
    ev.register("SignPackage", sign_package_fn as NativeFn, Attributes::empty());
    ev.register("PublishPackage", publish_package_fn as NativeFn, Attributes::empty());
    ev.register("InstallPackage", install_package_fn as NativeFn, Attributes::empty());
    ev.register("UpdatePackage", update_package_fn as NativeFn, Attributes::empty());
    ev.register("RemovePackage", remove_package_fn as NativeFn, Attributes::empty());
    ev.register("LoginRegistry", login_registry_fn as NativeFn, Attributes::empty());
    ev.register("LogoutRegistry", logout_registry_fn as NativeFn, Attributes::empty());
    ev.register("WhoAmI", whoami_registry_fn as NativeFn, Attributes::empty());
    ev.register("PackageAudit", package_audit_fn as NativeFn, Attributes::empty());
    ev.register("PackageVerify", package_verify_fn as NativeFn, Attributes::empty());
}

pub fn register_package_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str)->bool) {
    register_if(ev, pred, "Using", using_fn as NativeFn, Attributes::HOLD_FIRST);
    register_if(ev, pred, "Unuse", unuse_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "ReloadPackage", reload_package_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "WithPackage", with_package_fn as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "BeginModule", begin_module_fn as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "EndModule", end_module_fn as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "Export", export_fn as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "Private", private_fn as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "CurrentModule", current_module_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "ModulePath", module_path_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "SetModulePath", set_module_path_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "PackageInfo", package_info_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "PackageVersion", package_version_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "PackagePath", package_path_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "ListInstalledPackages", list_installed_packages_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "NewPackage", new_package_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "NewModule", new_module_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "ImportedSymbols", imported_symbols_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "LoadedPackages", loaded_packages_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "RegisterExports", register_exports_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "PackageExports", package_exports_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "BuildPackage", build_package_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "TestPackage", test_package_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "LintPackage", lint_package_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "PackPackage", pack_package_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "GenerateSBOM", generate_sbom_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "SignPackage", sign_package_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "PublishPackage", publish_package_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "InstallPackage", install_package_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "UpdatePackage", update_package_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "RemovePackage", remove_package_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "LoginRegistry", login_registry_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "LogoutRegistry", logout_registry_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "WhoAmI", whoami_registry_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "PackageAudit", package_audit_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "PackageVerify", package_verify_fn as NativeFn, Attributes::empty());
}

fn uneval(head: &str, args: Vec<Value>) -> Value { Value::Expr { head: Box::new(Value::Symbol(head.into())), args } }

fn failure(tag: &str, msg: &str) -> Value {
    Value::Assoc([
        ("tag".to_string(), Value::String(tag.to_string())),
        ("message".to_string(), Value::String(msg.to_string())),
    ].into_iter().collect())
}

// ---------- Environment helpers ----------

fn get_env(ev: &mut Evaluator, name: &str) -> Value {
    ev.eval(Value::Symbol(name.into()))
}

fn set_env(ev: &mut Evaluator, name: &str, v: Value) {
    let _ = ev.eval(Value::Expr { head: Box::new(Value::Symbol("Set".into())), args: vec![Value::Symbol(name.into()), v] });
}

fn default_package_path() -> Vec<String> {
    vec![
        "./.lyra/packages".to_string(),
        "./packages".to_string(),
    ]
}

fn read_package_path(ev: &mut Evaluator) -> Vec<String> {
    match get_env(ev, "$PackagePath") {
        Value::List(vs) => vs.into_iter().filter_map(|v| if let Value::String(s)=v { Some(s) } else { None }).collect(),
        _ => default_package_path(),
    }
}

fn write_package_path(ev: &mut Evaluator, paths: Vec<String>) {
    set_env(ev, "$PackagePath", Value::List(paths.into_iter().map(Value::String).collect()));
}

fn loaded_packages_assoc(ev: &mut Evaluator) -> std::collections::HashMap<String, Value> {
    match get_env(ev, "$LoadedPackages") {
        Value::Assoc(m) => m,
        _ => std::collections::HashMap::new(),
    }
}

fn write_loaded_packages(ev: &mut Evaluator, m: std::collections::HashMap<String, Value>) { set_env(ev, "$LoadedPackages", Value::Assoc(m)); }

fn get_module_exports(ev: &mut Evaluator) -> Vec<String> {
    match get_env(ev, "$ModuleExports") {
        Value::List(vs) => vs.into_iter().filter_map(|v| if let Value::String(s)=v { Some(s) } else { None }).collect(),
        _ => Vec::new(),
    }
}

fn write_module_exports(ev: &mut Evaluator, names: Vec<String>) { set_env(ev, "$ModuleExports", Value::List(names.into_iter().map(Value::String).collect())); }

fn imported_assoc(ev: &mut Evaluator) -> std::collections::HashMap<String, Value> {
    match get_env(ev, "$ImportedSymbols") {
        Value::Assoc(m) => m,
        _ => std::collections::HashMap::new(),
    }
}

fn write_imported_assoc(ev: &mut Evaluator, m: std::collections::HashMap<String, Value>) {
    set_env(ev, "$ImportedSymbols", Value::Assoc(m));
}

fn unset_var(ev: &mut Evaluator, name: &str) {
    let _ = ev.eval(Value::Expr { head: Box::new(Value::Symbol("Unset".into())), args: vec![Value::Symbol(name.into())] });
}

fn pkg_exports_assoc(ev: &mut Evaluator) -> std::collections::HashMap<String, Vec<String>> {
    match get_env(ev, "$PackageExports") {
        Value::Assoc(m) => {
            let mut out: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
            for (k, v) in m {
                if let Value::List(vs) = v {
                    let mut xs: Vec<String> = Vec::new();
                    for it in vs {
                        match it {
                            Value::String(s) => xs.push(s),
                            Value::Symbol(s) => xs.push(s),
                            _ => {}
                        }
                    }
                    out.insert(k, xs);
                }
            }
            out
        }
        _ => std::collections::HashMap::new(),
    }
}

fn write_pkg_exports(ev: &mut Evaluator, m: std::collections::HashMap<String, Vec<String>>) {
    let mut out = std::collections::HashMap::new();
    for (k, vs) in m { out.insert(k, Value::List(vs.into_iter().map(Value::String).collect())); }
    set_env(ev, "$PackageExports", Value::Assoc(out));
}

// ---------- Core functions ----------

fn using_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Using["name" | {"name", ver?} | <|Name->..|>, opts?]
    if args.is_empty() { return uneval("Using", args); }
    let (name, _ver_req) = match ev.eval(args[0].clone()) {
        Value::String(s) | Value::Symbol(s) => (s, None),
        Value::List(vs) if vs.len()>=1 => {
            match &vs[0] { Value::String(s)|Value::Symbol(s) => (s.clone(), vs.get(1).and_then(|v| if let Value::String(t)=v { Some(t.clone()) } else { None })), _ => return uneval("Using", args) }
        }
        Value::Assoc(m) => {
            let nm = m.get("name").or_else(|| m.get("Name")).and_then(|v| if let Value::String(s)=v { Some(s.clone()) } else { None });
            match nm { Some(s) => (s, m.get("version").or_else(|| m.get("Version")).and_then(|v| if let Value::String(t)=v { Some(t.clone()) } else { None })), None => return uneval("Using", args) }
        }
        other => return uneval("Using", vec![other]),
    };
    // Resolve path
    let paths = read_package_path(ev);
    let mut found: Option<PathBuf> = None;
    for base in paths {
        let p = Path::new(&base).join(&name);
        if p.is_dir() {
            let has_manifest = p.join("lyra.toml").exists() || p.join("LYRA-PKG.json").exists();
            let has_src = p.join("src").is_dir();
            if has_manifest || has_src { found = Some(p); break; }
        }
    }
    let pkg_path = match found { Some(p)=>p, None => return failure("Package::notFound", &format!("Package '{}' not found on $PackagePath", name)) };
    // Idempotent: record in $LoadedPackages; attach import options if provided
    let import_spec = if args.len()>=2 {
        match ev.eval(args[1].clone()) {
            Value::Assoc(m) => {
                let mut spec = std::collections::HashMap::new();
                if let Some(v) = m.get("Import").or_else(|| m.get("import")) { spec.insert("import".into(), v.clone()); }
                if let Some(v) = m.get("Except").or_else(|| m.get("except")) { spec.insert("except".into(), v.clone()); }
                Value::Assoc(spec)
            }
            _ => Value::Assoc(std::collections::HashMap::new()),
        }
    } else { Value::Assoc(std::collections::HashMap::new()) };
    // Compute effective imported set representation for $ImportedSymbols
    let mut imported = imported_assoc(ev);
    let exports_map = pkg_exports_assoc(ev);
    let exports_opt = exports_map.get(&name);
    let eff: Value = match &import_spec {
        Value::Assoc(m) => {
            let imp = m.get("import");
            let exc = m.get("except");
            match (imp, exc) {
                (Some(Value::Symbol(s)), _) | (Some(Value::String(s)), _) if s == "All" => {
                    if let Some(exports) = exports_opt {
                        // Resolve to explicit list = exports minus except
                        let mut items: Vec<String> = exports.clone();
                        if let Some(Value::List(exl)) = exc {
                            let mut exset: std::collections::HashSet<String> = std::collections::HashSet::new();
                            for v in exl { if let Value::String(s)=v { exset.insert(s.clone()); } else if let Value::Symbol(s)=v { exset.insert(s.clone()); } }
                            items.retain(|s| !exset.contains(s));
                        }
                        Value::List(items.into_iter().map(Value::String).collect())
                    } else {
                        // store as <|all->True, except->{...}?|> until exports are known
                        let mut a = std::collections::HashMap::new();
                        a.insert("all".into(), Value::Boolean(true));
                        if let Some(Value::List(vs)) = exc { a.insert("except".into(), Value::List(vs.clone())); }
                        Value::Assoc(a)
                    }
                }
                (Some(Value::List(vs)), _) => {
                    // list minus except if any (only string/symbol items kept)
                    let mut items: Vec<String> = Vec::new();
                    for v in vs { if let Value::String(s) = v { items.push(s.clone()); } else if let Value::Symbol(s) = v { items.push(s.clone()); } }
                    // intersect with exports if available
                    if let Some(exports) = exports_opt {
                        let allow: std::collections::HashSet<String> = exports.iter().cloned().collect();
                        items.retain(|s| allow.contains(s));
                    }
                    if let Some(Value::List(exl)) = exc {
                        let mut exset: std::collections::HashSet<String> = std::collections::HashSet::new();
                        for v in exl { if let Value::String(s)=v { exset.insert(s.clone()); } else if let Value::Symbol(s)=v { exset.insert(s.clone()); } }
                        items.retain(|s| !exset.contains(s));
                    }
                    Value::List(items.into_iter().map(Value::String).collect())
                }
                _ => Value::List(vec![]),
            }
        }
        _ => Value::List(vec![]),
    };
    imported.insert(name.clone(), eff);
    write_imported_assoc(ev, imported);
    let mut loaded = loaded_packages_assoc(ev);
    loaded.insert(name.clone(), Value::Assoc([
        ("name".into(), Value::String(name.clone())),
        ("path".into(), Value::String(pkg_path.to_string_lossy().into())),
        ("options".into(), import_spec),
    ].into_iter().collect()));
    write_loaded_packages(ev, loaded);
    Value::Boolean(true)
}

fn unuse_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return uneval("Unuse", args); }
    let name = match ev.eval(args[0].clone()) { Value::String(s)|Value::Symbol(s)=>s, other=> return uneval("Unuse", vec![other]) };
    // Remove loaded record
    let mut loaded = loaded_packages_assoc(ev);
    loaded.remove(&name);
    write_loaded_packages(ev, loaded);
    // Soft-hide imported symbols from env if we have an explicit list
    let mut imp = imported_assoc(ev);
    if let Some(v) = imp.remove(&name) {
        match v {
            Value::List(vs) => {
                for it in vs { if let Value::String(s)=it { unset_var(ev, &s); } else if let Value::Symbol(s)=it { unset_var(ev, &s); } }
            }
            Value::Assoc(m) => {
                // all->True case: remove exports minus except if exports known
                if let Some(Value::Boolean(true)) = m.get("all") {
                    let except: std::collections::HashSet<String> = m.get("except").and_then(|v| match v { Value::List(vs)=>Some(vs.clone()), _=>None }).map(|vs| vs.into_iter().filter_map(|v| match v { Value::String(s)=>Some(s), Value::Symbol(s)=>Some(s), _=>None }).collect()).unwrap_or_default();
                    let emap = pkg_exports_assoc(ev);
                    if let Some(exports) = emap.get(&name) {
                        for s in exports { if !except.contains(s) { unset_var(ev, s); } }
                    }
                }
            }
            _ => {}
        }
        write_imported_assoc(ev, imp);
    }
    Value::Boolean(true)
}

fn reload_package_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return uneval("ReloadPackage", args); }
    let name = match ev.eval(args[0].clone()) { Value::String(s)|Value::Symbol(s)=>s, other=> return uneval("ReloadPackage", vec![other]) };
    // Currently same as Using for stub
    using_fn(ev, vec![Value::String(name)])
}

fn with_package_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // WithPackage[name, expr]
    if args.len()!=2 { return uneval("WithPackage", args); }
    let name = match ev.eval(args[0].clone()) { Value::String(s)|Value::Symbol(s)=>s, other=> return uneval("WithPackage", vec![other, args[1].clone()]) };
    // Prepend to $PackagePath for the duration
    let paths = read_package_path(ev);
    let mut found: Option<String> = None;
    // derive a candidate path by searching and keeping exact path
    for base in paths.iter() {
        let p = Path::new(base).join(&name);
        if p.is_dir() { found = Some(base.clone()); break; }
    }
    let saved = get_env(ev, "$PackagePath");
    if let Some(base) = found {
        let mut newp = vec![base.clone()];
        newp.extend(paths.into_iter());
        write_package_path(ev, newp);
    }
    let result = ev.eval(args[1].clone());
    // restore
    set_env(ev, "$PackagePath", saved);
    result
}

// ---------- Module lifecycle ----------

fn begin_module_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return uneval("BeginModule", args); }
    let name = match ev.eval(args[0].clone()) { Value::String(s)|Value::Symbol(s)=>s, other=> return uneval("BeginModule", vec![other]) };
    set_env(ev, "$ModuleContext", Value::String(name.clone()));
    write_module_exports(ev, Vec::new());
    Value::String(name)
}

fn end_module_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if !args.is_empty() { return uneval("EndModule", args); }
    // For now, just clear context; export semantics will be wired later
    set_env(ev, "$ModuleContext", Value::Symbol("Null".into()));
    Value::Boolean(true)
}

fn export_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return uneval("Export", args); }
    let mut exports = get_module_exports(ev);
    match ev.eval(args[0].clone()) {
        Value::List(vs) => {
            for v in vs { if let Value::Symbol(s)|Value::String(s) = v { exports.push(s); } }
        }
        Value::Symbol(s) | Value::String(s) => exports.push(s),
        other => return uneval("Export", vec![other]),
    }
    // Dedup
    exports.sort(); exports.dedup();
    write_module_exports(ev, exports);
    Value::Boolean(true)
}

fn private_fn(ev: &mut Evaluator, _args: Vec<Value>) -> Value {
    // Placeholder: no-op for now
    Value::Boolean(true)
}

fn current_module_fn(ev: &mut Evaluator, _args: Vec<Value>) -> Value { get_env(ev, "$ModuleContext") }

fn module_path_fn(ev: &mut Evaluator, _args: Vec<Value>) -> Value {
    Value::List(read_package_path(ev).into_iter().map(Value::String).collect())
}

fn set_module_path_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return uneval("SetModulePath", args); }
    let paths: Vec<String> = match ev.eval(args[0].clone()) {
        Value::List(vs) => vs.into_iter().filter_map(|v| if let Value::String(s)=v { Some(s) } else { None }).collect(),
        Value::String(s) => vec![s],
        other => return uneval("SetModulePath", vec![other]),
    };
    write_package_path(ev, paths);
    Value::Boolean(true)
}

// ---------- Introspection ----------

fn package_info_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let name_opt = args.get(0).map(|v| ev.eval(v.clone())).and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s), _=>None });
    if let Some(name) = name_opt {
        // Check loaded first
        if let Value::Assoc(m) = get_env(ev, "$LoadedPackages") {
            if let Some(info) = m.get(&name) { return info.clone(); }
        }
        // Search on path
        let paths = read_package_path(ev);
        for base in paths {
            let p = Path::new(&base).join(&name);
            if p.is_dir() {
                let mut assoc = std::collections::HashMap::new();
                assoc.insert("name".into(), Value::String(name.clone()));
                assoc.insert("path".into(), Value::String(p.to_string_lossy().into()));
                if p.join("lyra.toml").exists() {
                    if let Some(ver) = read_manifest_version(&p.join("lyra.toml")) { assoc.insert("version".into(), Value::String(ver)); }
                }
                return Value::Assoc(assoc);
            }
        }
        return failure("Package::notFound", &format!("Package '{}' not found", name));
    }
    // No name: return current module/package info if any
    if let Value::Assoc(m) = get_env(ev, "$LoadedPackages") {
        return Value::List(m.into_iter().map(|(_,v)| v).collect());
    }
    Value::List(vec![])
}

fn package_version_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return uneval("PackageVersion", args); }
    match package_info_fn(ev, vec![args[0].clone()]) {
        Value::Assoc(m) => m.get("version").cloned().unwrap_or(Value::Symbol("Null".into())),
        other => other,
    }
}

fn package_path_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return module_path_fn(ev, args); }
    match package_info_fn(ev, args) {
        Value::Assoc(m) => m.get("path").cloned().unwrap_or(Value::Symbol("Null".into())),
        other => other,
    }
}

fn imported_symbols_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()==1 {
        let name = match ev.eval(args[0].clone()) { Value::String(s)|Value::Symbol(s)=>s, _=> return uneval("ImportedSymbols", args) };
        if let Value::Assoc(m) = get_env(ev, "$ImportedSymbols") { return m.get(&name).cloned().unwrap_or(Value::List(vec![])); }
        Value::List(vec![])
    } else if args.is_empty() {
        get_env(ev, "$ImportedSymbols")
    } else { uneval("ImportedSymbols", args) }
}

fn loaded_packages_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if !args.is_empty() { return uneval("LoadedPackages", args); }
    get_env(ev, "$LoadedPackages")
}

fn register_exports_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // RegisterExports[name, {sym..}] or RegisterExports[<|name->..., exports->{...}|>]
    if args.is_empty() { return uneval("RegisterExports", args); }
    let (name, ex_list) = match ev.eval(args[0].clone()) {
        Value::Assoc(m) => {
            let nm = m.get("name").or_else(|| m.get("Name"));
            let ex = m.get("exports").or_else(|| m.get("Exports"));
            match (nm, ex) { (Some(Value::String(n))|Some(Value::Symbol(n)), Some(v)) => (n.clone(), v.clone()), _ => return uneval("RegisterExports", vec![Value::Assoc(m)]) }
        }
        Value::String(s) | Value::Symbol(s) => {
            if args.len()<2 { return uneval("RegisterExports", args); }
            (s, ev.eval(args[1].clone()))
        }
        other => return uneval("RegisterExports", vec![other])
    };
    let mut exports = Vec::new();
    match ex_list {
        Value::List(vs) => { for v in vs { if let Value::String(s)=v { exports.push(s); } else if let Value::Symbol(s)=v { exports.push(s); } } }
        Value::String(s) => exports.push(s),
        Value::Symbol(s) => exports.push(s),
        _ => {}
    }
    let mut m = pkg_exports_assoc(ev);
    m.insert(name, exports);
    write_pkg_exports(ev, m);
    Value::Boolean(true)
}

fn package_exports_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let m = pkg_exports_assoc(ev);
    if args.len()==1 {
        let name = match ev.eval(args[0].clone()) { Value::String(s)|Value::Symbol(s)=>s, _=> return uneval("PackageExports", args) };
        if let Some(vs) = m.get(&name) { return Value::List(vs.iter().cloned().map(Value::String).collect()); }
        return Value::List(vec![]);
    }
    let mut out = std::collections::HashMap::new();
    for (k,vs) in m { out.insert(k, Value::List(vs.into_iter().map(Value::String).collect())); }
    Value::Assoc(out)
}

fn list_installed_packages_fn(ev: &mut Evaluator, _args: Vec<Value>) -> Value {
    let mut out: Vec<Value> = Vec::new();
    let paths = read_package_path(ev);
    for base in paths {
        let dir = Path::new(&base);
        if !dir.is_dir() { continue; }
        if let Ok(read) = fs::read_dir(dir) {
            for ent in read.flatten() {
                if let Ok(ft) = ent.file_type() { if !ft.is_dir() { continue; } }
                let p = ent.path();
                let name = ent.file_name().to_string_lossy().to_string();
                let mut assoc = std::collections::HashMap::new();
                assoc.insert("name".into(), Value::String(name.clone()));
                assoc.insert("path".into(), Value::String(p.to_string_lossy().into()));
                if p.join("lyra.toml").exists() {
                    if let Some(ver) = read_manifest_version(&p.join("lyra.toml")) { assoc.insert("version".into(), Value::String(ver)); }
                }
                out.push(Value::Assoc(assoc));
            }
        }
    }
    Value::List(out)
}

// ---------- Scaffolding ----------

fn new_package_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return uneval("NewPackage", args); }
    let name = match ev.eval(args[0].clone()) { Value::String(s)|Value::Symbol(s)=>s, other=> return uneval("NewPackage", vec![other]) };
    let path = PathBuf::from(&name);
    let (dir_path, pkg_name_opt) = if path.components().count()>1 { (path, None) } else { (PathBuf::from(&name), Some(name.clone())) };
    if let Err(e) = fs::create_dir_all(dir_path.join("src")) { return failure("Package::scaffold", &e.to_string()); }
    // Write minimal manifest
    let manifest_path = dir_path.join("lyra.toml");
    let manifest = if let Some(ref n) = pkg_name_opt { format!("[package]\nname = \"{}\"\nversion = \"0.1.0\"\n", n) } else { "[package]\nversion = \"0.1.0\"\n".into() };
    if let Err(e) = fs::write(&manifest_path, manifest) { return failure("Package::scaffold", &e.to_string()); }
    // Write a starter module file
    let src_main = dir_path.join("src").join("main.lyra");
    let starter = "BeginModule[\"Main\"]\n(* export your API here *)\nEndModule[]\n";
    let _ = fs::write(&src_main, starter);
    Value::Assoc([
        ("path".into(), Value::String(dir_path.to_string_lossy().into())),
        ("name".into(), Value::String(pkg_name_opt.unwrap_or_else(|| "".into()))),
    ].into_iter().collect())
}

fn new_module_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<2 { return uneval("NewModule", args); }
    let pkg_path = match ev.eval(args[0].clone()) { Value::String(s)|Value::Symbol(s)=>PathBuf::from(s), other=> return uneval("NewModule", vec![other, args[1].clone()]) };
    let name = match ev.eval(args[1].clone()) { Value::String(s)|Value::Symbol(s)=>s, other=> return uneval("NewModule", vec![Value::String(pkg_path.to_string_lossy().into()), other]) };
    let dest = pkg_path.join("src").join(format!("{}.lyra", name));
    if let Some(parent) = dest.parent() { if let Err(e) = fs::create_dir_all(parent) { return failure("Package::scaffold", &e.to_string()); } }
    let contents = format!("BeginModule[\"{}\"]\n(* start coding *)\nEndModule[]\n", name);
    if let Err(e) = fs::write(&dest, contents) { return failure("Package::scaffold", &e.to_string()); }
    Value::String(dest.to_string_lossy().into())
}

// ---------- Helpers ----------

fn read_manifest_version(path: &Path) -> Option<String> {
    if let Ok(s) = fs::read_to_string(path) {
        for line in s.lines() {
            let l = line.trim();
            if l.starts_with("version") {
                if let Some(idx) = l.find('=') { return Some(l[idx+1..].trim().trim_matches('"').to_string()); }
            }
        }
    }
    None
}

// ---------- PM stubs ----------

fn pm_na(fun: &str) -> Value { failure("PackageManagerNotAvailable", &format!("{} requires lyra-pm; not yet available", fun)) }

fn build_package_fn(_ev: &mut Evaluator, _args: Vec<Value>) -> Value { pm_na("BuildPackage") }
fn test_package_fn(_ev: &mut Evaluator, _args: Vec<Value>) -> Value { pm_na("TestPackage") }
fn lint_package_fn(_ev: &mut Evaluator, _args: Vec<Value>) -> Value { pm_na("LintPackage") }
fn pack_package_fn(_ev: &mut Evaluator, _args: Vec<Value>) -> Value { pm_na("PackPackage") }
fn generate_sbom_fn(_ev: &mut Evaluator, _args: Vec<Value>) -> Value { pm_na("GenerateSBOM") }
fn sign_package_fn(_ev: &mut Evaluator, _args: Vec<Value>) -> Value { pm_na("SignPackage") }
fn publish_package_fn(_ev: &mut Evaluator, _args: Vec<Value>) -> Value { pm_na("PublishPackage") }
fn install_package_fn(_ev: &mut Evaluator, _args: Vec<Value>) -> Value { pm_na("InstallPackage") }
fn update_package_fn(_ev: &mut Evaluator, _args: Vec<Value>) -> Value { pm_na("UpdatePackage") }
fn remove_package_fn(_ev: &mut Evaluator, _args: Vec<Value>) -> Value { pm_na("RemovePackage") }
fn login_registry_fn(_ev: &mut Evaluator, _args: Vec<Value>) -> Value { pm_na("LoginRegistry") }
fn logout_registry_fn(_ev: &mut Evaluator, _args: Vec<Value>) -> Value { pm_na("LogoutRegistry") }
fn whoami_registry_fn(_ev: &mut Evaluator, _args: Vec<Value>) -> Value { pm_na("WhoAmI") }
fn package_audit_fn(_ev: &mut Evaluator, _args: Vec<Value>) -> Value { pm_na("PackageAudit") }
fn package_verify_fn(_ev: &mut Evaluator, _args: Vec<Value>) -> Value { pm_na("PackageVerify") }
