use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use chrono::{DateTime, Utc};

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

fn failure(tag: &str, msg: &str) -> Value {
    Value::Assoc(HashMap::from([
        ("message".into(), Value::String(msg.to_string())),
        ("tag".into(), Value::String(tag.to_string())),
    ]))
}

#[derive(Clone, Debug)]
struct MountRec {
    _name: String,
    target: String, // canonical scheme URL (e.g., file:///abs, s3://bucket/prefix)
    provider: String,
    read_only: bool,
    options: HashMap<String, Value>,
}

static MOUNTS: OnceLock<Mutex<HashMap<String, MountRec>>> = OnceLock::new();
fn mounts_reg() -> &'static Mutex<HashMap<String, MountRec>> { MOUNTS.get_or_init(|| Mutex::new(HashMap::new())) }

pub fn register_vfs(ev: &mut Evaluator) {
    ev.register("VfsMount", vfs_mount as NativeFn, Attributes::empty());
    ev.register("VfsUnmount", vfs_unmount as NativeFn, Attributes::empty());
    ev.register("VfsMounts", vfs_mounts as NativeFn, Attributes::empty());
    ev.register("VfsResolve", vfs_resolve as NativeFn, Attributes::empty());
    ev.register("VfsRead", vfs_read as NativeFn, Attributes::empty());
    ev.register("VfsStat", vfs_stat as NativeFn, Attributes::empty());
    ev.register("VfsExistsQ", vfs_exists_q as NativeFn, Attributes::empty());
    ev.register("VfsList", vfs_list as NativeFn, Attributes::empty());
    ev.register("VfsWrite", vfs_write as NativeFn, Attributes::empty());
    ev.register("VfsDelete", vfs_delete as NativeFn, Attributes::empty());
    ev.register("VfsCopy", vfs_copy as NativeFn, Attributes::empty());
    ev.register("VfsMove", vfs_move as NativeFn, Attributes::empty());
    ev.register("VfsGlob", vfs_glob as NativeFn, Attributes::empty());
    ev.register("VfsSetConfig", vfs_set_config as NativeFn, Attributes::empty());
    ev.register("VfsGetConfig", vfs_get_config as NativeFn, Attributes::empty());
}

fn to_string(ev: &mut Evaluator, v: Value) -> String {
    match ev.eval(v) { Value::String(s) | Value::Symbol(s) => s, other => lyra_core::pretty::format_value(&other) }
}

fn ensure_scheme(url: &str, ev: &Evaluator) -> String {
    // If it looks like scheme:// return as-is; if starts with vfs:// keep; else treat as file path resolved to absolute file://
    if url.contains("://") { return url.to_string(); }
    let base = if let Some(Value::String(s)) = ev.get_env("CurrentDir") { PathBuf::from(s) } else { std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")) };
    let p = PathBuf::from(url);
    let abs = if p.is_absolute() { p } else { base.join(p) };
    format!("file://{}", abs.to_string_lossy())
}

fn parse_path(ev: &Evaluator, path: &str) -> Result<(String, String, Option<String>), String> {
    // Returns (scheme, canonical_url, mount_name?)
    if let Some(rest) = path.strip_prefix("vfs://") {
        let seg: Vec<&str> = rest.splitn(2, '/').collect();
        let name = seg.get(0).cloned().unwrap_or("");
        if name.is_empty() { return Err("empty mount name".into()); }
        let rel = seg.get(1).cloned().unwrap_or("");
        let reg = mounts_reg().lock().unwrap();
        if let Some(m) = reg.get(name) {
            let mut canon = m.target.trim_end_matches('/').to_string();
            if !rel.is_empty() { canon.push('/'); canon.push_str(rel); }
            let scheme = m.provider.clone();
            return Ok((scheme, canon, Some(name.to_string())));
        } else {
            return Err(format!("unknown mount: {}", name));
        }
    }
    let full = ensure_scheme(path, ev);
    let scheme = full.split("://").next().unwrap_or("").to_string();
    Ok((scheme, full, None))
}

fn capabilities_allow(ev: &mut Evaluator, required: &str) -> bool {
    let caps_v = ev.eval(Value::Expr { head: Box::new(Value::Symbol("ToolsGetCapabilities".into())), args: vec![] });
    match caps_v {
        Value::List(vs) => {
            if vs.is_empty() { return true; }
            let req = required.to_ascii_lowercase();
            vs.iter().any(|v| matches!(v, Value::String(s) | Value::Symbol(s) if s.eq_ignore_ascii_case(&req)))
        }
        _ => true,
    }
}

fn enforce_mount_rw(mount: &Option<String>, write: bool) -> Option<Value> {
    if !write { return None; }
    if let Some(name) = mount.as_ref() {
        let reg = mounts_reg().lock().unwrap();
        if let Some(rec) = reg.get(name) {
            if rec.read_only {
                return Some(failure("VFS::denied", &format!("readOnly mount '{}' denies write", name)));
            }
        }
    }
    None
}

fn vfs_mount(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("VfsMount".into())), args }; }
    let name = to_string(ev, args[0].clone());
    let mut target = to_string(ev, args[1].clone());
    let opts: HashMap<String, Value> = match args.get(2).map(|v| ev.eval(v.clone())) { Some(Value::Assoc(m)) => m, _ => HashMap::new() };
    if !target.contains("://") { // local path → file://
        let base = if let Some(Value::String(s)) = ev.get_env("CurrentDir") { PathBuf::from(s) } else { std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")) };
        let p = PathBuf::from(&target);
        let abs = if p.is_absolute() { p } else { base.join(p) };
        target = format!("file://{}", abs.to_string_lossy());
    }
    let provider = target.split("://").next().unwrap_or("").to_string();
    let read_only = matches!(opts.get("readOnly"), Some(Value::Boolean(true)));
    let rec = MountRec { _name: name.clone(), target: target.clone(), provider, read_only, options: opts };
    mounts_reg().lock().unwrap().insert(name, rec);
    Value::Boolean(true)
}

fn vfs_unmount(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::Expr { head: Box::new(Value::Symbol("VfsUnmount".into())), args }; }
    let name = to_string(ev, args[0].clone());
    let mut reg = mounts_reg().lock().unwrap();
    Value::Boolean(reg.remove(&name).is_some())
}

fn vfs_mounts(_ev: &mut Evaluator, _args: Vec<Value>) -> Value {
    let reg = mounts_reg().lock().unwrap();
    let mut out: HashMap<String, Value> = HashMap::new();
    for (k, m) in reg.iter() {
        out.insert(k.clone(), Value::Assoc(HashMap::from([
            ("target".into(), Value::String(m.target.clone())),
            ("provider".into(), Value::String(m.provider.clone())),
            ("readOnly".into(), Value::Boolean(m.read_only)),
            ("options".into(), Value::Assoc(m.options.clone())),
        ])));
    }
    Value::Assoc(out)
}

fn vfs_resolve(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::Expr { head: Box::new(Value::Symbol("VfsResolve".into())), args }; }
    let p = to_string(ev, args[0].clone());
    match parse_path(ev, &p) {
        Ok((scheme, canon, mnt)) => {
            Value::Assoc(HashMap::from([
                ("scheme".into(), Value::String(scheme)),
                ("canonical".into(), Value::String(canon)),
                ("mount".into(), mnt.map(Value::String).unwrap_or(Value::Symbol("Null".into()))),
                ("isLocal".into(), Value::Boolean(p.starts_with('/') || p.starts_with("file://"))),
            ]))
        }
        Err(e) => failure("VFS::resolve", &e),
    }
}

static VFS_CONFIG: OnceLock<Mutex<HashMap<String, Value>>> = OnceLock::new();
fn cfg_reg() -> &'static Mutex<HashMap<String, Value>> { VFS_CONFIG.get_or_init(|| Mutex::new(HashMap::new())) }

fn vfs_set_config(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("VfsSetConfig".into())), args }; }
    let provider = to_string(ev, args[0].clone()).to_ascii_lowercase();
    let val = match ev.eval(args[1].clone()) { Value::Assoc(m) => Value::Assoc(m), _ => return failure("VFS::config", "cfg must be assoc") };
    cfg_reg().lock().unwrap().insert(provider, val);
    Value::Boolean(true)
}

fn vfs_get_config(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::Expr { head: Box::new(Value::Symbol("VfsGetConfig".into())), args }; }
    let provider = to_string(ev, args[0].clone()).to_ascii_lowercase();
    cfg_reg().lock().unwrap().get(&provider).cloned().unwrap_or(Value::Assoc(HashMap::new()))
}

fn vfs_read(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("VfsRead".into())), args }; }
    let p = to_string(ev, args[0].clone());
    let as_bytes = matches!(args.get(1).and_then(|v| match ev.eval(v.clone()) { Value::Assoc(m) => m.get("as").cloned(), _ => None }), Some(Value::String(s)) if s.to_ascii_lowercase()=="bytes");
    match parse_path(ev, &p) {
        Ok((scheme, canon, _mnt)) => {
            let need = if scheme == "file" { "fs" } else { "net" };
            if !capabilities_allow(ev, need) { return failure("VFS::denied", &format!("requires capability {}", need)); }
            match scheme.as_str() {
                "file" => {
                    let local = canon.trim_start_matches("file://");
                    match std::fs::read(local) {
                        Ok(buf) => { if as_bytes { Value::PackedArray { shape: vec![buf.len()], data: buf.iter().map(|b| *b as f64).collect() } } else { match String::from_utf8(buf) { Ok(s) => Value::String(s), Err(e) => failure("VFS::read", &format!("utf8: {}", e)) } } }
                        Err(e) => failure("VFS::read", &e.to_string()),
                    }
                }
                "http" | "https" => {
                    #[cfg(feature = "net")]
                    {
                        // Request bytes or text; then extract and normalize
                        let opts = if as_bytes {
                            Value::Assoc(HashMap::from([(String::from("As"), Value::String(String::from("Bytes")))]))
                        } else {
                            Value::Assoc(HashMap::from([(String::from("As"), Value::String(String::from("Text")))]))
                        };
                        let resp = ev.eval(Value::Expr { head: Box::new(Value::Symbol("HttpGet".into())), args: vec![Value::String(canon.clone()), opts] });
                        if as_bytes {
                            if let Value::Assoc(m) = resp {
                                if let Some(Value::List(bs)) = m.get("bytes") {
                                    let buf: Vec<f64> = bs.iter().filter_map(|v| if let Value::Integer(n)=v { Some(*n as f64) } else { None }).collect();
                                    return Value::PackedArray { shape: vec![buf.len()], data: buf };
                                }
                            }
                            return failure("VFS::http", "no bytes in response");
                        } else {
                            if let Value::Assoc(m) = resp {
                                if let Some(Value::String(s)) = m.get("body") { return Value::String(s.clone()); }
                            }
                            return failure("VFS::http", "no body in response");
                        }
                    }
                    #[cfg(not(feature = "net"))]
                    { return failure("VFS::notSupported", "http/https requires net feature"); }
                }
                "s3" => {
                    #[cfg(feature = "vfs_s3")]
                    { return s3_read(ev, &canon, as_bytes); }
                    #[cfg(not(feature = "vfs_s3"))]
                    { return failure("VFS::notSupported", "s3 support disabled (enable vfs_s3 feature)"); }
                }
                _ => failure("VFS::notSupported", &format!("read for scheme {}", scheme)),
            }
        }
        Err(e) => failure("VFS::read", &e),
    }
}

fn vfs_stat(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::Expr { head: Box::new(Value::Symbol("VfsStat".into())), args }; }
    let p = to_string(ev, args[0].clone());
    match parse_path(ev, &p) {
        Ok((scheme, canon, _)) => {
            let need = if scheme == "file" { "fs" } else { "net" };
            if !capabilities_allow(ev, need) { return failure("VFS::denied", &format!("requires capability {}", need)); }
            if scheme == "file" {
                let local = canon.trim_start_matches("file://");
                let path = Path::new(local);
                let exists = path.exists();
                let (typ, size, mtime) = if exists {
                    let meta = match std::fs::metadata(path) { Ok(m) => m, Err(_) => return Value::Assoc(HashMap::from([(String::from("exists"), Value::Boolean(false))])) };
                    let t = if meta.is_dir() { "dir" } else { "file" };
                    let sz = if meta.is_file() { Some(meta.len() as i64) } else { None };
                    let mt = meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_millis() as i64);
                    (t, sz, mt)
                } else { ("unknown", None, None) };
                return Value::Assoc(HashMap::from([
                    ("exists".into(), Value::Boolean(exists)),
                    ("type".into(), Value::String(typ.into())),
                    ("size".into(), size.map(Value::Integer).unwrap_or(Value::Symbol("Null".into()))),
                    ("mtime".into(), mtime.map(Value::Integer).unwrap_or(Value::Symbol("Null".into()))),
                ]));
            } else if scheme == "http" || scheme == "https" {
                #[cfg(feature = "net")]
                {
                    let resp = ev.eval(Value::Expr { head: Box::new(Value::Symbol("HttpHead".into())), args: vec![Value::String(canon.clone()), Value::Assoc(HashMap::new())] });
                    if let Value::Assoc(m) = resp {
                        let status = m.get("status").and_then(|v| if let Value::Integer(n)=v { Some(*n as i64) } else { None }).unwrap_or(0);
                        let headers = m.get("headers");
                        let mut size: Option<i64> = None;
                        let mut ctype: Option<String> = None;
                        let mut etag: Option<String> = None;
                        let mut mtime_ms: Option<i64> = None;
                        if let Some(Value::Assoc(hm)) = headers {
                            for (k, v) in hm.iter() {
                                let kl = k.to_ascii_lowercase();
                                if kl == "content-length" { if let Value::String(s) = v { size = s.parse::<i64>().ok(); } }
                                if kl == "content-type" { if let Value::String(s) = v { ctype = Some(s.clone()); } }
                                if kl == "etag" { if let Value::String(s) = v { etag = Some(s.trim_matches('"').to_string()); } }
                                if kl == "last-modified" { if let Value::String(s) = v { if let Ok(dt) = DateTime::parse_from_rfc2822(s) { mtime_ms = Some(dt.with_timezone(&Utc).timestamp_millis()); } } }
                            }
                        }
                        return Value::Assoc(HashMap::from([
                            ("exists".into(), Value::Boolean(status>=200 && status<300)),
                            ("type".into(), Value::String("file".into())),
                            ("size".into(), size.map(Value::Integer).unwrap_or(Value::Symbol("Null".into()))),
                            ("mtime".into(), mtime_ms.map(Value::Integer).unwrap_or(Value::Symbol("Null".into()))),
                            ("contentType".into(), ctype.map(Value::String).unwrap_or(Value::Symbol("Null".into()))),
                            ("etag".into(), etag.map(Value::String).unwrap_or(Value::Symbol("Null".into()))),
                        ]));
                    }
                    return failure("VFS::http", "unexpected head response");
                }
                #[cfg(not(feature = "net"))]
                { return failure("VFS::notSupported", "http/https requires net feature"); }
            } else if scheme == "s3" {
                #[cfg(feature = "vfs_s3")]
                { return s3_stat(ev, &canon); }
                #[cfg(not(feature = "vfs_s3"))]
                { return failure("VFS::notSupported", "s3 support disabled (enable vfs_s3 feature)"); }
            }
            failure("VFS::notSupported", &format!("stat for scheme {}", scheme))
        }
        Err(e) => failure("VFS::stat", &e),
    }
}

fn vfs_exists_q(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::Expr { head: Box::new(Value::Symbol("VfsExistsQ".into())), args }; }
    let p = to_string(ev, args[0].clone());
    match parse_path(ev, &p) { Ok((scheme, canon, _)) => match scheme.as_str() { "file" => Value::Boolean(Path::new(canon.trim_start_matches("file://")).exists()), _ => Value::Boolean(false) }, Err(_) => Value::Boolean(false) }
}

fn list_dir_rec(path: &Path, recursive: bool, out: &mut Vec<String>) {
    if let Ok(rd) = std::fs::read_dir(path) {
        for ent in rd.flatten() {
            let p = ent.path();
            out.push(p.to_string_lossy().to_string());
            if recursive && p.is_dir() { list_dir_rec(&p, recursive, out); }
        }
    }
}

fn vfs_list(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("VfsList".into())), args }; }
    let p = to_string(ev, args[0].clone());
    let (recursive, _glob, dotfiles) = match args.get(1).map(|v| ev.eval(v.clone())) {
        Some(Value::Assoc(m)) => {
            let rec = matches!(m.get("recursive"), Some(Value::Boolean(true)));
            let dots = matches!(m.get("dotfiles"), Some(Value::Boolean(true)));
            (rec, None::<String>, dots)
        }
        _ => (false, None, true),
    };
    match parse_path(ev, &p) {
        Ok((scheme, canon, _)) => match scheme.as_str() {
            "file" => {
                let local = canon.trim_start_matches("file://");
                let mut items: Vec<String> = Vec::new();
                let base = Path::new(local);
                if base.is_dir() { list_dir_rec(base, recursive, &mut items) } else if base.exists() { items.push(base.to_string_lossy().to_string()); }
                if !dotfiles { items.retain(|s| !Path::new(s).file_name().and_then(|x| x.to_str()).unwrap_or("").starts_with('.')); }
                items.sort();
                Value::List(items.into_iter().map(Value::String).collect())
            }
            "s3" => {
                #[cfg(feature = "vfs_s3")]
                { return s3_list(ev, &canon, recursive); }
                #[cfg(not(feature = "vfs_s3"))]
                { failure("VFS::notSupported", "s3 support disabled (enable vfs_s3 feature)") }
            }
            _ => failure("VFS::notSupported", &format!("list for scheme {}", scheme)),
        },
        Err(e) => failure("VFS::list", &e),
    }
}

fn vfs_write(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("VfsWrite".into())), args }; }
    let p = to_string(ev, args[0].clone());
    let data = ev.eval(args[1].clone());
    let (as_bytes, encoding, overwrite) = match args.get(2).map(|v| ev.eval(v.clone())) {
        Some(Value::Assoc(m)) => {
            let asv = m.get("as").and_then(|v| if let Value::String(s)=v { Some(s.to_ascii_lowercase()) } else { None }).unwrap_or_else(|| "text".into());
            let enc = m.get("encoding").and_then(|v| if let Value::String(s)=v { Some(s.clone()) } else { None }).unwrap_or_else(|| "utf-8".into());
            let ow = matches!(m.get("overwrite"), Some(Value::Boolean(true)));
            (asv=="bytes", enc, ow)
        }
        _ => (false, "utf-8".into(), true),
    };
    match parse_path(ev, &p) {
        Ok((scheme, canon, _)) => match scheme.as_str() {
            "file" => {
                let local = canon.trim_start_matches("file://");
                let path = std::path::Path::new(local);
                if path.exists() && !overwrite { return failure("VFS::exists", "file exists and overwrite=false"); }
                if let Some(parent) = path.parent() { let _ = std::fs::create_dir_all(parent); }
                let res = if as_bytes {
                    // Data as bytes: accept PackedArray of u8-ish or String (write utf-8 bytes)
                    match &data {
                        Value::PackedArray { data, .. } => {
                            let buf: Vec<u8> = data.iter().map(|f| (*f as i64).clamp(0,255) as u8).collect();
                            std::fs::write(path, buf)
                        }
                        Value::String(s) => std::fs::write(path, s.as_bytes()),
                        _ => std::fs::write(path, lyra_core::pretty::format_value(&data).as_bytes()),
                    }
                } else {
                    // as text with encoding (utf-8 only for now)
                    let s = match &data { Value::String(s) => s.clone(), _ => lyra_core::pretty::format_value(&data) };
                    if encoding.to_ascii_lowercase() != "utf-8" { return failure("VFS::notSupported", "encodings other than utf-8 not supported yet"); }
                    std::fs::write(path, s.as_bytes())
                };
                match res { Ok(_) => Value::Boolean(true), Err(e) => failure("VFS::write", &e.to_string()) }
            }
            "s3" => {
                #[cfg(feature = "vfs_s3")]
                { return s3_write(ev, &canon, &data, as_bytes); }
                #[cfg(not(feature = "vfs_s3"))]
                { failure("VFS::notSupported", "s3 support disabled (enable vfs_s3 feature)") }
            }
            _ => failure("VFS::notSupported", &format!("write for scheme {}", scheme)),
        },
        Err(e) => failure("VFS::write", &e),
    }
}

fn vfs_delete(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("VfsDelete".into())), args }; }
    let p = to_string(ev, args[0].clone());
    let recursive = match args.get(1).map(|v| ev.eval(v.clone())) { Some(Value::Assoc(m)) => matches!(m.get("recursive"), Some(Value::Boolean(true))), _ => false };
    match parse_path(ev, &p) { Ok((scheme, canon, mnt)) => match scheme.as_str() {
        "file" => {
            if !capabilities_allow(ev, "fs") { return failure("VFS::denied", "requires capability fs"); }
            if let Some(denied) = enforce_mount_rw(&mnt, true) { return denied; }
            let local = canon.trim_start_matches("file://");
            let path = std::path::Path::new(local);
            let res = if recursive { std::fs::remove_dir_all(path) } else { if path.is_dir() { std::fs::remove_dir(path) } else { std::fs::remove_file(path) } };
            match res { Ok(_) => Value::Boolean(true), Err(e) => failure("VFS::delete", &e.to_string()) }
        }
        "s3" => {
            if !capabilities_allow(ev, "net") { return failure("VFS::denied", "requires capability net"); }
            #[cfg(feature = "vfs_s3")]
            { return s3_delete(ev, &canon); }
            #[cfg(not(feature = "vfs_s3"))]
            { failure("VFS::notSupported", "s3 support disabled (enable vfs_s3 feature)") }
        }
        _ => failure("VFS::notSupported", &format!("delete for scheme {}", scheme)),
    }, Err(e) => failure("VFS::delete", &e) }
}

fn copy_rec(src: &Path, dst: &Path) -> std::io::Result<()> {
    if src.is_dir() {
        std::fs::create_dir_all(dst)?;
        for ent in std::fs::read_dir(src)? { let ent = ent?; let sp = ent.path(); let dp = dst.join(ent.file_name()); copy_rec(&sp, &dp)?; }
        Ok(())
    } else {
        if let Some(parent) = dst.parent() { let _ = std::fs::create_dir_all(parent); }
        std::fs::copy(src, dst).map(|_| ())
    }
}

fn vfs_copy(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("VfsCopy".into())), args }; }
    let src = to_string(ev, args[0].clone());
    let dst = to_string(ev, args[1].clone());
    let recursive = match args.get(2).map(|v| ev.eval(v.clone())) { Some(Value::Assoc(m)) => matches!(m.get("recursive"), Some(Value::Boolean(true))), _ => false };
    match (parse_path(ev, &src), parse_path(ev, &dst)) {
        (Ok((ss, sc, _sm)), Ok((ds, dc, dm))) if ss=="file" && ds=="file" => {
            if !capabilities_allow(ev, "fs") { return failure("VFS::denied", "requires capability fs"); }
            if let Some(denied) = enforce_mount_rw(&dm, true) { return denied; }
            let sp = Path::new(sc.trim_start_matches("file://"));
            let dp = Path::new(dc.trim_start_matches("file://"));
            let res = if recursive { copy_rec(sp, dp) } else { if sp.is_dir() { return failure("VFS::copy", "src is directory; use recursive->true"); } else { if let Some(parent)=dp.parent() { let _ = std::fs::create_dir_all(parent); } std::fs::copy(sp, dp).map(|_| ()) } };
            match res { Ok(_) => Value::Boolean(true), Err(e) => failure("VFS::copy", &e.to_string()) }
        }
        (Ok((ss, _sc, _)), Ok((ds, _dc, _))) if ss=="s3" && ds=="s3" => {
            if !capabilities_allow(ev, "net") { return failure("VFS::denied", "requires capability net"); }
            #[cfg(feature = "vfs_s3")]
            { return s3_copy(ev, &sc, &dc); }
            #[cfg(not(feature = "vfs_s3"))]
            { return failure("VFS::notSupported", "s3 support disabled (enable vfs_s3 feature)"); }
        }
        _ => failure("VFS::notSupported", "copy across providers not implemented"),
    }
}

fn vfs_move(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("VfsMove".into())), args }; }
    let src = to_string(ev, args[0].clone());
    let dst = to_string(ev, args[1].clone());
    match (parse_path(ev, &src), parse_path(ev, &dst)) {
        (Ok((ss, sc, sm)), Ok((ds, dc, dm))) if ss=="file" && ds=="file" => {
            if !capabilities_allow(ev, "fs") { return failure("VFS::denied", "requires capability fs"); }
            if let Some(denied) = enforce_mount_rw(&dm, true) { return denied; }
            if let Some(denied) = enforce_mount_rw(&sm, true) { return denied; }
            let sp = Path::new(sc.trim_start_matches("file://"));
            let dp = Path::new(dc.trim_start_matches("file://"));
            if let Some(parent)=dp.parent() { let _ = std::fs::create_dir_all(parent); }
            match std::fs::rename(sp, dp) { Ok(_) => Value::Boolean(true), Err(e) => failure("VFS::move", &e.to_string()) }
        }
        (Ok((ss, _sc, _)), Ok((ds, _dc, _))) if ss=="s3" && ds=="s3" => {
            if !capabilities_allow(ev, "net") { return failure("VFS::denied", "requires capability net"); }
            #[cfg(feature = "vfs_s3")]
            {
                let res = s3_copy(ev, &sc, &dc);
                if let Value::Boolean(true) = res { s3_delete(ev, &sc) } else { res }
            }
            #[cfg(not(feature = "vfs_s3"))]
            { failure("VFS::notSupported", "s3 support disabled (enable vfs_s3 feature)") }
        }
        _ => failure("VFS::notSupported", "move across providers not implemented"),
    }
}

// ---- S3 provider (feature vfs_s3) ----

#[cfg(feature = "vfs_s3")]
fn s3_parse(url: &str) -> Result<(String,String,String), String> {
    // s3://bucket/key
    let rest = url.strip_prefix("s3://").ok_or("invalid s3 url")?;
    let mut it = rest.splitn(2, '/');
    let bucket = it.next().unwrap_or("");
    let key = it.next().unwrap_or("");
    if bucket.is_empty() { return Err("missing bucket".into()); }
    // region from config
    Ok((bucket.to_string(), key.to_string(), String::new()))
}

#[cfg(feature = "vfs_s3")]
fn s3_cfg(ev: &Evaluator, mount_opts: Option<&HashMap<String, Value>>) -> (String, String, String, Option<String>) {
    // region, accessKeyId, secretAccessKey, sessionToken
    let cfg = cfg_reg().lock().unwrap().get("aws").cloned();
    let (mut region, mut ak, mut sk, mut st): (Option<String>, Option<String>, Option<String>, Option<String>) = (None,None,None,None);
    if let Some(Value::Assoc(m)) = cfg.as_ref() {
        region = m.get("region").and_then(|v| if let Value::String(s)=v { Some(s.clone()) } else { None });
        ak = m.get("accessKeyId").and_then(|v| if let Value::String(s)=v { Some(s.clone()) } else { None });
        sk = m.get("secretAccessKey").and_then(|v| if let Value::String(s)=v { Some(s.clone()) } else { None });
        st = m.get("sessionToken").and_then(|v| if let Value::String(s)=v { Some(s.clone()) } else { None });
    }
    if let Some(opts) = mount_opts { if region.is_none() { region = opts.get("region").and_then(|v| if let Value::String(s)=v { Some(s.clone()) } else { None }); } }
    if region.is_none() { region = std::env::var("AWS_REGION").ok().or_else(|| std::env::var("AWS_DEFAULT_REGION").ok()); }
    if ak.is_none() { ak = std::env::var("AWS_ACCESS_KEY_ID").ok(); }
    if sk.is_none() { sk = std::env::var("AWS_SECRET_ACCESS_KEY").ok(); }
    if st.is_none() { st = std::env::var("AWS_SESSION_TOKEN").ok(); }
    (region.unwrap_or_else(|| "us-east-1".into()), ak.unwrap_or_default(), sk.unwrap_or_default(), st)
}

#[cfg(feature = "vfs_s3")]
fn s3_req(
    ev: &Evaluator,
    url: &str,
    method: &str,
    body: Option<&[u8]>,
    extra_headers: Option<&HashMap<String, String>>,
    query_params: Option<&Vec<(String, String)>>,
) -> Result<(u16, HashMap<String,String>, Vec<u8>), String> {
    use sha2::{Digest, Sha256};
    use hmac::{Hmac, Mac};
    use chrono::Utc;
    let (bucket, key, _r) = s3_parse(url)?;
    let reg = mounts_reg().lock().unwrap();
    // attempt to find mount by prefix match to get mount options
    let mopts: Option<HashMap<String, Value>> = reg.values().find_map(|m| if m.target.starts_with("s3://") { Some(m.options.clone()) } else { None });
    let (region, access, secret, token) = s3_cfg(ev, mopts.as_ref());
    if access.is_empty() || secret.is_empty() { return Err("missing AWS credentials".into()); }
    let host = format!("{}.s3.{}.amazonaws.com", bucket, region);
    let path = format!("/{}", percent_encoding::utf8_percent_encode(&key, percent_encoding::NON_ALPHANUMERIC));
    // Canonical query string
    let mut qps: Vec<(String, String)> = query_params.cloned().unwrap_or_default();
    qps.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let canonical_query = if qps.is_empty() { String::new() } else {
        qps.iter()
            .map(|(k, v)| {
                format!(
                    "{}={}",
                    percent_encoding::utf8_percent_encode(k, percent_encoding::NON_ALPHANUMERIC),
                    percent_encoding::utf8_percent_encode(v, percent_encoding::NON_ALPHANUMERIC)
                )
            })
            .collect::<Vec<_>>()
            .join("&")
    };
    let payload = body.unwrap_or(&[]);
    let mut hasher = Sha256::new(); hasher.update(payload); let payload_hash = format!("{:x}", hasher.finalize());
    let now = Utc::now();
    let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
    let date_stamp = now.format("%Y%m%d").to_string();
    // Collect headers (lowercased names) and build canonical headers/signed headers
    let mut hdrs: Vec<(String, String)> = Vec::new();
    hdrs.push(("host".into(), host.clone()));
    hdrs.push(("x-amz-content-sha256".into(), payload_hash.clone()));
    hdrs.push(("x-amz-date".into(), amz_date.clone()));
    if let Some(t) = token.as_ref() { hdrs.push(("x-amz-security-token".into(), t.clone())); }
    if let Some(ex) = extra_headers {
        for (k, v) in ex.iter() { hdrs.push((k.to_ascii_lowercase(), v.clone())); }
    }
    hdrs.sort_by(|a, b| a.0.cmp(&b.0));
    let mut canonical_headers = String::new();
    let mut signed_vec: Vec<String> = Vec::new();
    for (k, v) in hdrs.iter() {
        signed_vec.push(k.clone());
        canonical_headers.push_str(&format!("{}:{}\n", k, v.trim()));
    }
    let signed_headers = signed_vec.join(";");
    let canonical_request = format!("{}\n{}\n{}\n{}\n{}\n{}", method, path, canonical_query, canonical_headers, signed_headers, payload_hash);
    let mut hcr = Sha256::new(); hcr.update(canonical_request.as_bytes()); let cr_hash = format!("{:x}", hcr.finalize());
    let scope = format!("{}/{}/s3/aws4_request", date_stamp, region);
    let string_to_sign = format!("AWS4-HMAC-SHA256\n{}\n{}\n{}", amz_date, scope, cr_hash);
    type HmacSha256 = Hmac<Sha256>;
    let k_date = HmacSha256::new_from_slice(format!("AWS4{}", secret).as_bytes()).unwrap().chain_update(date_stamp.as_bytes()).finalize().into_bytes();
    let k_region = HmacSha256::new_from_slice(&k_date).unwrap().chain_update(region.as_bytes()).finalize().into_bytes();
    let k_service = HmacSha256::new_from_slice(&k_region).unwrap().chain_update(b"s3").finalize().into_bytes();
    let k_signing = HmacSha256::new_from_slice(&k_service).unwrap().chain_update(b"aws4_request").finalize().into_bytes();
    let signature = HmacSha256::new_from_slice(&k_signing).unwrap().chain_update(string_to_sign.as_bytes()).finalize().into_bytes();
    let sig_hex = signature.iter().map(|b| format!("{:02x}", b)).collect::<String>();
    let auth = format!("AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}", access, scope, signed_headers, sig_hex);
    #[cfg(feature = "net_https")]
    {
        let client = reqwest::blocking::Client::new();
        let url_full = if canonical_query.is_empty() { format!("https://{}{}", host, path) } else { format!("https://{}{}?{}", host, path, canonical_query) };
        let mut req = client.request(reqwest::Method::from_bytes(method.as_bytes()).unwrap(), &url_full)
            .header("x-amz-date", amz_date)
            .header("x-amz-content-sha256", payload_hash.clone())
            .header("authorization", auth)
            .header("host", host.clone());
        if let Some(t) = token.as_ref() { req = req.header("x-amz-security-token", t); }
        if let Some(ex) = extra_headers { for (k, v) in ex.iter() { req = req.header(k, v); } }
        if method == "PUT" || method == "POST" { req = req.body(payload.to_vec()); }
        if let Some(q) = query_params { for (k, v) in q.iter() { let _ = (k, v); /* already encoded in url_full */ } }
        let resp = req.send().map_err(|e| e.to_string())?;
        let status = resp.status().as_u16();
        let mut headers: HashMap<String, String> = HashMap::new();
        for (k, v) in resp.headers().iter() { headers.insert(k.to_string(), v.to_str().unwrap_or("").into()); }
        let bytes = resp.bytes().map_err(|e| e.to_string())?.to_vec();
        return Ok((status, headers, bytes));
    }
    #[cfg(not(feature = "net_https"))]
    { Err("https client disabled (enable net_https)".into()) }
}

#[cfg(feature = "vfs_s3")]
fn s3_read(ev: &mut Evaluator, url: &str, as_bytes: bool) -> Value {
    match s3_req(ev, url, "GET", None, None, None) {
        Ok((_st,_h,body)) => {
            if as_bytes {
                let buf: Vec<f64> = body.iter().map(|b| *b as f64).collect();
                Value::PackedArray { shape: vec![buf.len()], data: buf }
            } else {
                Value::String(String::from_utf8_lossy(&body).to_string())
            }
        }
        Err(e) => failure("VFS::s3", &e)
    }
}
#[cfg(feature = "vfs_s3")]
fn s3_stat(ev: &mut Evaluator, url: &str) -> Value {
    match s3_req(ev, url, "HEAD", None, None, None) { Ok((st,h,_)) => {
        let exists = st==200;
        let size = h.get("content-length").and_then(|s| s.parse::<i64>().ok());
        let etag = h.get("etag").cloned().map(|s| s.trim_matches('"').to_string());
        let ctype = h.get("content-type").cloned();
        let mtime_ms = h.get("last-modified").and_then(|s| DateTime::parse_from_rfc2822(s).ok()).map(|dt| dt.with_timezone(&Utc).timestamp_millis());
        Value::Assoc(HashMap::from([
            ("exists".into(), Value::Boolean(exists)),
            ("type".into(), Value::String("file".into())),
            ("size".into(), size.map(Value::Integer).unwrap_or(Value::Symbol("Null".into()))),
            ("mtime".into(), mtime_ms.map(Value::Integer).unwrap_or(Value::Symbol("Null".into()))),
            ("contentType".into(), ctype.map(Value::String).unwrap_or(Value::Symbol("Null".into()))),
            ("etag".into(), etag.map(Value::String).unwrap_or(Value::Symbol("Null".into()))),
        ]))
    } Err(e) => failure("VFS::s3", &e) }
}
#[cfg(feature = "vfs_s3")]
fn s3_write(ev: &mut Evaluator, url: &str, data: &Value, as_bytes: bool) -> Value {
    let buf: Vec<u8> = if as_bytes { match data { Value::PackedArray { data, .. } => data.iter().map(|f| (*f as i64).clamp(0,255) as u8).collect(), Value::String(s)=>s.as_bytes().to_vec(), _ => lyra_core::pretty::format_value(data).into_bytes() } } else { match data { Value::String(s)=>s.as_bytes().to_vec(), _ => lyra_core::pretty::format_value(data).into_bytes() } };
    match s3_req(ev, url, "PUT", Some(&buf), None, None) { Ok((_st,_h,_)) => Value::Boolean(true), Err(e) => failure("VFS::s3", &e) }
}
#[cfg(feature = "vfs_s3")]
fn s3_delete(ev: &mut Evaluator, url: &str) -> Value { match s3_req(ev, url, "DELETE", None, None, None) { Ok((_st,_h,_)) => Value::Boolean(true), Err(e) => failure("VFS::s3", &e) } }
#[cfg(feature = "vfs_s3")]
fn s3_copy(ev: &mut Evaluator, src: &str, dst: &str) -> Value {
    // Prefer CopyObject: PUT to dst with x-amz-copy-source header
    let (sb, sk, _r) = match s3_parse(src) { Ok(t) => t, Err(e) => return failure("VFS::s3", &e) };
    let source_hdr = format!("/{}{}{}", sb, if sk.is_empty() { "" } else { "/" }, percent_encoding::utf8_percent_encode(&sk, percent_encoding::NON_ALPHANUMERIC));
    let mut ex: HashMap<String, String> = HashMap::new();
    ex.insert("x-amz-copy-source".into(), source_hdr);
    match s3_req(ev, dst, "PUT", None, Some(&ex), None) {
        Ok((st,_h,_)) if st==200 || st==201 => Value::Boolean(true),
        Ok((_st,_h,_)) | Err(_) => {
            // Fallback: GET+PUT
            match s3_req(ev, src, "GET", None, None, None) { Ok((_s,_h,body)) => match s3_req(ev, dst, "PUT", Some(&body), None, None) { Ok((_s2,_h2,_)) => Value::Boolean(true), Err(e)=>failure("VFS::s3", &e) }, Err(e)=>failure("VFS::s3", &e) }
        }
    }
}

#[cfg(feature = "vfs_s3")]
fn s3_list(ev: &mut Evaluator, url: &str, recursive: bool) -> Value {
    let (bucket, prefix, _r) = match s3_parse(url) { Ok(t) => t, Err(e) => return failure("VFS::s3", &e) };
    let mut cont: Option<String> = None;
    let mut items: Vec<Value> = Vec::new();
    loop {
        let mut params: Vec<(String, String)> = vec![("list-type".into(), "2".into())];
        if !prefix.is_empty() { params.push(("prefix".into(), prefix.clone())); }
        if !recursive { params.push(("delimiter".into(), "/".into())); }
        if let Some(t) = cont.as_ref() { params.push(("continuation-token".into(), t.clone())); }
        let url_bucket_root = format!("s3://{}/", bucket);
        match s3_req(ev, &url_bucket_root, "GET", None, None, Some(&params)) {
            Ok((_st,_h,body)) => {
                let text = String::from_utf8_lossy(&body);
                // Extract object keys
                if let Ok(re) = regex::Regex::new(r"<Key>([^<]+)</Key>") {
                    for cap in re.captures_iter(&text) {
                        if let Some(m) = cap.get(1) {
                            let key = m.as_str();
                            items.push(Value::String(format!("s3://{}/{}", bucket, key)));
                        }
                    }
                }
                // Non-recursive: include CommonPrefixes as directory entries
                if !recursive {
                    if let Ok(re) = regex::Regex::new(r"<CommonPrefixes>\s*<Prefix>([^<]+)/</Prefix>\s*</CommonPrefixes>") {
                        for cap in re.captures_iter(&text) {
                            if let Some(m) = cap.get(1) {
                                let pfx = m.as_str();
                                items.push(Value::String(format!("s3://{}/{}", bucket, pfx)));
                            }
                        }
                    }
                }
                // Continuation
                let mut has_more = false;
                if let Ok(re_trunc) = regex::Regex::new(r"<IsTruncated>true</IsTruncated>") { has_more = re_trunc.is_match(&text); }
                if has_more {
                    if let Ok(re_next) = regex::Regex::new(r"<NextContinuationToken>([^<]+)</NextContinuationToken>") {
                        if let Some(capn) = re_next.captures(&text) { cont = capn.get(1).map(|m| m.as_str().to_string()); } else { cont = None; }
                    } else { cont = None; }
                    if cont.is_none() { break; }
                } else { break; }
            }
            Err(e) => return failure("VFS::s3", &e),
        }
    }
    Value::List(items)
}

fn simple_glob_match(p: &str, pat: &str) -> bool {
    if pat == "**" { return true; }
    if !pat.contains('*') && !pat.contains('?') { return p == pat; }
    // very naive: convert * -> .*, ? -> . and anchor
    let mut re = String::from("^");
    for ch in pat.chars() {
        match ch {
            '*' => re.push_str(".*"),
            '?' => re.push('.'),
            '.' | '+' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '\\' | '^' | '$' => { re.push('\\'); re.push(ch); },
            _ => re.push(ch),
        }
    }
    re.push('$');
    match regex::Regex::new(&re) { Ok(rx) => rx.is_match(p), Err(_) => p.contains(pat.trim_matches('*')) }
}

fn vfs_glob(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("VfsGlob".into())), args }; }
    let base = to_string(ev, args[0].clone());
    let pattern = to_string(ev, args[1].clone());
    let mut opts: HashMap<String, Value> = match args.get(2).map(|v| ev.eval(v.clone())) { Some(Value::Assoc(m)) => m, _ => HashMap::new() };
    // Ensure recursive list then filter by pattern over relative path
    opts.insert("recursive".into(), Value::Boolean(true));
    let listed = vfs_list(ev, vec![Value::String(base.clone()), Value::Assoc(opts)]);
    match listed {
        Value::List(items) => {
            // compute root path for relative matching
            let root = match parse_path(ev, &base) { Ok((_s, canon, _)) => canon, Err(_) => base.clone() };
            let root_local = root.trim_start_matches("file://").to_string();
            let mut out: Vec<Value> = Vec::new();
            for v in items {
                if let Value::String(p) = v {
                    let rel = if p.starts_with(&root_local) { p[root_local.len()..].trim_start_matches(std::path::MAIN_SEPARATOR).replace(std::path::MAIN_SEPARATOR, "/") } else { p.clone() };
                    if simple_glob_match(&rel, &pattern) { out.push(Value::String(p)); }
                }
            }
            Value::List(out)
        }
        other => other,
    }
}
