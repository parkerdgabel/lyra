use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

fn failure(tag: &str, msg: &str) -> Value {
    Value::Assoc(
        vec![
            ("message".to_string(), Value::String(msg.to_string())),
            ("tag".to_string(), Value::String(tag.to_string())),
        ]
        .into_iter()
        .collect(),
    )
}

pub fn register_fs(ev: &mut Evaluator) {
    ev.register("MakeDirectory", mkdir as NativeFn, Attributes::empty());
    ev.register("Remove", remove as NativeFn, Attributes::empty());
    ev.register("Copy", copy as NativeFn, Attributes::empty());
    ev.register("Move", move_fn as NativeFn, Attributes::empty());
    ev.register("Touch", touch as NativeFn, Attributes::empty());
    ev.register("Symlink", symlink as NativeFn, Attributes::empty());
    ev.register("Glob", glob as NativeFn, Attributes::empty());
    ev.register("ReadBytes", read_bytes as NativeFn, Attributes::empty());
    ev.register("WriteBytes", write_bytes as NativeFn, Attributes::empty());
    ev.register("TempFile", temp_file as NativeFn, Attributes::empty());
    ev.register("TempDir", temp_dir as NativeFn, Attributes::empty());
    ev.register("WatchDirectory", watch_directory as NativeFn, Attributes::HOLD_ALL);
    ev.register("CancelWatch", cancel_watch as NativeFn, Attributes::empty());
    #[cfg(feature = "fs_archive")]
    {
        ev.register("ZipCreate", zip_create as NativeFn, Attributes::empty());
        ev.register("ZipExtract", zip_extract as NativeFn, Attributes::empty());
        ev.register("TarCreate", tar_create as NativeFn, Attributes::empty());
        ev.register("TarExtract", tar_extract as NativeFn, Attributes::empty());
        ev.register("Gzip", gzip_fn as NativeFn, Attributes::empty());
        ev.register("Gunzip", gunzip_fn as NativeFn, Attributes::empty());
    }
}

pub fn register_fs_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    crate::register_if(ev, pred, "MakeDirectory", mkdir as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "Remove", remove as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "Copy", copy as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "Move", move_fn as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "Touch", touch as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "Symlink", symlink as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "Glob", glob as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "ReadBytes", read_bytes as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "WriteBytes", write_bytes as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "TempFile", temp_file as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "TempDir", temp_dir as NativeFn, Attributes::empty());
    crate::register_if(
        ev,
        pred,
        "WatchDirectory",
        watch_directory as NativeFn,
        Attributes::HOLD_ALL,
    );
    crate::register_if(ev, pred, "CancelWatch", cancel_watch as NativeFn, Attributes::empty());
    #[cfg(feature = "fs_archive")]
    {
        crate::register_if(ev, pred, "ZipCreate", zip_create as NativeFn, Attributes::empty());
        crate::register_if(ev, pred, "ZipExtract", zip_extract as NativeFn, Attributes::empty());
        crate::register_if(ev, pred, "TarCreate", tar_create as NativeFn, Attributes::empty());
        crate::register_if(ev, pred, "TarExtract", tar_extract as NativeFn, Attributes::empty());
        crate::register_if(ev, pred, "Gzip", gzip_fn as NativeFn, Attributes::empty());
        crate::register_if(ev, pred, "Gunzip", gunzip_fn as NativeFn, Attributes::empty());
    }
}

fn to_string(ev: &mut Evaluator, v: Value) -> String {
    match ev.eval(v) {
        Value::String(s) | Value::Symbol(s) => s,
        other => lyra_core::pretty::format_value(&other),
    }
}

fn mkdir(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("MakeDirectory".into())), args };
    }
    let path = to_string(ev, args[0].clone());
    let parents = match args.get(1).and_then(|v| {
        if let Value::Assoc(m) = ev.eval(v.clone()) {
            m.get("Parents").cloned()
        } else {
            None
        }
    }) {
        Some(Value::Boolean(true)) => true,
        _ => false,
    };
    let res = if parents { fs::create_dir_all(&path) } else { fs::create_dir(&path) };
    match res {
        Ok(_) => Value::Boolean(true),
        Err(e) => failure("FS::mkdir", &e.to_string()),
    }
}

fn remove(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("Remove".into())), args };
    }
    let path = to_string(ev, args[0].clone());
    let recursive = match args.get(1).and_then(|v| {
        if let Value::Assoc(m) = ev.eval(v.clone()) {
            m.get("Recursive").cloned()
        } else {
            None
        }
    }) {
        Some(Value::Boolean(true)) => true,
        _ => false,
    };
    let p = Path::new(&path);
    let res = if recursive {
        fs::remove_dir_all(p)
    } else {
        if p.is_dir() {
            fs::remove_dir(p)
        } else {
            fs::remove_file(p)
        }
    };
    match res {
        Ok(_) => Value::Boolean(true),
        Err(e) => failure("FS::rm", &e.to_string()),
    }
}

fn copy(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("Copy".into())), args };
    }
    let src = to_string(ev, args[0].clone());
    let dst = to_string(ev, args[1].clone());
    let recursive = match args.get(2).and_then(|v| {
        if let Value::Assoc(m) = ev.eval(v.clone()) {
            m.get("Recursive").cloned()
        } else {
            None
        }
    }) {
        Some(Value::Boolean(true)) => true,
        _ => false,
    };
    if recursive {
        return copy_dir_recursive(Path::new(&src), Path::new(&dst));
    }
    match fs::copy(&src, &dst) {
        Ok(_) => Value::Boolean(true),
        Err(e) => failure("FS::cp", &e.to_string()),
    }
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> Value {
    let md = fs::metadata(src);
    if md.is_err() {
        return failure("FS::cp", "Source not found");
    }
    if md.unwrap().is_dir() {
        if let Err(e) = fs::create_dir_all(dst) {
            return failure("FS::cp", &e.to_string());
        }
        match fs::read_dir(src) {
            Ok(rd) => {
                for entry in rd {
                    if let Ok(ent) = entry {
                        let p = ent.path();
                        let dp = dst.join(ent.file_name());
                        let res = if p.is_dir() {
                            copy_dir_recursive(&p, &dp)
                        } else {
                            match fs::copy(&p, &dp) {
                                Ok(_) => Value::Boolean(true),
                                Err(e) => failure("FS::cp", &e.to_string()),
                            }
                        };
                        if !matches!(res, Value::Boolean(true)) {
                            return res;
                        }
                    }
                }
            }
            Err(e) => {
                return failure("FS::cp", &e.to_string());
            }
        }
        Value::Boolean(true)
    } else {
        match fs::copy(src, dst) {
            Ok(_) => Value::Boolean(true),
            Err(e) => failure("FS::cp", &e.to_string()),
        }
    }
}

fn move_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("Move".into())), args };
    }
    let src = to_string(ev, args[0].clone());
    let dst = to_string(ev, args[1].clone());
    match fs::rename(&src, &dst) {
        Ok(_) => Value::Boolean(true),
        Err(_) => {
            // fallback copy+remove
            let res = copy(ev, vec![Value::String(src.clone()), Value::String(dst.clone())]);
            if matches!(res, Value::Boolean(true)) {
                let _ = remove(ev, vec![Value::String(src)]);
                return Value::Boolean(true);
            }
            res
        }
    }
}

fn touch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("Touch".into())), args };
    }
    let path = to_string(ev, args[0].clone());
    match fs::OpenOptions::new().create(true).append(true).open(&path) {
        Ok(_) => Value::Boolean(true),
        Err(e) => failure("FS::touch", &e.to_string()),
    }
}

#[cfg(unix)]
fn symlink_impl(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(src, dst)
}
#[cfg(windows)]
fn symlink_impl(src: &Path, dst: &Path) -> std::io::Result<()> {
    if src.is_dir() {
        std::os::windows::fs::symlink_dir(src, dst)
    } else {
        std::os::windows::fs::symlink_file(src, dst)
    }
}

fn symlink(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("Symlink".into())), args };
    }
    let target = to_string(ev, args[0].clone());
    let linkp = to_string(ev, args[1].clone());
    match symlink_impl(Path::new(&target), Path::new(&linkp)) {
        Ok(_) => Value::Boolean(true),
        Err(e) => failure("FS::symlink", &e.to_string()),
    }
}

fn read_bytes(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("ReadBytes".into())), args };
    }
    let path = to_string(ev, args[0].clone());
    match fs::read(&path) {
        Ok(b) => Value::String(String::from_utf8_lossy(&b).to_string()),
        Err(e) => failure("FS::read", &e.to_string()),
    }
}

fn write_bytes(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("WriteBytes".into())), args };
    }
    let path = to_string(ev, args[0].clone());
    let data = match ev.eval(args[1].clone()) {
        Value::String(s) => s.into_bytes(),
        _ => Vec::new(),
    };
    match fs::write(&path, data) {
        Ok(_) => Value::Boolean(true),
        Err(e) => failure("FS::write", &e.to_string()),
    }
}

fn temp_file(_ev: &mut Evaluator, _args: Vec<Value>) -> Value {
    let mut path = std::env::temp_dir();
    let fname = format!("lyra-{}-{}.tmp", std::process::id(), crate_ts_ms());
    path.push(fname);
    match fs::OpenOptions::new().create_new(true).write(true).open(&path) {
        Ok(_) => Value::String(path.to_string_lossy().to_string()),
        Err(e) => failure("FS::temp", &e.to_string()),
    }
}

fn temp_dir(_ev: &mut Evaluator, _args: Vec<Value>) -> Value {
    let mut path = std::env::temp_dir();
    let dname = format!("lyra-{}-{}", std::process::id(), crate_ts_ms());
    path.push(dname);
    match fs::create_dir(&path) {
        Ok(_) => Value::String(path.to_string_lossy().to_string()),
        Err(e) => failure("FS::temp", &e.to_string()),
    }
}

fn crate_ts_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn matches_pattern(name: &str, pat: &str) -> bool {
    // Simple glob: * and ? only, no path separators logic
    fn helper(s: &[u8], p: &[u8]) -> bool {
        if p.is_empty() {
            return s.is_empty();
        }
        match p[0] {
            b'*' => {
                for i in 0..=s.len() {
                    if helper(&s[i..], &p[1..]) {
                        return true;
                    }
                }
                false
            }
            b'?' => {
                if s.is_empty() {
                    false
                } else {
                    helper(&s[1..], &p[1..])
                }
            }
            c => {
                if !s.is_empty() && s[0] == c {
                    helper(&s[1..], &p[1..])
                } else {
                    false
                }
            }
        }
    }
    helper(name.as_bytes(), pat.as_bytes())
}

fn path_glob_match(path: &str, pattern: &str) -> bool {
    let sep = std::path::MAIN_SEPARATOR;
    let pat = pattern.replace('\\', "/");
    let path = path.replace(sep, "/");
    let pcomps: Vec<&str> = pat.split('/').collect();
    let scomps: Vec<&str> = path.split('/').collect();
    fn comp_match(name: &str, pat: &str) -> bool {
        matches_pattern(name, pat)
    }
    fn rec(pi: usize, si: usize, p: &[&str], s: &[&str]) -> bool {
        if pi == p.len() {
            return si == s.len();
        }
        if p[pi] == "**" {
            for k in si..=s.len() {
                if rec(pi + 1, k, p, s) {
                    return true;
                }
            }
            false
        } else {
            if si < s.len() && comp_match(s[si], p[pi]) {
                rec(pi + 1, si + 1, p, s)
            } else {
                false
            }
        }
    }
    rec(0, 0, &pcomps, &scomps)
}

fn glob(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("Glob".into())), args };
    }
    let patterns: Vec<String> = match ev.eval(args[0].clone()) {
        Value::List(xs) => xs
            .into_iter()
            .filter_map(|v| match v {
                Value::String(s) | Value::Symbol(s) => Some(s),
                _ => None,
            })
            .collect(),
        Value::String(s) | Value::Symbol(s) => vec![s],
        other => {
            return Value::Expr { head: Box::new(Value::Symbol("Glob".into())), args: vec![other] }
        }
    };
    let opts = if args.len() > 1 {
        if let Value::Assoc(m) = ev.eval(args[1].clone()) {
            m
        } else {
            HashMap::new()
        }
    } else {
        HashMap::new()
    };
    let cwd = opts
        .get("Cwd")
        .and_then(|v| {
            if let Value::String(s) | Value::Symbol(s) = v {
                Some(PathBuf::from(s))
            } else {
                None
            }
        })
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let recursive = matches!(opts.get("Recursive"), Some(Value::Boolean(true)));
    let dotfiles = matches!(opts.get("Dotfiles"), Some(Value::Boolean(true)));
    let includes: Vec<String> = opts
        .get("Include")
        .and_then(|v| {
            if let Value::List(xs) = v {
                Some(
                    xs.iter()
                        .filter_map(|v| {
                            if let Value::String(s) | Value::Symbol(s) = v {
                                Some(s.clone())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            }
        })
        .unwrap_or_default();
    let excludes: Vec<String> = opts
        .get("Exclude")
        .and_then(|v| {
            if let Value::List(xs) = v {
                Some(
                    xs.iter()
                        .filter_map(|v| {
                            if let Value::String(s) | Value::Symbol(s) = v {
                                Some(s.clone())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            }
        })
        .unwrap_or_default();
    let mut results: Vec<String> = Vec::new();
    fn walk_dir(base: &Path, recursive: bool, f: &mut dyn FnMut(&Path)) {
        if base.is_file() {
            f(base);
            return;
        }
        if let Ok(rd) = fs::read_dir(base) {
            for ent in rd {
                if let Ok(ent) = ent {
                    let p = ent.path();
                    if p.is_dir() {
                        if recursive {
                            walk_dir(&p, true, f);
                        }
                    } else {
                        f(&p);
                    }
                }
            }
        }
    }
    let mut visit = |p: &Path| {
        let rel = p.strip_prefix(&cwd).unwrap_or(p).to_string_lossy().to_string();
        let rel_norm = rel.replace(std::path::MAIN_SEPARATOR, "/");
        let filename = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if !dotfiles && filename.starts_with('.') {
            return;
        }
        let mut matched_any = false;
        for pat in &patterns {
            if path_glob_match(&rel_norm, &pat.replace('\\', "/")) {
                matched_any = true;
                break;
            }
        }
        if matched_any {
            let mut included = true;
            if !includes.is_empty() {
                included =
                    includes.iter().any(|inc| path_glob_match(&rel_norm, &inc.replace('\\', "/")));
            }
            if included && !excludes.is_empty() {
                if excludes.iter().any(|ex| path_glob_match(&rel_norm, &ex.replace('\\', "/"))) {
                    included = false;
                }
            }
            if included {
                results.push(p.to_string_lossy().to_string());
            }
        }
    };
    walk_dir(&cwd, recursive, &mut visit);
    results.sort();
    results.dedup();
    Value::List(results.into_iter().map(Value::String).collect())
}

#[derive(Default)]
struct WatchState {
    #[allow(dead_code)]
    id: i64,
    #[allow(dead_code)]
    path: String,
    active: bool,
}
static WATCH_REG: OnceLock<Mutex<HashMap<i64, WatchState>>> = OnceLock::new();
static NEXT_WATCH_ID: OnceLock<std::sync::atomic::AtomicI64> = OnceLock::new();
fn wreg() -> &'static Mutex<HashMap<i64, WatchState>> {
    WATCH_REG.get_or_init(|| Mutex::new(HashMap::new()))
}
fn next_wid() -> i64 {
    let a = NEXT_WATCH_ID.get_or_init(|| std::sync::atomic::AtomicI64::new(1));
    a.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}
fn whandle(id: i64) -> Value {
    Value::Assoc(HashMap::from([
        ("__type".into(), Value::String("FSWatch".into())),
        ("id".into(), Value::Integer(id)),
    ]))
}
fn w_get_id(v: &Value) -> Option<i64> {
    if let Value::Assoc(m) = v {
        if m.get("__type") == Some(&Value::String("FSWatch".into())) {
            if let Some(Value::Integer(id)) = m.get("id") {
                return Some(*id);
            }
        }
    }
    None
}

fn watch_directory(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("WatchDirectory".into())), args };
    }
    let path = match ev.eval(args[0].clone()) {
        Value::String(s) | Value::Symbol(s) => s,
        other => {
            return Value::Expr {
                head: Box::new(Value::Symbol("WatchDirectory".into())),
                args: vec![other, args[1].clone()],
            }
        }
    };
    let handler = args[1].clone();
    let (recursive, debounce_ms) = match args.get(2).map(|v| ev.eval(v.clone())) {
        Some(Value::Assoc(m)) => {
            let rec = matches!(m.get("Recursive"), Some(Value::Boolean(true)));
            let db = match m.get("DebounceMs") {
                Some(Value::Integer(n)) if *n > 0 => *n as u64,
                _ => 50,
            };
            (rec, db)
        }
        _ => (true, 50),
    };
    #[cfg(feature = "fs_watch")]
    {
        use notify::{
            event::{CreateKind, EventKind, ModifyKind, RemoveKind},
            Config, RecursiveMode, Watcher,
        };
        let id = next_wid();
        wreg().lock().unwrap().insert(id, WatchState { id, path: path.clone(), active: true });
        let rec_mode =
            if recursive { RecursiveMode::Recursive } else { RecursiveMode::NonRecursive };
        // We need a 'static handler; clone needed values
        let path_clone = path.clone();
        let mut watcher =
            match notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
                if let Ok(event) = res {
                    let ev_kind = match event.kind {
                        EventKind::Create(CreateKind::Any)
                        | EventKind::Create(CreateKind::File)
                        | EventKind::Create(CreateKind::Folder) => "create",
                        EventKind::Modify(ModifyKind::Any)
                        | EventKind::Modify(ModifyKind::Data(_))
                        | EventKind::Modify(ModifyKind::Metadata(_)) => "modify",
                        EventKind::Remove(RemoveKind::Any)
                        | EventKind::Remove(RemoveKind::File)
                        | EventKind::Remove(RemoveKind::Folder) => "delete",
                        EventKind::Modify(ModifyKind::Name(_)) => "rename",
                        _ => "modify",
                    };
                    let first_path = event
                        .paths
                        .first()
                        .map(|p| p.to_string_lossy().to_string())
                        .unwrap_or(path_clone.clone());
                    let payload = Value::Assoc(HashMap::from([
                        ("Event".into(), Value::String(ev_kind.to_string())),
                        ("Path".into(), Value::String(first_path)),
                    ]));
                    let mut ev2 = Evaluator::new();
                    crate::register_all(&mut ev2);
                    let _ = ev2
                        .eval(Value::Expr { head: Box::new(handler.clone()), args: vec![payload] });
                }
            }) {
                Ok(w) => w,
                Err(e) => return failure("FS::watch", &format!("watcher: {}", e)),
            };
        if let Err(e) = watcher.configure(
            Config::default().with_poll_interval(std::time::Duration::from_millis(debounce_ms)),
        ) {
            return failure("FS::watch", &format!("config: {}", e));
        }
        if let Err(e) = watcher.watch(Path::new(&path), rec_mode) {
            return failure("FS::watch", &format!("watch: {}", e));
        }
        // Keep watcher alive by leaking it into a background thread that parks
        std::thread::spawn(move || {
            let _w = watcher;
            loop {
                std::thread::sleep(std::time::Duration::from_secs(3600));
            }
        });
        return whandle(id);
    }
    #[cfg(not(feature = "fs_watch"))]
    {
        let _ = (path, handler, recursive, debounce_ms);
        failure("FS::watch", "Watch feature disabled")
    }
}

fn cancel_watch(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("CancelWatch".into())), args };
    }
    if let Some(id) = w_get_id(&args[0]) {
        let mut r = wreg().lock().unwrap();
        if let Some(st) = r.get_mut(&id) {
            st.active = false;
            r.remove(&id);
            return Value::Boolean(true);
        }
    }
    Value::Boolean(false)
}

#[cfg(feature = "fs_archive")]
fn zip_add_path(
    zip: &mut zip::ZipWriter<std::fs::File>,
    p: &Path,
    name: &str,
    count: &mut i32,
    bytes: &mut u64,
) -> Result<(), String> {
    use zip::write::FileOptions;
    if p.is_dir() {
        for ent in fs::read_dir(p).map_err(|e| e.to_string())? {
            let ent = ent.map_err(|e| e.to_string())?;
            let path = ent.path();
            let child_name =
                format!("{}/{}", name.trim_end_matches('/'), ent.file_name().to_string_lossy());
            zip_add_path(zip, &path, &child_name, count, bytes)?;
        }
        Ok(())
    } else {
        zip.start_file(name, FileOptions::default()).map_err(|e| e.to_string())?;
        let mut f = fs::File::open(p).map_err(|e| e.to_string())?;
        let n = std::io::copy(&mut f, zip).map_err(|e| e.to_string())?;
        *count += 1;
        *bytes += n;
        Ok(())
    }
}

#[cfg(feature = "fs_archive")]
fn zip_create(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("ZipCreate".into())), args };
    }
    let dest = to_string(ev, args[0].clone());
    let inputs = ev.eval(args[1].clone());
    let opts = if args.len() > 2 {
        if let Value::Assoc(m) = ev.eval(args[2].clone()) {
            m
        } else {
            HashMap::new()
        }
    } else {
        HashMap::new()
    };
    let basedir = opts
        .get("BaseDir")
        .and_then(
            |v| if let Value::String(s) | Value::Symbol(s) = v { Some(s.clone()) } else { None },
        )
        .unwrap_or_else(|| String::from(""));
    let file = match fs::File::create(&dest) {
        Ok(f) => f,
        Err(e) => return failure("FS::zip", &e.to_string()),
    };
    let mut zip = zip::ZipWriter::new(file);
    let mut count = 0;
    let mut bytes = 0u64;
    match inputs {
        Value::List(xs) => {
            for v in xs {
                if let Value::String(s) | Value::Symbol(s) = v {
                    let p = Path::new(&s);
                    let name = if basedir.is_empty() {
                        p.to_string_lossy().to_string()
                    } else {
                        Path::new(&basedir).join(p).to_string_lossy().to_string()
                    };
                    let name = name
                        .trim_start_matches(&basedir)
                        .trim_start_matches(std::path::MAIN_SEPARATOR)
                        .to_string();
                    if let Err(e) = zip_add_path(&mut zip, p, &name, &mut count, &mut bytes) {
                        return failure("FS::zip", &e);
                    }
                }
            }
        }
        _ => return failure("FS::zip", "Inputs must be list of paths"),
    }
    let _ = zip.finish();
    Value::Assoc(HashMap::from([
        ("path".into(), Value::String(dest)),
        ("files".into(), Value::Integer(count as i64)),
        ("bytes".into(), Value::Integer(bytes as i64)),
    ]))
}

#[cfg(feature = "fs_archive")]
fn zip_extract(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("ZipExtract".into())), args };
    }
    let src = to_string(ev, args[0].clone());
    let dest = to_string(ev, args[1].clone());
    let file = match fs::File::open(&src) {
        Ok(f) => f,
        Err(e) => return failure("FS::zip", &e.to_string()),
    };
    let mut archive = match zip::ZipArchive::new(file) {
        Ok(a) => a,
        Err(e) => return failure("FS::zip", &e.to_string()),
    };
    if let Err(e) = fs::create_dir_all(&dest) {
        return failure("FS::zip", &e.to_string());
    }
    let mut files = 0;
    for i in 0..archive.len() {
        let mut f = archive.by_index(i).map_err(|e| e.to_string()).unwrap();
        let outpath = Path::new(&dest).join(f.mangled_name());
        if (*f.name()).ends_with('/') {
            let _ = fs::create_dir_all(&outpath);
        } else {
            if let Some(p) = outpath.parent() {
                let _ = fs::create_dir_all(p);
            }
            let mut out = fs::File::create(&outpath).map_err(|e| e.to_string()).unwrap();
            let _ = std::io::copy(&mut f, &mut out);
            files += 1;
        }
    }
    Value::Assoc(HashMap::from([
        ("path".into(), Value::String(dest)),
        ("files".into(), Value::Integer(files)),
    ]))
}

#[cfg(feature = "fs_archive")]
fn tar_create(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("TarCreate".into())), args };
    }
    let dest = to_string(ev, args[0].clone());
    let inputs = ev.eval(args[1].clone());
    let gz = if args.len() > 2 {
        matches!(ev.eval(args[2].clone()), Value::Assoc(m) if m.get("Gzip") == Some(&Value::Boolean(true)))
    } else {
        false
    };
    let file = match fs::File::create(&dest) {
        Ok(f) => f,
        Err(e) => return failure("FS::tar", &e.to_string()),
    };
    let writer: Box<dyn std::io::Write> = if gz {
        Box::new(flate2::write::GzEncoder::new(file, flate2::Compression::default()))
    } else {
        Box::new(file)
    };
    let mut builder = tar::Builder::new(writer);
    match inputs {
        Value::List(xs) => {
            for v in xs {
                if let Value::String(s) | Value::Symbol(s) = v {
                    let p = Path::new(&s);
                    if let Err(e) = builder.append_path(p) {
                        return failure("FS::tar", &e.to_string());
                    }
                }
            }
        }
        _ => return failure("FS::tar", "Inputs must be list of paths"),
    }
    if let Err(e) = builder.finish() {
        return failure("FS::tar", &e.to_string());
    }
    Value::Assoc(HashMap::from([
        ("path".into(), Value::String(dest)),
        ("files".into(), Value::Integer(0)),
    ]))
}

#[cfg(feature = "fs_archive")]
fn tar_extract(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("TarExtract".into())), args };
    }
    let src = to_string(ev, args[0].clone());
    let dest = to_string(ev, args[1].clone());
    if let Err(e) = fs::create_dir_all(&dest) {
        return failure("FS::tar", &e.to_string());
    }
    let file = match fs::File::open(&src) {
        Ok(f) => f,
        Err(e) => return failure("FS::tar", &e.to_string()),
    };
    let reader: Box<dyn std::io::Read> = if src.ends_with(".gz") {
        Box::new(flate2::read::GzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let mut archive = tar::Archive::new(reader);
    if let Err(e) = archive.unpack(&dest) {
        return failure("FS::tar", &e.to_string());
    }
    Value::Assoc(HashMap::from([
        ("path".into(), Value::String(dest)),
        ("files".into(), Value::Integer(0)),
    ]))
}

#[cfg(feature = "fs_archive")]
fn gzip_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("Gzip".into())), args };
    }
    // Accept Path or bytes (string)
    let data = ev.eval(args[0].clone());
    let out_path = args.get(1).and_then(|v| {
        if let Value::Assoc(m) = ev.eval(v.clone()) {
            m.get("Out").and_then(|x| {
                if let Value::String(s) | Value::Symbol(s) = x {
                    Some(s.clone())
                } else {
                    None
                }
            })
        } else {
            None
        }
    });
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    match data {
        Value::String(s) => {
            let _ = encoder.write_all(s.as_bytes());
        }
        Value::Symbol(s) => {
            let _ = encoder.write_all(s.as_bytes());
        }
        _ => {
            let p = to_string(ev, args[0].clone());
            match fs::read(&p) {
                Ok(b) => {
                    let _ = encoder.write_all(&b);
                }
                Err(e) => return failure("FS::gzip", &e.to_string()),
            }
        }
    }
    let out = encoder.finish().map_err(|e| e.to_string()).unwrap_or_default();
    if let Some(path) = out_path {
        match fs::write(&path, &out) {
            Ok(_) => Value::Assoc(HashMap::from([
                ("path".into(), Value::String(path)),
                ("bytes_written".into(), Value::Integer(out.len() as i64)),
            ])),
            Err(e) => failure("FS::gzip", &e.to_string()),
        }
    } else {
        Value::String(String::from_utf8_lossy(&out).to_string())
    }
}

#[cfg(feature = "fs_archive")]
fn gunzip_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("Gunzip".into())), args };
    }
    let data = ev.eval(args[0].clone());
    let out_path = args.get(1).and_then(|v| {
        if let Value::Assoc(m) = ev.eval(v.clone()) {
            m.get("Out").and_then(|x| {
                if let Value::String(s) | Value::Symbol(s) = x {
                    Some(s.clone())
                } else {
                    None
                }
            })
        } else {
            None
        }
    });
    let input_bytes = match data {
        Value::String(s) | Value::Symbol(s) => s.into_bytes(),
        _ => {
            let p = to_string(ev, args[0].clone());
            match fs::read(&p) {
                Ok(b) => b,
                Err(e) => return failure("FS::gzip", &e.to_string()),
            }
        }
    };
    let mut decoder = flate2::read::GzDecoder::new(std::io::Cursor::new(input_bytes));
    let mut out: Vec<u8> = Vec::new();
    match std::io::copy(&mut decoder, &mut out) {
        Ok(_) => {}
        Err(e) => return failure("FS::gzip", &e.to_string()),
    }
    if let Some(path) = out_path {
        match fs::write(&path, &out) {
            Ok(_) => Value::Assoc(HashMap::from([
                ("path".into(), Value::String(path)),
                ("bytes_written".into(), Value::Integer(out.len() as i64)),
            ])),
            Err(e) => failure("FS::gzip", &e.to_string()),
        }
    } else {
        Value::String(String::from_utf8_lossy(&out).to_string())
    }
}
