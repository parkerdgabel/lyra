use lyra_core::value::Value;
use lyra_runtime::{Evaluator};
use lyra_runtime::attrs::Attributes;
use serde_json as sj;
use serde_json::ser::{PrettyFormatter, Serializer};
use serde::Serialize;

#[cfg(feature = "tools")] use crate::tools::add_specs;
#[cfg(feature = "tools")] use crate::tool_spec;
#[cfg(feature = "tools")] use crate::{schema_str, schema_bool};
#[cfg(feature = "tools")] use std::collections::HashMap;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

pub fn register_io(ev: &mut Evaluator) {
    ev.register("ReadFile", read_file as NativeFn, Attributes::empty());
    ev.register("WriteFile", write_file as NativeFn, Attributes::empty());
    ev.register("ReadLines", read_lines as NativeFn, Attributes::empty());
    ev.register("FileExistsQ", file_exists_q as NativeFn, Attributes::empty());
    ev.register("ListDirectory", list_directory as NativeFn, Attributes::empty());
    ev.register("Stat", stat_fn as NativeFn, Attributes::empty());
    ev.register("PathJoin", path_join as NativeFn, Attributes::empty());
    ev.register("PathSplit", path_split as NativeFn, Attributes::empty());
    ev.register("CanonicalPath", canonical_path as NativeFn, Attributes::empty());
    ev.register("ExpandPath", expand_path as NativeFn, Attributes::empty());
    ev.register("CurrentDirectory", current_directory as NativeFn, Attributes::empty());
    ev.register("SetDirectory", set_directory as NativeFn, Attributes::empty());
    ev.register("Basename", basename_fn as NativeFn, Attributes::empty());
    ev.register("Dirname", dirname_fn as NativeFn, Attributes::empty());
    ev.register("FileStem", file_stem_fn as NativeFn, Attributes::empty());
    ev.register("FileExtension", file_extension_fn as NativeFn, Attributes::empty());
    ev.register("GetEnv", get_env as NativeFn, Attributes::empty());
    ev.register("SetEnv", set_env as NativeFn, Attributes::empty());
    ev.register("ReadStdin", read_stdin as NativeFn, Attributes::empty());
    ev.register("ToJson", to_json as NativeFn, Attributes::empty());
    ev.register("FromJson", from_json as NativeFn, Attributes::empty());
    ev.register("ParseCSV", parse_csv as NativeFn, Attributes::empty());
    ev.register("ReadCSV", read_csv as NativeFn, Attributes::empty());
    ev.register("RenderCSV", render_csv as NativeFn, Attributes::empty());
    ev.register("WriteCSV", write_csv as NativeFn, Attributes::empty());

    #[cfg(feature = "tools")]
    add_specs(vec![
        tool_spec!("ReadFile", summary: "Read entire file as string", params: ["path"], tags: ["io","fs"], input_schema: schema_str!(), output_schema: schema_str!(), effects: ["fs.read"]),
        tool_spec!("WriteFile", summary: "Write stringified content to file", params: ["path","content"], tags: ["io","fs"], input_schema: lyra_core::value::Value::Assoc(HashMap::new()), effects: ["fs.write"]),
        tool_spec!("ReadLines", summary: "Read file and split into lines", params: ["path"], tags: ["io","fs"], input_schema: schema_str!(), effects: ["fs.read"]),
        tool_spec!("FileExistsQ", summary: "Check if file or directory exists", params: ["path"], tags: ["io","fs"], input_schema: schema_str!(), output_schema: schema_bool!()),
        tool_spec!("ListDirectory", summary: "List directory entries (names only)", params: ["path"], tags: ["io","fs"], input_schema: schema_str!(), output_schema: lyra_core::value::Value::Assoc(HashMap::from([(String::from("type"), lyra_core::value::Value::String(String::from("array")))])), effects: ["fs.read"]),
        tool_spec!("Stat", summary: "Get basic file metadata", params: ["path"], tags: ["io","fs"], input_schema: schema_str!(), output_schema: lyra_core::value::Value::Assoc(HashMap::from([(String::from("type"), lyra_core::value::Value::String(String::from("object")))])), effects: ["fs.read"]),
        tool_spec!("PathJoin", summary: "Join path segments", params: ["parts"], tags: ["io","path"]),
        tool_spec!("PathSplit", summary: "Split path into segments", params: ["path"], tags: ["io","path"], output_schema: lyra_core::value::Value::Assoc(HashMap::from([(String::from("type"), lyra_core::value::Value::String(String::from("array")))]))),
        tool_spec!("CanonicalPath", summary: "Resolve symlinks and norm path", params: ["path"], tags: ["io","path"], effects: ["fs.read"]),
        tool_spec!("ExpandPath", summary: "Expand ~ to home", params: ["path"], tags: ["io","path"]),
        tool_spec!("CurrentDirectory", summary: "Get current working directory", params: [], tags: ["io","path"]),
        tool_spec!("SetDirectory", summary: "Change current directory", params: ["path"], tags: ["io","path"], effects: ["fs.chdir"]),
        tool_spec!("Basename", summary: "Last path component", params: ["path"], tags: ["io","path"]),
        tool_spec!("Dirname", summary: "Parent directory path", params: ["path"], tags: ["io","path"]),
        tool_spec!("FileStem", summary: "Filename without extension", params: ["path"], tags: ["io","path"]),
        tool_spec!("FileExtension", summary: "File extension (no dot)", params: ["path"], tags: ["io","path"]),
        tool_spec!("GetEnv", summary: "Read environment variable", params: ["name"], tags: ["io","env"], effects: ["env.read"]),
        tool_spec!("SetEnv", summary: "Set environment variable", params: ["name","value"], tags: ["io","env"], effects: ["env.write"]),
        tool_spec!("ReadStdin", summary: "Read all text from stdin", params: [], tags: ["io","stdio"], effects: ["stdio.read"]),
        tool_spec!("ToJson", summary: "Serialize value to JSON string", params: ["value","opts"], tags: ["io","json"]),
        tool_spec!("FromJson", summary: "Parse JSON string to value", params: ["json"], tags: ["io","json"]),
        tool_spec!("ParseCSV", summary: "Parse CSV string to rows", params: ["csv","opts"], tags: ["io","csv"]),
        tool_spec!("ReadCSV", summary: "Read and parse CSV file", params: ["path","opts"], tags: ["io","csv"], effects: ["fs.read"]),
        tool_spec!("RenderCSV", summary: "Render rows to CSV string", params: ["rows","opts"], tags: ["io","csv"]),
        tool_spec!("WriteCSV", summary: "Write rows to CSV file", params: ["path","rows","opts"], tags: ["io","csv"], effects: ["fs.write"]),
    ]);
}

fn read_file(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("ReadFile".into())), args } }
    let path = match &args[0] { Value::String(s)|Value::Symbol(s)=>s.clone(), other=>return Value::Expr { head: Box::new(Value::Symbol("ReadFile".into())), args: vec![other.clone()] } };
    match std::fs::read_to_string(&path) {
        Ok(s) => Value::String(s),
        Err(e) => failure("IO::read", &format!("ReadFile: {}", e)),
    }
}

fn write_file(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("WriteFile".into())), args } }
    let path = match &args[0] { Value::String(s)|Value::Symbol(s)=>s.clone(), other=>return Value::Expr { head: Box::new(Value::Symbol("WriteFile".into())), args: vec![other.clone(), args[1].clone()] } };
    let content_v = ev.eval(args[1].clone());
    let content = match &content_v { Value::String(s)=>s.clone(), _ => lyra_core::pretty::format_value(&content_v) };
    match std::fs::write(&path, content) {
        Ok(_) => Value::Boolean(true),
        Err(e) => failure("IO::write", &format!("WriteFile: {}", e)),
    }
}

fn read_lines(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("ReadLines".into())), args } }
    let path = match &args[0] { Value::String(s)|Value::Symbol(s)=>s.clone(), other=>return Value::Expr { head: Box::new(Value::Symbol("ReadLines".into())), args: vec![other.clone()] } };
    match std::fs::read_to_string(&path) {
        Ok(s) => Value::List(s.lines().map(|x| Value::String(x.to_string())).collect()),
        Err(e) => failure("IO::read", &format!("ReadLines: {}", e)),
    }
}

fn file_exists_q(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("FileExistsQ".into())), args } }
    let path = match &args[0] { Value::String(s)|Value::Symbol(s)=>s.clone(), other=>return Value::Expr { head: Box::new(Value::Symbol("FileExistsQ".into())), args: vec![other.clone()] } };
    Value::Boolean(std::path::Path::new(&path).exists())
}

fn list_directory(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("ListDirectory".into())), args } }
    let path = match &args[0] { Value::String(s)|Value::Symbol(s)=>s.clone(), other=>return Value::Expr { head: Box::new(Value::Symbol("ListDirectory".into())), args: vec![other.clone()] } };
    let mut out: Vec<Value> = Vec::new();
    match std::fs::read_dir(&path) {
        Ok(read_dir) => {
            for entry in read_dir.flatten() {
                if let Some(name) = entry.file_name().to_str() { out.push(Value::String(name.to_string())); }
            }
            Value::List(out)
        }
        Err(e) => failure("IO::list", &format!("ListDirectory: {}", e)),
    }
}

fn stat_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Stat".into())), args } }
    let path = match &args[0] { Value::String(s)|Value::Symbol(s)=>s.clone(), other=>return Value::Expr { head: Box::new(Value::Symbol("Stat".into())), args: vec![other.clone()] } };
    match std::fs::metadata(&path) {
        Ok(meta) => {
            let is_dir = meta.is_dir();
            let is_file = meta.is_file();
            let size = meta.len() as i64;
            let modified = meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs() as i64).unwrap_or(0);
            let mut m: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
            m.insert("isDir".into(), Value::Boolean(is_dir));
            m.insert("isFile".into(), Value::Boolean(is_file));
            m.insert("size".into(), Value::Integer(size));
            m.insert("modified".into(), Value::Integer(modified));
            Value::Assoc(m)
        }
        Err(e) => failure("IO::stat", &format!("Stat: {}", e)),
    }
}

fn failure(tag: &str, msg: &str) -> Value {
    Value::Assoc(vec![
        ("message".to_string(), Value::String(msg.to_string())),
        ("tag".to_string(), Value::String(tag.to_string())),
    ].into_iter().collect())
}

fn to_string_arg(ev: &mut Evaluator, v: Value) -> String {
    match ev.eval(v) {
        Value::String(s) | Value::Symbol(s) => s,
        other => lyra_core::pretty::format_value(&other),
    }
}

fn path_join(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    use std::path::{PathBuf};
    if args.is_empty() { return Value::String(String::new()); }
    let parts: Vec<String> = if args.len()==1 {
        match ev.eval(args[0].clone()) {
            Value::List(items) => items.into_iter().map(|it| to_string_arg(ev, it)).collect(),
            other => vec![to_string_arg(ev, other)],
        }
    } else { args.into_iter().map(|a| to_string_arg(ev, a)).collect() };
    let mut pb = PathBuf::new();
    for p in parts { pb.push(p); }
    Value::String(pb.to_string_lossy().to_string())
}

fn path_split(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("PathSplit".into())), args } }
    let path = to_string_arg(ev, args[0].clone());
    let p = std::path::Path::new(&path);
    let comps: Vec<Value> = p.components().map(|c| Value::String(c.as_os_str().to_string_lossy().to_string())).collect();
    Value::List(comps)
}

fn canonical_path(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("CanonicalPath".into())), args } }
    let path = to_string_arg(ev, args[0].clone());
    match std::fs::canonicalize(&path) {
        Ok(pb) => Value::String(pb.to_string_lossy().to_string()),
        Err(e) => failure("IO::canonical", &format!("CanonicalPath: {}", e)),
    }
}

fn expand_path(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("ExpandPath".into())), args } }
    let path = to_string_arg(ev, args[0].clone());
    if path == "~" || path.starts_with("~/") {
        let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")).unwrap_or_else(|_| String::from(""));
        let rest = if path == "~" { String::new() } else { path[2..].to_string() };
        let mut pb = std::path::PathBuf::from(home);
        if !rest.is_empty() { pb.push(rest); }
        Value::String(pb.to_string_lossy().to_string())
    } else { Value::String(path) }
}

fn current_directory(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if !args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("CurrentDirectory".into())), args } }
    match std::env::current_dir() { Ok(p) => Value::String(p.to_string_lossy().to_string()), Err(e) => failure("IO::cwd", &format!("CurrentDirectory: {}", e)) }
}

fn set_directory(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("SetDirectory".into())), args } }
    let path = to_string_arg(ev, args[0].clone());
    match std::env::set_current_dir(&path) { Ok(_) => Value::Boolean(true), Err(e) => failure("IO::chdir", &format!("SetDirectory: {}", e)) }
}

fn basename_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Basename".into())), args } }
    let path = to_string_arg(ev, args[0].clone());
    let p = std::path::Path::new(&path);
    Value::String(p.file_name().unwrap_or_default().to_string_lossy().to_string())
}

fn dirname_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Dirname".into())), args } }
    let path = to_string_arg(ev, args[0].clone());
    let p = std::path::Path::new(&path);
    Value::String(p.parent().unwrap_or_else(|| std::path::Path::new("")).to_string_lossy().to_string())
}

fn file_stem_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("FileStem".into())), args } }
    let path = to_string_arg(ev, args[0].clone());
    let p = std::path::Path::new(&path);
    Value::String(p.file_stem().unwrap_or_default().to_string_lossy().to_string())
}

fn file_extension_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("FileExtension".into())), args } }
    let path = to_string_arg(ev, args[0].clone());
    let p = std::path::Path::new(&path);
    Value::String(p.extension().unwrap_or_default().to_string_lossy().to_string())
}

fn get_env(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("GetEnv".into())), args } }
    let name = to_string_arg(ev, args[0].clone());
    match std::env::var(&name) { Ok(v) => Value::String(v), Err(_) => Value::Symbol("Null".into()) }
}

fn set_env(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("SetEnv".into())), args } }
    let name = to_string_arg(ev, args[0].clone());
    let val = to_string_arg(ev, args[1].clone());
    std::env::set_var(name, val);
    Value::Boolean(true)
}

fn read_stdin(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if !args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("ReadStdin".into())), args } }
    use std::io::Read;
    let mut s = String::new();
    match std::io::stdin().read_to_string(&mut s) { Ok(_) => Value::String(s), Err(e) => failure("IO::stdin", &format!("ReadStdin: {}", e)) }
}

// ---------------- JSON ----------------
struct JsonOpts { pretty: bool, sort_keys: bool, ensure_ascii: bool, trailing_newline: bool, indent: Option<Vec<u8>> }

fn json_opts_from(ev: &mut Evaluator, v: Option<Value>) -> JsonOpts {
    let mut o = JsonOpts { pretty: false, sort_keys: false, ensure_ascii: false, trailing_newline: false, indent: None };
    if let Some(Value::Assoc(m)) = v.map(|x| ev.eval(x)) {
        if let Some(Value::Boolean(b)) = m.get("Pretty") { o.pretty = *b; }
        if let Some(Value::Boolean(b)) = m.get("SortKeys") { o.sort_keys = *b; }
        if let Some(Value::Boolean(b)) = m.get("EnsureAscii") { o.ensure_ascii = *b; }
        if let Some(Value::Boolean(b)) = m.get("TrailingNewline") { o.trailing_newline = *b; }
        if let Some(Value::Integer(n)) = m.get("Indent") { let n = (*n).clamp(0, 8) as usize; if n>0 { o.indent = Some(vec![b' '; n]); } }
        if let Some(Value::String(s)) = m.get("Indent") { let bytes = s.as_bytes().to_vec(); if !bytes.is_empty() { o.indent = Some(bytes); } }
    }
    o
}

fn to_json(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("ToJson".into())), args } }
    let v = ev.eval(args[0].clone());
    let opts = json_opts_from(ev, args.get(1).cloned());
    let j = value_to_json_opts(&v, opts.sort_keys);
    let mut out: Vec<u8> = Vec::new();
    if opts.pretty {
        let indent = opts.indent.unwrap_or_else(|| vec![b' '; 2]);
        let fmt = PrettyFormatter::with_indent(&indent);
        let mut ser = Serializer::with_formatter(&mut out, fmt);
        if let Err(e) = j.serialize(&mut ser) { return failure("IO::json", &format!("ToJson: {}", e)); }
    } else {
        let mut ser = Serializer::new(&mut out);
        if let Err(e) = j.serialize(&mut ser) { return failure("IO::json", &format!("ToJson: {}", e)); }
    }
    let mut s = String::from_utf8_lossy(&out).to_string();
    if opts.ensure_ascii {
        s = ensure_ascii_json(&s);
    }
    if opts.trailing_newline { s.push('\n'); }
    Value::String(s)
}

fn from_json(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("FromJson".into())), args } }
    let s = to_string_arg(ev, args[0].clone());
    match sj::from_str::<sj::Value>(&s) {
        Ok(j) => json_to_value(&j),
        Err(e) => failure("IO::json", &format!("FromJson: {}", e)),
    }
}

fn value_to_json_opts(v: &Value, sort_keys: bool) -> sj::Value {
    match v {
        Value::Integer(n) => sj::Value::Number((*n).into()),
        Value::Real(f) => sj::json!(*f),
        Value::String(s) => sj::Value::String(s.clone()),
        Value::Symbol(s) => { if s=="Null" { sj::Value::Null } else { sj::Value::String(s.clone()) } },
        Value::Boolean(b) => sj::Value::Bool(*b),
        Value::List(items) => sj::Value::Array(items.iter().map(|x| value_to_json_opts(x, sort_keys)).collect()),
        Value::Assoc(m) => {
            let mut obj = serde_json::Map::new();
            if sort_keys {
                let mut keys: Vec<&String> = m.keys().collect();
                keys.sort();
                for k in keys { obj.insert(k.clone(), value_to_json_opts(m.get(k).unwrap(), sort_keys)); }
            } else {
                for (k, vv) in m.iter() { obj.insert(k.clone(), value_to_json_opts(vv, sort_keys)); }
            }
            sj::Value::Object(obj)
        }
        _ => sj::Value::String(lyra_core::pretty::format_value(v)),
    }
}

fn json_to_value(j: &sj::Value) -> Value {
    match j {
        sj::Value::Null => Value::Symbol("Null".into()),
        sj::Value::Bool(b) => Value::Boolean(*b),
        sj::Value::Number(n) => {
            if let Some(i) = n.as_i64() { Value::Integer(i) }
            else if let Some(f) = n.as_f64() { Value::Real(f) }
            else { Value::String(n.to_string()) }
        }
        sj::Value::String(s) => Value::String(s.clone()),
        sj::Value::Array(arr) => Value::List(arr.iter().map(json_to_value).collect()),
        sj::Value::Object(map) => Value::Assoc(map.iter().map(|(k,v)| (k.clone(), json_to_value(v))).collect()),
    }
}

fn ensure_ascii_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        if ch <= '\u{7F}' { out.push(ch); }
        else {
            let mut buf = [0u16; 2];
            for unit in ch.encode_utf16(&mut buf) { out.push_str(&format!("\\u{:04X}", unit)); }
        }
    }
    out
}

// ---------------- CSV ----------------
#[derive(Clone)]
struct CsvOpts { delim: char, quote: char, header: bool, eol: &'static str, columns: Option<Vec<String>>, headers: Option<Vec<String>> }

fn csv_opts_from(ev: &mut Evaluator, v: Option<Value>) -> CsvOpts {
    let mut o = CsvOpts { delim: ',', quote: '"', header: true, eol: "\n", columns: None, headers: None };
    if let Some(Value::Assoc(m)) = v.map(|x| ev.eval(x)) {
        if let Some(Value::String(s))|Some(Value::Symbol(s)) = m.get("Delimiter") { if let Some(c) = s.chars().next() { o.delim = c; } }
        if let Some(Value::String(s))|Some(Value::Symbol(s)) = m.get("Quote") { if let Some(c) = s.chars().next() { o.quote = c; } }
        if let Some(Value::Boolean(b)) = m.get("Header") { o.header = *b; }
        if let Some(Value::String(s)) = m.get("Eol") { if s == "\r\n" { o.eol = "\r\n"; } }
        if let Some(Value::List(cols_v)) = m.get("Columns") {
            let cols: Vec<String> = cols_v.iter().filter_map(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None }).collect();
            if !cols.is_empty() { o.columns = Some(cols); }
        }
        if let Some(Value::List(cols_v)) = m.get("Headers") {
            let cols: Vec<String> = cols_v.iter().filter_map(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None }).collect();
            if !cols.is_empty() { o.headers = Some(cols); }
        }
    }
    o
}

fn parse_csv_records(s: &str, opts: &CsvOpts) -> Vec<Vec<String>> {
    let mut out: Vec<Vec<String>> = Vec::new();
    let mut rec: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut it = s.chars().peekable();
    let mut in_q = false;
    while let Some(ch) = it.next() {
        if in_q {
            if ch == opts.quote {
                if let Some(nextc) = it.peek() { if *nextc == opts.quote { field.push(opts.quote); it.next(); } else { in_q = false; } }
                else { in_q = false; }
            } else { field.push(ch); }
        } else {
            if ch == opts.quote { in_q = true; }
            else if ch == opts.delim { rec.push(field.clone()); field.clear(); }
            else if ch == '\n' || ch == '\r' {
                // handle CRLF
                if ch == '\r' { if let Some('\n') = it.peek().copied() { it.next(); } }
                rec.push(field.clone()); field.clear();
                out.push(rec); rec = Vec::new();
            } else { field.push(ch); }
        }
    }
    // flush last
    if in_q { /* unbalanced quote: treat as text */ }
    rec.push(field);
    if !rec.is_empty() { out.push(rec); }
    out
}

fn parse_csv(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("ParseCSV".into())), args } }
    let s = to_string_arg(ev, args[0].clone());
    let opts = csv_opts_from(ev, args.get(1).cloned());
    let rows = parse_csv_records(&s, &opts);
    csv_rows_to_values(rows, &opts)
}

fn read_csv(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("ReadCSV".into())), args } }
    let path = to_string_arg(ev, args[0].clone());
    let s = match std::fs::read_to_string(&path) { Ok(x)=>x, Err(e)=> return failure("IO::csv", &format!("ReadCSV: {}", e)) };
    let opts = csv_opts_from(ev, args.get(1).cloned());
    let rows = parse_csv_records(&s, &opts);
    csv_rows_to_values(rows, &opts)
}

fn csv_rows_to_values(rows: Vec<Vec<String>>, opts: &CsvOpts) -> Value {
    if rows.is_empty() { return Value::List(vec![]); }
    if opts.headers.is_some() || opts.header {
        let headers: Vec<String> = if let Some(h) = &opts.headers { h.clone() } else { rows[0].clone() };
        let mut out: Vec<Value> = Vec::new();
        let start_idx = if opts.headers.is_some() { 0 } else { 1 };
        for r in rows.into_iter().skip(start_idx) {
            let mut m: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
            for (i, h) in headers.iter().enumerate() {
                if let Some(cell) = r.get(i) { m.insert(h.clone(), Value::String(cell.clone())); }
                else { m.insert(h.clone(), Value::Symbol("Null".into())); }
            }
            out.push(Value::Assoc(m));
        }
        Value::List(out)
    } else {
        Value::List(rows.into_iter().map(|r| Value::List(r.into_iter().map(Value::String).collect())).collect())
    }
}

fn render_csv(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("RenderCSV".into())), args } }
    let data = ev.eval(args[0].clone());
    let opts = csv_opts_from(ev, args.get(1).cloned());
    match render_csv_string(&data, &opts) { Ok(s)=>Value::String(s), Err(e)=>failure("IO::csv", &e) }
}

fn write_csv(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("WriteCSV".into())), args } }
    let path = to_string_arg(ev, args[0].clone());
    let data = ev.eval(args[1].clone());
    let opts = csv_opts_from(ev, args.get(2).cloned());
    match render_csv_string(&data, &opts) { Ok(s)=> match std::fs::write(&path, s) { Ok(_)=>Value::Boolean(true), Err(e)=>failure("IO::csv", &format!("WriteCSV: {}", e)) }, Err(e)=>failure("IO::csv", &e) }
}

fn render_csv_string(data: &Value, opts: &CsvOpts) -> Result<String, String> {
    fn esc(cell: &str, delim: char, quote: char) -> String {
        let must_quote = cell.contains(delim) || cell.contains(quote) || cell.contains('\n') || cell.contains('\r');
        if must_quote { format!("{}{}{}", quote, cell.replace(quote, &format!("{}{}", quote, quote)), quote) } else { cell.to_string() }
    }
    match data {
        Value::List(rows) => {
            if rows.is_empty() { return Ok(String::new()); }
            // Decide format by inspecting first row
            match &rows[0] {
                Value::Assoc(first) => {
                    // columns from opts or derived from first row (sorted for stability)
                    let cols: Vec<String> = if let Some(cols) = &opts.columns { cols.clone() } else { let mut v: Vec<String> = first.keys().cloned().collect(); v.sort(); v };
                    let mut out = String::new();
                    if opts.header {
                        out.push_str(&cols.iter().map(|h| esc(h, opts.delim, opts.quote)).collect::<Vec<_>>().join(&opts.delim.to_string()));
                        out.push_str(opts.eol);
                    }
                    for r in rows {
                        if let Value::Assoc(m) = r {
                            let cells: Vec<String> = cols.iter().map(|k| {
                                let v = m.get(k).cloned().unwrap_or(Value::Symbol("Null".into()));
                                let s = match v { Value::String(s)=>s, Value::Symbol(ref s) if s=="Null" => String::new(), _ => lyra_core::pretty::format_value(&v) };
                                esc(&s, opts.delim, opts.quote)
                            }).collect();
                            out.push_str(&cells.join(&opts.delim.to_string()));
                            out.push_str(opts.eol);
                        }
                    }
                    Ok(out)
                }
                Value::List(_) => {
                    let mut out = String::new();
                    for r in rows {
                        if let Value::List(cells_v) = r {
                            let cells: Vec<String> = cells_v.iter().map(|v| match v { Value::String(s)=>esc(s, opts.delim, opts.quote), _=>esc(&lyra_core::pretty::format_value(v), opts.delim, opts.quote) }).collect();
                            out.push_str(&cells.join(&opts.delim.to_string()));
                            out.push_str(opts.eol);
                        }
                    }
                    Ok(out)
                }
                _ => Err("RenderCSV: expected list of assoc or list of lists".into()),
            }
        }
        _ => Err("RenderCSV: expected list".into()),
    }
}
