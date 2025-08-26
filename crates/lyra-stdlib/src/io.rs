use lyra_core::value::Value;
use lyra_runtime::{Evaluator};
use std::io::Read;
use lyra_runtime::attrs::Attributes;
use serde_json as sj;
use serde_json::ser::{PrettyFormatter, Serializer};
use serde::Serialize;
use crate::register_if;
use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicI64, Ordering};

#[cfg(feature = "tools")] use crate::tools::add_specs;
#[cfg(feature = "tools")] use crate::tool_spec;
#[cfg(feature = "tools")] use crate::{schema_str, schema_bool};
#[cfg(feature = "tools")] use std::collections::HashMap;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

pub fn register_io(ev: &mut Evaluator) {
    ev.register("ReadFile", read_file as NativeFn, Attributes::empty());
    ev.register("WriteFile", write_file as NativeFn, Attributes::empty());
    ev.register("Puts", puts as NativeFn, Attributes::empty());
    ev.register("Gets", gets as NativeFn, Attributes::empty());
    ev.register("PutsAppend", puts_append as NativeFn, Attributes::empty());
    ev.register("ReadLines", read_lines as NativeFn, Attributes::empty());
    ev.register("FileExistsQ", file_exists_q as NativeFn, Attributes::empty());
    ev.register("ListDirectory", list_directory as NativeFn, Attributes::empty());
    ev.register("Stat", stat_fn as NativeFn, Attributes::empty());
    ev.register("PathJoin", path_join as NativeFn, Attributes::empty());
    ev.register("PathSplit", path_split as NativeFn, Attributes::empty());
    ev.register("CanonicalPath", canonical_path as NativeFn, Attributes::empty());
    ev.register("ExpandPath", expand_path as NativeFn, Attributes::empty());
    ev.register("PathNormalize", canonical_path as NativeFn, Attributes::empty());
    ev.register("PathRelative", path_relative as NativeFn, Attributes::empty());
    ev.register("PathResolve", path_resolve as NativeFn, Attributes::empty());
    ev.register("CurrentDirectory", current_directory as NativeFn, Attributes::empty());
    ev.register("SetDirectory", set_directory as NativeFn, Attributes::empty());
    ev.register("Basename", basename_fn as NativeFn, Attributes::empty());
    ev.register("Dirname", dirname_fn as NativeFn, Attributes::empty());
    ev.register("FileStem", file_stem_fn as NativeFn, Attributes::empty());
    ev.register("FileExtension", file_extension_fn as NativeFn, Attributes::empty());
    ev.register("PathExtname", file_extension_fn as NativeFn, Attributes::empty());
    ev.register("GetEnv", get_env as NativeFn, Attributes::empty());
    ev.register("SetEnv", set_env as NativeFn, Attributes::empty());
    ev.register("DotenvLoad", dotenv_load as NativeFn, Attributes::empty());
    ev.register("ConfigFind", config_find as NativeFn, Attributes::empty());
    ev.register("XdgDirs", xdg_dirs as NativeFn, Attributes::empty());
    ev.register("EnvExpand", env_expand as NativeFn, Attributes::empty());
    ev.register("ReadStdin", read_stdin as NativeFn, Attributes::empty());
    // CLI ergonomics
    ev.register("ArgsParse", args_parse as NativeFn, Attributes::empty());
    ev.register("Prompt", prompt as NativeFn, Attributes::empty());
    ev.register("Confirm", confirm as NativeFn, Attributes::empty());
    ev.register("PasswordPrompt", password_prompt as NativeFn, Attributes::empty());
    ev.register("Select", select_fn as NativeFn, Attributes::empty());
    ev.register("ProgressBar", progress_bar as NativeFn, Attributes::empty());
    ev.register("ProgressAdvance", progress_advance as NativeFn, Attributes::empty());
    ev.register("ProgressFinish", progress_finish as NativeFn, Attributes::empty());
    ev.register("ToJson", to_json as NativeFn, Attributes::empty());
    ev.register("FromJson", from_json as NativeFn, Attributes::empty());
    // Aliases for common JSON ops
    ev.register("JsonStringify", to_json as NativeFn, Attributes::empty());
    ev.register("JsonParse", from_json as NativeFn, Attributes::empty());
    // YAML/TOML parse/stringify
    ev.register("YamlParse", yaml_parse as NativeFn, Attributes::empty());
    ev.register("YamlStringify", yaml_stringify as NativeFn, Attributes::empty());
    ev.register("TomlParse", toml_parse as NativeFn, Attributes::empty());
    ev.register("TomlStringify", toml_stringify as NativeFn, Attributes::empty());
    ev.register("ParseCSV", parse_csv as NativeFn, Attributes::empty());
    ev.register("ReadCSV", read_csv as NativeFn, Attributes::empty());
    ev.register("RenderCSV", render_csv as NativeFn, Attributes::empty());
    ev.register("WriteCSV", write_csv as NativeFn, Attributes::empty());
    // Aliases for CSV
    ev.register("CsvRead", parse_csv as NativeFn, Attributes::empty());
    ev.register("CsvWrite", render_csv as NativeFn, Attributes::empty());
    // Bytes / encoding utilities
    ev.register("Base64Encode", base64_encode as NativeFn, Attributes::empty());
    ev.register("Base64Decode", base64_decode as NativeFn, Attributes::empty());
    ev.register("HexEncode", hex_encode_fn as NativeFn, Attributes::empty());
    ev.register("HexDecode", hex_decode_fn as NativeFn, Attributes::empty());
    ev.register("TextEncode", text_encode as NativeFn, Attributes::empty());
    ev.register("TextDecode", text_decode as NativeFn, Attributes::empty());
    ev.register("BytesConcat", bytes_concat as NativeFn, Attributes::empty());
    ev.register("BytesSlice", bytes_slice as NativeFn, Attributes::empty());
    ev.register("BytesLength", bytes_length as NativeFn, Attributes::empty());

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
        tool_spec!("JsonStringify", summary: "Alias of ToJson", params: ["value","opts"], tags: ["io","json"]),
        tool_spec!("JsonParse", summary: "Alias of FromJson", params: ["json"], tags: ["io","json"]),
        tool_spec!("YamlParse", summary: "Parse YAML string to value", params: ["yaml"], tags: ["io","yaml"]),
        tool_spec!("YamlStringify", summary: "Render value as YAML", params: ["value","opts"], tags: ["io","yaml"]),
        tool_spec!("TomlParse", summary: "Parse TOML string to value", params: ["toml"], tags: ["io","toml"]),
        tool_spec!("TomlStringify", summary: "Render value as TOML", params: ["value"], tags: ["io","toml"]),
        tool_spec!("ParseCSV", summary: "Parse CSV string to rows", params: ["csv","opts"], tags: ["io","csv"]),
        tool_spec!("ReadCSV", summary: "Read and parse CSV file", params: ["path","opts"], tags: ["io","csv"], effects: ["fs.read"]),
        tool_spec!("RenderCSV", summary: "Render rows to CSV string", params: ["rows","opts"], tags: ["io","csv"]),
        tool_spec!("WriteCSV", summary: "Write rows to CSV file", params: ["path","rows","opts"], tags: ["io","csv"], effects: ["fs.write"]),
        tool_spec!("CsvRead", summary: "Alias of ParseCSV", params: ["csv","opts"], tags: ["io","csv"]),
        tool_spec!("CsvWrite", summary: "Alias of RenderCSV", params: ["rows","opts"], tags: ["io","csv"]),
        tool_spec!("Base64Encode", summary: "Encode bytes to base64", params: ["bytes","opts"], tags: ["io","bytes","encoding"]),
        tool_spec!("Base64Decode", summary: "Decode base64 to bytes", params: ["text","opts"], tags: ["io","bytes","encoding"]),
        tool_spec!("HexEncode", summary: "Encode bytes to hex", params: ["bytes"], tags: ["io","bytes","encoding"]),
        tool_spec!("HexDecode", summary: "Decode hex to bytes", params: ["text"], tags: ["io","bytes","encoding"]),
        tool_spec!("TextEncode", summary: "Encode text to bytes (utf-8)", params: ["text","opts"], tags: ["io","bytes","encoding"]),
        tool_spec!("TextDecode", summary: "Decode bytes to text (utf-8)", params: ["bytes","opts"], tags: ["io","bytes","encoding"]),
        tool_spec!("BytesConcat", summary: "Concatenate byte arrays", params: ["chunks"], tags: ["io","bytes"], output_schema: lyra_core::value::Value::Assoc(HashMap::from([(String::from("type"), lyra_core::value::Value::String(String::from("array")))]))),
        tool_spec!("BytesSlice", summary: "Slice a byte array", params: ["bytes","start","end"], tags: ["io","bytes"], output_schema: lyra_core::value::Value::Assoc(HashMap::from([(String::from("type"), lyra_core::value::Value::String(String::from("array")))]))),
        tool_spec!("BytesLength", summary: "Length of byte array", params: ["bytes"], tags: ["io","bytes"]),
    ]);
}

pub fn register_io_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str)->bool) {
    register_if(ev, pred, "ReadFile", read_file as NativeFn, Attributes::empty());
    register_if(ev, pred, "WriteFile", write_file as NativeFn, Attributes::empty());
    register_if(ev, pred, "ReadLines", read_lines as NativeFn, Attributes::empty());
    register_if(ev, pred, "FileExistsQ", file_exists_q as NativeFn, Attributes::empty());
    register_if(ev, pred, "ListDirectory", list_directory as NativeFn, Attributes::empty());
    register_if(ev, pred, "Stat", stat_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "PathJoin", path_join as NativeFn, Attributes::empty());
    register_if(ev, pred, "PathSplit", path_split as NativeFn, Attributes::empty());
    register_if(ev, pred, "CanonicalPath", canonical_path as NativeFn, Attributes::empty());
    register_if(ev, pred, "ExpandPath", expand_path as NativeFn, Attributes::empty());
    register_if(ev, pred, "CurrentDirectory", current_directory as NativeFn, Attributes::empty());
    register_if(ev, pred, "SetDirectory", set_directory as NativeFn, Attributes::empty());
    register_if(ev, pred, "Basename", basename_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Dirname", dirname_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "FileStem", file_stem_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "FileExtension", file_extension_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "GetEnv", get_env as NativeFn, Attributes::empty());
    register_if(ev, pred, "SetEnv", set_env as NativeFn, Attributes::empty());
    register_if(ev, pred, "ReadStdin", read_stdin as NativeFn, Attributes::empty());
    register_if(ev, pred, "ArgsParse", args_parse as NativeFn, Attributes::empty());
    register_if(ev, pred, "Prompt", prompt as NativeFn, Attributes::empty());
    register_if(ev, pred, "Confirm", confirm as NativeFn, Attributes::empty());
    register_if(ev, pred, "PasswordPrompt", password_prompt as NativeFn, Attributes::empty());
    register_if(ev, pred, "Select", select_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "ProgressBar", progress_bar as NativeFn, Attributes::empty());
    register_if(ev, pred, "ProgressAdvance", progress_advance as NativeFn, Attributes::empty());
    register_if(ev, pred, "ProgressFinish", progress_finish as NativeFn, Attributes::empty());
    register_if(ev, pred, "ToJson", to_json as NativeFn, Attributes::empty());
    register_if(ev, pred, "FromJson", from_json as NativeFn, Attributes::empty());
    register_if(ev, pred, "JsonStringify", to_json as NativeFn, Attributes::empty());
    register_if(ev, pred, "JsonParse", from_json as NativeFn, Attributes::empty());
    register_if(ev, pred, "YamlParse", yaml_parse as NativeFn, Attributes::empty());
    register_if(ev, pred, "YamlStringify", yaml_stringify as NativeFn, Attributes::empty());
    register_if(ev, pred, "TomlParse", toml_parse as NativeFn, Attributes::empty());
    register_if(ev, pred, "TomlStringify", toml_stringify as NativeFn, Attributes::empty());
    register_if(ev, pred, "ParseCSV", parse_csv as NativeFn, Attributes::empty());
    register_if(ev, pred, "ReadCSV", read_csv as NativeFn, Attributes::empty());
    register_if(ev, pred, "RenderCSV", render_csv as NativeFn, Attributes::empty());
    register_if(ev, pred, "WriteCSV", write_csv as NativeFn, Attributes::empty());
    register_if(ev, pred, "CsvRead", parse_csv as NativeFn, Attributes::empty());
    register_if(ev, pred, "CsvWrite", render_csv as NativeFn, Attributes::empty());
    register_if(ev, pred, "Base64Encode", base64_encode as NativeFn, Attributes::empty());
    register_if(ev, pred, "Base64Decode", base64_decode as NativeFn, Attributes::empty());
    register_if(ev, pred, "HexEncode", hex_encode_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "HexDecode", hex_decode_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "TextEncode", text_encode as NativeFn, Attributes::empty());
    register_if(ev, pred, "TextDecode", text_decode as NativeFn, Attributes::empty());
    register_if(ev, pred, "BytesConcat", bytes_concat as NativeFn, Attributes::empty());
    register_if(ev, pred, "BytesSlice", bytes_slice as NativeFn, Attributes::empty());
    register_if(ev, pred, "BytesLength", bytes_length as NativeFn, Attributes::empty());
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

// Puts[value] -> prints to stdout with newline
// Puts[value, path] -> writes stringified value to file (overwrites)
fn puts(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v] => { let s = to_string_arg(ev, v.clone()); println!("{}", s); Value::Boolean(true) }
        [v, p] => {
            let s = to_string_arg(ev, v.clone());
            let path = to_string_arg(ev, p.clone());
            match std::fs::write(&path, s.as_bytes()) { Ok(_)=> Value::Boolean(true), Err(e)=> failure("IO::puts", &e.to_string()) }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("Puts".into())), args }
    }
}

// Gets[path] -> reads entire file as string
// Future: Gets[] could read stdin
fn gets(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        let mut s = String::new();
        match std::io::stdin().read_to_string(&mut s) { Ok(_)=> Value::String(s), Err(e)=> failure("IO::gets", &e.to_string()) }
    } else if args.len()==1 {
        let path = to_string_arg(ev, args[0].clone());
        match std::fs::read_to_string(&path) { Ok(s)=> Value::String(s), Err(e)=> failure("IO::gets", &e.to_string()) }
    } else {
        Value::Expr { head: Box::new(Value::Symbol("Gets".into())), args }
    }
}

fn puts_append(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("PutsAppend".into())), args } }
    let s = to_string_arg(ev, args[0].clone());
    let path = to_string_arg(ev, args[1].clone());
    match std::fs::OpenOptions::new().create(true).append(true).open(&path).and_then(|mut f| std::io::Write::write_all(&mut f, s.as_bytes())) { Ok(_)=> Value::Boolean(true), Err(e)=> failure("IO::puts", &e.to_string()) }
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

fn path_relative(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("PathRelative".into())), args } }
    let base = to_string_arg(ev, args[0].clone());
    let path = to_string_arg(ev, args[1].clone());
    let bp = std::path::Path::new(&base);
    let pp = std::path::Path::new(&path);
    match pathdiff::diff_paths(pp, bp) { Some(p)=> Value::String(p.to_string_lossy().to_string()), None => Value::String(path) }
}

fn path_resolve(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("PathResolve".into())), args } }
    let base = to_string_arg(ev, args[0].clone());
    let rel = to_string_arg(ev, args[1].clone());
    let p = std::path::Path::new(&base).join(rel);
    Value::String(p.to_string_lossy().to_string())
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

// ---------------- Config / Env helpers ----------------
fn dotenv_load(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let (path_opt, override_vars) = match args.as_slice() {
        [] => (None, false),
        [p] => { let s = to_string_arg(ev, p.clone()); (Some(s), false) },
        [p, o] => { let s = to_string_arg(ev, p.clone()); let ov = matches!(ev.eval(o.clone()), Value::Assoc(m) if m.get("Override").cloned()==Some(Value::Boolean(true))); (Some(s), ov) },
        _ => (None, false)
    };
    let path = path_opt.unwrap_or_else(|| {
        let cwd = std::env::current_dir().unwrap_or_default(); cwd.join(".env").to_string_lossy().to_string()
    });
    let content = match std::fs::read_to_string(&path) { Ok(s)=>s, Err(e)=> return failure("IO::env", &format!("DotenvLoad: {}", e)) };
    let mut count = 0;
    for line in content.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') { continue; }
        let mut parts = t.splitn(2, '=');
        if let (Some(k), Some(v)) = (parts.next(), parts.next()) {
            let key = k.trim().to_string();
            let val = v.trim().trim_matches('"').trim_matches('\'').to_string();
            if override_vars || std::env::var(&key).is_err() { std::env::set_var(&key, &val); count += 1; }
        }
    }
    Value::Assoc(std::collections::HashMap::from([ ("path".into(), Value::String(path)), ("loaded".into(), Value::Integer(count)) ]))
}

fn config_find(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let (names, start_dir) = match args.as_slice() {
        [] => (vec![".env".to_string(), "lyra.toml".to_string(), "config.yaml".to_string()], std::env::current_dir().unwrap_or_default()),
        [n] => {
            let mut v = Vec::new();
            match ev.eval(n.clone()) {
                Value::List(xs) => { for x in xs { if let Value::String(s)|Value::Symbol(s) = x { v.push(s); } } }
                Value::String(s)|Value::Symbol(s) => v.push(s),
                _ => {}
            }
            (if v.is_empty() { vec![".env".into()] } else { v }, std::env::current_dir().unwrap_or_default())
        }
        [n, sdir] => {
            let mut v = Vec::new();
            match ev.eval(n.clone()) {
                Value::List(xs) => { for x in xs { if let Value::String(s)|Value::Symbol(s) = x { v.push(s); } } }
                Value::String(s)|Value::Symbol(s) => v.push(s),
                _ => {}
            }
            let sd = std::path::PathBuf::from(to_string_arg(ev, sdir.clone()));
            (v, sd)
        }
        _ => (vec![".env".into()], std::env::current_dir().unwrap_or_default())
    };
    let mut dir = start_dir.as_path();
    loop {
        for name in &names {
            let p = dir.join(name);
            if p.exists() { return Value::Assoc(std::collections::HashMap::from([ ("path".into(), Value::String(p.to_string_lossy().to_string())), ("name".into(), Value::String(name.clone())) ])); }
        }
        if let Some(parent) = dir.parent() { dir = parent; } else { break; }
    }
    Value::Symbol("Null".into())
}

fn xdg_dirs(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let app = args.get(0).and_then(|v| match ev.eval(v.clone()) { Value::String(s)|Value::Symbol(s)=>Some(s), _=>None });
    let home = std::env::var("HOME").unwrap_or_else(|_| "".into());
    let cfg = std::env::var("XDG_CONFIG_HOME").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| format!("{}/.config", home));
    let cache = std::env::var("XDG_CACHE_HOME").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| format!("{}/.cache", home));
    let data = std::env::var("XDG_DATA_HOME").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| format!("{}/.local/share", home));
    let (cfg, cache, data) = if let Some(appn) = app { (format!("{}/{}", cfg, appn), format!("{}/{}", cache, appn), format!("{}/{}", data, appn)) } else { (cfg, cache, data) };
    Value::Assoc(std::collections::HashMap::from([
        ("config_dir".into(), Value::String(cfg)),
        ("cache_dir".into(), Value::String(cache)),
        ("data_dir".into(), Value::String(data)),
    ]))
}

fn env_expand(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("EnvExpand".into())), args } }
    let text = to_string_arg(ev, args[0].clone());
    let (vars, style_shell) = if args.len()>1 { if let Value::Assoc(m) = ev.eval(args[1].clone()) { let vars = m.get("Vars").and_then(|v| if let Value::Assoc(mm)=v { Some(mm.clone()) } else { None }).unwrap_or_default(); let style_shell = match m.get("Style") { Some(Value::String(s))|Some(Value::Symbol(s)) if s=="windows" => false, _=> true }; (vars, style_shell) } else { (std::collections::HashMap::new(), true) } } else { (std::collections::HashMap::new(), true) };
    let mut out = String::new();
    if style_shell {
        let bytes = text.as_bytes(); let mut i = 0usize;
        while i < bytes.len() {
            if bytes[i]==b'$' {
                if i+1 < bytes.len() && bytes[i+1]==b'{' {
                    // ${VAR}
                    let mut j = i+2; while j < bytes.len() && bytes[j]!=b'}' { j+=1; }
                    if j < bytes.len() {
                        let name = String::from_utf8_lossy(&bytes[i+2..j]).to_string();
                        let val = vars.get(&name).cloned().or_else(|| std::env::var(&name).ok().map(Value::String)).unwrap_or(Value::String(String::new()));
                        match val { Value::String(s)|Value::Symbol(s)=> out.push_str(&s), other => out.push_str(&lyra_core::pretty::format_value(&other)) }
                        i = j+1; continue;
                    }
                }
                // $VAR
                let mut j = i+1; while j < bytes.len() { let c = bytes[j] as char; if c.is_alphanumeric() || c=='_' { j+=1; } else { break; } }
                if j>i+1 {
                    let name = String::from_utf8_lossy(&bytes[i+1..j]).to_string();
                    let val = vars.get(&name).cloned().or_else(|| std::env::var(&name).ok().map(Value::String)).unwrap_or(Value::String(String::new()));
                    match val { Value::String(s)|Value::Symbol(s)=> out.push_str(&s), other => out.push_str(&lyra_core::pretty::format_value(&other)) }
                    i = j; continue;
                }
            }
            out.push(bytes[i] as char); i+=1;
        }
    } else {
        // Windows style %VAR%
        let mut i = 0; let b = text.as_bytes();
        while i < b.len() {
            if b[i]==b'%' {
                if let Some(jrel) = b[i+1..].iter().position(|&ch| ch==b'%') { let j = i+1+jrel; let name = String::from_utf8_lossy(&b[i+1..j]).to_string(); let val = vars.get(&name).cloned().or_else(|| std::env::var(&name).ok().map(Value::String)).unwrap_or(Value::String(String::new())); match val { Value::String(s)|Value::Symbol(s)=> out.push_str(&s), other => out.push_str(&lyra_core::pretty::format_value(&other)) }; i = j+1; continue; }
            }
            out.push(b[i] as char); i+=1;
        }
    }
    Value::String(out)
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

// ------------- CLI Ergonomics -------------
fn args_parse(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // ArgsParse[spec?, argv?]
    let (spec, argv) = match args.as_slice() {
        [] => (Value::Assoc(std::collections::HashMap::new()), std::env::args().skip(1).collect::<Vec<_>>()),
        [s] => (ev.eval(s.clone()), std::env::args().skip(1).collect::<Vec<_>>()),
        [s, av] => {
            let v = match ev.eval(av.clone()) { Value::List(xs) => xs.into_iter().filter_map(|v| if let Value::String(s)|Value::Symbol(s)=v { Some(s) } else { None }).collect(), _ => vec![] };
            (ev.eval(s.clone()), v)
        }
        _ => (Value::Assoc(std::collections::HashMap::new()), vec![])
    };
    let mut opts_spec: std::collections::HashMap<String, (String, String, bool, bool)> = std::collections::HashMap::new(); // name -> (short, typ, repeat, is_flag)
    if let Value::Assoc(m) = spec {
        if let Some(Value::List(items)) = m.get("Options") {
            for it in items {
                if let Value::Assoc(o) = it {
                    let name = o.get("Name").and_then(|v| if let Value::String(s)|Value::Symbol(s)=v { Some(s.clone()) } else { None }).unwrap_or_default();
                    let short = o.get("Short").and_then(|v| if let Value::String(s)|Value::Symbol(s)=v { Some(s.clone()) } else { None }).unwrap_or_default();
                    let typ = o.get("Type").and_then(|v| if let Value::String(s)|Value::Symbol(s)=v { Some(s.clone()) } else { None }).unwrap_or_else(|| "string".into());
                    let rpt = matches!(o.get("Repeat"), Some(Value::Boolean(true)));
                    opts_spec.insert(name, (short, typ, rpt, false));
                }
            }
        }
        if let Some(Value::List(items)) = m.get("Flags") {
            for it in items {
                if let Value::Assoc(o) = it {
                    let name = o.get("Name").and_then(|v| if let Value::String(s)|Value::Symbol(s)=v { Some(s.clone()) } else { None }).unwrap_or_default();
                    let short = o.get("Short").and_then(|v| if let Value::String(s)|Value::Symbol(s)=v { Some(s.clone()) } else { None }).unwrap_or_default();
                    opts_spec.insert(name, (short, "bool".into(), false, true));
                }
            }
        }
    }
    let mut options: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
    let mut flags: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
    let mut rest: Vec<Value> = Vec::new();
    let mut i = 0;
    while i < argv.len() {
        let a = &argv[i];
        if a.starts_with("--") {
            let mut kv = a[2..].splitn(2, '=');
            let k = kv.next().unwrap().to_string();
            let v_opt = kv.next().map(|s| s.to_string());
            if let Some((_, typ, rpt, is_flag)) = opts_spec.get(&k).cloned() {
                if is_flag { flags.insert(k, Value::Boolean(true)); }
                else {
                    let val = if let Some(vs) = v_opt { vs } else { i+=1; if i<argv.len() { argv[i].clone() } else { String::new() } };
                    let v = match typ.as_str() { "int"=> Value::Integer(val.parse::<i64>().unwrap_or(0)), "float"=> Value::Real(val.parse::<f64>().unwrap_or(0.0)), "bool"=> Value::Boolean(val=="true"), _=> Value::String(val) };
                    if rpt { let e = options.entry(k).or_insert_with(|| Value::List(vec![])); if let Value::List(xs) = e { xs.push(v); } } else { options.insert(k, v); }
                }
            }
        } else if a.starts_with('-') && a.len()>1 {
            let s = a[1..].to_string();
            // treat combined like -abc; set flags or take next as value for option type
            for ch in s.chars() {
                let key = opts_spec.iter().find(|(_, (short,_,_,_))| short==&ch.to_string()).map(|(name, (..))| name.clone());
                if let Some(name) = key { if let Some((_, typ, rpt, is_flag)) = opts_spec.get(&name).cloned() { if is_flag { flags.insert(name, Value::Boolean(true)); } else { i+=1; let val = if i<argv.len() { argv[i].clone() } else { String::new() }; let v = match typ.as_str() { "int"=> Value::Integer(val.parse::<i64>().unwrap_or(0)), "float"=> Value::Real(val.parse::<f64>().unwrap_or(0.0)), "bool"=> Value::Boolean(val=="true"), _=> Value::String(val) }; if rpt { let e = options.entry(name).or_insert_with(|| Value::List(vec![])); if let Value::List(xs)=e { xs.push(v); } } else { options.insert(name, v); } } } }
            }
        } else { rest.push(Value::String(a.clone())); }
        i+=1;
    }
    Value::Assoc(std::collections::HashMap::from([
        ("Options".into(), Value::Assoc(options)),
        ("Flags".into(), Value::Assoc(flags)),
        ("Rest".into(), Value::List(rest)),
    ]))
}

fn prompt(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Prompt".into())), args } }
    let msg = to_string_arg(ev, args[0].clone());
    print!("{} ", msg); let _ = std::io::Write::flush(&mut std::io::stdout());
    let mut s = String::new();
    match std::io::stdin().read_line(&mut s) { Ok(_)=> Value::String(s.trim_end_matches(['\n','\r']).to_string()), Err(e)=> failure("IO::prompt", &e.to_string()) }
}

fn confirm(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Confirm".into())), args } }
    let msg = to_string_arg(ev, args[0].clone());
    print!("{} [y/N] ", msg); let _ = std::io::Write::flush(&mut std::io::stdout());
    let mut s = String::new();
    match std::io::stdin().read_line(&mut s) { Ok(_)=> { let t = s.trim().to_ascii_lowercase(); Value::Boolean(t=="y" || t=="yes" ) }, Err(e)=> failure("IO::prompt", &e.to_string()) }
}

fn password_prompt(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("PasswordPrompt".into())), args } }
    let msg = to_string_arg(ev, args[0].clone());
    // Plaintext fallback (no masking)
    print!("{} ", msg); let _ = std::io::Write::flush(&mut std::io::stdout());
    let mut s = String::new();
    match std::io::stdin().read_line(&mut s) { Ok(_)=> Value::String(s.trim_end_matches(['\n','\r']).to_string()), Err(e)=> failure("IO::prompt", &e.to_string()) }
}

fn select_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("Select".into())), args } }
    let msg = to_string_arg(ev, args[0].clone());
    let choices = match ev.eval(args[1].clone()) { Value::List(xs)=> xs.into_iter().filter_map(|v| match v { Value::String(s)|Value::Symbol(s)=>Some((s.clone(), s)), Value::Assoc(m)=> m.get("name").and_then(|n| if let Value::String(ns)|Value::Symbol(ns)=n { Some(ns.clone()) } else { None }).zip(m.get("value").and_then(|n| if let Value::String(ns)|Value::Symbol(ns)=n { Some(ns.clone()) } else { None })), _=> None }).collect::<Vec<_>>(), _ => vec![] };
    println!("{}", msg);
    for (i,(name,_)) in choices.iter().enumerate() { println!("  {}. {}", i+1, name); }
    print!("Select (1-{}): ", choices.len()); let _ = std::io::Write::flush(&mut std::io::stdout());
    let mut s = String::new(); let _ = std::io::stdin().read_line(&mut s);
    if let Ok(n) = s.trim().parse::<usize>() { if n>=1 && n<=choices.len() { return Value::String(choices[n-1].1.clone()); } }
    Value::Symbol("Null".into())
}

#[derive(Clone)] struct PState { total: i64, cur: i64 }
static PB_REG: OnceLock<Mutex<std::collections::HashMap<i64, PState>>> = OnceLock::new();
static PB_NEXT: OnceLock<AtomicI64> = OnceLock::new();
fn pb_reg() -> &'static Mutex<std::collections::HashMap<i64, PState>> { PB_REG.get_or_init(|| Mutex::new(std::collections::HashMap::new())) }
fn pb_next() -> i64 { PB_NEXT.get_or_init(|| AtomicI64::new(1)).fetch_add(1, Ordering::Relaxed) }

fn progress_bar(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("ProgressBar".into())), args } }
    let total = match ev.eval(args[0].clone()) { Value::Integer(n) if n>0 => n, _ => 0 };
    let id = pb_next();
    pb_reg().lock().unwrap().insert(id, PState { total, cur: 0 });
    Value::Integer(id)
}

fn progress_advance(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("ProgressAdvance".into())), args } }
    let id = match ev.eval(args[0].clone()) { Value::Integer(n)=>n, _=>0 };
    let n = if args.len()>1 { match ev.eval(args[1].clone()) { Value::Integer(n)=>n, _=>1 } } else { 1 };
    if let Some(st) = pb_reg().lock().unwrap().get_mut(&id) { st.cur += n; }
    Value::Boolean(true)
}

fn progress_finish(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("ProgressFinish".into())), args } }
    let id = match ev.eval(args[0].clone()) { Value::Integer(n)=>n, _=>0 };
    pb_reg().lock().unwrap().remove(&id);
    Value::Boolean(true)
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

// ---------------- YAML ----------------
fn yaml_parse(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("YamlParse".into())), args } }
    let s = to_string_arg(ev, args[0].clone());
    match serde_yaml::from_str::<serde_yaml::Value>(&s) {
        Ok(y) => yaml_to_value(&y),
        Err(e) => failure("IO::yaml", &format!("YamlParse: {}", e)),
    }
}

fn yaml_to_value(y: &serde_yaml::Value) -> Value {
    match y {
        serde_yaml::Value::Null => Value::Symbol("Null".into()),
        serde_yaml::Value::Bool(b) => Value::Boolean(*b),
        serde_yaml::Value::Number(n) => {
            if let Some(i) = n.as_i64() { Value::Integer(i) }
            else if let Some(f) = n.as_f64() { Value::Real(f) }
            else { Value::String(n.to_string()) }
        }
        serde_yaml::Value::String(s) => Value::String(s.clone()),
        serde_yaml::Value::Sequence(arr) => Value::List(arr.iter().map(yaml_to_value).collect()),
        serde_yaml::Value::Mapping(map) => {
            let mut out = std::collections::HashMap::new();
            for (k, v) in map.iter() {
                let kk = match k { serde_yaml::Value::String(s) => s.clone(), other => format!("{:?}", other) };
                out.insert(kk, yaml_to_value(v));
            }
            Value::Assoc(out)
        }
        _ => Value::Symbol("Null".into()),
    }
}

fn yaml_stringify(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("YamlStringify".into())), args } }
    let v = ev.eval(args[0].clone());
    let j = value_to_json_opts(&v, false);
    match serde_yaml::to_string(&j) {
        Ok(s) => Value::String(s),
        Err(e) => failure("IO::yaml", &format!("YamlStringify: {}", e)),
    }
}

// ---------------- TOML ----------------
fn toml_parse(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("TomlParse".into())), args } }
    let s = to_string_arg(ev, args[0].clone());
    match s.parse::<toml::Value>() {
        Ok(t) => toml_to_value(&t),
        Err(e) => failure("IO::toml", &format!("TomlParse: {}", e)),
    }
}

fn toml_to_value(t: &toml::Value) -> Value {
    match t {
        toml::Value::String(s) => Value::String(s.clone()),
        toml::Value::Integer(i) => Value::Integer(*i),
        toml::Value::Float(f) => Value::Real(*f),
        toml::Value::Boolean(b) => Value::Boolean(*b),
        toml::Value::Datetime(dt) => Value::String(dt.to_string()),
        toml::Value::Array(arr) => Value::List(arr.iter().map(toml_to_value).collect()),
        toml::Value::Table(map) => Value::Assoc(map.iter().map(|(k,v)| (k.clone(), toml_to_value(v))).collect()),
    }
}

fn toml_stringify(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("TomlStringify".into())), args } }
    let v = ev.eval(args[0].clone());
    // Route via JSON mapping for broad coverage
    let j = value_to_json_opts(&v, true);
    // Convert serde_json::Value to toml::Value recursively
    fn json_to_toml(j: &serde_json::Value) -> toml::Value {
        match j {
            serde_json::Value::Null => toml::Value::String("".into()),
            serde_json::Value::Bool(b) => toml::Value::Boolean(*b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() { toml::Value::Integer(i) }
                else if let Some(f) = n.as_f64() { toml::Value::Float(f) }
                else { toml::Value::String(n.to_string()) }
            }
            serde_json::Value::String(s) => toml::Value::String(s.clone()),
            serde_json::Value::Array(arr) => toml::Value::Array(arr.iter().map(json_to_toml).collect()),
            serde_json::Value::Object(map) => {
                let mut tbl = toml::map::Map::new();
                for (k,v) in map.iter() { tbl.insert(k.clone(), json_to_toml(v)); }
                toml::Value::Table(tbl)
            }
        }
    }
    let t = json_to_toml(&j);
    match toml::to_string_pretty(&t) {
        Ok(s) => Value::String(s),
        Err(e) => failure("IO::toml", &format!("TomlStringify: {}", e)),
    }
}

// ---------------- Bytes / Encoding ----------------
fn bytes_from_value(ev: &mut Evaluator, v: Value) -> Result<Vec<u8>, String> {
    match ev.eval(v) {
        Value::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                match it { Value::Integer(i) if i>=0 && i<=255 => out.push(i as u8), _ => return Err("Bytes must be list of 0..255".into()) }
            }
            Ok(out)
        }
        // Support Binary[base64url] fallback
        Value::Expr{ head, args } => {
            if let Value::Symbol(sym) = *head {
                if sym == "Binary" && !args.is_empty() {
                    if let Value::String(s) = &args[0] { return base64url_to_bytes(s).map_err(|e| format!("{}", e)); }
                }
            }
            Err("Expected bytes".into())
        }
        Value::String(s) => Ok(s.into_bytes()),
        _ => Err("Expected bytes".into()),
    }
}

fn bytes_to_value(bytes: &[u8]) -> Value { Value::List(bytes.iter().map(|b| Value::Integer(*b as i64)).collect()) }

fn base64_encode(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Base64Encode".into())), args } }
    let data = match bytes_from_value(ev, args[0].clone()) { Ok(b)=>b, Err(e)=> return failure("IO::bytes", &format!("Base64Encode: {}", e)) };
    use base64::Engine as _;
    let s = base64::engine::general_purpose::STANDARD.encode(data);
    Value::String(s)
}

fn base64_decode(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Base64Decode".into())), args } }
    let s = to_string_arg(ev, args[0].clone());
    use base64::Engine as _;
    match base64::engine::general_purpose::STANDARD.decode(s) { Ok(b)=> bytes_to_value(&b), Err(e)=> failure("IO::bytes", &format!("Base64Decode: {}", e)) }
}

fn hex_encode_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("HexEncode".into())), args } }
    let data = match bytes_from_value(ev, args[0].clone()) { Ok(b)=>b, Err(e)=> return failure("IO::bytes", &format!("HexEncode: {}", e)) };
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(data.len()*2);
    for b in data { out.push(HEX[(b>>4) as usize] as char); out.push(HEX[(b & 0x0f) as usize] as char); }
    Value::String(out)
}

fn from_hex(c: u8) -> Result<u8, String> { match c { b'0'..=b'9'=>Ok(c-b'0'), b'a'..=b'f'=>Ok(c-b'a'+10), b'A'..=b'F'=>Ok(c-b'A'+10), _=>Err("invalid hex".into()) } }

fn hex_decode_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("HexDecode".into())), args } }
    let s = to_string_arg(ev, args[0].clone());
    if s.len()%2 != 0 { return failure("IO::bytes", "HexDecode: hex length must be even"); }
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len()/2);
    for i in (0..bytes.len()).step_by(2) {
        let h = match from_hex(bytes[i]) { Ok(x)=>x, Err(e)=> return failure("IO::bytes", &format!("HexDecode: {}", e)) };
        let l = match from_hex(bytes[i+1]) { Ok(x)=>x, Err(e)=> return failure("IO::bytes", &format!("HexDecode: {}", e)) };
        out.push((h<<4)|l);
    }
    bytes_to_value(&out)
}

fn text_encode(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("TextEncode".into())), args } }
    let s = to_string_arg(ev, args[0].clone());
    // Currently only utf-8 supported
    bytes_to_value(s.as_bytes())
}

fn text_decode(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("TextDecode".into())), args } }
    let b = match bytes_from_value(ev, args[0].clone()) { Ok(x)=>x, Err(e)=> return failure("IO::bytes", &format!("TextDecode: {}", e)) };
    match String::from_utf8(b) { Ok(s)=> Value::String(s), Err(_)=> failure("IO::bytes", "TextDecode: invalid utf-8") }
}

fn bytes_concat(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("BytesConcat".into())), args } }
    let list_v = ev.eval(args[0].clone());
    if let Value::List(chunks) = list_v {
        let mut out: Vec<u8> = Vec::new();
        for c in chunks { match bytes_from_value(ev, c) { Ok(mut b)=> out.append(&mut b), Err(e)=> return failure("IO::bytes", &format!("BytesConcat: {}", e)) } }
        return bytes_to_value(&out);
    }
    failure("IO::bytes", "BytesConcat: expected list of byte arrays")
}

fn bytes_slice(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("BytesSlice".into())), args } }
    let data = match bytes_from_value(ev, args[0].clone()) { Ok(b)=>b, Err(e)=> return failure("IO::bytes", &format!("BytesSlice: {}", e)) };
    let start = match ev.eval(args[1].clone()) { Value::Integer(i) if i>=0 => i as usize, _ => 0 };
    let end = if args.len()>2 { match ev.eval(args[2].clone()) { Value::Integer(i) if i>=0 => (i as usize).min(data.len()), _ => data.len() } } else { data.len() };
    let s = if start <= end && start <= data.len() { &data[start..end] } else { &[] };
    bytes_to_value(s)
}

fn bytes_length(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("BytesLength".into())), args } }
    match bytes_from_value(ev, args[0].clone()) { Ok(b)=> Value::Integer(b.len() as i64), Err(e)=> failure("IO::bytes", &format!("BytesLength: {}", e)) }
}

fn base64url_to_bytes(s: &str) -> Result<Vec<u8>, String> {
    use base64::Engine as _;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(s).map_err(|e| e.to_string())
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
