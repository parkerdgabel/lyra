use lyra_core::pretty::format_value;
use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
use std::collections::HashMap;
use std::io::Write;
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

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Level {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}
impl Level {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "trace" => Some(Level::Trace),
            "debug" => Some(Level::Debug),
            "info" => Some(Level::Info),
            "warn" => Some(Level::Warn),
            "error" => Some(Level::Error),
            _ => None,
        }
    }
    fn as_str(&self) -> &'static str {
        match self {
            Level::Trace => "trace",
            Level::Debug => "debug",
            Level::Info => "info",
            Level::Warn => "warn",
            Level::Error => "error",
        }
    }
}

#[derive(Clone)]
struct LoggerConf {
    level: Level,
    #[allow(dead_code)]
    to_stderr: bool,
    json: bool,
    file: Option<String>,
    include_time: bool,
    include_span: bool,
}

static LOGGER: OnceLock<Mutex<LoggerConf>> = OnceLock::new();
fn logger() -> &'static Mutex<LoggerConf> {
    LOGGER.get_or_init(|| {
        Mutex::new(LoggerConf {
            level: Level::Info,
            to_stderr: true,
            json: false,
            file: None,
            include_time: true,
            include_span: false,
        })
    })
}

thread_local! { static LOG_CONTEXT: std::cell::RefCell<Vec<HashMap<String, Value>>> = std::cell::RefCell::new(Vec::new()); }

pub fn register_logging(ev: &mut Evaluator) {
    ev.register("ConfigureLogging", configure_logging as NativeFn, Attributes::empty());
    ev.register("Log", log_fn as NativeFn, Attributes::empty());
    ev.register("WithLogger", with_logger as NativeFn, Attributes::HOLD_ALL);
    ev.register("SetLogLevel", set_log_level as NativeFn, Attributes::empty());
    ev.register("GetLogger", get_logger as NativeFn, Attributes::empty());
}

pub fn register_logging_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    crate::register_if(
        ev,
        pred,
        "ConfigureLogging",
        configure_logging as NativeFn,
        Attributes::empty(),
    );
    crate::register_if(ev, pred, "Log", log_fn as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "WithLogger", with_logger as NativeFn, Attributes::HOLD_ALL);
    crate::register_if(ev, pred, "SetLogLevel", set_log_level as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "GetLogger", get_logger as NativeFn, Attributes::empty());
}

fn configure_logging(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("ConfigureLogging".into())), args };
    }
    let opts = ev.eval(args[0].clone());
    if let Value::Assoc(m) = opts {
        let mut conf = logger().lock().unwrap().clone();
        if let Some(Value::String(s)) | Some(Value::Symbol(s)) = m.get("Level") {
            if let Some(l) = Level::from_str(s) {
                conf.level = l;
            }
        }
        if let Some(Value::String(s)) | Some(Value::Symbol(s)) = m.get("Format") {
            conf.json = s.to_lowercase() == "json";
        }
        if let Some(Value::Assoc(file)) = m.get("File") {
            if let Some(Value::String(p)) | Some(Value::Symbol(p)) = file.get("Path") {
                conf.file = Some(p.clone());
            }
        }
        if let Some(Value::Boolean(b)) = m.get("IncludeTime") {
            conf.include_time = *b;
        }
        if let Some(Value::Boolean(b)) = m.get("IncludeSpan") {
            conf.include_span = *b;
        }
        // If File.Path set, try opening once to validate
        if let Some(path) = conf.file.clone() {
            if std::fs::OpenOptions::new().create(true).append(true).open(&path).is_err() {
                return failure("Log::config", &format!("Cannot open log file: {}", path));
            }
        }
        *logger().lock().unwrap() = conf;
        return Value::Boolean(true);
    }
    failure("Log::config", "Expected options assoc")
}

fn emit(level: Level, msg: String, meta: Option<HashMap<String, Value>>) -> Result<(), String> {
    let conf = logger().lock().unwrap().clone();
    if level < conf.level {
        return Ok(());
    }
    let ts = if conf.include_time { Some(crate_ts_ms()) } else { None };
    let ctx = LOG_CONTEXT.with(|c| c.borrow().iter().cloned().collect::<Vec<_>>());
    if conf.json {
        let mut obj: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
        obj.insert("level".into(), serde_json::Value::String(level.as_str().into()));
        if let Some(t) = ts {
            obj.insert("ts".into(), serde_json::Value::Number((t).into()));
        }
        obj.insert("msg".into(), serde_json::Value::String(msg));
        // merge context
        for m in ctx {
            for (k, v) in m {
                obj.insert(k, val_to_json(&v));
            }
        }
        if let Some(m) = meta {
            for (k, v) in m {
                obj.insert(k, val_to_json(&v));
            }
        }
        let line = serde_json::Value::Object(obj).to_string() + "\n";
        write_out(conf, line.as_bytes())
    } else {
        let mut parts: Vec<String> = Vec::new();
        if let Some(t) = ts {
            parts.push(format!("{}", t));
        }
        parts.push(level.as_str().to_string());
        parts.push(msg);
        if let Some(m) = meta {
            if !m.is_empty() {
                parts.push(format!("meta={}", format_value(&Value::Assoc(m))));
            }
        }
        let line = parts.join(" ") + "\n";
        write_out(conf, line.as_bytes())
    }
}

fn write_out(conf: LoggerConf, bytes: &[u8]) -> Result<(), String> {
    if let Some(path) = conf.file {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| e.to_string())?;
        f.write_all(bytes).map_err(|e| e.to_string())?;
        return Ok(());
    }
    let stderr = std::io::stderr();
    let mut h = stderr.lock();
    h.write_all(bytes).map_err(|e| e.to_string())
}

fn crate_ts_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn val_to_json(v: &Value) -> serde_json::Value {
    match v {
        Value::Integer(n) => serde_json::Value::Number((*n).into()),
        Value::Real(f) => serde_json::json!(*f),
        Value::String(s) => serde_json::Value::String(s.clone()),
        Value::Symbol(s) => serde_json::Value::String(s.clone()),
        Value::Boolean(b) => serde_json::Value::Bool(*b),
        Value::List(items) => serde_json::Value::Array(items.iter().map(val_to_json).collect()),
        Value::Assoc(m) => {
            let mut obj = serde_json::Map::new();
            for (k, vv) in m.iter() {
                obj.insert(k.clone(), val_to_json(vv));
            }
            serde_json::Value::Object(obj)
        }
        _ => serde_json::Value::String(format_value(v)),
    }
}

fn log_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("Log".into())), args };
    }
    let level_s = match ev.eval(args[0].clone()) {
        Value::String(s) | Value::Symbol(s) => s,
        other => format_value(&other),
    };
    let level = Level::from_str(&level_s).unwrap_or(Level::Info);
    let msg = match ev.eval(args[1].clone()) {
        Value::String(s) | Value::Symbol(s) => s,
        other => format_value(&other),
    };
    let meta = if args.len() >= 3 {
        match ev.eval(args[2].clone()) {
            Value::Assoc(m) => Some(m),
            _ => None,
        }
    } else {
        None
    };
    match emit(level, msg, meta) {
        Ok(_) => Value::Boolean(true),
        Err(e) => failure("Log::emit", &e),
    }
}

fn with_logger(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("WithLogger".into())), args };
    }
    let ctx = match ev.eval(args[0].clone()) {
        Value::Assoc(m) => m,
        _ => HashMap::new(),
    };
    LOG_CONTEXT.with(|c| c.borrow_mut().push(ctx));
    let out = ev.eval(args[1].clone());
    LOG_CONTEXT.with(|c| {
        let _ = c.borrow_mut().pop();
    });
    out
}

fn set_log_level(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("SetLogLevel".into())), args };
    }
    let s = match ev.eval(args[0].clone()) {
        Value::String(s) | Value::Symbol(s) => s,
        _ => String::new(),
    };
    if let Some(l) = Level::from_str(&s) {
        logger().lock().unwrap().level = l;
        return Value::Boolean(true);
    }
    failure("Log::config", "Unknown log level")
}

fn get_logger(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if !args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("GetLogger".into())), args };
    }
    let c = logger().lock().unwrap().clone();
    Value::Assoc(HashMap::from([
        ("Level".into(), Value::String(c.level.as_str().into())),
        ("Format".into(), Value::String(if c.json { "json" } else { "text" }.into())),
        ("To".into(), Value::String(if c.file.is_some() { "file" } else { "stderr" }.into())),
        ("IncludeTime".into(), Value::Boolean(c.include_time)),
        ("IncludeSpan".into(), Value::Boolean(c.include_span)),
    ]))
}
