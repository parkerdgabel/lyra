use lyra_core::value::Value;
#[cfg(feature = "tools")]
use crate::tool_spec;
#[cfg(feature = "tools")]
use crate::tools::add_specs;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::process::{Command, Stdio};
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

pub fn register_process(ev: &mut Evaluator) {
    ev.register("Run", run as NativeFn, Attributes::empty());
    ev.register("Which", which as NativeFn, Attributes::empty());
    ev.register("CommandExistsQ", command_exists_q as NativeFn, Attributes::empty());
    ev.register("Popen", popen as NativeFn, Attributes::empty());
    ev.register("WriteProcess", write_process as NativeFn, Attributes::empty());
    ev.register("ReadProcess", read_process as NativeFn, Attributes::empty());
    ev.register("WaitProcess", wait_process as NativeFn, Attributes::empty());
    ev.register("KillProcess", kill_process as NativeFn, Attributes::empty());
    ev.register("Pipe", pipe as NativeFn, Attributes::empty());
    ev.register("ProcessInfo", process_info as NativeFn, Attributes::empty());

    #[cfg(feature = "tools")]
    add_specs(vec![
        tool_spec!(
            "Run",
            summary: "Run a process and capture output",
            params: ["cmd","args?","opts?"],
            tags: ["process","proc","os"],
            examples: [ Value::String("Run[\"echo\", {\"hi\"}]".into()) ]
        ),
        tool_spec!(
            "Which",
            summary: "Resolve command path from PATH",
            params: ["cmd"],
            tags: ["process","proc","os"]
        ),
        tool_spec!(
            "CommandExistsQ",
            summary: "Does a command exist in PATH?",
            params: ["cmd"],
            tags: ["process","proc","os"]
        ),
        tool_spec!(
            "Popen",
            summary: "Spawn process and return handle",
            params: ["cmd","args?","opts?"],
            tags: ["process","proc","os"]
        ),
        tool_spec!(
            "WriteProcess",
            summary: "Write to process stdin",
            params: ["proc","data"],
            tags: ["process","proc","os"]
        ),
        tool_spec!(
            "ReadProcess",
            summary: "Read from process stdout/stderr",
            params: ["proc","opts?"],
            tags: ["process","proc","os"]
        ),
        tool_spec!(
            "WaitProcess",
            summary: "Wait for process to exit",
            params: ["proc"],
            tags: ["process","proc","os"]
        ),
        tool_spec!(
            "KillProcess",
            summary: "Send signal to process",
            params: ["proc","signal?"],
            tags: ["process","proc","os"]
        ),
        tool_spec!(
            "Pipe",
            summary: "Compose processes via pipes",
            params: ["cmds"],
            tags: ["process","proc","os"]
        ),
        tool_spec!(
            "ProcessInfo",
            summary: "Inspect process handle (pid, running, exit)",
            params: ["proc"],
            tags: ["process","proc","introspect"],
            examples: [ Value::String("p := Popen[\"sleep\", {\"0.1\"}]; ProcessInfo[p]".into()) ]
        ),
    ]);
}

pub fn register_process_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    crate::register_if(ev, pred, "Run", run as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "Which", which as NativeFn, Attributes::empty());
    crate::register_if(
        ev,
        pred,
        "CommandExistsQ",
        command_exists_q as NativeFn,
        Attributes::empty(),
    );
    crate::register_if(ev, pred, "Popen", popen as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "WriteProcess", write_process as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "ReadProcess", read_process as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "WaitProcess", wait_process as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "KillProcess", kill_process as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "Pipe", pipe as NativeFn, Attributes::empty());
    crate::register_if(ev, pred, "ProcessInfo", process_info as NativeFn, Attributes::empty());
}

fn to_string_arg(ev: &mut Evaluator, v: Value) -> String {
    match ev.eval(v) {
        Value::String(s) | Value::Symbol(s) => s,
        other => lyra_core::pretty::format_value(&other),
    }
}

fn run(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("Run".into())), args };
    }
    let cmd = to_string_arg(ev, args[0].clone());
    let argv: Vec<String> = match args.get(1).map(|v| ev.eval(v.clone())) {
        Some(Value::List(vs)) => vs.into_iter().map(|v| to_string_arg(ev, v)).collect(),
        _ => Vec::new(),
    };
    let opts = args.get(2).map(|v| ev.eval(v.clone()));
    let mut c = Command::new(&cmd);
    c.args(&argv);
    c.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut input: Option<Vec<u8>> = None;
    if let Some(Value::Assoc(m)) = opts.as_ref() {
        if let Some(Value::String(dir)) | Some(Value::Symbol(dir)) = m.get("Cwd") {
            c.current_dir(dir);
        }
        if let Some(Value::Assoc(env)) = m.get("Env") {
            for (k, v) in env {
                let vs = match v {
                    Value::String(s) | Value::Symbol(s) => s.clone(),
                    _ => lyra_core::pretty::format_value(v),
                };
                c.env(k, vs);
            }
        }
        if let Some(val) = m.get("Input") {
            match val {
                Value::String(s) => input = Some(s.clone().into_bytes()),
                _ => {}
            }
        }
    }
    let timeout_ms = match opts.as_ref().and_then(|v| {
        if let Value::Assoc(m) = v {
            m.get("TimeoutMs")
        } else {
            None
        }
    }) {
        Some(Value::Integer(n)) if *n > 0 => Some(*n as u64),
        _ => None,
    };
    let mut child = match c.spawn() {
        Ok(ch) => ch,
        Err(e) => return failure("Process::run", &format!("spawn: {}", e)),
    };
    if let Some(data) = input.take() {
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(&data);
        }
    }
    let start = std::time::Instant::now();
    let (status, out, err) = (|| {
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        if let Some(mut s) = child.stdout.take() {
            let _ = s.read_to_end(&mut stdout);
        }
        if let Some(mut s) = child.stderr.take() {
            let _ = s.read_to_end(&mut stderr);
        }
        let status = child.wait();
        (status, stdout, stderr)
    })();
    let mut timed_out = false;
    if let Some(ms) = timeout_ms {
        if start.elapsed().as_millis() as u64 > ms {
            timed_out = true;
        }
    }
    match status {
        Ok(s) => Value::Assoc(HashMap::from([
            ("Status".into(), Value::Integer(s.code().unwrap_or(-1) as i64)),
            ("Stdout".into(), Value::String(String::from_utf8_lossy(&out).to_string())),
            ("Stderr".into(), Value::String(String::from_utf8_lossy(&err).to_string())),
            ("TimedOut".into(), Value::Boolean(timed_out)),
        ])),
        Err(e) => failure("Process::run", &format!("wait: {}", e)),
    }
}

fn which(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("Which".into())), args };
    }
    let name = match &args[0] {
        Value::String(s) | Value::Symbol(s) => s.clone(),
        _ => return Value::Symbol("Null".into()),
    };
    let path_var = std::env::var_os("PATH");
    if let Some(paths) = path_var {
        for dir in std::env::split_paths(&paths) {
            let p = dir.join(&name);
            if p.exists() {
                return Value::String(p.to_string_lossy().to_string());
            }
        }
    }
    Value::Symbol("Null".into())
}

fn command_exists_q(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [x] => match ev.eval(x.clone()) {
            Value::String(s) | Value::Symbol(s) => {
                let v = which(ev, vec![Value::String(s)]);
                Value::Boolean(!matches!(v, Value::Symbol(ref n) if n=="Null"))
            }
            _ => Value::Boolean(false),
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("CommandExistsQ".into())), args },
    }
}

#[derive(Default)]
struct ProcState {
    child: Option<std::process::Child>,
    #[allow(dead_code)]
    id: i64,
}
static PROC_REG: OnceLock<Mutex<HashMap<i64, ProcState>>> = OnceLock::new();
static NEXT_PROC_ID: OnceLock<std::sync::atomic::AtomicI64> = OnceLock::new();
fn preg() -> &'static Mutex<HashMap<i64, ProcState>> {
    PROC_REG.get_or_init(|| Mutex::new(HashMap::new()))
}
fn next_id() -> i64 {
    let a = NEXT_PROC_ID.get_or_init(|| std::sync::atomic::AtomicI64::new(1));
    a.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}
fn phandle(id: i64, pid: i32) -> Value {
    Value::Assoc(HashMap::from([
        ("__type".into(), Value::String("Process".into())),
        ("id".into(), Value::Integer(id)),
        ("pid".into(), Value::Integer(pid as i64)),
    ]))
}
fn get_pid(v: &Value) -> Option<i64> {
    if let Value::Assoc(m) = v {
        if m.get("__type") == Some(&Value::String("Process".into())) {
            if let Some(Value::Integer(id)) = m.get("id") {
                return Some(*id);
            }
        }
    }
    None
}

fn popen(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("Popen".into())), args };
    }
    let cmd = to_string_arg(ev, args[0].clone());
    let argv: Vec<String> = match args.get(1).map(|v| ev.eval(v.clone())) {
        Some(Value::List(vs)) => vs.into_iter().map(|v| to_string_arg(ev, v)).collect(),
        _ => Vec::new(),
    };
    let opts = args.get(2).map(|v| ev.eval(v.clone()));
    let mut c = Command::new(&cmd);
    c.args(&argv);
    c.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    if let Some(Value::Assoc(m)) = opts.as_ref() {
        if let Some(Value::String(dir)) | Some(Value::Symbol(dir)) = m.get("Cwd") {
            c.current_dir(dir);
        }
        if let Some(Value::Assoc(env)) = m.get("Env") {
            for (k, v) in env {
                let vs = match v {
                    Value::String(s) | Value::Symbol(s) => s.clone(),
                    _ => lyra_core::pretty::format_value(v),
                };
                c.env(k, vs);
            }
        }
    }
    match c.spawn() {
        Ok(ch) => {
            let id = next_id();
            let pid = ch.id() as i32;
            preg().lock().unwrap().insert(id, ProcState { child: Some(ch), id });
            phandle(id, pid)
        }
        Err(e) => failure("Process::spawn", &e.to_string()),
    }
}

fn process_info(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("ProcessInfo".into())), args };
    }
    let a0 = &args[0];
    let id = match get_pid(a0) { Some(i) => i, None => return Value::Expr { head: Box::new(Value::Symbol("ProcessInfo".into())), args: vec![a0.clone()] } };
    let mut reg = preg().lock().unwrap();
    if let Some(st) = reg.get_mut(&id) {
        let mut running = false;
        let mut exit_code: Option<i64> = None;
        if let Some(ch) = st.child.as_mut() {
            match ch.try_wait() {
                Ok(Some(status)) => {
                    running = false;
                    exit_code = status.code().map(|c| c as i64);
                }
                Ok(None) => { running = true; }
                Err(_) => {}
            }
        }
        let pid = match a0 {
            Value::Assoc(m) => match m.get("pid") { Some(Value::Integer(p)) => *p, _ => -1 },
            _ => -1,
        };
        let mut m: HashMap<String, Value> = HashMap::new();
        m.insert("Type".into(), Value::String("Process".into()));
        m.insert("Id".into(), Value::Integer(id));
        m.insert("Pid".into(), Value::Integer(pid));
        m.insert("Running".into(), Value::Boolean(running));
        match exit_code {
            Some(c) => { m.insert("ExitCode".into(), Value::Integer(c)); },
            None => {},
        }
        return Value::Assoc(m);
    }
    Value::Expr { head: Box::new(Value::Symbol("ProcessInfo".into())), args: vec![a0.clone()] }
}

fn write_process(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("WriteProcess".into())), args };
    }
    if let Some(id) = get_pid(&args[0]) {
        let mut reg = preg().lock().unwrap();
        if let Some(st) = reg.get_mut(&id) {
            if let Some(ch) = st.child.as_mut() {
                if let Some(stdin) = ch.stdin.as_mut() {
                    let data = match ev.eval(args[1].clone()) {
                        Value::String(s) => s.into_bytes(),
                        _ => Vec::new(),
                    };
                    if let Err(e) = stdin.write_all(&data) {
                        return failure("Process::io", &e.to_string());
                    }
                    return Value::Integer(data.len() as i64);
                }
            }
        }
    }
    failure("Process::io", "Invalid process handle or no stdin")
}

fn read_process(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("ReadProcess".into())), args };
    }
    let stream = args
        .get(1)
        .and_then(|v| {
            if let Value::Assoc(m) = ev.eval(v.clone()) {
                m.get("Stream").and_then(|x| {
                    if let Value::String(s) | Value::Symbol(s) = x {
                        Some(s.clone())
                    } else {
                        None
                    }
                })
            } else {
                None
            }
        })
        .unwrap_or_else(|| "stdout".into());
    if let Some(id) = get_pid(&args[0]) {
        let mut reg = preg().lock().unwrap();
        if let Some(st) = reg.get_mut(&id) {
            if let Some(ch) = st.child.as_mut() {
                let mut buf = Vec::new();
                match stream.as_str() {
                    "stderr" => {
                        if let Some(s) = ch.stderr.as_mut() {
                            let _ = s.read_to_end(&mut buf);
                        }
                    }
                    _ => {
                        if let Some(s) = ch.stdout.as_mut() {
                            let _ = s.read_to_end(&mut buf);
                        }
                    }
                }
                return Value::String(String::from_utf8_lossy(&buf).to_string());
            }
        }
    }
    failure("Process::io", "Invalid process handle")
}

fn wait_process(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("WaitProcess".into())), args };
    }
    let timeout_ms =
        args.get(1).and_then(|v| if let Value::Integer(n) = v { Some(*n as u64) } else { None });
    if let Some(id) = get_pid(&args[0]) {
        let start = std::time::Instant::now();
        loop {
            let done = {
                let mut reg = preg().lock().unwrap();
                if let Some(st) = reg.get_mut(&id) {
                    if let Some(ch) = st.child.as_mut() {
                        if let Ok(Some(status)) = ch.try_wait() {
                            return Value::Assoc(HashMap::from([(
                                "Status".into(),
                                Value::Integer(status.code().unwrap_or(-1) as i64),
                            )]));
                        }
                    }
                }
                false
            };
            if done {
                break;
            }
            if let Some(ms) = timeout_ms {
                if start.elapsed().as_millis() as u64 >= ms {
                    return failure("Process::timeout", "WaitProcess timed out");
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }
    failure("Process::io", "Invalid process handle")
}

fn kill_process(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("KillProcess".into())), args };
    }
    if let Some(id) = get_pid(&args[0]) {
        let mut reg = preg().lock().unwrap();
        if let Some(st) = reg.get_mut(&id) {
            if let Some(ch) = st.child.as_mut() {
                return match ch.kill() {
                    Ok(_) => Value::Boolean(true),
                    Err(e) => failure("Process::signal", &e.to_string()),
                };
            }
        }
    }
    failure("Process::signal", "Invalid process handle")
}

fn pipe(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("Pipe".into())), args };
    }
    let pipeline_v = ev.eval(args[0].clone());
    let steps: Vec<Vec<String>> = match pipeline_v {
        Value::List(items) => items
            .into_iter()
            .filter_map(|it| match it {
                Value::List(xs) => Some(xs.into_iter().map(|v| to_string_arg(ev, v)).collect()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    };
    if steps.is_empty() {
        return failure("Process::run", "Empty pipeline");
    }
    // Build pipeline
    let mut children: Vec<std::process::Child> = Vec::new();
    let mut prev_stdout: Option<std::process::ChildStdout> = None;
    for (i, step) in steps.iter().enumerate() {
        let (cmd, argsv) = (&step[0], &step[1..]);
        let mut c = Command::new(cmd);
        c.args(argsv);
        if i == 0 {
            c.stdin(Stdio::inherit());
        } else {
            if let Some(out) = prev_stdout.take() {
                c.stdin(Stdio::from(out));
            }
        }
        if i == steps.len() - 1 {
            c.stdout(Stdio::piped());
        } else {
            c.stdout(Stdio::piped());
        }
        c.stderr(Stdio::piped());
        let mut ch = match c.spawn() {
            Ok(x) => x,
            Err(e) => return failure("Process::run", &format!("spawn: {}", e)),
        };
        prev_stdout = ch.stdout.take();
        children.push(ch);
    }
    // Read final stdout and stderr of last child
    let mut last = children.pop().unwrap();
    let mut out = Vec::new();
    let mut err = Vec::new();
    if let Some(mut s) = last.stdout.take() {
        let _ = s.read_to_end(&mut out);
    }
    if let Some(mut s) = last.stderr.take() {
        let _ = s.read_to_end(&mut err);
    }
    let status = last.wait();
    // wait others
    for mut ch in children {
        let _ = ch.wait();
    }
    match status {
        Ok(s) => Value::Assoc(HashMap::from([
            ("Status".into(), Value::Integer(s.code().unwrap_or(-1) as i64)),
            ("Stdout".into(), Value::String(String::from_utf8_lossy(&out).to_string())),
            ("Stderr".into(), Value::String(String::from_utf8_lossy(&err).to_string())),
            ("TimedOut".into(), Value::Boolean(false)),
        ])),
        Err(e) => failure("Process::run", &e.to_string()),
    }
}
