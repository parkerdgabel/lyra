use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use std::fs;

pub fn caret_line(text: &str, col: usize) -> String {
    let mut out = String::new();
    let mut c = 1usize;
    for ch in text.chars() {
        if c >= col {
            break;
        }
        match ch {
            '\t' => out.push('\t'),
            _ => out.push(' '),
        }
        c += 1;
    }
    out.push('^');
    out
}

pub fn spawn_pager(text: &str) -> std::io::Result<()> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let pager = std::env::var("PAGER").unwrap_or_else(|_| "less -R".to_string());
    let mut parts = pager.split_whitespace();
    let cmd = parts.next().unwrap_or("less");
    let args: Vec<&str> = parts.collect();
    let mut child = Command::new(cmd).args(&args).stdin(Stdio::piped()).spawn()?;
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(text.as_bytes());
    }
    let _ = child.wait();
    Ok(())
}

pub fn save_env_json(ev: &mut Evaluator, path: &str) -> anyhow::Result<()> {
    use std::collections::HashMap;
    let mut map: HashMap<String, Value> = HashMap::new();
    for name in ev.env_keys() {
        let val = ev.eval(Value::Symbol(name.clone()));
        map.insert(name, val);
    }
    let s = serde_json::to_string_pretty(&Value::Assoc(map))?;
    fs::write(path, s)?;
    Ok(())
}

pub fn load_env_json(ev: &mut Evaluator, path: &str) -> anyhow::Result<()> {
    let s = fs::read_to_string(path)?;
    let v: Value = serde_json::from_str(&s)?;
    if let Value::Assoc(m) = v {
        for (k, val) in m.into_iter() {
            let _ = ev.eval(Value::Expr { head: Box::new(Value::Symbol("Set".into())), args: vec![Value::Symbol(k), val] });
        }
    }
    Ok(())
}

pub fn collect_pkg_exports(ev: &mut Evaluator) -> std::collections::HashMap<String, Vec<String>> {
    use std::collections::HashMap;
    let mut out: HashMap<String, Vec<String>> = HashMap::new();
    let loaded = ev.eval(Value::Expr { head: Box::new(Value::Symbol("LoadedPackages".into())), args: vec![] });
    if let Value::Assoc(m) = loaded {
        for name in m.keys() {
            let q = Value::Expr { head: Box::new(Value::Symbol("PackageExports".into())), args: vec![Value::String(name.clone())] };
            let ex = ev.eval(q);
            if let Value::List(vs) = ex {
                let syms: Vec<String> = vs.into_iter().filter_map(|v| if let Value::String(s) = v { Some(s) } else { None }).collect();
                out.insert(name.clone(), syms);
            }
        }
    }
    out
}

