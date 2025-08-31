use crate::register_if;
use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
use std::collections::HashMap;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

fn failure(tag: &str, msg: &str) -> Value {
    Value::Assoc(HashMap::from([
        ("message".into(), Value::String("Failure".into())),
        ("tag".into(), Value::String(tag.into())),
        ("detail".into(), Value::String(msg.into())),
    ]))
}

fn as_str(v: &Value) -> Option<String> {
    match v {
        Value::String(s) | Value::Symbol(s) => Some(s.clone()),
        _ => None,
    }
}

fn fs_exists(p: &str) -> bool {
    std::path::Path::new(p).exists()
}

// --- Formatter (naive AST roundtrip)

fn format_text(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("FormatLyraText".into())), args };
    }
    let src = match &args[0] {
        Value::String(s) => s.clone(),
        other => lyra_core::pretty::format_value(other),
    };
    let mut p = lyra_parser::Parser::from_source(&src);
    let exprs = match p.parse_all() {
        Ok(vs) => vs,
        Err(e) => return failure("FormatLyra::parse", &format!("{}", e)),
    };
    let mut out = String::new();
    for (i, e) in exprs.iter().enumerate() {
        if i > 0 {
            out.push_str("\n");
        }
        out.push_str(&lyra_core::pretty::format_value(e));
    }
    Value::String(out)
}

fn format_file(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("FormatLyraFile".into())), args };
    }
    let path = match &args[0] {
        Value::String(s) => s.clone(),
        Value::Symbol(s) => s.clone(),
        _ => return failure("FormatLyraFile::arg", "expected path string"),
    };
    let content = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => return failure("FormatLyraFile::read", &e.to_string()),
    };
    let mut p = lyra_parser::Parser::from_source(&content);
    let exprs = match p.parse_all() {
        Ok(vs) => vs,
        Err(e) => return failure("FormatLyraFile::parse", &format!("{}", e)),
    };
    let mut out = String::new();
    for (i, e) in exprs.iter().enumerate() {
        if i > 0 {
            out.push_str("\n");
        }
        out.push_str(&lyra_core::pretty::format_value(e));
    }
    let changed = if out != content {
        match std::fs::write(&path, &out) {
            Ok(_) => true,
            Err(_) => false,
        }
    } else {
        false
    };
    Value::Assoc(HashMap::from([
        ("path".into(), Value::String(path)),
        ("changed".into(), Value::Boolean(changed)),
    ]))
}

fn format_any(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("FormatLyra".into())), args };
    }
    // Accept: text, path, or list of paths
    let a0 = ev.eval(args[0].clone());
    match a0 {
        Value::String(s) => {
            if fs_exists(&s) {
                format_file(ev, vec![Value::String(s)])
            } else {
                format_text(ev, vec![Value::String(s)])
            }
        }
        Value::List(vs) => {
            let mut out: Vec<Value> = Vec::new();
            for v in vs {
                if let Some(p) = as_str(&v) {
                    out.push(format_file(ev, vec![Value::String(p)]));
                }
            }
            Value::List(out)
        }
        other => format_text(ev, vec![other]),
    }
}

// --- Linter (basic style + parseability)
fn lint_text(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("LintLyraText".into())), args };
    }
    let src = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return failure("LintLyraText::arg", "expected string"),
    };
    let mut issues: Vec<Value> = Vec::new();
    // Parseability
    let mut p = lyra_parser::Parser::from_source(&src);
    if let Err(e) = p.parse_all() {
        issues.push(Value::Assoc(HashMap::from([
            ("rule".into(), Value::String("parse-error".into())),
            ("message".into(), Value::String(format!("{}", e))),
        ])));
    }
    // Style: tabs, trailing ws, long lines
    for (i, line) in src.lines().enumerate() {
        if line.contains('\t') {
            issues.push(Value::Assoc(HashMap::from([
                ("rule".into(), Value::String("no-tabs".into())),
                ("line".into(), Value::Integer((i + 1) as i64)),
                ("message".into(), Value::String("tab character found".into())),
            ])));
        }
        if line.ends_with(' ') || line.ends_with('\t') {
            issues.push(Value::Assoc(HashMap::from([
                ("rule".into(), Value::String("no-trailing-whitespace".into())),
                ("line".into(), Value::Integer((i + 1) as i64)),
                ("message".into(), Value::String("trailing whitespace".into())),
            ])));
        }
        if line.chars().count() > 120 {
            issues.push(Value::Assoc(HashMap::from([
                ("rule".into(), Value::String("max-line-length".into())),
                ("line".into(), Value::Integer((i + 1) as i64)),
                ("message".into(), Value::String("line exceeds 120 chars".into())),
            ])));
        }
    }
    // Semantic pass (parsed)
    let mut p2 = lyra_parser::Parser::from_source(&src);
    if let Ok(exprs) = p2.parse_all() {
        for e in exprs {
            collect_semantic_issues(&e, "", &mut issues);
        }
    }
    Value::List(issues)
}

fn lint_file(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("LintLyraFile".into())), args };
    }
    let path = match &args[0] {
        Value::String(s) | Value::Symbol(s) => s.clone(),
        _ => return failure("LintLyraFile::arg", "expected path string"),
    };
    let content = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => return failure("LintLyraFile::read", &e.to_string()),
    };
    let issues = lint_text(ev, vec![Value::String(content)]);
    Value::Assoc(HashMap::from([("path".into(), Value::String(path)), ("issues".into(), issues)]))
}

fn lint_any(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("LintLyra".into())), args };
    }
    let a0 = ev.eval(args[0].clone());
    match a0 {
        Value::String(s) => {
            if fs_exists(&s) {
                lint_file(ev, vec![Value::String(s)])
            } else {
                lint_text(ev, vec![Value::String(s)])
            }
        }
        Value::List(vs) => {
            let mut out: Vec<Value> = Vec::new();
            for v in vs {
                if let Some(p) = as_str(&v) {
                    out.push(lint_file(ev, vec![Value::String(p)]));
                }
            }
            Value::List(out)
        }
        other => lint_text(ev, vec![other]),
    }
}

fn collect_semantic_issues(v: &Value, path: &str, out: &mut Vec<Value>) {
    use Value::*;
    match v {
        Expr { head, args } => {
            let hname = if let Symbol(s) = &**head {
                s.as_str()
            } else if let String(s) = &**head {
                s.as_str()
            } else {
                ""
            };
            // Deprecated names (prefer canonical forms)
            let mut deprecated: Option<(&str, &str)> = None;
            match hname {
                "AssocGet" => deprecated = Some(("AssocGet", "Get")),
                "Lookup" => deprecated = Some(("Lookup", "Get")),
                "AssocContainsKeyQ" => deprecated = Some(("AssocContainsKeyQ", "ContainsKeyQ")),
                "StringLength" => deprecated = Some(("StringLength", "Length")),
                "StringSplit" => deprecated = Some(("StringSplit", "Split")),
                "SetUnion" => deprecated = Some(("SetUnion", "Union")),
                "SetIntersection" => deprecated = Some(("SetIntersection", "Intersection")),
                "SetDifference" => deprecated = Some(("SetDifference", "Difference")),
                "ListUnion" => deprecated = Some(("ListUnion", "Union")),
                "ListIntersection" => deprecated = Some(("ListIntersection", "Intersection")),
                "ListDifference" => deprecated = Some(("ListDifference", "Difference")),
                "HttpServer" => deprecated = Some(("HttpServer", "HttpServe/HttpServeRoutes")),
                _ => {}
            }
            if let Some((old, newn)) = deprecated {
                out.push(Value::Assoc(HashMap::from([
                    ("rule".into(), Value::String("deprecated-name".into())),
                    ("path".into(), Value::String(path.into())),
                    ("message".into(), Value::String(format!("Use {} instead of {}", newn, old))),
                ])));
            }
            // Equal[x,x]
            if hname == "Equal" && args.len() == 2 && args[0] == args[1] {
                out.push(Value::Assoc(HashMap::from([
                    ("rule".into(), Value::String("equal-self".into())),
                    ("path".into(), Value::String(path.into())),
                    ("message".into(), Value::String("comparison to self is always true".into())),
                ])));
            }
            // Redundant ops
            if hname == "Plus" {
                for a in args {
                    let is_zero = match a {
                        Integer(0) => true,
                        Real(f) if *f == 0.0 => true,
                        _ => false,
                    };
                    if is_zero {
                        out.push(Value::Assoc(HashMap::from([
                            ("rule".into(), Value::String("redundant-plus-zero".into())),
                            ("path".into(), Value::String(path.into())),
                            ("message".into(), Value::String("adding zero has no effect".into())),
                        ])));
                        break;
                    }
                }
            }
            if hname == "Times" {
                for a in args {
                    let is_one = match a {
                        Integer(1) => true,
                        Real(f) if *f == 1.0 => true,
                        _ => false,
                    };
                    if is_one {
                        out.push(Value::Assoc(HashMap::from([
                            ("rule".into(), Value::String("redundant-times-one".into())),
                            ("path".into(), Value::String(path.into())),
                            (
                                "message".into(),
                                Value::String("multiplying by 1 has no effect".into()),
                            ),
                        ])));
                        break;
                    }
                }
            }
            if hname == "Power" && args.len() == 2 {
                if matches!(args[1], Integer(1)) {
                    out.push(Value::Assoc(HashMap::from([
                        ("rule".into(), Value::String("redundant-power-one".into())),
                        ("path".into(), Value::String(path.into())),
                        ("message".into(), Value::String("x^1 equals x".into())),
                    ])));
                }
                if matches!(args[1], Integer(0)) {
                    out.push(Value::Assoc(HashMap::from([
                        ("rule".into(), Value::String("power-zero-const".into())),
                        ("path".into(), Value::String(path.into())),
                        ("message".into(), Value::String("x^0 equals 1".into())),
                    ])));
                }
            }
            if hname == "Map" && args.len() >= 2 {
                if let List(xs) = &args[1] {
                    if xs.is_empty() {
                        out.push(Value::Assoc(HashMap::from([
                            ("rule".into(), Value::String("map-empty".into())),
                            ("path".into(), Value::String(path.into())),
                            (
                                "message".into(),
                                Value::String("mapping over empty list does nothing".into()),
                            ),
                        ])));
                    }
                }
            }
            // Recurse
            for (i, a) in args.iter().enumerate() {
                let sub =
                    if path.is_empty() { format!("[{}]", i) } else { format!("{}[{}]", path, i) };
                collect_semantic_issues(a, &sub, out);
            }
        }
        List(xs) => {
            for (i, a) in xs.iter().enumerate() {
                let sub =
                    if path.is_empty() { format!("[{}]", i) } else { format!("{}[{}]", path, i) };
                collect_semantic_issues(a, &sub, out);
            }
        }
        Assoc(m) => {
            // Option key casing: prefer lowerCamelCase for known keys
            let key_map: HashMap<&str, &str> = HashMap::from([
                ("TimeoutMs", "timeoutMs"),
                ("MaxThreads", "maxThreads"),
                ("TimeBudgetMs", "timeBudgetMs"),
                ("Port", "port"),
                ("Host", "host"),
                ("ReadTimeoutMs", "readTimeoutMs"),
                ("WriteTimeoutMs", "writeTimeoutMs"),
                ("Headers", "headers"),
                ("FollowRedirects", "followRedirects"),
                ("DisableTlsVerify", "disableTlsVerify"),
                ("MaxBodyBytes", "maxBodyBytes"),
                ("Query", "query"),
                ("Form", "form"),
                ("Json", "json"),
                ("Multipart", "multipart"),
                ("As", "as"),
                ("Cookies", "cookies"),
            ]);
            for (k, a) in m.iter() {
                if let Some(newk) = key_map.get(k.as_str()) {
                    let msg = format!("Use lowerCamelCase option key: {}", newk);
                    out.push(Value::Assoc(HashMap::from([
                        ("rule".into(), Value::String("option-casing".into())),
                        ("path".into(), Value::String(path.into())),
                        ("message".into(), Value::String(msg)),
                    ])));
                }
                let sub =
                    if path.is_empty() { format!(".{}", k) } else { format!("{}.{}", path, k) };
                collect_semantic_issues(a, &sub, out);
            }
        }
        _ => {}
    }
}

// --- Config and Secrets
fn config_load(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // ConfigLoad[<|"Env"->{k..}, "DotEnv"->True, "Files"->{...}, "Overrides"-><|...|>|>]
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("ConfigLoad".into())), args };
    }
    let opts = match ev.eval(args[0].clone()) {
        Value::Assoc(m) => m,
        _ => HashMap::new(),
    };
    let mut out: HashMap<String, Value> = HashMap::new();
    // Env
    if let Some(Value::List(keys)) = opts.get("Env") {
        for k in keys {
            if let Some(name) = as_str(k) {
                if let Ok(v) = std::env::var(&name) {
                    out.insert(name, Value::String(v));
                }
            }
        }
    }
    // DotEnv
    let want_dotenv = match opts.get("DotEnv") {
        Some(Value::Boolean(b)) => *b,
        _ => false,
    };
    if want_dotenv {
        if let Ok(s) = std::fs::read_to_string(".env") {
            for line in s.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                if let Some((k, v)) = line.split_once('=') {
                    out.insert(k.trim().into(), Value::String(v.trim().trim_matches('"').into()));
                }
            }
        }
    }
    // Files
    if let Some(Value::List(paths)) = opts.get("Files") {
        for p in paths {
            if let Some(path) = as_str(p) {
                if let Ok(s) = std::fs::read_to_string(&path) {
                    let v = if path.ends_with(".json") {
                        serde_json::from_str::<serde_json::Value>(&s).ok().map(json_to_value)
                    } else if path.ends_with(".yaml") || path.ends_with(".yml") {
                        serde_yaml::from_str::<serde_json::Value>(&s).ok().map(json_to_value)
                    } else if path.ends_with(".toml") {
                        toml::from_str::<toml::Value>(&s).ok().map(toml_to_value)
                    } else {
                        None
                    };
                    if let Some(Value::Assoc(m)) = v {
                        for (k, vv) in m {
                            out.insert(k, vv);
                        }
                    }
                }
            }
        }
    }
    // Overrides
    if let Some(Value::Assoc(ovr)) = opts.get("Overrides") {
        for (k, v) in ovr {
            out.insert(k.clone(), v.clone());
        }
    }
    Value::Assoc(out)
}

fn json_to_value(j: serde_json::Value) -> Value {
    match j {
        serde_json::Value::Null => Value::Symbol("Null".into()),
        serde_json::Value::Bool(b) => Value::Boolean(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Integer(i)
            } else if let Some(f) = n.as_f64() {
                Value::Real(f)
            } else {
                Value::Real(0.0)
            }
        }
        serde_json::Value::String(s) => Value::String(s),
        serde_json::Value::Array(a) => Value::List(a.into_iter().map(json_to_value).collect()),
        serde_json::Value::Object(m) => {
            Value::Assoc(m.into_iter().map(|(k, v)| (k, json_to_value(v))).collect())
        }
    }
}
fn toml_to_value(t: toml::Value) -> Value {
    match t {
        toml::Value::Boolean(b) => Value::Boolean(b),
        toml::Value::Integer(i) => Value::Integer(i as i64),
        toml::Value::Float(f) => Value::Real(f),
        toml::Value::String(s) => Value::String(s),
        toml::Value::Array(a) => Value::List(a.into_iter().map(toml_to_value).collect()),
        toml::Value::Table(m) => {
            Value::Assoc(m.into_iter().map(|(k, v)| (k, toml_to_value(v))).collect())
        }
        _ => Value::Symbol("Null".into()),
    }
}

fn secrets_get(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // SecretsGet[key, provider] where provider: <|"Env"->True|> or <|"File"->path|>
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("SecretsGet".into())), args };
    }
    let key = match as_str(&ev.eval(args[0].clone())) {
        Some(s) => s,
        None => return failure("SecretsGet::key", "expected string"),
    };
    let provider = match ev.eval(args[1].clone()) {
        Value::Assoc(m) => m,
        _ => HashMap::new(),
    };
    if let Some(Value::Boolean(true)) = provider.get("Env") {
        if let Ok(v) = std::env::var(&key) {
            return Value::String(v);
        }
        return failure("SecretsGet::env", &format!("{} not set", key));
    }
    if let Some(Value::String(path)) = provider.get("File") {
        match std::fs::read_to_string(path) {
            Ok(s) => return Value::String(s.trim().into()),
            Err(e) => return failure("SecretsGet::file", &e.to_string()),
        }
    }
    failure("SecretsGet::provider", "unsupported provider")
}

// --- Release helpers
fn version_bump(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // VersionBump[level, paths] where level in {"major","minor","patch"}
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("VersionBump".into())), args };
    }
    let level = match &args[0] {
        Value::String(s) | Value::Symbol(s) => s.clone(),
        _ => "patch".into(),
    };
    let paths = match &args[1] {
        Value::List(vs) => vs.iter().filter_map(as_str).collect::<Vec<_>>(),
        v => vec![as_str(v).unwrap_or_default()],
    };
    let mut results: Vec<Value> = Vec::new();
    for p in paths {
        if p.is_empty() {
            continue;
        }
        if !fs_exists(&p) {
            results.push(failure("VersionBump::missing", &p));
            continue;
        }
        let s = match std::fs::read_to_string(&p) {
            Ok(s) => s,
            Err(e) => {
                results.push(failure("VersionBump::read", &e.to_string()));
                continue;
            }
        };
        let mut changed = false;
        let out = if p.ends_with(".toml") {
            match s.parse::<toml::Value>() {
                Ok(mut tv) => {
                    // Try update [package].version or workspace.package.version
                    let bump = |v: &str| -> String {
                        let mut parts: Vec<i64> =
                            v.split('.').filter_map(|x| x.parse::<i64>().ok()).collect();
                        while parts.len() < 3 {
                            parts.push(0);
                        }
                        match level.as_str() {
                            "major" => {
                                parts[0] += 1;
                                parts[1] = 0;
                                parts[2] = 0
                            }
                            "minor" => {
                                parts[1] += 1;
                                parts[2] = 0
                            }
                            _ => parts[2] += 1,
                        }
                        format!("{}.{}.{}", parts[0], parts[1], parts[2])
                    };
                    let mut bumped = false;
                    if let Some(tbl) = tv.get_mut("package").and_then(|x| x.as_table_mut()) {
                        if let Some(ver) = tbl.get_mut("version") {
                            if let Some(sv) = ver.as_str() {
                                *ver = toml::Value::String(bump(sv));
                                bumped = true;
                            }
                        }
                    }
                    if !bumped {
                        if let Some(ws) = tv.get_mut("workspace").and_then(|x| x.as_table_mut()) {
                            if let Some(pkg) = ws.get_mut("package").and_then(|x| x.as_table_mut())
                            {
                                if let Some(ver) = pkg.get_mut("version") {
                                    if let Some(sv) = ver.as_str() {
                                        *ver = toml::Value::String(bump(sv));
                                        bumped = true;
                                    }
                                }
                            }
                        }
                    }
                    if bumped {
                        changed = true;
                        tv.to_string()
                    } else {
                        s.clone()
                    }
                }
                Err(_) => s.clone(),
            }
        } else {
            s.clone()
        };
        if changed {
            if let Err(e) = std::fs::write(&p, &out) {
                results.push(failure("VersionBump::write", &e.to_string()));
                continue;
            }
        }
        results.push(Value::Assoc(HashMap::from([
            ("path".into(), Value::String(p.clone())),
            ("changed".into(), Value::Boolean(changed)),
        ])));
    }
    Value::List(results)
}

fn changelog_generate(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // ChangelogGenerate[range?] => markdown string built from GitLog
    let range = if args.len() >= 1 { as_str(&ev.eval(args[0].clone())) } else { None };
    let mut git_args: Vec<Value> = vec![];
    if let Some(r) = range {
        git_args.push(Value::String(r));
    }
    let log = ev.eval(Value::expr(Value::Symbol("GitLog".into()), git_args));
    let mut sections: HashMap<&str, Vec<String>> = HashMap::from([
        ("feat", Vec::new()),
        ("fix", Vec::new()),
        ("docs", Vec::new()),
        ("refactor", Vec::new()),
        ("perf", Vec::new()),
        ("chore", Vec::new()),
    ]);
    if let Value::List(items) = log {
        for it in items {
            if let Value::Assoc(m) = it {
                if let Some(msg) = m.get("message").and_then(as_str) {
                    let lower = msg.to_lowercase();
                    let kind = if lower.starts_with("feat") {
                        "feat"
                    } else if lower.starts_with("fix") {
                        "fix"
                    } else if lower.starts_with("docs") {
                        "docs"
                    } else if lower.starts_with("refactor") {
                        "refactor"
                    } else if lower.starts_with("perf") {
                        "perf"
                    } else {
                        "chore"
                    };
                    if let Some(v) = sections.get_mut(kind) {
                        v.push(msg);
                    }
                }
            }
        }
    }
    let mut out = String::new();
    for (k, title) in [
        ("feat", "Features"),
        ("fix", "Fixes"),
        ("docs", "Docs"),
        ("refactor", "Refactors"),
        ("perf", "Performance"),
        ("chore", "Chores"),
    ] {
        if let Some(xs) = sections.get(k) {
            if !xs.is_empty() {
                out.push_str(&format!("## {}\n", title));
                for m in xs {
                    out.push_str(&format!("- {}\n", m));
                }
                out.push('\n');
            }
        }
    }
    Value::String(out)
}

fn release_tag(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // ReleaseTag[version, opts?] uses GitTag and optionally GitPush
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("ReleaseTag".into())), args };
    }
    let ver = match as_str(&ev.eval(args[0].clone())) {
        Some(s) => s,
        None => return failure("ReleaseTag::arg", "expected version string"),
    };
    let opts = if args.len() >= 2 {
        match ev.eval(args[1].clone()) {
            Value::Assoc(m) => m,
            _ => HashMap::new(),
        }
    } else {
        HashMap::new()
    };
    let tag_name = format!("v{}", ver);
    let _ = ev.eval(Value::expr(
        Value::Symbol("GitTag".into()),
        vec![Value::String(tag_name.clone()), Value::String(format!("Release {}", ver))],
    ));
    if let Some(Value::Boolean(true)) = opts.get("Push") {
        let _ = ev.eval(Value::expr(
            Value::Symbol("GitPush".into()),
            vec![Value::String("origin".into()), Value::String(tag_name.clone())],
        ));
    }
    Value::String(tag_name)
}

fn describe_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("Describe".into())), args };
    }
    let name = as_str(&args[0]).unwrap_or_else(|| "Suite".into());
    let items = match ev.eval(args[1].clone()) {
        Value::List(vs) => vs,
        other => vec![other],
    };
    Value::Assoc(HashMap::from([
        ("type".into(), Value::String("suite".into())),
        ("name".into(), Value::String(name)),
        ("items".into(), Value::List(items)),
        (
            "options".into(),
            if args.len() >= 3 { ev.eval(args[2].clone()) } else { Value::Assoc(HashMap::new()) },
        ),
    ]))
}

fn it_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("It".into())), args };
    }
    let name = as_str(&args[0]).unwrap_or_else(|| "Case".into());
    Value::Assoc(HashMap::from([
        ("type".into(), Value::String("case".into())),
        ("name".into(), Value::String(name)),
        ("body".into(), args[1].clone()),
        (
            "options".into(),
            if args.len() >= 3 { args[2].clone() } else { Value::Assoc(HashMap::new()) },
        ),
    ]))
}

pub fn register_dev(ev: &mut Evaluator) {
    // BDD sugar
    ev.register("Describe", describe_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("It", it_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("FormatLyraText", format_text as NativeFn, Attributes::empty());
    ev.register("FormatLyraFile", format_file as NativeFn, Attributes::empty());
    ev.register("FormatLyra", format_any as NativeFn, Attributes::empty());
    ev.register("LintLyraText", lint_text as NativeFn, Attributes::empty());
    ev.register("LintLyraFile", lint_file as NativeFn, Attributes::empty());
    ev.register("LintLyra", lint_any as NativeFn, Attributes::empty());
    // Existing helpers
    ev.register("ConfigLoad", config_load as NativeFn, Attributes::empty());
    ev.register("SecretsGet", secrets_get as NativeFn, Attributes::empty());
    // Aliases and naming-consistent wrappers
    ev.register("GetSecret", secrets_get as NativeFn, Attributes::empty());
    ev.register("LoadConfig", load_config as NativeFn, Attributes::empty());
    ev.register("Env", env_read as NativeFn, Attributes::empty());
    ev.register("VersionBump", version_bump as NativeFn, Attributes::empty());
    ev.register("ChangelogGenerate", changelog_generate as NativeFn, Attributes::empty());
    ev.register("ReleaseTag", release_tag as NativeFn, Attributes::empty());
}

pub fn register_dev_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    register_if(ev, pred, "FormatLyraText", format_text as NativeFn, Attributes::empty());
    register_if(ev, pred, "FormatLyraFile", format_file as NativeFn, Attributes::empty());
    register_if(ev, pred, "FormatLyra", format_any as NativeFn, Attributes::empty());
    register_if(ev, pred, "LintLyraText", lint_text as NativeFn, Attributes::empty());
    register_if(ev, pred, "LintLyraFile", lint_file as NativeFn, Attributes::empty());
    register_if(ev, pred, "LintLyra", lint_any as NativeFn, Attributes::empty());
    register_if(ev, pred, "ConfigLoad", config_load as NativeFn, Attributes::empty());
    register_if(ev, pred, "SecretsGet", secrets_get as NativeFn, Attributes::empty());
    register_if(ev, pred, "GetSecret", secrets_get as NativeFn, Attributes::empty());
    register_if(ev, pred, "LoadConfig", load_config as NativeFn, Attributes::empty());
    register_if(ev, pred, "Env", env_read as NativeFn, Attributes::empty());
    register_if(ev, pred, "VersionBump", version_bump as NativeFn, Attributes::empty());
    register_if(ev, pred, "ChangelogGenerate", changelog_generate as NativeFn, Attributes::empty());
    register_if(ev, pred, "ReleaseTag", release_tag as NativeFn, Attributes::empty());
}

// ---- Naming-consistent wrappers ----
// LoadConfig[<| files->{...}, envPrefix->"APP_", overrides-><|...|> |>]
fn load_config(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    use std::collections::HashMap as Map;
    let opts = match args.as_slice() {
        [v] => match ev.eval(v.clone()) { Value::Assoc(m) => m, _ => Map::new() },
        _ => Map::new(),
    };
    let mut out: Map<String, Value> = Map::new();
    // files: merge JSON/YAML/TOML maps
    if let Some(Value::List(paths)) = opts.get("files") {
        for p in paths {
            if let Value::String(path) | Value::Symbol(path) = p {
                if let Ok(s) = std::fs::read_to_string(path) {
                    let v = if path.ends_with(".json") {
                        serde_json::from_str::<serde_json::Value>(&s).ok().map(json_to_value)
                    } else if path.ends_with(".yaml") || path.ends_with(".yml") {
                        serde_yaml::from_str::<serde_json::Value>(&s).ok().map(json_to_value)
                    } else if path.ends_with(".toml") {
                        toml::from_str::<toml::Value>(&s).ok().map(toml_to_value)
                    } else { None };
                    if let Some(Value::Assoc(m)) = v { for (k,vv) in m { out.insert(k, vv); } }
                }
            }
        }
    }
    // envPrefix: include any process env vars starting with prefix (no stripping)
    if let Some(Value::String(prefix)) | Some(Value::Symbol(prefix)) = opts.get("envPrefix") {
        for (k, v) in std::env::vars() {
            if k.starts_with(prefix) {
                out.insert(k, Value::String(v));
            }
        }
    }
    // overrides
    if let Some(Value::Assoc(ovr)) = opts.get("overrides") {
        for (k, v) in ovr { out.insert(k.clone(), v.clone()); }
    }
    Value::Assoc(out)
}

// Env[<| keys->{"A","B"}, required->{"A"}, defaults-><|B->"x"|> |>]
fn env_read(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    use std::collections::HashMap as Map;
    let opts = match args.as_slice() { [v] => match ev.eval(v.clone()) { Value::Assoc(m) => m, _ => Map::new() }, _ => Map::new() };
    let mut keys: Vec<String> = Vec::new();
    if let Some(Value::List(vs)) = opts.get("keys") {
        for v in vs { if let Value::String(s) | Value::Symbol(s) = v { keys.push(s.clone()); } }
    }
    let mut required: Vec<String> = Vec::new();
    if let Some(Value::List(vs)) = opts.get("required") {
        for v in vs { if let Value::String(s) | Value::Symbol(s) = v { required.push(s.clone()); } }
    }
    let defaults: Map<String, Value> = match opts.get("defaults") { Some(Value::Assoc(m)) => m.clone(), _ => Map::new() };
    let mut out: Map<String, Value> = Map::new();
    for k in keys.iter() {
        match std::env::var(k) {
            Ok(val) => { out.insert(k.clone(), Value::String(val)); },
            Err(_) => {
                if let Some(d) = defaults.get(k) { out.insert(k.clone(), d.clone()); } else {
                    // keep Null to signal missing unless required triggers failure later
                    out.insert(k.clone(), Value::Symbol("Null".into()));
                }
            }
        }
    }
    // Enforce required
    for rk in required.iter() {
        if !out.contains_key(rk) || matches!(out.get(rk), Some(Value::Symbol(s)) if s=="Null") {
            return failure("Env::required", &format!("missing required env {}", rk));
        }
    }
    Value::Assoc(out)
}
