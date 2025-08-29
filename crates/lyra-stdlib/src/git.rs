use lyra_core::value::Value;
#[cfg(feature = "tools")]
use crate::tool_spec;
#[cfg(feature = "tools")]
use crate::tools::add_specs;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
use std::collections::HashMap;
use std::io::Write;
use std::process::{Command, Stdio};

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

fn as_str(v: &Value) -> Option<String> {
    match v {
        Value::String(s) | Value::Symbol(s) => Some(s.clone()),
        _ => None,
    }
}

fn cwd_from(ev: &mut Evaluator, args: &[Value]) -> Option<String> {
    for a in args {
        if let Value::Assoc(m) = ev.eval(a.clone()) {
            if let Some(Value::String(s)) | Some(Value::Symbol(s)) = m.get("Cwd").or_else(|| m.get("cwd")) {
                return Some(s.clone());
            }
            if let Some(Value::String(s)) | Some(Value::Symbol(s)) = m.get("Path").or_else(|| m.get("path")) {
                return Some(s.clone());
            }
        }
    }
    ev.get_env("CurrentDir").and_then(|v| as_str(&v))
}

fn run_git(git_args: &[&str], cwd: Option<String>, input: Option<&str>) -> (i32, String, String) {
    let git = "git";
    let mut cmd = Command::new(git);
    cmd.args(git_args);
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    match cmd.spawn() {
        Ok(mut child) => {
            if let Some(data) = input {
                if let Some(mut stdin) = child.stdin.take() {
                    let _ = stdin.write_all(data.as_bytes());
                }
            }
            let out = child.wait_with_output();
            match out {
                Ok(o) => (
                    o.status.code().unwrap_or(-1),
                    String::from_utf8_lossy(&o.stdout).to_string(),
                    String::from_utf8_lossy(&o.stderr).to_string(),
                ),
                Err(e) => (-1, String::new(), format!("spawn/wait error: {}", e)),
            }
        }
        Err(e) => (-1, String::new(), format!("spawn error: {}", e)),
    }
}

pub fn register_git(ev: &mut Evaluator) {
    ev.register("GitVersion", git_version as NativeFn, Attributes::empty());
    ev.register("GitRoot", git_root as NativeFn, Attributes::empty());
    ev.register("GitInit", git_init as NativeFn, Attributes::empty());
    ev.register("GitStatus", git_status as NativeFn, Attributes::empty());
    ev.register("GitAdd", git_add as NativeFn, Attributes::LISTABLE);
    ev.register("GitCommit", git_commit as NativeFn, Attributes::empty());
    ev.register("GitCurrentBranch", git_current_branch as NativeFn, Attributes::empty());
    ev.register("GitBranchList", git_branch_list as NativeFn, Attributes::empty());
    ev.register("GitBranch", git_branch as NativeFn, Attributes::empty());
    ev.register("GitSwitch", git_switch as NativeFn, Attributes::empty());
    ev.register("GitDiff", git_diff as NativeFn, Attributes::empty());
    ev.register("GitApply", git_apply as NativeFn, Attributes::empty());
    ev.register("GitLog", git_log as NativeFn, Attributes::empty());
    ev.register("GitRemoteList", git_remote_list as NativeFn, Attributes::empty());
    ev.register("GitFetch", git_fetch as NativeFn, Attributes::empty());
    ev.register("GitPull", git_pull as NativeFn, Attributes::empty());
    ev.register("GitPush", git_push as NativeFn, Attributes::empty());

    // High-level helpers
    ev.register("GitEnsureRepo", git_ensure_repo as NativeFn, Attributes::empty());
    ev.register("GitStatusSummary", git_status_summary as NativeFn, Attributes::empty());
    ev.register("GitSmartCommit", git_smart_commit as NativeFn, Attributes::empty());
    ev.register("GitFeatureBranch", git_create_feature_branch as NativeFn, Attributes::empty());
    ev.register("GitSyncUpstream", git_sync_upstream as NativeFn, Attributes::empty());

    #[cfg(feature = "tools")]
    add_specs(vec![
        tool_spec!("GitVersion", summary: "Get git client version string", params: [], tags: ["git","vcs"]),
        tool_spec!("GitRoot", summary: "Path to repository root (or Null)", params: [], tags: ["git","vcs"]),
        tool_spec!("GitInit", summary: "Initialize a new git repository", params: ["opts?"], tags: ["git","repo"]),
        tool_spec!("GitStatus", summary: "Status (porcelain) with branch/ahead/behind/changes", params: ["opts?"], tags: ["git","status"]),
        tool_spec!("GitAdd", summary: "Stage files for commit", params: ["paths","opts?"], tags: ["git","index"], examples: [Value::String("GitAdd[\"src/main.rs\"]".into())]),
        tool_spec!("GitCommit", summary: "Create a commit with message", params: ["message","opts?"], tags: ["git","commit"], examples: [Value::String("GitCommit[\"feat: add api\"]".into())]),
        tool_spec!("GitCurrentBranch", summary: "Current branch name", params: [], tags: ["git","branch"]),
        tool_spec!("GitBranchList", summary: "List local branches", params: [], tags: ["git","branch"]),
        tool_spec!("GitBranch", summary: "Create a new branch", params: ["name","opts?"], tags: ["git","branch"], examples: [Value::String("GitBranch[\"feature/x\"]".into())]),
        tool_spec!("GitSwitch", summary: "Switch to branch (optionally create)", params: ["name","opts?"], tags: ["git","branch"]),
        tool_spec!("GitDiff", summary: "Diff against base and optional paths", params: ["opts?"], tags: ["git","diff"]),
        tool_spec!("GitApply", summary: "Apply a patch (or check only)", params: ["patch","opts?"], tags: ["git","patch"]),
        tool_spec!("GitLog", summary: "List commits with formatting options", params: ["opts?"], tags: ["git","log"]),
        tool_spec!("GitRemoteList", summary: "List remotes", params: [], tags: ["git","remote"]),
        tool_spec!("GitFetch", summary: "Fetch from remote", params: ["remote?"], tags: ["git","remote"]),
        tool_spec!("GitPull", summary: "Pull from remote", params: ["remote?","opts?"], tags: ["git","remote"]),
        tool_spec!("GitPush", summary: "Push to remote", params: ["opts?"], tags: ["git","remote"]),
        // High-level helpers
        tool_spec!("GitEnsureRepo", summary: "Ensure Cwd is a git repo (init if needed)", params: ["opts?"], tags: ["git","repo"]),
        tool_spec!("GitStatusSummary", summary: "Summarize status counts and branch", params: ["opts?"], tags: ["git","status"]),
        tool_spec!("GitSmartCommit", summary: "Stage + conventional commit (auto msg option)", params: ["opts?"], tags: ["git","commit"]),
        tool_spec!("GitFeatureBranch", summary: "Create and switch to a feature branch", params: ["opts?"], tags: ["git","branch"]),
        tool_spec!("GitSyncUpstream", summary: "Fetch, rebase/merge, and push upstream", params: ["opts?"], tags: ["git","sync"]),
    ]);
}

pub fn register_git_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    super::register_if(ev, pred, "GitVersion", git_version as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "GitRoot", git_root as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "GitInit", git_init as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "GitStatus", git_status as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "GitAdd", git_add as NativeFn, Attributes::LISTABLE);
    super::register_if(ev, pred, "GitCommit", git_commit as NativeFn, Attributes::empty());
    super::register_if(
        ev,
        pred,
        "GitCurrentBranch",
        git_current_branch as NativeFn,
        Attributes::empty(),
    );
    super::register_if(ev, pred, "GitBranchList", git_branch_list as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "GitBranch", git_branch as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "GitSwitch", git_switch as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "GitDiff", git_diff as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "GitApply", git_apply as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "GitLog", git_log as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "GitRemoteList", git_remote_list as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "GitFetch", git_fetch as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "GitPull", git_pull as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "GitPush", git_push as NativeFn, Attributes::empty());

    super::register_if(ev, pred, "GitEnsureRepo", git_ensure_repo as NativeFn, Attributes::empty());
    super::register_if(
        ev,
        pred,
        "GitStatusSummary",
        git_status_summary as NativeFn,
        Attributes::empty(),
    );
    super::register_if(
        ev,
        pred,
        "GitSmartCommit",
        git_smart_commit as NativeFn,
        Attributes::empty(),
    );
    super::register_if(ev, pred, "GitFeatureBranch", git_create_feature_branch as NativeFn, Attributes::empty());
    super::register_if(
        ev,
        pred,
        "GitSyncUpstream",
        git_sync_upstream as NativeFn,
        Attributes::empty(),
    );
}

fn ok_bool(ok: bool) -> Value {
    Value::Boolean(ok)
}
fn booly(v: Option<&Value>) -> bool {
    match v {
        Some(Value::Boolean(b)) => *b,
        Some(Value::String(s)) | Some(Value::Symbol(s)) => {
            let ls = s.to_lowercase();
            ls == "true" || ls == "on" || ls == "1"
        }
        _ => false,
    }
}

fn git_version(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let cwd = cwd_from(ev, &args);
    let (code, out, _err) = run_git(&["--version"], cwd, None);
    if code == 0 {
        Value::String(out.trim().to_string())
    } else {
        Value::Symbol("Null".into())
    }
}

fn git_root(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let cwd = cwd_from(ev, &args);
    let (code, out, _err) = run_git(&["rev-parse", "--show-toplevel"], cwd, None);
    if code == 0 {
        Value::String(out.trim().to_string())
    } else {
        Value::Symbol("Null".into())
    }
}

fn git_init(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // GitInit[<|Path->"...", Bare->False, InitialBranch->"main"|>]
    let mut path: Option<String> = None;
    let mut bare = false;
    let mut ib: Option<String> = None;
    if let Some(Value::Assoc(m)) = args.get(0) {
        if let Some(Value::String(s)) | Some(Value::Symbol(s)) = m.get("Path").or_else(|| m.get("path")) {
            path = Some(s.clone());
        }
        if let Some(v) = m.get("Bare") {
            bare = booly(Some(v));
        }
        if let Some(Value::String(s)) | Some(Value::Symbol(s)) = m.get("InitialBranch").or_else(|| m.get("initialBranch")) {
            ib = Some(s.clone());
        }
    }
    let mut argv: Vec<&str> = vec!["init"];
    if bare {
        argv.push("--bare");
    }
    if let Some(s) = ib.as_deref() {
        argv.push("-b");
        argv.push(s);
    }
    if let Some(p) = path.as_deref() {
        argv.push(p);
    }
    let cwd = cwd_from(ev, &args);
    let (code, _out, _err) = run_git(&argv, cwd, None);
    ok_bool(code == 0)
}

fn parse_status_porcelain(s: &str) -> (HashMap<String, Value>, Vec<Value>) {
    let mut hdr: HashMap<String, Value> = HashMap::new();
    let mut changes: Vec<Value> = Vec::new();
    for line in s.lines() {
        if line.starts_with("## ") {
            // Example: ## main...origin/main [ahead 1]
            let rest = &line[3..];
            let mut branch = rest.to_string();
            let mut ahead = 0i64;
            let mut behind = 0i64;
            if let Some((b, rb)) = rest.split_once(" [") {
                branch = b.to_string();
                let r = rb.trim_end_matches(']');
                for part in r.split(',') {
                    let t = part.trim();
                    if let Some(n) = t.strip_prefix("ahead ") {
                        ahead = n.parse().unwrap_or(0);
                    }
                    if let Some(n) = t.strip_prefix("behind ") {
                        behind = n.parse().unwrap_or(0);
                    }
                }
            }
            hdr.insert("Branch".into(), Value::String(branch));
            hdr.insert("Ahead".into(), Value::Integer(ahead));
            hdr.insert("Behind".into(), Value::Integer(behind));
        } else {
            // XY PATH or rename: R100 from -> to
            if line.len() >= 3 {
                let x = line.chars().nth(0).unwrap_or(' ');
                let y = line.chars().nth(1).unwrap_or(' ');
                let rest = &line[3..];
                let (path, orig): (String, Option<String>) = if rest.contains(" -> ") {
                    let mut parts = rest.splitn(2, " -> ");
                    let a = parts.next().unwrap_or("").to_string();
                    let b = parts.next().unwrap_or("").to_string();
                    (b, Some(a))
                } else {
                    (rest.to_string(), None)
                };
                let status = match (x, y) {
                    ('?', '?') => "untracked",
                    ('A', _) | (_, 'A') => "added",
                    ('M', _) | (_, 'M') => "modified",
                    ('D', _) | (_, 'D') => "deleted",
                    ('R', _) | (_, 'R') => "renamed",
                    _ => "changed",
                };
                let mut m = HashMap::new();
                m.insert("path".into(), Value::String(path));
                m.insert("x".into(), Value::String(x.to_string()));
                m.insert("y".into(), Value::String(y.to_string()));
                m.insert("status".into(), Value::String(status.into()));
                if let Some(o) = orig {
                    m.insert("orig".into(), Value::String(o));
                }
                changes.push(Value::Assoc(m));
            }
        }
    }
    (hdr, changes)
}

fn git_status(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // GitStatus[<|Cwd->..., Porcelain->True|>] -> {Branch,Ahead,Behind,Changes:[...], Raw}
    let porcelain = args
        .get(0)
        .and_then(|v| if let Value::Assoc(m) = v { m.get("Porcelain") } else { None })
        .map(|v| booly(Some(v)))
        .unwrap_or(true);
    let mut argv = vec!["status"];
    if porcelain {
        argv.push("--porcelain=v1");
        argv.push("--branch");
    }
    let cwd = cwd_from(ev, &args);
    let (code, out, err) = run_git(&argv, cwd, None);
    if code != 0 {
        return Value::Assoc(HashMap::from([(String::from("Error"), Value::String(err))]));
    }
    if porcelain {
        let (hdr, changes) = parse_status_porcelain(&out);
        let mut m = hdr;
        m.insert("Changes".into(), Value::List(changes));
        m.insert("Raw".into(), Value::String(out));
        Value::Assoc(m)
    } else {
        Value::String(out)
    }
}

fn git_add(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // GitAdd[{paths...}|path, <|Cwd->...|>]
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("GitAdd".into())), args };
    }
    let mut paths: Vec<String> = Vec::new();
    match ev.eval(args[0].clone()) {
        Value::List(vs) => {
            for v in vs {
                if let Some(s) = as_str(&v) {
                    paths.push(s);
                }
            }
        }
        v => {
            if let Some(s) = as_str(&v) {
                paths.push(s);
            }
        }
    }
    if paths.is_empty() {
        paths.push(".".into());
    }
    let str_s: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
    let mut argv: Vec<&str> = vec!["add", "--"];
    argv.extend(str_s);
    let cwd = cwd_from(ev, &args);
    let (code, _out, _err) = run_git(&argv, cwd, None);
    ok_bool(code == 0)
}

fn git_commit(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // GitCommit[Message, <|All->False, Cwd->...|>]
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("GitCommit".into())), args };
    }
    let msg = as_str(&ev.eval(args[0].clone())).unwrap_or_else(|| "commit".into());
    let all = args.get(1).and_then(|v| {
        if let Value::Assoc(m) = ev.eval(v.clone()) {
            m.get("All").cloned()
        } else {
            None
        }
    });
    let mut argv: Vec<&str> = vec!["commit", "-m", &msg];
    if booly(all.as_ref()) {
        argv.push("-a");
    }
    let cwd = cwd_from(ev, &args);
    let (code, _out, err) = run_git(&argv, cwd.clone(), None);
    if code != 0 {
        return Value::Assoc(HashMap::from([(String::from("Error"), Value::String(err))]));
    }
    // return last commit sha and message
    let (code2, out2, _e2) = run_git(&["rev-parse", "HEAD"], cwd, None);
    if code2 == 0 {
        Value::Assoc(HashMap::from([
            (String::from("Sha"), Value::String(out2.trim().into())),
            (String::from("Message"), Value::String(msg)),
        ]))
    } else {
        ok_bool(true)
    }
}

fn git_current_branch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let cwd = cwd_from(ev, &args);
    let (code, out, _err) = run_git(&["rev-parse", "--abbrev-ref", "HEAD"], cwd, None);
    if code == 0 {
        Value::String(out.trim().to_string())
    } else {
        Value::Symbol("Null".into())
    }
}

fn git_branch_list(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let cwd = cwd_from(ev, &args);
    let (code, out, _err) = run_git(&["branch", "--list", "--format=%(refname:short)"], cwd, None);
    if code != 0 {
        return Value::List(vec![]);
    }
    let list: Vec<Value> = out
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| Value::String(l.trim().into()))
        .collect();
    Value::List(list)
}

fn git_branch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // GitBranch[name, <|Start->ref|>]
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("GitBranch".into())), args };
    }
    let name = as_str(&ev.eval(args[0].clone())).unwrap_or_else(|| "feature".into());
    let start = args.get(1).and_then(|v| {
        if let Value::Assoc(m) = ev.eval(v.clone()) {
            m.get("Start").and_then(as_str)
        } else {
            None
        }
    });
    let mut argv: Vec<&str> = vec!["branch", &name];
    if let Some(s) = start.as_deref() {
        argv.push(s);
    }
    let cwd = cwd_from(ev, &args);
    let (code, _out, _err) = run_git(&argv, cwd, None);
    ok_bool(code == 0)
}

fn git_switch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // GitSwitch[name, <|Create->False|>]
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("GitSwitch".into())), args };
    }
    let name = as_str(&ev.eval(args[0].clone())).unwrap_or_default();
    let create = args
        .get(1)
        .and_then(|v| {
            if let Value::Assoc(m) = ev.eval(v.clone()) {
                Some(booly(m.get("Create")))
            } else {
                None
            }
        })
        .unwrap_or(false);
    let mut argv: Vec<&str> = vec!["switch"];
    if create {
        argv.push("-c");
    }
    argv.push(&name);
    let cwd = cwd_from(ev, &args);
    let (code, _out, _err) = run_git(&argv, cwd, None);
    ok_bool(code == 0)
}

fn git_diff(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // GitDiff[<|Base->"HEAD", Paths->{...}|>]
    let mut base: Option<String> = None;
    let mut paths: Vec<String> = Vec::new();
    if let Some(Value::Assoc(m)) = args.get(0).map(|v| ev.eval(v.clone())) {
        if let Some(Value::String(s)) | Some(Value::Symbol(s)) = m.get("Base").or_else(|| m.get("base")) {
            base = Some(s.clone());
        }
        if let Some(Value::List(vs)) = m.get("Paths").or_else(|| m.get("paths")) {
            for v in vs {
                if let Some(s) = as_str(&v) {
                    paths.push(s);
                }
            }
        }
    }
    let mut argv: Vec<&str> = vec!["diff"];
    if let Some(b) = base.as_deref() {
        argv.push(b);
    }
    if !paths.is_empty() {
        argv.push("--");
        for p in &paths {
            argv.push(p);
        }
    }
    let cwd = cwd_from(ev, &args);
    let (code, out, err) = run_git(&argv, cwd, None);
    if code == 0 {
        Value::String(out)
    } else {
        Value::Assoc(HashMap::from([(String::from("Error"), Value::String(err))]))
    }
}

fn git_apply(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // GitApply[patch, <|Check->False|>]
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("GitApply".into())), args };
    }
    let patch = as_str(&ev.eval(args[0].clone())).unwrap_or_default();
    let check = args
        .get(1)
        .and_then(|v| {
            if let Value::Assoc(m) = ev.eval(v.clone()) {
                Some(booly(m.get("Check")))
            } else {
                None
            }
        })
        .unwrap_or(false);
    let mut argv: Vec<&str> = vec!["apply"];
    if check {
        argv.push("--check");
    }
    let cwd = cwd_from(ev, &args);
    let (code, _out, err) = run_git(&argv, cwd, Some(&patch));
    if code == 0 {
        Value::Boolean(true)
    } else {
        Value::Assoc(HashMap::from([(String::from("Error"), Value::String(err))]))
    }
}

fn git_log(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // GitLog[<|Limit->n, Format->"%H|%an|%ae|%ad|%s"|>]
    let mut limit: i64 = 20;
    let mut fmt = "%H|%an|%ae|%ad|%s".to_string();
    if let Some(Value::Assoc(m)) = args.get(0).map(|v| ev.eval(v.clone())) {
        if let Some(Value::Integer(n)) = m.get("Limit").or_else(|| m.get("limit")) {
            if *n > 0 {
                limit = *n;
            }
        }
        if let Some(Value::String(s)) | Some(Value::Symbol(s)) = m.get("Format").or_else(|| m.get("format")) {
            fmt = s.clone();
        }
    }
    let pretty = format!("--pretty={}", fmt);
    let max = format!("-n{}", limit);
    let cwd = cwd_from(ev, &args);
    let (code, out, _err) = run_git(&["log", &max, &pretty], cwd, None);
    if code != 0 {
        return Value::List(vec![]);
    }
    let list: Vec<Value> = out
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| Value::String(l.to_string()))
        .collect();
    Value::List(list)
}

fn git_remote_list(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let cwd = cwd_from(ev, &args);
    let (code, out, _err) = run_git(&["remote", "-v"], cwd, None);
    if code != 0 {
        return Value::List(vec![]);
    }
    let mut rems: HashMap<String, (String, String)> = HashMap::new();
    for line in out.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 {
            let name = parts[0].to_string();
            let url = parts[1].to_string();
            let kind = parts[2].trim_matches('(').trim_matches(')').to_string();
            let entry = rems.entry(name).or_insert((String::new(), String::new()));
            if kind == "fetch" {
                entry.0 = url;
            } else if kind == "push" {
                entry.1 = url;
            }
        }
    }
    let list: Vec<Value> = rems
        .into_iter()
        .map(|(k, (f, p))| {
            Value::Assoc(HashMap::from([
                (String::from("name"), Value::String(k)),
                (String::from("fetch"), Value::String(f)),
                (String::from("push"), Value::String(p)),
            ]))
        })
        .collect();
    Value::List(list)
}

fn git_fetch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // GitFetch[<|Remote->"origin", Prune->True|>]
    let mut remote = "origin".to_string();
    let mut prune = false;
    if let Some(Value::Assoc(m)) = args.get(0).map(|v| ev.eval(v.clone())) {
        if let Some(Value::String(s)) | Some(Value::Symbol(s)) = m.get("Remote") {
            remote = s.clone();
        }
        if let Some(v) = m.get("Prune") {
            prune = booly(Some(v));
        }
    }
    let mut argv: Vec<String> = vec!["fetch".into(), remote];
    if prune {
        argv.push("--prune".into());
    }
    let refs: Vec<String> = args
        .get(1)
        .and_then(|v| {
            if let Value::List(vs) = ev.eval(v.clone()) {
                Some(vs.into_iter().filter_map(|x| as_str(&x)).collect())
            } else {
                None
            }
        })
        .unwrap_or_default();
    for r in refs {
        argv.push(r);
    }
    let refv: Vec<&str> = argv.iter().map(|s| s.as_str()).collect();
    let cwd = cwd_from(ev, &args);
    let (code, _out, err) = run_git(&refv, cwd, None);
    if code == 0 {
        Value::Boolean(true)
    } else {
        Value::Assoc(HashMap::from([(String::from("Error"), Value::String(err))]))
    }
}

fn git_pull(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // GitPull[<|Rebase->True, Remote->..., Branch->...|>]
    let mut argv_s: Vec<String> = vec!["pull".into()];
    if let Some(Value::Assoc(m)) = args.get(0).map(|v| ev.eval(v.clone())) {
        if booly(m.get("Rebase")) {
            argv_s.push("--rebase".into());
        }
        if let Some(Value::String(r)) | Some(Value::Symbol(r)) = m.get("Remote") {
            argv_s.push(r.clone());
            if let Some(Value::String(b)) | Some(Value::Symbol(b)) = m.get("Branch") {
                argv_s.push(b.clone());
            }
        }
    }
    let argv: Vec<&str> = argv_s.iter().map(|s| s.as_str()).collect();
    let cwd = cwd_from(ev, &args);
    let (code, _out, err) = run_git(&argv, cwd, None);
    if code == 0 {
        Value::Boolean(true)
    } else {
        Value::Assoc(HashMap::from([(String::from("Error"), Value::String(err))]))
    }
}

fn git_push(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // GitPush[<|Remote->"origin", Branch->current, SetUpstream->False, Force->False|>]
    let mut remote = "origin".to_string();
    let mut branch: Option<String> = None;
    let mut u = false;
    let mut f = false;
    if let Some(Value::Assoc(m)) = args.get(0).map(|v| ev.eval(v.clone())) {
        if let Some(Value::String(s)) | Some(Value::Symbol(s)) = m.get("Remote") {
            remote = s.clone();
        }
        if let Some(Value::String(s)) | Some(Value::Symbol(s)) = m.get("Branch") {
            branch = Some(s.clone());
        }
        u = booly(m.get("SetUpstream"));
        f = booly(m.get("Force"));
    }
    let mut argv: Vec<String> = vec!["push".into(), remote];
    if u {
        argv.push("-u".into());
    }
    if f {
        argv.push("--force".into());
    }
    if branch.is_none() {
        // get current
        let cwd = cwd_from(ev, &args);
        let (_c, out, _e) = run_git(&["rev-parse", "--abbrev-ref", "HEAD"], cwd, None);
        let b = out.trim();
        if !b.is_empty() {
            branch = Some(b.to_string());
        }
    }
    if let Some(b) = branch {
        argv.push(b);
    }
    let refv: Vec<&str> = argv.iter().map(|s| s.as_str()).collect();
    let cwd = cwd_from(ev, &args);
    let (code, _out, err) = run_git(&refv, cwd, None);
    if code == 0 {
        Value::Boolean(true)
    } else {
        Value::Assoc(HashMap::from([(String::from("Error"), Value::String(err))]))
    }
}

// ---------------- High-level helpers ----------------

fn is_repo(cwd: Option<String>) -> bool {
    let (code, _out, _err) = run_git(&["rev-parse", "--git-dir"], cwd, None);
    code == 0
}

fn git_ensure_repo(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // GitEnsureRepo[<|Path->..., InitialBranch->"main", GitIgnore->"...", User-><|Name,Email|>|>]
    let mut path = cwd_from(ev, &args).unwrap_or_else(|| ".".into());
    let mut initial = Some("main".to_string());
    let mut gitignore: Option<String> = None;
    let mut user: Option<(String, String)> = None;
    if let Some(Value::Assoc(m)) = args.get(0).map(|v| ev.eval(v.clone())) {
        if let Some(Value::String(s)) | Some(Value::Symbol(s)) = m.get("Path") {
            path = s.clone();
        }
        if let Some(Value::String(s)) | Some(Value::Symbol(s)) = m.get("InitialBranch") {
            initial = Some(s.clone());
        }
        if let Some(Value::String(s)) = m.get("GitIgnore") {
            gitignore = Some(s.clone());
        }
        if let Some(Value::Assoc(u)) = m.get("User") {
            let name = u.get("Name").and_then(as_str).unwrap_or_default();
            let email = u.get("Email").and_then(as_str).unwrap_or_default();
            if !name.is_empty() && !email.is_empty() {
                user = Some((name, email));
            }
        }
    }
    let cwd = Some(path.clone());
    if !is_repo(cwd.clone()) {
        // init
        let mut argv: Vec<&str> = vec!["init"];
        if let Some(b) = initial.as_deref() {
            argv.push("-b");
            argv.push(b);
        }
        let (code, _o, e) = run_git(&argv, cwd.clone(), None);
        if code != 0 {
            return Value::Assoc(HashMap::from([(String::from("Error"), Value::String(e))]));
        }
    }
    if let Some((n, e)) = user {
        let _ = run_git(&["config", "user.name", &n], cwd.clone(), None);
        let _ = run_git(&["config", "user.email", &e], cwd.clone(), None);
    }
    if let Some(gi) = gitignore {
        // simple write .gitignore
        let p = format!("{}/.gitignore", path);
        let _ = std::fs::write(&p, gi);
    }
    Value::Assoc(HashMap::from([(String::from("Path"), Value::String(path))]))
}

fn git_status_summary(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let s = git_status(ev, args);
    match s {
        Value::Assoc(mut m) => {
            let mut added = 0;
            let mut modified = 0;
            let mut deleted = 0;
            let mut renamed = 0;
            let mut untracked = 0;
            let mut changed = 0;
            if let Some(Value::List(vs)) = m.get("Changes") {
                for v in vs {
                    if let Value::Assoc(cm) = v {
                        changed += 1;
                        match cm.get("status").and_then(as_str).as_deref() {
                            Some("added") => added += 1,
                            Some("modified") => modified += 1,
                            Some("deleted") => deleted += 1,
                            Some("renamed") => renamed += 1,
                            Some("untracked") => untracked += 1,
                            _ => {}
                        }
                    }
                }
            }
            m.insert(
                "Counts".into(),
                Value::Assoc(HashMap::from([
                    ("added".into(), Value::Integer(added)),
                    ("modified".into(), Value::Integer(modified)),
                    ("deleted".into(), Value::Integer(deleted)),
                    ("renamed".into(), Value::Integer(renamed)),
                    ("untracked".into(), Value::Integer(untracked)),
                    ("changed".into(), Value::Integer(changed)),
                ])),
            );
            Value::Assoc(m)
        }
        other => other,
    }
}

fn git_smart_commit(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // GitSmartCommit[<|Cwd->..., Message->..., All->False, Include->{...}, Exclude->{...}, Conventional->True, AutoMessage->False|>]
    let mut msg: Option<String> = None;
    let mut all = false;
    let mut include: Vec<String> = Vec::new();
    let mut _exclude: Vec<String> = Vec::new();
    let mut conventional = true;
    let mut auto = false;
    if let Some(Value::Assoc(m)) = args.get(0).map(|v| ev.eval(v.clone())) {
        if let Some(Value::String(s)) | Some(Value::Symbol(s)) = m.get("Message") {
            msg = Some(s.clone());
        }
        if let Some(v) = m.get("All") {
            all = booly(Some(v));
        }
        if let Some(Value::List(vs)) = m.get("Include") {
            for v in vs {
                if let Some(s) = as_str(v) {
                    include.push(s);
                }
            }
        }
        if let Some(Value::List(vs)) = m.get("Exclude") {
            for v in vs {
                if let Some(s) = as_str(v) {
                    _exclude.push(s);
                }
            }
        }
        if let Some(v) = m.get("Conventional") {
            conventional = booly(Some(v));
        }
        if let Some(v) = m.get("AutoMessage").or_else(|| m.get("autoMessage")) {
            auto = booly(Some(v));
        }
    }
    let cwd = cwd_from(ev, &args);
    // Stage files
    if include.is_empty() && !all {
        let _ = run_git(&["add", "--", "."], cwd.clone(), None);
    } else if !include.is_empty() {
        let mut av: Vec<&str> = vec!["add", "--"];
        for s in &include {
            av.push(s.as_str());
        }
        let _ = run_git(&av, cwd.clone(), None);
    }
    // Auto message heuristic
    let message = match msg {
        Some(m) => m,
        None => {
            let summary = git_status_summary(
                ev,
                vec![Value::Assoc(HashMap::from([(
                    String::from("Cwd"),
                    Value::String(cwd.clone().unwrap_or_else(|| ".".into())),
                )]))],
            );
            let mut m = "chore: update".to_string();
            if let Value::Assoc(am) = summary {
                if let Some(Value::Assoc(c)) = am.get("Counts") {
                    let a = c
                        .get("added")
                        .and_then(|v| if let Value::Integer(i) = v { Some(*i) } else { None })
                        .unwrap_or(0);
                    let mo = c
                        .get("modified")
                        .and_then(|v| if let Value::Integer(i) = v { Some(*i) } else { None })
                        .unwrap_or(0);
                    let d = c
                        .get("deleted")
                        .and_then(|v| if let Value::Integer(i) = v { Some(*i) } else { None })
                        .unwrap_or(0);
                    m = format!("chore: {} added, {} modified, {} deleted", a, mo, d);
                }
            }
            if !conventional {
                m = m.trim_start_matches("chore: ").to_string();
            }
            if !auto {
                m = "update".into();
            }
            m
        }
    };
    // Commit
    let mut argv: Vec<&str> = vec!["commit", "-m", &message];
    if all {
        argv.push("-a");
    }
    let (code, _o, e) = run_git(&argv, cwd, None);
    if code == 0 {
        Value::Assoc(HashMap::from([(String::from("Message"), Value::String(message))]))
    } else {
        Value::Assoc(HashMap::from([(String::from("Error"), Value::String(e))]))
    }
}

fn git_create_feature_branch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
// GitFeatureBranch[<|Cwd->..., Name->..., From->base|>]
    let mut name_opt: Option<String> = None;
    let mut from: Option<String> = None;
    if let Some(Value::Assoc(m)) = args.get(0).map(|v| ev.eval(v.clone())) {
        if let Some(Value::String(s)) | Some(Value::Symbol(s)) = m.get("Name").or_else(|| m.get("name")) {
            name_opt = Some(s.clone());
        }
        if let Some(Value::String(s)) | Some(Value::Symbol(s)) = m.get("From").or_else(|| m.get("from")) {
            from = Some(s.clone());
        }
    }
    let name =
        name_opt.unwrap_or_else(|| format!("feat/{}", chrono::Utc::now().format("%Y%m%d%H%M%S")));
    let mut argv: Vec<&str> = vec!["switch", "-c", &name];
    if let Some(b) = from.as_deref() {
        argv.push(b);
    }
    let cwd = cwd_from(ev, &args);
    let (code, _o, e) = run_git(&argv, cwd, None);
    if code == 0 {
        Value::String(name)
    } else {
        Value::Assoc(HashMap::from([(String::from("Error"), Value::String(e))]))
    }
}

fn git_sync_upstream(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // GitSyncUpstream[<|Cwd->..., Remote->"origin", Rebase->True|>]
    let mut remote = "origin".to_string();
    let mut rebase = true;
    if let Some(Value::Assoc(m)) = args.get(0).map(|v| ev.eval(v.clone())) {
        if let Some(Value::String(s)) | Some(Value::Symbol(s)) = m.get("Remote").or_else(|| m.get("remote")) {
            remote = s.clone();
        }
        if let Some(v) = m.get("Rebase").or_else(|| m.get("rebase")) {
            rebase = booly(Some(v));
        }
    }
    let cwd = cwd_from(ev, &args);
    let _ = run_git(&["fetch", &remote], cwd.clone(), None);
    let pull_args: Vec<&str> =
        if rebase { vec!["pull", "--rebase", &remote] } else { vec!["pull", &remote] };
    let (pc, _po, pe) = run_git(&pull_args, cwd.clone(), None);
    if pc != 0 {
        return Value::Assoc(HashMap::from([(String::from("Error"), Value::String(pe))]));
    }
    // fast-forward push
    let (puc, _uo, ue) = run_git(&["push"], cwd, None);
    if puc == 0 {
        Value::Assoc(HashMap::from([(String::from("Pushed"), Value::Boolean(true))]))
    } else {
        Value::Assoc(HashMap::from([(String::from("Error"), Value::String(ue))]))
    }
}
