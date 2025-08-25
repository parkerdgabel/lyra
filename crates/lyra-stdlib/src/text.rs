use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
use crate::register_if;
use regex::Regex;
#[cfg(feature = "text_multipattern")] use aho_corasick::AhoCorasick;
#[cfg(feature = "text_glob")] use ignore::{WalkBuilder, overrides::OverrideBuilder};
#[cfg(feature = "text_encoding_detect")] use chardetng::EncodingDetector;
#[cfg(feature = "text_encoding_detect")] use encoding_rs::Encoding;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

pub fn register_text(ev: &mut Evaluator) {
    ev.register("TextFind", text_find as NativeFn, Attributes::empty());
    ev.register("TextCount", text_count as NativeFn, Attributes::empty());
    ev.register("TextFilesWithMatch", text_files_with_match as NativeFn, Attributes::empty());
    ev.register("TextLines", text_lines as NativeFn, Attributes::empty());
    ev.register("TextReplace", text_replace as NativeFn, Attributes::empty());
    ev.register("TextDetectEncoding", text_detect_encoding as NativeFn, Attributes::empty());
    ev.register("TextSearch", text_search as NativeFn, Attributes::empty());
}

pub fn register_text_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str)->bool) {
    register_if(ev, pred, "TextFind", text_find as NativeFn, Attributes::empty());
    register_if(ev, pred, "TextCount", text_count as NativeFn, Attributes::empty());
    register_if(ev, pred, "TextFilesWithMatch", text_files_with_match as NativeFn, Attributes::empty());
    register_if(ev, pred, "TextLines", text_lines as NativeFn, Attributes::empty());
    register_if(ev, pred, "TextReplace", text_replace as NativeFn, Attributes::empty());
    register_if(ev, pred, "TextDetectEncoding", text_detect_encoding as NativeFn, Attributes::empty());
    register_if(ev, pred, "TextSearch", text_search as NativeFn, Attributes::empty());
}

fn ok_summary(files: usize, files_with: usize, total: usize, dur_ms: u128) -> Value {
    Value::Assoc([
        ("filesSearched".into(), Value::Integer(files as i64)),
        ("filesWithMatch".into(), Value::Integer(files_with as i64)),
        ("totalMatches".into(), Value::Integer(total as i64)),
        ("durationMs".into(), Value::Integer(dur_ms as i64)),
    ].into_iter().collect())
}

pub(crate) fn failure(tag: &str, msg: &str) -> Value {
    Value::Assoc([
        ("message".to_string(), Value::String(msg.to_string())),
        ("tag".to_string(), Value::String(tag.to_string())),
    ].into_iter().collect())
}

#[derive(Clone)]
enum PatternKind {
    Regex(Regex),
    #[cfg(feature = "text_multipattern")] Multi { ac: AhoCorasick, pats: Vec<String> },
}

fn parse_pattern(ev: &mut Evaluator, v: Value, opts: Option<&std::collections::HashMap<String, Value>>) -> Result<PatternKind, String> {
    let mut case_insensitive = false;
    let mut multiline = false;
    if let Some(o) = opts {
        if let Some(Value::Assoc(ctx)) = o.get("context") { let _ = ctx; }
        if let Some(Value::Boolean(b)) = o.get("caseInsensitive") { case_insensitive = *b; }
        if let Some(Value::Boolean(b)) = o.get("multiline") { multiline = *b; }
    }
    let (mut regex_src, is_regex, multi) = match ev.eval(v) {
        Value::Assoc(m) => {
            if let Some(Value::List(list)) = m.get("multi").or_else(|| m.get("Multi")) {
                let mut pats = Vec::new();
                for it in list { match ev.eval(it.clone()) { Value::String(s)=>pats.push(s), other=> return Err(format!("Invalid multi item: {}", lyra_core::pretty::format_value(&other))) } }
                return Ok({
                    #[cfg(feature = "text_multipattern")] {
                        PatternKind::Multi { ac: AhoCorasick::new(&pats).map_err(|e| e.to_string())?, pats }
                    }
                    #[cfg(not(feature = "text_multipattern"))] { return Err("multi pattern requires feature 'text_multipattern'".into()); }
                });
            } else if let Some(Value::String(s)) = m.get("regex").or_else(|| m.get("Regex")) { (s.clone(), true, false) }
            else if let Some(Value::String(s)) = m.get("literal").or_else(|| m.get("Literal")) { (regex::escape(s), false, false) }
            else { return Err("Missing 'regex' or 'literal' in pattern".into()); }
        }
        Value::String(s) | Value::Symbol(s) => (s, true, false),
        other => return Err(format!("Invalid pattern: {}", lyra_core::pretty::format_value(&other))),
    };
    if case_insensitive { regex_src = format!("(?i){}", regex_src); }
    if multiline && is_regex { regex_src = format!("(?m){}", regex_src); }
    let re = Regex::new(&regex_src).map_err(|e| e.to_string())?;
    Ok(PatternKind::Regex(re))
}

#[derive(Default)]
struct Ctx {
    before: usize,
    after: usize,
}

fn parse_opts(ev: &mut Evaluator, v: Option<Value>) -> (std::collections::HashMap<String, Value>, Ctx) {
    let m = v.map(|vv| ev.eval(vv)).and_then(|v| match v { Value::Assoc(m)=>Some(m), _=>None }).unwrap_or_default();
    let mut ctx = Ctx::default();
    if let Some(Value::Assoc(cm)) = m.get("context") {
        if let Some(Value::Integer(n)) = cm.get("before") { ctx.before = (*n).max(0) as usize; }
        if let Some(Value::Integer(n)) = cm.get("after") { ctx.after = (*n).max(0) as usize; }
    }
    (m, ctx)
}

#[derive(Clone)]
pub(crate) enum InputKind {
    Text(String),
    Files(Vec<String>),
}

pub(crate) fn parse_input(ev: &mut Evaluator, v: Value) -> Result<InputKind, String> {
    match ev.eval(v) {
        Value::Assoc(m) => {
            if let Some(Value::String(s)) = m.get("text").or_else(|| m.get("Text")) {
                Ok(InputKind::Text(s.clone()))
            } else if let Some(Value::String(p)) = m.get("path").or_else(|| m.get("Path")) {
                Ok(InputKind::Files(vec![p.clone()]))
            } else if let Some(Value::List(ps)) = m.get("paths").or_else(|| m.get("Paths")) {
                let mut out = Vec::new();
                for p in ps { match ev.eval(p.clone()) { Value::String(s)=>out.push(s), other=> return Err(format!("Invalid path: {}", lyra_core::pretty::format_value(&other))) } }
                Ok(InputKind::Files(out))
            } else if let Some(Value::String(g)) = m.get("glob").or_else(|| m.get("Glob")) {
                Ok(InputKind::Files(resolve_glob_paths(&[g.clone()], &m)))
            } else if let Some(Value::List(gs)) = m.get("globs").or_else(|| m.get("Globs")) {
                let mut pats = Vec::new();
                for p in gs { match ev.eval(p.clone()) { Value::String(s)=>pats.push(s), other=> return Err(format!("Invalid glob: {}", lyra_core::pretty::format_value(&other))) } }
                Ok(InputKind::Files(resolve_glob_paths(&pats, &m)))
            } else {
                Err("Provide {text}|{path}|{paths}".into())
            }
        }
        Value::String(s) => Ok(InputKind::Text(s)),
        other => Err(format!("Invalid input: {}", lyra_core::pretty::format_value(&other))),
    }
}

fn resolve_glob_paths(patterns: &[String], opts: &std::collections::HashMap<String, Value>) -> Vec<String> {
    #[cfg(feature = "text_glob")] {
        let mut ob = OverrideBuilder::new(".");
        for p in patterns { let _ = ob.add(p); }
        if let Some(Value::List(ex)) = opts.get("excludeGlobs").or_else(|| opts.get("ExcludeGlobs")) {
            for v in ex { if let Value::String(s) = v { let _ = ob.add(&format!("!{}", s)); } }
        }
        let ov = ob.build().unwrap();
        let mut wb = WalkBuilder::new(".");
        wb.overrides(ov);
        let follow = opts.get("followSymlinks").and_then(|v| if let Value::Boolean(b)=v { Some(*b) } else { None }).unwrap_or(false);
        wb.follow_links(follow);
        let hidden = opts.get("hidden").and_then(|v| if let Value::Boolean(b)=v { Some(*b) } else { None }).unwrap_or(false);
        wb.hidden(!hidden);
        let respect = opts.get("respectVCSIgnores").and_then(|v| if let Value::Boolean(b)=v { Some(*b) } else { None }).unwrap_or(true);
        wb.git_global(respect).git_ignore(respect).git_exclude(respect);
        let mut out = Vec::new();
        for r in wb.build() {
            if let Ok(de) = r { if de.file_type().map(|t| t.is_file()).unwrap_or(false) {
                if let Some(p) = de.path().to_str() { out.push(p.to_string()); }
            }}
        }
        out
    }
    #[cfg(not(feature = "text_glob"))] {
        patterns.iter().cloned().collect()
    }
}

fn is_probably_binary(bytes: &[u8]) -> bool {
    // simple heuristic: any NUL byte indicates binary
    bytes.iter().any(|&b| b == 0)
}

pub(crate) fn read_file_to_string(path: &str) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if is_probably_binary(&bytes) { return Err("BinaryFileSkipped".into()); }
    let s = String::from_utf8_lossy(&bytes).to_string();
    Ok(s)
}

fn text_find(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // TextFind(input, pattern, opts?) -> { matches: [...], summary: {...} }
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("TextFind".into())), args } }
    let start = std::time::Instant::now();
    let input = match parse_input(ev, args[0].clone()) { Ok(i)=>i, Err(e)=> return failure("Text::input", &e) };
    let (opts_map, ctx) = parse_opts(ev, args.get(2).cloned());
    let pat = match parse_pattern(ev, args[1].clone(), Some(&opts_map)) { Ok(p)=>p, Err(e)=> return failure("Text::pattern", &e) };

    let mut matches: Vec<Value> = Vec::new();
    let mut files_scanned = 0usize;
    let mut files_with = 0usize;
    let mut total = 0usize;
    let max_total = opts_map.get("maxMatches").and_then(|v| if let Value::Integer(n)=v { Some(*n as usize) } else { None });
    let max_per_file = opts_map.get("maxMatchesPerFile").and_then(|v| if let Value::Integer(n)=v { Some(*n as usize) } else { None });
    let timeout_ms = opts_map.get("timeoutMs").and_then(|v| if let Value::Integer(n)=v { Some(*n as u128) } else { None });

    match input {
        InputKind::Text(s) => {
            let (line_starts, lines) = line_indices(&s);
            for (lnum, line) in lines.iter().enumerate() {
                let base = line_starts[lnum];
                let mut per_line_count = 0usize;
                for (mstart, mend, caps) in iter_matches(&pat, line) {
                    total += 1;
                    per_line_count += 1;
                    let mut m = std::collections::HashMap::new();
                    m.insert("file".into(), Value::String(String::new()));
                    m.insert("line".into(), Value::String(line.clone()));
                    m.insert("lineNumber".into(), Value::Integer((lnum + 1) as i64));
                    m.insert("charRange".into(), Value::List(vec![Value::Integer(mstart as i64), Value::Integer(mend as i64)]));
                    m.insert("byteRange".into(), Value::List(vec![Value::Integer((base + mstart) as i64), Value::Integer((base + mend) as i64)]));
                    if !caps.is_empty() { m.insert("captures".into(), Value::List(caps.iter().map(|(s,e)| Value::List(vec![Value::Integer(*s as i64), Value::Integer(*e as i64)])).collect())); }
                    if ctx.before > 0 || ctx.after > 0 { m.insert("context".into(), extract_context(&lines, lnum, ctx.before, ctx.after)); }
                    matches.push(Value::Assoc(m));
                    if let Some(limit) = max_per_file { if per_line_count >= limit { break; } }
                    if let Some(limit) = max_total { if total >= limit { break; } }
                    if let Some(ms) = timeout_ms { if start.elapsed().as_millis() > ms { break; } }
                }
                if let Some(limit) = max_total { if total >= limit { break; } }
                if let Some(ms) = timeout_ms { if start.elapsed().as_millis() > ms { break; } }
            }
            files_scanned = 1;
            files_with = if total > 0 { 1 } else { 0 };
        }
        InputKind::Files(paths) => {
            for p in paths {
                files_scanned += 1;
                match read_file_to_string(&p) {
                    Ok(s) => {
                        let (line_starts, lines) = line_indices(&s);
                        let mut had = false;
                        let mut per_file_count = 0usize;
                        for (lnum, line) in lines.iter().enumerate() {
                            let base = line_starts[lnum];
                            for (mstart, mend, caps) in iter_matches(&pat, line) {
                                total += 1; had = true; per_file_count += 1;
                                let mut m = std::collections::HashMap::new();
                                m.insert("file".into(), Value::String(p.clone()));
                                m.insert("line".into(), Value::String(line.clone()));
                                m.insert("lineNumber".into(), Value::Integer((lnum + 1) as i64));
                                m.insert("charRange".into(), Value::List(vec![Value::Integer(mstart as i64), Value::Integer(mend as i64)]));
                                m.insert("byteRange".into(), Value::List(vec![Value::Integer((base + mstart) as i64), Value::Integer((base + mend) as i64)]));
                                if !caps.is_empty() { m.insert("captures".into(), Value::List(caps.iter().map(|(s,e)| Value::List(vec![Value::Integer(*s as i64), Value::Integer(*e as i64)])).collect())); }
                                if ctx.before > 0 || ctx.after > 0 { m.insert("context".into(), extract_context(&lines, lnum, ctx.before, ctx.after)); }
                                matches.push(Value::Assoc(m));
                                if let Some(limit) = max_per_file { if per_file_count >= limit { break; } }
                                if let Some(limit) = max_total { if total >= limit { break; } }
                                if let Some(ms) = timeout_ms { if start.elapsed().as_millis() > ms { break; } }
                            }
                            if let Some(limit) = max_total { if total >= limit { break; } }
                            if let Some(ms) = timeout_ms { if start.elapsed().as_millis() > ms { break; } }
                        }
                        if had { files_with += 1; }
                    }
                    Err(e) => { if e != "BinaryFileSkipped" { return failure("Text::read", &format!("{}: {}", p, e)); } }
                }
                if let Some(limit) = max_total { if total >= limit { break; } }
                if let Some(ms) = timeout_ms { if start.elapsed().as_millis() > ms { break; } }
            }
        }
    }

    Value::Assoc([
        ("matches".into(), Value::List(matches)),
        ("summary".into(), ok_summary(files_scanned, files_with, total, start.elapsed().as_millis())),
    ].into_iter().collect())
}

fn text_count(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // TextCount(input, pattern, opts?) -> { perFile:[{file,count}], total, durationMs }
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("TextCount".into())), args } }
    let start = std::time::Instant::now();
    let input = match parse_input(ev, args[0].clone()) { Ok(i)=>i, Err(e)=> return failure("Text::input", &e) };
    let (opts_map, _) = parse_opts(ev, args.get(2).cloned());
    let pat = match parse_pattern(ev, args[1].clone(), Some(&opts_map)) { Ok(p)=>p, Err(e)=> return failure("Text::pattern", &e) };
    let mut per_file: Vec<Value> = Vec::new();
    let mut total = 0usize;
    match input {
        InputKind::Text(s) => {
            let mut c = 0usize;
            for (_,_,_) in iter_matches(&pat, &s) { c += 1; }
            total = c;
            per_file.push(Value::Assoc([("file".into(), Value::String(String::new())),("count".into(), Value::Integer(c as i64))].into_iter().collect()));
        }
        InputKind::Files(paths) => {
            for p in paths {
                match read_file_to_string(&p) {
                    Ok(s) => { let mut c=0; for (_,_,_) in iter_matches(&pat, &s) { c+=1; } total += c; per_file.push(Value::Assoc([("file".into(), Value::String(p)), ("count".into(), Value::Integer(c as i64))].into_iter().collect())); }
                    Err(e) => { if e != "BinaryFileSkipped" { return failure("Text::read", &format!("{}: {}", p, e)); } }
                }
            }
        }
    }
    Value::Assoc([
        ("perFile".into(), Value::List(per_file)),
        ("total".into(), Value::Integer(total as i64)),
        ("durationMs".into(), Value::Integer(start.elapsed().as_millis() as i64)),
    ].into_iter().collect())
}

fn text_files_with_match(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // TextFilesWithMatch(input, pattern, opts?) -> { files: [path], durationMs }
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("TextFilesWithMatch".into())), args } }
    let start = std::time::Instant::now();
    let input = match parse_input(ev, args[0].clone()) { Ok(i)=>i, Err(e)=> return failure("Text::input", &e) };
    let (opts_map, _) = parse_opts(ev, args.get(2).cloned());
    let pat = match parse_pattern(ev, args[1].clone(), Some(&opts_map)) { Ok(p)=>p, Err(e)=> return failure("Text::pattern", &e) };
    let mut out: Vec<Value> = Vec::new();
    match input {
        InputKind::Text(s) => { if iter_matches(&pat, &s).next().is_some() { out.push(Value::String(String::new())); } }
        InputKind::Files(paths) => {
            for p in paths {
                match read_file_to_string(&p) {
                    Ok(s) => { if iter_matches(&pat, &s).next().is_some() { out.push(Value::String(p)); } }
                    Err(e) => { if e != "BinaryFileSkipped" { return failure("Text::read", &format!("{}: {}", p, e)); } }
                }
            }
        }
    }
    Value::Assoc([
        ("files".into(), Value::List(out)),
        ("durationMs".into(), Value::Integer(start.elapsed().as_millis() as i64)),
    ].into_iter().collect())
}

fn text_lines(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // TextLines(input, pattern, opts?) -> { lines: [{file,lineNumber,text,matches:[{charRange}]}], durationMs }
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("TextLines".into())), args } }
    let start = std::time::Instant::now();
    let input = match parse_input(ev, args[0].clone()) { Ok(i)=>i, Err(e)=> return failure("Text::input", &e) };
    let (opts_map, _) = parse_opts(ev, args.get(2).cloned());
    let pat = match parse_pattern(ev, args[1].clone(), Some(&opts_map)) { Ok(p)=>p, Err(e)=> return failure("Text::pattern", &e) };
    let mut lines_out: Vec<Value> = Vec::new();
    match input {
        InputKind::Text(s) => collect_lines_for_blob(None, &s, &pat, &mut lines_out),
        InputKind::Files(paths) => {
            for p in paths {
                match read_file_to_string(&p) {
                    Ok(s) => collect_lines_for_blob(Some(&p), &s, &pat, &mut lines_out),
                    Err(e) => { if e != "BinaryFileSkipped" { return failure("Text::read", &format!("{}: {}", p, e)); } }
                }
            }
        }
    }
    Value::Assoc([
        ("lines".into(), Value::List(lines_out)),
        ("durationMs".into(), Value::Integer(start.elapsed().as_millis() as i64)),
    ].into_iter().collect())
}

fn text_replace(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // TextReplace(input, pattern, replacement, {inPlace?:bool, dryRun?:bool, backupExt?:string})
    if args.len() < 3 { return Value::Expr { head: Box::new(Value::Symbol("TextReplace".into())), args } }
    let start = std::time::Instant::now();
    let input = match parse_input(ev, args[0].clone()) { Ok(i)=>i, Err(e)=> return failure("Text::input", &e) };
    let (opts_map, _) = parse_opts(ev, args.get(3).cloned());
    let pat = match parse_pattern(ev, args[1].clone(), Some(&opts_map)) { Ok(p)=>p, Err(e)=> return failure("Text::pattern", &e) };
    let replacement = match ev.eval(args[2].clone()) { Value::String(s)=>s, v=> lyra_core::pretty::format_value(&v) };
    let in_place = match opts_map.get("inPlace") { Some(Value::Boolean(b)) => *b, _ => false };
    let dry_run = match opts_map.get("dryRun") { Some(Value::Boolean(b)) => *b, _ => !in_place };
    let backup_ext = opts_map.get("backupExt").and_then(|v| if let Value::String(s)=v { Some(s.clone()) } else { None });

    let mut files_changed = 0usize;
    let mut total_repls = 0usize;
    let mut edits: Vec<Value> = Vec::new();

    match input {
        InputKind::Text(s) => {
            let (new_s, n) = replace_all(&pat, &s, &replacement);
            total_repls = n;
            edits.push(Value::Assoc([
                ("file".into(), Value::String(String::new())),
                ("replacements".into(), Value::Integer(n as i64)),
                ("preview".into(), Value::String(new_s)),
            ].into_iter().collect()));
        }
        InputKind::Files(paths) => {
            for p in paths {
                match read_file_to_string(&p) {
                    Ok(orig) => {
                        let (new_s, n) = replace_all(&pat, &orig, &replacement);
                        if n > 0 { files_changed += 1; total_repls += n; }
                        if !dry_run && n > 0 && in_place {
                            if let Some(ext) = &backup_ext { let _ = std::fs::write(format!("{}{}", p, ext), orig.as_bytes()); }
                            if let Err(e) = std::fs::write(&p, new_s.as_bytes()) { return failure("Text::write", &format!("{}: {}", p, e)); }
                            edits.push(Value::Assoc([("file".into(), Value::String(p)), ("replacements".into(), Value::Integer(n as i64))].into_iter().collect()));
                        } else {
                            edits.push(Value::Assoc([
                                ("file".into(), Value::String(p)),
                                ("replacements".into(), Value::Integer(n as i64)),
                                ("preview".into(), Value::String(new_s)),
                            ].into_iter().collect()));
                        }
                    }
                    Err(e) => { if e != "BinaryFileSkipped" { return failure("Text::read", &format!("{}: {}", p, e)); } }
                }
            }
        }
    }

    Value::Assoc([
        ("edits".into(), Value::List(edits)),
        ("filesChanged".into(), Value::Integer(files_changed as i64)),
        ("totalReplacements".into(), Value::Integer(total_repls as i64)),
        ("durationMs".into(), Value::Integer(start.elapsed().as_millis() as i64)),
    ].into_iter().collect())
}

fn line_indices(s: &str) -> (Vec<usize>, Vec<String>) {
    let mut starts = vec![0usize];
    let mut lines: Vec<String> = Vec::new();
    let mut last = 0usize;
    for (i, ch) in s.char_indices() {
        if ch == '\n' {
            lines.push(s[last..i].to_string());
            starts.push(i + 1);
            last = i + 1;
        }
    }
    if last <= s.len() { lines.push(s[last..].to_string()); }
    (starts, lines)
}

fn iter_matches<'a>(pat: &'a PatternKind, line: &'a str) -> Box<dyn Iterator<Item=(usize,usize,Vec<(usize,usize)>)> + 'a> {
    match pat {
        PatternKind::Regex(re) => {
            Box::new(re.find_iter(line).map(move |m| {
                let mut caps = Vec::new();
                if let Some(c) = re.captures(line) {
                    for i in 1..c.len() { if let Some(m) = c.get(i) { caps.push((m.start(), m.end())); } }
                }
                (m.start(), m.end(), caps)
            }))
        }
        #[cfg(feature = "text_multipattern")]
        PatternKind::Multi { ac, .. } => {
            Box::new(ac.find_iter(line.as_bytes()).map(|m| (m.start(), m.end(), Vec::new())))
        }
    }
}

fn extract_context(lines: &[String], idx: usize, before: usize, after: usize) -> Value {
    let start = idx.saturating_sub(before);
    let end = std::cmp::min(lines.len(), idx + 1 + after);
    let mut bm: Vec<Value> = Vec::new();
    let mut am: Vec<Value> = Vec::new();
    for i in start..idx { bm.push(Value::String(lines[i].clone())); }
    for i in (idx+1)..end { am.push(Value::String(lines[i].clone())); }
    Value::Assoc([
        ("before".into(), Value::List(bm)),
        ("after".into(), Value::List(am)),
    ].into_iter().collect())
}

fn collect_lines_for_blob(file: Option<&str>, s: &str, pat: &PatternKind, out: &mut Vec<Value>) {
    for (lnum, line) in s.lines().enumerate() {
        let mut ms: Vec<Value> = Vec::new();
        for (mstart, mend, _) in iter_matches(pat, line) {
            ms.push(Value::Assoc([
                ("charRange".into(), Value::List(vec![Value::Integer(mstart as i64), Value::Integer(mend as i64)])),
            ].into_iter().collect()));
        }
        if !ms.is_empty() {
            out.push(Value::Assoc([
                ("file".into(), match file { Some(p)=>Value::String(p.to_string()), None=>Value::String(String::new()) }),
                ("lineNumber".into(), Value::Integer((lnum + 1) as i64)),
                ("text".into(), Value::String(line.to_string())),
                ("matches".into(), Value::List(ms)),
            ].into_iter().collect()));
        }
    }
}

fn replace_all(pat: &PatternKind, s: &str, replacement: &str) -> (String, usize) {
    match pat {
        PatternKind::Regex(re) => {
            let mut count = 0usize;
            let out = re.replace_all(s, |_: &regex::Captures| { count += 1; replacement.to_string() });
            (out.into_owned(), count)
        }
        #[cfg(feature = "text_multipattern")]
        PatternKind::Multi { ac, .. } => {
            let mut count = 0usize;
            let mut out = String::with_capacity(s.len());
            let mut last = 0usize;
            for m in ac.find_iter(s.as_bytes()) {
                out.push_str(&s[last..m.start()]);
                out.push_str(replacement);
                last = m.end();
                count += 1;
            }
            out.push_str(&s[last..]);
            (out, count)
        }
    }
}

fn text_detect_encoding(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // TextDetectEncoding({paths|path}) -> { files: [{file, encoding}], durationMs }
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("TextDetectEncoding".into())), args } }
    let start = std::time::Instant::now();
    let input = match parse_input(ev, args[0].clone()) { Ok(i)=>i, Err(e)=> return failure("Text::input", &e) };
    let mut out: Vec<Value> = Vec::new();
    match input {
        InputKind::Text(s) => {
            let enc = detect_bytes_encoding(s.as_bytes());
            out.push(Value::Assoc([("file".into(), Value::String(String::new())), ("encoding".into(), Value::String(enc))].into_iter().collect()));
        }
        InputKind::Files(paths) => {
            for p in paths {
                match std::fs::read(&p) {
                    Ok(bytes) => {
                        let enc = detect_bytes_encoding(&bytes);
                        out.push(Value::Assoc([("file".into(), Value::String(p)), ("encoding".into(), Value::String(enc))].into_iter().collect()));
                    }
                    Err(e) => return failure("Text::read", &format!("{}: {}", p, e)),
                }
            }
        }
    }
    Value::Assoc([
        ("files".into(), Value::List(out)),
        ("durationMs".into(), Value::Integer(start.elapsed().as_millis() as i64)),
    ].into_iter().collect())
}

fn detect_bytes_encoding(bytes: &[u8]) -> String {
    // Try UTF-8 first
    if std::str::from_utf8(bytes).is_ok() { return "utf-8".into(); }
    #[cfg(feature = "text_encoding_detect")] {
        let mut det = EncodingDetector::new();
        det.feed(bytes, true);
        let enc = det.guess(None, true);
        return enc.name().to_ascii_lowercase();
    }
    // Fallback
    "unknown".into()
}

fn text_search(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // TextSearch(input, query, opts?) -> { engine: "grep"|"fuzzy"|"index", data: <underlying result>, durationMs }
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("TextSearch".into())), args } }
    let start = std::time::Instant::now();
    let raw_input = ev.eval(args[0].clone());
    let query = ev.eval(args[1].clone());
    let opts = args.get(2).map(|v| ev.eval(v.clone())).and_then(|v| match v { Value::Assoc(m)=>Some(m), _=>None }).unwrap_or_default();

    let task = opts.get("task").and_then(|v| if let Value::String(s)=v { Some(s.to_lowercase()) } else { None }).unwrap_or_else(|| "find".into());
    // Mode selection: explicit mode wins
    let explicit_mode = opts.get("mode").and_then(|v| if let Value::String(s)=v { Some(s.to_lowercase()) } else { None });
    let index_path_opt = opts.get("indexPath").and_then(|v| if let Value::String(s)=v { Some(s.clone()) } else { None })
        .or_else(|| if let Value::Assoc(m) = &raw_input { m.get("indexPath").and_then(|v| if let Value::String(s)=v { Some(s.clone()) } else { None }) } else { None });
    let fuzzy_flag = opts.get("fuzzy").and_then(|v| if let Value::Boolean(b)=v { Some(*b) } else { None }).unwrap_or(false);

    let engine = if task != "find" {
        // Non-find tasks use grep engine
        "grep".into()
    } else if let Some(m) = explicit_mode {
        m
    } else if index_path_opt.is_some() {
        "index".into()
    } else if fuzzy_flag {
        "fuzzy".into()
    } else {
        // Heuristics: if query is assoc with 'regex' or 'literal' or 'multi' => grep; plain string => grep; otherwise fuzzy if enabled and small-ish needle
        match &query {
            Value::Assoc(m) => {
                if m.contains_key("regex") || m.contains_key("Regex") || m.contains_key("literal") || m.contains_key("Literal") || m.contains_key("multi") || m.contains_key("Multi") { "grep".into() } else { "fuzzy".into() }
            }
            Value::String(s) | Value::Symbol(s) => {
                if s.len() <= 64 { "grep".into() } else { "fuzzy".into() }
            }
            _ => "grep".into(),
        }
    };

    let data = match engine.as_str() {
        "index" => {
            let idx = index_path_opt.unwrap();
            let q = match &query { Value::Assoc(m)=> m.get("q").and_then(|v| if let Value::String(s)=v { Some(s.clone()) } else { None }).unwrap_or_default(), Value::String(s)=>s.clone(), Value::Symbol(s)=>s.clone(), _=> String::new() };
            ev.eval(Value::expr(Value::Symbol("IndexSearch".into()), vec![Value::String(idx), Value::String(q)]))
        }
        "fuzzy" => {
            // Use text_fuzzy if available; decide Text vs Files
            match &raw_input {
                Value::Assoc(m) if m.contains_key("path") || m.contains_key("paths") || m.contains_key("Path") || m.contains_key("Paths") || m.contains_key("glob") || m.contains_key("globs") => {
                    let needle = match &query { Value::Assoc(ma)=> ma.get("needle").and_then(|v| if let Value::String(s)=v { Some(s.clone()) } else { None }).unwrap_or_else(|| String::new()), Value::String(s)=>s.clone(), Value::Symbol(s)=>s.clone(), _=> String::new() };
                    ev.eval(Value::expr(Value::Symbol("FuzzyFindInFiles".into()), vec![Value::Assoc(m.clone()), Value::String(needle)]))
                }
                Value::Assoc(m) if m.contains_key("text") || m.contains_key("Text") => {
                    let text = m.get("text").or_else(|| m.get("Text")).cloned().unwrap();
                    let needle = match &query { Value::Assoc(ma)=> ma.get("needle").and_then(|v| if let Value::String(s)=v { Some(s.clone()) } else { None }).unwrap_or_else(|| String::new()), Value::String(s)=>s.clone(), Value::Symbol(s)=>s.clone(), _=> String::new() };
                    ev.eval(Value::expr(Value::Symbol("FuzzyFindInText".into()), vec![text, Value::String(needle)]))
                }
                Value::String(s) => {
                    let needle = match &query { Value::Assoc(ma)=> ma.get("needle").and_then(|v| if let Value::String(s)=v { Some(s.clone()) } else { None }).unwrap_or_else(|| String::new()), Value::String(ss)=>ss.clone(), Value::Symbol(ss)=>ss.clone(), _=> String::new() };
                    ev.eval(Value::expr(Value::Symbol("FuzzyFindInText".into()), vec![Value::String(s.clone()), Value::String(needle)]))
                }
                other => failure("TextSearch::fuzzy", &format!("Unsupported input for fuzzy: {}", lyra_core::pretty::format_value(other))),
            }
        }
        _ => {
            // Grep-backed tasks
            let pattern = match &query {
                Value::Assoc(_) => query.clone(),
                Value::String(s) | Value::Symbol(s) => Value::Assoc([("literal".into(), Value::String(s.clone()))].into_iter().collect()),
                _ => Value::Assoc([].into_iter().collect()),
            };
            match task.as_str() {
                "count" => ev.eval(Value::expr(Value::Symbol("TextCount".into()), vec![raw_input.clone(), pattern, Value::Assoc(opts.clone())])),
                "lines" => ev.eval(Value::expr(Value::Symbol("TextLines".into()), vec![raw_input.clone(), pattern, Value::Assoc(opts.clone())])),
                "files" | "filesWithMatch" => ev.eval(Value::expr(Value::Symbol("TextFilesWithMatch".into()), vec![raw_input.clone(), pattern, Value::Assoc(opts.clone())])),
                "replace" => {
                    let repl = if let Value::Assoc(m) = &query { m.get("replacement").and_then(|v| if let Value::String(s)=v { Some(s.clone()) } else { None }) } else { None }
                        .or_else(|| opts.get("replacement").and_then(|v| if let Value::String(s)=v { Some(s.clone()) } else { None }))
                        .unwrap_or_default();
                    ev.eval(Value::expr(Value::Symbol("TextReplace".into()), vec![raw_input.clone(), pattern, Value::String(repl), Value::Assoc(opts.clone())]))
                }
                _ => ev.eval(Value::expr(Value::Symbol("TextFind".into()), vec![raw_input.clone(), pattern, Value::Assoc(opts.clone())])),
            }
        }
    };
    Value::Assoc([
        ("engine".into(), Value::String(engine)),
        ("data".into(), data),
        ("durationMs".into(), Value::Integer(start.elapsed().as_millis() as i64)),
    ].into_iter().collect())
}
