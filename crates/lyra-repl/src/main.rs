use anyhow::Result;
use colored::Colorize;
use lyra_runtime::{Evaluator, set_default_registrar};
use lyra_stdlib as stdlib;
use lyra_parser::Parser;
use rustyline::{Editor, Helper, error::ReadlineError, Context};
use rustyline::config::Configurer;
use rustyline::history::DefaultHistory;
use rustyline::completion::{Completer, Pair};
use rustyline::hint::Hinter;
use rustyline::highlight::Highlighter;
use rustyline::validate::{Validator, ValidationContext, ValidationResult};
use rustyline::{Event, EventHandler, ExternalPrinter};
use rustyline::{KeyEvent, KeyCode, Modifiers};
use rustyline::Cmd;
use lyra_core::value::Value;
use std::fs;

#[derive(Clone, Debug)]
struct BuiltinEntry { name: String, summary: String, attrs: Vec<String> }

#[allow(dead_code)]
struct ReplHelper {
    builtins: Vec<BuiltinEntry>,
    env_names: Vec<String>,
    doc_index: std::sync::Arc<DocIndex>,
    cfg: std::sync::Arc<std::sync::Mutex<ReplConfig>>,
}

impl Helper for ReplHelper {}

impl Highlighter for ReplHelper {}

impl Validator for ReplHelper {
    fn validate(&self, _ctx: &mut ValidationContext) -> rustyline::Result<ValidationResult> {
        // Simple bracket balance validator with string and comment awareness
        // Accepts when balanced; returns Incomplete to prompt for more lines
        if scan_unbalanced(_ctx.input()).is_some() { return Ok(ValidationResult::Incomplete); }
        Ok(ValidationResult::Valid(None))
    }
    fn validate_while_typing(&self) -> bool { false }
}

impl Hinter for ReplHelper {
    type Hint = String;
    fn hint(&self, line: &str, pos: usize, _ctx: &Context<'_>) -> Option<String> {
        // Show inline summary when the current token is a known builtin
        let (start, word) = current_symbol_token(line, pos);
        if start == pos || word.is_empty() { return None; }
        // Exact match or unique prefix
        let mut matches: Vec<&BuiltinEntry> = self.builtins.iter().filter(|b| b.name.starts_with(&word)).collect();
        if matches.is_empty() { return None; }
        if matches.len() > 1 {
            if let Some(exact) = matches.iter().find(|b| b.name == word) { matches = vec![exact]; } else { return None; }
        }
        let s = matches[0];
        if s.summary.is_empty() { None } else { Some(format!(" — {}", s.summary)) }
    }
}

impl Completer for ReplHelper {
    type Candidate = Pair;
    fn complete(&self, line: &str, pos: usize, _ctx: &Context<'_>) -> rustyline::Result<(usize, Vec<Pair>)> {
        let (start, word) = current_symbol_token(line, pos);
        if word.is_empty() {
            return Ok((pos, Vec::new()));
        }
        let mut cands: Vec<Pair> = Vec::new();
        for b in self.builtins.iter().filter(|b| b.name.starts_with(&word)) {
            let attrs = if b.attrs.is_empty() { String::new() } else { format!(" [{}]", b.attrs.join(", ")) };
            let display = if b.summary.is_empty() { format!("{}{}", b.name, attrs) } else { format!("{} — {}{}", b.name, b.summary, attrs) };
            // Auto-insert '[' for function-like builtins
            cands.push(Pair { display, replacement: format!("{}[", b.name) });
        }
        for n in self.env_names.iter().filter(|n| n.starts_with(&word)) {
            cands.push(Pair { display: format!("{} — (var)", n), replacement: n.clone() });
        }
        Ok((start, cands))
    }
}

fn current_symbol_token(line: &str, pos: usize) -> (usize, String) {
    let chars: Vec<char> = line.chars().collect();
    let mut i = pos.min(chars.len());
    while i > 0 {
        let c = chars[i-1];
        if c.is_alphanumeric() || c == '_' || c == '$' { i -= 1; } else { break; }
    }
    (i, chars[i..pos.min(chars.len())].iter().collect())
}

fn scan_unbalanced(s: &str) -> Option<(char, usize)> {
    // returns Some((open_char, position)) if any open bracket is unclosed
    let mut stack: Vec<(char, usize)> = Vec::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0usize;
    let mut in_string = false;
    let mut in_comment = 0usize; // allow nesting (* *)
    while i < chars.len() {
        let c = chars[i];
        // handle comment open/close
        if !in_string {
            if i+1 < chars.len() && chars[i]=='(' && chars[i+1]=='*' {
                in_comment += 1; i += 2; continue;
            }
            if in_comment>0 && i+1 < chars.len() && chars[i]=='*' && chars[i+1]==')' {
                in_comment -= 1; i += 2; continue;
            }
        }
        if in_comment>0 { i += 1; continue; }
        if c=='"' {
            // toggle string unless escaped
            let escaped = i>0 && chars[i-1]=='\\';
            if !escaped { in_string = !in_string; }
            i += 1; continue;
        }
        if in_string { i += 1; continue; }
        match c {
            '('|'['|'{' => stack.push((c, i)),
            ')'|']'|'}' => {
                if let Some((open,_)) = stack.pop() {
                    if !matches!((open,c), ('(',')')|('[',']')|('{','}')) {
                        // mismatched, treat as unbalanced
                        return Some((open, i));
                    }
                } else {
                    // extra closer; ignore but continue
                }
            }
            _=>{}
        }
        i += 1;
    }
    stack.last().cloned()
}

struct DocKeyHandler { printer: std::sync::Mutex<Box<dyn ExternalPrinter + Send>>, doc_index: std::sync::Arc<DocIndex>, cfg: std::sync::Arc<std::sync::Mutex<ReplConfig>> }

impl rustyline::ConditionalEventHandler for DocKeyHandler {
    fn handle(
        &self,
        _evt: &Event,
        _n: rustyline::RepeatCount,
        _positive: bool,
        ctx: &rustyline::EventContext,
    ) -> Option<Cmd> {
        let line = ctx.line();
        let pos = ctx.pos();
        let (_, word) = current_symbol_token(line, pos);
        let msg = if word.is_empty() {
            "(no symbol at cursor)".to_string()
        } else {
            build_doc_card_str_from_index(&self.doc_index, &word)
        };
        let use_pager = self.cfg.lock().map(|c| c.pager_on).unwrap_or(false);
        if use_pager { let _ = spawn_pager(&msg); }
        else if let Ok(mut p) = self.printer.lock() { let _ = p.print(msg); }
        Some(Cmd::Noop)
    }
}

fn main() -> Result<()> {
    // Build evaluator and register stdlib first (for builtin discovery)
    println!("{}", "Lyra REPL (prototype)".bright_yellow().bold());
    // Ensure spawned Evaluators (e.g., in Futures) inherit stdlib
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);

    // Discover builtins for completion and docs
    let builtins = discover_builtins(&mut ev);
    let doc_index = build_doc_index(&mut ev);
    let cfg = std::sync::Arc::new(std::sync::Mutex::new(ReplConfig { pager_on: false }));
    let helper = ReplHelper { builtins, env_names: Vec::new(), doc_index: doc_index.clone(), cfg: cfg.clone() };

    // Editor with helper for completion and hints
    let mut rl = Editor::<ReplHelper, DefaultHistory>::new()?;
    // Enable fuzzy completion when available
    rl.set_completion_type(rustyline::CompletionType::Fuzzy);
    rl.set_helper(Some(helper));
    // External printer for popups
    let printer = rl.create_external_printer().expect("external printer");
    // Bind F1 and Alt-h to popup docs for symbol at cursor (non-destructive)
    rl.bind_sequence(
        Event::from(KeyEvent(KeyCode::F(1), Modifiers::NONE)),
        EventHandler::Conditional(Box::new(DocKeyHandler { printer: std::sync::Mutex::new(Box::new(printer)), doc_index: doc_index.clone(), cfg: cfg.clone() }))
    );
    let printer2 = rl.create_external_printer().expect("external printer");
    rl.bind_sequence(
        Event::from(KeyEvent::alt('h')),
        EventHandler::Conditional(Box::new(DocKeyHandler { printer: std::sync::Mutex::new(Box::new(printer2)), doc_index, cfg: cfg.clone() }))
    );
    // Bind Shift-Enter and Ctrl-Enter to insert newline explicitly
    rl.bind_sequence(Event::from(KeyEvent(KeyCode::Enter, Modifiers::SHIFT)), Cmd::Newline);
    rl.bind_sequence(Event::from(KeyEvent(KeyCode::Enter, Modifiers::CTRL)), Cmd::Newline);
    // Print quick help of new features
    println!("{}", "Tip: F1/Alt-h for docs · Tab to complete · :more to expand".dimmed());
    // State for output rendering
    let mut last_output: Option<Value> = None;
    let mut truncate_cfg = TruncateCfg { max_list: 20, max_assoc: 20, max_string: 120 };
    let mut explain_on: bool = false;
    let mut profile_on: bool = false;
    let mut watches: Vec<(String, String)> = Vec::new();
    loop {
        // Refresh environment names for completion before prompt
        if let Some(h) = rl.helper_mut() { h.env_names = ev.env_keys(); }
        match rl.readline("> ") {
            Ok(line) => {
                let _ = rl.add_history_entry(line.as_str());
                let trimmed = line.trim();
                if trimmed.is_empty() { continue; }
                // Allow setting truncation config: :set truncate key=value
                if let Some(rest) = trimmed.strip_prefix(":set ") {
                    let parts: Vec<&str> = rest.split('=').map(|s| s.trim()).collect();
                    if parts.len()==2 {
                        match parts[0] {
                            "max_list" => if let Ok(n)=parts[1].parse::<usize>() { truncate_cfg.max_list=n },
                            "max_assoc" => if let Ok(n)=parts[1].parse::<usize>() { truncate_cfg.max_assoc=n },
                            "max_string" => if let Ok(n)=parts[1].parse::<usize>() { truncate_cfg.max_string=n },
                            _ => {}
                        }
                        println!("{} {}", "Set".green().bold(), rest);
                        continue;
                    }
                }
                // :doc Symbol -> show summary via builtin_help
                if let Some(rest) = trimmed.strip_prefix(":doc ") {
                    let sym = rest.trim();
                    render_doc_card(&mut ev, sym);
                    continue;
                }
                // :env -> list variable names
                if trimmed==":env" { render_env(&ev); continue; }
                // :funcs [filter] -> list builtins
                if let Some(rest) = trimmed.strip_prefix(":funcs") {
                    let filter = rest.trim();
                    render_funcs(&mut ev, if filter.is_empty() { None } else { Some(filter) });
                    continue;
                }
                // :sym Name -> show attributes and definition counts
                if let Some(rest) = trimmed.strip_prefix(":sym ") {
                    let name = rest.trim();
                    render_symbol_info(&mut ev, name);
                    continue;
                }
                // :defs Name -> show definitions (lhs -> rhs)
                if let Some(rest) = trimmed.strip_prefix(":defs ") {
                    let name = rest.trim();
                    render_defs(&mut ev, name, truncate_cfg);
                    continue;
                }
                // :explain on|off
                if let Some(rest) = trimmed.strip_prefix(":explain ") {
                    let v = rest.trim().eq_ignore_ascii_case("on");
                    explain_on = v;
                    println!("Explain {}", if v {"on".green()} else {"off".yellow()});
                    continue;
                }
                // :profile on|off
                if let Some(rest) = trimmed.strip_prefix(":profile ") {
                    let v = rest.trim().eq_ignore_ascii_case("on");
                    profile_on = v;
                    println!("Profile {}", if v {"on".green()} else {"off".yellow()});
                    continue;
                }
                // :pager on|off
                if let Some(rest) = trimmed.strip_prefix(":pager ") {
                    let on = rest.trim().eq_ignore_ascii_case("on");
                    if let Ok(mut c) = cfg.lock() { c.pager_on = on; }
                    println!("Pager {}", if on {"on".green()} else {"off".yellow()});
                    continue;
                }
                // :watch add|rm name expr
                if let Some(rest) = trimmed.strip_prefix(":watch ") {
                    let parts: Vec<&str> = rest.splitn(3, ' ').collect();
                    if parts.len()>=2 {
                        match parts[0] {
                            "add" if parts.len()==3 => { watches.push((parts[1].to_string(), parts[2].to_string())); println!("Added watch {}", parts[1]); },
                            "rm" => { let name=parts[1]; let before=watches.len(); watches.retain(|(n,_)| n!=name); println!("Removed {}", name); if watches.len()==before { println!("{}", "(not found)".dimmed()); } },
                            "list" => { for (n,e) in &watches { println!("{}: {}", n, e); } if watches.is_empty() { println!("{}", "(no watches)".dimmed()); } },
                            _ => println!("Usage: :watch add <name> <expr> | :watch rm <name> | :watch list"),
                        }
                        continue;
                    }
                }
                // :save file.json -> save env
                if let Some(rest) = trimmed.strip_prefix(":save ") {
                    let file = rest.trim(); if !file.is_empty() { if let Err(e)=save_env_json(&mut ev, file) { eprintln!("save error: {e}"); } else { println!("Saved {}", file); } }
                    continue;
                }
                // :load file.json -> load env (overwrites vars)
                if let Some(rest) = trimmed.strip_prefix(":load ") {
                    let file = rest.trim(); if !file.is_empty() { if let Err(e)=load_env_json(&mut ev, file) { eprintln!("load error: {e}"); } else { println!("Loaded {}", file); } }
                    continue;
                }
                // :more -> reprint last output fully
                if trimmed==":more" {
                    if let Some(v) = &last_output { println!("{}", format_value_color(v, None)); } else { println!("{}", "No previous output".dimmed()); }
                    continue;
                }
                // Simple help system: ?help or ?Symbol
                if trimmed.starts_with('?') {
                    handle_help(trimmed);
                    continue;
                }
                let mut p = Parser::from_source(&line);
                match p.parse_all() {
                    Ok(values) => {
                        for v in values {
                            let t0 = std::time::Instant::now();
                            let out = ev.eval(v);
                            let elapsed = t0.elapsed();
                            let suffix = result_suffix(&out, elapsed);
                            // Save last output
                            last_output = Some(out.clone());
                            println!("{}{}", format_value_color(&out, Some(truncate_cfg)), suffix);
                            if explain_on || profile_on { let steps = explain_steps(&mut ev, out); if explain_on { render_explain_steps(&steps); } if profile_on { render_profile_summary(&steps); } }
                            // Evaluate watches
                            if !watches.is_empty() {
                                for (n, expr_src) in &watches {
                                    let mut p2 = Parser::from_source(expr_src);
                                    if let Ok(vs) = p2.parse_all() { if let Some(expr) = vs.last() { let val = ev.eval(expr.clone()); println!("{} = {}", n.bold(), format_value_color(&val, Some(truncate_cfg))); } }
                                }
                            }
                        }
                    }
                    Err(e) => eprintln!("{} {}", "Error:".red().bold(), e),
                }
            }
            Err(ReadlineError::Interrupted) => { println!("^C"); continue; }
            Err(ReadlineError::Eof) => { println!("^D"); break; }
            Err(e) => { eprintln!("readline error: {e}"); break; }
        }
    }
    Ok(())
}

fn handle_help(q: &str) {
    let topic = q.trim_start_matches('?').trim();
    if topic.is_empty() || topic.eq_ignore_ascii_case("help") {
        let header = format!("{}", "Lyra REPL help".bright_green().bold());
        let body = "  - ?help: show this help\n  - ?Symbol: show a short description (e.g., ?Plus)\n  - Tab: autocomplete builtins; inline doc hints as you type\n  - Expressions use f[x, y], {a, b}, <|k->v|>\n  - Try: Explain[Plus[1, 2]] or Schema[<|\"a\"->1|>]";
        println!("{}\n{}", header, body);
        return;
    }
    let desc = builtin_help(topic);
    println!("{} {} — {}", "Help:".cyan().bold(), topic, desc);
}

fn builtin_help(sym: &str) -> &'static str {
    match sym {
        "Plus" => "Add numbers; Listable, Flat, Orderless.",
        "Times" => "Multiply numbers; Listable, Flat, Orderless.",
        "Minus" => "Subtract or unary negate.",
        "Divide" => "Divide two numbers.",
        "Power" => "Exponentiation (right-associative in parser).",
        "Map" => "Map a function over a list: Map[f, {..}].",
        "Replace" => "Replace first match by rule(s).",
        "ReplaceAll" => "Replace all matches by rule(s).",
        "ReplaceFirst" => "Replace first element(s) matching pattern.",
        "Set" => "Assignment: Set[symbol, value].",
        "With" => "Lexically bind symbols within a body.",
        "Schema" => "Return a minimal schema for a value/association.",
        "Explain" => "Explain evaluation; stub returns minimal trace info.",
        _ => "No documentation yet.",
    }
}

fn discover_builtins(ev: &mut Evaluator) -> Vec<BuiltinEntry> {
    // Try DescribeBuiltins[] then merge in our builtin_help summaries
    let mut entries: Vec<BuiltinEntry> = Vec::new();
    let resp = ev.eval(Value::Expr { head: Box::new(Value::Symbol("DescribeBuiltins".into())), args: vec![] });
    if let Value::List(items) = resp {
        for it in items {
            if let Value::Assoc(m) = it {
                if let Some(Value::String(name)) = m.get("name") {
                    let sum = builtin_help(name);
                    let attrs: Vec<String> = match m.get("attributes") { Some(Value::List(vs)) => vs.iter().filter_map(|v| if let Value::String(s)=v { Some(s.clone()) } else { None }).collect(), _ => Vec::new() };
                    entries.push(BuiltinEntry { name: name.clone(), summary: sum.to_string(), attrs });
                }
            }
        }
        entries.sort_by(|a,b| a.name.cmp(&b.name));
        entries.dedup_by(|a,b| a.name==b.name);
        return entries;
    }
    // Fallback to a minimal static set if DescribeBuiltins is unavailable
    let names = ["Plus","Times","Minus","Divide","Power","Replace","ReplaceAll","ReplaceFirst","Set","With","Schema","Explain"]; 
    for n in names { entries.push(BuiltinEntry { name: n.to_string(), summary: builtin_help(n).to_string(), attrs: Vec::new() }); }
    entries
}

#[derive(Clone, Copy)]
struct TruncateCfg { max_list: usize, max_assoc: usize, max_string: usize }

fn result_suffix(v: &Value, elapsed: std::time::Duration) -> String {
    let ms = (elapsed.as_secs_f64()*1000.0).round() as i64;
    let size = match v { Value::List(items) => Some(items.len()), Value::Assoc(m) => Some(m.len()), _=>None };
    match size { Some(n) => format!("  {}", format!("({} ms, {} items)", ms, n).dimmed()), None => format!("  {}", format!("({} ms)", ms).dimmed()) }
}

fn format_value_color(v: &Value, trunc: Option<TruncateCfg>) -> String {
    match v {
        Value::Integer(n) => format!("{}", n).bright_blue().to_string(),
        Value::Real(x) => format!("{}", x).bright_blue().to_string(),
        Value::BigReal(s) => s.bright_blue().to_string(),
        Value::Rational { num, den } => format!("{}/{}", num, den).bright_blue().to_string(),
        Value::Complex { re, im } => format!("{}{}{}i", format_value_color(re, trunc), "+".bright_blue(), format_value_color(im, trunc)).to_string(),
        Value::PackedArray { shape, .. } => format!("{}[{}]", "PackedArray".yellow(), shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x")).to_string(),
        Value::String(s) => {
            let max = trunc.map(|t| t.max_string).unwrap_or(usize::MAX);
            if s.len() > max { format!("\"{}…\"", &s[..max]).green().to_string() } else { format!("\"{}\"", s).green().to_string() }
        }
        Value::Symbol(s) => s.normal().to_string(),
        Value::Boolean(b) => if *b { "True".purple().to_string() } else { "False".purple().to_string() },
        Value::List(items) => {
            if let Some(table) = try_format_table(items, trunc) { return table; }
            let max = trunc.map(|t| t.max_list).unwrap_or(usize::MAX);
            let mut parts: Vec<String> = Vec::new();
            let take = items.len().min(max);
            for i in 0..take { parts.push(format_value_color(&items[i], trunc)); }
            if items.len()>take { parts.push(format!("{}", format!("… (+{} more)", items.len()-take).dimmed())); }
            format!("{{{}}}", parts.join(", "))
        }
        Value::Assoc(m) => {
            let max = trunc.map(|t| t.max_assoc).unwrap_or(usize::MAX);
            let mut keys: Vec<_> = m.keys().collect(); keys.sort();
            let take = keys.len().min(max);
            let mut parts: Vec<String> = Vec::new();
            for k in &keys[..take] {
                parts.push(format!("{} -> {}", format!("\"{}\"", k).cyan(), format_value_color(m.get(*k).unwrap(), trunc)));
            }
            if keys.len()>take { parts.push(format!("{}", format!("… (+{} more)", keys.len()-take).dimmed())); }
            format!("<|{}|>", parts.join(", "))
        }
        Value::Expr { head, args } => {
            let head_s = match &**head { Value::Symbol(s) => s.yellow().to_string(), other => format_value_color(other, trunc) };
            let args_s: Vec<String> = args.iter().map(|a| format_value_color(a, trunc)).collect();
            format!("{}[{}]", head_s, args_s.join(", "))
        }
        Value::Slot(None) => "#".to_string(),
        Value::Slot(Some(n)) => format!("#{}", n),
        Value::PureFunction { params: _, body } => format!("{}[{}]", "Function".yellow(), format_value_color(body, trunc)),
    }
}

fn try_format_table(items: &Vec<Value>, trunc: Option<TruncateCfg>) -> Option<String> {
    // Render a list of associations sharing same small key set as a simple table
    if items.is_empty() { return None; }
    let mut keys: Vec<String> = Vec::new();
    for v in items {
        if let Value::Assoc(m) = v {
            if keys.is_empty() { keys = m.keys().cloned().collect(); keys.sort(); }
            else if m.keys().count()!=keys.len() || !keys.iter().all(|k| m.contains_key(k)) { return None; }
        } else { return None; }
    }
    if keys.len()==0 || keys.len()>6 { return None; }
    let max_rows = trunc.map(|t| t.max_list).unwrap_or(usize::MAX);
    let rows_take = items.len().min(max_rows);
    // Compute column widths
    let mut colw: Vec<usize> = keys.iter().map(|k| k.len()).collect();
    let mut cells: Vec<Vec<String>> = Vec::new();
    for i in 0..rows_take {
        let m = if let Value::Assoc(m) = &items[i] { m } else { unreachable!() };
        let mut row: Vec<String> = Vec::new();
        for (j,k) in keys.iter().enumerate() {
            let s = format_value_color(m.get(k).unwrap(), Some(trunc.unwrap_or(TruncateCfg{max_list:8,max_assoc:8,max_string:60})));
            if s.len()>colw[j] { colw[j]=s.len(); }
            row.push(s);
        }
        cells.push(row);
    }
    // Header
    let mut out = String::new();
    out.push_str(&keys.iter().enumerate().map(|(j,k)| format!("{:<width$}", k.cyan(), width=colw[j])).collect::<Vec<_>>().join("  "));
    out.push('\n');
    // Rows
    for row in cells {
        let line = row.into_iter().enumerate().map(|(j,cell)| format!("{:<width$}", cell, width=colw[j])).collect::<Vec<_>>().join("  ");
        out.push_str(&line);
        out.push('\n');
    }
    if items.len()>rows_take { out.push_str(&format!("{}\n", format!("… (+{} more rows)", items.len()-rows_take).dimmed())); }
    Some(out)
}

fn render_doc_card(ev: &mut Evaluator, sym: &str) {
    // Lookup builtin details; fall back to summary only
    let resp = ev.eval(Value::Expr { head: Box::new(Value::Symbol("DescribeBuiltins".into())), args: vec![] });
    let mut attrs: Vec<String> = Vec::new();
    if let Value::List(items) = resp {
        for it in items {
            if let Value::Assoc(m) = it {
                if let Some(Value::String(name)) = m.get("name") {
                    if name == sym {
                        if let Some(Value::List(vs)) = m.get("attributes") { attrs = vs.iter().filter_map(|v| if let Value::String(s)=v { Some(s.clone()) } else { None }).collect(); }
                        break;
                    }
                }
            }
        }
    }
    let summary = builtin_help(sym);
    let header = format!("{} {}", sym.bold(), attrs.join(", "));
    println!("{}\n  {} {}", header, "Summary:".bold(), summary);
}

struct DocIndex { map: std::collections::HashMap<String, (String, Vec<String>)> }

fn build_doc_index(ev: &mut Evaluator) -> std::sync::Arc<DocIndex> {
    use std::collections::HashMap;
    let mut map: HashMap<String, (String, Vec<String>)> = HashMap::new();
    let resp = ev.eval(Value::Expr { head: Box::new(Value::Symbol("DescribeBuiltins".into())), args: vec![] });
    if let Value::List(items) = resp {
        for it in items {
            if let Value::Assoc(m) = it {
                if let Some(Value::String(name)) = m.get("name") {
                    let summary = builtin_help(name).to_string();
                    let attrs: Vec<String> = match m.get("attributes") { Some(Value::List(vs)) => vs.iter().filter_map(|v| if let Value::String(s)=v { Some(s.clone()) } else { None }).collect(), _=>Vec::new() };
                    map.insert(name.clone(), (summary, attrs));
                }
            }
        }
    }
    std::sync::Arc::new(DocIndex { map })
}

fn build_doc_card_str_from_index(index: &std::sync::Arc<DocIndex>, sym: &str) -> String {
    if let Some((summary, attrs)) = index.map.get(sym) {
        let header = if attrs.is_empty() { sym.bold().to_string() } else { format!("{} {}", sym.bold(), attrs.join(", ")) };
        format!("{}\n  {} {}", header, "Summary:".bold(), summary)
    } else {
        format!("{}\n  {} {}", sym.bold(), "Summary:".bold(), "No documentation yet.")
    }
}

fn render_env(ev: &Evaluator) {
    let mut names = ev.env_keys();
    names.sort();
    if names.is_empty() { println!("{}", "(no variables)".dimmed()); return; }
    let items = names.join(", ");
    println!("{} {}", "Env:".bold(), items);
}

fn render_funcs(ev: &mut Evaluator, filter: Option<&str>) {
    let resp = ev.eval(Value::Expr { head: Box::new(Value::Symbol("DescribeBuiltins".into())), args: vec![] });
    let mut rows: Vec<(String, Vec<String>)> = Vec::new();
    if let Value::List(items) = resp {
        for it in items {
            if let Value::Assoc(m) = it {
                if let Some(Value::String(name)) = m.get("name") {
                    if let Some(f) = filter { if !name.to_lowercase().contains(&f.to_lowercase()) { continue; } }
                    let attrs: Vec<String> = match m.get("attributes") { Some(Value::List(vs)) => vs.iter().filter_map(|v| if let Value::String(s)=v { Some(s.clone()) } else { None }).collect(), _=>Vec::new() };
                    rows.push((name.clone(), attrs));
                }
            }
        }
    }
    rows.sort_by(|a,b| a.0.cmp(&b.0));
    for (name, attrs) in rows.into_iter() {
        let attr = if attrs.is_empty() { String::new() } else { format!(" [{}]", attrs.join(", ")) };
        println!("{}{}", name, attr.dimmed());
    }
}

fn render_symbol_info(ev: &mut Evaluator, name: &str) {
    // Attributes via DescribeBuiltins
    let resp = ev.eval(Value::Expr { head: Box::new(Value::Symbol("DescribeBuiltins".into())), args: vec![] });
    let mut attrs: Vec<String> = Vec::new();
    if let Value::List(items) = resp {
        for it in items {
            if let Value::Assoc(m) = it {
                if let Some(Value::String(n)) = m.get("name") { if n==name { if let Some(Value::List(vs)) = m.get("attributes") { attrs = vs.iter().filter_map(|v| if let Value::String(s)=v { Some(s.clone()) } else { None }).collect(); } break; } }
            }
        }
    }
    // Definition counts
    let symv = Value::Symbol(name.to_string());
    let own = ev.eval(Value::Expr { head: Box::new(Value::Symbol("GetOwnValues".into())), args: vec![symv.clone()] });
    let down = ev.eval(Value::Expr { head: Box::new(Value::Symbol("GetDownValues".into())), args: vec![symv.clone()] });
    let up = ev.eval(Value::Expr { head: Box::new(Value::Symbol("GetUpValues".into())), args: vec![symv.clone()] });
    let sub = ev.eval(Value::Expr { head: Box::new(Value::Symbol("GetSubValues".into())), args: vec![symv.clone()] });
    let count = |v: &Value| -> usize { if let Value::List(items)=v { items.len() } else { 0 } };
    println!("{} {}", "Symbol:".bold(), name);
    if !attrs.is_empty() { println!("  {} {}", "Attributes:".bold(), attrs.join(", ")); }
    println!("  {} {}", "OwnValues:".bold(), count(&own));
    println!("  {} {}", "DownValues:".bold(), count(&down));
    println!("  {} {}", "UpValues:".bold(), count(&up));
    println!("  {} {}", "SubValues:".bold(), count(&sub));
}

fn render_defs(ev: &mut Evaluator, name: &str, trunc: TruncateCfg) {
    let symv = Value::Symbol(name.to_string());
    let kinds = [("OwnValues","GetOwnValues"),("DownValues","GetDownValues"),("UpValues","GetUpValues"),("SubValues","GetSubValues")];
    for (label, getter) in kinds {
        let vals = ev.eval(Value::Expr { head: Box::new(Value::Symbol(getter.into())), args: vec![symv.clone()] });
        match vals {
            Value::List(items) if !items.is_empty() => {
                println!("{} {}:", label.bold(), name);
                for r in items {
                    match r {
                        Value::Expr { head, args } if matches!(*head, Value::Symbol(ref s) if s=="Rule") && args.len()==2 => {
                            let lhs = &args[0]; let rhs = &args[1];
                            println!("  {} {} {}", format_value_color(lhs, Some(trunc)), "->".bold(), format_value_color(rhs, Some(trunc)));
                        }
                        other => println!("  {}", format_value_color(&other, Some(trunc))),
                    }
                }
            }
            _ => {}
        }
    }
}

fn explain_steps(ev: &mut Evaluator, expr: Value) -> Vec<Value> {
    let res = ev.eval(Value::Expr { head: Box::new(Value::Symbol("Explain".into())), args: vec![expr] });
    if let Value::Assoc(m) = res {
        if let Some(Value::List(steps)) = m.get("steps") {
            return steps.clone();
        }
    }
    Vec::new()
}

fn render_explain_steps(steps: &Vec<Value>) {
    for s in steps {
        if let Value::Assoc(sm) = s {
            let action = match sm.get("action") { Some(Value::String(x))=>x.as_str(), _=>"" };
            let head = match sm.get("head") { Some(Value::Symbol(x))=>x.as_str(), _=>"" };
            let extra = if let Some(Value::Assoc(data)) = sm.get("data") {
                if let Some(v) = data.get("count") { format!(" count={}", format_value_color(v, None)) }
                else if let Some(v) = data.get("finalOrder") { format!(" finalOrder={}", format_value_color(v, Some(TruncateCfg{max_list:6, max_assoc:6, max_string:80}))) }
                else { String::new() }
            } else { String::new() };
            println!("  {} {}{}", action.to_string().blue(), head.yellow(), extra.dimmed());
        }
    }
}

fn render_profile_summary(steps: &Vec<Value>) {
    use std::collections::HashMap;
    let mut actions: HashMap<String, usize> = HashMap::new();
    let mut heads: HashMap<String, usize> = HashMap::new();
    for s in steps {
        if let Value::Assoc(sm) = s {
            if let Some(Value::String(a)) = sm.get("action") { *actions.entry(a.clone()).or_insert(0)+=1; }
            if let Some(Value::Symbol(h)) = sm.get("head") { *heads.entry(h.clone()).or_insert(0)+=1; }
        }
    }
    let mut top_a: Vec<_> = actions.into_iter().collect(); top_a.sort_by(|a,b| b.1.cmp(&a.1));
    let mut top_h: Vec<_> = heads.into_iter().collect(); top_h.sort_by(|a,b| b.1.cmp(&a.1));
    let ta = top_a.into_iter().take(3).map(|(k,v)| format!("{}:{}", k, v)).collect::<Vec<_>>().join(", ");
    let th = top_h.into_iter().take(3).map(|(k,v)| format!("{}:{}", k, v)).collect::<Vec<_>>().join(", ");
    println!("{} {} | {} {}", "Profile actions:".bold(), ta, "heads:".bold(), th);
}

#[allow(dead_code)]
fn explain_steps_to_string(steps: &Vec<Value>) -> String {
    let mut out = String::new();
    for s in steps {
        if let Value::Assoc(sm) = s {
            let action = match sm.get("action") { Some(Value::String(x))=>x.as_str(), _=>"" };
            let head = match sm.get("head") { Some(Value::Symbol(x))=>x.as_str(), _=>"" };
            let extra = if let Some(Value::Assoc(data)) = sm.get("data") {
                if let Some(v) = data.get("count") { format!(" count={}", format_value_color(v, None)) }
                else if let Some(v) = data.get("finalOrder") { format!(" finalOrder={}", format_value_color(v, Some(TruncateCfg{max_list:6, max_assoc:6, max_string:80}))) }
                else { String::new() }
            } else { String::new() };
            out.push_str(&format!("  {} {}{}\n", action.to_string().blue(), head.yellow(), extra.dimmed()));
        }
    }
    out
}

struct ReplConfig { pager_on: bool }

fn spawn_pager(text: &str) -> std::io::Result<()> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let pager = std::env::var("PAGER").unwrap_or_else(|_| "less -R".to_string());
    let mut parts = pager.split_whitespace();
    let cmd = parts.next().unwrap_or("less");
    let args: Vec<&str> = parts.collect();
    let mut child = Command::new(cmd)
        .args(&args)
        .stdin(Stdio::piped())
        .spawn()?;
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(text.as_bytes());
    }
    let _ = child.wait();
    Ok(())
}

fn save_env_json(ev: &mut Evaluator, path: &str) -> anyhow::Result<()> {
    use std::collections::HashMap;
    let mut map: HashMap<String, Value> = HashMap::new();
    for name in ev.env_keys() { let val = ev.eval(Value::Symbol(name.clone())); map.insert(name, val); }
    let s = serde_json::to_string_pretty(&Value::Assoc(map))?;
    fs::write(path, s)?; Ok(())
}

fn load_env_json(ev: &mut Evaluator, path: &str) -> anyhow::Result<()> {
    let s = fs::read_to_string(path)?;
    let v: Value = serde_json::from_str(&s)?;
    if let Value::Assoc(m) = v {
        for (k,val) in m.into_iter() {
            // Set[k, val]
            let _ = ev.eval(Value::Expr { head: Box::new(Value::Symbol("Set".into())), args: vec![Value::Symbol(k), val] });
        }
    }
    Ok(())
}
