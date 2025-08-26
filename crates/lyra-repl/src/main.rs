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
use terminal_size::{Width, Height, terminal_size};

fn history_path() -> Option<std::path::PathBuf> {
    if let Ok(home) = std::env::var("HOME") {
        let mut p = std::path::PathBuf::from(home);
        p.push(".lyra_history");
        Some(p)
    } else { None }
}

#[derive(Clone, Debug)]
struct BuiltinEntry { name: String, summary: String, attrs: Vec<String> }

#[allow(dead_code)]
struct ReplHelper {
    builtins: Vec<BuiltinEntry>,
    env_names: Vec<String>,
    doc_index: std::sync::Arc<DocIndex>,
    cfg: std::sync::Arc<std::sync::Mutex<ReplConfig>>,
    pkg_exports: std::collections::HashMap<String, Vec<String>>, // package -> exports
    repl_cmds: Vec<String>,
    common_option_keys: Vec<String>,
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
        // Filesystem path completion if current token looks like a path or is quoted
        if let Some((start, word, quoted)) = current_path_token(line, pos) {
            let cands = complete_paths(&word);
            let pairs = cands.into_iter().map(|(disp, repl)| Pair { display: disp.clone(), replacement: if quoted { format!("{}", repl) } else { repl } }).collect();
            return Ok((start, pairs));
        }
        // REPL commands (":...")
        let trimmed = line[..pos].trim_start();
        if trimmed.starts_with(':') {
            let cmd_word = trimmed.trim_start_matches(':');
            let (start, word) = current_symbol_token(cmd_word, word_start_pos_in_substr(line, pos));
            let mut cands: Vec<Pair> = Vec::new();
            // commands
            for c in self.repl_cmds.iter().filter(|c| c.starts_with(&word)) {
                cands.push(Pair { display: format!(":{}", c), replacement: format!(":{} ", c) });
            }
            // sub-args for specific commands
            if cmd_word.starts_with("explain ") || cmd_word.starts_with("profile ") || cmd_word.starts_with("pager ") || cmd_word.starts_with("record ") {
                for v in ["on","off"] { if v.starts_with(&word) { cands.push(Pair { display: v.into(), replacement: format!("{}", v) }); } }
            }
            if cmd_word.starts_with("mode ") {
                for v in ["vi","emacs"] { if v.starts_with(&word) { cands.push(Pair { display: v.into(), replacement: v.into() }); } }
            }
            if cmd_word.starts_with("format ") {
                if cmd_word.starts_with("format assoc ") {
                    for v in ["auto","inline","pretty"] { if v.starts_with(&word) { cands.push(Pair { display: v.into(), replacement: v.into() }); } }
                } else if cmd_word.starts_with("format output ") {
                    for v in ["expr","json"] { if v.starts_with(&word) { cands.push(Pair { display: v.into(), replacement: v.into() }); } }
                } else {
                    for v in ["assoc","output"] { if v.starts_with(&word) { cands.push(Pair { display: v.into(), replacement: format!("{} ", v) }); } }
                }
            }
            return Ok((pos.saturating_sub(word.len()), cands));
        }

        // Using[...] context-aware options
        if let Some(ctx) = using_context(line, pos) {
            let mut cands: Vec<Pair> = Vec::new();
            match ctx {
                UsingCtx::PackageName(prefix) => {
                    // suggest loaded package names (keys in pkg_exports)
                    for name in self.pkg_exports.keys().filter(|n| n.starts_with(&prefix)) {
                        cands.push(Pair { display: format!("{} — (package)", name), replacement: name.to_string() });
                    }
                }
                UsingCtx::OptionKey(prefix) => {
                    for k in ["Import","Except"].into_iter() {
                        if k.starts_with(&prefix) { cands.push(Pair { display: k.to_string(), replacement: k.to_string() }); }
                    }
                }
                UsingCtx::ImportValue { pkg, prefix } => {
                    if let Some(exports) = self.pkg_exports.get(&pkg) {
                        for e in exports.iter().filter(|s| s.starts_with(&prefix)) {
                            cands.push(Pair { display: e.clone(), replacement: format!("\"{}\"", e) });
                        }
                        if "All".starts_with(&prefix) { cands.push(Pair { display: "All".into(), replacement: "All".into() }); }
                    } else {
                        if "All".starts_with(&prefix) { cands.push(Pair { display: "All".into(), replacement: "All".into() }); }
                    }
                }
            }
            return Ok((pos, cands));
        }
        // Association key completion inside <| ... |>
        if let Some(prefix) = assoc_key_prefix(line, pos) {
            let mut cands: Vec<Pair> = Vec::new();
            for k in &self.common_option_keys { if k.starts_with(&prefix) { cands.push(Pair { display: k.clone(), replacement: k.clone() }); } }
            return Ok((pos, cands));
        }

        // Default: builtins and env vars
        let (start, word) = current_symbol_token(line, pos);
        if word.is_empty() { return Ok((pos, Vec::new())); }
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

fn word_start_pos_in_substr(line: &str, pos: usize) -> usize {
    let mut i = pos.min(line.len());
    let bytes = line.as_bytes();
    while i>0 {
        let c = bytes[i-1] as char;
        if c.is_alphanumeric() || c=='_' || c=='$' { i-=1; } else { break; }
    }
    i
}

fn current_path_token(line: &str, pos: usize) -> Option<(usize, String, bool)> {
    // If inside quotes or token looks like a path (starts with ./, ../, ~/, or contains '/')
    let chars: Vec<char> = line.chars().collect();
    let mut i = pos.min(chars.len());
    let mut in_quotes = false;
    let mut j = i;
    // scan backwards for start quote on this token
    let mut k = i;
    while k>0 { if chars[k-1]=='"' { in_quotes = true; break; } if chars[k-1].is_whitespace() || chars[k-1]==',' { break; } k-=1; }
    // find token start
    while j>0 {
        let c = chars[j-1];
        if c.is_whitespace() || c==',' || c=='[' || c=='{' || c=='|' || c=='>' { break; }
        j-=1;
    }
    let token: String = chars[j..pos.min(chars.len())].iter().collect();
    let looks_path = token.starts_with("./") || token.starts_with("../") || token.starts_with("~/") || token.contains('/') || in_quotes;
    if looks_path { Some((j, token, in_quotes)) } else { None }
}

fn complete_paths(token: &str) -> Vec<(String, String)> {
    use std::path::{Path, PathBuf};
    let expanded = if token.starts_with("~/") {
        std::env::var("HOME").ok().map(|h| format!("{}{}", h, &token[1..])).unwrap_or(token.to_string())
    } else { token.to_string() };
    let (dir, base) = if expanded.ends_with('/') { (expanded.clone(), String::new()) } else { match expanded.rsplit_once('/') { Some((d,b))=>(format!("{}/", d), b.to_string()), None => ("./".into(), expanded.clone()) } };
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(Path::new(&dir)) {
        for e in rd.flatten() {
            let fname = e.file_name().to_string_lossy().to_string();
            if !fname.starts_with(&base) { continue; }
            let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
            let display = if is_dir { format!("{}/", fname) } else { fname.clone() };
            let replacement = format!("{}{}", if token.starts_with("/") || token.starts_with("~/") { dir.clone() } else { String::new() }, display.clone());
            out.push((display, replacement));
        }
    }
    out
}

enum UsingCtx { PackageName(String), OptionKey(String), ImportValue { pkg: String, prefix: String } }

fn using_context(line: &str, pos: usize) -> Option<UsingCtx> {
    // Very light-weight detection: Using[ pkg, <| ... |>
    let upos = line.find("Using[")?;
    if pos <= upos { return None; }
    // find argument region
    let after = &line[upos+6..];
    // If cursor is within first arg segment, suggest package name
    if let Some(brack) = after.find(']') {
        let args_str = &after[..brack];
        // cursor relative position inside args
        let rel = pos.saturating_sub(upos+6);
        // first arg ends at comma or end
        let first_end = args_str.find(',').unwrap_or(args_str.len());
        if rel <= first_end {
            // prefix for package name
            let prefix = args_str[..rel].trim().trim_matches('"').to_string();
            return Some(UsingCtx::PackageName(prefix));
        }
        // inside options association?
        if let Some(assoc_start) = args_str.find("<|") {
            if rel >= assoc_start {
                // detect key or value
                let before_cursor = &args_str[..rel];
                if before_cursor.rsplit("->").count() == 1 {
                    // likely typing a key
                    let key_prefix = before_cursor.rsplit(|c: char| c==',' || c=='<' || c=='|').next().unwrap_or("").trim().to_string();
                    return Some(UsingCtx::OptionKey(key_prefix));
                } else {
                    // typing a value; try to detect Import key and package name
                    let pkg = args_str[..first_end].trim().trim_matches('"').to_string();
                    let val_prefix = before_cursor.rsplit(|c: char| c==',' || c=='{' || c=='[' || c=='>' || c=='|').next().unwrap_or("").trim().trim_matches('"').to_string();
                    return Some(UsingCtx::ImportValue { pkg, prefix: val_prefix });
                }
            }
        }
    }
    None
}

fn assoc_key_prefix(line: &str, pos: usize) -> Option<String> {
    // If cursor is inside an association literal and currently typing a key (before ->)
    // Rough heuristic: find last '<|' before pos and ensure there's no matching '|>' between.
    let left = &line[..pos];
    let last_open = left.rfind("<|")?;
    if line[last_open..].contains("|>") { return None; }
    // Now, find last '->' before pos; if none, we are in key position
    let key_start = if let Some(after_open) = left[last_open..].rfind(|c: char| c==',' || c=='<' || c=='|') {
        last_open + after_open + 1
    } else { last_open + 2 };
    // prefix
    let prefix = left[key_start..].trim();
    if prefix.contains("->") { return None; }
    Some(prefix.to_string())
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

fn bracket_depth(s: &str) -> usize {
    let mut stack: Vec<char> = Vec::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0usize;
    let mut in_string = false;
    let mut in_comment = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if !in_string {
            if i+1 < chars.len() && chars[i]=='(' && chars[i+1]=='*' { in_comment += 1; i += 2; continue; }
            if in_comment>0 && i+1 < chars.len() && chars[i]=='*' && chars[i+1]==')' { in_comment -= 1; i += 2; continue; }
        }
        if in_comment>0 { i += 1; continue; }
        if c=='"' { let escaped = i>0 && chars[i-1]=='\\'; if !escaped { in_string = !in_string; } i+=1; continue; }
        if in_string { i+=1; continue; }
        match c { '('|'['|'{' => stack.push(c), ')'|']'|'}' => { let _ = stack.pop(); }, _=>{} }
        i+=1;
    }
    stack.len()
}

fn term_width_cols() -> usize {
    if let Some((Width(w), Height(_h))) = terminal_size() { w as usize } else { 100 }
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
    let cfg = std::sync::Arc::new(std::sync::Mutex::new(ReplConfig { pager_on: false, assoc_mode: AssocMode::Auto, print_mode: PrintMode::Expr }));
    let repl_cmds = vec![
        "help".into(), "env".into(), "funcs".into(), "sym".into(), "defs".into(),
        "explain".into(), "profile".into(), "pager".into(), "watch".into(), "save".into(),
        "load".into(), "more".into(), "json".into(), "history".into(), "using".into(),
        "edit".into(), "imports".into(), "loaded".into(), "exports".into(), "register-exports".into(),
        "newpkg".into(), "newmod".into(), "mode".into(), "set".into(),
    ];
    let common_option_keys = vec![
        "MaxThreads".into(), "TimeBudgetMs".into(), "Import".into(), "Except".into(),
        "replacement".into(), "inPlace".into(), "dryRun".into(), "backupExt".into(),
    ];
    let helper = ReplHelper { builtins, env_names: Vec::new(), doc_index: doc_index.clone(), cfg: cfg.clone(), pkg_exports: std::collections::HashMap::new(), repl_cmds, common_option_keys };

    // Editor with helper for completion and hints
    let mut rl = Editor::<ReplHelper, DefaultHistory>::new()?;
    rl.set_history_ignore_dups(true);
    rl.set_history_ignore_space(true);
    rl.set_edit_mode(rustyline::EditMode::Emacs);
    // Enable bracketed paste and fuzzy completion when available
    rl.enable_bracketed_paste(true);
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
    // Load history if present
    if let Some(p) = history_path() { let _ = rl.load_history(&p); }
    // Print quick help of new features
    println!("{}", "Tip: F1/Alt-h for docs · Tab to complete · :format assoc/output · :history · :help".dimmed());
    // State for output rendering
    let mut last_output: Option<Value> = None;
    let mut truncate_cfg = recommended_truncation_for_width(term_width_cols());
    let mut explain_on: bool = false;
    let mut profile_on: bool = false;
    let mut record_on: bool = false;
    let mut record_path: Option<std::path::PathBuf> = None;
    let mut in_counter: usize = 1;
    let mut watches: Vec<(String, String)> = Vec::new();
    let mut out_buf: Vec<Value> = Vec::new();
    let mut out_counter: usize = 0;
    loop {
        // Refresh environment names and package exports for completion before prompt
        if let Some(h) = rl.helper_mut() {
            h.env_names = ev.env_keys();
            h.pkg_exports = collect_pkg_exports(&mut ev);
        }
        match read_repl_input(&mut rl, &format!("In[{}]> ", in_counter)) {
            Ok(line) => {
                // Add to history once per submitted buffer
                let _ = rl.add_history_entry(line.as_str());
                let trimmed = line.trim();
                if trimmed.is_empty() { continue; }
                if record_on { append_transcript(&record_path, &format!("In[{}]: {}\n", in_counter, line)); }
                // Built-in lightweight commands (not expressions)
                if trimmed==":help" { print_repl_help(); continue; }
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
                // :format assoc auto|inline|pretty  OR  :format output expr|json
                if let Some(rest) = trimmed.strip_prefix(":format ") {
                    let mut parts = rest.split_whitespace();
                    let cat = parts.next().unwrap_or("");
                    let val = parts.next().unwrap_or("");
                    match (cat, val.to_lowercase().as_str()) {
                        ("assoc", "auto") => if let Ok(mut c)=cfg.lock() { c.assoc_mode = AssocMode::Auto; println!("format assoc -> auto"); },
                        ("assoc", "inline") => if let Ok(mut c)=cfg.lock() { c.assoc_mode = AssocMode::Inline; println!("format assoc -> inline"); },
                        ("assoc", "pretty") => if let Ok(mut c)=cfg.lock() { c.assoc_mode = AssocMode::Pretty; println!("format assoc -> pretty"); },
                        ("output", "expr") => if let Ok(mut c)=cfg.lock() { c.print_mode = PrintMode::Expr; println!("format output -> expr"); },
                        ("output", "json") => if let Ok(mut c)=cfg.lock() { c.print_mode = PrintMode::Json; println!("format output -> json"); },
                        _ => println!("Usage: :format assoc auto|inline|pretty | :format output expr|json"),
                    }
                    continue;
                }
                // :mode vi|emacs
                if let Some(rest) = trimmed.strip_prefix(":mode ") {
                    let m = rest.trim().to_lowercase();
                    match m.as_str() { "vi"=> rl.set_edit_mode(rustyline::EditMode::Vi), "emacs"=> rl.set_edit_mode(rustyline::EditMode::Emacs), _=>{} }
                    println!("Mode set to {}", m);
                    continue;
                }
                // :record on|off [file]
                if let Some(rest) = trimmed.strip_prefix(":record ") {
                    let mut parts = rest.split_whitespace();
                    match parts.next().unwrap_or("") {
                        "on" => {
                            let path = parts.next().map(|s| s.to_string()).unwrap_or_else(|| {
                                let mut p = std::env::temp_dir(); p.push("lyra_transcript.txt"); p.to_string_lossy().to_string()
                            });
                            record_path = Some(std::path::PathBuf::from(path.clone()));
                            record_on = true;
                            println!("Recording to {}", path);
                        }
                        "off" => { record_on = false; println!("Recording off"); }
                        _ => println!("Usage: :record on [file] | :record off"),
                    }
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
                    if let Some(v) = &last_output { println!("{}", format_value_color(v, None, AssocMode::Auto)); } else { println!("{}", "No previous output".dimmed()); }
                    continue;
                }
                // :json -> print last output as pretty JSON
                if trimmed==":json" {
                    if let Some(v) = &last_output {
                        match serde_json::to_string_pretty(v) { Ok(s)=>println!("{}", s), Err(e)=> eprintln!("json error: {e}") }
                    } else { println!("{}", "No previous output".dimmed()); }
                    continue;
                }
                // :history [N | /substr]
                if let Some(rest) = trimmed.strip_prefix(":history") {
                    let arg = rest.trim();
                    print_history(&rl, arg);
                    continue;
                }
                // :using name [--all|--import a,b] [--except x,y]
                if let Some(rest) = trimmed.strip_prefix(":using ") {
                    if let Err(e) = handle_using_cmd(&mut ev, rest) { eprintln!("{} {}", "Using:".red().bold(), e); }
                    continue;
                }
                // :edit [file] -> open EDITOR to write/eval a buffer
                if let Some(rest) = trimmed.strip_prefix(":edit") {
                    let path = rest.trim();
                    if let Err(e) = edit_and_eval(&mut ev, path) { eprintln!("edit error: {e}"); }
                    continue;
                }
                // Simple help system: ?help or ?Symbol
                if trimmed.starts_with('?') {
                    handle_help(trimmed);
                    continue;
                }
                // Echo input like In[n]:= expr
                println!("In[{}]:= {}", in_counter, line.trim());
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
                            // Maintain % binding and an OutList for retrieval via OutList[[n]]
                            out_counter += 1;
                            out_buf.push(out.clone());
                            let _ = ev.eval(Value::Expr { head: Box::new(Value::Symbol("Set".into())), args: vec![Value::Symbol("%".into()), out.clone()] });
                            let _ = ev.eval(Value::Expr { head: Box::new(Value::Symbol("Set".into())), args: vec![Value::Symbol("OutList".into()), Value::List(out_buf.clone())] });
                            let (assoc_mode, print_mode) = { if let Ok(c)=cfg.lock() { (c.assoc_mode, c.print_mode) } else { (AssocMode::Auto, PrintMode::Expr) } };
                            let rendered = match print_mode { PrintMode::Expr => format_value_color(&out, Some(truncate_cfg), assoc_mode), PrintMode::Json => serde_json::to_string_pretty(&out).unwrap_or_else(|_| "<json error>".into()) };
                            let out_line = format!("Out[{}]= {}{}", out_counter, rendered, suffix);
                            println!("{}", out_line);
                            if record_on { append_transcript(&record_path, &format!("{}\n", strip_ansi(&out_line))); }
                            if explain_on || profile_on { let steps = explain_steps(&mut ev, out); if explain_on { render_explain_steps(&steps); } if profile_on { render_profile_summary(&steps); } }
                            // Evaluate watches
                            if !watches.is_empty() {
                                for (n, expr_src) in &watches {
                                    let mut p2 = Parser::from_source(expr_src);
                                    if let Ok(vs) = p2.parse_all() { if let Some(expr) = vs.last() { let val = ev.eval(expr.clone()); println!("{} = {}", n.bold(), format_value_color(&val, Some(truncate_cfg), AssocMode::Auto)); } }
                                }
                            }
                        }
                    }
                    Err(e) => render_parse_error(&rl, &line, &e.to_string()),
                }
            }
            Err(ReadlineError::Interrupted) => { println!("^C"); continue; }
            Err(ReadlineError::Eof) => { println!("^D"); break; }
            Err(e) => { eprintln!("readline error: {e}"); break; }
        }
        in_counter += 1;
    }
    // Save history on exit
    if let Some(p) = history_path() { let _ = rl.save_history(&p); }
    Ok(())
}

fn read_repl_input(rl: &mut Editor<ReplHelper, DefaultHistory>, first_prompt: &str) -> rustyline::Result<String> {
    let mut buf = String::new();
    let mut first = true;
    loop {
        let prompt = if first { first_prompt } else { "… " };
        first = false;
        let indent = {
            let depth = bracket_depth(&buf);
            if depth == 0 { String::new() } else { "  ".repeat(depth) }
        };
        let line = if indent.is_empty() { rl.readline(prompt)? } else { rl.readline_with_initial(prompt, (&indent, ""))? };
        // ctrl-d at empty second prompt should bubble up
        buf.push_str(&line);
        if scan_unbalanced(&buf).is_some() { buf.push('\n'); continue; }
        break;
    }
    Ok(buf)
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

fn print_repl_help() {
    let lines = [
        ":help                — show REPL commands",
        ":env                 — list variables",
        ":funcs [filter]      — list builtins",
        ":sym Name            — show symbol info",
        ":defs Name           — list definitions",
        ":explain on|off      — toggle step tracing",
        ":profile on|off      — toggle profile summary",
        ":pager on|off        — toggle pager for docs",
        ":format assoc auto|inline|pretty",
        ":format output expr|json",
        ":watch add/rm/list   — manage watches",
        ":save file.json      — save environment",
        ":load file.json      — load environment",
        ":more                — expand last output",
        ":json                — print last output as JSON",
        ":edit [file]         — open $EDITOR to edit/eval",
        "OutList[[n]]        — retrieve nth output; % is last",
        ":history [N|/substr] — show history",
        ":using name [--all|--import a,b] [--except x,y]",
    ];
    println!("{}", "Lyra REPL commands".bright_green().bold());
    for l in lines { println!("  {}", l); }
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

fn format_value_color(v: &Value, trunc: Option<TruncateCfg>, assoc_mode: AssocMode) -> String {
    match v {
        Value::Integer(n) => format!("{}", n).bright_blue().to_string(),
        Value::Real(x) => format!("{}", x).bright_blue().to_string(),
        Value::BigReal(s) => s.bright_blue().to_string(),
        Value::Rational { num, den } => format!("{}/{}", num, den).bright_blue().to_string(),
        Value::Complex { re, im } => format!("{}{}{}i", format_value_color(re, trunc, assoc_mode), "+".bright_blue(), format_value_color(im, trunc, assoc_mode)).to_string(),
        Value::PackedArray { shape, .. } => format!("{}[{}]", "PackedArray".yellow(), shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x")).to_string(),
        Value::String(s) => {
            let max = trunc.map(|t| t.max_string).unwrap_or(usize::MAX);
            if s.len() > max { format!("\"{}…\"", &s[..max]).green().to_string() } else { format!("\"{}\"", s).green().to_string() }
        }
        Value::Symbol(s) => s.normal().to_string(),
        Value::Boolean(b) => if *b { "True".purple().to_string() } else { "False".purple().to_string() },
        Value::List(items) => {
            if let Some(table) = try_format_table(items, trunc, assoc_mode) { return table; }
            let max = trunc.map(|t| t.max_list).unwrap_or(usize::MAX);
            let mut parts: Vec<String> = Vec::new();
            let take = items.len().min(max);
            for i in 0..take { parts.push(format_value_color(&items[i], trunc, assoc_mode)); }
            if items.len()>take { parts.push(format!("{}", format!("… (+{} more)", items.len()-take).dimmed())); }
            let inline = format!("{{{}}}", parts.join(", "));
            let cols = term_width_cols();
            let pretty = items.len()>=8 || inline.len()>cols.saturating_sub(4);
            if !pretty { inline } else { format_list_pretty(&parts, cols) }
        }
        Value::Assoc(m) => {
            let max = trunc.map(|t| t.max_assoc).unwrap_or(usize::MAX);
            let mut keys: Vec<_> = m.keys().collect(); keys.sort();
            let take = keys.len().min(max);
            // First build inline; if too long or many keys, pretty-print multiline
            let mut inline_parts: Vec<String> = Vec::new();
            for k in &keys[..take] {
                inline_parts.push(format!("{} -> {}", format!("\"{}\"", k).cyan(), format_value_color(m.get(*k).unwrap(), trunc, assoc_mode)));
            }
            if keys.len()>take { inline_parts.push(format!("{}", format!("… (+{} more)", keys.len()-take).dimmed())); }
            let inline = format!("<|{}|>", inline_parts.join(", "));
            let cols = term_width_cols();
            let pretty = match assoc_mode { AssocMode::Inline => false, AssocMode::Pretty => true, AssocMode::Auto => keys.len()>=4 || inline.len()>cols.saturating_sub(4) };
            if !pretty { return inline; }
            // Pretty multiline with aligned arrows
            let key_strs: Vec<String> = keys[..take].iter().map(|k| format!("\"{}\"", k)).collect();
            let keyw = key_strs.iter().map(|s| s.len()).max().unwrap_or(0).min(60);
            let mut out = String::new();
            out.push_str("<|\n");
            for (i, k) in keys[..take].iter().enumerate() {
                let kq = format!("\"{}\"", k).cyan();
                let pad = if keyw>key_strs[i].len() { keyw-key_strs[i].len() } else { 0 };
                let val = format_value_color(m.get(*k).unwrap(), trunc, assoc_mode);
                out.push_str(&format!("  {:<width$} -> {}", kq, val, width=keyw));
                if i+1 < take { out.push(','); }
                out.push('\n');
            }
            if keys.len()>take { out.push_str(&format!("  {}\n", format!("… (+{} more)", keys.len()-take).dimmed())); }
            out.push_str("|>");
            out
        }
        Value::Expr { head, args } => {
            let head_s = match &**head { Value::Symbol(s) => s.yellow().to_string(), other => format_value_color(other, trunc, assoc_mode) };
            let args_s: Vec<String> = args.iter().map(|a| format_value_color(a, trunc, assoc_mode)).collect();
            format!("{}[{}]", head_s, args_s.join(", "))
        }
        Value::Slot(None) => "#".to_string(),
        Value::Slot(Some(n)) => format!("#{}", n),
        Value::PureFunction { params: _, body } => format!("{}[{}]", "Function".yellow(), format_value_color(body, trunc, assoc_mode)),
    }
}

fn try_format_table(items: &Vec<Value>, trunc: Option<TruncateCfg>, assoc_mode: AssocMode) -> Option<String> {
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
            let s = format_value_color(m.get(k).unwrap(), Some(trunc.unwrap_or(TruncateCfg{max_list:8,max_assoc:8,max_string:60})), assoc_mode);
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

fn format_list_pretty(parts: &Vec<String>, cols: usize) -> String {
    // Wrap items across lines to fit within terminal width
    let mut out = String::new();
    out.push('{'); out.push('\n');
    let indent = "  ";
    let wrap_width = cols.saturating_sub(4).max(40);
    let mut line = String::new();
    for (i, p) in parts.iter().enumerate() {
        let item = if i+1 < parts.len() { format!("{}{}", p, ", ") } else { p.clone() };
        if indent.len() + line.len() + item.len() > wrap_width {
            out.push_str(indent);
            out.push_str(&line);
            out.push('\n');
            line.clear();
        }
        line.push_str(&item);
    }
    if !line.is_empty() { out.push_str(indent); out.push_str(&line); out.push('\n'); }
    out.push('}');
    out
}

fn recommended_truncation_for_width(cols: usize) -> TruncateCfg {
    // Simple heuristic based on terminal width
    let max_string = cols.saturating_sub(40).max(60);
    let max_list = if cols < 90 { 12 } else if cols < 120 { 20 } else { 30 };
    let max_assoc = if cols < 90 { 12 } else if cols < 120 { 20 } else { 30 };
    TruncateCfg { max_list, max_assoc, max_string }
}

fn render_parse_error(rl: &Editor<ReplHelper, DefaultHistory>, line: &str, msg: &str) {
    eprintln!("{} {}", "Error:".red().bold(), msg);
    // If clearly unbalanced, hint the missing closer
    if let Some((open, _pos)) = scan_unbalanced(line) {
        let closer = match open { '('=>')', '['=>']', '{'=>'}', _=>'?'};
        eprintln!("{} {} -> missing '{}'", "Hint:".yellow().bold(), "Unclosed bracket", closer);
    }
    // Suggest similar builtins/vars based on trailing token
    let (_, word) = current_symbol_token(line, line.len());
    if word.len() >= 2 {
        if let Some(h) = rl.helper() {
            let mut cands: Vec<(&str, f64)> = Vec::new();
            for b in &h.builtins { cands.push((&b.name, strsim::jaro_winkler(&word, &b.name))); }
            for n in &h.env_names { cands.push((&n, strsim::jaro_winkler(&word, n))); }
            cands.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut shown = 0;
            for (name, score) in cands.into_iter() { if score > 0.85 { if shown==0 { eprintln!("{} {}:", "Maybe:".yellow().bold(), word); } eprintln!("  {}", name); shown+=1; if shown>=3 { break; } } }
        }
    }
}

fn print_history(rl: &Editor<ReplHelper, DefaultHistory>, arg: &str) {
    let hist = rl.history();
    let entries: Vec<_> = hist.iter().collect();
    if entries.is_empty() { println!("{}", "(no history)".dimmed()); return; }
    if arg.starts_with('/') {
        let pat = arg.trim_start_matches('/').to_lowercase();
        for (i, e) in entries.iter().enumerate() {
            let s = e.to_string();
            if s.to_lowercase().contains(&pat) { println!("{:>4}: {}", i+1, s); }
        }
        return;
    }
    let n: usize = if arg.is_empty() { 20 } else { arg.parse().unwrap_or(20) };
    let start = entries.len().saturating_sub(n);
    for (i, e) in entries.iter().enumerate().skip(start) {
        println!("{:>4}: {}", i+1, e);
    }
}

fn handle_using_cmd(ev: &mut Evaluator, rest: &str) -> anyhow::Result<()> {
    let mut name = String::new();
    let mut import_all = false;
    let mut imports: Vec<String> = Vec::new();
    let mut excepts: Vec<String> = Vec::new();
    let mut parts = rest.split_whitespace();
    if let Some(n) = parts.next() { name = n.trim_matches('"').to_string(); }
    for tok in parts {
        if tok.eq_ignore_ascii_case("--all") { import_all = true; continue; }
        if tok.eq_ignore_ascii_case("--import") { /* handled via next token */ continue; }
        if tok.eq_ignore_ascii_case("--except") { /* handled via next token */ continue; }
        // comma lists for previous flag
        if import_all { /* already set; ignore */ }
        else if imports.is_empty() && tok.starts_with("--import") { /* --import a,b */
            let v = tok.trim_start_matches("--import").trim_start_matches('='); if !v.is_empty() { imports = v.split(',').map(|s| s.trim().trim_matches('"').to_string()).filter(|s| !s.is_empty()).collect(); }
        } else if excepts.is_empty() && tok.starts_with("--except") {
            let v = tok.trim_start_matches("--except").trim_start_matches('='); if !v.is_empty() { excepts = v.split(',').map(|s| s.trim().trim_matches('"').to_string()).filter(|s| !s.is_empty()).collect(); }
        } else {
            // also allow bare a,b after --import or --except in previous token; simple heuristic
            if imports.is_empty() && rest.contains("--import ") {
                if let Some(seg) = rest.split("--import ").nth(1) { let seg = seg.split_whitespace().next().unwrap_or(""); if !seg.is_empty() { imports = seg.split(',').map(|s| s.trim().trim_matches('"').to_string()).filter(|s| !s.is_empty()).collect(); } }
            }
            if excepts.is_empty() && rest.contains("--except ") {
                if let Some(seg) = rest.split("--except ").nth(1) { let seg = seg.split_whitespace().next().unwrap_or(""); if !seg.is_empty() { excepts = seg.split(',').map(|s| s.trim().trim_matches('"').to_string()).filter(|s| !s.is_empty()).collect(); } }
            }
        }
    }
    if name.is_empty() { anyhow::bail!("usage: :using name [--all|--import a,b] [--except x,y]"); }
    let call = if import_all { format!("Using[\"{}\", <|Import->All|>]", name) }
        else if !imports.is_empty() && excepts.is_empty() { format!("Using[\"{}\", <|Import->{{{}}}|>]", name, imports.iter().map(|s| format!("\"{}\"", s)).collect::<Vec<_>>().join(", ")) }
        else if !imports.is_empty() && !excepts.is_empty() { format!("Using[\"{}\", <|Import->{{{}}}, Except->{{{}}}|>]", name, imports.iter().map(|s| format!("\"{}\"", s)).collect::<Vec<_>>().join(", "), excepts.iter().map(|s| format!("\"{}\"", s)).collect::<Vec<_>>().join(", ")) }
        else { format!("Using[\"{}\"]", name) };
    let mut p = Parser::from_source(&call);
    match p.parse_all() { Ok(vs) => { for v in vs { let _ = ev.eval(v); } Ok(()) }, Err(e) => Err(anyhow::anyhow!(e.to_string())) }
}

fn append_transcript(path: &Option<std::path::PathBuf>, text: &str) {
    if let Some(p) = path {
        let _ = std::fs::OpenOptions::new().create(true).append(true).open(p).and_then(|mut f| std::io::Write::write_all(&mut f, text.as_bytes()));
    }
}

fn strip_ansi(s: &str) -> String {
    // very simple strip for color codes: remove \x1b[ ... m sequences
    let mut out = String::new();
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i]==0x1b && i+1 < bytes.len() && bytes[i+1]==b'[' {
            i+=2; while i<bytes.len() && (bytes[i] as char)!='m' { i+=1; } if i<bytes.len() { i+=1; }
        } else { out.push(bytes[i] as char); i+=1; }
    }
    out
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
                            println!("  {} {} {}", format_value_color(lhs, Some(trunc), AssocMode::Auto), "->".bold(), format_value_color(rhs, Some(trunc), AssocMode::Auto));
                        }
                        other => println!("  {}", format_value_color(&other, Some(trunc), AssocMode::Auto)),
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
                if let Some(v) = data.get("count") { format!(" count={}", format_value_color(v, None, AssocMode::Auto)) }
                else if let Some(v) = data.get("finalOrder") { format!(" finalOrder={}", format_value_color(v, Some(TruncateCfg{max_list:6, max_assoc:6, max_string:80}), AssocMode::Auto)) }
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
                if let Some(v) = data.get("count") { format!(" count={}", format_value_color(v, None, AssocMode::Auto)) }
                else if let Some(v) = data.get("finalOrder") { format!(" finalOrder={}", format_value_color(v, Some(TruncateCfg{max_list:6, max_assoc:6, max_string:80}), AssocMode::Auto)) }
                else { String::new() }
            } else { String::new() };
            out.push_str(&format!("  {} {}{}\n", action.to_string().blue(), head.yellow(), extra.dimmed()));
        }
    }
    out
}

#[derive(Clone, Copy)]
enum AssocMode { Auto, Inline, Pretty }

#[derive(Clone, Copy)]
enum PrintMode { Expr, Json }

struct ReplConfig { pager_on: bool, assoc_mode: AssocMode, print_mode: PrintMode }

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

fn collect_pkg_exports(ev: &mut Evaluator) -> std::collections::HashMap<String, Vec<String>> {
    use std::collections::HashMap;
    let mut out: HashMap<String, Vec<String>> = HashMap::new();
    let loaded = ev.eval(Value::Expr { head: Box::new(Value::Symbol("LoadedPackages".into())), args: vec![] });
    if let Value::Assoc(m) = loaded {
        for name in m.keys() {
            let q = Value::Expr { head: Box::new(Value::Symbol("PackageExports".into())), args: vec![Value::String(name.clone())] };
            let ex = ev.eval(q);
            if let Value::List(vs) = ex {
                let syms: Vec<String> = vs.into_iter().filter_map(|v| if let Value::String(s)=v { Some(s) } else { None }).collect();
                out.insert(name.clone(), syms);
            }
        }
    }
    out
}

fn edit_and_eval(ev: &mut Evaluator, path: &str) -> anyhow::Result<()> {
    use std::io::Write;
    use std::process::Command;
    let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vi".to_string());
    let file_path = if path.is_empty() {
        let mut p = std::env::temp_dir();
        p.push("lyra_edit.lyra");
        if !p.exists() { let _ = std::fs::File::create(&p); }
        p
    } else { std::path::PathBuf::from(path) };
    // Launch editor and wait
    let status = Command::new(editor).arg(&file_path).status();
    match status { Ok(s) if s.success() => {}, _ => {} }
    // Read file and evaluate
    let src = std::fs::read_to_string(&file_path)?;
    if src.trim().is_empty() { println!("{}", "(empty)".dimmed()); return Ok(()); }
    let mut p = Parser::from_source(&src);
    match p.parse_all() {
        Ok(values) => {
            for v in values { let out = ev.eval(v); println!("{}", format_value_color(&out, Some(recommended_truncation_for_width(term_width_cols())), AssocMode::Auto)); }
            Ok(())
        }
        Err(e) => { eprintln!("{} {}", "Error:".red().bold(), e); Ok(()) }
    }
}
