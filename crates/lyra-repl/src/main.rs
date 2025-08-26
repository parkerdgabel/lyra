use anyhow::Result;
use colored::Colorize;
use lyra_core::value::Value;
use lyra_parser::Parser;
use lyra_runtime::{set_default_registrar, Evaluator};
use lyra_stdlib as stdlib;
use rustyline::completion::{Completer, Pair};
use rustyline::config::Configurer;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::history::DefaultHistory;
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::Cmd;
use rustyline::Movement;
use rustyline::{error::ReadlineError, Context, Editor, Helper};
use rustyline::{Event, EventHandler, ExternalPrinter};
use rustyline::{KeyCode, KeyEvent, Modifiers};
use std::fs;
use terminal_size::{terminal_size, Height, Width};

#[cfg(feature = "reedline")]
mod reedline_mode {
    use super::*;
    use colored::Colorize;
    use lyra_core::value::Value;
    use lyra_runtime::set_default_registrar;
    use lyra_runtime::Evaluator;
    use lyra_stdlib as stdlib;
    use nu_ansi_term::Style as NuStyle;
    use reedline::MenuBuilder;
    use reedline::{
        default_emacs_keybindings, Completer as RLCompleter, DefaultPrompt, DescriptionMenu, Emacs,
        Highlighter as RLHighlighter, Hinter as RLHinter, Reedline, ReedlineEvent, ReedlineMenu,
        Signal, Span, StyledText, Suggestion,
    };

    #[derive(Clone)]
    struct LyraCompleter {
        builtins: std::sync::Arc<Vec<BuiltinEntry>>,
        env_names: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        pkg_exports:
            std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, Vec<String>>>>,
        assoc_keys: std::sync::Arc<Vec<String>>,
        doc_index: std::sync::Arc<DocIndex>,
    }

    impl RLCompleter for LyraCompleter {
        fn complete(&self, line: &str, pos: usize) -> Vec<Suggestion> {
            let mut out: Vec<(i64, Suggestion)> = Vec::new();
            let (tok_start, tok_word) = current_symbol_token(line, pos);
            // Using[...] context
            if let Some(ctx) = using_context(line, pos) {
                match ctx {
                    UsingCtx::PackageName(prefix) => {
                        if let Ok(pkgs) = self.pkg_exports.lock() {
                            for name in pkgs.keys() {
                                if let Some(s) = fuzzy_score(name, &prefix) {
                                    let start = pos.saturating_sub(prefix.len());
                                    out.push((
                                        s + 1200,
                                        Suggestion {
                                            value: name.clone(),
                                            description: Some("(package)".into()),
                                            span: Span { start, end: pos },
                                            ..Default::default()
                                        },
                                    ));
                                }
                            }
                        }
                    }
                    UsingCtx::OptionKey(prefix) => {
                        for k in ["Import", "Except"].into_iter() {
                            if let Some(s) = fuzzy_score(k, &prefix) {
                                let start = pos.saturating_sub(prefix.len());
                                out.push((
                                    s + 1100,
                                    Suggestion {
                                        value: k.to_string(),
                                        description: None,
                                        span: Span { start, end: pos },
                                        ..Default::default()
                                    },
                                ));
                            }
                        }
                    }
                    UsingCtx::ImportValue { pkg, prefix } => {
                        if let Ok(pkgs) = self.pkg_exports.lock() {
                            if let Some(exports) = pkgs.get(&pkg) {
                                for e in exports {
                                    if let Some(s) = fuzzy_score(e, &prefix) {
                                        let start = pos.saturating_sub(prefix.len());
                                        out.push((
                                            s + 1300,
                                            Suggestion {
                                                value: format!("\"{}\"", e),
                                                description: Some("(export)".into()),
                                                span: Span { start, end: pos },
                                                ..Default::default()
                                            },
                                        ));
                                    }
                                }
                                if let Some(s) = fuzzy_score("All", &prefix) {
                                    let start = pos.saturating_sub(prefix.len());
                                    out.push((
                                        s + 1000,
                                        Suggestion {
                                            value: "All".into(),
                                            description: Some("(all)".into()),
                                            span: Span { start, end: pos },
                                            ..Default::default()
                                        },
                                    ));
                                }
                            }
                        }
                    }
                }
                out.sort_by(|a, b| b.0.cmp(&a.0));
                return out.into_iter().map(|(_, s)| s).collect();
            }
            // Assoc key
            if let Some(prefix) = assoc_key_prefix(line, pos) {
                for k in self.assoc_keys.iter() {
                    if let Some(s) = fuzzy_score(k, &prefix) {
                        let start = pos.saturating_sub(prefix.len());
                        out.push((
                            s + 1000,
                            Suggestion {
                                value: k.clone(),
                                description: Some("(key)".into()),
                                span: Span { start, end: pos },
                                ..Default::default()
                            },
                        ));
                    }
                }
                out.sort_by(|a, b| b.0.cmp(&a.0));
                return out.into_iter().map(|(_, s)| s).collect();
            }
            // Assoc value
            if let Some((key, vprefix, quoted)) = assoc_value_context(line, pos) {
                let mut values: Vec<String> = Vec::new();
                let lower = key.to_lowercase();
                if ["inplace", "dryrun", "verbose", "pretty"].contains(&lower.as_str()) {
                    values.extend(vec!["True".into(), "False".into()]);
                }
                if lower == "cache" {
                    values.extend(vec!["session".into(), "none".into()]);
                }
                if lower == "output" {
                    values.extend(vec!["expr".into(), "json".into()]);
                }
                if lower == "assoc" || lower == "assocrender" {
                    values.extend(vec!["auto".into(), "inline".into(), "pretty".into()]);
                }
                if lower == "import" {
                    values.push("All".into());
                }
                values.extend(vec!["Null".into(), "0".into(), "1".into(), "\"\"".into()]);
                for val in values.into_iter() {
                    if let Some(s) = fuzzy_score(&val, &vprefix) {
                        let rep = if val == "True"
                            || val == "False"
                            || val == "Null"
                            || val.chars().all(|c| c.is_ascii_digit())
                            || val.starts_with('"')
                        {
                            val.clone()
                        } else {
                            format!("\"{}\"", val)
                        };
                        let start = pos.saturating_sub(vprefix.len());
                        out.push((
                            s + 900,
                            Suggestion {
                                value: if quoted { rep.clone() } else { rep },
                                description: Some("(value)".into()),
                                span: Span { start, end: pos },
                                ..Default::default()
                            },
                        ));
                    }
                }
                out.sort_by(|a, b| b.0.cmp(&a.0));
                return out.into_iter().map(|(_, s)| s).collect();
            }
            // Default symbol/env completion
            let (start, word) = current_symbol_token(line, pos);
            if word.is_empty() {
                return Vec::new();
            }
            for b in self.builtins.iter() {
                if let Some(s) = fuzzy_score(&b.name, &word) {
                    if b.name == word {
                        continue;
                    }
                    // Build description with summary and usage
                    let mut desc: Option<String> = None;
                    if let Some((summary, _attrs, params)) = self.doc_index.map.get(&b.name) {
                        let usage = if !params.is_empty() {
                            format!("Usage: {}[{}]", b.name, params.join(", "))
                        } else {
                            String::new()
                        };
                        let txt = match (summary.is_empty(), usage.is_empty()) {
                            (true, true) => None,
                            (false, true) => Some(summary.clone()),
                            (true, false) => Some(usage),
                            (false, false) => Some(format!("{} — {}", summary, usage)),
                        };
                        desc = txt;
                    } else if !b.summary.is_empty() {
                        desc = Some(b.summary.clone());
                    }
                    out.push((
                        s + 1000,
                        Suggestion {
                            value: format!("{}[", b.name),
                            description: desc,
                            span: Span { start, end: pos },
                            ..Default::default()
                        },
                    ));
                }
            }
            if let Ok(envs) = self.env_names.lock() {
                for n in envs.iter() {
                    if let Some(s) = fuzzy_score(n, &word) {
                        if n == &word {
                            continue;
                        }
                        out.push((
                            s + 800,
                            Suggestion {
                                value: n.clone(),
                                description: Some("(var)".into()),
                                span: Span { start, end: pos },
                                ..Default::default()
                            },
                        ));
                    }
                }
            }
            out.sort_by(|a, b| b.0.cmp(&a.0));
            // Guard: avoid emitting a single suggestion identical to current token
            let mut suggs: Vec<Suggestion> = out.into_iter().map(|(_, s)| s).collect();
            if suggs.len() == 1 {
                let s = &suggs[0];
                // If replacement equals current token content, suppress
                if s.span.start <= s.span.end {
                    let replaced = &line[s.span.start..s.span.end];
                    if s.value == replaced || s.value == tok_word {
                        return Vec::new();
                    }
                }
            }
            suggs
        }
    }

    struct LyraRlHighlighter {
        builtins: std::sync::Arc<Vec<BuiltinEntry>>,
        env_names: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl RLHighlighter for LyraRlHighlighter {
        fn highlight(&self, line: &str, _cursor: usize) -> StyledText {
            let mut st = StyledText::new();
            let colored = match self.env_names.lock() {
                Ok(names) => super::highlight_lyra_line(line, &self.builtins, Some(&names)),
                _ => super::highlight_lyra_line(line, &self.builtins, None),
            };
            st.push((NuStyle::new(), colored));
            st
        }
    }

    struct LyraHinter {
        doc_index: std::sync::Arc<DocIndex>,
        last: String,
    }

    impl RLHinter for LyraHinter {
        fn handle(
            &mut self,
            line: &str,
            pos: usize,
            _history: &dyn reedline::History,
            use_ansi_coloring: bool,
        ) -> String {
            let mut out = String::new();
            if let Some(diag) = super::compute_live_diag(line) {
                out = format!("error: {}", diag.msg);
                self.last = out.clone();
                return out;
            }
            if let Some((head, arg_idx)) = current_call_context(line, pos) {
                if let Some((_, _attrs, params)) = self.doc_index.map.get(&head) {
                    let name = if arg_idx < params.len() {
                        params[arg_idx].clone()
                    } else if !params.is_empty() {
                        format!("{} (extra)", params.last().cloned().unwrap_or_default())
                    } else {
                        String::new()
                    };
                    if !name.is_empty() {
                        out = format!("  next: {}", name);
                    }
                }
            }
            self.last = out.clone();
            out
        }
        fn complete_hint(&self) -> String {
            self.last.clone()
        }
        fn next_hint_token(&self) -> String {
            self.last.clone()
        }
    }

    pub fn run() -> anyhow::Result<()> {
        println!("{}", "Lyra REPL (reedline prototype)".bright_yellow().bold());
        set_default_registrar(stdlib::register_all);
        let mut ev = Evaluator::new();
        stdlib::register_all(&mut ev);

        let mut rl = Reedline::create();
        // Build completer
        let builtins = discover_builtins(&mut ev);
        let doc_index = build_doc_index(&mut ev);
        let env_names_shared: std::sync::Arc<std::sync::Mutex<Vec<String>>> =
            std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let pkg_exports_shared: std::sync::Arc<
            std::sync::Mutex<std::collections::HashMap<String, Vec<String>>>,
        > = std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
        let assoc_keys = std::sync::Arc::new(vec![
            "MaxThreads".into(),
            "TimeBudgetMs".into(),
            "Import".into(),
            "Except".into(),
            "replacement".into(),
            "inPlace".into(),
            "dryRun".into(),
            "backupExt".into(),
        ]);
        let completer = LyraCompleter {
            builtins: std::sync::Arc::new(builtins),
            env_names: env_names_shared.clone(),
            pkg_exports: pkg_exports_shared.clone(),
            assoc_keys: assoc_keys.clone(),
            doc_index: doc_index.clone(),
        };
        rl = rl.with_completer(Box::new(completer));
        rl = rl.with_highlighter(Box::new(LyraRlHighlighter {
            builtins: std::sync::Arc::new(builtins.clone()),
            env_names: env_names_shared.clone(),
        }));
        // Add a description menu for completions (shows summary/usage from Suggestion.description)
        let completion_menu = Box::new(DescriptionMenu::default().with_name("lyra_menu"));
        // Keybindings: Tab opens completion menu (no auto-next to avoid edge hangs)
        let mut keybindings = default_emacs_keybindings();
        keybindings.add_binding(
            reedline::KeyModifiers::NONE,
            reedline::KeyCode::Tab,
            ReedlineEvent::Menu("lyra_menu".to_string()),
        );
        let edit_mode = Box::new(Emacs::new(keybindings));
        rl = rl.with_menu(ReedlineMenu::EngineCompleter(completion_menu)).with_edit_mode(edit_mode);
        // Add a simple hinter for next parameter hints
        rl = rl.with_hinter(Box::new(LyraHinter {
            doc_index: doc_index.clone(),
            last: String::new(),
        }));
        let prompt = DefaultPrompt::default();
        loop {
            // refresh env and package export data
            if let Ok(mut n) = env_names_shared.lock() {
                *n = ev.env_keys();
            }
            if let Ok(mut p) = pkg_exports_shared.lock() {
                *p = collect_pkg_exports(&mut ev);
            }
            match rl.read_line(&prompt) {
                Ok(Signal::Success(line)) => {
                    if line.trim().is_empty() {
                        continue;
                    }
                    let mut p = lyra_parser::Parser::from_source(&line);
                    match p.parse_all() {
                        Ok(values) => {
                            for v in values {
                                let out = ev.eval(v);
                                println!("{}", lyra_core::pretty::format_value(&out));
                            }
                        }
                        Err(e) => eprintln!("parse error: {}", e),
                    }
                }
                Ok(Signal::CtrlC) => {
                    println!("^C");
                    continue;
                }
                Ok(Signal::CtrlD) | Err(_) => {
                    println!("^D");
                    break;
                }
            }
        }
        Ok(())
    }
}

fn history_path() -> Option<std::path::PathBuf> {
    if let Ok(home) = std::env::var("HOME") {
        let mut p = std::path::PathBuf::from(home);
        p.push(".lyra_history");
        Some(p)
    } else {
        None
    }
}

fn print_parse_error(file: &str, src: &str, pos: usize, msg: &str) {
    let (line, col) = byte_to_line_col(src, pos);
    eprintln!("{}:{}:{}: error: {}", file, line, col, msg);
    if let Some((ln, text)) = line_text(src, line) {
        eprintln!("   {} | {}", ln, text);
        let caret = caret_line(&text, col);
        eprintln!("     | {}", caret);
    }
}

fn byte_to_line_col(src: &str, pos: usize) -> (usize, usize) {
    let mut line = 1usize;
    let mut col = 1usize;
    let mut idx = 0usize;
    for ch in src.chars() {
        if idx >= pos {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
        idx += ch.len_utf8();
    }
    (line, col)
}

fn line_text(src: &str, line: usize) -> Option<(usize, String)> {
    let mut cur = 1usize;
    let mut start = 0usize;
    for (i, ch) in src.char_indices() {
        if ch == '\n' {
            if cur == line {
                return Some((line, src[start..i].to_string()));
            }
            cur += 1;
            start = i + 1;
        }
    }
    if cur == line {
        return Some((line, src[start..].to_string()));
    }
    None
}

fn caret_line(text: &str, col: usize) -> String {
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

#[derive(Clone, Debug)]
struct BuiltinEntry {
    name: String,
    summary: String,
    attrs: Vec<String>,
}

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

impl Highlighter for ReplHelper {
    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> std::borrow::Cow<'l, str> {
        use std::borrow::Cow;
        Cow::Owned(highlight_lyra_line(line, &self.builtins, Some(&self.env_names)))
    }
    fn highlight_hint<'h>(&self, hint: &'h str) -> std::borrow::Cow<'h, str> {
        use std::borrow::Cow;
        Cow::Owned(hint.dimmed().to_string())
    }
    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(
        &self,
        prompt: &'p str,
        _default: bool,
    ) -> std::borrow::Cow<'b, str> {
        use std::borrow::Cow;
        Cow::Owned(prompt.bright_black().to_string())
    }
    fn highlight_char(&self, _line: &str, _pos: usize, _forced: bool) -> bool {
        true
    }
}

#[derive(Clone, Debug)]
struct LiveDiag {
    start: usize,
    end: usize,
    msg: String,
}

fn compute_live_diag(s: &str) -> Option<LiveDiag> {
    let ch: Vec<char> = s.chars().collect();
    let mut i = 0usize;
    let mut in_string = false;
    let mut string_open: Option<usize> = None;
    let mut esc = false;
    let mut comment_stack: Vec<usize> = Vec::new();
    let mut bracket_stack: Vec<(char, usize)> = Vec::new();
    while i < ch.len() {
        if !in_string {
            if i + 1 < ch.len() && ch[i] == '(' && ch[i + 1] == '*' {
                comment_stack.push(i);
                i += 2;
                continue;
            }
            if i + 1 < ch.len() && ch[i] == '*' && ch[i + 1] == ')' {
                if comment_stack.pop().is_none() {
                    return Some(LiveDiag { start: i, end: i + 2, msg: "unexpected '*)'".into() });
                }
                i += 2;
                continue;
            }
        }
        if comment_stack.len() > 0 {
            i += 1;
            continue;
        }
        if ch[i] == '"' {
            if !in_string {
                in_string = true;
                string_open = Some(i);
                esc = false;
                i += 1;
                continue;
            }
            if !esc {
                in_string = false;
                string_open = None;
                i += 1;
                continue;
            }
        }
        if in_string {
            esc = ch[i] == '\\' && !esc;
            i += 1;
            continue;
        }
        match ch[i] {
            '(' | '[' | '{' => bracket_stack.push((ch[i], i)),
            ')' | ']' | '}' => {
                if let Some((open, _pos)) = bracket_stack.pop() {
                    if !matches!((open, ch[i]), ('(', ')') | ('[', ']') | ('{', '}')) {
                        return Some(LiveDiag {
                            start: i,
                            end: i + 1,
                            msg: format!("mismatched '{}'", ch[i]),
                        });
                    }
                } else {
                    return Some(LiveDiag {
                        start: i,
                        end: i + 1,
                        msg: format!("unexpected '{}'", ch[i]),
                    });
                }
            }
            _ => {}
        }
        i += 1;
    }
    if let Some(pos) = comment_stack.pop() {
        return Some(LiveDiag { start: pos, end: s.len(), msg: "unclosed comment".into() });
    }
    if let Some(pos) = string_open {
        return Some(LiveDiag { start: pos, end: s.len(), msg: "unclosed string".into() });
    }
    if let Some((open, pos)) = bracket_stack.pop() {
        return Some(LiveDiag { start: pos, end: pos + 1, msg: format!("unclosed '{}'", open) });
    }
    None
}

fn parser_backed_diag(s: &str) -> Option<LiveDiag> {
    let mut p = Parser::from_source(s);
    match p.parse_all_detailed() {
        Ok(_) => None,
        Err(e) => {
            let start = e.pos.min(s.len());
            let mut bytes_after: Vec<usize> = s[start..].char_indices().map(|(i, _)| i).collect();
            bytes_after.push(s.len() - start);
            let end = if bytes_after.len() > 1 { start + bytes_after[1] } else { start };
            Some(LiveDiag { start, end, msg: e.message })
        }
    }
}

fn highlight_lyra_line(
    s: &str,
    builtins: &Vec<BuiltinEntry>,
    env_names: Option<&[String]>,
) -> String {
    let diag = compute_live_diag(s).or_else(|| parser_backed_diag(s));
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0usize;
    let mut out = String::new();
    let in_string = false;
    let mut in_comment: usize = 0; // allow simple nested (* *) for highlighting

    while i < chars.len() {
        // comment close
        if in_comment > 0 {
            if i + 1 < chars.len() && chars[i] == '*' && chars[i + 1] == ')' {
                let styled = "*)".bright_black().to_string();
                out.push_str(&styled);
                i += 2;
                in_comment -= 1;
                continue;
            }
            // flush until potential close or end
            let mut j = i;
            while j < chars.len() {
                if j + 1 < chars.len() && chars[j] == '*' && chars[j + 1] == ')' {
                    break;
                }
                j += 1;
            }
            out.push_str(&chars[i..j].iter().collect::<String>().bright_black().to_string());
            i = j;
            continue;
        }

        // comment open
        if !in_string && i + 1 < chars.len() && chars[i] == '(' && chars[i + 1] == '*' {
            let is_err = matches!(diag, Some(ref d) if d.start==i && d.end>=i+2);
            let styled = if is_err {
                "(*".red().underline().to_string()
            } else {
                "(*".bright_black().to_string()
            };
            out.push_str(&styled);
            i += 2;
            in_comment += 1;
            continue;
        }

        // strings (detect assoc key if followed by -> after quotes)
        if !in_string && chars[i] == '"' {
            let mut j = i + 1;
            let mut escaped = false;
            while j < chars.len() {
                let c = chars[j];
                if c == '"' && !escaped {
                    j += 1;
                    break;
                }
                escaped = c == '\\' && !escaped;
                if c != '\\' {
                    escaped = false;
                }
                j += 1;
            }
            // Lookahead for -> to treat as association key
            let mut k = j;
            while k < chars.len() && chars[k].is_whitespace() {
                k += 1;
            }
            let is_key = k + 1 < chars.len() && chars[k] == '-' && chars[k + 1] == '>';
            let is_err = matches!(diag, Some(ref d) if i>=d.start && j<=d.end && d.msg.contains("unclosed string"));
            let styled = if is_err {
                chars[i..j].iter().collect::<String>().red().underline().to_string()
            } else if is_key {
                chars[i..j].iter().collect::<String>().cyan().to_string()
            } else {
                chars[i..j].iter().collect::<String>().green().to_string()
            };
            out.push_str(&styled);
            i = j;
            continue;
        }

        // association delimiters <| and |>
        if i + 1 < chars.len() && chars[i] == '<' && chars[i + 1] == '|' {
            out.push_str(&"<|".cyan().to_string());
            i += 2;
            continue;
        }
        if i + 1 < chars.len() && chars[i] == '|' && chars[i + 1] == '>' {
            out.push_str(&"|>".cyan().to_string());
            i += 2;
            continue;
        }

        // multi-char operators
        macro_rules! match_emit {
            ($lit:expr, $style:ident) => {{
                if s[i..].starts_with($lit) {
                    out.push_str(&($lit).$style().to_string());
                    i += $lit.len();
                    continue;
                }
            }};
        }
        match_emit!("->", magenta);
        match_emit!("/;", magenta);
        match_emit!("==", magenta);
        match_emit!("!=", magenta);
        match_emit!("<=", magenta);
        match_emit!(">=", magenta);
        match_emit!("||", magenta);
        match_emit!("&&", magenta);
        match_emit!("<<", magenta);
        match_emit!(">>", magenta);
        match_emit!("=>", magenta);
        match_emit!("@@@", magenta);
        match_emit!("@@", magenta);
        match_emit!("/@", magenta);
        match_emit!("|>", magenta); // generic pipeline if typed alone

        // brackets and punctuation
        if matches!(chars[i], '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';') {
            let is_err = matches!(diag, Some(ref d) if i>=d.start && i<d.end);
            let styled = if is_err {
                chars[i].to_string().red().underline().to_string()
            } else {
                chars[i].to_string().bright_black().to_string()
            };
            out.push_str(&styled);
            i += 1;
            continue;
        }

        // single-char operators
        if matches!(chars[i], '+' | '-' | '*' | '/' | '^' | '=' | '<' | '>' | '!' | '@' | '|') {
            out.push_str(&chars[i].to_string().magenta().to_string());
            i += 1;
            continue;
        }

        // numbers (hex 0x.. or decimal with optional '.')
        if chars[i].is_ascii_digit() {
            let mut j = i + 1;
            if i + 1 < chars.len()
                && (chars[i] == '0')
                && (chars[i + 1] == 'x' || chars[i + 1] == 'X')
            {
                j = i + 2;
                while j < chars.len() && chars[j].is_ascii_hexdigit() {
                    j += 1;
                }
            } else {
                while j < chars.len() && (chars[j].is_ascii_digit() || chars[j] == '.') {
                    j += 1;
                }
            }
            out.push_str(&chars[i..j].iter().collect::<String>().bright_blue().to_string());
            i = j;
            continue;
        }

        // slots: # or #n
        if chars[i] == '#' {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_ascii_digit() {
                j += 1;
            }
            out.push_str(&chars[i..j].iter().collect::<String>().purple().to_string());
            i = j;
            continue;
        }

        // identifiers (including $ and _)
        if chars[i].is_alphabetic() || chars[i] == '_' || chars[i] == '$' {
            let mut j = i + 1;
            while j < chars.len() {
                let c = chars[j];
                if c.is_alphanumeric() || c == '_' || c == '$' {
                    j += 1;
                } else {
                    break;
                }
            }
            let tok: String = chars[i..j].iter().collect();
            let lowered = tok.as_str();
            // assoc key if followed by ->
            let mut k = j;
            while k < chars.len() && chars[k].is_whitespace() {
                k += 1;
            }
            let is_assoc_key = k + 1 < chars.len() && chars[k] == '-' && chars[k + 1] == '>';
            if lowered == "True" || lowered == "False" || lowered == "Null" {
                out.push_str(&tok.purple().to_string());
            } else if is_assoc_key {
                out.push_str(&tok.cyan().to_string());
            } else if builtins.iter().any(|b| b.name == tok) {
                // If this is a call head, brighten
                let mut k2 = j;
                while k2 < chars.len() && chars[k2].is_whitespace() {
                    k2 += 1;
                }
                let is_head = k2 < chars.len() && chars[k2] == '[';
                if is_head {
                    out.push_str(&tok.yellow().bold().to_string());
                } else {
                    out.push_str(&tok.yellow().to_string());
                }
            } else if env_names.and_then(|ns| Some(ns.contains(&tok))).unwrap_or(false) {
                out.push_str(&tok.bright_cyan().to_string());
            } else {
                // Non-builtin call head: highlight subtly
                let mut k2 = j;
                while k2 < chars.len() && chars[k2].is_whitespace() {
                    k2 += 1;
                }
                let is_head = k2 < chars.len() && chars[k2] == '[';
                if is_head {
                    out.push_str(&tok.bright_yellow().to_string());
                } else {
                    out.push_str(&tok);
                }
            }
            i = j;
            continue;
        }

        // default: passthrough
        out.push(chars[i]);
        i += 1;
    }

    out
}

impl Validator for ReplHelper {
    fn validate(&self, _ctx: &mut ValidationContext) -> rustyline::Result<ValidationResult> {
        // Simple bracket balance validator with string and comment awareness
        // Accepts when balanced; returns Incomplete to prompt for more lines
        if scan_unbalanced(_ctx.input()).is_some() {
            return Ok(ValidationResult::Incomplete);
        }
        Ok(ValidationResult::Valid(None))
    }
    fn validate_while_typing(&self) -> bool {
        false
    }
}

impl Hinter for ReplHelper {
    type Hint = String;
    fn hint(&self, line: &str, pos: usize, _ctx: &Context<'_>) -> Option<String> {
        if let Some(diag) = compute_live_diag(line) {
            return Some(format!("error: {}", diag.msg));
        }
        // Parameter hint: if cursor is inside Name[ ... ], show next param name
        if let Some((head, arg_idx)) = current_call_context(line, pos) {
            if let Some((_, _attrs, params, _examples)) = self.doc_index.map.get(&head) {
                let name = if arg_idx < params.len() {
                    params[arg_idx].clone()
                } else if !params.is_empty() {
                    format!("{} (extra)", params.last().cloned().unwrap_or_default())
                } else {
                    String::new()
                };
                if !name.is_empty() {
                    return Some(format!("  next: {}", name).dimmed().to_string());
                }
            }
        }
        // Show inline summary when the current token is a known builtin
        let (start, word) = current_symbol_token(line, pos);
        if start == pos || word.is_empty() {
            return None;
        }
        // Exact match or unique prefix
        let mut matches: Vec<&BuiltinEntry> =
            self.builtins.iter().filter(|b| b.name.starts_with(&word)).collect();
        if matches.is_empty() {
            return None;
        }
        if matches.len() > 1 {
            if let Some(exact) = matches.iter().find(|b| b.name == word) {
                matches = vec![exact];
            } else {
                return None;
            }
        }
        let s = matches[0];
        if s.summary.is_empty() {
            None
        } else {
            Some(format!(" — {}", s.summary))
        }
    }
}

impl Completer for ReplHelper {
    type Candidate = Pair;
    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        // Filesystem path completion if current token looks like a path or is quoted
        if let Some((start, word, quoted)) = current_path_token(line, pos) {
            let cands = complete_paths(&word);
            let pairs = cands
                .into_iter()
                .map(|(disp, repl)| Pair {
                    display: disp.clone(),
                    replacement: if quoted { format!("{}", repl) } else { repl },
                })
                .collect();
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
            if cmd_word.starts_with("explain ")
                || cmd_word.starts_with("profile ")
                || cmd_word.starts_with("pager ")
                || cmd_word.starts_with("record ")
            {
                for v in ["on", "off"] {
                    if v.starts_with(&word) {
                        cands.push(Pair { display: v.into(), replacement: format!("{}", v) });
                    }
                }
            }
            if cmd_word.starts_with("mode ") {
                for v in ["vi", "emacs"] {
                    if v.starts_with(&word) {
                        cands.push(Pair { display: v.into(), replacement: v.into() });
                    }
                }
            }
            if cmd_word.starts_with("format ") {
                if cmd_word.starts_with("format assoc ") {
                    for v in ["auto", "inline", "pretty"] {
                        if v.starts_with(&word) {
                            cands.push(Pair { display: v.into(), replacement: v.into() });
                        }
                    }
                } else if cmd_word.starts_with("format output ") {
                    for v in ["expr", "json"] {
                        if v.starts_with(&word) {
                            cands.push(Pair { display: v.into(), replacement: v.into() });
                        }
                    }
                } else {
                    for v in ["assoc", "output"] {
                        if v.starts_with(&word) {
                            cands.push(Pair { display: v.into(), replacement: format!("{} ", v) });
                        }
                    }
                }
            }
            // Guard: if single suggestion identical to current token, suppress
            if cands.len() == 1 {
                if cands[0].replacement.trim() == format!(":{}", word) {
                    return Ok((pos, Vec::new()));
                }
            }
            return Ok((pos.saturating_sub(word.len()), cands));
        }

        // Using[...] context-aware options
        if let Some(ctx) = using_context(line, pos) {
            let mut cands: Vec<Pair> = Vec::new();
            match ctx {
                UsingCtx::PackageName(prefix) => {
                    // suggest loaded package names (keys in pkg_exports) with fuzzy match
                    let mut scored: Vec<(i64, Pair)> = Vec::new();
                    for name in self.pkg_exports.keys() {
                        if let Some(s) = fuzzy_score(name, &prefix) {
                            scored.push((
                                s,
                                Pair {
                                    display: format!("{} — (package)", name),
                                    replacement: name.to_string(),
                                },
                            ));
                        }
                    }
                    scored.sort_by(|a, b| b.0.cmp(&a.0));
                    cands.extend(scored.into_iter().map(|(_, p)| p));
                }
                UsingCtx::OptionKey(prefix) => {
                    let mut scored: Vec<(i64, Pair)> = Vec::new();
                    for k in ["Import", "Except"].into_iter() {
                        if let Some(s) = fuzzy_score(k, &prefix) {
                            scored.push((
                                s,
                                Pair { display: k.to_string(), replacement: k.to_string() },
                            ));
                        }
                    }
                    scored.sort_by(|a, b| b.0.cmp(&a.0));
                    cands.extend(scored.into_iter().map(|(_, p)| p));
                }
                UsingCtx::ImportValue { pkg, prefix } => {
                    if let Some(exports) = self.pkg_exports.get(&pkg) {
                        let mut scored: Vec<(i64, Pair)> = Vec::new();
                        for e in exports.iter() {
                            if let Some(s) = fuzzy_score(e, &prefix) {
                                scored.push((
                                    s,
                                    Pair { display: e.clone(), replacement: format!("\"{}\"", e) },
                                ));
                            }
                        }
                        if let Some(s) = fuzzy_score("All", &prefix) {
                            scored.push((
                                s,
                                Pair { display: "All".into(), replacement: "All".into() },
                            ));
                        }
                        scored.sort_by(|a, b| b.0.cmp(&a.0));
                        cands.extend(scored.into_iter().map(|(_, p)| p));
                    } else {
                        if let Some(_) = fuzzy_score("All", &prefix) {
                            cands.push(Pair { display: "All".into(), replacement: "All".into() });
                        }
                    }
                }
            }
            return Ok((pos, cands));
        }
        // Association key completion inside <| ... |>
        if let Some(prefix) = assoc_key_prefix(line, pos) {
            let mut scored: Vec<(i64, Pair)> = Vec::new();
            for k in &self.common_option_keys {
                if let Some(s) = fuzzy_score(k, &prefix) {
                    scored.push((s, Pair { display: k.clone(), replacement: k.clone() }));
                }
            }
            scored.sort_by(|a, b| b.0.cmp(&a.0));
            let pairs: Vec<Pair> = scored.into_iter().map(|(_, p)| p).collect();
            if pairs.len() == 1 && pairs[0].replacement == prefix {
                return Ok((pos, Vec::new()));
            }
            return Ok((pos.saturating_sub(prefix.len()), pairs));
        }
        // Association value suggestions inside <| key -> value |>
        if let Some((key, vprefix, quoted)) = assoc_value_context(line, pos) {
            let mut values: Vec<String> = Vec::new();
            let lower = key.to_lowercase();
            if ["inplace", "dryrun", "verbose", "pretty"].contains(&lower.as_str()) {
                values.extend(vec!["True".into(), "False".into()]);
            }
            if lower == "cache" {
                values.extend(vec!["session".into(), "none".into()]);
            }
            if lower == "output" {
                values.extend(vec!["expr".into(), "json".into()]);
            }
            if lower == "assoc" || lower == "assocrender" {
                values.extend(vec!["auto".into(), "inline".into(), "pretty".into()]);
            }
            if lower == "import" {
                values.push("All".into());
            }
            // Common helpful literals
            values.extend(vec!["Null".into(), "0".into(), "1".into(), "\"\"".into()]);
            let mut scored: Vec<(i64, Pair)> = Vec::new();
            for val in values.into_iter() {
                let disp = val.clone();
                let rep = if disp == "True"
                    || disp == "False"
                    || disp == "Null"
                    || disp.chars().all(|c| c.is_ascii_digit())
                    || disp.starts_with('"')
                {
                    disp.clone()
                } else {
                    format!("\"{}\"", disp)
                };
                if let Some(s) = fuzzy_score(&disp, &vprefix) {
                    scored.push((
                        s,
                        Pair { display: disp, replacement: if quoted { rep.clone() } else { rep } },
                    ));
                }
            }
            scored.sort_by(|a, b| b.0.cmp(&a.0));
            let pairs: Vec<Pair> = scored.into_iter().map(|(_, p)| p).collect();
            if pairs.len() == 1 && pairs[0].replacement.trim_matches('"') == vprefix {
                return Ok((pos, Vec::new()));
            }
            let start = pos.saturating_sub(vprefix.len());
            return Ok((start, pairs));
        }

        // Default: builtins and env vars
        let (start, word) = current_symbol_token(line, pos);
        if word.is_empty() {
            return Ok((pos, Vec::new()));
        }
        let mut scored: Vec<(i64, Pair)> = Vec::new();
        for b in &self.builtins {
            if let Some(s) = fuzzy_score(&b.name, &word) {
                if b.name == word {
                    continue;
                }
                let attrs = if b.attrs.is_empty() {
                    String::new()
                } else {
                    format!(" [{}]", b.attrs.join(", "))
                };
                let display = if b.summary.is_empty() {
                    format!("{}{}", b.name, attrs)
                } else {
                    format!("{} — {}{}", b.name, b.summary, attrs)
                };
                scored.push((s + 1000, Pair { display, replacement: format!("{}[", b.name) }));
            }
        }
        for n in &self.env_names {
            if let Some(s) = fuzzy_score(n, &word) {
                if n == &word {
                    continue;
                }
                scored.push((
                    s + 800,
                    Pair { display: format!("{} — (var)", n), replacement: n.clone() },
                ));
            }
        }
        scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.display.cmp(&b.1.display)));
        let pairs: Vec<Pair> = scored.into_iter().take(200).map(|(_, p)| p).collect();
        // Guard: if exactly one candidate and it doesn't add any characters over the current word,
        // return no completions to avoid editor stalls on single-candidate autopick.
        if pairs.len() == 1 {
            let rep = pairs[0].replacement.as_str();
            if rep == word {
                return Ok((start, Vec::new()));
            }
        }
        Ok((start, pairs))
    }
}

// Simple fuzzy matcher: prefix > substring > subsequence with bonuses
fn fuzzy_score(candidate: &str, pattern: &str) -> Option<i64> {
    if pattern.is_empty() {
        return None;
    }
    let c_low = candidate.to_lowercase();
    let p_low = pattern.to_lowercase();
    if c_low == p_low {
        return Some(10_000);
    }
    if c_low.starts_with(&p_low) {
        return Some(5_000 + (100 - (candidate.len() as i64)));
    }
    if c_low.contains(&p_low) {
        return Some(3_000 + (50 - (candidate.len() as i64)));
    }
    // subsequence
    let mut score: i64 = 0;
    let mut last_idx: isize = -1;
    let mut consec: i64 = 0;
    let bytes: Vec<char> = candidate.chars().collect();
    let mut i = 0usize;
    for pc in p_low.chars() {
        let mut found = false;
        while i < bytes.len() {
            let cb = bytes[i].to_ascii_lowercase();
            if cb == pc {
                // base hit
                score += 10;
                // consecutive bonus
                if last_idx >= 0 && (i as isize) == last_idx + 1 {
                    consec += 1;
                    score += 5 * consec;
                } else {
                    consec = 0;
                }
                // word/start bonus
                if i == 0 {
                    score += 10;
                } else {
                    let prev = bytes[i - 1];
                    if prev == '_' || prev == '-' || prev == ':' || prev == '/' {
                        score += 8;
                    }
                    if prev.is_lowercase() && bytes[i].is_uppercase() {
                        score += 6;
                    }
                }
                last_idx = i as isize;
                i += 1;
                found = true;
                break;
            }
            i += 1;
        }
        if !found {
            return None;
        }
    }
    score -= ((candidate.len() as i64) - (pattern.len() as i64)).max(0);
    Some(score)
}

fn current_symbol_token(line: &str, pos: usize) -> (usize, String) {
    let chars: Vec<char> = line.chars().collect();
    let mut i = pos.min(chars.len());
    while i > 0 {
        let c = chars[i - 1];
        if c.is_alphanumeric() || c == '_' || c == '$' {
            i -= 1;
        } else {
            break;
        }
    }
    (i, chars[i..pos.min(chars.len())].iter().collect())
}

// Detect if cursor is inside a call like Name[...], returning (Name, current_arg_index)
fn current_call_context(line: &str, pos: usize) -> Option<(String, usize)> {
    // Scan backwards to find matching '[' at depth 0
    let bytes: Vec<char> = line.chars().collect();
    let mut i = pos.min(bytes.len());
    let mut depth: i32 = 0;
    let mut in_string = false;
    // Find the '[' that opens current bracket group
    while i > 0 {
        i -= 1;
        let c = bytes[i];
        if c == '"' {
            let escaped = i > 0 && bytes[i - 1] == '\\';
            if !escaped {
                in_string = !in_string;
            }
            continue;
        }
        if in_string {
            continue;
        }
        match c {
            ']' => depth += 1,
            '[' => {
                if depth == 0 {
                    break;
                } else {
                    depth -= 1;
                }
            }
            _ => {}
        }
    }
    if i == 0 && !(bytes.get(0) == Some(&'[')) {
        return None;
    }
    // Extract head symbol before '['
    if i == 0 {
        return None;
    }
    let mut j = i;
    while j > 0 {
        let ch = bytes[j - 1];
        if ch.is_alphanumeric() || ch == '_' || ch == '$' {
            j -= 1;
        } else {
            break;
        }
    }
    if j == i {
        return None;
    }
    let head: String = bytes[j..i].iter().collect();
    // Count commas at depth 0 inside the bracket from i+1 .. pos
    let mut k = i + 1;
    depth = 0;
    in_string = false;
    let mut arg_idx: usize = 0;
    while k < pos.min(bytes.len()) {
        let ch = bytes[k];
        if ch == '"' {
            let esc = k > 0 && bytes[k - 1] == '\\';
            if !esc {
                in_string = !in_string;
            }
            k += 1;
            continue;
        }
        if in_string {
            k += 1;
            continue;
        }
        match ch {
            '[' | '{' | '(' => depth += 1,
            ']' | '}' | ')' => {
                if depth > 0 {
                    depth -= 1;
                }
            }
            ',' => {
                if depth == 0 {
                    arg_idx += 1;
                }
            }
            _ => {}
        }
        k += 1;
    }
    Some((head, arg_idx))
}

fn word_start_pos_in_substr(line: &str, pos: usize) -> usize {
    let mut i = pos.min(line.len());
    let bytes = line.as_bytes();
    while i > 0 {
        let c = bytes[i - 1] as char;
        if c.is_alphanumeric() || c == '_' || c == '$' {
            i -= 1;
        } else {
            break;
        }
    }
    i
}

fn current_path_token(line: &str, pos: usize) -> Option<(usize, String, bool)> {
    // If inside quotes or token looks like a path (starts with ./, ../, ~/, or contains '/')
    let chars: Vec<char> = line.chars().collect();
    let i = pos.min(chars.len());
    let mut in_quotes = false;
    let mut j = i;
    // scan backwards for start quote on this token
    let mut k = i;
    while k > 0 {
        if chars[k - 1] == '"' {
            in_quotes = true;
            break;
        }
        if chars[k - 1].is_whitespace() || chars[k - 1] == ',' {
            break;
        }
        k -= 1;
    }
    // find token start
    while j > 0 {
        let c = chars[j - 1];
        if c.is_whitespace() || c == ',' || c == '[' || c == '{' || c == '|' || c == '>' {
            break;
        }
        j -= 1;
    }
    let token: String = chars[j..pos.min(chars.len())].iter().collect();
    let looks_path = token.starts_with("./")
        || token.starts_with("../")
        || token.starts_with("~/")
        || token.contains('/')
        || in_quotes;
    if looks_path {
        Some((j, token, in_quotes))
    } else {
        None
    }
}

fn complete_paths(token: &str) -> Vec<(String, String)> {
    use std::path::Path;
    let expanded = if token.starts_with("~/") {
        std::env::var("HOME")
            .ok()
            .map(|h| format!("{}{}", h, &token[1..]))
            .unwrap_or(token.to_string())
    } else {
        token.to_string()
    };
    let (dir, base) = if expanded.ends_with('/') {
        (expanded.clone(), String::new())
    } else {
        match expanded.rsplit_once('/') {
            Some((d, b)) => (format!("{}/", d), b.to_string()),
            None => ("./".into(), expanded.clone()),
        }
    };
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(Path::new(&dir)) {
        for e in rd.flatten() {
            let fname = e.file_name().to_string_lossy().to_string();
            if !fname.starts_with(&base) {
                continue;
            }
            let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
            let display = if is_dir { format!("{}/", fname) } else { fname.clone() };
            let replacement = format!(
                "{}{}",
                if token.starts_with("/") || token.starts_with("~/") {
                    dir.clone()
                } else {
                    String::new()
                },
                display.clone()
            );
            out.push((display, replacement));
        }
    }
    out
}

enum UsingCtx {
    PackageName(String),
    OptionKey(String),
    ImportValue { pkg: String, prefix: String },
}

fn using_context(line: &str, pos: usize) -> Option<UsingCtx> {
    // Very light-weight detection: Using[ pkg, <| ... |>
    let upos = line.find("Using[")?;
    if pos <= upos {
        return None;
    }
    // find argument region
    let after = &line[upos + 6..];
    // If cursor is within first arg segment, suggest package name
    if let Some(brack) = after.find(']') {
        let args_str = &after[..brack];
        // cursor relative position inside args
        let rel = pos.saturating_sub(upos + 6);
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
                    let key_prefix = before_cursor
                        .rsplit(|c: char| c == ',' || c == '<' || c == '|')
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    return Some(UsingCtx::OptionKey(key_prefix));
                } else {
                    // typing a value; try to detect Import key and package name
                    let pkg = args_str[..first_end].trim().trim_matches('"').to_string();
                    let val_prefix = before_cursor
                        .rsplit(|c: char| c == ',' || c == '{' || c == '[' || c == '>' || c == '|')
                        .next()
                        .unwrap_or("")
                        .trim()
                        .trim_matches('"')
                        .to_string();
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
    if line[last_open..].contains("|>") {
        return None;
    }
    // Now, find last '->' before pos; if none, we are in key position
    let key_start = if let Some(after_open) =
        left[last_open..].rfind(|c: char| c == ',' || c == '<' || c == '|')
    {
        last_open + after_open + 1
    } else {
        last_open + 2
    };
    // prefix
    let prefix = left[key_start..].trim();
    if prefix.contains("->") {
        return None;
    }
    Some(prefix.to_string())
}

// If cursor is inside an association literal and after '->', return (key, value_prefix, quoted?)
fn assoc_value_context(line: &str, pos: usize) -> Option<(String, String, bool)> {
    let left = &line[..pos];
    let open = left.rfind("<|")?;
    if line[open..].contains("|>") {
        return None;
    }
    let seg = &left[open + 2..];
    let arrow = seg.rfind("->")?;
    // Key is before arrow up to previous comma or start of segment
    let key_start = seg[..arrow].rfind(',').map(|i| i + 1).unwrap_or(0);
    let key = seg[key_start..arrow].trim().trim_matches('"').to_string();
    // Value prefix is after arrow
    let after = &seg[arrow + 2..];
    let vprefix =
        after.rsplit(|c: char| c == ',' || c == '<' || c == '|').next().unwrap_or("").trim();
    let quoted = vprefix.starts_with('"');
    let vprefix_clean = vprefix.trim_matches('"').to_string();
    Some((key, vprefix_clean, quoted))
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
            if i + 1 < chars.len() && chars[i] == '(' && chars[i + 1] == '*' {
                in_comment += 1;
                i += 2;
                continue;
            }
            if in_comment > 0 && i + 1 < chars.len() && chars[i] == '*' && chars[i + 1] == ')' {
                in_comment -= 1;
                i += 2;
                continue;
            }
        }
        if in_comment > 0 {
            i += 1;
            continue;
        }
        if c == '"' {
            // toggle string unless escaped
            let escaped = i > 0 && chars[i - 1] == '\\';
            if !escaped {
                in_string = !in_string;
            }
            i += 1;
            continue;
        }
        if in_string {
            i += 1;
            continue;
        }
        match c {
            '(' | '[' | '{' => stack.push((c, i)),
            ')' | ']' | '}' => {
                if let Some((open, _)) = stack.pop() {
                    if !matches!((open, c), ('(', ')') | ('[', ']') | ('{', '}')) {
                        // mismatched, treat as unbalanced
                        return Some((open, i));
                    }
                } else {
                    // extra closer; ignore but continue
                }
            }
            _ => {}
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
            if i + 1 < chars.len() && chars[i] == '(' && chars[i + 1] == '*' {
                in_comment += 1;
                i += 2;
                continue;
            }
            if in_comment > 0 && i + 1 < chars.len() && chars[i] == '*' && chars[i + 1] == ')' {
                in_comment -= 1;
                i += 2;
                continue;
            }
        }
        if in_comment > 0 {
            i += 1;
            continue;
        }
        if c == '"' {
            let escaped = i > 0 && chars[i - 1] == '\\';
            if !escaped {
                in_string = !in_string;
            }
            i += 1;
            continue;
        }
        if in_string {
            i += 1;
            continue;
        }
        match c {
            '(' | '[' | '{' => stack.push(c),
            ')' | ']' | '}' => {
                let _ = stack.pop();
            }
            _ => {}
        }
        i += 1;
    }
    stack.len()
}

fn term_width_cols() -> usize {
    if let Some((Width(w), Height(_h))) = terminal_size() {
        w as usize
    } else {
        100
    }
}

struct DocKeyHandler {
    printer: std::sync::Mutex<Box<dyn ExternalPrinter + Send>>,
    doc_index: std::sync::Arc<DocIndex>,
    cfg: std::sync::Arc<std::sync::Mutex<ReplConfig>>,
}

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
        if use_pager {
            let _ = spawn_pager(&msg);
        } else if let Ok(mut p) = self.printer.lock() {
            let _ = p.print(msg);
        }
        Some(Cmd::Noop)
    }
}

// Show quick docs when pressing Tab and the current token is an exact symbol
struct TabDocAssistHandler {
    printer: std::sync::Mutex<Box<dyn ExternalPrinter + Send>>,
    doc_index: std::sync::Arc<DocIndex>,
    cfg: std::sync::Arc<std::sync::Mutex<ReplConfig>>,
}

impl rustyline::ConditionalEventHandler for TabDocAssistHandler {
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
        if word.is_empty() {
            return None;
        }
        // Only trigger on exact match to avoid noise while exploring
        if !self.doc_index.map.contains_key(&word) {
            return None;
        }
        let msg = build_doc_card_str_from_index(&self.doc_index, &word);
        let use_pager = self.cfg.lock().map(|c| c.pager_on).unwrap_or(false);
        if use_pager {
            let _ = spawn_pager(&msg);
        } else if let Ok(mut p) = self.printer.lock() {
            let _ = p.print(msg);
        }
        // Swallow Tab when showing docs on exact match to avoid auto-complete side effects
        Some(Cmd::Noop)
    }
}

// Completion menu: state + handlers for showing and picking candidates
#[derive(Default, Clone)]
struct CompletionMenuState {
    items: Vec<Pair>,
    prefix_len: usize,
    selected_idx: usize,
    active: bool,
}

fn render_completion_panel(state: &CompletionMenuState, docs: &DocIndex) -> String {
    let mut msg = String::new();
    msg.push_str(&format!("{}\n", "Completions (Alt-/, Alt-j/k, Enter to pick)".bold()));
    for (i, p) in state.items.iter().enumerate() {
        let sel =
            if i == state.selected_idx { ">".bright_green().to_string() } else { " ".to_string() };
        let num = if i < 9 { format!("Alt-{}", i + 1) } else { "Alt-0".into() };
        msg.push_str(&format!(" {} {:>6}  {}\n", sel, num.dimmed(), p.display.as_str()));
        if i == state.selected_idx {
                if let Some(name) = candidate_symbol_name(&p) {
                if let Some((summary, _attrs, _params, _examples)) = docs.map.get(&name) {
                    if !summary.is_empty() {
                        msg.push_str(&format!("       {} {}\n", "Summary:".dimmed(), summary));
                    }
                }
            }
        }
    }
    msg
}

fn candidate_symbol_name(p: &Pair) -> Option<String> {
    let r = p.replacement.as_str();
    if r.ends_with('[') {
        return Some(r.trim_end_matches('[').to_string());
    }
    if r.starts_with('"') && r.ends_with('"') && r.len() >= 2 {
        return Some(r[1..r.len() - 1].to_string());
    }
    if !r.is_empty() {
        return Some(r.to_string());
    }
    None
}

struct ShowCompletionMenuHandler {
    printer: std::sync::Mutex<Box<dyn ExternalPrinter + Send>>,
    builtins: std::sync::Arc<Vec<BuiltinEntry>>,
    env_names: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    state: std::sync::Arc<std::sync::Mutex<CompletionMenuState>>,
    pkg_exports: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, Vec<String>>>>,
    assoc_keys: std::sync::Arc<Vec<String>>,
    doc_index: std::sync::Arc<DocIndex>,
}

impl rustyline::ConditionalEventHandler for ShowCompletionMenuHandler {
    fn handle(
        &self,
        _evt: &Event,
        _n: rustyline::RepeatCount,
        _positive: bool,
        ctx: &rustyline::EventContext,
    ) -> Option<Cmd> {
        let line = ctx.line();
        let pos = ctx.pos();
        let (start, word) = current_symbol_token(line, pos);
        if word.is_empty() {
            return None;
        }
        let mut scored: Vec<(i64, Pair)> = Vec::new();
        // Context-aware: Using[...] and assoc key
        if let Some(ctx) = using_context(line, pos) {
            match ctx {
                UsingCtx::PackageName(prefix) => {
                    if let Ok(pkgs) = self.pkg_exports.lock() {
                        for name in pkgs.keys() {
                            if let Some(s) = fuzzy_score(name, &prefix) {
                                scored.push((
                                    s + 1200,
                                    Pair {
                                        display: format!("{} — (package)", name),
                                        replacement: name.clone(),
                                    },
                                ));
                            }
                        }
                    }
                }
                UsingCtx::OptionKey(prefix) => {
                    for k in ["Import", "Except"].into_iter() {
                        if let Some(s) = fuzzy_score(k, &prefix) {
                            scored.push((
                                s + 1100,
                                Pair { display: k.to_string(), replacement: k.to_string() },
                            ));
                        }
                    }
                }
                UsingCtx::ImportValue { pkg, prefix } => {
                    if let Ok(pkgs) = self.pkg_exports.lock() {
                        if let Some(exports) = pkgs.get(&pkg) {
                            for e in exports {
                                if let Some(s) = fuzzy_score(e, &prefix) {
                                    scored.push((
                                        s + 1300,
                                        Pair {
                                            display: e.clone(),
                                            replacement: format!("\"{}\"", e),
                                        },
                                    ));
                                }
                            }
                            if let Some(s) = fuzzy_score("All", &prefix) {
                                scored.push((
                                    s + 1000,
                                    Pair { display: "All".into(), replacement: "All".into() },
                                ));
                            }
                        }
                    }
                }
            }
        } else if let Some(prefix) = assoc_key_prefix(line, pos) {
            for k in self.assoc_keys.iter() {
                if let Some(s) = fuzzy_score(k, &prefix) {
                    scored.push((s + 1000, Pair { display: k.clone(), replacement: k.clone() }));
                }
            }
        } else if let Some((key, vprefix, quoted)) = assoc_value_context(line, pos) {
            let mut values: Vec<String> = Vec::new();
            let lower = key.to_lowercase();
            if ["inplace", "dryrun", "verbose", "pretty"].contains(&lower.as_str()) {
                values.extend(vec!["True".into(), "False".into()]);
            }
            if lower == "cache" {
                values.extend(vec!["session".into(), "none".into()]);
            }
            if lower == "output" {
                values.extend(vec!["expr".into(), "json".into()]);
            }
            if lower == "assoc" || lower == "assocrender" {
                values.extend(vec!["auto".into(), "inline".into(), "pretty".into()]);
            }
            if lower == "import" {
                values.push("All".into());
            }
            values.extend(vec!["Null".into(), "0".into(), "1".into(), "\"\"".into()]);
            for val in values.into_iter() {
                if let Some(s) = fuzzy_score(&val, &vprefix) {
                    let rep = if val == "True"
                        || val == "False"
                        || val == "Null"
                        || val.chars().all(|c| c.is_ascii_digit())
                        || val.starts_with('"')
                    {
                        val.clone()
                    } else {
                        format!("\"{}\"", val)
                    };
                    scored.push((
                        s + 900,
                        Pair { display: val, replacement: if quoted { rep.clone() } else { rep } },
                    ));
                }
            }
        } else {
            // Default: builtins + env vars
            for b in self.builtins.iter() {
                if let Some(s) = fuzzy_score(&b.name, &word) {
                    if b.name == word {
                        continue;
                    }
                    let attrs = if b.attrs.is_empty() {
                        String::new()
                    } else {
                        format!(" [{}]", b.attrs.join(", "))
                    };
                    let display = if b.summary.is_empty() {
                        format!("{}{}", b.name, attrs)
                    } else {
                        format!("{} — {}{}", b.name, b.summary, attrs)
                    };
                    scored.push((s + 1000, Pair { display, replacement: format!("{}[", b.name) }));
                }
            }
            if let Ok(envs) = self.env_names.lock() {
                for n in envs.iter() {
                    if let Some(s) = fuzzy_score(n, &word) {
                        if n == &word {
                            continue;
                        }
                        scored.push((
                            s + 800,
                            Pair { display: format!("{} — (var)", n), replacement: n.clone() },
                        ));
                    }
                }
            }
        }
        if scored.is_empty() {
            return None;
        }
        scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.display.cmp(&b.1.display)));
        let pairs: Vec<Pair> = scored.into_iter().take(10).map(|(_, p)| p).collect();
        if let Ok(mut st) = self.state.lock() {
            st.items = pairs;
            st.prefix_len = pos.saturating_sub(start);
            st.selected_idx = 0;
            st.active = !st.items.is_empty();
            if let Ok(mut pr) = self.printer.lock() {
                let _ = pr.print(render_completion_panel(&st, &self.doc_index));
            }
        }
        Some(Cmd::Noop)
    }
}

struct PickCompletionHandler {
    state: std::sync::Arc<std::sync::Mutex<CompletionMenuState>>,
    index: usize,
}

impl rustyline::ConditionalEventHandler for PickCompletionHandler {
    fn handle(
        &self,
        _evt: &Event,
        _n: rustyline::RepeatCount,
        _positive: bool,
        _ctx: &rustyline::EventContext,
    ) -> Option<Cmd> {
        if let Ok(st) = self.state.lock() {
            let idx = if self.index == 9 { 9 } else { self.index.saturating_sub(1) };
            if idx < st.items.len() {
                let repl = st.items[idx].replacement.clone();
                let back = st.prefix_len.max(0);
                return Some(Cmd::Replace(Movement::BackwardChar(back), Some(repl)));
            }
        }
        Some(Cmd::Noop)
    }
}

struct CycleCompletionMenuHandler {
    printer: std::sync::Mutex<Box<dyn ExternalPrinter + Send>>,
    state: std::sync::Arc<std::sync::Mutex<CompletionMenuState>>,
    delta: isize,
    doc_index: std::sync::Arc<DocIndex>,
}

impl rustyline::ConditionalEventHandler for CycleCompletionMenuHandler {
    fn handle(
        &self,
        _evt: &Event,
        _n: rustyline::RepeatCount,
        _positive: bool,
        _ctx: &rustyline::EventContext,
    ) -> Option<Cmd> {
        if let Ok(mut st) = self.state.lock() {
            if !st.active || st.items.is_empty() {
                return None;
            }
            let len = st.items.len();
            let cur = st.selected_idx as isize;
            let mut next = cur + self.delta;
            if next < 0 {
                next = len as isize - 1;
            }
            if next >= len as isize {
                next = 0;
            }
            st.selected_idx = next as usize;
            if let Ok(mut pr) = self.printer.lock() {
                let _ = pr.print(render_completion_panel(&st, &self.doc_index));
            }
            return Some(Cmd::Noop);
        }
        None
    }
}

struct AcceptCompletionMenuHandler {
    state: std::sync::Arc<std::sync::Mutex<CompletionMenuState>>,
}

impl rustyline::ConditionalEventHandler for AcceptCompletionMenuHandler {
    fn handle(
        &self,
        _evt: &Event,
        _n: rustyline::RepeatCount,
        _positive: bool,
        _ctx: &rustyline::EventContext,
    ) -> Option<Cmd> {
        if let Ok(mut st) = self.state.lock() {
            if !st.active || st.items.is_empty() {
                return None;
            }
            let idx = st.selected_idx.min(st.items.len() - 1);
            let repl = st.items[idx].replacement.clone();
            let back = st.prefix_len.max(0);
            st.active = false;
            return Some(Cmd::Replace(Movement::BackwardChar(back), Some(repl)));
        }
        None
    }
}

fn print_usage() {
    eprintln!("Lyra — Unified CLI\nUSAGE:\n  lyra                Start interactive REPL\n  lyra mcp            Run MCP server over stdio\n  lyra --help         Show this help\n  lyra --version      Show version");
}

fn main() -> Result<()> {
    // Unified CLI: early check for MCP mode or help/version
    {
        let mut args_iter = std::env::args().skip(1);
        if let Some(first) = args_iter.next() {
            match first.as_str() {
                "mcp" | "--mcp" => {
                    return lyra_mcp::run_stdio_server();
                }
                "fmt" | "format" => {
                    let files: Vec<String> = args_iter.collect();
                    if files.is_empty() {
                        eprintln!("usage: lyra fmt <files...>");
                        return Ok(());
                    }
                    lyra_runtime::set_default_registrar(lyra_stdlib::register_all);
                    let mut ev = Evaluator::new();
                    lyra_stdlib::register_all(&mut ev);
                    let mut changed = 0i64;
                    let total = files.len() as i64;
                    for f in files {
                        match std::fs::read_to_string(&f) {
                            Ok(src) => {
                                let mut p = lyra_parser::Parser::from_source(&src);
                                if let Err(e) = p.parse_all_detailed() {
                                    print_parse_error(&f, &src, e.pos, &e.message);
                                    continue;
                                }
                                let v = ev.eval(Value::Expr {
                                    head: Box::new(Value::Symbol("FormatLyraFile".into())),
                                    args: vec![Value::String(f.clone())],
                                });
                                match v {
                                    Value::Assoc(m) => {
                                        let ch =
                                            matches!(m.get("changed"), Some(Value::Boolean(true)));
                                        if ch {
                                            changed += 1;
                                        }
                                        println!("{}: {}", f, if ch { "formatted" } else { "ok" });
                                    }
                                    _ => println!("{}: error", f),
                                }
                            }
                            Err(e) => {
                                eprintln!("{}: read error: {}", f, e);
                            }
                        }
                    }
                    println!("formatted {} / {}", changed, total);
                    return Ok(());
                }
                "--reedline" => {
                    #[cfg(feature = "reedline")]
                    {
                        return reedline_mode::run();
                    }
                    #[cfg(not(feature = "reedline"))]
                    {
                        eprintln!("Rebuild with --features reedline to use reedline REPL");
                        return Ok(());
                    }
                }
                "lint" => {
                    let files: Vec<String> = args_iter.collect();
                    if files.is_empty() {
                        eprintln!("usage: lyra lint <files...>");
                        return Ok(());
                    }
                    lyra_runtime::set_default_registrar(lyra_stdlib::register_all);
                    let mut ev = Evaluator::new();
                    lyra_stdlib::register_all(&mut ev);
                    let mut total_issues = 0i64;
                    for f in files {
                        match std::fs::read_to_string(&f) {
                            Ok(src) => {
                                let mut p = lyra_parser::Parser::from_source(&src);
                                if let Err(e) = p.parse_all_detailed() {
                                    print_parse_error(&f, &src, e.pos, &e.message);
                                    total_issues += 1;
                                    continue;
                                }
                                let v = ev.eval(Value::Expr {
                                    head: Box::new(Value::Symbol("LintLyraFile".into())),
                                    args: vec![Value::String(f.clone())],
                                });
                                match v {
                                    Value::Assoc(m) => {
                                        if let Some(Value::List(issues)) = m.get("issues") {
                                            let n = issues.len() as i64;
                                            total_issues += n;
                                            println!("{}: {} issues", f, n);
                                        } else {
                                            println!("{}: ok", f);
                                        }
                                    }
                                    _ => println!("{}: error", f),
                                }
                            }
                            Err(e) => {
                                eprintln!("{}: read error: {}", f, e);
                                total_issues += 1;
                            }
                        }
                    }
                    println!("total issues: {}", total_issues);
                    return Ok(());
                }
                "test" => {
                    let path = if let Some(p) = args_iter.next() {
                        p
                    } else {
                        eprintln!("usage: lyra test <file.lyra>");
                        return Ok(());
                    };
                    lyra_runtime::set_default_registrar(lyra_stdlib::register_all);
                    let mut ev = Evaluator::new();
                    lyra_stdlib::register_all(&mut ev);
                    match std::fs::read_to_string(&path) {
                        Ok(src) => {
                            let mut parser = lyra_parser::Parser::from_source(&src);
                            match parser.parse_all_with_ranges() {
                                Ok(exprs) => {
                                    let mut last = Value::Symbol("Null".into());
                                    for (e, s, _e) in exprs {
                                        ev.set_current_span(Some((s, _e)));
                                        let out = ev.eval(e);
                                        ev.set_current_span(None);
                                        if let Value::Assoc(m) = &out {
                                            let is_err = matches!(
                                                m.get("error"),
                                                Some(Value::Boolean(true))
                                            ) || m.get("message").is_some();
                                            if is_err {
                                                let (mut pos, mut msg) = (s, String::from("error"));
                                                if let Some(Value::String(mtxt)) = m.get("message")
                                                {
                                                    msg = mtxt.clone();
                                                }
                                                if let Some(Value::Assoc(sp)) = m.get("span") {
                                                    if let (
                                                        Some(Value::Integer(ss)),
                                                        Some(Value::Integer(_ee)),
                                                    ) = (sp.get("start"), sp.get("end"))
                                                    {
                                                        pos = *ss as usize;
                                                    }
                                                }
                                                print_parse_error(&path, &src, pos, &msg);
                                            }
                                        }
                                        last = out;
                                    }
                                    let spec = ev.eval(Value::Expr {
                                        head: Box::new(Value::Symbol("TestSpec".into())),
                                        args: vec![last.clone()],
                                    });
                                    println!("{}", lyra_core::pretty::format_value(&spec));
                                    return Ok(());
                                }
                                Err(e) => {
                                    print_parse_error(&path, &src, e.pos, &e.message);
                                    return Ok(());
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("error: {}", e);
                            return Ok(());
                        }
                    }
                }
                "-h" | "--help" | "help" => {
                    print_usage();
                    return Ok(());
                }
                "--version" | "version" => {
                    println!("{}", env!("CARGO_PKG_VERSION"));
                    return Ok(());
                }
                _ => { /* fall through to REPL for unknown args */ }
            }
        }
    }
    // Build evaluator and register stdlib first (for builtin discovery)
    println!("{}", "Lyra REPL (prototype)".bright_yellow().bold());
    // Ensure spawned Evaluators (e.g., in Futures) inherit stdlib
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);

    // Discover builtins for completion and docs
    let builtins = discover_builtins(&mut ev);
    let doc_index = build_doc_index(&mut ev);
    let cfg = std::sync::Arc::new(std::sync::Mutex::new(ReplConfig {
        pager_on: false,
        assoc_mode: AssocMode::Auto,
        print_mode: PrintMode::Expr,
    }));
    let repl_cmds = vec![
        "help".into(),
        "env".into(),
        "funcs".into(),
        "sym".into(),
        "defs".into(),
        "explain".into(),
        "profile".into(),
        "pager".into(),
        "watch".into(),
        "save".into(),
        "load".into(),
        "more".into(),
        "json".into(),
        "history".into(),
        "using".into(),
        "edit".into(),
        "imports".into(),
        "loaded".into(),
        "exports".into(),
        "register-exports".into(),
        "newpkg".into(),
        "newmod".into(),
        "mode".into(),
        "set".into(),
    ];
    let common_option_keys = vec![
        "MaxThreads".into(),
        "TimeBudgetMs".into(),
        "Import".into(),
        "Except".into(),
        "replacement".into(),
        "inPlace".into(),
        "dryRun".into(),
        "backupExt".into(),
    ];
    let assoc_keys_arc = std::sync::Arc::new(common_option_keys.clone());
    let helper = ReplHelper {
        builtins: builtins.clone(),
        env_names: Vec::new(),
        doc_index: doc_index.clone(),
        cfg: cfg.clone(),
        pkg_exports: std::collections::HashMap::new(),
        repl_cmds,
        common_option_keys,
    };

    // Editor with helper for completion and hints
    // Try to enable a more discoverable completion experience (list/menu style)
    let rl_cfg = rustyline::Config::builder()
        .history_ignore_dups(true)
        .expect("config")
        .history_ignore_space(true)
        .auto_add_history(false)
        .completion_type(rustyline::CompletionType::Circular)
        .completion_prompt_limit(50)
        .build();
    let mut rl = Editor::<ReplHelper, DefaultHistory>::with_config(rl_cfg)?;
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
        EventHandler::Conditional(Box::new(DocKeyHandler {
            printer: std::sync::Mutex::new(Box::new(printer)),
            doc_index: doc_index.clone(),
            cfg: cfg.clone(),
        })),
    );
    let printer2 = rl.create_external_printer().expect("external printer");
    rl.bind_sequence(
        Event::from(KeyEvent::alt('h')),
        EventHandler::Conditional(Box::new(DocKeyHandler {
            printer: std::sync::Mutex::new(Box::new(printer2)),
            doc_index: doc_index.clone(),
            cfg: cfg.clone(),
        })),
    );
    // Bind Tab to also show quick docs on exact symbol match (non-destructive)
    let printer3 = rl.create_external_printer().expect("external printer");
    rl.bind_sequence(
        Event::from(KeyEvent(KeyCode::Tab, Modifiers::NONE)),
        EventHandler::Conditional(Box::new(TabDocAssistHandler {
            printer: std::sync::Mutex::new(Box::new(printer3)),
            doc_index: doc_index.clone(),
            cfg: cfg.clone(),
        })),
    );
    // Note: rustyline v13 lists candidates with CompletionType::List; cycling bindings depend on upstream support
    // Completion menu state + handlers
    let menu_state = std::sync::Arc::new(std::sync::Mutex::new(CompletionMenuState::default()));
    let env_names_shared: std::sync::Arc<std::sync::Mutex<Vec<String>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let printer_menu = rl.create_external_printer().expect("external printer");
    let pkg_exports_shared: std::sync::Arc<
        std::sync::Mutex<std::collections::HashMap<String, Vec<String>>>,
    > = std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
    rl.bind_sequence(
        Event::from(KeyEvent::alt('/')),
        EventHandler::Conditional(Box::new(ShowCompletionMenuHandler {
            printer: std::sync::Mutex::new(Box::new(printer_menu)),
            builtins: std::sync::Arc::new(builtins.clone()),
            env_names: env_names_shared.clone(),
            state: menu_state.clone(),
            pkg_exports: pkg_exports_shared.clone(),
            assoc_keys: assoc_keys_arc.clone(),
            doc_index: doc_index.clone(),
        })),
    );
    // Alt-j / Alt-k to move selection
    let printer_menu2 = rl.create_external_printer().expect("external printer");
    rl.bind_sequence(
        Event::from(KeyEvent::alt('j')),
        EventHandler::Conditional(Box::new(CycleCompletionMenuHandler {
            printer: std::sync::Mutex::new(Box::new(printer_menu2)),
            state: menu_state.clone(),
            delta: 1,
            doc_index: doc_index.clone(),
        })),
    );
    let printer_menu3 = rl.create_external_printer().expect("external printer");
    rl.bind_sequence(
        Event::from(KeyEvent::alt('k')),
        EventHandler::Conditional(Box::new(CycleCompletionMenuHandler {
            printer: std::sync::Mutex::new(Box::new(printer_menu3)),
            state: menu_state.clone(),
            delta: -1,
            doc_index: doc_index.clone(),
        })),
    );
    // Enter accepts selection if panel is active
    rl.bind_sequence(
        Event::from(KeyEvent(KeyCode::Enter, Modifiers::NONE)),
        EventHandler::Conditional(Box::new(AcceptCompletionMenuHandler {
            state: menu_state.clone(),
        })),
    );
    // Alt-1..9 and Alt-0 to pick
    for (i, key) in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'].iter().enumerate() {
        rl.bind_sequence(
            Event::from(KeyEvent::alt(*key)),
            EventHandler::Conditional(Box::new(PickCompletionHandler {
                state: menu_state.clone(),
                index: i + 1,
            })),
        );
    }
    // Bind Shift-Enter and Ctrl-Enter to insert newline explicitly
    rl.bind_sequence(Event::from(KeyEvent(KeyCode::Enter, Modifiers::SHIFT)), Cmd::Newline);
    rl.bind_sequence(Event::from(KeyEvent(KeyCode::Enter, Modifiers::CTRL)), Cmd::Newline);
    // Load history if present
    if let Some(p) = history_path() {
        let _ = rl.load_history(&p);
    }
    // Print quick help of new features
    println!("{}", "Tip: Tab = fuzzy-complete (and docs on exact); F1/Alt-h = docs; :format assoc/output; :history; :help".dimmed());
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
            let names = ev.env_keys();
            h.env_names = names.clone();
            if let Ok(mut m) = env_names_shared.lock() {
                *m = names;
            }
            let exports = collect_pkg_exports(&mut ev);
            h.pkg_exports = exports.clone();
            if let Ok(mut pe) = pkg_exports_shared.lock() {
                *pe = exports;
            }
        }
        match read_repl_input(&mut rl, &format!("In[{}]> ", in_counter)) {
            Ok(line) => {
                // Add to history once per submitted buffer
                let _ = rl.add_history_entry(line.as_str());
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if record_on {
                    append_transcript(&record_path, &format!("In[{}]: {}\n", in_counter, line));
                }
                // Built-in lightweight commands (not expressions)
                if trimmed == ":help" {
                    print_repl_help();
                    continue;
                }
                // Allow setting truncation config: :set truncate key=value
                if let Some(rest) = trimmed.strip_prefix(":set ") {
                    let parts: Vec<&str> = rest.split('=').map(|s| s.trim()).collect();
                    if parts.len() == 2 {
                        match parts[0] {
                            "max_list" => {
                                if let Ok(n) = parts[1].parse::<usize>() {
                                    truncate_cfg.max_list = n
                                }
                            }
                            "max_assoc" => {
                                if let Ok(n) = parts[1].parse::<usize>() {
                                    truncate_cfg.max_assoc = n
                                }
                            }
                            "max_string" => {
                                if let Ok(n) = parts[1].parse::<usize>() {
                                    truncate_cfg.max_string = n
                                }
                            }
                            _ => {}
                        }
                        println!("{} {}", "Set".green().bold(), rest);
                        continue;
                    }
                }
                // :doc Name [--json] -> show documentation card or JSON
                if let Some(rest) = trimmed.strip_prefix(":doc ") {
                    let mut parts = rest.split_whitespace().collect::<Vec<_>>();
                    let use_json = parts.iter().any(|p| *p == "--json" || *p == "-j");
                    parts.retain(|p| *p != "--json" && *p != "-j");
                    let sym = parts.first().copied().unwrap_or("");
                    if sym.is_empty() {
                        println!("Usage: :doc Name [--json]");
                        continue;
                    }
                    if use_json {
                        // Documentation["Name"] -> JSON
                        let card = ev.eval(Value::Expr {
                            head: Box::new(Value::Symbol("Documentation".into())),
                            args: vec![Value::String(sym.to_string())],
                        });
                        match serde_json::to_string_pretty(&card) {
                            Ok(s) => println!("{}", s),
                            Err(e) => eprintln!("doc json error: {}", e),
                        }
                    } else {
                        render_doc_card(&mut ev, sym);
                    }
                    continue;
                }
                // :env -> list variable names
                if trimmed == ":env" {
                    render_env(&ev);
                    continue;
                }
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
                    println!("Explain {}", if v { "on".green() } else { "off".yellow() });
                    continue;
                }
                // :profile on|off
                if let Some(rest) = trimmed.strip_prefix(":profile ") {
                    let v = rest.trim().eq_ignore_ascii_case("on");
                    profile_on = v;
                    println!("Profile {}", if v { "on".green() } else { "off".yellow() });
                    continue;
                }
                // :pager on|off
                if let Some(rest) = trimmed.strip_prefix(":pager ") {
                    let on = rest.trim().eq_ignore_ascii_case("on");
                    if let Ok(mut c) = cfg.lock() {
                        c.pager_on = on;
                    }
                    println!("Pager {}", if on { "on".green() } else { "off".yellow() });
                    continue;
                }
                // :format assoc auto|inline|pretty  OR  :format output expr|json
                if let Some(rest) = trimmed.strip_prefix(":format ") {
                    let mut parts = rest.split_whitespace();
                    let cat = parts.next().unwrap_or("");
                    let val = parts.next().unwrap_or("");
                    match (cat, val.to_lowercase().as_str()) {
                        ("assoc", "auto") => {
                            if let Ok(mut c) = cfg.lock() {
                                c.assoc_mode = AssocMode::Auto;
                                println!("format assoc -> auto");
                            }
                        }
                        ("assoc", "inline") => {
                            if let Ok(mut c) = cfg.lock() {
                                c.assoc_mode = AssocMode::Inline;
                                println!("format assoc -> inline");
                            }
                        }
                        ("assoc", "pretty") => {
                            if let Ok(mut c) = cfg.lock() {
                                c.assoc_mode = AssocMode::Pretty;
                                println!("format assoc -> pretty");
                            }
                        }
                        ("output", "expr") => {
                            if let Ok(mut c) = cfg.lock() {
                                c.print_mode = PrintMode::Expr;
                                println!("format output -> expr");
                            }
                        }
                        ("output", "json") => {
                            if let Ok(mut c) = cfg.lock() {
                                c.print_mode = PrintMode::Json;
                                println!("format output -> json");
                            }
                        }
                        _ => println!(
                            "Usage: :format assoc auto|inline|pretty | :format output expr|json"
                        ),
                    }
                    continue;
                }
                // :mode vi|emacs
                if let Some(rest) = trimmed.strip_prefix(":mode ") {
                    let m = rest.trim().to_lowercase();
                    match m.as_str() {
                        "vi" => rl.set_edit_mode(rustyline::EditMode::Vi),
                        "emacs" => rl.set_edit_mode(rustyline::EditMode::Emacs),
                        _ => {}
                    }
                    println!("Mode set to {}", m);
                    continue;
                }
                // :record on|off [file]
                if let Some(rest) = trimmed.strip_prefix(":record ") {
                    let mut parts = rest.split_whitespace();
                    match parts.next().unwrap_or("") {
                        "on" => {
                            let path = parts.next().map(|s| s.to_string()).unwrap_or_else(|| {
                                let mut p = std::env::temp_dir();
                                p.push("lyra_transcript.txt");
                                p.to_string_lossy().to_string()
                            });
                            record_path = Some(std::path::PathBuf::from(path.clone()));
                            record_on = true;
                            println!("Recording to {}", path);
                        }
                        "off" => {
                            record_on = false;
                            println!("Recording off");
                        }
                        _ => println!("Usage: :record on [file] | :record off"),
                    }
                    continue;
                }
                // :watch add|rm name expr
                if let Some(rest) = trimmed.strip_prefix(":watch ") {
                    let parts: Vec<&str> = rest.splitn(3, ' ').collect();
                    if parts.len() >= 2 {
                        match parts[0] {
                            "add" if parts.len() == 3 => {
                                watches.push((parts[1].to_string(), parts[2].to_string()));
                                println!("Added watch {}", parts[1]);
                            }
                            "rm" => {
                                let name = parts[1];
                                let before = watches.len();
                                watches.retain(|(n, _)| n != name);
                                println!("Removed {}", name);
                                if watches.len() == before {
                                    println!("{}", "(not found)".dimmed());
                                }
                            }
                            "list" => {
                                for (n, e) in &watches {
                                    println!("{}: {}", n, e);
                                }
                                if watches.is_empty() {
                                    println!("{}", "(no watches)".dimmed());
                                }
                            }
                            _ => println!(
                                "Usage: :watch add <name> <expr> | :watch rm <name> | :watch list"
                            ),
                        }
                        continue;
                    }
                }
                // :save file.json -> save env
                if let Some(rest) = trimmed.strip_prefix(":save ") {
                    let file = rest.trim();
                    if !file.is_empty() {
                        if let Err(e) = save_env_json(&mut ev, file) {
                            eprintln!("save error: {e}");
                        } else {
                            println!("Saved {}", file);
                        }
                    }
                    continue;
                }
                // :load file.json -> load env (overwrites vars)
                if let Some(rest) = trimmed.strip_prefix(":load ") {
                    let file = rest.trim();
                    if !file.is_empty() {
                        if let Err(e) = load_env_json(&mut ev, file) {
                            eprintln!("load error: {e}");
                        } else {
                            println!("Loaded {}", file);
                        }
                    }
                    continue;
                }
                // :more -> reprint last output fully
                if trimmed == ":more" {
                    if let Some(v) = &last_output {
                        println!("{}", format_value_color(v, None, AssocMode::Auto));
                    } else {
                        println!("{}", "No previous output".dimmed());
                    }
                    continue;
                }
                // :json -> print last output as pretty JSON
                if trimmed == ":json" {
                    if let Some(v) = &last_output {
                        match serde_json::to_string_pretty(v) {
                            Ok(s) => println!("{}", s),
                            Err(e) => eprintln!("json error: {e}"),
                        }
                    } else {
                        println!("{}", "No previous output".dimmed());
                    }
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
                    if let Err(e) = handle_using_cmd(&mut ev, rest) {
                        eprintln!("{} {}", "Using:".red().bold(), e);
                    }
                    continue;
                }
                // :edit [file] -> open EDITOR to write/eval a buffer
                if let Some(rest) = trimmed.strip_prefix(":edit") {
                    let path = rest.trim();
                    if let Err(e) = edit_and_eval(&mut ev, path) {
                        eprintln!("edit error: {e}");
                    }
                    continue;
                }
                // Simple help system: ?help or ?Symbol
                if trimmed.starts_with('?') {
                    handle_help(trimmed, &doc_index);
                    continue;
                }
                // Echo input like In[n]:= expr
                println!("In[{}]:= {}", in_counter, line.trim());
                let mut p = Parser::from_source(&line);
                match p.parse_all_with_ranges() {
                    Ok(values) => {
                        for (v, start, _end) in values {
                            // Attach top-level span for better runtime error reporting
                            ev.set_current_span(Some((start, _end)));
                            let t0 = std::time::Instant::now();
                            let out = ev.eval(v);
                            ev.set_current_span(None);
                            let elapsed = t0.elapsed();
                            let suffix = result_suffix(&out, elapsed);
                            // Save last output
                            last_output = Some(out.clone());
                            // Maintain % binding and an OutList for retrieval via OutList[[n]]
                            out_counter += 1;
                            out_buf.push(out.clone());
                            let _ = ev.eval(Value::Expr {
                                head: Box::new(Value::Symbol("Set".into())),
                                args: vec![Value::Symbol("%".into()), out.clone()],
                            });
                            let _ = ev.eval(Value::Expr {
                                head: Box::new(Value::Symbol("Set".into())),
                                args: vec![
                                    Value::Symbol("OutList".into()),
                                    Value::List(out_buf.clone()),
                                ],
                            });
                            let (assoc_mode, print_mode) = {
                                if let Ok(c) = cfg.lock() {
                                    (c.assoc_mode, c.print_mode)
                                } else {
                                    (AssocMode::Auto, PrintMode::Expr)
                                }
                            };
                            // Detect failure shape and print as error with caret under top-level expr
                            let mut is_error = false;
                            let mut err_msg = String::new();
                            let mut span_from_val: Option<(usize, usize)> = None;
                            if let Value::Assoc(m) = &out {
                                if matches!(m.get("error"), Some(Value::Boolean(true)))
                                    || m.get("message").is_some()
                                {
                                    is_error = true;
                                }
                                if let Some(Value::String(msg)) = m.get("message") {
                                    err_msg = msg.clone();
                                }
                                if let Some(Value::Assoc(sp)) = m.get("span") {
                                    let s = sp.get("start").and_then(|v| {
                                        if let Value::Integer(n) = v {
                                            Some(*n as usize)
                                        } else {
                                            None
                                        }
                                    });
                                    let e = sp.get("end").and_then(|v| {
                                        if let Value::Integer(n) = v {
                                            Some(*n as usize)
                                        } else {
                                            None
                                        }
                                    });
                                    if let (Some(sv), Some(ev2)) = (s, e) {
                                        span_from_val = Some((sv, ev2));
                                    }
                                }
                            }
                            if is_error {
                                let err_start = span_from_val.map(|(s, _e)| s).unwrap_or(start);
                                let (line_no, col) = byte_to_line_col(&line, err_start);
                                eprintln!("{}:{}:{}: error: {}", "<repl>", line_no, col, err_msg);
                                if let Some((_ln, text)) = line_text(&line, line_no) {
                                    eprintln!("     | {}", text);
                                    eprintln!("     | {}", caret_line(&text, col));
                                }
                            }
                            let rendered = match print_mode {
                                PrintMode::Expr => {
                                    format_value_color(&out, Some(truncate_cfg), assoc_mode)
                                }
                                PrintMode::Json => serde_json::to_string_pretty(&out)
                                    .unwrap_or_else(|_| "<json error>".into()),
                            };
                            let out_line = format!("Out[{}]= {}{}", out_counter, rendered, suffix);
                            println!("{}", out_line);
                            if record_on {
                                append_transcript(
                                    &record_path,
                                    &format!("{}\n", strip_ansi(&out_line)),
                                );
                            }
                            if explain_on || profile_on {
                                let steps = explain_steps(&mut ev, out);
                                if explain_on {
                                    render_explain_steps(&steps);
                                }
                                if profile_on {
                                    render_profile_summary(&steps);
                                }
                            }
                            // Evaluate watches
                            if !watches.is_empty() {
                                for (n, expr_src) in &watches {
                                    let mut p2 = Parser::from_source(expr_src);
                                    if let Ok(vs) = p2.parse_all() {
                                        if let Some(expr) = vs.last() {
                                            let val = ev.eval(expr.clone());
                                            println!(
                                                "{} = {}",
                                                n.bold(),
                                                format_value_color(
                                                    &val,
                                                    Some(truncate_cfg),
                                                    AssocMode::Auto
                                                )
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => render_parse_error(&rl, &line, &e.message),
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("^C");
                continue;
            }
            Err(ReadlineError::Eof) => {
                println!("^D");
                break;
            }
            Err(e) => {
                eprintln!("readline error: {e}");
                break;
            }
        }
        in_counter += 1;
    }
    // Save history on exit
    if let Some(p) = history_path() {
        let _ = rl.save_history(&p);
    }
    Ok(())
}

fn read_repl_input(
    rl: &mut Editor<ReplHelper, DefaultHistory>,
    first_prompt: &str,
) -> rustyline::Result<String> {
    let mut buf = String::new();
    let mut first = true;
    loop {
        let prompt = if first { first_prompt } else { "… " };
        first = false;
        let indent = {
            let depth = bracket_depth(&buf);
            if depth == 0 {
                String::new()
            } else {
                "  ".repeat(depth)
            }
        };
        let line = if indent.is_empty() {
            rl.readline(prompt)?
        } else {
            rl.readline_with_initial(prompt, (&indent, ""))?
        };
        // ctrl-d at empty second prompt should bubble up
        buf.push_str(&line);
        if scan_unbalanced(&buf).is_some() {
            buf.push('\n');
            continue;
        }
        break;
    }
    Ok(buf)
}

fn handle_help(q: &str, index: &std::sync::Arc<DocIndex>) {
    let topic = q.trim_start_matches('?').trim();
    if topic.is_empty() || topic.eq_ignore_ascii_case("help") {
        let header = format!("{}", "Lyra REPL help".bright_green().bold());
        let body = "  - ?help: show this help\n  - ?Symbol: show a short description (e.g., ?Plus)\n  - Tab: autocomplete builtins; inline doc hints as you type\n  - Expressions use f[x, y], {a, b}, <|k->v|>\n  - Try: Explain[Plus[1, 2]] or Schema[<|\"a\"->1|>]";
        println!("{}\n{}", header, body);
        return;
    }
    // Prefer full documentation from index (seeded by Documentation[])
    // Fallback in the builder prints "No documentation yet." if unknown.
    let card = build_doc_card_str_from_index(index, topic);
    println!("{}", card);
}

fn print_repl_help() {
    let lines = [
        ":help                — show REPL commands",
        ":env                 — list variables",
        ":funcs [filter]      — list builtins",
        ":doc Name [--json]   — show documentation (card or JSON)",
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
    for l in lines {
        println!("  {}", l);
    }
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
    let resp = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("DescribeBuiltins".into())),
        args: vec![],
    });
    if let Value::List(items) = resp {
        for it in items {
            if let Value::Assoc(m) = it {
                if let Some(Value::String(name)) = m.get("name") {
                    // Prefer summary from DescribeBuiltins (filled by docs registry), fallback to builtin_help
                    let sum = match m.get("summary") {
                        Some(Value::String(s)) if !s.is_empty() => s.as_str(),
                        _ => builtin_help(name),
                    };
                    let attrs: Vec<String> = match m.get("attributes") {
                        Some(Value::List(vs)) => vs
                            .iter()
                            .filter_map(|v| {
                                if let Value::String(s) = v {
                                    Some(s.clone())
                                } else {
                                    None
                                }
                            })
                            .collect(),
                        _ => Vec::new(),
                    };
                    entries.push(BuiltinEntry {
                        name: name.clone(),
                        summary: sum.to_string(),
                        attrs,
                    });
                }
            }
        }
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        entries.dedup_by(|a, b| a.name == b.name);
        return entries;
    }
    // Fallback to a minimal static set if DescribeBuiltins is unavailable
    let names = [
        "Plus",
        "Times",
        "Minus",
        "Divide",
        "Power",
        "Replace",
        "ReplaceAll",
        "ReplaceFirst",
        "Set",
        "With",
        "Schema",
        "Explain",
    ];
    for n in names {
        entries.push(BuiltinEntry {
            name: n.to_string(),
            summary: builtin_help(n).to_string(),
            attrs: Vec::new(),
        });
    }
    entries
}

#[derive(Clone, Copy)]
struct TruncateCfg {
    max_list: usize,
    max_assoc: usize,
    max_string: usize,
}

fn result_suffix(v: &Value, elapsed: std::time::Duration) -> String {
    let ms = (elapsed.as_secs_f64() * 1000.0).round() as i64;
    let size = match v {
        Value::List(items) => Some(items.len()),
        Value::Assoc(m) => Some(m.len()),
        _ => None,
    };
    match size {
        Some(n) => format!("  {}", format!("({} ms, {} items)", ms, n).dimmed()),
        None => format!("  {}", format!("({} ms)", ms).dimmed()),
    }
}

fn format_value_color(v: &Value, trunc: Option<TruncateCfg>, assoc_mode: AssocMode) -> String {
    match v {
        Value::Integer(n) => format!("{}", n).bright_blue().to_string(),
        Value::Real(x) => format!("{}", x).bright_blue().to_string(),
        Value::BigReal(s) => s.bright_blue().to_string(),
        Value::Rational { num, den } => format!("{}/{}", num, den).bright_blue().to_string(),
        Value::Complex { re, im } => format!(
            "{}{}{}i",
            format_value_color(re, trunc, assoc_mode),
            "+".bright_blue(),
            format_value_color(im, trunc, assoc_mode)
        )
        .to_string(),
        Value::PackedArray { shape, .. } => format!(
            "{}[{}]",
            "PackedArray".yellow(),
            shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x")
        )
        .to_string(),
        Value::String(s) => {
            let max = trunc.map(|t| t.max_string).unwrap_or(usize::MAX);
            if s.len() > max {
                format!("\"{}…\"", &s[..max]).green().to_string()
            } else {
                format!("\"{}\"", s).green().to_string()
            }
        }
        Value::Symbol(s) => s.normal().to_string(),
        Value::Boolean(b) => {
            if *b {
                "True".purple().to_string()
            } else {
                "False".purple().to_string()
            }
        }
        Value::List(items) => {
            if let Some(table) = try_format_table(items, trunc, assoc_mode) {
                return table;
            }
            let max = trunc.map(|t| t.max_list).unwrap_or(usize::MAX);
            let mut parts: Vec<String> = Vec::new();
            let take = items.len().min(max);
            for i in 0..take {
                parts.push(format_value_color(&items[i], trunc, assoc_mode));
            }
            if items.len() > take {
                parts.push(format!("{}", format!("… (+{} more)", items.len() - take).dimmed()));
            }
            let inline = format!("{{{}}}", parts.join(", "));
            let cols = term_width_cols();
            let pretty = items.len() >= 8 || inline.len() > cols.saturating_sub(4);
            if !pretty {
                inline
            } else {
                format_list_pretty(&parts, cols)
            }
        }
        Value::Assoc(m) => {
            let max = trunc.map(|t| t.max_assoc).unwrap_or(usize::MAX);
            let mut keys: Vec<_> = m.keys().collect();
            keys.sort();
            let take = keys.len().min(max);
            // First build inline; if too long or many keys, pretty-print multiline
            let mut inline_parts: Vec<String> = Vec::new();
            for k in &keys[..take] {
                inline_parts.push(format!(
                    "{} -> {}",
                    format!("\"{}\"", k).cyan(),
                    format_value_color(m.get(*k).unwrap(), trunc, assoc_mode)
                ));
            }
            if keys.len() > take {
                inline_parts
                    .push(format!("{}", format!("… (+{} more)", keys.len() - take).dimmed()));
            }
            let inline = format!("<|{}|>", inline_parts.join(", "));
            let cols = term_width_cols();
            let pretty = match assoc_mode {
                AssocMode::Inline => false,
                AssocMode::Pretty => true,
                AssocMode::Auto => keys.len() >= 4 || inline.len() > cols.saturating_sub(4),
            };
            if !pretty {
                return inline;
            }
            // Pretty multiline with aligned arrows
            let key_strs: Vec<String> = keys[..take].iter().map(|k| format!("\"{}\"", k)).collect();
            let keyw = key_strs.iter().map(|s| s.len()).max().unwrap_or(0).min(60);
            let mut out = String::new();
            out.push_str("<|\n");
            for (i, k) in keys[..take].iter().enumerate() {
                let kq = format!("\"{}\"", k).cyan();
                let pad = if keyw > key_strs[i].len() { keyw - key_strs[i].len() } else { 0 };
                let val = format_value_color(m.get(*k).unwrap(), trunc, assoc_mode);
                out.push_str(&format!("  {:<width$} -> {}", kq, val, width = keyw));
                if i + 1 < take {
                    out.push(',');
                }
                out.push('\n');
            }
            if keys.len() > take {
                out.push_str(&format!(
                    "  {}\n",
                    format!("… (+{} more)", keys.len() - take).dimmed()
                ));
            }
            out.push_str("|>");
            out
        }
        Value::Expr { head, args } => {
            let head_s = match &**head {
                Value::Symbol(s) => s.yellow().to_string(),
                other => format_value_color(other, trunc, assoc_mode),
            };
            let args_s: Vec<String> =
                args.iter().map(|a| format_value_color(a, trunc, assoc_mode)).collect();
            format!("{}[{}]", head_s, args_s.join(", "))
        }
        Value::Slot(None) => "#".to_string(),
        Value::Slot(Some(n)) => format!("#{}", n),
        Value::PureFunction { params: _, body } => {
            format!("{}[{}]", "Function".yellow(), format_value_color(body, trunc, assoc_mode))
        }
    }
}

fn try_format_table(
    items: &Vec<Value>,
    trunc: Option<TruncateCfg>,
    assoc_mode: AssocMode,
) -> Option<String> {
    // Render a list of associations sharing same small key set as a simple table
    if items.is_empty() {
        return None;
    }
    let mut keys: Vec<String> = Vec::new();
    for v in items {
        if let Value::Assoc(m) = v {
            if keys.is_empty() {
                keys = m.keys().cloned().collect();
                keys.sort();
            } else if m.keys().count() != keys.len() || !keys.iter().all(|k| m.contains_key(k)) {
                return None;
            }
        } else {
            return None;
        }
    }
    if keys.len() == 0 || keys.len() > 6 {
        return None;
    }
    let max_rows = trunc.map(|t| t.max_list).unwrap_or(usize::MAX);
    let rows_take = items.len().min(max_rows);
    // Compute column widths
    let mut colw: Vec<usize> = keys.iter().map(|k| k.len()).collect();
    let mut cells: Vec<Vec<String>> = Vec::new();
    for i in 0..rows_take {
        let m = if let Value::Assoc(m) = &items[i] { m } else { unreachable!() };
        let mut row: Vec<String> = Vec::new();
        for (j, k) in keys.iter().enumerate() {
            let s = format_value_color(
                m.get(k).unwrap(),
                Some(trunc.unwrap_or(TruncateCfg { max_list: 8, max_assoc: 8, max_string: 60 })),
                assoc_mode,
            );
            if s.len() > colw[j] {
                colw[j] = s.len();
            }
            row.push(s);
        }
        cells.push(row);
    }
    // Header
    let mut out = String::new();
    out.push_str(
        &keys
            .iter()
            .enumerate()
            .map(|(j, k)| format!("{:<width$}", k.cyan(), width = colw[j]))
            .collect::<Vec<_>>()
            .join("  "),
    );
    out.push('\n');
    // Rows
    for row in cells {
        let line = row
            .into_iter()
            .enumerate()
            .map(|(j, cell)| format!("{:<width$}", cell, width = colw[j]))
            .collect::<Vec<_>>()
            .join("  ");
        out.push_str(&line);
        out.push('\n');
    }
    if items.len() > rows_take {
        out.push_str(&format!(
            "{}\n",
            format!("… (+{} more rows)", items.len() - rows_take).dimmed()
        ));
    }
    Some(out)
}

fn format_list_pretty(parts: &Vec<String>, cols: usize) -> String {
    // Wrap items across lines to fit within terminal width
    let mut out = String::new();
    out.push('{');
    out.push('\n');
    let indent = "  ";
    let wrap_width = cols.saturating_sub(4).max(40);
    let mut line = String::new();
    for (i, p) in parts.iter().enumerate() {
        let item = if i + 1 < parts.len() { format!("{}{}", p, ", ") } else { p.clone() };
        if indent.len() + line.len() + item.len() > wrap_width {
            out.push_str(indent);
            out.push_str(&line);
            out.push('\n');
            line.clear();
        }
        line.push_str(&item);
    }
    if !line.is_empty() {
        out.push_str(indent);
        out.push_str(&line);
        out.push('\n');
    }
    out.push('}');
    out
}

fn recommended_truncation_for_width(cols: usize) -> TruncateCfg {
    // Simple heuristic based on terminal width
    let max_string = cols.saturating_sub(40).max(60);
    let max_list = if cols < 90 {
        12
    } else if cols < 120 {
        20
    } else {
        30
    };
    let max_assoc = if cols < 90 {
        12
    } else if cols < 120 {
        20
    } else {
        30
    };
    TruncateCfg { max_list, max_assoc, max_string }
}

fn render_parse_error(rl: &Editor<ReplHelper, DefaultHistory>, line: &str, msg: &str) {
    eprintln!("{} {}", "Error:".red().bold(), msg);
    // If clearly unbalanced, hint the missing closer
    if let Some((open, _pos)) = scan_unbalanced(line) {
        let closer = match open {
            '(' => ')',
            '[' => ']',
            '{' => '}',
            _ => '?',
        };
        eprintln!("{} {} -> missing '{}'", "Hint:".yellow().bold(), "Unclosed bracket", closer);
    }
    // Suggest similar builtins/vars based on trailing token
    let (_, word) = current_symbol_token(line, line.len());
    if word.len() >= 2 {
        if let Some(h) = rl.helper() {
            let mut cands: Vec<(&str, f64)> = Vec::new();
            for b in &h.builtins {
                cands.push((&b.name, strsim::jaro_winkler(&word, &b.name)));
            }
            for n in &h.env_names {
                cands.push((&n, strsim::jaro_winkler(&word, n)));
            }
            cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut shown = 0;
            for (name, score) in cands.into_iter() {
                if score > 0.85 {
                    if shown == 0 {
                        eprintln!("{} {}:", "Maybe:".yellow().bold(), word);
                    }
                    eprintln!("  {}", name);
                    shown += 1;
                    if shown >= 3 {
                        break;
                    }
                }
            }
        }
    }
}

fn print_history(rl: &Editor<ReplHelper, DefaultHistory>, arg: &str) {
    let hist = rl.history();
    let entries: Vec<_> = hist.iter().collect();
    if entries.is_empty() {
        println!("{}", "(no history)".dimmed());
        return;
    }
    if arg.starts_with('/') {
        let pat = arg.trim_start_matches('/').to_lowercase();
        for (i, e) in entries.iter().enumerate() {
            let s = e.to_string();
            if s.to_lowercase().contains(&pat) {
                println!("{:>4}: {}", i + 1, s);
            }
        }
        return;
    }
    let n: usize = if arg.is_empty() { 20 } else { arg.parse().unwrap_or(20) };
    let start = entries.len().saturating_sub(n);
    for (i, e) in entries.iter().enumerate().skip(start) {
        println!("{:>4}: {}", i + 1, e);
    }
}

fn handle_using_cmd(ev: &mut Evaluator, rest: &str) -> anyhow::Result<()> {
    let mut name = String::new();
    let mut import_all = false;
    let mut imports: Vec<String> = Vec::new();
    let mut excepts: Vec<String> = Vec::new();
    let mut parts = rest.split_whitespace();
    if let Some(n) = parts.next() {
        name = n.trim_matches('"').to_string();
    }
    for tok in parts {
        if tok.eq_ignore_ascii_case("--all") {
            import_all = true;
            continue;
        }
        if tok.eq_ignore_ascii_case("--import") {
            /* handled via next token */
            continue;
        }
        if tok.eq_ignore_ascii_case("--except") {
            /* handled via next token */
            continue;
        }
        // comma lists for previous flag
        if import_all { /* already set; ignore */
        } else if imports.is_empty() && tok.starts_with("--import") {
            /* --import a,b */
            let v = tok.trim_start_matches("--import").trim_start_matches('=');
            if !v.is_empty() {
                imports = v
                    .split(',')
                    .map(|s| s.trim().trim_matches('"').to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
        } else if excepts.is_empty() && tok.starts_with("--except") {
            let v = tok.trim_start_matches("--except").trim_start_matches('=');
            if !v.is_empty() {
                excepts = v
                    .split(',')
                    .map(|s| s.trim().trim_matches('"').to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
        } else {
            // also allow bare a,b after --import or --except in previous token; simple heuristic
            if imports.is_empty() && rest.contains("--import ") {
                if let Some(seg) = rest.split("--import ").nth(1) {
                    let seg = seg.split_whitespace().next().unwrap_or("");
                    if !seg.is_empty() {
                        imports = seg
                            .split(',')
                            .map(|s| s.trim().trim_matches('"').to_string())
                            .filter(|s| !s.is_empty())
                            .collect();
                    }
                }
            }
            if excepts.is_empty() && rest.contains("--except ") {
                if let Some(seg) = rest.split("--except ").nth(1) {
                    let seg = seg.split_whitespace().next().unwrap_or("");
                    if !seg.is_empty() {
                        excepts = seg
                            .split(',')
                            .map(|s| s.trim().trim_matches('"').to_string())
                            .filter(|s| !s.is_empty())
                            .collect();
                    }
                }
            }
        }
    }
    if name.is_empty() {
        anyhow::bail!("usage: :using name [--all|--import a,b] [--except x,y]");
    }
    let call = if import_all {
        format!("Using[\"{}\", <|Import->All|>]", name)
    } else if !imports.is_empty() && excepts.is_empty() {
        format!(
            "Using[\"{}\", <|Import->{{{}}}|>]",
            name,
            imports.iter().map(|s| format!("\"{}\"", s)).collect::<Vec<_>>().join(", ")
        )
    } else if !imports.is_empty() && !excepts.is_empty() {
        format!(
            "Using[\"{}\", <|Import->{{{}}}, Except->{{{}}}|>]",
            name,
            imports.iter().map(|s| format!("\"{}\"", s)).collect::<Vec<_>>().join(", "),
            excepts.iter().map(|s| format!("\"{}\"", s)).collect::<Vec<_>>().join(", ")
        )
    } else {
        format!("Using[\"{}\"]", name)
    };
    let mut p = Parser::from_source(&call);
    match p.parse_all() {
        Ok(vs) => {
            for v in vs {
                let _ = ev.eval(v);
            }
            Ok(())
        }
        Err(e) => Err(anyhow::anyhow!(e.to_string())),
    }
}

fn append_transcript(path: &Option<std::path::PathBuf>, text: &str) {
    if let Some(p) = path {
        let _ = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
            .and_then(|mut f| std::io::Write::write_all(&mut f, text.as_bytes()));
    }
}

fn strip_ansi(s: &str) -> String {
    // very simple strip for color codes: remove \x1b[ ... m sequences
    let mut out = String::new();
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == 0x1b && i + 1 < bytes.len() && bytes[i + 1] == b'[' {
            i += 2;
            while i < bytes.len() && (bytes[i] as char) != 'm' {
                i += 1;
            }
            if i < bytes.len() {
                i += 1;
            }
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

fn render_doc_card(ev: &mut Evaluator, sym: &str) {
    // Lookup builtin details via Documentation/DescribeBuiltins
    let resp = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("DescribeBuiltins".into())),
        args: vec![],
    });
    let mut attrs: Vec<String> = Vec::new();
    if let Value::List(items) = resp {
        for it in items {
            if let Value::Assoc(m) = it {
                if let Some(Value::String(name)) = m.get("name") {
                    if name == sym {
                        if let Some(Value::List(vs)) = m.get("attributes") {
                            attrs = vs
                                .iter()
                                .filter_map(|v| {
                                    if let Value::String(s) = v {
                                        Some(s.clone())
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                        }
                        break;
                    }
                }
            }
        }
    }
    // Prefer Documentation[] summary/params
    let doc = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("Documentation".into())),
        args: vec![Value::String(sym.to_string())],
    });
    let mut summary = builtin_help(sym).to_string();
    let mut params: Vec<String> = Vec::new();
    if let Value::Assoc(m) = doc {
        if let Some(Value::String(s)) = m.get("summary") {
            if !s.is_empty() {
                summary = s.clone();
            }
        }
        if let Some(Value::List(vs)) = m.get("params") {
            params = vs
                .iter()
                .filter_map(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
                .collect();
        }
    }
    let header = format!("{} {}", sym.bold(), attrs.join(", "));
    println!(
        "{}\n  {} {}{}",
        header,
        "Summary:".bold(),
        summary,
        if params.is_empty() {
            String::new()
        } else {
            format!("\n  {} {}[{}]", "Usage:".bold(), sym, params.join(", "))
        }
    );
}

struct DocIndex {
    // name -> (summary, attributes, params, examples)
    map: std::collections::HashMap<String, (String, Vec<String>, Vec<String>, Vec<String>)>,
}

fn build_doc_index(ev: &mut Evaluator) -> std::sync::Arc<DocIndex> {
    use std::collections::HashMap;
    let mut map: HashMap<String, (String, Vec<String>, Vec<String>, Vec<String>)> = HashMap::new();
    let resp = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("DescribeBuiltins".into())),
        args: vec![],
    });
    if let Value::List(items) = resp {
        for it in items {
            if let Value::Assoc(m) = it {
                if let Some(Value::String(name)) = m.get("name") {
                    // Try Documentation first
                    let mut summary = String::new();
                    let mut params: Vec<String> = Vec::new();
                    let doc = ev.eval(Value::Expr {
                        head: Box::new(Value::Symbol("Documentation".into())),
                        args: vec![Value::String(name.clone())],
                    });
                    let mut examples: Vec<String> = Vec::new();
                    if let Value::Assoc(dm) = doc {
                        if let Some(Value::String(s)) = dm.get("summary") {
                            summary = s.clone();
                        }
                        if let Some(Value::List(vs)) = dm.get("params") {
                            params = vs
                                .iter()
                                .filter_map(|v| {
                                    if let Value::String(s) = v {
                                        Some(s.clone())
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                        }
                        if let Some(Value::List(vs)) = dm.get("examples") {
                            examples = vs
                                .iter()
                                .filter_map(|v| {
                                    if let Value::String(s) = v {
                                        Some(s.clone())
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                        }
                    }
                    if summary.is_empty() {
                        summary = builtin_help(name).to_string();
                    }
                    let attrs: Vec<String> = match m.get("attributes") {
                        Some(Value::List(vs)) => vs
                            .iter()
                            .filter_map(|v| {
                                if let Value::String(s) = v {
                                    Some(s.clone())
                                } else {
                                    None
                                }
                            })
                            .collect(),
                        _ => Vec::new(),
                    };
                    map.insert(name.clone(), (summary, attrs, params, examples));
                }
            }
        }
    }
    std::sync::Arc::new(DocIndex { map })
}

fn build_doc_card_str_from_index(index: &std::sync::Arc<DocIndex>, sym: &str) -> String {
    if let Some((summary, attrs, params, examples)) = index.map.get(sym) {
        let header = if attrs.is_empty() {
            sym.bold().to_string()
        } else {
            format!("{} {}", sym.bold(), attrs.join(", "))
        };
        let mut s = format!("{}\n  {} {}", header, "Summary:".bold(), summary);
        if !params.is_empty() {
            s.push_str(&format!("\n  {} {}[{}]", "Usage:".bold(), sym, params.join(", ")));
        }
        if !examples.is_empty() {
            s.push_str(&format!("\n  {}", "Examples:".bold()));
            // Show up to 3 examples for brevity
            for ex in examples.iter().take(3) {
                s.push_str(&format!("\n    {}", ex));
            }
            if examples.len() > 3 {
                s.push_str(&format!("\n    {}", format!("(+{} more)", examples.len() - 3).dimmed()));
            }
        }
        s
    } else {
        format!("{}\n  {} {}", sym.bold(), "Summary:".bold(), "No documentation yet.")
    }
}

fn render_env(ev: &Evaluator) {
    let mut names = ev.env_keys();
    names.sort();
    if names.is_empty() {
        println!("{}", "(no variables)".dimmed());
        return;
    }
    let items = names.join(", ");
    println!("{} {}", "Env:".bold(), items);
}

fn render_funcs(ev: &mut Evaluator, filter: Option<&str>) {
    let resp = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("DescribeBuiltins".into())),
        args: vec![],
    });
    let mut rows: Vec<(String, Vec<String>)> = Vec::new();
    if let Value::List(items) = resp {
        for it in items {
            if let Value::Assoc(m) = it {
                if let Some(Value::String(name)) = m.get("name") {
                    if let Some(f) = filter {
                        if !name.to_lowercase().contains(&f.to_lowercase()) {
                            continue;
                        }
                    }
                    let attrs: Vec<String> = match m.get("attributes") {
                        Some(Value::List(vs)) => vs
                            .iter()
                            .filter_map(|v| {
                                if let Value::String(s) = v {
                                    Some(s.clone())
                                } else {
                                    None
                                }
                            })
                            .collect(),
                        _ => Vec::new(),
                    };
                    rows.push((name.clone(), attrs));
                }
            }
        }
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, attrs) in rows.into_iter() {
        let attr =
            if attrs.is_empty() { String::new() } else { format!(" [{}]", attrs.join(", ")) };
        println!("{}{}", name, attr.dimmed());
    }
}

fn render_symbol_info(ev: &mut Evaluator, name: &str) {
    // Attributes via DescribeBuiltins
    let resp = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("DescribeBuiltins".into())),
        args: vec![],
    });
    let mut attrs: Vec<String> = Vec::new();
    if let Value::List(items) = resp {
        for it in items {
            if let Value::Assoc(m) = it {
                if let Some(Value::String(n)) = m.get("name") {
                    if n == name {
                        if let Some(Value::List(vs)) = m.get("attributes") {
                            attrs = vs
                                .iter()
                                .filter_map(|v| {
                                    if let Value::String(s) = v {
                                        Some(s.clone())
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                        }
                        break;
                    }
                }
            }
        }
    }
    // Definition counts
    let symv = Value::Symbol(name.to_string());
    let own = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("GetOwnValues".into())),
        args: vec![symv.clone()],
    });
    let down = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("GetDownValues".into())),
        args: vec![symv.clone()],
    });
    let up = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("GetUpValues".into())),
        args: vec![symv.clone()],
    });
    let sub = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("GetSubValues".into())),
        args: vec![symv.clone()],
    });
    let count = |v: &Value| -> usize {
        if let Value::List(items) = v {
            items.len()
        } else {
            0
        }
    };
    println!("{} {}", "Symbol:".bold(), name);
    if !attrs.is_empty() {
        println!("  {} {}", "Attributes:".bold(), attrs.join(", "));
    }
    println!("  {} {}", "OwnValues:".bold(), count(&own));
    println!("  {} {}", "DownValues:".bold(), count(&down));
    println!("  {} {}", "UpValues:".bold(), count(&up));
    println!("  {} {}", "SubValues:".bold(), count(&sub));
}

fn render_defs(ev: &mut Evaluator, name: &str, trunc: TruncateCfg) {
    let symv = Value::Symbol(name.to_string());
    let kinds = [
        ("OwnValues", "GetOwnValues"),
        ("DownValues", "GetDownValues"),
        ("UpValues", "GetUpValues"),
        ("SubValues", "GetSubValues"),
    ];
    for (label, getter) in kinds {
        let vals = ev.eval(Value::Expr {
            head: Box::new(Value::Symbol(getter.into())),
            args: vec![symv.clone()],
        });
        match vals {
            Value::List(items) if !items.is_empty() => {
                println!("{} {}:", label.bold(), name);
                for r in items {
                    match r {
                        Value::Expr { head, args }
                            if matches!(*head, Value::Symbol(ref s) if s=="Rule")
                                && args.len() == 2 =>
                        {
                            let lhs = &args[0];
                            let rhs = &args[1];
                            println!(
                                "  {} {} {}",
                                format_value_color(lhs, Some(trunc), AssocMode::Auto),
                                "->".bold(),
                                format_value_color(rhs, Some(trunc), AssocMode::Auto)
                            );
                        }
                        other => println!(
                            "  {}",
                            format_value_color(&other, Some(trunc), AssocMode::Auto)
                        ),
                    }
                }
            }
            _ => {}
        }
    }
}

fn explain_steps(ev: &mut Evaluator, expr: Value) -> Vec<Value> {
    let res =
        ev.eval(Value::Expr { head: Box::new(Value::Symbol("Explain".into())), args: vec![expr] });
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
            let action = match sm.get("action") {
                Some(Value::String(x)) => x.as_str(),
                _ => "",
            };
            let head = match sm.get("head") {
                Some(Value::Symbol(x)) => x.as_str(),
                _ => "",
            };
            let extra = if let Some(Value::Assoc(data)) = sm.get("data") {
                if let Some(v) = data.get("count") {
                    format!(" count={}", format_value_color(v, None, AssocMode::Auto))
                } else if let Some(v) = data.get("finalOrder") {
                    format!(
                        " finalOrder={}",
                        format_value_color(
                            v,
                            Some(TruncateCfg { max_list: 6, max_assoc: 6, max_string: 80 }),
                            AssocMode::Auto
                        )
                    )
                } else {
                    String::new()
                }
            } else {
                String::new()
            };
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
            if let Some(Value::String(a)) = sm.get("action") {
                *actions.entry(a.clone()).or_insert(0) += 1;
            }
            if let Some(Value::Symbol(h)) = sm.get("head") {
                *heads.entry(h.clone()).or_insert(0) += 1;
            }
        }
    }
    let mut top_a: Vec<_> = actions.into_iter().collect();
    top_a.sort_by(|a, b| b.1.cmp(&a.1));
    let mut top_h: Vec<_> = heads.into_iter().collect();
    top_h.sort_by(|a, b| b.1.cmp(&a.1));
    let ta = top_a
        .into_iter()
        .take(3)
        .map(|(k, v)| format!("{}:{}", k, v))
        .collect::<Vec<_>>()
        .join(", ");
    let th = top_h
        .into_iter()
        .take(3)
        .map(|(k, v)| format!("{}:{}", k, v))
        .collect::<Vec<_>>()
        .join(", ");
    println!("{} {} | {} {}", "Profile actions:".bold(), ta, "heads:".bold(), th);
}

#[allow(dead_code)]
fn explain_steps_to_string(steps: &Vec<Value>) -> String {
    let mut out = String::new();
    for s in steps {
        if let Value::Assoc(sm) = s {
            let action = match sm.get("action") {
                Some(Value::String(x)) => x.as_str(),
                _ => "",
            };
            let head = match sm.get("head") {
                Some(Value::Symbol(x)) => x.as_str(),
                _ => "",
            };
            let extra = if let Some(Value::Assoc(data)) = sm.get("data") {
                if let Some(v) = data.get("count") {
                    format!(" count={}", format_value_color(v, None, AssocMode::Auto))
                } else if let Some(v) = data.get("finalOrder") {
                    format!(
                        " finalOrder={}",
                        format_value_color(
                            v,
                            Some(TruncateCfg { max_list: 6, max_assoc: 6, max_string: 80 }),
                            AssocMode::Auto
                        )
                    )
                } else {
                    String::new()
                }
            } else {
                String::new()
            };
            out.push_str(&format!(
                "  {} {}{}\n",
                action.to_string().blue(),
                head.yellow(),
                extra.dimmed()
            ));
        }
    }
    out
}

#[derive(Clone, Copy)]
enum AssocMode {
    Auto,
    Inline,
    Pretty,
}

#[derive(Clone, Copy)]
enum PrintMode {
    Expr,
    Json,
}

struct ReplConfig {
    pager_on: bool,
    assoc_mode: AssocMode,
    print_mode: PrintMode,
}

fn spawn_pager(text: &str) -> std::io::Result<()> {
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

fn save_env_json(ev: &mut Evaluator, path: &str) -> anyhow::Result<()> {
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

fn load_env_json(ev: &mut Evaluator, path: &str) -> anyhow::Result<()> {
    let s = fs::read_to_string(path)?;
    let v: Value = serde_json::from_str(&s)?;
    if let Value::Assoc(m) = v {
        for (k, val) in m.into_iter() {
            // Set[k, val]
            let _ = ev.eval(Value::Expr {
                head: Box::new(Value::Symbol("Set".into())),
                args: vec![Value::Symbol(k), val],
            });
        }
    }
    Ok(())
}

fn collect_pkg_exports(ev: &mut Evaluator) -> std::collections::HashMap<String, Vec<String>> {
    use std::collections::HashMap;
    let mut out: HashMap<String, Vec<String>> = HashMap::new();
    let loaded = ev
        .eval(Value::Expr { head: Box::new(Value::Symbol("LoadedPackages".into())), args: vec![] });
    if let Value::Assoc(m) = loaded {
        for name in m.keys() {
            let q = Value::Expr {
                head: Box::new(Value::Symbol("PackageExports".into())),
                args: vec![Value::String(name.clone())],
            };
            let ex = ev.eval(q);
            if let Value::List(vs) = ex {
                let syms: Vec<String> = vs
                    .into_iter()
                    .filter_map(|v| if let Value::String(s) = v { Some(s) } else { None })
                    .collect();
                out.insert(name.clone(), syms);
            }
        }
    }
    out
}

fn edit_and_eval(ev: &mut Evaluator, path: &str) -> anyhow::Result<()> {
    use std::process::Command;
    let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vi".to_string());
    let file_path = if path.is_empty() {
        let mut p = std::env::temp_dir();
        p.push("lyra_edit.lyra");
        if !p.exists() {
            let _ = std::fs::File::create(&p);
        }
        p
    } else {
        std::path::PathBuf::from(path)
    };
    // Launch editor and wait
    let status = Command::new(editor).arg(&file_path).status();
    match status {
        Ok(s) if s.success() => {}
        _ => {}
    }
    // Read file and evaluate
    let src = std::fs::read_to_string(&file_path)?;
    if src.trim().is_empty() {
        println!("{}", "(empty)".dimmed());
        return Ok(());
    }
    let mut p = Parser::from_source(&src);
    match p.parse_all() {
        Ok(values) => {
            for v in values {
                let out = ev.eval(v);
                println!(
                    "{}",
                    format_value_color(
                        &out,
                        Some(recommended_truncation_for_width(term_width_cols())),
                        AssocMode::Auto
                    )
                );
            }
            Ok(())
        }
        Err(e) => {
            eprintln!("{} {}", "Error:".red().bold(), e);
            Ok(())
        }
    }
}
