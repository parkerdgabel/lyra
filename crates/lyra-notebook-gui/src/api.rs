use anyhow::Result;
use lazy_static::lazy_static;
use lyra_notebook_core as nbcore;
use nbcore::ops as nbops;
use lyra_notebook_kernel as kernel;
use parking_lot::Mutex;
use std::collections::HashMap;
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use lyra_core::value::Value;
use std::time::Instant;
use std::path::PathBuf;
use std::fs;
// Bring base64 Engine trait into scope for `.encode`/`.decode` methods
use base64::Engine as _;

lazy_static! {
    pub(crate) static ref SESSION_REG: Mutex<HashMap<Uuid, kernel::Session>> = Mutex::new(HashMap::new());
}

pub fn open_notebook(path: &str) -> Result<kernel::OpenResponse> {
    let nb = nbcore::read_notebook(path)?;
    // In GUI, prefer rich display by default
    lyra_stdlib::display::set_prefer_display(true);
    let mut settings = kernel::SessionSettings::default();
    // Compute cache dir next to notebook file
    let mut cache_dir = PathBuf::from(path);
    if cache_dir.is_file() { cache_dir.pop(); }
    if cache_dir.as_os_str().is_empty() { cache_dir = PathBuf::from("."); }
    cache_dir.push(".lyra-cache");
    let _ = fs::create_dir_all(&cache_dir);
    settings.cache_dir = Some(cache_dir.to_string_lossy().to_string());
    settings.enable_cache = true;
    let mut sess = kernel::Session::new(nb, settings);
    let sid = sess.id();
    let nb_copy = sess.notebook().clone();
    SESSION_REG.lock().insert(sid, sess);
    Ok(kernel::OpenResponse { session_id: sid, notebook: nb_copy })
}

pub fn new_notebook(title: Option<&str>) -> Result<kernel::OpenResponse> {
    let mut opts = nbops::NotebookCreateOpts::default();
    if let Some(t) = title { opts.title = Some(t.to_string()); }
    let mut nb = nbops::notebook_create(opts);
    // Add an initial empty Lyra code cell for convenience
    let cell = nbops::cell_create(nbcore::schema::CellType::Code, "", nbops::CellCreateOpts::default());
    nb = nbops::cell_insert(&nb, cell, nbops::InsertPos::Index(0));
    // In GUI, prefer rich display by default
    lyra_stdlib::display::set_prefer_display(true);
    let settings = kernel::SessionSettings::default();
    let mut sess = kernel::Session::new(nb, settings);
    let sid = sess.id();
    let nb_copy = sess.notebook().clone();
    SESSION_REG.lock().insert(sid, sess);
    Ok(kernel::OpenResponse { session_id: sid, notebook: nb_copy })
}

pub fn save_notebook(session_id: Uuid, path: &str, include_outputs: bool, pretty: bool) -> Result<bool> {
    let map = SESSION_REG.lock();
    if let Some(sess) = map.get(&session_id) {
        let opts = nbcore::io::WriteOpts { include_outputs, pretty };
        nbcore::write_notebook(path, sess.notebook(), opts)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

pub fn execute_cell(session_id: Uuid, cell_id: Uuid) -> Result<kernel::ExecResult> {
    let mut map = SESSION_REG.lock();
    if let Some(sess) = map.get_mut(&session_id) {
        Ok(sess.execute_cell(cell_id, kernel::ExecutionOpts::default()))
    } else {
        anyhow::bail!("Unknown session")
    }
}

pub fn execute_cell_nocache(session_id: Uuid, cell_id: Uuid) -> Result<kernel::ExecResult> {
    let mut map = SESSION_REG.lock();
    if let Some(sess) = map.get_mut(&session_id) {
        let mut opts = kernel::ExecutionOpts::default();
        opts.ignore_cache = true;
        Ok(sess.execute_cell(cell_id, opts))
    } else {
        anyhow::bail!("Unknown session")
    }
}

pub fn execute_all(session_id: Uuid, ids: Option<Vec<Uuid>>, method: Option<String>, ignore_cache: bool) -> Result<Vec<kernel::ExecResult>> {
    let mut map = SESSION_REG.lock();
    if let Some(sess) = map.get_mut(&session_id) {
        let use_ids: Vec<Uuid> = ids.unwrap_or_else(|| sess.notebook().cells.iter().map(|c| c.id).collect());
        let mut opts = kernel::ExecutionOpts::default();
        opts.ignore_cache = ignore_cache;
        Ok(sess.execute_many(&use_ids, method.as_deref().unwrap_or("Linear"), opts))
    } else {
        anyhow::bail!("Unknown session")
    }
}

pub fn interrupt(session_id: Uuid) -> bool {
    let mut map = SESSION_REG.lock();
    if let Some(sess) = map.get_mut(&session_id) {
        sess.interrupt()
    } else { false }
}

pub fn update_session_notebook(session_id: Uuid, nb: nbcore::schema::Notebook) -> bool {
    let mut map = SESSION_REG.lock();
    if let Some(sess) = map.get_mut(&session_id) {
        *sess.notebook_mut() = nb;
        true
    } else { false }
}

pub fn preview_value(session_id: Uuid, value_json: String, limit: u32) -> Result<String> {
    let mut map = SESSION_REG.lock();
    if let Some(sess) = map.get_mut(&session_id) {
        // Accept either raw typed JSON or versioned envelope { __meta: {x-lyra-version}, value: <typed> }
        let raw: serde_json::Value = serde_json::from_str(&value_json)?;
        let v: lyra_core::value::Value = if let Some(val) = raw.get("value") {
            serde_json::from_value(val.clone())?
        } else {
            serde_json::from_value(raw)?
        };
        let out = sess.preview_rows(v, limit as usize);
        // Return typed Value JSON (not wrapped) for preview rows to keep the client fast
        Ok(serde_json::to_string(&out)?)
    } else {
        anyhow::bail!("Unknown session")
    }
}

pub fn execute_cell_events(session_id: Uuid, cell_id: Uuid) -> Result<Vec<kernel::ExecEvent>> {
    let mut map = SESSION_REG.lock();
    if let Some(sess) = map.get_mut(&session_id) {
        Ok(sess.execute_cell_events(cell_id, kernel::ExecutionOpts::default()))
    } else {
        anyhow::bail!("Unknown session")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecTextResult {
    pub duration_ms: u128,
    pub outputs: Vec<kernel::DisplayItem>,
    pub error: Option<String>,
}

pub fn execute_text(session_id: Uuid, text: String) -> Result<ExecTextResult> {
    let mut map = SESSION_REG.lock();
    if let Some(sess) = map.get_mut(&session_id) {
        let (dur, outs, err) = sess.eval_text_to_display(&text);
        Ok(ExecTextResult { duration_ms: dur, outputs: outs, error: err })
    } else {
        anyhow::bail!("Unknown session")
    }
}

pub fn ping(session_id: Uuid) -> bool {
    let map = SESSION_REG.lock();
    map.contains_key(&session_id)
}

// Cache UX
pub fn set_cache_enabled(session_id: Uuid, on: bool) -> bool {
    let mut map = SESSION_REG.lock();
    if let Some(sess) = map.get_mut(&session_id) { sess.set_cache_enabled(on); true } else { false }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheGcResult { pub freed_bytes: u64, pub total_after: u64 }

pub fn cache_clear(session_id: Uuid) -> bool {
    let mut map = SESSION_REG.lock();
    if let Some(sess) = map.get_mut(&session_id) { sess.clear_cache_memory(); sess.clear_cache_disk(); true } else { false }
}

pub fn cache_gc(session_id: Uuid, max_bytes: u64) -> CacheGcResult {
    let map = SESSION_REG.lock();
    if let Some(sess) = map.get(&session_id) { let (freed, after) = sess.cache_gc(max_bytes); CacheGcResult { freed_bytes: freed, total_after: after } } else { CacheGcResult { freed_bytes: 0, total_after: 0 } }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheInfo { pub enabled: bool, pub disk_bytes: u64, pub files: u64 }

pub fn cache_info(session_id: Uuid) -> CacheInfo {
    let map = SESSION_REG.lock();
    if let Some(sess) = map.get(&session_id) {
        let (bytes, files) = sess.cache_disk_size_bytes();
        CacheInfo { enabled: true && sess.notebook().version.len() >= 0 && sess as *const _ != std::ptr::null(), disk_bytes: bytes, files }
    } else { CacheInfo { enabled: false, disk_bytes: 0, files: 0 } }
}

// Cache salt
pub fn cache_set_salt(session_id: Uuid, salt: Option<String>) -> bool {
    let mut map = SESSION_REG.lock();
    if let Some(sess) = map.get_mut(&session_id) { sess.set_cache_salt(salt); true } else { false }
}

pub fn cache_get_salt(session_id: Uuid) -> Option<String> {
    let map = SESSION_REG.lock();
    map.get(&session_id).and_then(|s| s.cache_salt())
}

pub fn add_cell(session_id: Uuid, typ: &str) -> Result<nbcore::schema::Notebook> {
    let mut map = SESSION_REG.lock();
    if let Some(sess) = map.get_mut(&session_id) {
        let mut nb = sess.notebook().clone();
        let cell_type = match typ {
            "Code" => nbcore::schema::CellType::Code,
            "Markdown" => nbcore::schema::CellType::Markdown,
            "Text" => nbcore::schema::CellType::Text,
            _ => nbcore::schema::CellType::Code,
        };
        let input = match cell_type { nbcore::schema::CellType::Code => "", _ => "" };
        let cell = nbops::cell_create(cell_type, input, nbops::CellCreateOpts::default());
        nb = nbops::cell_insert(&nb, cell, nbops::InsertPos::Index(nb.cells.len()));
        *sess.notebook_mut() = nb.clone();
        Ok(nb)
    } else { anyhow::bail!("Unknown session") }
}

pub fn delete_cell(session_id: Uuid, cell_id: Uuid) -> Result<nbcore::schema::Notebook> {
    let mut map = SESSION_REG.lock();
    if let Some(sess) = map.get_mut(&session_id) {
        let nb = sess.notebook().clone();
        let nb2 = nbops::cell_delete(&nb, cell_id);
        *sess.notebook_mut() = nb2.clone();
        Ok(nb2)
    } else { anyhow::bail!("Unknown session") }
}

// --- Editor helpers ---

lazy_static! {
    static ref BUILTINS: Mutex<Option<Vec<String>>> = Mutex::new(None);
}

pub fn editor_builtins() -> Vec<String> {
    if let Some(v) = BUILTINS.lock().clone() { return v; }
    let mut ev = lyra_runtime::Evaluator::new();
    lyra_stdlib::register_all(&mut ev);
    let mut names: Vec<String> = Vec::new();
    let resp = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("DescribeBuiltins".into())),
        args: vec![],
    });
    if let Value::List(items) = resp {
        for it in items {
            if let Value::Assoc(m) = it {
                if let Some(Value::String(n)) = m.get("name") {
                    names.push(n.clone());
                }
            }
        }
    }
    names.sort();
    names.dedup();
    *BUILTINS.lock() = Some(names.clone());
    names
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditorDoc {
    pub name: String,
    pub summary: String,
    pub params: Vec<String>,
    pub examples: Vec<String>,
}

lazy_static! {
    static ref DOCS: Mutex<Option<std::collections::HashMap<String, EditorDoc>>> = Mutex::new(None);
}

fn ensure_docs() -> std::collections::HashMap<String, EditorDoc> {
    if let Some(map) = DOCS.lock().clone() { return map; }
    let mut ev = lyra_runtime::Evaluator::new();
    lyra_stdlib::register_all(&mut ev);
    let resp = ev.eval(Value::Expr { head: Box::new(Value::Symbol("Documentation".into())), args: vec![] });
    let mut map: std::collections::HashMap<String, EditorDoc> = std::collections::HashMap::new();
    if let Value::List(items) = resp {
        for it in items {
            if let Value::Assoc(m) = it {
                let name = match m.get("name") { Some(Value::String(s)) => s.clone(), _ => String::new() };
                if name.is_empty() || name.starts_with("__") { continue; }
                let summary = match m.get("summary") { Some(Value::String(s)) => s.clone(), _ => String::new() };
                let params = match m.get("params") { Some(Value::List(ps)) => ps.iter().filter_map(|v| if let Value::String(s)=v { Some(s.clone()) } else { None }).collect(), _ => Vec::new() };
                let examples = match m.get("examples") { Some(Value::List(xs)) => xs.iter().filter_map(|v| if let Value::String(s)=v { Some(s.clone()) } else { None }).collect(), _ => Vec::new() };
                map.insert(name.clone(), EditorDoc { name, summary, params, examples });
            }
        }
    }
    *DOCS.lock() = Some(map.clone());
    map
}

pub fn editor_doc(name: String) -> Option<EditorDoc> {
    let map = ensure_docs();
    map.get(&name).cloned()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditorDiagnostic {
    pub message: String,
    pub start_line: u32,
    pub start_col: u32,
    pub end_line: u32,
    pub end_col: u32,
    pub severity: String, // "Error" | "Warning" | "Info"
}

// --- LSP basics: defs/refs/rename ---
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditorLocation {
    pub cell_id: String,
    pub start_line: u32,
    pub start_col: u32,
    pub end_line: u32,
    pub end_col: u32,
}

fn offsets_to_line_col(text: &str, start: usize, end: usize) -> (u32, u32, u32, u32) {
    let mut line: u32 = 0;
    let mut col: u32 = 0;
    let mut s_line = 0u32; let mut s_col = 0u32;
    let mut e_line = 0u32; let mut e_col = 0u32;
    for (i, ch) in text.char_indices() {
        if i == start { s_line = line; s_col = col; }
        if i == end   { e_line = line; e_col = col; }
        if ch == '\n' { line += 1; col = 0; } else { col += 1; }
        if i > end { break; }
    }
    // If end lies at text length
    if end >= text.len() { e_line = line; e_col = col; }
    (s_line, s_col, e_line, e_col)
}

fn find_word_occurrences(text: &str, name: &str) -> Vec<(usize, usize)> {
    if name.is_empty() { return vec![]; }
    let re = regex::Regex::new(&format!(r"(?m)\b{}\b", regex::escape(name))).unwrap();
    let code = code_mask(text);
    re.find_iter(text)
        .filter(|m| code.get(m.start()).copied().unwrap_or(true))
        .map(|m| (m.start(), m.end()))
        .collect()
}

// Find definition spans for a given name: matches `Name :=` and `Name[...] :=`
fn find_defs(text: &str, name: &str) -> Vec<(usize, usize)> {
    if name.is_empty() { return vec![]; }
    let code = code_mask(text);
    let mut out = vec![];
    // Simple variable definition: Name :=
    let re_var = regex::Regex::new(&format!(r"(?m)\b{}\b\s*: =?", regex::escape(name)).replace(" ", "")).unwrap();
    for m in re_var.find(text) {
        if code.get(m.start()).copied().unwrap_or(true) { out.push((m.start(), m.start()+name.len())); }
    }
    // Function head definition: Name[ ... ] :=
    let re_head = regex::Regex::new(&format!(r"(?m)\b{}\b\s*\[", regex::escape(name))).unwrap();
    for m in re_head.find_iter(text) {
        if !code.get(m.start()).copied().unwrap_or(true) { continue; }
        // Find matching closing ']' then check for ':='
        let mut i = m.end(); let bytes = text.as_bytes(); let n = bytes.len(); let mut depth = 1i32;
        while i < n {
            if bytes[i] == b'"' { // skip strings
                i += 1; while i < n { if bytes[i] == b'\\' { i += 2; continue; } if bytes[i] == b'"' { i+=1; break; } i+=1; }
                continue;
            }
            if bytes[i] == b'(' && i+1<n && bytes[i+1]==b'*' { // skip comments
                i += 2; while i+1 < n { if bytes[i]==b'*' && bytes[i+1]==b')' { i+=2; break; } i+=1; }
                continue;
            }
            if bytes[i] == b'[' { depth += 1; }
            else if bytes[i] == b']' { depth -= 1; if depth == 0 { i += 1; break; } }
            i += 1;
        }
        // After ']' (or if unmatched), scan for ':='
        let mut j = i; while j+1 < n && text[j..].starts_with(':') == false { if !text[j..].starts_with(' ') { break; } j+=1; }
        if j+1 < n && &text[j..j+2] == ":=" { out.push((m.start(), m.start()+name.len())); }
    }
    out
}

pub fn editor_defs(session_id: Uuid, name: &str) -> Vec<EditorLocation> {
    let map = SESSION_REG.lock();
    let mut out = vec![];
    if let Some(sess) = map.get(&session_id) {
        for c in &sess.notebook().cells {
            if c.r#type != nbcore::schema::CellType::Code || c.language.to_lowercase() != "lyra" { continue; }
            for (s, e) in find_defs(&c.input, name) {
                let (sl, sc, el, ec) = offsets_to_line_col(&c.input, s, e);
                out.push(EditorLocation { cell_id: c.id.to_string(), start_line: sl, start_col: sc, end_line: el, end_col: ec });
            }
        }
    }
    out
}

pub fn editor_refs(session_id: Uuid, name: &str) -> Vec<EditorLocation> {
    let map = SESSION_REG.lock();
    let mut out = vec![];
    if let Some(sess) = map.get(&session_id) {
        for c in &sess.notebook().cells {
            if c.r#type != nbcore::schema::CellType::Code || c.language.to_lowercase() != "lyra" { continue; }
            let defs = find_defs(&c.input, name);
            let def_spans: std::collections::HashSet<usize> = defs.iter().map(|(s, _)| *s).collect();
            for (s, e) in find_word_occurrences(&c.input, name) {
                // Skip occurrences that start at a def span start (avoid double counting definition token)
                if def_spans.contains(&s) { continue; }
                let (sl, sc, el, ec) = offsets_to_line_col(&c.input, s, e);
                out.push(EditorLocation { cell_id: c.id.to_string(), start_line: sl, start_col: sc, end_line: el, end_col: ec });
            }
        }
    }
    out
}

pub fn editor_rename(session_id: Uuid, old: &str, new_name: &str) -> Result<nbcore::schema::Notebook> {
    if old.is_empty() || new_name.is_empty() { anyhow::bail!("empty name"); }
    let mut map = SESSION_REG.lock();
    let sess = map.get_mut(&session_id).ok_or_else(|| anyhow::anyhow!("Unknown session"))?;
    let mut nb = sess.notebook().clone();
    let re = regex::Regex::new(&format!(r"(?m)\b{}\b", regex::escape(old)))?;
    for c in nb.cells.iter_mut() {
        if c.r#type != nbcore::schema::CellType::Code || c.language.to_lowercase() != "lyra" { continue; }
        // Only replace in code regions (skip comments/strings)
        let code = code_mask(&c.input);
        let mut last = 0usize;
        let mut out = String::with_capacity(c.input.len());
        for m in re.find_iter(&c.input) {
            let s = m.start(); let e = m.end();
            if !code.get(s).copied().unwrap_or(true) { continue; }
            out.push_str(&c.input[last..s]);
            out.push_str(new_name);
            last = e;
        }
        out.push_str(&c.input[last..]);
        if !out.is_empty() { c.input = out; }
    }
    *sess.notebook_mut() = nb.clone();
    Ok(nb)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_offsets_line_col() {
        let s = "let a = 1\nlet foo = a + 1\n";
        let idx = s.find("foo").unwrap();
        let (sl, sc, el, ec) = offsets_to_line_col(s, idx, idx+3);
        assert_eq!((sl, sc, el, ec), (1, 4, 1, 7));
    }
    #[test]
    fn test_find_defs_and_refs() {
        let text = "(* comment *)\nlet foo = 1\nlet bar = foo + 2\nfoo[bar]\n";
        let defs = find_let_defs(text, "foo");
        assert_eq!(defs.len(), 1);
        let refs = find_word_occurrences(text, "foo");
        assert!(refs.len() >= 2);
    }
}

// Build a boolean mask of code regions: true for code, false for inside comments/strings
fn code_mask(text: &str) -> Vec<bool> {
    let bytes = text.as_bytes();
    let n = bytes.len();
    let mut mask = vec![true; n];
    let mut i = 0usize;
    let mut in_string = false;
    let mut in_comment = 0u32; // allow simple nesting
    while i < n {
        if in_string {
            mask[i] = false;
            if bytes[i] == b'\\' { // escape next
                if i + 1 < n { mask[i+1] = false; i += 2; continue; }
            }
            if bytes[i] == b'"' { in_string = false; }
            i += 1;
            continue;
        }
        if in_comment > 0 {
            mask[i] = false;
            // detect "*)"
            if bytes[i] == b'*' && i + 1 < n && bytes[i+1] == b')' {
                mask[i+1] = false;
                in_comment = in_comment.saturating_sub(1);
                i += 2; continue;
            }
            // allow simple nested comments "(*"
            if bytes[i] == b'(' && i + 1 < n && bytes[i+1] == b'*' {
                mask[i+1] = false; in_comment = in_comment.saturating_add(1); i += 2; continue;
            }
            i += 1; continue;
        }
        // Not in string/comment
        if bytes[i] == b'"' { mask[i] = false; in_string = true; i += 1; continue; }
        if bytes[i] == b'(' && i + 1 < n && bytes[i+1] == b'*' {
            mask[i] = false; mask[i+1] = false; in_comment = 1; i += 2; continue;
        }
        i += 1;
    }
    mask
}

pub fn editor_diagnostics(text: String) -> Vec<EditorDiagnostic> {
    // Use parser detailed errors to get index; map to line/column
    let mut p = lyra_parser::Parser::from_source(&text);
    match p.parse_all_detailed() {
        Ok(_) => vec![],
        Err(e) => {
            let (sl, sc) = offset_to_line_col(&text, e.pos);
            vec![EditorDiagnostic {
                message: format!("{}", e.message),
                start_line: sl,
                start_col: sc,
                end_line: sl,
                end_col: sc + 1,
                severity: "Error".into(),
            }]
        }
    }
}

fn offset_to_line_col(text: &str, pos: usize) -> (u32, u32) {
    let mut idx: usize = 0;
    let mut line: u32 = 0;
    let mut col: u32 = 0;
    for ch in text.chars() {
        if idx >= pos { break; }
        if ch == '\n' { line += 1; col = 0; } else { col += 1; }
        idx += ch.len_utf8();
    }
    (line, col)
}

// --- Language info and context ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LangOperator { pub symbol: String, pub fixity: String, pub precedence: u32, pub associativity: String }

pub fn lang_operators() -> Vec<LangOperator> {
    // Minimal operator table aligned with parser precedence
    vec![
        LangOperator { symbol: "//".into(), fixity: "postfix".into(), precedence: 30, associativity: "left".into() },
        LangOperator { symbol: "/@".into(), fixity: "postfix".into(), precedence: 30, associativity: "left".into() },
        LangOperator { symbol: "//.".into(), fixity: "infix".into(), precedence: 40, associativity: "left".into() },
        LangOperator { symbol: "/.".into(), fixity: "infix".into(), precedence: 40, associativity: "left".into() },
        LangOperator { symbol: "^".into(), fixity: "infix".into(), precedence: 50, associativity: "right".into() },
        LangOperator { symbol: "*".into(), fixity: "infix".into(), precedence: 60, associativity: "left".into() },
        LangOperator { symbol: "/".into(), fixity: "infix".into(), precedence: 60, associativity: "left".into() },
        LangOperator { symbol: "+".into(), fixity: "infix".into(), precedence: 70, associativity: "left".into() },
        LangOperator { symbol: "-".into(), fixity: "infix".into(), precedence: 70, associativity: "left".into() },
        LangOperator { symbol: "==".into(), fixity: "infix".into(), precedence: 80, associativity: "left".into() },
        LangOperator { symbol: "!=".into(), fixity: "infix".into(), precedence: 80, associativity: "left".into() },
        LangOperator { symbol: "<".into(), fixity: "infix".into(), precedence: 80, associativity: "left".into() },
        LangOperator { symbol: "<=".into(), fixity: "infix".into(), precedence: 80, associativity: "left".into() },
        LangOperator { symbol: ">".into(), fixity: "infix".into(), precedence: 80, associativity: "left".into() },
        LangOperator { symbol: ">=".into(), fixity: "infix".into(), precedence: 80, associativity: "left".into() },
        LangOperator { symbol: "&&".into(), fixity: "infix".into(), precedence: 90, associativity: "left".into() },
        LangOperator { symbol: "||".into(), fixity: "infix".into(), precedence: 100, associativity: "left".into() },
        LangOperator { symbol: "!".into(), fixity: "prefix".into(), precedence: 20, associativity: "right".into() },
        LangOperator { symbol: "-".into(), fixity: "prefix".into(), precedence: 20, associativity: "right".into() },
    ]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditorContext {
    pub region: String, // code | string | comment
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arg_index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operator: Option<String>,
}

pub fn editor_context(text: String, offset: u32) -> EditorContext {
    let off = offset as usize;
    let mask = code_mask(&text);
    let region = if off < mask.len() && !mask[off] { // decide string vs comment via a secondary scan
        // quick heuristic: walk backwards to find '"' or '(*'
        let bytes = text.as_bytes();
        let mut i = off.min(bytes.len().saturating_sub(1));
        let mut typ = "comment";
        while i > 0 {
            if bytes[i] == b'"' { typ = "string"; break; }
            if i>0 && bytes[i-1]==b'(' && bytes[i]==b'*' { typ = "comment"; break; }
            i -= 1;
        }
        typ.into()
    } else { "code".into() };

    // Head symbol and argument index heuristics
    let mut head: Option<String> = None; let mut arg_index: Option<u32> = None;
    if region == "code" {
        // find nearest unmatched '[' to the left and capture head token before it
        let mut i = off.min(text.len());
        let mut depth: i32 = 0; let bytes = text.as_bytes();
        while i > 0 { i -= 1; let b = bytes[i];
            if b == b']' { depth += 1; }
            else if b == b'[' { if depth == 0 { // found
                // scan backward for symbol name
                let mut j = i; while j>0 && bytes[j-1].is_ascii_whitespace() { j-=1; }
                let mut k = j; while k>0 { let ch = bytes[k-1] as char; if ch.is_ascii_alphanumeric() || ch == '_' { k-=1; } else { break; } }
                if k < j { head = Some(text[k..j].to_string()); }
                // compute argument index by counting commas from i..off
                let mut idx = 0u32; let mut p = i+1; let mut nested = 0i32; while p < off && p < bytes.len() {
                    if bytes[p] == b'[' { nested += 1; }
                    else if bytes[p] == b']' { if nested>0 { nested -= 1; } else { break; } }
                    else if bytes[p] == b',' && nested==0 { idx += 1; }
                    p+=1;
                }
                arg_index = Some(idx);
                break;
            } else { depth -= 1; } }
        }
    }

    // Operator at/near caret
    let mut operator: Option<String> = None;
    if region == "code" {
        let ops = lang_operators();
        let window = text.get(off.saturating_sub(2)..text.len().min(off+3)).unwrap_or("");
        for op in ops {
            if window.contains(&op.symbol) { operator = Some(op.symbol); break; }
        }
    }
    EditorContext { region, head, arg_index, operator }
}

// --- Data Table scaffolding ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnMeta {
    pub name: String,
    pub r#type: String,
    pub nullable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frozen: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableSchema {
    pub columns: Vec<ColumnMeta>,
    pub row_count_approx: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageInfo { pub offset: u64, pub limit: u64, pub total_approx: u64 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableQueryResp {
    pub rows: serde_json::Value,
    pub page: PageInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFilter { pub col: String, pub op: String, pub value: serde_json::Value }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuerySort { pub col: String, pub dir: String }

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TableQuery {
    #[serde(default)]
    pub filters: Vec<QueryFilter>,
    #[serde(default)]
    pub sort: Vec<QuerySort>,
    #[serde(default)]
    pub columns: Option<Vec<String>>, // projection
    pub offset: Option<u64>,
    pub limit: Option<u64>,
    pub search: Option<String>,
}

#[derive(Debug)]
struct TableEntry { value: Value, created_at: Instant, rows_cache: Option<Vec<Value>> }

lazy_static! {
    static ref TABLE_REG: Mutex<HashMap<Uuid, HashMap<String, TableEntry>>> = Mutex::new(HashMap::new());
}

pub fn table_open(session_id: Uuid, value_json: String) -> Result<String> {
    // Accept either raw typed JSON or versioned envelope
    let raw: serde_json::Value = serde_json::from_str(&value_json)?;
    let v: Value = if let Some(val) = raw.get("value") {
        serde_json::from_value(val.clone())?
    } else {
        serde_json::from_value(raw)?
    };
    let handle = Uuid::new_v4().to_string();
    let mut map = TABLE_REG.lock();
    let inner = map.entry(session_id).or_insert_with(HashMap::new);
    inner.insert(handle.clone(), TableEntry { value: v, created_at: Instant::now(), rows_cache: None });
    Ok(handle)
}

pub fn table_close(session_id: Uuid, handle: String) -> Result<bool> {
    let mut map = TABLE_REG.lock();
    if let Some(inner) = map.get_mut(&session_id) {
        Ok(inner.remove(&handle).is_some())
    } else { Ok(false) }
}

pub fn table_schema(session_id: Uuid, handle: String, timeout_ms: Option<u64>) -> Result<TableSchema> {
    let map = TABLE_REG.lock();
    let inner = map.get(&session_id).ok_or_else(|| anyhow::anyhow!("Unknown session"))?;
    let entry = inner.get(&handle).ok_or_else(|| anyhow::anyhow!("Unknown handle"))?;
    let (cols, rows) = infer_schema_or_collect(session_id, handle.clone(), entry, timeout_ms)?;
    Ok(TableSchema { columns: cols, row_count_approx: rows as u64 })
}

pub fn table_query(session_id: Uuid, handle: String, q: TableQuery, timeout_ms: Option<u64>) -> Result<TableQueryResp> {
    // Ensure rows are materialized if needed (Frame)
    let mut items: Vec<Value> = collect_rows_if_needed(session_id, handle.clone(), timeout_ms)?;
    // Apply basic search/filters/sort, then paginate
    let off = q.offset.unwrap_or(0) as usize;
    let lim = q.limit.unwrap_or(100) as usize;
    let mut refs: Vec<&Value> = items.iter().collect();
    // Search (case-insensitive contains across string-like fields)
    if let Some(s) = q.search.as_ref() {
        let needle = s.to_lowercase();
        refs = refs.into_iter().filter(|row| row_matches_search(row, &needle)).collect();
    }
    // Filters
    if !q.filters.is_empty() {
        refs = refs.into_iter().filter(|row| row_matches_filters(row, &q.filters)).collect();
    }
    // Sort (single or multi)
    if !q.sort.is_empty() {
        let sort = q.sort.clone();
        refs.sort_by(|a,b| compare_rows(a,b,&sort));
    }
    let total = refs.len();
    let end = std::cmp::min(total, off.saturating_add(lim));
    let slice = if off < end { &refs[off..end] } else { &[] };
    let rows = slice.iter().map(|v| value_row_to_json(v)).collect::<Vec<_>>();
    Ok(TableQueryResp { rows: serde_json::Value::Array(rows), page: PageInfo { offset: off as u64, limit: lim as u64, total_approx: total as u64 } })
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ColumnStats {
    #[serde(skip_serializing_if = "Option::is_none")] pub nulls: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")] pub uniques_approx: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")] pub min: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")] pub max: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")] pub mean: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")] pub stddev: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")] pub histogram: Option<Histogram>,
    #[serde(skip_serializing_if = "Option::is_none")] pub topk: Option<Vec<(String, u64)>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Histogram { pub bins: Vec<u64>, pub edges: Vec<f64> }

pub fn table_stats(session_id: Uuid, handle: String, columns: Option<Vec<String>>, query: Option<TableQuery>, timeout_ms: Option<u64>) -> Result<serde_json::Value> {
    let map = TABLE_REG.lock();
    let inner = map.get(&session_id).ok_or_else(|| anyhow::anyhow!("Unknown session"))?;
    let entry = inner.get(&handle).ok_or_else(|| anyhow::anyhow!("Unknown handle"))?;
    // If it's a Frame, ensure rows are collected (with optional budget)
    if is_frame_value(&entry.value) {
        let _ = ensure_rows_cache_with_budget(session_id, handle.clone(), timeout_ms)?;
    }
    let mut out = serde_json::Map::new();
    // Build working set applying search/filters (ignore sort)
    let mut items: Vec<&Value> = match &entry.value { Value::List(v) => v.iter().collect(), _ => Vec::new() };
    if let Some(q) = query.as_ref() {
        if let Some(s) = q.search.as_ref() { let needle = s.to_lowercase(); items = items.into_iter().filter(|row| row_matches_search(row, &needle)).collect(); }
        if !q.filters.is_empty() { items = items.into_iter().filter(|row| row_matches_filters(row, &q.filters)).collect(); }
    }
    let sel: Vec<String> = if let Some(cols) = columns { cols } else { infer_default_column_names(items.as_slice()) };
    for col in sel {
        let stats = compute_column_stats(&items, &col, 20, 100_000);
        out.insert(col, serde_json::to_value(stats).unwrap_or(serde_json::Value::Null));
    }
    Ok(serde_json::Value::Object(out))
}

pub fn table_export(session_id: Uuid, handle: String, format: String, query: Option<TableQuery>, columns: Option<Vec<String>>) -> Result<String> {
    if format.to_lowercase() != "csv" { anyhow::bail!("only csv supported"); }
    // Materialize rows
    let items: Vec<Value> = collect_rows_if_needed(session_id, handle.clone(), Some(10_000))?; // soft budget for Frames
    // Apply query (search/filters/sort)
    let mut refs: Vec<&Value> = items.iter().collect();
    if let Some(q) = query.as_ref() {
        if let Some(s) = q.search.as_ref() { let needle = s.to_lowercase(); refs = refs.into_iter().filter(|row| row_matches_search(row, &needle)).collect(); }
        if !q.filters.is_empty() { refs = refs.into_iter().filter(|row| row_matches_filters(row, &q.filters)).collect(); }
        if !q.sort.is_empty() { let sort = q.sort.clone(); refs.sort_by(|a,b| compare_rows(a,b,&sort)); }
    }
    // Determine headers
    let headers: Vec<String> = if let Some(cols) = columns.as_ref() { if !cols.is_empty() { cols.clone() } else { infer_default_column_names(&refs) } } else { infer_default_column_names(&refs) };
    // Build CSV
    let mut out = String::new();
    // headers
    for (i, h) in headers.iter().enumerate() { if i>0 { out.push(','); } out.push('"'); out.push_str(&h.replace("\"", "\"\"")); out.push('"'); }
    out.push('\n');
    for row in refs.iter() {
        if let Value::Assoc(m) = row { 
            for (i, h) in headers.iter().enumerate() { 
                if i>0 { out.push(','); }
                let cell = m.get(h).cloned().unwrap_or(Value::Symbol("Null".into()));
                let s = match cell { Value::String(s) => s, v => format!("{}", lyra_core::pretty::format_value(&v)) };
                out.push('"'); out.push_str(&s.replace("\"", "\"\"")); out.push('"');
            }
            out.push('\n');
        }
    }
    let b64 = base64::engine::general_purpose::STANDARD.encode(out.as_bytes());
    Ok(format!("data:text/csv;base64,{}", b64))
}

fn infer_schema(value: &Value) -> (Vec<ColumnMeta>, usize) {
    match value {
        Value::List(items) => {
            // Array of objects
            for it in items {
                if let Value::Assoc(m) = it {
                    let mut cols: Vec<ColumnMeta> = m.keys().cloned().map(|k| ColumnMeta { name: k, r#type: "any".into(), nullable: true, width: None, frozen: None, order: None }).collect();
                    cols.sort_by(|a,b| a.name.cmp(&b.name));
                    return (cols, items.len());
                }
            }
            // Array of arrays
            for it in items {
                if let Value::List(row) = it {
                    let cols: Vec<ColumnMeta> = (0..row.len()).map(|i| ColumnMeta { name: format!("c{}", i+1), r#type: "any".into(), nullable: true, width: None, frozen: None, order: None }).collect();
                    return (cols, items.len());
                }
            }
            (vec![], items.len())
        }
        _ => (vec![], 0),
    }
}

fn count_rows(value: &Value) -> usize {
    match value { Value::List(items) => items.len(), _ => 0 }
}

fn extract_rows_json(value: &Value, offset: usize, limit: usize) -> serde_json::Value {
    match value {
        Value::List(items) => {
            let end = std::cmp::min(items.len(), offset.saturating_add(limit));
            let mut out: Vec<serde_json::Value> = Vec::new();
            for it in &items[offset..end] {
                match it {
                    Value::Assoc(m) => {
                        let mut obj = serde_json::Map::new();
                        for (k, v) in m.iter() { obj.insert(k.clone(), serde_json::to_value(v).unwrap_or(serde_json::Value::Null)); }
                        out.push(serde_json::Value::Object(obj));
                    }
                    Value::List(row) => {
                        let arr: Vec<serde_json::Value> = row.iter().map(|v| serde_json::to_value(v).unwrap_or(serde_json::Value::Null)).collect();
                        out.push(serde_json::Value::Array(arr));
                    }
                    other => {
                        out.push(serde_json::to_value(other).unwrap_or(serde_json::Value::Null));
                    }
                }
            }
            serde_json::Value::Array(out)
        }
        _ => serde_json::Value::Array(vec![]),
    }
}

fn infer_schema_or_collect(session_id: Uuid, handle: String, entry: &TableEntry, timeout_ms: Option<u64>) -> Result<(Vec<ColumnMeta>, usize)> {
    // If already list, infer directly
    if matches!(entry.value, Value::List(_)) { return Ok(infer_schema(&entry.value)); }
    // If Frame, collect rows once and cache
    if is_frame_value(&entry.value) {
        let rows = ensure_rows_cache_with_budget(session_id, handle, timeout_ms)?;
        let v = Value::List(rows);
        return Ok(infer_schema(&v));
    }
    Ok((vec![], 0))
}

fn is_frame_value(v: &Value) -> bool {
    match v { Value::Assoc(m) => matches!(m.get("__type"), Some(Value::String(s)) if s=="Frame"), _ => false }
}

fn ensure_rows_cache_with_budget(session_id: Uuid, handle: String, timeout_ms: Option<u64>) -> Result<Vec<Value>> {
    // fast path: cached
    if let Some(rows) = {
        let map = TABLE_REG.lock();
        map.get(&session_id)
            .and_then(|inner| inner.get(&handle))
            .and_then(|e| e.rows_cache.clone())
    } { return Ok(rows); }
    // Check if frame and capture value clone without holding lock
    let frame_val_opt: Option<Value> = {
        let map = TABLE_REG.lock();
        map.get(&session_id)
            .and_then(|inner| inner.get(&handle))
            .and_then(|e| if is_frame_value(&e.value) { Some(e.value.clone()) } else { None })
    };
    let Some(frame_val) = frame_val_opt else { return Ok(vec![]); };
    // Use kernel session to collect rows
    let mut sreg = crate::api::SESSION_REG.lock();
    let sess = sreg.get_mut(&session_id).ok_or_else(|| anyhow::anyhow!("Unknown session"))?;
    let expr = Value::Expr { head: Box::new(Value::Symbol("FrameCollect".into())), args: vec![frame_val] };
    let rows_v = if let Some(ms) = timeout_ms { sess.eval_with_time_budget(expr, ms as i64) } else { sess.eval_in_scope(expr) };
    let rows = match rows_v { Value::List(v) => v, _ => vec![] };
    drop(sreg);
    // store cache
    let mut map2 = TABLE_REG.lock();
    if let Some(inner2) = map2.get_mut(&session_id) { if let Some(e2) = inner2.get_mut(&handle) { e2.rows_cache = Some(rows.clone()); } }
    Ok(rows)
}
fn collect_rows_if_needed(session_id: Uuid, handle: String, timeout_ms: Option<u64>) -> Result<Vec<Value>> {
    ensure_rows_cache_with_budget(session_id, handle, timeout_ms)
}
// (Legacy helper removed; use collect_rows_if_needed(session_id, handle, timeout_ms))

fn infer_default_column_names(items: &[&Value]) -> Vec<String> {
    // If we see an object row, return its keys; if array row, return cN up to first row len
    for it in items.iter() {
        if let Value::Assoc(m) = it { return m.keys().cloned().collect(); }
    }
    for it in items.iter() {
        if let Value::List(row) = it { return (0..row.len()).map(|i| format!("c{}", i+1)).collect(); }
    }
    vec![]
}

fn compute_column_stats(items: &[&Value], col: &str, bins_target: usize, sample_cap: usize) -> ColumnStats {
    let mut nulls: u64 = 0;
    let mut nums: Vec<f64> = Vec::new(); nums.reserve(std::cmp::min(items.len(), sample_cap));
    let mut cats: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    let mut used: usize = 0;
    for it in items.iter() {
        if used >= sample_cap { break; }
        used += 1;
        match get_cell(it, col) {
            Some(Value::Integer(i)) => nums.push(*i as f64),
            Some(Value::Real(f)) => nums.push(*f),
            Some(Value::String(s)) => { *cats.entry(s.clone()).or_insert(0) += 1; }
            Some(Value::Boolean(b)) => { *cats.entry(b.to_string()).or_insert(0) += 1; }
            Some(Value::Assoc(_)) | Some(Value::List(_)) => { nulls += 1; }
            None => nulls += 1,
            _ => {}
        }
    }
    let mut out = ColumnStats::default();
    if !nums.is_empty() {
        let mut min = f64::INFINITY; let mut max = f64::NEG_INFINITY; let mut mean = 0.0f64; let mut m2 = 0.0f64; let mut n = 0f64;
        for &x in nums.iter() {
            if x < min { min = x; } if x > max { max = x; }
            n += 1.0; let delta = x - mean; mean += delta / n; m2 += delta * (x - mean);
        }
        let var = if n > 1.0 { m2 / (n - 1.0) } else { 0.0 };
        if let Some(n) = serde_json::Number::from_f64(min) { out.min = Some(serde_json::Value::Number(n)); }
        if let Some(n) = serde_json::Number::from_f64(max) { out.max = Some(serde_json::Value::Number(n)); }
        out.mean = Some(mean);
        out.stddev = Some(var.sqrt());
        // histogram
        let bins_n = std::cmp::max(1, std::cmp::min(bins_target as i64, 64)) as usize;
        let mut edges: Vec<f64> = Vec::with_capacity(bins_n+1);
        if min.is_finite() && max.is_finite() {
            if (max - min).abs() < std::f64::EPSILON { edges = vec![min, max]; }
            else { let step = (max - min) / (bins_n as f64); for i in 0..=bins_n { edges.push(min + step * (i as f64)); } }
        }
        let mut bins = vec![0u64; std::cmp::max(1, edges.len().saturating_sub(1))];
        if edges.len() >= 2 {
            for &x in nums.iter() {
                if x <= edges[0] { bins[0] += 1; continue; }
                if x >= *edges.last().unwrap() { *bins.last_mut().unwrap() += 1; continue; }
                let pos = ((x - edges[0]) / (edges.last().unwrap() - edges[0])) * (bins.len() as f64);
                let idx = std::cmp::min(bins.len()-1, pos.floor() as usize);
                bins[idx] += 1;
            }
        }
        out.histogram = Some(Histogram { bins, edges });
    }
    if !cats.is_empty() {
        let mut arr: Vec<(String, u64)> = cats.into_iter().collect();
        arr.sort_by(|a,b| b.1.cmp(&a.1));
        out.topk = Some(arr.into_iter().take(10).collect());
        out.uniques_approx = Some(out.topk.as_ref().map(|v| v.len() as u64).unwrap_or(0));
    }
    out.nulls = Some(nulls);
    out
}

fn row_matches_search(row: &Value, needle: &str) -> bool {
    match row {
        Value::Assoc(m) => {
            for (_k, v) in m.iter() {
                if value_contains(v, needle) { return true; }
            }
            false
        }
        Value::List(arr) => {
            for v in arr.iter() { if value_contains(v, needle) { return true; } }
            false
        }
        _ => false,
    }
}

fn value_contains(v: &Value, needle: &str) -> bool {
    value_to_string(v).to_lowercase().contains(needle)
}

fn row_matches_filters(row: &Value, filters: &Vec<QueryFilter>) -> bool {
    for f in filters.iter() {
        if !row_matches_filter(row, f) { return false; }
    }
    true
}

fn get_cell<'a>(row: &'a Value, col: &str) -> Option<&'a Value> {
    match row {
        Value::Assoc(m) => m.get(col),
        Value::List(arr) => {
            // support numeric index or cN form
            if let Ok(i) = col.parse::<usize>() { return arr.get(i); }
            if let Some(rem) = col.strip_prefix('c') { if let Ok(i) = rem.parse::<usize>() { return arr.get(i.saturating_sub(1)); } }
            None
        }
        _ => None,
    }
}

fn row_matches_filter(row: &Value, f: &QueryFilter) -> bool {
    let op = f.op.as_str();
    let v = &f.value;
    // Try to map JSON value to comparable types against lyra_core::Value
    let cell = get_cell(row, &f.col);
    match op {
        "isNull" => cell.is_none(),
        "contains" => {
            let needle = v.as_str().unwrap_or("").to_lowercase();
            match cell { Some(Value::String(s)) => s.to_lowercase().contains(&needle), Some(other) => value_to_string(other).to_lowercase().contains(&needle), None => false }
        }
        "eq" | "neq" | "lt" | "lte" | "gt" | "gte" => {
            let ord = match (cell, v) {
                (Some(Value::Integer(a)), serde_json::Value::Number(n)) => n.as_i64().map(|b| a.cmp(&b)),
                (Some(Value::Real(a)), serde_json::Value::Number(n)) => n.as_f64().map(|b| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)),
                (Some(Value::String(a)), serde_json::Value::String(b)) => Some(a.cmp(b)),
                (Some(Value::Boolean(a)), serde_json::Value::Bool(b)) => Some(a.cmp(b)),
                // fallback string compare
                (Some(a), other) => Some(value_to_string(a).cmp(&other.to_string())),
                (None, _) => None,
            };
            let ok = match (op, ord) {
                ("eq", Some(std::cmp::Ordering::Equal)) => true,
                ("neq", Some(std::cmp::Ordering::Equal)) => false,
                ("neq", _) => true,
                ("lt", Some(std::cmp::Ordering::Less)) => true,
                ("lte", Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)) => true,
                ("gt", Some(std::cmp::Ordering::Greater)) => true,
                ("gte", Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)) => true,
                _ => false,
            };
            ok
        }
        _ => true,
    }
}

fn compare_rows(a: &Value, b: &Value, sort: &Vec<QuerySort>) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;
    for s in sort.iter() {
        let av = get_cell(a, &s.col);
        let bv = get_cell(b, &s.col);
        let ord = match (av, bv) {
            (Some(Value::Integer(x)), Some(Value::Integer(y))) => x.cmp(y),
            (Some(Value::Real(x)), Some(Value::Real(y))) => x.partial_cmp(y).unwrap_or(Equal),
            (Some(Value::String(x)), Some(Value::String(y))) => x.cmp(y),
            // fallback to string compare
            (Some(x), Some(y)) => value_to_string(x).cmp(&value_to_string(y)),
            (None, None) => Equal,
            (None, Some(_)) => Greater, // missing last
            (Some(_), None) => Less,
        };
        let ord = if s.dir.eq_ignore_ascii_case("desc") { ord.reverse() } else { ord };
        if ord != Equal { return ord; }
    }
    Equal
}

fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Integer(i) => i.to_string(),
        Value::Real(f) => {
            let mut s = f.to_string();
            if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s }
        }
        Value::Boolean(b) => b.to_string(),
        other => serde_json::to_value(other).map(|j| j.to_string()).unwrap_or_default(),
    }
}

fn value_row_to_json(v: &Value) -> serde_json::Value {
    match v {
        Value::Assoc(m) => {
            let mut obj = serde_json::Map::new();
            for (k, vv) in m.iter() { obj.insert(k.clone(), serde_json::to_value(vv).unwrap_or(serde_json::Value::Null)); }
            serde_json::Value::Object(obj)
        }
        Value::List(arr) => {
            serde_json::Value::Array(arr.iter().map(|vv| serde_json::to_value(vv).unwrap_or(serde_json::Value::Null)).collect())
        }
        _ => serde_json::to_value(v).unwrap_or(serde_json::Value::Null),
    }
}
