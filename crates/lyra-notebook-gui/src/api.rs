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

lazy_static! {
    pub(crate) static ref SESSION_REG: Mutex<HashMap<Uuid, kernel::Session>> = Mutex::new(HashMap::new());
}

pub fn open_notebook(path: &str) -> Result<kernel::OpenResponse> {
    let nb = nbcore::read_notebook(path)?;
    // In GUI, prefer rich display by default
    lyra_stdlib::display::set_prefer_display(true);
    let settings = kernel::SessionSettings::default();
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

pub fn execute_all(session_id: Uuid, ids: Option<Vec<Uuid>>, method: Option<String>) -> Result<Vec<kernel::ExecResult>> {
    let mut map = SESSION_REG.lock();
    if let Some(sess) = map.get_mut(&session_id) {
        let use_ids: Vec<Uuid> = ids.unwrap_or_else(|| sess.notebook().cells.iter().map(|c| c.id).collect());
        Ok(sess.execute_many(&use_ids, method.as_deref().unwrap_or("Linear"), kernel::ExecutionOpts::default()))
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
        let v: lyra_core::value::Value = serde_json::from_str(&value_json)?;
        let out = sess.preview_rows(v, limit as usize);
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
    let v: Value = serde_json::from_str(&value_json)?;
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

pub fn table_schema(session_id: Uuid, handle: String) -> Result<TableSchema> {
    let map = TABLE_REG.lock();
    let inner = map.get(&session_id).ok_or_else(|| anyhow::anyhow!("Unknown session"))?;
    let entry = inner.get(&handle).ok_or_else(|| anyhow::anyhow!("Unknown handle"))?;
    let (cols, rows) = infer_schema_or_collect(session_id, handle.clone(), entry)?;
    Ok(TableSchema { columns: cols, row_count_approx: rows as u64 })
}

pub fn table_query(session_id: Uuid, handle: String, q: TableQuery) -> Result<TableQueryResp> {
    // Ensure rows are materialized if needed (Frame)
    let mut items: Vec<Value> = collect_rows_if_needed(session_id, handle.clone())?;
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

pub fn table_stats(session_id: Uuid, handle: String, columns: Option<Vec<String>>, query: Option<TableQuery>) -> Result<serde_json::Value> {
    let map = TABLE_REG.lock();
    let inner = map.get(&session_id).ok_or_else(|| anyhow::anyhow!("Unknown session"))?;
    let entry = inner.get(&handle).ok_or_else(|| anyhow::anyhow!("Unknown handle"))?;
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

pub fn table_export(_session_id: Uuid, _handle: String, _format: String) -> Result<String> {
    anyhow::bail!("Export not implemented yet")
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

fn infer_schema_or_collect(session_id: Uuid, handle: String, entry: &TableEntry) -> Result<(Vec<ColumnMeta>, usize)> {
    // If already list, infer directly
    if matches!(entry.value, Value::List(_)) { return Ok(infer_schema(&entry.value)); }
    // If Frame, collect rows once and cache
    if is_frame_value(&entry.value) {
        let rows = ensure_rows_cache(session_id, handle)?;
        let v = Value::List(rows);
        return Ok(infer_schema(&v));
    }
    Ok((vec![], 0))
}

fn is_frame_value(v: &Value) -> bool {
    match v { Value::Assoc(m) => matches!(m.get("__type"), Some(Value::String(s)) if s=="Frame"), _ => false }
}

fn ensure_rows_cache(session_id: Uuid, handle: String) -> Result<Vec<Value>> {
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
    let rows_v = sess.eval_in_scope(expr);
    let rows = match rows_v { Value::List(v) => v, _ => vec![] };
    drop(sreg);
    // store cache
    let mut map2 = TABLE_REG.lock();
    if let Some(inner2) = map2.get_mut(&session_id) { if let Some(e2) = inner2.get_mut(&handle) { e2.rows_cache = Some(rows.clone()); } }
    Ok(rows)
}

fn collect_rows_if_needed(session_id: Uuid, handle: String) -> Result<Vec<Value>> {
    let map = TABLE_REG.lock();
    let inner = map.get(&session_id).ok_or_else(|| anyhow::anyhow!("Unknown session"))?;
    let entry = inner.get(&handle).ok_or_else(|| anyhow::anyhow!("Unknown handle"))?;
    match &entry.value {
        Value::List(v) => Ok(v.clone()),
        v if is_frame_value(v) => ensure_rows_cache(session_id, handle),
        _ => Ok(vec![]),
    }
}

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
