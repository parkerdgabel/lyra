use crate::types::{DisplayItem, ExecEvent, ExecResult};
use lyra_core::pretty::format_value;
use lyra_core::value::Value;
use lyra_notebook_core as nb;
use lyra_parser::Parser;
use lyra_runtime::Evaluator;
// removed unused imports
use std::time::Instant;
use uuid::Uuid;
use serde_json as sj;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SessionSettings {
    pub capture_explain: bool,
    pub max_threads: Option<i64>,
    pub time_budget_ms: Option<i64>,
    pub enable_cache: bool,
    pub cache_dir: Option<String>,
    pub cache_salt: Option<String>,
}

impl Default for SessionSettings {
    fn default() -> Self {
        Self { capture_explain: false, max_threads: None, time_budget_ms: None, enable_cache: true, cache_dir: None, cache_salt: None }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ExecutionOpts {
    pub with_init_cells: bool,
    pub ignore_cache: bool,
}

pub struct Session {
    id: Uuid,
    nb: nb::schema::Notebook,
    ev: Evaluator,
    settings: SessionSettings,
    scope_id: Option<i64>,
    cache: HashMap<String, Vec<DisplayItem>>,
}

impl Session {
    pub fn new(nb: nb::schema::Notebook, settings: SessionSettings) -> Self {
        let id = Uuid::now_v7();
        let mut ev = Evaluator::new();
        // Register stdlib so that symbols like Range, Plus, etc. evaluate
        lyra_stdlib::register_all(&mut ev);
        let mut s = Self { id, nb, ev, settings, scope_id: None, cache: HashMap::new() };
        s.start_scope();
        s
    }

    pub fn id(&self) -> Uuid { self.id }
    pub fn notebook(&self) -> &nb::schema::Notebook { &self.nb }
    pub fn notebook_mut(&mut self) -> &mut nb::schema::Notebook { &mut self.nb }

    // Cache controls
    pub fn set_cache_enabled(&mut self, on: bool) { self.settings.enable_cache = on; }
    pub fn clear_cache_memory(&mut self) { self.cache.clear(); }
    pub fn clear_cache_disk(&self) -> bool {
        let Some(dir) = self.settings.cache_dir.as_ref() else { return false; };
        let Ok(rd) = std::fs::read_dir(dir) else { return false; };
        let mut ok_any = false;
        for ent in rd.flatten() {
            if let Some(name) = ent.file_name().to_str() {
                if name.ends_with(".json") { let _ = std::fs::remove_file(ent.path()); ok_any = true; }
            }
        }
        ok_any
    }
    pub fn cache_gc(&self, max_bytes: u64) -> (u64, u64) {
        let Some(dir) = self.settings.cache_dir.as_ref() else { return (0, 0); };
        let Ok(rd) = std::fs::read_dir(dir) else { return (0, 0); };
        let mut files: Vec<(std::path::PathBuf, u64, std::time::SystemTime)> = Vec::new();
        for ent in rd.flatten() {
            let p = ent.path();
            if p.extension().and_then(|s| s.to_str()) != Some("json") { continue; }
            if let Ok(md) = ent.metadata() {
                let sz = md.len(); let mt = md.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                files.push((p, sz, mt));
            }
        }
        files.sort_by_key(|(_, _, mt)| *mt); // oldest first
        let mut total: u64 = files.iter().map(|(_, sz, _)| *sz).sum();
        if total <= max_bytes { return (0, total); }
        let mut freed: u64 = 0;
        for (p, sz, _) in files {
            if total <= max_bytes { break; }
            let _ = std::fs::remove_file(p);
            total = total.saturating_sub(sz);
            freed = freed.saturating_add(sz);
        }
        (freed, total)
    }
    pub fn cache_disk_size_bytes(&self) -> (u64, u64) {
        let Some(dir) = self.settings.cache_dir.as_ref() else { return (0, 0) };
        let Ok(rd) = std::fs::read_dir(dir) else { return (0, 0) };
        let mut total: u64 = 0; let mut files: u64 = 0;
        for ent in rd.flatten() {
            if ent.path().extension().and_then(|s| s.to_str()) != Some("json") { continue; }
            if let Ok(md) = ent.metadata() { total = total.saturating_add(md.len()); files = files.saturating_add(1); }
        }
        (total, files)
    }

    pub fn set_cache_salt<S: Into<Option<String>>>(&mut self, salt: S) { self.settings.cache_salt = salt.into(); }
    pub fn cache_salt(&self) -> Option<String> { self.settings.cache_salt.clone() }

    pub fn reset_env(&mut self) { self.ev = Evaluator::new(); }

    fn scope_val(id: i64) -> Value {
        Value::Expr { head: Box::new(Value::Symbol("ScopeId".into())), args: vec![Value::Integer(id)] }
    }

    fn start_scope(&mut self) -> bool {
        // Build options association for StartScope
        let mut map = lyra_core::value::AssocMap::new();
        if let Some(n) = self.settings.max_threads { map.insert("maxThreads".into(), Value::Integer(n)); }
        if let Some(ms) = self.settings.time_budget_ms { map.insert("timeBudgetMs".into(), Value::Integer(ms)); }
        let opts = Value::Assoc(map);
        let call = Value::Expr { head: Box::new(Value::Symbol("StartScope".into())), args: vec![opts] };
        let res = self.ev.eval(call);
        if let Value::Expr { head, args } = res {
            if matches!(*head, Value::Symbol(ref s) if s=="ScopeId") {
                if let Some(Value::Integer(id)) = args.get(0) {
                    self.scope_id = Some(*id);
                    return true;
                }
            }
        }
        false
    }

    fn end_scope(&mut self) -> bool {
        if let Some(id) = self.scope_id.take() {
            let call = Value::Expr { head: Box::new(Value::Symbol("EndScope".into())), args: vec![Self::scope_val(id)] };
            let v = self.ev.eval(call);
            return matches!(v, Value::Boolean(true));
        }
        false
    }

    pub fn interrupt(&mut self) -> bool {
        if let Some(id) = self.scope_id {
            let call = Value::Expr { head: Box::new(Value::Symbol("CancelScope".into())), args: vec![Self::scope_val(id)] };
            let v = self.ev.eval(call);
            // Start a fresh scope for future work
            let _ = self.end_scope();
            let _ = self.start_scope();
            return matches!(v, Value::Boolean(true));
        }
        false
    }

    pub fn eval_in_scope(&mut self, expr: Value) -> Value {
        if let Some(id) = self.scope_id { 
            let call = Value::Expr { head: Box::new(Value::Symbol("InScope".into())), args: vec![Self::scope_val(id), expr] };
            self.ev.eval(call)
        } else {
            self.ev.eval(expr)
        }
    }

    pub fn eval_with_time_budget(&mut self, expr: Value, budget_ms: i64) -> Value {
        // Start a temporary scope with a custom time budget, evaluate, then end it and restore previous scope
        let prev = self.scope_id;
        // Build options association for StartScope
        let mut map = lyra_core::value::AssocMap::new();
        map.insert("timeBudgetMs".into(), Value::Integer(budget_ms));
        let opts = Value::Assoc(map);
        // Start temporary scope
        let call = Value::Expr { head: Box::new(Value::Symbol("StartScope".into())), args: vec![opts] };
        let res = self.ev.eval(call);
        if let Value::Expr { head, args } = res {
            if matches!(*head, Value::Symbol(ref s) if s=="ScopeId") {
                if let Some(Value::Integer(id)) = args.get(0) {
                    // Switch to temporary scope, eval, then end and restore
                    self.scope_id = Some(*id);
                    let val = self.eval_in_scope(expr);
                    let _ = self.end_scope();
                    self.scope_id = prev;
                    return val;
                }
            }
        }
        // Fallback: evaluate in current scope if temp scope failed
        self.scope_id = prev;
        self.eval_in_scope(expr)
    }

    pub fn preview_rows(&mut self, value: Value, limit: usize) -> Value {
        // For Dataset: Head[value, n] yields first rows (as list of assoc)
        // For Frame: FrameCollect[FrameHead[value, n]] yields list of assoc
        let expr = match &value {
            Value::Assoc(m) if matches!(m.get("__type"), Some(Value::String(s)) if s=="Dataset") => {
                Value::Expr { head: Box::new(Value::Symbol("Head".into())), args: vec![value, Value::Integer(limit as i64)] }
            }
            Value::Assoc(m) if matches!(m.get("__type"), Some(Value::String(s)) if s=="Frame") => {
                let head = Value::Expr { head: Box::new(Value::Symbol("FrameHead".into())), args: vec![value.clone(), Value::Integer(limit as i64)] };
                Value::Expr { head: Box::new(Value::Symbol("FrameCollect".into())), args: vec![head] }
            }
            _ => value,
        };
        self.eval_in_scope(expr)
    }

    pub fn execute_cell(&mut self, cell_id: Uuid, opts: ExecutionOpts) -> ExecResult {
        let start = Instant::now();
        let mut outputs: Vec<DisplayItem> = Vec::new();
        let mut err: Option<String> = None;

        // Locate cell
        let pos = match self.nb.cells.iter().position(|c| c.id == cell_id) {
            Some(i) => i,
            None => {
                return ExecResult { cell_id, duration_ms: 0, outputs, error: Some("Cell not found".into()) };
            }
        };
        let ctype = self.nb.cells[pos].r#type.clone();
        let lang = self.nb.cells[pos].language.clone();
        let input = self.nb.cells[pos].input.clone();
        if !matches!(ctype, nb::schema::CellType::Code) || lang.as_str() != "Lyra" {
            return ExecResult { cell_id, duration_ms: 0, outputs, error: Some("Not a Lyra code cell".into()) };
        }

        // Parse
        let mut parser = Parser::from_source(&input);
        let parsed = parser.parse_all();
        let exprs = match parsed {
            Ok(vs) => vs,
            Err(e) => {
                err = Some(format!("Parse error: {}", e));
                vec![]
            }
        };

        // Cache check
        let cache_key = if self.settings.enable_cache && !opts.ignore_cache {
            Some(self.compute_cache_key(&input))
        } else { None };
        if let Some(ref key) = cache_key {
            if let Some(saved) = self.cache.get(key) {
                outputs = saved.clone();
                // Write outputs + metadata back to notebook cell
                let mut new_cell = self.nb.cells[pos].clone();
                new_cell.output = outputs
                    .iter()
                    .map(|it| nb::schema::DisplayData { mime: it.mime.clone(), data: it.data.clone(), schema: None, meta: None })
                    .collect();
                let dur_ms = start.elapsed().as_millis();
                let meta = &mut new_cell.meta;
                let count = meta.get("execCount").and_then(|v| v.as_i64()).unwrap_or(0) + 1;
                meta.insert("execCount".into(), sj::Value::Number((count as i64).into()));
                meta.insert("timingMs".into(), sj::Value::Number(sj::Number::from(dur_ms as u64)));
                meta.insert("cached".into(), sj::Value::Bool(true));
                meta.remove("error");
                self.nb.cells[pos] = new_cell;
                return ExecResult { cell_id, duration_ms: dur_ms, outputs, error: None };
            }
            // Try disk cache
            if let Some(on_disk) = self.try_load_cache(key) {
                outputs = on_disk.clone();
                // mirror to memory
                self.cache.insert(key.clone(), on_disk);
                let mut new_cell = self.nb.cells[pos].clone();
                new_cell.output = outputs
                    .iter()
                    .map(|it| nb::schema::DisplayData { mime: it.mime.clone(), data: it.data.clone(), schema: None, meta: None })
                    .collect();
                let dur_ms = start.elapsed().as_millis();
                let meta = &mut new_cell.meta;
                let count = meta.get("execCount").and_then(|v| v.as_i64()).unwrap_or(0) + 1;
                meta.insert("execCount".into(), sj::Value::Number((count as i64).into()));
                meta.insert("timingMs".into(), sj::Value::Number(sj::Number::from(dur_ms as u64)));
                meta.insert("cached".into(), sj::Value::Bool(true));
                meta.remove("error");
                self.nb.cells[pos] = new_cell;
                return ExecResult { cell_id, duration_ms: dur_ms, outputs, error: None };
            }
        }

        // Evaluate each expression and capture outputs
        for expr in exprs {
            let val = self.eval_in_scope(expr);
            // Detect rich display as data URL to emit typed output
            let mut emitted_rich = false;
            if let Value::String(s) = &val {
                if let Some((mime, data)) = Self::data_url_mime_and_data(s) {
                    outputs.push(DisplayItem { mime, data });
                    emitted_rich = true;
                }
            }
            if !emitted_rich {
                // Main value as pretty text
                outputs.push(DisplayItem::text(format_value(&val)));
            }
            // Always include machine-readable lyra value JSON
            outputs.push(DisplayItem::lyra_value_json(&val));

            if self.settings.capture_explain {
                // Explain[expr]
                let expl = Value::Expr { head: Box::new(Value::Symbol("Explain".into())), args: vec![val.clone()] };
                let eout = self.eval_in_scope(expl);
                outputs.push(DisplayItem { mime: "application/lyra+explain".into(), data: serde_json::to_string(&eout).unwrap_or_default() });
            }
        }

        // Write outputs + metadata back to notebook cell
        let mut new_cell = self.nb.cells[pos].clone();
        new_cell.output = outputs
            .iter()
            .map(|it| nb::schema::DisplayData { mime: it.mime.clone(), data: it.data.clone(), schema: None, meta: None })
            .collect();
        // Update metadata: execCount, timingMs, error?
        let dur_ms = start.elapsed().as_millis();
        let meta = &mut new_cell.meta;
        // execCount
        let count = meta.get("execCount").and_then(|v| v.as_i64()).unwrap_or(0) + 1;
        meta.insert("execCount".into(), sj::Value::Number((count as i64).into()));
        // timingMs
        meta.insert(
            "timingMs".into(),
            sj::Value::Number(sj::Number::from(dur_ms as u64)),
        );
        // error?
        if let Some(msg) = &err { meta.insert("error".into(), sj::Value::String(msg.clone())); } else { meta.remove("error"); }
        self.nb.cells[pos] = new_cell;

        // Save into cache on success
        if self.settings.enable_cache && err.is_none() {
            if let Some(key) = cache_key { 
                self.cache.insert(key.clone(), outputs.clone());
                let _ = self.try_store_cache(&key, &outputs);
            }
        }

        ExecResult { cell_id, duration_ms: dur_ms, outputs, error: err }
    }

    pub fn execute_many(&mut self, ids: &[Uuid], method: &str, opts: ExecutionOpts) -> Vec<ExecResult> {
        // Optional: run init cells first
        if opts.with_init_cells {
            // Execute all initialization cells in document order
            let init_ids: Vec<Uuid> = self.nb.cells.iter().filter(|c| c.attrs.contains(nb::schema::CellAttrs::INITIALIZATION)).map(|c| c.id).collect();
            for cid in init_ids { let _ = self.execute_cell(cid, ExecutionOpts::default()); }
        }
        // Linear or DAG
        let ordered: Vec<Uuid> = if method.eq_ignore_ascii_case("DAG") {
            self.order_by_deps(ids)
        } else {
            ids.to_vec()
        };
        ordered.iter().map(|cid| self.execute_cell(*cid, ExecutionOpts::default())).collect()
    }

    pub fn eval_text_to_display(&mut self, text: &str) -> (u128, Vec<DisplayItem>, Option<String>) {
        let start = Instant::now();
        let mut outputs: Vec<DisplayItem> = Vec::new();
        let mut err: Option<String> = None;
        let mut parser = Parser::from_source(text);
        let parsed = parser.parse_all();
        let exprs = match parsed { Ok(vs) => vs, Err(e) => { err = Some(format!("Parse error: {}", e)); vec![] } };
        for expr in exprs {
            let val = self.eval_in_scope(expr);
            let mut emitted_rich = false;
            if let Value::String(s) = &val {
                if let Some((mime, data)) = Self::data_url_mime_and_data(s) {
                    outputs.push(DisplayItem { mime, data });
                    emitted_rich = true;
                }
            }
            if !emitted_rich { outputs.push(DisplayItem::text(format_value(&val))); }
            outputs.push(DisplayItem::lyra_value_json(&val));
        }
        let dur = start.elapsed().as_millis();
        (dur, outputs, err)
    }

    pub fn execute_cell_events(&mut self, cell_id: Uuid, opts: ExecutionOpts) -> Vec<ExecEvent> {
        let mut events = Vec::new();
        let res = self.execute_cell_with_cb(cell_id, opts, |ev| events.push(ev));
        events.push(ExecEvent::Finished { result: res });
        events
    }

    pub fn execute_cell_with_cb<F>(&mut self, cell_id: Uuid, opts: ExecutionOpts, mut cb: F) -> ExecResult
    where
        F: FnMut(ExecEvent),
    {
        cb(ExecEvent::Started { cell_id });
        let start = Instant::now();

        let pos = match self.nb.cells.iter().position(|c| c.id == cell_id) {
            Some(i) => i,
            None => {
                cb(ExecEvent::Error { cell_id, message: "Cell not found".into() });
                return ExecResult { cell_id, duration_ms: 0, outputs: vec![], error: Some("Cell not found".into()) };
            }
        };
        let c = self.nb.cells[pos].clone();
        if !matches!(c.r#type, nb::schema::CellType::Code) || c.language.as_str() != "Lyra" {
            cb(ExecEvent::Error { cell_id, message: "Not a Lyra code cell".into() });
            return ExecResult { cell_id, duration_ms: 0, outputs: vec![], error: Some("Not a Lyra code cell".into()) };
        }

        // Parse
        let mut parser = Parser::from_source(&c.input);
        let exprs = match parser.parse_all() {
            Ok(vs) => vs,
            Err(e) => {
                let msg = format!("Parse error: {}", e);
                cb(ExecEvent::Error { cell_id, message: msg.clone() });
                // Update metadata only
                let dur_ms = start.elapsed().as_millis();
                let mut new_cell = c.clone();
                let meta = &mut new_cell.meta;
                let count = meta.get("execCount").and_then(|v| v.as_i64()).unwrap_or(0) + 1;
                meta.insert("execCount".into(), sj::Value::Number((count as i64).into()));
                meta.insert("timingMs".into(), sj::Value::Number(sj::Number::from(dur_ms as u64)));
                meta.insert("error".into(), sj::Value::String(msg.clone()));
                self.nb.cells[pos] = new_cell;
                return ExecResult { cell_id, duration_ms: dur_ms, outputs: vec![], error: Some(msg) };
            }
        };
        // Cache check
        if self.settings.enable_cache && !opts.ignore_cache {
            let key = self.compute_cache_key(&c.input);
            if let Some(saved) = self.cache.get(&key) {
                for it in saved.iter().cloned() { cb(ExecEvent::Output { cell_id, item: it }); }
                let dur_ms = start.elapsed().as_millis();
                return ExecResult { cell_id, duration_ms: dur_ms, outputs: saved.clone(), error: None };
            } else if let Some(on_disk) = self.try_load_cache(&key) {
                self.cache.insert(key.clone(), on_disk.clone());
                for it in on_disk.iter().cloned() { cb(ExecEvent::Output { cell_id, item: it }); }
                let dur_ms = start.elapsed().as_millis();
                return ExecResult { cell_id, duration_ms: dur_ms, outputs: on_disk, error: None };
            }
        }
        let mut outputs: Vec<DisplayItem> = Vec::new();
        let mut err: Option<String> = None;
        for expr in exprs {
            let val = self.eval_in_scope(expr);
            let mut emitted_rich = false;
            if let Value::String(s) = &val {
                if let Some((mime, data)) = Self::data_url_mime_and_data(s) {
                    let it = DisplayItem { mime, data };
                    cb(ExecEvent::Output { cell_id, item: it.clone() });
                    outputs.push(it);
                    emitted_rich = true;
                }
            }
            if !emitted_rich {
                let t = DisplayItem::text(format_value(&val));
                cb(ExecEvent::Output { cell_id, item: t.clone() });
                outputs.push(t);
            }
            let mj = DisplayItem::lyra_value_json(&val);
            cb(ExecEvent::Output { cell_id, item: mj.clone() });
            outputs.push(mj);
            if self.settings.capture_explain {
                let expl = Value::Expr { head: Box::new(Value::Symbol("Explain".into())), args: vec![val.clone()] };
                let eout = self.eval_in_scope(expl);
                let it = DisplayItem { mime: "application/lyra+explain".into(), data: serde_json::to_string(&eout).unwrap_or_default() };
                cb(ExecEvent::Output { cell_id, item: it.clone() });
                outputs.push(it);
            }
        }
        let dur_ms = start.elapsed().as_millis();
        // Update cell outputs + metadata in notebook
        let mut new_cell = c.clone();
        new_cell.output = outputs
            .iter()
            .map(|it| nb::schema::DisplayData { mime: it.mime.clone(), data: it.data.clone(), schema: None, meta: None })
            .collect();
        let meta = &mut new_cell.meta;
        let count = meta.get("execCount").and_then(|v| v.as_i64()).unwrap_or(0) + 1;
        meta.insert("execCount".into(), sj::Value::Number((count as i64).into()));
        meta.insert("timingMs".into(), sj::Value::Number(sj::Number::from(dur_ms as u64)));
        if let Some(msg) = &err { meta.insert("error".into(), sj::Value::String(msg.clone())); } else { meta.remove("error"); }
        self.nb.cells[pos] = new_cell;
        // Save to cache on success
        if self.settings.enable_cache && err.is_none() {
            let key = self.compute_cache_key(&c.input);
            self.cache.insert(key.clone(), outputs.clone());
            let _ = self.try_store_cache(&key, &outputs);
        }

        ExecResult { cell_id, duration_ms: dur_ms, outputs, error: err }
    }
}

impl Session {
    fn cache_path(&self, key: &str) -> Option<std::path::PathBuf> {
        let dir = self.settings.cache_dir.as_ref()?;
        let mut p = std::path::PathBuf::from(dir);
        p.push(format!("{}.json", key));
        Some(p)
    }
    fn try_load_cache(&self, key: &str) -> Option<Vec<DisplayItem>> {
        let path = self.cache_path(key)?;
        match std::fs::read_to_string(path) {
            Ok(s) => serde_json::from_str::<Vec<DisplayItem>>(&s).ok(),
            Err(_) => None,
        }
    }
    fn try_store_cache(&self, key: &str, val: &Vec<DisplayItem>) -> Result<(), ()> {
        let path = match self.cache_path(key) { Some(p) => p, None => return Err(()) };
        if let Some(parent) = path.parent() { let _ = std::fs::create_dir_all(parent); }
        let s = match serde_json::to_string(val) { Ok(s) => s, Err(_) => return Err(()) };
        std::fs::write(path, s).map_err(|_| ())
    }
    fn data_url_mime_and_data(s: &str) -> Option<(String, String)> {
        if let Some(rest) = s.strip_prefix("data:") {
            let mut parts = rest.splitn(2, ',');
            let head = parts.next().unwrap_or("");
            // head looks like: "mime[;params]" e.g., "image/png;base64"
            let mime = head.split(';').next().unwrap_or("").trim();
            if !mime.is_empty() {
                return Some((mime.to_string(), s.to_string()));
            }
        }
        None
    }
    fn order_by_deps(&self, ids: &[Uuid]) -> Vec<Uuid> {
        use std::collections::{HashMap, HashSet, VecDeque};
        let idset: HashSet<Uuid> = ids.iter().copied().collect();
        let mut indeg: HashMap<Uuid, usize> = HashMap::new();
        let mut adj: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
        for cid in ids {
            indeg.entry(*cid).or_insert(0);
        }
        // Build graph from meta.deps UUID strings
        for c in &self.nb.cells {
            if !idset.contains(&c.id) { continue; }
            if let Some(deps) = c.meta.get("deps").and_then(|v| v.as_array()) {
                for d in deps {
                    if let Some(s) = d.as_str() {
                        if let Ok(dep_id) = Uuid::try_parse(s) {
                            if idset.contains(&dep_id) {
                                adj.entry(dep_id).or_insert_with(Vec::new).push(c.id);
                                *indeg.entry(c.id).or_insert(0) += 1;
                            }
                        }
                    }
                }
            }
        }
        let mut q: VecDeque<Uuid> = indeg.iter().filter(|(_, &deg)| deg == 0).map(|(k, _)| *k).collect();
        let mut out: Vec<Uuid> = Vec::new();
        while let Some(u) = q.pop_front() {
            out.push(u);
            if let Some(neis) = adj.get(&u) {
                for v in neis { if let Some(e) = indeg.get_mut(v) { *e -= 1; if *e == 0 { q.push_back(*v); } } }
            }
        }
        if out.len() == ids.len() { out } else { ids.to_vec() }
    }
}

impl Session {
    fn compute_cache_key(&self, input: &str) -> String {
        let kernel_ver = env!("CARGO_PKG_VERSION");
        let nb_ver = nb::schema::CURRENT_VERSION;
        let mut hasher = blake3::Hasher::new();
        hasher.update(kernel_ver.as_bytes()); hasher.update(b"|");
        hasher.update(nb_ver.as_bytes()); hasher.update(b"|");
        if let Some(s) = self.settings.cache_salt.as_ref() { hasher.update(s.as_bytes()); hasher.update(b"|"); }
        hasher.update(input.as_bytes());
        hasher.finalize().to_hex().to_string()
    }
}
