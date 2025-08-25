use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_runtime::attrs::Attributes;
use std::collections::HashMap;
use std::sync::{OnceLock, Mutex};
#[cfg(feature = "db_sqlite")] use base64;
#[cfg(feature = "db_sqlite")] use base64::Engine;

// Database connection abstraction (initial scaffolding)
// We start with a Mock connector that stores in-memory tables for dev/testing.

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

#[derive(Clone)]
pub enum ConnectorKind {
    Mock,
    #[cfg(feature = "db_sqlite")] Sqlite,
    #[cfg(feature = "db_duckdb")] DuckDb,
}

#[derive(Clone)]
struct ConnectionState {
    dsn: String,
    kind: ConnectorKind,
    mock_tables: HashMap<String, Vec<Value>>, // table -> Vec<Assoc rows>
    #[cfg(feature = "db_sqlite")] sqlite_conn: Option<std::sync::Arc<std::sync::Mutex<rusqlite::Connection>>>,
    #[cfg(feature = "db_duckdb")] duckdb_conn: Option<std::sync::Arc<std::sync::Mutex<duckdb::Connection>>>,
    in_tx: bool,
}

static CONN_REG: OnceLock<Mutex<HashMap<i64, ConnectionState>>> = OnceLock::new();
static NEXT_CONN_ID: OnceLock<std::sync::atomic::AtomicI64> = OnceLock::new();

fn conn_reg() -> &'static Mutex<HashMap<i64, ConnectionState>> { CONN_REG.get_or_init(|| Mutex::new(HashMap::new())) }
fn next_conn_id() -> i64 { let a = NEXT_CONN_ID.get_or_init(|| std::sync::atomic::AtomicI64::new(1)); a.fetch_add(1, std::sync::atomic::Ordering::Relaxed) }

fn conn_handle(id: i64) -> Value {
    Value::Assoc(HashMap::from([
        ("__type".to_string(), Value::String("Connection".into())),
        ("id".to_string(), Value::Integer(id)),
    ]))
}

fn get_conn(v: &Value) -> Option<i64> {
    if let Value::Assoc(m) = v {
        if matches!(m.get("__type"), Some(Value::String(s)) if s=="Connection") {
            if let Some(Value::Integer(id)) = m.get("id") { return Some(*id); }
        }
    }
    None
}

// Public helper used by dataset.rs to fetch rows for DbTable on Mock connections
pub fn fetch_table_rows(conn_id: i64, table: &str) -> Option<Vec<Value>> {
    let reg = conn_reg().lock().unwrap();
    let st = reg.get(&conn_id)?;
    match st.kind.clone() {
        ConnectorKind::Mock => st.mock_tables.get(table).cloned(),
        #[cfg(feature = "db_sqlite")] ConnectorKind::Sqlite => {
            let dsn = st.dsn.clone();
            drop(reg);
            fetch_sqlite_rows(&dsn, &format!("SELECT * FROM {}", table)).ok()
        }
        #[cfg(feature = "db_duckdb")] ConnectorKind::DuckDb => {
            let dsn = st.dsn.clone();
            drop(reg);
            fetch_duckdb_rows(&dsn, &format!("SELECT * FROM {}", table)).ok()
        }
    }
}

fn connect(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Connect".into())), args } }
    // Accept DSN string or options assoc with DSN key
    let (dsn, kind) = match ev.eval(args[0].clone()) {
        Value::String(s) | Value::Symbol(s) => {
            let kind =
                if s.starts_with("mock:") || s.starts_with("mock://") { ConnectorKind::Mock }
                else if s.starts_with("sqlite:") || s.starts_with("sqlite://") { #[cfg(feature = "db_sqlite")] { ConnectorKind::Sqlite } #[cfg(not(feature = "db_sqlite"))] { ConnectorKind::Mock } }
                else if s.starts_with("duckdb:") || s.starts_with("duckdb://") { #[cfg(feature = "db_duckdb")] { ConnectorKind::DuckDb } #[cfg(not(feature = "db_duckdb"))] { ConnectorKind::Mock } }
                else { ConnectorKind::Mock };
            (s, kind)
        }
        Value::Assoc(m) => {
            let dsn = match m.get("DSN") { Some(Value::String(s))|Some(Value::Symbol(s)) => s.clone(), _ => "mock://".to_string() };
            let kind =
                if dsn.starts_with("mock:") || dsn.starts_with("mock://") { ConnectorKind::Mock }
                else if dsn.starts_with("sqlite:") || dsn.starts_with("sqlite://") { #[cfg(feature = "db_sqlite")] { ConnectorKind::Sqlite } #[cfg(not(feature = "db_sqlite"))] { ConnectorKind::Mock } }
                else if dsn.starts_with("duckdb:") || dsn.starts_with("duckdb://") { #[cfg(feature = "db_duckdb")] { ConnectorKind::DuckDb } #[cfg(not(feature = "db_duckdb"))] { ConnectorKind::Mock } }
                else { ConnectorKind::Mock };
            (dsn, kind)
        }
        other => return Value::Expr { head: Box::new(Value::Symbol("Connect".into())), args: vec![other] },
    };
    let id = next_conn_id();
    // Initialize persistent connection for SQL engines
    #[cfg(feature = "db_sqlite")] let mut sqlite_conn: Option<std::sync::Arc<std::sync::Mutex<rusqlite::Connection>>> = None;
    #[cfg(feature = "db_sqlite")] {
        if matches!(kind, ConnectorKind::Sqlite) { if let Ok(c) = sqlite_open(&dsn) { sqlite_conn = Some(std::sync::Arc::new(std::sync::Mutex::new(c))); } }
    }
    #[cfg(feature = "db_duckdb")] let mut duckdb_conn: Option<std::sync::Arc<std::sync::Mutex<duckdb::Connection>>> = None;
    #[cfg(feature = "db_duckdb")] {
        if matches!(kind, ConnectorKind::DuckDb) { if let Ok(c) = duckdb_open(&dsn) { duckdb_conn = Some(std::sync::Arc::new(std::sync::Mutex::new(c))); } }
    }
    conn_reg().lock().unwrap().insert(id, ConnectionState { dsn, kind, mock_tables: HashMap::new(), #[cfg(feature = "db_sqlite")] sqlite_conn, #[cfg(feature = "db_duckdb")] duckdb_conn, in_tx: false });
    conn_handle(id)
}

fn disconnect(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Disconnect".into())), args } }
    let id = match get_conn(&args[0]) { Some(id)=>id, None => return Value::Expr { head: Box::new(Value::Symbol("Disconnect".into())), args } };
    let mut reg = conn_reg().lock().unwrap();
    reg.remove(&id);
    Value::Symbol("Null".into())
}

fn ping(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Ping".into())), args } }
    let id = match get_conn(&args[0]) { Some(_)=>true, None=>false };
    Value::Boolean(id)
}

fn list_tables(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("ListTables".into())), args } }
    let id = match get_conn(&args[0]) { Some(id)=>id, None => return Value::Expr { head: Box::new(Value::Symbol("ListTables".into())), args } };
    let reg = conn_reg().lock().unwrap();
    let st = match reg.get(&id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("ListTables".into())), args } };
    drop(reg);
    match st.kind {
        ConnectorKind::Mock => Value::List(st.mock_tables.keys().cloned().map(Value::String).collect()),
        #[cfg(feature = "db_sqlite")] ConnectorKind::Sqlite => {
            match list_sqlite_tables(&st.dsn) { Ok(names)=> Value::List(names.into_iter().map(Value::String).collect()), Err(_)=> Value::List(vec![]) }
        }
        #[cfg(feature = "db_duckdb")] ConnectorKind::DuckDb => {
            match list_duckdb_tables(&st.dsn) { Ok(names)=> Value::List(names.into_iter().map(Value::String).collect()), Err(_)=> Value::List(vec![]) }
        }
    }
}

// Mock-only: register an in-memory table
fn register_table(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=3 { return Value::Expr { head: Box::new(Value::Symbol("RegisterTable".into())), args } }
    let id = match get_conn(&args[0]) { Some(id)=>id, None => return Value::Expr { head: Box::new(Value::Symbol("RegisterTable".into())), args } };
    let table = match ev.eval(args[1].clone()) { Value::String(s)|Value::Symbol(s)=>s, other=> return Value::Expr { head: Box::new(Value::Symbol("RegisterTable".into())), args: vec![args[0].clone(), other, args[2].clone()] } };
    let rows = match ev.eval(args[2].clone()) { Value::List(vs)=>vs, other=> return Value::Expr { head: Box::new(Value::Symbol("RegisterTable".into())), args: vec![args[0].clone(), Value::String(table), other] } };
    let mut reg = conn_reg().lock().unwrap();
    let st = match reg.get_mut(&id) { Some(s)=>s, None=> return Value::Expr { head: Box::new(Value::Symbol("RegisterTable".into())), args } };
    st.mock_tables.insert(table, rows);
    Value::Boolean(true)
}

// Table[conn, "schema.table"] -> returns a Dataset that references a DB table (DbTable plan)
fn table_to_dataset(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("Table".into())), args } }
    let conn_id = match get_conn(&args[0]) { Some(id)=>id, None => return Value::Expr { head: Box::new(Value::Symbol("Table".into())), args } };
    let name = match ev.eval(args[1].clone()) { Value::String(s)|Value::Symbol(s)=>s, other=> return Value::Expr { head: Box::new(Value::Symbol("Table".into())), args: vec![args[0].clone(), other] } };
    // Delegate to dataset module via an escape hatch: build a Dataset from a DbTable plan using a private symbol call
    // We construct Expr: __DatasetFromDbTable[conn_id, name]
    let call = Value::Expr { head: Box::new(Value::Symbol("__DatasetFromDbTable".into())), args: vec![Value::Integer(conn_id), Value::String(name)] };
    ev.eval(call)
}

// SQL[conn, "select ...", params?] -> returns a Dataset (rows) for SELECT; other statements return affected rows count
fn sql_query(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("SQL".into())), args } }
    let conn_id = match get_conn(&args[0]) { Some(id)=>id, None => return Value::Expr { head: Box::new(Value::Symbol("SQL".into())), args } };
    let mut sql = match ev.eval(args[1].clone()) { Value::String(s)|Value::Symbol(s)=>s, other=> return Value::Expr { head: Box::new(Value::Symbol("SQL".into())), args: vec![args[0].clone(), other] } };
    let params_map = if args.len()>=3 { match ev.eval(args[2].clone()) { Value::Assoc(m)=> Some(m), _=> None } } else { None };
    // For Mock connector, support a trivial form: "SELECT * FROM table" only
    let reg = conn_reg().lock().unwrap();
    let st = match reg.get(&conn_id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("SQL".into())), args } };
    drop(reg);
    match st.kind {
        ConnectorKind::Mock => {
            let mut parts = sql.trim().split_whitespace().collect::<Vec<_>>();
            if parts.len()>=4 && parts[0].eq_ignore_ascii_case("select") && parts[1]=="*" && parts[2].eq_ignore_ascii_case("from") {
                let table = parts[3].trim_matches('"').to_string();
                let rows = fetch_table_rows(conn_id, &table).unwrap_or_default();
                // Turn rows into a Dataset via DatasetFromRows
                return ev.eval(Value::Expr { head: Box::new(Value::Symbol("DatasetFromRows".into())), args: vec![Value::List(rows)] });
            }
            // Not supported by mock
            Value::Expr { head: Box::new(Value::Symbol("SQL".into())), args }
        }
        #[cfg(feature = "db_sqlite")] ConnectorKind::Sqlite => {
            // Only support SELECT queries for now; detect by leading keyword
            if sql.trim().to_ascii_lowercase().starts_with("select ") {
                if let Some(pm) = &params_map {
                    match fetch_sqlite_rows_prepared(&st.dsn, &sql, pm) {
                        Ok(rows) => return ev.eval(Value::Expr { head: Box::new(Value::Symbol("DatasetFromRows".into())), args: vec![Value::List(rows)] }),
                        Err(_) => {}
                    }
                }
                match fetch_sqlite_rows(&st.dsn, &sql) {
                    Ok(rows) => return ev.eval(Value::Expr { head: Box::new(Value::Symbol("DatasetFromRows".into())), args: vec![Value::List(rows)] }),
                    Err(_) => {}
                }
            }
            Value::Expr { head: Box::new(Value::Symbol("SQL".into())), args }
        }
        #[cfg(feature = "db_duckdb")] ConnectorKind::DuckDb => {
            if sql.trim().to_ascii_lowercase().starts_with("select ") {
                let sql = if let Some(pm) = &params_map { substitute_params(&sql, pm) } else { sql };
                match fetch_duckdb_rows(&st.dsn, &sql) {
                    Ok(rows) => return ev.eval(Value::Expr { head: Box::new(Value::Symbol("DatasetFromRows".into())), args: vec![Value::List(rows)] }),
                    Err(_) => {}
                }
            }
            Value::Expr { head: Box::new(Value::Symbol("SQL".into())), args }
        }
    }
}

// ---------- SQL parameter substitution ----------
fn substitute_params(sql: &str, params: &HashMap<String, Value>) -> String {
    let mut out = String::with_capacity(sql.len()+16);
    let bytes = sql.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'$' {
            let start = i+1;
            let mut j = start;
            while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j]==b'_') { j += 1; }
            if j > start {
                let key = &sql[start..j];
                if let Some(v) = params.get(key) {
                    out.push_str(&value_to_sql_literal(v));
                    i = j; continue;
                }
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

fn value_to_sql_literal(v: &Value) -> String {
    match v {
        Value::Integer(n) => n.to_string(),
        Value::Real(f) => f.to_string(),
        Value::Boolean(b) => if *b { "TRUE".into() } else { "FALSE".into() },
        Value::String(s) => format!("'{}'", s.replace("'", "''")),
        Value::Symbol(s) if s=="Null" => "NULL".into(),
        _ => format!("'{}'", lyra_core::pretty::format_value(v).replace("'", "''")),
    }
}

// ---------- Dataset pushdown execution helper ----------
fn sql_to_rows(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("__SQLToRows".into())), args } }
    let conn_id = match &args[0] { Value::Integer(id)=>*id, _ => return Value::Expr { head: Box::new(Value::Symbol("__SQLToRows".into())), args } };
    let sql = match &args[1] { Value::String(s)|Value::Symbol(s)=>s.clone(), _ => return Value::Expr { head: Box::new(Value::Symbol("__SQLToRows".into())), args } };
    let reg = conn_reg().lock().unwrap();
    let st = match reg.get(&conn_id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("__SQLToRows".into())), args } };
    drop(reg);
    let rows = match st.kind {
        ConnectorKind::Mock => {
            let mut parts = sql.trim().split_whitespace().collect::<Vec<_>>();
            if parts.len()>=4 && parts[0].eq_ignore_ascii_case("select") && parts[1]=="*" && parts[2].eq_ignore_ascii_case("from") {
                let table = parts[3].trim_matches('"').to_string();
                fetch_table_rows(conn_id, &table).unwrap_or_default()
            } else { vec![] }
        }
        #[cfg(feature = "db_sqlite")] ConnectorKind::Sqlite => fetch_sqlite_rows(&st.dsn, &sql).unwrap_or_default(),
        #[cfg(feature = "db_duckdb")] ConnectorKind::DuckDb => fetch_duckdb_rows(&st.dsn, &sql).unwrap_or_default(),
    };
    Value::List(rows)
}

// ---------- Streaming cursors (simulated) ----------
#[derive(Clone)]
struct CursorState { kind: ConnectorKind, dsn: String, sql: String, offset: i64, fetch_size: i64 }

static CUR_REG: OnceLock<Mutex<HashMap<i64, CursorState>>> = OnceLock::new();
static NEXT_CUR_ID: OnceLock<std::sync::atomic::AtomicI64> = OnceLock::new();
fn cur_reg() -> &'static Mutex<HashMap<i64, CursorState>> { CUR_REG.get_or_init(|| Mutex::new(HashMap::new())) }
fn next_cur_id() -> i64 { let a = NEXT_CUR_ID.get_or_init(|| std::sync::atomic::AtomicI64::new(1)); a.fetch_add(1, std::sync::atomic::Ordering::Relaxed) }

fn cursor_handle(id: i64) -> Value {
    Value::Assoc(HashMap::from([
        ("__type".to_string(), Value::String("Cursor".into())),
        ("id".to_string(), Value::Integer(id)),
    ]))
}

fn get_cursor(v: &Value) -> Option<i64> {
    if let Value::Assoc(m) = v {
        if matches!(m.get("__type"), Some(Value::String(s)) if s=="Cursor") {
            if let Some(Value::Integer(id)) = m.get("id") { return Some(*id); }
        }
    }
    None
}

fn sql_cursor(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("SQLCursor".into())), args } }
    let conn_id = match get_conn(&args[0]) { Some(id)=>id, None => return Value::Expr { head: Box::new(Value::Symbol("SQLCursor".into())), args } };
    let mut sql = match ev.eval(args[1].clone()) { Value::String(s)|Value::Symbol(s)=>s, other=> return Value::Expr { head: Box::new(Value::Symbol("SQLCursor".into())), args: vec![args[0].clone(), other] } };
    if args.len()>=3 { if let Value::Assoc(m) = ev.eval(args[2].clone()) { sql = substitute_params(&sql, &m); } }
    let fetch_size = if args.len()>=4 { if let Value::Assoc(m) = ev.eval(args[3].clone()) { if let Some(Value::Integer(n)) = m.get("FetchSize") { *n } else { 1000 } } else { 1000 } } else { 1000 };
    let reg = conn_reg().lock().unwrap();
    let st = match reg.get(&conn_id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("SQLCursor".into())), args } };
    drop(reg);
    let id = next_cur_id();
    cur_reg().lock().unwrap().insert(id, CursorState { kind: st.kind, dsn: st.dsn, sql, offset: 0, fetch_size });
    cursor_handle(id)
}

fn fetch_cursor(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Fetch".into())), args } }
    let id = match get_cursor(&args[0]) { Some(id)=>id, None => return Value::Expr { head: Box::new(Value::Symbol("Fetch".into())), args } };
    let n = if args.len()>=2 { match ev.eval(args[1].clone()) { Value::Integer(k)=>k, _=>0 } } else { 0 };
    let mut reg = cur_reg().lock().unwrap();
    let st = match reg.get_mut(&id) { Some(s)=>s, None=> return Value::Expr { head: Box::new(Value::Symbol("Fetch".into())), args } };
    let batch = if n>0 { n } else { st.fetch_size };
    let paged_sql = format!("{} LIMIT {} OFFSET {}", st.sql, batch, st.offset);
    let rows = match st.kind {
        ConnectorKind::Mock => Vec::new(),
        #[cfg(feature = "db_sqlite")] ConnectorKind::Sqlite => fetch_sqlite_rows(&st.dsn, &paged_sql).unwrap_or_default(),
        #[cfg(feature = "db_duckdb")] ConnectorKind::DuckDb => fetch_duckdb_rows(&st.dsn, &paged_sql).unwrap_or_default(),
    };
    let fetched = rows.len() as i64;
    st.offset += fetched;
    Value::List(rows)
}

fn close_cursor(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Close".into())), args } }
    if let Some(id) = get_cursor(&args[0]) { cur_reg().lock().unwrap().remove(&id); return Value::Boolean(true); }
    Value::Expr { head: Box::new(Value::Symbol("Close".into())), args }
}

// ---------- Writes ----------
fn insert_rows(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 3 { return Value::Expr { head: Box::new(Value::Symbol("InsertRows".into())), args } }
    let conn_id = match get_conn(&args[0]) { Some(id)=>id, None=> return Value::Expr { head: Box::new(Value::Symbol("InsertRows".into())), args } };
    let table = match ev.eval(args[1].clone()) { Value::String(s)|Value::Symbol(s)=>s, other=> return Value::Expr { head: Box::new(Value::Symbol("InsertRows".into())), args: vec![args[0].clone(), other, args[2].clone()] } };
    let rows_v = ev.eval(args[2].clone());
    let rows = match &rows_v { Value::List(vs)=>vs.clone(), _=> return Value::Expr { head: Box::new(Value::Symbol("InsertRows".into())), args: vec![args[0].clone(), Value::String(table), rows_v] } };
    let reg = conn_reg().lock().unwrap();
    let st = match reg.get(&conn_id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("InsertRows".into())), args } };
    drop(reg);
    // Build column set from first row
    let mut cols: Vec<String> = Vec::new();
    for r in &rows {
        if let Value::Assoc(m) = r { for k in m.keys() { if !cols.contains(k) { cols.push(k.clone()); } } }
    }
    if cols.is_empty() { return Value::Integer(0); }
    // Build multi-values INSERT
    let mut sql = String::new();
    sql.push_str(&format!("INSERT INTO {} ({} ) VALUES ", table, cols.join(", ")));
    let mut first = true;
    let mut count = 0;
    for r in &rows {
        if let Value::Assoc(m) = r {
            if !first { sql.push_str(", "); } else { first=false; }
            let vals: Vec<String> = cols.iter().map(|c| value_to_sql_literal(m.get(c).unwrap_or(&Value::Symbol("Null".into())))).collect();
            sql.push_str(&format!("({})", vals.join(", ")));
            count += 1;
        }
    }
    match st.kind {
        ConnectorKind::Mock => { /* store in mock */
            let mut reg = conn_reg().lock().unwrap();
            if let Some(state) = reg.get_mut(&conn_id) {
                let ent = state.mock_tables.entry(table).or_default();
                ent.extend(rows.clone());
            }
            Value::Integer(count)
        }
        #[cfg(feature = "db_sqlite")] ConnectorKind::Sqlite => {
            if let Some(conn) = st.sqlite_conn.as_ref() {
                let mut guard = conn.lock().unwrap();
                if !st.in_tx { let _ = guard.execute_batch("BEGIN"); }
                let res = guard.execute_batch(&sql).map(|_| Value::Integer(count)).unwrap_or(Value::Integer(0));
                if !st.in_tx { let _ = guard.execute_batch("COMMIT"); }
                return res;
            }
            Value::Integer(0)
        }
        #[cfg(feature = "db_duckdb")] ConnectorKind::DuckDb => {
            if let Some(conn) = st.duckdb_conn.as_ref() {
                let mut guard = conn.lock().unwrap();
                if !st.in_tx { let _ = guard.execute_batch("BEGIN TRANSACTION"); }
                let res = guard.execute_batch(&sql).map(|_| Value::Integer(count)).unwrap_or(Value::Integer(0));
                if !st.in_tx { let _ = guard.execute_batch("COMMIT"); }
                return res;
            }
            Value::Integer(0)
        }
    }
}

fn upsert_rows(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // UpsertRows[conn, table, rows, <|Keys->{...}|>] via delete+insert fallback
    if args.len() < 3 { return Value::Expr { head: Box::new(Value::Symbol("UpsertRows".into())), args } }
    let conn_id = match get_conn(&args[0]) { Some(id)=>id, None=> return Value::Expr { head: Box::new(Value::Symbol("UpsertRows".into())), args } };
    let table = match ev.eval(args[1].clone()) { Value::String(s)|Value::Symbol(s)=>s, other=> return Value::Expr { head: Box::new(Value::Symbol("UpsertRows".into())), args: vec![args[0].clone(), other, args[2].clone()] } };
    let rows_v = ev.eval(args[2].clone());
    let rows = match &rows_v { Value::List(vs)=>vs.clone(), _=> return Value::Expr { head: Box::new(Value::Symbol("UpsertRows".into())), args: vec![args[0].clone(), Value::String(table), rows_v] } };
    let keys: Vec<String> = if args.len()>=4 { if let Value::Assoc(m) = ev.eval(args[3].clone()) { if let Some(Value::List(ks))=m.get("Keys") { ks.iter().filter_map(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None }).collect() } else { vec![] } } else { vec![] } } else { vec![] };
    let reg = conn_reg().lock().unwrap();
    let mut st = match reg.get(&conn_id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("UpsertRows".into())), args } };
    drop(reg);
    let mut inserted = 0;
    match st.kind {
        ConnectorKind::Mock => {
            let mut reg = conn_reg().lock().unwrap();
            if let Some(state) = reg.get_mut(&conn_id) {
                let ent = state.mock_tables.entry(table).or_default();
                for r in rows {
                    if let Value::Assoc(m) = &r {
                        if !keys.is_empty() {
                            ent.retain(|row| match row { Value::Assoc(mm)=> { !keys.iter().all(|k| mm.get(k)==m.get(k)) }, _=> true });
                        }
                        ent.push(r);
                        inserted += 1;
                    }
                }
            }
            Value::Integer(inserted)
        }
        #[cfg(feature = "db_sqlite")] ConnectorKind::Sqlite => {
            // Use generic delete+insert approach
            let mut sql = String::new();
            if !st.in_tx {
                sql.push_str("BEGIN; ");
            }
            for r in rows {
                if let Value::Assoc(m) = r {
                    if !keys.is_empty() {
                        let conds: Vec<String> = keys.iter().map(|k| format!("{} = {}", k, value_to_sql_literal(m.get(k).unwrap_or(&Value::Symbol("Null".into()))))).collect();
                        sql.push_str(&format!("DELETE FROM {} WHERE {}; ", table, conds.join(" AND ")));
                    }
                    let cols: Vec<String> = m.keys().cloned().collect();
                    let vals: Vec<String> = cols.iter().map(|c| value_to_sql_literal(m.get(c).unwrap())).collect();
                    sql.push_str(&format!("INSERT INTO {} ({}) VALUES ({}); ", table, cols.join(", "), vals.join(", "))); inserted += 1;
                }
            }
            if !st.in_tx { sql.push_str("COMMIT;"); }
            if let Some(conn) = st.sqlite_conn.as_ref() { let _ = conn.lock().unwrap().execute_batch(&sql); }
            Value::Integer(inserted)
        }
        #[cfg(feature = "db_duckdb")] ConnectorKind::DuckDb => {
            let mut sql = String::new();
            if !st.in_tx { sql.push_str("BEGIN TRANSACTION; "); }
            for r in rows {
                if let Value::Assoc(m) = r {
                    if !keys.is_empty() {
                        let conds: Vec<String> = keys.iter().map(|k| format!("{} = {}", k, value_to_sql_literal(m.get(k).unwrap_or(&Value::Symbol("Null".into()))))).collect();
                        sql.push_str(&format!("DELETE FROM {} WHERE {}; ", table, conds.join(" AND ")));
                    }
                    let cols: Vec<String> = m.keys().cloned().collect();
                    let vals: Vec<String> = cols.iter().map(|c| value_to_sql_literal(m.get(c).unwrap())).collect();
                    sql.push_str(&format!("INSERT INTO {} ({}) VALUES ({}); ", table, cols.join(", "), vals.join(", "))); inserted += 1;
                }
            }
            if !st.in_tx { sql.push_str("COMMIT;"); }
            if let Some(conn) = st.duckdb_conn.as_ref() { let _ = conn.lock().unwrap().execute_batch(&sql); }
            Value::Integer(inserted)
        }
    }
}

fn write_dataset(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // WriteDataset[ds, conn, table, <|Mode->"Append"|"Overwrite"|>]
    if args.len() < 3 { return Value::Expr { head: Box::new(Value::Symbol("WriteDataset".into())), args } }
    let ds = ev.eval(args[0].clone());
    let conn = args[1].clone();
    let table = ev.eval(args[2].clone());
    let mode = if args.len()>=4 { if let Value::Assoc(m) = ev.eval(args[3].clone()) { match m.get("Mode") { Some(Value::String(s))|Some(Value::Symbol(s)) => s.clone(), _=> "Append".into() } } else { "Append".into() } } else { "Append".into() };
    let rows = match ds { Value::Assoc(_) | Value::List(_) | Value::Expr{..} => ev.eval(Value::Expr { head: Box::new(Value::Symbol("Collect".into())), args: vec![ds] }), other => other };
    if let Value::String(tname) | Value::Symbol(tname) = table {
        if mode.eq_ignore_ascii_case("Overwrite") {
            // Best-effort truncate via DELETE
            let conn_id = match get_conn(&conn) { Some(id)=>id, None=> return Value::Expr { head: Box::new(Value::Symbol("WriteDataset".into())), args } };
            let reg = conn_reg().lock().unwrap();
            let st = match reg.get(&conn_id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("WriteDataset".into())), args } };
            drop(reg);
            match st.kind {
                ConnectorKind::Mock => {
                    let mut reg = conn_reg().lock().unwrap();
                    if let Some(state) = reg.get_mut(&conn_id) { state.mock_tables.insert(tname.clone(), Vec::new()); }
                }
                #[cfg(feature = "db_sqlite")] ConnectorKind::Sqlite => { if let Some(c) = st.sqlite_conn.as_ref() { let _ = c.lock().unwrap().execute(&format!("DELETE FROM {}", tname), []); } }
                #[cfg(feature = "db_duckdb")] ConnectorKind::DuckDb => { if let Some(c) = st.duckdb_conn.as_ref() { let _ = c.lock().unwrap().execute(&format!("DELETE FROM {}", tname), []); } }
            }
        }
        return insert_rows(ev, vec![conn, Value::String(tname), rows]);
    }
    Value::Expr { head: Box::new(Value::Symbol("WriteDataset".into())), args }
}

// ---------- Transactions ----------
fn begin_tx(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Begin".into())), args } }
    let id = match get_conn(&args[0]) { Some(id)=>id, None=> return Value::Expr { head: Box::new(Value::Symbol("Begin".into())), args } };
    let mut reg = conn_reg().lock().unwrap();
    if let Some(st) = reg.get_mut(&id) {
        match st.kind {
            ConnectorKind::Mock => { st.in_tx = true; return Value::Boolean(true); }
            #[cfg(feature = "db_sqlite")] ConnectorKind::Sqlite => { if let Some(c) = &st.sqlite_conn { let _ = c.lock().unwrap().execute_batch("BEGIN"); st.in_tx = true; return Value::Boolean(true); } }
            #[cfg(feature = "db_duckdb")] ConnectorKind::DuckDb => { if let Some(c) = &st.duckdb_conn { let _ = c.lock().unwrap().execute_batch("BEGIN TRANSACTION"); st.in_tx = true; return Value::Boolean(true); } }
        }
    }
    Value::Boolean(false)
}

fn commit_tx(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Commit".into())), args } }
    let id = match get_conn(&args[0]) { Some(id)=>id, None=> return Value::Expr { head: Box::new(Value::Symbol("Commit".into())), args } };
    let mut reg = conn_reg().lock().unwrap();
    if let Some(st) = reg.get_mut(&id) {
        match st.kind {
            ConnectorKind::Mock => { st.in_tx = false; return Value::Boolean(true); }
            #[cfg(feature = "db_sqlite")] ConnectorKind::Sqlite => { if let Some(c) = &st.sqlite_conn { let _ = c.lock().unwrap().execute_batch("COMMIT"); st.in_tx = false; return Value::Boolean(true); } }
            #[cfg(feature = "db_duckdb")] ConnectorKind::DuckDb => { if let Some(c) = &st.duckdb_conn { let _ = c.lock().unwrap().execute_batch("COMMIT"); st.in_tx = false; return Value::Boolean(true); } }
        }
    }
    Value::Boolean(false)
}

fn rollback_tx(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Rollback".into())), args } }
    let id = match get_conn(&args[0]) { Some(id)=>id, None=> return Value::Expr { head: Box::new(Value::Symbol("Rollback".into())), args } };
    let mut reg = conn_reg().lock().unwrap();
    if let Some(st) = reg.get_mut(&id) {
        match st.kind {
            ConnectorKind::Mock => { st.in_tx = false; return Value::Boolean(true); }
            #[cfg(feature = "db_sqlite")] ConnectorKind::Sqlite => { if let Some(c) = &st.sqlite_conn { let _ = c.lock().unwrap().execute_batch("ROLLBACK"); st.in_tx = false; return Value::Boolean(true); } }
            #[cfg(feature = "db_duckdb")] ConnectorKind::DuckDb => { if let Some(c) = &st.duckdb_conn { let _ = c.lock().unwrap().execute_batch("ROLLBACK"); st.in_tx = false; return Value::Boolean(true); } }
        }
    }
    Value::Boolean(false)
}

// ExplainSQL[dataset] -> String explaining pushdown and produced SQL (scaffolding)
fn explain_sql(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("ExplainSQL".into())), args } }
    // Placeholder: real implementation will walk Dataset plan and render SQL
    Value::String("ExplainSQL not yet implemented".into())
}

pub fn register_db(ev: &mut Evaluator) {
    ev.register("Connect", connect as NativeFn, Attributes::empty());
    ev.register("Disconnect", disconnect as NativeFn, Attributes::empty());
    ev.register("Ping", ping as NativeFn, Attributes::empty());
    ev.register("ListTables", list_tables as NativeFn, Attributes::empty());
    ev.register("RegisterTable", register_table as NativeFn, Attributes::empty());
    ev.register("Table", table_to_dataset as NativeFn, Attributes::empty());
    ev.register("SQL", sql_query as NativeFn, Attributes::empty());
    ev.register("Exec", exec_query as NativeFn, Attributes::empty());
    // ExplainSQL is implemented in dataset.rs to access the logical plan
    // Internal helpers
    ev.register("__SQLToRows", sql_to_rows as NativeFn, Attributes::empty());
    ev.register("SQLCursor", sql_cursor as NativeFn, Attributes::empty());
    ev.register("Fetch", fetch_cursor as NativeFn, Attributes::empty());
    ev.register("Close", close_cursor as NativeFn, Attributes::empty());
    // Writes and transactions
    ev.register("InsertRows", insert_rows as NativeFn, Attributes::empty());
    ev.register("UpsertRows", upsert_rows as NativeFn, Attributes::empty());
    ev.register("WriteDataset", write_dataset as NativeFn, Attributes::empty());
    ev.register("Begin", begin_tx as NativeFn, Attributes::empty());
    ev.register("Commit", commit_tx as NativeFn, Attributes::empty());
    ev.register("Rollback", rollback_tx as NativeFn, Attributes::empty());
}

// Exec[conn, sql, params?] -> executes non-SELECT (DDL/DML); returns Boolean success
fn exec_query(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("Exec".into())), args } }
    let conn_id = match get_conn(&args[0]) { Some(id)=>id, None => return Value::Expr { head: Box::new(Value::Symbol("Exec".into())), args } };
    let mut sql = match ev.eval(args[1].clone()) { Value::String(s)|Value::Symbol(s)=>s, other=> return Value::Expr { head: Box::new(Value::Symbol("Exec".into())), args: vec![args[0].clone(), other] } };
    let params_map = if args.len()>=3 { match ev.eval(args[2].clone()) { Value::Assoc(m)=> Some(m), _=> None } } else { None };
    let reg = conn_reg().lock().unwrap();
    let st = match reg.get(&conn_id) { Some(s)=>s.clone(), None=> return Value::Expr { head: Box::new(Value::Symbol("Exec".into())), args } };
    drop(reg);
    match st.kind {
        ConnectorKind::Mock => Value::Boolean(true),
        #[cfg(feature = "db_sqlite")] ConnectorKind::Sqlite => {
            if let Some(c) = st.sqlite_conn.as_ref() {
                if let Some(pm) = &params_map {
                    let ok = exec_sqlite_prepared_conn(&mut *c.lock().unwrap(), &sql, pm).is_ok();
                    Value::Boolean(ok)
                } else {
                    let ok = c.lock().unwrap().execute_batch(&sql).is_ok(); Value::Boolean(ok)
                }
            } else { Value::Boolean(false) }
        }
        #[cfg(feature = "db_duckdb")] ConnectorKind::DuckDb => {
            if let Some(c) = st.duckdb_conn.as_ref() {
                let sql = if let Some(pm) = &params_map { substitute_params(&sql, pm) } else { sql };
                let ok = c.lock().unwrap().execute_batch(&sql).is_ok(); Value::Boolean(ok)
            } else { Value::Boolean(false) }
        }
    }
}

// ---------- SQLite helpers (feature-gated) ----------
#[cfg(feature = "db_sqlite")]
fn sqlite_open(dsn: &str) -> rusqlite::Result<rusqlite::Connection> {
    // DSN accepted forms: sqlite::memory:, sqlite:path, sqlite://path
    if dsn == "sqlite::memory:" || dsn == "sqlite:memory" {
        rusqlite::Connection::open_in_memory()
    } else if let Some(path) = dsn.strip_prefix("sqlite://") {
        rusqlite::Connection::open(path)
    } else if let Some(path) = dsn.strip_prefix("sqlite:") {
        rusqlite::Connection::open(path)
    } else {
        rusqlite::Connection::open(dsn)
    }
}

#[cfg(feature = "db_sqlite")]
fn list_sqlite_tables(dsn: &str) -> rusqlite::Result<Vec<String>> {
    let conn = sqlite_open(dsn)?;
    let mut stmt = conn.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY 1")?;
    let iter = stmt.query_map([], |row| row.get::<usize, String>(0))?;
    let mut out = Vec::new();
    for r in iter { out.push(r?); }
    Ok(out)
}

#[cfg(feature = "db_sqlite")]
fn fetch_sqlite_rows(dsn: &str, sql: &str) -> rusqlite::Result<Vec<Value>> {
    use rusqlite::types::ValueRef;
    let conn = sqlite_open(dsn)?;
    let mut stmt = conn.prepare(sql)?;
    let col_count = stmt.column_count();
    let col_names: Vec<String> = (0..col_count).map(|i| stmt.column_name(i).unwrap_or("").to_string()).collect();
    let mut rows = stmt.query([])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        let mut m: HashMap<String, Value> = HashMap::new();
        for i in 0..col_count {
            let key = col_names.get(i).cloned().unwrap_or_else(|| format!("col{}", i+1));
            let v = match row.get_ref(i)? {
                ValueRef::Null => Value::Symbol("Null".into()),
                ValueRef::Integer(n) => Value::Integer(n),
                ValueRef::Real(f) => Value::Real(f),
                ValueRef::Text(t) => Value::String(String::from_utf8_lossy(t).to_string()),
                ValueRef::Blob(b) => Value::String(base64::engine::general_purpose::STANDARD.encode(b)),
            };
            m.insert(key, v);
        }
        out.push(Value::Assoc(m));
    }
    Ok(out)
}

#[cfg(feature = "db_sqlite")]
fn to_sqlite_value(v: &Value) -> rusqlite::types::Value {
    use rusqlite::types::Value as SV;
    match v {
        Value::Symbol(s) if s=="Null" => SV::Null,
        Value::Integer(n) => SV::Integer(*n),
        Value::Real(f) => SV::Real(*f),
        Value::Boolean(b) => SV::Integer(if *b {1} else {0}),
        Value::String(s) => SV::Text(s.clone()),
        _ => SV::Text(lyra_core::pretty::format_value(v)),
    }
}

#[cfg(feature = "db_sqlite")]
fn prepare_sql_and_params(sql: &str, params: &HashMap<String, Value>) -> (String, Vec<rusqlite::types::Value>) {
    let mut out = String::with_capacity(sql.len()+16);
    let bytes = sql.as_bytes();
    let mut i = 0;
    let mut vals: Vec<rusqlite::types::Value> = Vec::new();
    while i < bytes.len() {
        if bytes[i] == b'$' {
            let start = i+1; let mut j = start;
            while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j]==b'_') { j += 1; }
            if j > start {
                let key = &sql[start..j];
                if let Some(v) = params.get(key) {
                    out.push('?');
                    vals.push(to_sqlite_value(v));
                    i = j; continue;
                }
            }
        }
        out.push(bytes[i] as char); i += 1;
    }
    (out, vals)
}

#[cfg(feature = "db_sqlite")]
fn fetch_sqlite_rows_prepared(dsn: &str, sql: &str, params: &HashMap<String, Value>) -> rusqlite::Result<Vec<Value>> {
    use rusqlite::params_from_iter;
    let conn = sqlite_open(dsn)?;
    let (sql2, vals) = prepare_sql_and_params(sql, params);
    let mut stmt = conn.prepare(&sql2)?;
    let col_count = stmt.column_count();
    let col_names: Vec<String> = (0..col_count).map(|i| stmt.column_name(i).unwrap_or("").to_string()).collect();
    let mut rows = stmt.query(params_from_iter(vals))?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        let mut m: HashMap<String, Value> = HashMap::new();
        for i in 0..col_count {
            let key = col_names.get(i).cloned().unwrap_or_else(|| format!("col{}", i+1));
            let v = match row.get_ref(i)? {
                rusqlite::types::ValueRef::Null => Value::Symbol("Null".into()),
                rusqlite::types::ValueRef::Integer(n) => Value::Integer(n),
                rusqlite::types::ValueRef::Real(f) => Value::Real(f),
                rusqlite::types::ValueRef::Text(t) => Value::String(String::from_utf8_lossy(t).to_string()),
                rusqlite::types::ValueRef::Blob(b) => Value::String(base64::engine::general_purpose::STANDARD.encode(b)),
            };
            m.insert(key, v);
        }
        out.push(Value::Assoc(m));
    }
    Ok(out)
}

#[cfg(feature = "db_sqlite")]
fn exec_sqlite_prepared_conn(conn: &mut rusqlite::Connection, sql: &str, params: &HashMap<String, Value>) -> rusqlite::Result<()> {
    use rusqlite::params_from_iter;
    let (sql2, vals) = prepare_sql_and_params(sql, params);
    let mut stmt = conn.prepare(&sql2)?;
    let _ = stmt.execute(params_from_iter(vals))?;
    Ok(())
}

// ---------- DuckDB helpers (feature-gated) ----------
#[cfg(feature = "db_duckdb")]
fn duckdb_open(dsn: &str) -> duckdb::Result<duckdb::Connection> {
    // DSN forms: duckdb::memory:, duckdb:memory, duckdb://file
    if dsn == "duckdb::memory:" || dsn == "duckdb:memory" {
        duckdb::Connection::open_in_memory()
    } else if let Some(path) = dsn.strip_prefix("duckdb://") {
        duckdb::Connection::open(path)
    } else if let Some(path) = dsn.strip_prefix("duckdb:") {
        duckdb::Connection::open(path)
    } else {
        duckdb::Connection::open(dsn)
    }
}

#[cfg(feature = "db_duckdb")]
fn list_duckdb_tables(dsn: &str) -> duckdb::Result<Vec<String>> {
    let conn = duckdb_open(dsn)?;
    let mut stmt = conn.prepare("SHOW TABLES")?;
    let mut rows = stmt.query([])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        // DuckDB SHOW TABLES returns: schema, name, type; grab name (col 1 or 2 depending on version). Try common positions.
        let name = row.get::<usize, Option<String>>(1).ok().flatten()
            .or_else(|| row.get::<usize, Option<String>>(0).ok().flatten())
            .unwrap_or_default();
        if !name.is_empty() { out.push(name); }
    }
    Ok(out)
}

#[cfg(feature = "db_duckdb")]
fn fetch_duckdb_rows(dsn: &str, sql: &str) -> duckdb::Result<Vec<Value>> {
    let conn = duckdb_open(dsn)?;
    let mut stmt = conn.prepare(sql)?;
    let mut rows = stmt.query([])?;
    let mut out = Vec::new();
    let col_count = stmt.column_count();
    let col_names: Vec<String> = (0..col_count).map(|i| stmt.column_name(i).unwrap_or("").to_string()).collect();
    while let Some(row) = rows.next()? {
        let mut m: HashMap<String, Value> = HashMap::new();
        for i in 0..col_count {
            let key = col_names.get(i).cloned().unwrap_or_else(|| format!("col{}", i+1));
            // Try common types; fallback to string
            let v = row.get::<usize, Option<i64>>(i).ok().flatten().map(Value::Integer)
                .or_else(|| row.get::<usize, Option<f64>>(i).ok().flatten().map(Value::Real))
                .or_else(|| row.get::<usize, Option<String>>(i).ok().flatten().map(Value::String))
                .or_else(|| row.get::<usize, Option<bool>>(i).ok().flatten().map(Value::Boolean))
                .unwrap_or(Value::Symbol("Null".into()));
            m.insert(key, v);
        }
        out.push(Value::Assoc(m));
    }
    Ok(out)
}
