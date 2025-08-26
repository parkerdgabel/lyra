use crate::register_if;
use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
#[cfg(feature = "text_index")]
use rusqlite::{params, Connection};

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

pub fn register_text_index(ev: &mut Evaluator) {
    ev.register("IndexCreate", index_create as NativeFn, Attributes::empty());
    ev.register("IndexAdd", index_add as NativeFn, Attributes::empty());
    ev.register("IndexSearch", index_search as NativeFn, Attributes::empty());
    ev.register("IndexInfo", index_info as NativeFn, Attributes::empty());
}

pub fn register_text_index_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    register_if(ev, pred, "IndexCreate", index_create as NativeFn, Attributes::empty());
    register_if(ev, pred, "IndexAdd", index_add as NativeFn, Attributes::empty());
    register_if(ev, pred, "IndexSearch", index_search as NativeFn, Attributes::empty());
    register_if(ev, pred, "IndexInfo", index_info as NativeFn, Attributes::empty());
}

fn index_create(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // IndexCreate(path) -> { indexPath }
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("IndexCreate".into())), args };
    }
    let path = match &args[0] {
        Value::String(s) => s.clone(),
        v => format!("{}", lyra_core::pretty::format_value(v)),
    };
    #[cfg(feature = "text_index")]
    {
        // Treat `path` as a SQLite file path
        if let Some(dir) = std::path::Path::new(&path).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        match Connection::open(&path) {
            Ok(conn) => {
                let _ = conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(id UNINDEXED, body);",
                    [],
                );
                Value::Assoc([("indexPath".into(), Value::String(path))].into_iter().collect())
            }
            Err(e) => super::text::failure("Index::create", &e.to_string()),
        }
    }
    #[cfg(not(feature = "text_index"))]
    {
        super::text::failure("Index::create", "feature 'text_index' not enabled")
    }
}

fn index_add(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // IndexAdd(indexPath, docs:[{id, body}]) -> { added }
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("IndexAdd".into())), args };
    }
    let path = match &args[0] {
        Value::String(s) => s.clone(),
        v => format!("{}", lyra_core::pretty::format_value(v)),
    };
    let docs = match &args[1] {
        Value::List(v) => v.clone(),
        _ => return super::text::failure("Index::add", "docs must be a list"),
    };
    #[cfg(feature = "text_index")]
    {
        let mut conn = match Connection::open(&path) {
            Ok(c) => c,
            Err(e) => return super::text::failure("Index::open", &e.to_string()),
        };
        let tx = match conn.transaction() {
            Ok(t) => t,
            Err(e) => return super::text::failure("Index::tx", &e.to_string()),
        };
        let mut added = 0i64;
        for d in docs {
            if let Value::Assoc(m) = d {
                let id = m
                    .get("id")
                    .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
                    .unwrap_or_default();
                let body = m
                    .get("body")
                    .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
                    .unwrap_or_default();
                let _ = tx.execute("DELETE FROM docs WHERE id=?1", params![id]);
                if let Err(e) =
                    tx.execute("INSERT INTO docs(id, body) VALUES (?1, ?2)", params![id, body])
                {
                    return super::text::failure("Index::insert", &e.to_string());
                }
                added += 1;
            }
        }
        if let Err(e) = tx.commit() {
            return super::text::failure("Index::commit", &e.to_string());
        }
        Value::Assoc([("added".into(), Value::Integer(added))].into_iter().collect())
    }
    #[cfg(not(feature = "text_index"))]
    {
        super::text::failure("Index::add", "feature 'text_index' not enabled")
    }
}

fn index_search(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // IndexSearch(indexPath, q) -> { hits:[{id, score, body}], total }
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("IndexSearch".into())), args };
    }
    let path = match &args[0] {
        Value::String(s) => s.clone(),
        v => format!("{}", lyra_core::pretty::format_value(v)),
    };
    let q = match &args[1] {
        Value::String(s) => s.clone(),
        v => format!("{}", lyra_core::pretty::format_value(v)),
    };
    #[cfg(feature = "text_index")]
    {
        let conn = match Connection::open(&path) {
            Ok(c) => c,
            Err(e) => return super::text::failure("Index::open", &e.to_string()),
        };
        let mut stmt = match conn.prepare("SELECT id, body FROM docs WHERE docs MATCH ?1 LIMIT 20")
        {
            Ok(s) => s,
            Err(e) => return super::text::failure("Index::query", &e.to_string()),
        };
        let rows = match stmt.query_map(params![q], |row| {
            let id: String = row.get(0)?;
            let body: String = row.get(1)?;
            Ok((id, body))
        }) {
            Ok(r) => r,
            Err(e) => return super::text::failure("Index::search", &e.to_string()),
        };
        let mut hits: Vec<Value> = Vec::new();
        for r in rows {
            if let Ok((id, body)) = r {
                hits.push(Value::Assoc(
                    [("id".into(), Value::String(id)), ("body".into(), Value::String(body))]
                        .into_iter()
                        .collect(),
                ));
            }
        }
        Value::Assoc(
            [("hits".into(), Value::List(hits)), ("total".into(), Value::Integer(-1))]
                .into_iter()
                .collect(),
        )
    }
    #[cfg(not(feature = "text_index"))]
    {
        super::text::failure("Index::search", "feature 'text_index' not enabled")
    }
}

fn index_info(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("IndexInfo".into())), args };
    }
    let path = match &args[0] {
        Value::String(s) => s.clone(),
        v => format!("{}", lyra_core::pretty::format_value(v)),
    };
    #[cfg(feature = "text_index")]
    {
        match Connection::open(&path) {
            Ok(conn) => {
                let total: i64 =
                    conn.query_row("SELECT count(*) FROM docs", [], |r| r.get(0)).unwrap_or(0);
                Value::Assoc(
                    [
                        ("indexPath".into(), Value::String(path)),
                        ("numDocs".into(), Value::Integer(total)),
                    ]
                    .into_iter()
                    .collect(),
                )
            }
            Err(e) => super::text::failure("Index::open", &e.to_string()),
        }
    }
    #[cfg(not(feature = "text_index"))]
    {
        super::text::failure("Index::info", "feature 'text_index' not enabled")
    }
}
