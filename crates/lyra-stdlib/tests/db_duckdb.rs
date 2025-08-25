#![cfg(feature = "db_duckdb")]
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;
use lyra_core::value::Value;

fn eval_str(ev: &mut Evaluator, src: &str) -> Value {
    let mut p = lyra_parser::Parser::from_source(src);
    let exprs = p.parse_all().unwrap();
    let mut last = Value::Symbol("Null".into());
    for e in exprs { last = ev.eval(e); }
    last
}

#[test]
fn duckdb_join_and_list_tables() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let conn = eval_str(&mut ev, "Connect[\"duckdb::memory:\"]");
    // Create tables
    let _ = eval_str(&mut ev, &format!("Exec[{}, \"CREATE TABLE a(id INTEGER, x INTEGER); CREATE TABLE b(id INTEGER, y INTEGER)\"]", lyra_core::pretty::format_value(&conn)));
    // Insert rows
    let rows_a = Value::List(vec![
        Value::assoc(vec![("id", Value::Integer(1)), ("x", Value::Integer(10))]),
        Value::assoc(vec![("id", Value::Integer(2)), ("x", Value::Integer(20))]),
    ]);
    let rows_b = Value::List(vec![
        Value::assoc(vec![("id", Value::Integer(1)), ("y", Value::Integer(7))]),
    ]);
    let _ = ev.eval(Value::expr(Value::Symbol("InsertRows".into()), vec![conn.clone(), Value::String("a".into()), rows_a]));
    let _ = ev.eval(Value::expr(Value::Symbol("InsertRows".into()), vec![conn.clone(), Value::String("b".into()), rows_b]));
    // Join pushdown
    let join_ds = eval_str(&mut ev, &format!("Join[Table[{}, \"a\"], Table[{}, \"b\"], {{\"id\"}}, <|How->\"inner\"|>]", lyra_core::pretty::format_value(&conn), lyra_core::pretty::format_value(&conn)));
    let ex = eval_str(&mut ev, &format!("ExplainSQL[{}]", lyra_core::pretty::format_value(&join_ds)));
    let s = lyra_core::pretty::format_value(&ex);
    assert!(s.to_lowercase().contains("join"));
    // Execute and verify row with id=1 present
    let mat = eval_str(&mut ev, &format!("Collect[{}]", lyra_core::pretty::format_value(&join_ds)));
    let txt = lyra_core::pretty::format_value(&mat);
    assert!(txt.contains("\"id\" -> 1"));
    // ListTables should include a and b
    let lt = eval_str(&mut ev, &format!("ListTables[{}]", lyra_core::pretty::format_value(&conn)));
    let lt_s = lyra_core::pretty::format_value(&lt).to_lowercase();
    assert!(lt_s.contains("a") && lt_s.contains("b"));
}

