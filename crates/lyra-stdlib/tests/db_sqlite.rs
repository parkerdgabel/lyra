#![cfg(feature = "db_sqlite")]
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
fn sqlite_roundtrip_and_explain() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let conn = eval_str(&mut ev, "Connect[\"sqlite::memory:\"]");
    // Create table
    let _ = eval_str(&mut ev, &format!("Exec[{}, \"CREATE TABLE t(id INTEGER, v INTEGER)\"]", lyra_core::pretty::format_value(&conn)));
    // Insert rows
    let rows = Value::List(vec![
        Value::assoc(vec![("id", Value::Integer(1)), ("v", Value::Integer(10))]),
        Value::assoc(vec![("id", Value::Integer(2)), ("v", Value::Integer(5))]),
        Value::assoc(vec![("id", Value::Integer(1)), ("v", Value::Integer(7))]),
    ]);
    let _ = ev.eval(Value::expr(Value::Symbol("InsertRows".into()), vec![conn.clone(), Value::String("t".into()), rows]));
    // Query with Filter pushdown
    let q = eval_str(&mut ev, &format!("Table[{}, \"t\"] |> FilterRows[(row)=>Greater[Part[row,\"v\"], 6], #] &", lyra_core::pretty::format_value(&conn)));
    let ex = eval_str(&mut ev, &format!("ExplainSQL[{}]", lyra_core::pretty::format_value(&q)));
    let s = lyra_core::pretty::format_value(&ex);
    assert!(s.contains("WHERE") && s.contains("v > 6"));
    // GroupBy/Agg pushdown with multiple aggs
    let g = eval_str(&mut ev, &format!("Agg[GroupBy[Table[{}, \"t\"], {{\"id\"}}], <| \"cnt\"->Count[], \"sumv\"->Sum[col[\"v\"]], \"avgv\"->Avg[col[\"v\"]], \"minv\"->Min[col[\"v\"]], \"maxv\"->Max[col[\"v\"]] |> ]", lyra_core::pretty::format_value(&conn)));
    let ex2 = eval_str(&mut ev, &format!("ExplainSQL[{}]", lyra_core::pretty::format_value(&g)));
    let s2 = lyra_core::pretty::format_value(&ex2);
    assert!(s2.contains("COUNT(*) AS cnt"));
    assert!(s2.contains("SUM(v) AS sumv"));
    assert!(s2.contains("AVG(v) AS avgv"));
    assert!(s2.contains("MIN(v) AS minv"));
    assert!(s2.contains("MAX(v) AS maxv"));
    assert!(s2.contains("GROUP BY id"));
    // Execute and check results
    let res = eval_str(&mut ev, &format!("Collect[{}]", lyra_core::pretty::format_value(&g)));
    let txt = lyra_core::pretty::format_value(&res);
    assert!(txt.contains("\"id\" -> 1") && txt.contains("\"cnt\" -> 2"));

    // Distinct on subset (keys)
    let d = eval_str(&mut ev, &format!("Distinct[Table[{}, \"t\"], {{\"id\"}}]", lyra_core::pretty::format_value(&conn)));
    let d_ex = eval_str(&mut ev, &format!("ExplainSQL[{}]", lyra_core::pretty::format_value(&d)));
    let d_s = lyra_core::pretty::format_value(&d_ex);
    assert!(d_s.to_lowercase().contains("select distinct id from"));

    // DistinctOn selects first/last by order
    let do_first = eval_str(&mut ev, &format!(
        "Collect[DistinctOn[Table[{}, \"t\"], {{\"id\"}}, <|OrderBy->{{\"v\"->\"asc\"}}, Keep->\"first\"|>]]",
        lyra_core::pretty::format_value(&conn)));
    let do_first_s = lyra_core::pretty::format_value(&do_first);
    assert!(do_first_s.contains("\"id\" -> 1") && do_first_s.contains("\"v\" -> 7"));
    let do_last = eval_str(&mut ev, &format!(
        "Collect[DistinctOn[Table[{}, \"t\"], {{\"id\"}}, <|OrderBy->{{\"v\"->\"asc\"}}, Keep->\"last\"|>]]",
        lyra_core::pretty::format_value(&conn)));
    let do_last_s = lyra_core::pretty::format_value(&do_last);
    assert!(do_last_s.contains("\"id\" -> 1") && do_last_s.contains("\"v\" -> 10"));
}
