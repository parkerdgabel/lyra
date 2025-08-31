#![cfg(feature = "db_duckdb")]
use lyra_runtime::Evaluator;

fn eval_str(code: &str) -> lyra_core::value::Value {
    let mut ev = Evaluator::new();
    lyra_stdlib::register_all(&mut ev);
    let mut p = lyra_parser::Parser::from_source(code);
    let exprs = p.parse_all().unwrap();
    let mut out = lyra_core::value::Value::Symbol("Null".into());
    for e in exprs { out = ev.eval(e); }
    out
}

#[test]
fn read_arrow_lazy_explain_sql_contains_read_feather() {
    let code = r#"
ds := ReadArrow[\"/tmp/nonexistent.feather\", <|lazy->True|>];
ExplainSQL[ds]
"#;
    let v = eval_str(code);
    match v { lyra_core::value::Value::String(s) => {
        assert!(s.contains("read_feather("));
    }, other => panic!("unexpected: {}", lyra_core::pretty::format_value(&other)) }
}

