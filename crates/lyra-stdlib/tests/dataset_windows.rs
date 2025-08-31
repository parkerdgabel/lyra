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
fn window_row_number_and_lead() {
    let code = r#"
ds := DatasetFromRows[{<|"userId"->"u1","ts"->1,"v"->10|>,<|"userId"->"u1","ts"->2,"v"->20|>,<|"userId"->"u2","ts"->1,"v"->5|>}];
win := Window[ds, <|"partitionBy"->{"userId"}, "orderBy"->{"ts"}|>, <|"rowNum"->RowNumber[], "nextV"->Lead["v",1,0]|>];
Collect[win]
"#;
    let v = eval_str(code);
    match v { lyra_core::value::Value::List(rows) => {
        assert_eq!(rows.len(), 3);
    }, other => panic!("unexpected: {}", lyra_core::pretty::format_value(&other)) }
}
