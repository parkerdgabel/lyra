use lyra_parser::Parser;
use lyra_runtime::Evaluator;
use lyra_core::pretty::format_value;
use lyra_core::value::Value;

fn eval_one(src: &str) -> String {
    let mut p = Parser::from_source(src);
    let vals = p.parse_all().expect("parse");
    let mut ev = Evaluator::new();
    format_value(&ev.eval(vals.into_iter().last().unwrap()))
}

#[test]
fn future_await_simple() {
    let s = eval_one("Await[Future[Plus[1,2]]]" );
    assert_eq!(s, "3");
}

#[test]
fn parallel_map_squares() {
    let s = eval_one("ParallelMap[(x)=>Times[x,x], {1,2,3,4}]" );
    assert_eq!(s, "{1, 4, 9, 16}");
}

#[test]
fn explain_and_schema_exist() {
    let mut ev = Evaluator::new();
    let plus = Value::expr(Value::Symbol("Plus".into()), vec![Value::Integer(1), Value::Integer(2)]);
    let explain = Value::expr(Value::Symbol("Explain".into()), vec![plus]);
    let out1 = ev.eval(explain);
    let s1 = format_value(&out1);
    assert!(s1.contains("Explain[") == false);

    let assoc = Value::assoc(vec![("a", Value::Integer(1))]);
    let schema = Value::expr(Value::Symbol("Schema".into()), vec![assoc]);
    let out2 = ev.eval(schema);
    let s2 = format_value(&out2);
    assert!(s2.contains("\"name\""));
}
