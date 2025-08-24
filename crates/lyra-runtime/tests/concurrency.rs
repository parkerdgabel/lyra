use lyra_parser::Parser;
use lyra_runtime::Evaluator;
use lyra_core::pretty::format_value;
use lyra_core::value::Value;
use lyra_stdlib as stdlib;
use lyra_runtime::set_default_registrar;
use std::time::Instant;

fn eval_one(src: &str) -> String {
    let mut p = Parser::from_source(src);
    let vals = p.parse_all().expect("parse");
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
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
    stdlib::register_all(&mut ev);
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

#[test]
fn map_async_and_gather() {
    let s = eval_one("Gather[MapAsync[(x)=>Times[x,x], {1,2,3}]]");
    assert_eq!(s, "{1, 4, 9}");
}

#[test]
fn map_async_nested_and_partial_failures() {
    // nested structure
    let s1 = eval_one("Gather[MapAsync[EvenQ, {{1,2}, {3,4}}]]");
    assert_eq!(s1, "{{False, True}, {False, True}}" );
    // partial failures using Fail builtin
    let s2 = eval_one("Gather[MapAsync[(x)=>If[EvenQ[x], Fail[\"oops\"], Times[x,x]], {1,2,3}]]");
    assert!(s2.contains("\"oops\""));
}

#[test]
fn cancel_future_and_await_failure() {
    use lyra_parser::Parser;
    use lyra_runtime::Evaluator;
    use lyra_core::pretty::format_value;
    // Ensure spawned evaluators inherit stdlib registrations
    set_default_registrar(stdlib::register_all);
    // Build one evaluator for multiple statements
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let mut p = Parser::from_source("f = Future[BusyWait[50]]; Cancel[f]; Await[f]");
    let vals = p.parse_all().expect("parse");
    let mut last = lyra_core::value::Value::Symbol("Null".into());
    for v in vals { last = ev.eval(v); }
    let s = format_value(&last);
    assert!(s.contains("Cancel::abort"), "Got: {}", s);
}

#[test]
fn await_unknown_future_failure() {
    let s = eval_one("Await[999999]");
    assert!(s.contains("Await::invfuture"));
}

#[test]
fn parallel_table_basic() {
    let s = eval_one("ParallelTable[Times[i,i], {i, 1, 5}]" );
    assert_eq!(s, "{1, 4, 9, 16, 25}");
}

#[test]
fn scope_time_budget_exceeded() {
    let s = eval_one("Scope[<|TimeBudgetMs->10|>, BusyWait[50]]");
    assert!(s.contains("TimeBudget::exceeded"), "Got: {}", s);
}

#[test]
fn scope_max_threads_limits_parallel_table() {
    // Using BusyWait[20] ~ 100ms per item; with MaxThreads->2 over 6 items => ~3 batches => >= 200ms
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let mut p = Parser::from_source("Scope[<|MaxThreads->2|>, ParallelTable[BusyWait[20], {i,1,6}]]");
    let vals = p.parse_all().expect("parse");
    let expr = vals.into_iter().last().unwrap();
    let start = Instant::now();
    let _out = ev.eval(expr);
    let elapsed = start.elapsed();
    assert!(elapsed.as_millis() >= 150, "Too fast: {:?}", elapsed);
}
