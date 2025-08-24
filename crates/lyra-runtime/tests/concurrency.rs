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

#[test]
fn scope_max_threads_limits_parallel_map() {
    // With MaxThreads->2 over 6 items, BusyWait[20] should throttle
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let mut p = Parser::from_source("Scope[<|MaxThreads->2|>, ParallelMap[(x)=>BusyWait[20], {1,2,3,4,5,6}]]");
    let vals = p.parse_all().expect("parse");
    let expr = vals.into_iter().last().unwrap();
    let start = Instant::now();
    let _out = ev.eval(expr);
    let elapsed = start.elapsed();
    assert!(elapsed.as_millis() >= 150, "Too fast: {:?}", elapsed);
}

#[test]
fn cancel_scope_group_cancels_futures() {
    // Start a scope, launch two futures under it, cancel scope, both should cancel
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // 1) sid = StartScope[<||>]
    let mut p = Parser::from_source("sid = StartScope[<||>]");
    let vals = p.parse_all().expect("parse1");
    for v in vals { let _ = ev.eval(v); }
    // 2) InScope[sid, {Set[f, Future[BusyWait[50]]], Set[g, Future[BusyWait[50]]]}]
    let mut p2 = Parser::from_source("InScope[sid, {Set[f, Future[BusyWait[50]]], Set[g, Future[BusyWait[50]]]}]");
    let vals2 = p2.parse_all().expect("parse2");
    for v in vals2 { let _ = ev.eval(v); }
    // 3) CancelScope[sid]
    let mut p3 = Parser::from_source("CancelScope[sid]");
    let vals3 = p3.parse_all().expect("parse3");
    for v in vals3 { let _ = ev.eval(v); }
    // 4) {Await[f], Await[g]}
    let mut p4 = Parser::from_source("{Await[f], Await[g]}");
    let vals4 = p4.parse_all().expect("parse4");
    let last = vals4.into_iter().last().unwrap();
    let out = ev.eval(last);
    let s = format_value(&out);
    assert!(s.contains("Cancel::abort"), "Got: {}", s);
}

#[test]
fn parallel_map_with_options_overrides_scope() {
    // Global scope has 2 threads; per-call opts restrict to 1
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let mut p = Parser::from_source("Scope[<|MaxThreads->2|>, ParallelMap[(x)=>BusyWait[10], {1,2,3,4}, <|MaxThreads->1|>]]");
    let vals = p.parse_all().expect("parse");
    let expr = vals.into_iter().last().unwrap();
    let start = Instant::now();
    let _out = ev.eval(expr);
    let elapsed = start.elapsed();
    // With ~50ms per item (BusyWait[10]) and MaxThreads 1, total >= 150ms for 4 items (queueing effects)
    assert!(elapsed.as_millis() >= 120, "Too fast: {:?}", elapsed);
}

#[test]
fn parallel_evaluate_basic() {
    let s = eval_one("ParallelEvaluate[{Plus[1,2], Times[2,3]}]");
    assert_eq!(s, "{3, 6}");
}

#[test]
fn map_async_with_options_throttles() {
    // MapAsync spawns Futures; per-call opts restrict to 1
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let mut p = Parser::from_source("Scope[<|MaxThreads->4|>, Gather[MapAsync[(x)=>BusyWait[10], {1,2,3,4}, <|MaxThreads->1|>]]]");
    let vals = p.parse_all().expect("parse");
    let expr = vals.into_iter().last().unwrap();
    let start = Instant::now();
    let _out = ev.eval(expr);
    let elapsed = start.elapsed();
    // With ~50ms per item (BusyWait[10]) and MaxThreads 1, total >= ~120ms for 4 items including overhead
    assert!(elapsed.as_millis() >= 120, "Too fast: {:?}", elapsed);
}

#[test]
fn end_scope_cleans_registry() {
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // sid = StartScope[<||>]
    let mut p1 = Parser::from_source("sid = StartScope[<||>]");
    for v in p1.parse_all().unwrap() { let _ = ev.eval(v); }
    // First EndScope[sid] -> True
    let mut p2 = Parser::from_source("EndScope[sid]");
    let out1 = ev.eval(p2.parse_all().unwrap().into_iter().last().unwrap());
    let s1 = format_value(&out1);
    assert_eq!(s1, "True");
    // Second EndScope[sid] -> False
    let mut p3 = Parser::from_source("EndScope[sid]");
    let out2 = ev.eval(p3.parse_all().unwrap().into_iter().last().unwrap());
    let s2 = format_value(&out2);
    assert_eq!(s2, "False");
}

#[test]
fn channel_roundtrip_and_close() {
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let mut p = Parser::from_source("ch = BoundedChannel[2]; Send[ch, 1]; Send[ch, 2]; {Receive[ch], Receive[ch], CloseChannel[ch], Receive[ch]}");
    let vals = p.parse_all().expect("parse");
    let mut last = Value::Symbol("Null".into());
    for v in vals { last = ev.eval(v); }
    let s = format_value(&last);
    assert_eq!(s, "{1, 2, True, Null}");
}

#[test]
fn actor_tell_and_stop() {
    // Spin up a no-op actor and send messages, then stop
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let mut p = Parser::from_source("a = Actor[(m)=>Null]; Tell[a, 123]; StopActor[a]");
    let vals = p.parse_all().expect("parse");
    let mut last = Value::Symbol("Null".into());
    for v in vals { last = ev.eval(v); }
    let s = format_value(&last);
    assert_eq!(s, "True");
}
