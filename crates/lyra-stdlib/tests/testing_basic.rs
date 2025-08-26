#![cfg(feature = "testing")]

use lyra_runtime::eval::set_default_registrar;
use lyra_runtime::Evaluator;
use std::sync::Once;

static INIT: Once = Once::new();
fn ensure_default() {
    INIT.call_once(|| set_default_registrar(stdlib::register_all));
}
use lyra_core::pretty::format_value;
use lyra_core::value::Value;
use lyra_stdlib as stdlib;

fn run_simple_suite(ev: &mut Evaluator) -> Value {
    // Build: TestRun[TestSuite["Sample", { TestCase["adds numbers", AssertEqual[1+2, 3]], TestCase["assert true", Assert[True]] }]]
    let adds = Value::expr(
        Value::Symbol("AssertEqual".into()),
        vec![
            Value::expr(Value::Symbol("Plus".into()), vec![Value::Integer(1), Value::Integer(2)]),
            Value::Integer(3),
        ],
    );
    let ok = Value::expr(Value::Symbol("Assert".into()), vec![Value::Boolean(true)]);
    let case1 = Value::expr(
        Value::Symbol("TestCase".into()),
        vec![Value::String("adds numbers".into()), adds],
    );
    let case2 = Value::expr(
        Value::Symbol("TestCase".into()),
        vec![Value::String("assert true".into()), ok],
    );
    let suite = Value::expr(
        Value::Symbol("TestSuite".into()),
        vec![Value::String("Sample".into()), Value::list(vec![case1, case2])],
    );
    let run = Value::expr(Value::Symbol("TestRun".into()), vec![suite]);
    ev.eval(run)
}

#[test]
fn basic_testing_run() {
    let mut ev = Evaluator::new();
    ensure_default();
    stdlib::register_all(&mut ev);

    let v = run_simple_suite(&mut ev);
    let txt = format_value(&v);
    assert!(txt.contains("\"passed\" -> 2"), "expected 2 passes, got: {}", txt);
    assert!(txt.contains("\"failed\" -> 0"), "expected 0 fails, got: {}", txt);
}

#[test]
fn failing_case_is_reported() {
    let mut ev = Evaluator::new();
    ensure_default();
    stdlib::register_all(&mut ev);

    // Build: TestRun[TestSuite["S", { TestCase["oops", AssertEqual[1+2,4]] }]]
    let adds_wrong = Value::expr(
        Value::Symbol("AssertEqual".into()),
        vec![
            Value::expr(Value::Symbol("Plus".into()), vec![Value::Integer(1), Value::Integer(2)]),
            Value::Integer(4),
        ],
    );
    let case = Value::expr(
        Value::Symbol("TestCase".into()),
        vec![Value::String("oops".into()), adds_wrong],
    );
    let suite = Value::expr(
        Value::Symbol("TestSuite".into()),
        vec![Value::String("S".into()), Value::list(vec![case])],
    );
    let run = Value::expr(Value::Symbol("TestRun".into()), vec![suite]);
    let v = ev.eval(run);
    let txt = format_value(&v);
    assert!(txt.contains("\"failed\" -> 1"), "expected 1 fail, got: {}", txt);
}

#[test]
fn assert_equal_with_tolerance() {
    let mut ev = Evaluator::new();
    ensure_default();
    stdlib::register_all(&mut ev);
    // AssertEqual[1.000000001, 1.0, <|Tolerance->1e-8|>]
    let opts = Value::assoc(vec![("Tolerance", Value::Real(1e-8))]);
    let case = Value::expr(
        Value::Symbol("TestCase".into()),
        vec![
            Value::String("tol".into()),
            Value::expr(
                Value::Symbol("AssertEqual".into()),
                vec![Value::Real(1.000000001), Value::Real(1.0), opts],
            ),
        ],
    );
    let suite = Value::expr(
        Value::Symbol("TestSuite".into()),
        vec![Value::String("S".into()), Value::list(vec![case])],
    );
    let run = Value::expr(Value::Symbol("TestRun".into()), vec![suite]);
    let v = ev.eval(run);
    let txt = format_value(&v);
    assert!(txt.contains("\"passed\" -> 1"), "expected tolerance pass, got: {}", txt);
}

#[test]
fn retries_recorded() {
    let mut ev = Evaluator::new();
    ensure_default();
    stdlib::register_all(&mut ev);
    // Case fails via Fail[]; ensure retries recorded
    // TestCase["fail", Fail[], <|Retries->2|>]
    let opts = Value::assoc(vec![("Retries", Value::Integer(2))]);
    let body = Value::expr(Value::Symbol("Fail".into()), vec![]);
    let case = Value::expr(
        Value::Symbol("TestCase".into()),
        vec![Value::String("fail".into()), body, opts],
    );
    let suite = Value::expr(
        Value::Symbol("TestSuite".into()),
        vec![Value::String("S".into()), Value::list(vec![case])],
    );
    let run = Value::expr(Value::Symbol("TestRun".into()), vec![suite]);
    let v = ev.eval(run);
    let txt = format_value(&v);
    assert!(txt.contains("\"failed\" -> 1"), "expected failure, got: {}", txt);
    assert!(txt.contains("\"retries\" -> 2"), "expected retries recorded, got: {}", txt);
}

#[test]
fn fixtures_and_hooks() {
    let mut ev = Evaluator::new();
    ensure_default();
    stdlib::register_all(&mut ev);

    // Fixture x (all): setup returns 41; teardown checks value is 41 via AssertEqual[v,41]
    let setup_x = Value::pure_function(Some(vec![]), Value::Integer(41));
    let teardown_x = Value::pure_function(
        Some(vec!["v".into()]),
        Value::expr(
            Value::Symbol("AssertEqual".into()),
            vec![Value::Symbol("v".into()), Value::Integer(41)],
        ),
    );
    let fx_all = Value::expr(
        Value::Symbol("Fixture".into()),
        vec![
            Value::String("x".into()),
            setup_x,
            teardown_x.clone(),
            Value::assoc(vec![("Scope", Value::String("all".into()))]),
        ],
    );

    // Fixture y (each): setup returns 5; teardown no-op (Assert[True])
    let setup_y = Value::pure_function(Some(vec![]), Value::Integer(5));
    let teardown_y = Value::pure_function(
        Some(vec!["_".into()]),
        Value::expr(Value::Symbol("Assert".into()), vec![Value::Boolean(true)]),
    );
    let fx_each = Value::expr(
        Value::Symbol("Fixture".into()),
        vec![
            Value::String("y".into()),
            setup_y,
            teardown_y,
            Value::assoc(vec![("Scope", Value::String("each".into()))]),
        ],
    );

    // Case uses x and y: returns x + y + 1
    let body = Value::expr(
        Value::Symbol("Plus".into()),
        vec![
            Value::expr(
                Value::Symbol("Plus".into()),
                vec![Value::Symbol("x".into()), Value::Symbol("y".into())],
            ),
            Value::Integer(1),
        ],
    );
    let case = Value::expr(
        Value::Symbol("TestCase".into()),
        vec![Value::String("uses fixtures".into()), body],
    );

    // Hooks (no-op): BeforeAll/AfterAll/BeforeEach/AfterEach
    let hook_true = Value::expr(Value::Symbol("Assert".into()), vec![Value::Boolean(true)]);
    let h_ba = Value::expr(Value::Symbol("BeforeAll".into()), vec![hook_true.clone()]);
    let h_aa = Value::expr(Value::Symbol("AfterAll".into()), vec![hook_true.clone()]);
    let h_be = Value::expr(Value::Symbol("BeforeEach".into()), vec![hook_true.clone()]);
    let h_ae = Value::expr(Value::Symbol("AfterEach".into()), vec![hook_true.clone()]);

    let suite = Value::expr(
        Value::Symbol("TestSuite".into()),
        vec![
            Value::String("Hooks".into()),
            Value::list(vec![h_ba, h_be, fx_all, fx_each, case, h_ae, h_aa]),
        ],
    );
    let run = Value::expr(Value::Symbol("TestRun".into()), vec![suite]);
    let v = ev.eval(run);
    let txt = format_value(&v);
    assert!(txt.contains("\"passed\" -> 1"), "expected pass with fixtures, got: {}", txt);
}

#[test]
fn assert_throws_tag_and_predicate() {
    let mut ev = Evaluator::new();
    ensure_default();
    stdlib::register_all(&mut ev);
    // AssertThrows[Fail["boom"], "boom"]
    let thrown = Value::expr(Value::Symbol("Fail".into()), vec![Value::String("boom".into())]);
    let at_tag = Value::expr(
        Value::Symbol("AssertThrows".into()),
        vec![thrown.clone(), Value::String("boom".into())],
    );
    let r1 = format_value(&ev.eval(at_tag));
    assert_eq!(r1, "True");
    // Predicate variant
    let pred = Value::pure_function(Some(vec!["e".into()]), Value::Boolean(true));
    let at_pred = Value::expr(Value::Symbol("AssertThrows".into()), vec![thrown, pred]);
    let r2 = format_value(&ev.eval(at_pred));
    assert_eq!(r2, "True");
}
