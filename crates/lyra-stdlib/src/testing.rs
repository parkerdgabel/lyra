use crate::register_if;
use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
use std::collections::HashMap;
use std::time::Instant;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

// --- Registration
pub fn register_testing(ev: &mut Evaluator) {
    // Assertions
    ev.register("Assert", assert_fn as NativeFn, Attributes::empty());
    ev.register("AssertEqual", assert_equal_fn as NativeFn, Attributes::empty());
    ev.register("AssertThrows", assert_throws_fn as NativeFn, Attributes::HOLD_ALL);

    // Construction
    ev.register("TestCase", test_case_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("TestSuite", test_suite_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("BeforeAll", before_all_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("AfterAll", after_all_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("BeforeEach", before_each_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("AfterEach", after_each_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Fixture", fixture_fn as NativeFn, Attributes::HOLD_ALL);

    // Runner and reporter
    ev.register("TestRun", test_run_fn as NativeFn, Attributes::empty());
    ev.register("TestSpec", test_spec_report_fn as NativeFn, Attributes::empty());
    ev.register("TestJSON", test_json_report_fn as NativeFn, Attributes::empty());
}

pub fn register_testing_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    register_if(ev, pred, "Assert", assert_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "AssertEqual", assert_equal_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "AssertThrows", assert_throws_fn as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "TestCase", test_case_fn as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "TestSuite", test_suite_fn as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "BeforeAll", before_all_fn as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "AfterAll", after_all_fn as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "BeforeEach", before_each_fn as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "AfterEach", after_each_fn as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "Fixture", fixture_fn as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "TestRun", test_run_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "TestSpec", test_spec_report_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "TestJSON", test_json_report_fn as NativeFn, Attributes::empty());
}

// --- Helpers
fn is_failure(v: &Value) -> bool {
    match v {
        Value::Assoc(m) => match m.get("message") {
            Some(Value::String(s))
                if s == "Failure"
                    || s == "Time budget exceeded"
                    || s == "Computation cancelled" =>
            {
                true
            }
            _ => match m.get("tag") {
                Some(Value::String(tag))
                    if tag.starts_with("TimeBudget::") || tag.starts_with("Cancel::") =>
                {
                    true
                }
                _ => false,
            },
        },
        _ => false,
    }
}

fn failure(tag: &str, extra: Vec<(String, Value)>) -> Value {
    let mut m: HashMap<String, Value> = HashMap::new();
    m.insert("message".into(), Value::String("Failure".into()));
    m.insert("tag".into(), Value::String(tag.into()));
    for (k, v) in extra {
        m.insert(k, v);
    }
    Value::Assoc(m)
}

fn as_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Symbol(s) => Some(s.clone()),
        _ => None,
    }
}

fn deep_equal(a: &Value, b: &Value) -> bool {
    a == b
}

fn to_assoc(mut pairs: Vec<(impl Into<String>, Value)>) -> Value {
    let mut m: HashMap<String, Value> = HashMap::new();
    for (k, v) in pairs.drain(..) {
        m.insert(k.into(), v);
    }
    Value::Assoc(m)
}

// --- Assertions
fn assert_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("Assert".into())), args };
    }
    let cond = ev.eval(args[0].clone());
    match cond {
        Value::Boolean(true) => Value::Symbol("True".into()),
        other => {
            let msg = if let Some(v) = args.get(1) {
                as_string(&ev.eval(v.clone())).unwrap_or_else(|| "Assertion failed".into())
            } else {
                "Assertion failed".into()
            };
            failure(
                "Assert",
                vec![
                    ("expected".into(), Value::Boolean(true)),
                    ("actual".into(), other),
                    ("message".into(), Value::String(msg)),
                ],
            )
        }
    }
}

fn assert_equal_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("AssertEqual".into())), args };
    }
    let a = ev.eval(args[0].clone());
    let b = ev.eval(args[1].clone());
    // Optional options assoc as 3rd arg
    let (tol_opt, numeric_opt) = if args.len() >= 3 {
        match ev.eval(args[2].clone()) {
            Value::Assoc(m) => {
                let tol = match m.get("Tolerance") {
                    Some(Value::Real(f)) => Some(*f),
                    Some(Value::Integer(i)) => Some(*i as f64),
                    _ => None,
                };
                let numeric = match m.get("Numeric") {
                    Some(Value::Boolean(b)) => *b,
                    _ => tol.is_some(),
                };
                (tol, numeric)
            }
            _ => (None, false),
        }
    } else {
        (None, false)
    };

    if deep_equal(&a, &b) {
        return Value::Symbol("True".into());
    }

    if numeric_opt || tol_opt.is_some() {
        if let (Some(x), Some(y)) = (to_f64(&a), to_f64(&b)) {
            let tol = tol_opt.unwrap_or(0.0);
            if (x - y).abs() <= tol {
                return Value::Symbol("True".into());
            }
        }
    }
    failure("AssertEqual", vec![("expected".into(), b), ("actual".into(), a)])
}

fn assert_throws_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("AssertThrows".into())), args };
    }
    // Evaluate the expression and expect a Failure-shape assoc
    let v = ev.eval(args[0].clone());
    if !is_failure(&v) {
        return failure("AssertThrows", vec![("actual".into(), v)]);
    }
    // Optional matcher: tag string/symbol, list of tags, or predicate function
    if args.len() >= 2 {
        let matcher = ev.eval(args[1].clone());
        // Extract failure tag
        let tag = match &v {
            Value::Assoc(m) => m.get("tag").and_then(|x| match x {
                Value::String(s) => Some(s.clone()),
                Value::Symbol(s) => Some(s.clone()),
                _ => None,
            }),
            _ => None,
        };
        match matcher {
            Value::String(s) | Value::Symbol(s) => {
                if tag.as_deref() == Some(&s) {
                    Value::Symbol("True".into())
                } else {
                    failure(
                        "AssertThrows",
                        vec![("expectedTag".into(), Value::String(s)), ("actual".into(), v)],
                    )
                }
            }
            Value::List(ts) => {
                let mut ok = false;
                for t in &ts {
                    if let Value::String(s) = t.clone() {
                        if tag.as_deref() == Some(&s) {
                            ok = true;
                            break;
                        }
                    } else if let Value::Symbol(s) = t {
                        if tag.as_deref() == Some(&s) {
                            ok = true;
                            break;
                        }
                    }
                }
                if ok {
                    Value::Symbol("True".into())
                } else {
                    failure(
                        "AssertThrows",
                        vec![
                            ("expectedTags".into(), Value::List(ts.clone())),
                            ("actual".into(), v),
                        ],
                    )
                }
            }
            Value::PureFunction { .. } => {
                let pred = ev.eval(Value::expr(matcher, vec![v.clone()]));
                if matches!(pred, Value::Boolean(true)) {
                    Value::Symbol("True".into())
                } else {
                    failure(
                        "AssertThrows",
                        vec![("predicate".into(), Value::Boolean(false)), ("actual".into(), v)],
                    )
                }
            }
            other => failure(
                "AssertThrows",
                vec![("unsupportedMatcher".into(), other), ("actual".into(), v)],
            ),
        }
    } else {
        Value::Symbol("True".into())
    }
}

// --- Construction
// TestCase[name, body, opts?] => <| type:"case", name, body, opts |> (body held)
fn test_case_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("TestCase".into())), args };
    }
    let name = match &args[0] {
        Value::String(s) | Value::Symbol(s) => s.clone(),
        other => format!("{}", stringify_value(other)),
    };
    let body = args[1].clone(); // held
    let opts = if args.len() >= 3 { args[2].clone() } else { Value::Assoc(HashMap::new()) };
    to_assoc(vec![
        ("type", Value::String("case".into())),
        ("name", Value::String(name)),
        ("body", body),
        ("options", opts),
    ])
}

// TestSuite[name, items, opts?] where items is a List of case/suite
fn test_suite_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("TestSuite".into())), args };
    }
    let name = match &args[0] {
        Value::String(s) | Value::Symbol(s) => s.clone(),
        other => format!("{}", stringify_value(other)),
    };
    let items_v = ev.eval(args[1].clone());
    let items = match items_v {
        Value::List(vs) => vs,
        other => vec![other],
    };
    let opts =
        if args.len() >= 3 { ev.eval(args[2].clone()) } else { Value::Assoc(HashMap::new()) };
    to_assoc(vec![
        ("type", Value::String("suite".into())),
        ("name", Value::String(name)),
        ("items", Value::List(items)),
        ("options", opts),
    ])
}

// Hooks and fixtures builders
fn before_all_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("BeforeAll".into())), args };
    }
    to_assoc(vec![
        ("type", Value::String("hook".into())),
        ("kind", Value::String("BeforeAll".into())),
        ("body", args[0].clone()),
    ])
}
fn after_all_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("AfterAll".into())), args };
    }
    to_assoc(vec![
        ("type", Value::String("hook".into())),
        ("kind", Value::String("AfterAll".into())),
        ("body", args[0].clone()),
    ])
}
fn before_each_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("BeforeEach".into())), args };
    }
    to_assoc(vec![
        ("type", Value::String("hook".into())),
        ("kind", Value::String("BeforeEach".into())),
        ("body", args[0].clone()),
    ])
}
fn after_each_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("AfterEach".into())), args };
    }
    to_assoc(vec![
        ("type", Value::String("hook".into())),
        ("kind", Value::String("AfterEach".into())),
        ("body", args[0].clone()),
    ])
}

// Fixture[name, setup, teardown, opts?]
fn fixture_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 3 {
        return Value::Expr { head: Box::new(Value::Symbol("Fixture".into())), args };
    }
    let name = match &args[0] {
        Value::String(s) | Value::Symbol(s) => s.clone(),
        other => format!("{}", stringify_value(other)),
    };
    let setup = args[1].clone();
    let teardown = args[2].clone();
    let opts =
        if args.len() >= 4 { ev.eval(args[3].clone()) } else { Value::Assoc(HashMap::new()) };
    to_assoc(vec![
        ("type", Value::String("fixture".into())),
        ("name", Value::String(name)),
        ("setup", setup),
        ("teardown", teardown),
        ("options", opts),
    ])
}

// --- Runner & Reporter
// TestRun[target, opts?] => <| summary, items |>; treats any assoc with message=="Failure" as failure
fn test_run_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("TestRun".into())), args };
    }
    let target = ev.eval(args[0].clone());
    let opts =
        if args.len() >= 2 { ev.eval(args[1].clone()) } else { Value::Assoc(HashMap::new()) };

    let mut out_items: Vec<Value> = Vec::new();
    let t0 = Instant::now();
    let mut counts = (0i64, 0i64, 0i64); // pass, fail, skip

    run_any(ev, &target, &opts, &mut out_items, &mut counts);

    let dur_ms = t0.elapsed().as_millis() as i64;
    let summary = to_assoc(vec![
        ("passed", Value::Integer(counts.0)),
        ("failed", Value::Integer(counts.1)),
        ("skipped", Value::Integer(counts.2)),
        ("durationMs", Value::Integer(dur_ms)),
    ]);
    to_assoc(vec![("summary", summary), ("items", Value::List(out_items))])
}

fn run_any(
    ev: &mut Evaluator,
    target: &Value,
    _opts: &Value,
    out_items: &mut Vec<Value>,
    counts: &mut (i64, i64, i64),
) {
    match target {
        Value::Assoc(m) => {
            match m.get("type") {
                Some(Value::String(t)) if t == "case" => {
                    out_items.push(run_case(ev, m, counts));
                }
                Some(Value::String(t)) if t == "suite" => {
                    let (suite_res, p, f, s) = run_suite(ev, m);
                    counts.0 += p;
                    counts.1 += f;
                    counts.2 += s;
                    out_items.push(suite_res);
                }
                _ => {
                    // Treat assoc-like case (direct case map)
                    out_items.push(run_case(ev, m, counts));
                }
            }
        }
        Value::List(vs) => {
            for it in vs {
                run_any(ev, it, _opts, out_items, counts);
            }
        }
        other => {
            // If user passed a naked expression, wrap as single case
            let case = to_assoc(vec![
                ("type", Value::String("case".into())),
                ("name", Value::String("Case".into())),
                ("body", other.clone()),
                ("options", Value::Assoc(HashMap::new())),
            ]);
            run_any(ev, &case, _opts, out_items, counts);
        }
    }
}

// Internal helpers for runner
fn run_callable(ev: &mut Evaluator, callable: &Value, args: Vec<Value>) -> Value {
    match callable {
        Value::PureFunction { .. } => ev.eval(Value::expr(callable.clone(), args)),
        other => {
            if args.is_empty() {
                ev.eval(other.clone())
            } else {
                ev.eval(other.clone())
            }
        }
    }
}

fn run_suite(ev: &mut Evaluator, m: &HashMap<String, Value>) -> (Value, i64, i64, i64) {
    let name = m.get("name").and_then(as_string).unwrap_or_else(|| "Suite".into());
    let mut before_all: Vec<Value> = Vec::new();
    let mut after_all: Vec<Value> = Vec::new();
    let mut before_each: Vec<Value> = Vec::new();
    let mut after_each: Vec<Value> = Vec::new();
    let mut fixtures_all: Vec<(String, Value, Value)> = Vec::new();
    let mut fixtures_each: Vec<(String, Value, Value)> = Vec::new();
    let mut exec_items: Vec<Value> = Vec::new();
    if let Some(Value::List(items)) = m.get("items") {
        for it in items {
            if let Value::Assoc(im) = it {
                match (im.get("type"), im.get("kind")) {
                    (Some(Value::String(t)), _) if t == "hook" => {
                        if let (Some(k), Some(body)) =
                            (im.get("kind").and_then(as_string), im.get("body"))
                        {
                            match k.as_str() {
                                "BeforeAll" => before_all.push(body.clone()),
                                "AfterAll" => after_all.push(body.clone()),
                                "BeforeEach" => before_each.push(body.clone()),
                                "AfterEach" => after_each.push(body.clone()),
                                _ => {}
                            }
                        }
                    }
                    (Some(Value::String(t)), _) if t == "fixture" => {
                        if let (Some(n), Some(setup), Some(teardown)) = (
                            im.get("name").and_then(as_string),
                            im.get("setup"),
                            im.get("teardown"),
                        ) {
                            let scope = match im.get("options") {
                                Some(Value::Assoc(o)) => match o.get("Scope") {
                                    Some(Value::String(s)) => s.clone(),
                                    _ => "each".into(),
                                },
                                _ => "each".into(),
                            };
                            if scope == "all" {
                                fixtures_all.push((n, setup.clone(), teardown.clone()));
                            } else {
                                fixtures_each.push((n, setup.clone(), teardown.clone()));
                            }
                        }
                    }
                    _ => exec_items.push(Value::Assoc(im.clone())),
                }
            } else {
                exec_items.push(it.clone());
            }
        }
    }
    // Setup all-scope fixtures once, collect bindings
    let mut all_bindings: std::collections::HashMap<String, Value> =
        std::collections::HashMap::new();
    let mut all_vals: Vec<(String, Value, Value)> = Vec::new(); // (name, value, teardown)
    for (name, setup, teardown) in &fixtures_all {
        let res = run_callable(ev, setup, vec![]);
        all_bindings.insert(name.clone(), res.clone());
        all_vals.push((name.clone(), res, teardown.clone()));
    }
    // Run BeforeAll hooks under all fixture bindings
    for h in &before_all {
        let w = Value::expr(
            Value::Symbol("With".into()),
            vec![Value::Assoc(all_bindings.clone()), h.clone()],
        );
        let _ = ev.eval(w);
    }

    let mut suite_items: Vec<Value> = Vec::new();
    let mut scounts = (0i64, 0i64, 0i64);
    for it in exec_items {
        match it {
            Value::Assoc(ref im) if im.get("type") == Some(&Value::String("suite".into())) => {
                let (sub, p, f, s) = run_suite(ev, im);
                scounts.0 += p;
                scounts.1 += f;
                scounts.2 += s;
                suite_items.push(sub);
            }
            Value::Assoc(im) if im.get("type") == Some(&Value::String("case".into())) => {
                // Setup each fixtures and build bindings
                let mut bind_map: std::collections::HashMap<String, Value> = all_bindings.clone();
                let mut each_vals: Vec<(String, Value, Value)> = Vec::new();
                for (name, setup, teardown) in &fixtures_each {
                    let res = run_callable(ev, setup, vec![]);
                    bind_map.insert(name.clone(), res.clone());
                    each_vals.push((name.clone(), res, teardown.clone()));
                }
                // Run BeforeEach under bindings
                for h in &before_each {
                    let w = Value::expr(
                        Value::Symbol("With".into()),
                        vec![Value::Assoc(bind_map.clone()), h.clone()],
                    );
                    let _ = ev.eval(w);
                }
                // Run case under bindings
                let item = run_case_with_bindings(
                    ev,
                    &im,
                    &mut scounts,
                    Some(Value::Assoc(bind_map.clone())),
                );
                // Run AfterEach under bindings
                for h in &after_each {
                    let w = Value::expr(
                        Value::Symbol("With".into()),
                        vec![Value::Assoc(bind_map.clone()), h.clone()],
                    );
                    let _ = ev.eval(w);
                }
                // Teardown each fixtures with their values
                for (_name, val, teardown) in each_vals {
                    let _ = run_callable(ev, &teardown, vec![val]);
                }
                suite_items.push(item);
            }
            other => {
                suite_items.push(other);
            }
        }
    }

    // AfterAll under bindings and teardown all fixtures
    for h in &after_all {
        let w = Value::expr(
            Value::Symbol("With".into()),
            vec![Value::Assoc(all_bindings.clone()), h.clone()],
        );
        let _ = ev.eval(w);
    }
    for (_name, val, teardown) in all_vals {
        let _ = run_callable(ev, &teardown, vec![val]);
    }

    let suite_res = to_assoc(vec![
        ("type", Value::String("suite".into())),
        ("name", Value::String(name)),
        ("items", Value::List(suite_items)),
        (
            "summary",
            to_assoc(vec![
                ("passed", Value::Integer(scounts.0)),
                ("failed", Value::Integer(scounts.1)),
                ("skipped", Value::Integer(scounts.2)),
            ]),
        ),
    ]);
    (suite_res, scounts.0, scounts.1, scounts.2)
}

fn run_case(ev: &mut Evaluator, m: &HashMap<String, Value>, counts: &mut (i64, i64, i64)) -> Value {
    let name = m.get("name").and_then(as_string).unwrap_or_else(|| "Case".into());
    let body = m.get("body").cloned().unwrap_or(Value::Symbol("Null".into()));
    let t0 = Instant::now();
    // Extract options
    let (timeout_ms, mut retries) = match m.get("options") {
        Some(Value::Assoc(opts)) => {
            let to = match opts.get("TimeoutMs") {
                Some(Value::Integer(n)) if *n > 0 => *n as i64,
                _ => 0,
            };
            let r = match opts.get("Retries") {
                Some(Value::Integer(n)) if *n > 0 => *n as i64,
                _ => 0,
            };
            (to, r)
        }
        _ => (0, 0),
    };
    // Optionally wrap with Scope for timeout
    let mut eval_target = match &body {
        Value::PureFunction { .. } => Value::expr(body.clone(), vec![]),
        other => other.clone(),
    };
    if timeout_ms > 0 {
        let mut o: HashMap<String, Value> = HashMap::new();
        o.insert("TimeBudgetMs".into(), Value::Integer(timeout_ms));
        eval_target =
            Value::expr(Value::Symbol("Scope".into()), vec![Value::Assoc(o), eval_target]);
    }
    // Evaluate with retries
    let mut used_retries: i64 = 0;
    let mut res = ev.eval(eval_target.clone());
    while retries > 0 && is_failure(&res) {
        used_retries += 1;
        retries -= 1;
        res = ev.eval(eval_target.clone());
    }
    let dur_ms = t0.elapsed().as_millis() as i64;
    let (status, err) = if is_failure(&res) {
        counts.1 += 1;
        ("fail", Some(res))
    } else {
        counts.0 += 1;
        ("pass", None)
    };
    let mut pairs: Vec<(String, Value)> = vec![
        ("type".to_string(), Value::String("case".into())),
        ("name".to_string(), Value::String(name)),
        ("status".to_string(), Value::String(status.into())),
        ("durationMs".to_string(), Value::Integer(dur_ms)),
    ];
    if used_retries > 0 {
        pairs.push(("retries".to_string(), Value::Integer(used_retries)));
    }
    if let Some(e) = err {
        pairs.push(("error".to_string(), e));
    }
    to_assoc(pairs)
}

fn run_case_with_bindings(
    ev: &mut Evaluator,
    m: &HashMap<String, Value>,
    counts: &mut (i64, i64, i64),
    bindings: Option<Value>,
) -> Value {
    let name = m.get("name").and_then(as_string).unwrap_or_else(|| "Case".into());
    let body = m.get("body").cloned().unwrap_or(Value::Symbol("Null".into()));
    let t0 = Instant::now();
    // Extract options
    let (timeout_ms, mut retries) = match m.get("options") {
        Some(Value::Assoc(opts)) => {
            let to = match opts.get("TimeoutMs") {
                Some(Value::Integer(n)) if *n > 0 => *n as i64,
                _ => 0,
            };
            let r = match opts.get("Retries") {
                Some(Value::Integer(n)) if *n > 0 => *n as i64,
                _ => 0,
            };
            (to, r)
        }
        _ => (0, 0),
    };
    let mut eval_target = match &body {
        Value::PureFunction { .. } => Value::expr(body.clone(), vec![]),
        other => other.clone(),
    };
    if timeout_ms > 0 {
        let mut o: HashMap<String, Value> = HashMap::new();
        o.insert("TimeBudgetMs".into(), Value::Integer(timeout_ms));
        eval_target =
            Value::expr(Value::Symbol("Scope".into()), vec![Value::Assoc(o), eval_target]);
    }
    if let Some(b) = bindings {
        eval_target = Value::expr(Value::Symbol("With".into()), vec![b, eval_target]);
    }
    let mut used_retries: i64 = 0;
    let mut res = ev.eval(eval_target.clone());
    while retries > 0 && is_failure(&res) {
        used_retries += 1;
        retries -= 1;
        res = ev.eval(eval_target.clone());
    }
    let dur_ms = t0.elapsed().as_millis() as i64;
    let (status, err) = if is_failure(&res) {
        counts.1 += 1;
        ("fail", Some(res))
    } else {
        counts.0 += 1;
        ("pass", None)
    };
    let mut pairs: Vec<(String, Value)> = vec![
        ("type".to_string(), Value::String("case".into())),
        ("name".to_string(), Value::String(name)),
        ("status".to_string(), Value::String(status.into())),
        ("durationMs".to_string(), Value::Integer(dur_ms)),
    ];
    if used_retries > 0 {
        pairs.push(("retries".to_string(), Value::Integer(used_retries)));
    }
    if let Some(e) = err {
        pairs.push(("error".to_string(), e));
    }
    to_assoc(pairs)
}

// TestSpec[resultOrSuite] => string
fn test_spec_report_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("TestSpec".into())), args };
    }
    let v = ev.eval(args[0].clone());
    let result = match v {
        Value::Assoc(ref m) if m.contains_key("summary") && m.contains_key("items") => v.clone(),
        Value::Assoc(_) | Value::List(_) => test_run_fn(ev, vec![v.clone()]),
        other => test_run_fn(ev, vec![other.clone()]),
    };
    Value::String(spec_string(&result))
}

// TestJSON[resultOrSuite] => string JSON
fn test_json_report_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("TestJSON".into())), args };
    }
    let v = ev.eval(args[0].clone());
    let result = match v {
        Value::Assoc(ref m) if m.contains_key("summary") && m.contains_key("items") => v.clone(),
        Value::Assoc(_) | Value::List(_) => test_run_fn(ev, vec![v.clone()]),
        other => test_run_fn(ev, vec![other.clone()]),
    };
    let s = serde_json::to_string(&result).unwrap_or_else(|_| String::from("{}"));
    Value::String(s)
}

// Format a minimal spec-style string from a TestRun result
fn spec_string(result: &Value) -> String {
    match result {
        Value::Assoc(m) => {
            let mut out = String::new();
            if let Some(Value::List(items)) = m.get("items") {
                for it in items {
                    spec_item(it, 0, &mut out);
                }
            }
            if let Some(Value::Assoc(summary)) = m.get("summary") {
                let p = summary
                    .get("passed")
                    .and_then(|v| if let Value::Integer(i) = v { Some(*i) } else { None })
                    .unwrap_or(0);
                let f = summary
                    .get("failed")
                    .and_then(|v| if let Value::Integer(i) = v { Some(*i) } else { None })
                    .unwrap_or(0);
                let ms = summary
                    .get("durationMs")
                    .and_then(|v| if let Value::Integer(i) = v { Some(*i) } else { None })
                    .unwrap_or(0);
                out.push_str(&format!("\n{} passing, {} failing ({} ms)\n", p, f, ms));
            }
            out
        }
        _ => String::from(""),
    }
}

fn spec_item(v: &Value, indent: usize, out: &mut String) {
    let pad = " ".repeat(indent * 2);
    match v {
        Value::Assoc(m) => match m.get("type") {
            Some(Value::String(t)) if t == "suite" => {
                let name = m.get("name").and_then(as_string).unwrap_or_else(|| "Suite".into());
                out.push_str(&format!("{}{}\n", pad, name));
                if let Some(Value::List(items)) = m.get("items") {
                    for it in items {
                        spec_item(it, indent + 1, out);
                    }
                }
            }
            Some(Value::String(t)) if t == "case" => {
                let name = m.get("name").and_then(as_string).unwrap_or_else(|| "Case".into());
                let status = m.get("status").and_then(as_string).unwrap_or_else(|| "pass".into());
                let mark = if status == "pass" { '✓' } else { '✗' };
                out.push_str(&format!("{}{} {}\n", pad, mark, name));
            }
            _ => {}
        },
        _ => {}
    }
}

// stringify helper for fallback names
fn stringify_value(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Symbol(s) => s.clone(),
        Value::Integer(i) => i.to_string(),
        Value::Real(f) => format!("{}", f),
        _ => String::from("Value"),
    }
}

fn to_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Integer(i) => Some(*i as f64),
        Value::Real(f) => Some(*f),
        Value::Rational { num, den } => Some(*num as f64 / *den as f64),
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}
