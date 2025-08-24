use lyra_core::value::Value;
use lyra_rewrite as rw;

fn sym(s: &str) -> Value { Value::Symbol(s.into()) }
fn int(n: i64) -> Value { Value::Integer(n) }
fn call(h: &str, args: Vec<Value>) -> Value { Value::expr(sym(h), args) }

#[test]
fn blank_and_namedblank_match() {
    // Rule: f[_] -> 1
    let rules = vec![(call("f", vec![call("Blank", vec![])]), int(1))];
    let out = rw::engine::rewrite_once(call("f", vec![int(5)]), &rules);
    assert_eq!(out, int(1));

    // Rule: f[x_Integer] -> x
    let lhs = call("f", vec![call("NamedBlank", vec![sym("x"), sym("Integer")])]);
    let rhs = sym("x");
    let rules = vec![(lhs, rhs)];
    let out1 = rw::engine::rewrite_once(call("f", vec![int(7)]), &rules);
    assert_eq!(out1, int(7));
    let out2 = rw::engine::rewrite_once(call("f", vec![Value::String("s".into())]), &rules);
    assert_eq!(out2, call("f", vec![Value::String("s".into())]));
}

#[test]
fn alternative_match() {
    // Rule: Alternative[a, b] -> X applied at top-level
    let lhs = call("Alternative", vec![sym("a"), sym("b")]);
    let rhs = sym("X");
    let rules = vec![(lhs, rhs)];
    assert_eq!(rw::engine::rewrite_once(sym("a"), &rules), sym("X"));
    assert_eq!(rw::engine::rewrite_once(sym("b"), &rules), sym("X"));
    assert_eq!(rw::engine::rewrite_once(sym("c"), &rules), sym("c"));
}

#[test]
fn patterntest_evenq_default() {
    // Rule: PatternTest[_, EvenQ] -> 1 applied to integers
    let lhs = call("PatternTest", vec![call("Blank", vec![]), sym("EvenQ")]);
    let rules = vec![(lhs, int(1))];
    assert_eq!(rw::engine::rewrite_once(int(2), &rules), int(1));
    assert_eq!(rw::engine::rewrite_once(int(3), &rules), int(3));
}

#[test]
fn condition_with_ctx_predicate() {
    // Rule: Condition[x_, IsTwo[x]] -> 42, where IsTwo is evaluated via ctx
    let lhs = call("Condition", vec![call("NamedBlank", vec![sym("x")]), call("IsTwo", vec![sym("x")])]);
    let rules = vec![(lhs, int(42))];
    let pred = |_pred: &Value, _arg: &Value| -> bool { true }; // unused
    let cond = |cond: &Value, binds: &rw::matcher::Bindings| -> bool {
        // cond is IsTwo[x], where x is bound
        if let Value::Expr { head, args } = cond {
            if matches!(**head, Value::Symbol(ref s) if s=="IsTwo") {
                if let Some(Value::Symbol(xname)) = args.get(0) {
                    if let Some(Value::Integer(n)) = binds.get(xname) { return *n == 2; }
                }
            }
        }
        false
    };
    let ctx = rw::matcher::MatcherCtx { eval_pred: Some(&pred), eval_cond: Some(&cond) };
    assert_eq!(rw::engine::rewrite_once_with_ctx(&ctx, int(2), &rules), int(42));
    assert_eq!(rw::engine::rewrite_once_with_ctx(&ctx, int(3), &rules), int(3));
}

#[test]
fn sequence_splicing_in_args() {
    // Rule: f[xs__] -> Sequence[xs]; applying inside g should splice args
    let lhs = call("f", vec![call("NamedBlankSequence", vec![sym("xs")])]);
    let rhs = sym("xs");
    let rules = vec![(lhs, rhs)];
    let expr = call("g", vec![call("f", vec![int(1), int(2), int(3)]), int(9)]);
    let out = rw::engine::rewrite_once(expr, &rules);
    assert_eq!(out, call("g", vec![int(1), int(2), int(3), int(9)]));
}
