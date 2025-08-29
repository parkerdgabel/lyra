use lyra_core::value::Value;
use lyra_runtime::{eval::Evaluator, set_default_registrar};
use lyra_stdlib as stdlib;

fn sym(s: &str) -> Value { Value::Symbol(s.into()) }
fn int(n: i64) -> Value { Value::Integer(n) }
fn list(xs: Vec<Value>) -> Value { Value::List(xs) }

#[test]
fn replace_basic_rule() {
    // Replace[f[1], { f[x_] -> 42 }] => 42
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    let lhs = Value::Expr {
        head: Box::new(sym("f")),
        args: vec![Value::Expr { head: Box::new(sym("NamedBlank")), args: vec![sym("x")] }],
    };
    let rule = Value::Expr { head: Box::new(sym("Rule")), args: vec![lhs, int(42)] };
    let expr = Value::Expr { head: Box::new(sym("Replace")), args: vec![
        Value::Expr { head: Box::new(sym("f")), args: vec![int(1)] },
        list(vec![rule])
    ] };
    let out = ev.eval(expr);
    assert_eq!(out, int(42));
}

#[test]
fn replace_all_list_elements() {
    // ReplaceAll[{f[1], f[2]}, { f[x_] -> x }] => {1,2}
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    let lhs = Value::Expr { head: Box::new(sym("f")), args: vec![Value::Expr { head: Box::new(sym("NamedBlank")), args: vec![sym("x")] }] };
    let rhs = sym("x");
    let rule = Value::Expr { head: Box::new(sym("Rule")), args: vec![lhs, rhs] };
    let target = list(vec![
        Value::Expr { head: Box::new(sym("f")), args: vec![int(1)] },
        Value::Expr { head: Box::new(sym("f")), args: vec![int(2)] },
    ]);
    let expr = Value::Expr { head: Box::new(sym("ReplaceAll")), args: vec![target, list(vec![rule])] };
    let out = ev.eval(expr);
    assert_eq!(out, list(vec![int(1), int(2)]));
}

#[test]
fn replace_repeated_chain() {
    // rules: a->b, b->c; ReplaceRepeated[a, rules] => c
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    let r1 = Value::Expr { head: Box::new(sym("Rule")), args: vec![sym("a"), sym("b")] };
    let r2 = Value::Expr { head: Box::new(sym("Rule")), args: vec![sym("b"), sym("c")] };
    let expr = Value::Expr { head: Box::new(sym("ReplaceRepeated")), args: vec![sym("a"), list(vec![r1, r2])] };
    let out = ev.eval(expr);
    assert_eq!(out, sym("c"));
}

#[test]
fn replace_first_only_one() {
    // ReplaceFirst on a nested list only changes first match
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    let lhs = Value::Expr { head: Box::new(sym("f")), args: vec![Value::Expr { head: Box::new(sym("NamedBlank")), args: vec![sym("x")] }] };
    let rule = Value::Expr { head: Box::new(sym("Rule")), args: vec![lhs, int(0)] };
    let target = list(vec![
        Value::Expr { head: Box::new(sym("f")), args: vec![int(1)] },
        Value::Expr { head: Box::new(sym("f")), args: vec![int(2)] },
        Value::Expr { head: Box::new(sym("f")), args: vec![int(3)] },
    ]);
    let expr = Value::Expr { head: Box::new(sym("ReplaceFirst")), args: vec![target, list(vec![rule])] };
    let out = ev.eval(expr);
    // Expect only the first f[...] replaced
    assert_eq!(out, list(vec![int(0), Value::Expr { head: Box::new(sym("f")), args: vec![int(2)] }, Value::Expr { head: Box::new(sym("f")), args: vec![int(3)] }]));
}

#[test]
fn thread_with_symbol_plus() {
    // Thread[Plus, {1,2,3}, {10,20,30}] => {11,22,33}
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    let expr = Value::Expr { head: Box::new(sym("Thread")), args: vec![
        sym("Plus"),
        list(vec![int(1), int(2), int(3)]),
        list(vec![int(10), int(20), int(30)]),
    ] };
    let out = ev.eval(expr);
    assert_eq!(out, list(vec![int(11), int(22), int(33)]));
}

#[test]
fn thread_with_pure_function() {
    // Thread[Function[#1 * 2], {3,4}] => {6,8}
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    let pf = Value::PureFunction { params: None, body: Box::new(Value::Expr { head: Box::new(sym("Times")), args: vec![Value::Slot(None), int(2)] }) };
    let expr = Value::Expr { head: Box::new(sym("Thread")), args: vec![
        pf,
        list(vec![int(3), int(4)]),
    ] };
    let out = ev.eval(expr);
    assert_eq!(out, list(vec![int(6), int(8)]));
}

