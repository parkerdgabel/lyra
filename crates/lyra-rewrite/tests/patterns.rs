use lyra_core::value::Value;
use lyra_rewrite as rw;

fn sym(s: &str) -> Value { Value::Symbol(s.into()) }
fn int(n: i64) -> Value { Value::Integer(n) }

#[test]
fn repeated_matches_multiple_args() {
    // Rule: f[Repeated[_]] -> 42 matches f[1,2,3]
    let lhs = Value::Expr { head: Box::new(sym("f")), args: vec![
        Value::Expr { head: Box::new(sym("Repeated")), args: vec![ Value::Expr { head: Box::new(sym("Blank")), args: vec![] } ] }
    ]};
    let rhs = int(42);
    let rules = vec![(lhs, rhs)];
    let expr = Value::Expr { head: Box::new(sym("f")), args: vec![ int(1), int(2), int(3) ] };
    let out = rw::engine::rewrite_once(expr, &rules);
    assert_eq!(out, int(42));
}

#[test]
fn repeatednull_matches_empty_and_many() {
    // Rule: h[RepeatedNull[_]] -> 7 matches h[] and h[1,2]
    let lhs = Value::Expr { head: Box::new(sym("h")), args: vec![
        Value::Expr { head: Box::new(sym("RepeatedNull")), args: vec![ Value::Expr { head: Box::new(sym("Blank")), args: vec![] } ] }
    ]};
    let rhs = int(7);
    let rules = vec![(lhs, rhs)];
    let expr0 = Value::Expr { head: Box::new(sym("h")), args: vec![] };
    let expr1 = Value::Expr { head: Box::new(sym("h")), args: vec![ int(1), int(2) ] };
    let out0 = rw::engine::rewrite_once(expr0, &rules);
    let out1 = rw::engine::rewrite_once(expr1, &rules);
    assert_eq!(out0, int(7));
    assert_eq!(out1, int(7));
}

#[test]
fn optional_allows_missing_arg() {
    // Rule: g[x_, Optional[_]] -> 9 matches g[1] and g[1,2]
    let lhs = Value::Expr { head: Box::new(sym("g")), args: vec![
        Value::Expr { head: Box::new(sym("NamedBlank")), args: vec![ sym("x") ] },
        Value::Expr { head: Box::new(sym("Optional")), args: vec![ Value::Expr { head: Box::new(sym("Blank")), args: vec![] } ] },
    ]};
    let rhs = int(9);
    let rules = vec![(lhs, rhs)];
    let expr0 = Value::Expr { head: Box::new(sym("g")), args: vec![ int(1) ] };
    let expr1 = Value::Expr { head: Box::new(sym("g")), args: vec![ int(1), int(2) ] };
    let out0 = rw::engine::rewrite_once(expr0, &rules);
    let out1 = rw::engine::rewrite_once(expr1, &rules);
    assert_eq!(out0, int(9));
    assert_eq!(out1, int(9));
}

