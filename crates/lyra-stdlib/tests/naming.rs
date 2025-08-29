use lyra_core::value::Value;
use lyra_runtime::{Evaluator, set_default_registrar};
use lyra_stdlib as stdlib;

fn sym(s: &str) -> Value { Value::Symbol(s.into()) }

#[test]
fn legacy_names_are_not_registered() {
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let legacy = vec![
        "AssocGet",
        "AssocContainsKeyQ",
        "StringSplit",
        "StringLength",
        "SetUnion",
        "SetIntersection",
        "SetDifference",
        "ListUnion",
        "ListIntersection",
        "ListDifference",
        "HttpServer",
        "Lookup",
        "URLEncode",
        "URLDecode",
    ];
    for name in legacy {
        let expr = Value::Expr { head: Box::new(sym(name)), args: vec![] };
        let out = ev.eval(expr.clone());
        assert_eq!(out, expr, "legacy name {} should not be a registered builtin", name);
    }
}
