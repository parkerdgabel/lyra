use lyra_parser::Parser;
use lyra_runtime::Evaluator;
use lyra_core::pretty::format_value;
use lyra_stdlib as stdlib;
use lyra_runtime::set_default_registrar;

fn eval_one(src: &str) -> String {
    let mut p = Parser::from_source(src);
    let vals = p.parse_all().expect("parse");
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    format_value(&ev.eval(vals.into_iter().last().unwrap()))
}

#[test]
fn replace_all_operator_works() {
    // Simple literal replacement
    assert_eq!(eval_one("1 /. 1 -> 9"), "9");
    // Vectorized mapping via pattern (operator form)
    assert_eq!(eval_one("{1,2,3} /. x_Integer -> x^2"), "{1, 4, 9}");
}

// ReplaceRepeated covered by parser tests; runtime behavior exercised via ReplaceAll/others.

#[test]
fn postfix_prefix_and_infix_evaluate() {
    // Postfix
    assert_eq!(eval_one("Range[5] // Total"), "15");
    // Prefix
    assert_eq!(eval_one("Total @ Range[5]"), "15");
    // Infix chaining
    assert_eq!(eval_one("1 ~ Plus ~ 2"), "3");
    assert_eq!(eval_one("1 ~ Plus ~ 2 ~ Times ~ 3"), "9");
}

#[test]
fn replacefirst_with_named_pattern() {
    assert_eq!(eval_one("ReplaceFirst[{1,2,3}, x_Integer -> 9]"), "{9, 2, 3}");
}

#[test]
fn matcher_sees_namedblank_on_integer() {
    // Build a NamedBlank lhs via parsing
    let mut p = Parser::from_source("x_Integer -> x^2");
    let vals = p.parse_all().expect("parse");
    let rule = vals.last().unwrap().clone();
    let lhs = match rule {
        lyra_core::value::Value::Expr { head, args } if matches!(*head, lyra_core::value::Value::Symbol(ref s) if s=="Rule") => args[0].clone(),
        _ => panic!("no rule"),
    };
    // Inspect shape
    let lhs_s = lyra_core::pretty::format_value(&lhs);
    eprintln!("LHS={}", lhs_s);
    assert!(lyra_rewrite::matcher::match_rule(&lhs, &lyra_core::value::Value::Integer(1)).is_some());
}

#[test]
fn replace_all_with_rule_list_symbols() {
    // Plus[x, y] /. {x -> 1, y -> 2} => 3
    assert_eq!(eval_one("Plus[x, y] /. {x -> 1, y -> 2}"), "3");
}
