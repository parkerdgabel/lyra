use lyra_parser::Parser;
use lyra_core::pretty::format_value;

fn fmt(src: &str) -> String {
    let mut p = Parser::from_source(src);
    let vals = p.parse_all().expect("parse");
    format_value(&vals.last().unwrap())
}

#[test]
fn precedence_arith_power() {
    assert_eq!(fmt("1 + 2 * 3"), "Plus[1, Times[2, 3]]");
    assert_eq!(fmt("(1 + 2) * 3"), "Times[Plus[1, 2], 3]");
    assert_eq!(fmt("2 ^ 3 ^ 2"), "Power[2, Power[3, 2]]");
}

#[test]
fn comparisons_and_booleans() {
    let s1 = fmt("1 + 2 == 3 && 4 > 3");
    assert!(s1.contains("And["));
    let s2 = fmt("1 != 2 || 2 <= 2");
    assert!(s2.contains("Or["));
}

#[test]
fn rule_operators_replace_all_and_repeated() {
    // ReplaceAll
    assert_eq!(fmt("x /. y -> z"), "ReplaceAll[x, Rule[y, z]]");
    // ReplaceRepeated
    assert_eq!(fmt("x //. y -> z"), "ReplaceRepeated[x, Rule[y, z]]");
    // Typed named blank in rule LHS
    assert_eq!(
        fmt("{1,2,3} /. x_Integer -> x^2"),
        "ReplaceAll[{1, 2, 3}, Rule[x_Integer, Power[x, 2]]]"
    );
}

#[test]
fn replace_all_with_rule_list() {
    // Replace using a list of rules on the RHS
    assert_eq!(
        fmt("Plus[x, y] /. {x -> 1, y -> 2}"),
        "ReplaceAll[Plus[x, y], {Rule[x, 1], Rule[y, 2]}]"
    );
}

#[test]
fn prefix_postfix_and_infix_forms() {
    // Prefix f @ x => f[x]
    assert_eq!(fmt("f @ x"), "f[x]");
    // Postfix x // f => f[x]
    assert_eq!(fmt("x // f"), "f[x]");
    // Chain postfix
    assert_eq!(fmt("x // f // g"), "g[f[x]]");
    // Chain prefix (right-assoc)
    assert_eq!(fmt("f @ g @ x"), "f[g[x]]");
    // Infix a ~ f ~ b => f[a, b]
    assert_eq!(fmt("a ~ f ~ b"), "f[a, b]");
    // Mixed chaining: a ~ f ~ b ~ g ~ c => g[f[a, b], c]
    assert_eq!(fmt("a ~ f ~ b ~ g ~ c"), "g[f[a, b], c]");
}

#[test]
fn rule_lhs_condition_parenthesized() {
    // (x_Integer /; x > 0) -> x - 1
    let s = fmt("(x_Integer /; x > 0) -> x - 1");
    assert!(s.contains("Rule[Condition["));
}

#[test]
fn replace_all_rule_roundtrip_desugared() {
    let src = "Rule[ReplaceAll[{1, 2, 3}, Rule[x_Integer, Power[x, 2]]], {1, 4, 9}]";
    let mut p = Parser::from_source(src);
    let vals = p.parse_all().expect("parse");
    let formatted = format_value(vals.last().unwrap());
    assert_eq!(formatted, src);
}
