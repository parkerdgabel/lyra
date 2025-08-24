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
