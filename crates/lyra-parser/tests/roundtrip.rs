use lyra_core::pretty::format_value;
use lyra_parser::Parser;

fn roundtrip_ok(src: &str) {
    let mut p = Parser::from_source(src);
    let vals = p.parse_all().expect("parse");
    assert!(!vals.is_empty());
    let printed = format_value(vals.last().unwrap());
    // Reparse printed form should succeed
    let mut p2 = Parser::from_source(&printed);
    let _vals2 = p2.parse_all().expect("reparse pretty");
}

#[test]
fn rt_call() {
    roundtrip_ok("f[1,2,3]");
}

#[test]
fn rt_list() {
    roundtrip_ok("{1, 2, 3}");
}

// Association roundtrip disabled pending parser assoc fix
#[test]
fn rt_assoc() {
    roundtrip_ok("<|\"a\"->1, \"b\"->2|>");
}

#[test]
fn rt_alternative() {
    roundtrip_ok("x_Integer | y_String");
}

#[test]
fn rt_lambda() {
    roundtrip_ok("(x)=>Plus[x,1]");
}
