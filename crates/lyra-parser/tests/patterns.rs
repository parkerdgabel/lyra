use lyra_core::value::Value;
use lyra_parser::Parser;

fn parse_ok(src: &str) -> Vec<Value> {
    let mut p = Parser::from_source(src);
    p.parse_all().expect("parse ok")
}

#[test]
fn head_alternatives_without_parens() {
    let vals = parse_ok("f|g[x_]");
    assert_eq!(vals.len(), 1);
    match &vals[0] {
        Value::Expr { head, args } => {
            match &**head {
                Value::Expr { head: h2, args: alts } => {
                    match &**h2 {
                        Value::Symbol(s) => assert_eq!(s, "Alternative"),
                        _ => panic!(),
                    }
                    assert!(alts.len() >= 2);
                }
                _ => panic!("expected Alternative head"),
            }
            assert!(!args.is_empty());
        }
        _ => panic!("expected call"),
    }
}

#[test]
fn sequence_patterns_parse() {
    parse_ok("f[x__, y___]");
    parse_ok("f[x__Integer, y___String]");
    parse_ok("f[a|b, c]");
}
