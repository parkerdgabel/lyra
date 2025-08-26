use lyra_core::pretty::format_value;
use lyra_parser::Parser;

fn parse_fmt(src: &str) -> String {
    let mut p = Parser::from_source(src);
    let vals = p.parse_all().expect("parse");
    format_value(vals.last().unwrap())
}

#[test]
fn assoc_in_call_args_simple() {
    let s = parse_fmt("KeySort[<|\"b\"->2, \"a\"->1|>]");
    assert!(s.starts_with("KeySort["));
}

#[test]
fn assoc_literal_standalone() {
    let s = parse_fmt("<|\"b\"->2, \"a\"->1|>");
    assert!(s.starts_with("<|"));
}

#[test]
fn assoc_empty_standalone() {
    let s = parse_fmt("<||>");
    assert_eq!(s, "<||>");
}

#[test]
fn assoc_symbol_keys() {
    let s1 = parse_fmt("<|a->1|>");
    assert!(s1.starts_with("<|"));
    let s2 = parse_fmt("<|a->1, b->2|>");
    assert!(s2.starts_with("<|"));
}

#[test]
fn assoc_in_call_args_with_lambda() {
    let s = parse_fmt("SortBy[(k,v)=>v, <|\"b\"->2, \"a\"->1|>]");
    assert!(s.starts_with("SortBy["));
}

#[test]
fn assoc_multiple_args_and_lambda() {
    let s = parse_fmt("Merge[<|\"a\"->1|>, <|\"a\"->10, \"b\"->2|>, (xs)=>Total[xs]]");
    assert!(s.starts_with("Merge["));
}
