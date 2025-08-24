use lyra_parser::Parser;
use lyra_runtime::Evaluator;
use lyra_core::pretty::format_value;

fn eval_one(src: &str) -> String {
    let mut p = Parser::from_source(src);
    let vals = p.parse_all().expect("parse");
    let mut ev = Evaluator::new();
    format_value(&ev.eval(vals.into_iter().last().unwrap()))
}

#[test]
fn sequence_splicing_in_calls() {
    assert_eq!(eval_one("Plus[Sequence[1,2], 3]"), "6");
    assert_eq!(eval_one("Times[2, Sequence[3, 4]]"), "24");
}

#[test]
fn replace_all_simple_rule() {
    assert_eq!(eval_one("ReplaceAll[Plus[1,2], 3 -> 42]"), "42");
}

#[test]
fn replace_first_in_list_on_items() {
    assert_eq!(eval_one("ReplaceFirst[{1,2,3}, _Integer -> 9]"), "{9, 2, 3}");
}

#[test]
fn merge_with_combiner() {
    assert_eq!(eval_one("Merge[<|\"a\"->1|>, <|\"a\"->10, \"b\"->2|>, (xs)=>Total[xs]]"), "<|\"a\" -> 11, \"b\" -> 2|>");
}

#[test]
fn key_sort_and_sort_by() {
    assert_eq!(eval_one("KeySort[<|\"b\"->2, \"a\"->1|>]"), "<|\"a\" -> 1, \"b\" -> 2|>");
    assert_eq!(eval_one("SortBy[(k,v)=>v, <|\"b\"->2, \"a\"->1|>]"), "<|\"a\" -> 1, \"b\" -> 2|>");
}
