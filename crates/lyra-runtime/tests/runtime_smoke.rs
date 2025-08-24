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

// If semantics are covered implicitly via evaluator; dedicated tests can be added later.

#[test]
fn stdlib_min_basics() {
    assert_eq!(eval_one("EvenQ[2]"), "True");
    assert_eq!(eval_one("OddQ[3]"), "True");
    assert_eq!(eval_one("EvenQ[{1,2,3}]"), "{False, True, False}");
    assert_eq!(eval_one("StringLength[\"abc\"]"), "3");
    assert_eq!(eval_one("ToUpper[\"abC\"]"), "\"ABC\"");
    assert_eq!(eval_one("ToLower[\"AbC\"]"), "\"abc\"");
    assert_eq!(eval_one("StringJoin[{\"a\", \"b\"}]"), "\"ab\"");
    assert_eq!(eval_one("StringSplit[\"a b  c\"]"), "{\"a\", \"b\", \"c\"}");
    assert_eq!(eval_one("StringSplit[\"a,b,c\", \",\"]"), "{\"a\", \"b\", \"c\"}");
    assert_eq!(eval_one("StringSplit[\"abc\", \"\"]"), "{\"a\", \"b\", \"c\"}");
    assert_eq!(eval_one("StartsWith[\"hello\", \"he\"]"), "True");
    assert_eq!(eval_one("EndsWith[\"hello\", \"lo\"]"), "True");
    assert_eq!(eval_one("StringReplace[\"foo bar foo\", \"foo\", \"qux\"]"), "\"qux bar qux\"");
    assert_eq!(eval_one("StringReverse[\"abc\"]"), "\"cba\"");
    assert_eq!(eval_one("StringPadLeft[\"7\", 3, \"0\"]"), "\"007\"");
    assert_eq!(eval_one("StringPadRight[\"hi\", 4]"), "\"hi  \"");
    assert_eq!(eval_one("StartsWith[{\"a\", \"ab\"}, \"a\"]"), "{True, True}");
    assert_eq!(eval_one("EndsWith[{\"ab\", \"ba\"}, \"a\"]"), "{False, True}");
    assert_eq!(eval_one("Abs[{-2, 0, 3}]"), "{2, 0, 3}");
    assert_eq!(eval_one("Length[{1,2,3}]"), "3");
    assert_eq!(eval_one("Range[1,5]"), "{1, 2, 3, 4, 5}");
    assert_eq!(eval_one("Join[{1,2}, {3}]"), "{1, 2, 3}");
    assert_eq!(eval_one("Reverse[{1,2}]"), "{2, 1}");
    assert_eq!(eval_one("Min[3,1,2]"), "1");
    assert_eq!(eval_one("Max[{1,5,2}]"), "5");
    assert_eq!(eval_one("StringTrim[\"  hi  \"]"), "\"hi\"");
    assert_eq!(eval_one("StringContains[\"hello\", \"ell\"]"), "True");
    assert_eq!(eval_one("StringContains[\"hello\", \"xyz\"]"), "False");
}

#[test]
fn echo_attribute_tests() {
    assert_eq!(eval_one("OrderlessEcho[b, a]"), "{a, b}");
    assert_eq!(eval_one("FlatEcho[FlatEcho[1,2], 3]"), "{1, 2, 3}");
    assert_eq!(eval_one("FlatOrderlessEcho[FlatOrderlessEcho[b,a], c]"), "{a, b, c}");
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

#[test]
fn explain_traces_counts_and_order() {
    // Listable threading count
    let s1 = eval_one("Explain[Plus[{1,2,3}, 10]]");
    assert!(s1.contains("\"ListableThread\""));
    assert!(s1.contains("\"count\" -> 3"));
    // Orderless final order
    let s2 = eval_one("Explain[OrderlessEcho[c, a, b]]");
    assert!(s2.contains("\"OrderlessSort\""));
    assert!(s2.contains("\"finalOrder\" -> {a, b, c}"));
}

#[test]
fn explain_traces_hold_and_flat() {
    // HOLD_ALL on FlatEcho: held positions should be 1 and 2
    let s1 = eval_one("Explain[FlatEcho[1, 2]]");
    assert!(s1.contains("\"Hold\""));
    assert!(s1.contains("\"held\" -> {1, 2}"));
    // Flat flatten adds inner args count
    let s2 = eval_one("Explain[FlatEcho[FlatEcho[1,2], 3]]");
    assert!(s2.contains("\"FlatFlatten\""));
    assert!(s2.contains("\"added\" -> 2"));
}
