use lyra_core::pretty::format_value;
use lyra_parser::Parser;
use lyra_runtime::set_default_registrar;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;

fn eval_one(src: &str) -> String {
    let mut p = Parser::from_source(src);
    let vals = p.parse_all().expect("parse");
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    format_value(&ev.eval(vals.into_iter().last().unwrap()))
}

#[test]
fn select_withcolumns_and_collect() {
    let out = eval_one(
        "Collect[WithColumns[DatasetFromRows[{<|\"a\"->1, \"b\"->2|>, <|\"a\"->3, \"b\"->4|>}], <|\"c\"-> (row)=> Plus[Part[row, \"a\"], Part[row, \"b\"]]|>]]",
    );
    assert!(out.contains("\"c\" -> 3"));
    assert!(out.contains("\"c\" -> 7"));
}

#[test]
fn filter_dataset_and_count() {
    let out = eval_one(
        "Count[FilterRows[(row)=> Greater[Part[row, \"a\"], 1], DatasetFromRows[{<|\"a\"->1|>, <|\"a\"->2|>}]]]",
    );
    assert_eq!(out, "1");
}

#[test]
fn groupby_and_aggregate_sum_count() {
    let out = eval_one(
        "Collect[Agg[GroupBy[DatasetFromRows[{<|\"k\"->\"x\", \"v\"->1|>, <|\"k\"->\"x\", \"v\"->2|>, <|\"k\"->\"y\", \"v\"->3|>}], {\"k\"}], <|\"sum\"->Sum[\"v\"], \"count\"->Count[]|>]]",
    );
    eprintln!("out= {}", out);
    // keys may be present; primary checks are on aggregates
    assert!(out.contains("\"count\" -> 2"));
    assert!(out.contains("\"count\" -> 1"));
}

#[test]
fn distinct_by_and_distinct_cols() {
    let by = eval_one(
        "Count[DistinctBy[DatasetFromRows[{<|\"id\"->1, \"v\"->9|>, <|\"id\"->1, \"v\"->10|>, <|\"id\"->2, \"v\"->5|>}], {\"id\"}]]",
    );
    assert_eq!(by, "2");
    let dist = eval_one(
        "Count[Distinct[DatasetFromRows[{<|\"a\"->1|>, <|\"a\"->1|>, <|\"a\"->2|>}], {\"a\"}]]",
    );
    assert_eq!(dist, "2");
}

#[test]
fn head_tail_offset_dataset_counts() {
    let cnt_head = eval_one(
        "Count[Head[DatasetFromRows[{<|\"a\"->1|>, <|\"a\"->2|>, <|\"a\"->3|>}], 2]]",
    );
    assert_eq!(cnt_head, "2");
    let cnt_offset = eval_one(
        "Count[Offset[DatasetFromRows[{<|\"a\"->1|>, <|\"a\"->2|>, <|\"a\"->3|>}], 1]]",
    );
    assert_eq!(cnt_offset, "2");
    let cnt_tail = eval_one(
        "Count[Tail[DatasetFromRows[{<|\"a\"->1|>, <|\"a\"->2|>, <|\"a\"->3|>}], 2]]",
    );
    assert_eq!(cnt_tail, "2");
}

#[test]
fn join_inner_and_left() {
    let left = "DatasetFromRows[{<|\"id\"->1, \"a\"->\"x\"|>, <|\"id\"->2, \"a\"->\"y\"|>, <|\"id\"->3, \"a\"->\"z\"|>}]";
    let right = "DatasetFromRows[{<|\"id\"->1, \"b\"->10|>, <|\"id\"->3, \"b\"->30|>}]";
    let inner_cnt = eval_one(&format!("Count[Join[{}, {}, {{\"id\"}}]]", left, right));
    assert_eq!(inner_cnt, "2");
    let left_cnt = eval_one(&format!("Count[Join[{}, {}, {{\"id\"}}, <|\"How\"->\"left\"|>]]", left, right));
    assert_eq!(left_cnt, "3");
}
