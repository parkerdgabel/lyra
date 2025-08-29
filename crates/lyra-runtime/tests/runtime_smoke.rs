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
    assert_eq!(eval_one("Range[0,10,2]"), "{0, 2, 4, 6, 8, 10}");
    assert_eq!(eval_one("Range[5,1,-2]"), "{5, 3, 1}");
    // step=0 should emit a Failure; OnFailure should catch and return default
    assert_eq!(eval_one("OnFailure[Range[1,5,0], \"bad step\"]"), "\"bad step\"");
    assert_eq!(eval_one("Join[{1,2}, {3}]"), "{1, 2, 3}");
    assert_eq!(eval_one("Reverse[{1,2}]"), "{2, 1}");
    assert_eq!(eval_one("Min[3,1,2]"), "1");
    assert_eq!(eval_one("Max[{1,5,2}]"), "5");
    assert_eq!(eval_one("StringTrim[\"  hi  \"]"), "\"hi\"");
    assert_eq!(eval_one("StringContains[\"hello\", \"ell\"]"), "True");
    assert_eq!(eval_one("StringContains[\"hello\", \"xyz\"]"), "False");
}

#[test]
fn stdlib_string_filters() {
    // HTML
    assert_eq!(eval_one("HtmlEscape[\"<a&b>\"]"), "\"&lt;a&amp;b&gt;\"");
    assert_eq!(eval_one("HtmlUnescape[\"&lt;a&amp;b&gt;\"]"), "\"<a&b>\"");
    // URL
    assert_eq!(eval_one("UrlEncode[\"a b/©\"]"), "\"a%20b%2F%C2%A9\"");
    assert_eq!(eval_one("UrlDecode[\"a%20b%2F%C2%A9\"]"), "\"a b/©\"");
    // Form encoding and more helpers
    assert_eq!(eval_one("UrlFormEncode[\"a b+c\"]"), "\"a+b%2Bc\"");
    assert_eq!(eval_one("UrlFormDecode[\"a+b%2Bc\"]"), "\"a b+c\"");
    assert_eq!(eval_one("Slugify[\"Hello, World!\"]"), "\"hello-world\"");
    assert_eq!(eval_one("StringTruncate[\"abcdef\", 4]"), "\"abc…\"");
    assert_eq!(eval_one("StringTruncate[\"abcdef\", 4, \"...\"]"), "\"a...\"");
    assert_eq!(eval_one("CamelCase[\"hello world_test\"]"), "\"helloWorldTest\"");
    assert_eq!(eval_one("SnakeCase[\"HelloWorld Test\"]"), "\"hello_world_test\"");
    assert_eq!(eval_one("KebabCase[\"HelloWorld Test\"]"), "\"hello-world-test\"");
}

#[test]
fn stdlib_string_extras() {
    assert_eq!(eval_one("StringJoinWith[\",\", {\"a\", \"b\"}]"), "\"a,b\"");
    assert_eq!(eval_one("StringSlice[\"abcdef\", 2, 3]"), "\"cde\"");
    assert_eq!(eval_one("StringSlice[\"abcdef\", -2, 2]"), "\"ef\"");
    assert_eq!(eval_one("IndexOf[\"hello\", \"l\"]"), "2");
    assert_eq!(eval_one("IndexOf[\"hello\", \"l\", 3]"), "3");
    assert_eq!(eval_one("LastIndexOf[\"hello\", \"l\"]"), "3");
    assert_eq!(eval_one("LastIndexOf[\"hello\", \"l\", 2]"), "2");
    assert_eq!(eval_one("StringTrimLeft[\"  hi  \"]"), "\"hi  \"");
    assert_eq!(eval_one("StringTrimRight[\"  hi  \"]"), "\"  hi\"");
    assert_eq!(eval_one("StringReplaceFirst[\"foo foo\", \"foo\", \"bar\"]"), "\"bar foo\"");
    assert_eq!(eval_one("StringRepeat[\"ab\", 3]"), "\"ababab\"");
    assert_eq!(eval_one("IsBlank[\"   \"]"), "True");
    assert_eq!(eval_one("IsBlank[\" a \"]"), "False");
    assert_eq!(eval_one("Capitalize[\"hELLo\"]"), "\"Hello\"");
    assert_eq!(eval_one("TitleCase[\"hELLo   woRLD\"]"), "\"Hello   World\"");
    assert_eq!(eval_one("StringTrimPrefix[\"foobar\", \"foo\"]"), "\"bar\"");
    assert_eq!(eval_one("StringTrimPrefix[\"foobar\", \"bar\"]"), "\"foobar\"");
    assert_eq!(eval_one("StringTrimSuffix[\"foobar\", \"bar\"]"), "\"foo\"");
    assert_eq!(eval_one("StringTrimSuffix[\"foobar\", \"foo\"]"), "\"foobar\"");
    assert_eq!(eval_one("StringTrimChars[\"--ab--\", \"-a\"]"), "\"b\"");
    assert_eq!(eval_one("EqualsIgnoreCase[\"Hello\", \"hELLo\"]"), "True");
    assert_eq!(
        eval_one("SplitLines[JoinLines[{\"a\", \"b\", \"c\", \"d\"}]]"),
        "{\"a\", \"b\", \"c\", \"d\"}"
    );
    assert_eq!(eval_one("StringChars[\"ab\"]"), "{\"a\", \"b\"}");
    assert_eq!(eval_one("StringFromChars[{\"a\", \"b\", \"c\"}]"), "\"abc\"");
}

#[test]
fn stdlib_string_templating() {
    // Interpolate evaluates inner expressions
    assert_eq!(eval_one("StringInterpolate[\"sum={Total[{1,2,3}]}\"]"), "\"sum=6\"");
    // Interpolate with bindings
    // Use backslash to avoid parser-time interpolation of braces
    assert_eq!(
        eval_one("StringInterpolateWith[\"Hello \\{name\\}!\", <|name->\"Lyra\"|>]"),
        "\"Hello Lyra!\""
    );
    // Format with positional args
    assert_eq!(eval_one("StringFormat[\"\\{0\\}-\\{1\\}\", {\"a\", 123}]"), "\"a-123\"");
    // Format with map
    assert_eq!(
        eval_one("StringFormatMap[\"\\{a\\}:\\{b\\}\", <|\"a\"->1, \"b\"->\"x\"|>]"),
        "\"1:x\""
    );
}

#[test]
fn dataset_dispatch_minimal_list_paths() {
    // Distinct on lists dispatches to Unique
    assert_eq!(eval_one("Distinct[{1,1,2}]"), "{1, 2}");
    // Offset and Head dispatch for lists still work via Drop/Take
    assert_eq!(eval_one("Head[{1,2,3}, 2]"), "{1, 2}");
    assert_eq!(eval_one("Offset[{1,2,3}, 1]"), "{2, 3}");
}

#[test]
fn dataset_dispatch_dataset_ops() {
    // DistinctBy over a dataset
    assert_eq!(
        eval_one("Count[DistinctBy[DatasetFromRows[{<|\"id\"->1|>,<|\"id\"->1|>}], {\"id\"}]]"),
        "1"
    );
}

#[test]
fn stdlib_template_render_basic() {
    // Basic Mustache replacement
    assert_eq!(
        eval_one("TemplateRender[\"Hello {{name}}!\", <|\"name\"->\"Lyra\"|>]"),
        "\"Hello Lyra!\""
    );
    // HTML-escape only variable content for double braces
    assert_eq!(
        eval_one("TemplateRender[\"<i>{{name}}</i>\", <|\"name\"->\"<Lyra>\"|>]"),
        "\"<i>&lt;Lyra&gt;</i>\""
    );
    // Triple mustache unescaped
    assert_eq!(
        eval_one("TemplateRender[\"<i>{{{name}}}</i>\", <|\"name\"->\"<Lyra>\"|>]"),
        "\"<i><Lyra></i>\""
    );
}

#[test]
fn with_basic_binding() {
    assert_eq!(eval_one("With[<|\"x\"->1|>, x]"), "1");
}

// TemplateRender has separate examples in README; parser string escaping makes inline tests noisy.

#[test]
fn echo_attribute_tests() {
    assert_eq!(eval_one("OrderlessEcho[b, a]"), "{a, b}");
    assert_eq!(eval_one("FlatEcho[FlatEcho[1,2], 3]"), "{1, 2, 3}");
    assert_eq!(eval_one("FlatOrderlessEcho[FlatOrderlessEcho[b,a], c]"), "{a, b, c}");
}

#[test]
fn replace_all_simple_rule() {
    // With HoldFirst semantics for ReplaceAll, target is not evaluated before matching,
    // so Plus[1,2] does not match 3 and evaluates after replacement to 3.
    assert_eq!(eval_one("ReplaceAll[Plus[1,2], 3 -> 42]"), "3");
}

#[test]
fn replace_first_in_list_on_items() {
    assert_eq!(eval_one("ReplaceFirst[{1,2,3}, _Integer -> 9]"), "{9, 2, 3}");
}

#[test]
fn merge_with_combiner() {
    assert_eq!(
        eval_one("Merge[<|\"a\"->1|>, <|\"a\"->10, \"b\"->2|>, (xs)=>Total[xs]]"),
        "<|\"a\" -> 11, \"b\" -> 2|>"
    );
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

#[test]
fn tools_dry_run_and_export_openai() {
    // Register a simple http.get spec with schema
    let _ = eval_one("ToolsRegister[<|\"id\"->\"net.http.get@1\", \"name\"->\"http.get\", \"impl\"->\"HttpGet\", \"params\"->{\"url\", \"retries\"}, \"summary\"->\"Fetch a URL\", \"tags\"->{\"http\", \"json\"}, \"effects\"->{\"net\"}, \"input_schema\"-><|\"type\"->\"object\", \"properties\"-><|\"url\"-><|\"type\"->\"string\"|>, \"retries\"-><|\"type\"->\"integer\"|>|>, \"required\"->{\"url\"}|> |>]");

    // Missing required "retries" should be ok (since only url required), but wrong type should fail
    let res =
        eval_one("ToolsDryRun[\"net.http.get@1\", <|\"url\"->\"https://a\", \"retries\"->\"x\"|>]");
    assert!(res.contains("\"ok\" -> False") || res.contains("\"ok\"->False"));
    assert!(res.contains("\"errors\""));

    // Export OpenAI and check function name present (parameters structure may vary)
    let export = eval_one("ToolsExportOpenAI[]");
    assert!(export.contains("\"name\" -> \"http_get\""));
}

#[test]
fn tools_capabilities_and_filters() {
    // Register a net tool with effect "net"
    let _ = eval_one("ToolsRegister[<|\"id\"->\"net.http.get@1\", \"name\"->\"http.get\", \"impl\"->\"HttpGet\", \"params\"->{\"url\"}, \"effects\"->{\"net\"}|>]");
    // Without granting capabilities, it is discoverable
    let all = eval_one("ToolsList[<|\"effects\"->{\"net\"}|>]");
    assert!(all.contains("http.get"));
    // Explicit capabilities filter should allow viewing matching tools
    let filtered = eval_one("ToolsList[<|\"capabilities\"->{\"net\"}|>]");
    assert!(filtered.contains("http.get"));
    let filtered_fs = eval_one("ToolsList[<|\"capabilities\"->{\"fs\"}|>]");
    assert!(!filtered_fs.contains("http.get"));
}

#[test]
fn tools_schema_validation_for_stdlib_specs() {
    // Not expects boolean
    let not_bad = eval_one("ToolsDryRun[\"Not\", <|\"x\"->1|>]");
    assert!(not_bad.contains("\"ok\" -> False") || not_bad.contains("\"ok\"->False"));
    // EvenQ expects integer
    let even_bad = eval_one("ToolsDryRun[\"EvenQ\", <|\"n\"->\"2\"|>]");
    assert!(even_bad.contains("\"ok\" -> False") || even_bad.contains("\"ok\"->False"));
    // Join expects arrays a and b
    let join_bad = eval_one("ToolsDryRun[\"Join\", <|\"a\"->1, \"b\"->2|>]");
    assert!(join_bad.contains("\"ok\" -> False") || join_bad.contains("\"ok\"->False"));
    // Join with proper arrays should succeed
    let join_ok = eval_one("ToolsDryRun[\"Join\", <|\"a\"->{1}, \"b\"->{2}|>]");
    assert!(join_ok.contains("\"ok\" -> True") || join_ok.contains("\"ok\"->True"));
    // Range expects integers a and b
    let range_bad = eval_one("ToolsDryRun[\"Range\", <|\"a\"->1, \"b\"->\"9\"|>]");
    assert!(range_bad.contains("\"ok\" -> False") || range_bad.contains("\"ok\"->False"));
}
