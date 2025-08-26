use lyra_runtime::Evaluator;
use lyra_core::pretty::format_value;
use lyra_parser::Parser;
use lyra_stdlib as stdlib;

fn eval_one(src: &str) -> String {
    let mut p = Parser::from_source(src);
    let vals = p.parse_all().expect("parse");
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    format_value(&ev.eval(vals.into_iter().last().unwrap()))
}

#[test]
fn time_basics() {
    let v = eval_one("NumberQ[NowMs[]]");
    assert_eq!(v, "True");
    let s = eval_one("DateFormat[DateTime[\"2020-01-01T00:00:00Z\"], \"%Y-%m-%d\"]");
    assert_eq!(s, "\"2020-01-01\"");
}

#[test]
fn fs_basics() {
    let tf = eval_one("TempFile[]");
    assert!(tf.starts_with("\""));
    let p = tf.trim_matches('"');
    let wrote = eval_one(&format!("WriteBytes[\"{}\", \"abc\"]", p));
    assert_eq!(wrote, "True");
    let s = eval_one(&format!("ReadBytes[\"{}\"]", p));
    assert_eq!(s, "\"abc\"");
}

#[test]
fn process_smoke() {
    // Best-effort: only run if 'echo' exists
    let has_echo = eval_one("CommandExistsQ[\"echo\"]");
    if has_echo == "True" {
        let out = eval_one("Run[\"echo\", {\"hello\"}]");
        assert!(out.contains("hello"));
    }
}

