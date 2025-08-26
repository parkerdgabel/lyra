#![cfg(feature = "text")]

use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;

fn write_file(path: &str, contents: &str) {
    let p = std::path::Path::new(path);
    if let Some(dir) = p.parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    std::fs::write(p, contents.as_bytes()).unwrap();
}

#[test]
fn text_search_grep_mode() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let dir = format!("{}/target/test_text_search", env!("CARGO_MANIFEST_DIR"));
    let f1 = format!("{}/g1.txt", dir);
    write_file(&f1, "alpha\nbeta\nLyra\n");
    let res = ev.eval(Value::expr(
        Value::Symbol("TextSearch".into()),
        vec![
            Value::Assoc([("path".into(), Value::String(f1.clone()))].into_iter().collect()),
            Value::Assoc([("literal".into(), Value::String("Lyra".into()))].into_iter().collect()),
        ],
    ));
    if let Value::Assoc(m) = res {
        assert!(matches!(m.get("engine"), Some(Value::String(s)) if s=="grep"));
    } else {
        panic!("TextSearch invalid");
    }
}

#[test]
#[cfg(feature = "text_fuzzy")]
fn text_search_fuzzy_mode() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let res = ev.eval(Value::expr(
        Value::Symbol("TextSearch".into()),
        vec![
            Value::Assoc(
                [("text".into(), Value::String("hello lyra\ntext search".into()))]
                    .into_iter()
                    .collect(),
            ),
            Value::String("hl".into()),
            Value::Assoc([("mode".into(), Value::String("fuzzy".into()))].into_iter().collect()),
        ],
    ));
    if let Value::Assoc(m) = res {
        assert!(matches!(m.get("engine"), Some(Value::String(s)) if s=="fuzzy"));
    } else {
        panic!("TextSearch invalid");
    }
}

#[test]
#[cfg(feature = "text_index")]
fn text_search_index_mode() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let idx = format!("{}/target/test_index/searchwrap.sqlite", env!("CARGO_MANIFEST_DIR"));
    let _ =
        ev.eval(Value::expr(Value::Symbol("IndexCreate".into()), vec![Value::String(idx.clone())]));
    let _ = ev.eval(Value::expr(
        Value::Symbol("IndexAdd".into()),
        vec![
            Value::String(idx.clone()),
            Value::List(vec![Value::Assoc(
                [
                    ("id".into(), Value::String("1".into())),
                    ("body".into(), Value::String("Lyra search".into())),
                ]
                .into_iter()
                .collect(),
            )]),
        ],
    ));
    let res = ev.eval(Value::expr(
        Value::Symbol("TextSearch".into()),
        vec![
            Value::Assoc([("indexPath".into(), Value::String(idx))].into_iter().collect()),
            Value::Assoc([("q".into(), Value::String("Lyra".into()))].into_iter().collect()),
        ],
    ));
    if let Value::Assoc(m) = res {
        assert!(matches!(m.get("engine"), Some(Value::String(s)) if s=="index"));
    } else {
        panic!("TextSearch invalid");
    }
}

#[test]
fn text_search_count_task() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let dir = format!("{}/target/test_text_search_task", env!("CARGO_MANIFEST_DIR"));
    let f = format!("{}/c1.txt", dir);
    write_file(&f, "one two one\n");
    let res = ev.eval(Value::expr(
        Value::Symbol("TextSearch".into()),
        vec![
            Value::Assoc([("path".into(), Value::String(f.clone()))].into_iter().collect()),
            Value::Assoc([("literal".into(), Value::String("one".into()))].into_iter().collect()),
            Value::Assoc([("task".into(), Value::String("count".into()))].into_iter().collect()),
        ],
    ));
    if let Value::Assoc(m) = res {
        assert!(matches!(m.get("engine"), Some(Value::String(s)) if s=="grep"));
    } else {
        panic!("invalid");
    }
}

#[test]
fn text_search_replace_task() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let dir = format!("{}/target/test_text_search_task2", env!("CARGO_MANIFEST_DIR"));
    let f = format!("{}/r1.txt", dir);
    write_file(&f, "abc def\n");
    let _ = ev.eval(Value::expr(
        Value::Symbol("TextSearch".into()),
        vec![
            Value::Assoc([("path".into(), Value::String(f.clone()))].into_iter().collect()),
            Value::Assoc([("literal".into(), Value::String("def".into()))].into_iter().collect()),
            Value::Assoc(
                [
                    ("task".into(), Value::String("replace".into())),
                    ("replacement".into(), Value::String("xyz".into())),
                    ("inPlace".into(), Value::Boolean(true)),
                ]
                .into_iter()
                .collect(),
            ),
        ],
    ));
    let s = std::fs::read_to_string(&f).unwrap();
    assert!(s.contains("xyz"));
}
