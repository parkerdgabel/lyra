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
fn text_find_count_lines_replace() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);

    let dir = format!("{}/target/test_text", env!("CARGO_MANIFEST_DIR"));
    let f1 = format!("{}/a.txt", dir);
    let f2 = format!("{}/b.txt", dir);
    write_file(&f1, "Lyra is great\nlyra rocks\nno match here\n");
    write_file(&f2, "Something else\nAnother line\n");

    // Find (literal, case-insensitive) across files
    let res = ev.eval(Value::expr(
        Value::Symbol("TextFind".into()),
        vec![
            Value::Assoc(
                [(
                    "paths".into(),
                    Value::List(vec![Value::String(f1.clone()), Value::String(f2.clone())]),
                )]
                .into_iter()
                .collect(),
            ),
            Value::Assoc([("literal".into(), Value::String("lyra".into()))].into_iter().collect()),
            Value::Assoc(
                [
                    ("caseInsensitive".into(), Value::Boolean(true)),
                    (
                        "context".into(),
                        Value::Assoc(
                            [
                                ("before".into(), Value::Integer(1)),
                                ("after".into(), Value::Integer(1)),
                            ]
                            .into_iter()
                            .collect(),
                        ),
                    ),
                ]
                .into_iter()
                .collect(),
            ),
        ],
    ));
    let (mut total, mut files_with) = (0i64, 0i64);
    if let Value::Assoc(m) = res {
        if let Some(Value::Assoc(s)) = m.get("summary") {
            if let Some(Value::Integer(t)) = s.get("totalMatches") {
                total = *t;
            }
            if let Some(Value::Integer(fw)) = s.get("filesWithMatch") {
                files_with = *fw;
            }
        }
    } else {
        panic!("TextFind invalid result");
    }
    assert_eq!(total, 2);
    assert_eq!(files_with, 1);

    // Count
    let cnt = ev.eval(Value::expr(
        Value::Symbol("TextCount".into()),
        vec![
            Value::Assoc(
                [(
                    "paths".into(),
                    Value::List(vec![Value::String(f1.clone()), Value::String(f2.clone())]),
                )]
                .into_iter()
                .collect(),
            ),
            Value::Assoc(
                [("regex".into(), Value::String("(?i)lyra".into()))].into_iter().collect(),
            ),
        ],
    ));
    let mut total_c = -1i64;
    if let Value::Assoc(m) = cnt {
        if let Some(Value::Integer(t)) = m.get("total") {
            total_c = *t;
        }
    } else {
        panic!("TextCount invalid");
    }
    assert_eq!(total_c, 2);

    // Lines
    let lines = ev.eval(Value::expr(
        Value::Symbol("TextLines".into()),
        vec![
            Value::Assoc([("path".into(), Value::String(f1.clone()))].into_iter().collect()),
            Value::Assoc([("literal".into(), Value::String("Lyra".into()))].into_iter().collect()),
        ],
    ));
    let mut got_line = false;
    if let Value::Assoc(m) = lines {
        if let Some(Value::List(ls)) = m.get("lines") {
            got_line = !ls.is_empty();
        }
    } else {
        panic!("TextLines invalid");
    }
    assert!(got_line);

    // Replace dry-run
    let rep = ev.eval(Value::expr(
        Value::Symbol("TextReplace".into()),
        vec![
            Value::Assoc([("path".into(), Value::String(f1.clone()))].into_iter().collect()),
            Value::Assoc(
                [("regex".into(), Value::String("(?i)lyra".into()))].into_iter().collect(),
            ),
            Value::String("LYRA".into()),
            Value::Assoc([("dryRun".into(), Value::Boolean(true))].into_iter().collect()),
        ],
    ));
    if let Value::Assoc(m) = rep {
        if let Some(Value::List(es)) = m.get("edits") {
            assert!(!es.is_empty());
        } else {
            panic!("no edits");
        }
    } else {
        panic!("TextReplace invalid");
    }

    // Replace in-place
    let _rep2 = ev.eval(Value::expr(
        Value::Symbol("TextReplace".into()),
        vec![
            Value::Assoc([("path".into(), Value::String(f1.clone()))].into_iter().collect()),
            Value::Assoc([("literal".into(), Value::String("rocks".into()))].into_iter().collect()),
            Value::String("rules".into()),
            Value::Assoc([("inPlace".into(), Value::Boolean(true))].into_iter().collect()),
        ],
    ));
    let new_f1 = std::fs::read_to_string(&f1).unwrap();
    assert!(new_f1.contains("rules"));
}
