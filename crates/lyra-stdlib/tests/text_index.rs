#![cfg(feature = "text_index")]

use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;

#[test]
fn index_create_add_search() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);

    let dir = format!("{}/target/test_index/basic", env!("CARGO_MANIFEST_DIR"));
    // Create index
    let _ = ev.eval(Value::expr(Value::Symbol("Index".into()), vec![Value::String(dir.clone())]));
    // Add docs
    let _ = ev.eval(Value::expr(
        Value::Symbol("IndexAdd".into()),
        vec![
            Value::String(dir.clone()),
            Value::List(vec![
                Value::Assoc(
                    [
                        ("id".into(), Value::String("1".into())),
                        ("body".into(), Value::String("Lyra text search is neat".into())),
                    ]
                    .into_iter()
                    .collect(),
                ),
                Value::Assoc(
                    [
                        ("id".into(), Value::String("2".into())),
                        ("body".into(), Value::String("Full text indexing with tantivy".into())),
                    ]
                    .into_iter()
                    .collect(),
                ),
            ]),
        ],
    ));
    // Search
    let res = ev.eval(Value::expr(
        Value::Symbol("IndexSearch".into()),
        vec![Value::String(dir), Value::String("lyra OR tantivy".into())],
    ));
    let mut ok = false;
    if let Value::Assoc(m) = res {
        if let Some(Value::List(hits)) = m.get("hits") {
            ok = !hits.is_empty();
        }
    }
    assert!(ok);
}
