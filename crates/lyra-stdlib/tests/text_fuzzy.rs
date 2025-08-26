#![cfg(feature = "text_fuzzy")]

use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;

#[test]
fn fuzzy_list_basic() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let res = ev.eval(Value::expr(
        Value::Symbol("FuzzyFindInList".into()),
        vec![
            Value::List(vec![
                Value::String("foobar".into()),
                Value::String("bar".into()),
                Value::String("foo".into()),
            ]),
            Value::String("fb".into()),
        ],
    ));
    let mut ok = false;
    if let Value::Assoc(m) = res {
        if let Some(Value::List(items)) = m.get("items") {
            ok = !items.is_empty();
        }
    }
    assert!(ok);
}
