use lyra::stdlib::common::options as opt;
use lyra::vm::{Value, VmError};
use std::collections::HashMap;

fn obj(pairs: Vec<(&str, Value)>) -> HashMap<String, Value> {
    let mut m = HashMap::new();
    for (k, v) in pairs { m.insert(k.to_string(), v); }
    m
}

#[test]
fn test_get_bool_and_int_real_string_enum() {
    let opts = obj(vec![
        ("flag", Value::Boolean(true)),
        ("count", Value::Integer(42)),
        ("pi", Value::Real(3.14)),
        ("name", Value::String("lyra".into())),
        ("mode", Value::String("Fast".into())),
    ]);

    assert_eq!(opt::get_bool(&opts, "flag", false).unwrap(), true);
    assert_eq!(opt::get_bool(&opts, "missing", true).unwrap(), true);
    assert_eq!(opt::get_int(&opts, "count", 0).unwrap(), 42);
    assert_eq!(opt::get_int(&opts, "missing", 7).unwrap(), 7);
    assert!((opt::get_real(&opts, "pi", 0.0).unwrap() - 3.14).abs() < 1e-9);
    assert_eq!(opt::get_real(&opts, "missing", 2.0).unwrap(), 2.0);
    assert_eq!(opt::get_string(&opts, "name", "default").unwrap(), "lyra");
    assert_eq!(opt::get_string(&opts, "missing", "default").unwrap(), "default");
    assert_eq!(opt::get_enum(&opts, "mode", &["fast", "slow"], "slow").unwrap(), "Fast");
}

#[test]
fn test_require_keys_and_type_errors() {
    let opts = obj(vec![
        ("ok", Value::Boolean(true)),
        ("count", Value::Integer(1)),
    ]);

    // require_keys should pass
    assert!(opt::require_keys(&opts, &["ok", "count"]).is_ok());

    // type error cases
    let bad_bool = obj(vec![("flag", Value::String("no".into()))]);
    let err = opt::get_bool(&bad_bool, "flag", true).unwrap_err();
    assert!(matches!(err, VmError::TypeError { .. }));

    let err = opt::get_int(&obj(vec![("n", Value::String("x".into()))]), "n", 0).unwrap_err();
    assert!(matches!(err, VmError::TypeError { .. }));
}
