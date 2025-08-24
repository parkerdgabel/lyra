use lyra::stdlib::utilities::serialization::{yaml_parse, yaml_stringify, toml_parse, toml_stringify};
use lyra::vm::Value;

#[test]
fn test_yaml_parse_and_stringify() {
    let y = "a: 1\nb: [x, y]\n".to_string();
    let v = yaml_parse(&[Value::String(y.clone())]).unwrap();
    if let Value::Object(m) = &v {
        assert_eq!(m.get("a"), Some(&Value::Integer(1)));
        if let Some(Value::List(lst)) = m.get("b") {
            assert_eq!(lst.len(), 2);
        } else { panic!("expected list"); }
    } else { panic!("expected object"); }

    // stringify roundtrip should produce YAML string
    let s = yaml_stringify(&[v]).unwrap();
    if let Value::String(out) = s { assert!(out.contains("a:")); } else { panic!("expected string"); }
}

#[test]
fn test_toml_parse_and_stringify() {
    let t = "title = \"T\"\n[db]\nport = 123\n".to_string();
    let v = toml_parse(&[Value::String(t)]).unwrap();
    if let Value::Object(m) = &v { assert!(m.contains_key("db")); } else { panic!("expected object"); }
    let s = toml_stringify(&[v]).unwrap();
    if let Value::String(out) = s { assert!(out.contains("db")); } else { panic!("expected string"); }
}

