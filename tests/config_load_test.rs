use lyra::stdlib::utilities::config::config_load;
use lyra::vm::Value;
use std::fs;

#[test]
fn test_config_load_yaml_and_toml_auto() {
    let dir = tempfile::tempdir().unwrap();
    let ypath = dir.path().join("c.yaml");
    fs::write(&ypath, b"root:\n  a: 1\n").unwrap();
    let tpath = dir.path().join("c.toml");
    fs::write(&tpath, b"[root]\na = 2\n").unwrap();

    let yv = config_load(&[Value::String(ypath.to_string_lossy().to_string())]).unwrap();
    if let Value::Object(m) = yv { assert!(m.contains_key("root")); } else { panic!("expected object"); }

    let tv = config_load(&[Value::String(tpath.to_string_lossy().to_string())]).unwrap();
    if let Value::Object(m) = tv { assert!(m.contains_key("root")); } else { panic!("expected object"); }
}

