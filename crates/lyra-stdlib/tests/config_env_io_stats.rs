use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;

fn ev_all() -> Evaluator {
    let mut e = Evaluator::new();
    stdlib::register_all(&mut e);
    e
}

fn sym(s: &str) -> Value { Value::Symbol(s.into()) }

#[test]
fn load_config_env_and_files() {
    let mut ev = ev_all();
    // temp files
    let dir = tempfile::tempdir().unwrap();
    let p_json = dir.path().join("a.json");
    let p_yaml = dir.path().join("b.yaml");
    let p_toml = dir.path().join("c.toml");
    std::fs::write(&p_json, "{\"ja\":1}").unwrap();
    std::fs::write(&p_yaml, "yb: 2\n").unwrap();
    std::fs::write(&p_toml, "tc = 3\n").unwrap();

    std::env::set_var("APP_TEST_ONLY", "yes");

    let cfg = ev.eval(Value::Expr {
        head: Box::new(sym("LoadConfig")),
        args: vec![Value::Assoc(std::collections::HashMap::from([
            ("files".into(), Value::List(vec![
                Value::String(p_json.to_string_lossy().to_string()),
                Value::String(p_yaml.to_string_lossy().to_string()),
                Value::String(p_toml.to_string_lossy().to_string()),
            ])),
            ("envPrefix".into(), Value::String("APP_TEST_".into())),
            ("overrides".into(), Value::Assoc(std::collections::HashMap::from([
                ("ovr".into(), Value::String("ok".into())),
            ]))),
        ]))],
    });
    let m = match cfg { Value::Assoc(m) => m, other => panic!("expected assoc, got {:?}", other) };
    assert_eq!(m.get("ja"), Some(&Value::Integer(1)));
    assert_eq!(m.get("yb"), Some(&Value::Integer(2)));
    assert_eq!(m.get("tc"), Some(&Value::Integer(3)));
    assert_eq!(m.get("ovr"), Some(&Value::String("ok".into())));
    assert_eq!(m.get(&"APP_TEST_ONLY".to_string()), Some(&Value::String("yes".into())));
}

#[test]
fn env_read_defaults_and_required() {
    let mut ev = ev_all();
    std::env::set_var("FOO", "bar");
    let ok = ev.eval(Value::Expr {
        head: Box::new(sym("Env")),
        args: vec![Value::Assoc(std::collections::HashMap::from([
            ("keys".into(), Value::List(vec![Value::String("FOO".into()), Value::String("BAZ".into())])),
            ("required".into(), Value::List(vec![Value::String("FOO".into()), Value::String("BAZ".into())])),
            ("defaults".into(), Value::Assoc(std::collections::HashMap::from([
                ("BAZ".into(), Value::String("x".into())),
            ]))),
        ]))],
    });
    match ok {
        Value::Assoc(m) => {
            assert_eq!(m.get("FOO"), Some(&Value::String("bar".into())));
            assert_eq!(m.get("BAZ"), Some(&Value::String("x".into())));
        }
        other => panic!("expected assoc, got {:?}", other),
    }

    let fail = ev.eval(Value::Expr {
        head: Box::new(sym("Env")),
        args: vec![Value::Assoc(std::collections::HashMap::from([
            ("keys".into(), Value::List(vec![Value::String("MISSING".into())])),
            ("required".into(), Value::List(vec![Value::String("MISSING".into())])),
        ]))],
    });
    match fail { Value::Assoc(m) => assert_eq!(m.get("message"), Some(&Value::String("Failure".into()))), _ => panic!("expected Failure") }
}

#[test]
fn dotenv_and_json_file_io() {
    let mut ev = ev_all();
    let dir = tempfile::tempdir().unwrap();
    // dotenv
    let p_env = dir.path().join(".env");
    std::fs::write(&p_env, "LD_A=1\nLD_B=2\n").unwrap();
    let res = ev.eval(Value::Expr {
        head: Box::new(sym("LoadDotenv")),
        args: vec![Value::Assoc(std::collections::HashMap::from([
            ("path".into(), Value::String(p_env.to_string_lossy().to_string())),
            ("override".into(), Value::Boolean(true)),
        ]))],
    });
    match res { Value::Assoc(m) => assert!(matches!(m.get("loaded"), Some(Value::Integer(n)) if *n == 2)), other => panic!("expected assoc: {:?}", other) }
    assert_eq!(std::env::var("LD_A").unwrap(), "1");

    // WriteJson/ReadJson
    let p_json = dir.path().join("z.json");
    let v = Value::Assoc(std::collections::HashMap::from([(String::from("a"), Value::Integer(1))]));
    let w = ev.eval(Value::Expr { head: Box::new(sym("WriteJson")), args: vec![Value::String(p_json.to_string_lossy().to_string()), v.clone()] });
    assert_eq!(w, Value::Boolean(true));
    let r = ev.eval(Value::Expr { head: Box::new(sym("ReadJson")), args: vec![Value::String(p_json.to_string_lossy().to_string())] });
    assert_eq!(r, v);
}

#[test]
fn stats_helpers() {
    let mut ev = ev_all();
    // DescriptiveStats
    let s = ev.eval(Value::Expr { head: Box::new(sym("DescriptiveStats")), args: vec![Value::List(vec![1,2,3,4].into_iter().map(Value::Integer).collect())] });
    match s { Value::Assoc(m) => {
        assert_eq!(m.get("count"), Some(&Value::Integer(4)));
        assert!(matches!(m.get("mean"), Some(Value::Real(x)) if (*x-2.5).abs() < 1e-9));
        assert!(matches!(m.get("p25"), Some(Value::Real(x)) if (*x-1.75).abs() < 1e-9));
    } _ => panic!("expected assoc") }

    // RollingStats
    let r = ev.eval(Value::Expr { head: Box::new(sym("RollingStats")), args: vec![Value::List(vec![1,2,3,4].into_iter().map(Value::Integer).collect()), Value::Integer(3)] });
    match r { Value::List(vs) => {
        assert_eq!(vs.len(), 2);
        if let Value::Assoc(m0) = &vs[0] { assert!(matches!(m0.get("mean"), Some(Value::Real(x)) if (*x-2.0).abs() < 1e-9)); } else { panic!("expected assoc"); }
        if let Value::Assoc(m1) = &vs[1] { assert!(matches!(m1.get("mean"), Some(Value::Real(x)) if (*x-3.0).abs() < 1e-9)); } else { panic!("expected assoc"); }
    } _ => panic!("expected list") }

    // RandomSample determinism with seed
    let a = ev.eval(Value::Expr { head: Box::new(sym("RandomSample")), args: vec![Value::List(vec![1,2,3,4].into_iter().map(Value::Integer).collect()), Value::Integer(2), Value::Assoc(std::collections::HashMap::from([(String::from("seed"), Value::Integer(1))]))] });
    let b = ev.eval(Value::Expr { head: Box::new(sym("RandomSample")), args: vec![Value::List(vec![1,2,3,4].into_iter().map(Value::Integer).collect()), Value::Integer(2), Value::Assoc(std::collections::HashMap::from([(String::from("seed"), Value::Integer(1))]))] });
    assert_eq!(a, b);
}

