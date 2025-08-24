use lyra::stdlib::StandardLibrary;
use lyra::vm::Value;

fn get(stdlib: &StandardLibrary, name: &str) -> fn(&[Value]) -> lyra::vm::VmResult<Value> {
    stdlib.get_function(name).expect(name)
}

#[test]
fn schema_detects_spectral() {
    let stdlib = StandardLibrary::new();
    let schema = get(&stdlib, "Schema");

    let mut m = std::collections::HashMap::new();
    m.insert("frequencies".to_string(), Value::List(vec![Value::Real(0.0)]));
    m.insert("magnitudes".to_string(), Value::List(vec![Value::Real(1.0)]));
    m.insert("phases".to_string(), Value::List(vec![Value::Real(0.0)]));
    m.insert("sampleRate".to_string(), Value::Real(1000.0));
    m.insert("method".to_string(), Value::String("FFT".into()));
    let v = Value::Object(m);
    let out = schema(&[v]).unwrap();
    match out { Value::Object(o) => {
        assert_eq!(o.get("name").and_then(|v| v.as_string()).unwrap(), "SpectralResult/v1");
    } _ => panic!("expected object") }
}

#[test]
fn schema_detects_root() {
    let stdlib = StandardLibrary::new();
    let schema = get(&stdlib, "Schema");
    let mut m = std::collections::HashMap::new();
    m.insert("root".to_string(), Value::Real(3.1415));
    m.insert("iterations".to_string(), Value::Integer(8));
    m.insert("functionValue".to_string(), Value::Real(0.0));
    m.insert("converged".to_string(), Value::Boolean(true));
    m.insert("errorEstimate".to_string(), Value::Real(1e-6));
    m.insert("method".to_string(), Value::String("Bisection".into()));
    let out = schema(&[Value::Object(m)]).unwrap();
    match out { Value::Object(o) => {
        assert_eq!(o.get("name").and_then(|v| v.as_string()).unwrap(), "RootResult/v1");
    } _ => panic!("expected object") }
}

