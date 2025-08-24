use lyra::stdlib::StandardLibrary;
use lyra::vm::Value;

fn get(stdlib: &StandardLibrary, name: &str) -> fn(&[Value]) -> lyra::vm::VmResult<Value> {
    stdlib.get_function(name).expect(name)
}

fn assert_object(v: Value, name: &str) -> std::collections::HashMap<String, Value> {
    match v {
        Value::Object(m) => m,
        other => panic!("expected Association from {} but got {:?}", name, other),
    }
}

fn main() {
    let stdlib = StandardLibrary::new();

    // FFT
    let fft = get(&stdlib, "FFT");
    let sig = Value::List(vec![Value::Real(1.0), Value::Real(0.0), Value::Real(-1.0), Value::Real(0.0)]);
    let m = assert_object(fft(&[sig]).expect("FFT ok"), "FFT");
    assert_eq!(m.get("method").and_then(|v| v.as_string()).unwrap(), "FFT");

    // FiniteDifference
    let finite_diff = get(&stdlib, "FiniteDifference");
    let d = assert_object(
        finite_diff(&[Value::Function("Sin".into()), Value::Real(1.0), Value::Real(1e-6)])
            .expect("FiniteDifference ok"),
        "FiniteDifference",
    );
    assert_eq!(d.get("method").and_then(|v| v.as_string()).unwrap(), "Central");

    // HashFunction
    let hash = get(&stdlib, "HashFunction");
    let h = assert_object(hash(&[Value::String("abc".into()), Value::String("sha256".into())]).expect("Hash ok"), "HashFunction");
    assert_eq!(h.get("algorithm").and_then(|v| v.as_string()).unwrap(), "sha256");

    // RAGQuery
    let rag_query = get(&stdlib, "RAGQuery");
    let rq = assert_object(
        rag_query(&[Value::String("What is AI?".into()), Value::List(vec![]), Value::String("gpt-4".into()), Value::String("tmpl".into())])
            .expect("RAGQuery ok"),
        "RAGQuery",
    );
    assert!(rq.get("answer").is_some());

    println!("integration_smoke: OK");
}

