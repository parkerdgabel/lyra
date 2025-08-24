use lyra::stdlib::StandardLibrary;
use lyra::vm::{Value};

fn get(stdlib: &StandardLibrary, name: &str) -> fn(&[Value]) -> lyra::vm::VmResult<Value> {
    stdlib.get_function(name).expect(name)
}

#[test]
fn smoke_fft_and_periodogram() {
    let stdlib = StandardLibrary::new();
    let fft = get(&stdlib, "FFT");
    let periodogram = get(&stdlib, "Periodogram");

    let sig = Value::List(vec![Value::Real(1.0), Value::Real(0.0), Value::Real(-1.0), Value::Real(0.0)]);
    let v = fft(&[sig.clone()]).expect("FFT ok");
    match v {
        Value::Object(m) => {
            assert_eq!(m.get("method").and_then(|v| v.as_string()).unwrap(), "FFT");
            assert!(m.get("frequencies").is_some());
            assert!(m.get("magnitudes").is_some());
            assert!(m.get("phases").is_some());
        }
        other => panic!("expected Association from FFT, got {:?}", other),
    }

    let pv = periodogram(&[sig]).expect("Periodogram ok");
    match pv {
        Value::Object(m) => {
            assert_eq!(m.get("method").and_then(|v| v.as_string()).unwrap(), "Periodogram");
            assert!(m.get("frequencies").is_some());
            assert!(m.get("magnitudes").is_some());
        }
        other => panic!("expected Association from Periodogram, got {:?}", other),
    }
}

#[test]
fn smoke_numerical_results() {
    let stdlib = StandardLibrary::new();
    let finite_diff = get(&stdlib, "FiniteDifference");
    let bisection = get(&stdlib, "Bisection");
    let simpson = get(&stdlib, "Simpson");

    // FiniteDifference[Sin, 1.0, 1e-6]
    let v = finite_diff(&[Value::Function("Sin".into()), Value::Real(1.0), Value::Real(1e-6)])
        .expect("FiniteDifference ok");
    match v {
        Value::Object(m) => {
            assert!(m.get("value").is_some());
            assert_eq!(m.get("method").and_then(|v| v.as_string()).unwrap(), "Central");
        }
        other => panic!("expected Association from FiniteDifference, got {:?}", other),
    }

    // Bisection[Sin, 3.0, 4.0]
    let root = bisection(&[Value::Function("Sin".into()), Value::Real(3.0), Value::Real(4.0)])
        .expect("Bisection ok");
    match root {
        Value::Object(m) => {
            assert!(m.get("root").is_some());
            assert_eq!(m.get("method").and_then(|v| v.as_string()).unwrap(), "Bisection");
        }
        other => panic!("expected Association from Bisection, got {:?}", other),
    }

    // Simpson[Sin, {0, Pi}, 200]
    let sim = simpson(&[Value::Function("Sin".into()), Value::List(vec![Value::Real(0.0), Value::Real(std::f64::consts::PI)]), Value::Integer(200)])
        .expect("Simpson ok");
    match sim {
        Value::Object(m) => {
            assert!(m.get("value").is_some());
            assert_eq!(m.get("method").and_then(|v| v.as_string()).unwrap(), "Simpson");
        }
        other => panic!("expected Association from Simpson, got {:?}", other),
    }
}

#[test]
fn smoke_number_theory_and_rag() {
    let stdlib = StandardLibrary::new();
    let hash = get(&stdlib, "HashFunction");
    let rag_query = get(&stdlib, "RAGQuery");

    let hv = hash(&[Value::String("abc".into()), Value::String("sha256".into())]).expect("HashFunction ok");
    match hv {
        Value::Object(m) => {
            assert_eq!(m.get("algorithm").and_then(|v| v.as_string()).unwrap(), "sha256");
            assert!(m.get("hexString").is_some());
        }
        other => panic!("expected Association from HashFunction, got {:?}", other),
    }

    let contexts = Value::List(vec![]);
    let rv = rag_query(&[Value::String("What is AI?".into()), contexts, Value::String("gpt-4".into()), Value::String("tmpl".into())])
        .expect("RAGQuery ok");
    match rv {
        Value::Object(m) => {
            assert!(m.get("question").is_some());
            assert!(m.get("answer").is_some());
        }
        other => panic!("expected Association from RAGQuery, got {:?}", other),
    }
}

