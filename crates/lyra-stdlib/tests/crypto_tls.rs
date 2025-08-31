use lyra_runtime::Evaluator;

fn eval_str(code: &str) -> lyra_core::value::Value {
    let mut ev = Evaluator::new();
    lyra_stdlib::register_all(&mut ev);
    let mut p = lyra_parser::Parser::from_source(code);
    let exprs = p.parse_all().unwrap();
    let mut out = lyra_core::value::Value::Symbol("Null".into());
    for e in exprs { out = ev.eval(e); }
    out
}

#[test]
fn tls_self_signed_generates_pems() {
    let v = eval_str("TlsSelfSigned[<|hosts->{\"localhost\"}, subject-><|CN->\"localhost\"|>|>]");
    match v {
        lyra_core::value::Value::Assoc(m) => {
            let cert = m.get("certPem").and_then(|v| if let lyra_core::value::Value::String(s)=v { Some(s) } else { None }).cloned().unwrap();
            let key = m.get("privateKeyPem").and_then(|v| if let lyra_core::value::Value::String(s)=v { Some(s) } else { None }).cloned().unwrap();
            assert!(cert.contains("BEGIN CERTIFICATE"));
            assert!(key.contains("BEGIN PRIVATE KEY"));
        }
        other => panic!("unexpected: {}", lyra_core::pretty::format_value(&other)),
    }
}

