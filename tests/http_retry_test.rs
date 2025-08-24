use lyra::stdlib::network::http::http_retry;
use lyra::vm::Value;
use axum::{routing::get, Router};
use std::net::TcpListener;
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

#[test]
fn test_http_retry_retries_on_status() {
    // Start a tiny axum server that returns 500 for first 2 requests then 200
    let attempts = Arc::new(AtomicUsize::new(0));
    let attempts_cl = attempts.clone();
    let app = Router::new().route("/", get(move || {
        let n = attempts_cl.fetch_add(1, Ordering::SeqCst);
        async move {
            if n < 2 { (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "err") }
            else { (axum::http::StatusCode::OK, "ok") }
        }
    }));

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async move {
            axum::serve(tokio::net::TcpListener::from_std(listener).unwrap(), app)
                .with_graceful_shutdown(async move { /* no-op */ })
                .await
                .unwrap();
        });
    });

    let url = format!("http://{}/", addr);
    let opts = Value::Object(
        [
            ("retries".to_string(), Value::Integer(3)),
            ("retryOn".to_string(), Value::List(vec![Value::Integer(500)])),
            ("backoff".to_string(), Value::Object([
                ("type".to_string(), Value::String("fixed".to_string())),
                ("baseMs".to_string(), Value::Integer(10)),
                ("maxMs".to_string(), Value::Integer(10)),
                ("jitter".to_string(), Value::Boolean(false)),
            ].into_iter().collect())),
        ]
        .into_iter()
        .collect(),
    );
    let resp = http_retry(&[Value::String(url), opts]).unwrap();
    if let Value::LyObj(obj) = resp {
        if let Some(r) = obj.downcast_ref::<lyra::stdlib::network::core::NetworkResponse>() {
            assert_eq!(r.status, 200);
            assert!(attempts.load(Ordering::SeqCst) >= 3);
        } else { panic!("wrong object") }
    } else { panic!("expected LyObj") }
}

