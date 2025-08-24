use lyra::vm::Value;
use lyra::stdlib::network::graphql::graphql_query_response;
use axum::{routing::post, Router};
use std::net::TcpListener;

#[test]
fn test_graphql_query_response_against_local_server() {
    // Local GraphQL-like endpoint: always returns {"data":{"ok":true}}
    let app = Router::new().route("/graphql", post(|| async {
        (axum::http::StatusCode::OK, axum::Json(serde_json::json!({"data": {"ok": true}})))
    }));

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            axum::serve(tokio::net::TcpListener::from_std(listener).unwrap(), app)
                .await
                .unwrap();
        });
    });

    let url = format!("http://{}/graphql", addr);
    let resp_val = graphql_query_response(&[Value::String(url), Value::String("{ ok }".to_string())]).unwrap();
    if let Value::LyObj(obj) = resp_val {
        // Call methods on GraphQLResponse
        let has_errors = obj.call_method("HasErrors", &[]).unwrap();
        assert_eq!(has_errors, Value::Boolean(false));
        let data = obj.call_method("Data", &[]).unwrap();
        match data { Value::Object(m) => assert!(m.contains_key("ok")), _ => panic!("expected object") }
    } else {
        panic!("expected GraphQLResponse LyObj");
    }
}

