#![cfg(feature = "cloud")]
use lyra::stdlib::io::object_store::object_store_presign;
use lyra::vm::Value;

// This test requires valid cloud credentials and network access.
// It is ignored by default.
#[test]
#[ignore]
fn test_s3_presign_get() {
    // Set BUCKET and KEY in env or hardcode a test bucket accessible to your creds
    let bucket = std::env::var("LYRA_TEST_S3_BUCKET").expect("set LYRA_TEST_S3_BUCKET");
    let key = std::env::var("LYRA_TEST_S3_KEY").unwrap_or_else(|_| "test.txt".into());
    let uri = format!("s3://{}/{}", bucket, key);
    let url = object_store_presign(&[Value::String(uri), Value::Object(std::collections::HashMap::new())]).unwrap();
    if let Value::String(s) = url { assert!(s.starts_with("http")); } else { panic!("expected string presigned url") }
}

