#![cfg(feature = "cloud")]

use lyra::vm::Value;
use lyra::stdlib::io::object_store as os;

// These tests require local emulators and are ignored by default.

#[ignore]
#[test]
fn s3_localstack_presign() {
    // Requires localstack running at http://localhost:4566 and credentials
    let uri = "s3://test-bucket/demo".to_string();
    let opts = Value::Object([
        ("method".to_string(), Value::String("GET".to_string())),
        ("expiresSec".to_string(), Value::Integer(60)),
        ("providerOpts".to_string(), Value::Object([
            ("endpoint".to_string(), Value::String("http://localhost:4566".to_string())),
            ("region".to_string(), Value::String("us-east-1".to_string())),
        ].into_iter().collect())),
    ].into_iter().collect());
    let url = os::object_store_presign(&[Value::String(uri), opts]).unwrap();
    if let Value::String(u) = url { assert!(u.contains("localhost:4566")); } else { panic!("expected url string") }
}

#[ignore]
#[test]
fn gcs_emulator_list() {
    // Requires GCS emulator; set STORAGE_EMULATOR_HOST accordingly or use providerOpts.emulatorHost
    let uri = "gs://test-bucket/".to_string();
    let opts = Value::Object([
        ("providerOpts".to_string(), Value::Object([
            ("emulatorHost".to_string(), Value::String("http://localhost:4443".to_string())),
        ].into_iter().collect())),
    ].into_iter().collect());
    let _ = os::object_store_list(&[Value::String(uri), opts]);
}

#[ignore]
#[test]
fn azure_azurite_head() {
    // Requires Azurite running and a container; provide account/accessKey/blobEndpoint
    let uri = "az://devstoreaccount1/test-blob".to_string();
    let opts = Value::Object([
        ("providerOpts".to_string(), Value::Object([
            ("account".to_string(), Value::String("devstoreaccount1".to_string())),
            ("accessKey".to_string(), Value::String("Eby8vdM02xNOcqFe......==".to_string())),
            ("blobEndpoint".to_string(), Value::String("http://127.0.0.1:10000/devstoreaccount1".to_string())),
        ].into_iter().collect())),
    ].into_iter().collect());
    let _ = os::object_store_head(&[Value::String(uri), opts]);
}

