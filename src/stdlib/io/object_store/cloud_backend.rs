//! Cloud object store integration behind the `cloud` feature.

#![cfg(feature = "cloud")]

use crate::vm::{Value, VmError, VmResult};
use futures_util::TryStreamExt;
use object_store::path::Path;
use object_store::ObjectStore;
use std::sync::Arc;
use std::collections::HashMap;
use tokio::runtime::Runtime;

fn to_value_bytes(bytes: Vec<u8>) -> Value {
    Value::List(bytes.into_iter().map(|b| Value::Integer(b as i64)).collect())
}

fn parse_bucket_and_key(rest: &str) -> Result<(String, String), VmError> {
    let mut parts = rest.splitn(2, '/');
    let bucket = parts.next().unwrap_or("");
    let key = parts.next().unwrap_or("");
    if bucket.is_empty() {
        return Err(VmError::Runtime("Missing bucket/container name in URI".into()));
    }
    Ok((bucket.to_string(), key.to_string()))
}

fn rt() -> Result<Runtime, VmError> {
    Runtime::new().map_err(|e| VmError::Runtime(format!("Tokio runtime error: {}", e)))
}

struct TempEnvGuard {
    saved: Vec<(String, Option<String>)>,
}

impl TempEnvGuard {
    fn set(vars: &[(String, String)]) -> Self {
        let mut saved = Vec::new();
        for (k, v) in vars {
            let prev = std::env::var(k).ok();
            std::env::set_var(k, v);
            saved.push((k.clone(), prev));
        }
        TempEnvGuard { saved }
    }
}

impl Drop for TempEnvGuard {
    fn drop(&mut self) {
        for (k, prev) in self.saved.drain(..) {
            match prev {
                Some(val) => std::env::set_var(k, val),
                None => std::env::remove_var(k),
            }
        }
    }
}

// S3 implementation using object_store
#[allow(dead_code)]
fn s3_store(bucket: &str, opts: Option<&HashMap<String, Value>>) -> Result<Arc<dyn ObjectStore>, VmError> {
    use object_store::aws::AmazonS3Builder;
    let mut builder = AmazonS3Builder::new().with_bucket_name(bucket);
    if let Some(o) = opts {
        if let Some(Value::String(region)) = o.get("region") { builder = builder.with_region(region); }
        if let Some(Value::String(endpoint)) = o.get("endpoint") { builder = builder.with_endpoint(endpoint); }
        if let Some(Value::String(ak)) = o.get("accessKey") { builder = builder.with_access_key_id(ak); }
        if let Some(Value::String(sk)) = o.get("secretKey") { builder = builder.with_secret_access_key(sk); }
        if let Some(Value::String(tok)) = o.get("sessionToken") { builder = builder.with_token(tok); }
    }
    let store = builder.build().map_err(|e| VmError::Runtime(format!("S3 build error: {}", e)))?;
    Ok(Arc::new(store))
}

#[allow(dead_code)]
fn gcs_store(bucket: &str, _opts: Option<&HashMap<String, Value>>) -> Result<Arc<dyn ObjectStore>, VmError> {
    use object_store::gcp::GoogleCloudStorageBuilder;
    // Allow emulator override via opts.providerOpts.emulatorHost -> STORAGE_EMULATOR_HOST
    let _guard = if let Some(opts) = _opts {
        if let Some(Value::String(host)) = opts.get("emulatorHost") {
            Some(TempEnvGuard::set(&[("STORAGE_EMULATOR_HOST".to_string(), host.clone())]))
        } else { None }
    } else { None };
    let store = GoogleCloudStorageBuilder::new()
        .with_bucket_name(bucket)
        .build()
        .map_err(|e| VmError::Runtime(format!("GCS build error: {}", e)))?;
    Ok(Arc::new(store))
}

#[allow(dead_code)]
fn azure_store(container: &str, _opts: Option<&HashMap<String, Value>>) -> Result<Arc<dyn ObjectStore>, VmError> {
    use object_store::azure::MicrosoftAzureBuilder;
    // Allow emulator/account override via environment-backed configuration
    let mut vars: Vec<(String, String)> = Vec::new();
    if let Some(opts) = _opts {
        if let Some(Value::String(account)) = opts.get("account") {
            vars.push(("AZURE_STORAGE_ACCOUNT".to_string(), account.clone()));
        }
        if let Some(Value::String(key)) = opts.get("accessKey") {
            vars.push(("AZURE_STORAGE_ACCESS_KEY".to_string(), key.clone()));
        }
        if let Some(Value::String(ep)) = opts.get("blobEndpoint") {
            vars.push(("AZURE_STORAGE_BLOB_ENDPOINT".to_string(), ep.clone()));
        }
    }
    let _guard = if vars.is_empty() { None } else { Some(TempEnvGuard::set(&vars)) };
    let store = MicrosoftAzureBuilder::new()
        .with_container_name(container)
        .build()
        .map_err(|e| VmError::Runtime(format!("Azure build error: {}", e)))?;
    Ok(Arc::new(store))
}

pub fn cloud_read_uri(provider: &str, rest: String, opts: Option<&HashMap<String, Value>>) -> VmResult<Value> {
    match provider {
        "s3" => {
            let (bucket, key) = parse_bucket_and_key(&rest)?;
            let store = s3_store(&bucket, opts)?;
            let path = Path::from(key);
            let rt = rt()?;
            let bytes = rt.block_on(async { store.get(&path).await })
                .map_err(|e| VmError::Runtime(format!("S3 get error: {}", e)))?
                .bytes()
                .await
                .map_err(|e| VmError::Runtime(format!("S3 read bytes error: {}", e)))?
                .to_vec();
            Ok(to_value_bytes(bytes))
        }
        "gcs" => {
            let (bucket, key) = parse_bucket_and_key(&rest)?;
            let store = gcs_store(&bucket, opts)?;
            let path = Path::from(key);
            let rt = rt()?;
            let bytes = rt.block_on(async { store.get(&path).await })
                .map_err(|e| VmError::Runtime(format!("GCS get error: {}", e)))?
                .bytes()
                .await
                .map_err(|e| VmError::Runtime(format!("GCS read bytes error: {}", e)))?
                .to_vec();
            Ok(to_value_bytes(bytes))
        }
        "azure" => {
            let (container, key) = parse_bucket_and_key(&rest)?;
            let store = azure_store(&container, opts)?;
            let path = Path::from(key);
            let rt = rt()?;
            let bytes = rt.block_on(async { store.get(&path).await })
                .map_err(|e| VmError::Runtime(format!("Azure get error: {}", e)))?
                .bytes()
                .await
                .map_err(|e| VmError::Runtime(format!("Azure read bytes error: {}", e)))?
                .to_vec();
            Ok(to_value_bytes(bytes))
        }
        _ => Err(VmError::Runtime(format!("{} provider not implemented yet", provider))),
    }
}

pub fn cloud_write_uri(provider: &str, rest: String, bytes: &[u8], opts: Option<&HashMap<String, Value>>) -> VmResult<Value> {
    match provider {
        "s3" => {
            let (bucket, key) = parse_bucket_and_key(&rest)?;
            let store = s3_store(&bucket, opts)?;
            let path = Path::from(key);
            let data = bytes.to_vec();
            let rt = rt()?;
            rt.block_on(async { store.put(&path, data.into()).await })
                .map_err(|e| VmError::Runtime(format!("S3 put error: {}", e)))?;
            Ok(Value::Boolean(true))
        }
        "gcs" => {
            let (bucket, key) = parse_bucket_and_key(&rest)?;
            let store = gcs_store(&bucket, opts)?;
            let path = Path::from(key);
            let data = bytes.to_vec();
            let rt = rt()?;
            rt.block_on(async { store.put(&path, data.into()).await })
                .map_err(|e| VmError::Runtime(format!("GCS put error: {}", e)))?;
            Ok(Value::Boolean(true))
        }
        "azure" => {
            let (container, key) = parse_bucket_and_key(&rest)?;
            let store = azure_store(&container, opts)?;
            let path = Path::from(key);
            let data = bytes.to_vec();
            let rt = rt()?;
            rt.block_on(async { store.put(&path, data.into()).await })
                .map_err(|e| VmError::Runtime(format!("Azure put error: {}", e)))?;
            Ok(Value::Boolean(true))
        }
        _ => Err(VmError::Runtime(format!("{} provider not implemented yet", provider))),
    }
}

pub fn cloud_delete_uri(provider: &str, rest: String, opts: Option<&std::collections::HashMap<String, Value>>) -> VmResult<Value> {
    match provider {
        "s3" => {
            let (bucket, key) = parse_bucket_and_key(&rest)?;
            let store = s3_store(&bucket, opts)?;
            let path = Path::from(key);
            let rt = rt()?;
            rt.block_on(async { store.delete(&path).await })
                .map_err(|e| VmError::Runtime(format!("S3 delete error: {}", e)))?;
            Ok(Value::Boolean(true))
        }
        "gcs" => {
            let (bucket, key) = parse_bucket_and_key(&rest)?;
            let store = gcs_store(&bucket, opts)?;
            let path = Path::from(key);
            let rt = rt()?;
            rt.block_on(async { store.delete(&path).await })
                .map_err(|e| VmError::Runtime(format!("GCS delete error: {}", e)))?;
            Ok(Value::Boolean(true))
        }
        "azure" => {
            let (container, key) = parse_bucket_and_key(&rest)?;
            let store = azure_store(&container, opts)?;
            let path = Path::from(key);
            let rt = rt()?;
            rt.block_on(async { store.delete(&path).await })
                .map_err(|e| VmError::Runtime(format!("Azure delete error: {}", e)))?;
            Ok(Value::Boolean(true))
        }
        _ => Err(VmError::Runtime(format!("{} provider not implemented yet", provider))),
    }
}

pub fn cloud_list_uri(provider: &str, rest: String, opts: Option<&std::collections::HashMap<String, Value>>) -> VmResult<Value> {
    match provider {
        "s3" => {
            let (bucket, prefix) = parse_bucket_and_key(&rest)?;
            let store = s3_store(&bucket, opts)?;
            let path = if prefix.is_empty() { Path::from("") } else { Path::from(prefix) };
            let rt = rt()?;
            let entries = rt
                .block_on(async {
                    let mut stream = store.list(Some(&path));
                    let mut out = Vec::new();
                    while let Some(meta) = stream.try_next().await? {
                        out.push(Value::String(meta.location.to_string()))
                    }
                    Ok::<Vec<Value>, object_store::Error>(out)
                })
                .map_err(|e| VmError::Runtime(format!("S3 list error: {}", e)))?;
            Ok(Value::List(entries))
        }
        "gcs" => {
            let (bucket, prefix) = parse_bucket_and_key(&rest)?;
            let store = gcs_store(&bucket, opts)?;
            let path = if prefix.is_empty() { Path::from("") } else { Path::from(prefix) };
            let rt = rt()?;
            let entries = rt
                .block_on(async {
                    let mut stream = store.list(Some(&path));
                    let mut out = Vec::new();
                    while let Some(meta) = stream.try_next().await? {
                        out.push(Value::String(meta.location.to_string()))
                    }
                    Ok::<Vec<Value>, object_store::Error>(out)
                })
                .map_err(|e| VmError::Runtime(format!("GCS list error: {}", e)))?;
            Ok(Value::List(entries))
        }
        "azure" => {
            let (container, prefix) = parse_bucket_and_key(&rest)?;
            let store = azure_store(&container, opts)?;
            let path = if prefix.is_empty() { Path::from("") } else { Path::from(prefix) };
            let rt = rt()?;
            let entries = rt
                .block_on(async {
                    let mut stream = store.list(Some(&path));
                    let mut out = Vec::new();
                    while let Some(meta) = stream.try_next().await? {
                        out.push(Value::String(meta.location.to_string()))
                    }
                    Ok::<Vec<Value>, object_store::Error>(out)
                })
                .map_err(|e| VmError::Runtime(format!("Azure list error: {}", e)))?;
            Ok(Value::List(entries))
        }
        _ => Err(VmError::Runtime(format!("{} provider not implemented yet", provider))),
    }
}

pub fn cloud_head_uri(provider: &str, rest: String, opts: Option<&std::collections::HashMap<String, Value>>) -> VmResult<Value> {
    match provider {
        "s3" => {
            let (bucket, key) = parse_bucket_and_key(&rest)?;
            let store = s3_store(&bucket, opts)?;
            let path = Path::from(key);
            let rt = rt()?;
            let meta = rt
                .block_on(async { store.head(&path).await })
                .map_err(|e| VmError::Runtime(format!("S3 head error: {}", e)))?;
            let mut m = std::collections::HashMap::new();
            m.insert("size".to_string(), Value::Integer(meta.size as i64));
            if let Some(etag) = meta.e_tag { m.insert("etag".into(), Value::String(etag)); }
            Ok(Value::Object(m))
        }
        "gcs" => {
            let (bucket, key) = parse_bucket_and_key(&rest)?;
            let store = gcs_store(&bucket, opts)?;
            let path = Path::from(key);
            let rt = rt()?;
            let meta = rt
                .block_on(async { store.head(&path).await })
                .map_err(|e| VmError::Runtime(format!("GCS head error: {}", e)))?;
            let mut m = std::collections::HashMap::new();
            m.insert("size".to_string(), Value::Integer(meta.size as i64));
            if let Some(etag) = meta.e_tag { m.insert("etag".into(), Value::String(etag)); }
            Ok(Value::Object(m))
        }
        "azure" => {
            let (container, key) = parse_bucket_and_key(&rest)?;
            let store = azure_store(&container, opts)?;
            let path = Path::from(key);
            let rt = rt()?;
            let meta = rt
                .block_on(async { store.head(&path).await })
                .map_err(|e| VmError::Runtime(format!("Azure head error: {}", e)))?;
            let mut m = std::collections::HashMap::new();
            m.insert("size".to_string(), Value::Integer(meta.size as i64));
            if let Some(etag) = meta.e_tag { m.insert("etag".into(), Value::String(etag)); }
            Ok(Value::Object(m))
        }
        _ => Err(VmError::Runtime(format!("{} provider not implemented yet", provider))),
    }
}

pub fn cloud_presign_uri(provider: &str, rest: String, method: &str, expires: u64, opts: Option<&std::collections::HashMap<String, Value>>) -> VmResult<Value> {
    use object_store::presign::PresignOperation;
    let expires = std::time::Duration::from_secs(expires);
    let op_from = |path: Path| -> Result<PresignOperation, VmError> {
        match method {
            m if m.eq_ignore_ascii_case("GET") => Ok(PresignOperation::Get { path, expires }),
            m if m.eq_ignore_ascii_case("PUT") => Ok(PresignOperation::Put { path, expires }),
            _ => Err(VmError::TypeError { expected: "method in {GET, PUT}".into(), actual: method.to_string() }),
        }
    };
    match provider {
        "s3" => {
            let (bucket, key) = parse_bucket_and_key(&rest)?;
            let store = s3_store(&bucket, opts)?;
            let path = Path::from(key);
            let op = op_from(path)?;
            let rt = rt()?;
            let req = rt
                .block_on(async { store.presign(&op).await })
                .map_err(|e| VmError::Runtime(format!("S3 presign error: {}", e)))?;
            Ok(Value::String(req.url().to_string()))
        }
        "gcs" => {
            let (bucket, key) = parse_bucket_and_key(&rest)?;
            let store = gcs_store(&bucket, opts)?;
            let path = Path::from(key);
            let op = op_from(path)?;
            let rt = rt()?;
            let req = rt
                .block_on(async { store.presign(&op).await })
                .map_err(|e| VmError::Runtime(format!("GCS presign error: {}", e)))?;
            Ok(Value::String(req.url().to_string()))
        }
        "azure" => {
            let (container, key) = parse_bucket_and_key(&rest)?;
            let store = azure_store(&container, opts)?;
            let path = Path::from(key);
            let op = op_from(path)?;
            let rt = rt()?;
            let req = rt
                .block_on(async { store.presign(&op).await })
                .map_err(|e| VmError::Runtime(format!("Azure presign error: {}", e)))?;
            Ok(Value::String(req.url().to_string()))
        }
        _ => Err(VmError::Runtime(format!("{} presign not implemented yet", provider))),
    }
}
