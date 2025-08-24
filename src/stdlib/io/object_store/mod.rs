//! Object Store (Phase A): file:// and store-backed memory
//!
//! Generic APIs: ObjectStoreOpen/Read/Write/Delete/List/Head/Copy/Presign
//! Provider wrappers (S3/GCS/Azure) return Not Implemented in Phase A.

use std::any::Any;
use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmError, VmResult};

#[derive(Debug, Clone)]
enum Provider { File, Memory }

#[derive(Debug, Clone)]
pub struct ObjectStore {
    provider: Provider,
    root: String, // for file:// this is a base directory; for memory, a namespace
    memory: Option<Arc<Mutex<HashMap<String, Vec<u8>>>>>,
}

impl ObjectStore {
    fn new_file(root: String) -> Self { Self { provider: Provider::File, root, memory: None } }
    fn new_memory(ns: String) -> Self { Self { provider: Provider::Memory, root: ns, memory: Some(Arc::new(Mutex::new(HashMap::new()))) } }
}

impl Foreign for ObjectStore {
    fn type_name(&self) -> &'static str { "ObjectStore" }
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Provider" => Ok(Value::String(match self.provider { Provider::File => "file".to_string(), Provider::Memory => "memory".to_string() })),
            "Root" => Ok(Value::String(self.root.clone())),
            _ => Err(ForeignError::UnknownMethod { method: method.to_string(), type_name: self.type_name().to_string() })
        }
    }
    fn clone_boxed(&self) -> Box<dyn Foreign> { Box::new(self.clone()) }
    fn as_any(&self) -> &dyn Any { self }
}

#[derive(Debug, Clone)]
pub enum ReaderKind {
    File { path: PathBuf, pos: u64, closed: bool },
    Memory { data: Arc<Vec<u8>>, pos: u64, closed: bool },
}

#[derive(Debug, Clone)]
pub struct ObjectReader { state: Arc<Mutex<ReaderKind>> }

impl ObjectReader {
    fn read_n(&self, n: usize) -> VmResult<Value> {
        let mut guard = self.state.lock().unwrap();
        match &mut *guard {
            ReaderKind::File { path, pos, closed } => {
                if *closed { return Err(VmError::Runtime("Reader is closed".into())); }
                let mut f = fs::File::open(path).map_err(|e| VmError::Runtime(format!("Open failed: {}", e)))?;
                f.seek(SeekFrom::Start(*pos)).map_err(|e| VmError::Runtime(format!("Seek failed: {}", e)))?;
                let mut buf = vec![0u8; n];
                let read = f.read(&mut buf).map_err(|e| VmError::Runtime(format!("Read failed: {}", e)))?;
                buf.truncate(read);
                *pos += read as u64;
                Ok(Value::List(buf.into_iter().map(|b| Value::Integer(b as i64)).collect()))
            }
            ReaderKind::Memory { data, pos, closed } => {
                if *closed { return Err(VmError::Runtime("Reader is closed".into())); }
                let d = data.as_ref();
                let start = (*pos) as usize;
                if start >= d.len() { return Ok(Value::List(Vec::new())); }
                let end = (start + n).min(d.len());
                let slice = &d[start..end];
                *pos = end as u64;
                Ok(Value::List(slice.iter().copied().map(|b| Value::Integer(b as i64)).collect()))
            }
        }
    }
}

impl Foreign for ObjectReader {
    fn type_name(&self) -> &'static str { "ObjectReader" }
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Read" => {
                let n = if args.len() == 1 { if let Value::Integer(i) = args[0] { if i > 0 { i as usize } else { 8192 } } else { 8192 } } else { 8192 };
                self.read_n(n).map_err(|e| ForeignError::RuntimeError { message: e.to_string() })
            }
            "Seek" => {
                if args.len() != 1 { return Err(ForeignError::UnknownMethod { method: method.to_string(), type_name: self.type_name().to_string() }); }
                let pos = match args[0] { Value::Integer(i) if i >= 0 => i as u64, _ => 0 };
                let mut guard = self.state.lock().unwrap();
                match &mut *guard { ReaderKind::File { pos: p, .. } | ReaderKind::Memory { pos: p, .. } => { *p = pos; Ok(Value::Boolean(true)) } }
            }
            "BytesRead" => {
                let guard = self.state.lock().unwrap();
                match &*guard { ReaderKind::File { pos, .. } | ReaderKind::Memory { pos, .. } => Ok(Value::Integer(*pos as i64)) }
            }
            "Close" => {
                let mut guard = self.state.lock().unwrap();
                match &mut *guard { ReaderKind::File { closed, .. } | ReaderKind::Memory { closed, .. } => { *closed = true; Ok(Value::Boolean(true)) } }
            }
            _ => Err(ForeignError::UnknownMethod { method: method.to_string(), type_name: self.type_name().to_string() })
        }
    }
    fn clone_boxed(&self) -> Box<dyn Foreign> { Box::new(self.clone()) }
    fn as_any(&self) -> &dyn Any { self }
}

fn parse_uri(uri: &str) -> Result<(String, String), VmError> {
    // returns (scheme, path) naive for Phase A
    let lower = uri.to_lowercase();
    if let Some(rest) = lower.strip_prefix("file://") { return Ok(("file".to_string(), rest.to_string())); }
    if let Some(rest) = lower.strip_prefix("memory://") { return Ok(("memory".to_string(), rest.to_string())); }
    if let Some(rest) = lower.strip_prefix("s3://") { return Ok(("s3".to_string(), rest.to_string())); }
    if let Some(rest) = lower.strip_prefix("gs://") { return Ok(("gcs".to_string(), rest.to_string())); }
    if let Some(rest) = lower.strip_prefix("az://") { return Ok(("azure".to_string(), rest.to_string())); }
    Err(VmError::Runtime(format!("Unsupported URI scheme: {}", uri)))
}

fn bytes_to_value(bytes: Vec<u8>) -> Value {
    Value::List(bytes.into_iter().map(|b| Value::Integer(b as i64)).collect())
}

fn value_to_bytes(v: &Value) -> VmResult<Vec<u8>> {
    match v {
        Value::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items { match it { Value::Integer(i) if *i >= 0 && *i <= 255 => out.push(*i as u8), _ => return Err(VmError::TypeError { expected: "List of byte integers 0..255".to_string(), actual: format!("{:?}", it) }) } }
            Ok(out)
        }
        Value::String(s) => Ok(s.as_bytes().to_vec()),
        other => Err(VmError::TypeError { expected: "Bytes as List[Integer] or String".to_string(), actual: format!("{:?}", other) })
    }
}

pub fn object_store_open(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 { return Err(VmError::TypeError { expected: "ObjectStoreOpen[uri|opts]".to_string(), actual: format!("{} args", args.len()) }); }
    // opts placeholder removed; not used in Phase A
    match &args[0] {
        Value::String(uri) => {
            let (scheme, _rest) = parse_uri(uri)?;
            match scheme.as_str() {
                "file" => Ok(Value::LyObj(LyObj::new(Box::new(ObjectStore::new_file(String::from(_rest)))))),
                "memory" => Ok(Value::LyObj(LyObj::new(Box::new(ObjectStore::new_memory(String::from(_rest)))))),
                "s3" | "gcs" | "azure" => Err(VmError::Runtime("ObjectStoreOpen for cloud providers is not yet supported; use uri-based operations".to_string())),
                _ => Err(VmError::Runtime("Unsupported provider".to_string())),
            }
        }
        Value::Object(m) => {
            let provider = m.get("provider").and_then(|v| if let Value::String(s)=v { Some(s.clone()) } else { None }).unwrap_or("file".to_string());
            let root = m.get("root").and_then(|v| if let Value::String(s)=v { Some(s.clone()) } else { None }).unwrap_or(".".to_string());
            match provider.to_lowercase().as_str() {
                "file" => Ok(Value::LyObj(LyObj::new(Box::new(ObjectStore::new_file(root))))),
                "memory" => Ok(Value::LyObj(LyObj::new(Box::new(ObjectStore::new_memory(root))))),
                _ => Err(VmError::Runtime("Unsupported provider in Phase A".to_string())),
            }
        }
        v => Err(VmError::TypeError { expected: "String uri or opts Association".to_string(), actual: format!("{:?}", v) })
    }
}

pub fn object_store_read(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 { return Err(VmError::TypeError { expected: "ObjectStoreRead[uri|{store, key}, opts?]".to_string(), actual: format!("{} args", args.len()) }); }
    // opts: stream -> True|False, chunkSize; cloud opts pass-through
    let mut stream = false;
    let mut _chunk = 8192usize;
    let opts_obj: Option<&HashMap<String, Value>> = if args.len() == 2 { if let Value::Object(m) = &args[1] { Some(m) } else { None } } else { None };
    if let Some(m) = opts_obj {
        if let Some(v) = m.get("stream") {
            stream = match v { Value::Boolean(b) => *b, Value::Integer(i) => *i != 0, _ => stream };
        }
        if let Some(v) = m.get("chunkSize") {
            if let Value::Integer(i) = v { if *i > 0 { _chunk = *i as usize; } }
        }
    }
    match &args[0] {
        Value::String(uri) => {
            let (scheme, rest) = parse_uri(uri)?;
            match scheme.as_str() {
                "file" => {
                    if stream {
                        let reader = ObjectReader { state: Arc::new(Mutex::new(ReaderKind::File { path: PathBuf::from(rest), pos: 0, closed: false })) };
                        Ok(Value::LyObj(LyObj::new(Box::new(reader))))
                    } else {
                        let path = PathBuf::from(rest);
                        let mut f = fs::File::open(&path).map_err(|e| VmError::Runtime(format!("File open failed: {}", e)))?;
                        let mut buf = Vec::new(); f.read_to_end(&mut buf).map_err(|e| VmError::Runtime(format!("File read failed: {}", e)))?;
                        Ok(bytes_to_value(buf))
                    }
                }
                "memory" => Err(VmError::Runtime("Use {store,key} form for memory provider".to_string())),
                "s3" | "gcs" | "azure" => {
                    #[cfg(feature = "cloud")]
                    {
                        // Extract providerOpts if present
                        let provider_opts: Option<&HashMap<String, Value>> = match _opts_obj.and_then(|o| o.get("providerOpts")) {
                            Some(Value::Object(p)) => Some(p),
                            _ => None,
                        };
                        return cloud_read_uri(scheme.as_str(), rest, provider_opts);
                    }
                    #[allow(unreachable_code)]
                    { Err(VmError::Runtime("Cloud providers require 'cloud' feature".to_string())) }
                }
                _ => Err(VmError::Runtime("Unsupported provider".to_string())),
            }
        }
        Value::List(v) if v.len() == 2 => {
            let store = match &v[0] { Value::LyObj(obj) => obj, _ => return Err(VmError::TypeError { expected: "ObjectStore in first element".to_string(), actual: format!("{:?}", v[0]) }) };
            let key = match &v[1] { Value::String(s) => s.clone(), _ => return Err(VmError::TypeError { expected: "String key".to_string(), actual: format!("{:?}", v[1]) }) };
            if let Some(os) = store.downcast_ref::<ObjectStore>() {
                match os.provider {
                    Provider::Memory => {
                        let guard = os.memory.as_ref().unwrap().lock().unwrap();
                        if let Some(bytes) = guard.get(&key) {
                            if stream {
                                let reader = ObjectReader { state: Arc::new(Mutex::new(ReaderKind::Memory { data: Arc::new(bytes.clone()), pos: 0, closed: false })) };
                                Ok(Value::LyObj(LyObj::new(Box::new(reader))))
                            } else {
                                Ok(bytes_to_value(bytes.clone()))
                            }
                        } else { Ok(Value::Missing) }
                    }
                    Provider::File => {
                        let p = PathBuf::from(&os.root).join(key);
                        if stream {
                            let reader = ObjectReader { state: Arc::new(Mutex::new(ReaderKind::File { path: p, pos: 0, closed: false })) };
                            Ok(Value::LyObj(LyObj::new(Box::new(reader))))
                        } else {
                            let mut f = fs::File::open(&p).map_err(|e| VmError::Runtime(format!("File open failed: {}", e)))?;
                            let mut buf = Vec::new(); f.read_to_end(&mut buf).map_err(|e| VmError::Runtime(format!("File read failed: {}", e)))?;
                            Ok(bytes_to_value(buf))
                        }
                    }
                }
            } else { Err(VmError::TypeError { expected: "ObjectStore".to_string(), actual: store.type_name().to_string() }) }
        }
        v => Err(VmError::TypeError { expected: "uri String or {store,key}".to_string(), actual: format!("{:?}", v) })
    }
}

pub fn object_store_write(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 { return Err(VmError::TypeError { expected: "ObjectStoreWrite[uri|{store,key}, data, opts?]".to_string(), actual: format!("{} args", args.len()) }); }
    let data = &args[1];
    let bytes = value_to_bytes(data)?;
    match &args[0] {
        Value::String(uri) => {
            let (scheme, rest) = parse_uri(uri)?;
            match scheme.as_str() {
                "file" => {
                    let path = PathBuf::from(rest);
                    if let Some(parent) = path.parent() { let _ = fs::create_dir_all(parent); }
                    let mut f = fs::File::create(&path).map_err(|e| VmError::Runtime(format!("File create failed: {}", e)))?;
                    f.write_all(&bytes).map_err(|e| VmError::Runtime(format!("File write failed: {}", e)))?;
                    Ok(Value::Boolean(true))
                }
                "memory" => Err(VmError::Runtime("Use {store,key} form for memory provider".to_string())),
                "s3" | "gcs" | "azure" => {
                    #[cfg(feature = "cloud")]
                    {
                        let provider_opts: Option<&HashMap<String, Value>> = match _opts_obj.and_then(|o| o.get("providerOpts")) {
                            Some(Value::Object(p)) => Some(p),
                            _ => None,
                        };
                        return cloud_write_uri(scheme.as_str(), rest, &bytes, provider_opts);
                    }
                    #[allow(unreachable_code)]
                    { Err(VmError::Runtime("Cloud providers require 'cloud' feature".to_string())) }
                }
                _ => Err(VmError::Runtime("Unsupported provider".to_string())),
            }
        }
        Value::List(v) if v.len() == 2 => {
            let store = match &v[0] { Value::LyObj(obj) => obj, _ => return Err(VmError::TypeError { expected: "ObjectStore in first element".to_string(), actual: format!("{:?}", v[0]) }) };
            let key = match &v[1] { Value::String(s) => s.clone(), _ => return Err(VmError::TypeError { expected: "String key".to_string(), actual: format!("{:?}", v[1]) }) };
            if let Some(os) = store.downcast_ref::<ObjectStore>() {
                match os.provider {
                    Provider::Memory => {
                        let mut guard = os.memory.as_ref().unwrap().lock().unwrap();
                        guard.insert(key, bytes);
                        Ok(Value::Boolean(true))
                    }
                    Provider::File => {
                        let p = PathBuf::from(&os.root).join(key);
                        if let Some(parent) = p.parent() { let _ = fs::create_dir_all(parent); }
                        let mut f = fs::File::create(&p).map_err(|e| VmError::Runtime(format!("File create failed: {}", e)))?;
                        f.write_all(&bytes).map_err(|e| VmError::Runtime(format!("File write failed: {}", e)))?;
                        Ok(Value::Boolean(true))
                    }
                }
            } else { Err(VmError::TypeError { expected: "ObjectStore".to_string(), actual: store.type_name().to_string() }) }
        }
        v => Err(VmError::TypeError { expected: "uri String or {store,key}".to_string(), actual: format!("{:?}", v) })
    }
}

pub fn object_store_delete(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 { return Err(VmError::TypeError { expected: "ObjectStoreDelete[uri|{store,key}, opts?]".to_string(), actual: format!("{} args", args.len()) }); }
    let _opts_obj: Option<&HashMap<String, Value>> = if args.len() == 2 { if let Value::Object(m) = &args[1] { Some(m) } else { None } } else { None };
    match &args[0] {
        Value::String(uri) => {
            let (scheme, rest) = parse_uri(uri)?;
            match scheme.as_str() {
                "file" => { let p = PathBuf::from(rest); fs::remove_file(&p).map_err(|e| VmError::Runtime(format!("Delete failed: {}", e)))?; Ok(Value::Boolean(true)) }
                "s3" | "gcs" | "azure" => {
                    #[cfg(feature = "cloud")]
                    {
                        let provider_opts: Option<&HashMap<String, Value>> = match _opts_obj.and_then(|o| o.get("providerOpts")) {
                            Some(Value::Object(p)) => Some(p),
                            _ => None,
                        };
                        return cloud_delete_uri(scheme.as_str(), rest, provider_opts);
                    }
                    #[allow(unreachable_code)]
                    { Err(VmError::Runtime("Cloud providers require 'cloud' feature".to_string())) }
                }
                _ => Err(VmError::Runtime("Unsupported provider".to_string())),
            }
        }
        Value::List(v) if v.len() == 2 => {
            let store = match &v[0] { Value::LyObj(obj) => obj, _ => return Err(VmError::TypeError { expected: "ObjectStore in first element".to_string(), actual: format!("{:?}", v[0]) }) };
            let key = match &v[1] { Value::String(s) => s.clone(), _ => return Err(VmError::TypeError { expected: "String key".to_string(), actual: format!("{:?}", v[1]) }) };
            if let Some(os) = store.downcast_ref::<ObjectStore>() {
                match os.provider {
                    Provider::Memory => { let mut g = os.memory.as_ref().unwrap().lock().unwrap(); g.remove(&key); Ok(Value::Boolean(true)) }
                    Provider::File => { let p = PathBuf::from(&os.root).join(key); fs::remove_file(&p).map_err(|e| VmError::Runtime(format!("Delete failed: {}", e)))?; Ok(Value::Boolean(true)) }
                }
            } else { Err(VmError::TypeError { expected: "ObjectStore".to_string(), actual: store.type_name().to_string() }) }
        }
        v => Err(VmError::TypeError { expected: "uri String or {store,key}".to_string(), actual: format!("{:?}", v) })
    }
}

pub fn object_store_list(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 { return Err(VmError::TypeError { expected: "ObjectStoreList[uri|{store,prefix}, opts?]".to_string(), actual: format!("{} args", args.len()) }); }
    let _opts_obj: Option<&HashMap<String, Value>> = if args.len() == 2 { if let Value::Object(m) = &args[1] { Some(m) } else { None } } else { None };
    match &args[0] {
        Value::String(uri) => {
            let (scheme, rest) = parse_uri(uri)?;
            match scheme.as_str() {
                "file" => {
                    let p = PathBuf::from(rest);
                    let entries = fs::read_dir(&p).map_err(|e| VmError::Runtime(format!("List failed: {}", e)))?;
                    let mut out = Vec::new();
                    for e in entries { let e = e.map_err(|e| VmError::Runtime(format!("Dir entry error: {}", e)))?; out.push(Value::String(e.file_name().to_string_lossy().to_string())); }
                    Ok(Value::List(out))
                }
                "s3" | "gcs" | "azure" => {
                    #[cfg(feature = "cloud")]
                    {
                        let provider_opts: Option<&HashMap<String, Value>> = match _opts_obj.and_then(|o| o.get("providerOpts")) {
                            Some(Value::Object(p)) => Some(p),
                            _ => None,
                        };
                        return cloud_list_uri(scheme.as_str(), rest, provider_opts);
                    }
                    #[allow(unreachable_code)]
                    { Err(VmError::Runtime("Cloud providers require 'cloud' feature".to_string())) }
                }
                _ => Err(VmError::Runtime("Unsupported provider".to_string())),
            }
        }
        Value::List(v) if v.len() == 2 => {
            let store = match &v[0] { Value::LyObj(obj) => obj, _ => return Err(VmError::TypeError { expected: "ObjectStore in first element".to_string(), actual: format!("{:?}", v[0]) }) };
            let prefix = match &v[1] { Value::String(s) => s.clone(), _ => return Err(VmError::TypeError { expected: "String prefix".to_string(), actual: format!("{:?}", v[1]) }) };
            if let Some(os) = store.downcast_ref::<ObjectStore>() {
                match os.provider {
                    Provider::Memory => { let g = os.memory.as_ref().unwrap().lock().unwrap(); let mut out = Vec::new(); for k in g.keys() { if k.starts_with(&prefix) { out.push(Value::String(k.clone())); } } Ok(Value::List(out)) }
                    Provider::File => { let p = PathBuf::from(&os.root).join(prefix); let entries = fs::read_dir(&p).map_err(|e| VmError::Runtime(format!("List failed: {}", e)))?; let mut out = Vec::new(); for e in entries { let e = e.map_err(|e| VmError::Runtime(format!("Dir entry error: {}", e)))?; out.push(Value::String(e.file_name().to_string_lossy().to_string())); } Ok(Value::List(out)) }
                }
            } else { Err(VmError::TypeError { expected: "ObjectStore".to_string(), actual: store.type_name().to_string() }) }
        }
        v => Err(VmError::TypeError { expected: "uri String or {store,prefix}".to_string(), actual: format!("{:?}", v) })
    }
}

pub fn object_store_head(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 { return Err(VmError::TypeError { expected: "ObjectStoreHead[uri|{store,key}, opts?]".to_string(), actual: format!("{} args", args.len()) }); }
    let opts_obj: Option<&HashMap<String, Value>> = if args.len() == 2 { if let Value::Object(m) = &args[1] { Some(m) } else { None } } else { None };
    match &args[0] {
        Value::String(uri) => {
            let (scheme, rest) = parse_uri(uri)?;
            match scheme.as_str() {
                "file" => { let md = fs::metadata(PathBuf::from(rest)).map_err(|e| VmError::Runtime(format!("Head failed: {}", e)))?; let mut m = HashMap::new(); m.insert("size".to_string(), Value::Integer(md.len() as i64)); Ok(Value::Object(m)) }
                "s3" | "gcs" | "azure" => {
                    #[cfg(feature = "cloud")]
                    {
                        let provider_opts: Option<&HashMap<String, Value>> = match opts_obj.and_then(|o| o.get("providerOpts")) {
                            Some(Value::Object(p)) => Some(p),
                            _ => None,
                        };
                        return cloud_head_uri(scheme.as_str(), rest, provider_opts);
                    }
                    #[allow(unreachable_code)]
                    { Err(VmError::Runtime("Cloud providers require 'cloud' feature".to_string())) }
                }
                _ => Err(VmError::Runtime("Unsupported provider".to_string())),
            }
        }
        Value::List(v) if v.len() == 2 => {
            let store = match &v[0] { Value::LyObj(obj) => obj, _ => return Err(VmError::TypeError { expected: "ObjectStore in first element".to_string(), actual: format!("{:?}", v[0]) }) };
            let key = match &v[1] { Value::String(s) => s.clone(), _ => return Err(VmError::TypeError { expected: "String key".to_string(), actual: format!("{:?}", v[1]) }) };
            if let Some(os) = store.downcast_ref::<ObjectStore>() {
                match os.provider {
                    Provider::Memory => { let g = os.memory.as_ref().unwrap().lock().unwrap(); if let Some(bytes) = g.get(&key) { let mut m = HashMap::new(); m.insert("size".to_string(), Value::Integer(bytes.len() as i64)); Ok(Value::Object(m)) } else { Ok(Value::Missing) } }
                    Provider::File => { let md = fs::metadata(PathBuf::from(&os.root).join(key)).map_err(|e| VmError::Runtime(format!("Head failed: {}", e)))?; let mut m = HashMap::new(); m.insert("size".to_string(), Value::Integer(md.len() as i64)); Ok(Value::Object(m)) }
                }
            } else { Err(VmError::TypeError { expected: "ObjectStore".to_string(), actual: store.type_name().to_string() }) }
        }
        v => Err(VmError::TypeError { expected: "uri String or {store,key}".to_string(), actual: format!("{:?}", v) })
    }
}

pub fn object_store_copy(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 { return Err(VmError::TypeError { expected: "ObjectStoreCopy[src, dst]".to_string(), actual: format!("{} args", args.len()) }); }
    // Support String URIs or {store,key}
    // Helper to resolve into a bytes source and a writer sink
    enum Src { File(PathBuf), Mem(Arc<Vec<u8>>), StoreKey(ObjectStore, String) }
    enum Dst { File(PathBuf), StoreKey(ObjectStore, String) }

    // Parse src
    let src = match &args[0] {
        Value::String(uri) => {
            let (scheme, rest) = parse_uri(uri)?;
            match scheme.as_str() {
                "file" => Src::File(PathBuf::from(rest)),
                _ => return Err(VmError::Runtime("Only file:// supported for URI copy in this phase".to_string())),
            }
        }
        Value::List(v) if v.len() == 2 => {
            let store = match &v[0] { Value::LyObj(obj) => obj, _ => return Err(VmError::TypeError { expected: "ObjectStore in first element".to_string(), actual: format!("{:?}", v[0]) }) };
            let key = match &v[1] { Value::String(s) => s.clone(), _ => return Err(VmError::TypeError { expected: "String key".to_string(), actual: format!("{:?}", v[1]) }) };
            if let Some(os) = store.downcast_ref::<ObjectStore>() { Src::StoreKey(os.clone(), key) } else { return Err(VmError::TypeError { expected: "ObjectStore".to_string(), actual: store.type_name().to_string() }) }
        }
        v => return Err(VmError::TypeError { expected: "src as uri or {store,key}".to_string(), actual: format!("{:?}", v) })
    };
    // Parse dst
    let dst = match &args[1] {
        Value::String(uri) => {
            let (scheme, rest) = parse_uri(uri)?;
            match scheme.as_str() { "file" => Dst::File(PathBuf::from(rest)), _ => return Err(VmError::Runtime("Only file:// supported for URI copy in this phase".to_string())) }
        }
        Value::List(v) if v.len() == 2 => {
            let store = match &v[0] { Value::LyObj(obj) => obj, _ => return Err(VmError::TypeError { expected: "ObjectStore in first element".to_string(), actual: format!("{:?}", v[0]) }) };
            let key = match &v[1] { Value::String(s) => s.clone(), _ => return Err(VmError::TypeError { expected: "String key".to_string(), actual: format!("{:?}", v[1]) }) };
            if let Some(os) = store.downcast_ref::<ObjectStore>() { Dst::StoreKey(os.clone(), key) } else { return Err(VmError::TypeError { expected: "ObjectStore".to_string(), actual: store.type_name().to_string() }) }
        }
        v => return Err(VmError::TypeError { expected: "dst as uri or {store,key}".to_string(), actual: format!("{:?}", v) })
    };

    // Read src bytes efficiently
    let bytes: Vec<u8> = match src {
        Src::File(p) => {
            let mut f = fs::File::open(&p).map_err(|e| VmError::Runtime(format!("Open failed: {}", e)))?;
            let mut buf = Vec::new();
            f.read_to_end(&mut buf).map_err(|e| VmError::Runtime(format!("Read failed: {}", e)))?;
            buf
        }
        Src::Mem(data) => data.as_ref().clone(),
        Src::StoreKey(os, key) => {
            match os.provider {
                Provider::Memory => {
                    let g = os.memory.as_ref().unwrap().lock().unwrap();
                    g.get(&key).cloned().unwrap_or_default()
                }
                Provider::File => {
                    let p = PathBuf::from(&os.root).join(key);
                    let mut f = fs::File::open(&p).map_err(|e| VmError::Runtime(format!("Open failed: {}", e)))?;
                    let mut buf = Vec::new(); f.read_to_end(&mut buf).map_err(|e| VmError::Runtime(format!("Read failed: {}", e)))?; buf
                }
            }
        }
    };

    // Write to dst
    match dst {
        Dst::File(p) => {
            if let Some(parent) = p.parent() { let _ = fs::create_dir_all(parent); }
            let mut f = fs::File::create(&p).map_err(|e| VmError::Runtime(format!("Create failed: {}", e)))?;
            f.write_all(&bytes).map_err(|e| VmError::Runtime(format!("Write failed: {}", e)))?;
        }
        Dst::StoreKey(os, key) => {
            match os.provider {
                Provider::Memory => { let mut g = os.memory.as_ref().unwrap().lock().unwrap(); g.insert(key, bytes); }
                Provider::File => { let p = PathBuf::from(&os.root).join(key); if let Some(parent) = p.parent() { let _ = fs::create_dir_all(parent); } let mut f = fs::File::create(&p).map_err(|e| VmError::Runtime(format!("Create failed: {}", e)))?; f.write_all(&bytes).map_err(|e| VmError::Runtime(format!("Write failed: {}", e)))?; }
            }
        }
    }

    Ok(Value::Boolean(true))
}
pub fn object_store_presign(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 { return Err(VmError::TypeError { expected: "ObjectStorePresign[uri, opts?]".to_string(), actual: format!("{} args", args.len()) }); }
    let uri = match &args[0] { Value::String(s) => s.clone(), v => return Err(VmError::TypeError { expected: "uri String".into(), actual: format!("{:?}", v) }) };
    // default method GET, expiresSec 900
    let mut _method = "GET".to_string();
    let mut _expires = 900u64;
    let mut _provider_opts: Option<&HashMap<String, Value>> = None;
    if args.len() == 2 {
        if let Value::Object(m) = &args[1] {
            if let Some(Value::String(s)) = m.get("method") { _method = s.to_uppercase(); }
            if let Some(Value::Integer(i)) = m.get("expiresSec") { if *i > 0 { _expires = *i as u64; } }
            if let Some(Value::Object(p)) = m.get("providerOpts") { _provider_opts = Some(p); }
        } else {
            return Err(VmError::TypeError { expected: "opts Association".into(), actual: format!("{:?}", args[1]) });
        }
    }
    let (scheme, _rest) = parse_uri(&uri)?;
    match scheme.as_str() {
        "s3" | "gcs" | "azure" => {
            #[cfg(feature = "cloud")]
            { return cloud_presign_uri(scheme.as_str(), _rest, &_method, _expires, _provider_opts); }
            #[allow(unreachable_code)]
            { Err(VmError::Runtime("Cloud providers require 'cloud' feature".to_string())) }
        }
        _ => Err(VmError::Runtime("Presign only supported for cloud providers".to_string())),
    }
}

pub fn object_store_exists(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 { return Err(VmError::TypeError { expected: "ObjectStoreExists[uri|{store,key}]".into(), actual: format!("{} args", args.len()) }); }
    match &args[0] {
        Value::String(uri) => {
            let (scheme, rest) = parse_uri(uri)?;
            match scheme.as_str() {
                "file" => Ok(Value::Boolean(PathBuf::from(rest).exists())),
                "s3" | "gcs" | "azure" => {
                    #[cfg(feature = "cloud")]
                    { match cloud_head_uri(scheme.as_str(), rest) { Ok(_) => Ok(Value::Boolean(true)), Err(_) => Ok(Value::Boolean(false)) } }
                    #[allow(unreachable_code)]
                    { Ok(Value::Boolean(false)) }
                }
                _ => Err(VmError::Runtime("Unsupported provider".into())),
            }
        }
        Value::List(v) if v.len() == 2 => {
            let store = match &v[0] { Value::LyObj(obj) => obj, _ => return Err(VmError::TypeError { expected: "ObjectStore in first element".into(), actual: format!("{:?}", v[0]) }) };
            let key = match &v[1] { Value::String(s) => s.clone(), _ => return Err(VmError::TypeError { expected: "String key".into(), actual: format!("{:?}", v[1]) }) };
            if let Some(os) = store.downcast_ref::<ObjectStore>() {
                match os.provider {
                    Provider::Memory => {
                        let g = os.memory.as_ref().unwrap().lock().unwrap();
                        Ok(Value::Boolean(g.contains_key(&key)))
                    }
                    Provider::File => {
                        let p = PathBuf::from(&os.root).join(key);
                        Ok(Value::Boolean(p.exists()))
                    }
                }
            } else { Err(VmError::TypeError { expected: "ObjectStore".into(), actual: store.type_name().into() }) }
        }
        v => Err(VmError::TypeError { expected: "uri String or {store,key}".into(), actual: format!("{:?}", v) })
    }
}

pub fn object_store_move(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 { return Err(VmError::TypeError { expected: "ObjectStoreMove[src, dst]".into(), actual: format!("{} args", args.len()) }); }
    // Implement as copy then delete src
    object_store_copy(&[args[0].clone(), args[1].clone()])?;
    // Best-effort delete src
    let _ = object_store_delete(&[args[0].clone()]);
    Ok(Value::Boolean(true))
}

// Provider-specific wrappers (stubs)
macro_rules! stub {
    ($name:ident) => { pub fn $name(_args: &[Value]) -> VmResult<Value> { Err(VmError::Runtime(concat!(stringify!($name), " is not implemented in Phase A").to_string())) } };
}

#[cfg(not(feature = "cloud"))]
stub!(s3_read); #[cfg(not(feature = "cloud"))] stub!(s3_write); #[cfg(not(feature = "cloud"))] stub!(s3_list); #[cfg(not(feature = "cloud"))] stub!(s3_delete); #[cfg(not(feature = "cloud"))] stub!(s3_head); #[cfg(not(feature = "cloud"))] stub!(s3_presign);
#[cfg(not(feature = "cloud"))]
stub!(gcs_read); #[cfg(not(feature = "cloud"))] stub!(gcs_write); #[cfg(not(feature = "cloud"))] stub!(gcs_list); #[cfg(not(feature = "cloud"))] stub!(gcs_delete); #[cfg(not(feature = "cloud"))] stub!(gcs_head); #[cfg(not(feature = "cloud"))] stub!(gcs_presign);
#[cfg(not(feature = "cloud"))]
stub!(azure_blob_read); #[cfg(not(feature = "cloud"))] stub!(azure_blob_write); #[cfg(not(feature = "cloud"))] stub!(azure_blob_list); #[cfg(not(feature = "cloud"))] stub!(azure_blob_delete); #[cfg(not(feature = "cloud"))] stub!(azure_blob_head); #[cfg(not(feature = "cloud"))] stub!(azure_blob_presign);

#[cfg(feature = "cloud")]
mod cloud_backend;
#[cfg(feature = "cloud")]
use cloud_backend::{cloud_read_uri, cloud_write_uri, cloud_delete_uri, cloud_list_uri, cloud_head_uri, cloud_presign_uri};

pub fn register_object_store_functions() -> HashMap<String, fn(&[Value]) -> VmResult<Value>> {
    let mut f = HashMap::new();
    f.insert("ObjectStoreOpen".to_string(), object_store_open as fn(&[Value]) -> VmResult<Value>);
    f.insert("ObjectStoreRead".to_string(), object_store_read as fn(&[Value]) -> VmResult<Value>);
    f.insert("ObjectStoreWrite".to_string(), object_store_write as fn(&[Value]) -> VmResult<Value>);
    f.insert("ObjectStoreDelete".to_string(), object_store_delete as fn(&[Value]) -> VmResult<Value>);
    f.insert("ObjectStoreList".to_string(), object_store_list as fn(&[Value]) -> VmResult<Value>);
    f.insert("ObjectStoreHead".to_string(), object_store_head as fn(&[Value]) -> VmResult<Value>);
    f.insert("ObjectStoreCopy".to_string(), object_store_copy as fn(&[Value]) -> VmResult<Value>);
    f.insert("ObjectStorePresign".to_string(), object_store_presign as fn(&[Value]) -> VmResult<Value>);
    // Additional helpers
    f.insert("ObjectStoreExists".to_string(), object_store_exists as fn(&[Value]) -> VmResult<Value>);
    f.insert("ObjectStoreMove".to_string(), object_store_move as fn(&[Value]) -> VmResult<Value>);
    // Provider wrappers
    f.insert("S3Read".to_string(), s3_read as fn(&[Value]) -> VmResult<Value>);
    f.insert("S3Write".to_string(), s3_write as fn(&[Value]) -> VmResult<Value>);
    f.insert("S3List".to_string(), s3_list as fn(&[Value]) -> VmResult<Value>);
    f.insert("S3Delete".to_string(), s3_delete as fn(&[Value]) -> VmResult<Value>);
    f.insert("S3Head".to_string(), s3_head as fn(&[Value]) -> VmResult<Value>);
    f.insert("S3Presign".to_string(), s3_presign as fn(&[Value]) -> VmResult<Value>);
    f.insert("GCSRead".to_string(), gcs_read as fn(&[Value]) -> VmResult<Value>);
    f.insert("GCSWrite".to_string(), gcs_write as fn(&[Value]) -> VmResult<Value>);
    f.insert("GCSList".to_string(), gcs_list as fn(&[Value]) -> VmResult<Value>);
    f.insert("GCSDelete".to_string(), gcs_delete as fn(&[Value]) -> VmResult<Value>);
    f.insert("GCSHead".to_string(), gcs_head as fn(&[Value]) -> VmResult<Value>);
    f.insert("GCSPresign".to_string(), gcs_presign as fn(&[Value]) -> VmResult<Value>);
    f.insert("AzureBlobRead".to_string(), azure_blob_read as fn(&[Value]) -> VmResult<Value>);
    f.insert("AzureBlobWrite".to_string(), azure_blob_write as fn(&[Value]) -> VmResult<Value>);
    f.insert("AzureBlobList".to_string(), azure_blob_list as fn(&[Value]) -> VmResult<Value>);
    f.insert("AzureBlobDelete".to_string(), azure_blob_delete as fn(&[Value]) -> VmResult<Value>);
    f.insert("AzureBlobHead".to_string(), azure_blob_head as fn(&[Value]) -> VmResult<Value>);
    f.insert("AzureBlobPresign".to_string(), azure_blob_presign as fn(&[Value]) -> VmResult<Value>);
    f
}
