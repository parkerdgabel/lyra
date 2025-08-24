//! Simple in-VM cache primitives (LRU-like) for Phase A

use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmError, VmResult};

#[derive(Debug, Clone)]
pub struct Cache {
    capacity: usize,
    default_ttl: Option<Duration>,
    state: Arc<Mutex<CacheState>>,
}

#[derive(Debug, Clone)]
struct Entry { value: Value, expires_at: Option<Instant> }

#[derive(Debug, Clone)]
struct CacheState {
    store: HashMap<String, Entry>,
    order: VecDeque<String>,
    hits: usize,
    misses: usize,
}

impl CacheState {
    fn new() -> Self { Self { store: HashMap::new(), order: VecDeque::new(), hits: 0, misses: 0 } }

    fn touch(&mut self, capacity: usize, key: &str) {
        if let Some(pos) = self.order.iter().position(|k| k == key) { self.order.remove(pos); }
        self.order.push_back(key.to_string());
        if self.order.len() > capacity { self.evict_one(); }
    }

    fn evict_one(&mut self) {
        if let Some(old) = self.order.pop_front() { self.store.remove(&old); }
    }
}

impl Cache {
    pub fn new(capacity: usize, default_ttl: Option<Duration>) -> Self {
        Self { capacity, default_ttl, state: Arc::new(Mutex::new(CacheState::new())) }
    }
}

impl Foreign for Cache {
    fn type_name(&self) -> &'static str { "Cache" }

    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Capacity" => Ok(Value::Integer(self.capacity as i64)),
            "Size" => {
                let st = self.state.lock().unwrap();
                Ok(Value::Integer(st.store.len() as i64))
            }
            "Hits" => {
                let st = self.state.lock().unwrap();
                Ok(Value::Integer(st.hits as i64))
            }
            "Misses" => {
                let st = self.state.lock().unwrap();
                Ok(Value::Integer(st.misses as i64))
            }
            _ => Err(ForeignError::UnknownMethod { method: method.to_string(), type_name: self.type_name().to_string() })
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> { Box::new(self.clone()) }
    fn as_any(&self) -> &dyn Any { self }
}

pub fn cache_create(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError { expected: "CacheCreate[type, capacity, opts?]".to_string(), actual: format!("{} args", args.len()) });
    }
    // type is currently ignored; kept for API compatibility ("LRU"|"LFU")
    let _cache_type = match &args[0] { Value::String(s) => s.clone(), v => return Err(VmError::TypeError { expected: "String cache type".to_string(), actual: format!("{:?}", v) }) };
    let capacity = match &args[1] { Value::Integer(i) if *i > 0 => *i as usize, v => return Err(VmError::TypeError { expected: "positive Integer capacity".to_string(), actual: format!("{:?}", v) }) };
    // opts: ttlMs
    let default_ttl = if args.len() == 3 {
        match &args[2] {
            Value::Object(m) => {
                if let Some(Value::Integer(ms)) = m.get("ttlMs") { if *ms > 0 { Some(Duration::from_millis(*ms as u64)) } else { None } } else { None }
            }
            v => return Err(VmError::TypeError { expected: "opts Association".to_string(), actual: format!("{:?}", v) })
        }
    } else { None };
    let cache = Cache::new(capacity, default_ttl);
    Ok(Value::LyObj(LyObj::new(Box::new(cache))))
}

pub fn cache_get(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 { return Err(VmError::TypeError { expected: "CacheGet[cache, key]".to_string(), actual: format!("{} args", args.len()) }); }
    let (cache_obj, key) = (&args[0], &args[1]);
    let key = match key { Value::String(s) => s.clone(), v => return Err(VmError::TypeError { expected: "String key".to_string(), actual: format!("{:?}", v) }) };
    if let Value::LyObj(obj) = cache_obj {
        if let Some(cache) = obj.downcast_ref::<Cache>() {
            let mut st = cache.state.lock().unwrap();
            let now = Instant::now();
            
            // Check for entry and clone value if found
            let entry_result = if let Some(entry) = st.store.get(&key) {
                if let Some(exp) = entry.expires_at { 
                    if exp <= now { 
                        None // expired 
                    } else {
                        Some(entry.value.clone())
                    }
                } else {
                    Some(entry.value.clone())
                }
            } else {
                None
            };
            
            match entry_result {
                Some(value) => {
                    st.hits += 1; 
                    st.touch(cache.capacity, &key); 
                    Ok(value)
                },
                None => {
                    // Miss or expired
                    st.misses += 1;
                    st.store.remove(&key);
                    Ok(Value::Missing)
                }
            }
        } else { Err(VmError::TypeError { expected: "Cache object".to_string(), actual: obj.type_name().to_string() }) }
    } else { Err(VmError::TypeError { expected: "Cache object".to_string(), actual: format!("{:?}", cache_obj) }) }
}

pub fn cache_put(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 || args.len() > 4 { return Err(VmError::TypeError { expected: "CachePut[cache, key, value, opts?]".to_string(), actual: format!("{} args", args.len()) }); }
    let (cache_obj, key, value) = (&args[0], &args[1], &args[2]);
    let key = match key { Value::String(s) => s.clone(), v => return Err(VmError::TypeError { expected: "String key".to_string(), actual: format!("{:?}", v) }) };
    if let Value::LyObj(obj) = cache_obj {
        if let Some(cache) = obj.downcast_ref::<Cache>() {
            let ttl_override = if args.len() == 4 {
                match &args[3] { Value::Object(m) => {
                    if let Some(Value::Integer(ms)) = m.get("ttlMs") { if *ms > 0 { Some(Duration::from_millis(*ms as u64)) } else { None } } else { None }
                }
                v => return Err(VmError::TypeError { expected: "opts Association".to_string(), actual: format!("{:?}", v) })
                }
            } else { None };
            let mut st = cache.state.lock().unwrap();
            let expires_at = ttl_override.or(cache.default_ttl).map(|d| Instant::now() + d);
            st.store.insert(key.clone(), Entry { value: value.clone(), expires_at });
            st.touch(cache.capacity, &key);
            Ok(Value::Boolean(true))
        } else { Err(VmError::TypeError { expected: "Cache object".to_string(), actual: obj.type_name().to_string() }) }
    } else { Err(VmError::TypeError { expected: "Cache object".to_string(), actual: format!("{:?}", cache_obj) }) }
}

pub fn cache_invalidate(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 { return Err(VmError::TypeError { expected: "CacheInvalidate[cache, key|All]".to_string(), actual: format!("{} args", args.len()) }); }
    let (cache_obj, key) = (&args[0], &args[1]);
    if let Value::LyObj(obj) = cache_obj {
        if let Some(cache) = obj.downcast_ref::<Cache>() {
            let mut st = cache.state.lock().unwrap();
            match key {
                Value::Symbol(s) if s == "All" => { st.store.clear(); st.order.clear(); Ok(Value::Boolean(true)) }
                Value::String(s) => { st.store.remove(s); st.order.retain(|k| k != s); Ok(Value::Boolean(true)) }
                v => Err(VmError::TypeError { expected: "String key or All".to_string(), actual: format!("{:?}", v) })
            }
        } else { Err(VmError::TypeError { expected: "Cache object".to_string(), actual: obj.type_name().to_string() }) }
    } else { Err(VmError::TypeError { expected: "Cache object".to_string(), actual: format!("{:?}", cache_obj) }) }
}

pub fn cache_stats(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 { return Err(VmError::TypeError { expected: "CacheStats[cache]".to_string(), actual: format!("{} args", args.len()) }); }
    let cache_obj = &args[0];
    if let Value::LyObj(obj) = cache_obj {
        if let Some(cache) = obj.downcast_ref::<Cache>() {
            let st = cache.state.lock().unwrap();
            let mut m = HashMap::new();
            m.insert("capacity".to_string(), Value::Integer(cache.capacity as i64));
            m.insert("size".to_string(), Value::Integer(st.store.len() as i64));
            m.insert("hits".to_string(), Value::Integer(st.hits as i64));
            m.insert("misses".to_string(), Value::Integer(st.misses as i64));
            Ok(Value::Object(m))
        } else { Err(VmError::TypeError { expected: "Cache object".to_string(), actual: obj.type_name().to_string() }) }
    } else { Err(VmError::TypeError { expected: "Cache object".to_string(), actual: format!("{:?}", cache_obj) }) }
}

pub fn register_cache_functions() -> std::collections::HashMap<String, fn(&[Value]) -> VmResult<Value>> {
    let mut f = std::collections::HashMap::new();
    f.insert("CacheCreate".to_string(), cache_create as fn(&[Value]) -> VmResult<Value>);
    f.insert("CacheGet".to_string(), cache_get as fn(&[Value]) -> VmResult<Value>);
    f.insert("CachePut".to_string(), cache_put as fn(&[Value]) -> VmResult<Value>);
    f.insert("CacheInvalidate".to_string(), cache_invalidate as fn(&[Value]) -> VmResult<Value>);
    f.insert("CacheStats".to_string(), cache_stats as fn(&[Value]) -> VmResult<Value>);
    f
}
