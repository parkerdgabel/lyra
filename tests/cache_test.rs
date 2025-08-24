use lyra::stdlib::utilities::cache::{cache_create, cache_get, cache_put, cache_invalidate, cache_stats};
use lyra::vm::Value;
use std::collections::HashMap;
use std::thread;
use std::time::Duration;

fn opts(pairs: Vec<(&str, Value)>) -> Value {
    let mut m = HashMap::new();
    for (k, v) in pairs { m.insert(k.to_string(), v); }
    Value::Object(m)
}

#[test]
fn test_cache_put_get_and_stats() {
    // CacheCreate["LRU", 2]
    let cache = cache_create(&[Value::String("LRU".into()), Value::Integer(2)]).unwrap();
    // Put a=1, b=2
    assert_eq!(cache_put(&[cache.clone(), Value::String("a".into()), Value::Integer(1)]).unwrap(), Value::Boolean(true));
    assert_eq!(cache_put(&[cache.clone(), Value::String("b".into()), Value::Integer(2)]).unwrap(), Value::Boolean(true));
    // Get a (hit) and c (miss)
    assert_eq!(cache_get(&[cache.clone(), Value::String("a".into())]).unwrap(), Value::Integer(1));
    assert_eq!(cache_get(&[cache.clone(), Value::String("c".into())]).unwrap(), Value::Missing);
    // Stats reflect 1 hit, 1 miss, size 2
    let s = cache_stats(&[cache.clone()]).unwrap();
    if let Value::Object(m) = s {
        assert_eq!(m.get("size"), Some(&Value::Integer(2)));
        assert_eq!(m.get("hits"), Some(&Value::Integer(1)));
        assert_eq!(m.get("misses"), Some(&Value::Integer(1)));
    } else { panic!("expected stats object"); }
}

#[test]
fn test_cache_ttl_expiry() {
    // CacheCreate["LRU", 2, {ttlMs -> 5}]
    let cache = cache_create(&[
        Value::String("LRU".into()),
        Value::Integer(2),
        opts(vec![("ttlMs", Value::Integer(5))]),
    ]).unwrap();

    assert_eq!(cache_put(&[cache.clone(), Value::String("k".into()), Value::String("v".into())]).unwrap(), Value::Boolean(true));
    // wait for expiration
    thread::sleep(Duration::from_millis(15));
    assert_eq!(cache_get(&[cache.clone(), Value::String("k".into())]).unwrap(), Value::Missing);
}

#[test]
fn test_cache_eviction_order() {
    let cache = cache_create(&[Value::String("LRU".into()), Value::Integer(2)]).unwrap();
    // a, b fill cache
    cache_put(&[cache.clone(), Value::String("a".into()), Value::Integer(1)]).unwrap();
    cache_put(&[cache.clone(), Value::String("b".into()), Value::Integer(2)]).unwrap();
    // touch a to make it most-recent
    assert_eq!(cache_get(&[cache.clone(), Value::String("a".into())]).unwrap(), Value::Integer(1));
    // insert c, should evict b
    cache_put(&[cache.clone(), Value::String("c".into()), Value::Integer(3)]).unwrap();
    assert_eq!(cache_get(&[cache.clone(), Value::String("b".into())]).unwrap(), Value::Missing);
    assert_eq!(cache_get(&[cache.clone(), Value::String("a".into())]).unwrap(), Value::Integer(1));
    assert_eq!(cache_get(&[cache.clone(), Value::String("c".into())]).unwrap(), Value::Integer(3));
    // invalidate All
    assert_eq!(cache_invalidate(&[cache.clone(), Value::Symbol("All".into())]).unwrap(), Value::Boolean(true));
    let s = cache_stats(&[cache.clone()]).unwrap();
    if let Value::Object(m) = s { assert_eq!(m.get("size"), Some(&Value::Integer(0))); } else { panic!("expected stats object"); }
}

