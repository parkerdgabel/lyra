use lyra::vm::Value;
use lyra::stdlib::async_ops::{AsyncFuture, ThreadPool, Channel};
use lyra::stdlib::StandardLibrary;
use lyra::foreign::LyObj;

/// Test that async API methods use PascalCase naming for Wolfram Language compatibility
#[test]
fn test_async_future_pascal_case_naming() {
    // Create a Future object directly
    let future = AsyncFuture::resolved(Value::Integer(42));
    let future_obj = LyObj::new(Box::new(future));
    
    assert_eq!(future_obj.type_name(), "Future");
    
    // Test PascalCase method names
    // IsResolved (should replace isResolved)
    let is_resolved = future_obj.call_method("IsResolved", &[]);
    assert!(is_resolved.is_ok(), "IsResolved method should exist and work");
    
    // Resolve (should replace resolve)
    let resolve = future_obj.call_method("Resolve", &[]);
    assert!(resolve.is_ok(), "Resolve method should exist and work");
    
    // Old camelCase names should no longer work
    let old_is_resolved = future_obj.call_method("isResolved", &[]);
    assert!(old_is_resolved.is_err(), "Old camelCase 'isResolved' should not work");
    
    let old_resolve = future_obj.call_method("resolve", &[]);
    assert!(old_resolve.is_err(), "Old camelCase 'resolve' should not work");
}

#[test]
fn test_async_thread_pool_pascal_case_naming() {
    // Create a ThreadPool object directly
    let thread_pool = ThreadPool::new(2);
    let thread_pool_obj = LyObj::new(Box::new(thread_pool));
    
    assert_eq!(thread_pool_obj.type_name(), "ThreadPool");
    
    // Test PascalCase method names
    // WorkerCount (should replace workerCount)
    let worker_count = thread_pool_obj.call_method("WorkerCount", &[]);
    assert!(worker_count.is_ok(), "WorkerCount method should exist and work");
    
    // PendingTasks (should replace pendingTasks) 
    let pending_tasks = thread_pool_obj.call_method("PendingTasks", &[]);
    assert!(pending_tasks.is_ok(), "PendingTasks method should exist and work");
    
    // Submit a task to test other methods
    let submit_result = thread_pool_obj.call_method("Submit", &[Value::Function("Add".to_string()), Value::Integer(1), Value::Integer(2)]);
    assert!(submit_result.is_ok(), "Submit method should work");
    
    if let Ok(Value::Integer(task_id)) = submit_result {
        // IsCompleted (should replace isCompleted)
        let is_completed = thread_pool_obj.call_method("IsCompleted", &[Value::Integer(task_id)]);
        assert!(is_completed.is_ok(), "IsCompleted method should exist and work");
        
        // GetResult (should replace getResult)
        let get_result = thread_pool_obj.call_method("GetResult", &[Value::Integer(task_id)]);
        assert!(get_result.is_ok(), "GetResult method should exist and work");
    }
    
    // Old camelCase names should no longer work
    let old_worker_count = thread_pool_obj.call_method("workerCount", &[]);
    assert!(old_worker_count.is_err(), "Old camelCase 'workerCount' should not work");
    
    let old_pending_tasks = thread_pool_obj.call_method("pendingTasks", &[]);
    assert!(old_pending_tasks.is_err(), "Old camelCase 'pendingTasks' should not work");
    
    let old_is_completed = thread_pool_obj.call_method("isCompleted", &[Value::Integer(0)]);
    assert!(old_is_completed.is_err(), "Old camelCase 'isCompleted' should not work");
    
    let old_get_result = thread_pool_obj.call_method("getResult", &[Value::Integer(0)]);
    assert!(old_get_result.is_err(), "Old camelCase 'getResult' should not work");
}

#[test]
fn test_async_channel_pascal_case_naming() {
    // Create a Channel object directly
    let channel = Channel::unbounded();
    let channel_obj = LyObj::new(Box::new(channel));
    
    assert_eq!(channel_obj.type_name(), "Channel");
    
    // Test PascalCase method names
    // IsClosed (should replace isClosed)
    let is_closed = channel_obj.call_method("IsClosed", &[]);
    assert!(is_closed.is_ok(), "IsClosed method should exist and work");
    
    // IsEmpty (should replace isEmpty)
    let is_empty = channel_obj.call_method("IsEmpty", &[]);
    assert!(is_empty.is_ok(), "IsEmpty method should exist and work");
    
    // TrySend (should replace trySend)
    let try_send = channel_obj.call_method("TrySend", &[Value::Integer(42)]);
    assert!(try_send.is_ok(), "TrySend method should exist and work");
    
    // TryReceive (should replace tryReceive)
    let try_receive = channel_obj.call_method("TryReceive", &[]);
    assert!(try_receive.is_ok(), "TryReceive method should exist and work");
    
    // Old camelCase names should no longer work
    let old_is_closed = channel_obj.call_method("isClosed", &[]);
    assert!(old_is_closed.is_err(), "Old camelCase 'isClosed' should not work");
    
    let old_is_empty = channel_obj.call_method("isEmpty", &[]);
    assert!(old_is_empty.is_err(), "Old camelCase 'isEmpty' should not work");
    
    let old_try_send = channel_obj.call_method("trySend", &[Value::Integer(1)]);
    assert!(old_try_send.is_err(), "Old camelCase 'trySend' should not work");
    
    let old_try_receive = channel_obj.call_method("tryReceive", &[]);
    assert!(old_try_receive.is_err(), "Old camelCase 'tryReceive' should not work");
}

#[test]
fn test_help_system_for_async_objects() {
    // Create a Future object directly
    let future = AsyncFuture::resolved(Value::Integer(42));
    let future_obj = LyObj::new(Box::new(future));
    
    // Test Help method (will be implemented)
    let help_result = future_obj.call_method("Help", &[]);
    // This should work after we implement the help system
    
    // Test ListMethods method (will be implemented)
    let methods_result = future_obj.call_method("ListMethods", &[]);
    // This should work after we implement the introspection system
}

#[test] 
fn test_async_stdlib_function_availability() {
    let stdlib = StandardLibrary::new();
    
    // Test that all async functions are available with correct PascalCase names
    let async_functions = vec![
        "Promise",
        "Await", 
        "AsyncFunction",
        "ThreadPool",
        "Channel",
        "BoundedChannel",
        "Send",
        "Receive", 
        "TrySend",
        "TryReceive",
        "ChannelClose",
        "Parallel",
        "ParallelMap",
        "ParallelReduce",
        "Pipeline",
        "All",
        "Any",
    ];
    
    for func_name in async_functions {
        let function = stdlib.get_function(func_name);
        assert!(function.is_some(), "Async function {} should be available in stdlib", func_name);
    }
}