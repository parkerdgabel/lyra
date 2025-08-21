use lyra::vm::{Value, VmError};
use lyra::stdlib::StandardLibrary;

/// Helper function to execute a function with arguments
fn exec_fn(stdlib: &StandardLibrary, name: &str, args: &[Value]) -> Result<Value, VmError> {
    match stdlib.get_function(name) {
        Some(f) => f(args),
        None => Err(VmError::Runtime(format!("Function {} not found", name))),
    }
}

/// Helper to create a list value
fn list(items: Vec<Value>) -> Value {
    Value::List(items)
}

/// Helper to create an integer value
fn int(n: i64) -> Value {
    Value::Integer(n)
}

/// Helper to create a string value
fn str(s: &str) -> Value {
    Value::String(s.to_string())
}

#[cfg(test)]
mod set_operations_tests {
    use super::*;

    #[test]
    fn test_set_create_removes_duplicates() {
        let stdlib = StandardLibrary::new();
        let input = list(vec![int(1), int(2), int(2), int(3), int(1)]);
        
        let result = exec_fn(&stdlib, "SetCreate", &[input]);
        
        // Should create a set with unique elements {1, 2, 3}
        assert!(result.is_ok());
        let set_val = result.unwrap();
        if let Value::List(items) = set_val {
            assert_eq!(items.len(), 3);
            assert!(items.contains(&int(1)));
            assert!(items.contains(&int(2)));
            assert!(items.contains(&int(3)));
        } else {
            panic!("Expected List value for set");
        }
    }

    #[test]
    fn test_set_create_empty_list() {
        let stdlib = StandardLibrary::new();
        let input = list(vec![]);
        
        let result = exec_fn(&stdlib, "SetCreate", &[input]);
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), list(vec![]));
    }

    #[test]
    fn test_set_union() {
        let stdlib = StandardLibrary::new();
        let set1 = list(vec![int(1), int(2)]);
        let set2 = list(vec![int(2), int(3)]);
        
        let result = exec_fn(&stdlib, "SetUnion", &[set1, set2]);
        
        assert!(result.is_ok());
        let union_set = result.unwrap();
        if let Value::List(items) = union_set {
            assert_eq!(items.len(), 3);
            assert!(items.contains(&int(1)));
            assert!(items.contains(&int(2)));
            assert!(items.contains(&int(3)));
        } else {
            panic!("Expected List value for union");
        }
    }

    #[test]
    fn test_set_intersection() {
        let stdlib = StandardLibrary::new();
        let set1 = list(vec![int(1), int(2), int(3)]);
        let set2 = list(vec![int(2), int(3), int(4)]);
        
        let result = exec_fn(&stdlib, "SetIntersection", &[set1, set2]);
        
        assert!(result.is_ok());
        let intersection_set = result.unwrap();
        if let Value::List(items) = intersection_set {
            assert_eq!(items.len(), 2);
            assert!(items.contains(&int(2)));
            assert!(items.contains(&int(3)));
        } else {
            panic!("Expected List value for intersection");
        }
    }

    #[test]
    fn test_set_difference() {
        let stdlib = StandardLibrary::new();
        let set1 = list(vec![int(1), int(2), int(3)]);
        let set2 = list(vec![int(2), int(3), int(4)]);
        
        let result = exec_fn(&stdlib, "SetDifference", &[set1, set2]);
        
        assert!(result.is_ok());
        let diff_set = result.unwrap();
        if let Value::List(items) = diff_set {
            assert_eq!(items.len(), 1);
            assert!(items.contains(&int(1)));
        } else {
            panic!("Expected List value for difference");
        }
    }

    #[test]
    fn test_set_contains_true() {
        let stdlib = StandardLibrary::new();
        let set = list(vec![int(1), int(2), int(3)]);
        
        let result = exec_fn(&stdlib, "SetContains", &[set, int(2)]);
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Boolean(true));
    }

    #[test]
    fn test_set_contains_false() {
        let stdlib = StandardLibrary::new();
        let set = list(vec![int(1), int(2), int(3)]);
        
        let result = exec_fn(&stdlib, "SetContains", &[set, int(4)]);
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Boolean(false));
    }

    #[test]
    fn test_set_add() {
        let stdlib = StandardLibrary::new();
        let set = list(vec![int(1), int(2)]);
        
        let result = exec_fn(&stdlib, "SetAdd", &[set, int(3)]);
        
        assert!(result.is_ok());
        let new_set = result.unwrap();
        if let Value::List(items) = new_set {
            assert_eq!(items.len(), 3);
            assert!(items.contains(&int(1)));
            assert!(items.contains(&int(2)));
            assert!(items.contains(&int(3)));
        } else {
            panic!("Expected List value for set after add");
        }
    }

    #[test]
    fn test_set_add_duplicate() {
        let stdlib = StandardLibrary::new();
        let set = list(vec![int(1), int(2)]);
        
        let result = exec_fn(&stdlib, "SetAdd", &[set, int(2)]);
        
        assert!(result.is_ok());
        let new_set = result.unwrap();
        if let Value::List(items) = new_set {
            assert_eq!(items.len(), 2); // Should not add duplicate
            assert!(items.contains(&int(1)));
            assert!(items.contains(&int(2)));
        } else {
            panic!("Expected List value for set after add duplicate");
        }
    }

    #[test]
    fn test_set_remove() {
        let stdlib = StandardLibrary::new();
        let set = list(vec![int(1), int(2), int(3)]);
        
        let result = exec_fn(&stdlib, "SetRemove", &[set, int(2)]);
        
        assert!(result.is_ok());
        let new_set = result.unwrap();
        if let Value::List(items) = new_set {
            assert_eq!(items.len(), 2);
            assert!(items.contains(&int(1)));
            assert!(items.contains(&int(3)));
            assert!(!items.contains(&int(2)));
        } else {
            panic!("Expected List value for set after remove");
        }
    }

    #[test]
    fn test_set_size() {
        let stdlib = StandardLibrary::new();
        let set = list(vec![int(1), int(2), int(3)]);
        
        let result = exec_fn(&stdlib, "SetSize", &[set]);
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), int(3));
    }

    #[test]
    fn test_set_size_empty() {
        let stdlib = StandardLibrary::new();
        let set = list(vec![]);
        
        let result = exec_fn(&stdlib, "SetSize", &[set]);
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), int(0));
    }
}

#[cfg(test)]
mod dictionary_operations_tests {
    use super::*;

    #[test]
    fn test_dict_create_from_pairs() {
        let stdlib = StandardLibrary::new();
        let pairs = list(vec![
            list(vec![str("key1"), int(10)]),
            list(vec![str("key2"), int(20)]),
        ]);
        
        let result = exec_fn(&stdlib, "DictCreate", &[pairs]);
        
        assert!(result.is_ok());
        // Result should be a Foreign Dictionary object
        let dict = result.unwrap();
        assert!(matches!(dict, Value::LyObj(_)));
    }

    #[test]
    fn test_dict_create_empty() {
        let stdlib = StandardLibrary::new();
        let pairs = list(vec![]);
        
        let result = exec_fn(&stdlib, "DictCreate", &[pairs]);
        
        assert!(result.is_ok());
        let dict = result.unwrap();
        assert!(matches!(dict, Value::LyObj(_)));
    }

    #[test]
    fn test_dict_get_existing_key() {
        let stdlib = StandardLibrary::new();
        let pairs = list(vec![
            list(vec![str("key1"), int(10)]),
            list(vec![str("key2"), int(20)]),
        ]);
        let dict = exec_fn(&stdlib, "DictCreate", &[pairs]).unwrap();
        
        let result = exec_fn(&stdlib, "DictGet", &[dict, str("key1")]);
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), int(10));
    }

    #[test]
    fn test_dict_get_missing_key_with_default() {
        let stdlib = StandardLibrary::new();
        let pairs = list(vec![
            list(vec![str("key1"), int(10)]),
        ]);
        let dict = exec_fn(&stdlib, "DictCreate", &[pairs]).unwrap();
        
        let result = exec_fn(&stdlib, "DictGet", &[dict, str("missing"), int(-1)]);
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), int(-1));
    }

    #[test]
    fn test_dict_get_missing_key_no_default() {
        let stdlib = StandardLibrary::new();
        let pairs = list(vec![
            list(vec![str("key1"), int(10)]),
        ]);
        let dict = exec_fn(&stdlib, "DictCreate", &[pairs]).unwrap();
        
        let result = exec_fn(&stdlib, "DictGet", &[dict, str("missing")]);
        
        // Should return Missing or None for non-existent key
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Missing);
    }

    #[test]
    fn test_dict_set_new_key() {
        let stdlib = StandardLibrary::new();
        let pairs = list(vec![
            list(vec![str("key1"), int(10)]),
        ]);
        let dict = exec_fn(&stdlib, "DictCreate", &[pairs]).unwrap();
        
        let result = exec_fn(&stdlib, "DictSet", &[dict.clone(), str("key2"), int(20)]);
        
        assert!(result.is_ok());
        
        // Verify the key was added
        let get_result = exec_fn(&stdlib, "DictGet", &[dict, str("key2")]);
        assert!(get_result.is_ok());
        assert_eq!(get_result.unwrap(), int(20));
    }

    #[test]
    fn test_dict_set_existing_key() {
        let stdlib = StandardLibrary::new();
        let pairs = list(vec![
            list(vec![str("key1"), int(10)]),
        ]);
        let dict = exec_fn(&stdlib, "DictCreate", &[pairs]).unwrap();
        
        let result = exec_fn(&stdlib, "DictSet", &[dict.clone(), str("key1"), int(99)]);
        
        assert!(result.is_ok());
        
        // Verify the key was updated
        let get_result = exec_fn(&stdlib, "DictGet", &[dict, str("key1")]);
        assert!(get_result.is_ok());
        assert_eq!(get_result.unwrap(), int(99));
    }

    #[test]
    fn test_dict_delete() {
        let stdlib = StandardLibrary::new();
        let pairs = list(vec![
            list(vec![str("key1"), int(10)]),
            list(vec![str("key2"), int(20)]),
        ]);
        let dict = exec_fn(&stdlib, "DictCreate", &[pairs]).unwrap();
        
        let result = exec_fn(&stdlib, "DictDelete", &[dict.clone(), str("key1")]);
        
        assert!(result.is_ok());
        
        // Verify the key was deleted
        let get_result = exec_fn(&stdlib, "DictGet", &[dict, str("key1")]);
        assert!(get_result.is_ok());
        assert_eq!(get_result.unwrap(), Value::Missing);
    }

    #[test]
    fn test_dict_keys() {
        let stdlib = StandardLibrary::new();
        let pairs = list(vec![
            list(vec![str("key1"), int(10)]),
            list(vec![str("key2"), int(20)]),
        ]);
        let dict = exec_fn(&stdlib, "DictCreate", &[pairs]).unwrap();
        
        let result = exec_fn(&stdlib, "DictKeys", &[dict]);
        
        assert!(result.is_ok());
        let keys = result.unwrap();
        if let Value::List(key_list) = keys {
            assert_eq!(key_list.len(), 2);
            assert!(key_list.contains(&str("key1")));
            assert!(key_list.contains(&str("key2")));
        } else {
            panic!("Expected List value for dict keys");
        }
    }

    #[test]
    fn test_dict_values() {
        let stdlib = StandardLibrary::new();
        let pairs = list(vec![
            list(vec![str("key1"), int(10)]),
            list(vec![str("key2"), int(20)]),
        ]);
        let dict = exec_fn(&stdlib, "DictCreate", &[pairs]).unwrap();
        
        let result = exec_fn(&stdlib, "DictValues", &[dict]);
        
        assert!(result.is_ok());
        let values = result.unwrap();
        if let Value::List(value_list) = values {
            assert_eq!(value_list.len(), 2);
            assert!(value_list.contains(&int(10)));
            assert!(value_list.contains(&int(20)));
        } else {
            panic!("Expected List value for dict values");
        }
    }

    #[test]
    fn test_dict_contains_key() {
        let stdlib = StandardLibrary::new();
        let pairs = list(vec![
            list(vec![str("key1"), int(10)]),
        ]);
        let dict = exec_fn(&stdlib, "DictCreate", &[pairs]).unwrap();
        
        let result_exists = exec_fn(&stdlib, "DictContains", &[dict.clone(), str("key1")]);
        let result_missing = exec_fn(&stdlib, "DictContains", &[dict, str("missing")]);
        
        assert!(result_exists.is_ok());
        assert_eq!(result_exists.unwrap(), Value::Boolean(true));
        
        assert!(result_missing.is_ok());
        assert_eq!(result_missing.unwrap(), Value::Boolean(false));
    }

    #[test]
    fn test_dict_merge() {
        let stdlib = StandardLibrary::new();
        let pairs1 = list(vec![
            list(vec![str("key1"), int(10)]),
            list(vec![str("key2"), int(20)]),
        ]);
        let pairs2 = list(vec![
            list(vec![str("key2"), int(99)]), // Override key2
            list(vec![str("key3"), int(30)]),
        ]);
        let dict1 = exec_fn(&stdlib, "DictCreate", &[pairs1]).unwrap();
        let dict2 = exec_fn(&stdlib, "DictCreate", &[pairs2]).unwrap();
        
        let result = exec_fn(&stdlib, "DictMerge", &[dict1, dict2]);
        
        assert!(result.is_ok());
        let merged = result.unwrap();
        
        // Verify merged content
        let get_key1 = exec_fn(&stdlib, "DictGet", &[merged.clone(), str("key1")]);
        let get_key2 = exec_fn(&stdlib, "DictGet", &[merged.clone(), str("key2")]);
        let get_key3 = exec_fn(&stdlib, "DictGet", &[merged, str("key3")]);
        
        assert_eq!(get_key1.unwrap(), int(10));
        assert_eq!(get_key2.unwrap(), int(99)); // Should be overridden value
        assert_eq!(get_key3.unwrap(), int(30));
    }

    #[test]
    fn test_dict_size() {
        let stdlib = StandardLibrary::new();
        let pairs = list(vec![
            list(vec![str("key1"), int(10)]),
            list(vec![str("key2"), int(20)]),
        ]);
        let dict = exec_fn(&stdlib, "DictCreate", &[pairs]).unwrap();
        
        let result = exec_fn(&stdlib, "DictSize", &[dict]);
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), int(2));
    }

    #[test]
    fn test_dict_size_empty() {
        let stdlib = StandardLibrary::new();
        let pairs = list(vec![]);
        let dict = exec_fn(&stdlib, "DictCreate", &[pairs]).unwrap();
        
        let result = exec_fn(&stdlib, "DictSize", &[dict]);
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), int(0));
    }
}

#[cfg(test)]
mod queue_operations_tests {
    use super::*;

    #[test]
    fn test_queue_create() {
        let stdlib = StandardLibrary::new();
        
        let result = exec_fn(&stdlib, "QueueCreate", &[]);
        
        assert!(result.is_ok());
        let queue = result.unwrap();
        assert!(matches!(queue, Value::LyObj(_)));
    }

    #[test]
    fn test_queue_enqueue_dequeue() {
        let stdlib = StandardLibrary::new();
        let queue = exec_fn(&stdlib, "QueueCreate", &[]).unwrap();
        
        // Enqueue elements
        let result1 = exec_fn(&stdlib, "QueueEnqueue", &[queue.clone(), int(10)]);
        let result2 = exec_fn(&stdlib, "QueueEnqueue", &[queue.clone(), int(20)]);
        
        assert!(result1.is_ok());
        assert!(result2.is_ok());
        
        // Dequeue elements (FIFO order)
        let dequeue1 = exec_fn(&stdlib, "QueueDequeue", &[queue.clone()]);
        let dequeue2 = exec_fn(&stdlib, "QueueDequeue", &[queue]);
        
        assert!(dequeue1.is_ok());
        assert!(dequeue2.is_ok());
        assert_eq!(dequeue1.unwrap(), int(10)); // First in, first out
        assert_eq!(dequeue2.unwrap(), int(20));
    }

    #[test]
    fn test_queue_dequeue_empty() {
        let stdlib = StandardLibrary::new();
        let queue = exec_fn(&stdlib, "QueueCreate", &[]).unwrap();
        
        let result = exec_fn(&stdlib, "QueueDequeue", &[queue]);
        
        // Should return Missing or error for empty queue
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Missing);
    }

    #[test]
    fn test_queue_size() {
        let stdlib = StandardLibrary::new();
        let queue = exec_fn(&stdlib, "QueueCreate", &[]).unwrap();
        
        // Initially empty
        let size_empty = exec_fn(&stdlib, "QueueSize", &[queue.clone()]);
        assert!(size_empty.is_ok());
        assert_eq!(size_empty.unwrap(), int(0));
        
        // After enqueuing
        exec_fn(&stdlib, "QueueEnqueue", &[queue.clone(), int(10)]).unwrap();
        exec_fn(&stdlib, "QueueEnqueue", &[queue.clone(), int(20)]).unwrap();
        
        let size_full = exec_fn(&stdlib, "QueueSize", &[queue.clone()]);
        assert!(size_full.is_ok());
        assert_eq!(size_full.unwrap(), int(2));
        
        // After dequeuing
        exec_fn(&stdlib, "QueueDequeue", &[queue.clone()]).unwrap();
        
        let size_after_dequeue = exec_fn(&stdlib, "QueueSize", &[queue]);
        assert!(size_after_dequeue.is_ok());
        assert_eq!(size_after_dequeue.unwrap(), int(1));
    }
}

#[cfg(test)]
mod stack_operations_tests {
    use super::*;

    #[test]
    fn test_stack_create() {
        let stdlib = StandardLibrary::new();
        
        let result = exec_fn(&stdlib, "StackCreate", &[]);
        
        assert!(result.is_ok());
        let stack = result.unwrap();
        assert!(matches!(stack, Value::LyObj(_)));
    }

    #[test]
    fn test_stack_push_pop() {
        let stdlib = StandardLibrary::new();
        let stack = exec_fn(&stdlib, "StackCreate", &[]).unwrap();
        
        // Push elements
        let result1 = exec_fn(&stdlib, "StackPush", &[stack.clone(), int(10)]);
        let result2 = exec_fn(&stdlib, "StackPush", &[stack.clone(), int(20)]);
        
        assert!(result1.is_ok());
        assert!(result2.is_ok());
        
        // Pop elements (LIFO order)
        let pop1 = exec_fn(&stdlib, "StackPop", &[stack.clone()]);
        let pop2 = exec_fn(&stdlib, "StackPop", &[stack]);
        
        assert!(pop1.is_ok());
        assert!(pop2.is_ok());
        assert_eq!(pop1.unwrap(), int(20)); // Last in, first out
        assert_eq!(pop2.unwrap(), int(10));
    }

    #[test]
    fn test_stack_pop_empty() {
        let stdlib = StandardLibrary::new();
        let stack = exec_fn(&stdlib, "StackCreate", &[]).unwrap();
        
        let result = exec_fn(&stdlib, "StackPop", &[stack]);
        
        // Should return Missing or error for empty stack
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Missing);
    }

    #[test]
    fn test_stack_size() {
        let stdlib = StandardLibrary::new();
        let stack = exec_fn(&stdlib, "StackCreate", &[]).unwrap();
        
        // Initially empty
        let size_empty = exec_fn(&stdlib, "StackSize", &[stack.clone()]);
        assert!(size_empty.is_ok());
        assert_eq!(size_empty.unwrap(), int(0));
        
        // After pushing
        exec_fn(&stdlib, "StackPush", &[stack.clone(), int(10)]).unwrap();
        exec_fn(&stdlib, "StackPush", &[stack.clone(), int(20)]).unwrap();
        
        let size_full = exec_fn(&stdlib, "StackSize", &[stack.clone()]);
        assert!(size_full.is_ok());
        assert_eq!(size_full.unwrap(), int(2));
        
        // After popping
        exec_fn(&stdlib, "StackPop", &[stack.clone()]).unwrap();
        
        let size_after_pop = exec_fn(&stdlib, "StackSize", &[stack]);
        assert!(size_after_pop.is_ok());
        assert_eq!(size_after_pop.unwrap(), int(1));
    }
}

#[cfg(test)]
mod edge_cases_and_errors_tests {
    use super::*;

    #[test]
    fn test_invalid_argument_counts() {
        let stdlib = StandardLibrary::new();
        
        // SetCreate with no arguments
        let result = exec_fn(&stdlib, "SetCreate", &[]);
        assert!(result.is_err());
        
        // SetUnion with one argument
        let set = list(vec![int(1)]);
        let result = exec_fn(&stdlib, "SetUnion", &[set]);
        assert!(result.is_err());
        
        // DictGet with no arguments
        let result = exec_fn(&stdlib, "DictGet", &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_argument_types() {
        let stdlib = StandardLibrary::new();
        
        // SetCreate with non-list argument
        let result = exec_fn(&stdlib, "SetCreate", &[int(42)]);
        assert!(result.is_err());
        
        // SetUnion with non-list arguments
        let result = exec_fn(&stdlib, "SetUnion", &[int(1), int(2)]);
        assert!(result.is_err());
        
        // QueueEnqueue with non-queue first argument
        let result = exec_fn(&stdlib, "QueueEnqueue", &[int(42), int(10)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_large_collections() {
        let stdlib = StandardLibrary::new();
        
        // Create large set
        let large_list: Vec<Value> = (0..1000).map(|i| int(i)).collect();
        let input = list(large_list);
        
        let result = exec_fn(&stdlib, "SetCreate", &[input]);
        assert!(result.is_ok());
        
        let set = result.unwrap();
        let size_result = exec_fn(&stdlib, "SetSize", &[set]);
        assert!(size_result.is_ok());
        assert_eq!(size_result.unwrap(), int(1000));
    }

    #[test]
    fn test_mixed_value_types() {
        let stdlib = StandardLibrary::new();
        
        // Set with different value types
        let mixed_list = list(vec![
            int(1),
            str("hello"),
            Value::Boolean(true),
            Value::Float(3.14),
        ]);
        
        let result = exec_fn(&stdlib, "SetCreate", &[mixed_list]);
        assert!(result.is_ok());
        
        let set = result.unwrap();
        let size_result = exec_fn(&stdlib, "SetSize", &[set.clone()]);
        assert_eq!(size_result.unwrap(), int(4));
        
        // Test contains with different types
        let contains_int = exec_fn(&stdlib, "SetContains", &[set.clone(), int(1)]);
        let contains_str = exec_fn(&stdlib, "SetContains", &[set.clone(), str("hello")]);
        let contains_bool = exec_fn(&stdlib, "SetContains", &[set, Value::Boolean(true)]);
        
        assert_eq!(contains_int.unwrap(), Value::Boolean(true));
        assert_eq!(contains_str.unwrap(), Value::Boolean(true));
        assert_eq!(contains_bool.unwrap(), Value::Boolean(true));
    }
}