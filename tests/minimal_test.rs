use lyra::vm::{Value, VirtualMachine};

#[test]
fn test_vm_creation() {
    let vm = VirtualMachine::new();
    // Just test that we can create a VM without panicking
    assert!(true);
}

#[test]
fn test_value_creation() {
    let val = Value::Integer(42);
    match val {
        Value::Integer(n) => assert_eq!(n, 42),
        _ => panic!("Expected Integer value"),
    }
}