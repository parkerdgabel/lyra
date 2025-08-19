use lyra::vm::{Value, VirtualMachine};
use lyra::compiler::Compiler;
use lyra::foreign::LyObj;
use lyra::stdlib::data::ForeignTensor;
use lyra::stdlib::tensor::array_to_lyobj;
use ndarray::{ArrayD, IxDyn};

#[test]
fn test_tensor_foreign_object_creation() {
    // Test that we can create a tensor using the foreign object system
    let tensor_data = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let foreign_tensor_value = array_to_lyobj(tensor_data);
    
    // Verify it's a foreign object
    match &foreign_tensor_value {
        Value::LyObj(lyobj) => {
            assert_eq!(lyobj.type_name(), "Tensor");
        }
        _ => panic!("Expected LyObj, got {:?}", foreign_tensor_value),
    }
}

#[test]
fn test_tensor_arithmetic_with_foreign_objects() {
    use lyra::stdlib::tensor::{tensor_add};
    
    // Create two tensors as foreign objects
    let tensor_a = array_to_lyobj(ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap());
    let tensor_b = array_to_lyobj(ArrayD::from_shape_vec(IxDyn(&[2]), vec![3.0, 4.0]).unwrap());
    
    // Test addition
    let result = tensor_add(&tensor_a, &tensor_b).expect("Addition should succeed");
    
    // Verify result is also a foreign object
    match &result {
        Value::LyObj(lyobj) => {
            assert_eq!(lyobj.type_name(), "Tensor");
            // The result should be [4.0, 6.0]
        }
        _ => panic!("Expected LyObj result, got {:?}", result),
    }
}

#[test]
fn test_mixed_tensor_arithmetic() {
    use lyra::stdlib::tensor::{tensor_add};
    
    // Test foreign tensor + foreign tensor (no more legacy tensors)
    let tensor_a = array_to_lyobj(ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap());
    let tensor_b = array_to_lyobj(ArrayD::from_shape_vec(IxDyn(&[2]), vec![3.0, 4.0]).unwrap());
    
    // Both tensors are now foreign objects
    let result = tensor_add(&tensor_a, &tensor_b);
    
    // Result should always be foreign object format
    match result {
        Ok(Value::LyObj(lyobj)) => {
            assert_eq!(lyobj.type_name(), "Tensor");
        }
        Ok(other) => panic!("Expected LyObj result, got {:?}", other),
        Err(e) => panic!("Addition should succeed, got error: {:?}", e),
    }
}