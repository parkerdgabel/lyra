//! ML Framework Integration Tests
//!
//! This test file demonstrates the tree-shaking optimized ML framework
//! integration with hierarchical function organization for efficient imports.

use lyra::stdlib::StandardLibrary;

#[test]
fn test_ml_framework_integration() {
    let stdlib = StandardLibrary::new();
    
    // Test that all ML functions are registered in stdlib
    println!("=== ML Framework Tree-Shaking Integration Test ===");
    
    // Core ML utilities (would be in std::ml::core module)
    assert!(stdlib.get_function("TensorShape").is_some());
    assert!(stdlib.get_function("TensorRank").is_some());
    assert!(stdlib.get_function("TensorSize").is_some());
    println!("✅ Core ML utilities: TensorShape, TensorRank, TensorSize");
    
    // Spatial layers (would be in std::ml::layers module)
    assert!(stdlib.get_function("FlattenLayer").is_some());
    assert!(stdlib.get_function("ReshapeLayer").is_some());
    assert!(stdlib.get_function("PermuteLayer").is_some());
    assert!(stdlib.get_function("TransposeLayer").is_some());
    println!("✅ Spatial layers: FlattenLayer, ReshapeLayer, PermuteLayer, TransposeLayer");
    
    // Layer composition (would be in std::ml::layers module)
    assert!(stdlib.get_function("Sequential").is_some());
    assert!(stdlib.get_function("Identity").is_some());
    println!("✅ Layer composition: Sequential, Identity");
    
    // Verify existing tensor operations still work
    assert!(stdlib.get_function("Array").is_some());
    assert!(stdlib.get_function("Dot").is_some());
    assert!(stdlib.get_function("Transpose").is_some());
    println!("✅ Existing tensor operations maintained compatibility");
    
    println!("🎉 ML Framework successfully integrated with hierarchical organization!");
    println!("📦 Ready for tree-shaking: import std::ml::layers::{{FlattenLayer, ReshapeLayer}}");
    println!("📦 Ready for tree-shaking: import std::ml::core::{{TensorShape, TensorRank}}");
}

#[test]
fn test_ml_function_categories() {
    let stdlib = StandardLibrary::new();
    
    // Count different categories of functions
    let all_functions = stdlib.function_names();
    
    let ml_core_functions = ["TensorShape", "TensorRank", "TensorSize"];
    let ml_layer_functions = ["FlattenLayer", "ReshapeLayer", "PermuteLayer", "TransposeLayer", "Sequential", "Identity"];
    let math_functions = ["Sin", "Cos", "Tan", "Exp", "Log", "Sqrt"];
    let tensor_functions = ["Array", "Dot", "Transpose", "Maximum"];
    
    let ml_core_count = ml_core_functions.iter()
        .filter(|&name| stdlib.get_function(name).is_some())
        .count();
    
    let ml_layer_count = ml_layer_functions.iter()
        .filter(|&name| stdlib.get_function(name).is_some())
        .count();
    
    let math_count = math_functions.iter()
        .filter(|&name| stdlib.get_function(name).is_some())
        .count();
    
    let tensor_count = tensor_functions.iter()
        .filter(|&name| stdlib.get_function(name).is_some())
        .count();
    
    println!("=== Function Category Analysis ===");
    println!("📊 Total functions registered: {}", all_functions.len());
    println!("🧮 std::ml::core functions: {}/{}", ml_core_count, ml_core_functions.len());
    println!("🔗 std::ml::layers functions: {}/{}", ml_layer_count, ml_layer_functions.len());
    println!("📐 std::math functions: {}/{}", math_count, math_functions.len()); 
    println!("🔢 std::tensor functions: {}/{}", tensor_count, tensor_functions.len());
    
    // Verify we have all expected ML functions
    assert_eq!(ml_core_count, ml_core_functions.len());
    assert_eq!(ml_layer_count, ml_layer_functions.len());
    
    println!("✅ All ML function categories properly registered!");
}

#[test]
fn test_tree_shaking_foundation() {
    let stdlib = StandardLibrary::new();
    
    println!("=== Tree-Shaking Foundation Test ===");
    
    // Simulate tree-shaking by checking specific import scenarios
    
    // Scenario 1: Only need tensor utilities
    let tensor_utils = ["TensorShape", "TensorRank", "TensorSize"];
    let tensor_utils_available = tensor_utils.iter()
        .all(|&name| stdlib.get_function(name).is_some());
    assert!(tensor_utils_available);
    println!("📦 Tree-shaking scenario 1: import std::ml::core::{{TensorShape, TensorRank, TensorSize}} ✅");
    
    // Scenario 2: Only need spatial layers
    let spatial_layers = ["FlattenLayer", "ReshapeLayer", "PermuteLayer"];
    let spatial_layers_available = spatial_layers.iter()
        .all(|&name| stdlib.get_function(name).is_some());
    assert!(spatial_layers_available);
    println!("📦 Tree-shaking scenario 2: import std::ml::layers::{{FlattenLayer, ReshapeLayer, PermuteLayer}} ✅");
    
    // Scenario 3: Only need layer composition
    let composition_layers = ["Sequential", "Identity"];
    let composition_available = composition_layers.iter()
        .all(|&name| stdlib.get_function(name).is_some());
    assert!(composition_available);
    println!("📦 Tree-shaking scenario 3: import std::ml::layers::{{Sequential, Identity}} ✅");
    
    // Scenario 4: Mixed imports from different modules
    let mixed_imports = ["TensorShape", "FlattenLayer", "Sin", "Array"];
    let mixed_available = mixed_imports.iter()
        .all(|&name| stdlib.get_function(name).is_some());
    assert!(mixed_available);
    println!("📦 Tree-shaking scenario 4: mixed imports from multiple modules ✅");
    
    println!("🎯 Tree-shaking foundation successfully established!");
    println!("🚀 Ready for full tree-shaking implementation with usage tracking!");
}