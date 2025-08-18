use lyra::linker::registry::create_global_registry;

#[test]
fn test_unified_function_index_space() {
    println!("\n🔍 UNIFIED FUNCTION INDEX SPACE VALIDATION");
    println!("==========================================");

    let registry = create_global_registry().unwrap();

    // Test total function count (Foreign 32 + Stdlib 47 = 79)
    let total_functions = registry.get_total_function_count();
    println!("📊 Total functions registered: {}", total_functions);
    assert!(total_functions >= 70, "Should have at least 70 functions (32 Foreign + 40+ Stdlib)");
    assert!(total_functions <= 90, "Should have no more than 90 functions for reasonable bounds");

    // Test Foreign method indices (should be 0-31)
    println!("\n🎯 FOREIGN METHOD INDEX VALIDATION (0-31):");
    let foreign_methods = [
        ("Series", "Length"),
        ("Series", "Type"),
        ("Tensor", "Add"),
        ("Tensor", "Dimensions"),
        ("Table", "RowCount"),
    ];

    for (type_name, method_name) in foreign_methods {
        if let Some(index) = registry.get_method_index(type_name, method_name) {
            println!("  {}::{} → index {}", type_name, method_name, index);
            assert!(index <= 31, "Foreign method {}::{} should have index ≤ 31, got {}", 
                   type_name, method_name, index);
        }
    }

    // Test stdlib function indices (should be 32+)
    println!("\n📚 STDLIB FUNCTION INDEX VALIDATION (32+):");
    let stdlib_functions = [
        "Sin", "Cos", "Length", "StringJoin", "Array"
    ];

    for function_name in stdlib_functions {
        if let Some(index) = registry.get_stdlib_index(function_name) {
            println!("  {} → index {}", function_name, index);
            assert!(index >= 32, "Stdlib function {} should have index ≥ 32, got {}", 
                   function_name, index);
            assert!(index < 80, "Stdlib function {} should have index < 80, got {}", 
                   function_name, index);
        }
    }

    // Test no index conflicts
    println!("\n🔒 INDEX CONFLICT VALIDATION:");
    let mut all_indices = std::collections::HashSet::new();
    let mut duplicate_indices = Vec::new();

    // Check Foreign method indices
    for type_name in registry.get_type_names() {
        for method_name in registry.get_type_methods(&type_name) {
            if let Some(index) = registry.get_method_index(&type_name, &method_name) {
                if !all_indices.insert(index) {
                    duplicate_indices.push((format!("{}::{}", type_name, method_name), index));
                }
            }
        }
    }

    // Check stdlib function indices
    let stdlib_names = [
        "Sin", "Cos", "Tan", "Exp", "Log", "Sqrt", 
        "Length", "Head", "Tail", "Append", "Flatten",
        "StringJoin", "StringLength", "StringTake", "StringDrop",
        "Array", "ArrayDimensions", "ArrayRank", "Transpose"
    ];

    for function_name in stdlib_names {
        if let Some(index) = registry.get_stdlib_index(function_name) {
            if !all_indices.insert(index) {
                duplicate_indices.push((function_name.to_string(), index));
            }
        }
    }

    if duplicate_indices.is_empty() {
        println!("  ✅ No index conflicts found");
    } else {
        for (func_name, index) in &duplicate_indices {
            println!("  ❌ Conflict: {} has duplicate index {}", func_name, index);
        }
        panic!("Found {} index conflicts", duplicate_indices.len());
    }

    println!("\n✅ UNIFIED INDEX SPACE VALIDATION COMPLETE");
    println!("  • Foreign methods: indices 0-31 ✅");
    println!("  • Stdlib functions: indices 32+ ✅");
    println!("  • No conflicts: {} unique indices ✅", all_indices.len());
    println!("  • Total functions: {} ✅", total_functions);
}

#[test]
fn test_function_index_lookup_performance() {
    println!("\n⚡ FUNCTION INDEX LOOKUP PERFORMANCE");
    println!("===================================");

    let registry = create_global_registry().unwrap();
    
    // Test Foreign method lookup speed
    let start = std::time::Instant::now();
    for _ in 0..10000 {
        let _ = registry.get_method_index("Series", "Length");
        let _ = registry.get_method_index("Tensor", "Add");
        let _ = registry.get_method_index("Table", "RowCount");
    }
    let foreign_duration = start.elapsed();

    // Test stdlib function lookup speed  
    let start = std::time::Instant::now();
    for _ in 0..10000 {
        let _ = registry.get_stdlib_index("Sin");
        let _ = registry.get_stdlib_index("Length");
        let _ = registry.get_stdlib_index("Array");
    }
    let stdlib_duration = start.elapsed();

    println!("📊 Index lookup performance:");
    println!("  Foreign methods: {:6.2}μs per lookup", foreign_duration.as_micros() as f64 / 30000.0);
    println!("  Stdlib functions: {:6.2}μs per lookup", stdlib_duration.as_micros() as f64 / 30000.0);

    // Both should be very fast (sub-10ms for 10k operations = very reasonable)
    assert!(foreign_duration.as_micros() < 10000, "Foreign lookup should be <10ms for 10k operations");
    assert!(stdlib_duration.as_micros() < 10000, "Stdlib lookup should be <10ms for 10k operations");

    println!("  ✅ Both lookup types are high-performance");
}

#[test] 
fn test_function_registry_completeness() {
    println!("\n🔍 FUNCTION REGISTRY COMPLETENESS CHECK");
    println!("======================================");

    let registry = create_global_registry().unwrap();

    // Verify key Foreign methods are registered
    let required_foreign_methods = [
        ("Series", "Length"), ("Series", "Type"), ("Series", "Get"), ("Series", "Append"),
        ("Tensor", "Add"), ("Tensor", "Dimensions"), ("Tensor", "Transpose"), ("Tensor", "Dot"),
        ("Table", "RowCount"), ("Table", "ColumnCount"), ("Table", "GetRow"), ("Table", "GetColumn"),
    ];

    println!("🎯 Required Foreign methods:");
    for (type_name, method_name) in required_foreign_methods {
        let has_method = registry.has_method(type_name, method_name);
        let status = if has_method { "✅" } else { "❌" };
        println!("  {} {}::{}", status, type_name, method_name);
        assert!(has_method, "Required Foreign method {}::{} not found", type_name, method_name);
    }

    // Verify key stdlib functions are registered
    let required_stdlib_functions = [
        "Sin", "Cos", "Tan", "Length", "Head", "Tail", "Append", 
        "StringJoin", "Array", "Transpose", "Dot", "Maximum"
    ];

    println!("\n📚 Required stdlib functions:");
    for function_name in required_stdlib_functions {
        let has_function = registry.has_stdlib_function(function_name);
        let status = if has_function { "✅" } else { "❌" };
        println!("  {} {}", status, function_name);
        assert!(has_function, "Required stdlib function {} not found", function_name);
    }

    println!("\n✅ Registry completeness validation passed");
}