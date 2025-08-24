use std::sync::{Arc, RwLock};

use lyra::linker::FunctionRegistry;
use lyra::modules::registry::ModuleRegistry;
use lyra::stdlib::StandardLibrary;
use lyra::vm::Value;

#[test]
fn audit_modules_present_and_exports() {
    let func_reg = Arc::new(RwLock::new(FunctionRegistry::new()));
    let registry = ModuleRegistry::new(func_reg);

    // Key modules we expect
    let expected = vec![
        "std::ml::core",
        "std::ml::layers",
        "std::image",
        "std::vision",
        "std::analytics::timeseries",
        "std::network",
        "std::numerical",
        "std::clustering",
        "std::geometry",
        "std::number_theory",
        "std::combinatorics",
        "std::data_processing",
        "std::topology",
    ];

    let modules = registry.list_modules();
    for ns in expected {
        assert!(modules.iter().any(|m| m == ns), "Missing module: {}", ns);
    }

    // ML exports exist
    let core_exports = registry.get_module_exports("std::ml::core");
    assert!(core_exports.iter().any(|e| e == "AIForward"));
    assert!(core_exports.iter().any(|e| e == "NetGraph"));

    let layer_exports = registry.get_module_exports("std::ml::layers");
    assert!(layer_exports.iter().any(|e| e == "Sequential"));
    assert!(layer_exports.iter().any(|e| e == "Conv2D"));
}

#[test]
fn stdlib_print_and_mean() {
    let stdlib = StandardLibrary::new();

    // Print returns Missing
    let print_fn = stdlib.get_function("Print").expect("Print registered");
    let out = print_fn(&[Value::String("hello".to_string())]).expect("Print should not error");
    match out { Value::Missing => {}, v => panic!("Expected Missing, got {:?}", v) }

    // Mean works and returns numeric
    let mean_fn = stdlib.get_function("Mean").expect("Mean registered");
    let out = mean_fn(&[Value::List(vec![1.into(), 2.into(), 3.into()])]).expect("Mean ok");
    match out { Value::Real(f) => assert!((f - 2.0).abs() < 1e-9), _ => panic!("Expected Real") }
}

#[test]
fn ml_ai_forward_sequential() {
    let stdlib = StandardLibrary::new();
    let aifn = stdlib.get_function("AIForward").expect("AIForward registered");
    let ops = Value::List(vec![Value::String("ReLU".into()), Value::String("Sigmoid".into())]);
    let input = Value::List(vec![(-1).into(), 0.into(), 2.into()]);
    let out = aifn(&[ops, input]).expect("AIForward ok");
    match out {
        Value::List(v) => {
            assert_eq!(v.len(), 3);
            // After ReLU + Sigmoid, all outputs should be in [0,1]
            for val in v {
                match val { Value::Real(f) => assert!(f >= 0.0 && f <= 1.0), _ => panic!("Expected Real") }
            }
        }
        _ => panic!("Expected List")
    }
}
#![allow(warnings)]
