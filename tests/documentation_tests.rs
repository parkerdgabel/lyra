use lyra::vm::{Value, Vm, VmResult};
use lyra::stdlib::StandardLibrary;

fn create_test_vm() -> Vm {
    let stdlib = StandardLibrary::new();
    Vm::new(stdlib)
}

fn call_function(vm: &mut Vm, name: &str, args: Vec<Value>) -> VmResult<Value> {
    if let Some(func) = vm.stdlib.get_function(name) {
        func(&args)
    } else {
        panic!("Function {} not found", name);
    }
}

#[test]
fn test_help_general() {
    let mut vm = create_test_vm();
    let result = call_function(&mut vm, "Help", vec![]).unwrap();
    
    if let Value::String(help_text) = result {
        assert!(help_text.contains("Lyra Symbolic Computation Engine Help"));
        assert!(help_text.contains("Available function categories"));
        assert!(help_text.contains("List"));
        assert!(help_text.contains("String"));
        assert!(help_text.contains("Math"));
    } else {
        panic!("Expected String result from Help[]");
    }
}

#[test]
fn test_help_specific_function() {
    let mut vm = create_test_vm();
    let result = call_function(&mut vm, "Help", vec![Value::String("Length".to_string())]).unwrap();
    
    if let Value::String(help_text) = result {
        assert!(help_text.contains("Function: Length"));
        assert!(help_text.contains("Category: List"));
        assert!(help_text.contains("Returns the length"));
        assert!(help_text.contains("Examples:"));
        assert!(help_text.contains("Length[{1, 2, 3, 4}]"));
    } else {
        panic!("Expected String result from Help[\"Length\"]");
    }
}

#[test]
fn test_help_category() {
    let mut vm = create_test_vm();
    let result = call_function(&mut vm, "Help", vec![Value::String("String".to_string())]).unwrap();
    
    if let Value::String(help_text) = result {
        assert!(help_text.contains("Functions in category 'String'"));
        assert!(help_text.contains("StringJoin"));
    } else {
        panic!("Expected String result from Help[\"String\"]");
    }
}

#[test]
fn test_function_info() {
    let mut vm = create_test_vm();
    let result = call_function(&mut vm, "FunctionInfo", vec![Value::String("StringJoin".to_string())]).unwrap();
    
    if let Value::List(info_list) = result {
        // Should return structured metadata
        assert!(!info_list.is_empty());
        
        // Check that we have name, category, description, etc.
        let has_name = info_list.iter().any(|item| {
            if let Value::List(pair) = item {
                if let (Some(Value::String(key)), Some(Value::String(_value))) = (pair.get(0), pair.get(1)) {
                    key == "Name"
                } else { false }
            } else { false }
        });
        assert!(has_name);
    } else {
        panic!("Expected List result from FunctionInfo");
    }
}

#[test]
fn test_function_list_all() {
    let mut vm = create_test_vm();
    let result = call_function(&mut vm, "FunctionList", vec![]).unwrap();
    
    if let Value::List(functions) = result {
        assert!(functions.len() > 10); // Should have many functions
        
        // Check that all items are strings
        for func in &functions {
            assert!(matches!(func, Value::String(_)));
        }
        
        // Check for some known functions
        let func_names: Vec<String> = functions.iter()
            .filter_map(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
            .collect();
        
        assert!(func_names.contains(&"Length".to_string()));
        assert!(func_names.contains(&"StringJoin".to_string()));
        assert!(func_names.contains(&"Sin".to_string()));
    } else {
        panic!("Expected List result from FunctionList[]");
    }
}

#[test]
fn test_function_list_pattern() {
    let mut vm = create_test_vm();
    let result = call_function(&mut vm, "FunctionList", vec![Value::String("String*".to_string())]).unwrap();
    
    if let Value::List(functions) = result {
        assert!(!functions.is_empty());
        
        // All returned functions should start with "String"
        for func in &functions {
            if let Value::String(name) = func {
                assert!(name.starts_with("String"));
            } else {
                panic!("Expected String in function list");
            }
        }
    } else {
        panic!("Expected List result from FunctionList[\"String*\"]");
    }
}

#[test]
fn test_function_search() {
    let mut vm = create_test_vm();
    let result = call_function(&mut vm, "FunctionSearch", vec![Value::String("string".to_string())]).unwrap();
    
    if let Value::List(functions) = result {
        assert!(!functions.is_empty());
        
        // Should find string-related functions
        let func_names: Vec<String> = functions.iter()
            .filter_map(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
            .collect();
        
        // Should contain some string functions
        assert!(func_names.iter().any(|name| name.contains("String")));
    } else {
        panic!("Expected List result from FunctionSearch");
    }
}

#[test]
fn test_functions_by_category() {
    let mut vm = create_test_vm();
    let result = call_function(&mut vm, "FunctionsByCategory", vec![Value::String("List".to_string())]).unwrap();
    
    if let Value::List(functions) = result {
        assert!(!functions.is_empty());
        
        let func_names: Vec<String> = functions.iter()
            .filter_map(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
            .collect();
        
        // Should contain list functions
        assert!(func_names.contains(&"Length".to_string()));
    } else {
        panic!("Expected List result from FunctionsByCategory");
    }
}

#[test]
fn test_generate_code_python() {
    let mut vm = create_test_vm();
    let result = call_function(&mut vm, "GenerateCode", vec![
        Value::String("Map[Square, {1, 2, 3, 4}]".to_string()),
        Value::String("Python".to_string())
    ]).unwrap();
    
    if let Value::String(code) = result {
        // Should generate some Python code
        assert!(!code.is_empty());
    } else {
        panic!("Expected String result from GenerateCode");
    }
}

#[test]
fn test_generate_code_javascript() {
    let mut vm = create_test_vm();
    let result = call_function(&mut vm, "GenerateCode", vec![
        Value::String("StringTemplate[\"Hello {name}\"]".to_string()),
        Value::String("JavaScript".to_string())
    ]).unwrap();
    
    if let Value::String(code) = result {
        // Should generate some JavaScript code
        assert!(!code.is_empty());
        assert!(code.contains("//"));
    } else {
        panic!("Expected String result from GenerateCode");
    }
}

#[test]
fn test_code_template() {
    let mut vm = create_test_vm();
    let result = call_function(&mut vm, "CodeTemplate", vec![
        Value::String("MapReduce".to_string()),
        Value::List(vec![]) // Parameters (can be empty for this test)
    ]).unwrap();
    
    if let Value::String(template) = result {
        assert!(template.contains("Map-Reduce Template"));
        assert!(template.contains("ParallelMap"));
        assert!(template.contains("ParallelReduce"));
    } else {
        panic!("Expected String result from CodeTemplate");
    }
}

#[test]
fn test_generate_stub_python() {
    let mut vm = create_test_vm();
    let result = call_function(&mut vm, "GenerateStub", vec![
        Value::String("Length".to_string()),
        Value::String("Python".to_string())
    ]).unwrap();
    
    if let Value::String(stub) = result {
        assert!(stub.contains("def "));
        assert!(stub.contains("length"));
        assert!(stub.contains("def"));
        assert!(stub.contains("pass"));
    } else {
        panic!("Expected String result from GenerateStub");
    }
}

#[test]
fn test_generate_documentation_markdown() {
    let mut vm = create_test_vm();
    let result = call_function(&mut vm, "GenerateDocumentation", vec![
        Value::List(vec![Value::String("Length".to_string()), Value::String("StringJoin".to_string())]),
        Value::String("Markdown".to_string())
    ]).unwrap();
    
    if let Value::String(docs) = result {
        assert!(docs.contains("# Lyra Function Reference"));
        assert!(docs.contains("## Length"));
        assert!(docs.contains("## StringJoin"));
        assert!(docs.contains("**Category:**"));
        assert!(docs.contains("**Signature:**"));
    } else {
        panic!("Expected String result from GenerateDocumentation");
    }
}

#[test]
fn test_api_reference() {
    let mut vm = create_test_vm();
    let result = call_function(&mut vm, "APIReference", vec![
        Value::String("String".to_string())
    ]).unwrap();
    
    if let Value::String(reference) = result {
        assert!(reference.contains("# API Reference: String"));
        assert!(reference.contains("StringJoin"));
    } else {
        panic!("Expected String result from APIReference");
    }
}

#[test]
fn test_example_usage() {
    let mut vm = create_test_vm();
    let result = call_function(&mut vm, "ExampleUsage", vec![
        Value::String("Length".to_string())
    ]).unwrap();
    
    if let Value::List(examples) = result {
        assert!(!examples.is_empty());
        
        // Each example should be a list with code, description, and output
        for example in &examples {
            if let Value::List(example_parts) = example {
                assert_eq!(example_parts.len(), 3); // code, description, output
                assert!(matches!(example_parts[0], Value::String(_)));
                assert!(matches!(example_parts[1], Value::String(_)));
                assert!(matches!(example_parts[2], Value::String(_)));
            } else {
                panic!("Expected List for each example");
            }
        }
    } else {
        panic!("Expected List result from ExampleUsage");
    }
}

#[test]
fn test_generate_examples() {
    let mut vm = create_test_vm();
    let result = call_function(&mut vm, "GenerateExamples", vec![
        Value::String("UnknownFunction".to_string())
    ]).unwrap();
    
    if let Value::List(examples) = result {
        assert!(!examples.is_empty());
        
        // Should generate basic examples even for unknown functions
        for example in &examples {
            if let Value::String(code) = example {
                assert!(code.contains("UnknownFunction"));
            } else {
                panic!("Expected String for generated example");
            }
        }
    } else {
        panic!("Expected List result from GenerateExamples");
    }
}

#[test]
fn test_export_notebook_jupyter() {
    let mut vm = create_test_vm();
    let result = call_function(&mut vm, "ExportNotebook", vec![
        Value::List(vec![
            Value::String("Length[{1, 2, 3}]".to_string()),
            Value::String("StringJoin[\"Hello\", \" \", \"World\"]".to_string())
        ]),
        Value::String("Jupyter".to_string())
    ]).unwrap();
    
    if let Value::String(notebook) = result {
        assert!(notebook.contains("\"cells\":"));
        assert!(notebook.contains("\"cell_type\": \"code\""));
        assert!(notebook.contains("Length"));
        assert!(notebook.contains("StringJoin"));
        assert!(notebook.contains("\"nbformat\": 4"));
    } else {
        panic!("Expected String result from ExportNotebook");
    }
}

#[test]
fn test_export_notebook_html() {
    let mut vm = create_test_vm();
    let result = call_function(&mut vm, "ExportNotebook", vec![
        Value::List(vec![
            Value::String("Sin[0]".to_string()),
            Value::String("Cos[Pi/2]".to_string())
        ]),
        Value::String("HTML".to_string())
    ]).unwrap();
    
    if let Value::String(html) = result {
        assert!(html.contains("<html>"));
        assert!(html.contains("<body>"));
        assert!(html.contains("Sin[0]"));
        assert!(html.contains("Cos[Pi/2]"));
        assert!(html.contains("</html>"));
    } else {
        panic!("Expected String result from ExportNotebook HTML");
    }
}

#[test]
fn test_help_nonexistent_function() {
    let mut vm = create_test_vm();
    let result = call_function(&mut vm, "Help", vec![Value::String("NonExistentFunction".to_string())]).unwrap();
    
    if let Value::String(help_text) = result {
        assert!(help_text.contains("NonExistentFunction"));
        // Should provide suggestions or indicate function not found
    } else {
        panic!("Expected String result from Help with non-existent function");
    }
}

#[test]
fn test_function_info_error() {
    let mut vm = create_test_vm();
    let result = call_function(&mut vm, "FunctionInfo", vec![Value::String("NonExistentFunction".to_string())]);
    
    // Should return an error for non-existent functions
    assert!(result.is_err());
}

#[test]
fn test_generate_code_unsupported_language() {
    let mut vm = create_test_vm();
    let result = call_function(&mut vm, "GenerateCode", vec![
        Value::String("Test expression".to_string()),
        Value::String("UnsupportedLanguage".to_string())
    ]).unwrap();
    
    if let Value::String(code) = result {
        assert!(code.contains("Unsupported language"));
    } else {
        panic!("Expected String result for unsupported language");
    }
}

#[test]
fn test_functions_by_category_empty() {
    let mut vm = create_test_vm();
    let result = call_function(&mut vm, "FunctionsByCategory", vec![
        Value::String("NonExistentCategory".to_string())
    ]).unwrap();
    
    if let Value::List(functions) = result {
        assert!(functions.is_empty()); // Should return empty list for non-existent category
    } else {
        panic!("Expected List result from FunctionsByCategory");
    }
}

#[test] 
fn test_integration_with_actual_stdlib_functions() {
    let mut vm = create_test_vm();
    
    // Test that FunctionList actually returns real stdlib functions
    let result = call_function(&mut vm, "FunctionList", vec![]).unwrap();
    
    if let Value::List(functions) = result {
        let func_names: Vec<String> = functions.iter()
            .filter_map(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
            .collect();
        
        // Test that the documented functions actually exist in the stdlib
        for func_name in &func_names {
            let exists = vm.stdlib.get_function(func_name).is_some();
            assert!(exists, "Function {} listed but not found in stdlib", func_name);
        }
        
        // Test that some core functions are present
        assert!(func_names.contains(&"Length".to_string()));
        assert!(func_names.contains(&"Map".to_string()));
        assert!(func_names.contains(&"StringJoin".to_string()));
        assert!(func_names.contains(&"Sin".to_string()));
    } else {
        panic!("Expected List from FunctionList");
    }
}