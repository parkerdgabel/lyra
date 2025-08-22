use lyra::stdlib::StandardLibrary;

fn main() {
    println!("Testing data module consolidation...");
    
    // Create StandardLibrary instance
    let stdlib = StandardLibrary::new();
    
    // Test that data processing functions are registered
    let data_functions = vec![
        "JSONParse", "JSONStringify", "JSONQuery", "JSONMerge", "JSONValidate",
        "CSVParse", "CSVStringify", "CSVToTable", "TableToCSV",
        "DataTransform", "DataFilter", "DataGroup", "DataJoin", "DataSort", "DataSelect", "DataRename",
        "ValidateData", "InferSchema", "ConvertTypes", "NormalizeData",
        "DataQuery", "DataIndex", "DataAggregate"
    ];
    
    let mut missing_functions = Vec::new();
    let mut found_functions = Vec::new();
    
    for function_name in data_functions {
        if stdlib.get_function(function_name).is_some() {
            found_functions.push(function_name);
        } else {
            missing_functions.push(function_name);
        }
    }
    
    println!("Found {} data processing functions:", found_functions.len());
    for func in &found_functions {
        println!("  ✓ {}", func);
    }
    
    if !missing_functions.is_empty() {
        println!("Missing {} functions:", missing_functions.len());
        for func in &missing_functions {
            println!("  ✗ {}", func);
        }
        panic!("Some data functions are missing!");
    } else {
        println!("✅ All 23 data processing functions successfully registered!");
        println!("✅ Data module consolidation test PASSED!");
    }
}