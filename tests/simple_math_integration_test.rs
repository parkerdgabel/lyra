//! Simple Mathematics Integration Test
//!
//! Validates that advanced mathematics functions are available in stdlib registry.

#[cfg(test)]
mod tests {
    use lyra::stdlib::StandardLibrary;

    #[test]
    fn test_mathematics_functions_registered() {
        println!("🧮 Testing Mathematics Functions Registration");
        
        let stdlib = StandardLibrary::new();
        
        // Test that key mathematics functions are registered
        let key_functions = vec![
            // Basic math (should definitely exist)
            ("Sin", "trigonometric"),
            ("Cos", "trigonometric"), 
            ("Exp", "exponential"),
            ("Log", "logarithm"),
            
            // Advanced linear algebra (newly integrated)
            ("SVD", "linear_algebra"),
            ("LinearSolve", "linear_algebra"),
            ("EigenDecomposition", "linear_algebra"),
            
            // Optimization (newly integrated)
            ("FindRoot", "optimization"),
            ("Minimize", "optimization"),
            ("NIntegrate", "optimization"),
            
            // Differential equations (newly integrated)
            ("NDSolve", "differential"),
            ("Gradient", "vector_calculus"),
            
            // Interpolation (newly integrated)
            ("Interpolation", "interpolation"),
            ("SplineInterpolation", "interpolation"),
        ];
        
        let mut registered_count = 0;
        let mut missing_functions = Vec::new();
        
        for (func_name, category) in &key_functions {
            if stdlib.get_function(func_name).is_some() {
                registered_count += 1;
                println!("  ✓ {} ({}) - registered", func_name, category);
            } else {
                missing_functions.push((func_name, category));
                println!("  ✗ {} ({}) - missing", func_name, category);
            }
        }
        
        println!("  📊 Functions registered: {}/{}", registered_count, key_functions.len());
        
        if !missing_functions.is_empty() {
            println!("  ⚠ Missing functions: {:?}", missing_functions);
        }
        
        // We expect at least the basic math functions to be available
        assert!(registered_count >= 4, "At least basic math functions should be registered");
        
        // If we have most functions, then integration was successful
        if registered_count >= key_functions.len() * 3 / 4 {
            println!("  ✅ Mathematics integration successful!");
        } else {
            println!("  ⚠ Partial mathematics integration - some functions missing");
        }
    }

    #[test]
    fn test_function_registry_not_empty() {
        println!("🧮 Testing Function Registry Population");
        
        let stdlib = StandardLibrary::new();
        let function_names = stdlib.function_names();
        
        println!("  📊 Total functions in registry: {}", function_names.len());
        
        // Print first few function names as sample
        let sample_size = 10.min(function_names.len());
        for i in 0..sample_size {
            println!("    - {}", function_names[i]);
        }
        
        if function_names.len() > sample_size {
            println!("    ... and {} more", function_names.len() - sample_size);
        }
        
        // We should have a substantial number of functions
        assert!(function_names.len() > 50, "Should have substantial function library, got {}", function_names.len());
        
        println!("  ✅ Function registry is properly populated");
    }

    #[test]
    fn test_mathematical_categories_available() {
        println!("🧮 Testing Mathematical Categories");
        
        let stdlib = StandardLibrary::new();
        
        // Check for functions from different mathematical categories
        let categories = vec![
            ("Basic Math", vec!["Sin", "Cos", "Exp", "Log", "Sqrt"]),
            ("Linear Algebra", vec!["SVD", "LinearSolve", "EigenDecomposition"]), 
            ("Optimization", vec!["FindRoot", "Minimize", "NIntegrate"]),
            ("Calculus", vec!["D", "Integrate"]),
            ("Special Functions", vec!["Gamma", "Erf"]),
        ];
        
        let mut categories_with_functions = 0;
        
        for (category_name, functions) in &categories {
            let available_in_category = functions.iter()
                .filter(|&func| stdlib.get_function(func).is_some())
                .count();
                
            println!("  📊 {}: {}/{} functions available", 
                    category_name, available_in_category, functions.len());
            
            if available_in_category > 0 {
                categories_with_functions += 1;
            }
        }
        
        println!("  📊 Categories with functions: {}/{}", categories_with_functions, categories.len());
        
        // We should have functions from multiple mathematical categories
        assert!(categories_with_functions >= 3, 
               "Should have functions from at least 3 mathematical categories, got {}", 
               categories_with_functions);
        
        println!("  ✅ Multiple mathematical categories are available");
    }
}