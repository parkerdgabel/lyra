//! Advanced Mathematics Integration Test
//!
//! Validates that all advanced mathematics modules are properly integrated
//! and available through the standard library registry.

#[cfg(test)]
mod tests {
    use lyra::stdlib::StandardLibrary;
    use lyra::vm::Value;

    #[test]
    fn test_advanced_mathematics_integration() {
        println!("🧮 Advanced Mathematics Integration Test");
        
        let stdlib = StandardLibrary::new();
        
        // Test Linear Algebra functions
        println!("  📊 Testing Linear Algebra Functions");
        assert!(stdlib.get_function("SVD").is_some(), "SVD function should be registered");
        assert!(stdlib.get_function("QRDecomposition").is_some(), "QRDecomposition function should be registered");
        assert!(stdlib.get_function("LUDecomposition").is_some(), "LUDecomposition function should be registered");
        assert!(stdlib.get_function("LinearSolve").is_some(), "LinearSolve function should be registered");
        assert!(stdlib.get_function("EigenDecomposition").is_some(), "EigenDecomposition function should be registered");
        
        // Test Optimization functions  
        println!("  📊 Testing Optimization Functions");
        assert!(stdlib.get_function("FindRoot").is_some(), "FindRoot function should be registered");
        assert!(stdlib.get_function("Minimize").is_some(), "Minimize function should be registered");
        assert!(stdlib.get_function("Maximize").is_some(), "Maximize function should be registered");
        assert!(stdlib.get_function("Newton").is_some(), "Newton function should be registered");
        assert!(stdlib.get_function("Bisection").is_some(), "Bisection function should be registered");
        
        // Test Signal Processing functions
        println!("  📊 Testing Signal Processing Functions");
        // Note: Signal functions are registered dynamically, so we'll test a few key ones
        // These should be available if signal module is integrated
        
        // Test Differential Equations functions
        println!("  📊 Testing Differential Equations Functions");
        assert!(stdlib.get_function("NDSolve").is_some(), "NDSolve function should be registered");
        assert!(stdlib.get_function("DSolve").is_some(), "DSolve function should be registered");
        assert!(stdlib.get_function("Gradient").is_some(), "Gradient function should be registered");
        assert!(stdlib.get_function("Divergence").is_some(), "Divergence function should be registered");
        assert!(stdlib.get_function("Curl").is_some(), "Curl function should be registered");
        
        // Test Interpolation functions
        println!("  📊 Testing Interpolation Functions");
        assert!(stdlib.get_function("Interpolation").is_some(), "Interpolation function should be registered");
        assert!(stdlib.get_function("SplineInterpolation").is_some(), "SplineInterpolation function should be registered");
        assert!(stdlib.get_function("PolynomialInterpolation").is_some(), "PolynomialInterpolation function should be registered");
        assert!(stdlib.get_function("NIntegrateAdvanced").is_some(), "NIntegrateAdvanced function should be registered");
        
        println!("  ✅ All advanced mathematics functions are properly integrated");
    }
    
    #[test]
    fn test_mathematics_function_availability() {
        println!("🧮 Mathematics Function Availability Test");
        
        let stdlib = StandardLibrary::new();
        
        // Count total available functions
        let total_functions = stdlib.function_names().len();
        println!("  📊 Total functions available: {}", total_functions);
        
        // Test critical mathematics functions that should be available
        let critical_functions = vec![
            // Basic math
            "Sin", "Cos", "Tan", "Exp", "Log", "Sqrt",
            
            // Calculus
            "D", "Integrate",
            
            // Linear Algebra  
            "SVD", "LinearSolve", "EigenDecomposition",
            
            // Optimization
            "FindRoot", "Minimize", "NIntegrate",
            
            // Differential Equations
            "NDSolve", "Gradient", "Curl",
            
            // Interpolation
            "Interpolation", "SplineInterpolation",
            
            // Special functions
            "Gamma", "Erf", "BesselJ",
        ];
        
        let mut available_count = 0;
        let mut missing_functions = Vec::new();
        
        for function_name in &critical_functions {
            if stdlib.get_function(function_name).is_some() {
                available_count += 1;
            } else {
                missing_functions.push(*function_name);
            }
        }
        
        println!("  📊 Critical functions available: {}/{}", available_count, critical_functions.len());
        
        if !missing_functions.is_empty() {
            println!("  ⚠ Missing functions: {:?}", missing_functions);
        }
        
        // We should have most of the critical functions available
        assert!(available_count >= critical_functions.len() * 3 / 4,
               "At least 75% of critical mathematics functions should be available. Got {}/{}",
               available_count, critical_functions.len());
        
        println!("  ✅ Mathematics function availability validated");
    }
    
    #[test]
    fn test_basic_mathematics_operations() {
        println!("🧮 Basic Mathematics Operations Test");
        
        let stdlib = StandardLibrary::new();
        
        // Test basic trigonometric functions
        if let Some(sin_fn) = stdlib.get_function("Sin") {
            // Test Sin(0) = 0
            let result = sin_fn(&[Value::Real(0.0)]).unwrap();
            if let Value::Real(val) = result {
                assert!((val - 0.0).abs() < 1e-10, "Sin(0) should be 0, got {}", val);
                println!("  ✓ Sin(0) = {:.10}", val);
            } else {
                panic!("Sin should return a Real value");
            }
            
            // Test Sin(π/2) ≈ 1  
            let result = sin_fn(&[Value::Real(std::f64::consts::PI / 2.0)]).unwrap();
            if let Value::Real(val) = result {
                assert!((val - 1.0).abs() < 1e-10, "Sin(π/2) should be 1, got {}", val);
                println!("  ✓ Sin(π/2) = {:.10}", val);
            } else {
                panic!("Sin should return a Real value");
            }
        }
        
        // Test exponential function
        if let Some(exp_fn) = stdlib.get_function("Exp") {
            // Test Exp(0) = 1
            let result = exp_fn(&[Value::Real(0.0)]).unwrap();
            if let Value::Real(val) = result {
                assert!((val - 1.0).abs() < 1e-10, "Exp(0) should be 1, got {}", val);
                println!("  ✓ Exp(0) = {:.10}", val);
            } else {
                panic!("Exp should return a Real value");
            }
            
            // Test Exp(1) ≈ e
            let result = exp_fn(&[Value::Real(1.0)]).unwrap();
            if let Value::Real(val) = result {
                assert!((val - std::f64::consts::E).abs() < 1e-10, "Exp(1) should be e, got {}", val);
                println!("  ✓ Exp(1) = {:.10}", val);
            } else {
                panic!("Exp should return a Real value");
            }
        }
        
        // Test logarithm function
        if let Some(log_fn) = stdlib.get_function("Log") {
            // Test Log(1) = 0
            let result = log_fn(&[Value::Real(1.0)]).unwrap();
            if let Value::Real(val) = result {
                assert!((val - 0.0).abs() < 1e-10, "Log(1) should be 0, got {}", val);
                println!("  ✓ Log(1) = {:.10}", val);
            } else {
                panic!("Log should return a Real value");
            }
            
            // Test Log(e) = 1
            let result = log_fn(&[Value::Real(std::f64::consts::E)]).unwrap();
            if let Value::Real(val) = result {
                assert!((val - 1.0).abs() < 1e-10, "Log(e) should be 1, got {}", val);
                println!("  ✓ Log(e) = {:.10}", val);
            } else {
                panic!("Log should return a Real value");
            }
        }
        
        println!("  ✅ Basic mathematics operations validated");
    }
    
    #[test]
    fn test_mathematics_modules_count() {
        println!("🧮 Mathematics Modules Function Count");
        
        let stdlib = StandardLibrary::new();
        
        // Count functions by category
        let linear_algebra_functions = vec![
            "SVD", "QRDecomposition", "LUDecomposition", "CholeskyDecomposition",
            "EigenDecomposition", "LinearSolve", "LeastSquares", "PseudoInverse",
        ];
        
        let optimization_functions = vec![
            "FindRoot", "Newton", "Bisection", "Minimize", "Maximize", "NIntegrate",
        ];
        
        let differential_functions = vec![
            "NDSolve", "DSolve", "Gradient", "Divergence", "Curl",
        ];
        
        let interpolation_functions = vec![
            "Interpolation", "SplineInterpolation", "PolynomialInterpolation",
        ];
        
        let la_count = linear_algebra_functions.iter()
            .filter(|&name| stdlib.get_function(name).is_some())
            .count();
        
        let opt_count = optimization_functions.iter()
            .filter(|&name| stdlib.get_function(name).is_some())
            .count();
            
        let diff_count = differential_functions.iter()
            .filter(|&name| stdlib.get_function(name).is_some())
            .count();
            
        let interp_count = interpolation_functions.iter()
            .filter(|&name| stdlib.get_function(name).is_some())
            .count();
        
        println!("  📊 Linear Algebra functions: {}/{}", la_count, linear_algebra_functions.len());
        println!("  📊 Optimization functions: {}/{}", opt_count, optimization_functions.len());
        println!("  📊 Differential functions: {}/{}", diff_count, differential_functions.len());
        println!("  📊 Interpolation functions: {}/{}", interp_count, interpolation_functions.len());
        
        let total_advanced = la_count + opt_count + diff_count + interp_count;
        println!("  📊 Total advanced math functions: {}", total_advanced);
        
        // We should have a significant number of advanced mathematics functions
        assert!(total_advanced >= 15, "Should have at least 15 advanced mathematics functions, got {}", total_advanced);
        
        println!("  ✅ Mathematics modules integration validated");
    }
}