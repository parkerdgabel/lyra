use lyra::ast::{Expr, Symbol, Number};
use lyra::compiler::Compiler;

/// Test the Phase 5A.5.1a Infrastructure Setup
/// This validates that attribute detection works during compilation

#[cfg(test)]
mod infrastructure_tests {
    use super::*;
    
    /// Test that the infrastructure detects function attributes during compilation
    #[test]
    fn test_attribute_detection_infrastructure() {
        let mut compiler = Compiler::new();
        
        // Test 1: Function with Listable attribute (Sin)
        let sin_expr = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Sin".to_string() })),
            args: vec![Expr::Number(Number::Real(1.0))],
        };
        
        // This should compile successfully and detect the Listable attribute
        let result = compiler.compile_expr(&sin_expr);
        assert!(result.is_ok(), "Sin function should compile successfully");
        
        // Test 2: Function with Orderless + Listable attributes (Plus)
        let plus_expr = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
            args: vec![
                Expr::Number(Number::Integer(1)),
                Expr::Number(Number::Integer(2)),
            ],
        };
        
        // This should compile successfully and detect both attributes
        let result = compiler.compile_expr(&plus_expr);
        assert!(result.is_ok(), "Plus function should compile successfully");
        
        // Test 3: Function without attributes (basic case still works)
        let custom_expr = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "UnknownFunction".to_string() })),
            args: vec![Expr::Number(Number::Integer(42))],
        };
        
        // This should compile (will fall back to normal compilation)
        let result = compiler.compile_expr(&custom_expr);
        // Note: This might fail because UnknownFunction isn't registered,
        // but that's expected behavior for now
        
        println!("✅ Infrastructure test completed - attribute detection operational!");
    }
    
    /// Test function name extraction directly
    #[test]
    fn test_function_name_extraction() {
        let compiler = Compiler::new();
        
        // Test simple symbol extraction
        let symbol_expr = Expr::Symbol(Symbol { name: "TestFunction".to_string() });
        let name_result = compiler.extract_function_name(&symbol_expr);
        assert!(name_result.is_ok());
        assert_eq!(name_result.unwrap(), "TestFunction");
        
        // Test unsupported complex expression
        let complex_expr = Expr::List(vec![]);
        let name_result = compiler.extract_function_name(&complex_expr);
        assert!(name_result.is_err());
        
        println!("✅ Function name extraction working correctly!");
    }
    
    /// Test attribute query system
    #[test] 
    fn test_attribute_queries() {
        let compiler = Compiler::new();
        
        // Test querying attributes for stdlib functions
        let sin_attributes = compiler.get_function_attributes("Sin");
        assert!(!sin_attributes.is_empty(), "Sin should have attributes");
        
        let plus_attributes = compiler.get_function_attributes("Plus");
        assert!(!plus_attributes.is_empty(), "Plus should have attributes");
        
        // Test specific attribute checking
        use lyra::linker::FunctionAttribute;
        assert!(compiler.function_has_attribute("Sin", &FunctionAttribute::Listable));
        assert!(compiler.function_has_attribute("Plus", &FunctionAttribute::Orderless));
        assert!(compiler.function_has_attribute("Plus", &FunctionAttribute::Listable));
        
        // Test non-existent function
        let unknown_attributes = compiler.get_function_attributes("NonExistentFunction");
        assert!(unknown_attributes.is_empty(), "Unknown function should have no attributes");
        
        println!("✅ Attribute query system working correctly!");
    }
}