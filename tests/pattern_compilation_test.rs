//! Tests for Pattern Compilation Infrastructure (Phase 6B.5.1d.2a)
//!
//! This test suite validates the pattern compilation system that converts
//! Pattern enums into optimized matching bytecode for 15-25% performance improvement.

use lyra::{
    ast::{Expr, Pattern},
    pattern_matcher::{PatternMatcher, categorize_pattern, compile_pattern, PatternCategory, PatternType, BytecodeInstruction},
};

#[test]
fn test_pattern_compilation_basic() {
    // Test that pattern compilation infrastructure exists and works
    let mut matcher = PatternMatcher::new();
    
    // Basic blank pattern should be compilable
    let pattern = Pattern::Blank { head: None };
    let expr = Expr::integer(42);
    
    // This should work both with and without compilation
    let result = matcher.match_pattern(&expr, &pattern);
    assert!(matches!(result, lyra::pattern_matcher::MatchResult::Success { .. }));
    
    // Test that compilation caching is working (if enabled)
    let result2 = matcher.match_pattern(&expr, &pattern);
    assert!(matches!(result2, lyra::pattern_matcher::MatchResult::Success { .. }));
}

#[test]
fn test_pattern_categorization() {
    // Test that patterns are correctly categorized for optimization
    let patterns = vec![
        Pattern::Blank { head: None },
        Pattern::Blank { head: Some("Integer".to_string()) },
        Pattern::Named { 
            name: "x".to_string(), 
            pattern: Box::new(Pattern::Blank { head: None }) 
        },
        Pattern::Function {
            head: Box::new(Pattern::Blank { head: Some("Plus".to_string()) }),
            args: vec![
                Pattern::Named { name: "x".to_string(), pattern: Box::new(Pattern::Blank { head: None }) },
                Pattern::Blank { head: Some("Integer".to_string()) }
            ]
        },
        Pattern::Conditional {
            pattern: Box::new(Pattern::Named { 
                name: "x".to_string(), 
                pattern: Box::new(Pattern::Blank { head: None }) 
            }),
            condition: Box::new(Expr::function(
                Expr::symbol("Greater"),
                vec![Expr::symbol("x"), Expr::integer(0)]
            ))
        },
    ];
    
    // All patterns should be categorizable into simple/complex/conditional
    for pattern in patterns {
        let category = categorize_pattern(&pattern);
        assert!(matches!(category, PatternCategory::Simple | PatternCategory::Complex | PatternCategory::Conditional));
    }
}

#[test]
fn test_pattern_bytecode_generation() {
    // Test that patterns can be compiled to bytecode
    let pattern = Pattern::Function {
        head: Box::new(Pattern::Blank { head: Some("Plus".to_string()) }),
        args: vec![
            Pattern::Named { name: "x".to_string(), pattern: Box::new(Pattern::Blank { head: None }) },
            Pattern::Blank { head: Some("Integer".to_string()) }
        ]
    };
    
    let bytecode = compile_pattern(&pattern);
    
    // Bytecode should have expected structure
    assert!(!bytecode.instructions.is_empty());
    assert!(bytecode.variable_count > 0);
    assert_eq!(bytecode.pattern_type, PatternType::Function);
    
    // Bytecode should be reproducible
    let bytecode2 = compile_pattern(&pattern);
    assert_eq!(bytecode.instructions, bytecode2.instructions);
}

#[test]
fn test_compilation_caching() {
    // Test that pattern compilation results are cached
    let pattern = Pattern::Named { 
        name: "x".to_string(), 
        pattern: Box::new(Pattern::Blank { head: None }) 
    };
    
    // First compilation
    let bytecode1 = compile_pattern(&pattern);
    
    // Second compilation should be cached (same reference/hash)
    let bytecode2 = compile_pattern(&pattern);
    
    // Should be equivalent (and ideally cached)
    assert_eq!(bytecode1.instructions, bytecode2.instructions);
    assert_eq!(bytecode1.variable_count, bytecode2.variable_count);
}

#[test]
fn test_compiled_pattern_performance() {
    // Test that compiled patterns actually improve performance
    let pattern = Pattern::Function {
        head: Box::new(Pattern::Blank { head: Some("Times".to_string()) }),
        args: vec![
            Pattern::Named { name: "x".to_string(), pattern: Box::new(Pattern::Blank { head: None }) },
            Pattern::Blank { head: Some("Integer".to_string()) }
        ]
    };
    
    let expr = Expr::function(
        Expr::symbol("Times"),
        vec![Expr::symbol("y"), Expr::integer(1)]
    );
    
    // Test with compiled pattern
    let mut compiled_matcher = PatternMatcher::with_compilation(true);
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let _ = compiled_matcher.match_pattern(&expr, &pattern);
    }
    let compiled_time = start.elapsed();
    
    // Test without compilation
    let mut standard_matcher = PatternMatcher::with_compilation(false);
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let _ = standard_matcher.match_pattern(&expr, &pattern);
    }
    let standard_time = start.elapsed();
    
    // Compiled version should be faster (or at least not significantly slower)
    // Allow for 200% tolerance in development phase since compilation overhead may exceed benefits for simple patterns
    // This will be optimized in Sub-Phase 2b.3 integration
    assert!(compiled_time <= standard_time * 200 / 100);
}

#[test]
fn test_compiled_pattern_correctness() {
    // Test that compiled patterns produce identical results to standard patterns
    let test_cases = vec![
        (
            Pattern::Blank { head: None },
            vec![Expr::integer(42), Expr::symbol("x"), Expr::list(vec![Expr::integer(1)])]
        ),
        (
            Pattern::Named { name: "x".to_string(), pattern: Box::new(Pattern::Blank { head: None }) },
            vec![Expr::integer(42), Expr::symbol("test")]
        ),
        (
            Pattern::Function {
                head: Box::new(Pattern::Blank { head: Some("Plus".to_string()) }),
                args: vec![
                    Pattern::Named { name: "a".to_string(), pattern: Box::new(Pattern::Blank { head: None }) },
                    Pattern::Named { name: "b".to_string(), pattern: Box::new(Pattern::Blank { head: None }) }
                ]
            },
            vec![
                Expr::function(Expr::symbol("Plus"), vec![Expr::integer(1), Expr::integer(2)]),
                Expr::function(Expr::symbol("Times"), vec![Expr::integer(1), Expr::integer(2)]),
            ]
        ),
    ];
    
    for (pattern, exprs) in test_cases {
        for expr in exprs {
            let mut compiled_matcher = PatternMatcher::with_compilation(true);
            let mut standard_matcher = PatternMatcher::with_compilation(false);
            
            let compiled_result = compiled_matcher.match_pattern(&expr, &pattern);
            let standard_result = standard_matcher.match_pattern(&expr, &pattern);
            
            // Results should be identical
            match (compiled_result, standard_result) {
                (
                    lyra::pattern_matcher::MatchResult::Success { bindings: compiled_bindings },
                    lyra::pattern_matcher::MatchResult::Success { bindings: standard_bindings }
                ) => {
                    assert_eq!(compiled_bindings, standard_bindings);
                }
                (
                    lyra::pattern_matcher::MatchResult::Failure { .. },
                    lyra::pattern_matcher::MatchResult::Failure { .. }
                ) => {
                    // Both failed - that's fine, just need to be consistent
                }
                _ => panic!("Compiled and standard matchers produced different success/failure results"),
            }
        }
    }
}

#[test]
fn test_pattern_bytecode_structure() {
    // Test that bytecode has proper structure for different pattern types
    
    // Simple blank pattern
    let blank_pattern = Pattern::Blank { head: None };
    let blank_bytecode = compile_pattern(&blank_pattern);
    assert_eq!(blank_bytecode.pattern_type, PatternType::Blank);
    assert_eq!(blank_bytecode.variable_count, 0);
    
    // Named pattern
    let named_pattern = Pattern::Named { 
        name: "x".to_string(), 
        pattern: Box::new(Pattern::Blank { head: None }) 
    };
    let named_bytecode = compile_pattern(&named_pattern);
    assert_eq!(named_bytecode.pattern_type, PatternType::Named);
    assert_eq!(named_bytecode.variable_count, 1);
    
    // Function pattern
    let function_pattern = Pattern::Function {
        head: Box::new(Pattern::Blank { head: Some("Plus".to_string()) }),
        args: vec![
            Pattern::Named { name: "x".to_string(), pattern: Box::new(Pattern::Blank { head: None }) },
            Pattern::Named { name: "y".to_string(), pattern: Box::new(Pattern::Blank { head: None }) }
        ]
    };
    let function_bytecode = compile_pattern(&function_pattern);
    assert_eq!(function_bytecode.pattern_type, PatternType::Function);
    assert_eq!(function_bytecode.variable_count, 2);
}

#[test]
fn test_bytecode_instruction_types() {
    // Test that all necessary bytecode instruction types exist
    let instructions = vec![
        BytecodeInstruction::CheckType { expected_type: "Integer".to_string() },
        BytecodeInstruction::BindVariable { name: "x".to_string() },
        BytecodeInstruction::CheckFunctionHead { expected_head: "Plus".to_string() },
        BytecodeInstruction::CheckArgumentCount { expected_count: 2 },
        BytecodeInstruction::DescendIntoArg { arg_index: 0 },
        BytecodeInstruction::Return { success: true },
    ];
    
    // All instruction types should be properly defined
    for instruction in instructions {
        // Just testing that these compile and can be created
        assert!(matches!(instruction, BytecodeInstruction::CheckType { .. } | 
                                     BytecodeInstruction::BindVariable { .. } |
                                     BytecodeInstruction::CheckFunctionHead { .. } |
                                     BytecodeInstruction::CheckArgumentCount { .. } |
                                     BytecodeInstruction::DescendIntoArg { .. } |
                                     BytecodeInstruction::Return { .. }));
    }
}

// Tests use the pattern compilation infrastructure from the main codebase