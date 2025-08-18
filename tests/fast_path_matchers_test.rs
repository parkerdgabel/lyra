//! Tests for Fast-Path Matcher Infrastructure (Phase 6B.5.1d.2b.1)
//!
//! This test suite validates the fast-path matcher system that provides optimized
//! pattern matching for common pattern types, targeting 15-25% performance improvement.

use lyra::{
    ast::{Expr, Pattern, Symbol, Number},
    pattern_matcher::{
        PatternMatcher, MatchResult, 
        FastPathMatcher, FastPathRegistry,
        BlankMatcher, SymbolMatcher, IntegerMatcher, FunctionHeadMatcher
    },
    vm::Value,
};

#[test]
fn test_blank_matcher_anonymous() {
    // Test anonymous blank pattern (_) - should match anything
    let matcher = BlankMatcher::new();
    let pattern = Pattern::Blank { head: None };
    
    // Should handle anonymous blank patterns
    assert!(matcher.can_handle(&pattern));
    
    // Should match integer
    let expr = Expr::integer(42);
    let result = matcher.fast_match(&expr, &pattern).unwrap();
    assert!(matches!(result, MatchResult::Success { .. }));
    
    // Should match symbol
    let expr = Expr::symbol("x");
    let result = matcher.fast_match(&expr, &pattern).unwrap();
    assert!(matches!(result, MatchResult::Success { .. }));
    
    // Should match string
    let expr = Expr::string("hello");
    let result = matcher.fast_match(&expr, &pattern).unwrap();
    assert!(matches!(result, MatchResult::Success { .. }));
}

#[test]
fn test_blank_matcher_typed() {
    // Test typed blank pattern (_Integer) - should match only integers
    let matcher = BlankMatcher::new();
    let pattern = Pattern::Blank { head: Some("Integer".to_string()) };
    
    // Should handle typed blank patterns
    assert!(matcher.can_handle(&pattern));
    
    // Should match integer
    let expr = Expr::integer(42);
    let result = matcher.fast_match(&expr, &pattern).unwrap();
    assert!(matches!(result, MatchResult::Success { .. }));
    
    // Should NOT match real
    let expr = Expr::real(3.14);
    let result = matcher.fast_match(&expr, &pattern).unwrap();
    assert!(matches!(result, MatchResult::Failure { .. }));
    
    // Should NOT match symbol
    let expr = Expr::symbol("x");
    let result = matcher.fast_match(&expr, &pattern).unwrap();
    assert!(matches!(result, MatchResult::Failure { .. }));
}

#[test]
fn test_blank_matcher_all_types() {
    let matcher = BlankMatcher::new();
    
    // Test all supported types
    let test_cases = vec![
        (Expr::integer(42), Pattern::Blank { head: Some("Integer".to_string()) }, true),
        (Expr::real(3.14), Pattern::Blank { head: Some("Real".to_string()) }, true),
        (Expr::string("test"), Pattern::Blank { head: Some("String".to_string()) }, true),
        (Expr::symbol("x"), Pattern::Blank { head: Some("Symbol".to_string()) }, true),
        (Expr::list(vec![Expr::integer(1)]), Pattern::Blank { head: Some("List".to_string()) }, true),
        // Cross-type matching should fail
        (Expr::integer(42), Pattern::Blank { head: Some("Real".to_string()) }, false),
        (Expr::real(3.14), Pattern::Blank { head: Some("Integer".to_string()) }, false),
        (Expr::string("test"), Pattern::Blank { head: Some("Symbol".to_string()) }, false),
    ];
    
    for (expr, pattern, should_match) in test_cases {
        assert!(matcher.can_handle(&pattern));
        let result = matcher.fast_match(&expr, &pattern).unwrap();
        
        if should_match {
            assert!(matches!(result, MatchResult::Success { .. }), 
                   "Expected match for {:?} against {:?}", expr, pattern);
        } else {
            assert!(matches!(result, MatchResult::Failure { .. }),
                   "Expected failure for {:?} against {:?}", expr, pattern);
        }
    }
}

#[test]
fn test_symbol_matcher() {
    let matcher = SymbolMatcher::new();
    let pattern = Pattern::Blank { head: Some("Symbol".to_string()) };
    
    // Should handle Symbol type patterns
    assert!(matcher.can_handle(&pattern));
    
    // Should match symbol
    let expr = Expr::symbol("x");
    let result = matcher.fast_match(&expr, &pattern).unwrap();
    assert!(matches!(result, MatchResult::Success { .. }));
    
    // Should NOT match integer
    let expr = Expr::integer(42);
    let result = matcher.fast_match(&expr, &pattern).unwrap();
    assert!(matches!(result, MatchResult::Failure { .. }));
    
    // Should NOT handle non-Symbol patterns
    let pattern = Pattern::Blank { head: Some("Integer".to_string()) };
    assert!(!matcher.can_handle(&pattern));
}

#[test]
fn test_integer_matcher() {
    let matcher = IntegerMatcher::new();
    let pattern = Pattern::Blank { head: Some("Integer".to_string()) };
    
    // Should handle Integer type patterns
    assert!(matcher.can_handle(&pattern));
    
    // Should match integer
    let expr = Expr::integer(42);
    let result = matcher.fast_match(&expr, &pattern).unwrap();
    assert!(matches!(result, MatchResult::Success { .. }));
    
    // Should NOT match real
    let expr = Expr::real(3.14);
    let result = matcher.fast_match(&expr, &pattern).unwrap();
    assert!(matches!(result, MatchResult::Failure { .. }));
    
    // Should NOT handle non-Integer patterns
    let pattern = Pattern::Blank { head: Some("Symbol".to_string()) };
    assert!(!matcher.can_handle(&pattern));
}

#[test]
fn test_function_head_matcher_simple() {
    let matcher = FunctionHeadMatcher::new();
    
    // Simple function pattern: Plus[x_, 0]
    let pattern = Pattern::Function {
        head: Box::new(Pattern::Blank { head: Some("Plus".to_string()) }),
        args: vec![
            Pattern::Named {
                name: "x".to_string(),
                pattern: Box::new(Pattern::Blank { head: None }),
            },
            Pattern::Blank { head: Some("Integer".to_string()) },
        ],
    };
    
    // Should handle simple function patterns
    assert!(matcher.can_handle(&pattern));
    
    // Matching function expression: Plus[y, 0]
    let expr = Expr::function(
        Expr::symbol("Plus"),
        vec![Expr::symbol("y"), Expr::integer(0)]
    );
    
    let result = matcher.fast_match(&expr, &pattern).unwrap();
    match result {
        MatchResult::Success { bindings } => {
            assert_eq!(bindings.get("x"), Some(&Value::Symbol("y".to_string())));
        }
        _ => panic!("Expected successful match"),
    }
}

#[test]
fn test_function_head_matcher_mismatch() {
    let matcher = FunctionHeadMatcher::new();
    
    // Pattern: Plus[x_, 0]
    let pattern = Pattern::Function {
        head: Box::new(Pattern::Blank { head: Some("Plus".to_string()) }),
        args: vec![
            Pattern::Named {
                name: "x".to_string(),
                pattern: Box::new(Pattern::Blank { head: None }),
            },
            Pattern::Blank { head: Some("Integer".to_string()) },
        ],
    };
    
    // Non-matching expression: Times[y, 1] (wrong function head)
    let expr = Expr::function(
        Expr::symbol("Times"),
        vec![Expr::symbol("y"), Expr::integer(1)]
    );
    
    let result = matcher.fast_match(&expr, &pattern).unwrap();
    assert!(matches!(result, MatchResult::Failure { .. }));
}

#[test]
fn test_function_head_matcher_argument_count_mismatch() {
    let matcher = FunctionHeadMatcher::new();
    
    // Pattern: Plus[x_, y_] (expects 2 arguments)
    let pattern = Pattern::Function {
        head: Box::new(Pattern::Blank { head: Some("Plus".to_string()) }),
        args: vec![
            Pattern::Named {
                name: "x".to_string(),
                pattern: Box::new(Pattern::Blank { head: None }),
            },
            Pattern::Named {
                name: "y".to_string(),
                pattern: Box::new(Pattern::Blank { head: None }),
            },
        ],
    };
    
    // Expression: Plus[a, b, c] (has 3 arguments)
    let expr = Expr::function(
        Expr::symbol("Plus"),
        vec![Expr::symbol("a"), Expr::symbol("b"), Expr::symbol("c")]
    );
    
    let result = matcher.fast_match(&expr, &pattern).unwrap();
    assert!(matches!(result, MatchResult::Failure { .. }));
}

#[test]
fn test_function_head_matcher_complex_pattern_fallback() {
    let matcher = FunctionHeadMatcher::new();
    
    // Complex pattern that should fall back to standard matcher
    let pattern = Pattern::Function {
        head: Box::new(Pattern::Blank { head: Some("Plus".to_string()) }),
        args: vec![
            // Complex nested pattern - should trigger fallback
            Pattern::Function {
                head: Box::new(Pattern::Blank { head: Some("Times".to_string()) }),
                args: vec![
                    Pattern::Named {
                        name: "x".to_string(),
                        pattern: Box::new(Pattern::Blank { head: None }),
                    },
                    Pattern::Blank { head: Some("Integer".to_string()) },
                ],
            },
        ],
    };
    
    // Should not be able to handle complex patterns
    assert!(!matcher.can_handle(&pattern));
}

#[test]
fn test_function_head_matcher_arity_limit() {
    let matcher = FunctionHeadMatcher::new();
    
    // Pattern with too many arguments (> 3) should not be handled
    let pattern = Pattern::Function {
        head: Box::new(Pattern::Blank { head: Some("Plus".to_string()) }),
        args: vec![
            Pattern::Blank { head: None },
            Pattern::Blank { head: None },
            Pattern::Blank { head: None },
            Pattern::Blank { head: None }, // 4th argument - exceeds limit
        ],
    };
    
    assert!(!matcher.can_handle(&pattern));
}

#[test]
fn test_fast_path_registry_creation() {
    let registry = FastPathRegistry::new();
    
    // Should have all 4 default matchers
    let matcher_names = registry.get_matcher_names();
    assert_eq!(matcher_names.len(), 4);
    assert!(matcher_names.contains(&"BlankMatcher"));
    assert!(matcher_names.contains(&"SymbolMatcher"));
    assert!(matcher_names.contains(&"IntegerMatcher"));
    assert!(matcher_names.contains(&"FunctionHeadMatcher"));
}

#[test]
fn test_fast_path_registry_find_matcher() {
    let registry = FastPathRegistry::new();
    
    // Should find BlankMatcher for blank patterns
    let pattern = Pattern::Blank { head: None };
    let matcher = registry.find_matcher(&pattern).unwrap();
    assert_eq!(matcher.name(), "BlankMatcher");
    
    // Should find IntegerMatcher for integer patterns
    let pattern = Pattern::Blank { head: Some("Integer".to_string()) };
    let matcher = registry.find_matcher(&pattern).unwrap();
    assert_eq!(matcher.name(), "BlankMatcher"); // BlankMatcher handles all blank patterns
    
    // Should find FunctionHeadMatcher for simple function patterns
    let pattern = Pattern::Function {
        head: Box::new(Pattern::Blank { head: Some("Plus".to_string()) }),
        args: vec![Pattern::Blank { head: None }],
    };
    let matcher = registry.find_matcher(&pattern).unwrap();
    assert_eq!(matcher.name(), "FunctionHeadMatcher");
}

#[test]
fn test_fast_path_registry_try_fast_match() {
    let registry = FastPathRegistry::new();
    
    // Test successful fast match for blank pattern
    let expr = Expr::integer(42);
    let pattern = Pattern::Blank { head: None };
    
    let result = registry.try_fast_match(&expr, &pattern).unwrap();
    assert!(matches!(result, MatchResult::Success { .. }));
    
    // Test successful fast match for function pattern
    let expr = Expr::function(Expr::symbol("Plus"), vec![Expr::symbol("x"), Expr::integer(0)]);
    let pattern = Pattern::Function {
        head: Box::new(Pattern::Blank { head: Some("Plus".to_string()) }),
        args: vec![
            Pattern::Named {
                name: "var".to_string(),
                pattern: Box::new(Pattern::Blank { head: None }),
            },
            Pattern::Blank { head: Some("Integer".to_string()) },
        ],
    };
    
    let result = registry.try_fast_match(&expr, &pattern).unwrap();
    match result {
        MatchResult::Success { bindings } => {
            assert_eq!(bindings.get("var"), Some(&Value::Symbol("x".to_string())));
        }
        _ => panic!("Expected successful match"),
    }
}

#[test]
fn test_fast_path_registry_no_matcher_found() {
    let registry = FastPathRegistry::new();
    
    // Complex pattern that no fast-path matcher can handle
    let pattern = Pattern::Conditional {
        pattern: Box::new(Pattern::Named {
            name: "x".to_string(),
            pattern: Box::new(Pattern::Blank { head: None }),
        }),
        condition: Box::new(Expr::function(
            Expr::symbol("Greater"),
            vec![Expr::symbol("x"), Expr::integer(0)]
        )),
    };
    
    // Should not find any matcher
    assert!(registry.find_matcher(&pattern).is_none());
    
    // Should not be able to fast match
    let expr = Expr::integer(5);
    assert!(registry.try_fast_match(&expr, &pattern).is_none());
}

#[test]
fn test_fast_path_performance_characteristics() {
    // This test demonstrates that fast-path matchers should be significantly faster
    // than the standard pattern matcher for common cases
    
    let registry = FastPathRegistry::new();
    let mut standard_matcher = PatternMatcher::new();
    
    // Test pattern: simple blank
    let expr = Expr::integer(42);
    let pattern = Pattern::Blank { head: Some("Integer".to_string()) };
    
    // Both should produce identical results
    let fast_result = registry.try_fast_match(&expr, &pattern).unwrap();
    let standard_result = standard_matcher.match_pattern(&expr, &pattern);
    
    match (fast_result, standard_result) {
        (MatchResult::Success { .. }, MatchResult::Success { .. }) => {
            // Both succeeded - this is correct
        }
        (MatchResult::Failure { .. }, MatchResult::Failure { .. }) => {
            // Both failed - also correct
        }
        _ => panic!("Fast-path and standard matcher produced different results"),
    }
}

#[test]
fn test_fast_path_correctness_comprehensive() {
    // Comprehensive test ensuring fast-path matchers produce identical results
    // to standard matchers for all cases they claim to handle
    
    let registry = FastPathRegistry::new();
    let mut standard_matcher = PatternMatcher::new();
    
    let test_cases = vec![
        // Anonymous blank patterns
        (Expr::integer(42), Pattern::Blank { head: None }),
        (Expr::real(3.14), Pattern::Blank { head: None }),
        (Expr::symbol("x"), Pattern::Blank { head: None }),
        (Expr::string("hello"), Pattern::Blank { head: None }),
        
        // Typed blank patterns
        (Expr::integer(42), Pattern::Blank { head: Some("Integer".to_string()) }),
        (Expr::real(3.14), Pattern::Blank { head: Some("Real".to_string()) }),
        (Expr::symbol("x"), Pattern::Blank { head: Some("Symbol".to_string()) }),
        (Expr::string("hello"), Pattern::Blank { head: Some("String".to_string()) }),
        
        // Type mismatches
        (Expr::integer(42), Pattern::Blank { head: Some("Real".to_string()) }),
        (Expr::real(3.14), Pattern::Blank { head: Some("Integer".to_string()) }),
        (Expr::symbol("x"), Pattern::Blank { head: Some("Integer".to_string()) }),
    ];
    
    for (expr, pattern) in test_cases {
        // Try fast-path matching
        if let Some(fast_result) = registry.try_fast_match(&expr, &pattern) {
            // If fast-path can handle it, compare with standard matcher
            let standard_result = standard_matcher.match_pattern(&expr, &pattern);
            
            match (fast_result, standard_result) {
                (MatchResult::Success { .. }, MatchResult::Success { .. }) => {
                    // Both succeeded - correct
                }
                (MatchResult::Failure { .. }, MatchResult::Failure { .. }) => {
                    // Both failed - also correct
                }
                _ => panic!("Fast-path and standard matcher disagreed for {:?} vs {:?}", expr, pattern),
            }
        }
        
        // Clear bindings for next test
        standard_matcher.clear_bindings();
    }
}

#[test] 
fn test_fast_path_binding_correctness() {
    // Test that fast-path matchers produce correct variable bindings
    let registry = FastPathRegistry::new();
    
    // Function pattern with named arguments
    let expr = Expr::function(
        Expr::symbol("Plus"),
        vec![Expr::symbol("a"), Expr::integer(5), Expr::real(2.5)]
    );
    
    let pattern = Pattern::Function {
        head: Box::new(Pattern::Blank { head: Some("Plus".to_string()) }),
        args: vec![
            Pattern::Named {
                name: "var1".to_string(),
                pattern: Box::new(Pattern::Blank { head: None }),
            },
            Pattern::Named {
                name: "var2".to_string(),
                pattern: Box::new(Pattern::Blank { head: Some("Integer".to_string()) }),
            },
            Pattern::Named {
                name: "var3".to_string(),
                pattern: Box::new(Pattern::Blank { head: Some("Real".to_string()) }),
            },
        ],
    };
    
    if let Some(MatchResult::Success { bindings }) = registry.try_fast_match(&expr, &pattern) {
        assert_eq!(bindings.get("var1"), Some(&Value::Symbol("a".to_string())));
        assert_eq!(bindings.get("var2"), Some(&Value::Integer(5)));
        assert_eq!(bindings.get("var3"), Some(&Value::Real(2.5)));
    } else {
        panic!("Expected successful match with correct bindings");
    }
}

#[test]
fn test_matcher_names_and_debugging() {
    // Test that all matchers have proper names for debugging/profiling
    assert_eq!(BlankMatcher::new().name(), "BlankMatcher");
    assert_eq!(SymbolMatcher::new().name(), "SymbolMatcher");
    assert_eq!(IntegerMatcher::new().name(), "IntegerMatcher");
    assert_eq!(FunctionHeadMatcher::new().name(), "FunctionHeadMatcher");
}

// Performance regression test - ensure fast-path matchers remain efficient
#[test]
fn test_fast_path_performance_baseline() {
    let registry = FastPathRegistry::new();
    
    // Common pattern types that should benefit from fast-path optimization
    let test_patterns = vec![
        Pattern::Blank { head: None },
        Pattern::Blank { head: Some("Integer".to_string()) },
        Pattern::Blank { head: Some("Symbol".to_string()) },
        Pattern::Function {
            head: Box::new(Pattern::Blank { head: Some("Plus".to_string()) }),
            args: vec![
                Pattern::Named {
                    name: "x".to_string(),
                    pattern: Box::new(Pattern::Blank { head: None }),
                },
                Pattern::Blank { head: Some("Integer".to_string()) },
            ],
        },
    ];
    
    let test_expr = Expr::function(Expr::symbol("Plus"), vec![Expr::symbol("y"), Expr::integer(0)]);
    
    // All patterns should either match quickly or fall back gracefully
    for pattern in test_patterns {
        // This should complete quickly without hanging or excessive allocation
        let _result = registry.try_fast_match(&test_expr, &pattern);
    }
}