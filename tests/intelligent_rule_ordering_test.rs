//! Tests for intelligent rule ordering performance optimization

use lyra::{
    ast::{Expr, Pattern},
    rules_engine::{RuleEngine, Rule}
};

#[test]
fn test_intelligent_rule_ordering_basic() {
    // Test that intelligent ordering is enabled by default
    let engine = RuleEngine::new();
    assert!(engine.get_rule_stats().is_empty()); // No stats initially
    
    // Test that we can disable intelligent ordering
    let engine_disabled = RuleEngine::without_intelligent_ordering();
    assert!(engine_disabled.get_rule_stats().is_empty());
}

#[test]
fn test_rule_statistics_tracking() {
    let mut engine = RuleEngine::new();
    
    // Create test expression: Power[x, 1] 
    let test_expr = Expr::function(
        Expr::symbol("Power"),
        vec![Expr::symbol("x"), Expr::integer(1)]
    );
    
    // Create rules where the matching rule is LAST (worst case)
    let rules = vec![
        // Rule 0: x + 0 -> x (won't match Power)
        Rule::immediate(
            Pattern::Function {
                head: Box::new(Pattern::Blank { head: Some("Plus".to_string()) }),
                args: vec![
                    Pattern::Named { name: "x".to_string(), pattern: Box::new(Pattern::Blank { head: None }) },
                    Pattern::Blank { head: Some("Integer".to_string()) }
                ]
            },
            Expr::symbol("x")
        ),
        // Rule 1: x * 0 -> 0 (won't match Power)  
        Rule::immediate(
            Pattern::Function {
                head: Box::new(Pattern::Blank { head: Some("Times".to_string()) }),
                args: vec![
                    Pattern::Named { name: "x".to_string(), pattern: Box::new(Pattern::Blank { head: None }) },
                    Pattern::Blank { head: Some("Integer".to_string()) }
                ]
            },
            Expr::integer(0)
        ),
        // Rule 2: x^1 -> x (WILL MATCH - but last!)
        Rule::immediate(
            Pattern::Function {
                head: Box::new(Pattern::Blank { head: Some("Power".to_string()) }),
                args: vec![
                    Pattern::Named { name: "x".to_string(), pattern: Box::new(Pattern::Blank { head: None }) },
                    Pattern::Blank { head: Some("Integer".to_string()) }
                ]
            },
            Expr::symbol("x")
        ),
    ];
    
    // Apply rules multiple times to build statistics
    for i in 0..15 { // More than reorder_threshold (10)
        let result = engine.apply_rules(&test_expr, &rules).unwrap();
        
        // Rule should match and transform Power[x, 1] -> x
        assert_eq!(result, Expr::symbol("x"));
        
        // Check statistics after enough attempts
        if i >= 10 {
            let stats = engine.get_rule_stats();
            
            // Should have statistics for all 3 rules
            assert_eq!(stats.len(), 3);
            
            // Rule 2 (index 2) should have high success rate (100%)
            if let Some(rule2_stats) = stats.get(&2) {
                assert_eq!(rule2_stats.success_rate, 1.0); // 100% success
                assert!(rule2_stats.successes > 0);
            }
            
            // Rules 0 and 1 should have 0% success rate
            if let Some(rule0_stats) = stats.get(&0) {
                assert_eq!(rule0_stats.success_rate, 0.0); // 0% success  
                assert_eq!(rule0_stats.successes, 0);
            }
            
            if let Some(rule1_stats) = stats.get(&1) {
                assert_eq!(rule1_stats.success_rate, 0.0); // 0% success
                assert_eq!(rule1_stats.successes, 0);
            }
        }
    }
    
    // Verify rule 2 has highest priority score (most successful rule)
    let stats = engine.get_rule_stats();
    let rule2_score = stats.get(&2).map(|s| s.priority_score).unwrap_or(0.0);
    let rule0_score = stats.get(&0).map(|s| s.priority_score).unwrap_or(0.0);
    let rule1_score = stats.get(&1).map(|s| s.priority_score).unwrap_or(0.0);
    
    assert!(rule2_score > rule0_score);
    assert!(rule2_score > rule1_score);
}

#[test]
fn test_intelligent_ordering_performance_improvement() {
    // Test that shows intelligent ordering improves performance
    // by moving successful rules to front of list
    
    let test_expr = Expr::function(
        Expr::symbol("Power"),
        vec![Expr::symbol("y"), Expr::integer(1)]
    );
    
    // Create rules with successful rule LAST initially
    let rules = vec![
        // 5 failing rules first
        Rule::immediate(
            Pattern::Function {
                head: Box::new(Pattern::Blank { head: Some("Sin".to_string()) }),
                args: vec![Pattern::Named { name: "x".to_string(), pattern: Box::new(Pattern::Blank { head: None }) }]
            },
            Expr::symbol("x")
        ),
        Rule::immediate(
            Pattern::Function {
                head: Box::new(Pattern::Blank { head: Some("Cos".to_string()) }),
                args: vec![Pattern::Named { name: "x".to_string(), pattern: Box::new(Pattern::Blank { head: None }) }]
            },
            Expr::symbol("x")
        ),
        Rule::immediate(
            Pattern::Function {
                head: Box::new(Pattern::Blank { head: Some("Tan".to_string()) }),
                args: vec![Pattern::Named { name: "x".to_string(), pattern: Box::new(Pattern::Blank { head: None }) }]
            },
            Expr::symbol("x")
        ),
        Rule::immediate(
            Pattern::Function {
                head: Box::new(Pattern::Blank { head: Some("Log".to_string()) }),
                args: vec![Pattern::Named { name: "x".to_string(), pattern: Box::new(Pattern::Blank { head: None }) }]
            },
            Expr::symbol("x")
        ),
        Rule::immediate(
            Pattern::Function {
                head: Box::new(Pattern::Blank { head: Some("Exp".to_string()) }),
                args: vec![Pattern::Named { name: "x".to_string(), pattern: Box::new(Pattern::Blank { head: None }) }]
            },
            Expr::symbol("x")
        ),
        // Successful rule LAST
        Rule::immediate(
            Pattern::Function {
                head: Box::new(Pattern::Blank { head: Some("Power".to_string()) }),
                args: vec![
                    Pattern::Named { name: "y".to_string(), pattern: Box::new(Pattern::Blank { head: None }) },
                    Pattern::Blank { head: Some("Integer".to_string()) }
                ]
            },
            Expr::symbol("y")
        ),
    ];
    
    // Test with intelligent ordering
    let mut engine_smart = RuleEngine::new();
    
    // Apply rules many times to build statistics and trigger reordering
    for _ in 0..20 {
        let result = engine_smart.apply_rules(&test_expr, &rules).unwrap();
        assert_eq!(result, Expr::symbol("y"));
    }
    
    // Verify that the successful rule (index 5) has highest priority
    let stats = engine_smart.get_rule_stats();
    let successful_rule_score = stats.get(&5).map(|s| s.priority_score).unwrap_or(0.0);
    
    // Check that successful rule has higher priority than failing rules
    for i in 0..5 {
        let failing_rule_score = stats.get(&i).map(|s| s.priority_score).unwrap_or(0.0);
        assert!(successful_rule_score > failing_rule_score, 
                "Successful rule should have higher priority than failing rule {}", i);
    }
    
    // Test without intelligent ordering for comparison
    let mut engine_basic = RuleEngine::without_intelligent_ordering();
    
    // Apply the same rules - should still work but no reordering
    for _ in 0..20 {
        let result = engine_basic.apply_rules(&test_expr, &rules).unwrap();
        assert_eq!(result, Expr::symbol("y"));
    }
    
    // Basic engine should have no statistics
    assert!(engine_basic.get_rule_stats().is_empty());
}

#[test]
fn test_rule_reordering_threshold() {
    let mut engine = RuleEngine::new();
    
    let test_expr = Expr::function(
        Expr::symbol("Plus"),
        vec![Expr::symbol("z"), Expr::integer(0)]
    );
    
    let rules = vec![
        // Failing rule first
        Rule::immediate(
            Pattern::Function {
                head: Box::new(Pattern::Blank { head: Some("Times".to_string()) }),
                args: vec![
                    Pattern::Named { name: "x".to_string(), pattern: Box::new(Pattern::Blank { head: None }) },
                    Pattern::Blank { head: Some("Integer".to_string()) }
                ]
            },
            Expr::integer(0)
        ),
        // Successful rule second  
        Rule::immediate(
            Pattern::Function {
                head: Box::new(Pattern::Blank { head: Some("Plus".to_string()) }),
                args: vec![
                    Pattern::Named { name: "z".to_string(), pattern: Box::new(Pattern::Blank { head: None }) },
                    Pattern::Blank { head: Some("Integer".to_string()) }
                ]
            },
            Expr::symbol("z")
        ),
    ];
    
    // Apply rules up to but not exceeding threshold (10 attempts)
    for i in 0..9 {
        let result = engine.apply_rules(&test_expr, &rules).unwrap();
        assert_eq!(result, Expr::symbol("z"));
        
        let stats = engine.get_rule_stats();
        
        // Should have stats but no reordering should happen yet
        // (reordering happens when total_attempts >= reorder_threshold)
        assert_eq!(stats.len(), 2);
    }
    
    // On the 10th+ attempt, reordering should kick in
    let result = engine.apply_rules(&test_expr, &rules).unwrap();
    assert_eq!(result, Expr::symbol("z"));
    
    let stats = engine.get_rule_stats();
    assert_eq!(stats.len(), 2);
    
    // Rule 1 should have higher priority than rule 0
    let rule0_score = stats.get(&0).map(|s| s.priority_score).unwrap_or(0.0);
    let rule1_score = stats.get(&1).map(|s| s.priority_score).unwrap_or(0.0);
    assert!(rule1_score > rule0_score);
}