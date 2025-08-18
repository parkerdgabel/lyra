use criterion::{black_box, criterion_group, criterion_main, Criterion, BatchSize};
use lyra::{
    ast::{Expr, Pattern},
    rules_engine::{RuleEngine, Rule}
};

/// Benchmark rule ordering impact on performance
fn rule_ordering_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("rule_ordering");
    
    // Create test expression that will match the LAST rule (worst case scenario)
    let test_expr = Expr::function(
        Expr::symbol("Power"),
        vec![Expr::symbol("x"), Expr::integer(1)]
    );
    
    // Create rules where the matching rule is at different positions
    // Rule 1: x + 0 -> x (won't match)
    let rule1 = Rule::immediate(
        Pattern::Function {
            head: Box::new(Pattern::Blank { head: Some("Plus".to_string()) }),
            args: vec![
                Pattern::Named { name: "x".to_string(), pattern: Box::new(Pattern::Blank { head: None }) },
                Pattern::Blank { head: Some("Integer".to_string()) }
            ]
        },
        Expr::symbol("x")
    );
    
    // Rule 2: x * 0 -> 0 (won't match) 
    let rule2 = Rule::immediate(
        Pattern::Function {
            head: Box::new(Pattern::Blank { head: Some("Times".to_string()) }),
            args: vec![
                Pattern::Named { name: "x".to_string(), pattern: Box::new(Pattern::Blank { head: None }) },
                Pattern::Blank { head: Some("Integer".to_string()) }
            ]
        },
        Expr::integer(0)
    );
    
    // Rule 3: x * 1 -> x (won't match)
    let rule3 = Rule::immediate(
        Pattern::Function {
            head: Box::new(Pattern::Blank { head: Some("Times".to_string()) }),
            args: vec![
                Pattern::Named { name: "x".to_string(), pattern: Box::new(Pattern::Blank { head: None }) },
                Pattern::Blank { head: Some("Integer".to_string()) }
            ]
        },
        Expr::symbol("x")
    );
    
    // Rule 4: x^0 -> 1 (won't match)
    let rule4 = Rule::immediate(
        Pattern::Function {
            head: Box::new(Pattern::Blank { head: Some("Power".to_string()) }),
            args: vec![
                Pattern::Named { name: "x".to_string(), pattern: Box::new(Pattern::Blank { head: None }) },
                Pattern::Blank { head: Some("Integer".to_string()) }
            ]
        },
        Expr::integer(1)
    );
    
    // Rule 5: x^1 -> x (WILL MATCH - but last!)
    let rule5 = Rule::immediate(
        Pattern::Function {
            head: Box::new(Pattern::Blank { head: Some("Power".to_string()) }),
            args: vec![
                Pattern::Named { name: "x".to_string(), pattern: Box::new(Pattern::Blank { head: None }) },
                Pattern::Blank { head: Some("Integer".to_string()) }
            ]
        },
        Expr::symbol("x")
    );
    
    // Test with matching rule in different positions
    group.bench_function("matching_rule_position_1", |b| {
        let rules = vec![rule5.clone()]; // Matching rule first
        b.iter_batched(
            || (RuleEngine::new(), test_expr.clone(), rules.clone()),
            |(mut engine, expr, rules)| {
                engine.apply_rules(black_box(&expr), black_box(&rules)).unwrap()
            },
            BatchSize::SmallInput
        );
    });
    
    group.bench_function("matching_rule_position_2", |b| {
        let rules = vec![rule1.clone(), rule5.clone()]; // Matching rule second
        b.iter_batched(
            || (RuleEngine::new(), test_expr.clone(), rules.clone()),
            |(mut engine, expr, rules)| {
                engine.apply_rules(black_box(&expr), black_box(&rules)).unwrap()
            },
            BatchSize::SmallInput
        );
    });
    
    group.bench_function("matching_rule_position_3", |b| {
        let rules = vec![rule1.clone(), rule2.clone(), rule5.clone()]; // Matching rule third
        b.iter_batched(
            || (RuleEngine::new(), test_expr.clone(), rules.clone()),
            |(mut engine, expr, rules)| {
                engine.apply_rules(black_box(&expr), black_box(&rules)).unwrap()
            },
            BatchSize::SmallInput
        );
    });
    
    group.bench_function("matching_rule_position_5", |b| {
        let rules = vec![rule1.clone(), rule2.clone(), rule3.clone(), rule4.clone(), rule5.clone()]; // Matching rule last (worst case)
        b.iter_batched(
            || (RuleEngine::new(), test_expr.clone(), rules.clone()),
            |(mut engine, expr, rules)| {
                engine.apply_rules(black_box(&expr), black_box(&rules)).unwrap()
            },
            BatchSize::SmallInput
        );
    });
    
    // Test with different success rates - some rules that never match
    group.bench_function("many_failing_rules", |b| {
        let mut rules = Vec::new();
        
        // Add 20 rules that will never match our test expression
        for i in 0..20 {
            rules.push(Rule::immediate(
                Pattern::Function {
                    head: Box::new(Pattern::Blank { head: Some("NonExistentFunction".to_string()) }),
                    args: vec![
                        Pattern::Named { name: format!("var{}", i), pattern: Box::new(Pattern::Blank { head: None }) }
                    ]
                },
                Expr::integer(i)
            ));
        }
        
        // Add the matching rule at the end
        rules.push(rule5.clone());
        
        b.iter_batched(
            || (RuleEngine::new(), test_expr.clone(), rules.clone()),
            |(mut engine, expr, rules)| {
                engine.apply_rules(black_box(&expr), black_box(&rules)).unwrap()
            },
            BatchSize::SmallInput
        );
    });
    
    group.finish();
}

/// Benchmark rule application frequency patterns
fn rule_frequency_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("rule_frequency_patterns");
    
    // Common mathematical simplifications - these should be prioritized
    let common_rules = vec![
        // x + 0 -> x (very common)
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
        // x * 1 -> x (very common)  
        Rule::immediate(
            Pattern::Function {
                head: Box::new(Pattern::Blank { head: Some("Times".to_string()) }),
                args: vec![
                    Pattern::Named { name: "x".to_string(), pattern: Box::new(Pattern::Blank { head: None }) },
                    Pattern::Blank { head: Some("Integer".to_string()) }
                ]
            },
            Expr::symbol("x")
        ),
    ];
    
    // Rare mathematical rules - should be deprioritized
    let rare_rules = vec![
        // Complex trigonometric identities (rarely used)
        Rule::immediate(
            Pattern::Function {
                head: Box::new(Pattern::Blank { head: Some("Sin".to_string()) }),
                args: vec![
                    Pattern::Function {
                        head: Box::new(Pattern::Blank { head: Some("ArcSin".to_string()) }),
                        args: vec![
                            Pattern::Named { name: "x".to_string(), pattern: Box::new(Pattern::Blank { head: None }) }
                        ]
                    }
                ]
            },
            Expr::symbol("x")
        ),
    ];
    
    // Test expressions that match common vs rare rules
    let common_expr = Expr::function(Expr::symbol("Plus"), vec![Expr::symbol("y"), Expr::integer(0)]);
    let rare_expr = Expr::function(
        Expr::symbol("Sin"), 
        vec![Expr::function(Expr::symbol("ArcSin"), vec![Expr::symbol("z")])]
    );
    
    // Benchmark rule ordering: common first vs rare first
    group.bench_function("common_rules_first", |b| {
        let mut rules = common_rules.clone();
        rules.extend(rare_rules.clone());
        
        b.iter_batched(
            || (RuleEngine::new(), common_expr.clone(), rules.clone()),
            |(mut engine, expr, rules)| {
                engine.apply_rules(black_box(&expr), black_box(&rules)).unwrap()
            },
            BatchSize::SmallInput
        );
    });
    
    group.bench_function("rare_rules_first", |b| {
        let mut rules = rare_rules.clone();
        rules.extend(common_rules.clone());
        
        b.iter_batched(
            || (RuleEngine::new(), common_expr.clone(), rules.clone()),
            |(mut engine, expr, rules)| {
                engine.apply_rules(black_box(&expr), black_box(&rules)).unwrap()
            },
            BatchSize::SmallInput
        );
    });
    
    group.finish();
}

criterion_group!(
    rule_ordering_benches,
    rule_ordering_benchmark,
    rule_frequency_benchmark
);
criterion_main!(rule_ordering_benches);