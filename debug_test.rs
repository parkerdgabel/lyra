use lyra::{
    ast::{Expr, Pattern},
    pattern_matcher::{PatternMatcher, MatchResult},
    vm::Value,
};

fn main() {
    println!("🔍 DEBUG: Function Pattern Matching Issue");
    
    let mut matcher = PatternMatcher::new();
    
    let expr = Expr::function(
        Expr::symbol("Plus"),
        vec![Expr::symbol("x"), Expr::integer(0)]
    );
    
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
    
    println!("Expression: {:?}", expr);
    println!("Pattern: {:?}", pattern);
    
    let result = matcher.match_pattern(&expr, &pattern);
    println!("Match result: {:?}", result);
    
    // Also test with standard matcher to see if it's a routing issue
    let mut standard_matcher = PatternMatcher::with_fast_path_disabled();
    let standard_result = standard_matcher.match_pattern(&expr, &pattern);
    println!("Standard matcher result: {:?}", standard_result);
    
    // Test simpler patterns to isolate the issue
    println!("\n🧪 Testing simpler patterns:");
    
    // Test just the integer pattern
    let int_pattern = Pattern::Blank { head: Some("Integer".to_string()) };
    let int_expr = Expr::integer(0);
    let int_result = matcher.match_pattern(&int_expr, &int_pattern);
    println!("Integer pattern match: {:?}", int_result);
    
    // Test just a blank pattern
    let blank_pattern = Pattern::Blank { head: None };
    let blank_result = matcher.match_pattern(&int_expr, &blank_pattern);
    println!("Blank pattern match: {:?}", blank_result);
}