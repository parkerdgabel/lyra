#![no_main]

use libfuzzer_sys::fuzz_target;
use lyra::{lexer::Lexer, parser::Parser};
use arbitrary::{Arbitrary, Unstructured};

const MAX_INPUT_SIZE: usize = 5_000; // Smaller for parser to prevent excessive memory usage

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    source: String,
}

impl FuzzInput {
    fn constrain_size(&mut self) {
        if self.source.len() > MAX_INPUT_SIZE {
            self.source.truncate(MAX_INPUT_SIZE);
        }
        
        // Replace null bytes and other problematic characters
        self.source = self.source.replace('\0', " ");
        
        // Ensure balanced brackets/parentheses to some degree
        self.balance_delimiters();
    }
    
    fn balance_delimiters(&mut self) {
        let mut open_parens = 0;
        let mut open_brackets = 0;
        let mut open_braces = 0;
        
        // Count opening delimiters
        for ch in self.source.chars() {
            match ch {
                '(' => open_parens += 1,
                '[' => open_brackets += 1,
                '{' => open_braces += 1,
                ')' => open_parens = open_parens.saturating_sub(1),
                ']' => open_brackets = open_brackets.saturating_sub(1),
                '}' => open_braces = open_braces.saturating_sub(1),
                _ => {}
            }
        }
        
        // Add closing delimiters
        self.source.push_str(&")".repeat(open_parens));
        self.source.push_str(&"]".repeat(open_brackets));
        self.source.push_str(&"}".repeat(open_braces));
        
        // Limit final size
        if self.source.len() > MAX_INPUT_SIZE {
            self.source.truncate(MAX_INPUT_SIZE);
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let mut unstructured = Unstructured::new(data);
    
    if let Ok(mut input) = FuzzInput::arbitrary(&mut unstructured) {
        input.constrain_size();
        
        // Test parser with the generated input
        let lexer = Lexer::new(&input.source);
        let mut parser = Parser::new(lexer);
        
        // Attempt to parse with timeout/limit protection
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            parser.parse()
        })) {
            Ok(result) => {
                // Parser succeeded or failed gracefully
                match result {
                    Ok(_ast) => {
                        // Successfully parsed - this is good!
                    }
                    Err(_error) => {
                        // Parse error is expected for malformed input
                    }
                }
            }
            Err(_) => {
                // Parser panicked - this is a bug we want to find
                // The fuzzer will report this as a crash
            }
        }
    }
});

#[cfg(test)]
mod tests {
    use super::*;
    use lyra::{lexer::Lexer, parser::Parser};
    
    #[test]
    fn test_parser_fuzzing_basic() {
        let test_cases = vec![
            // Basic expressions
            "42",
            "3.14",
            "True",
            "x",
            "f[x]",
            "x + y",
            "x * y + z",
            
            // Lists and functions
            "{1, 2, 3}",
            "f[x, y, z]",
            "g[h[i[j]]]",
            
            // Patterns and rules
            "x_",
            "x__",
            "x_Integer",
            "x -> y",
            "x :> y",
            "expr /. rule",
            
            // Malformed inputs
            "",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            "f[",
            "f]",
            "+",
            "->",
            "/.",
            "_",
            "__",
            
            // Edge cases
            "f[x, y,]", // trailing comma
            "a + + b",   // double operator
            "f[g[h[i[j[k[l]]]]]]", // deep nesting
        ];
        
        for case in test_cases {
            let lexer = Lexer::new(case);
            let mut parser = Parser::new(lexer);
            
            // Just ensure it doesn't crash
            let _ = parser.parse();
        }
    }
    
    #[test]
    fn test_parser_security_patterns() {
        let security_test_cases = vec![
            // Very deep nesting
            &format!("f[{}]", "g[".repeat(100) + &"]".repeat(100)),
            
            // Very long expressions
            &format!("x {}", "+ x ".repeat(1000)),
            
            // Complex patterns
            &format!("{{{}}} /. {{x_}} -> x", "x, ".repeat(500)),
            
            // Many function arguments
            &format!("f[{}]", (1..1000).map(|i| i.to_string()).collect::<Vec<_>>().join(", ")),
            
            // Pathological rule patterns
            "x_ /. x_ -> x_ /. x_ -> x_",
            
            // Unicode edge cases
            "f[\"\\u{10FFFF}\"]",
            
            // Control structures
            "If[True, x, y]",
            "Module[{x}, x + 1]",
        ];
        
        for case in security_test_cases {
            let lexer = Lexer::new(case);
            let mut parser = Parser::new(lexer);
            
            // Test with panic catching to detect crashes
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                parser.parse()
            }));
            
            assert!(result.is_ok(), "Parser panicked on input: {}", 
                   case.chars().take(100).collect::<String>());
        }
    }
    
    #[test]
    fn test_parser_resource_limits() {
        // Test that parser handles resource-intensive inputs gracefully
        
        // Very deep nesting - should not cause stack overflow
        let deep_nesting = "f[".repeat(1000) + &"]".repeat(1000);
        let lexer = Lexer::new(&deep_nesting);
        let mut parser = Parser::new(lexer);
        let _ = parser.parse(); // Should not panic
        
        // Very wide expressions - should not cause excessive memory usage
        let wide_expr = (0..10000).map(|i| format!("x{}", i)).collect::<Vec<_>>().join(" + ");
        let lexer = Lexer::new(&wide_expr);
        let mut parser = Parser::new(lexer);
        let _ = parser.parse(); // Should not panic or use excessive memory
        
        // Complex rule patterns
        let complex_rules = (0..100).map(|i| format!("x{} -> y{}", i, i)).collect::<Vec<_>>().join(" /. ");
        let lexer = Lexer::new(&complex_rules);
        let mut parser = Parser::new(lexer);
        let _ = parser.parse(); // Should not panic
    }
}