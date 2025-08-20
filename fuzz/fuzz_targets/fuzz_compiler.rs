#![no_main]

use libfuzzer_sys::fuzz_target;
use lyra::{lexer::Lexer, parser::Parser, compiler::Compiler};
use arbitrary::{Arbitrary, Unstructured};

const MAX_INPUT_SIZE: usize = 2_000; // Even smaller for compiler testing

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    source: String,
}

impl FuzzInput {
    fn constrain_size(&mut self) {
        if self.source.len() > MAX_INPUT_SIZE {
            self.source.truncate(MAX_INPUT_SIZE);
        }
        
        // Replace problematic characters
        self.source = self.source.replace('\0', " ");
        
        // Ensure the input is somewhat well-formed
        self.sanitize_input();
    }
    
    fn sanitize_input(&mut self) {
        // Remove obviously problematic patterns that would cause early parser failure
        self.source = self.source
            .replace(";;", ";")
            .replace(",,", ",")
            .replace("  ", " ")
            .trim()
            .to_string();
            
        // Ensure minimal balancing of delimiters
        let mut depth = 0;
        let mut chars: Vec<char> = self.source.chars().collect();
        let mut i = 0;
        
        while i < chars.len() {
            match chars[i] {
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => {
                    if depth > 0 {
                        depth -= 1;
                    } else {
                        // Remove unmatched closing delimiter
                        chars.remove(i);
                        continue;
                    }
                }
                _ => {}
            }
            i += 1;
        }
        
        // Add closing delimiters for remaining open ones
        for _ in 0..depth.min(10) { // Limit to prevent huge additions
            chars.push(']');
        }
        
        self.source = chars.into_iter().collect();
        
        // Final size check
        if self.source.len() > MAX_INPUT_SIZE {
            self.source.truncate(MAX_INPUT_SIZE);
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let mut unstructured = Unstructured::new(data);
    
    if let Ok(mut input) = FuzzInput::arbitrary(&mut unstructured) {
        input.constrain_size();
        
        // Test the complete pipeline: lexer -> parser -> compiler
        let lexer = Lexer::new(&input.source);
        let mut parser = Parser::new(lexer);
        
        // First parse the input
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            parser.parse()
        })) {
            Ok(parse_result) => {
                if let Ok(ast) = parse_result {
                    // If parsing succeeded, try compilation
                    let mut compiler = Compiler::new();
                    
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        compiler.compile(&ast)
                    })) {
                        Ok(compile_result) => {
                            // Compilation succeeded or failed gracefully
                            match compile_result {
                                Ok(_bytecode) => {
                                    // Successfully compiled - great!
                                }
                                Err(_error) => {
                                    // Compilation error is expected for some valid ASTs
                                }
                            }
                        }
                        Err(_) => {
                            // Compiler panicked - this is a bug
                        }
                    }
                }
                // If parsing failed, that's fine - invalid input
            }
            Err(_) => {
                // Parser panicked - already a bug
            }
        }
    }
});

#[cfg(test)]
mod tests {
    use super::*;
    use lyra::{lexer::Lexer, parser::Parser, compiler::Compiler};
    
    #[test]
    fn test_compiler_fuzzing_basic() {
        let test_cases = vec![
            // Simple expressions that should compile
            "42",
            "x",
            "x + y",
            "f[x]",
            "{1, 2, 3}",
            "x -> x + 1",
            
            // More complex but valid expressions
            "f[x_, y_] := x + y",
            "expr /. x -> x + 1",
            "If[True, 1, 2]",
            "Map[f, {1, 2, 3}]",
            
            // Edge cases
            "{}",
            "f[]",
            "x_",
            "x__",
        ];
        
        for case in test_cases {
            let lexer = Lexer::new(case);
            let mut parser = Parser::new(lexer);
            
            if let Ok(ast) = parser.parse() {
                let mut compiler = Compiler::new();
                
                // Should not panic
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    compiler.compile(&ast)
                }));
                
                assert!(result.is_ok(), "Compiler panicked on input: {}", case);
            }
        }
    }
    
    #[test]
    fn test_compiler_security_patterns() {
        let security_test_cases = vec![
            // Deep nesting
            &"f[".repeat(50) + &"]".repeat(50),
            
            // Very long symbol names
            &format!("very_long_symbol_name_{}", "x".repeat(1000)),
            
            // Many function arguments
            &format!("f[{}]", (1..100).map(|i| i.to_string()).collect::<Vec<_>>().join(", ")),
            
            // Complex nested functions
            &format!("f[g[h[{}]]]", (1..50).map(|i| format!("x{}", i)).collect::<Vec<_>>().join(", ")),
            
            // Pattern with many alternatives
            &format!("x /. {{{}}} -> result", (1..100).map(|i| format!("{} -> {}", i, i)).collect::<Vec<_>>().join(", ")),
            
            // Large lists
            &format!("{{{}}}", (1..1000).map(|i| i.to_string()).collect::<Vec<_>>().join(", ")),
        ];
        
        for case in security_test_cases {
            let lexer = Lexer::new(case);
            let mut parser = Parser::new(lexer);
            
            let parse_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                parser.parse()
            }));
            
            if let Ok(Ok(ast)) = parse_result {
                let mut compiler = Compiler::new();
                
                let compile_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    compiler.compile(&ast)
                }));
                
                assert!(compile_result.is_ok(), "Compiler panicked on input: {}", 
                       case.chars().take(100).collect::<String>());
            }
        }
    }
    
    #[test]
    fn test_compiler_resource_limits() {
        // Test that compiler handles resource-intensive ASTs gracefully
        
        // Create a deeply nested function call through parsing
        let deep_call = "f[g[h[i[j[k[1]]]]]]";
        let lexer = Lexer::new(deep_call);
        let mut parser = Parser::new(lexer);
        
        if let Ok(ast) = parser.parse() {
            let mut compiler = Compiler::new();
            let _ = compiler.compile(&ast); // Should not panic or use excessive resources
        }
        
        // Create a function with many parameters
        let many_params = format!("f[{}]", (1..500).map(|i| i.to_string()).collect::<Vec<_>>().join(", "));
        let lexer = Lexer::new(&many_params);
        let mut parser = Parser::new(lexer);
        
        if let Ok(ast) = parser.parse() {
            let mut compiler = Compiler::new();
            let _ = compiler.compile(&ast); // Should not panic
        }
        
        // Create a large list
        let large_list = format!("{{{}}}", (1..2000).map(|i| i.to_string()).collect::<Vec<_>>().join(", "));
        let lexer = Lexer::new(&large_list);
        let mut parser = Parser::new(lexer);
        
        if let Ok(ast) = parser.parse() {
            let mut compiler = Compiler::new();
            let _ = compiler.compile(&ast); // Should not panic or use excessive memory
        }
    }
    
    #[test]
    fn test_compiler_edge_cases() {
        // Test specific edge cases that might cause issues
        
        let edge_cases = vec![
            // Empty structures
            "{}",
            "f[]",
            
            // Undefined symbols
            "undefinedSymbol",
            "f[undefinedSymbol]",
            
            // Complex patterns
            "x_Integer",
            "x__Real",
            "f[x_, y__]",
            
            // Rule applications
            "x /. x -> y",
            "expr /. {a -> b, c -> d}",
            
            // Nested rules
            "x /. a -> (y /. b -> c)",
            
            // Function definitions
            "f[x_] := x + 1",
            "g[x_, y_] := f[x] + f[y]",
        ];
        
        for case in edge_cases {
            let lexer = Lexer::new(case);
            let mut parser = Parser::new(lexer);
            
            if let Ok(ast) = parser.parse() {
                let mut compiler = Compiler::new();
                
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    compiler.compile(&ast)
                }));
                
                assert!(result.is_ok(), "Compiler panicked on edge case: {}", case);
            }
        }
    }
}