#![no_main]

use libfuzzer_sys::fuzz_target;
use lyra::lexer::Lexer;
use arbitrary::{Arbitrary, Unstructured};

// Maximum input size to prevent infinite loops and excessive memory usage
const MAX_INPUT_SIZE: usize = 10_000;

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    source: String,
}

impl FuzzInput {
    fn constrain_size(&mut self) {
        if self.source.len() > MAX_INPUT_SIZE {
            self.source.truncate(MAX_INPUT_SIZE);
        }
    }
}

fuzz_target!(|data: &[u8]| {
    // Use Arbitrary to generate structured input
    let mut unstructured = Unstructured::new(data);
    
    if let Ok(mut input) = FuzzInput::arbitrary(&mut unstructured) {
        input.constrain_size();
        
        // Test lexer with the generated input
        let mut lexer = Lexer::new(&input.source);
        
        // Consume all tokens with resource limits
        let mut token_count = 0;
        const MAX_TOKENS: usize = 10_000;
        
        loop {
            if token_count >= MAX_TOKENS {
                break;
            }
            
            match lexer.next_token() {
                Ok(token) => {
                    token_count += 1;
                    
                    // Check for EOF token to terminate
                    if matches!(token.token_type, lyra::lexer::TokenType::EOF) {
                        break;
                    }
                }
                Err(_) => {
                    // Lexer error is expected for invalid input
                    break;
                }
            }
        }
    }
});

#[cfg(test)]
mod tests {
    use super::*;
    use lyra::lexer::Lexer;
    
    #[test]
    fn test_lexer_fuzzing_basic() {
        // Test with some known problematic patterns
        let test_cases = vec![
            "",
            "\"",
            "\"\"\"",
            "\\",
            "\0",
            "\n\r\t",
            "(*",
            "*)",
            "(*(*(*",
            "\"unclosed string",
            "123.456.789",
            "var_name_with_underscore",
            "sym`bol",
            "[[[[[]]]]",
            "f[x, y, z",
            "a + b * c / d",
            "Rule[x_, y_]",
            "{1, 2, 3, 4, 5}",
        ];
        
        for case in test_cases {
            let mut lexer = Lexer::new(case);
            
            // Just ensure it doesn't crash
            let mut count = 0;
            while count < 100 {
                match lexer.next_token() {
                    Ok(token) => {
                        if matches!(token.token_type, lyra::lexer::TokenType::EOF) {
                            break;
                        }
                    }
                    Err(_) => break,
                }
                count += 1;
            }
        }
    }
    
    #[test]
    fn test_lexer_security_patterns() {
        // Test patterns that might cause security issues
        let security_test_cases = vec![
            // Very long identifiers
            &"a".repeat(10000),
            // Very long strings
            &format!("\"{}\"", "x".repeat(10000)),
            // Deep nesting
            &"[".repeat(1000) + &"]".repeat(1000),
            // Many tokens
            &"a ".repeat(5000),
            // Unicode edge cases
            "\"\\u{0000}\\u{FFFF}\"",
            // Control characters
            "\x01\x02\x03\x04\x05",
            // Mixed patterns
            &format!("f[{}]", "x, ".repeat(1000)),
        ];
        
        for case in security_test_cases {
            let mut lexer = Lexer::new(case);
            
            let mut token_count = 0;
            while token_count < 20000 {
                match lexer.next_token() {
                    Ok(token) => {
                        if matches!(token.token_type, lyra::lexer::TokenType::EOF) {
                            break;
                        }
                    }
                    Err(_) => break,
                }
                token_count += 1;
            }
            
            // Ensure we don't get stuck in infinite loops
            assert!(token_count < 20000, "Lexer might be in infinite loop for input: {}", case.chars().take(50).collect::<String>());
        }
    }
}