use lyra::{
    ast::Expr, 
    parser::Parser,
    lexer::Lexer,
};

/// Helper function to parse statements
fn parse_statements(input: &str) -> Result<Vec<Expr>, Box<dyn std::error::Error>> {
    let mut lexer = Lexer::new(input);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    Ok(parser.parse()?)
}

/// Simple test to check assignment parsing
#[test]
fn test_assignment_parsing_only() {
    // Test immediate assignment parsing
    let statements = parse_statements("x = 42").expect("Failed to parse assignment");
    assert_eq!(statements.len(), 1);
    
    if let Expr::Assignment { lhs, rhs, delayed } = &statements[0] {
        assert!(!delayed); // Should be immediate assignment
    } else {
        panic!("Expected Assignment expression, got {:?}", &statements[0]);
    }
    
    // Test delayed assignment parsing
    let statements = parse_statements("x := 42").expect("Failed to parse delayed assignment");
    assert_eq!(statements.len(), 1);
    
    if let Expr::Assignment { lhs, rhs, delayed } = &statements[0] {
        assert!(*delayed); // Should be delayed assignment
    } else {
        panic!("Expected Assignment expression, got {:?}", &statements[0]);
    }
}