use crate::{
    ast::{Expr, Pattern},
    error::{Error, Result},
    lexer::{Lexer, Token, TokenKind},
};

pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, current: 0 }
    }

    pub fn from_source(source: &str) -> Result<Self> {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize()?;
        Ok(Parser::new(tokens))
    }

    pub fn parse(&mut self) -> Result<Vec<Expr>> {
        let mut statements = Vec::new();

        while !self.is_at_end() {
            statements.push(self.statement()?);
            // Skip optional semicolon
            self.match_token(&TokenKind::Semicolon);
        }

        Ok(statements)
    }

    pub fn parse_expression(&mut self) -> Result<Expr> {
        self.expression()
    }

    fn statement(&mut self) -> Result<Expr> {
        self.assignment()
    }

    fn assignment(&mut self) -> Result<Expr> {
        let expr = self.arrow_function()?;

        if self.match_token(&TokenKind::Set) {
            let rhs = self.assignment()?;
            return Ok(Expr::assignment(expr, rhs, false));
        }

        if self.match_token(&TokenKind::SetDelayed) {
            let rhs = self.assignment()?;
            return Ok(Expr::assignment(expr, rhs, true));
        }

        Ok(expr)
    }

    fn arrow_function(&mut self) -> Result<Expr> {
        // Check for parenthesized parameter list at the start
        if self.check(&TokenKind::LeftParen) {
            let saved_pos = self.current;

            // Try to parse as arrow function
            if let Ok(params) = self.try_parse_arrow_params() {
                if self.match_token(&TokenKind::Arrow) {
                    let body = self.arrow_function()?; // Recursive call for nested arrow functions
                    return Ok(Expr::arrow_function(params, body));
                }
            }

            // If not an arrow function, restore position and parse normally
            self.current = saved_pos;
        }

        self.pipeline()
    }

    fn try_parse_arrow_params(&mut self) -> Result<Vec<String>> {
        self.consume(
            &TokenKind::LeftParen,
            "Expected '(' for arrow function parameters",
        )?;

        let mut params = Vec::new();

        if !self.check(&TokenKind::RightParen) {
            loop {
                if let Some(TokenKind::Symbol(param_name)) = self.peek().map(|t| &t.kind) {
                    params.push(param_name.clone());
                    self.advance();
                } else {
                    return Err(Error::Parse {
                        message: "Expected parameter name in arrow function".to_string(),
                        position: self.current_position(),
                    });
                }

                if !self.match_token(&TokenKind::Comma) {
                    break;
                }
            }
        }

        self.consume(
            &TokenKind::RightParen,
            "Expected ')' after arrow function parameters",
        )?;
        Ok(params)
    }

    fn pipeline(&mut self) -> Result<Expr> {
        let mut stages = vec![self.replacement()?];

        while self.match_token(&TokenKind::Pipeline) {
            stages.push(self.replacement()?);
        }

        if stages.len() == 1 {
            Ok(stages.into_iter().next().unwrap())
        } else {
            Ok(Expr::pipeline(stages))
        }
    }

    fn replacement(&mut self) -> Result<Expr> {
        let mut expr = self.rule()?;

        while self.match_token(&TokenKind::ReplaceAll) {
            let rules = self.rule()?;
            expr = Expr::replace(expr, rules);
        }

        while self.match_token(&TokenKind::ReplaceRepeated) {
            let rules = self.rule()?;
            expr = Expr::replace_repeated(expr, rules);
        }

        Ok(expr)
    }

    fn rule(&mut self) -> Result<Expr> {
        let expr = self.range_expr()?;

        if self.match_token(&TokenKind::Rule) {
            let rhs = self.rule()?;
            return Ok(Expr::rule(expr, rhs, false));
        }

        if self.match_token(&TokenKind::RuleDelayed) {
            let rhs = self.rule()?;
            return Ok(Expr::rule(expr, rhs, true));
        }

        Ok(expr)
    }

    fn range_expr(&mut self) -> Result<Expr> {
        let mut expr = self.or()?;

        // Check for range syntax: expr ;; end [;; step]
        if self.check(&TokenKind::Range) {
            self.advance(); // consume ";;"
            let end = self.or()?;

            // Check for optional step
            let step = if self.check(&TokenKind::Range) {
                self.advance(); // consume second ";;"
                Some(self.or()?)
            } else {
                None
            };

            expr = Expr::range(expr, end, step);
        }

        Ok(expr)
    }

    fn or(&mut self) -> Result<Expr> {
        let mut expr = self.and()?;

        while self.match_token(&TokenKind::Or) {
            let right = self.and()?;
            expr = Expr::function(Expr::symbol("Or"), vec![expr, right]);
        }

        Ok(expr)
    }

    fn and(&mut self) -> Result<Expr> {
        let mut expr = self.equality()?;

        while self.match_token(&TokenKind::And) {
            let right = self.equality()?;
            expr = Expr::function(Expr::symbol("And"), vec![expr, right]);
        }

        Ok(expr)
    }

    fn equality(&mut self) -> Result<Expr> {
        let mut expr = self.comparison()?;

        while let Some(op) = self.match_tokens(&[TokenKind::Equal, TokenKind::NotEqual]) {
            let right = self.comparison()?;
            let op_symbol = match op {
                TokenKind::Equal => "Equal",
                TokenKind::NotEqual => "Unequal",
                _ => unreachable!(),
            };
            expr = Expr::function(Expr::symbol(op_symbol), vec![expr, right]);
        }

        Ok(expr)
    }

    fn comparison(&mut self) -> Result<Expr> {
        let mut expr = self.term()?;

        while let Some(op) = self.match_tokens(&[
            TokenKind::Greater,
            TokenKind::GreaterEqual,
            TokenKind::Less,
            TokenKind::LessEqual,
        ]) {
            let right = self.term()?;
            let op_symbol = match op {
                TokenKind::Greater => "Greater",
                TokenKind::GreaterEqual => "GreaterEqual",
                TokenKind::Less => "Less",
                TokenKind::LessEqual => "LessEqual",
                _ => unreachable!(),
            };
            expr = Expr::function(Expr::symbol(op_symbol), vec![expr, right]);
        }

        Ok(expr)
    }

    fn term(&mut self) -> Result<Expr> {
        let mut expr = self.factor()?;

        while let Some(op) = self.match_tokens(&[TokenKind::Minus, TokenKind::Plus]) {
            let right = self.factor()?;
            let op_symbol = match op {
                TokenKind::Minus => "Plus",
                TokenKind::Plus => "Plus",
                _ => unreachable!(),
            };
            if matches!(op, TokenKind::Minus) {
                let neg_right =
                    Expr::function(Expr::symbol("Times"), vec![Expr::integer(-1), right]);
                expr = Expr::function(Expr::symbol(op_symbol), vec![expr, neg_right]);
            } else {
                expr = Expr::function(Expr::symbol(op_symbol), vec![expr, right]);
            }
        }

        Ok(expr)
    }

    fn factor(&mut self) -> Result<Expr> {
        let mut expr = self.power()?;

        while let Some(op) = self.match_tokens(&[TokenKind::Times, TokenKind::Divide, TokenKind::Modulo]) {
            let right = self.power()?;
            let op_symbol = match op {
                TokenKind::Times => "Times",
                TokenKind::Divide => "Divide",
                TokenKind::Modulo => "Modulo",
                _ => unreachable!(),
            };
            expr = Expr::function(Expr::symbol(op_symbol), vec![expr, right]);
        }

        Ok(expr)
    }

    fn power(&mut self) -> Result<Expr> {
        let mut expr = self.unary()?;

        if self.match_token(&TokenKind::Power) {
            let right = self.power()?; // Right associative
            expr = Expr::function(Expr::symbol("Power"), vec![expr, right]);
        }

        Ok(expr)
    }

    fn unary(&mut self) -> Result<Expr> {
        if let Some(op) = self.match_tokens(&[TokenKind::Not, TokenKind::Minus, TokenKind::Plus]) {
            let expr = self.unary()?;
            match op {
                TokenKind::Not => Ok(Expr::function(Expr::symbol("Not"), vec![expr])),
                TokenKind::Minus => Ok(Expr::function(
                    Expr::symbol("Times"),
                    vec![Expr::integer(-1), expr],
                )),
                TokenKind::Plus => Ok(expr), // Unary plus is a no-op
                _ => unreachable!(),
            }
        } else {
            self.postfix()
        }
    }

    fn postfix(&mut self) -> Result<Expr> {
        let mut expr = self.primary()?;

        loop {
            if self.match_token(&TokenKind::LeftBracket) {
                // Check if this is a double bracket [[
                if self.match_token(&TokenKind::LeftBracket) {
                    // This is part access [[...]]
                    let indices = self.argument_list()?;
                    self.consume(&TokenKind::RightBracket, "Expected ']' after part indices")?;
                    self.consume(
                        &TokenKind::RightBracket,
                        "Expected second ']' after part indices",
                    )?;
                    expr = Expr::function(
                        Expr::symbol("Part"),
                        vec![expr].into_iter().chain(indices).collect(),
                    );
                } else {
                    // This is a regular function call [...]
                    let args = self.argument_list()?;
                    self.consume(
                        &TokenKind::RightBracket,
                        "Expected ']' after function arguments",
                    )?;
                    
                    // Check for return type annotation: f[args]: ReturnType
                    if self.check(&TokenKind::Colon) {
                        self.advance(); // consume ':'
                        let return_type = self.or()?; // Parse return type expression
                        expr = Expr::typed_function(expr, args, return_type);
                    } else {
                        expr = Expr::function(expr, args);
                    }
                }
            } else if self.match_token(&TokenKind::Dot) {
                // This is a dot-call obj.method[args]
                if let Some(TokenKind::Symbol(method_name)) = self.peek().map(|t| &t.kind) {
                    let method_name = method_name.clone();
                    self.advance(); // consume the method name

                    // Expect '[' for arguments
                    self.consume(
                        &TokenKind::LeftBracket,
                        "Expected '[' after method name in dot-call",
                    )?;
                    let args = self.argument_list()?;
                    self.consume(
                        &TokenKind::RightBracket,
                        "Expected ']' after dot-call arguments",
                    )?;

                    expr = Expr::dot_call(expr, method_name, args);
                } else {
                    return Err(Error::Parse {
                        message: "Expected method name after '.' in dot-call".to_string(),
                        position: self.current_position(),
                    });
                }
            } else {
                break;
            }
        }

        Ok(expr)
    }

    fn primary(&mut self) -> Result<Expr> {
        if let Some(token) = self.advance() {
            match token.kind.clone() {
                TokenKind::Integer(n) => Ok(Expr::integer(n)),
                TokenKind::Real(n) => Ok(Expr::real(n)),
                TokenKind::String(s) => Ok(Expr::string(s)),
                TokenKind::Symbol(name) => {
                    // Check for modern typed pattern x:_Integer
                    if self.check(&TokenKind::Colon) {
                        self.advance(); // consume ':'
                        let type_pattern = self.or()?; // Parse type expression
                        Ok(Expr::typed_pattern(name, type_pattern))
                    }
                    // Check for traditional pattern suffix x_
                    else if self.check(&TokenKind::Blank)
                        || self.check(&TokenKind::BlankSequence)
                        || self.check(&TokenKind::BlankNullSequence)
                    {
                        let mut pattern = self.parse_pattern_suffix(Some(name))?;

                        // Check for predicate pattern x_?Positive
                        if self.check(&TokenKind::Question) {
                            self.advance(); // consume '?'
                            let test = self.or()?; // Parse predicate expression
                            pattern = Pattern::Predicate {
                                pattern: Box::new(pattern),
                                test: Box::new(test),
                            };
                        }

                        // Check for conditional pattern x_ /; condition
                        if self.check(&TokenKind::Condition) {
                            self.advance(); // consume '/;'
                            let condition = self.or()?; // Parse condition expression
                            pattern = Pattern::Conditional {
                                pattern: Box::new(pattern),
                                condition: Box::new(condition),
                            };
                        }

                        Ok(Expr::Pattern(pattern))
                    } else {
                        Ok(Expr::symbol(name))
                    }
                }
                TokenKind::Blank => {
                    // Check if there's a head type after the blank
                    let head =
                        if let Some(TokenKind::Symbol(head_name)) = self.peek().map(|t| &t.kind) {
                            let head_name = head_name.clone();
                            self.advance();
                            Some(head_name)
                        } else {
                            None
                        };

                    let mut pattern = Pattern::Blank { head };

                    // Check for predicate pattern _?Positive
                    if self.check(&TokenKind::Question) {
                        self.advance(); // consume '?'
                        let test = self.or()?; // Parse predicate expression
                        pattern = Pattern::Predicate {
                            pattern: Box::new(pattern),
                            test: Box::new(test),
                        };
                    }

                    // Check for conditional pattern _ /; condition
                    if self.check(&TokenKind::Condition) {
                        self.advance(); // consume '/;'
                        let condition = self.or()?; // Parse condition expression
                        pattern = Pattern::Conditional {
                            pattern: Box::new(pattern),
                            condition: Box::new(condition),
                        };
                    }

                    Ok(Expr::Pattern(pattern))
                }
                TokenKind::BlankSequence => {
                    Ok(Expr::Pattern(Pattern::BlankSequence { head: None }))
                }
                TokenKind::BlankNullSequence => {
                    Ok(Expr::Pattern(Pattern::BlankNullSequence { head: None }))
                }
                TokenKind::LeftParen => {
                    let expr = self.expression()?;
                    self.consume(&TokenKind::RightParen, "Expected ')' after expression")?;
                    Ok(expr)
                }
                TokenKind::LeftBrace => {
                    let elements = if self.check(&TokenKind::RightBrace) {
                        Vec::new()
                    } else {
                        self.argument_list()?
                    };
                    self.consume(&TokenKind::RightBrace, "Expected '}' after list elements")?;
                    Ok(Expr::list(elements))
                }
                TokenKind::LeftAssoc => {
                    // Parse association <|key->value, key2->value2|>
                    let pairs = if self.check(&TokenKind::Pipeline) {
                        // Empty association <||>
                        Vec::new()
                    } else {
                        self.association_pairs()?
                    };
                    self.consume(
                        &TokenKind::Pipeline,
                        "Expected '|>' after association elements",
                    )?;
                    Ok(Expr::association(pairs))
                }
                _ => Err(Error::Parse {
                    message: format!("Unexpected token: {:?}", token.kind),
                    position: token.position,
                }),
            }
        } else {
            Err(Error::Parse {
                message: "Unexpected end of input".to_string(),
                position: self.tokens.last().map(|t| t.position).unwrap_or(0),
            })
        }
    }

    fn parse_pattern_suffix(&mut self, name: Option<String>) -> Result<Pattern> {
        if self.match_token(&TokenKind::Blank) {
            let head = if let Some(TokenKind::Symbol(head_name)) = self.peek().map(|t| &t.kind) {
                let head_name = head_name.clone();
                self.advance();
                Some(head_name)
            } else {
                None
            };

            if let Some(name) = name {
                Ok(Pattern::Named {
                    name,
                    pattern: Box::new(Pattern::Blank { head }),
                })
            } else {
                Ok(Pattern::Blank { head })
            }
        } else if self.match_token(&TokenKind::BlankSequence) {
            let head = if let Some(TokenKind::Symbol(head_name)) = self.peek().map(|t| &t.kind) {
                let head_name = head_name.clone();
                self.advance();
                Some(head_name)
            } else {
                None
            };

            if let Some(name) = name {
                Ok(Pattern::Named {
                    name,
                    pattern: Box::new(Pattern::BlankSequence { head }),
                })
            } else {
                Ok(Pattern::BlankSequence { head })
            }
        } else if self.match_token(&TokenKind::BlankNullSequence) {
            let head = if let Some(TokenKind::Symbol(head_name)) = self.peek().map(|t| &t.kind) {
                let head_name = head_name.clone();
                self.advance();
                Some(head_name)
            } else {
                None
            };

            if let Some(name) = name {
                Ok(Pattern::Named {
                    name,
                    pattern: Box::new(Pattern::BlankNullSequence { head }),
                })
            } else {
                Ok(Pattern::BlankNullSequence { head })
            }
        } else {
            Err(Error::Parse {
                message: "Expected pattern suffix".to_string(),
                position: self.current_position(),
            })
        }
    }

    fn argument_list(&mut self) -> Result<Vec<Expr>> {
        let mut args = Vec::new();

        if !self.check_any(&[TokenKind::RightBracket, TokenKind::RightBrace]) {
            args.push(self.expression()?);

            while self.match_token(&TokenKind::Comma) {
                args.push(self.expression()?);
            }
        }

        Ok(args)
    }

    fn association_pairs(&mut self) -> Result<Vec<(Expr, Expr)>> {
        let mut pairs = Vec::new();

        // Parse first pair - use or() to avoid parsing rules
        let key = self.or()?;
        self.consume(&TokenKind::Rule, "Expected '->' in association pair")?;
        let value = self.or()?;
        pairs.push((key, value));

        // Parse additional pairs
        while self.match_token(&TokenKind::Comma) {
            let key = self.or()?;
            self.consume(&TokenKind::Rule, "Expected '->' in association pair")?;
            let value = self.or()?;
            pairs.push((key, value));
        }

        Ok(pairs)
    }

    fn expression(&mut self) -> Result<Expr> {
        self.assignment()
    }

    // Helper methods
    fn match_token(&mut self, token_type: &TokenKind) -> bool {
        if self.check(token_type) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn match_tokens(&mut self, types: &[TokenKind]) -> Option<TokenKind> {
        for token_type in types {
            if self.check(token_type) {
                let token = self.advance().unwrap();
                return Some(token.kind.clone());
            }
        }
        None
    }

    fn check(&self, token_type: &TokenKind) -> bool {
        if let Some(token) = self.peek() {
            std::mem::discriminant(&token.kind) == std::mem::discriminant(token_type)
        } else {
            false
        }
    }

    fn check_any(&self, token_types: &[TokenKind]) -> bool {
        token_types.iter().any(|t| self.check(t))
    }

    fn advance(&mut self) -> Option<&Token> {
        if !self.is_at_end() {
            self.current += 1;
        }
        self.previous()
    }

    fn is_at_end(&self) -> bool {
        self.peek()
            .map_or(true, |token| matches!(token.kind, TokenKind::Eof))
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.current)
    }

    fn previous(&self) -> Option<&Token> {
        if self.current > 0 {
            self.tokens.get(self.current - 1)
        } else {
            None
        }
    }

    fn consume(&mut self, token_type: &TokenKind, message: &str) -> Result<&Token> {
        if self.check(token_type) {
            Ok(self.advance().unwrap())
        } else {
            Err(Error::Parse {
                message: message.to_string(),
                position: self.current_position(),
            })
        }
    }

    fn current_position(&self) -> usize {
        self.peek().map(|t| t.position).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Pattern;
    use pretty_assertions::assert_eq;

    fn parse_expression(input: &str) -> Result<Expr> {
        let mut parser = Parser::from_source(input)?;
        parser.parse_expression()
    }

    fn parse_statements(input: &str) -> Result<Vec<Expr>> {
        let mut parser = Parser::from_source(input)?;
        parser.parse()
    }

    #[test]
    fn test_integer_literal() {
        let result = parse_expression("42").unwrap();
        assert_eq!(result, Expr::integer(42));
    }

    #[test]
    fn test_real_literal() {
        let result = parse_expression("3.14").unwrap();
        assert_eq!(result, Expr::real(3.14));
    }

    #[test]
    fn test_string_literal() {
        let result = parse_expression("\"hello\"").unwrap();
        assert_eq!(result, Expr::string("hello"));
    }

    #[test]
    fn test_symbol() {
        let result = parse_expression("x").unwrap();
        assert_eq!(result, Expr::symbol("x"));
    }

    #[test]
    fn test_list() {
        let result = parse_expression("{1, 2, 3}").unwrap();
        assert_eq!(
            result,
            Expr::list(vec![Expr::integer(1), Expr::integer(2), Expr::integer(3),])
        );
    }

    #[test]
    fn test_empty_list() {
        let result = parse_expression("{}").unwrap();
        assert_eq!(result, Expr::list(vec![]));
    }

    #[test]
    fn test_function_call() {
        let result = parse_expression("f[x, y]").unwrap();
        assert_eq!(
            result,
            Expr::function(
                Expr::symbol("f"),
                vec![Expr::symbol("x"), Expr::symbol("y")]
            )
        );
    }

    #[test]
    fn test_function_call_no_args() {
        let result = parse_expression("f[]").unwrap();
        assert_eq!(result, Expr::function(Expr::symbol("f"), vec![]));
    }

    #[test]
    fn test_nested_function_calls() {
        let result = parse_expression("f[g[x]]").unwrap();
        assert_eq!(
            result,
            Expr::function(
                Expr::symbol("f"),
                vec![Expr::function(Expr::symbol("g"), vec![Expr::symbol("x")])]
            )
        );
    }

    #[test]
    fn test_arithmetic_addition() {
        let result = parse_expression("1 + 2").unwrap();
        assert_eq!(
            result,
            Expr::function(
                Expr::symbol("Plus"),
                vec![Expr::integer(1), Expr::integer(2)]
            )
        );
    }

    #[test]
    fn test_arithmetic_subtraction() {
        let result = parse_expression("5 - 3").unwrap();
        assert_eq!(
            result,
            Expr::function(
                Expr::symbol("Plus"),
                vec![
                    Expr::integer(5),
                    Expr::function(
                        Expr::symbol("Times"),
                        vec![Expr::integer(-1), Expr::integer(3)]
                    )
                ]
            )
        );
    }

    #[test]
    fn test_arithmetic_multiplication() {
        let result = parse_expression("2 * 3").unwrap();
        assert_eq!(
            result,
            Expr::function(
                Expr::symbol("Times"),
                vec![Expr::integer(2), Expr::integer(3)]
            )
        );
    }

    #[test]
    fn test_arithmetic_division() {
        let result = parse_expression("8 / 2").unwrap();
        assert_eq!(
            result,
            Expr::function(
                Expr::symbol("Divide"),
                vec![Expr::integer(8), Expr::integer(2)]
            )
        );
    }

    #[test]
    fn test_arithmetic_power() {
        let result = parse_expression("2 ^ 3").unwrap();
        assert_eq!(
            result,
            Expr::function(
                Expr::symbol("Power"),
                vec![Expr::integer(2), Expr::integer(3)]
            )
        );
    }

    #[test]
    fn test_power_right_associative() {
        let result = parse_expression("2 ^ 3 ^ 4").unwrap();
        assert_eq!(
            result,
            Expr::function(
                Expr::symbol("Power"),
                vec![
                    Expr::integer(2),
                    Expr::function(
                        Expr::symbol("Power"),
                        vec![Expr::integer(3), Expr::integer(4)]
                    )
                ]
            )
        );
    }

    #[test]
    fn test_operator_precedence() {
        let result = parse_expression("1 + 2 * 3").unwrap();
        assert_eq!(
            result,
            Expr::function(
                Expr::symbol("Plus"),
                vec![
                    Expr::integer(1),
                    Expr::function(
                        Expr::symbol("Times"),
                        vec![Expr::integer(2), Expr::integer(3)]
                    )
                ]
            )
        );
    }

    #[test]
    fn test_parentheses() {
        let result = parse_expression("(1 + 2) * 3").unwrap();
        assert_eq!(
            result,
            Expr::function(
                Expr::symbol("Times"),
                vec![
                    Expr::function(
                        Expr::symbol("Plus"),
                        vec![Expr::integer(1), Expr::integer(2)]
                    ),
                    Expr::integer(3)
                ]
            )
        );
    }

    #[test]
    fn test_unary_minus() {
        let result = parse_expression("-5").unwrap();
        assert_eq!(
            result,
            Expr::function(
                Expr::symbol("Times"),
                vec![Expr::integer(-1), Expr::integer(5)]
            )
        );
    }

    #[test]
    fn test_unary_plus() {
        let result = parse_expression("+5").unwrap();
        assert_eq!(result, Expr::integer(5));
    }

    #[test]
    fn test_logical_not() {
        let result = parse_expression("!True").unwrap();
        assert_eq!(
            result,
            Expr::function(Expr::symbol("Not"), vec![Expr::symbol("True")])
        );
    }

    #[test]
    fn test_comparison_equal() {
        let result = parse_expression("x == y").unwrap();
        assert_eq!(
            result,
            Expr::function(
                Expr::symbol("Equal"),
                vec![Expr::symbol("x"), Expr::symbol("y")]
            )
        );
    }

    #[test]
    fn test_comparison_less() {
        let result = parse_expression("x < y").unwrap();
        assert_eq!(
            result,
            Expr::function(
                Expr::symbol("Less"),
                vec![Expr::symbol("x"), Expr::symbol("y")]
            )
        );
    }

    #[test]
    fn test_logical_and() {
        let result = parse_expression("x && y").unwrap();
        assert_eq!(
            result,
            Expr::function(
                Expr::symbol("And"),
                vec![Expr::symbol("x"), Expr::symbol("y")]
            )
        );
    }

    #[test]
    fn test_logical_or() {
        let result = parse_expression("x || y").unwrap();
        assert_eq!(
            result,
            Expr::function(
                Expr::symbol("Or"),
                vec![Expr::symbol("x"), Expr::symbol("y")]
            )
        );
    }

    #[test]
    fn test_rule() {
        let result = parse_expression("x -> x^2").unwrap();
        assert_eq!(
            result,
            Expr::rule(
                Expr::symbol("x"),
                Expr::function(
                    Expr::symbol("Power"),
                    vec![Expr::symbol("x"), Expr::integer(2)]
                ),
                false
            )
        );
    }

    #[test]
    fn test_rule_delayed() {
        let result = parse_expression("x :> RandomReal[]").unwrap();
        assert_eq!(
            result,
            Expr::rule(
                Expr::symbol("x"),
                Expr::function(Expr::symbol("RandomReal"), vec![]),
                true
            )
        );
    }

    #[test]
    fn test_replace_all() {
        let result = parse_expression("expr /. rule").unwrap();
        assert_eq!(
            result,
            Expr::replace(Expr::symbol("expr"), Expr::symbol("rule"))
        );
    }

    #[test]
    fn test_assignment() {
        let result = parse_statements("x = 5").unwrap();
        assert_eq!(
            result,
            vec![Expr::assignment(Expr::symbol("x"), Expr::integer(5), false)]
        );
    }

    #[test]
    fn test_assignment_delayed() {
        let result = parse_statements("x := RandomReal[]").unwrap();
        assert_eq!(
            result,
            vec![Expr::assignment(
                Expr::symbol("x"),
                Expr::function(Expr::symbol("RandomReal"), vec![]),
                true
            )]
        );
    }

    #[test]
    fn test_function_definition() {
        let result = parse_statements("f[x_] = x^2").unwrap();
        assert_eq!(
            result,
            vec![Expr::assignment(
                Expr::function(
                    Expr::symbol("f"),
                    vec![Expr::Pattern(Pattern::Named {
                        name: "x".to_string(),
                        pattern: Box::new(Pattern::Blank { head: None })
                    })]
                ),
                Expr::function(
                    Expr::symbol("Power"),
                    vec![Expr::symbol("x"), Expr::integer(2)]
                ),
                false
            )]
        );
    }

    #[test]
    fn test_blank_pattern() {
        let result = parse_expression("_").unwrap();
        assert_eq!(result, Expr::Pattern(Pattern::Blank { head: None }));
    }

    #[test]
    fn test_typed_blank_pattern() {
        let result = parse_expression("_Integer").unwrap();
        assert_eq!(
            result,
            Expr::Pattern(Pattern::Blank {
                head: Some("Integer".to_string())
            })
        );
    }

    #[test]
    fn test_named_pattern() {
        let result = parse_expression("x_").unwrap();
        assert_eq!(
            result,
            Expr::Pattern(Pattern::Named {
                name: "x".to_string(),
                pattern: Box::new(Pattern::Blank { head: None })
            })
        );
    }

    #[test]
    fn test_named_typed_pattern() {
        let result = parse_expression("x_Integer").unwrap();
        assert_eq!(
            result,
            Expr::Pattern(Pattern::Named {
                name: "x".to_string(),
                pattern: Box::new(Pattern::Blank {
                    head: Some("Integer".to_string())
                })
            })
        );
    }

    #[test]
    fn test_blank_sequence() {
        let result = parse_expression("__").unwrap();
        assert_eq!(result, Expr::Pattern(Pattern::BlankSequence { head: None }));
    }

    #[test]
    fn test_blank_null_sequence() {
        let result = parse_expression("___").unwrap();
        assert_eq!(
            result,
            Expr::Pattern(Pattern::BlankNullSequence { head: None })
        );
    }

    #[test]
    fn test_part_access() {
        let result = parse_expression("list[[1]]").unwrap();
        assert_eq!(
            result,
            Expr::function(
                Expr::symbol("Part"),
                vec![Expr::symbol("list"), Expr::integer(1)]
            )
        );
    }

    #[test]
    fn test_part_access_multiple_indices() {
        let result = parse_expression("matrix[[1, 2]]").unwrap();
        assert_eq!(
            result,
            Expr::function(
                Expr::symbol("Part"),
                vec![Expr::symbol("matrix"), Expr::integer(1), Expr::integer(2)]
            )
        );
    }

    #[test]
    fn test_complex_expression() {
        let result = parse_expression("f[g[x, y], {1, 2, 3}]").unwrap();
        assert_eq!(
            result,
            Expr::function(
                Expr::symbol("f"),
                vec![
                    Expr::function(
                        Expr::symbol("g"),
                        vec![Expr::symbol("x"), Expr::symbol("y")]
                    ),
                    Expr::list(vec![Expr::integer(1), Expr::integer(2), Expr::integer(3)])
                ]
            )
        );
    }

    #[test]
    fn test_error_unexpected_token() {
        let result = parse_expression(")");
        assert!(result.is_err());
        match result.unwrap_err() {
            Error::Parse { message, .. } => {
                assert!(message.contains("Unexpected token"));
            }
            _ => panic!("Expected parse error"),
        }
    }

    #[test]
    fn test_error_unclosed_parentheses() {
        let result = parse_expression("(1 + 2");
        assert!(result.is_err());
        match result.unwrap_err() {
            Error::Parse { message, .. } => {
                assert!(message.contains("Expected ')'"));
            }
            _ => panic!("Expected parse error"),
        }
    }

    #[test]
    fn test_error_unclosed_brackets() {
        let result = parse_expression("f[x, y");
        assert!(result.is_err());
        match result.unwrap_err() {
            Error::Parse { message, .. } => {
                assert!(message.contains("Expected ']'"));
            }
            _ => panic!("Expected parse error"),
        }
    }

    #[test]
    fn test_multiple_statements() {
        let result = parse_statements("x = 1; y = 2").unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(
            result[0],
            Expr::assignment(Expr::symbol("x"), Expr::integer(1), false)
        );
        assert_eq!(
            result[1],
            Expr::assignment(Expr::symbol("y"), Expr::integer(2), false)
        );
    }

    // Association tests
    #[test]
    fn test_empty_association() {
        let result = parse_expression("<||>").unwrap();
        assert_eq!(result, Expr::association(vec![]));
    }

    #[test]
    fn test_single_pair_association() {
        let result = parse_expression("<|\"name\" -> \"Ada\"|>").unwrap();
        assert_eq!(
            result,
            Expr::association(vec![(Expr::string("name"), Expr::string("Ada"))])
        );
    }

    #[test]
    fn test_multiple_pairs_association() {
        let result = parse_expression("<|\"name\" -> \"Ada\", \"age\" -> 37|>").unwrap();
        assert_eq!(
            result,
            Expr::association(vec![
                (Expr::string("name"), Expr::string("Ada")),
                (Expr::string("age"), Expr::integer(37))
            ])
        );
    }

    #[test]
    fn test_association_with_symbol_keys() {
        let result = parse_expression("<|x -> 1, y -> 2|>").unwrap();
        assert_eq!(
            result,
            Expr::association(vec![
                (Expr::symbol("x"), Expr::integer(1)),
                (Expr::symbol("y"), Expr::integer(2))
            ])
        );
    }

    #[test]
    fn test_association_with_expression_values() {
        let result = parse_expression("<|\"sum\" -> x + y, \"product\" -> x * y|>").unwrap();
        assert_eq!(
            result,
            Expr::association(vec![
                (
                    Expr::string("sum"),
                    Expr::function(
                        Expr::symbol("Plus"),
                        vec![Expr::symbol("x"), Expr::symbol("y")]
                    )
                ),
                (
                    Expr::string("product"),
                    Expr::function(
                        Expr::symbol("Times"),
                        vec![Expr::symbol("x"), Expr::symbol("y")]
                    )
                )
            ])
        );
    }

    #[test]
    fn test_nested_association() {
        let result = parse_expression("<|\"nested\" -> <|\"inner\" -> 42|>|>").unwrap();
        assert_eq!(
            result,
            Expr::association(vec![(
                Expr::string("nested"),
                Expr::association(vec![(Expr::string("inner"), Expr::integer(42))])
            )])
        );
    }

    // Pipeline tests
    #[test]
    fn test_simple_pipeline() {
        let result = parse_expression("x |> f").unwrap();
        assert_eq!(
            result,
            Expr::pipeline(vec![Expr::symbol("x"), Expr::symbol("f")])
        );
    }

    #[test]
    fn test_multi_stage_pipeline() {
        let result = parse_expression("x |> f |> g |> h").unwrap();
        assert_eq!(
            result,
            Expr::pipeline(vec![
                Expr::symbol("x"),
                Expr::symbol("f"),
                Expr::symbol("g"),
                Expr::symbol("h")
            ])
        );
    }

    #[test]
    fn test_pipeline_with_function_calls() {
        let result = parse_expression("data |> Map[f] |> Select[g]").unwrap();
        assert_eq!(
            result,
            Expr::pipeline(vec![
                Expr::symbol("data"),
                Expr::function(Expr::symbol("Map"), vec![Expr::symbol("f")]),
                Expr::function(Expr::symbol("Select"), vec![Expr::symbol("g")])
            ])
        );
    }

    #[test]
    fn test_pipeline_with_complex_expressions() {
        let result = parse_expression("{1, 2, 3} |> Map[x -> x * 2] |> Total[]").unwrap();
        assert_eq!(
            result,
            Expr::pipeline(vec![
                Expr::list(vec![Expr::integer(1), Expr::integer(2), Expr::integer(3)]),
                Expr::function(
                    Expr::symbol("Map"),
                    vec![Expr::rule(
                        Expr::symbol("x"),
                        Expr::function(
                            Expr::symbol("Times"),
                            vec![Expr::symbol("x"), Expr::integer(2)]
                        ),
                        false
                    )]
                ),
                Expr::function(Expr::symbol("Total"), vec![])
            ])
        );
    }

    #[test]
    fn test_pipeline_precedence_with_assignment() {
        let result = parse_statements("result = x |> f |> g").unwrap();
        assert_eq!(
            result,
            vec![Expr::assignment(
                Expr::symbol("result"),
                Expr::pipeline(vec![
                    Expr::symbol("x"),
                    Expr::symbol("f"),
                    Expr::symbol("g")
                ]),
                false
            )]
        );
    }

    #[test]
    fn test_no_pipeline() {
        // Single expression should not create a pipeline
        let result = parse_expression("x").unwrap();
        assert_eq!(result, Expr::symbol("x"));
    }

    // Dot-call tests
    #[test]
    fn test_simple_dot_call() {
        let result = parse_expression("obj.method[]").unwrap();
        assert_eq!(
            result,
            Expr::dot_call(Expr::symbol("obj"), "method", vec![])
        );
    }

    #[test]
    fn test_dot_call_with_args() {
        let result = parse_expression("obj.method[x, y]").unwrap();
        assert_eq!(
            result,
            Expr::dot_call(
                Expr::symbol("obj"),
                "method",
                vec![Expr::symbol("x"), Expr::symbol("y")]
            )
        );
    }

    #[test]
    fn test_chained_dot_calls() {
        let result = parse_expression("obj.first[].second[x]").unwrap();
        assert_eq!(
            result,
            Expr::dot_call(
                Expr::dot_call(Expr::symbol("obj"), "first", vec![]),
                "second",
                vec![Expr::symbol("x")]
            )
        );
    }

    #[test]
    fn test_dot_call_on_complex_object() {
        let result = parse_expression("{1, 2, 3}.map[f]").unwrap();
        assert_eq!(
            result,
            Expr::dot_call(
                Expr::list(vec![Expr::integer(1), Expr::integer(2), Expr::integer(3)]),
                "map",
                vec![Expr::symbol("f")]
            )
        );
    }

    #[test]
    fn test_dot_call_with_function_call_object() {
        let result = parse_expression("getData[].transform[f]").unwrap();
        assert_eq!(
            result,
            Expr::dot_call(
                Expr::function(Expr::symbol("getData"), vec![]),
                "transform",
                vec![Expr::symbol("f")]
            )
        );
    }

    #[test]
    fn test_dot_call_in_pipeline() {
        let result = parse_expression("data |> obj.process[config]").unwrap();
        assert_eq!(
            result,
            Expr::pipeline(vec![
                Expr::symbol("data"),
                Expr::dot_call(Expr::symbol("obj"), "process", vec![Expr::symbol("config")])
            ])
        );
    }

    #[test]
    fn test_mixed_postfix_operations() {
        let result = parse_expression("obj.method[x][[1]]").unwrap();
        assert_eq!(
            result,
            Expr::function(
                Expr::symbol("Part"),
                vec![
                    Expr::dot_call(Expr::symbol("obj"), "method", vec![Expr::symbol("x")]),
                    Expr::integer(1)
                ]
            )
        );
    }

    // Advanced pattern tests
    #[test]
    fn test_typed_pattern_modern() {
        let result = parse_expression("x:_Integer").unwrap();
        assert_eq!(
            result,
            Expr::typed_pattern(
                "x",
                Expr::Pattern(Pattern::Blank {
                    head: Some("Integer".to_string())
                })
            )
        );
    }

    #[test]
    fn test_typed_pattern_complex() {
        let result = parse_expression("value:_Real").unwrap();
        assert_eq!(
            result,
            Expr::typed_pattern(
                "value",
                Expr::Pattern(Pattern::Blank {
                    head: Some("Real".to_string())
                })
            )
        );
    }

    #[test]
    fn test_predicate_pattern_named() {
        let result = parse_expression("x_?Positive").unwrap();
        assert_eq!(
            result,
            Expr::Pattern(Pattern::Predicate {
                pattern: Box::new(Pattern::Named {
                    name: "x".to_string(),
                    pattern: Box::new(Pattern::Blank { head: None })
                }),
                test: Box::new(Expr::symbol("Positive"))
            })
        );
    }

    #[test]
    fn test_predicate_pattern_blank() {
        let result = parse_expression("_?EvenQ").unwrap();
        assert_eq!(
            result,
            Expr::Pattern(Pattern::Predicate {
                pattern: Box::new(Pattern::Blank { head: None }),
                test: Box::new(Expr::symbol("EvenQ"))
            })
        );
    }

    #[test]
    fn test_conditional_pattern_named() {
        let result = parse_expression("x_ /; x > 0").unwrap();
        assert_eq!(
            result,
            Expr::Pattern(Pattern::Conditional {
                pattern: Box::new(Pattern::Named {
                    name: "x".to_string(),
                    pattern: Box::new(Pattern::Blank { head: None })
                }),
                condition: Box::new(Expr::function(
                    Expr::symbol("Greater"),
                    vec![Expr::symbol("x"), Expr::integer(0)]
                ))
            })
        );
    }

    #[test]
    fn test_conditional_pattern_blank() {
        let result = parse_expression("_ /; # > 5").unwrap();
        assert_eq!(
            result,
            Expr::Pattern(Pattern::Conditional {
                pattern: Box::new(Pattern::Blank { head: None }),
                condition: Box::new(Expr::function(
                    Expr::symbol("Greater"),
                    vec![Expr::symbol("#"), Expr::integer(5)]
                ))
            })
        );
    }

    #[test]
    fn test_combined_pattern_predicate_and_conditional() {
        let result = parse_expression("x_?Positive /; x < 100").unwrap();
        assert_eq!(
            result,
            Expr::Pattern(Pattern::Conditional {
                pattern: Box::new(Pattern::Predicate {
                    pattern: Box::new(Pattern::Named {
                        name: "x".to_string(),
                        pattern: Box::new(Pattern::Blank { head: None })
                    }),
                    test: Box::new(Expr::symbol("Positive"))
                }),
                condition: Box::new(Expr::function(
                    Expr::symbol("Less"),
                    vec![Expr::symbol("x"), Expr::integer(100)]
                ))
            })
        );
    }

    // Range tests
    #[test]
    fn test_simple_range() {
        let result = parse_expression("1;;10").unwrap();
        assert_eq!(
            result,
            Expr::range(Expr::integer(1), Expr::integer(10), None)
        );
    }

    #[test]
    fn test_range_with_step() {
        let result = parse_expression("0;;1;;0.1").unwrap();
        assert_eq!(
            result,
            Expr::range(Expr::integer(0), Expr::integer(1), Some(Expr::real(0.1)))
        );
    }

    #[test]
    fn test_range_with_expressions() {
        let result = parse_expression("x;;y + 1").unwrap();
        assert_eq!(
            result,
            Expr::range(
                Expr::symbol("x"),
                Expr::function(
                    Expr::symbol("Plus"),
                    vec![Expr::symbol("y"), Expr::integer(1)]
                ),
                None
            )
        );
    }

    #[test]
    fn test_range_negative_numbers() {
        let result = parse_expression("-5;;5;;2").unwrap();
        assert_eq!(
            result,
            Expr::range(
                Expr::function(
                    Expr::symbol("Times"),
                    vec![Expr::integer(-1), Expr::integer(5)]
                ),
                Expr::integer(5),
                Some(Expr::integer(2))
            )
        );
    }

    #[test]
    fn test_range_in_list() {
        let result = parse_expression("{1;;5, 10;;15}").unwrap();
        assert_eq!(
            result,
            Expr::list(vec![
                Expr::range(Expr::integer(1), Expr::integer(5), None),
                Expr::range(Expr::integer(10), Expr::integer(15), None)
            ])
        );
    }

    #[test]
    fn test_range_in_function_call() {
        let result = parse_expression("Range[1;;10]").unwrap();
        assert_eq!(
            result,
            Expr::function(
                Expr::symbol("Range"),
                vec![Expr::range(Expr::integer(1), Expr::integer(10), None)]
            )
        );
    }

    // Arrow function tests
    #[test]
    fn test_simple_arrow_function() {
        let result = parse_expression("(x) => x + 1").unwrap();
        assert_eq!(
            result,
            Expr::arrow_function(
                vec!["x".to_string()],
                Expr::function(
                    Expr::symbol("Plus"),
                    vec![Expr::symbol("x"), Expr::integer(1)]
                )
            )
        );
    }

    #[test]
    fn test_arrow_function_no_params() {
        let result = parse_expression("() => 42").unwrap();
        assert_eq!(result, Expr::arrow_function(vec![], Expr::integer(42)));
    }

    #[test]
    fn test_arrow_function_multiple_params() {
        let result = parse_expression("(x, y) => x * y").unwrap();
        assert_eq!(
            result,
            Expr::arrow_function(
                vec!["x".to_string(), "y".to_string()],
                Expr::function(
                    Expr::symbol("Times"),
                    vec![Expr::symbol("x"), Expr::symbol("y")]
                )
            )
        );
    }

    #[test]
    fn test_arrow_function_complex_body() {
        let result = parse_expression("(x) => f[x + 1]").unwrap();
        assert_eq!(
            result,
            Expr::arrow_function(
                vec!["x".to_string()],
                Expr::function(
                    Expr::symbol("f"),
                    vec![Expr::function(
                        Expr::symbol("Plus"),
                        vec![Expr::symbol("x"), Expr::integer(1)]
                    )]
                )
            )
        );
    }

    #[test]
    fn test_arrow_function_in_pipeline() {
        let result = parse_expression("data |> Map[(x) => x^2]").unwrap();
        assert_eq!(
            result,
            Expr::pipeline(vec![
                Expr::symbol("data"),
                Expr::function(
                    Expr::symbol("Map"),
                    vec![Expr::arrow_function(
                        vec!["x".to_string()],
                        Expr::function(
                            Expr::symbol("Power"),
                            vec![Expr::symbol("x"), Expr::integer(2)]
                        )
                    )]
                )
            ])
        );
    }

    #[test]
    fn test_arrow_function_assignment() {
        let result = parse_statements("square = (x) => x^2").unwrap();
        assert_eq!(
            result,
            vec![Expr::assignment(
                Expr::symbol("square"),
                Expr::arrow_function(
                    vec!["x".to_string()],
                    Expr::function(
                        Expr::symbol("Power"),
                        vec![Expr::symbol("x"), Expr::integer(2)]
                    )
                ),
                false
            )]
        );
    }

    #[test]
    fn test_nested_arrow_functions() {
        let result = parse_expression("(x) => (y) => x + y").unwrap();
        assert_eq!(
            result,
            Expr::arrow_function(
                vec!["x".to_string()],
                Expr::arrow_function(
                    vec!["y".to_string()],
                    Expr::function(
                        Expr::symbol("Plus"),
                        vec![Expr::symbol("x"), Expr::symbol("y")]
                    )
                )
            )
        );
    }
}
