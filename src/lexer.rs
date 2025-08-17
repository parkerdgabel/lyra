use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum InterpolationPart {
    Text(String),
    Expression(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub position: usize,
    pub length: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Literals
    Integer(i64),
    Real(f64),
    Rational(i64, i64), // numerator, denominator
    Complex(f64, f64),  // real, imaginary
    BigInt(String),     // 123n
    BigDecimal(String), // 1.23d100
    HexInteger(String), // 16^^FF
    String(String),
    InterpolatedString(Vec<InterpolationPart>),
    Symbol(String),
    ContextSymbol(String), // std`net`http`Get

    // Operators
    Plus,
    Minus,
    Times,
    Divide,
    Power,

    // Comparison
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,

    // Logical
    And,
    Or,
    Not,

    // Assignment and Rules
    Set,         // =
    SetDelayed,  // :=
    Rule,        // ->
    RuleDelayed, // :>
    ReplaceAll,  // /.

    // Modern operators
    Pipeline,    // |>
    Postfix,     // //
    Prefix,      // @
    Arrow,       // =>
    Range,       // ;;
    Condition,   // /;
    Alternative, // |

    // Grouping
    LeftParen,    // (
    RightParen,   // )
    LeftBracket,  // [
    RightBracket, // ]
    LeftBrace,    // {
    RightBrace,   // }

    // Associations
    LeftAssoc,  // <|
    RightAssoc, // |>

    // Part access
    LeftDoubleBracket,  // [[
    RightDoubleBracket, // ]]

    // Separators
    Comma,
    Semicolon,
    Dot,      // .
    Colon,    // :
    Question, // ?

    // Patterns
    Blank,             // _
    BlankSequence,     // __
    BlankNullSequence, // ___

    // Special
    StringJoin, // <>
    Backtick,   // ` (for contexts)

    // Whitespace and comments (usually ignored)
    Whitespace,
    Comment(String),

    // End of input
    Eof,
}

pub struct Lexer<'a> {
    input: &'a str,
    position: usize,
    current_char: Option<char>,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        let mut lexer = Lexer {
            input,
            position: 0,
            current_char: None,
        };
        lexer.current_char = lexer.input.chars().next();
        lexer
    }

    pub fn tokenize(&mut self) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();

        loop {
            let token = self.next_token()?;
            let is_eof = matches!(token.kind, TokenKind::Eof);

            if !matches!(token.kind, TokenKind::Whitespace | TokenKind::Comment(_)) {
                tokens.push(token);
            }

            if is_eof {
                break;
            }
        }

        Ok(tokens)
    }

    fn next_token(&mut self) -> Result<Token> {
        self.skip_whitespace();

        let start_pos = self.position;

        match self.current_char {
            None => Ok(Token {
                kind: TokenKind::Eof,
                position: start_pos,
                length: 0,
            }),
            Some(ch) => {
                match ch {
                    // Single character tokens
                    '+' => {
                        self.advance();
                        Ok(Token {
                            kind: TokenKind::Plus,
                            position: start_pos,
                            length: 1,
                        })
                    }
                    '-' => {
                        self.advance();
                        if self.current_char == Some('>') {
                            self.advance();
                            Ok(Token {
                                kind: TokenKind::Rule,
                                position: start_pos,
                                length: 2,
                            })
                        } else {
                            Ok(Token {
                                kind: TokenKind::Minus,
                                position: start_pos,
                                length: 1,
                            })
                        }
                    }
                    '*' => {
                        self.advance();
                        Ok(Token {
                            kind: TokenKind::Times,
                            position: start_pos,
                            length: 1,
                        })
                    }
                    '/' => {
                        self.advance();
                        if self.current_char == Some('.') {
                            self.advance();
                            Ok(Token {
                                kind: TokenKind::ReplaceAll,
                                position: start_pos,
                                length: 2,
                            })
                        } else if self.current_char == Some('/') {
                            self.advance();
                            Ok(Token {
                                kind: TokenKind::Postfix,
                                position: start_pos,
                                length: 2,
                            })
                        } else if self.current_char == Some(';') {
                            self.advance();
                            Ok(Token {
                                kind: TokenKind::Condition,
                                position: start_pos,
                                length: 2,
                            })
                        } else {
                            Ok(Token {
                                kind: TokenKind::Divide,
                                position: start_pos,
                                length: 1,
                            })
                        }
                    }
                    '^' => {
                        self.advance();
                        Ok(Token {
                            kind: TokenKind::Power,
                            position: start_pos,
                            length: 1,
                        })
                    }
                    '(' => {
                        self.advance();
                        Ok(Token {
                            kind: TokenKind::LeftParen,
                            position: start_pos,
                            length: 1,
                        })
                    }
                    ')' => {
                        self.advance();
                        Ok(Token {
                            kind: TokenKind::RightParen,
                            position: start_pos,
                            length: 1,
                        })
                    }
                    '[' => {
                        self.advance();
                        Ok(Token {
                            kind: TokenKind::LeftBracket,
                            position: start_pos,
                            length: 1,
                        })
                    }
                    ']' => {
                        self.advance();
                        Ok(Token {
                            kind: TokenKind::RightBracket,
                            position: start_pos,
                            length: 1,
                        })
                    }
                    '{' => {
                        self.advance();
                        Ok(Token {
                            kind: TokenKind::LeftBrace,
                            position: start_pos,
                            length: 1,
                        })
                    }
                    '}' => {
                        self.advance();
                        Ok(Token {
                            kind: TokenKind::RightBrace,
                            position: start_pos,
                            length: 1,
                        })
                    }
                    ',' => {
                        self.advance();
                        Ok(Token {
                            kind: TokenKind::Comma,
                            position: start_pos,
                            length: 1,
                        })
                    }
                    ';' => {
                        self.advance();
                        if self.current_char == Some(';') {
                            self.advance();
                            Ok(Token {
                                kind: TokenKind::Range,
                                position: start_pos,
                                length: 2,
                            })
                        } else {
                            Ok(Token {
                                kind: TokenKind::Semicolon,
                                position: start_pos,
                                length: 1,
                            })
                        }
                    }
                    '.' => {
                        self.advance();
                        Ok(Token {
                            kind: TokenKind::Dot,
                            position: start_pos,
                            length: 1,
                        })
                    }
                    '?' => {
                        self.advance();
                        Ok(Token {
                            kind: TokenKind::Question,
                            position: start_pos,
                            length: 1,
                        })
                    }
                    '`' => {
                        self.advance();
                        Ok(Token {
                            kind: TokenKind::Backtick,
                            position: start_pos,
                            length: 1,
                        })
                    }
                    '=' => {
                        self.advance();
                        if self.current_char == Some('=') {
                            self.advance();
                            Ok(Token {
                                kind: TokenKind::Equal,
                                position: start_pos,
                                length: 2,
                            })
                        } else if self.current_char == Some('>') {
                            self.advance();
                            Ok(Token {
                                kind: TokenKind::Arrow,
                                position: start_pos,
                                length: 2,
                            })
                        } else {
                            Ok(Token {
                                kind: TokenKind::Set,
                                position: start_pos,
                                length: 1,
                            })
                        }
                    }
                    '@' => {
                        self.advance();
                        Ok(Token {
                            kind: TokenKind::Prefix,
                            position: start_pos,
                            length: 1,
                        })
                    }
                    '!' => {
                        self.advance();
                        if self.current_char == Some('=') {
                            self.advance();
                            Ok(Token {
                                kind: TokenKind::NotEqual,
                                position: start_pos,
                                length: 2,
                            })
                        } else {
                            Ok(Token {
                                kind: TokenKind::Not,
                                position: start_pos,
                                length: 1,
                            })
                        }
                    }
                    '<' => {
                        self.advance();
                        if self.current_char == Some('=') {
                            self.advance();
                            Ok(Token {
                                kind: TokenKind::LessEqual,
                                position: start_pos,
                                length: 2,
                            })
                        } else if self.current_char == Some('>') {
                            self.advance();
                            Ok(Token {
                                kind: TokenKind::StringJoin,
                                position: start_pos,
                                length: 2,
                            })
                        } else if self.current_char == Some('|') {
                            self.advance();
                            Ok(Token {
                                kind: TokenKind::LeftAssoc,
                                position: start_pos,
                                length: 2,
                            })
                        } else {
                            Ok(Token {
                                kind: TokenKind::Less,
                                position: start_pos,
                                length: 1,
                            })
                        }
                    }
                    '>' => {
                        self.advance();
                        if self.current_char == Some('=') {
                            self.advance();
                            Ok(Token {
                                kind: TokenKind::GreaterEqual,
                                position: start_pos,
                                length: 2,
                            })
                        } else {
                            Ok(Token {
                                kind: TokenKind::Greater,
                                position: start_pos,
                                length: 1,
                            })
                        }
                    }
                    '&' => {
                        self.advance();
                        if self.current_char == Some('&') {
                            self.advance();
                            Ok(Token {
                                kind: TokenKind::And,
                                position: start_pos,
                                length: 2,
                            })
                        } else {
                            Err(Error::Lexer {
                                message: "Unexpected character '&'".to_string(),
                                position: start_pos,
                            })
                        }
                    }
                    '|' => {
                        self.advance();
                        if self.current_char == Some('|') {
                            self.advance();
                            Ok(Token {
                                kind: TokenKind::Or,
                                position: start_pos,
                                length: 2,
                            })
                        } else if self.current_char == Some('>') {
                            self.advance();
                            Ok(Token {
                                kind: TokenKind::Pipeline,
                                position: start_pos,
                                length: 2,
                            })
                        } else {
                            Ok(Token {
                                kind: TokenKind::Alternative,
                                position: start_pos,
                                length: 1,
                            })
                        }
                    }
                    ':' => {
                        self.advance();
                        if self.current_char == Some('=') {
                            self.advance();
                            Ok(Token {
                                kind: TokenKind::SetDelayed,
                                position: start_pos,
                                length: 2,
                            })
                        } else if self.current_char == Some('>') {
                            self.advance();
                            Ok(Token {
                                kind: TokenKind::RuleDelayed,
                                position: start_pos,
                                length: 2,
                            })
                        } else {
                            Ok(Token {
                                kind: TokenKind::Colon,
                                position: start_pos,
                                length: 1,
                            })
                        }
                    }
                    '_' => {
                        self.advance();
                        if self.current_char == Some('_') {
                            self.advance();
                            if self.current_char == Some('_') {
                                self.advance();
                                Ok(Token {
                                    kind: TokenKind::BlankNullSequence,
                                    position: start_pos,
                                    length: 3,
                                })
                            } else {
                                Ok(Token {
                                    kind: TokenKind::BlankSequence,
                                    position: start_pos,
                                    length: 2,
                                })
                            }
                        } else {
                            Ok(Token {
                                kind: TokenKind::Blank,
                                position: start_pos,
                                length: 1,
                            })
                        }
                    }
                    '"' => self.read_string(start_pos),
                    c if c.is_ascii_digit() => {
                        // Check for hex notation (e.g., 16^^FF)
                        if self.peek_hex_notation() {
                            self.read_hex_number(start_pos)
                        } else {
                            self.read_number(start_pos)
                        }
                    }
                    c if c.is_alphabetic() || c == '$' || c == '#' => self.read_symbol(start_pos),
                    _ => Err(Error::Lexer {
                        message: format!("Unexpected character '{}'", ch),
                        position: start_pos,
                    }),
                }
            }
        }
    }

    fn advance(&mut self) {
        if self.current_char.is_some() {
            self.position += self.current_char.unwrap().len_utf8();
            self.current_char = self.input.chars().nth(self.char_position());
        }
    }

    fn char_position(&self) -> usize {
        self.input[..self.position].chars().count()
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.current_char {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn read_string(&mut self, start_pos: usize) -> Result<Token> {
        self.advance(); // Skip opening quote
        let mut string_value = String::new();

        while let Some(ch) = self.current_char {
            if ch == '"' {
                self.advance(); // Skip closing quote
                return Ok(Token {
                    kind: TokenKind::String(string_value),
                    position: start_pos,
                    length: self.position - start_pos,
                });
            } else if ch == '\\' {
                self.advance();
                match self.current_char {
                    Some('"') => {
                        string_value.push('"');
                        self.advance();
                    }
                    Some('\\') => {
                        string_value.push('\\');
                        self.advance();
                    }
                    Some('n') => {
                        string_value.push('\n');
                        self.advance();
                    }
                    Some('t') => {
                        string_value.push('\t');
                        self.advance();
                    }
                    Some('r') => {
                        string_value.push('\r');
                        self.advance();
                    }
                    Some(c) => {
                        string_value.push(c);
                        self.advance();
                    }
                    None => {
                        return Err(Error::Lexer {
                            message: "Unterminated string escape".to_string(),
                            position: self.position,
                        })
                    }
                }
            } else {
                string_value.push(ch);
                self.advance();
            }
        }

        Err(Error::Lexer {
            message: "Unterminated string".to_string(),
            position: start_pos,
        })
    }

    fn peek_hex_notation(&self) -> bool {
        // Look for pattern like 16^^FF
        let chars: Vec<char> = self.input[self.position..].chars().collect();
        if chars.len() < 4 {
            return false;
        }

        // Find the ^^
        for i in 1..chars.len() - 1 {
            if chars[i] == '^' && chars.get(i + 1) == Some(&'^') {
                // Check if everything before ^^ is digits (base)
                if chars[..i].iter().all(|c| c.is_ascii_digit()) {
                    return true;
                }
            }
        }
        false
    }

    fn read_hex_number(&mut self, start_pos: usize) -> Result<Token> {
        let mut number_str = String::new();

        // Read the entire hex notation
        while let Some(ch) = self.current_char {
            if ch.is_alphanumeric() || ch == '^' {
                number_str.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        Ok(Token {
            kind: TokenKind::HexInteger(number_str),
            position: start_pos,
            length: self.position - start_pos,
        })
    }

    fn read_number(&mut self, start_pos: usize) -> Result<Token> {
        let mut number_str = String::new();
        let mut is_real = false;

        // Read integer part
        while let Some(ch) = self.current_char {
            if ch.is_ascii_digit() {
                number_str.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        // Check for BigInt suffix 'n'
        if self.current_char == Some('n') {
            self.advance();
            return Ok(Token {
                kind: TokenKind::BigInt(number_str),
                position: start_pos,
                length: self.position - start_pos,
            });
        }

        // Check for decimal point
        if self.current_char == Some('.') {
            let next_char = self.input.chars().nth(self.char_position() + 1);
            if next_char.map_or(false, |c| c.is_ascii_digit()) {
                is_real = true;
                number_str.push('.');
                self.advance();

                // Read fractional part
                while let Some(ch) = self.current_char {
                    if ch.is_ascii_digit() {
                        number_str.push(ch);
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
        }

        // Check for exponent
        if matches!(self.current_char, Some('e') | Some('E')) {
            is_real = true;
            number_str.push(self.current_char.unwrap());
            self.advance();

            if matches!(self.current_char, Some('+') | Some('-')) {
                number_str.push(self.current_char.unwrap());
                self.advance();
            }

            while let Some(ch) = self.current_char {
                if ch.is_ascii_digit() {
                    number_str.push(ch);
                    self.advance();
                } else {
                    break;
                }
            }
        }

        // Check for BigDecimal suffix (e.g., 1.23d100)
        if self.current_char == Some('d') {
            self.advance();
            let mut precision = String::new();
            while let Some(ch) = self.current_char {
                if ch.is_ascii_digit() {
                    precision.push(ch);
                    self.advance();
                } else {
                    break;
                }
            }
            return Ok(Token {
                kind: TokenKind::BigDecimal(format!("{}d{}", number_str, precision)),
                position: start_pos,
                length: self.position - start_pos,
            });
        }

        // Check for rational division (e.g., 3/4 when both are integers)
        if self.current_char == Some('/') && !is_real {
            // Look ahead to see if there's another integer
            let saved_pos = self.position;
            let saved_char = self.current_char;
            self.advance(); // skip '/'

            if self.current_char.map_or(false, |c| c.is_ascii_digit()) {
                let mut denom_str = String::new();
                while let Some(ch) = self.current_char {
                    if ch.is_ascii_digit() {
                        denom_str.push(ch);
                        self.advance();
                    } else {
                        break;
                    }
                }

                // Parse both parts as integers for rational
                if let (Ok(num), Ok(denom)) = (number_str.parse::<i64>(), denom_str.parse::<i64>())
                {
                    return Ok(Token {
                        kind: TokenKind::Rational(num, denom),
                        position: start_pos,
                        length: self.position - start_pos,
                    });
                }
            }

            // Restore position if not a rational
            self.position = saved_pos;
            self.current_char = saved_char;
        }

        let token_kind = if is_real {
            match number_str.parse::<f64>() {
                Ok(value) => TokenKind::Real(value),
                Err(_) => {
                    return Err(Error::Lexer {
                        message: format!("Invalid real number: {}", number_str),
                        position: start_pos,
                    })
                }
            }
        } else {
            match number_str.parse::<i64>() {
                Ok(value) => TokenKind::Integer(value),
                Err(_) => {
                    return Err(Error::Lexer {
                        message: format!("Invalid integer: {}", number_str),
                        position: start_pos,
                    })
                }
            }
        };

        Ok(Token {
            kind: token_kind,
            position: start_pos,
            length: self.position - start_pos,
        })
    }

    fn read_symbol(&mut self, start_pos: usize) -> Result<Token> {
        let mut symbol_name = String::new();
        let mut is_context = false;

        while let Some(ch) = self.current_char {
            if ch.is_alphanumeric() || ch == '$' || ch == '#' {
                symbol_name.push(ch);
                self.advance();
            } else if ch == '_' {
                // Only include underscore if it's in the middle of a word AND
                // the next character is not uppercase (to handle x_Integer pattern)
                let next_char = self.input.chars().nth(self.char_position() + 1);
                if next_char.map_or(false, |c| {
                    (c.is_alphanumeric() || c == '$' || c == '#') && !c.is_uppercase()
                }) {
                    symbol_name.push(ch);
                    self.advance();
                } else {
                    break; // Stop here, let _ be handled as a separate token
                }
            } else if ch == '`' {
                symbol_name.push(ch);
                is_context = true;
                self.advance();
            } else {
                break;
            }
        }

        let token_kind = if is_context {
            TokenKind::ContextSymbol(symbol_name)
        } else {
            TokenKind::Symbol(symbol_name)
        };

        Ok(Token {
            kind: token_kind,
            position: start_pos,
            length: self.position - start_pos,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn tokenize_string(input: &str) -> Result<Vec<TokenKind>> {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        Ok(tokens.into_iter().map(|t| t.kind).collect())
    }

    #[test]
    fn test_integer_literals() {
        let result = tokenize_string("42").unwrap();
        assert_eq!(result, vec![TokenKind::Integer(42), TokenKind::Eof]);
    }

    #[test]
    fn test_real_literals() {
        let result = tokenize_string("3.14").unwrap();
        assert_eq!(result, vec![TokenKind::Real(3.14), TokenKind::Eof]);

        let result = tokenize_string("1.23e-4").unwrap();
        assert_eq!(result, vec![TokenKind::Real(1.23e-4), TokenKind::Eof]);
    }

    #[test]
    fn test_string_literals() {
        let result = tokenize_string("\"hello\"").unwrap();
        assert_eq!(
            result,
            vec![TokenKind::String("hello".to_string()), TokenKind::Eof]
        );

        let result = tokenize_string("\"hello\\nworld\"").unwrap();
        assert_eq!(
            result,
            vec![
                TokenKind::String("hello\nworld".to_string()),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_symbols() {
        let result = tokenize_string("x").unwrap();
        assert_eq!(
            result,
            vec![TokenKind::Symbol("x".to_string()), TokenKind::Eof]
        );

        let result = tokenize_string("Symbol123").unwrap();
        assert_eq!(
            result,
            vec![TokenKind::Symbol("Symbol123".to_string()), TokenKind::Eof]
        );
    }

    #[test]
    fn test_arithmetic_operators() {
        let result = tokenize_string("+ - * / ^").unwrap();
        assert_eq!(
            result,
            vec![
                TokenKind::Plus,
                TokenKind::Minus,
                TokenKind::Times,
                TokenKind::Divide,
                TokenKind::Power,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_comparison_operators() {
        let result = tokenize_string("== != < <= > >=").unwrap();
        assert_eq!(
            result,
            vec![
                TokenKind::Equal,
                TokenKind::NotEqual,
                TokenKind::Less,
                TokenKind::LessEqual,
                TokenKind::Greater,
                TokenKind::GreaterEqual,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_logical_operators() {
        let result = tokenize_string("&& || !").unwrap();
        assert_eq!(
            result,
            vec![
                TokenKind::And,
                TokenKind::Or,
                TokenKind::Not,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_assignment_and_rules() {
        let result = tokenize_string("= := -> :>").unwrap();
        assert_eq!(
            result,
            vec![
                TokenKind::Set,
                TokenKind::SetDelayed,
                TokenKind::Rule,
                TokenKind::RuleDelayed,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_grouping_brackets() {
        let result = tokenize_string("() [] {} [[ ]]").unwrap();
        assert_eq!(
            result,
            vec![
                TokenKind::LeftParen,
                TokenKind::RightParen,
                TokenKind::LeftBracket,
                TokenKind::RightBracket,
                TokenKind::LeftBrace,
                TokenKind::RightBrace,
                TokenKind::LeftBracket,
                TokenKind::LeftBracket,
                TokenKind::RightBracket,
                TokenKind::RightBracket,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_patterns() {
        let result = tokenize_string("_ __ ___").unwrap();
        assert_eq!(
            result,
            vec![
                TokenKind::Blank,
                TokenKind::BlankSequence,
                TokenKind::BlankNullSequence,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_special_operators() {
        let result = tokenize_string("/. <>").unwrap();
        assert_eq!(
            result,
            vec![TokenKind::ReplaceAll, TokenKind::StringJoin, TokenKind::Eof]
        );
    }

    #[test]
    fn test_modern_operators() {
        let result = tokenize_string("|> // @ => ;; /;").unwrap();
        assert_eq!(
            result,
            vec![
                TokenKind::Pipeline,
                TokenKind::Postfix,
                TokenKind::Prefix,
                TokenKind::Arrow,
                TokenKind::Range,
                TokenKind::Condition,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_association_brackets() {
        let result = tokenize_string("<| |>").unwrap();
        assert_eq!(
            result,
            vec![
                TokenKind::LeftAssoc,
                TokenKind::Pipeline, // Note: |> is parsed as pipeline, need to handle in parser
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_new_separators() {
        let result = tokenize_string(". : ? `").unwrap();
        assert_eq!(
            result,
            vec![
                TokenKind::Dot,
                TokenKind::Colon,
                TokenKind::Question,
                TokenKind::Backtick,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_alternative_pattern() {
        let result = tokenize_string("x | y").unwrap();
        assert_eq!(
            result,
            vec![
                TokenKind::Symbol("x".to_string()),
                TokenKind::Alternative,
                TokenKind::Symbol("y".to_string()),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_extended_numbers() {
        // BigInt
        let result = tokenize_string("123n").unwrap();
        assert_eq!(
            result,
            vec![TokenKind::BigInt("123".to_string()), TokenKind::Eof]
        );

        // BigDecimal
        let result = tokenize_string("1.23d100").unwrap();
        assert_eq!(
            result,
            vec![
                TokenKind::BigDecimal("1.23d100".to_string()),
                TokenKind::Eof
            ]
        );

        // Hex notation
        let result = tokenize_string("16^^FF").unwrap();
        assert_eq!(
            result,
            vec![TokenKind::HexInteger("16^^FF".to_string()), TokenKind::Eof]
        );

        // Rational
        let result = tokenize_string("3/4").unwrap();
        assert_eq!(result, vec![TokenKind::Rational(3, 4), TokenKind::Eof]);
    }

    #[test]
    fn test_context_symbols() {
        let result = tokenize_string("std`net`http`Get").unwrap();
        assert_eq!(
            result,
            vec![
                TokenKind::ContextSymbol("std`net`http`Get".to_string()),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_underscore_in_symbols() {
        // Underscore in middle should be included
        let result = tokenize_string("price_usd").unwrap();
        assert_eq!(
            result,
            vec![TokenKind::Symbol("price_usd".to_string()), TokenKind::Eof]
        );

        // Underscore at end should be separate
        let result = tokenize_string("x_").unwrap();
        assert_eq!(
            result,
            vec![
                TokenKind::Symbol("x".to_string()),
                TokenKind::Blank,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_function_call() {
        let result = tokenize_string("f[x, y]").unwrap();
        assert_eq!(
            result,
            vec![
                TokenKind::Symbol("f".to_string()),
                TokenKind::LeftBracket,
                TokenKind::Symbol("x".to_string()),
                TokenKind::Comma,
                TokenKind::Symbol("y".to_string()),
                TokenKind::RightBracket,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_list() {
        let result = tokenize_string("{1, 2, 3}").unwrap();
        assert_eq!(
            result,
            vec![
                TokenKind::LeftBrace,
                TokenKind::Integer(1),
                TokenKind::Comma,
                TokenKind::Integer(2),
                TokenKind::Comma,
                TokenKind::Integer(3),
                TokenKind::RightBrace,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_complex_expression() {
        let result = tokenize_string("f[x_] = x^2 + 1").unwrap();
        assert_eq!(
            result,
            vec![
                TokenKind::Symbol("f".to_string()),
                TokenKind::LeftBracket,
                TokenKind::Symbol("x".to_string()),
                TokenKind::Blank,
                TokenKind::RightBracket,
                TokenKind::Set,
                TokenKind::Symbol("x".to_string()),
                TokenKind::Power,
                TokenKind::Integer(2),
                TokenKind::Plus,
                TokenKind::Integer(1),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_error_unterminated_string() {
        let result = tokenize_string("\"hello");
        assert!(result.is_err());
        match result.unwrap_err() {
            Error::Lexer { message, position } => {
                assert_eq!(message, "Unterminated string");
                assert_eq!(position, 0);
            }
            _ => panic!("Expected lexer error"),
        }
    }

    #[test]
    fn test_error_invalid_character() {
        let result = tokenize_string("%");
        assert!(result.is_err());
        match result.unwrap_err() {
            Error::Lexer { message, .. } => {
                assert!(message.contains("Unexpected character"));
            }
            _ => panic!("Expected lexer error"),
        }
    }

    #[test]
    fn test_whitespace_handling() {
        let result = tokenize_string("  1   +   2  ").unwrap();
        assert_eq!(
            result,
            vec![
                TokenKind::Integer(1),
                TokenKind::Plus,
                TokenKind::Integer(2),
                TokenKind::Eof
            ]
        );
    }
}
