use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    Symbol(Symbol),
    Number(Number),
    String(String),
    List(Vec<Expr>),
    Function {
        head: Box<Expr>,
        args: Vec<Expr>,
    },
    Pattern(Pattern),
    Rule {
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        delayed: bool,
    },
    Assignment {
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        delayed: bool,
    },
    Replace {
        expr: Box<Expr>,
        rules: Box<Expr>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Symbol {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Number {
    Integer(i64),
    Real(f64),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Pattern {
    Blank {
        head: Option<String>,
    },
    BlankSequence {
        head: Option<String>,
    },
    BlankNullSequence {
        head: Option<String>,
    },
    Named {
        name: String,
        pattern: Box<Pattern>,
    },
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Symbol(s) => write!(f, "{}", s.name),
            Expr::Number(n) => write!(f, "{}", n),
            Expr::String(s) => write!(f, "\"{}\"", s),
            Expr::List(items) => {
                write!(f, "{{")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "}}")
            }
            Expr::Function { head, args } => {
                write!(f, "{}[", head)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, "]")
            }
            Expr::Pattern(p) => write!(f, "{}", p),
            Expr::Rule { lhs, rhs, delayed } => {
                let arrow = if *delayed { ":>" } else { "->" };
                write!(f, "{} {} {}", lhs, arrow, rhs)
            }
            Expr::Assignment { lhs, rhs, delayed } => {
                let op = if *delayed { ":=" } else { "=" };
                write!(f, "{} {} {}", lhs, op, rhs)
            }
            Expr::Replace { expr, rules } => {
                write!(f, "{} /. {}", expr, rules)
            }
        }
    }
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Number::Integer(n) => write!(f, "{}", n),
            Number::Real(n) => write!(f, "{}", n),
        }
    }
}

impl fmt::Display for Pattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Pattern::Blank { head } => {
                write!(f, "_")?;
                if let Some(h) = head {
                    write!(f, "{}", h)?;
                }
                Ok(())
            }
            Pattern::BlankSequence { head } => {
                write!(f, "__")?;
                if let Some(h) = head {
                    write!(f, "{}", h)?;
                }
                Ok(())
            }
            Pattern::BlankNullSequence { head } => {
                write!(f, "___")?;
                if let Some(h) = head {
                    write!(f, "{}", h)?;
                }
                Ok(())
            }
            Pattern::Named { name, pattern } => {
                write!(f, "{}{}", name, pattern)
            }
        }
    }
}

impl Expr {
    pub fn symbol(name: impl Into<String>) -> Self {
        Expr::Symbol(Symbol { name: name.into() })
    }

    pub fn integer(value: i64) -> Self {
        Expr::Number(Number::Integer(value))
    }

    pub fn real(value: f64) -> Self {
        Expr::Number(Number::Real(value))
    }

    pub fn string(value: impl Into<String>) -> Self {
        Expr::String(value.into())
    }

    pub fn list(items: Vec<Expr>) -> Self {
        Expr::List(items)
    }

    pub fn function(head: Expr, args: Vec<Expr>) -> Self {
        Expr::Function {
            head: Box::new(head),
            args,
        }
    }

    pub fn rule(lhs: Expr, rhs: Expr, delayed: bool) -> Self {
        Expr::Rule {
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            delayed,
        }
    }

    pub fn assignment(lhs: Expr, rhs: Expr, delayed: bool) -> Self {
        Expr::Assignment {
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            delayed,
        }
    }

    pub fn replace(expr: Expr, rules: Expr) -> Self {
        Expr::Replace {
            expr: Box::new(expr),
            rules: Box::new(rules),
        }
    }

    pub fn blank(head: Option<String>) -> Self {
        Expr::Pattern(Pattern::Blank { head })
    }

    pub fn blank_sequence(head: Option<String>) -> Self {
        Expr::Pattern(Pattern::BlankSequence { head })
    }

    pub fn blank_null_sequence(head: Option<String>) -> Self {
        Expr::Pattern(Pattern::BlankNullSequence { head })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_symbol_display() {
        let sym = Expr::symbol("x");
        assert_eq!(sym.to_string(), "x");
    }

    #[test]
    fn test_number_display() {
        let int = Expr::integer(42);
        let real = Expr::real(3.14);
        assert_eq!(int.to_string(), "42");
        assert_eq!(real.to_string(), "3.14");
    }

    #[test]
    fn test_string_display() {
        let s = Expr::string("hello");
        assert_eq!(s.to_string(), "\"hello\"");
    }

    #[test]
    fn test_list_display() {
        let list = Expr::list(vec![
            Expr::integer(1),
            Expr::integer(2),
            Expr::integer(3),
        ]);
        assert_eq!(list.to_string(), "{1, 2, 3}");
    }

    #[test]
    fn test_function_display() {
        let func = Expr::function(
            Expr::symbol("f"),
            vec![Expr::symbol("x"), Expr::integer(2)],
        );
        assert_eq!(func.to_string(), "f[x, 2]");
    }

    #[test]
    fn test_pattern_display() {
        let blank = Expr::blank(None);
        let typed_blank = Expr::blank(Some("Integer".to_string()));
        let blank_seq = Expr::blank_sequence(Some("String".to_string()));
        
        assert_eq!(blank.to_string(), "_");
        assert_eq!(typed_blank.to_string(), "_Integer");
        assert_eq!(blank_seq.to_string(), "__String");
    }

    #[test]
    fn test_rule_display() {
        let rule = Expr::rule(
            Expr::symbol("x"),
            Expr::function(Expr::symbol("Power"), vec![Expr::symbol("x"), Expr::integer(2)]),
            false,
        );
        assert_eq!(rule.to_string(), "x -> Power[x, 2]");
    }

    #[test]
    fn test_assignment_display() {
        let assign = Expr::assignment(
            Expr::function(Expr::symbol("f"), vec![Expr::blank(None)]),
            Expr::function(Expr::symbol("Power"), vec![Expr::symbol("x"), Expr::integer(2)]),
            false,
        );
        assert_eq!(assign.to_string(), "f[_] = Power[x, 2]");
    }

    #[test]
    fn test_replace_display() {
        let replace = Expr::replace(
            Expr::symbol("x"),
            Expr::rule(Expr::symbol("x"), Expr::integer(5), false),
        );
        assert_eq!(replace.to_string(), "x /. x -> 5");
    }

    #[test]
    fn test_expr_equality() {
        let expr1 = Expr::function(
            Expr::symbol("Plus"),
            vec![Expr::integer(1), Expr::integer(2)],
        );
        let expr2 = Expr::function(
            Expr::symbol("Plus"),
            vec![Expr::integer(1), Expr::integer(2)],
        );
        assert_eq!(expr1, expr2);
    }

    #[test]
    fn test_nested_expressions() {
        let nested = Expr::function(
            Expr::symbol("f"),
            vec![
                Expr::function(
                    Expr::symbol("g"),
                    vec![Expr::symbol("x")],
                ),
                Expr::list(vec![Expr::integer(1), Expr::integer(2)]),
            ],
        );
        assert_eq!(nested.to_string(), "f[g[x], {1, 2}]");
    }
}