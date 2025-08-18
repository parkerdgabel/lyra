use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InterpolationPart {
    Text(String),
    Expression(Box<Expr>),
}

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
        repeated: bool, // false for /., true for //.
    },
    // Modern syntax extensions
    Association(Vec<(Expr, Expr)>),
    Pipeline {
        stages: Vec<Expr>,
    },
    DotCall {
        object: Box<Expr>,
        method: String,
        args: Vec<Expr>,
    },
    Range {
        start: Box<Expr>,
        end: Box<Expr>,
        step: Option<Box<Expr>>,
    },
    ArrowFunction {
        params: Vec<String>,
        body: Box<Expr>,
    },
    InterpolatedString(Vec<InterpolationPart>),
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
    // Modern pattern extensions
    Typed {
        name: String,
        type_pattern: Box<Expr>,
    },
    Predicate {
        pattern: Box<Pattern>,
        test: Box<Expr>,
    },
    Alternative {
        patterns: Vec<Pattern>,
    },
    Conditional {
        pattern: Box<Pattern>,
        condition: Box<Expr>,
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
            Expr::Replace { expr, rules, repeated } => {
                let operator = if *repeated { "//." } else { "/." };
                write!(f, "{} {} {}", expr, operator, rules)
            }
            Expr::Association(pairs) => {
                write!(f, "<|")?;
                for (i, (key, value)) in pairs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{} -> {}", key, value)?;
                }
                write!(f, "|>")
            }
            Expr::Pipeline { stages } => {
                for (i, stage) in stages.iter().enumerate() {
                    if i > 0 {
                        write!(f, " |> ")?;
                    }
                    write!(f, "{}", stage)?;
                }
                Ok(())
            }
            Expr::DotCall {
                object,
                method,
                args,
            } => {
                write!(f, "{}.", object)?;
                write!(f, "{}[", method)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, "]")
            }
            Expr::Range { start, end, step } => {
                write!(f, "{};; {}", start, end)?;
                if let Some(s) = step {
                    write!(f, ";; {}", s)?;
                }
                Ok(())
            }
            Expr::ArrowFunction { params, body } => {
                write!(f, "(")?;
                for (i, param) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", param)?;
                }
                write!(f, ") => {}", body)
            }
            Expr::InterpolatedString(parts) => {
                write!(f, "\"")?;
                for part in parts {
                    match part {
                        InterpolationPart::Text(text) => write!(f, "{}", text)?,
                        InterpolationPart::Expression(expr) => write!(f, "#{{{}}}", expr)?,
                    }
                }
                write!(f, "\"")
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
            Pattern::Typed { name, type_pattern } => {
                write!(f, "{}:{}", name, type_pattern)
            }
            Pattern::Predicate { pattern, test } => {
                write!(f, "{}?{}", pattern, test)
            }
            Pattern::Alternative { patterns } => {
                for (i, pattern) in patterns.iter().enumerate() {
                    if i > 0 {
                        write!(f, " | ")?;
                    }
                    write!(f, "{}", pattern)?;
                }
                Ok(())
            }
            Pattern::Conditional { pattern, condition } => {
                write!(f, "{} /; {}", pattern, condition)
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
            repeated: false,
        }
    }

    pub fn replace_repeated(expr: Expr, rules: Expr) -> Self {
        Expr::Replace {
            expr: Box::new(expr),
            rules: Box::new(rules),
            repeated: true,
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

    // Modern expression constructors
    pub fn association(pairs: Vec<(Expr, Expr)>) -> Self {
        Expr::Association(pairs)
    }

    pub fn pipeline(stages: Vec<Expr>) -> Self {
        Expr::Pipeline { stages }
    }

    pub fn dot_call(object: Expr, method: impl Into<String>, args: Vec<Expr>) -> Self {
        Expr::DotCall {
            object: Box::new(object),
            method: method.into(),
            args,
        }
    }

    pub fn range(start: Expr, end: Expr, step: Option<Expr>) -> Self {
        Expr::Range {
            start: Box::new(start),
            end: Box::new(end),
            step: step.map(Box::new),
        }
    }

    pub fn arrow_function(params: Vec<String>, body: Expr) -> Self {
        Expr::ArrowFunction {
            params,
            body: Box::new(body),
        }
    }

    pub fn interpolated_string(parts: Vec<InterpolationPart>) -> Self {
        Expr::InterpolatedString(parts)
    }

    // Modern pattern constructors
    pub fn typed_pattern(name: impl Into<String>, type_pattern: Expr) -> Self {
        Expr::Pattern(Pattern::Typed {
            name: name.into(),
            type_pattern: Box::new(type_pattern),
        })
    }

    pub fn predicate_pattern(pattern: Pattern, test: Expr) -> Self {
        Expr::Pattern(Pattern::Predicate {
            pattern: Box::new(pattern),
            test: Box::new(test),
        })
    }

    pub fn alternative_pattern(patterns: Vec<Pattern>) -> Self {
        Expr::Pattern(Pattern::Alternative { patterns })
    }

    pub fn conditional_pattern(pattern: Pattern, condition: Expr) -> Self {
        Expr::Pattern(Pattern::Conditional {
            pattern: Box::new(pattern),
            condition: Box::new(condition),
        })
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
        let list = Expr::list(vec![Expr::integer(1), Expr::integer(2), Expr::integer(3)]);
        assert_eq!(list.to_string(), "{1, 2, 3}");
    }

    #[test]
    fn test_function_display() {
        let func = Expr::function(Expr::symbol("f"), vec![Expr::symbol("x"), Expr::integer(2)]);
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
            Expr::function(
                Expr::symbol("Power"),
                vec![Expr::symbol("x"), Expr::integer(2)],
            ),
            false,
        );
        assert_eq!(rule.to_string(), "x -> Power[x, 2]");
    }

    #[test]
    fn test_assignment_display() {
        let assign = Expr::assignment(
            Expr::function(Expr::symbol("f"), vec![Expr::blank(None)]),
            Expr::function(
                Expr::symbol("Power"),
                vec![Expr::symbol("x"), Expr::integer(2)],
            ),
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
                Expr::function(Expr::symbol("g"), vec![Expr::symbol("x")]),
                Expr::list(vec![Expr::integer(1), Expr::integer(2)]),
            ],
        );
        assert_eq!(nested.to_string(), "f[g[x], {1, 2}]");
    }

    // Modern syntax tests
    #[test]
    fn test_association_display() {
        let assoc = Expr::association(vec![
            (Expr::string("name"), Expr::string("Ada")),
            (Expr::string("age"), Expr::integer(37)),
        ]);
        assert_eq!(assoc.to_string(), "<|\"name\" -> \"Ada\", \"age\" -> 37|>");
    }

    #[test]
    fn test_pipeline_display() {
        let pipeline = Expr::pipeline(vec![
            Expr::symbol("x"),
            Expr::symbol("f"),
            Expr::function(Expr::symbol("g"), vec![Expr::integer(2)]),
        ]);
        assert_eq!(pipeline.to_string(), "x |> f |> g[2]");
    }

    #[test]
    fn test_dot_call_display() {
        let dot_call = Expr::dot_call(
            Expr::symbol("obj"),
            "method",
            vec![Expr::symbol("x"), Expr::integer(1)],
        );
        assert_eq!(dot_call.to_string(), "obj.method[x, 1]");
    }

    #[test]
    fn test_range_display() {
        let range1 = Expr::range(Expr::integer(1), Expr::integer(10), None);
        assert_eq!(range1.to_string(), "1;; 10");

        let range2 = Expr::range(Expr::integer(0), Expr::integer(1), Some(Expr::real(0.1)));
        assert_eq!(range2.to_string(), "0;; 1;; 0.1");
    }

    #[test]
    fn test_arrow_function_display() {
        let arrow = Expr::arrow_function(
            vec!["x".to_string()],
            Expr::function(
                Expr::symbol("Plus"),
                vec![Expr::symbol("x"), Expr::integer(1)],
            ),
        );
        assert_eq!(arrow.to_string(), "(x) => Plus[x, 1]");

        let multi_param = Expr::arrow_function(
            vec!["x".to_string(), "y".to_string()],
            Expr::function(
                Expr::symbol("Plus"),
                vec![Expr::symbol("x"), Expr::symbol("y")],
            ),
        );
        assert_eq!(multi_param.to_string(), "(x, y) => Plus[x, y]");
    }

    #[test]
    fn test_interpolated_string_display() {
        let interpolated = Expr::interpolated_string(vec![
            InterpolationPart::Text("Hello ".to_string()),
            InterpolationPart::Expression(Box::new(Expr::symbol("name"))),
            InterpolationPart::Text("!".to_string()),
        ]);
        assert_eq!(interpolated.to_string(), "\"Hello #{name}!\"");
    }

    #[test]
    fn test_typed_pattern_display() {
        let typed = Expr::typed_pattern("x", Expr::symbol("Integer"));
        assert_eq!(typed.to_string(), "x:Integer");
    }

    #[test]
    fn test_predicate_pattern_display() {
        let predicate =
            Expr::predicate_pattern(Pattern::Blank { head: None }, Expr::symbol("Positive"));
        assert_eq!(predicate.to_string(), "_?Positive");
    }

    #[test]
    fn test_alternative_pattern_display() {
        let alternative = Expr::alternative_pattern(vec![
            Pattern::Blank {
                head: Some("Integer".to_string()),
            },
            Pattern::Blank {
                head: Some("Real".to_string()),
            },
        ]);
        assert_eq!(alternative.to_string(), "_Integer | _Real");
    }

    #[test]
    fn test_conditional_pattern_display() {
        let conditional = Expr::conditional_pattern(
            Pattern::Named {
                name: "x".to_string(),
                pattern: Box::new(Pattern::Blank { head: None }),
            },
            Expr::function(
                Expr::symbol("Greater"),
                vec![Expr::symbol("x"), Expr::integer(0)],
            ),
        );
        assert_eq!(conditional.to_string(), "x_ /; Greater[x, 0]");
    }

    #[test]
    fn test_modern_expressions_equality() {
        let assoc1 = Expr::association(vec![(Expr::string("key"), Expr::integer(1))]);
        let assoc2 = Expr::association(vec![(Expr::string("key"), Expr::integer(1))]);
        assert_eq!(assoc1, assoc2);

        let pipeline1 = Expr::pipeline(vec![Expr::symbol("x"), Expr::symbol("f")]);
        let pipeline2 = Expr::pipeline(vec![Expr::symbol("x"), Expr::symbol("f")]);
        assert_eq!(pipeline1, pipeline2);
    }

    #[test]
    fn test_complex_modern_expression() {
        let complex = Expr::pipeline(vec![
            Expr::association(vec![(
                Expr::string("data"),
                Expr::list(vec![Expr::integer(1), Expr::integer(2)]),
            )]),
            Expr::dot_call(
                Expr::symbol("data"),
                "map",
                vec![Expr::arrow_function(
                    vec!["x".to_string()],
                    Expr::function(
                        Expr::symbol("Times"),
                        vec![Expr::symbol("x"), Expr::integer(2)],
                    ),
                )],
            ),
        ]);
        assert_eq!(
            complex.to_string(),
            "<|\"data\" -> {1, 2}|> |> data.map[(x) => Times[x, 2]]"
        );
    }
}
