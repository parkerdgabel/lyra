use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Hash, Serialize, Deserialize)]
pub enum InterpolationPart {
    Text(String),
    Expression(Box<Expr>),
}

#[derive(Debug, Clone, PartialEq, Hash, Serialize, Deserialize)]
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
    TypedFunction {
        head: Box<Expr>,
        params: Vec<Expr>,
        return_type: Box<Expr>,
    },
    InterpolatedString(Vec<InterpolationPart>),
    // Pure function support
    PureFunction {
        body: Box<Expr>,
        max_slot: Option<usize>, // highest slot number found, for optimization
    },
    Slot {
        number: Option<usize>, // None for #, Some(n) for #n
    },
}

#[derive(Debug, Clone, PartialEq, Hash, Serialize, Deserialize)]
pub struct Symbol {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Number {
    Integer(i64),
    Real(f64),
}

impl std::hash::Hash for Number {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Number::Integer(i) => {
                0u8.hash(state);
                i.hash(state);
            }
            Number::Real(f) => {
                1u8.hash(state);
                // Use a safe hash for f64 by converting to bits
                f.to_bits().hash(state);
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Hash, Serialize, Deserialize)]
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
    Function {
        head: Box<Pattern>,
        args: Vec<Pattern>,
    },
    // Exact match pattern for symbols, numbers, strings, etc.
    Exact {
        value: Box<Expr>,
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
            Expr::TypedFunction { head, params, return_type } => {
                write!(f, "{}[", head)?;
                for (i, param) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", param)?;
                }
                write!(f, "]: {}", return_type)
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
            Expr::PureFunction { body, .. } => {
                write!(f, "{} &", body)
            }
            Expr::Slot { number } => {
                match number {
                    Some(n) => write!(f, "#{}", n),
                    None => write!(f, "#"),
                }
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
            Pattern::Function { head, args } => {
                write!(f, "{}[", head)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, "]")
            }
            Pattern::Exact { value } => {
                write!(f, "{}", value)
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

    pub fn typed_function(head: Expr, params: Vec<Expr>, return_type: Expr) -> Self {
        Expr::TypedFunction {
            head: Box::new(head),
            params,
            return_type: Box::new(return_type),
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

    // Pure function constructors
    pub fn pure_function(body: Expr) -> Self {
        let max_slot = Self::find_max_slot(&body);
        Expr::PureFunction {
            body: Box::new(body),
            max_slot,
        }
    }

    pub fn slot() -> Self {
        Expr::Slot { number: None }
    }

    pub fn numbered_slot(number: usize) -> Self {
        Expr::Slot {
            number: Some(number),
        }
    }

    // Helper function to find the maximum slot number in an expression
    fn find_max_slot(expr: &Expr) -> Option<usize> {
        match expr {
            Expr::Slot { number } => *number,
            Expr::Function { head, args } => {
                let head_max = Self::find_max_slot(head);
                let args_max = args.iter().filter_map(|arg| Self::find_max_slot(arg)).max();
                match (head_max, args_max) {
                    (Some(h), Some(a)) => Some(h.max(a)),
                    (Some(h), None) => Some(h),
                    (None, Some(a)) => Some(a),
                    (None, None) => None,
                }
            }
            Expr::List(items) => items.iter().filter_map(|item| Self::find_max_slot(item)).max(),
            Expr::Rule { lhs, rhs, .. } => {
                let lhs_max = Self::find_max_slot(lhs);
                let rhs_max = Self::find_max_slot(rhs);
                match (lhs_max, rhs_max) {
                    (Some(l), Some(r)) => Some(l.max(r)),
                    (Some(l), None) => Some(l),
                    (None, Some(r)) => Some(r),
                    (None, None) => None,
                }
            }
            Expr::Assignment { lhs, rhs, .. } => {
                let lhs_max = Self::find_max_slot(lhs);
                let rhs_max = Self::find_max_slot(rhs);
                match (lhs_max, rhs_max) {
                    (Some(l), Some(r)) => Some(l.max(r)),
                    (Some(l), None) => Some(l),
                    (None, Some(r)) => Some(r),
                    (None, None) => None,
                }
            }
            Expr::Replace { expr, rules, .. } => {
                let expr_max = Self::find_max_slot(expr);
                let rules_max = Self::find_max_slot(rules);
                match (expr_max, rules_max) {
                    (Some(e), Some(r)) => Some(e.max(r)),
                    (Some(e), None) => Some(e),
                    (None, Some(r)) => Some(r),
                    (None, None) => None,
                }
            }
            Expr::Association(pairs) => pairs
                .iter()
                .flat_map(|(k, v)| [Self::find_max_slot(k), Self::find_max_slot(v)])
                .flatten()
                .max(),
            Expr::Pipeline { stages } => stages.iter().filter_map(|stage| Self::find_max_slot(stage)).max(),
            Expr::DotCall { object, args, .. } => {
                let obj_max = Self::find_max_slot(object);
                let args_max = args.iter().filter_map(|arg| Self::find_max_slot(arg)).max();
                match (obj_max, args_max) {
                    (Some(o), Some(a)) => Some(o.max(a)),
                    (Some(o), None) => Some(o),
                    (None, Some(a)) => Some(a),
                    (None, None) => None,
                }
            }
            Expr::Range { start, end, step } => {
                let start_max = Self::find_max_slot(start);
                let end_max = Self::find_max_slot(end);
                let step_max = step.as_ref().and_then(|s| Self::find_max_slot(s));
                [start_max, end_max, step_max].into_iter().flatten().max()
            }
            Expr::ArrowFunction { body, .. } => Self::find_max_slot(body),
            Expr::TypedFunction { head, params, return_type } => {
                let head_max = Self::find_max_slot(head);
                let params_max = params.iter().filter_map(|p| Self::find_max_slot(p)).max();
                let return_max = Self::find_max_slot(return_type);
                [head_max, params_max, return_max].into_iter().flatten().max()
            }
            Expr::InterpolatedString(parts) => parts
                .iter()
                .filter_map(|part| match part {
                    InterpolationPart::Expression(expr) => Self::find_max_slot(expr),
                    _ => None,
                })
                .max(),
            Expr::PureFunction { body, .. } => Self::find_max_slot(body),
            // These don't contain slots
            Expr::Symbol(_) | Expr::Number(_) | Expr::String(_) | Expr::Pattern(_) => None,
        }
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

    // Tests for PureFunction and Slot AST nodes
    #[test]
    fn test_slot_creation() {
        let slot = Expr::slot();
        assert_eq!(slot, Expr::Slot { number: None });
        
        let numbered_slot = Expr::numbered_slot(1);
        assert_eq!(numbered_slot, Expr::Slot { number: Some(1) });
        
        let numbered_slot_5 = Expr::numbered_slot(5);
        assert_eq!(numbered_slot_5, Expr::Slot { number: Some(5) });
    }

    #[test]
    fn test_slot_display() {
        let slot = Expr::slot();
        assert_eq!(slot.to_string(), "#");
        
        let slot1 = Expr::numbered_slot(1);
        assert_eq!(slot1.to_string(), "#1");
        
        let slot2 = Expr::numbered_slot(2);
        assert_eq!(slot2.to_string(), "#2");
        
        let slot10 = Expr::numbered_slot(10);
        assert_eq!(slot10.to_string(), "#10");
    }

    #[test]
    fn test_pure_function_creation() {
        let body = Expr::function(
            Expr::symbol("Plus"),
            vec![Expr::slot(), Expr::integer(1)]
        );
        let pure_func = Expr::pure_function(body.clone());
        
        match pure_func {
            Expr::PureFunction { body: func_body, max_slot } => {
                assert_eq!(*func_body, body);
                assert_eq!(max_slot, None); // # has no number
            },
            _ => panic!("Expected PureFunction"),
        }
    }

    #[test]
    fn test_pure_function_display() {
        let simple_slot = Expr::pure_function(Expr::slot());
        assert_eq!(simple_slot.to_string(), "# &");
        
        let slot_plus_one = Expr::pure_function(
            Expr::function(
                Expr::symbol("Plus"),
                vec![Expr::slot(), Expr::integer(1)]
            )
        );
        assert_eq!(slot_plus_one.to_string(), "Plus[#, 1] &");
        
        let numbered_slots = Expr::pure_function(
            Expr::function(
                Expr::symbol("Plus"),
                vec![Expr::numbered_slot(1), Expr::numbered_slot(2)]
            )
        );
        assert_eq!(numbered_slots.to_string(), "Plus[#1, #2] &");
    }

    #[test]
    fn test_find_max_slot_simple() {
        // Test basic slot detection
        assert_eq!(Expr::find_max_slot(&Expr::slot()), None);
        assert_eq!(Expr::find_max_slot(&Expr::numbered_slot(1)), Some(1));
        assert_eq!(Expr::find_max_slot(&Expr::numbered_slot(5)), Some(5));
        
        // Test non-slot expressions
        assert_eq!(Expr::find_max_slot(&Expr::integer(42)), None);
        assert_eq!(Expr::find_max_slot(&Expr::symbol("x")), None);
        assert_eq!(Expr::find_max_slot(&Expr::string("hello")), None);
    }

    #[test]
    fn test_find_max_slot_in_functions() {
        // Function with slots in arguments
        let func_with_slots = Expr::function(
            Expr::symbol("Plus"),
            vec![
                Expr::numbered_slot(1),
                Expr::numbered_slot(3),
                Expr::numbered_slot(2)
            ]
        );
        assert_eq!(Expr::find_max_slot(&func_with_slots), Some(3));
        
        // Function with slot in head
        let func_with_slot_head = Expr::function(
            Expr::numbered_slot(2),
            vec![Expr::integer(1), Expr::integer(2)]
        );
        assert_eq!(Expr::find_max_slot(&func_with_slot_head), Some(2));
        
        // Function with no slots
        let func_no_slots = Expr::function(
            Expr::symbol("Plus"),
            vec![Expr::integer(1), Expr::integer(2)]
        );
        assert_eq!(Expr::find_max_slot(&func_no_slots), None);
    }

    #[test]
    fn test_find_max_slot_in_lists() {
        let list_with_slots = Expr::list(vec![
            Expr::numbered_slot(1),
            Expr::integer(42),
            Expr::numbered_slot(4),
            Expr::numbered_slot(2)
        ]);
        assert_eq!(Expr::find_max_slot(&list_with_slots), Some(4));
        
        let list_no_slots = Expr::list(vec![
            Expr::integer(1),
            Expr::string("hello"),
            Expr::symbol("x")
        ]);
        assert_eq!(Expr::find_max_slot(&list_no_slots), None);
    }

    #[test]
    fn test_find_max_slot_in_rules() {
        let rule_with_slots = Expr::rule(
            Expr::numbered_slot(2),
            Expr::function(
                Expr::symbol("Times"),
                vec![Expr::numbered_slot(2), Expr::numbered_slot(5)]
            ),
            false
        );
        assert_eq!(Expr::find_max_slot(&rule_with_slots), Some(5));
    }

    #[test]
    fn test_find_max_slot_in_assignments() {
        let assignment_with_slots = Expr::assignment(
            Expr::symbol("x"),
            Expr::numbered_slot(3),
            false
        );
        assert_eq!(Expr::find_max_slot(&assignment_with_slots), Some(3));
    }

    #[test]
    fn test_find_max_slot_in_pure_functions() {
        let nested_pure_func = Expr::pure_function(
            Expr::function(
                Expr::symbol("Plus"),
                vec![Expr::numbered_slot(1), Expr::numbered_slot(7)]
            )
        );
        assert_eq!(Expr::find_max_slot(&nested_pure_func), Some(7));
    }

    #[test]
    fn test_find_max_slot_complex_nesting() {
        // Deep nesting: Plus[#1, Times[#3, List[#2, #8]]]
        let complex_expr = Expr::function(
            Expr::symbol("Plus"),
            vec![
                Expr::numbered_slot(1),
                Expr::function(
                    Expr::symbol("Times"),
                    vec![
                        Expr::numbered_slot(3),
                        Expr::list(vec![
                            Expr::numbered_slot(2),
                            Expr::numbered_slot(8)
                        ])
                    ]
                )
            ]
        );
        assert_eq!(Expr::find_max_slot(&complex_expr), Some(8));
    }

    #[test]
    fn test_pure_function_max_slot_calculation() {
        // Test that max_slot is calculated correctly during creation
        let body_with_slots = Expr::function(
            Expr::symbol("Plus"),
            vec![
                Expr::numbered_slot(1),
                Expr::numbered_slot(3),
                Expr::slot() // # has no number
            ]
        );
        let pure_func = Expr::pure_function(body_with_slots);
        
        match pure_func {
            Expr::PureFunction { max_slot, .. } => {
                assert_eq!(max_slot, Some(3));
            },
            _ => panic!("Expected PureFunction"),
        }
    }

    #[test]
    fn test_pure_function_empty_body() {
        let empty_pure_func = Expr::pure_function(Expr::integer(42));
        
        match empty_pure_func {
            Expr::PureFunction { max_slot, .. } => {
                assert_eq!(max_slot, None);
            },
            _ => panic!("Expected PureFunction"),
        }
    }

    #[test]
    fn test_serialization_deserialization() {
        use serde_json;
        
        // Test Slot serialization
        let slot = Expr::slot();
        let slot_json = serde_json::to_string(&slot).unwrap();
        let slot_deserialized: Expr = serde_json::from_str(&slot_json).unwrap();
        assert_eq!(slot, slot_deserialized);
        
        // Test numbered slot serialization
        let numbered_slot = Expr::numbered_slot(5);
        let numbered_slot_json = serde_json::to_string(&numbered_slot).unwrap();
        let numbered_slot_deserialized: Expr = serde_json::from_str(&numbered_slot_json).unwrap();
        assert_eq!(numbered_slot, numbered_slot_deserialized);
        
        // Test PureFunction serialization
        let pure_func = Expr::pure_function(
            Expr::function(
                Expr::symbol("Plus"),
                vec![Expr::numbered_slot(1), Expr::integer(1)]
            )
        );
        let pure_func_json = serde_json::to_string(&pure_func).unwrap();
        let pure_func_deserialized: Expr = serde_json::from_str(&pure_func_json).unwrap();
        assert_eq!(pure_func, pure_func_deserialized);
    }

    #[test]
    fn test_hash_and_equality() {
        use std::collections::HashSet;
        
        // Test that equal slots have equal hashes
        let slot1 = Expr::slot();
        let slot2 = Expr::slot();
        assert_eq!(slot1, slot2);
        
        let numbered_slot1 = Expr::numbered_slot(1);
        let numbered_slot1_copy = Expr::numbered_slot(1);
        assert_eq!(numbered_slot1, numbered_slot1_copy);
        
        // Test that different slots are not equal
        let numbered_slot2 = Expr::numbered_slot(2);
        assert_ne!(numbered_slot1, numbered_slot2);
        assert_ne!(slot1, numbered_slot1);
        
        // Test hash set usage
        let mut slot_set = HashSet::new();
        slot_set.insert(slot1);
        slot_set.insert(numbered_slot1.clone());
        slot_set.insert(numbered_slot2);
        slot_set.insert(numbered_slot1_copy); // Should not increase size
        
        assert_eq!(slot_set.len(), 3);
    }
}
