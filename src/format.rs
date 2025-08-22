use crate::ast::{Expr, InterpolationPart, Number, Pattern, Symbol};
use std::fmt::{self, Write};

#[derive(Clone, Debug)]
pub struct FormatterConfig {
    pub max_width: usize,
    pub indent: usize,
}

impl Default for FormatterConfig {
    fn default() -> Self {
        Self {
            max_width: 100,
            indent: 4,
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum FormatError {
    #[error("formatting failed: {0}")]
    Fmt(String),
}

pub fn format_str(input: &str, cfg: &FormatterConfig) -> Result<String, FormatError> {
    // Parse with your existing parser
    let mut parser = crate::parser::Parser::from_source(input)
        .map_err(|e| FormatError::Fmt(e.to_string()))?;
    let stmts = parser
        .parse()
        .map_err(|e| FormatError::Fmt(e.to_string()))?;

    let mut out = String::with_capacity(input.len());
    for (i, stmt) in stmts.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        format_expr(stmt, 0, cfg, &mut out).map_err(|e| FormatError::Fmt(e.to_string()))?;
    }
    Ok(out)
}

fn format_expr(e: &Expr, level: usize, cfg: &FormatterConfig, out: &mut String) -> fmt::Result {
    match e {
        Expr::Number(n) => format_number(n, out),
        Expr::String(s) => write!(out, "\"{}\"", escape_str(s)),
        Expr::Symbol(Symbol { name }) => write!(out, "{}", name),
        Expr::List(items) => format_list(items, level, cfg, out),
        Expr::Function { head, args } => format_call(head, args, level, cfg, out),
        Expr::Pattern(p) => format_pattern(p, level, cfg, out),
        Expr::Rule { lhs, rhs, delayed } => {
            format_expr(lhs, level, cfg, out)?;
            let arrow = if *delayed { " :> " } else { " -> " };
            write!(out, "{}", arrow)?;
            format_expr(rhs, level, cfg, out)
        }
        Expr::Assignment { lhs, rhs, delayed } => {
            format_expr(lhs, level, cfg, out)?;
            let op = if *delayed { " := " } else { " = " };
            write!(out, "{}", op)?;
            format_expr(rhs, level, cfg, out)
        }
        Expr::Replace { expr, rules, repeated } => {
            format_expr(expr, level, cfg, out)?;
            let operator = if *repeated { " //. " } else { " /. " };
            write!(out, "{}", operator)?;
            format_expr(rules, level, cfg, out)
        }
        Expr::Association(pairs) => format_association(pairs, level, cfg, out),
        Expr::Pipeline { stages } => format_pipeline(stages, level, cfg, out),
        Expr::DotCall {
            object,
            method,
            args,
        } => {
            format_expr(object, level, cfg, out)?;
            write!(out, ".{}[", method)?;
            for (i, arg) in args.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ")?;
                }
                format_expr(arg, level, cfg, out)?;
            }
            write!(out, "]")
        }
        Expr::Range { start, end, step } => {
            format_expr(start, level, cfg, out)?;
            write!(out, ";; ")?;
            format_expr(end, level, cfg, out)?;
            if let Some(s) = step {
                write!(out, ";; ")?;
                format_expr(s, level, cfg, out)?;
            }
            Ok(())
        }
        Expr::ArrowFunction { params, body } => {
            write!(out, "(")?;
            for (i, param) in params.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ")?;
                }
                write!(out, "{}", param)?;
            }
            write!(out, ") => ")?;
            format_expr(body, level, cfg, out)
        }
        Expr::TypedFunction { head, params, return_type } => {
            format_expr(head, level, cfg, out)?;
            write!(out, "[")?;
            for (i, param) in params.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ")?;
                }
                format_expr(param, level, cfg, out)?;
            }
            write!(out, "]: ")?;
            format_expr(return_type, level, cfg, out)
        }
        Expr::InterpolatedString(parts) => {
            write!(out, "\"")?;
            for part in parts {
                match part {
                    InterpolationPart::Text(text) => write!(out, "{}", escape_str(text))?,
                    InterpolationPart::Expression(expr) => {
                        write!(out, "#{{")?;
                        format_expr(expr, level, cfg, out)?;
                        write!(out, "}}")?;
                    }
                }
            }
            write!(out, "\"")
        }
        Expr::PureFunction { body, .. } => {
            format_expr(body, level, cfg, out)?;
            write!(out, " &")
        }
        Expr::Slot { number } => {
            match number {
                Some(n) => write!(out, "#{}", n),
                None => write!(out, "#"),
            }
        }
    }
}

fn format_number(n: &Number, out: &mut String) -> fmt::Result {
    match n {
        Number::Integer(i) => write!(out, "{}", i),
        Number::Real(f) => {
            if f.fract() == 0.0 {
                write!(out, "{:.1}", f)
            } else {
                write!(out, "{}", f)
            }
        }
    }
}

fn format_list(elems: &[Expr], level: usize, cfg: &FormatterConfig, out: &mut String) -> fmt::Result {
    if elems.is_empty() {
        write!(out, "{{}}")
    } else if should_format_inline(elems, level, cfg) {
        write!(out, "{{")?;
        for (i, el) in elems.iter().enumerate() {
            if i > 0 {
                write!(out, ", ")?;
            }
            format_expr(el, level, cfg, out)?;
        }
        write!(out, "}}")
    } else {
        write!(out, "{{")?;
        let next_indent = (level + 1) * cfg.indent;
        for (i, el) in elems.iter().enumerate() {
            if i == 0 {
                write!(out, "\n")?;
            } else {
                write!(out, ",\n")?;
            }
            for _ in 0..next_indent {
                write!(out, " ")?;
            }
            format_expr(el, level + 1, cfg, out)?;
        }
        write!(out, "\n")?;
        for _ in 0..(level * cfg.indent) {
            write!(out, " ")?;
        }
        write!(out, "}}")
    }
}

fn format_call(head: &Expr, args: &[Expr], level: usize, cfg: &FormatterConfig, out: &mut String) -> fmt::Result {
    format_expr(head, level, cfg, out)?;
    write!(out, "[")?;
    for (i, arg) in args.iter().enumerate() {
        if i > 0 {
            write!(out, ", ")?;
        }
        format_expr(arg, level, cfg, out)?;
    }
    write!(out, "]")
}

fn format_pattern(p: &Pattern, level: usize, cfg: &FormatterConfig, out: &mut String) -> fmt::Result {
    match p {
        Pattern::Blank { head } => {
            write!(out, "_")?;
            if let Some(h) = head {
                write!(out, "{}", h)?;
            }
            Ok(())
        }
        Pattern::BlankSequence { head } => {
            write!(out, "__")?;
            if let Some(h) = head {
                write!(out, "{}", h)?;
            }
            Ok(())
        }
        Pattern::BlankNullSequence { head } => {
            write!(out, "___")?;
            if let Some(h) = head {
                write!(out, "{}", h)?;
            }
            Ok(())
        }
        Pattern::Named { name, pattern } => {
            write!(out, "{}", name)?;
            format_pattern(pattern, level, cfg, out)
        }
        Pattern::Function { head, args } => {
            format_pattern(head, level, cfg, out)?;
            write!(out, "[")?;
            for (i, arg) in args.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ")?;
                }
                format_pattern(arg, level, cfg, out)?;
            }
            write!(out, "]")
        }
        Pattern::Exact { value } => {
            format_expr(value, level, cfg, out)
        }
        Pattern::Typed { name, type_pattern } => {
            write!(out, "{}:", name)?;
            format_expr(type_pattern, level, cfg, out)
        }
        Pattern::Predicate { pattern, test } => {
            format_pattern(pattern, level, cfg, out)?;
            write!(out, "?")?;
            format_expr(test, level, cfg, out)
        }
        Pattern::Alternative { patterns } => {
            for (i, pattern) in patterns.iter().enumerate() {
                if i > 0 {
                    write!(out, " | ")?;
                }
                format_pattern(pattern, level, cfg, out)?;
            }
            Ok(())
        }
        Pattern::Conditional { pattern, condition } => {
            format_pattern(pattern, level, cfg, out)?;
            write!(out, " /; ")?;
            format_expr(condition, level, cfg, out)
        }
    }
}

fn format_association(pairs: &[(Expr, Expr)], level: usize, cfg: &FormatterConfig, out: &mut String) -> fmt::Result {
    write!(out, "<|")?;
    for (i, (key, value)) in pairs.iter().enumerate() {
        if i > 0 {
            write!(out, ", ")?;
        }
        format_expr(key, level, cfg, out)?;
        write!(out, " -> ")?;
        format_expr(value, level, cfg, out)?;
    }
    write!(out, "|>")
}

fn format_pipeline(stages: &[Expr], level: usize, cfg: &FormatterConfig, out: &mut String) -> fmt::Result {
    for (i, stage) in stages.iter().enumerate() {
        if i > 0 {
            write!(out, " |> ")?;
        }
        format_expr(stage, level, cfg, out)?;
    }
    Ok(())
}

fn should_format_inline(elems: &[Expr], _level: usize, cfg: &FormatterConfig) -> bool {
    // Simple heuristic: format inline if there are few elements and they're simple
    if elems.len() > 5 {
        return false;
    }
    
    let estimated_width: usize = elems.iter().map(|e| estimate_width(e)).sum::<usize>()
        + (elems.len().saturating_sub(1)) * 2; // ", " separators
    
    estimated_width + 2 <= cfg.max_width // +2 for braces
}

fn estimate_width(expr: &Expr) -> usize {
    match expr {
        Expr::Symbol(Symbol { name }) => name.len(),
        Expr::Number(Number::Integer(i)) => i.to_string().len(),
        Expr::Number(Number::Real(f)) => f.to_string().len(),
        Expr::String(s) => s.len() + 2, // quotes
        Expr::List(items) => {
            2 + items.iter().map(estimate_width).sum::<usize>() + items.len().saturating_sub(1) * 2
        }
        Expr::Function { head, args } => {
            estimate_width(head) + 2 + args.iter().map(estimate_width).sum::<usize>() 
                + args.len().saturating_sub(1) * 2
        }
        _ => 20, // conservative estimate for complex expressions
    }
}

fn escape_str(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_simple_expr() {
        let input = "42";
        let result = format_str(input, &FormatterConfig::default()).unwrap();
        assert_eq!(result, "42");
    }

    #[test]
    fn test_format_function_call() {
        let input = "f[x,y]";
        let result = format_str(input, &FormatterConfig::default()).unwrap();
        assert_eq!(result, "f[x, y]");
    }

    #[test]
    fn test_format_list() {
        let input = "{1,2,3}";
        let result = format_str(input, &FormatterConfig::default()).unwrap();
        assert_eq!(result, "{1, 2, 3}");
    }

    #[test]
    fn test_format_rule() {
        let input = "x->Power[x,2]";
        let result = format_str(input, &FormatterConfig::default()).unwrap();
        assert_eq!(result, "x -> Power[x, 2]");
    }

    #[test]
    fn test_format_assignment() {
        let input = "f[x_]=Power[x,2]";
        let result = format_str(input, &FormatterConfig::default()).unwrap();
        assert_eq!(result, "f[x_] = Power[x, 2]");
    }

    #[test]
    fn test_format_idempotent() {
        let input = "f[x, y] = {1, 2, 3}";
        let result1 = format_str(input, &FormatterConfig::default()).unwrap();
        let result2 = format_str(&result1, &FormatterConfig::default()).unwrap();
        assert_eq!(result1, result2);
    }
}