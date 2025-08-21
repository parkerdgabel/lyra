//! Advanced string operations for the Lyra standard library
//! 
//! This module provides comprehensive string processing capabilities including:
//! - String templating with variable substitution
//! - Regular expression operations
//! - String formatting and manipulation
//! - Encoding/decoding functions
//! - Case transformation utilities

use crate::vm::{Value, VmError, VmResult};
use crate::foreign::{Foreign, ForeignError, LyObj};
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use regex::Regex;
use base64::{Engine as _, engine::general_purpose};
use html_escape::{encode_text, decode_html_entities};
use percent_encoding::{utf8_percent_encode, percent_decode_str, AsciiSet, CONTROLS};

/// URL encoding character set
const FRAGMENT: &AsciiSet = &CONTROLS.add(b' ').add(b'"').add(b'<').add(b'>').add(b'`');

/// Compiled string template with variable substitution support
#[derive(Debug, Clone)]
pub struct StringTemplate {
    template: String,
    compiled_parts: Vec<TemplatePart>,
}

#[derive(Debug, Clone)]
enum TemplatePart {
    Literal(String),
    Variable(String),
    ConditionalBlock {
        variable: String,
        then_content: String,
        else_content: Option<String>,
    },
}

impl Foreign for StringTemplate {
    fn type_name(&self) -> &'static str {
        "StringTemplate"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "render" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                match &args[0] {
                    Value::List(rules) => {
                        let mut substitutions = HashMap::new();
                        for rule in rules {
                            if let Value::List(rule_parts) = rule {
                                if rule_parts.len() == 2 {
                                    if let (Value::String(key), value) = (&rule_parts[0], &rule_parts[1]) {
                                        substitutions.insert(key.clone(), value.clone());
                                    }
                                }
                            }
                        }
                        Ok(Value::String(self.render(&substitutions)?))
                    },
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "List of rules".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                }
            },
            "template" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::String(self.template.clone()))
            },
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl StringTemplate {
    /// Create a new StringTemplate from a template string
    pub fn new(template: String) -> Result<Self, ForeignError> {
        let compiled_parts = Self::compile_template(&template)?;
        Ok(StringTemplate {
            template,
            compiled_parts,
        })
    }

    /// Compile a template string into parts for efficient rendering
    fn compile_template(template: &str) -> Result<Vec<TemplatePart>, ForeignError> {
        let mut parts = Vec::new();
        let mut chars = template.char_indices().peekable();
        let mut current_literal = String::new();

        while let Some((i, ch)) = chars.next() {
            if ch == '{' {
                // Check for escaped braces
                if let Some((_, next_ch)) = chars.peek() {
                    if *next_ch == '{' {
                        chars.next(); // consume the second '{'
                        current_literal.push('{');
                        continue;
                    }
                }

                // Save any accumulated literal
                if !current_literal.is_empty() {
                    parts.push(TemplatePart::Literal(current_literal.clone()));
                    current_literal.clear();
                }

                // Find the closing brace
                let start = i + 1;
                let mut end = start;
                let mut brace_count = 1;
                let mut found_end = false;

                for (j, ch) in template[start..].char_indices() {
                    let pos = start + j;
                    match ch {
                        '{' => brace_count += 1,
                        '}' => {
                            brace_count -= 1;
                            if brace_count == 0 {
                                end = pos;
                                found_end = true;
                                // Skip the closing brace in the main iterator
                                chars.nth(j);
                                break;
                            }
                        },
                        _ => {}
                    }
                }

                if !found_end {
                    return Err(ForeignError::RuntimeError {
                        message: format!("Unclosed variable brace starting at position {}", i),
                    });
                }

                let variable_expr = &template[start..end];
                
                // Parse conditional expressions like {count|item|items}
                if let Some(pipe_pos) = variable_expr.find('|') {
                    let var_name = variable_expr[..pipe_pos].trim().to_string();
                    let conditions = &variable_expr[pipe_pos + 1..];
                    
                    // Simple conditional: {var|singular|plural}
                    if let Some(second_pipe) = conditions.find('|') {
                        let then_content = conditions[..second_pipe].to_string();
                        let else_content = Some(conditions[second_pipe + 1..].to_string());
                        
                        parts.push(TemplatePart::ConditionalBlock {
                            variable: var_name,
                            then_content,
                            else_content,
                        });
                    } else {
                        // Single condition: {var|suffix}
                        parts.push(TemplatePart::ConditionalBlock {
                            variable: var_name,
                            then_content: conditions.to_string(),
                            else_content: None,
                        });
                    }
                } else {
                    // Simple variable substitution
                    parts.push(TemplatePart::Variable(variable_expr.trim().to_string()));
                }
            } else if ch == '}' {
                // Check for escaped braces
                if let Some((_, next_ch)) = chars.peek() {
                    if *next_ch == '}' {
                        chars.next(); // consume the second '}'
                        current_literal.push('}');
                        continue;
                    }
                }
                // Unmatched closing brace
                return Err(ForeignError::RuntimeError {
                    message: format!("Unmatched closing brace at position {}", i),
                });
            } else {
                current_literal.push(ch);
            }
        }

        // Add any remaining literal
        if !current_literal.is_empty() {
            parts.push(TemplatePart::Literal(current_literal));
        }

        Ok(parts)
    }

    /// Render the template with the given substitutions
    fn render(&self, substitutions: &HashMap<String, Value>) -> Result<String, ForeignError> {
        let mut result = String::new();

        for part in &self.compiled_parts {
            match part {
                TemplatePart::Literal(text) => {
                    result.push_str(text);
                },
                TemplatePart::Variable(var_name) => {
                    if let Some(value) = substitutions.get(var_name) {
                        result.push_str(&value_to_string(value));
                    } else {
                        return Err(ForeignError::RuntimeError {
                            message: format!("Missing template variable: {}", var_name),
                        });
                    }
                },
                TemplatePart::ConditionalBlock { variable, then_content, else_content } => {
                    if let Some(value) = substitutions.get(variable) {
                        let should_use_then = match value {
                            Value::Boolean(b) => *b,
                            Value::Integer(n) => *n != 0 && *n != 1,
                            Value::Real(f) => *f != 0.0 && *f != 1.0,
                            Value::String(s) => !s.is_empty(),
                            Value::List(l) => l.len() != 1,
                            _ => false,
                        };
                        
                        if should_use_then {
                            result.push_str(then_content);
                        } else if let Some(else_text) = else_content {
                            result.push_str(else_text);
                        }
                    }
                },
            }
        }

        Ok(result)
    }
}

/// Compiled regular expression pattern
#[derive(Debug, Clone)]
pub struct RegularExpression {
    pattern: String,
    regex: Arc<Regex>,
    flags: RegexFlags,
}

#[derive(Debug, Clone, Default)]
struct RegexFlags {
    case_insensitive: bool,
    multiline: bool,
    dot_matches_newline: bool,
}

impl Foreign for RegularExpression {
    fn type_name(&self) -> &'static str {
        "RegularExpression"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "pattern" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::String(self.pattern.clone()))
            },
            "test" | "match" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                match &args[0] {
                    Value::String(text) => {
                        Ok(Value::Boolean(self.regex.is_match(text)))
                    },
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "String".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                }
            },
            "extract" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                match &args[0] {
                    Value::String(text) => {
                        let matches: Vec<Value> = self.regex.find_iter(text)
                            .map(|m| Value::String(m.as_str().to_string()))
                            .collect();
                        Ok(Value::List(matches))
                    },
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "String".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                }
            },
            "replace" => {
                if args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }
                
                match (&args[0], &args[1]) {
                    (Value::String(text), Value::String(replacement)) => {
                        let result = self.regex.replace_all(text, replacement.as_str());
                        Ok(Value::String(result.to_string()))
                    },
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "String, String".to_string(),
                        actual: format!("{:?}, {:?}", args[0], args[1]),
                    }),
                }
            },
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl RegularExpression {
    /// Create a new RegularExpression from a pattern string
    pub fn new(pattern: String) -> Result<Self, ForeignError> {
        let regex = Regex::new(&pattern)
            .map_err(|e| ForeignError::RuntimeError {
                message: format!("Invalid regex pattern: {}", e),
            })?;
        
        Ok(RegularExpression {
            pattern,
            regex: Arc::new(regex),
            flags: RegexFlags::default(),
        })
    }
}

/// Convert a Value to its string representation
fn value_to_string(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Integer(n) => n.to_string(),
        Value::Real(f) => f.to_string(),
        Value::Boolean(b) => b.to_string(),
        Value::Symbol(s) => s.clone(),
        Value::Missing => "Missing".to_string(),
        Value::List(items) => {
            let elements: Vec<String> = items.iter().map(value_to_string).collect();
            format!("{{{}}}", elements.join(", "))
        },
        Value::Function(name) => format!("Function[{}]", name),
        Value::LyObj(obj) => format!("{}[...]", obj.type_name()),
        Value::Quote(expr) => format!("Hold[{:?}]", expr),
        Value::Pattern(pat) => format!("Pattern[{:?}]", pat),
        Value::Rule { lhs, rhs } => format!("{} -> {}", value_to_string(lhs), value_to_string(rhs)),
    }
}

// ============================================================================
// Standard Library Functions
// ============================================================================

/// Create a StringTemplate object
/// Usage: StringTemplate["Hello {name}, you have {count} {item|s}"]
pub fn string_template(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::String(template) => {
            match StringTemplate::new(template.clone()) {
                Ok(template_obj) => Ok(Value::LyObj(LyObj::new(Box::new(template_obj)))),
                Err(e) => Err(VmError::Runtime(format!("Template compilation failed: {}", e))),
            }
        },
        _ => Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Create a RegularExpression object
/// Usage: RegularExpression["\\w+@\\w+\\.\\w+"]
pub fn regular_expression(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::String(pattern) => {
            match RegularExpression::new(pattern.clone()) {
                Ok(regex_obj) => Ok(Value::LyObj(LyObj::new(Box::new(regex_obj)))),
                Err(e) => Err(VmError::Runtime(format!("Regex compilation failed: {}", e))),
            }
        },
        _ => Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Test if a string matches a regex pattern
/// Usage: StringMatch["test@example.com", RegularExpression["\\w+@\\w+\\.\\w+"]]
pub fn string_match(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String as first argument".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    match &args[1] {
        Value::LyObj(obj) => {
            match obj.call_method("test", &[Value::String(text.clone())]) {
                Ok(result) => Ok(result),
                Err(e) => Err(VmError::Runtime(format!("StringMatch failed: {}", e))),
            }
        },
        _ => Err(VmError::TypeError {
            expected: "RegularExpression as second argument".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    }
}

/// Extract matches from a string using a regex pattern
/// Usage: StringExtract["emails: test@example.com, user@domain.org", RegularExpression["\\w+@\\w+\\.\\w+"]]
pub fn string_extract(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String as first argument".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    match &args[1] {
        Value::LyObj(obj) => {
            match obj.call_method("extract", &[Value::String(text.clone())]) {
                Ok(result) => Ok(result),
                Err(e) => Err(VmError::Runtime(format!("StringExtract failed: {}", e))),
            }
        },
        _ => Err(VmError::TypeError {
            expected: "RegularExpression as second argument".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    }
}

/// Replace matches in a string using a regex pattern
/// Usage: StringReplace["hello world", RegularExpression["\\s+"], "-"]
pub fn string_replace(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "exactly 3 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String as first argument".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let replacement = match &args[2] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String as third argument".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };

    match &args[1] {
        Value::LyObj(obj) => {
            match obj.call_method("replace", &[Value::String(text.clone()), Value::String(replacement.clone())]) {
                Ok(result) => Ok(result),
                Err(e) => Err(VmError::Runtime(format!("StringReplace failed: {}", e))),
            }
        },
        _ => Err(VmError::TypeError {
            expected: "RegularExpression as second argument".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    }
}

/// Split a string by delimiter
/// Usage: StringSplit["hello,world,test", ","] -> {"hello", "world", "test"}
pub fn string_split(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String as first argument".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let delimiter = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String as second argument".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let parts: Vec<Value> = text.split(delimiter)
        .map(|s| Value::String(s.to_string()))
        .collect();

    Ok(Value::List(parts))
}

/// Trim whitespace from a string
/// Usage: StringTrim["  hello  "] -> "hello"
pub fn string_trim(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::String(s) => Ok(Value::String(s.trim().to_string())),
        _ => Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Check if string contains substring
/// Usage: StringContains["hello world", "world"] -> True
pub fn string_contains(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String as first argument".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let substring = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String as second argument".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    Ok(Value::Boolean(text.contains(substring)))
}

/// Check if string starts with prefix
/// Usage: StringStartsWith["hello world", "hello"] -> True
pub fn string_starts_with(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String as first argument".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let prefix = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String as second argument".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    Ok(Value::Boolean(text.starts_with(prefix)))
}

/// Check if string ends with suffix
/// Usage: StringEndsWith["hello world", "world"] -> True
pub fn string_ends_with(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String as first argument".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let suffix = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String as second argument".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    Ok(Value::Boolean(text.ends_with(suffix)))
}

/// Reverse a string
/// Usage: StringReverse["hello"] -> "olleh"
pub fn string_reverse(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::String(s) => {
            let reversed: String = s.chars().rev().collect();
            Ok(Value::String(reversed))
        },
        _ => Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Repeat a string n times
/// Usage: StringRepeat["hello", 3] -> "hellohellohello"
pub fn string_repeat(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String as first argument".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let count = match &args[1] {
        Value::Integer(n) => {
            if *n < 0 {
                return Err(VmError::TypeError {
                    expected: "non-negative integer".to_string(),
                    actual: format!("negative integer: {}", n),
                });
            }
            *n as usize
        },
        _ => return Err(VmError::TypeError {
            expected: "Integer as second argument".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    Ok(Value::String(text.repeat(count)))
}

/// Convert string to uppercase
/// Usage: ToUpperCase["hello"] -> "HELLO"
pub fn to_upper_case(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::String(s) => Ok(Value::String(s.to_uppercase())),
        _ => Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Convert string to lowercase
/// Usage: ToLowerCase["HELLO"] -> "hello"
pub fn to_lower_case(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::String(s) => Ok(Value::String(s.to_lowercase())),
        _ => Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Convert string to title case
/// Usage: TitleCase["hello world"] -> "Hello World"
pub fn title_case(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::String(s) => {
            let title = s.split_whitespace()
                .map(|word| {
                    let mut chars = word.chars();
                    match chars.next() {
                        None => String::new(),
                        Some(first) => first.to_uppercase().chain(chars.as_str().to_lowercase().chars()).collect(),
                    }
                })
                .collect::<Vec<String>>()
                .join(" ");
            Ok(Value::String(title))
        },
        _ => Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Convert string to camelCase
/// Usage: CamelCase["hello world"] -> "helloWorld"
pub fn camel_case(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::String(s) => {
            let words: Vec<&str> = s.split(|c: char| !c.is_alphanumeric()).collect();
            let camel = words.iter()
                .enumerate()
                .map(|(i, word)| {
                    if i == 0 {
                        word.to_lowercase()
                    } else {
                        let mut chars = word.chars();
                        match chars.next() {
                            None => String::new(),
                            Some(first) => first.to_uppercase().chain(chars.as_str().to_lowercase().chars()).collect(),
                        }
                    }
                })
                .collect::<String>();
            Ok(Value::String(camel))
        },
        _ => Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Convert string to snake_case
/// Usage: SnakeCase["Hello World"] -> "hello_world"
pub fn snake_case(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::String(s) => {
            let snake = s.chars()
                .enumerate()
                .map(|(i, c)| {
                    if c.is_uppercase() && i > 0 {
                        format!("_{}", c.to_lowercase())
                    } else if c.is_alphanumeric() {
                        c.to_lowercase().to_string()
                    } else {
                        "_".to_string()
                    }
                })
                .collect::<String>()
                .split("_")
                .filter(|s| !s.is_empty())
                .collect::<Vec<&str>>()
                .join("_");
            Ok(Value::String(snake))
        },
        _ => Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Encode data to Base64
/// Usage: Base64Encode["Hello World"] -> "SGVsbG8gV29ybGQ="
pub fn base64_encode(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::String(s) => {
            let encoded = general_purpose::STANDARD.encode(s.as_bytes());
            Ok(Value::String(encoded))
        },
        _ => Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Decode Base64 data
/// Usage: Base64Decode["SGVsbG8gV29ybGQ="] -> "Hello World"
pub fn base64_decode(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::String(s) => {
            match general_purpose::STANDARD.decode(s) {
                Ok(bytes) => {
                    match String::from_utf8(bytes) {
                        Ok(decoded) => Ok(Value::String(decoded)),
                        Err(_) => Err(VmError::Runtime("Invalid UTF-8 in Base64 decoded data".to_string())),
                    }
                },
                Err(e) => Err(VmError::Runtime(format!("Base64 decode failed: {}", e))),
            }
        },
        _ => Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// URL encode a string
/// Usage: URLEncode["hello world"] -> "hello%20world"
pub fn url_encode(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::String(s) => {
            let encoded = utf8_percent_encode(s, FRAGMENT).to_string();
            Ok(Value::String(encoded))
        },
        _ => Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// URL decode a string
/// Usage: URLDecode["hello%20world"] -> "hello world"
pub fn url_decode(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::String(s) => {
            match percent_decode_str(s).decode_utf8() {
                Ok(decoded) => Ok(Value::String(decoded.to_string())),
                Err(e) => Err(VmError::Runtime(format!("URL decode failed: {}", e))),
            }
        },
        _ => Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// HTML escape a string
/// Usage: HTMLEscape["<div>Hello</div>"] -> "&lt;div&gt;Hello&lt;/div&gt;"
pub fn html_escape(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::String(s) => {
            let escaped = encode_text(s).to_string();
            Ok(Value::String(escaped))
        },
        _ => Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// HTML unescape a string
/// Usage: HTMLUnescape["&lt;div&gt;Hello&lt;/div&gt;"] -> "<div>Hello</div>"
pub fn html_unescape(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::String(s) => {
            let unescaped = decode_html_entities(s).to_string();
            Ok(Value::String(unescaped))
        },
        _ => Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// JSON escape a string (escape quotes and backslashes for JSON)
/// Usage: JSONEscape["Hello \"world\""] -> "Hello \\\"world\\\""
pub fn json_escape(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::String(s) => {
            let escaped = s.chars()
                .map(|c| match c {
                    '"' => "\\\"".to_string(),
                    '\\' => "\\\\".to_string(),
                    '\n' => "\\n".to_string(),
                    '\r' => "\\r".to_string(),
                    '\t' => "\\t".to_string(),
                    c => c.to_string(),
                })
                .collect::<String>();
            Ok(Value::String(escaped))
        },
        _ => Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Printf-style string formatting
/// Usage: StringFormat["{} has {} {}", "Alice", 5, "messages"] -> "Alice has 5 messages"
pub fn string_format(args: &[Value]) -> VmResult<Value> {
    if args.is_empty() {
        return Err(VmError::TypeError {
            expected: "at least 1 argument".to_string(),
            actual: "0 arguments".to_string(),
        });
    }

    let format_string = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String as first argument".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let format_args = &args[1..];
    let mut result = format_string.clone();
    let mut arg_index = 0;

    // Simple {} placeholder replacement
    while let Some(pos) = result.find("{}") {
        if arg_index >= format_args.len() {
            return Err(VmError::Runtime(
                "Not enough arguments for format string".to_string()
            ));
        }
        
        let replacement = value_to_string(&format_args[arg_index]);
        result.replace_range(pos..pos+2, &replacement);
        arg_index += 1;
    }

    if arg_index < format_args.len() {
        return Err(VmError::Runtime(
            "Too many arguments for format string".to_string()
        ));
    }

    Ok(Value::String(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_template_basic() {
        let template = StringTemplate::new("Hello {name}!".to_string()).unwrap();
        let mut substitutions = HashMap::new();
        substitutions.insert("name".to_string(), Value::String("World".to_string()));
        
        let result = template.render(&substitutions).unwrap();
        assert_eq!(result, "Hello World!");
    }

    #[test]
    fn test_string_template_conditional() {
        let template = StringTemplate::new("You have {count} {item|s}".to_string()).unwrap();
        
        // Test singular
        let mut substitutions = HashMap::new();
        substitutions.insert("count".to_string(), Value::Integer(1));
        substitutions.insert("item".to_string(), Value::String("message".to_string()));
        let result = template.render(&substitutions).unwrap();
        assert_eq!(result, "You have 1 message");
        
        // Test plural
        substitutions.insert("count".to_string(), Value::Integer(5));
        let result = template.render(&substitutions).unwrap();
        assert_eq!(result, "You have 5 messages");
    }

    #[test]
    fn test_regular_expression() {
        let regex = RegularExpression::new(r"\w+@\w+\.\w+".to_string()).unwrap();
        
        // Test pattern method
        let pattern_result = regex.call_method("pattern", &[]).unwrap();
        assert_eq!(pattern_result, Value::String(r"\w+@\w+\.\w+".to_string()));
        
        // Test match method
        let match_result = regex.call_method("test", &[Value::String("test@example.com".to_string())]).unwrap();
        assert_eq!(match_result, Value::Boolean(true));
        
        let no_match_result = regex.call_method("test", &[Value::String("invalid email".to_string())]).unwrap();
        assert_eq!(no_match_result, Value::Boolean(false));
    }

    #[test]
    fn test_string_split() {
        let args = vec![
            Value::String("hello,world,test".to_string()),
            Value::String(",".to_string()),
        ];
        let result = string_split(&args).unwrap();
        
        if let Value::List(parts) = result {
            assert_eq!(parts.len(), 3);
            assert_eq!(parts[0], Value::String("hello".to_string()));
            assert_eq!(parts[1], Value::String("world".to_string()));
            assert_eq!(parts[2], Value::String("test".to_string()));
        } else {
            panic!("Expected List result");
        }
    }

    #[test]
    fn test_string_trim() {
        let args = vec![Value::String("  hello world  ".to_string())];
        let result = string_trim(&args).unwrap();
        assert_eq!(result, Value::String("hello world".to_string()));
    }

    #[test]
    fn test_string_contains() {
        let args = vec![
            Value::String("hello world".to_string()),
            Value::String("world".to_string()),
        ];
        let result = string_contains(&args).unwrap();
        assert_eq!(result, Value::Boolean(true));
        
        let args = vec![
            Value::String("hello world".to_string()),
            Value::String("foo".to_string()),
        ];
        let result = string_contains(&args).unwrap();
        assert_eq!(result, Value::Boolean(false));
    }

    #[test]
    fn test_case_operations() {
        // Test uppercase
        let args = vec![Value::String("hello".to_string())];
        let result = to_upper_case(&args).unwrap();
        assert_eq!(result, Value::String("HELLO".to_string()));
        
        // Test lowercase
        let args = vec![Value::String("HELLO".to_string())];
        let result = to_lower_case(&args).unwrap();
        assert_eq!(result, Value::String("hello".to_string()));
        
        // Test title case
        let args = vec![Value::String("hello world".to_string())];
        let result = title_case(&args).unwrap();
        assert_eq!(result, Value::String("Hello World".to_string()));
        
        // Test camel case
        let args = vec![Value::String("hello world".to_string())];
        let result = camel_case(&args).unwrap();
        assert_eq!(result, Value::String("helloWorld".to_string()));
        
        // Test snake case
        let args = vec![Value::String("Hello World".to_string())];
        let result = snake_case(&args).unwrap();
        assert_eq!(result, Value::String("hello_world".to_string()));
    }

    #[test]
    fn test_base64_encoding() {
        // Test encode
        let args = vec![Value::String("Hello World".to_string())];
        let encoded = base64_encode(&args).unwrap();
        assert_eq!(encoded, Value::String("SGVsbG8gV29ybGQ=".to_string()));
        
        // Test decode
        let args = vec![Value::String("SGVsbG8gV29ybGQ=".to_string())];
        let decoded = base64_decode(&args).unwrap();
        assert_eq!(decoded, Value::String("Hello World".to_string()));
    }

    #[test]
    fn test_url_encoding() {
        // Test encode
        let args = vec![Value::String("hello world".to_string())];
        let encoded = url_encode(&args).unwrap();
        assert_eq!(encoded, Value::String("hello%20world".to_string()));
        
        // Test decode
        let args = vec![Value::String("hello%20world".to_string())];
        let decoded = url_decode(&args).unwrap();
        assert_eq!(decoded, Value::String("hello world".to_string()));
    }

    #[test]
    fn test_string_format() {
        let args = vec![
            Value::String("Hello {}, you have {} {}!".to_string()),
            Value::String("Alice".to_string()),
            Value::Integer(5),
            Value::String("messages".to_string()),
        ];
        let result = string_format(&args).unwrap();
        assert_eq!(result, Value::String("Hello Alice, you have 5 messages!".to_string()));
    }

    #[test]
    fn test_string_repeat() {
        let args = vec![
            Value::String("hello".to_string()),
            Value::Integer(3),
        ];
        let result = string_repeat(&args).unwrap();
        assert_eq!(result, Value::String("hellohellohello".to_string()));
    }

    #[test]
    fn test_string_reverse() {
        let args = vec![Value::String("hello".to_string())];
        let result = string_reverse(&args).unwrap();
        assert_eq!(result, Value::String("olleh".to_string()));
    }
}