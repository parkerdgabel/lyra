//! Enhanced Error Messages for Lyra Programming Language
//!
//! This module provides user-friendly error messages with helpful suggestions,
//! recovery hints, and "Did you mean?" alternatives for common mistakes.

use std::collections::HashMap;
use crate::error::LyraError;
use crate::vm::VmError;

/// Enhanced error message generator with contextual suggestions
pub struct ErrorEnhancer {
    /// Function name typo database
    function_typos: HashMap<String, Vec<FunctionSuggestion>>,
    /// Common syntax error patterns
    syntax_patterns: Vec<SyntaxPattern>,
    /// Built-in function signatures for validation
    function_signatures: HashMap<String, FunctionSignature>,
    /// Domain error patterns for mathematical functions
    domain_patterns: Vec<DomainPattern>,
}

/// Function suggestion with confidence score
#[derive(Debug, Clone)]
pub struct FunctionSuggestion {
    pub suggested_name: String,
    pub confidence: f32,
    pub usage_example: String,
    pub description: String,
}

/// Syntax error pattern matching
#[derive(Debug, Clone)]
pub struct SyntaxPattern {
    pub pattern_name: String,
    pub recognizer: fn(&str) -> bool,
    pub message_generator: fn(&str) -> String,
    pub suggestions: Vec<String>,
    pub fix_examples: Vec<String>,
}

/// Function signature for validation
#[derive(Debug, Clone)]
pub struct FunctionSignature {
    pub name: String,
    pub min_args: usize,
    pub max_args: Option<usize>,
    pub param_types: Vec<String>,
    pub return_type: String,
    pub usage_example: String,
    pub description: String,
}

/// Domain error pattern for mathematical functions
#[derive(Debug, Clone)]
pub struct DomainPattern {
    pub function_name: String,
    pub error_condition: String,
    pub suggestion: String,
    pub alternative: String,
}

/// Enhanced error message with suggestions
#[derive(Debug, Clone)]
pub struct EnhancedError {
    pub original_error: String,
    pub friendly_message: String,
    pub error_category: ErrorCategory,
    pub position_info: Option<PositionInfo>,
    pub suggestions: Vec<ErrorSuggestion>,
    pub did_you_mean: Vec<String>,
    pub code_examples: Vec<CodeExample>,
    pub recovery_steps: Vec<String>,
    pub related_docs: Vec<String>,
}

/// Error category for targeted handling
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorCategory {
    UnknownFunction,
    BracketMismatch,
    ArgumentCount,
    ArgumentType,
    DivisionByZero,
    IndexOutOfBounds,
    ParseError,
    FileNotFound,
    PatternSyntax,
    RuntimeEvaluation,
}

/// Position information for precise error location
#[derive(Debug, Clone)]
pub struct PositionInfo {
    pub line: usize,
    pub column: usize,
    pub position: usize,
    pub context: String,
    pub pointer: String, // Visual pointer like "   ^"
}

/// Individual error suggestion
#[derive(Debug, Clone)]
pub struct ErrorSuggestion {
    pub title: String,
    pub description: String,
    pub confidence: f32,
    pub fix_code: Option<String>,
    pub why_this_helps: String,
}

/// Code example with explanation
#[derive(Debug, Clone)]  
pub struct CodeExample {
    pub title: String,
    pub correct_code: String,
    pub explanation: String,
    pub common_mistake: Option<String>,
}

impl ErrorEnhancer {
    /// Create a new error enhancer with built-in knowledge
    pub fn new() -> Self {
        let mut enhancer = Self {
            function_typos: HashMap::new(),
            syntax_patterns: Vec::new(),
            function_signatures: HashMap::new(),
            domain_patterns: Vec::new(),
        };
        
        enhancer.populate_function_database();
        enhancer.populate_syntax_patterns();
        enhancer.populate_function_signatures();
        enhancer.populate_domain_patterns();
        
        enhancer
    }
    
    /// Enhance any error with helpful suggestions
    pub fn enhance_error(&self, error: &LyraError, source_code: &str) -> EnhancedError {
        match error {
            LyraError::UnknownSymbol { symbol } => {
                self.enhance_unknown_function_error(symbol, source_code)
            },
            LyraError::Parse { message, position } => {
                self.enhance_parse_error(message, *position, source_code)
            },
            LyraError::Type { expected, actual } => {
                self.enhance_type_error(expected, actual, source_code)
            },
            LyraError::Vm(vm_error) => {
                self.enhance_vm_error(vm_error, source_code)
            },
            _ => self.enhance_generic_error(error, source_code),
        }
    }
    
    /// Enhance unknown function error with "Did you mean?" suggestions
    fn enhance_unknown_function_error(&self, symbol: &str, source_code: &str) -> EnhancedError {
        let suggestions = self.find_function_suggestions(symbol);
        let did_you_mean = suggestions.iter().map(|s| s.suggested_name.clone()).collect();
        
        let friendly_message = if suggestions.is_empty() {
            format!("I don't recognize the function '{}'.", symbol)
        } else {
            format!("The function '{}' isn't defined.", symbol)
        };
        
        let error_suggestions = suggestions.into_iter().map(|s| {
            ErrorSuggestion {
                title: format!("Use '{}' instead", s.suggested_name),
                description: s.description,
                confidence: s.confidence,
                fix_code: Some(source_code.replace(symbol, &s.suggested_name)),
                why_this_helps: format!("'{}' is a built-in function with similar spelling", s.suggested_name),
            }
        }).collect();
        
        let code_examples = vec![
            CodeExample {
                title: "Function Call Syntax".to_string(),
                correct_code: "Sin[Pi/2]  (* Use square brackets *)".to_string(),
                explanation: "Function calls in Lyra use square brackets [ ]".to_string(),
                common_mistake: Some("Sin(Pi/2)  (* Parentheses are incorrect *)".to_string()),
            },
            CodeExample {
                title: "Case Sensitivity".to_string(), 
                correct_code: "Sin[x]  (* Capital S *)".to_string(),
                explanation: "Function names are case-sensitive".to_string(),
                common_mistake: Some("sin[x]  (* Lowercase s won't work *)".to_string()),
            },
        ];
        
        EnhancedError {
            original_error: format!("Unknown symbol: {}", symbol),
            friendly_message,
            error_category: ErrorCategory::UnknownFunction,
            position_info: None,
            suggestions: error_suggestions,
            did_you_mean,
            code_examples,
            recovery_steps: vec![
                "Check the spelling and capitalization of the function name".to_string(),
                "Use the 'functions' command to see all available functions".to_string(),
                "Make sure to use square brackets [ ] for function calls".to_string(),
            ],
            related_docs: vec![
                "Function Reference Guide".to_string(),
                "Built-in Functions List".to_string(),
            ],
        }
    }
    
    /// Enhance parse error with position and syntax suggestions
    fn enhance_parse_error(&self, message: &str, position: usize, source_code: &str) -> EnhancedError {
        let position_info = self.create_position_info(position, source_code);
        let category = self.categorize_parse_error(message, source_code);
        
        let (friendly_message, suggestions, examples, recovery_steps) = match category {
            ErrorCategory::BracketMismatch => {
                self.create_bracket_error_details(source_code)
            },
            _ => self.create_generic_parse_error_details(message, source_code),
        };
        
        EnhancedError {
            original_error: message.to_string(),
            friendly_message,
            error_category: category,
            position_info: Some(position_info),
            suggestions,
            did_you_mean: vec![],
            code_examples: examples,
            recovery_steps,
            related_docs: vec!["Syntax Reference".to_string()],
        }
    }
    
    /// Enhance type error with conversion suggestions
    fn enhance_type_error(&self, expected: &str, actual: &str, source_code: &str) -> EnhancedError {
        let conversion_suggestion = self.suggest_type_conversion(expected, actual, source_code);
        
        let suggestions = vec![
            ErrorSuggestion {
                title: format!("Convert {} to {}", actual, expected),
                description: format!("The function expects {} but received {}", expected, actual),
                confidence: 0.9,
                fix_code: conversion_suggestion.clone(),
                why_this_helps: "Type conversion ensures the argument matches the expected type".to_string(),
            }
        ];
        
        let code_examples = vec![
            CodeExample {
                title: "Type Conversion".to_string(),
                correct_code: match expected.as_ref() {
                    "Number" => "ToNumber[\"123\"]  (* Convert string to number *)".to_string(),
                    "String" => "ToString[42]      (* Convert number to string *)".to_string(),  
                    "List" => "{value}           (* Wrap single value in list *)".to_string(),
                    _ => format!("Convert to {}", expected),
                },
                explanation: "Use built-in conversion functions to change types".to_string(),
                common_mistake: Some("Passing wrong type directly without conversion".to_string()),
            }
        ];
        
        EnhancedError {
            original_error: format!("Type error: expected {}, got {}", expected, actual),
            friendly_message: format!("Expected {} but got {}. The types don't match.", expected, actual),
            error_category: ErrorCategory::ArgumentType,
            position_info: None,
            suggestions,
            did_you_mean: vec![],
            code_examples,
            recovery_steps: vec![
                format!("Convert the {} to {} before using", actual, expected),
                "Check the function's documentation for expected parameter types".to_string(),
                "Verify your data has the correct type before passing it".to_string(),
            ],
            related_docs: vec!["Type System Guide".to_string(), "Conversion Functions".to_string()],
        }
    }
    
    /// Enhance VM errors (runtime errors)
    fn enhance_vm_error(&self, vm_error: &VmError, source_code: &str) -> EnhancedError {
        match vm_error {
            VmError::ArityError { function_name, expected, actual } => {
                self.enhance_arity_error(function_name, *expected, *actual, source_code)
            },
            VmError::ArgumentTypeError { function_name, param_index, expected, actual } => {
                self.enhance_argument_type_error(function_name, *param_index, expected, actual, source_code)
            },
            VmError::UnknownFunction { function_name } => {
                self.enhance_unknown_function_error(function_name, source_code)
            },
            VmError::DivisionByZero => {
                EnhancedError {
                    original_error: "Division by zero".to_string(),
                    friendly_message: "You can't divide by zero - it's mathematically undefined.".to_string(),
                    error_category: ErrorCategory::DivisionByZero,
                    position_info: None,
                    suggestions: vec![
                        ErrorSuggestion {
                            title: "Add zero check".to_string(),
                            description: "Check if the divisor is zero before dividing".to_string(),
                            confidence: 0.95,
                            fix_code: Some("If[divisor != 0, numerator/divisor, \"undefined\"]".to_string()),
                            why_this_helps: "Prevents division by zero errors at runtime".to_string(),
                        }
                    ],
                    did_you_mean: vec![],
                    code_examples: vec![
                        CodeExample {
                            title: "Safe Division".to_string(),
                            correct_code: "If[b != 0, a/b, Missing]  (* Check before dividing *)".to_string(),
                            explanation: "Always verify the divisor isn't zero".to_string(),
                            common_mistake: Some("a/b  (* Could fail if b is zero *)".to_string()),
                        }
                    ],
                    recovery_steps: vec![
                        "Check if any variables in your expression could be zero".to_string(),
                        "Add a condition to handle the zero case explicitly".to_string(),
                        "Use Missing or a default value when division by zero occurs".to_string(),
                    ],
                    related_docs: vec!["Conditional Expressions".to_string()],
                }
            },
            VmError::IndexError { index, length } => {
                EnhancedError {
                    original_error: format!("Index {} out of bounds for length {}", index, length),
                    friendly_message: format!("The list only has {} items, but you're trying to access item #{}.", length, index),
                    error_category: ErrorCategory::IndexOutOfBounds,
                    position_info: None,
                    suggestions: vec![
                        ErrorSuggestion {
                            title: "Check list bounds".to_string(),
                            description: format!("Use an index between 1 and {} (Lyra uses 1-based indexing)", length),
                            confidence: 0.95,
                            fix_code: Some(format!("If[index <= Length[list], list[[index]], Missing]")),
                            why_this_helps: "Bounds checking prevents index errors".to_string(),
                        }
                    ],
                    did_you_mean: vec![],
                    code_examples: vec![
                        CodeExample {
                            title: "Safe List Access".to_string(),
                            correct_code: "If[i <= Length[list], list[[i]], Missing]".to_string(),
                            explanation: "Check bounds before accessing list elements".to_string(),
                            common_mistake: Some("list[[i]]  (* Could fail if i > Length[list] *)".to_string()),
                        }
                    ],
                    recovery_steps: vec![
                        format!("Use an index between 1 and {} (Lyra uses 1-based indexing)", length),
                        "Check the length of your list with Length[list]".to_string(),
                        "Add bounds checking before accessing list elements".to_string(),
                    ],
                    related_docs: vec!["List Operations".to_string(), "Index-Based Access".to_string()],
                }
            },
            _ => self.enhance_generic_vm_error(vm_error, source_code),
        }
    }
    
    /// Find function suggestions based on similarity
    fn find_function_suggestions(&self, symbol: &str) -> Vec<FunctionSuggestion> {
        let mut suggestions = Vec::new();
        let symbol_lower = symbol.to_lowercase();
        
        // Check direct typo mappings first
        if let Some(typo_suggestions) = self.function_typos.get(&symbol_lower) {
            suggestions.extend(typo_suggestions.clone());
        }
        
        // Calculate similarity with all known functions
        let known_functions = vec![
            "Sin", "Cos", "Tan", "ArcSin", "ArcCos", "ArcTan",
            "Exp", "Log", "Log10", "Sqrt", "Abs", "Sign",
            "Length", "Head", "Tail", "Append", "Prepend",
            "Map", "Apply", "Select", "Sort", "Reverse",
            "StringJoin", "StringLength", "StringSplit",
            "Array", "Dot", "Transpose", "Maximum", "Minimum",
            "Sum", "Product", "Mean", "Variance",
        ];
        
        for func in known_functions {
            let similarity = self.calculate_similarity(symbol, func);
            if similarity > 0.4 && !suggestions.iter().any(|s| s.suggested_name == func) {
                suggestions.push(FunctionSuggestion {
                    suggested_name: func.to_string(),
                    confidence: similarity,
                    usage_example: self.get_function_example(func),
                    description: self.get_function_description(func),
                });
            }
        }
        
        // Sort by confidence and return top suggestions
        suggestions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        suggestions.truncate(3);
        
        suggestions
    }
    
    /// Calculate string similarity using Levenshtein distance
    fn calculate_similarity(&self, a: &str, b: &str) -> f32 {
        let a_lower = a.to_lowercase();
        let b_lower = b.to_lowercase();
        
        let max_len = a_lower.len().max(b_lower.len()) as f32;
        if max_len == 0.0 { return 1.0; }
        
        let distance = levenshtein_distance(&a_lower, &b_lower) as f32;
        1.0 - (distance / max_len)
    }
    
    /// Create position information with visual pointer
    fn create_position_info(&self, position: usize, source_code: &str) -> PositionInfo {
        let (line, column) = position_to_line_col(source_code, position);
        let lines: Vec<&str> = source_code.lines().collect();
        
        let context = if line > 0 && line <= lines.len() {
            lines[line - 1].to_string()
        } else {
            source_code.to_string()
        };
        
        let pointer = format!("{}^", " ".repeat(column.saturating_sub(1)));
        
        PositionInfo {
            line,
            column,
            position,
            context,
            pointer,
        }
    }
    
    /// Categorize parse error type
    fn categorize_parse_error(&self, message: &str, source_code: &str) -> ErrorCategory {
        if message.contains("bracket") || self.has_bracket_mismatch(source_code) {
            ErrorCategory::BracketMismatch
        } else if message.contains("Unexpected token") {
            ErrorCategory::ParseError
        } else {
            ErrorCategory::ParseError
        }
    }
    
    /// Check for bracket mismatches
    fn has_bracket_mismatch(&self, source_code: &str) -> bool {
        let mut stack = Vec::new();
        let mut in_string = false;
        
        for ch in source_code.chars() {
            match ch {
                '"' => in_string = !in_string,
                '[' if !in_string => stack.push('['),
                '(' if !in_string => stack.push('('),
                '{' if !in_string => stack.push('{'),
                ']' if !in_string => {
                    if stack.last() == Some(&'[') {
                        stack.pop();
                    } else {
                        return true; // Mismatch
                    }
                },
                ')' if !in_string => {
                    if stack.last() == Some(&'(') {
                        stack.pop();
                    } else {
                        return true; // Mismatch
                    }
                },
                '}' if !in_string => {
                    if stack.last() == Some(&'{') {
                        stack.pop();
                    } else {
                        return true; // Mismatch
                    }
                },
                _ => {}
            }
        }
        
        !stack.is_empty()
    }
    
    /// Create bracket error details
    fn create_bracket_error_details(&self, source_code: &str) -> (String, Vec<ErrorSuggestion>, Vec<CodeExample>, Vec<String>) {
        let analysis = self.analyze_brackets(source_code);
        
        let friendly_message = "There's a problem with brackets in your expression.".to_string();
        
        let suggestions = vec![
            ErrorSuggestion {
                title: analysis.suggestion.clone(),
                description: analysis.description.clone(),
                confidence: 0.9,
                fix_code: analysis.fix_code.clone(),
                why_this_helps: "Balanced brackets are required for valid syntax".to_string(),
            }
        ];
        
        let examples = vec![
            CodeExample {
                title: "Bracket Types".to_string(),
                correct_code: "f[x]     (* Function calls use square brackets *)\n{1,2,3}  (* Lists use curly braces *)\n(x + y)  (* Grouping uses parentheses *)".to_string(),
                explanation: "Each bracket type has a specific purpose".to_string(),
                common_mistake: Some("f(x)  (* Wrong bracket type for function calls *)".to_string()),
            }
        ];
        
        let recovery_steps = vec![
            "Count opening and closing brackets to ensure they match".to_string(),
            "Use square brackets [ ] for function calls".to_string(), 
            "Use curly braces { } for lists".to_string(),
            "Use parentheses ( ) for grouping expressions".to_string(),
        ];
        
        (friendly_message, suggestions, examples, recovery_steps)
    }
    
    /// Analyze bracket structure and suggest fixes
    fn analyze_brackets(&self, source_code: &str) -> BracketAnalysis {
        let mut square_count = 0;
        let mut paren_count = 0;
        let mut brace_count = 0;
        
        for ch in source_code.chars() {
            match ch {
                '[' => square_count += 1,
                ']' => square_count -= 1,
                '(' => paren_count += 1,
                ')' => paren_count -= 1,
                '{' => brace_count += 1,
                '}' => brace_count -= 1,
                _ => {}
            }
        }
        
        if square_count > 0 {
            BracketAnalysis {
                suggestion: format!("Add {} closing square bracket(s) ']'", square_count),
                description: "There are unmatched opening square brackets".to_string(),
                fix_code: Some(format!("{}{}", source_code, "]".repeat(square_count))),
            }
        } else if square_count < 0 {
            BracketAnalysis {
                suggestion: format!("Remove {} extra closing square bracket(s) ']'", square_count),
                description: "There are extra closing square brackets".to_string(),
                fix_code: None,
            }
        } else {
            BracketAnalysis {
                suggestion: "Check bracket balance".to_string(),
                description: "Bracket structure appears incorrect".to_string(),
                fix_code: None,
            }
        }
    }
    
    /// Create generic parse error details
    fn create_generic_parse_error_details(&self, message: &str, _source_code: &str) -> (String, Vec<ErrorSuggestion>, Vec<CodeExample>, Vec<String>) {
        let friendly_message = "There's a syntax error in your input.".to_string();
        
        let suggestions = vec![
            ErrorSuggestion {
                title: "Check syntax".to_string(),
                description: "Review the expression for syntax errors".to_string(),
                confidence: 0.7,
                fix_code: None,
                why_this_helps: "Correct syntax is required for parsing".to_string(),
            }
        ];
        
        let examples = vec![
            CodeExample {
                title: "Basic Syntax".to_string(),
                correct_code: "f[x] + g[y]  (* Function calls and operators *)".to_string(),
                explanation: "Follow Wolfram-inspired syntax rules".to_string(),
                common_mistake: None,
            }
        ];
        
        let recovery_steps = vec![
            "Check for missing operators between expressions".to_string(),
            "Verify all brackets and parentheses are properly matched".to_string(),
            "Make sure string literals are properly quoted".to_string(),
        ];
        
        (friendly_message, suggestions, examples, recovery_steps)
    }
    
    /// Suggest type conversion
    fn suggest_type_conversion(&self, expected: &str, actual: &str, source_code: &str) -> Option<String> {
        match (expected, actual) {
            ("Number", "String") => Some(format!("ToNumber[{}]", self.extract_value_from_source(source_code))),
            ("String", "Number") => Some(format!("ToString[{}]", self.extract_value_from_source(source_code))),
            ("List", _) => Some(format!("{{{}}}  (* Wrap in list *)", self.extract_value_from_source(source_code))),
            _ => None,
        }
    }
    
    /// Extract value from source code (simplified)
    fn extract_value_from_source(&self, source_code: &str) -> String {
        // This is a simplified extraction - in practice would need better parsing
        if let Some(start) = source_code.find('[') {
            if let Some(end) = source_code[start..].find(']') {
                return source_code[start+1..start+end].to_string();
            }
        }
        "value".to_string()
    }
    
    /// Enhance generic VM error
    fn enhance_generic_vm_error(&self, vm_error: &VmError, _source_code: &str) -> EnhancedError {
        EnhancedError {
            original_error: vm_error.to_string(),
            friendly_message: "A runtime error occurred during execution.".to_string(),
            error_category: ErrorCategory::RuntimeEvaluation,
            position_info: None,
            suggestions: vec![],
            did_you_mean: vec![],
            code_examples: vec![],
            recovery_steps: vec!["Check your input values and function calls".to_string()],
            related_docs: vec!["Runtime Error Guide".to_string()],
        }
    }
    
    /// Enhance generic error
    fn enhance_generic_error(&self, error: &LyraError, _source_code: &str) -> EnhancedError {
        EnhancedError {
            original_error: error.to_string(),
            friendly_message: "An error occurred.".to_string(),
            error_category: ErrorCategory::RuntimeEvaluation,
            position_info: None,
            suggestions: vec![],
            did_you_mean: vec![],
            code_examples: vec![],
            recovery_steps: vec![],
            related_docs: vec![],
        }
    }
    
    /// Get function usage example
    fn get_function_example(&self, func_name: &str) -> String {
        match func_name {
            "Sin" => "Sin[Pi/2]".to_string(),
            "Cos" => "Cos[0]".to_string(),
            "Length" => "Length[{1,2,3}]".to_string(),
            "Log" => "Log[E]".to_string(),
            "Sqrt" => "Sqrt[4]".to_string(),
            _ => format!("{}[...]", func_name),
        }
    }
    
    /// Get function description
    fn get_function_description(&self, func_name: &str) -> String {
        match func_name {
            "Sin" => "Trigonometric sine function".to_string(),
            "Cos" => "Trigonometric cosine function".to_string(),
            "Length" => "Returns the length of a list".to_string(),
            "Log" => "Natural logarithm function".to_string(),
            "Sqrt" => "Square root function".to_string(),
            _ => format!("Built-in {} function", func_name),
        }
    }
    
    /// Populate function typo database
    fn populate_function_database(&mut self) {
        let typos = vec![
            ("sin", vec!["Sin"]),
            ("sine", vec!["Sin"]),  
            ("cos", vec!["Cos"]),
            ("cosine", vec!["Cos"]),
            ("tan", vec!["Tan"]),
            ("tangent", vec!["Tan"]),
            ("log", vec!["Log"]),
            ("ln", vec!["Log"]),
            ("sqrt", vec!["Sqrt"]),
            ("sqroot", vec!["Sqrt"]),
            ("length", vec!["Length"]),
            ("len", vec!["Length"]),
            ("lenght", vec!["Length"]),
            ("stringjoin", vec!["StringJoin"]),
            ("join", vec!["StringJoin"]),
        ];
        
        for (typo, corrections) in typos {
            let suggestions = corrections.into_iter().map(|correct| {
                FunctionSuggestion {
                    suggested_name: correct.to_string(),
                    confidence: 0.9,
                    usage_example: self.get_function_example(correct),
                    description: self.get_function_description(correct),
                }
            }).collect();
            
            self.function_typos.insert(typo.to_string(), suggestions);
        }
    }
    
    /// Populate syntax patterns (placeholder)
    fn populate_syntax_patterns(&mut self) {
        // Implementation would include various syntax patterns
    }
    
    /// Populate function signatures (placeholder)
    fn populate_function_signatures(&mut self) {
        // Implementation would include function signature data
    }
    
    /// Populate domain patterns (placeholder)
    fn populate_domain_patterns(&mut self) {
        // Implementation would include mathematical domain restrictions
    }
    
    /// Enhance arity error with specific suggestions
    fn enhance_arity_error(&self, function_name: &str, expected: usize, actual: usize, source_code: &str) -> EnhancedError {
        let friendly_message = if actual < expected {
            format!("The function '{}' needs {} argument{}, but you only provided {}.", 
                   function_name, expected, if expected == 1 { "" } else { "s" }, actual)
        } else {
            format!("The function '{}' only takes {} argument{}, but you provided {}.", 
                   function_name, expected, if expected == 1 { "" } else { "s" }, actual)
        };
        
        let usage_example = self.get_function_example(function_name);
        
        let suggestions = vec![
            ErrorSuggestion {
                title: "Check function signature".to_string(),
                description: format!("{} expects exactly {} argument{}", 
                                   function_name, expected, if expected == 1 { "" } else { "s" }),
                confidence: 0.95,
                fix_code: Some(usage_example.clone()),
                why_this_helps: "Using the correct number of arguments will fix this error".to_string(),
            }
        ];
        
        let code_examples = vec![
            CodeExample {
                title: "Correct Usage".to_string(),
                correct_code: usage_example,
                explanation: format!("{} requires exactly {} parameter{}", 
                                   function_name, expected, if expected == 1 { "" } else { "s" }),
                common_mistake: Some(format!("Providing {} argument{} instead of {}", 
                                           actual, if actual == 1 { "" } else { "s" }, expected)),
            }
        ];
        
        EnhancedError {
            original_error: format!("Wrong number of arguments: {} expects {}, got {}", 
                                  function_name, expected, actual),
            friendly_message,
            error_category: ErrorCategory::ArgumentCount,
            position_info: None,
            suggestions,
            did_you_mean: vec![],
            code_examples,
            recovery_steps: vec![
                format!("Check the documentation for {} to see the expected arguments", function_name),
                format!("Provide exactly {} argument{}", expected, if expected == 1 { "" } else { "s" }),
                "Use '%help {}' to see the function signature".replace("{}", function_name),
            ],
            related_docs: vec![format!("{} Function Reference", function_name)],
        }
    }
    
    /// Enhance argument type error with conversion suggestions
    fn enhance_argument_type_error(&self, function_name: &str, param_index: usize, expected: &str, actual: &str, source_code: &str) -> EnhancedError {
        let param_display = if param_index == 0 { "1st".to_string() } 
                           else if param_index == 1 { "2nd".to_string() }
                           else if param_index == 2 { "3rd".to_string() }
                           else { format!("{}th", param_index + 1) };
        
        let friendly_message = format!("The {} parameter of '{}' should be {}, but you gave it {}.", 
                                     param_display, function_name, expected.to_lowercase(), actual.to_lowercase());
        
        let conversion_suggestion = self.suggest_type_conversion(expected, actual, source_code);
        
        let mut suggestions = vec![
            ErrorSuggestion {
                title: format!("Convert {} to {}", actual, expected),
                description: format!("Use a type conversion function to change {} to {}", actual, expected),
                confidence: 0.9,
                fix_code: conversion_suggestion,
                why_this_helps: format!("{} expects {} for parameter {}", function_name, expected, param_index + 1),
            }
        ];
        
        // Add specific suggestions based on the type mismatch
        match (expected, actual) {
            ("Number", "String") => {
                suggestions.push(ErrorSuggestion {
                    title: "Parse numeric string".to_string(),
                    description: "If your string contains a number, use ToNumber[] to convert it".to_string(),
                    confidence: 0.8,
                    fix_code: Some("ToNumber[\"123\"]".to_string()),
                    why_this_helps: "This converts string representations of numbers to actual numbers".to_string(),
                });
            },
            ("List", _) => {
                suggestions.push(ErrorSuggestion {
                    title: "Wrap in list".to_string(),
                    description: "Put your value inside curly braces to make it a list".to_string(),
                    confidence: 0.7,
                    fix_code: Some("{value}".to_string()),
                    why_this_helps: "Single values can be made into lists by wrapping them".to_string(),
                });
            },
            _ => {}
        }
        
        let code_examples = vec![
            CodeExample {
                title: "Type Conversion Examples".to_string(),
                correct_code: match expected {
                    "Number" => "ToNumber[\"42\"]      (* String to Number *)\nN[\"3.14\"]         (* String to Number *)".to_string(),
                    "String" => "ToString[42]        (* Number to String *)\nStringForm[expr]   (* Expression to String *)".to_string(),
                    "List" => "{item}             (* Single item to List *)\nList[item1, item2] (* Multiple items to List *)".to_string(),
                    _ => format!("Convert to {}", expected),
                },
                explanation: format!("Ways to convert {} to {}", actual, expected),
                common_mistake: Some(format!("Passing {} directly without conversion", actual)),
            }
        ];
        
        EnhancedError {
            original_error: format!("Argument type error in {}: parameter {} expects {}, got {}", 
                                  function_name, param_index, expected, actual),
            friendly_message,
            error_category: ErrorCategory::ArgumentType,
            position_info: None,
            suggestions,
            did_you_mean: vec![],
            code_examples,
            recovery_steps: vec![
                format!("Convert the {} parameter from {} to {}", param_display, actual, expected),
                format!("Check that parameter {} has the right type", param_index + 1),
                "Use type conversion functions like ToNumber[], ToString[], etc.".to_string(),
            ],
            related_docs: vec![
                format!("{} Function Reference", function_name),
                "Type Conversion Guide".to_string(),
            ],
        }
    }
}

/// Bracket analysis result
#[derive(Debug)]
struct BracketAnalysis {
    suggestion: String,
    description: String,
    fix_code: Option<String>,
}

/// Helper function for position calculation
fn position_to_line_col(source: &str, position: usize) -> (usize, usize) {
    let mut line = 1;
    let mut col = 1;

    for (i, ch) in source.chars().enumerate() {
        if i >= position {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }

    (line, col)
}

/// Calculate Levenshtein distance between two strings
fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();
    
    if a_len == 0 { return b_len; }
    if b_len == 0 { return a_len; }
    
    let mut matrix = vec![vec![0; b_len + 1]; a_len + 1];
    
    for i in 0..=a_len {
        matrix[i][0] = i;
    }
    for j in 0..=b_len {
        matrix[0][j] = j;
    }
    
    for i in 1..=a_len {
        for j in 1..=b_len {
            let cost = if a_chars[i-1] == b_chars[j-1] { 0 } else { 1 };
            matrix[i][j] = (matrix[i-1][j] + 1)
                .min(matrix[i][j-1] + 1)
                .min(matrix[i-1][j-1] + cost);
        }
    }
    
    matrix[a_len][b_len]
}

impl Default for ErrorEnhancer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_suggestions() {
        let enhancer = ErrorEnhancer::new();
        let suggestions = enhancer.find_function_suggestions("sine");
        assert!(!suggestions.is_empty());
        assert!(suggestions[0].suggested_name == "Sin");
    }
    
    #[test]
    fn test_bracket_analysis() {
        let enhancer = ErrorEnhancer::new();
        assert!(enhancer.has_bracket_mismatch("Sin[0"));
        assert!(!enhancer.has_bracket_mismatch("Sin[0]"));
    }
    
    #[test]
    fn test_similarity_calculation() {
        let enhancer = ErrorEnhancer::new();
        let similarity = enhancer.calculate_similarity("sine", "Sin");
        assert!(similarity > 0.5);
    }
}