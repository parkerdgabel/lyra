use thiserror::Error;
use std::fmt;

/// Unified error system for Lyra that consolidates all error types
/// and provides consistent error handling across the entire system.
/// 
/// This replaces the multiple error types scattered throughout the codebase
/// with a single, comprehensive error hierarchy that provides:
/// - Consistent error messaging
/// - Proper error chaining
/// - Context preservation
/// - Recovery suggestions
/// - Categorized error handling
#[derive(Error, Debug)]
pub enum LyraUnifiedError {
    // === Core System Errors ===
    #[error("Parse error at position {position}: {message}")]
    Parse { 
        message: String, 
        position: usize,
        source_context: Option<String>,
    },

    #[error("Lexer error at position {position}: {message}")]
    Lexer { 
        message: String, 
        position: usize,
        source_context: Option<String>,
    },

    #[error("Compilation error: {message}")]
    Compilation { 
        message: String,
        phase: CompilationPhase,
        suggestions: Vec<String>,
    },

    #[error("Runtime error: {message}")]
    Runtime { 
        message: String,
        context: RuntimeContext,
        recoverable: bool,
    },

    // === Type System Errors ===
    #[error("Type error: expected {expected}, got {actual}")]
    Type { 
        expected: String, 
        actual: String,
        location: Option<SourceLocation>,
        suggestions: Vec<String>,
    },

    #[error("Type inference failed: {message}")]
    TypeInference {
        message: String,
        expression: String,
        context: Vec<String>,
    },

    // === VM Execution Errors ===
    #[error("VM execution error: {kind:?}")]
    VmExecution {
        kind: VmErrorKind,
        instruction_pointer: Option<usize>,
        stack_trace: Vec<String>,
    },

    #[error("Stack operation error: {operation}")]
    Stack {
        operation: String,
        expected_size: Option<usize>,
        actual_size: usize,
    },

    // === Function and Method Errors ===
    #[error("Function call error: {function_name}")]
    FunctionCall {
        function_name: String,
        issue: FunctionCallIssue,
        available_signatures: Vec<String>,
    },

    #[error("Method call error: {type_name}::{method_name}")]
    MethodCall {
        type_name: String,
        method_name: String,
        issue: MethodCallIssue,
        available_methods: Vec<String>,
    },

    // === Symbol and Scope Errors ===
    #[error("Symbol error: {symbol_name}")]
    Symbol {
        symbol_name: String,
        issue: SymbolIssue,
        scope_context: Vec<String>,
    },

    // === I/O and Resource Errors ===
    #[error("I/O error: {operation}")]
    Io {
        operation: String,
        #[source]
        source: std::io::Error,
        path: Option<std::path::PathBuf>,
    },

    #[error("Resource error: {resource}")]
    Resource {
        resource: String,
        issue: ResourceIssue,
        cleanup_performed: bool,
    },

    // === Module and Import Errors ===
    #[error("Module error: {module_name}")]
    Module {
        module_name: String,
        issue: ModuleIssue,
        dependency_chain: Vec<String>,
    },

    // === Foreign Object Errors ===
    #[error("Foreign object error: {type_name}")]
    Foreign {
        type_name: String,
        operation: String,
        details: String,
    },

    // === Validation and Constraint Errors ===
    #[error("Validation error: {constraint}")]
    Validation {
        constraint: String,
        value: String,
        valid_range: Option<String>,
    },

    // === System and Configuration Errors ===
    #[error("Configuration error: {setting}")]
    Configuration {
        setting: String,
        issue: String,
        default_used: bool,
    },

    #[error("Internal system error: {component}")]
    Internal {
        component: String,
        details: String,
        bug_report_info: String,
    },

    // === Chained Error Context ===
    #[error("{message}")]
    WithContext {
        message: String,
        #[source]
        source: Box<LyraUnifiedError>,
        context_chain: Vec<String>,
    },
}

/// Compilation phase information for better error context
#[derive(Debug, Clone, PartialEq)]
pub enum CompilationPhase {
    Lexing,
    Parsing,
    TypeChecking,
    CodeGeneration,
    Optimization,
    Linking,
}

/// Runtime context for error reporting
#[derive(Debug, Clone)]
pub struct RuntimeContext {
    pub current_function: Option<String>,
    pub call_stack_depth: usize,
    pub local_variables: Vec<String>,
    pub evaluation_mode: String,
}

/// Source location information
#[derive(Debug, Clone, PartialEq)]
pub struct SourceLocation {
    pub line: usize,
    pub column: usize,
    pub file: Option<String>,
}

/// VM error categories
#[derive(Debug, Clone, PartialEq)]
pub enum VmErrorKind {
    StackUnderflow,
    StackOverflow,
    InvalidInstruction,
    InvalidOperand,
    DivisionByZero,
    IndexOutOfBounds { index: i64, length: usize },
    NullPointerAccess,
    MemoryExhausted,
    ExecutionTimeout,
    InfiniteLoop,
}

/// Function call issues
#[derive(Debug, Clone, PartialEq)]
pub enum FunctionCallIssue {
    UnknownFunction,
    ArityMismatch { expected: usize, actual: usize },
    TypeMismatch { parameter: String, expected: String, actual: String },
    AttributeViolation { attribute: String, reason: String },
    EvaluationError { reason: String },
}

/// Method call issues
#[derive(Debug, Clone, PartialEq)]
pub enum MethodCallIssue {
    UnknownMethod,
    ArityMismatch { expected: usize, actual: usize },
    TypeMismatch { parameter: String, expected: String, actual: String },
    ObjectStateInvalid { reason: String },
    AccessDenied { reason: String },
}

/// Symbol-related issues
#[derive(Debug, Clone, PartialEq)]
pub enum SymbolIssue {
    NotFound,
    NotDefined,
    AlreadyDefined,
    Protected,
    OutOfScope,
    TypeConflict { expected: String, actual: String },
    MutabilityViolation,
}

/// Resource-related issues
#[derive(Debug, Clone, PartialEq)]
pub enum ResourceIssue {
    NotFound,
    AccessDenied,
    Corrupted,
    InUse,
    Exhausted,
    VersionMismatch { expected: String, actual: String },
}

/// Module-related issues
#[derive(Debug, Clone, PartialEq)]
pub enum ModuleIssue {
    NotFound,
    LoadFailed { reason: String },
    DependencyError { dependency: String, reason: String },
    VersionConflict { required: String, available: String },
    CircularDependency,
    ExportNotFound { export: String },
}

impl LyraUnifiedError {
    /// Add context to an error
    pub fn with_context(self, context: &str) -> Self {
        match self {
            LyraUnifiedError::WithContext { message, source, mut context_chain } => {
                context_chain.insert(0, context.to_string());
                LyraUnifiedError::WithContext { message, source, context_chain }
            }
            other => LyraUnifiedError::WithContext {
                message: context.to_string(),
                source: Box::new(other),
                context_chain: vec![],
            }
        }
    }
    
    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            LyraUnifiedError::Runtime { recoverable, .. } => *recoverable,
            LyraUnifiedError::VmExecution { kind, .. } => {
                matches!(kind, 
                    VmErrorKind::IndexOutOfBounds { .. } |
                    VmErrorKind::DivisionByZero |
                    VmErrorKind::InvalidOperand
                )
            }
            LyraUnifiedError::FunctionCall { issue, .. } => {
                matches!(issue, 
                    FunctionCallIssue::ArityMismatch { .. } |
                    FunctionCallIssue::TypeMismatch { .. }
                )
            }
            LyraUnifiedError::Validation { .. } => true,
            LyraUnifiedError::WithContext { source, .. } => source.is_recoverable(),
            _ => false,
        }
    }
    
    /// Get error category for handling
    pub fn category(&self) -> ErrorCategory {
        match self {
            LyraUnifiedError::Parse { .. } | LyraUnifiedError::Lexer { .. } => ErrorCategory::Syntax,
            LyraUnifiedError::Compilation { .. } => ErrorCategory::Compilation,
            LyraUnifiedError::Type { .. } | LyraUnifiedError::TypeInference { .. } => ErrorCategory::Type,
            LyraUnifiedError::VmExecution { .. } | LyraUnifiedError::Stack { .. } => ErrorCategory::Runtime,
            LyraUnifiedError::FunctionCall { .. } | LyraUnifiedError::MethodCall { .. } => ErrorCategory::Call,
            LyraUnifiedError::Symbol { .. } => ErrorCategory::Symbol,
            LyraUnifiedError::Io { .. } | LyraUnifiedError::Resource { .. } => ErrorCategory::Resource,
            LyraUnifiedError::Module { .. } => ErrorCategory::Module,
            LyraUnifiedError::Foreign { .. } => ErrorCategory::Foreign,
            LyraUnifiedError::Validation { .. } => ErrorCategory::Validation,
            LyraUnifiedError::Configuration { .. } | LyraUnifiedError::Internal { .. } => ErrorCategory::System,
            LyraUnifiedError::Runtime { .. } => ErrorCategory::Runtime,
            LyraUnifiedError::WithContext { source, .. } => source.category(),
        }
    }
    
    /// Get recovery suggestions with enhanced "Did you mean?" functionality
    pub fn recovery_suggestions(&self) -> Vec<String> {
        match self {
            LyraUnifiedError::Parse { message, position, source_context } => {
                let mut suggestions = vec![
                    "Check for missing brackets [ ] or parentheses ( )".to_string(),
                    "Verify function call syntax: FunctionName[arg1, arg2]".to_string(),
                    "Make sure list syntax uses braces: {item1, item2}".to_string(),
                ];
                
                // Enhanced parse error suggestions based on context
                if let Some(context) = source_context {
                    if context.contains("(") && !context.contains("[") {
                        suggestions.insert(0, "Use square brackets [ ] for function calls, not parentheses ( )".to_string());
                    }
                    if context.chars().filter(|&c| c == '[').count() > context.chars().filter(|&c| c == ']').count() {
                        suggestions.insert(0, format!("Add {} closing bracket(s) ']'", 
                            context.chars().filter(|&c| c == '[').count() - context.chars().filter(|&c| c == ']').count()));
                    }
                }
                
                if message.contains("Unexpected token") {
                    suggestions.push("Check for missing operators (+, -, *, /) between expressions".to_string());
                }
                
                suggestions
            },
            
            LyraUnifiedError::Type { expected, actual, suggestions: type_suggestions, .. } => {
                let mut suggestions = type_suggestions.clone();
                
                // Enhanced type conversion suggestions
                match (expected.as_str(), actual.as_str()) {
                    ("Number" | "Integer" | "Real", "String") => {
                        suggestions.push("Convert string to number: ToNumber[\"123\"]".to_string());
                        suggestions.push("For numeric strings, check if they contain valid numbers".to_string());
                    },
                    ("String", "Number" | "Integer" | "Real") => {
                        suggestions.push("Convert number to string: ToString[42]".to_string());
                        suggestions.push("Use StringForm[expr] for complex expressions".to_string());
                    },
                    ("List", _) => {
                        suggestions.push("Wrap single value in list: {value}".to_string());
                        suggestions.push("Use List[item1, item2, ...] to create lists".to_string());
                    },
                    ("Boolean", _) => {
                        suggestions.push("Use comparison operators: x == y, x > y, etc.".to_string());
                        suggestions.push("Boolean values are True or False".to_string());
                    },
                    _ => {
                        suggestions.push(format!("Check if {} can be converted to {}", actual, expected));
                    }
                }
                
                suggestions
            },
            
            LyraUnifiedError::Compilation { suggestions: comp_suggestions, phase, .. } => {
                let mut suggestions = comp_suggestions.clone();
                
                match phase {
                    CompilationPhase::Parsing => {
                        suggestions.push("Check syntax against Lyra language reference".to_string());
                    },
                    CompilationPhase::TypeChecking => {
                        suggestions.push("Verify all variables are defined and types match".to_string());
                    },
                    _ => {}
                }
                
                suggestions
            },
            
            LyraUnifiedError::FunctionCall { function_name, issue, available_signatures } => {
                let mut suggestions = vec![];
                
                match issue {
                    FunctionCallIssue::UnknownFunction => {
                        suggestions.push("Check function name spelling and capitalization".to_string());
                        suggestions.extend(self.get_function_suggestions(function_name));
                        suggestions.push("Use '%functions' in REPL to see all available functions".to_string());
                    },
                    FunctionCallIssue::ArityMismatch { expected, actual } => {
                        suggestions.push(format!("Function '{}' expects {} argument{}, but got {}", 
                            function_name, expected, if *expected == 1 { "" } else { "s" }, actual));
                        if let Some(example) = self.get_function_example(function_name) {
                            suggestions.push(format!("Example usage: {}", example));
                        }
                    },
                    FunctionCallIssue::TypeMismatch { parameter, expected: param_expected, actual: param_actual } => {
                        suggestions.push(format!("Parameter '{}' expects {}, got {}", parameter, param_expected, param_actual));
                        suggestions.push(format!("Convert {} to {} before passing to function", param_actual, param_expected));
                    },
                    _ => {
                        suggestions.push("Check function documentation for proper usage".to_string());
                    }
                }
                
                if !available_signatures.is_empty() {
                    suggestions.push(format!("Available signatures: {}", available_signatures.join(", ")));
                }
                
                suggestions
            },
            
            LyraUnifiedError::MethodCall { type_name, method_name, issue, available_methods } => {
                let mut suggestions = vec![];
                
                match issue {
                    MethodCallIssue::UnknownMethod => {
                        suggestions.push(format!("Method '{}' not found for type '{}'", method_name, type_name));
                        suggestions.extend(self.get_method_suggestions(type_name, method_name));
                    },
                    MethodCallIssue::ArityMismatch { expected, actual } => {
                        suggestions.push(format!("Method '{}' expects {} argument{}, got {}", 
                            method_name, expected, if *expected == 1 { "" } else { "s" }, actual));
                    },
                    _ => {}
                }
                
                if !available_methods.is_empty() {
                    suggestions.push(format!("Available methods for {}: {}", type_name, available_methods.join(", ")));
                }
                
                suggestions
            },
            
            LyraUnifiedError::Symbol { symbol_name, issue, scope_context } => {
                let mut suggestions = vec![];
                
                match issue {
                    SymbolIssue::NotFound => {
                        suggestions.push(format!("Symbol '{}' is not defined", symbol_name));
                        suggestions.extend(self.get_symbol_suggestions(symbol_name));
                        suggestions.push("Check spelling and case sensitivity".to_string());
                        suggestions.push(format!("Define '{}' before using it: {} = value", symbol_name, symbol_name));
                    },
                    SymbolIssue::NotDefined => {
                        suggestions.push(format!("Initialize '{}' before using: {} = initialValue", symbol_name, symbol_name));
                    },
                    SymbolIssue::AlreadyDefined => {
                        suggestions.push(format!("Symbol '{}' is already defined in this scope", symbol_name));
                        suggestions.push("Use a different name or unset the existing symbol".to_string());
                    },
                    SymbolIssue::Protected => {
                        suggestions.push(format!("Symbol '{}' is protected and cannot be modified", symbol_name));
                        suggestions.push("Use a different symbol name".to_string());
                        suggestions.push("Protected symbols include built-in constants and functions".to_string());
                    },
                    SymbolIssue::OutOfScope => {
                        suggestions.push(format!("Symbol '{}' is not accessible in current scope", symbol_name));
                        if !scope_context.is_empty() {
                            suggestions.push(format!("Available in scope: {}", scope_context.join(", ")));
                        }
                    },
                    _ => {
                        suggestions.push("Check symbol usage and scope".to_string());
                    }
                }
                
                suggestions
            },
            
            LyraUnifiedError::VmExecution { kind, .. } => {
                match kind {
                    VmErrorKind::DivisionByZero => vec![
                        "Division by zero is undefined".to_string(),
                        "Check that the denominator is not zero before dividing".to_string(),
                        "Use conditional: If[divisor != 0, numerator/divisor, Missing]".to_string(),
                        "Consider using limits or alternative formulations".to_string(),
                    ],
                    VmErrorKind::IndexOutOfBounds { index, length } => vec![
                        format!("Index {} is out of bounds for list of length {}", index, length),
                        format!("Valid indices are 1 to {} (Lyra uses 1-based indexing)", length),
                        "Check list length with Length[list] before accessing elements".to_string(),
                        "Use bounds checking: If[index <= Length[list], list[[index]], Missing]".to_string(),
                    ],
                    VmErrorKind::StackOverflow => vec![
                        "Stack overflow - likely due to infinite recursion".to_string(),
                        "Check recursive functions for proper base cases".to_string(),
                        "Consider iterative alternatives to deep recursion".to_string(),
                    ],
                    VmErrorKind::StackUnderflow => vec![
                        "Internal error: stack underflow".to_string(),
                        "This may indicate a compiler bug".to_string(),
                    ],
                    VmErrorKind::InvalidInstruction => vec![
                        "Invalid instruction - this is likely a compiler bug".to_string(),
                        "Please report this error".to_string(),
                    ],
                    VmErrorKind::MemoryExhausted => vec![
                        "Out of memory".to_string(),
                        "Try processing data in smaller chunks".to_string(),
                        "Consider using streaming operations for large datasets".to_string(),
                    ],
                    VmErrorKind::ExecutionTimeout => vec![
                        "Execution timed out".to_string(),
                        "Operation took too long to complete".to_string(),
                        "Consider optimizing the algorithm or increasing timeout".to_string(),
                    ],
                    _ => vec!["Check the operation and try again".to_string()],
                }
            },
            
            LyraUnifiedError::Io { operation, path, .. } => {
                let mut suggestions = vec![];
                
                if operation.contains("read") || operation.contains("open") {
                    suggestions.push("Check that the file exists and you have read permissions".to_string());
                    if let Some(path_buf) = path {
                        suggestions.push(format!("Verify path: {}", path_buf.display()));
                        if let Some(parent) = path_buf.parent() {
                            suggestions.push(format!("Check that directory exists: {}", parent.display()));
                        }
                    }
                    suggestions.push("Use absolute paths to avoid confusion".to_string());
                }
                
                if operation.contains("write") {
                    suggestions.push("Check that you have write permissions".to_string());
                    suggestions.push("Ensure the parent directory exists".to_string());
                }
                
                suggestions
            },
            
            LyraUnifiedError::Validation { constraint, value, valid_range } => {
                let mut suggestions = vec![
                    format!("Value '{}' violates constraint: {}", value, constraint),
                ];
                
                if let Some(range) = valid_range {
                    suggestions.push(format!("Valid range: {}", range));
                }
                
                suggestions.push("Check input values and constraints".to_string());
                suggestions
            },
            
            LyraUnifiedError::WithContext { source, .. } => source.recovery_suggestions(),
            
            _ => vec!["Check the operation and try again".to_string()],
        }
    }
    
    /// Get detailed error information for debugging
    pub fn debug_info(&self) -> ErrorDebugInfo {
        ErrorDebugInfo {
            error_type: format!("{:?}", std::mem::discriminant(self)),
            is_recoverable: self.is_recoverable(),
            category: self.category(),
            suggestions: self.recovery_suggestions(),
            context_chain: self.get_context_chain(),
        }
    }
    
    /// Get the full context chain
    pub fn get_context_chain(&self) -> Vec<String> {
        match self {
            LyraUnifiedError::WithContext { context_chain, source, .. } => {
                let mut chain = context_chain.clone();
                chain.extend(source.get_context_chain());
                chain
            }
            _ => vec![],
        }
    }
    
    /// Get "Did you mean?" suggestions for unknown functions
    fn get_function_suggestions(&self, function_name: &str) -> Vec<String> {
        let known_functions = vec![
            // Mathematical functions
            "Sin", "Cos", "Tan", "ArcSin", "ArcCos", "ArcTan",
            "Sinh", "Cosh", "Tanh", "ArcSinh", "ArcCosh", "ArcTanh",
            "Exp", "Log", "Log10", "Log2", "Sqrt", "Power", "Abs", "Sign",
            "Floor", "Ceiling", "Round", "N", "Rationalize",
            "Max", "Min", "Clip", "Rescale",
            
            // List and data functions
            "Length", "Head", "Tail", "First", "Last", "Rest", "Most",
            "Append", "Prepend", "Join", "Flatten", "Reverse", "Sort",
            "Map", "Apply", "Select", "Cases", "Count", "Position",
            "Table", "Array", "Range", "ConstantArray",
            
            // String functions
            "StringJoin", "StringLength", "StringSplit", "StringReplace",
            "ToString", "ToExpression", "StringTake", "StringDrop",
            "StringMatchQ", "StringContainsQ",
            
            // Pattern and rule functions
            "Replace", "ReplaceAll", "ReplaceRepeated", "MatchQ",
            "Cases", "Position", "Count", "DeleteCases",
            
            // Type conversion
            "ToNumber", "ToString", "ToList", "ToInteger", "ToReal",
            
            // Control flow
            "If", "Which", "Switch", "Do", "For", "While",
            
            // Statistics
            "Mean", "Variance", "StandardDeviation", "Median", "Total",
            "Sum", "Product", "Maximum", "Minimum",
        ];
        
        let function_lower = function_name.to_lowercase();
        let mut suggestions = Vec::new();
        
        // Check for exact matches with different cases
        for func in &known_functions {
            if func.to_lowercase() == function_lower {
                suggestions.push(format!("Did you mean '{}'? (case-sensitive)", func));
                break;
            }
        }
        
        // Calculate similarity scores
        let mut scored_functions: Vec<(String, f32)> = known_functions
            .iter()
            .map(|func| {
                let similarity = calculate_string_similarity(function_name, func);
                (func.to_string(), similarity)
            })
            .filter(|(_, score)| *score > 0.4)
            .collect();
        
        // Sort by similarity score
        scored_functions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Add top suggestions
        for (func, score) in scored_functions.into_iter().take(3) {
            if score > 0.6 {
                suggestions.push(format!("Did you mean '{}'?", func));
            }
        }
        
        // Add common typos
        let typo_map = vec![
            ("sin", "Sin"), ("cos", "Cos"), ("tan", "Tan"),
            ("log", "Log"), ("sqrt", "Sqrt"), ("exp", "Exp"),
            ("length", "Length"), ("head", "Head"), ("tail", "Tail"),
            ("map", "Map"), ("apply", "Apply"), ("select", "Select"),
            ("stringjoin", "StringJoin"), ("join", "StringJoin"),
            ("tostring", "ToString"), ("tonumber", "ToNumber"),
        ];
        
        for (typo, correct) in typo_map {
            if function_name.to_lowercase() == typo {
                suggestions.insert(0, format!("Did you mean '{}'?", correct));
                break;
            }
        }
        
        suggestions
    }
    
    /// Get method suggestions for unknown methods
    fn get_method_suggestions(&self, type_name: &str, method_name: &str) -> Vec<String> {
        let method_map = vec![
            ("Series", vec!["head", "tail", "length", "mean", "sum", "sort", "reverse"]),
            ("Table", vec!["select", "filter", "join", "groupBy", "sort", "columns", "rows"]),
            ("Dataset", vec!["head", "describe", "select", "filter", "groupBy", "aggregate"]),
            ("Tensor", vec!["shape", "reshape", "transpose", "dot", "sum", "mean", "max", "min"]),
            ("String", vec!["length", "split", "join", "replace", "contains", "startsWith", "endsWith"]),
            ("List", vec!["length", "head", "tail", "append", "prepend", "reverse", "sort", "map"]),
        ];
        
        let mut suggestions = Vec::new();
        
        for (type_name_match, methods) in method_map {
            if type_name == type_name_match {
                for method in methods {
                    let similarity = calculate_string_similarity(method_name, method);
                    if similarity > 0.4 {
                        suggestions.push(format!("Did you mean '{}'?", method));
                    }
                }
                break;
            }
        }
        
        suggestions.sort();
        suggestions.dedup();
        suggestions
    }
    
    /// Get symbol suggestions for undefined symbols
    fn get_symbol_suggestions(&self, symbol_name: &str) -> Vec<String> {
        let common_constants = vec![
            "Pi", "E", "I", "Infinity", "True", "False", "Missing", "Null",
            "EulerGamma", "GoldenRatio", "Catalan",
        ];
        
        let common_variables = vec![
            "x", "y", "z", "t", "n", "i", "j", "k",
            "data", "list", "result", "value", "input", "output",
        ];
        
        let mut suggestions = Vec::new();
        
        // Check constants first
        for constant in common_constants {
            let similarity = calculate_string_similarity(symbol_name, constant);
            if similarity > 0.5 {
                suggestions.push(format!("Did you mean constant '{}'?", constant));
            }
        }
        
        // Check common variable names
        for var in common_variables {
            let similarity = calculate_string_similarity(symbol_name, var);
            if similarity > 0.6 {
                suggestions.push(format!("Did you mean variable '{}'?", var));
            }
        }
        
        suggestions
    }
    
    /// Get function usage example
    fn get_function_example(&self, function_name: &str) -> Option<String> {
        let examples = vec![
            ("Sin", "Sin[Pi/2]"),
            ("Cos", "Cos[0]"),
            ("Log", "Log[E]"),
            ("Sqrt", "Sqrt[4]"),
            ("Length", "Length[{1, 2, 3}]"),
            ("Head", "Head[{a, b, c}]"),
            ("Map", "Map[f, {1, 2, 3}]"),
            ("StringJoin", "StringJoin[\"Hello\", \" \", \"World\"]"),
            ("ToNumber", "ToNumber[\"123\"]"),
            ("ToString", "ToString[42]"),
            ("If", "If[x > 0, \"positive\", \"non-positive\"]"),
            ("Sum", "Sum[i, {i, 1, 10}]"),
            ("Mean", "Mean[{1, 2, 3, 4, 5}]"),
        ];
        
        for (name, example) in examples {
            if name == function_name {
                return Some(example.to_string());
            }
        }
        
        None
    }
}

/// Error categories for handling
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorCategory {
    Syntax,
    Compilation,
    Type,
    Runtime,
    Call,
    Symbol,
    Resource,
    Module,
    Foreign,
    Validation,
    System,
}

/// Debug information for errors
#[derive(Debug, Clone)]
pub struct ErrorDebugInfo {
    pub error_type: String,
    pub is_recoverable: bool,
    pub category: ErrorCategory,
    pub suggestions: Vec<String>,
    pub context_chain: Vec<String>,
}

impl fmt::Display for CompilationPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompilationPhase::Lexing => write!(f, "lexing"),
            CompilationPhase::Parsing => write!(f, "parsing"),
            CompilationPhase::TypeChecking => write!(f, "type checking"),
            CompilationPhase::CodeGeneration => write!(f, "code generation"),
            CompilationPhase::Optimization => write!(f, "optimization"),
            CompilationPhase::Linking => write!(f, "linking"),
        }
    }
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorCategory::Syntax => write!(f, "syntax"),
            ErrorCategory::Compilation => write!(f, "compilation"),
            ErrorCategory::Type => write!(f, "type"),
            ErrorCategory::Runtime => write!(f, "runtime"),
            ErrorCategory::Call => write!(f, "call"),
            ErrorCategory::Symbol => write!(f, "symbol"),
            ErrorCategory::Resource => write!(f, "resource"),
            ErrorCategory::Module => write!(f, "module"),
            ErrorCategory::Foreign => write!(f, "foreign"),
            ErrorCategory::Validation => write!(f, "validation"),
            ErrorCategory::System => write!(f, "system"),
        }
    }
}

// Type aliases for backward compatibility
pub type LyraResult<T> = Result<T, LyraUnifiedError>;


// Implement conversions from existing error types
impl From<std::io::Error> for LyraUnifiedError {
    fn from(err: std::io::Error) -> Self {
        LyraUnifiedError::Io {
            operation: "unknown".to_string(),
            source: err,
            path: None,
        }
    }
}

impl From<crate::vm::VmError> for LyraUnifiedError {
    fn from(err: crate::vm::VmError) -> Self {
        match err {
            crate::vm::VmError::StackUnderflow => LyraUnifiedError::VmExecution {
                kind: VmErrorKind::StackUnderflow,
                instruction_pointer: None,
                stack_trace: vec![],
            },
            crate::vm::VmError::DivisionByZero => LyraUnifiedError::VmExecution {
                kind: VmErrorKind::DivisionByZero,
                instruction_pointer: None,
                stack_trace: vec![],
            },
            crate::vm::VmError::IndexError { index, length } => LyraUnifiedError::VmExecution {
                kind: VmErrorKind::IndexOutOfBounds { index, length },
                instruction_pointer: None,
                stack_trace: vec![],
            },
            crate::vm::VmError::TypeError { expected, actual } => LyraUnifiedError::Type {
                expected,
                actual,
                location: None,
                suggestions: vec![],
            },
            other => LyraUnifiedError::Runtime {
                message: other.to_string(),
                context: RuntimeContext {
                    current_function: None,
                    call_stack_depth: 0,
                    local_variables: vec![],
                    evaluation_mode: "normal".to_string(),
                },
                recoverable: true,
            },
        }
    }
}

impl From<crate::compiler::CompilerError> for LyraUnifiedError {
    fn from(err: crate::compiler::CompilerError) -> Self {
        match err {
            crate::compiler::CompilerError::UnknownFunction(name) => LyraUnifiedError::FunctionCall {
                function_name: name,
                issue: FunctionCallIssue::UnknownFunction,
                available_signatures: vec![],
            },
            crate::compiler::CompilerError::InvalidArity { function, expected, actual } => {
                LyraUnifiedError::FunctionCall {
                    function_name: function,
                    issue: FunctionCallIssue::ArityMismatch { expected, actual },
                    available_signatures: vec![],
                }
            },
            other => LyraUnifiedError::Compilation {
                message: other.to_string(),
                phase: CompilationPhase::CodeGeneration,
                suggestions: vec![],
            },
        }
    }
}

impl From<crate::foreign::ForeignError> for LyraUnifiedError {
    fn from(err: crate::foreign::ForeignError) -> Self {
        match err {
            crate::foreign::ForeignError::UnknownMethod { type_name, method } => {
                LyraUnifiedError::MethodCall {
                    type_name,
                    method_name: method,
                    issue: MethodCallIssue::UnknownMethod,
                    available_methods: vec![],
                }
            },
            crate::foreign::ForeignError::InvalidArity { method, expected, actual } => {
                LyraUnifiedError::MethodCall {
                    type_name: "unknown".to_string(),
                    method_name: method,
                    issue: MethodCallIssue::ArityMismatch { expected, actual },
                    available_methods: vec![],
                }
            },
            other => LyraUnifiedError::Foreign {
                type_name: "unknown".to_string(),
                operation: "unknown".to_string(),
                details: other.to_string(),
            },
        }
    }
}

/// Calculate string similarity using Levenshtein distance
fn calculate_string_similarity(a: &str, b: &str) -> f32 {
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();
    
    let max_len = a_lower.len().max(b_lower.len()) as f32;
    if max_len == 0.0 { return 1.0; }
    
    let distance = levenshtein_distance(&a_lower, &b_lower) as f32;
    1.0 - (distance / max_len)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = LyraUnifiedError::Type {
            expected: "Integer".to_string(),
            actual: "String".to_string(),
            location: Some(SourceLocation { line: 1, column: 5, file: None }),
            suggestions: vec!["Convert to integer".to_string()],
        };
        
        assert_eq!(error.category(), ErrorCategory::Type);
        assert!(!error.is_recoverable());
        assert!(!error.recovery_suggestions().is_empty());
    }
    
    #[test]
    fn test_error_context() {
        let base_error = LyraUnifiedError::VmExecution {
            kind: VmErrorKind::DivisionByZero,
            instruction_pointer: Some(42),
            stack_trace: vec!["function1".to_string()],
        };
        
        let contextual_error = base_error
            .with_context("while evaluating expression")
            .with_context("in function main");
        
        let context_chain = contextual_error.get_context_chain();
        assert_eq!(context_chain, vec!["in function main", "while evaluating expression"]);
    }
    
    #[test]
    fn test_error_conversions() {
        let vm_error = crate::vm::VmError::DivisionByZero;
        let unified_error: LyraUnifiedError = vm_error.into();
        
        assert_eq!(unified_error.category(), ErrorCategory::Runtime);
        assert!(unified_error.is_recoverable());
    }
    
    #[test]
    fn test_error_debug_info() {
        let error = LyraUnifiedError::Symbol {
            symbol_name: "undefinedVar".to_string(),
            issue: SymbolIssue::NotFound,
            scope_context: vec!["global".to_string()],
        };
        
        let debug_info = error.debug_info();
        assert_eq!(debug_info.category, ErrorCategory::Symbol);
        assert!(!debug_info.suggestions.is_empty());
    }
}