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
    
    /// Get recovery suggestions
    pub fn recovery_suggestions(&self) -> Vec<String> {
        match self {
            LyraUnifiedError::Parse { .. } => vec![
                "Check for missing brackets [ ] or parentheses ( )".to_string(),
                "Verify function call syntax: FunctionName[arg1, arg2]".to_string(),
                "Make sure list syntax uses braces: {item1, item2}".to_string(),
            ],
            LyraUnifiedError::Type { suggestions, .. } => suggestions.clone(),
            LyraUnifiedError::Compilation { suggestions, .. } => suggestions.clone(),
            LyraUnifiedError::FunctionCall { available_signatures, .. } => {
                let mut suggestions = vec!["Check function name spelling".to_string()];
                if !available_signatures.is_empty() {
                    suggestions.push(format!("Available signatures: {}", available_signatures.join(", ")));
                }
                suggestions
            }
            LyraUnifiedError::MethodCall { available_methods, .. } => {
                let mut suggestions = vec!["Check method name spelling".to_string()];
                if !available_methods.is_empty() {
                    suggestions.push(format!("Available methods: {}", available_methods.join(", ")));
                }
                suggestions
            }
            LyraUnifiedError::Symbol { symbol_name, issue, .. } => {
                match issue {
                    SymbolIssue::NotFound => vec![
                        format!("Define the symbol '{}' before using it", symbol_name),
                        "Check spelling and case sensitivity".to_string(),
                    ],
                    SymbolIssue::Protected => vec![
                        format!("Symbol '{}' is protected and cannot be modified", symbol_name),
                        "Use a different symbol name".to_string(),
                    ],
                    _ => vec!["Check symbol usage and scope".to_string()],
                }
            }
            LyraUnifiedError::WithContext { source, .. } => source.recovery_suggestions(),
            _ => vec![],
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

/// Convert from existing error types to unified error
macro_rules! impl_from_error {
    ($error_type:ty, $variant:ident, $field:expr) => {
        impl From<$error_type> for LyraUnifiedError {
            fn from(err: $error_type) -> Self {
                LyraUnifiedError::$variant { $field: err.to_string() }
            }
        }
    };
}

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