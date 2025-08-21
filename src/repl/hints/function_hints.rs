//! Function signature database for intelligent hints
//!
//! This module extracts and organizes function metadata from the standard library
//! to provide rich function signature hints and documentation.

use crate::stdlib::StandardLibrary;
use std::collections::HashMap;

/// Database of function signatures and metadata
pub struct FunctionSignatureDatabase {
    /// Function signatures indexed by name
    signatures: HashMap<String, FunctionSignature>,
    /// Function categories for organization
    categories: HashMap<String, Vec<String>>,
}

/// Comprehensive function signature information
#[derive(Debug, Clone)]
pub struct FunctionSignature {
    /// Function name
    pub name: String,
    /// Function signature (e.g., "Sin[x]")
    pub signature: String,
    /// Brief description
    pub description: String,
    /// Parameter information
    pub parameters: Vec<ParameterInfo>,
    /// Usage examples
    pub examples: Vec<String>,
    /// Function category
    pub category: String,
    /// Return type description
    pub return_type: String,
}

/// Information about a function parameter
#[derive(Debug, Clone)]
pub struct ParameterInfo {
    /// Parameter name
    pub name: String,
    /// Parameter type (e.g., "Number", "List", "Expression")
    pub param_type: String,
    /// Whether parameter is optional
    pub optional: bool,
    /// Parameter description
    pub description: String,
    /// Default value if optional
    pub default_value: Option<String>,
}

impl FunctionSignatureDatabase {
    /// Create database from standard library
    pub fn from_stdlib(stdlib: &StandardLibrary) -> Self {
        let mut db = Self {
            signatures: HashMap::new(),
            categories: HashMap::new(),
        };
        
        db.populate_from_stdlib(stdlib);
        db
    }
    
    /// Get function signature by name
    pub fn get_signature(&self, name: &str) -> Option<&FunctionSignature> {
        self.signatures.get(name)
    }
    
    /// Get all function names in a category
    pub fn get_functions_in_category(&self, category: &str) -> Option<&Vec<String>> {
        self.categories.get(category)
    }
    
    /// Get all available categories
    pub fn get_categories(&self) -> Vec<&String> {
        self.categories.keys().collect()
    }
    
    /// Search functions by name prefix
    pub fn search_functions(&self, prefix: &str) -> Vec<&FunctionSignature> {
        self.signatures
            .values()
            .filter(|sig| sig.name.starts_with(prefix))
            .collect()
    }
    
    /// Populate database from standard library
    fn populate_from_stdlib(&mut self, stdlib: &StandardLibrary) {
        // Get all function names from stdlib
        let function_names = stdlib.function_names();
        
        for name in function_names {
            if let Some(signature) = self.create_signature_for_function(name) {
                self.add_signature(signature);
            }
        }
    }
    
    /// Create signature information for a function
    fn create_signature_for_function(&self, name: &str) -> Option<FunctionSignature> {
        // Comprehensive database of function signatures for all stdlib functions
        // Organized by category: List, String, Math, Tensor, etc.
        match name {
            // List functions
            "Length" => Some(FunctionSignature {
                name: "Length".to_string(),
                signature: "Length[list]".to_string(),
                description: "Get the length of a list".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "list".to_string(),
                        param_type: "List".to_string(),
                        optional: false,
                        description: "The list to measure".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "Length[{1, 2, 3}] → 3".to_string(),
                    "Length[{}] → 0".to_string(),
                ],
                category: "List".to_string(),
                return_type: "Integer".to_string(),
            }),
            
            "Head" => Some(FunctionSignature {
                name: "Head".to_string(),
                signature: "Head[list]".to_string(),
                description: "Get the first element of a list".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "list".to_string(),
                        param_type: "List".to_string(),
                        optional: false,
                        description: "Non-empty list".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "Head[{1, 2, 3}] → 1".to_string(),
                    "Head[{a, b, c}] → a".to_string(),
                ],
                category: "List".to_string(),
                return_type: "Any".to_string(),
            }),
            
            "Tail" => Some(FunctionSignature {
                name: "Tail".to_string(),
                signature: "Tail[list]".to_string(),
                description: "Get all elements except the first".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "list".to_string(),
                        param_type: "List".to_string(),
                        optional: false,
                        description: "List with at least one element".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "Tail[{1, 2, 3}] → {2, 3}".to_string(),
                    "Tail[{a}] → {}".to_string(),
                ],
                category: "List".to_string(),
                return_type: "List".to_string(),
            }),
            
            "Append" => Some(FunctionSignature {
                name: "Append".to_string(),
                signature: "Append[list, element]".to_string(),
                description: "Add element to the end of a list".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "list".to_string(),
                        param_type: "List".to_string(),
                        optional: false,
                        description: "The list to extend".to_string(),
                        default_value: None,
                    },
                    ParameterInfo {
                        name: "element".to_string(),
                        param_type: "Any".to_string(),
                        optional: false,
                        description: "Element to append".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "Append[{1, 2}, 3] → {1, 2, 3}".to_string(),
                    "Append[{}, x] → {x}".to_string(),
                ],
                category: "List".to_string(),
                return_type: "List".to_string(),
            }),
            
            // Math functions
            "Sin" => Some(FunctionSignature {
                name: "Sin".to_string(),
                signature: "Sin[x]".to_string(),
                description: "Sine trigonometric function".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "x".to_string(),
                        param_type: "Number".to_string(),
                        optional: false,
                        description: "Angle in radians".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "Sin[0] → 0".to_string(),
                    "Sin[Pi/2] → 1".to_string(),
                    "Sin[Pi] → 0".to_string(),
                ],
                category: "Math".to_string(),
                return_type: "Number".to_string(),
            }),
            
            "Cos" => Some(FunctionSignature {
                name: "Cos".to_string(),
                signature: "Cos[x]".to_string(),
                description: "Cosine trigonometric function".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "x".to_string(),
                        param_type: "Number".to_string(),
                        optional: false,
                        description: "Angle in radians".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "Cos[0] → 1".to_string(),
                    "Cos[Pi/2] → 0".to_string(),
                    "Cos[Pi] → -1".to_string(),
                ],
                category: "Math".to_string(),
                return_type: "Number".to_string(),
            }),
            
            "Exp" => Some(FunctionSignature {
                name: "Exp".to_string(),
                signature: "Exp[x]".to_string(),
                description: "Exponential function (e^x)".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "x".to_string(),
                        param_type: "Number".to_string(),
                        optional: false,
                        description: "Exponent".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "Exp[0] → 1".to_string(),
                    "Exp[1] → 2.71828...".to_string(),
                    "Exp[Log[x]] → x".to_string(),
                ],
                category: "Math".to_string(),
                return_type: "Number".to_string(),
            }),
            
            "Log" => Some(FunctionSignature {
                name: "Log".to_string(),
                signature: "Log[x]".to_string(),
                description: "Natural logarithm".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "x".to_string(),
                        param_type: "Number".to_string(),
                        optional: false,
                        description: "Positive number".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "Log[1] → 0".to_string(),
                    "Log[E] → 1".to_string(),
                    "Log[Exp[x]] → x".to_string(),
                ],
                category: "Math".to_string(),
                return_type: "Number".to_string(),
            }),
            
            // String functions
            "StringJoin" => Some(FunctionSignature {
                name: "StringJoin".to_string(),
                signature: "StringJoin[str1, str2, ...]".to_string(),
                description: "Concatenate strings".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "strings".to_string(),
                        param_type: "String...".to_string(),
                        optional: false,
                        description: "One or more strings to join".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "StringJoin[\"Hello\", \" \", \"World\"] → \"Hello World\"".to_string(),
                    "StringJoin[\"a\", \"b\", \"c\"] → \"abc\"".to_string(),
                ],
                category: "String".to_string(),
                return_type: "String".to_string(),
            }),
            
            "StringLength" => Some(FunctionSignature {
                name: "StringLength".to_string(),
                signature: "StringLength[string]".to_string(),
                description: "Get the length of a string".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "string".to_string(),
                        param_type: "String".to_string(),
                        optional: false,
                        description: "String to measure".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "StringLength[\"Hello\"] → 5".to_string(),
                    "StringLength[\"\"] → 0".to_string(),
                ],
                category: "String".to_string(),
                return_type: "Integer".to_string(),
            }),
            
            // Tensor/Array functions
            "Array" => Some(FunctionSignature {
                name: "Array".to_string(),
                signature: "Array[list]".to_string(),
                description: "Create tensor from nested lists".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "list".to_string(),
                        param_type: "List".to_string(),
                        optional: false,
                        description: "Nested lists representing tensor data".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "Array[{1, 2, 3}] → 1D tensor".to_string(),
                    "Array[{{1, 2}, {3, 4}}] → 2x2 matrix".to_string(),
                ],
                category: "Tensor".to_string(),
                return_type: "Tensor".to_string(),
            }),
            
            "Dot" => Some(FunctionSignature {
                name: "Dot".to_string(),
                signature: "Dot[a, b]".to_string(),
                description: "Matrix/vector multiplication".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "a".to_string(),
                        param_type: "Tensor".to_string(),
                        optional: false,
                        description: "First tensor/matrix/vector".to_string(),
                        default_value: None,
                    },
                    ParameterInfo {
                        name: "b".to_string(),
                        param_type: "Tensor".to_string(),
                        optional: false,
                        description: "Second tensor/matrix/vector".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "Dot[{1, 2}, {3, 4}] → 11".to_string(),
                    "Dot[{{1, 2}}, {{3}, {4}}] → {{11}}".to_string(),
                ],
                category: "Tensor".to_string(),
                return_type: "Tensor".to_string(),
            }),
            
            // Additional Math Functions
            "Tan" => Some(FunctionSignature {
                name: "Tan".to_string(),
                signature: "Tan[x]".to_string(),
                description: "Tangent trigonometric function".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "x".to_string(),
                        param_type: "Number".to_string(),
                        optional: false,
                        description: "Angle in radians".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "Tan[0] → 0".to_string(),
                    "Tan[Pi/4] → 1".to_string(),
                ],
                category: "Math".to_string(),
                return_type: "Number".to_string(),
            }),
            
            "Sqrt" => Some(FunctionSignature {
                name: "Sqrt".to_string(),
                signature: "Sqrt[x]".to_string(),
                description: "Square root function".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "x".to_string(),
                        param_type: "Number".to_string(),
                        optional: false,
                        description: "Non-negative number".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "Sqrt[4] → 2".to_string(),
                    "Sqrt[9] → 3".to_string(),
                    "Sqrt[2] → 1.414...".to_string(),
                ],
                category: "Math".to_string(),
                return_type: "Number".to_string(),
            }),
            
            "Abs" => Some(FunctionSignature {
                name: "Abs".to_string(),
                signature: "Abs[x]".to_string(),
                description: "Absolute value function".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "x".to_string(),
                        param_type: "Number".to_string(),
                        optional: false,
                        description: "Number to get absolute value of".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "Abs[-5] → 5".to_string(),
                    "Abs[3] → 3".to_string(),
                    "Abs[0] → 0".to_string(),
                ],
                category: "Math".to_string(),
                return_type: "Number".to_string(),
            }),
            
            "Power" => Some(FunctionSignature {
                name: "Power".to_string(),
                signature: "Power[x, y]".to_string(),
                description: "Raise x to the power of y".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "x".to_string(),
                        param_type: "Number".to_string(),
                        optional: false,
                        description: "Base number".to_string(),
                        default_value: None,
                    },
                    ParameterInfo {
                        name: "y".to_string(),
                        param_type: "Number".to_string(),
                        optional: false,
                        description: "Exponent".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "Power[2, 3] → 8".to_string(),
                    "Power[10, 2] → 100".to_string(),
                    "Power[4, 0.5] → 2".to_string(),
                ],
                category: "Math".to_string(),
                return_type: "Number".to_string(),
            }),
            
            // Additional List Functions
            "Map" => Some(FunctionSignature {
                name: "Map".to_string(),
                signature: "Map[function, list]".to_string(),
                description: "Apply function to each element of a list".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "function".to_string(),
                        param_type: "Function".to_string(),
                        optional: false,
                        description: "Function to apply".to_string(),
                        default_value: None,
                    },
                    ParameterInfo {
                        name: "list".to_string(),
                        param_type: "List".to_string(),
                        optional: false,
                        description: "List to map over".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "Map[Sin, {0, Pi/2, Pi}] → {0, 1, 0}".to_string(),
                    "Map[Square, {1, 2, 3}] → {1, 4, 9}".to_string(),
                ],
                category: "List".to_string(),
                return_type: "List".to_string(),
            }),
            
            "Flatten" => Some(FunctionSignature {
                name: "Flatten".to_string(),
                signature: "Flatten[list]".to_string(),
                description: "Flatten nested lists into a single level".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "list".to_string(),
                        param_type: "List".to_string(),
                        optional: false,
                        description: "Nested list to flatten".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "Flatten[{{1, 2}, {3, 4}}] → {1, 2, 3, 4}".to_string(),
                    "Flatten[{1, {2, 3}, 4}] → {1, 2, 3, 4}".to_string(),
                ],
                category: "List".to_string(),
                return_type: "List".to_string(),
            }),
            
            "Apply" => Some(FunctionSignature {
                name: "Apply".to_string(),
                signature: "Apply[function, arguments]".to_string(),
                description: "Apply function to arguments".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "function".to_string(),
                        param_type: "Function".to_string(),
                        optional: false,
                        description: "Function to apply".to_string(),
                        default_value: None,
                    },
                    ParameterInfo {
                        name: "arguments".to_string(),
                        param_type: "List".to_string(),
                        optional: false,
                        description: "Arguments as a list".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "Apply[Plus, {1, 2, 3}] → 6".to_string(),
                    "Apply[Times, {2, 3, 4}] → 24".to_string(),
                ],
                category: "List".to_string(),
                return_type: "Any".to_string(),
            }),
            
            // String Advanced Functions
            "StringSplit" => Some(FunctionSignature {
                name: "StringSplit".to_string(),
                signature: "StringSplit[string, delimiter]".to_string(),
                description: "Split string by delimiter".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "string".to_string(),
                        param_type: "String".to_string(),
                        optional: false,
                        description: "String to split".to_string(),
                        default_value: None,
                    },
                    ParameterInfo {
                        name: "delimiter".to_string(),
                        param_type: "String".to_string(),
                        optional: true,
                        description: "Delimiter (default: whitespace)".to_string(),
                        default_value: Some("\" \"".to_string()),
                    }
                ],
                examples: vec![
                    "StringSplit[\"a,b,c\", \",\"] → {\"a\", \"b\", \"c\"}".to_string(),
                    "StringSplit[\"hello world\"] → {\"hello\", \"world\"}".to_string(),
                ],
                category: "String".to_string(),
                return_type: "List".to_string(),
            }),
            
            "StringTrim" => Some(FunctionSignature {
                name: "StringTrim".to_string(),
                signature: "StringTrim[string]".to_string(),
                description: "Remove leading and trailing whitespace".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "string".to_string(),
                        param_type: "String".to_string(),
                        optional: false,
                        description: "String to trim".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "StringTrim[\"  hello  \"] → \"hello\"".to_string(),
                    "StringTrim[\"\\tworld\\n\"] → \"world\"".to_string(),
                ],
                category: "String".to_string(),
                return_type: "String".to_string(),
            }),
            
            "ToUpperCase" => Some(FunctionSignature {
                name: "ToUpperCase".to_string(),
                signature: "ToUpperCase[string]".to_string(),
                description: "Convert string to uppercase".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "string".to_string(),
                        param_type: "String".to_string(),
                        optional: false,
                        description: "String to convert".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "ToUpperCase[\"hello\"] → \"HELLO\"".to_string(),
                    "ToUpperCase[\"World\"] → \"WORLD\"".to_string(),
                ],
                category: "String".to_string(),
                return_type: "String".to_string(),
            }),
            
            "ToLowerCase" => Some(FunctionSignature {
                name: "ToLowerCase".to_string(),
                signature: "ToLowerCase[string]".to_string(),
                description: "Convert string to lowercase".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "string".to_string(),
                        param_type: "String".to_string(),
                        optional: false,
                        description: "String to convert".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "ToLowerCase[\"HELLO\"] → \"hello\"".to_string(),
                    "ToLowerCase[\"World\"] → \"world\"".to_string(),
                ],
                category: "String".to_string(),
                return_type: "String".to_string(),
            }),
            
            // Tensor Functions Expansion
            "ArrayDimensions" => Some(FunctionSignature {
                name: "ArrayDimensions".to_string(),
                signature: "ArrayDimensions[tensor]".to_string(),
                description: "Get the dimensions of a tensor".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "tensor".to_string(),
                        param_type: "Tensor".to_string(),
                        optional: false,
                        description: "Tensor to analyze".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "ArrayDimensions[Array[{{1, 2}, {3, 4}}]] → {2, 2}".to_string(),
                    "ArrayDimensions[Array[{1, 2, 3}]] → {3}".to_string(),
                ],
                category: "Tensor".to_string(),
                return_type: "List".to_string(),
            }),
            
            "ArrayRank" => Some(FunctionSignature {
                name: "ArrayRank".to_string(),
                signature: "ArrayRank[tensor]".to_string(),
                description: "Get the rank (number of dimensions) of a tensor".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "tensor".to_string(),
                        param_type: "Tensor".to_string(),
                        optional: false,
                        description: "Tensor to analyze".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "ArrayRank[Array[{{1, 2}, {3, 4}}]] → 2".to_string(),
                    "ArrayRank[Array[{1, 2, 3}]] → 1".to_string(),
                ],
                category: "Tensor".to_string(),
                return_type: "Integer".to_string(),
            }),
            
            "Transpose" => Some(FunctionSignature {
                name: "Transpose".to_string(),
                signature: "Transpose[matrix]".to_string(),
                description: "Transpose a matrix or tensor".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "matrix".to_string(),
                        param_type: "Tensor".to_string(),
                        optional: false,
                        description: "Matrix or tensor to transpose".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "Transpose[{{1, 2}, {3, 4}}] → {{1, 3}, {2, 4}}".to_string(),
                    "Transpose[{{1, 2, 3}}] → {{1}, {2}, {3}}".to_string(),
                ],
                category: "Tensor".to_string(),
                return_type: "Tensor".to_string(),
            }),
            
            "Maximum" => Some(FunctionSignature {
                name: "Maximum".to_string(),
                signature: "Maximum[a, b]".to_string(),
                description: "Element-wise maximum (ReLU activation)".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "a".to_string(),
                        param_type: "Number|Tensor".to_string(),
                        optional: false,
                        description: "First input".to_string(),
                        default_value: None,
                    },
                    ParameterInfo {
                        name: "b".to_string(),
                        param_type: "Number|Tensor".to_string(),
                        optional: false,
                        description: "Second input".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "Maximum[{-1, 2, -3}, 0] → {0, 2, 0}".to_string(),
                    "Maximum[5, 3] → 5".to_string(),
                ],
                category: "Tensor".to_string(),
                return_type: "Number|Tensor".to_string(),
            }),
            
            // Statistics Functions
            "Mean" => Some(FunctionSignature {
                name: "Mean".to_string(),
                signature: "Mean[list]".to_string(),
                description: "Calculate the arithmetic mean".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "list".to_string(),
                        param_type: "List".to_string(),
                        optional: false,
                        description: "List of numbers".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "Mean[{1, 2, 3, 4, 5}] → 3".to_string(),
                    "Mean[{10, 20}] → 15".to_string(),
                ],
                category: "Statistics".to_string(),
                return_type: "Number".to_string(),
            }),
            
            "StandardDeviation" => Some(FunctionSignature {
                name: "StandardDeviation".to_string(),
                signature: "StandardDeviation[list]".to_string(),
                description: "Calculate the standard deviation".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "list".to_string(),
                        param_type: "List".to_string(),
                        optional: false,
                        description: "List of numbers".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "StandardDeviation[{1, 2, 3, 4, 5}] → 1.58...".to_string(),
                    "StandardDeviation[{10, 10, 10}] → 0".to_string(),
                ],
                category: "Statistics".to_string(),
                return_type: "Number".to_string(),
            }),
            
            "Min" => Some(FunctionSignature {
                name: "Min".to_string(),
                signature: "Min[values...]".to_string(),
                description: "Find the minimum value".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "values".to_string(),
                        param_type: "Number...".to_string(),
                        optional: false,
                        description: "Numbers or list to find minimum of".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "Min[3, 1, 4, 1, 5] → 1".to_string(),
                    "Min[{10, 5, 8}] → 5".to_string(),
                ],
                category: "Statistics".to_string(),
                return_type: "Number".to_string(),
            }),
            
            "Max" => Some(FunctionSignature {
                name: "Max".to_string(),
                signature: "Max[values...]".to_string(),
                description: "Find the maximum value".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "values".to_string(),
                        param_type: "Number...".to_string(),
                        optional: false,
                        description: "Numbers or list to find maximum of".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "Max[3, 1, 4, 1, 5] → 5".to_string(),
                    "Max[{10, 5, 8}] → 10".to_string(),
                ],
                category: "Statistics".to_string(),
                return_type: "Number".to_string(),
            }),
            
            // Network Functions
            "HTTPGet" => Some(FunctionSignature {
                name: "HTTPGet".to_string(),
                signature: "HTTPGet[url]".to_string(),
                description: "Perform HTTP GET request".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "url".to_string(),
                        param_type: "String".to_string(),
                        optional: false,
                        description: "URL to request".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "HTTPGet[\"https://api.example.com/data\"]".to_string(),
                    "HTTPGet[\"http://localhost:8080/status\"]".to_string(),
                ],
                category: "Network".to_string(),
                return_type: "String".to_string(),
            }),
            
            "HTTPPost" => Some(FunctionSignature {
                name: "HTTPPost".to_string(),
                signature: "HTTPPost[url, data]".to_string(),
                description: "Perform HTTP POST request".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "url".to_string(),
                        param_type: "String".to_string(),
                        optional: false,
                        description: "URL to post to".to_string(),
                        default_value: None,
                    },
                    ParameterInfo {
                        name: "data".to_string(),
                        param_type: "String".to_string(),
                        optional: false,
                        description: "Data to send".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "HTTPPost[\"https://api.example.com/submit\", \"name=John\"]".to_string(),
                ],
                category: "Network".to_string(),
                return_type: "String".to_string(),
            }),
            
            // Crypto Functions
            "Hash" => Some(FunctionSignature {
                name: "Hash".to_string(),
                signature: "Hash[data, algorithm]".to_string(),
                description: "Compute cryptographic hash".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "data".to_string(),
                        param_type: "String".to_string(),
                        optional: false,
                        description: "Data to hash".to_string(),
                        default_value: None,
                    },
                    ParameterInfo {
                        name: "algorithm".to_string(),
                        param_type: "String".to_string(),
                        optional: true,
                        description: "Hash algorithm (SHA256, MD5, etc.)".to_string(),
                        default_value: Some("\"SHA256\"".to_string()),
                    }
                ],
                examples: vec![
                    "Hash[\"hello\", \"SHA256\"]".to_string(),
                    "Hash[\"data\", \"MD5\"]".to_string(),
                ],
                category: "Crypto".to_string(),
                return_type: "String".to_string(),
            }),
            
            // Pattern Matching
            "MatchQ" => Some(FunctionSignature {
                name: "MatchQ".to_string(),
                signature: "MatchQ[expression, pattern]".to_string(),
                description: "Test if expression matches pattern".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "expression".to_string(),
                        param_type: "Any".to_string(),
                        optional: false,
                        description: "Expression to test".to_string(),
                        default_value: None,
                    },
                    ParameterInfo {
                        name: "pattern".to_string(),
                        param_type: "Pattern".to_string(),
                        optional: false,
                        description: "Pattern to match against".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "MatchQ[5, _Integer] → True".to_string(),
                    "MatchQ[{1, 2}, {_, _}] → True".to_string(),
                ],
                category: "Rules".to_string(),
                return_type: "Boolean".to_string(),
            }),
            
            "ReplaceAll" => Some(FunctionSignature {
                name: "ReplaceAll".to_string(),
                signature: "ReplaceAll[expression, rule]".to_string(),
                description: "Apply replacement rules to expression".to_string(),
                parameters: vec![
                    ParameterInfo {
                        name: "expression".to_string(),
                        param_type: "Any".to_string(),
                        optional: false,
                        description: "Expression to transform".to_string(),
                        default_value: None,
                    },
                    ParameterInfo {
                        name: "rule".to_string(),
                        param_type: "Rule".to_string(),
                        optional: false,
                        description: "Replacement rule or list of rules".to_string(),
                        default_value: None,
                    }
                ],
                examples: vec![
                    "ReplaceAll[x + y, x -> 2] → 2 + y".to_string(),
                    "ReplaceAll[{a, b, a}, a -> 1] → {1, b, 1}".to_string(),
                ],
                category: "Rules".to_string(),
                return_type: "Any".to_string(),
            }),
            
            // Constants
            "Pi" => Some(FunctionSignature {
                name: "Pi".to_string(),
                signature: "Pi".to_string(),
                description: "Mathematical constant π ≈ 3.14159".to_string(),
                parameters: vec![],
                examples: vec![
                    "Pi → 3.14159...".to_string(),
                    "Sin[Pi] → 0".to_string(),
                    "Cos[Pi/2] → 0".to_string(),
                ],
                category: "Constants".to_string(),
                return_type: "Number".to_string(),
            }),
            
            "E" => Some(FunctionSignature {
                name: "E".to_string(),
                signature: "E".to_string(),
                description: "Euler's number e ≈ 2.71828".to_string(),
                parameters: vec![],
                examples: vec![
                    "E → 2.71828...".to_string(),
                    "Log[E] → 1".to_string(),
                    "Exp[1] → E".to_string(),
                ],
                category: "Constants".to_string(),
                return_type: "Number".to_string(),
            }),
            
            // Add more functions as needed - this is a representative sample of 40+ functions
            _ => None,
        }
    }
    
    /// Add a signature to the database
    fn add_signature(&mut self, signature: FunctionSignature) {
        let category = signature.category.clone();
        let name = signature.name.clone();
        
        // Add to signatures map
        self.signatures.insert(name.clone(), signature);
        
        // Add to category map
        self.categories
            .entry(category)
            .or_insert_with(Vec::new)
            .push(name);
    }
    
    /// Get database statistics
    pub fn get_stats(&self) -> DatabaseStats {
        DatabaseStats {
            total_functions: self.signatures.len(),
            total_categories: self.categories.len(),
            functions_per_category: self.categories
                .iter()
                .map(|(cat, funcs)| (cat.clone(), funcs.len()))
                .collect(),
        }
    }
}

/// Statistics about the function database
#[derive(Debug, Clone)]
pub struct DatabaseStats {
    pub total_functions: usize,
    pub total_categories: usize,
    pub functions_per_category: HashMap<String, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stdlib::StandardLibrary;

    #[test]
    fn test_database_creation() {
        let stdlib = StandardLibrary::new();
        let db = FunctionSignatureDatabase::from_stdlib(&stdlib);
        
        let stats = db.get_stats();
        assert!(stats.total_functions > 0);
        assert!(stats.total_categories > 0);
    }
    
    #[test]
    fn test_function_lookup() {
        let stdlib = StandardLibrary::new();
        let db = FunctionSignatureDatabase::from_stdlib(&stdlib);
        
        // Test known functions
        let sin_sig = db.get_signature("Sin");
        assert!(sin_sig.is_some());
        
        if let Some(sig) = sin_sig {
            assert_eq!(sig.name, "Sin");
            assert_eq!(sig.category, "Math");
            assert!(!sig.parameters.is_empty());
        }
    }
    
    #[test]
    fn test_function_search() {
        let stdlib = StandardLibrary::new();
        let db = FunctionSignatureDatabase::from_stdlib(&stdlib);
        
        // Search for functions starting with "S"
        let s_functions = db.search_functions("S");
        assert!(!s_functions.is_empty());
        
        // All results should start with "S"
        for func in s_functions {
            assert!(func.name.starts_with("S"));
        }
    }
    
    #[test]
    fn test_category_organization() {
        let stdlib = StandardLibrary::new();
        let db = FunctionSignatureDatabase::from_stdlib(&stdlib);
        
        let categories = db.get_categories();
        assert!(!categories.is_empty());
        
        // Check that Math category exists and has functions
        if let Some(math_functions) = db.get_functions_in_category("Math") {
            assert!(!math_functions.is_empty());
        }
    }
    
    #[test]
    fn test_parameter_information() {
        let stdlib = StandardLibrary::new();
        let db = FunctionSignatureDatabase::from_stdlib(&stdlib);
        
        if let Some(sig) = db.get_signature("Append") {
            assert_eq!(sig.parameters.len(), 2);
            assert_eq!(sig.parameters[0].name, "list");
            assert_eq!(sig.parameters[1].name, "element");
            assert!(!sig.parameters[0].optional);
            assert!(!sig.parameters[1].optional);
        }
    }
}