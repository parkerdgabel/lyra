//! Rich documentation database for function hints
//!
//! This module provides comprehensive documentation, examples, and cross-references
//! for all standard library functions.

use std::collections::HashMap;

/// Database of rich function documentation
pub struct DocumentationDatabase {
    /// Detailed documentation entries indexed by function name
    documentation: HashMap<String, DocumentationEntry>,
    /// Cross-references between related functions
    cross_references: HashMap<String, Vec<String>>,
    /// Function examples organized by complexity
    examples: HashMap<String, ExampleSet>,
    /// Error patterns and solutions
    error_patterns: HashMap<String, Vec<ErrorSolution>>,
}

/// Comprehensive documentation entry for a function
#[derive(Debug, Clone)]
pub struct DocumentationEntry {
    /// Function name
    pub name: String,
    /// Detailed description with mathematical background
    pub detailed_description: String,
    /// Usage notes and best practices
    pub usage_notes: Vec<String>,
    /// Common pitfalls and how to avoid them
    pub pitfalls: Vec<String>,
    /// Performance characteristics
    pub performance_notes: Vec<String>,
    /// Mathematical domain and constraints
    pub domain_constraints: Vec<String>,
    /// Related concepts and algorithms
    pub related_concepts: Vec<String>,
    /// Links to external documentation
    pub external_links: Vec<String>,
}

/// Set of examples organized by complexity level
#[derive(Debug, Clone)]
pub struct ExampleSet {
    /// Function name
    pub function_name: String,
    /// Basic examples for beginners
    pub basic_examples: Vec<Example>,
    /// Intermediate examples showing common patterns
    pub intermediate_examples: Vec<Example>,
    /// Advanced examples for complex use cases
    pub advanced_examples: Vec<Example>,
    /// Real-world application examples
    pub application_examples: Vec<Example>,
}

/// Individual example with code and explanation
#[derive(Debug, Clone)]
pub struct Example {
    /// Example code
    pub code: String,
    /// Expected result
    pub result: String,
    /// Explanation of what happens
    pub explanation: String,
    /// Keywords for searching
    pub keywords: Vec<String>,
    /// Whether this example is executable in REPL
    pub executable: bool,
}

/// Error pattern and solution mapping
#[derive(Debug, Clone)]
pub struct ErrorSolution {
    /// Error pattern or message
    pub error_pattern: String,
    /// Explanation of what causes this error
    pub cause: String,
    /// Solution or workaround
    pub solution: String,
    /// Example of correct usage
    pub correct_example: String,
}

impl DocumentationDatabase {
    /// Create a new documentation database
    pub fn new() -> Self {
        let mut db = Self {
            documentation: HashMap::new(),
            cross_references: HashMap::new(),
            examples: HashMap::new(),
            error_patterns: HashMap::new(),
        };
        
        db.populate_documentation();
        db.populate_examples();
        db.populate_error_patterns();
        db.build_cross_references();
        
        db
    }
    
    /// Get documentation for a function
    pub fn get_documentation(&self, function_name: &str) -> Option<&DocumentationEntry> {
        self.documentation.get(function_name)
    }
    
    /// Get examples for a function
    pub fn get_examples(&self, function_name: &str) -> Option<&ExampleSet> {
        self.examples.get(function_name)
    }
    
    /// Get related functions
    pub fn get_related_functions(&self, function_name: &str) -> Option<&Vec<String>> {
        self.cross_references.get(function_name)
    }
    
    /// Get error solutions for a function
    pub fn get_error_solutions(&self, function_name: &str) -> Vec<&ErrorSolution> {
        self.error_patterns
            .get(function_name)
            .map(|solutions| solutions.iter().collect())
            .unwrap_or_default()
    }
    
    /// Search examples by keyword
    pub fn search_examples(&self, keyword: &str) -> Vec<(String, Example)> {
        let mut results = Vec::new();
        
        for (func_name, example_set) in &self.examples {
            let all_example_refs = [
                &example_set.basic_examples,
                &example_set.intermediate_examples,
                &example_set.advanced_examples,
                &example_set.application_examples,
            ];
            
            for example_vec in all_example_refs {
                for example in example_vec {
                    if example.keywords.iter().any(|k| k.contains(keyword)) ||
                       example.explanation.to_lowercase().contains(&keyword.to_lowercase()) {
                        results.push((func_name.clone(), example.clone()));
                    }
                }
            }
        }
        
        results
    }
    
    /// Populate comprehensive documentation
    fn populate_documentation(&mut self) {
        // Mathematical Functions
        self.add_documentation("Sin", DocumentationEntry {
            name: "Sin".to_string(),
            detailed_description: "Sine trigonometric function. Computes sin(x) where x is in radians. The sine function is periodic with period 2π and maps the real line to [-1, 1].".to_string(),
            usage_notes: vec![
                "Input angles should be in radians, not degrees".to_string(),
                "Use Sin[x * Pi / 180] to convert degrees to radians".to_string(),
                "Sin is Listable: Sin[{x, y, z}] applies to each element".to_string(),
            ],
            pitfalls: vec![
                "Common mistake: using degrees instead of radians".to_string(),
                "Floating point precision: Sin[Pi] ≈ 0, not exactly 0".to_string(),
            ],
            performance_notes: vec![
                "Uses optimized libm implementation".to_string(),
                "Vectorized operations available for arrays".to_string(),
            ],
            domain_constraints: vec![
                "Input: any real number".to_string(),
                "Output: [-1, 1]".to_string(),
            ],
            related_concepts: vec![
                "Trigonometry".to_string(),
                "Periodic functions".to_string(),
                "Fourier analysis".to_string(),
            ],
            external_links: vec![
                "https://en.wikipedia.org/wiki/Sine".to_string(),
            ],
        });
        
        self.add_documentation("Cos", DocumentationEntry {
            name: "Cos".to_string(),
            detailed_description: "Cosine trigonometric function. Computes cos(x) where x is in radians. The cosine function is periodic with period 2π and maps the real line to [-1, 1].".to_string(),
            usage_notes: vec![
                "Input angles should be in radians".to_string(),
                "Cos[0] = 1, Cos[Pi/2] = 0, Cos[Pi] = -1".to_string(),
                "Cos is an even function: Cos[-x] = Cos[x]".to_string(),
            ],
            pitfalls: vec![
                "Degrees vs radians confusion".to_string(),
                "Floating point precision near zeros".to_string(),
            ],
            performance_notes: vec![
                "Highly optimized implementation".to_string(),
                "SIMD acceleration for large arrays".to_string(),
            ],
            domain_constraints: vec![
                "Input: any real number".to_string(),
                "Output: [-1, 1]".to_string(),
            ],
            related_concepts: vec![
                "Trigonometry".to_string(),
                "Even functions".to_string(),
                "Harmonic analysis".to_string(),
            ],
            external_links: vec![
                "https://en.wikipedia.org/wiki/Cosine".to_string(),
            ],
        });
        
        // List Functions
        self.add_documentation("Length", DocumentationEntry {
            name: "Length".to_string(),
            detailed_description: "Returns the number of elements in a list. For nested lists, only counts elements at the top level. For empty lists, returns 0.".to_string(),
            usage_notes: vec![
                "Only counts top-level elements, not nested ones".to_string(),
                "Works with any type of list elements".to_string(),
                "Constant time operation O(1)".to_string(),
            ],
            pitfalls: vec![
                "Doesn't count nested elements: Length[{{1, 2}, {3}}] = 2".to_string(),
                "Use Flatten first if you want total element count".to_string(),
            ],
            performance_notes: vec![
                "O(1) time complexity".to_string(),
                "No memory allocation".to_string(),
            ],
            domain_constraints: vec![
                "Input: any list".to_string(),
                "Output: non-negative integer".to_string(),
            ],
            related_concepts: vec![
                "List structure".to_string(),
                "Data organization".to_string(),
            ],
            external_links: vec![],
        });
        
        // String Functions
        self.add_documentation("StringJoin", DocumentationEntry {
            name: "StringJoin".to_string(),
            detailed_description: "Concatenates multiple strings into a single string. Accepts variable number of arguments and efficiently combines them without intermediate allocations.".to_string(),
            usage_notes: vec![
                "Accepts any number of string arguments".to_string(),
                "Non-string arguments are converted to strings".to_string(),
                "Empty strings are handled gracefully".to_string(),
            ],
            pitfalls: vec![
                "Watch for automatic type conversion of numbers".to_string(),
                "Large string concatenations may use significant memory".to_string(),
            ],
            performance_notes: vec![
                "Optimized for multiple concatenations".to_string(),
                "Linear time O(n) where n is total character count".to_string(),
            ],
            domain_constraints: vec![
                "Input: string arguments".to_string(),
                "Output: concatenated string".to_string(),
            ],
            related_concepts: vec![
                "String processing".to_string(),
                "Text manipulation".to_string(),
            ],
            external_links: vec![],
        });
        
        // Tensor Functions
        self.add_documentation("Dot", DocumentationEntry {
            name: "Dot".to_string(),
            detailed_description: "Matrix/vector multiplication following standard linear algebra conventions. Supports vector dot products, matrix-vector multiplication, and matrix-matrix multiplication with automatic shape inference.".to_string(),
            usage_notes: vec![
                "Automatically determines operation type from input shapes".to_string(),
                "Follows NumPy-style broadcasting rules".to_string(),
                "Optimized for neural network operations".to_string(),
            ],
            pitfalls: vec![
                "Shape compatibility: inner dimensions must match".to_string(),
                "Result shape depends on input shapes".to_string(),
                "Large matrices may require significant memory".to_string(),
            ],
            performance_notes: vec![
                "Uses optimized BLAS routines when available".to_string(),
                "Memory-efficient for large operations".to_string(),
                "SIMD acceleration for supported types".to_string(),
            ],
            domain_constraints: vec![
                "Input: compatible tensor shapes".to_string(),
                "Output: result tensor with inferred shape".to_string(),
            ],
            related_concepts: vec![
                "Linear algebra".to_string(),
                "Neural networks".to_string(),
                "Matrix operations".to_string(),
            ],
            external_links: vec![
                "https://en.wikipedia.org/wiki/Matrix_multiplication".to_string(),
            ],
        });
    }
    
    /// Populate example sets for functions
    fn populate_examples(&mut self) {
        // Sin examples
        self.add_examples("Sin", ExampleSet {
            function_name: "Sin".to_string(),
            basic_examples: vec![
                Example {
                    code: "Sin[0]".to_string(),
                    result: "0".to_string(),
                    explanation: "Sine of zero is zero".to_string(),
                    keywords: vec!["zero".to_string(), "basic".to_string()],
                    executable: true,
                },
                Example {
                    code: "Sin[Pi/2]".to_string(),
                    result: "1".to_string(),
                    explanation: "Sine of π/2 radians (90 degrees) is 1".to_string(),
                    keywords: vec!["pi".to_string(), "90 degrees".to_string()],
                    executable: true,
                },
            ],
            intermediate_examples: vec![
                Example {
                    code: "Sin[{0, Pi/6, Pi/4, Pi/3, Pi/2}]".to_string(),
                    result: "{0, 0.5, 0.707..., 0.866..., 1}".to_string(),
                    explanation: "Listable operation: applies to each element".to_string(),
                    keywords: vec!["listable".to_string(), "array".to_string()],
                    executable: true,
                },
                Example {
                    code: "Map[Sin, Range[0, 2*Pi, Pi/4]]".to_string(),
                    result: "sine wave values".to_string(),
                    explanation: "Generate sine wave values over one period".to_string(),
                    keywords: vec!["wave".to_string(), "period".to_string()],
                    executable: true,
                },
            ],
            advanced_examples: vec![
                Example {
                    code: "Plot[Sin[x], {x, 0, 2*Pi}]".to_string(),
                    result: "sine wave plot".to_string(),
                    explanation: "Visualize sine function over one period".to_string(),
                    keywords: vec!["plot".to_string(), "visualization".to_string()],
                    executable: false,
                },
            ],
            application_examples: vec![
                Example {
                    code: "oscillation = Sin[2*Pi*frequency*time]".to_string(),
                    result: "harmonic oscillation".to_string(),
                    explanation: "Model harmonic motion in physics".to_string(),
                    keywords: vec!["physics".to_string(), "oscillation".to_string()],
                    executable: false,
                },
            ],
        });
        
        // Length examples
        self.add_examples("Length", ExampleSet {
            function_name: "Length".to_string(),
            basic_examples: vec![
                Example {
                    code: "Length[{1, 2, 3}]".to_string(),
                    result: "3".to_string(),
                    explanation: "Count elements in a simple list".to_string(),
                    keywords: vec!["count".to_string(), "simple".to_string()],
                    executable: true,
                },
                Example {
                    code: "Length[{}]".to_string(),
                    result: "0".to_string(),
                    explanation: "Empty list has length zero".to_string(),
                    keywords: vec!["empty".to_string(), "zero".to_string()],
                    executable: true,
                },
            ],
            intermediate_examples: vec![
                Example {
                    code: "Length[{{1, 2}, {3, 4, 5}}]".to_string(),
                    result: "2".to_string(),
                    explanation: "Nested lists: only counts top-level elements".to_string(),
                    keywords: vec!["nested".to_string(), "top-level".to_string()],
                    executable: true,
                },
            ],
            advanced_examples: vec![
                Example {
                    code: "Map[Length, {{1, 2}, {3, 4, 5}, {6}}]".to_string(),
                    result: "{2, 3, 1}".to_string(),
                    explanation: "Get lengths of multiple sublists".to_string(),
                    keywords: vec!["sublists".to_string(), "multiple".to_string()],
                    executable: true,
                },
            ],
            application_examples: vec![
                Example {
                    code: "If[Length[data] > threshold, ProcessData[data], Warning[\"Insufficient data\"]]".to_string(),
                    result: "conditional processing".to_string(),
                    explanation: "Conditional processing based on data size".to_string(),
                    keywords: vec!["conditional".to_string(), "validation".to_string()],
                    executable: false,
                },
            ],
        });
        
        // Dot examples
        self.add_examples("Dot", ExampleSet {
            function_name: "Dot".to_string(),
            basic_examples: vec![
                Example {
                    code: "Dot[{1, 2, 3}, {4, 5, 6}]".to_string(),
                    result: "32".to_string(),
                    explanation: "Vector dot product: 1*4 + 2*5 + 3*6 = 32".to_string(),
                    keywords: vec!["vector".to_string(), "dot product".to_string()],
                    executable: true,
                },
            ],
            intermediate_examples: vec![
                Example {
                    code: "Dot[{{1, 2}, {3, 4}}, {1, 0}]".to_string(),
                    result: "{1, 3}".to_string(),
                    explanation: "Matrix-vector multiplication".to_string(),
                    keywords: vec!["matrix".to_string(), "vector".to_string()],
                    executable: true,
                },
            ],
            advanced_examples: vec![
                Example {
                    code: "Dot[{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}]".to_string(),
                    result: "{{19, 22}, {43, 50}}".to_string(),
                    explanation: "Matrix-matrix multiplication".to_string(),
                    keywords: vec!["matrix multiplication".to_string()],
                    executable: true,
                },
            ],
            application_examples: vec![
                Example {
                    code: "neurons = Dot[weights, inputs] + biases".to_string(),
                    result: "neural network layer".to_string(),
                    explanation: "Forward pass in neural network layer".to_string(),
                    keywords: vec!["neural network".to_string(), "forward pass".to_string()],
                    executable: false,
                },
            ],
        });
    }
    
    /// Populate error patterns and solutions
    fn populate_error_patterns(&mut self) {
        // Sin error patterns
        self.add_error_patterns("Sin", vec![
            ErrorSolution {
                error_pattern: "TypeError: Sin expects numeric input".to_string(),
                cause: "Passing non-numeric value to Sin function".to_string(),
                solution: "Ensure input is a number or numeric expression".to_string(),
                correct_example: "Sin[Pi/2] instead of Sin[\"pi/2\"]".to_string(),
            },
        ]);
        
        // Length error patterns
        self.add_error_patterns("Length", vec![
            ErrorSolution {
                error_pattern: "TypeError: Length expects list input".to_string(),
                cause: "Passing non-list value to Length function".to_string(),
                solution: "Ensure input is a list, or use appropriate function for other types".to_string(),
                correct_example: "Length[{1, 2, 3}] instead of Length[123]".to_string(),
            },
        ]);
        
        // Dot error patterns
        self.add_error_patterns("Dot", vec![
            ErrorSolution {
                error_pattern: "ShapeError: incompatible dimensions for dot product".to_string(),
                cause: "Inner dimensions don't match for matrix multiplication".to_string(),
                solution: "Ensure inner dimensions match: [m,n] · [n,p] = [m,p]".to_string(),
                correct_example: "Dot[{{1,2}}, {{3},{4}}] - 1x2 with 2x1 matrices".to_string(),
            },
        ]);
    }
    
    /// Build cross-references between related functions
    fn build_cross_references(&mut self) {
        // Math function relationships
        self.add_cross_reference("Sin", vec!["Cos".to_string(), "Tan".to_string(), "ArcSin".to_string()]);
        self.add_cross_reference("Cos", vec!["Sin".to_string(), "Tan".to_string(), "ArcCos".to_string()]);
        self.add_cross_reference("Tan", vec!["Sin".to_string(), "Cos".to_string(), "ArcTan".to_string()]);
        
        // List function relationships
        self.add_cross_reference("Length", vec!["Head".to_string(), "Tail".to_string(), "Dimensions".to_string()]);
        self.add_cross_reference("Head", vec!["Tail".to_string(), "Length".to_string(), "First".to_string()]);
        self.add_cross_reference("Tail", vec!["Head".to_string(), "Rest".to_string(), "Drop".to_string()]);
        
        // Tensor function relationships
        self.add_cross_reference("Dot", vec!["Transpose".to_string(), "MatrixMultiply".to_string(), "Cross".to_string()]);
        self.add_cross_reference("Transpose", vec!["Dot".to_string(), "Conjugate".to_string()]);
        
        // String function relationships
        self.add_cross_reference("StringJoin", vec!["StringSplit".to_string(), "StringReplace".to_string()]);
        self.add_cross_reference("StringLength", vec!["StringTake".to_string(), "StringDrop".to_string()]);
    }
    
    // Helper methods
    fn add_documentation(&mut self, name: &str, doc: DocumentationEntry) {
        self.documentation.insert(name.to_string(), doc);
    }
    
    fn add_examples(&mut self, name: &str, examples: ExampleSet) {
        self.examples.insert(name.to_string(), examples);
    }
    
    fn add_error_patterns(&mut self, name: &str, patterns: Vec<ErrorSolution>) {
        self.error_patterns.insert(name.to_string(), patterns);
    }
    
    fn add_cross_reference(&mut self, name: &str, related: Vec<String>) {
        self.cross_references.insert(name.to_string(), related);
    }
}

impl Default for DocumentationDatabase {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_documentation_retrieval() {
        let db = DocumentationDatabase::new();
        
        let sin_doc = db.get_documentation("Sin");
        assert!(sin_doc.is_some());
        
        if let Some(doc) = sin_doc {
            assert_eq!(doc.name, "Sin");
            assert!(!doc.detailed_description.is_empty());
            assert!(!doc.usage_notes.is_empty());
        }
    }
    
    #[test]
    fn test_examples_retrieval() {
        let db = DocumentationDatabase::new();
        
        let sin_examples = db.get_examples("Sin");
        assert!(sin_examples.is_some());
        
        if let Some(examples) = sin_examples {
            assert_eq!(examples.function_name, "Sin");
            assert!(!examples.basic_examples.is_empty());
        }
    }
    
    #[test]
    fn test_cross_references() {
        let db = DocumentationDatabase::new();
        
        let sin_related = db.get_related_functions("Sin");
        assert!(sin_related.is_some());
        
        if let Some(related) = sin_related {
            assert!(related.contains(&"Cos".to_string()));
        }
    }
    
    #[test]
    fn test_example_search() {
        let db = DocumentationDatabase::new();
        
        let results = db.search_examples("dot product");
        assert!(!results.is_empty());
        
        let dot_results: Vec<_> = results.iter()
            .filter(|(func_name, _)| *func_name == "Dot")
            .collect();
        assert!(!dot_results.is_empty());
    }
    
    #[test]
    fn test_error_solutions() {
        let db = DocumentationDatabase::new();
        
        let sin_errors = db.get_error_solutions("Sin");
        assert!(!sin_errors.is_empty());
        
        let first_error = &sin_errors[0];
        assert!(!first_error.error_pattern.is_empty());
        assert!(!first_error.solution.is_empty());
    }
}