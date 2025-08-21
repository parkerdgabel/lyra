//! Interactive examples database for function learning
//!
//! This module provides runnable code examples that users can execute
//! directly in the REPL with progressive complexity levels.

use std::collections::HashMap;

/// Database of interactive, executable examples
pub struct ExamplesDatabase {
    /// Examples organized by function and complexity
    examples: HashMap<String, CategoryExamples>,
    /// Example execution results cache
    execution_cache: HashMap<String, ExecutionResult>,
    /// Learning paths - sequences of related examples
    learning_paths: HashMap<String, LearningPath>,
}

/// Examples for a function category
#[derive(Debug, Clone)]
pub struct CategoryExamples {
    /// Category name (e.g., "Math", "List", "String")
    pub category: String,
    /// Examples organized by function
    pub function_examples: HashMap<String, FunctionExamples>,
}

/// Complete set of examples for a single function
#[derive(Debug, Clone)]
pub struct FunctionExamples {
    /// Function name
    pub function_name: String,
    /// Quick start examples (1-2 lines)
    pub quickstart: Vec<InteractiveExample>,
    /// Tutorial examples with explanations
    pub tutorial: Vec<InteractiveExample>,
    /// Common patterns and idioms
    pub patterns: Vec<InteractiveExample>,
    /// Real-world applications
    pub applications: Vec<InteractiveExample>,
    /// Performance examples and comparisons
    pub performance: Vec<InteractiveExample>,
}

/// Interactive example that can be executed in REPL
#[derive(Debug, Clone)]
pub struct InteractiveExample {
    /// Unique example ID
    pub id: String,
    /// Example title
    pub title: String,
    /// Executable code
    pub code: String,
    /// Expected output
    pub expected_output: String,
    /// Step-by-step explanation
    pub explanation: Vec<String>,
    /// Learning objectives
    pub objectives: Vec<String>,
    /// Prerequisites (other examples or concepts)
    pub prerequisites: Vec<String>,
    /// Difficulty level (1-5)
    pub difficulty: u8,
    /// Estimated execution time
    pub execution_time_ms: Option<u64>,
    /// Tags for searching and filtering
    pub tags: Vec<String>,
    /// Whether example modifies state
    pub stateful: bool,
}

/// Result of executing an example
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Whether execution succeeded
    pub success: bool,
    /// Actual output
    pub output: String,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Any error message
    pub error: Option<String>,
    /// Memory usage information
    pub memory_info: Option<MemoryInfo>,
}

/// Memory usage information
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Peak memory usage in bytes
    pub peak_memory: u64,
    /// Final memory usage in bytes
    pub final_memory: u64,
}

/// Learning path - sequence of related examples
#[derive(Debug, Clone)]
pub struct LearningPath {
    /// Path name
    pub name: String,
    /// Description of what this path teaches
    pub description: String,
    /// Ordered sequence of example IDs
    pub example_sequence: Vec<String>,
    /// Estimated completion time
    pub estimated_time_minutes: u32,
    /// Learning objectives for the entire path
    pub objectives: Vec<String>,
}

impl ExamplesDatabase {
    /// Create a new examples database
    pub fn new() -> Self {
        let mut db = Self {
            examples: HashMap::new(),
            execution_cache: HashMap::new(),
            learning_paths: HashMap::new(),
        };
        
        db.populate_examples();
        db.create_learning_paths();
        
        db
    }
    
    /// Get examples for a specific function
    pub fn get_function_examples(&self, function_name: &str) -> Option<&FunctionExamples> {
        for category in self.examples.values() {
            if let Some(func_examples) = category.function_examples.get(function_name) {
                return Some(func_examples);
            }
        }
        None
    }
    
    /// Get examples by category
    pub fn get_category_examples(&self, category: &str) -> Option<&CategoryExamples> {
        self.examples.get(category)
    }
    
    /// Search examples by tags or content
    pub fn search_examples(&self, query: &str) -> Vec<InteractiveExample> {
        let mut results = Vec::new();
        let query_lower = query.to_lowercase();
        
        for category in self.examples.values() {
            for func_examples in category.function_examples.values() {
                let all_example_refs = [
                    &func_examples.quickstart,
                    &func_examples.tutorial,
                    &func_examples.patterns,
                    &func_examples.applications,
                    &func_examples.performance,
                ];
                
                for example_vec in all_example_refs {
                    for example in example_vec {
                        if example.title.to_lowercase().contains(&query_lower) ||
                           example.code.to_lowercase().contains(&query_lower) ||
                           example.tags.iter().any(|tag| tag.to_lowercase().contains(&query_lower)) ||
                           example.explanation.iter().any(|exp| exp.to_lowercase().contains(&query_lower)) {
                            results.push(example.clone());
                        }
                    }
                }
            }
        }
        
        // Sort by difficulty, then by relevance
        results.sort_by(|a, b| {
            a.difficulty.cmp(&b.difficulty)
                .then_with(|| a.title.len().cmp(&b.title.len()))
        });
        
        results
    }
    
    /// Get examples by difficulty level
    pub fn get_examples_by_difficulty(&self, min_difficulty: u8, max_difficulty: u8) -> Vec<InteractiveExample> {
        let mut results = Vec::new();
        
        for category in self.examples.values() {
            for func_examples in category.function_examples.values() {
                let all_example_refs = [
                    &func_examples.quickstart,
                    &func_examples.tutorial,
                    &func_examples.patterns,
                    &func_examples.applications,
                    &func_examples.performance,
                ];
                
                for example_vec in all_example_refs {
                    for example in example_vec {
                        if example.difficulty >= min_difficulty && example.difficulty <= max_difficulty {
                            results.push(example.clone());
                        }
                    }
                }
            }
        }
        
        results.sort_by_key(|ex| ex.difficulty);
        results
    }
    
    /// Get learning path
    pub fn get_learning_path(&self, path_name: &str) -> Option<&LearningPath> {
        self.learning_paths.get(path_name)
    }
    
    /// Get all available learning paths
    pub fn get_all_learning_paths(&self) -> Vec<&LearningPath> {
        self.learning_paths.values().collect()
    }
    
    /// Get example by ID
    pub fn get_example_by_id(&self, example_id: &str) -> Option<InteractiveExample> {
        for category in self.examples.values() {
            for func_examples in category.function_examples.values() {
                let all_example_refs = [
                    &func_examples.quickstart,
                    &func_examples.tutorial,
                    &func_examples.patterns,
                    &func_examples.applications,
                    &func_examples.performance,
                ];
                
                for example_vec in all_example_refs {
                    for example in example_vec {
                        if example.id == example_id {
                            return Some(example.clone());
                        }
                    }
                }
            }
        }
        None
    }
    
    /// Cache execution result
    pub fn cache_execution_result(&mut self, example_id: String, result: ExecutionResult) {
        self.execution_cache.insert(example_id, result);
    }
    
    /// Get cached execution result
    pub fn get_cached_result(&self, example_id: &str) -> Option<&ExecutionResult> {
        self.execution_cache.get(example_id)
    }
    
    /// Populate the examples database
    fn populate_examples(&mut self) {
        // Math category examples
        let mut math_examples = CategoryExamples {
            category: "Math".to_string(),
            function_examples: HashMap::new(),
        };
        
        // Sin function examples
        math_examples.function_examples.insert("Sin".to_string(), FunctionExamples {
            function_name: "Sin".to_string(),
            quickstart: vec![
                InteractiveExample {
                    id: "sin_basic_1".to_string(),
                    title: "Basic sine values".to_string(),
                    code: "Sin[0]".to_string(),
                    expected_output: "0".to_string(),
                    explanation: vec![
                        "Sin[0] returns 0 because sine of 0 radians is 0".to_string(),
                        "This is a fundamental trigonometric identity".to_string(),
                    ],
                    objectives: vec!["Understand basic sine function".to_string()],
                    prerequisites: vec![],
                    difficulty: 1,
                    execution_time_ms: Some(1),
                    tags: vec!["trigonometry".to_string(), "basic".to_string()],
                    stateful: false,
                },
                InteractiveExample {
                    id: "sin_basic_2".to_string(),
                    title: "Sine of π/2".to_string(),
                    code: "Sin[Pi/2]".to_string(),
                    expected_output: "1".to_string(),
                    explanation: vec![
                        "Sin[π/2] = 1, which corresponds to 90 degrees".to_string(),
                        "This is the maximum value of the sine function".to_string(),
                    ],
                    objectives: vec!["Learn key sine values".to_string()],
                    prerequisites: vec!["sin_basic_1".to_string()],
                    difficulty: 1,
                    execution_time_ms: Some(1),
                    tags: vec!["trigonometry".to_string(), "pi".to_string()],
                    stateful: false,
                },
            ],
            tutorial: vec![
                InteractiveExample {
                    id: "sin_tutorial_1".to_string(),
                    title: "Sine function properties".to_string(),
                    code: "Map[Sin, {0, Pi/6, Pi/4, Pi/3, Pi/2}]".to_string(),
                    expected_output: "{0, 0.5, 0.707107, 0.866025, 1}".to_string(),
                    explanation: vec![
                        "This shows sine values for common angles".to_string(),
                        "Map applies Sin to each element in the list".to_string(),
                        "These correspond to 0°, 30°, 45°, 60°, and 90°".to_string(),
                    ],
                    objectives: vec![
                        "Understand listable functions".to_string(),
                        "Learn common trigonometric values".to_string(),
                    ],
                    prerequisites: vec!["sin_basic_2".to_string()],
                    difficulty: 2,
                    execution_time_ms: Some(5),
                    tags: vec!["trigonometry".to_string(), "map".to_string(), "listable".to_string()],
                    stateful: false,
                },
            ],
            patterns: vec![
                InteractiveExample {
                    id: "sin_pattern_1".to_string(),
                    title: "Generating sine waves".to_string(),
                    code: "data = Table[Sin[2*Pi*x/10], {x, 0, 20}]; Length[data]".to_string(),
                    expected_output: "21".to_string(),
                    explanation: vec![
                        "Table generates a list of sine wave values".to_string(),
                        "2*Pi*x/10 creates one complete cycle over 10 points".to_string(),
                        "This pattern is common in signal processing".to_string(),
                    ],
                    objectives: vec![
                        "Learn to generate periodic data".to_string(),
                        "Understand wave generation patterns".to_string(),
                    ],
                    prerequisites: vec!["sin_tutorial_1".to_string()],
                    difficulty: 3,
                    execution_time_ms: Some(10),
                    tags: vec!["waves".to_string(), "table".to_string(), "periodic".to_string()],
                    stateful: false,
                },
            ],
            applications: vec![
                InteractiveExample {
                    id: "sin_app_1".to_string(),
                    title: "Harmonic oscillator simulation".to_string(),
                    code: "amplitude = 2; frequency = 0.5; phase = 0; time = 1.0; amplitude * Sin[2*Pi*frequency*time + phase]".to_string(),
                    expected_output: "0".to_string(),
                    explanation: vec![
                        "This models a harmonic oscillator in physics".to_string(),
                        "amplitude controls the maximum displacement".to_string(),
                        "frequency determines how fast it oscillates".to_string(),
                        "phase shifts the wave in time".to_string(),
                    ],
                    objectives: vec![
                        "Apply trigonometry to physics".to_string(),
                        "Understand harmonic motion".to_string(),
                    ],
                    prerequisites: vec!["sin_pattern_1".to_string()],
                    difficulty: 4,
                    execution_time_ms: Some(5),
                    tags: vec!["physics".to_string(), "oscillator".to_string(), "simulation".to_string()],
                    stateful: false,
                },
            ],
            performance: vec![],
        });
        
        // List category examples
        let mut list_examples = CategoryExamples {
            category: "List".to_string(),
            function_examples: HashMap::new(),
        };
        
        // Length function examples
        list_examples.function_examples.insert("Length".to_string(), FunctionExamples {
            function_name: "Length".to_string(),
            quickstart: vec![
                InteractiveExample {
                    id: "length_basic_1".to_string(),
                    title: "Simple list length".to_string(),
                    code: "Length[{1, 2, 3, 4, 5}]".to_string(),
                    expected_output: "5".to_string(),
                    explanation: vec![
                        "Length counts the number of elements in a list".to_string(),
                        "This list has 5 elements, so Length returns 5".to_string(),
                    ],
                    objectives: vec!["Learn basic list operations".to_string()],
                    prerequisites: vec![],
                    difficulty: 1,
                    execution_time_ms: Some(1),
                    tags: vec!["lists".to_string(), "basic".to_string()],
                    stateful: false,
                },
            ],
            tutorial: vec![
                InteractiveExample {
                    id: "length_tutorial_1".to_string(),
                    title: "Nested lists".to_string(),
                    code: "Length[{{1, 2}, {3, 4, 5}, {6}}]".to_string(),
                    expected_output: "3".to_string(),
                    explanation: vec![
                        "Length only counts top-level elements".to_string(),
                        "Even though there are 6 numbers total, there are only 3 sublists".to_string(),
                        "Use Flatten first if you want to count all elements".to_string(),
                    ],
                    objectives: vec![
                        "Understand nested list structure".to_string(),
                        "Learn the difference between levels".to_string(),
                    ],
                    prerequisites: vec!["length_basic_1".to_string()],
                    difficulty: 2,
                    execution_time_ms: Some(2),
                    tags: vec!["lists".to_string(), "nested".to_string(), "levels".to_string()],
                    stateful: false,
                },
            ],
            patterns: vec![
                InteractiveExample {
                    id: "length_pattern_1".to_string(),
                    title: "Data validation pattern".to_string(),
                    code: "data = {1, 2, 3}; If[Length[data] > 0, Mean[data], \"No data\"]".to_string(),
                    expected_output: "2".to_string(),
                    explanation: vec![
                        "Common pattern: check if data exists before processing".to_string(),
                        "Length[data] > 0 tests for non-empty list".to_string(),
                        "This prevents errors in downstream calculations".to_string(),
                    ],
                    objectives: vec![
                        "Learn defensive programming patterns".to_string(),
                        "Understand conditional processing".to_string(),
                    ],
                    prerequisites: vec!["length_tutorial_1".to_string()],
                    difficulty: 3,
                    execution_time_ms: Some(5),
                    tags: vec!["validation".to_string(), "conditional".to_string(), "patterns".to_string()],
                    stateful: false,
                },
            ],
            applications: vec![],
            performance: vec![],
        });
        
        self.examples.insert("Math".to_string(), math_examples);
        self.examples.insert("List".to_string(), list_examples);
    }
    
    /// Create learning paths
    fn create_learning_paths(&mut self) {
        // Trigonometry learning path
        self.learning_paths.insert("trigonometry_basics".to_string(), LearningPath {
            name: "Trigonometry Basics".to_string(),
            description: "Learn fundamental trigonometric functions and their applications".to_string(),
            example_sequence: vec![
                "sin_basic_1".to_string(),
                "sin_basic_2".to_string(),
                "sin_tutorial_1".to_string(),
                "sin_pattern_1".to_string(),
                "sin_app_1".to_string(),
            ],
            estimated_time_minutes: 15,
            objectives: vec![
                "Understand sine, cosine, and tangent functions".to_string(),
                "Learn to generate periodic data".to_string(),
                "Apply trigonometry to real-world problems".to_string(),
            ],
        });
        
        // List processing learning path
        self.learning_paths.insert("list_processing".to_string(), LearningPath {
            name: "List Processing Fundamentals".to_string(),
            description: "Master list operations and data manipulation".to_string(),
            example_sequence: vec![
                "length_basic_1".to_string(),
                "length_tutorial_1".to_string(),
                "length_pattern_1".to_string(),
            ],
            estimated_time_minutes: 10,
            objectives: vec![
                "Understand list structure and operations".to_string(),
                "Learn defensive programming patterns".to_string(),
                "Master data validation techniques".to_string(),
            ],
        });
    }
}

impl Default for ExamplesDatabase {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_examples_retrieval() {
        let db = ExamplesDatabase::new();
        
        let sin_examples = db.get_function_examples("Sin");
        assert!(sin_examples.is_some());
        
        if let Some(examples) = sin_examples {
            assert_eq!(examples.function_name, "Sin");
            assert!(!examples.quickstart.is_empty());
        }
    }
    
    #[test]
    fn test_example_search() {
        let db = ExamplesDatabase::new();
        
        let results = db.search_examples("trigonometry");
        assert!(!results.is_empty());
        
        // Should find sine examples
        let sine_results: Vec<_> = results.iter()
            .filter(|ex| ex.tags.contains(&"trigonometry".to_string()))
            .collect();
        assert!(!sine_results.is_empty());
    }
    
    #[test]
    fn test_difficulty_filtering() {
        let db = ExamplesDatabase::new();
        
        let beginner_examples = db.get_examples_by_difficulty(1, 2);
        assert!(!beginner_examples.is_empty());
        
        // All should be difficulty 1 or 2
        for example in beginner_examples {
            assert!(example.difficulty >= 1 && example.difficulty <= 2);
        }
    }
    
    #[test]
    fn test_learning_paths() {
        let db = ExamplesDatabase::new();
        
        let trig_path = db.get_learning_path("trigonometry_basics");
        assert!(trig_path.is_some());
        
        if let Some(path) = trig_path {
            assert_eq!(path.name, "Trigonometry Basics");
            assert!(!path.example_sequence.is_empty());
        }
    }
    
    #[test]
    fn test_example_by_id() {
        let db = ExamplesDatabase::new();
        
        let example = db.get_example_by_id("sin_basic_1");
        assert!(example.is_some());
        
        if let Some(ex) = example {
            assert_eq!(ex.id, "sin_basic_1");
            assert_eq!(ex.code, "Sin[0]");
        }
    }
}