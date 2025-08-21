//! Documentation and Code Generation System for Lyra Stdlib
//!
//! This module implements Agent 8 - comprehensive documentation, help system,
//! and code generation capabilities for the Lyra symbolic computation engine.
//!
//! Features:
//! - Interactive help system with function discovery
//! - Comprehensive function metadata and examples
//! - Code generation to multiple target languages
//! - Documentation export in various formats

use crate::foreign::{Foreign, LyObj};
use crate::vm::{Value, VmResult, VmError};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use regex::Regex;

/// Comprehensive metadata for a stdlib function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionMetadata {
    pub name: String,
    pub category: String,
    pub description: String,
    pub signature: String,
    pub parameters: Vec<ParameterInfo>,
    pub return_type: String,
    pub examples: Vec<ExampleUsage>,
    pub related_functions: Vec<String>,
    pub tags: Vec<String>,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterInfo {
    pub name: String,
    pub description: String,
    pub param_type: String,
    pub optional: bool,
    pub default_value: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExampleUsage {
    pub code: String,
    pub description: String,
    pub expected_output: String,
}

/// Function registry with comprehensive metadata
#[derive(Debug, Clone)]
pub struct FunctionRegistry {
    pub functions: HashMap<String, FunctionMetadata>,
    pub categories: HashMap<String, Vec<String>>, // category -> function names
    pub tags: HashMap<String, Vec<String>>, // tag -> function names
}

impl FunctionRegistry {
    pub fn new() -> Self {
        let mut registry = FunctionRegistry {
            functions: HashMap::new(),
            categories: HashMap::new(),
            tags: HashMap::new(),
        };
        
        // Populate the registry with all stdlib functions
        registry.populate_metadata();
        registry
    }
    
    /// Populate the registry with metadata for all stdlib functions
    fn populate_metadata(&mut self) {
        // List operations
        self.add_function_metadata(FunctionMetadata {
            name: "Length".to_string(),
            category: "List".to_string(),
            description: "Returns the length of a list or string".to_string(),
            signature: "Length[expr]".to_string(),
            parameters: vec![ParameterInfo {
                name: "expr".to_string(),
                description: "List, string, or other expression with length".to_string(),
                param_type: "List | String".to_string(),
                optional: false,
                default_value: None,
            }],
            return_type: "Integer".to_string(),
            examples: vec![
                ExampleUsage {
                    code: "Length[{1, 2, 3, 4}]".to_string(),
                    description: "Length of a list".to_string(),
                    expected_output: "4".to_string(),
                },
                ExampleUsage {
                    code: "Length[\"hello\"]".to_string(),
                    description: "Length of a string".to_string(),
                    expected_output: "5".to_string(),
                },
            ],
            related_functions: vec!["Head".to_string(), "Tail".to_string(), "Append".to_string()],
            tags: vec!["list".to_string(), "string".to_string(), "basic".to_string()],
            version: "1.0".to_string(),
        });
        
        // String operations
        self.add_function_metadata(FunctionMetadata {
            name: "StringJoin".to_string(),
            category: "String".to_string(),
            description: "Joins multiple strings together".to_string(),
            signature: "StringJoin[list] or StringJoin[str1, str2, ...]".to_string(),
            parameters: vec![ParameterInfo {
                name: "strings".to_string(),
                description: "List of strings or multiple string arguments".to_string(),
                param_type: "List[String] | String...".to_string(),
                optional: false,
                default_value: None,
            }],
            return_type: "String".to_string(),
            examples: vec![
                ExampleUsage {
                    code: "StringJoin[{\"Hello\", \" \", \"World\"}]".to_string(),
                    description: "Join strings from a list".to_string(),
                    expected_output: "\"Hello World\"".to_string(),
                },
                ExampleUsage {
                    code: "StringJoin[\"A\", \"B\", \"C\"]".to_string(),
                    description: "Join multiple string arguments".to_string(),
                    expected_output: "\"ABC\"".to_string(),
                },
            ],
            related_functions: vec!["StringSplit".to_string(), "StringLength".to_string()],
            tags: vec!["string".to_string(), "manipulation".to_string()],
            version: "1.0".to_string(),
        });
        
        // Math operations
        self.add_function_metadata(FunctionMetadata {
            name: "Sin".to_string(),
            category: "Math".to_string(),
            description: "Computes the sine of an angle in radians".to_string(),
            signature: "Sin[x]".to_string(),
            parameters: vec![ParameterInfo {
                name: "x".to_string(),
                description: "Angle in radians".to_string(),
                param_type: "Real | Integer".to_string(),
                optional: false,
                default_value: None,
            }],
            return_type: "Real".to_string(),
            examples: vec![
                ExampleUsage {
                    code: "Sin[0]".to_string(),
                    description: "Sine of zero".to_string(),
                    expected_output: "0.0".to_string(),
                },
                ExampleUsage {
                    code: "Sin[Pi/2]".to_string(),
                    description: "Sine of π/2".to_string(),
                    expected_output: "1.0".to_string(),
                },
            ],
            related_functions: vec!["Cos".to_string(), "Tan".to_string(), "ArcSin".to_string()],
            tags: vec!["math".to_string(), "trigonometry".to_string()],
            version: "1.0".to_string(),
        });
        
        // Add more function metadata...
        self.populate_advanced_functions();
    }
    
    fn populate_advanced_functions(&mut self) {
        // JSON Processing
        self.add_function_metadata(FunctionMetadata {
            name: "JSONParse".to_string(),
            category: "Data".to_string(),
            description: "Parses a JSON string into Lyra data structures".to_string(),
            signature: "JSONParse[json_string]".to_string(),
            parameters: vec![ParameterInfo {
                name: "json_string".to_string(),
                description: "Valid JSON string to parse".to_string(),
                param_type: "String".to_string(),
                optional: false,
                default_value: None,
            }],
            return_type: "List | String | Integer | Real | Boolean".to_string(),
            examples: vec![
                ExampleUsage {
                    code: "JSONParse[\"{\\\"name\\\": \\\"Alice\\\", \\\"age\\\": 30}\"]".to_string(),
                    description: "Parse a JSON object".to_string(),
                    expected_output: "{\"name\" -> \"Alice\", \"age\" -> 30}".to_string(),
                },
            ],
            related_functions: vec!["JSONStringify".to_string(), "JSONQuery".to_string()],
            tags: vec!["data".to_string(), "json".to_string(), "parsing".to_string()],
            version: "1.0".to_string(),
        });
        
        // StringTemplate
        self.add_function_metadata(FunctionMetadata {
            name: "StringTemplate".to_string(),
            category: "String".to_string(),
            description: "Creates formatted strings using template substitution".to_string(),
            signature: "StringTemplate[template, substitutions]".to_string(),
            parameters: vec![
                ParameterInfo {
                    name: "template".to_string(),
                    description: "Template string with {placeholder} markers".to_string(),
                    param_type: "String".to_string(),
                    optional: false,
                    default_value: None,
                },
                ParameterInfo {
                    name: "substitutions".to_string(),
                    description: "Dictionary of placeholder -> value mappings".to_string(),
                    param_type: "List[Rule]".to_string(),
                    optional: false,
                    default_value: None,
                },
            ],
            return_type: "String".to_string(),
            examples: vec![
                ExampleUsage {
                    code: "StringTemplate[\"Hello {name}!\", {\"name\" -> \"World\"}]".to_string(),
                    description: "Basic template substitution".to_string(),
                    expected_output: "\"Hello World!\"".to_string(),
                },
            ],
            related_functions: vec!["StringFormat".to_string(), "StringReplace".to_string()],
            tags: vec!["string".to_string(), "template".to_string(), "formatting".to_string()],
            version: "1.0".to_string(),
        });
        
        // ParallelMap
        self.add_function_metadata(FunctionMetadata {
            name: "ParallelMap".to_string(),
            category: "Concurrency".to_string(),
            description: "Applies a function to each element of a list in parallel".to_string(),
            signature: "ParallelMap[function, list]".to_string(),
            parameters: vec![
                ParameterInfo {
                    name: "function".to_string(),
                    description: "Function to apply to each element".to_string(),
                    param_type: "Function".to_string(),
                    optional: false,
                    default_value: None,
                },
                ParameterInfo {
                    name: "list".to_string(),
                    description: "List of elements to process".to_string(),
                    param_type: "List".to_string(),
                    optional: false,
                    default_value: None,
                },
            ],
            return_type: "List".to_string(),
            examples: vec![
                ExampleUsage {
                    code: "ParallelMap[Square, {1, 2, 3, 4}]".to_string(),
                    description: "Square numbers in parallel".to_string(),
                    expected_output: "{1, 4, 9, 16}".to_string(),
                },
            ],
            related_functions: vec!["Map".to_string(), "ParallelReduce".to_string(), "ThreadPool".to_string()],
            tags: vec!["concurrency".to_string(), "parallel".to_string(), "map".to_string()],
            version: "1.0".to_string(),
        });
    }
    
    fn add_function_metadata(&mut self, metadata: FunctionMetadata) {
        let name = metadata.name.clone();
        let category = metadata.category.clone();
        
        // Add to categories index
        self.categories.entry(category).or_insert_with(Vec::new).push(name.clone());
        
        // Add to tags index
        for tag in &metadata.tags {
            self.tags.entry(tag.clone()).or_insert_with(Vec::new).push(name.clone());
        }
        
        // Add to main functions map
        self.functions.insert(name, metadata);
    }
    
    pub fn get_function(&self, name: &str) -> Option<&FunctionMetadata> {
        self.functions.get(name)
    }
    
    pub fn get_category(&self, category: &str) -> Option<&Vec<String>> {
        self.categories.get(category)
    }
    
    pub fn search_functions(&self, keyword: &str) -> Vec<String> {
        let keyword_lower = keyword.to_lowercase();
        self.functions.values()
            .filter(|func| {
                func.name.to_lowercase().contains(&keyword_lower) ||
                func.description.to_lowercase().contains(&keyword_lower) ||
                func.tags.iter().any(|tag| tag.to_lowercase().contains(&keyword_lower))
            })
            .map(|func| func.name.clone())
            .collect()
    }
    
    pub fn list_functions_by_pattern(&self, pattern: &str) -> Vec<String> {
        if pattern.ends_with('*') {
            let prefix = &pattern[..pattern.len()-1];
            self.functions.keys()
                .filter(|name| name.starts_with(prefix))
                .cloned()
                .collect()
        } else if pattern.starts_with('*') {
            let suffix = &pattern[1..];
            self.functions.keys()
                .filter(|name| name.ends_with(suffix))
                .cloned()
                .collect()
        } else {
            // Exact match or regex pattern
            match Regex::new(pattern) {
                Ok(regex) => self.functions.keys()
                    .filter(|name| regex.is_match(name))
                    .cloned()
                    .collect(),
                Err(_) => {
                    if self.functions.contains_key(pattern) {
                        vec![pattern.to_string()]
                    } else {
                        vec![]
                    }
                }
            }
        }
    }
    
    pub fn get_all_categories(&self) -> Vec<String> {
        let mut categories: Vec<String> = self.categories.keys().cloned().collect();
        categories.sort();
        categories
    }
    
    pub fn get_all_function_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.functions.keys().cloned().collect();
        names.sort();
        names
    }
}

/// Code generator for multiple target languages
pub struct CodeGenerator {
    templates: HashMap<String, HashMap<String, String>>, // language -> function -> template
}

impl CodeGenerator {
    pub fn new() -> Self {
        let mut generator = CodeGenerator {
            templates: HashMap::new(),
        };
        generator.initialize_templates();
        generator
    }
    
    fn initialize_templates(&mut self) {
        // Python templates
        let mut python_templates = HashMap::new();
        python_templates.insert("Map".to_string(), "list(map({func}, {list}))".to_string());
        python_templates.insert("Length".to_string(), "len({arg})".to_string());
        python_templates.insert("StringJoin".to_string(), "\"{sep}\".join({list})".to_string());
        python_templates.insert("Range".to_string(), "list(range({start}, {end}))".to_string());
        python_templates.insert("Sin".to_string(), "math.sin({x})".to_string());
        
        self.templates.insert("Python".to_string(), python_templates);
        
        // JavaScript templates
        let mut js_templates = HashMap::new();
        js_templates.insert("Map".to_string(), "{list}.map({func})".to_string());
        js_templates.insert("Length".to_string(), "{arg}.length".to_string());
        js_templates.insert("StringJoin".to_string(), "{list}.join({sep})".to_string());
        js_templates.insert("Range".to_string(), "[...Array({end} - {start}).keys()].map(i => i + {start})".to_string());
        js_templates.insert("Sin".to_string(), "Math.sin({x})".to_string());
        
        self.templates.insert("JavaScript".to_string(), js_templates);
        
        // Mathematica templates (for roundtrip compatibility)
        let mut mathematica_templates = HashMap::new();
        mathematica_templates.insert("Map".to_string(), "Map[{func}, {list}]".to_string());
        mathematica_templates.insert("Length".to_string(), "Length[{arg}]".to_string());
        mathematica_templates.insert("StringJoin".to_string(), "StringJoin[{list}]".to_string());
        mathematica_templates.insert("Range".to_string(), "Range[{start}, {end}]".to_string());
        mathematica_templates.insert("Sin".to_string(), "Sin[{x}]".to_string());
        
        self.templates.insert("Mathematica".to_string(), mathematica_templates);
    }
    
    pub fn generate_code(&self, expression: &str, target_language: &str) -> String {
        // Simple expression parsing and conversion
        // In a full implementation, this would parse the AST
        if let Some(_lang_templates) = self.templates.get(target_language) {
            // For now, return a placeholder implementation
            match target_language {
                "Python" => self.convert_to_python(expression),
                "JavaScript" => self.convert_to_javascript(expression),
                "Mathematica" => expression.to_string(), // Already in Mathematica format
                _ => format!("// Unsupported language: {}", target_language),
            }
        } else {
            format!("// Unsupported language: {}", target_language)
        }
    }
    
    fn convert_to_python(&self, expr: &str) -> String {
        // Simple pattern matching for common expressions
        if expr.contains("Map[") {
            "# Map operation\n# map(function, iterable)".to_string()
        } else if expr.contains("StringTemplate[") {
            "# String template\n# f\"Hello {name}\"".to_string()
        } else {
            format!("# Lyra expression: {}", expr)
        }
    }
    
    fn convert_to_javascript(&self, expr: &str) -> String {
        if expr.contains("Map[") {
            "// Map operation\n// array.map(function)".to_string()
        } else if expr.contains("StringTemplate[") {
            "// String template\n// `Hello ${name}`".to_string()
        } else {
            format!("// Lyra expression: {}", expr)
        }
    }
}

/// Documentation generator for multiple output formats
pub struct DocumentationGenerator {
    registry: FunctionRegistry,
}

impl DocumentationGenerator {
    pub fn new(registry: FunctionRegistry) -> Self {
        DocumentationGenerator { registry }
    }
    
    pub fn generate_help_text(&self, function_name: &str) -> String {
        if let Some(metadata) = self.registry.get_function(function_name) {
            let mut help = String::new();
            help.push_str(&format!("Function: {}\n", metadata.name));
            help.push_str(&format!("Category: {}\n", metadata.category));
            help.push_str(&format!("Signature: {}\n", metadata.signature));
            help.push_str(&format!("Description: {}\n\n", metadata.description));
            
            if !metadata.parameters.is_empty() {
                help.push_str("Parameters:\n");
                for param in &metadata.parameters {
                    help.push_str(&format!("  {} ({}): {}\n", 
                        param.name, param.param_type, param.description));
                }
                help.push('\n');
            }
            
            if !metadata.examples.is_empty() {
                help.push_str("Examples:\n");
                for example in &metadata.examples {
                    help.push_str(&format!("  {}\n", example.code));
                    help.push_str(&format!("    → {}\n", example.expected_output));
                }
                help.push('\n');
            }
            
            if !metadata.related_functions.is_empty() {
                help.push_str("Related functions: ");
                help.push_str(&metadata.related_functions.join(", "));
                help.push('\n');
            }
            
            help
        } else {
            format!("Function '{}' not found. Use FunctionList[] to see available functions.", function_name)
        }
    }
    
    pub fn generate_category_help(&self, category: &str) -> String {
        if let Some(functions) = self.registry.get_category(category) {
            let mut help = format!("Functions in category '{}':\n\n", category);
            for func_name in functions {
                if let Some(metadata) = self.registry.get_function(func_name) {
                    help.push_str(&format!("  {} - {}\n", metadata.name, metadata.description));
                }
            }
            help
        } else {
            let categories = self.registry.get_all_categories();
            format!("Category '{}' not found. Available categories: {}", category, categories.join(", "))
        }
    }
    
    pub fn generate_general_help(&self) -> String {
        let mut help = String::from("Lyra Symbolic Computation Engine Help\n");
        help.push_str("====================================\n\n");
        help.push_str("Available function categories:\n");
        
        for category in self.registry.get_all_categories() {
            if let Some(functions) = self.registry.get_category(&category) {
                help.push_str(&format!("  {} ({} functions)\n", category, functions.len()));
            }
        }
        
        help.push_str("\nUsage:\n");
        help.push_str("  Help[\"function_name\"] - Get help for a specific function\n");
        help.push_str("  Help[\"category\"] - Get help for a function category\n");
        help.push_str("  FunctionList[] - List all available functions\n");
        help.push_str("  FunctionList[\"pattern*\"] - List functions matching pattern\n");
        help.push_str("  FunctionSearch[\"keyword\"] - Search functions by keyword\n");
        
        help
    }
    
    pub fn generate_markdown(&self, functions: &[String]) -> String {
        let mut markdown = String::from("# Lyra Function Reference\n\n");
        
        for func_name in functions {
            if let Some(metadata) = self.registry.get_function(func_name) {
                markdown.push_str(&format!("## {}\n\n", metadata.name));
                markdown.push_str(&format!("**Category:** {}\n\n", metadata.category));
                markdown.push_str(&format!("**Signature:** `{}`\n\n", metadata.signature));
                markdown.push_str(&format!("{}\n\n", metadata.description));
                
                if !metadata.parameters.is_empty() {
                    markdown.push_str("### Parameters\n\n");
                    for param in &metadata.parameters {
                        markdown.push_str(&format!("- **{}** ({}): {}\n", 
                            param.name, param.param_type, param.description));
                    }
                    markdown.push_str("\n");
                }
                
                if !metadata.examples.is_empty() {
                    markdown.push_str("### Examples\n\n");
                    for example in &metadata.examples {
                        markdown.push_str("```wolfram\n");
                        markdown.push_str(&example.code);
                        markdown.push_str("\n```\n\n");
                        markdown.push_str(&format!("Output: `{}`\n\n", example.expected_output));
                    }
                }
                
                if !metadata.related_functions.is_empty() {
                    markdown.push_str("### Related Functions\n\n");
                    for related in &metadata.related_functions {
                        markdown.push_str(&format!("[{}](#{}), ", related, related.to_lowercase()));
                    }
                    markdown.push_str("\n\n");
                }
                
                markdown.push_str("---\n\n");
            }
        }
        
        markdown
    }
}

/// Global instances for the documentation system
lazy_static::lazy_static! {
    static ref FUNCTION_REGISTRY: FunctionRegistry = FunctionRegistry::new();
    static ref CODE_GENERATOR: CodeGenerator = CodeGenerator::new();
    static ref DOC_GENERATOR: DocumentationGenerator = DocumentationGenerator::new(FUNCTION_REGISTRY.clone());
}

// Help System Functions

/// General help and function categories
pub fn help(args: &[Value]) -> VmResult<Value> {
    match args.len() {
        0 => {
            let help_text = DOC_GENERATOR.generate_general_help();
            Ok(Value::String(help_text))
        }
        1 => {
            if let Value::String(query) = &args[0] {
                // Check if it's a function name or category
                if FUNCTION_REGISTRY.functions.contains_key(query) {
                    let help_text = DOC_GENERATOR.generate_help_text(query);
                    Ok(Value::String(help_text))
                } else if FUNCTION_REGISTRY.categories.contains_key(query) {
                    let help_text = DOC_GENERATOR.generate_category_help(query);
                    Ok(Value::String(help_text))
                } else {
                    // Try searching
                    let matches = FUNCTION_REGISTRY.search_functions(query);
                    if matches.is_empty() {
                        Ok(Value::String(format!("No functions found for query: '{}'", query)))
                    } else {
                        let mut result = format!("Functions matching '{}':\n", query);
                        for func_name in matches {
                            if let Some(metadata) = FUNCTION_REGISTRY.get_function(&func_name) {
                                result.push_str(&format!("  {} - {}\n", func_name, metadata.description));
                            }
                        }
                        Ok(Value::String(result))
                    }
                }
            } else {
                Err(VmError::TypeError {
                    expected: "String".to_string(),
                    actual: format!("{:?}", args[0]),
                })
            }
        }
        _ => Err(VmError::Runtime("Help takes 0 or 1 arguments".to_string())),
    }
}

/// Get specific function documentation and metadata
pub fn function_info(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("FunctionInfo requires exactly 1 argument".to_string()));
    }
    
    if let Value::String(func_name) = &args[0] {
        if let Some(metadata) = FUNCTION_REGISTRY.get_function(func_name) {
            // Return metadata as a structured value
            let mut info = Vec::new();
            info.push(Value::List(vec![Value::String("Name".to_string()), Value::String(metadata.name.clone())]));
            info.push(Value::List(vec![Value::String("Category".to_string()), Value::String(metadata.category.clone())]));
            info.push(Value::List(vec![Value::String("Description".to_string()), Value::String(metadata.description.clone())]));
            info.push(Value::List(vec![Value::String("Signature".to_string()), Value::String(metadata.signature.clone())]));
            info.push(Value::List(vec![Value::String("ReturnType".to_string()), Value::String(metadata.return_type.clone())]));
            
            let related_funcs: Vec<Value> = metadata.related_functions.iter()
                .map(|f| Value::String(f.clone()))
                .collect();
            info.push(Value::List(vec![Value::String("RelatedFunctions".to_string()), Value::List(related_funcs)]));
            
            Ok(Value::List(info))
        } else {
            Err(VmError::Runtime(format!("Function '{}' not found", func_name)))
        }
    } else {
        Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        })
    }
}

/// Get usage examples for a function
pub fn example_usage(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("ExampleUsage requires exactly 1 argument".to_string()));
    }
    
    if let Value::String(func_name) = &args[0] {
        if let Some(metadata) = FUNCTION_REGISTRY.get_function(func_name) {
            let examples: Vec<Value> = metadata.examples.iter()
                .map(|ex| Value::List(vec![
                    Value::String(ex.code.clone()),
                    Value::String(ex.description.clone()),
                    Value::String(ex.expected_output.clone()),
                ]))
                .collect();
            Ok(Value::List(examples))
        } else {
            Err(VmError::Runtime(format!("Function '{}' not found", func_name)))
        }
    } else {
        Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        })
    }
}

// Function Discovery Functions

/// List all available functions or functions matching a pattern
pub fn function_list(args: &[Value]) -> VmResult<Value> {
    match args.len() {
        0 => {
            let functions: Vec<Value> = FUNCTION_REGISTRY.get_all_function_names()
                .into_iter()
                .map(Value::String)
                .collect();
            Ok(Value::List(functions))
        }
        1 => {
            if let Value::String(pattern) = &args[0] {
                let matching_functions: Vec<Value> = FUNCTION_REGISTRY.list_functions_by_pattern(pattern)
                    .into_iter()
                    .map(Value::String)
                    .collect();
                Ok(Value::List(matching_functions))
            } else {
                Err(VmError::TypeError {
                    expected: "String".to_string(),
                    actual: format!("{:?}", args[0]),
                })
            }
        }
        _ => Err(VmError::Runtime("FunctionList takes 0 or 1 arguments".to_string())),
    }
}

/// Search functions by keyword
pub fn function_search(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("FunctionSearch requires exactly 1 argument".to_string()));
    }
    
    if let Value::String(keyword) = &args[0] {
        let matching_functions: Vec<Value> = FUNCTION_REGISTRY.search_functions(keyword)
            .into_iter()
            .map(Value::String)
            .collect();
        Ok(Value::List(matching_functions))
    } else {
        Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        })
    }
}

/// Get functions by category
pub fn functions_by_category(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("FunctionsByCategory requires exactly 1 argument".to_string()));
    }
    
    if let Value::String(category) = &args[0] {
        if let Some(functions) = FUNCTION_REGISTRY.get_category(category) {
            let function_values: Vec<Value> = functions.iter()
                .map(|f| Value::String(f.clone()))
                .collect();
            Ok(Value::List(function_values))
        } else {
            Ok(Value::List(vec![]))
        }
    } else {
        Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        })
    }
}

// Code Generation Functions

/// Generate code in target language
pub fn generate_code(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime("GenerateCode requires exactly 2 arguments: expression, language".to_string()));
    }
    
    let expression = match &args[0] {
        Value::String(s) => s.clone(),
        other => format!("{:?}", other), // Convert other values to string representation
    };
    
    if let Value::String(language) = &args[1] {
        let generated_code = CODE_GENERATOR.generate_code(&expression, language);
        Ok(Value::String(generated_code))
    } else {
        Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[1]),
        })
    }
}

/// Generate documentation in specified format
pub fn generate_documentation(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime("GenerateDocumentation requires exactly 2 arguments: functions, format".to_string()));
    }
    
    let function_names = match &args[0] {
        Value::List(items) => {
            let mut names = Vec::new();
            for item in items {
                if let Value::String(name) = item {
                    names.push(name.clone());
                } else {
                    return Err(VmError::TypeError {
                        expected: "List[String]".to_string(),
                        actual: format!("List containing {:?}", item),
                    });
                }
            }
            names
        }
        _ => return Err(VmError::TypeError {
            expected: "List[String]".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    if let Value::String(format) = &args[1] {
        let documentation = match format.as_str() {
            "Markdown" => DOC_GENERATOR.generate_markdown(&function_names),
            "Text" => {
                let mut text = String::new();
                for func_name in function_names {
                    text.push_str(&DOC_GENERATOR.generate_help_text(&func_name));
                    text.push_str("\n\n");
                }
                text
            }
            _ => return Err(VmError::Runtime(format!("Unsupported format: {}", format))),
        };
        
        Ok(Value::String(documentation))
    } else {
        Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[1]),
        })
    }
}

/// Code templates system
pub fn code_template(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime("CodeTemplate requires exactly 2 arguments: template_name, parameters".to_string()));
    }
    
    if let Value::String(template_name) = &args[0] {
        match template_name.as_str() {
            "MapReduce" => Ok(Value::String(
                "// Map-Reduce Template\n\
                 data = {1, 2, 3, 4, 5};\n\
                 mapped = ParallelMap[function, data];\n\
                 result = ParallelReduce[Add, mapped];".to_string()
            )),
            "WebAPI" => Ok(Value::String(
                "// Web API Template\n\
                 response = HTTPGet[\"https://api.example.com/data\"];\n\
                 data = JSONParse[response];\n\
                 result = DataFilter[data, condition];".to_string()
            )),
            "DataProcessing" => Ok(Value::String(
                "// Data Processing Template\n\
                 raw_data = Import[\"data.csv\"];\n\
                 cleaned = DataFilter[raw_data, IsValidData];\n\
                 transformed = DataTransform[cleaned, transformation_rules];\n\
                 Export[transformed, \"output.csv\"];".to_string()
            )),
            _ => Err(VmError::Runtime(format!("Unknown template: {}", template_name))),
        }
    } else {
        Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        })
    }
}

/// Generate API reference for a module
pub fn api_reference(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("APIReference requires exactly 1 argument".to_string()));
    }
    
    if let Value::String(module) = &args[0] {
        let mut reference = format!("# API Reference: {}\n\n", module);
        
        // Get all functions for the module/category
        let functions = match module.as_str() {
            "stdlib" => FUNCTION_REGISTRY.get_all_function_names(),
            category => FUNCTION_REGISTRY.get_category(category)
                .map(|funcs| funcs.clone())
                .unwrap_or_else(Vec::new),
        };
        
        // Generate documentation for each function
        reference.push_str(&DOC_GENERATOR.generate_markdown(&functions));
        
        Ok(Value::String(reference))
    } else {
        Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        })
    }
}

/// Generate function stub in target language
pub fn generate_stub(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime("GenerateStub requires exactly 2 arguments: function_name, language".to_string()));
    }
    
    if let (Value::String(func_name), Value::String(language)) = (&args[0], &args[1]) {
        if let Some(metadata) = FUNCTION_REGISTRY.get_function(func_name) {
            let stub = match language.as_str() {
                "Python" => {
                    let mut params: Vec<String> = metadata.parameters.iter()
                        .map(|p| p.name.clone())
                        .collect();
                    if params.is_empty() {
                        params.push("*args".to_string());
                    }
                    format!(
                        "def {}({}):\n    \"\"\"{}\"\"\"\n    # TODO: Implement\n    pass\n",
                        func_name.to_lowercase(),
                        params.join(", "),
                        metadata.description
                    )
                }
                "JavaScript" => {
                    let params: Vec<String> = metadata.parameters.iter()
                        .map(|p| p.name.clone())
                        .collect();
                    format!(
                        "function {}({}) {{\n    // {}\n    // TODO: Implement\n}}\n",
                        func_name,
                        params.join(", "),
                        metadata.description
                    )
                }
                _ => format!("// Stub generation not supported for {}", language),
            };
            Ok(Value::String(stub))
        } else {
            Err(VmError::Runtime(format!("Function '{}' not found", func_name)))
        }
    } else {
        Err(VmError::TypeError {
            expected: "String, String".to_string(),
            actual: format!("{:?}, {:?}", args[0], args[1]),
        })
    }
}

/// Auto-generate usage examples for a function
pub fn generate_examples(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("GenerateExamples requires exactly 1 argument".to_string()));
    }
    
    if let Value::String(func_name) = &args[0] {
        if let Some(metadata) = FUNCTION_REGISTRY.get_function(func_name) {
            // For now, return existing examples
            // In a full implementation, this could generate new examples based on function signature
            let examples: Vec<Value> = metadata.examples.iter()
                .map(|ex| Value::String(format!("{} // → {}", ex.code, ex.expected_output)))
                .collect();
            Ok(Value::List(examples))
        } else {
            // Generate basic examples based on function name patterns
            let basic_examples = match func_name.as_str() {
                name if name.starts_with("String") => vec![
                    format!("{}[\"example string\"]", func_name),
                    format!("{}[\"hello\", \"world\"]", func_name),
                ],
                name if name.contains("List") => vec![
                    format!("{}[{{1, 2, 3, 4}}]", func_name),
                    format!("{}[{{\"a\", \"b\", \"c\"}}]", func_name),
                ],
                _ => vec![
                    format!("{}[parameter]", func_name),
                    format!("{}[arg1, arg2]", func_name),
                ],
            };
            
            let examples: Vec<Value> = basic_examples.into_iter()
                .map(Value::String)
                .collect();
            Ok(Value::List(examples))
        }
    } else {
        Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        })
    }
}

/// Export as Jupyter notebook format
pub fn export_notebook(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime("ExportNotebook requires exactly 2 arguments: expressions, format".to_string()));
    }
    
    let expressions = match &args[0] {
        Value::List(items) => {
            let mut exprs = Vec::new();
            for item in items {
                if let Value::String(expr) = item {
                    exprs.push(expr.clone());
                } else {
                    exprs.push(format!("{:?}", item));
                }
            }
            exprs
        }
        _ => return Err(VmError::TypeError {
            expected: "List".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    if let Value::String(format) = &args[1] {
        match format.as_str() {
            "Jupyter" => {
                let mut notebook = String::from("{\n  \"cells\": [\n");
                for (i, expr) in expressions.iter().enumerate() {
                    notebook.push_str(&format!(
                        "    {{\n      \"cell_type\": \"code\",\n      \"source\": [\"{}\"],\n      \"outputs\": []\n    }}",
                        expr.replace("\"", "\\\"")
                    ));
                    if i < expressions.len() - 1 {
                        notebook.push(',');
                    }
                    notebook.push('\n');
                }
                notebook.push_str("  ],\n  \"metadata\": {},\n  \"nbformat\": 4,\n  \"nbformat_minor\": 4\n}");
                Ok(Value::String(notebook))
            }
            "HTML" => {
                let mut html = String::from("<html><head><title>Lyra Notebook</title></head><body>\n");
                for expr in expressions {
                    html.push_str(&format!("<div class=\"cell\"><code>{}</code></div>\n", expr));
                }
                html.push_str("</body></html>");
                Ok(Value::String(html))
            }
            _ => Err(VmError::Runtime(format!("Unsupported notebook format: {}", format))),
        }
    } else {
        Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[1]),
        })
    }
}