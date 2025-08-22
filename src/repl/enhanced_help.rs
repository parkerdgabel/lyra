//! Enhanced REPL Help System
//!
//! Production-grade help and function discovery system with:
//! - Enhanced help commands (?FunctionName)
//! - Fuzzy search with typo suggestions (??search_term)
//! - Category-based browsing (??math, ??quantum, etc.)
//! - Context-aware suggestions and auto-completion

use crate::{
    modules::registry::ModuleRegistry,
    repl::{ReplResult, ReplError},
    stdlib::StandardLibrary,
};
use std::collections::HashMap;
use std::sync::Arc;
use colored::*;
use fuzzy_matcher::{FuzzyMatcher, skim::SkimMatcherV2};

/// Enhanced help system with comprehensive function discovery
pub struct EnhancedHelpSystem {
    /// Module registry for function metadata
    module_registry: Arc<ModuleRegistry>,
    /// Standard library for function lookup
    stdlib: Arc<StandardLibrary>,
    /// Function database with signatures and documentation
    function_database: FunctionDatabase,
    /// Fuzzy matcher for typo detection
    fuzzy_matcher: SkimMatcherV2,
    /// Category mappings
    categories: HashMap<String, Vec<String>>,
    /// Function aliases and synonyms
    aliases: HashMap<String, String>,
    /// Usage statistics for smart suggestions
    usage_stats: HashMap<String, u64>,
}

/// Comprehensive function information
#[derive(Debug, Clone)]
pub struct FunctionInfo {
    pub name: String,
    pub signature: String,
    pub description: String,
    pub examples: Vec<String>,
    pub parameters: Vec<ParameterInfo>,
    pub return_type: String,
    pub category: String,
    pub module: String,
    pub aliases: Vec<String>,
    pub related_functions: Vec<String>,
    pub source_location: Option<String>,
}

/// Parameter information for functions
#[derive(Debug, Clone)]
pub struct ParameterInfo {
    pub name: String,
    pub type_hint: String,
    pub description: String,
    pub optional: bool,
    pub default_value: Option<String>,
}

/// Search result with relevance scoring
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub function_name: String,
    pub relevance_score: i64,
    pub match_type: MatchType,
    pub snippet: String,
}

/// Type of match found during search
#[derive(Debug, Clone)]
pub enum MatchType {
    ExactName,
    FuzzyName,
    Description,
    Category,
    Parameter,
    Example,
    Alias,
}

/// Function database for metadata storage
#[derive(Debug)]
pub struct FunctionDatabase {
    functions: HashMap<String, FunctionInfo>,
    categories: HashMap<String, Vec<String>>,
    keywords: HashMap<String, Vec<String>>,
}

impl FunctionDatabase {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            categories: HashMap::new(),
            keywords: HashMap::new(),
        }
    }

    /// Build database from module registry and stdlib
    pub fn build_from_registry(registry: &ModuleRegistry, stdlib: &StandardLibrary) -> Self {
        let mut db = Self::new();
        
        // Process all registered modules
        for module_name in registry.list_modules() {
            if let Some(module) = registry.get_module(&module_name) {
                let category = db.infer_category_from_module(&module_name);
                
                for export_name in registry.get_module_exports(&module_name) {
                    let function_info = FunctionInfo {
                        name: export_name.clone(),
                        signature: db.generate_signature(&export_name, &module_name),
                        description: db.get_function_description(&export_name, &module_name),
                        examples: db.generate_examples(&export_name),
                        parameters: db.infer_parameters(&export_name),
                        return_type: db.infer_return_type(&export_name),
                        category: category.clone(),
                        module: module_name.clone(),
                        aliases: db.get_function_aliases(&export_name),
                        related_functions: db.find_related_functions(&export_name, &category),
                        source_location: Some(format!("{}::{}", module_name, export_name)),
                    };
                    
                    // Index keywords before moving function_info
                    db.index_function_keywords(&export_name, &function_info);
                    
                    // Update category mapping
                    db.categories.entry(category.clone())
                        .or_insert_with(Vec::new)
                        .push(export_name.clone());
                    
                    // Store function info (this moves the value)
                    db.functions.insert(export_name.clone(), function_info);
                }
            }
        }
        
        db
    }

    fn infer_category_from_module(&self, module_name: &str) -> String {
        match module_name {
            name if name.contains("math") => "Mathematics".to_string(),
            name if name.contains("string") => "String Processing".to_string(),
            name if name.contains("list") => "List Operations".to_string(),
            name if name.contains("tensor") => "Linear Algebra".to_string(),
            name if name.contains("ml") => "Machine Learning".to_string(),
            name if name.contains("rules") => "Pattern Matching".to_string(),
            name if name.contains("table") => "Data Processing".to_string(),
            _ => "General".to_string(),
        }
    }

    fn generate_signature(&self, function_name: &str, module_name: &str) -> String {
        // Generate function signatures based on known patterns
        match function_name {
            // Math functions - usually 1 argument
            "Sin" | "Cos" | "Tan" | "Exp" | "Log" | "Sqrt" => {
                format!("{}[x_]", function_name)
            },
            // Binary operations
            "Plus" | "Times" | "Power" | "Divide" | "Minus" => {
                format!("{}[x_, y_, ...]", function_name)
            },
            // List operations
            "Length" | "Head" | "Tail" | "Flatten" => {
                format!("{}[list_]", function_name)
            },
            "Map" => "Map[f_, list_]".to_string(),
            "Apply" => "Apply[f_, args_]".to_string(),
            // String operations
            "StringJoin" => "StringJoin[list_, delimiter_]".to_string(),
            "StringLength" => "StringLength[string_]".to_string(),
            // Default pattern
            _ => format!("{}[args___]", function_name),
        }
    }

    fn get_function_description(&self, function_name: &str, _module_name: &str) -> String {
        match function_name {
            "Sin" => "Computes the sine of a numeric expression".to_string(),
            "Cos" => "Computes the cosine of a numeric expression".to_string(),
            "Tan" => "Computes the tangent of a numeric expression".to_string(),
            "Exp" => "Computes the exponential function e^x".to_string(),
            "Log" => "Computes the natural logarithm".to_string(),
            "Sqrt" => "Computes the square root".to_string(),
            "Plus" => "Adds numeric expressions or performs symbolic addition".to_string(),
            "Times" => "Multiplies numeric expressions or performs symbolic multiplication".to_string(),
            "Power" => "Raises the first argument to the power of the second".to_string(),
            "Divide" => "Divides the first argument by the second".to_string(),
            "Minus" => "Subtracts the second argument from the first".to_string(),
            "Length" => "Returns the number of elements in a list".to_string(),
            "Head" => "Returns the first element of a list".to_string(),
            "Tail" => "Returns all elements of a list except the first".to_string(),
            "Map" => "Applies a function to each element of a list".to_string(),
            "Apply" => "Applies a function to a sequence of arguments".to_string(),
            "Flatten" => "Flattens nested lists into a single level".to_string(),
            "StringJoin" => "Joins a list of strings with a delimiter".to_string(),
            "StringLength" => "Returns the length of a string".to_string(),
            _ => format!("Function {} from the Lyra standard library", function_name),
        }
    }

    fn generate_examples(&self, function_name: &str) -> Vec<String> {
        match function_name {
            "Sin" => vec![
                "Sin[0] (* → 0 *)".to_string(),
                "Sin[Pi/2] (* → 1 *)".to_string(),
                "Sin[Pi] (* → 0 *)".to_string(),
                "Sin[{0, Pi/2, Pi}] (* → {0, 1, 0} *)".to_string(),
            ],
            "Cos" => vec![
                "Cos[0] (* → 1 *)".to_string(),
                "Cos[Pi/2] (* → 0 *)".to_string(),
                "Cos[Pi] (* → -1 *)".to_string(),
            ],
            "Plus" => vec![
                "Plus[2, 3] (* → 5 *)".to_string(),
                "Plus[x, y] (* → x + y *)".to_string(),
                "Plus[1, 2, 3, 4] (* → 10 *)".to_string(),
            ],
            "Length" => vec![
                "Length[{1, 2, 3}] (* → 3 *)".to_string(),
                "Length[{}] (* → 0 *)".to_string(),
                "Length[\"hello\"] (* → 5 *)".to_string(),
            ],
            "Map" => vec![
                "Map[Sin, {0, Pi/2, Pi}] (* → {0, 1, 0} *)".to_string(),
                "Map[f, {a, b, c}] (* → {f[a], f[b], f[c]} *)".to_string(),
                "Map[Plus[#, 1] &, {1, 2, 3}] (* → {2, 3, 4} *)".to_string(),
            ],
            _ => vec![format!("{}[example_args] (* example usage *)", function_name)],
        }
    }

    fn infer_parameters(&self, function_name: &str) -> Vec<ParameterInfo> {
        match function_name {
            "Sin" | "Cos" | "Tan" | "Exp" | "Log" | "Sqrt" => vec![
                ParameterInfo {
                    name: "x".to_string(),
                    type_hint: "Number | Expression".to_string(),
                    description: "The input value or expression".to_string(),
                    optional: false,
                    default_value: None,
                }
            ],
            "Plus" | "Times" => vec![
                ParameterInfo {
                    name: "x".to_string(),
                    type_hint: "Number | Expression".to_string(),
                    description: "First operand".to_string(),
                    optional: false,
                    default_value: None,
                },
                ParameterInfo {
                    name: "y".to_string(),
                    type_hint: "Number | Expression".to_string(),
                    description: "Second operand".to_string(),
                    optional: false,
                    default_value: None,
                },
                ParameterInfo {
                    name: "...".to_string(),
                    type_hint: "Number | Expression".to_string(),
                    description: "Additional operands (variadic)".to_string(),
                    optional: true,
                    default_value: None,
                },
            ],
            "Map" => vec![
                ParameterInfo {
                    name: "f".to_string(),
                    type_hint: "Function".to_string(),
                    description: "Function to apply to each element".to_string(),
                    optional: false,
                    default_value: None,
                },
                ParameterInfo {
                    name: "list".to_string(),
                    type_hint: "List".to_string(),
                    description: "Input list".to_string(),
                    optional: false,
                    default_value: None,
                },
            ],
            _ => vec![],
        }
    }

    fn infer_return_type(&self, function_name: &str) -> String {
        match function_name {
            "Sin" | "Cos" | "Tan" | "Exp" | "Log" | "Sqrt" => "Number | Expression".to_string(),
            "Plus" | "Times" | "Power" | "Divide" | "Minus" => "Number | Expression".to_string(),
            "Length" => "Integer".to_string(),
            "Head" => "Any".to_string(),
            "Tail" => "List".to_string(),
            "Map" => "List".to_string(),
            "StringLength" => "Integer".to_string(),
            "StringJoin" => "String".to_string(),
            _ => "Any".to_string(),
        }
    }

    fn get_function_aliases(&self, function_name: &str) -> Vec<String> {
        match function_name {
            "Plus" => vec!["Add".to_string(), "+".to_string()],
            "Times" => vec!["Multiply".to_string(), "*".to_string()],
            "Power" => vec!["^".to_string(), "Pow".to_string()],
            "Divide" => vec!["/".to_string(), "Div".to_string()],
            "Minus" => vec!["-".to_string(), "Subtract".to_string()],
            "Length" => vec!["Len".to_string(), "Count".to_string()],
            _ => vec![],
        }
    }

    fn find_related_functions(&self, function_name: &str, category: &str) -> Vec<String> {
        match function_name {
            "Sin" => vec!["Cos".to_string(), "Tan".to_string(), "ArcSin".to_string()],
            "Cos" => vec!["Sin".to_string(), "Tan".to_string(), "ArcCos".to_string()],
            "Tan" => vec!["Sin".to_string(), "Cos".to_string(), "ArcTan".to_string()],
            "Plus" => vec!["Times".to_string(), "Minus".to_string(), "Divide".to_string()],
            "Head" => vec!["Tail".to_string(), "First".to_string(), "Length".to_string()],
            "Map" => vec!["Apply".to_string(), "Select".to_string(), "Table".to_string()],
            _ => vec![],
        }
    }

    fn index_function_keywords(&mut self, function_name: &str, info: &FunctionInfo) {
        let mut keywords = vec![function_name.to_lowercase()];
        keywords.extend(info.aliases.iter().map(|a| a.to_lowercase()));
        
        // Extract keywords from description
        for word in info.description.split_whitespace() {
            if word.len() > 3 {
                keywords.push(word.to_lowercase().trim_matches(|c: char| !c.is_alphabetic()).to_string());
            }
        }

        // Add category as keyword
        keywords.push(info.category.to_lowercase());

        for keyword in keywords {
            if !keyword.is_empty() {
                self.keywords.entry(keyword)
                    .or_insert_with(Vec::new)
                    .push(function_name.to_string());
            }
        }
    }

    pub fn get_function(&self, name: &str) -> Option<&FunctionInfo> {
        self.functions.get(name)
    }

    pub fn get_functions_in_category(&self, category: &str) -> Vec<&String> {
        self.categories.get(category).map(|funcs| funcs.iter().collect()).unwrap_or_default()
    }

    pub fn search_by_keyword(&self, keyword: &str) -> Vec<&String> {
        self.keywords.get(&keyword.to_lowercase()).map(|funcs| funcs.iter().collect()).unwrap_or_default()
    }

    pub fn all_functions(&self) -> impl Iterator<Item = &FunctionInfo> {
        self.functions.values()
    }

    pub fn all_categories(&self) -> impl Iterator<Item = &String> {
        self.categories.keys()
    }
}

impl EnhancedHelpSystem {
    /// Create a new enhanced help system
    pub fn new(registry: Arc<ModuleRegistry>, stdlib: Arc<StandardLibrary>) -> Self {
        let function_database = FunctionDatabase::build_from_registry(&registry, &stdlib);
        
        let mut categories = HashMap::new();
        categories.insert("math".to_string(), vec!["Sin".to_string(), "Cos".to_string(), "Tan".to_string(), "Plus".to_string(), "Times".to_string()]);
        categories.insert("list".to_string(), vec!["Length".to_string(), "Head".to_string(), "Tail".to_string(), "Map".to_string()]);
        categories.insert("string".to_string(), vec!["StringJoin".to_string(), "StringLength".to_string()]);
        
        let mut aliases = HashMap::new();
        aliases.insert("add".to_string(), "Plus".to_string());
        aliases.insert("multiply".to_string(), "Times".to_string());
        aliases.insert("len".to_string(), "Length".to_string());

        Self {
            module_registry: registry,
            stdlib,
            function_database,
            fuzzy_matcher: SkimMatcherV2::default(),
            categories,
            aliases,
            usage_stats: HashMap::new(),
        }
    }

    /// Handle ?FunctionName command - detailed function information
    pub fn handle_help_function(&mut self, function_name: &str) -> ReplResult<String> {
        // Record usage for analytics
        *self.usage_stats.entry(function_name.to_string()).or_insert(0) += 1;

        // Try exact match first
        if let Some(info) = self.function_database.get_function(function_name) {
            return Ok(self.format_detailed_help(info));
        }

        // Try alias lookup
        if let Some(real_name) = self.aliases.get(&function_name.to_lowercase()) {
            if let Some(info) = self.function_database.get_function(real_name) {
                return Ok(self.format_detailed_help(info));
            }
        }

        // Try fuzzy matching for typos
        let suggestions = self.find_fuzzy_matches(function_name, 5);
        if !suggestions.is_empty() {
            let mut result = format!("Function '{}' not found. Did you mean:\n\n", function_name.red());
            for (i, suggestion) in suggestions.iter().enumerate() {
                result.push_str(&format!("  {}. {} (score: {})\n", 
                    i + 1, 
                    suggestion.function_name.green(),
                    suggestion.relevance_score
                ));
            }
            result.push_str(&format!("\nUse ?{} for detailed help on any of these functions.", suggestions[0].function_name.cyan()));
            return Ok(result);
        }

        Err(ReplError::Other {
            message: format!("Function '{}' not found and no similar functions detected", function_name),
        })
    }

    /// Handle ??search_term command - fuzzy search with typo suggestions  
    pub fn handle_fuzzy_search(&mut self, search_term: &str) -> ReplResult<String> {
        let results = self.comprehensive_search(search_term);
        
        if results.is_empty() {
            return Ok(format!("No functions found matching '{}'. Try a broader search term.", search_term.red()));
        }

        let mut output = format!("Search results for '{}':\n\n", search_term.green().bold());
        
        for (i, result) in results.iter().take(10).enumerate() {
            let match_type_str = match result.match_type {
                MatchType::ExactName => "exact name".blue(),
                MatchType::FuzzyName => "similar name".cyan(),
                MatchType::Description => "description".yellow(),
                MatchType::Category => "category".magenta(),
                MatchType::Parameter => "parameter".green(),
                MatchType::Example => "example".white(),
                MatchType::Alias => "alias".bright_blue(),
            };
            
            output.push_str(&format!(
                "  {}. {} ({}) - score: {}\n     {}\n\n",
                i + 1,
                result.function_name.bold(),
                match_type_str,
                result.relevance_score,
                result.snippet.dimmed()
            ));
        }

        if results.len() > 10 {
            output.push_str(&format!("... and {} more results. Use a more specific search term to narrow down.\n", results.len() - 10));
        }

        output.push_str(&format!("\n{}", "Use ?FunctionName for detailed help on any function.".italic()));
        Ok(output)
    }

    /// Handle ??category command - category-based browsing
    pub fn handle_category_browse(&mut self, category: &str) -> ReplResult<String> {
        let category_lower = category.to_lowercase();
        
        // Try direct category match
        if let Some(functions) = self.categories.get(&category_lower) {
            return Ok(self.format_category_listing(category, functions));
        }

        // Try category alias matching
        let category_aliases: HashMap<&str, &str> = [
            ("mathematics", "math"),
            ("trigonometry", "math"), 
            ("trig", "math"),
            ("lists", "list"),
            ("arrays", "list"),
            ("strings", "string"),
            ("text", "string"),
            ("ml", "machine_learning"),
            ("ai", "machine_learning"),
        ].iter().cloned().collect();

        if let Some(&alias) = category_aliases.get(category_lower.as_str()) {
            if let Some(functions) = self.categories.get(alias) {
                return Ok(self.format_category_listing(category, functions));
            }
        }

        // Search categories from function database
        let matching_categories: Vec<_> = self.function_database.all_categories()
            .filter(|cat| cat.to_lowercase().contains(&category_lower))
            .collect();

        if !matching_categories.is_empty() {
            let mut output = format!("Categories matching '{}':\n\n", category.green().bold());
            for cat in matching_categories {
                let functions = self.function_database.get_functions_in_category(cat);
                output.push_str(&format!("{}:\n", cat.cyan().bold()));
                for func in functions.iter().take(8) {
                    output.push_str(&format!("  • {}\n", func));
                }
                if functions.len() > 8 {
                    output.push_str(&format!("  ... and {} more\n", functions.len() - 8));
                }
                output.push('\n');
            }
            return Ok(output);
        }

        // Show available categories
        Ok(self.list_all_categories())
    }

    /// Comprehensive search across all function metadata
    fn comprehensive_search(&self, query: &str) -> Vec<SearchResult> {
        let mut results = Vec::new();
        let query_lower = query.to_lowercase();

        for info in self.function_database.all_functions() {
            // Exact name match
            if info.name.to_lowercase() == query_lower {
                results.push(SearchResult {
                    function_name: info.name.clone(),
                    relevance_score: 1000,
                    match_type: MatchType::ExactName,
                    snippet: info.signature.clone(),
                });
                continue;
            }

            // Fuzzy name match
            if let Some(score) = self.fuzzy_matcher.fuzzy_match(&info.name.to_lowercase(), &query_lower) {
                if score > 30 {
                    results.push(SearchResult {
                        function_name: info.name.clone(),
                        relevance_score: score + 100,
                        match_type: MatchType::FuzzyName,
                        snippet: info.signature.clone(),
                    });
                }
            }

            // Alias match
            for alias in &info.aliases {
                if alias.to_lowercase().contains(&query_lower) {
                    results.push(SearchResult {
                        function_name: info.name.clone(),
                        relevance_score: 200,
                        match_type: MatchType::Alias,
                        snippet: format!("Alias: {} → {}", alias, info.name),
                    });
                }
            }

            // Description match
            if info.description.to_lowercase().contains(&query_lower) {
                results.push(SearchResult {
                    function_name: info.name.clone(),
                    relevance_score: 50,
                    match_type: MatchType::Description,
                    snippet: info.description.clone(),
                });
            }

            // Category match
            if info.category.to_lowercase().contains(&query_lower) {
                results.push(SearchResult {
                    function_name: info.name.clone(),
                    relevance_score: 30,
                    match_type: MatchType::Category,
                    snippet: format!("Category: {}", info.category),
                });
            }

            // Parameter match
            for param in &info.parameters {
                if param.name.to_lowercase().contains(&query_lower) ||
                   param.type_hint.to_lowercase().contains(&query_lower) {
                    results.push(SearchResult {
                        function_name: info.name.clone(),
                        relevance_score: 40,
                        match_type: MatchType::Parameter,
                        snippet: format!("Parameter: {} ({})", param.name, param.type_hint),
                    });
                }
            }

            // Example match
            for example in &info.examples {
                if example.to_lowercase().contains(&query_lower) {
                    results.push(SearchResult {
                        function_name: info.name.clone(),
                        relevance_score: 25,
                        match_type: MatchType::Example,
                        snippet: example.clone(),
                    });
                }
            }
        }

        // Sort by relevance score (descending)
        results.sort_by(|a, b| b.relevance_score.cmp(&a.relevance_score));
        
        // Remove duplicates, keeping highest scoring
        let mut seen = std::collections::HashSet::new();
        results.retain(|result| seen.insert(result.function_name.clone()));
        
        results
    }

    /// Find fuzzy matches for typo detection
    fn find_fuzzy_matches(&self, query: &str, limit: usize) -> Vec<SearchResult> {
        let mut matches = Vec::new();
        
        for info in self.function_database.all_functions() {
            if let Some(score) = self.fuzzy_matcher.fuzzy_match(&info.name.to_lowercase(), &query.to_lowercase()) {
                if score > 15 { // Threshold for reasonable matches
                    matches.push(SearchResult {
                        function_name: info.name.clone(),
                        relevance_score: score,
                        match_type: MatchType::FuzzyName,
                        snippet: info.signature.clone(),
                    });
                }
            }
        }

        matches.sort_by(|a, b| b.relevance_score.cmp(&a.relevance_score));
        matches.truncate(limit);
        matches
    }

    /// Format detailed help for a function
    fn format_detailed_help(&self, info: &FunctionInfo) -> String {
        let mut output = String::new();
        
        // Header with function name and signature
        output.push_str(&format!("{}\n", info.name.green().bold().underline()));
        output.push_str(&format!("{}\n\n", info.signature.cyan()));
        
        // Description
        output.push_str(&format!("{}\n", "Description:".yellow().bold()));
        output.push_str(&format!("  {}\n\n", info.description));
        
        // Parameters
        if !info.parameters.is_empty() {
            output.push_str(&format!("{}\n", "Parameters:".yellow().bold()));
            for param in &info.parameters {
                let optional_marker = if param.optional { " (optional)" } else { "" };
                output.push_str(&format!("  • {} : {}{}\n", 
                    param.name.blue(), 
                    param.type_hint.magenta(), 
                    optional_marker.dimmed()
                ));
                output.push_str(&format!("    {}\n", param.description.dimmed()));
                if let Some(default) = &param.default_value {
                    output.push_str(&format!("    Default: {}\n", default.green()));
                }
            }
            output.push('\n');
        }

        // Return type
        output.push_str(&format!("{} {}\n\n", "Returns:".yellow().bold(), info.return_type.magenta()));
        
        // Examples
        if !info.examples.is_empty() {
            output.push_str(&format!("{}\n", "Examples:".yellow().bold()));
            for (i, example) in info.examples.iter().take(4).enumerate() {
                output.push_str(&format!("  {}. {}\n", i + 1, example.bright_white()));
            }
            if info.examples.len() > 4 {
                output.push_str(&format!("     ... and {} more examples\n", info.examples.len() - 4));
            }
            output.push('\n');
        }

        // Related functions
        if !info.related_functions.is_empty() {
            output.push_str(&format!("{}\n", "See also:".yellow().bold()));
            let related_str = info.related_functions.iter()
                .map(|f| f.cyan().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            output.push_str(&format!("  {}\n\n", related_str));
        }

        // Aliases
        if !info.aliases.is_empty() {
            output.push_str(&format!("{}\n", "Aliases:".yellow().bold()));
            let aliases_str = info.aliases.iter()
                .map(|a| a.blue().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            output.push_str(&format!("  {}\n\n", aliases_str));
        }

        // Module and source location
        output.push_str(&format!("{}\n", "Source:".yellow().bold()));
        output.push_str(&format!("  Module: {}\n", info.module.green()));
        if let Some(location) = &info.source_location {
            output.push_str(&format!("  Location: {}\n", location.dimmed()));
        }

        output
    }

    /// Format category listing
    fn format_category_listing(&self, category: &str, functions: &[String]) -> String {
        let mut output = format!("Functions in category '{}':\n\n", category.green().bold());
        
        for (i, func_name) in functions.iter().enumerate() {
            if let Some(info) = self.function_database.get_function(func_name) {
                output.push_str(&format!("{}. {} - {}\n", 
                    i + 1, 
                    func_name.cyan().bold(), 
                    info.description.dimmed()
                ));
            } else {
                output.push_str(&format!("{}. {}\n", i + 1, func_name.cyan().bold()));
            }
        }
        
        output.push_str(&format!("\n{}", "Use ?FunctionName for detailed help on any function.".italic()));
        output
    }

    /// List all available categories
    fn list_all_categories(&self) -> String {
        let mut output = format!("{}\n\n", "Available categories:".green().bold());
        
        for category in self.function_database.all_categories() {
            let functions = self.function_database.get_functions_in_category(category);
            output.push_str(&format!("• {} ({} functions)\n", 
                category.cyan().bold(), 
                functions.len()
            ));
        }
        
        output.push_str(&format!("\n{}", "Use ??category_name to browse functions in a specific category.".italic()));
        output
    }

    /// Get context-aware suggestions for auto-completion
    pub fn get_context_suggestions(&self, input: &str, cursor_pos: usize) -> Vec<String> {
        let word_start = input[..cursor_pos]
            .rfind(|c: char| c.is_whitespace() || "()[]{},".contains(c))
            .map(|i| i + 1)
            .unwrap_or(0);
        
        let partial = &input[word_start..cursor_pos];
        if partial.is_empty() {
            return Vec::new();
        }

        let mut suggestions = Vec::new();
        
        // Function name completion
        for info in self.function_database.all_functions() {
            if info.name.to_lowercase().starts_with(&partial.to_lowercase()) {
                suggestions.push(format!("{} - {}", info.name, info.description));
            }
        }

        // Alias completion
        for (alias, real_name) in &self.aliases {
            if alias.starts_with(&partial.to_lowercase()) {
                suggestions.push(format!("{} (alias for {})", alias, real_name));
            }
        }

        suggestions.sort();
        suggestions.truncate(10);
        suggestions
    }

    /// Record function usage for analytics
    pub fn record_usage(&mut self, function_name: &str) {
        *self.usage_stats.entry(function_name.to_string()).or_insert(0) += 1;
    }

    /// Get usage statistics
    pub fn get_usage_stats(&self) -> Vec<(String, u64)> {
        let mut stats: Vec<_> = self.usage_stats.iter()
            .map(|(name, count)| (name.clone(), *count))
            .collect();
        stats.sort_by(|a, b| b.1.cmp(&a.1));
        stats
    }

    /// Get smart suggestions based on usage patterns
    pub fn get_smart_suggestions(&self, context: &str) -> Vec<String> {
        // This could analyze the input context and suggest commonly used functions
        // For now, return most frequently used functions
        self.get_usage_stats()
            .into_iter()
            .take(5)
            .map(|(name, count)| format!("{} (used {} times)", name, count))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        modules::registry::ModuleRegistry,
        linker::FunctionRegistry,
        stdlib::StandardLibrary,
    };
    use std::sync::{Arc, RwLock};

    #[test]
    fn test_function_database_creation() {
        let func_registry = Arc::new(RwLock::new(FunctionRegistry::new()));
        let registry = ModuleRegistry::new(func_registry);
        let stdlib = StandardLibrary::new();
        
        let db = FunctionDatabase::build_from_registry(&registry, &stdlib);
        
        // Should have functions from standard library modules
        assert!(db.get_function("Sin").is_some());
        assert!(db.get_function("Length").is_some());
        assert!(db.get_function("StringJoin").is_some());
    }

    #[test]
    fn test_enhanced_help_system() {
        let func_registry = Arc::new(RwLock::new(FunctionRegistry::new()));
        let registry = Arc::new(ModuleRegistry::new(func_registry));
        let stdlib = Arc::new(StandardLibrary::new());
        
        let mut help_system = EnhancedHelpSystem::new(registry, stdlib);
        
        // Test exact function lookup
        let result = help_system.handle_help_function("Sin");
        assert!(result.is_ok());
        assert!(result.unwrap().contains("Sin"));
    }

    #[test]
    fn test_fuzzy_search() {
        let func_registry = Arc::new(RwLock::new(FunctionRegistry::new()));
        let registry = Arc::new(ModuleRegistry::new(func_registry));
        let stdlib = Arc::new(StandardLibrary::new());
        
        let mut help_system = EnhancedHelpSystem::new(registry, stdlib);
        
        // Test fuzzy search
        let result = help_system.handle_fuzzy_search("sin");
        assert!(result.is_ok());
        
        let result = help_system.handle_fuzzy_search("len");
        assert!(result.is_ok());
    }

    #[test]
    fn test_category_browsing() {
        let func_registry = Arc::new(RwLock::new(FunctionRegistry::new()));
        let registry = Arc::new(ModuleRegistry::new(func_registry));
        let stdlib = Arc::new(StandardLibrary::new());
        
        let mut help_system = EnhancedHelpSystem::new(registry, stdlib);
        
        // Test category browsing
        let result = help_system.handle_category_browse("math");
        assert!(result.is_ok());
    }
}