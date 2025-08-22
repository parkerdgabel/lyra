//! Pattern matching analysis and validation for intelligent hints
//!
//! This module provides comprehensive pattern matching assistance including
//! pattern syntax validation, rule construction guidance, and match prediction.

use std::collections::HashMap;

/// Pattern matching analyzer for intelligent hints
pub struct PatternAnalyzer {
    /// Known pattern templates
    pattern_templates: HashMap<String, PatternTemplate>,
    /// Pattern validation rules
    validation_rules: Vec<ValidationRule>,
    /// Common pattern examples
    pattern_examples: HashMap<PatternType, Vec<PatternExample>>,
    /// Rule construction helpers
    rule_helpers: HashMap<String, RuleHelper>,
}

/// Pattern template for guided construction
#[derive(Debug, Clone)]
pub struct PatternTemplate {
    /// Template name
    pub name: String,
    /// Template pattern
    pub template: String,
    /// Description of what this pattern matches
    pub description: String,
    /// Parameters that can be customized
    pub parameters: Vec<PatternParameter>,
    /// Example usage
    pub examples: Vec<String>,
    /// Difficulty level (1-5)
    pub difficulty: u8,
}

/// Parameter in a pattern template
#[derive(Debug, Clone)]
pub struct PatternParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: String,
    /// Default value
    pub default_value: Option<String>,
    /// Description
    pub description: String,
}

/// Pattern validation rule
#[derive(Debug, Clone)]
pub struct ValidationRule {
    /// Rule identifier
    pub rule_id: String,
    /// Pattern to match against
    pub pattern_regex: String,
    /// Error message if validation fails
    pub error_message: String,
    /// Suggested fix
    pub suggested_fix: Option<String>,
    /// Severity (1=warning, 5=error)
    pub severity: u8,
}

/// Pattern type classification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PatternType {
    /// Blank pattern (x_)
    Blank,
    /// Blank sequence (x__)
    BlankSequence,
    /// Blank null sequence (x___)
    BlankNullSequence,
    /// Named pattern with type (x_Integer)
    TypedBlank,
    /// Conditional pattern (x_?condition)
    ConditionalPattern,
    /// Alternative patterns (x_|y_)
    AlternativePattern,
    /// Repeated patterns (x_..)
    RepeatedPattern,
    /// Optional patterns (x_.)
    OptionalPattern,
    /// Head patterns (f[x_])
    HeadPattern,
    /// Complex nested patterns
    NestedPattern,
}

/// Pattern example with explanation
#[derive(Debug, Clone)]
pub struct PatternExample {
    /// Pattern code
    pub pattern: String,
    /// What it matches
    pub matches: Vec<String>,
    /// What it doesn't match
    pub non_matches: Vec<String>,
    /// Explanation
    pub explanation: String,
    /// Use case
    pub use_case: String,
}

/// Rule construction helper
#[derive(Debug, Clone)]
pub struct RuleHelper {
    /// Rule type (replacement, delayed, condition)
    pub rule_type: RuleType,
    /// Pattern template
    pub pattern_template: String,
    /// Replacement template
    pub replacement_template: String,
    /// Description
    pub description: String,
    /// Common use cases
    pub use_cases: Vec<String>,
}

/// Type of rule
#[derive(Debug, Clone, PartialEq)]
pub enum RuleType {
    /// Immediate replacement (->)
    Replacement,
    /// Delayed replacement (:>)
    DelayedReplacement,
    /// Conditional rule (/; condition)
    ConditionalRule,
    /// Tagged rule
    TaggedRule,
}

/// Pattern analysis result
#[derive(Debug, Clone)]
pub struct PatternAnalysisResult {
    /// Detected pattern type
    pub pattern_type: Option<PatternType>,
    /// Validation errors
    pub validation_errors: Vec<ValidationError>,
    /// Suggestions for improvement
    pub suggestions: Vec<PatternSuggestion>,
    /// Variables captured by the pattern
    pub captured_variables: Vec<String>,
    /// Pattern complexity score (1-10)
    pub complexity_score: u8,
    /// Match predictions
    pub match_predictions: Vec<MatchPrediction>,
}

/// Validation error in pattern
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Error type
    pub error_type: String,
    /// Error message
    pub message: String,
    /// Position in pattern (if applicable)
    pub position: Option<usize>,
    /// Suggested fix
    pub suggested_fix: Option<String>,
    /// Severity level
    pub severity: u8,
}

/// Pattern improvement suggestion
#[derive(Debug, Clone)]
pub struct PatternSuggestion {
    /// Suggestion type
    pub suggestion_type: SuggestionType,
    /// Description of suggestion
    pub description: String,
    /// Original pattern part
    pub original: String,
    /// Suggested replacement
    pub suggested: String,
    /// Confidence (0.0 to 1.0)
    pub confidence: f32,
    /// Rationale
    pub rationale: String,
}

/// Type of pattern suggestion
#[derive(Debug, Clone, PartialEq)]
pub enum SuggestionType {
    /// Syntax improvement
    SyntaxImprovement,
    /// Performance optimization
    PerformanceOptimization,
    /// Readability enhancement
    ReadabilityEnhancement,
    /// Type safety improvement
    TypeSafety,
    /// Pattern simplification
    Simplification,
}

/// Match prediction for pattern
#[derive(Debug, Clone)]
pub struct MatchPrediction {
    /// Example expression that would match
    pub example_match: String,
    /// Explanation of why it matches
    pub match_explanation: String,
    /// Captured variable bindings
    pub variable_bindings: HashMap<String, String>,
    /// Confidence of prediction
    pub confidence: f32,
}

/// Rule construction hint
#[derive(Debug, Clone)]
pub struct RuleConstructionHint {
    /// Suggested rule pattern
    pub pattern: String,
    /// Suggested replacement
    pub replacement: String,
    /// Rule type recommendation
    pub rule_type: RuleType,
    /// Explanation of the rule
    pub explanation: String,
    /// Example usage
    pub example_usage: String,
}

impl PatternAnalyzer {
    /// Create a new pattern analyzer
    pub fn new() -> Self {
        let mut analyzer = Self {
            pattern_templates: HashMap::new(),
            validation_rules: Vec::new(),
            pattern_examples: HashMap::new(),
            rule_helpers: HashMap::new(),
        };
        
        analyzer.populate_pattern_templates();
        analyzer.populate_validation_rules();
        analyzer.populate_pattern_examples();
        analyzer.populate_rule_helpers();
        
        analyzer
    }
    
    /// Analyze a pattern and provide feedback
    pub fn analyze_pattern(&self, pattern: &str) -> PatternAnalysisResult {
        let pattern_type = self.classify_pattern(pattern);
        let validation_errors = self.validate_pattern(pattern);
        let suggestions = self.generate_suggestions(pattern, &pattern_type);
        let captured_variables = self.extract_captured_variables(pattern);
        let complexity_score = self.calculate_complexity(pattern);
        let match_predictions = self.predict_matches(pattern, &pattern_type);
        
        PatternAnalysisResult {
            pattern_type,
            validation_errors,
            suggestions,
            captured_variables,
            complexity_score,
            match_predictions,
        }
    }
    
    /// Get suggestions for pattern construction
    pub fn get_pattern_suggestions(&self, context: &str, intent: &str) -> Vec<PatternTemplate> {
        let mut suggestions = Vec::new();
        
        // Filter templates based on context and intent
        for template in self.pattern_templates.values() {
            if self.template_matches_context(template, context, intent) {
                suggestions.push(template.clone());
            }
        }
        
        // Sort by relevance and difficulty
        suggestions.sort_by(|a, b| {
            a.difficulty.cmp(&b.difficulty)
                .then_with(|| a.name.cmp(&b.name))
        });
        
        suggestions
    }
    
    /// Generate rule construction hints
    pub fn get_rule_construction_hints(&self, pattern: &str, context: &str) -> Vec<RuleConstructionHint> {
        let mut hints = Vec::new();
        
        // Analyze the pattern to suggest appropriate rules
        let analysis = self.analyze_pattern(pattern);
        
        if let Some(pattern_type) = analysis.pattern_type {
            match pattern_type {
                PatternType::Blank => {
                    hints.push(RuleConstructionHint {
                        pattern: format!("{} -> expr", pattern),
                        replacement: "replacement_expression".to_string(),
                        rule_type: RuleType::Replacement,
                        explanation: "Use -> for immediate replacement".to_string(),
                        example_usage: format!("{} -> Square[{}]", pattern, 
                                             analysis.captured_variables.get(0).unwrap_or(&"x".to_string())),
                    });
                    
                    hints.push(RuleConstructionHint {
                        pattern: format!("{} :> expr", pattern),
                        replacement: "dynamic_expression".to_string(),
                        rule_type: RuleType::DelayedReplacement,
                        explanation: "Use :> for delayed evaluation".to_string(),
                        example_usage: format!("{} :> RandomReal[]", pattern),
                    });
                },
                PatternType::TypedBlank => {
                    hints.push(RuleConstructionHint {
                        pattern: pattern.to_string(),
                        replacement: "type_specific_operation".to_string(),
                        rule_type: RuleType::Replacement,
                        explanation: "Typed patterns enable type-specific operations".to_string(),
                        example_usage: format!("{} -> specificOperation[{}]", pattern,
                                             analysis.captured_variables.get(0).unwrap_or(&"x".to_string())),
                    });
                },
                _ => {
                    // General suggestions for other pattern types
                    hints.push(RuleConstructionHint {
                        pattern: pattern.to_string(),
                        replacement: "custom_transformation".to_string(),
                        rule_type: RuleType::Replacement,
                        explanation: "Apply custom transformation".to_string(),
                        example_usage: format!("{} -> customFunction[captured_vars]", pattern),
                    });
                }
            }
        }
        
        hints
    }
    
    /// Validate pattern syntax
    pub fn validate_pattern_syntax(&self, pattern: &str) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        
        // Check basic syntax rules
        for rule in &self.validation_rules {
            if !self.pattern_matches_rule(pattern, rule) {
                errors.push(ValidationError {
                    error_type: rule.rule_id.clone(),
                    message: rule.error_message.clone(),
                    position: None, // Would need more sophisticated parsing
                    suggested_fix: rule.suggested_fix.clone(),
                    severity: rule.severity,
                });
            }
        }
        
        // Check for common mistakes
        errors.extend(self.check_common_pattern_mistakes(pattern));
        
        errors
    }
    
    /// Get pattern examples for learning
    pub fn get_pattern_examples(&self, pattern_type: &PatternType) -> Vec<PatternExample> {
        self.pattern_examples.get(pattern_type).cloned().unwrap_or_default()
    }
    
    /// Predict what expressions would match a pattern
    pub fn predict_pattern_matches(&self, pattern: &str) -> Vec<String> {
        let analysis = self.analyze_pattern(pattern);
        analysis.match_predictions
            .into_iter()
            .map(|pred| pred.example_match)
            .collect()
    }
    
    /// Check if a specific expression would match a pattern
    pub fn would_match(&self, pattern: &str, expression: &str) -> bool {
        // This is a simplified version - would need full pattern matching engine
        // For now, just do basic checks
        
        if pattern.contains("_") {
            // Blank patterns match most things
            return true;
        }
        
        if pattern == expression {
            return true;
        }
        
        false // Conservative default
    }
    
    /// Get pattern variable suggestions
    pub fn suggest_pattern_variables(&self, context: &str) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        // Common pattern variable names
        let common_vars = vec!["x_", "y_", "z_", "n_", "f_", "args__"];
        suggestions.extend(common_vars.iter().map(|s| s.to_string()));
        
        // Context-specific suggestions
        if context.contains("number") || context.contains("math") {
            suggestions.extend(vec!["n_Integer".to_string(), "x_Real".to_string()]);
        }
        
        if context.contains("list") || context.contains("array") {
            suggestions.extend(vec!["list_List".to_string(), "elements__".to_string()]);
        }
        
        if context.contains("function") {
            suggestions.extend(vec!["f_".to_string(), "args___".to_string()]);
        }
        
        suggestions
    }
    
    // Helper methods
    
    fn classify_pattern(&self, pattern: &str) -> Option<PatternType> {
        if pattern.contains("___") {
            Some(PatternType::BlankNullSequence)
        } else if pattern.contains("__") {
            Some(PatternType::BlankSequence)
        } else if pattern.contains("_?") {
            Some(PatternType::ConditionalPattern)
        } else if pattern.contains("_") && pattern.contains("[") {
            Some(PatternType::HeadPattern)
        } else if pattern.contains("_") && (pattern.contains("Integer") || pattern.contains("Real") || 
                                           pattern.contains("String") || pattern.contains("List")) {
            Some(PatternType::TypedBlank)
        } else if pattern.contains("_") {
            Some(PatternType::Blank)
        } else if pattern.contains("|") {
            Some(PatternType::AlternativePattern)
        } else {
            None
        }
    }
    
    fn validate_pattern(&self, pattern: &str) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        
        // Check for unbalanced brackets
        let mut bracket_count = 0;
        for ch in pattern.chars() {
            match ch {
                '[' => bracket_count += 1,
                ']' => bracket_count -= 1,
                _ => {}
            }
        }
        
        if bracket_count != 0 {
            errors.push(ValidationError {
                error_type: "unbalanced_brackets".to_string(),
                message: "Unbalanced brackets in pattern".to_string(),
                position: None,
                suggested_fix: Some("Check bracket balance".to_string()),
                severity: 4,
            });
        }
        
        // Check for invalid blank patterns
        if pattern.contains("____") {
            errors.push(ValidationError {
                error_type: "invalid_blank".to_string(),
                message: "Too many underscores in blank pattern".to_string(),
                position: None,
                suggested_fix: Some("Use _, __, or ___ for blank patterns".to_string()),
                severity: 3,
            });
        }
        
        errors
    }
    
    fn generate_suggestions(&self, pattern: &str, pattern_type: &Option<PatternType>) -> Vec<PatternSuggestion> {
        let mut suggestions = Vec::new();
        
        // Suggest improvements based on pattern type
        if let Some(pt) = pattern_type {
            match pt {
                PatternType::Blank => {
                    if !pattern.contains("_") {
                        suggestions.push(PatternSuggestion {
                            suggestion_type: SuggestionType::SyntaxImprovement,
                            description: "Consider using blank pattern".to_string(),
                            original: pattern.to_string(),
                            suggested: format!("{}_", pattern),
                            confidence: 0.8,
                            rationale: "Blank patterns are more flexible".to_string(),
                        });
                    }
                },
                PatternType::TypedBlank => {
                    // Check if type is appropriate
                    if pattern.contains("_String") && pattern.contains("Number") {
                        suggestions.push(PatternSuggestion {
                            suggestion_type: SuggestionType::TypeSafety,
                            description: "Consider more specific type".to_string(),
                            original: pattern.to_string(),
                            suggested: pattern.replace("_String", "_Real"),
                            confidence: 0.6,
                            rationale: "Real type is more appropriate for numbers".to_string(),
                        });
                    }
                },
                _ => {}
            }
        }
        
        suggestions
    }
    
    fn extract_captured_variables(&self, pattern: &str) -> Vec<String> {
        let mut variables = Vec::new();
        
        // Simple extraction - look for identifiers before underscores
        let mut current_var = String::new();
        let mut in_variable = false;
        
        for ch in pattern.chars() {
            match ch {
                'a'..='z' | 'A'..='Z' if !in_variable => {
                    current_var.clear();
                    current_var.push(ch);
                    in_variable = true;
                },
                'a'..='z' | 'A'..='Z' | '0'..='9' if in_variable => {
                    current_var.push(ch);
                },
                '_' if in_variable => {
                    variables.push(current_var.clone());
                    in_variable = false;
                },
                _ => {
                    in_variable = false;
                }
            }
        }
        
        variables
    }
    
    fn calculate_complexity(&self, pattern: &str) -> u8 {
        let mut complexity = 1;
        
        // Add complexity for various features
        if pattern.contains("___") { complexity += 3; }
        else if pattern.contains("__") { complexity += 2; }
        else if pattern.contains("_") { complexity += 1; }
        
        if pattern.contains("?") { complexity += 2; } // Conditional patterns
        if pattern.contains("|") { complexity += 2; } // Alternatives
        if pattern.contains("[") { complexity += 1; } // Head patterns
        
        // Count nesting levels
        let nesting_level = pattern.chars().filter(|&c| c == '[').count();
        complexity += nesting_level as u8;
        
        complexity.min(10)
    }
    
    fn predict_matches(&self, pattern: &str, pattern_type: &Option<PatternType>) -> Vec<MatchPrediction> {
        let mut predictions = Vec::new();
        
        if let Some(pt) = pattern_type {
            match pt {
                PatternType::Blank => {
                    predictions.push(MatchPrediction {
                        example_match: "42".to_string(),
                        match_explanation: "Blank patterns match any expression".to_string(),
                        variable_bindings: HashMap::new(),
                        confidence: 0.9,
                    });
                    
                    predictions.push(MatchPrediction {
                        example_match: "{1, 2, 3}".to_string(),
                        match_explanation: "Blank patterns match lists".to_string(),
                        variable_bindings: HashMap::new(),
                        confidence: 0.9,
                    });
                },
                PatternType::TypedBlank => {
                    if pattern.contains("_Integer") {
                        predictions.push(MatchPrediction {
                            example_match: "42".to_string(),
                            match_explanation: "Integer patterns match whole numbers".to_string(),
                            variable_bindings: HashMap::new(),
                            confidence: 0.95,
                        });
                    }
                    
                    if pattern.contains("_List") {
                        predictions.push(MatchPrediction {
                            example_match: "{a, b, c}".to_string(),
                            match_explanation: "List patterns match list expressions".to_string(),
                            variable_bindings: HashMap::new(),
                            confidence: 0.95,
                        });
                    }
                },
                _ => {}
            }
        }
        
        predictions
    }
    
    fn template_matches_context(&self, template: &PatternTemplate, context: &str, intent: &str) -> bool {
        let context_lower = context.to_lowercase();
        let intent_lower = intent.to_lowercase();
        let template_desc = template.description.to_lowercase();
        
        // Simple matching - could be more sophisticated
        template_desc.contains(&context_lower) || 
        template_desc.contains(&intent_lower) ||
        template.name.to_lowercase().contains(&intent_lower)
    }
    
    fn pattern_matches_rule(&self, _pattern: &str, _rule: &ValidationRule) -> bool {
        // Simplified - would implement proper regex matching
        true
    }
    
    fn check_common_pattern_mistakes(&self, pattern: &str) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        
        // Check for common typos
        if pattern.contains("__") && pattern.len() < 4 {
            errors.push(ValidationError {
                error_type: "incomplete_sequence".to_string(),
                message: "Blank sequence patterns need variable names".to_string(),
                position: None,
                suggested_fix: Some("Use 'name__' instead of '__'".to_string()),
                severity: 2,
            });
        }
        
        errors
    }
    
    fn populate_pattern_templates(&mut self) {
        // Populate with common pattern templates
        self.pattern_templates.insert("simple_blank".to_string(), PatternTemplate {
            name: "Simple Blank Pattern".to_string(),
            template: "x_".to_string(),
            description: "Matches any single expression".to_string(),
            parameters: vec![PatternParameter {
                name: "variable".to_string(),
                param_type: "identifier".to_string(),
                default_value: Some("x".to_string()),
                description: "Variable name to capture the match".to_string(),
            }],
            examples: vec!["x_ matches 42, {1,2}, Sin[x]".to_string()],
            difficulty: 1,
        });
        
        // Add more templates...
    }
    
    fn populate_validation_rules(&mut self) {
        // Populate with validation rules
        self.validation_rules.push(ValidationRule {
            rule_id: "valid_blank".to_string(),
            pattern_regex: r"[a-zA-Z][a-zA-Z0-9]*_+".to_string(),
            error_message: "Invalid blank pattern syntax".to_string(),
            suggested_fix: Some("Use format: variableName_".to_string()),
            severity: 3,
        });
        
        // Add more rules...
    }
    
    fn populate_pattern_examples(&mut self) {
        // Populate with examples for each pattern type
        let blank_examples = vec![
            PatternExample {
                pattern: "x_".to_string(),
                matches: vec!["42".to_string(), "Sin[y]".to_string(), "{1, 2}".to_string()],
                non_matches: vec![],
                explanation: "Matches any expression".to_string(),
                use_case: "General pattern matching".to_string(),
            }
        ];
        
        self.pattern_examples.insert(PatternType::Blank, blank_examples);
        
        // Add more examples...
    }
    
    fn populate_rule_helpers(&mut self) {
        // Populate with rule construction helpers
        self.rule_helpers.insert("replacement".to_string(), RuleHelper {
            rule_type: RuleType::Replacement,
            pattern_template: "pattern".to_string(),
            replacement_template: "replacement".to_string(),
            description: "Immediate replacement rule".to_string(),
            use_cases: vec!["Algebraic simplification".to_string(), "Function definition".to_string()],
        });
        
        // Add more helpers...
    }
}

impl Default for PatternAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_classification() {
        let analyzer = PatternAnalyzer::new();
        
        assert_eq!(analyzer.classify_pattern("x_"), Some(PatternType::Blank));
        assert_eq!(analyzer.classify_pattern("x__"), Some(PatternType::BlankSequence));
        assert_eq!(analyzer.classify_pattern("x___"), Some(PatternType::BlankNullSequence));
        assert_eq!(analyzer.classify_pattern("x_Integer"), Some(PatternType::TypedBlank));
        assert_eq!(analyzer.classify_pattern("x_?Positive"), Some(PatternType::ConditionalPattern));
    }
    
    #[test]
    fn test_pattern_validation() {
        let analyzer = PatternAnalyzer::new();
        
        let errors = analyzer.validate_pattern("x_[y_]");
        assert!(errors.is_empty()); // Should be valid
        
        let errors = analyzer.validate_pattern("x_[y_");
        assert!(!errors.is_empty()); // Should have unbalanced bracket error
    }
    
    #[test]
    fn test_variable_extraction() {
        let analyzer = PatternAnalyzer::new();
        
        let vars = analyzer.extract_captured_variables("f[x_, y_]");
        assert_eq!(vars, vec!["x", "y"]);
        
        let vars = analyzer.extract_captured_variables("expr_");
        assert_eq!(vars, vec!["expr"]);
    }
    
    #[test]
    fn test_complexity_calculation() {
        let analyzer = PatternAnalyzer::new();
        
        assert_eq!(analyzer.calculate_complexity("x_"), 2); // Basic blank
        assert_eq!(analyzer.calculate_complexity("f[x_, y__]"), 4); // Function with sequence
        assert!(analyzer.calculate_complexity("complex[nested[deep[x_]]]") > 5); // Deep nesting
    }
    
    #[test]
    fn test_pattern_suggestions() {
        let analyzer = PatternAnalyzer::new();
        
        let suggestions = analyzer.get_pattern_suggestions("math", "number");
        assert!(!suggestions.is_empty());
        
        let vars = analyzer.suggest_pattern_variables("number context");
        assert!(vars.contains(&"n_Integer".to_string()));
    }
}