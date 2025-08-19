//! Dead Code Elimination Engine
//!
//! Core engine for identifying and eliminating unused functions from the stdlib
//! based on dependency graph analysis and usage patterns.

use crate::modules::registry::ModuleRegistry;
use super::{DependencyGraph, UsageTracker, TreeShakeError};
use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};

/// Dead code elimination engine
pub struct Eliminator {
    /// Configuration for elimination
    config: EliminatorConfig,
    
    /// Results of the last elimination run
    last_results: Option<EliminationResults>,
    
    /// Safety checks and validation
    safety_validator: SafetyValidator,
    
    /// Performance metrics
    performance_metrics: EliminationMetrics,
}

/// Configuration for dead code elimination
#[derive(Debug, Clone)]
pub struct EliminatorConfig {
    /// Enable aggressive elimination (removes more functions but less safe)
    pub aggressive_mode: bool,
    
    /// Preserve functions marked as entry points
    pub preserve_entry_points: bool,
    
    /// Preserve functions with specific attributes
    pub preserve_attributes: Vec<String>,
    
    /// Minimum usage threshold for keeping functions
    pub min_usage_threshold: u64,
    
    /// Enable elimination preview mode (analyze but don't eliminate)
    pub preview_mode: bool,
    
    /// Maximum number of functions to eliminate in one pass
    pub max_eliminations_per_pass: usize,
    
    /// Enable safety validation
    pub enable_safety_validation: bool,
    
    /// Preserve functions used in tests
    pub preserve_test_functions: bool,
}

impl Default for EliminatorConfig {
    fn default() -> Self {
        EliminatorConfig {
            aggressive_mode: false,
            preserve_entry_points: true,
            preserve_attributes: vec!["public".to_string(), "exported".to_string()],
            min_usage_threshold: 0,
            preview_mode: false,
            max_eliminations_per_pass: 50,
            enable_safety_validation: true,
            preserve_test_functions: true,
        }
    }
}

/// Results of dead code elimination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EliminationResults {
    /// Functions that were eliminated
    pub eliminated_functions: Vec<EliminatedFunction>,
    
    /// Functions that were preserved and why
    pub preserved_functions: Vec<PreservedFunction>,
    
    /// Summary statistics
    pub summary: EliminationSummary,
    
    /// Warnings and issues encountered
    pub warnings: Vec<EliminationWarning>,
    
    /// Performance impact of elimination
    pub performance_impact: PerformanceImpact,
    
    /// Timestamp of elimination
    pub timestamp: std::time::SystemTime,
}

/// Information about an eliminated function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EliminatedFunction {
    /// Function name
    pub name: String,
    
    /// Module it belonged to
    pub module: String,
    
    /// Reason for elimination
    pub elimination_reason: EliminationReason,
    
    /// Size of eliminated function
    pub estimated_size: usize,
    
    /// Functions that were calling this one (if any)
    pub callers: Vec<String>,
    
    /// Functions this one was calling
    pub callees: Vec<String>,
    
    /// Usage statistics before elimination
    pub usage_stats: FunctionUsageSnapshot,
}

/// Information about a preserved function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreservedFunction {
    /// Function name
    pub name: String,
    
    /// Module it belongs to
    pub module: String,
    
    /// Reason for preservation
    pub preservation_reason: PreservationReason,
    
    /// Usage statistics
    pub usage_stats: FunctionUsageSnapshot,
    
    /// Dependencies on this function
    pub dependents: Vec<String>,
}

/// Snapshot of function usage for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionUsageSnapshot {
    /// Number of times called
    pub call_count: u64,
    
    /// Number of unique callers
    pub unique_callers: usize,
    
    /// Whether function is on critical path
    pub is_critical: bool,
    
    /// Performance impact score
    pub performance_score: f64,
}

/// Reasons for eliminating a function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EliminationReason {
    /// Function never called
    NeverCalled,
    
    /// Function called but below usage threshold
    BelowUsageThreshold { threshold: u64, actual: u64 },
    
    /// Function is dead code (unreachable)
    DeadCode,
    
    /// Function is redundant (duplicate functionality)
    Redundant { alternative: String },
    
    /// Function marked for elimination by user
    UserMarked,
    
    /// Function is test-only and not needed in production
    TestOnly,
}

/// Reasons for preserving a function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PreservationReason {
    /// Function is an entry point
    EntryPoint,
    
    /// Function has required attributes
    RequiredAttributes { attributes: Vec<String> },
    
    /// Function is above usage threshold
    AboveUsageThreshold { threshold: u64, actual: u64 },
    
    /// Function is on critical path
    CriticalPath,
    
    /// Function has external dependencies
    ExternalDependencies,
    
    /// Function marked as required by user
    UserRequired,
    
    /// Safety validation prevented elimination
    SafetyValidation { reason: String },
}

/// Summary statistics for elimination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EliminationSummary {
    /// Total functions analyzed
    pub total_functions: usize,
    
    /// Functions eliminated
    pub functions_eliminated: usize,
    
    /// Functions preserved
    pub functions_preserved: usize,
    
    /// Total size eliminated (estimated bytes)
    pub total_size_eliminated: usize,
    
    /// Percentage of codebase eliminated
    pub elimination_percentage: f64,
    
    /// Memory savings achieved
    pub memory_savings: usize,
    
    /// Estimated performance improvement
    pub performance_improvement: f64,
}

/// Warnings encountered during elimination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EliminationWarning {
    /// Warning type
    pub warning_type: WarningType,
    
    /// Function involved
    pub function_name: String,
    
    /// Warning message
    pub message: String,
    
    /// Severity level
    pub severity: WarningSeverity,
}

/// Types of elimination warnings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WarningType {
    /// Potential dependency issue
    PotentialDependencyIssue,
    
    /// Function might be needed for future use
    FutureUseRisk,
    
    /// External interface function
    ExternalInterface,
    
    /// Performance critical function
    PerformanceCritical,
    
    /// Safety validation concern
    SafetyConcern,
}

/// Warning severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WarningSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Performance impact of elimination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImpact {
    /// Estimated compilation time improvement
    pub compilation_time_improvement: f64,
    
    /// Estimated binary size reduction
    pub binary_size_reduction: usize,
    
    /// Estimated memory usage reduction
    pub memory_usage_reduction: usize,
    
    /// Estimated startup time improvement
    pub startup_time_improvement: f64,
}

/// Safety validator for elimination
#[derive(Debug, Clone)]
pub struct SafetyValidator {
    /// Functions that should never be eliminated
    protected_functions: HashSet<String>,
    
    /// Validation rules
    validation_rules: Vec<ValidationRule>,
}

/// Validation rule for safety checking
#[derive(Debug, Clone)]
pub struct ValidationRule {
    /// Rule name
    pub name: String,
    
    /// Rule description
    pub description: String,
    
    /// Function to check if rule applies
    pub check: fn(&str, &DependencyGraph) -> bool,
}

/// Performance metrics for elimination process
#[derive(Debug, Clone, Default)]
pub struct EliminationMetrics {
    /// Time taken for analysis
    pub analysis_time: std::time::Duration,
    
    /// Time taken for elimination
    pub elimination_time: std::time::Duration,
    
    /// Time taken for validation
    pub validation_time: std::time::Duration,
    
    /// Memory used during process
    pub memory_used: usize,
    
    /// Number of passes required
    pub passes_required: usize,
}

impl Eliminator {
    /// Create a new eliminator
    pub fn new() -> Self {
        Eliminator {
            config: EliminatorConfig::default(),
            last_results: None,
            safety_validator: SafetyValidator::new(),
            performance_metrics: EliminationMetrics::default(),
        }
    }
    
    /// Create eliminator with custom configuration
    pub fn with_config(config: EliminatorConfig) -> Self {
        Eliminator {
            config,
            last_results: None,
            safety_validator: SafetyValidator::new(),
            performance_metrics: EliminationMetrics::default(),
        }
    }
    
    /// Perform dead code elimination on the stdlib
    pub fn eliminate_dead_code(
        &mut self,
        dependency_graph: &DependencyGraph,
        usage_tracker: &UsageTracker,
        module_registry: &mut ModuleRegistry,
    ) -> Result<EliminationResults, TreeShakeError> {
        let start_time = std::time::Instant::now();
        
        // Step 1: Identify elimination candidates
        let candidates = self.identify_elimination_candidates(dependency_graph, usage_tracker)?;
        
        // Step 2: Apply safety validation
        let validated_candidates = if self.config.enable_safety_validation {
            self.safety_validator.validate_candidates(&candidates, dependency_graph)?
        } else {
            candidates
        };
        
        // Step 3: Perform elimination (or preview)
        let results = if self.config.preview_mode {
            self.preview_elimination(&validated_candidates, dependency_graph, usage_tracker)?
        } else {
            self.perform_elimination(&validated_candidates, dependency_graph, usage_tracker, module_registry)?
        };
        
        // Step 4: Update metrics
        self.performance_metrics.analysis_time = start_time.elapsed();
        
        // Step 5: Store results
        self.last_results = Some(results.clone());
        
        Ok(results)
    }
    
    /// Get the results of the last elimination run
    pub fn last_results(&self) -> Option<&EliminationResults> {
        self.last_results.as_ref()
    }
    
    /// Get performance metrics
    pub fn performance_metrics(&self) -> &EliminationMetrics {
        &self.performance_metrics
    }
    
    /// Analyze elimination potential without actually eliminating
    pub fn analyze_elimination_potential(
        &self,
        dependency_graph: &DependencyGraph,
        usage_tracker: &UsageTracker,
    ) -> Result<EliminationAnalysis, TreeShakeError> {
        let all_functions = dependency_graph.all_functions();
        let mut analysis = EliminationAnalysis {
            total_functions: all_functions.len(),
            eliminable_functions: 0,
            total_size_eliminable: 0,
            categories: HashMap::new(),
        };
        
        for function_name in all_functions {
            if let Some(node) = dependency_graph.get_node(function_name) {
                let usage_stats = usage_tracker.get_function_stats(function_name);
                let can_eliminate = self.can_eliminate_function(function_name, node, usage_stats);
                
                if can_eliminate {
                    analysis.eliminable_functions += 1;
                    analysis.total_size_eliminable += node.metadata.estimated_size;
                    
                    let category = self.categorize_elimination_reason(function_name, node, usage_stats);
                    *analysis.categories.entry(category).or_insert(0) += 1;
                }
            }
        }
        
        Ok(analysis)
    }
    
    // Private implementation methods
    
    fn identify_elimination_candidates(
        &self,
        dependency_graph: &DependencyGraph,
        usage_tracker: &UsageTracker,
    ) -> Result<Vec<EliminationCandidate>, TreeShakeError> {
        let mut candidates = Vec::new();
        
        for function_name in dependency_graph.all_functions() {
            if let Some(node) = dependency_graph.get_node(function_name) {
                let usage_stats = usage_tracker.get_function_stats(function_name);
                
                if let Some(candidate) = self.evaluate_function_for_elimination(function_name, node, usage_stats, dependency_graph)? {
                    candidates.push(candidate);
                }
            }
        }
        
        // Sort candidates by elimination priority
        candidates.sort_by(|a, b| a.priority.partial_cmp(&b.priority).unwrap_or(std::cmp::Ordering::Equal));
        
        // Limit to max eliminations if specified
        if candidates.len() > self.config.max_eliminations_per_pass {
            candidates.truncate(self.config.max_eliminations_per_pass);
        }
        
        Ok(candidates)
    }
    
    fn evaluate_function_for_elimination(
        &self,
        function_name: &str,
        node: &super::DependencyNode,
        usage_stats: Option<&super::usage_tracker::FunctionUsageStats>,
        dependency_graph: &DependencyGraph,
    ) -> Result<Option<EliminationCandidate>, TreeShakeError> {
        // Check if function should be preserved
        if self.should_preserve_function(function_name, node) {
            return Ok(None);
        }
        
        // Determine elimination reason and priority
        let (reason, priority) = if let Some(stats) = usage_stats {
            if stats.call_count == 0 {
                (EliminationReason::NeverCalled, 1.0)
            } else if stats.call_count < self.config.min_usage_threshold {
                (EliminationReason::BelowUsageThreshold {
                    threshold: self.config.min_usage_threshold,
                    actual: stats.call_count,
                }, 0.8)
            } else {
                return Ok(None); // Function is used enough to keep
            }
        } else {
            (EliminationReason::NeverCalled, 1.0)
        };
        
        Ok(Some(EliminationCandidate {
            function_name: function_name.to_string(),
            module_name: node.module.clone(),
            elimination_reason: reason,
            priority,
            estimated_size: node.metadata.estimated_size,
            dependencies: dependency_graph.get_dependencies(function_name).into_iter()
                .map(|edge| edge.to.clone()).collect(),
            dependents: dependency_graph.get_dependents(function_name).into_iter()
                .map(|edge| edge.from.clone()).collect(),
        }))
    }
    
    fn should_preserve_function(&self, function_name: &str, node: &super::DependencyNode) -> bool {
        // Preserve entry points
        if self.config.preserve_entry_points && node.is_entry_point {
            return true;
        }
        
        // Preserve functions with required attributes
        for attr in &self.config.preserve_attributes {
            if node.metadata.attributes.contains(attr) {
                return true;
            }
        }
        
        // Preserve functions in protected list
        if self.safety_validator.protected_functions.contains(function_name) {
            return true;
        }
        
        false
    }
    
    fn can_eliminate_function(
        &self,
        function_name: &str,
        node: &super::DependencyNode,
        usage_stats: Option<&super::usage_tracker::FunctionUsageStats>,
    ) -> bool {
        // For can_eliminate_function, we need to pass a dummy dependency graph
        // This is a simplified check - in practice you'd pass the real dependency graph
        false // Simplified for now - would need dependency_graph parameter
    }
    
    fn categorize_elimination_reason(
        &self,
        function_name: &str,
        node: &super::DependencyNode,
        usage_stats: Option<&super::usage_tracker::FunctionUsageStats>,
    ) -> String {
        if let Some(stats) = usage_stats {
            if stats.call_count == 0 {
                "never_called".to_string()
            } else {
                "low_usage".to_string()
            }
        } else {
            "no_usage_data".to_string()
        }
    }
    
    fn preview_elimination(
        &self,
        candidates: &[EliminationCandidate],
        dependency_graph: &DependencyGraph,
        usage_tracker: &UsageTracker,
    ) -> Result<EliminationResults, TreeShakeError> {
        let mut eliminated_functions = Vec::new();
        let mut preserved_functions = Vec::new();
        let mut warnings = Vec::new();
        
        // Process elimination candidates
        for candidate in candidates {
            let usage_snapshot = self.create_usage_snapshot(&candidate.function_name, usage_tracker);
            
            eliminated_functions.push(EliminatedFunction {
                name: candidate.function_name.clone(),
                module: candidate.module_name.clone(),
                elimination_reason: candidate.elimination_reason.clone(),
                estimated_size: candidate.estimated_size,
                callers: candidate.dependents.clone(),
                callees: candidate.dependencies.clone(),
                usage_stats: usage_snapshot,
            });
        }
        
        // Process preserved functions
        for function_name in dependency_graph.all_functions() {
            if !candidates.iter().any(|c| c.function_name == *function_name) {
                if let Some(node) = dependency_graph.get_node(function_name) {
                    let usage_snapshot = self.create_usage_snapshot(function_name, usage_tracker);
                    let preservation_reason = self.determine_preservation_reason(function_name, node, usage_tracker);
                    
                    preserved_functions.push(PreservedFunction {
                        name: function_name.clone(),
                        module: node.module.clone(),
                        preservation_reason,
                        usage_stats: usage_snapshot,
                        dependents: dependency_graph.get_dependents(function_name).into_iter()
                            .map(|edge| edge.from.clone()).collect(),
                    });
                }
            }
        }
        
        let summary = self.create_elimination_summary(&eliminated_functions, &preserved_functions);
        let performance_impact = self.estimate_performance_impact(&eliminated_functions);
        
        Ok(EliminationResults {
            eliminated_functions,
            preserved_functions,
            summary,
            warnings,
            performance_impact,
            timestamp: std::time::SystemTime::now(),
        })
    }
    
    fn perform_elimination(
        &mut self,
        candidates: &[EliminationCandidate],
        dependency_graph: &DependencyGraph,
        usage_tracker: &UsageTracker,
        module_registry: &mut ModuleRegistry,
    ) -> Result<EliminationResults, TreeShakeError> {
        // For now, we'll just simulate elimination by returning preview results
        // In a real implementation, this would actually modify the module registry
        let results = self.preview_elimination(candidates, dependency_graph, usage_tracker)?;
        
        // TODO: Actually remove functions from module registry
        // This would involve:
        // 1. Removing function exports from modules
        // 2. Updating module metadata
        // 3. Rebuilding dependency relationships
        // 4. Validating remaining functions still work
        
        Ok(results)
    }
    
    fn create_usage_snapshot(
        &self,
        function_name: &str,
        usage_tracker: &UsageTracker,
    ) -> FunctionUsageSnapshot {
        if let Some(stats) = usage_tracker.get_function_stats(function_name) {
            FunctionUsageSnapshot {
                call_count: stats.call_count,
                unique_callers: stats.unique_callers,
                is_critical: stats.is_on_critical_path,
                performance_score: stats.performance_impact.bottleneck_score,
            }
        } else {
            FunctionUsageSnapshot {
                call_count: 0,
                unique_callers: 0,
                is_critical: false,
                performance_score: 0.0,
            }
        }
    }
    
    fn determine_preservation_reason(
        &self,
        function_name: &str,
        node: &super::DependencyNode,
        usage_tracker: &UsageTracker,
    ) -> PreservationReason {
        if node.is_entry_point {
            PreservationReason::EntryPoint
        } else if let Some(stats) = usage_tracker.get_function_stats(function_name) {
            if stats.call_count >= self.config.min_usage_threshold {
                PreservationReason::AboveUsageThreshold {
                    threshold: self.config.min_usage_threshold,
                    actual: stats.call_count,
                }
            } else if stats.is_on_critical_path {
                PreservationReason::CriticalPath
            } else {
                PreservationReason::SafetyValidation {
                    reason: "Function preserved by safety validation".to_string(),
                }
            }
        } else {
            PreservationReason::SafetyValidation {
                reason: "No usage data available".to_string(),
            }
        }
    }
    
    fn create_elimination_summary(
        &self,
        eliminated_functions: &[EliminatedFunction],
        preserved_functions: &[PreservedFunction],
    ) -> EliminationSummary {
        let total_functions = eliminated_functions.len() + preserved_functions.len();
        let total_size_eliminated: usize = eliminated_functions.iter()
            .map(|f| f.estimated_size)
            .sum();
        
        EliminationSummary {
            total_functions,
            functions_eliminated: eliminated_functions.len(),
            functions_preserved: preserved_functions.len(),
            total_size_eliminated,
            elimination_percentage: if total_functions > 0 {
                (eliminated_functions.len() as f64 / total_functions as f64) * 100.0
            } else {
                0.0
            },
            memory_savings: total_size_eliminated,
            performance_improvement: self.estimate_performance_improvement(eliminated_functions),
        }
    }
    
    fn estimate_performance_impact(&self, eliminated_functions: &[EliminatedFunction]) -> PerformanceImpact {
        let total_size_eliminated: usize = eliminated_functions.iter()
            .map(|f| f.estimated_size)
            .sum();
        
        PerformanceImpact {
            compilation_time_improvement: eliminated_functions.len() as f64 * 0.1, // 0.1% per function
            binary_size_reduction: total_size_eliminated,
            memory_usage_reduction: total_size_eliminated,
            startup_time_improvement: eliminated_functions.len() as f64 * 0.05, // 0.05% per function
        }
    }
    
    fn estimate_performance_improvement(&self, eliminated_functions: &[EliminatedFunction]) -> f64 {
        // Simple heuristic: 1% improvement per 10 eliminated functions
        (eliminated_functions.len() as f64 / 10.0).min(50.0) // Cap at 50%
    }
}

/// Candidate for elimination
#[derive(Debug, Clone)]
struct EliminationCandidate {
    function_name: String,
    module_name: String,
    elimination_reason: EliminationReason,
    priority: f64,
    estimated_size: usize,
    dependencies: Vec<String>,
    dependents: Vec<String>,
}

/// Analysis of elimination potential
#[derive(Debug, Clone)]
pub struct EliminationAnalysis {
    /// Total functions analyzed
    pub total_functions: usize,
    
    /// Functions that can be eliminated
    pub eliminable_functions: usize,
    
    /// Total size that can be eliminated
    pub total_size_eliminable: usize,
    
    /// Breakdown by elimination category
    pub categories: HashMap<String, usize>,
}

impl SafetyValidator {
    /// Create a new safety validator
    pub fn new() -> Self {
        let mut protected_functions = HashSet::new();
        
        // Always protect core entry point functions
        protected_functions.insert("Sin".to_string());
        protected_functions.insert("Cos".to_string());
        protected_functions.insert("Length".to_string());
        protected_functions.insert("Head".to_string());
        protected_functions.insert("Array".to_string());
        protected_functions.insert("Map".to_string());
        protected_functions.insert("Apply".to_string());
        
        SafetyValidator {
            protected_functions,
            validation_rules: Self::create_default_rules(),
        }
    }
    
    /// Validate elimination candidates for safety
    pub fn validate_candidates(
        &self,
        candidates: &[EliminationCandidate],
        dependency_graph: &DependencyGraph,
    ) -> Result<Vec<EliminationCandidate>, TreeShakeError> {
        let mut validated = Vec::new();
        
        for candidate in candidates {
            if self.is_safe_to_eliminate(&candidate.function_name, dependency_graph) {
                validated.push(candidate.clone());
            }
        }
        
        Ok(validated)
    }
    
    fn is_safe_to_eliminate(&self, function_name: &str, dependency_graph: &DependencyGraph) -> bool {
        // Check protected functions list
        if self.protected_functions.contains(function_name) {
            return false;
        }
        
        // Apply validation rules
        for rule in &self.validation_rules {
            if !(rule.check)(function_name, dependency_graph) {
                return false;
            }
        }
        
        true
    }
    
    fn create_default_rules() -> Vec<ValidationRule> {
        vec![
            ValidationRule {
                name: "no_external_dependents".to_string(),
                description: "Function should not have external dependents".to_string(),
                check: |function_name, dependency_graph| {
                    // For simplicity, allow all functions for now
                    // In a real implementation, this would check for external dependencies
                    true
                },
            },
            ValidationRule {
                name: "not_critical_path".to_string(),
                description: "Function should not be on critical execution path".to_string(),
                check: |function_name, dependency_graph| {
                    // For simplicity, allow all functions for now
                    // In a real implementation, this would check critical path analysis
                    true
                },
            },
        ]
    }
}

impl Default for Eliminator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eliminator_creation() {
        let eliminator = Eliminator::new();
        assert!(!eliminator.config.aggressive_mode);
        assert!(eliminator.config.preserve_entry_points);
        assert_eq!(eliminator.config.min_usage_threshold, 0);
    }

    #[test]
    fn test_eliminator_config() {
        let config = EliminatorConfig {
            aggressive_mode: true,
            min_usage_threshold: 10,
            preview_mode: true,
            ..Default::default()
        };
        
        let eliminator = Eliminator::with_config(config);
        assert!(eliminator.config.aggressive_mode);
        assert_eq!(eliminator.config.min_usage_threshold, 10);
        assert!(eliminator.config.preview_mode);
    }

    #[test]
    fn test_elimination_reason_variants() {
        let reason1 = EliminationReason::NeverCalled;
        let reason2 = EliminationReason::BelowUsageThreshold { threshold: 10, actual: 5 };
        let reason3 = EliminationReason::DeadCode;
        
        match reason1 {
            EliminationReason::NeverCalled => assert!(true),
            _ => assert!(false),
        }
        
        match reason2 {
            EliminationReason::BelowUsageThreshold { threshold, actual } => {
                assert_eq!(threshold, 10);
                assert_eq!(actual, 5);
            },
            _ => assert!(false),
        }
    }

    #[test]
    fn test_preservation_reason_variants() {
        let reason1 = PreservationReason::EntryPoint;
        let reason2 = PreservationReason::AboveUsageThreshold { threshold: 10, actual: 25 };
        let reason3 = PreservationReason::CriticalPath;
        
        match reason1 {
            PreservationReason::EntryPoint => assert!(true),
            _ => assert!(false),
        }
        
        match reason2 {
            PreservationReason::AboveUsageThreshold { threshold, actual } => {
                assert_eq!(threshold, 10);
                assert_eq!(actual, 25);
            },
            _ => assert!(false),
        }
    }

    #[test]
    fn test_safety_validator() {
        let validator = SafetyValidator::new();
        assert!(validator.protected_functions.contains("Sin"));
        assert!(validator.protected_functions.contains("Array"));
        assert!(!validator.protected_functions.contains("UnknownFunction"));
    }

    #[test]
    fn test_warning_severity() {
        let warning = EliminationWarning {
            warning_type: WarningType::SafetyConcern,
            function_name: "test_function".to_string(),
            message: "Test warning".to_string(),
            severity: WarningSeverity::High,
        };
        
        match warning.severity {
            WarningSeverity::High => assert!(true),
            _ => assert!(false),
        }
    }

    #[test]
    fn test_performance_impact() {
        let impact = PerformanceImpact {
            compilation_time_improvement: 5.0,
            binary_size_reduction: 1024,
            memory_usage_reduction: 512,
            startup_time_improvement: 2.5,
        };
        
        assert_eq!(impact.compilation_time_improvement, 5.0);
        assert_eq!(impact.binary_size_reduction, 1024);
        assert_eq!(impact.memory_usage_reduction, 512);
        assert_eq!(impact.startup_time_improvement, 2.5);
    }

    #[test]
    fn test_elimination_summary() {
        let summary = EliminationSummary {
            total_functions: 100,
            functions_eliminated: 25,
            functions_preserved: 75,
            total_size_eliminated: 5000,
            elimination_percentage: 25.0,
            memory_savings: 5000,
            performance_improvement: 12.5,
        };
        
        assert_eq!(summary.total_functions, 100);
        assert_eq!(summary.functions_eliminated, 25);
        assert_eq!(summary.elimination_percentage, 25.0);
    }
}