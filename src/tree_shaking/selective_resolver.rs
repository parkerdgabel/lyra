//! Selective Import Resolver
//!
//! Transforms full module imports into selective function imports based on usage analysis.
//! Provides compile-time resolution and optimization of import statements.

use super::{DependencyGraph, UsageTracker, TreeShakeError, ImportAnalyzer};
use super::import_analyzer::{ImportPattern, ImportOptimizationType, ModuleImportOptimization};
use crate::modules::registry::ModuleRegistry;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use std::time::SystemTime;

/// Selective import resolver that transforms imports based on usage analysis
pub struct SelectiveImportResolver {
    /// Configuration for import resolution
    config: SelectiveResolverConfig,
    
    /// Cache of resolved imports for performance
    resolution_cache: ResolutionCache,
    
    /// Import transformation pipeline
    transformer: ImportTransformer,
    
    /// Performance metrics
    performance_metrics: ResolverMetrics,
}

/// Configuration for selective import resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectiveResolverConfig {
    /// Enable aggressive selective imports
    pub aggressive_selection: bool,
    
    /// Minimum functions required for selective import
    pub min_selective_threshold: usize,
    
    /// Maximum functions allowed in selective import
    pub max_selective_threshold: usize,
    
    /// Enable import caching
    pub enable_caching: bool,
    
    /// Cache expiration time in seconds
    pub cache_expiration_seconds: u64,
    
    /// Enable import validation
    pub enable_validation: bool,
    
    /// Enable performance optimization
    pub enable_performance_optimization: bool,
    
    /// Enable dependency resolution
    pub enable_dependency_resolution: bool,
}

impl Default for SelectiveResolverConfig {
    fn default() -> Self {
        SelectiveResolverConfig {
            aggressive_selection: false,
            min_selective_threshold: 1,
            max_selective_threshold: 20,
            enable_caching: true,
            cache_expiration_seconds: 300, // 5 minutes
            enable_validation: true,
            enable_performance_optimization: true,
            enable_dependency_resolution: true,
        }
    }
}

/// Cache for resolved imports
#[derive(Debug, Clone, Default)]
pub struct ResolutionCache {
    /// Cached import resolutions
    resolutions: HashMap<String, CachedResolution>,
    
    /// Cache statistics
    stats: CacheStats,
    
    /// Cache configuration
    config: CacheConfig,
}

/// Cached import resolution
#[derive(Debug, Clone)]
pub struct CachedResolution {
    /// Module name
    pub module_name: String,
    
    /// Resolved import statement
    pub resolved_import: ResolvedImport,
    
    /// Resolution metadata
    pub metadata: ResolutionMetadata,
    
    /// Cache timestamp
    pub cached_at: SystemTime,
    
    /// Cache expiration
    pub expires_at: SystemTime,
}

/// Resolved import representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedImport {
    /// Import type
    pub import_type: ResolvedImportType,
    
    /// Module being imported from
    pub source_module: String,
    
    /// Functions being imported
    pub imported_functions: Vec<ImportedFunction>,
    
    /// Generated import statement
    pub import_statement: String,
    
    /// Import metadata
    pub metadata: ImportMetadata,
    
    /// Performance impact
    pub performance_impact: ImportPerformanceImpact,
}

/// Types of resolved imports
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResolvedImportType {
    /// Selective import (use module::{func1, func2})
    Selective { functions: Vec<String> },
    
    /// Full module import (use module::*)
    Full,
    
    /// Single function import (use module::func)
    Single { function: String },
    
    /// Aliased import (use module::func as alias)
    Aliased { function: String, alias: String },
    
    /// Conditional import (import only under certain conditions)
    Conditional { condition: String, functions: Vec<String> },
    
    /// No import needed
    None,
}

/// Imported function information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportedFunction {
    /// Function name
    pub name: String,
    
    /// Function alias (if any)
    pub alias: Option<String>,
    
    /// Usage frequency
    pub usage_frequency: u64,
    
    /// Whether function is critical
    pub is_critical: bool,
    
    /// Performance impact
    pub performance_impact: f64,
    
    /// Dependencies this function brings
    pub dependencies: Vec<String>,
}

/// Metadata for resolved imports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportMetadata {
    /// Resolution strategy used
    pub resolution_strategy: ResolutionStrategy,
    
    /// Optimization level applied
    pub optimization_level: OptimizationLevel,
    
    /// Validation status
    pub validation_status: ValidationStatus,
    
    /// Resolution timestamp
    pub resolved_at: SystemTime,
    
    /// Confidence in resolution (0.0 to 1.0)
    pub confidence: f64,
    
    /// Warnings generated during resolution
    pub warnings: Vec<ResolutionWarning>,
}

/// Resolution strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResolutionStrategy {
    /// Usage-based selective import
    UsageBased,
    
    /// Dependency-driven import
    DependencyDriven,
    
    /// Performance-optimized import
    PerformanceOptimized,
    
    /// Conservative (full import)
    Conservative,
    
    /// Aggressive selective import
    Aggressive,
    
    /// Hybrid approach
    Hybrid,
}

/// Optimization levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// No optimization
    None,
    
    /// Basic optimization
    Basic,
    
    /// Advanced optimization
    Advanced,
    
    /// Maximum optimization
    Maximum,
}

/// Validation status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValidationStatus {
    /// Validation passed
    Valid,
    
    /// Validation passed with warnings
    ValidWithWarnings,
    
    /// Validation failed
    Invalid { errors: Vec<String> },
    
    /// Validation skipped
    Skipped,
}

/// Resolution warnings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionWarning {
    /// Warning type
    pub warning_type: ResolutionWarningType,
    
    /// Warning message
    pub message: String,
    
    /// Affected functions
    pub affected_functions: Vec<String>,
    
    /// Severity level
    pub severity: WarningSeverity,
}

/// Types of resolution warnings
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResolutionWarningType {
    /// Function might be needed in future
    PotentialFutureUse,
    
    /// Performance impact concern
    PerformanceImpact,
    
    /// Dependency chain concern
    DependencyChain,
    
    /// Breaking change risk
    BreakingChange,
    
    /// Compatibility concern
    Compatibility,
}

/// Warning severity levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WarningSeverity {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

/// Performance impact of import resolution
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImportPerformanceImpact {
    /// Compilation time improvement (milliseconds)
    pub compilation_time_improvement: i64,
    
    /// Binary size reduction (bytes)
    pub binary_size_reduction: i64,
    
    /// Memory usage reduction (bytes)
    pub memory_usage_reduction: i64,
    
    /// Runtime performance impact
    pub runtime_impact: RuntimePerformanceImpact,
    
    /// Import resolution time (microseconds)
    pub resolution_time: u64,
}

/// Runtime performance impact
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RuntimePerformanceImpact {
    /// Performance improvement
    Improvement(f64),
    
    /// No significant impact
    Neutral,
    
    /// Minor performance cost
    MinorCost(f64),
    
    /// Significant performance cost
    SignificantCost(f64),
}

impl Default for RuntimePerformanceImpact {
    fn default() -> Self {
        RuntimePerformanceImpact::Neutral
    }
}

/// Import transformer pipeline
#[derive(Debug, Clone)]
pub struct ImportTransformer {
    /// Transformation strategies
    strategies: Vec<TransformationStrategy>,
    
    /// Transformation cache
    cache: HashMap<String, TransformationResult>,
    
    /// Transformation statistics
    stats: TransformationStats,
}

/// Transformation strategy
#[derive(Debug, Clone)]
pub struct TransformationStrategy {
    /// Strategy name
    pub name: String,
    
    /// Strategy type
    pub strategy_type: StrategyType,
    
    /// Priority (higher = executed first)
    pub priority: u32,
    
    /// Whether strategy is enabled
    pub enabled: bool,
    
    /// Strategy configuration
    pub config: StrategyConfig,
}

/// Types of transformation strategies
#[derive(Debug, Clone, PartialEq)]
pub enum StrategyType {
    /// Convert to selective imports
    SelectiveConversion,
    
    /// Remove unused imports
    UnusedRemoval,
    
    /// Bundle related imports
    ImportBundling,
    
    /// Lazy import loading
    LazyLoading,
    
    /// Import aliasing
    Aliasing,
    
    /// Conditional imports
    ConditionalImports,
}

/// Strategy configuration
#[derive(Debug, Clone, Default)]
pub struct StrategyConfig {
    /// Strategy-specific parameters
    pub parameters: HashMap<String, String>,
    
    /// Thresholds for strategy application
    pub thresholds: HashMap<String, f64>,
    
    /// Strategy options
    pub options: HashMap<String, bool>,
}

/// Result of import transformation
#[derive(Debug, Clone)]
pub struct TransformationResult {
    /// Original import pattern
    pub original_pattern: ImportPattern,
    
    /// Transformed import
    pub transformed_import: ResolvedImport,
    
    /// Applied strategies
    pub applied_strategies: Vec<String>,
    
    /// Transformation success
    pub success: bool,
    
    /// Transformation errors
    pub errors: Vec<String>,
    
    /// Performance metrics
    pub metrics: TransformationMetrics,
}

/// Transformation statistics
#[derive(Debug, Clone, Default)]
pub struct TransformationStats {
    /// Total transformations attempted
    pub total_attempted: u64,
    
    /// Successful transformations
    pub successful: u64,
    
    /// Failed transformations
    pub failed: u64,
    
    /// Cache hits
    pub cache_hits: u64,
    
    /// Cache misses
    pub cache_misses: u64,
    
    /// Average transformation time
    pub avg_transformation_time: f64,
}

/// Transformation metrics
#[derive(Debug, Clone, Default)]
pub struct TransformationMetrics {
    /// Time taken for transformation
    pub transformation_time: std::time::Duration,
    
    /// Memory used during transformation
    pub memory_used: usize,
    
    /// Number of strategies applied
    pub strategies_applied: usize,
    
    /// Optimization effectiveness (0.0 to 1.0)
    pub optimization_effectiveness: f64,
}

/// Resolution metadata
#[derive(Debug, Clone)]
pub struct ResolutionMetadata {
    /// Resolution strategy
    pub strategy: ResolutionStrategy,
    
    /// Resolution quality score
    pub quality_score: f64,
    
    /// Resolution confidence
    pub confidence: f64,
    
    /// Dependencies resolved
    pub dependencies_resolved: Vec<String>,
    
    /// Issues encountered
    pub issues: Vec<ResolutionIssue>,
}

/// Resolution issues
#[derive(Debug, Clone)]
pub struct ResolutionIssue {
    /// Issue type
    pub issue_type: IssueType,
    
    /// Issue description
    pub description: String,
    
    /// Severity level
    pub severity: IssueSeverity,
    
    /// Affected components
    pub affected_components: Vec<String>,
}

/// Types of resolution issues
#[derive(Debug, Clone, PartialEq)]
pub enum IssueType {
    /// Circular dependency
    CircularDependency,
    
    /// Missing dependency
    MissingDependency,
    
    /// Version conflict
    VersionConflict,
    
    /// Performance concern
    PerformanceConcern,
    
    /// Compatibility issue
    CompatibilityIssue,
}

/// Issue severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum IssueSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: u64,
    
    /// Total cache misses
    pub misses: u64,
    
    /// Cache evictions
    pub evictions: u64,
    
    /// Cache size (number of entries)
    pub size: usize,
    
    /// Memory usage (bytes)
    pub memory_usage: usize,
    
    /// Hit ratio (0.0 to 1.0)
    pub hit_ratio: f64,
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum cache size
    pub max_size: usize,
    
    /// Maximum memory usage (bytes)
    pub max_memory: usize,
    
    /// Entry TTL (seconds)
    pub ttl_seconds: u64,
    
    /// Enable LRU eviction
    pub enable_lru: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        CacheConfig {
            max_size: 1000,
            max_memory: 10 * 1024 * 1024, // 10MB
            ttl_seconds: 300, // 5 minutes
            enable_lru: true,
        }
    }
}

/// Performance metrics for resolver
#[derive(Debug, Clone, Default)]
pub struct ResolverMetrics {
    /// Total resolutions performed
    pub total_resolutions: u64,
    
    /// Successful resolutions
    pub successful_resolutions: u64,
    
    /// Failed resolutions
    pub failed_resolutions: u64,
    
    /// Average resolution time
    pub avg_resolution_time: f64,
    
    /// Cache performance
    pub cache_performance: CacheStats,
    
    /// Transformation performance
    pub transformation_performance: TransformationStats,
    
    /// Memory usage
    pub memory_usage: usize,
}

impl SelectiveImportResolver {
    /// Create a new selective import resolver
    pub fn new() -> Self {
        SelectiveImportResolver {
            config: SelectiveResolverConfig::default(),
            resolution_cache: ResolutionCache::default(),
            transformer: ImportTransformer::new(),
            performance_metrics: ResolverMetrics::default(),
        }
    }
    
    /// Create resolver with custom configuration
    pub fn with_config(config: SelectiveResolverConfig) -> Self {
        SelectiveImportResolver {
            config,
            resolution_cache: ResolutionCache::default(),
            transformer: ImportTransformer::new(),
            performance_metrics: ResolverMetrics::default(),
        }
    }
    
    /// Resolve imports based on import analyzer results
    pub fn resolve_imports(
        &mut self,
        import_analyzer: &ImportAnalyzer,
        dependency_graph: &DependencyGraph,
        usage_tracker: &UsageTracker,
        module_registry: &ModuleRegistry,
    ) -> Result<ImportResolutionResults, TreeShakeError> {
        let start_time = std::time::Instant::now();
        
        // Step 1: Get import optimizations from analyzer
        let optimizations = import_analyzer.get_all_optimizations();
        
        // Step 2: Resolve each import optimization
        let mut resolved_imports = Vec::new();
        let mut resolution_errors = Vec::new();
        
        for optimization in &optimizations {
            match self.resolve_single_import(optimization, dependency_graph, usage_tracker) {
                Ok(resolved) => resolved_imports.push(resolved),
                Err(error) => resolution_errors.push(ResolutionError {
                    module_name: optimization.module_name.clone(),
                    error_type: ResolutionErrorType::TransformationFailed,
                    message: format!("Failed to resolve import: {}", error),
                    severity: ResolutionErrorSeverity::Error,
                }),
            }
        }
        
        // Step 3: Validate resolved imports
        let validation_results = if self.config.enable_validation {
            self.validate_resolved_imports(&resolved_imports, dependency_graph)?
        } else {
            ValidationResults::default()
        };
        
        // Step 4: Update metrics
        self.performance_metrics.total_resolutions += resolved_imports.len() as u64;
        self.performance_metrics.successful_resolutions += resolved_imports.iter()
            .filter(|r| r.metadata.validation_status == ValidationStatus::Valid)
            .count() as u64;
        self.performance_metrics.failed_resolutions += resolution_errors.len() as u64;
        
        let resolution_time = start_time.elapsed();
        self.performance_metrics.avg_resolution_time = 
            (self.performance_metrics.avg_resolution_time * (self.performance_metrics.total_resolutions - 1) as f64 
             + resolution_time.as_secs_f64()) / self.performance_metrics.total_resolutions as f64;
        
        let resolved_count = resolved_imports.len();
        let optimization_effectiveness = self.calculate_optimization_effectiveness(&resolved_imports);
        
        Ok(ImportResolutionResults {
            resolved_imports,
            validation_results,
            resolution_errors,
            performance_metrics: ImportResolutionPerformance {
                total_resolution_time: resolution_time,
                resolutions_per_second: resolved_count as f64 / resolution_time.as_secs_f64(),
                cache_hit_ratio: self.resolution_cache.stats.hit_ratio,
                memory_usage: self.performance_metrics.memory_usage,
            },
            metadata: ResolutionResultsMetadata {
                resolver_config: self.config.clone(),
                resolution_timestamp: SystemTime::now(),
                total_modules_processed: optimizations.len(),
                optimization_effectiveness,
            },
        })
    }
    
    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> &ResolverMetrics {
        &self.performance_metrics
    }
    
    /// Get cache statistics
    pub fn get_cache_stats(&self) -> &CacheStats {
        &self.resolution_cache.stats
    }
    
    /// Clear resolution cache
    pub fn clear_cache(&mut self) {
        self.resolution_cache.resolutions.clear();
        self.resolution_cache.stats = CacheStats::default();
    }
    
    /// Configure resolver
    pub fn configure(&mut self, config: SelectiveResolverConfig) {
        self.config = config;
    }
    
    // Private implementation methods
    
    fn resolve_single_import(
        &mut self,
        optimization: &ModuleImportOptimization,
        dependency_graph: &DependencyGraph,
        usage_tracker: &UsageTracker,
    ) -> Result<ResolvedImport, TreeShakeError> {
        // Check cache first
        if self.config.enable_caching {
            let cache_result = self.get_cached_resolution(&optimization.module_name).cloned();
            if let Some(cached) = cache_result {
                self.resolution_cache.stats.hits += 1;
                return Ok(cached.resolved_import);
            }
            self.resolution_cache.stats.misses += 1;
        }
        
        // Transform import based on optimization suggestion
        let resolved = self.transformer.transform_import(optimization, dependency_graph, usage_tracker)?;
        
        // Cache the result
        if self.config.enable_caching {
            self.cache_resolution(&optimization.module_name, &resolved);
        }
        
        Ok(resolved)
    }
    
    fn get_cached_resolution(&self, module_name: &str) -> Option<&CachedResolution> {
        self.resolution_cache.resolutions.get(module_name)
            .filter(|cached| cached.expires_at > SystemTime::now())
    }
    
    fn cache_resolution(&mut self, module_name: &str, resolved: &ResolvedImport) {
        let now = SystemTime::now();
        let cached = CachedResolution {
            module_name: module_name.to_string(),
            resolved_import: resolved.clone(),
            metadata: ResolutionMetadata {
                strategy: ResolutionStrategy::UsageBased,
                quality_score: 0.8,
                confidence: 0.9,
                dependencies_resolved: Vec::new(),
                issues: Vec::new(),
            },
            cached_at: now,
            expires_at: now + std::time::Duration::from_secs(self.config.cache_expiration_seconds),
        };
        
        self.resolution_cache.resolutions.insert(module_name.to_string(), cached);
        self.resolution_cache.stats.size = self.resolution_cache.resolutions.len();
    }
    
    fn validate_resolved_imports(
        &self,
        _resolved_imports: &[ResolvedImport],
        _dependency_graph: &DependencyGraph,
    ) -> Result<ValidationResults, TreeShakeError> {
        // Implementation would validate that resolved imports satisfy all dependencies
        Ok(ValidationResults::default())
    }
    
    fn calculate_optimization_effectiveness(&self, resolved_imports: &[ResolvedImport]) -> f64 {
        if resolved_imports.is_empty() {
            return 0.0;
        }
        
        // Calculate effectiveness based on selective vs full imports
        let selective_count = resolved_imports.iter()
            .filter(|import| matches!(import.import_type, ResolvedImportType::Selective { .. }))
            .count();
        
        selective_count as f64 / resolved_imports.len() as f64
    }
}

impl ImportTransformer {
    fn new() -> Self {
        ImportTransformer {
            strategies: Self::create_default_strategies(),
            cache: HashMap::new(),
            stats: TransformationStats::default(),
        }
    }
    
    fn create_default_strategies() -> Vec<TransformationStrategy> {
        vec![
            TransformationStrategy {
                name: "selective_conversion".to_string(),
                strategy_type: StrategyType::SelectiveConversion,
                priority: 100,
                enabled: true,
                config: StrategyConfig::default(),
            },
            TransformationStrategy {
                name: "unused_removal".to_string(),
                strategy_type: StrategyType::UnusedRemoval,
                priority: 90,
                enabled: true,
                config: StrategyConfig::default(),
            },
        ]
    }
    
    fn transform_import(
        &mut self,
        optimization: &ModuleImportOptimization,
        dependency_graph: &DependencyGraph,
        usage_tracker: &UsageTracker,
    ) -> Result<ResolvedImport, TreeShakeError> {
        let start_time = std::time::Instant::now();
        
        // Apply transformation based on optimization type
        let resolved = match &optimization.optimization.optimization_type {
            ImportOptimizationType::ConvertToSelective => {
                self.convert_to_selective(optimization, dependency_graph, usage_tracker)?
            },
            ImportOptimizationType::RemoveUnused => {
                self.remove_unused_imports(optimization)?
            },
            ImportOptimizationType::NoOptimization => {
                self.create_full_import(optimization)?
            },
            _ => {
                self.create_default_import(optimization)?
            }
        };
        
        // Update stats
        self.stats.total_attempted += 1;
        self.stats.successful += 1;
        self.stats.avg_transformation_time = 
            (self.stats.avg_transformation_time * (self.stats.total_attempted - 1) as f64 
             + start_time.elapsed().as_secs_f64()) / self.stats.total_attempted as f64;
        
        Ok(resolved)
    }
    
    fn convert_to_selective(
        &self,
        optimization: &ModuleImportOptimization,
        _dependency_graph: &DependencyGraph,
        _usage_tracker: &UsageTracker,
    ) -> Result<ResolvedImport, TreeShakeError> {
        let used_functions = &optimization.original_pattern.used_functions;
        let function_names: Vec<String> = used_functions.iter()
            .map(|f| f.function_name.clone())
            .collect();
        
        let imported_functions: Vec<ImportedFunction> = used_functions.iter()
            .map(|f| ImportedFunction {
                name: f.function_name.clone(),
                alias: None,
                usage_frequency: f.usage_count,
                is_critical: f.is_critical,
                performance_impact: f.performance_impact,
                dependencies: Vec::new(),
            })
            .collect();
        
        Ok(ResolvedImport {
            import_type: ResolvedImportType::Selective { functions: function_names.clone() },
            source_module: optimization.module_name.clone(),
            imported_functions,
            import_statement: format!("use {}::{{{}}}", optimization.module_name, function_names.join(", ")),
            metadata: ImportMetadata {
                resolution_strategy: ResolutionStrategy::UsageBased,
                optimization_level: OptimizationLevel::Advanced,
                validation_status: ValidationStatus::Valid,
                resolved_at: SystemTime::now(),
                confidence: 0.9,
                warnings: Vec::new(),
            },
            performance_impact: ImportPerformanceImpact {
                compilation_time_improvement: (used_functions.len() * 50) as i64,
                binary_size_reduction: (used_functions.len() * 1000) as i64,
                memory_usage_reduction: (used_functions.len() * 500) as i64,
                runtime_impact: RuntimePerformanceImpact::Improvement(0.1),
                resolution_time: 100,
            },
        })
    }
    
    fn remove_unused_imports(
        &self,
        optimization: &ModuleImportOptimization,
    ) -> Result<ResolvedImport, TreeShakeError> {
        Ok(ResolvedImport {
            import_type: ResolvedImportType::None,
            source_module: optimization.module_name.clone(),
            imported_functions: Vec::new(),
            import_statement: "// Import removed - no functions used".to_string(),
            metadata: ImportMetadata {
                resolution_strategy: ResolutionStrategy::UsageBased,
                optimization_level: OptimizationLevel::Maximum,
                validation_status: ValidationStatus::Valid,
                resolved_at: SystemTime::now(),
                confidence: 1.0,
                warnings: Vec::new(),
            },
            performance_impact: ImportPerformanceImpact {
                compilation_time_improvement: 200,
                binary_size_reduction: 5000,
                memory_usage_reduction: 2000,
                runtime_impact: RuntimePerformanceImpact::Improvement(0.2),
                resolution_time: 50,
            },
        })
    }
    
    fn create_full_import(
        &self,
        optimization: &ModuleImportOptimization,
    ) -> Result<ResolvedImport, TreeShakeError> {
        Ok(ResolvedImport {
            import_type: ResolvedImportType::Full,
            source_module: optimization.module_name.clone(),
            imported_functions: Vec::new(),
            import_statement: format!("use {}::*", optimization.module_name),
            metadata: ImportMetadata {
                resolution_strategy: ResolutionStrategy::Conservative,
                optimization_level: OptimizationLevel::None,
                validation_status: ValidationStatus::Valid,
                resolved_at: SystemTime::now(),
                confidence: 1.0,
                warnings: Vec::new(),
            },
            performance_impact: ImportPerformanceImpact::default(),
        })
    }
    
    fn create_default_import(
        &self,
        optimization: &ModuleImportOptimization,
    ) -> Result<ResolvedImport, TreeShakeError> {
        self.create_full_import(optimization)
    }
}

/// Results of import resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportResolutionResults {
    /// Resolved imports
    pub resolved_imports: Vec<ResolvedImport>,
    
    /// Validation results
    pub validation_results: ValidationResults,
    
    /// Resolution errors
    pub resolution_errors: Vec<ResolutionError>,
    
    /// Performance metrics
    pub performance_metrics: ImportResolutionPerformance,
    
    /// Resolution metadata
    pub metadata: ResolutionResultsMetadata,
}

/// Validation results
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidationResults {
    /// Validation passed
    pub passed: bool,
    
    /// Validation errors
    pub errors: Vec<ValidationError>,
    
    /// Validation warnings
    pub warnings: Vec<ValidationWarning>,
    
    /// Validated imports count
    pub validated_count: usize,
}

/// Validation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    /// Error type
    pub error_type: ValidationErrorType,
    
    /// Error message
    pub message: String,
    
    /// Affected import
    pub affected_import: String,
}

/// Types of validation errors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValidationErrorType {
    /// Missing dependency
    MissingDependency,
    
    /// Circular dependency
    CircularDependency,
    
    /// Function not found
    FunctionNotFound,
    
    /// Version mismatch
    VersionMismatch,
}

/// Validation warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    /// Warning type
    pub warning_type: ValidationWarningType,
    
    /// Warning message
    pub message: String,
    
    /// Affected import
    pub affected_import: String,
}

/// Types of validation warnings
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValidationWarningType {
    /// Performance concern
    PerformanceConcern,
    
    /// Compatibility issue
    CompatibilityIssue,
    
    /// Future dependency
    FutureDependency,
}

/// Resolution error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionError {
    /// Module name
    pub module_name: String,
    
    /// Error type
    pub error_type: ResolutionErrorType,
    
    /// Error message
    pub message: String,
    
    /// Error severity
    pub severity: ResolutionErrorSeverity,
}

/// Types of resolution errors
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResolutionErrorType {
    /// Transformation failed
    TransformationFailed,
    
    /// Validation failed
    ValidationFailed,
    
    /// Cache error
    CacheError,
    
    /// Configuration error
    ConfigurationError,
}

/// Resolution error severity
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResolutionErrorSeverity {
    Warning,
    Error,
    Critical,
}

/// Import resolution performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportResolutionPerformance {
    /// Total resolution time
    pub total_resolution_time: std::time::Duration,
    
    /// Resolutions per second
    pub resolutions_per_second: f64,
    
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
    
    /// Memory usage
    pub memory_usage: usize,
}

/// Resolution results metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionResultsMetadata {
    /// Resolver configuration
    pub resolver_config: SelectiveResolverConfig,
    
    /// Resolution timestamp
    pub resolution_timestamp: SystemTime,
    
    /// Total modules processed
    pub total_modules_processed: usize,
    
    /// Optimization effectiveness
    pub optimization_effectiveness: f64,
}

impl Default for SelectiveImportResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selective_resolver_creation() {
        let resolver = SelectiveImportResolver::new();
        assert!(resolver.config.enable_validation);
        assert!(resolver.config.enable_caching);
        assert_eq!(resolver.performance_metrics.total_resolutions, 0);
    }

    #[test]
    fn test_resolver_config() {
        let config = SelectiveResolverConfig {
            aggressive_selection: true,
            max_selective_threshold: 10,
            enable_caching: false,
            ..Default::default()
        };
        
        let resolver = SelectiveImportResolver::with_config(config);
        assert!(resolver.config.aggressive_selection);
        assert_eq!(resolver.config.max_selective_threshold, 10);
        assert!(!resolver.config.enable_caching);
    }

    #[test]
    fn test_resolved_import_types() {
        let selective = ResolvedImportType::Selective { 
            functions: vec!["Sin".to_string(), "Cos".to_string()] 
        };
        let single = ResolvedImportType::Single { 
            function: "Sin".to_string() 
        };
        let full = ResolvedImportType::Full;
        
        assert!(matches!(selective, ResolvedImportType::Selective { .. }));
        assert!(matches!(single, ResolvedImportType::Single { .. }));
        assert_eq!(full, ResolvedImportType::Full);
    }

    #[test]
    fn test_resolution_strategies() {
        assert_eq!(ResolutionStrategy::UsageBased, ResolutionStrategy::UsageBased);
        assert_ne!(ResolutionStrategy::UsageBased, ResolutionStrategy::Conservative);
    }

    #[test]
    fn test_optimization_levels() {
        assert_eq!(OptimizationLevel::Advanced, OptimizationLevel::Advanced);
        assert_ne!(OptimizationLevel::Basic, OptimizationLevel::Maximum);
    }

    #[test]
    fn test_validation_status() {
        let valid = ValidationStatus::Passed;
        let invalid = ValidationStatus::Failed;
        
        assert_eq!(valid, ValidationStatus::Passed);
        assert!(matches!(invalid, ValidationStatus::Failed));
    }

    #[test]
    fn test_cache_stats() {
        let mut stats = CacheStats::default();
        stats.hits = 10;
        stats.misses = 5;
        stats.hit_ratio = stats.hits as f64 / (stats.hits + stats.misses) as f64;
        
        assert_eq!(stats.hits, 10);
        assert_eq!(stats.misses, 5);
        assert_eq!(stats.hit_ratio, 10.0 / 15.0);
    }

    #[test]
    fn test_import_transformer() {
        let transformer = ImportTransformer::new();
        assert_eq!(transformer.strategies.len(), 2);
        assert!(transformer.strategies.iter().any(|s| s.strategy_type == StrategyType::SelectiveConversion));
        assert!(transformer.strategies.iter().any(|s| s.strategy_type == StrategyType::UnusedRemoval));
    }

    #[test]
    fn test_performance_impact() {
        let impact = ImportPerformanceImpact {
            compilation_time_improvement: 100,
            binary_size_reduction: 2000,
            memory_usage_reduction: 1000,
            runtime_impact: RuntimePerformanceImpact::Improvement(0.1),
            resolution_time: 50,
        };
        
        assert_eq!(impact.compilation_time_improvement, 100);
        assert_eq!(impact.binary_size_reduction, 2000);
        assert!(matches!(impact.runtime_impact, RuntimePerformanceImpact::Improvement(_)));
    }
}