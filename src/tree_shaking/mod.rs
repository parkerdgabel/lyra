//! Tree-Shaking System for Lyra
//!
//! This module implements a comprehensive tree-shaking system that analyzes
//! function dependencies across the stdlib hierarchy to enable dead code elimination.

pub mod dependency_graph;
pub mod graph_analyzer;
pub mod call_analyzer;
pub mod module_deps;
pub mod usage_tracker;
pub mod eliminator;
pub mod import_analyzer;
pub mod selective_resolver;
pub mod compile_time_resolver;
pub mod import_statement_generator;
pub mod dependency_validator;
pub mod import_cache;
pub mod intelligent_pipeline;

pub use dependency_graph::{DependencyGraph, DependencyNode, DependencyEdge, DependencyType, 
                           CallContext as DependencyCallContext, FunctionMetadata, 
                           PerformanceCharacteristics, NodeUsageStats, ComplexityMetrics};
pub use graph_analyzer::{GraphAnalyzer, CentralityMetrics, BottleneckAnalysis};
pub use call_analyzer::{CallAnalyzer, FunctionInfo, CallPattern};
pub use module_deps::{ModuleDependencyAnalyzer, ModuleNode, ModuleOptimization};
pub use usage_tracker::{UsageTracker, UsageStats, FunctionUsageStats};
pub use eliminator::{
    Eliminator, EliminatorConfig, EliminationResults, EliminatedFunction, 
    PreservedFunction, EliminationAnalysis
};
pub use import_analyzer::{
    ImportAnalyzer, ImportAnalyzerConfig, ImportPattern, ImportPatternType,
    ImportOptimizationType, OptimizationSuggestion, ImportOptimizationImpact, 
    ModuleImportOptimization, ImportDependencyGraph
};
pub use selective_resolver::{
    SelectiveImportResolver, SelectiveResolverConfig, ResolvedImport, ResolvedImportType,
    ImportResolutionResults, ResolutionStrategy, OptimizationLevel, ValidationStatus
};
pub use compile_time_resolver::{
    CompileTimeResolver, CompileTimeResolverConfig, CompileTimeResolutionResults,
    ResolvedDependency, DependencyResolutionType, LoadPriority
};
pub use import_statement_generator::{
    ImportStatementGenerator, StatementGeneratorConfig, ImportGenerationResults,
    OutputFormat, FormatGenerator, GenerationMetrics
};
pub use dependency_validator::{
    DependencyValidator, DependencyValidatorConfig, DependencyValidationResults, 
    CircularDependencyDetector, ValidationMetrics, CycleDetectionAlgorithm
};
pub use import_cache::{
    ImportCache, ImportCacheConfig, CachedImportData, CacheStatsSummary,
    CacheOptimizationReport, MemoryCacheStats, DiskCacheStats
};

// Integrated tree-shaking results and analysis (types defined below)

use crate::modules::registry::ModuleRegistry;
use std::collections::HashMap;

/// Tree-shaking coordinator that orchestrates the entire optimization process
pub struct TreeShaker {
    /// Usage tracking system
    usage_tracker: UsageTracker,
    
    /// Dependency graph analyzer
    dependency_graph: DependencyGraph,
    
    /// Module dependency analyzer
    module_deps: ModuleDependencyAnalyzer,
    
    /// Function call analyzer
    call_analyzer: CallAnalyzer,
    
    /// Graph analysis algorithms
    graph_analyzer: GraphAnalyzer,
    
    /// Dead code elimination engine
    eliminator: Eliminator,
    
    /// Import optimization analyzer
    import_analyzer: ImportAnalyzer,
    
    /// Selective import resolver
    selective_resolver: SelectiveImportResolver,
    
    /// Compile-time dependency resolver
    compile_time_resolver: CompileTimeResolver,
    
    /// Import statement generator
    import_statement_generator: ImportStatementGenerator,
    
    /// Dependency validator
    dependency_validator: DependencyValidator,
    
    /// Import cache system
    import_cache: ImportCache,
}

impl TreeShaker {
    /// Create a new tree-shaking system
    pub fn new() -> Self {
        TreeShaker {
            usage_tracker: UsageTracker::new(),
            dependency_graph: DependencyGraph::new(),
            module_deps: ModuleDependencyAnalyzer::new(),
            call_analyzer: CallAnalyzer::new(),
            graph_analyzer: GraphAnalyzer::new(),
            eliminator: Eliminator::new(),
            import_analyzer: ImportAnalyzer::new(),
            selective_resolver: SelectiveImportResolver::new(),
            compile_time_resolver: CompileTimeResolver::new(),
            import_statement_generator: ImportStatementGenerator::new(),
            dependency_validator: DependencyValidator::new(),
            import_cache: ImportCache::new(),
        }
    }
    
    /// Analyze stdlib dependencies and build dependency graph with integrated validation and caching
    pub fn analyze_stdlib(&mut self, module_registry: &ModuleRegistry) -> Result<TreeShakingResults, TreeShakeError> {
        let analysis_start_time = std::time::Instant::now();
        
        // Step 1: Check cache for existing analysis results
        let cache_key = self.generate_analysis_cache_key(module_registry);
        if let Some(cached_results) = self.get_cached_analysis_results(&cache_key)? {
            return Ok(cached_results);
        }
        
        // Step 2: Track all function usage patterns
        self.usage_tracker.track_stdlib_usage(module_registry)?;
        
        // Step 3: Analyze function calls and build dependency graph
        self.call_analyzer.analyze_stdlib_calls(module_registry, &mut self.dependency_graph)?;
        
        // Step 4: Analyze module-level dependencies
        self.module_deps.analyze_module_dependencies(module_registry, &mut self.dependency_graph)?;
        
        // Step 5: Run graph analysis algorithms
        self.graph_analyzer.analyze_graph(&mut self.dependency_graph)?;
        
        // Step 6: Validate dependencies (integrated validation)
        let validation_results = self.dependency_validator.validate_dependencies(
            &self.dependency_graph,
            &self.usage_tracker,
            module_registry,
        ).map_err(|e| TreeShakeError::ValidationError { message: e.to_string() })?;
        
        // Step 7: Cache validation results for future use
        self.cache_validation_results(&validation_results)?;
        
        // Step 8: Analyze import patterns with validation-guided optimization
        self.import_analyzer.analyze_imports(&self.dependency_graph, &self.usage_tracker, module_registry)?;
        
        // Step 9: Warm cache with frequently accessed import patterns
        let frequent_patterns = self.identify_frequent_import_patterns();
        self.warm_import_cache(frequent_patterns)?;
        
        // Step 10: Resolve selective imports with cache-first strategy
        let resolution_results = self.resolve_imports_with_caching(module_registry)?;
        
        // Step 11: Perform compile-time dependency resolution with caching
        let compile_time_results = self.resolve_compile_time_dependencies_with_caching(
            &resolution_results,
            module_registry,
        )?;
        
        // Step 12: Generate optimized import statements with validation
        let import_statements = self.generate_validated_import_statements(
            &resolution_results,
            &compile_time_results,
            &validation_results,
        )?;
        
        // Step 13: Create comprehensive results
        let results = TreeShakingResults {
            dependency_graph_summary: self.create_dependency_graph_summary(),
            usage_statistics: self.usage_tracker.stats().clone(),
            validation_results,
            resolution_results,
            compile_time_results,
            import_statements,
            performance_metrics: TreeShakingPerformanceMetrics {
                total_analysis_time: analysis_start_time.elapsed(),
                cache_hit_ratio: self.calculate_overall_cache_hit_ratio(),
                validation_time: self.dependency_validator.get_performance_metrics().total_validation_time,
                import_resolution_time: resolution_results.performance_metrics.total_resolution_time,
                optimization_effectiveness: self.calculate_optimization_effectiveness(),
            },
            recommendations: self.generate_optimization_recommendations(),
        };
        
        // Step 14: Cache the complete analysis results
        self.cache_analysis_results(&cache_key, &results)?;
        
        Ok(results)
    }
    
    /// Get the complete dependency graph
    pub fn dependency_graph(&self) -> &DependencyGraph {
        &self.dependency_graph
    }
    
    /// Get usage statistics
    pub fn usage_stats(&self) -> &UsageStats {
        self.usage_tracker.stats()
    }
    
    /// Get list of unused functions that can be eliminated
    pub fn unused_functions(&self) -> Vec<String> {
        self.graph_analyzer.find_unused_functions(&self.dependency_graph)
    }
    
    /// Get optimization recommendations
    pub fn optimization_recommendations(&self) -> Vec<OptimizationRecommendation> {
        self.graph_analyzer.generate_recommendations(&self.dependency_graph, &self.usage_tracker)
    }
    
    /// Perform dead code elimination
    pub fn eliminate_dead_code(
        &mut self, 
        module_registry: &mut ModuleRegistry
    ) -> Result<EliminationResults, TreeShakeError> {
        self.eliminator.eliminate_dead_code(
            &self.dependency_graph,
            &self.usage_tracker,
            module_registry,
        )
    }
    
    /// Preview dead code elimination without actually removing functions
    pub fn preview_elimination(&mut self) -> Result<EliminationResults, TreeShakeError> {
        let mut config = EliminatorConfig::default();
        config.preview_mode = true;
        self.eliminator = Eliminator::with_config(config);
        
        // Create a dummy module registry for preview
        let function_registry = std::sync::Arc::new(std::sync::RwLock::new(crate::linker::FunctionRegistry::new()));
        let mut dummy_registry = crate::modules::registry::ModuleRegistry::new(function_registry);
        self.eliminator.eliminate_dead_code(
            &self.dependency_graph,
            &self.usage_tracker,
            &mut dummy_registry,
        )
    }
    
    /// Analyze elimination potential without performing elimination
    pub fn analyze_elimination_potential(&self) -> Result<EliminationAnalysis, TreeShakeError> {
        self.eliminator.analyze_elimination_potential(&self.dependency_graph, &self.usage_tracker)
    }
    
    /// Configure elimination settings
    pub fn configure_elimination(&mut self, config: EliminatorConfig) {
        self.eliminator = Eliminator::with_config(config);
    }
    
    /// Get import optimization results
    pub fn get_import_optimizations(&self) -> Vec<&ModuleImportOptimization> {
        self.import_analyzer.get_all_optimizations()
    }
    
    /// Get optimized import statements for all modules
    pub fn get_optimized_imports(&self) -> Result<HashMap<String, String>, TreeShakeError> {
        self.import_analyzer.generate_optimized_imports()
    }
    
    /// Get import pattern for a specific module
    pub fn get_import_pattern(&self, module_name: &str) -> Option<&ImportPattern> {
        self.import_analyzer.get_import_pattern(module_name)
    }
    
    /// Get total import optimization impact
    pub fn get_import_optimization_impact(&self) -> ImportOptimizationImpact {
        self.import_analyzer.calculate_total_impact()
    }
    
    /// Get import dependency graph
    pub fn get_import_dependency_graph(&self) -> &ImportDependencyGraph {
        self.import_analyzer.get_import_graph()
    }
    
    /// Configure import analysis settings
    pub fn configure_import_analysis(&mut self, config: ImportAnalyzerConfig) {
        self.import_analyzer = ImportAnalyzer::with_config(config);
    }
    
    /// Resolve selective imports and get results
    pub fn resolve_selective_imports(
        &mut self,
        module_registry: &ModuleRegistry,
    ) -> Result<ImportResolutionResults, TreeShakeError> {
        self.selective_resolver.resolve_imports(
            &self.import_analyzer,
            &self.dependency_graph,
            &self.usage_tracker,
            module_registry,
        )
    }
    
    /// Get selective resolver performance metrics
    pub fn get_resolver_metrics(&self) -> &selective_resolver::ResolverMetrics {
        self.selective_resolver.get_performance_metrics()
    }
    
    /// Get import resolution cache statistics
    pub fn get_resolver_cache_stats(&self) -> &selective_resolver::CacheStats {
        self.selective_resolver.get_cache_stats()
    }
    
    /// Configure selective import resolver
    pub fn configure_selective_resolver(&mut self, config: SelectiveResolverConfig) {
        self.selective_resolver.configure(config);
    }
    
    /// Clear selective import resolution cache
    pub fn clear_resolver_cache(&mut self) {
        self.selective_resolver.clear_cache();
    }
    
    /// Perform compile-time dependency resolution
    pub fn resolve_compile_time_dependencies(
        &mut self,
        module_registry: &ModuleRegistry,
    ) -> Result<CompileTimeResolutionResults, TreeShakeError> {
        // First get selective import resolution results
        let resolution_results = self.selective_resolver.resolve_imports(
            &self.import_analyzer,
            &self.dependency_graph,
            &self.usage_tracker,
            module_registry,
        )?;
        
        // Then perform compile-time resolution
        self.compile_time_resolver.resolve_compile_time_dependencies(
            &resolution_results,
            &self.dependency_graph,
            &self.usage_tracker,
            module_registry,
        )
    }
    
    /// Get compile-time resolver performance metrics
    pub fn get_compile_time_metrics(&self) -> &compile_time_resolver::ResolverPerformanceMetrics {
        self.compile_time_resolver.get_performance_metrics()
    }
    
    /// Get compile-time resolver cache statistics
    pub fn get_compile_time_cache_stats(&self) -> &compile_time_resolver::CacheStatistics {
        self.compile_time_resolver.get_cache_statistics()
    }
    
    /// Configure compile-time resolver
    pub fn configure_compile_time_resolver(&mut self, config: CompileTimeResolverConfig) {
        self.compile_time_resolver.configure(config);
    }
    
    /// Clear compile-time resolver caches
    pub fn clear_compile_time_caches(&mut self) {
        self.compile_time_resolver.clear_caches();
    }
    
    /// Generate import statements from resolved imports
    pub fn generate_import_statements(
        &mut self,
        module_registry: &ModuleRegistry,
        target_format: Option<OutputFormat>,
    ) -> Result<ImportGenerationResults, TreeShakeError> {
        // First get selective import resolution results
        let resolution_results = self.selective_resolver.resolve_imports(
            &self.import_analyzer,
            &self.dependency_graph,
            &self.usage_tracker,
            module_registry,
        )?;
        
        // Then get compile-time resolution results
        let compile_time_results = self.compile_time_resolver.resolve_compile_time_dependencies(
            &resolution_results,
            &self.dependency_graph,
            &self.usage_tracker,
            module_registry,
        )?;
        
        // Finally generate import statements
        self.import_statement_generator.generate_import_statements(
            &resolution_results,
            &compile_time_results,
            target_format,
        )
    }
    
    /// Get import statement generator performance metrics
    pub fn get_import_generation_metrics(&self) -> &GenerationMetrics {
        self.import_statement_generator.get_performance_metrics()
    }
    
    /// Configure import statement generator
    pub fn configure_import_generator(&mut self, config: StatementGeneratorConfig) {
        self.import_statement_generator.configure(config);
    }
    
    /// Add custom format generator to import statement generator
    pub fn add_custom_format_generator(&mut self, format: OutputFormat, generator: Box<dyn FormatGenerator>) {
        self.import_statement_generator.add_format_generator(format, generator);
    }
    
    /// Clear import statement generator caches
    pub fn clear_import_generator_caches(&mut self) {
        self.import_statement_generator.clear_caches();
    }
    
    /// Validate dependencies using the dependency validator
    pub fn validate_dependencies(
        &mut self,
        module_registry: &ModuleRegistry,
    ) -> Result<DependencyValidationResults, TreeShakeError> {
        self.dependency_validator.validate_dependencies(
            &self.dependency_graph,
            &self.usage_tracker,
            module_registry,
        ).map_err(|e| TreeShakeError::ValidationError { message: e.to_string() })
    }
    
    /// Get dependency validator performance metrics
    pub fn get_validation_metrics(&self) -> &ValidationMetrics {
        self.dependency_validator.get_performance_metrics()
    }
    
    /// Configure dependency validator
    pub fn configure_dependency_validator(&mut self, config: DependencyValidatorConfig) {
        self.dependency_validator.configure(config);
    }
    
    /// Get import cache statistics
    pub fn get_import_cache_stats(&self) -> CacheStatsSummary {
        self.import_cache.get_cache_stats_summary()
    }
    
    /// Get cached import data
    pub fn get_cached_import_data(&mut self, key: &str) -> Result<Option<CachedImportData>, TreeShakeError> {
        self.import_cache.get(key)
    }
    
    /// Cache import data
    pub fn cache_import_data(&mut self, key: String, data: CachedImportData) -> Result<(), TreeShakeError> {
        self.import_cache.put(key, data, None)
    }
    
    /// Warm import cache with frequently accessed keys
    pub fn warm_import_cache(&mut self, keys: Vec<String>) -> Result<(), TreeShakeError> {
        self.import_cache.warm_cache(keys)
    }
    
    /// Optimize import cache performance
    pub fn optimize_import_cache(&mut self) -> Result<CacheOptimizationReport, TreeShakeError> {
        self.import_cache.optimize()
    }
    
    /// Clear import cache
    pub fn clear_import_cache(&mut self) -> Result<(), TreeShakeError> {
        self.import_cache.clear()
    }
    
    /// Configure import cache
    pub fn configure_import_cache(&mut self, config: ImportCacheConfig) -> Result<(), TreeShakeError> {
        self.import_cache = ImportCache::with_config(config);
        Ok(())
    }
    
    // Enhanced integration methods for validation and caching
    
    /// Generate cache key for analysis results
    fn generate_analysis_cache_key(&self, module_registry: &ModuleRegistry) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Include module registry state in cache key
        let module_list: Vec<_> = module_registry.list_modules();
        module_list.hash(&mut hasher);
        
        // Include configuration state
        format!("analysis_results_{:x}", hasher.finish())
    }
    
    /// Get cached analysis results
    fn get_cached_analysis_results(&mut self, cache_key: &str) -> Result<Option<TreeShakingResults>, TreeShakeError> {
        match self.import_cache.get(cache_key)? {
            Some(CachedImportData::CustomData { data_type, serialized_data }) 
                if data_type == "tree_shaking_results" => {
                let results: TreeShakingResults = bincode::deserialize(&serialized_data)
                    .map_err(|e| TreeShakeError::CacheError { 
                        message: format!("Failed to deserialize cached results: {}", e) 
                    })?;
                Ok(Some(results))
            },
            _ => Ok(None),
        }
    }
    
    /// Cache analysis results
    fn cache_analysis_results(&mut self, cache_key: &str, results: &TreeShakingResults) -> Result<(), TreeShakeError> {
        let serialized_data = bincode::serialize(results)
            .map_err(|e| TreeShakeError::CacheError { 
                message: format!("Failed to serialize results for caching: {}", e) 
            })?;
        
        let cache_data = CachedImportData::CustomData {
            data_type: "tree_shaking_results".to_string(),
            serialized_data,
        };
        
        self.import_cache.put(cache_key.to_string(), cache_data, None)
    }
    
    /// Cache validation results
    fn cache_validation_results(&mut self, results: &DependencyValidationResults) -> Result<(), TreeShakeError> {
        let cache_key = format!("validation_results_{}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap().as_secs());
        
        let cache_data = CachedImportData::DependencyReport(results.clone());
        self.import_cache.put(cache_key, cache_data, Some(std::time::Duration::from_secs(1800)))
    }
    
    /// Identify frequently accessed import patterns for cache warming
    fn identify_frequent_import_patterns(&self) -> Vec<String> {
        self.import_analyzer.get_all_optimizations()
            .iter()
            .filter(|opt| {
                // Include patterns that are accessed frequently
                opt.original_pattern.used_functions.iter()
                    .any(|func| func.usage_count > 5)
            })
            .map(|opt| format!("import_pattern_{}", opt.module_name))
            .collect()
    }
    
    /// Resolve imports with cache-first strategy
    fn resolve_imports_with_caching(&mut self, module_registry: &ModuleRegistry) -> Result<ImportResolutionResults, TreeShakeError> {
        let cache_key = "import_resolution_current";
        
        // Check cache first
        if let Some(cached_results) = self.get_cached_import_resolution_results(cache_key)? {
            return Ok(cached_results);
        }
        
        // Perform resolution
        let results = self.selective_resolver.resolve_imports(
            &self.import_analyzer, 
            &self.dependency_graph, 
            &self.usage_tracker, 
            module_registry
        )?;
        
        // Cache the results
        self.cache_import_resolution_results(cache_key, &results)?;
        
        Ok(results)
    }
    
    /// Get cached import resolution results
    fn get_cached_import_resolution_results(&mut self, cache_key: &str) -> Result<Option<ImportResolutionResults>, TreeShakeError> {
        match self.import_cache.get(cache_key)? {
            Some(CachedImportData::ResolvedImports(results)) => Ok(Some(results)),
            _ => Ok(None),
        }
    }
    
    /// Cache import resolution results
    fn cache_import_resolution_results(&mut self, cache_key: &str, results: &ImportResolutionResults) -> Result<(), TreeShakeError> {
        let cache_data = CachedImportData::ResolvedImports(results.clone());
        self.import_cache.put(cache_key.to_string(), cache_data, Some(std::time::Duration::from_secs(900)))
    }
    
    /// Resolve compile-time dependencies with caching
    fn resolve_compile_time_dependencies_with_caching(
        &mut self,
        resolution_results: &ImportResolutionResults,
        module_registry: &ModuleRegistry,
    ) -> Result<CompileTimeResolutionResults, TreeShakeError> {
        self.compile_time_resolver.resolve_compile_time_dependencies(
            resolution_results,
            &self.dependency_graph,
            &self.usage_tracker,
            module_registry,
        )
    }
    
    /// Generate validated import statements
    fn generate_validated_import_statements(
        &mut self,
        resolution_results: &ImportResolutionResults,
        compile_time_results: &CompileTimeResolutionResults,
        validation_results: &DependencyValidationResults,
    ) -> Result<ImportGenerationResults, TreeShakeError> {
        // Only generate imports for modules that passed validation
        if validation_results.has_critical_errors() {
            return Err(TreeShakeError::ValidationError {
                message: "Cannot generate imports due to critical validation errors".to_string(),
            });
        }
        
        self.import_statement_generator.generate_import_statements(
            resolution_results,
            compile_time_results,
            None, // Use default format
        )
    }
    
    /// Create dependency graph summary
    fn create_dependency_graph_summary(&self) -> DependencyGraphSummary {
        DependencyGraphSummary {
            total_nodes: self.dependency_graph.node_count(),
            total_edges: self.dependency_graph.edge_count(),
            circular_dependencies: self.dependency_graph.find_circular_dependencies().len(),
            unused_functions: self.unused_functions().len(),
            critical_path_length: self.graph_analyzer.find_critical_path_length(&self.dependency_graph),
        }
    }
    
    /// Calculate overall cache hit ratio across all systems
    fn calculate_overall_cache_hit_ratio(&self) -> f64 {
        let cache_stats = self.import_cache.get_cache_stats_summary();
        cache_stats.overall_hit_ratio
    }
    
    /// Calculate optimization effectiveness
    fn calculate_optimization_effectiveness(&self) -> f64 {
        let import_impact = self.import_analyzer.calculate_total_impact();
        import_impact.import_reduction_percentage / 100.0
    }
    
    /// Generate optimization recommendations based on integrated analysis
    fn generate_optimization_recommendations(&self) -> Vec<IntegratedOptimizationRecommendation> {
        let mut recommendations = Vec::new();
        
        // Add cache-based recommendations
        let cache_stats = self.import_cache.get_cache_stats_summary();
        if cache_stats.overall_hit_ratio < 0.8 {
            recommendations.push(IntegratedOptimizationRecommendation {
                recommendation_type: IntegratedOptimizationType::ImproveCache,
                description: "Cache hit ratio is below optimal threshold. Consider warming more frequently accessed patterns.".to_string(),
                expected_impact: OptimizationImpact {
                    size_reduction: 0,
                    compile_time_improvement: (cache_stats.average_response_time.as_millis() as u64 * 10),
                    runtime_impact: RuntimeImpact::Positive(0.1),
                    functions_affected: 0,
                },
                confidence: 0.8,
            });
        }
        
        // Add validation-based recommendations
        let validation_metrics = self.dependency_validator.get_performance_metrics();
        if validation_metrics.circular_dependencies_found > 0 {
            recommendations.push(IntegratedOptimizationRecommendation {
                recommendation_type: IntegratedOptimizationType::ResolveDependencies,
                description: format!("Found {} circular dependencies that should be resolved for better optimization.", 
                    validation_metrics.circular_dependencies_found),
                expected_impact: OptimizationImpact {
                    size_reduction: validation_metrics.circular_dependencies_found * 500,
                    compile_time_improvement: validation_metrics.circular_dependencies_found * 100,
                    runtime_impact: RuntimeImpact::Positive(0.05),
                    functions_affected: validation_metrics.circular_dependencies_found,
                },
                confidence: 0.9,
            });
        }
        
        recommendations
    }
}

/// Comprehensive tree-shaking analysis results
#[derive(Debug, Clone)]
pub struct TreeShakingResults {
    /// Summary of the dependency graph
    pub dependency_graph_summary: DependencyGraphSummary,
    
    /// Usage statistics
    pub usage_statistics: UsageStats,
    
    /// Dependency validation results
    pub validation_results: DependencyValidationResults,
    
    /// Import resolution results
    pub resolution_results: ImportResolutionResults,
    
    /// Compile-time resolution results
    pub compile_time_results: CompileTimeResolutionResults,
    
    /// Generated import statements
    pub import_statements: ImportGenerationResults,
    
    /// Performance metrics
    pub performance_metrics: TreeShakingPerformanceMetrics,
    
    /// Optimization recommendations
    pub recommendations: Vec<IntegratedOptimizationRecommendation>,
}

/// Summary of dependency graph characteristics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DependencyGraphSummary {
    /// Total number of nodes in the graph
    pub total_nodes: usize,
    
    /// Total number of edges in the graph
    pub total_edges: usize,
    
    /// Number of circular dependencies detected
    pub circular_dependencies: usize,
    
    /// Number of unused functions
    pub unused_functions: usize,
    
    /// Length of the critical path
    pub critical_path_length: usize,
}

/// Performance metrics for the entire tree-shaking process
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TreeShakingPerformanceMetrics {
    /// Total time for analysis
    pub total_analysis_time: std::time::Duration,
    
    /// Overall cache hit ratio
    pub cache_hit_ratio: f64,
    
    /// Time spent on validation
    pub validation_time: std::time::Duration,
    
    /// Time spent on import resolution
    pub import_resolution_time: std::time::Duration,
    
    /// Effectiveness of optimizations (0.0 to 1.0)
    pub optimization_effectiveness: f64,
}

/// Integrated optimization recommendation combining multiple analysis results
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IntegratedOptimizationRecommendation {
    /// Type of integrated optimization
    pub recommendation_type: IntegratedOptimizationType,
    
    /// Description of the recommendation
    pub description: String,
    
    /// Expected impact of applying this recommendation
    pub expected_impact: OptimizationImpact,
    
    /// Confidence level in this recommendation (0.0 to 1.0)
    pub confidence: f64,
}

/// Types of integrated optimizations that combine validation and caching insights
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum IntegratedOptimizationType {
    /// Improve cache performance
    ImproveCache,
    
    /// Resolve circular dependencies
    ResolveDependencies,
    
    /// Optimize import patterns based on validation
    OptimizeImports,
    
    /// Eliminate validated dead code
    EliminateDeadCode,
    
    /// Improve validation performance
    ImproveValidation,
    
    /// Bundle related imports more effectively
    OptimizeImportBundling,
    
    /// Reduce memory usage
    ReduceMemoryUsage,
    
    /// Improve compilation performance
    ImproveCompilationPerformance,
}

impl TreeShakingResults {
    /// Get overall success status
    pub fn is_successful(&self) -> bool {
        !self.validation_results.has_critical_errors() && 
        self.resolution_results.resolved_imports.len() > 0
    }
    
    /// Get total time saved from optimizations
    pub fn total_time_saved(&self) -> std::time::Duration {
        self.performance_metrics.total_analysis_time
    }
    
    /// Get total functions optimized
    pub fn functions_optimized(&self) -> usize {
        self.resolution_results.resolved_imports.len()
    }
    
    /// Get optimization summary
    pub fn get_optimization_summary(&self) -> OptimizationSummary {
        OptimizationSummary {
            total_modules_analyzed: self.resolution_results.resolved_imports.len(),
            circular_dependencies_found: self.dependency_graph_summary.circular_dependencies,
            unused_functions_eliminated: self.dependency_graph_summary.unused_functions,
            cache_hit_ratio: self.performance_metrics.cache_hit_ratio,
            optimization_effectiveness: self.performance_metrics.optimization_effectiveness,
            total_analysis_time: self.performance_metrics.total_analysis_time,
            recommendations_generated: self.recommendations.len(),
        }
    }
}

/// Summary of optimization results
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OptimizationSummary {
    /// Total modules analyzed
    pub total_modules_analyzed: usize,
    
    /// Circular dependencies found and resolved
    pub circular_dependencies_found: usize,
    
    /// Unused functions eliminated
    pub unused_functions_eliminated: usize,
    
    /// Cache hit ratio achieved
    pub cache_hit_ratio: f64,
    
    /// Overall optimization effectiveness
    pub optimization_effectiveness: f64,
    
    /// Total time spent on analysis
    pub total_analysis_time: std::time::Duration,
    
    /// Number of optimization recommendations generated
    pub recommendations_generated: usize,
}

/// Tree-shaking specific errors
#[derive(Debug, Clone)]
pub enum TreeShakeError {
    /// Dependency analysis failed
    DependencyAnalysisError { message: String },
    
    /// Circular dependency detected
    CircularDependency { functions: Vec<String> },
    
    /// Graph analysis failed
    GraphAnalysisError { message: String },
    
    /// Module resolution failed
    ModuleResolutionError { module: String, message: String },
    
    /// Cache operation failed
    CacheError { message: String },
    
    /// Validation failed
    ValidationError { message: String },
}

impl std::fmt::Display for TreeShakeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TreeShakeError::DependencyAnalysisError { message } => {
                write!(f, "Dependency analysis error: {}", message)
            }
            TreeShakeError::CircularDependency { functions } => {
                write!(f, "Circular dependency detected between functions: {:?}", functions)
            }
            TreeShakeError::GraphAnalysisError { message } => {
                write!(f, "Graph analysis error: {}", message)
            }
            TreeShakeError::ModuleResolutionError { module, message } => {
                write!(f, "Module resolution error in {}: {}", module, message)
            }
            TreeShakeError::CacheError { message } => {
                write!(f, "Cache error: {}", message)
            }
            TreeShakeError::ValidationError { message } => {
                write!(f, "Validation error: {}", message)
            }
        }
    }
}

impl std::error::Error for TreeShakeError {}

impl From<String> for TreeShakeError {
    fn from(message: String) -> Self {
        TreeShakeError::DependencyAnalysisError { message }
    }
}

/// Optimization recommendation from tree-shaking analysis
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Type of optimization
    pub recommendation_type: OptimizationType,
    
    /// Functions affected
    pub functions: Vec<String>,
    
    /// Expected impact
    pub impact: OptimizationImpact,
    
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    
    /// Description of the recommendation
    pub description: String,
}

/// Types of optimizations that can be performed
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationType {
    /// Remove completely unused functions
    DeadCodeElimination,
    
    /// Inline small frequently-called functions
    FunctionInlining,
    
    /// Combine related functions into single modules
    ModuleConsolidation,
    
    /// Split large modules with independent functions
    ModuleSplitting,
    
    /// Optimize import chains
    ImportOptimization,
    
    /// Lazy loading for rarely used functions
    LazyLoading,
}

/// Expected impact of an optimization
#[derive(Debug, Clone)]
pub struct OptimizationImpact {
    /// Estimated binary size reduction (bytes)
    pub size_reduction: usize,
    
    /// Estimated compilation time improvement (milliseconds)
    pub compile_time_improvement: u64,
    
    /// Estimated runtime performance impact
    pub runtime_impact: RuntimeImpact,
    
    /// Number of functions affected
    pub functions_affected: usize,
}

/// Runtime performance impact
#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeImpact {
    /// Improves performance
    Positive(f64),
    
    /// No significant impact
    Neutral,
    
    /// Minor performance cost
    NegligibleCost(f64),
    
    /// Performance trade-off for size benefits
    Negative(f64),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_shaker_creation() {
        let tree_shaker = TreeShaker::new();
        
        // Verify initial state
        assert_eq!(tree_shaker.dependency_graph().node_count(), 0);
        assert_eq!(tree_shaker.unused_functions().len(), 0);
    }

    #[test]
    fn test_optimization_recommendation() {
        let recommendation = OptimizationRecommendation {
            recommendation_type: OptimizationType::DeadCodeElimination,
            functions: vec!["unused_function".to_string()],
            impact: OptimizationImpact {
                size_reduction: 1024,
                compile_time_improvement: 50,
                runtime_impact: RuntimeImpact::Neutral,
                functions_affected: 1,
            },
            confidence: 0.95,
            description: "Remove unused function to reduce binary size".to_string(),
        };
        
        assert_eq!(recommendation.recommendation_type, OptimizationType::DeadCodeElimination);
        assert_eq!(recommendation.confidence, 0.95);
        assert_eq!(recommendation.functions.len(), 1);
    }

    #[test]
    fn test_runtime_impact_types() {
        assert_eq!(RuntimeImpact::Neutral, RuntimeImpact::Neutral);
        assert!(matches!(RuntimeImpact::Positive(0.1), RuntimeImpact::Positive(_)));
        assert!(matches!(RuntimeImpact::Negative(0.05), RuntimeImpact::Negative(_)));
    }
}