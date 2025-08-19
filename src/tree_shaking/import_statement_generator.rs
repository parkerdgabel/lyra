//! Import Statement Generator
//!
//! Transforms resolved import data into optimized import statements across multiple
//! target formats, enabling efficient compile-time code generation.

use super::{DependencyGraph, UsageTracker, TreeShakeError};
use super::selective_resolver::{ResolvedImport, ResolvedImportType, ImportResolutionResults};
use super::compile_time_resolver::{CompileTimeResolutionResults, ResolvedDependency};
use crate::modules::registry::ModuleRegistry;
use std::collections::{HashMap, HashSet, BTreeSet};
use serde::{Serialize, Deserialize};
use std::time::{SystemTime, Duration};

/// Import statement generator that creates optimized import code
pub struct ImportStatementGenerator {
    /// Configuration for statement generation
    config: StatementGeneratorConfig,
    
    /// Template engine for code generation
    template_engine: TemplateEngine,
    
    /// Import optimization pipeline
    optimization_pipeline: OptimizationPipeline,
    
    /// Format-specific generators
    format_generators: HashMap<OutputFormat, Box<dyn FormatGenerator>>,
    
    /// Performance metrics
    performance_metrics: GenerationMetrics,
    
    /// Generation context
    generation_context: GenerationContext,
}

/// Configuration for import statement generation
#[derive(Debug, Clone)]
pub struct StatementGeneratorConfig {
    /// Default output format
    pub default_format: OutputFormat,
    
    /// Enable import consolidation
    pub enable_consolidation: bool,
    
    /// Enable alias optimization
    pub enable_alias_optimization: bool,
    
    /// Enable dependency ordering
    pub enable_dependency_ordering: bool,
    
    /// Enable lazy loading hints
    pub enable_lazy_loading: bool,
    
    /// Enable tree-shaking annotations
    pub enable_tree_shaking_annotations: bool,
    
    /// Maximum imports per statement
    pub max_imports_per_statement: usize,
    
    /// Enable performance hints
    pub enable_performance_hints: bool,
    
    /// Enable syntax validation
    pub enable_syntax_validation: bool,
    
    /// Line length limit for generated code
    pub line_length_limit: usize,
    
    /// Indentation style
    pub indentation: IndentationStyle,
    
    /// Import style preferences
    pub import_style: ImportStylePreferences,
}

impl Default for StatementGeneratorConfig {
    fn default() -> Self {
        StatementGeneratorConfig {
            default_format: OutputFormat::ES6Module,
            enable_consolidation: true,
            enable_alias_optimization: true,
            enable_dependency_ordering: true,
            enable_lazy_loading: true,
            enable_tree_shaking_annotations: true,
            max_imports_per_statement: 10,
            enable_performance_hints: true,
            enable_syntax_validation: true,
            line_length_limit: 100,
            indentation: IndentationStyle::Spaces(2),
            import_style: ImportStylePreferences::default(),
        }
    }
}

/// Output formats supported by the generator
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OutputFormat {
    /// ES6 modules (import/export)
    ES6Module,
    
    /// CommonJS (require/module.exports)
    CommonJS,
    
    /// Rust use statements
    Rust,
    
    /// TypeScript imports with type annotations
    TypeScript,
    
    /// AMD modules
    AMD,
    
    /// UMD modules
    UMD,
    
    /// Custom format
    Custom { name: String },
}

/// Indentation styles
#[derive(Debug, Clone, PartialEq)]
pub enum IndentationStyle {
    /// Spaces with specified count
    Spaces(usize),
    
    /// Tabs
    Tabs,
    
    /// Mixed (tabs for indentation, spaces for alignment)
    Mixed,
}

/// Import style preferences
#[derive(Debug, Clone)]
pub struct ImportStylePreferences {
    /// Prefer single quotes over double quotes
    pub prefer_single_quotes: bool,
    
    /// Add trailing commas in multi-line imports
    pub trailing_commas: bool,
    
    /// Sort imports alphabetically
    pub sort_imports: bool,
    
    /// Group related imports together
    pub group_imports: bool,
    
    /// Add spacing around braces
    pub space_around_braces: bool,
    
    /// Prefer destructuring imports
    pub prefer_destructuring: bool,
}

impl Default for ImportStylePreferences {
    fn default() -> Self {
        ImportStylePreferences {
            prefer_single_quotes: true,
            trailing_commas: true,
            sort_imports: true,
            group_imports: true,
            space_around_braces: true,
            prefer_destructuring: true,
        }
    }
}

/// Template engine for generating import statements
#[derive(Debug, Clone)]
pub struct TemplateEngine {
    /// Available templates
    templates: HashMap<OutputFormat, ImportTemplateSet>,
    
    /// Template renderer
    renderer: TemplateRenderer,
    
    /// Template cache
    template_cache: TemplateCache,
}

/// Set of templates for a specific output format
#[derive(Debug, Clone)]
pub struct ImportTemplateSet {
    /// Template for selective imports
    pub selective_import: ImportTemplate,
    
    /// Template for full module imports
    pub full_import: ImportTemplate,
    
    /// Template for namespace imports
    pub namespace_import: ImportTemplate,
    
    /// Template for conditional imports
    pub conditional_import: ImportTemplate,
    
    /// Template for lazy imports
    pub lazy_import: ImportTemplate,
    
    /// Template for aliased imports
    pub aliased_import: ImportTemplate,
}

/// Import statement template
#[derive(Debug, Clone)]
pub struct ImportTemplate {
    /// Template pattern
    pub pattern: String,
    
    /// Variable substitutions
    pub variables: Vec<TemplateVariable>,
    
    /// Optional pre-processing
    pub preprocessor: Option<TemplatePreprocessor>,
    
    /// Optional post-processing
    pub postprocessor: Option<TemplatePostprocessor>,
    
    /// Template metadata
    pub metadata: TemplateMetadata,
}

/// Template variable definition
#[derive(Debug, Clone)]
pub struct TemplateVariable {
    /// Variable name
    pub name: String,
    
    /// Variable type
    pub var_type: VariableType,
    
    /// Default value
    pub default_value: Option<String>,
    
    /// Whether variable is required
    pub required: bool,
    
    /// Value transformation function
    pub transformer: Option<ValueTransformer>,
}

/// Types of template variables
#[derive(Debug, Clone, PartialEq)]
pub enum VariableType {
    /// String value
    String,
    
    /// List of strings
    StringList,
    
    /// Boolean value
    Boolean,
    
    /// Integer value
    Integer,
    
    /// Module path
    ModulePath,
    
    /// Function name
    FunctionName,
    
    /// Import alias
    ImportAlias,
}

/// Template metadata
#[derive(Debug, Clone, Default)]
pub struct TemplateMetadata {
    /// Template name
    pub name: String,
    
    /// Template description
    pub description: String,
    
    /// Template version
    pub version: String,
    
    /// Supported features
    pub features: Vec<String>,
    
    /// Performance characteristics
    pub performance: TemplatePerformance,
}

/// Template performance characteristics
#[derive(Debug, Clone, Default)]
pub struct TemplatePerformance {
    /// Expected rendering time
    pub render_time_micros: u64,
    
    /// Memory usage estimate
    pub memory_usage_bytes: usize,
    
    /// Output size estimate
    pub output_size_chars: usize,
}

/// Template renderer
#[derive(Debug, Clone, Default)]
pub struct TemplateRenderer {
    /// Rendering context
    context: RenderingContext,
    
    /// Rendering options
    options: RenderingOptions,
    
    /// Performance metrics
    metrics: RenderingMetrics,
}

/// Rendering context
#[derive(Debug, Clone, Default)]
pub struct RenderingContext {
    /// Current variables
    variables: HashMap<String, TemplateValue>,
    
    /// Rendering stack
    stack: Vec<String>,
    
    /// Output buffer
    output_buffer: String,
    
    /// Error collection
    errors: Vec<RenderingError>,
}

/// Template value types
#[derive(Debug, Clone)]
pub enum TemplateValue {
    /// String value
    String(String),
    
    /// List value
    List(Vec<String>),
    
    /// Boolean value
    Boolean(bool),
    
    /// Integer value
    Integer(i64),
    
    /// Object value
    Object(HashMap<String, TemplateValue>),
}

/// Rendering options
#[derive(Debug, Clone)]
pub struct RenderingOptions {
    /// Enable syntax highlighting
    pub enable_highlighting: bool,
    
    /// Enable minification
    pub enable_minification: bool,
    
    /// Enable pretty printing
    pub enable_pretty_printing: bool,
    
    /// Maximum line length
    pub max_line_length: usize,
    
    /// Indentation level
    pub indentation_level: usize,
}

impl Default for RenderingOptions {
    fn default() -> Self {
        RenderingOptions {
            enable_highlighting: false,
            enable_minification: false,
            enable_pretty_printing: true,
            max_line_length: 100,
            indentation_level: 0,
        }
    }
}

/// Rendering metrics
#[derive(Debug, Clone, Default)]
pub struct RenderingMetrics {
    /// Templates rendered
    pub templates_rendered: u64,
    
    /// Total rendering time
    pub total_render_time: Duration,
    
    /// Average rendering time
    pub average_render_time: Duration,
    
    /// Characters generated
    pub characters_generated: u64,
    
    /// Lines generated
    pub lines_generated: u64,
}

/// Rendering errors
#[derive(Debug, Clone)]
pub struct RenderingError {
    /// Error type
    pub error_type: RenderingErrorType,
    
    /// Error message
    pub message: String,
    
    /// Template name
    pub template_name: String,
    
    /// Line number
    pub line_number: Option<usize>,
    
    /// Column number
    pub column_number: Option<usize>,
}

/// Types of rendering errors
#[derive(Debug, Clone, PartialEq)]
pub enum RenderingErrorType {
    /// Variable not found
    VariableNotFound,
    
    /// Invalid variable type
    InvalidVariableType,
    
    /// Template syntax error
    SyntaxError,
    
    /// Template not found
    TemplateNotFound,
    
    /// Rendering timeout
    Timeout,
    
    /// Memory limit exceeded
    MemoryLimitExceeded,
}

/// Template cache for performance
#[derive(Debug, Clone, Default)]
pub struct TemplateCache {
    /// Cached rendered templates
    rendered_cache: HashMap<String, CachedTemplate>,
    
    /// Cache statistics
    cache_stats: CacheStats,
    
    /// Cache configuration
    cache_config: CacheConfig,
}

/// Cached template
#[derive(Debug, Clone)]
pub struct CachedTemplate {
    /// Cached content
    pub content: String,
    
    /// Cache timestamp
    pub cached_at: SystemTime,
    
    /// Cache expiration
    pub expires_at: SystemTime,
    
    /// Cache hits
    pub hits: u64,
    
    /// Template hash
    pub template_hash: u64,
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total hits
    pub hits: u64,
    
    /// Total misses
    pub misses: u64,
    
    /// Hit ratio
    pub hit_ratio: f64,
    
    /// Memory usage
    pub memory_usage: usize,
    
    /// Cache size
    pub cache_size: usize,
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum cache size
    pub max_size: usize,
    
    /// Cache TTL
    pub ttl_seconds: u64,
    
    /// Enable LRU eviction
    pub enable_lru: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        CacheConfig {
            max_size: 1000,
            ttl_seconds: 3600, // 1 hour
            enable_lru: true,
        }
    }
}

/// Import optimization pipeline
#[derive(Debug, Clone)]
pub struct OptimizationPipeline {
    /// Statement consolidator
    consolidator: StatementConsolidator,
    
    /// Alias manager
    alias_manager: AliasManager,
    
    /// Dependency orderer
    dependency_orderer: DependencyOrderer,
    
    /// Import bundler
    bundler: ImportBundler,
    
    /// Pipeline configuration
    config: PipelineConfig,
}

/// Statement consolidator
#[derive(Debug, Clone, Default)]
pub struct StatementConsolidator {
    /// Consolidation rules
    rules: Vec<ConsolidationRule>,
    
    /// Consolidation statistics
    stats: ConsolidationStats,
}

/// Consolidation rule
#[derive(Debug, Clone)]
pub struct ConsolidationRule {
    /// Rule name
    pub name: String,
    
    /// Rule pattern
    pub pattern: ConsolidationPattern,
    
    /// Rule action
    pub action: ConsolidationAction,
    
    /// Rule priority
    pub priority: u32,
    
    /// Rule enabled
    pub enabled: bool,
}

/// Consolidation patterns
#[derive(Debug, Clone)]
pub enum ConsolidationPattern {
    /// Same module imports
    SameModule,
    
    /// Related modules
    RelatedModules { pattern: String },
    
    /// Function prefix
    FunctionPrefix { prefix: String },
    
    /// Namespace pattern
    NamespacePattern { namespace: String },
    
    /// Custom pattern
    Custom { matcher: PatternMatcher },
}

/// Consolidation actions
#[derive(Debug, Clone)]
pub enum ConsolidationAction {
    /// Merge into single statement
    MergeSingle,
    
    /// Group by namespace
    GroupByNamespace,
    
    /// Create alias
    CreateAlias { alias: String },
    
    /// Bundle imports
    Bundle { bundle_name: String },
    
    /// Custom action
    Custom { action: ConsolidationActionHandler },
}

/// Consolidation statistics
#[derive(Debug, Clone, Default)]
pub struct ConsolidationStats {
    /// Total imports processed
    pub total_imports: usize,
    
    /// Imports consolidated
    pub consolidated_imports: usize,
    
    /// Consolidation ratio
    pub consolidation_ratio: f64,
    
    /// Time saved (estimated)
    pub time_saved_ms: u64,
    
    /// Size reduction (bytes)
    pub size_reduction_bytes: usize,
}

/// Alias manager for resolving naming conflicts
#[derive(Debug, Clone, Default)]
pub struct AliasManager {
    /// Alias mappings
    aliases: HashMap<String, String>,
    
    /// Conflict resolution strategies
    conflict_strategies: Vec<ConflictResolutionStrategy>,
    
    /// Reserved names
    reserved_names: HashSet<String>,
    
    /// Alias statistics
    stats: AliasStats,
}

/// Conflict resolution strategies
#[derive(Debug, Clone)]
pub enum ConflictResolutionStrategy {
    /// Add numeric suffix
    NumericSuffix,
    
    /// Add module prefix
    ModulePrefix,
    
    /// Use full path
    FullPath,
    
    /// Custom strategy
    Custom { resolver: ConflictResolver },
}

/// Alias statistics
#[derive(Debug, Clone, Default)]
pub struct AliasStats {
    /// Total aliases created
    pub aliases_created: usize,
    
    /// Conflicts resolved
    pub conflicts_resolved: usize,
    
    /// Average alias length
    pub average_alias_length: f64,
}

/// Dependency orderer for optimal import ordering
#[derive(Debug, Clone, Default)]
pub struct DependencyOrderer {
    /// Ordering strategies
    strategies: Vec<OrderingStrategy>,
    
    /// Dependency graph cache
    dependency_cache: HashMap<String, Vec<String>>,
    
    /// Ordering statistics
    stats: OrderingStats,
}

/// Import ordering strategies
#[derive(Debug, Clone)]
pub enum OrderingStrategy {
    /// Topological sorting
    Topological,
    
    /// Alphabetical sorting
    Alphabetical,
    
    /// Frequency-based ordering
    FrequencyBased,
    
    /// Size-based ordering
    SizeBased,
    
    /// Critical path first
    CriticalPathFirst,
    
    /// Custom strategy
    Custom { orderer: DependencyOrderingHandler },
}

/// Ordering statistics
#[derive(Debug, Clone, Default)]
pub struct OrderingStats {
    /// Dependencies ordered
    pub dependencies_ordered: usize,
    
    /// Ordering time
    pub ordering_time: Duration,
    
    /// Cycles detected
    pub cycles_detected: usize,
    
    /// Optimization effectiveness
    pub optimization_effectiveness: f64,
}

/// Import bundler for grouping related imports
#[derive(Debug, Clone, Default)]
pub struct ImportBundler {
    /// Bundling rules
    rules: Vec<BundlingRule>,
    
    /// Bundle cache
    bundle_cache: HashMap<String, ImportBundle>,
    
    /// Bundling statistics
    stats: BundlingStats,
}

/// Bundling rule
#[derive(Debug, Clone)]
pub struct BundlingRule {
    /// Rule name
    pub name: String,
    
    /// Bundle criteria
    pub criteria: BundlingCriteria,
    
    /// Bundle strategy
    pub strategy: BundlingStrategy,
    
    /// Rule priority
    pub priority: u32,
}

/// Bundling criteria
#[derive(Debug, Clone)]
pub enum BundlingCriteria {
    /// By module namespace
    ByNamespace { pattern: String },
    
    /// By usage frequency
    ByFrequency { threshold: u64 },
    
    /// By dependency relationship
    ByDependency,
    
    /// By size threshold
    BySize { threshold: usize },
    
    /// Custom criteria
    Custom { matcher: BundleMatcher },
}

/// Bundling strategies
#[derive(Debug, Clone)]
pub enum BundlingStrategy {
    /// Eager bundling
    Eager,
    
    /// Lazy bundling
    Lazy,
    
    /// Conditional bundling
    Conditional { condition: String },
    
    /// Dynamic bundling
    Dynamic,
}

/// Import bundle
#[derive(Debug, Clone)]
pub struct ImportBundle {
    /// Bundle name
    pub name: String,
    
    /// Bundled imports
    pub imports: Vec<String>,
    
    /// Bundle type
    pub bundle_type: BundleType,
    
    /// Bundle metadata
    pub metadata: BundleMetadata,
}

/// Bundle types
#[derive(Debug, Clone, PartialEq)]
pub enum BundleType {
    /// Static bundle
    Static,
    
    /// Dynamic bundle
    Dynamic,
    
    /// Lazy bundle
    Lazy,
    
    /// Conditional bundle
    Conditional,
}

/// Bundle metadata
#[derive(Debug, Clone, Default)]
pub struct BundleMetadata {
    /// Bundle size
    pub size_bytes: usize,
    
    /// Load priority
    pub priority: u32,
    
    /// Dependencies
    pub dependencies: Vec<String>,
    
    /// Performance impact
    pub performance_impact: f64,
}

/// Bundling statistics
#[derive(Debug, Clone, Default)]
pub struct BundlingStats {
    /// Bundles created
    pub bundles_created: usize,
    
    /// Imports bundled
    pub imports_bundled: usize,
    
    /// Bundle efficiency
    pub bundle_efficiency: f64,
    
    /// Size reduction
    pub size_reduction: f64,
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Enable consolidation
    pub enable_consolidation: bool,
    
    /// Enable alias management
    pub enable_alias_management: bool,
    
    /// Enable dependency ordering
    pub enable_dependency_ordering: bool,
    
    /// Enable bundling
    pub enable_bundling: bool,
    
    /// Pipeline timeout
    pub timeout_ms: u64,
    
    /// Maximum iterations
    pub max_iterations: u32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            enable_consolidation: true,
            enable_alias_management: true,
            enable_dependency_ordering: true,
            enable_bundling: true,
            timeout_ms: 5000,
            max_iterations: 10,
        }
    }
}

/// Format generator trait for different output formats
pub trait FormatGenerator: std::fmt::Debug {
    /// Generate import statement for the format
    fn generate_import(&self, import: &ResolvedImport, context: &GenerationContext) -> Result<String, GenerationError>;
    
    /// Generate multiple import statements
    fn generate_imports(&self, imports: &[ResolvedImport], context: &GenerationContext) -> Result<String, GenerationError>;
    
    /// Get format-specific metadata
    fn get_metadata(&self) -> FormatMetadata;
    
    /// Validate generated code
    fn validate(&self, code: &str) -> Result<ValidationResult, GenerationError>;
}

/// Format metadata
#[derive(Debug, Clone)]
pub struct FormatMetadata {
    /// Format name
    pub name: String,
    
    /// Format version
    pub version: String,
    
    /// Supported features
    pub features: Vec<FormatFeature>,
    
    /// File extensions
    pub file_extensions: Vec<String>,
}

/// Format features
#[derive(Debug, Clone, PartialEq)]
pub enum FormatFeature {
    /// Tree shaking support
    TreeShaking,
    
    /// Dynamic imports
    DynamicImports,
    
    /// Namespace imports
    NamespaceImports,
    
    /// Type annotations
    TypeAnnotations,
    
    /// Conditional imports
    ConditionalImports,
    
    /// Lazy loading
    LazyLoading,
}

/// Generation context
#[derive(Debug, Clone)]
pub struct GenerationContext {
    /// Current module
    current_module: Option<String>,
    
    /// Import cache
    import_cache: HashMap<String, String>,
    
    /// Generation options
    options: GenerationOptions,
    
    /// Context metadata
    metadata: ContextMetadata,
}

/// Generation options
#[derive(Debug, Clone, Default)]
pub struct GenerationOptions {
    /// Enable minification
    pub minify: bool,
    
    /// Enable source maps
    pub source_maps: bool,
    
    /// Enable comments
    pub comments: bool,
    
    /// Target version
    pub target_version: Option<String>,
}

/// Context metadata
#[derive(Debug, Clone)]
pub struct ContextMetadata {
    /// Generation timestamp
    pub generated_at: SystemTime,
    
    /// Generator version
    pub generator_version: String,
    
    /// Configuration hash
    pub config_hash: u64,
}

/// Generation errors
#[derive(Debug, Clone)]
pub struct GenerationError {
    /// Error type
    pub error_type: GenerationErrorType,
    
    /// Error message
    pub message: String,
    
    /// Error context
    pub context: Option<String>,
    
    /// Line number
    pub line: Option<usize>,
    
    /// Column number
    pub column: Option<usize>,
}

/// Types of generation errors
#[derive(Debug, Clone, PartialEq)]
pub enum GenerationErrorType {
    /// Template error
    TemplateError,
    
    /// Validation error
    ValidationError,
    
    /// Format not supported
    FormatNotSupported,
    
    /// Invalid configuration
    InvalidConfiguration,
    
    /// Generation timeout
    Timeout,
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Validation passed
    pub valid: bool,
    
    /// Validation errors
    pub errors: Vec<ValidationError>,
    
    /// Validation warnings
    pub warnings: Vec<ValidationWarning>,
    
    /// Validation metadata
    pub metadata: ValidationMetadata,
}

/// Validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Error message
    pub message: String,
    
    /// Error severity
    pub severity: ValidationSeverity,
    
    /// Line number
    pub line: Option<usize>,
    
    /// Column number
    pub column: Option<usize>,
}

/// Validation warning
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    /// Warning message
    pub message: String,
    
    /// Warning type
    pub warning_type: ValidationWarningType,
    
    /// Line number
    pub line: Option<usize>,
}

/// Validation severity
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationSeverity {
    /// Error
    Error,
    
    /// Warning
    Warning,
    
    /// Info
    Info,
}

/// Validation warning types
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationWarningType {
    /// Performance warning
    Performance,
    
    /// Style warning
    Style,
    
    /// Compatibility warning
    Compatibility,
    
    /// Best practice warning
    BestPractice,
}

/// Validation metadata
#[derive(Debug, Clone, Default)]
pub struct ValidationMetadata {
    /// Validation time
    pub validation_time: Duration,
    
    /// Validator version
    pub validator_version: String,
    
    /// Rules applied
    pub rules_applied: Vec<String>,
}

/// Generation performance metrics
#[derive(Debug, Clone, Default)]
pub struct GenerationMetrics {
    /// Total statements generated
    pub statements_generated: u64,
    
    /// Total generation time
    pub total_generation_time: Duration,
    
    /// Average generation time per statement
    pub avg_generation_time: Duration,
    
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
    
    /// Validation success rate
    pub validation_success_rate: f64,
    
    /// Memory usage
    pub memory_usage: usize,
    
    /// Optimization effectiveness
    pub optimization_effectiveness: f64,
}

// Trait definitions for extensibility

// Trait definitions for extensibility - using concrete types for simplicity

/// Template preprocessor function type
pub type TemplatePreprocessor = fn(&str, &RenderingContext) -> Result<String, RenderingError>;

/// Template postprocessor function type  
pub type TemplatePostprocessor = fn(&str, &RenderingContext) -> Result<String, RenderingError>;

/// Value transformer function type
pub type ValueTransformer = fn(&TemplateValue) -> Result<TemplateValue, RenderingError>;

/// Pattern matcher function type
pub type PatternMatcher = fn(&[ResolvedImport]) -> bool;

/// Consolidation action handler function type
pub type ConsolidationActionHandler = fn(&[ResolvedImport]) -> Result<Vec<ResolvedImport>, GenerationError>;

/// Conflict resolver function type
pub type ConflictResolver = fn(&[String]) -> Result<HashMap<String, String>, GenerationError>;

/// Dependency ordering handler function type
pub type DependencyOrderingHandler = fn(&[ResolvedDependency]) -> Result<Vec<ResolvedDependency>, GenerationError>;

/// Bundle matcher function type
pub type BundleMatcher = fn(&[ResolvedImport]) -> bool;

impl ImportStatementGenerator {
    /// Create a new import statement generator
    pub fn new() -> Self {
        ImportStatementGenerator {
            config: StatementGeneratorConfig::default(),
            template_engine: TemplateEngine::new(),
            optimization_pipeline: OptimizationPipeline::new(),
            format_generators: Self::create_default_generators(),
            performance_metrics: GenerationMetrics::default(),
            generation_context: GenerationContext::default(),
        }
    }
    
    /// Create generator with custom configuration
    pub fn with_config(config: StatementGeneratorConfig) -> Self {
        ImportStatementGenerator {
            config,
            template_engine: TemplateEngine::new(),
            optimization_pipeline: OptimizationPipeline::new(),
            format_generators: Self::create_default_generators(),
            performance_metrics: GenerationMetrics::default(),
            generation_context: GenerationContext::default(),
        }
    }
    
    /// Generate import statements from resolved imports
    pub fn generate_import_statements(
        &mut self,
        resolution_results: &ImportResolutionResults,
        compile_time_results: &CompileTimeResolutionResults,
        target_format: Option<OutputFormat>,
    ) -> Result<ImportGenerationResults, TreeShakeError> {
        let start_time = std::time::Instant::now();
        let format = target_format.unwrap_or(self.config.default_format.clone());
        
        // Step 1: Optimize imports through pipeline
        let optimized_imports = if self.config.enable_consolidation {
            self.optimization_pipeline.optimize_imports(&resolution_results.resolved_imports)?
        } else {
            resolution_results.resolved_imports.clone()
        };
        
        // Step 2: Get format-specific generator
        let generator = self.format_generators.get(&format)
            .ok_or_else(|| TreeShakeError::DependencyAnalysisError {
                message: format!("Unsupported output format: {:?}", format),
            })?;
        
        // Step 3: Generate statements
        let generated_statements = generator.generate_imports(&optimized_imports, &self.generation_context)
            .map_err(|e| TreeShakeError::DependencyAnalysisError { message: e.to_string() })?;
        
        // Step 4: Validate if enabled
        let validation_result = if self.config.enable_syntax_validation {
            Some(generator.validate(&generated_statements)
                .map_err(|e| TreeShakeError::DependencyAnalysisError { message: e.to_string() })?)
        } else {
            None
        };
        
        // Step 5: Update metrics
        self.update_metrics(start_time.elapsed(), &optimized_imports);
        
        Ok(ImportGenerationResults {
            generated_code: generated_statements,
            format_used: format,
            statements_generated: optimized_imports.len(),
            optimization_applied: self.config.enable_consolidation,
            validation_result,
            performance_metrics: ImportGenerationPerformance {
                generation_time: start_time.elapsed(),
                statements_per_second: optimized_imports.len() as f64 / start_time.elapsed().as_secs_f64(),
                cache_hit_ratio: self.performance_metrics.cache_hit_ratio,
                memory_usage: self.performance_metrics.memory_usage,
            },
            metadata: ImportGenerationMetadata {
                generator_version: env!("CARGO_PKG_VERSION").to_string(),
                generated_at: SystemTime::now(),
                config_hash: self.calculate_config_hash(),
                optimization_effectiveness: self.performance_metrics.optimization_effectiveness,
            },
        })
    }
    
    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> &GenerationMetrics {
        &self.performance_metrics
    }
    
    /// Configure generator
    pub fn configure(&mut self, config: StatementGeneratorConfig) {
        self.config = config;
    }
    
    /// Add custom format generator
    pub fn add_format_generator(&mut self, format: OutputFormat, generator: Box<dyn FormatGenerator>) {
        self.format_generators.insert(format, generator);
    }
    
    /// Clear caches
    pub fn clear_caches(&mut self) {
        self.template_engine.clear_cache();
        self.generation_context.import_cache.clear();
    }
    
    // Private implementation methods
    
    fn create_default_generators() -> HashMap<OutputFormat, Box<dyn FormatGenerator>> {
        let mut generators: HashMap<OutputFormat, Box<dyn FormatGenerator>> = HashMap::new();
        
        generators.insert(OutputFormat::ES6Module, Box::new(ES6ModuleGenerator::new()));
        generators.insert(OutputFormat::CommonJS, Box::new(CommonJSGenerator::new()));
        generators.insert(OutputFormat::Rust, Box::new(RustGenerator::new()));
        generators.insert(OutputFormat::TypeScript, Box::new(TypeScriptGenerator::new()));
        
        generators
    }
    
    fn update_metrics(&mut self, generation_time: Duration, imports: &[ResolvedImport]) {
        self.performance_metrics.statements_generated += imports.len() as u64;
        
        let total_time = self.performance_metrics.total_generation_time + generation_time;
        self.performance_metrics.total_generation_time = total_time;
        
        if self.performance_metrics.statements_generated > 0 {
            self.performance_metrics.avg_generation_time = Duration::from_nanos(
                total_time.as_nanos() as u64 / self.performance_metrics.statements_generated
            );
        }
    }
    
    fn calculate_config_hash(&self) -> u64 {
        // Simple hash calculation - in practice would use proper hasher
        42 // Placeholder
    }
}

// Default format generators

/// ES6 module generator
#[derive(Debug)]
pub struct ES6ModuleGenerator {
    config: ES6Config,
}

#[derive(Debug, Clone, Default)]
pub struct ES6Config {
    pub use_semicolons: bool,
    pub single_quotes: bool,
    pub sort_imports: bool,
}

impl ES6ModuleGenerator {
    pub fn new() -> Self {
        ES6ModuleGenerator {
            config: ES6Config::default(),
        }
    }
}

impl FormatGenerator for ES6ModuleGenerator {
    fn generate_import(&self, import: &ResolvedImport, _context: &GenerationContext) -> Result<String, GenerationError> {
        let quote = if self.config.single_quotes { "'" } else { "\"" };
        let semicolon = if self.config.use_semicolons { ";" } else { "" };
        
        let statement = match &import.import_type {
            ResolvedImportType::Selective { functions } => {
                if functions.len() == 1 {
                    format!("import {{ {} }} from {}{}{}{}", 
                           functions[0], quote, import.source_module, quote, semicolon)
                } else {
                    let mut sorted_functions = functions.clone();
                    if self.config.sort_imports {
                        sorted_functions.sort();
                    }
                    format!("import {{ {} }} from {}{}{}{}", 
                           sorted_functions.join(", "), quote, import.source_module, quote, semicolon)
                }
            },
            ResolvedImportType::Full => {
                format!("import * as {} from {}{}{}{}", 
                       import.source_module.replace("::", "_"), quote, import.source_module, quote, semicolon)
            },
            ResolvedImportType::Single { function } => {
                format!("import {{ {} }} from {}{}{}{}", 
                       function, quote, import.source_module, quote, semicolon)
            },
            ResolvedImportType::Aliased { function, alias } => {
                format!("import {{ {} as {} }} from {}{}{}{}", 
                       function, alias, quote, import.source_module, quote, semicolon)
            },
            _ => {
                return Err(GenerationError {
                    error_type: GenerationErrorType::FormatNotSupported,
                    message: "Import type not supported by ES6 generator".to_string(),
                    context: None,
                    line: None,
                    column: None,
                });
            }
        };
        
        Ok(statement)
    }
    
    fn generate_imports(&self, imports: &[ResolvedImport], context: &GenerationContext) -> Result<String, GenerationError> {
        let mut statements = Vec::new();
        
        for import in imports {
            statements.push(self.generate_import(import, context)?);
        }
        
        Ok(statements.join("\n"))
    }
    
    fn get_metadata(&self) -> FormatMetadata {
        FormatMetadata {
            name: "ES6 Modules".to_string(),
            version: "1.0.0".to_string(),
            features: vec![
                FormatFeature::TreeShaking,
                FormatFeature::DynamicImports,
                FormatFeature::NamespaceImports,
            ],
            file_extensions: vec!["js".to_string(), "mjs".to_string()],
        }
    }
    
    fn validate(&self, code: &str) -> Result<ValidationResult, GenerationError> {
        // Basic validation - check for proper import syntax
        let lines: Vec<&str> = code.lines().collect();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        
        for (i, line) in lines.iter().enumerate() {
            if line.trim().starts_with("import ") && !line.contains("from") && !line.trim().ends_with(';') {
                errors.push(ValidationError {
                    message: "Import statement missing 'from' clause".to_string(),
                    severity: ValidationSeverity::Error,
                    line: Some(i + 1),
                    column: None,
                });
            }
        }
        
        Ok(ValidationResult {
            valid: errors.is_empty(),
            errors,
            warnings,
            metadata: ValidationMetadata::default(),
        })
    }
}

/// CommonJS generator
#[derive(Debug)]
pub struct CommonJSGenerator;

impl CommonJSGenerator {
    pub fn new() -> Self {
        CommonJSGenerator
    }
}

impl FormatGenerator for CommonJSGenerator {
    fn generate_import(&self, import: &ResolvedImport, _context: &GenerationContext) -> Result<String, GenerationError> {
        let statement = match &import.import_type {
            ResolvedImportType::Selective { functions } => {
                format!("const {{ {} }} = require('{}');", 
                       functions.join(", "), import.source_module)
            },
            ResolvedImportType::Full => {
                format!("const {} = require('{}');", 
                       import.source_module.replace("::", "_"), import.source_module)
            },
            _ => {
                return Err(GenerationError {
                    error_type: GenerationErrorType::FormatNotSupported,
                    message: "Import type not fully supported by CommonJS generator".to_string(),
                    context: None,
                    line: None,
                    column: None,
                });
            }
        };
        
        Ok(statement)
    }
    
    fn generate_imports(&self, imports: &[ResolvedImport], context: &GenerationContext) -> Result<String, GenerationError> {
        let mut statements = Vec::new();
        
        for import in imports {
            statements.push(self.generate_import(import, context)?);
        }
        
        Ok(statements.join("\n"))
    }
    
    fn get_metadata(&self) -> FormatMetadata {
        FormatMetadata {
            name: "CommonJS".to_string(),
            version: "1.0.0".to_string(),
            features: vec![FormatFeature::NamespaceImports],
            file_extensions: vec!["js".to_string()],
        }
    }
    
    fn validate(&self, _code: &str) -> Result<ValidationResult, GenerationError> {
        Ok(ValidationResult {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            metadata: ValidationMetadata::default(),
        })
    }
}

/// Rust generator
#[derive(Debug)]
pub struct RustGenerator;

impl RustGenerator {
    pub fn new() -> Self {
        RustGenerator
    }
}

impl FormatGenerator for RustGenerator {
    fn generate_import(&self, import: &ResolvedImport, _context: &GenerationContext) -> Result<String, GenerationError> {
        let statement = match &import.import_type {
            ResolvedImportType::Selective { functions } => {
                format!("use {}::{{{}}};", 
                       import.source_module, functions.join(", "))
            },
            ResolvedImportType::Full => {
                format!("use {}::*;", import.source_module)
            },
            _ => {
                return Err(GenerationError {
                    error_type: GenerationErrorType::FormatNotSupported,
                    message: "Import type not fully supported by Rust generator".to_string(),
                    context: None,
                    line: None,
                    column: None,
                });
            }
        };
        
        Ok(statement)
    }
    
    fn generate_imports(&self, imports: &[ResolvedImport], context: &GenerationContext) -> Result<String, GenerationError> {
        let mut statements = Vec::new();
        
        for import in imports {
            statements.push(self.generate_import(import, context)?);
        }
        
        Ok(statements.join("\n"))
    }
    
    fn get_metadata(&self) -> FormatMetadata {
        FormatMetadata {
            name: "Rust".to_string(),
            version: "1.0.0".to_string(),
            features: vec![FormatFeature::TreeShaking, FormatFeature::NamespaceImports],
            file_extensions: vec!["rs".to_string()],
        }
    }
    
    fn validate(&self, _code: &str) -> Result<ValidationResult, GenerationError> {
        Ok(ValidationResult {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            metadata: ValidationMetadata::default(),
        })
    }
}

/// TypeScript generator
#[derive(Debug)]
pub struct TypeScriptGenerator;

impl TypeScriptGenerator {
    pub fn new() -> Self {
        TypeScriptGenerator
    }
}

impl FormatGenerator for TypeScriptGenerator {
    fn generate_import(&self, import: &ResolvedImport, context: &GenerationContext) -> Result<String, GenerationError> {
        // Delegate to ES6 generator for basic functionality
        let es6_generator = ES6ModuleGenerator::new();
        es6_generator.generate_import(import, context)
    }
    
    fn generate_imports(&self, imports: &[ResolvedImport], context: &GenerationContext) -> Result<String, GenerationError> {
        let es6_generator = ES6ModuleGenerator::new();
        es6_generator.generate_imports(imports, context)
    }
    
    fn get_metadata(&self) -> FormatMetadata {
        FormatMetadata {
            name: "TypeScript".to_string(),
            version: "1.0.0".to_string(),
            features: vec![
                FormatFeature::TreeShaking,
                FormatFeature::DynamicImports,
                FormatFeature::NamespaceImports,
                FormatFeature::TypeAnnotations,
            ],
            file_extensions: vec!["ts".to_string(), "tsx".to_string()],
        }
    }
    
    fn validate(&self, code: &str) -> Result<ValidationResult, GenerationError> {
        let es6_generator = ES6ModuleGenerator::new();
        es6_generator.validate(code)
    }
}

// Implementation of remaining components

impl TemplateEngine {
    pub fn new() -> Self {
        TemplateEngine {
            templates: HashMap::new(),
            renderer: TemplateRenderer::default(),
            template_cache: TemplateCache::default(),
        }
    }
    
    pub fn clear_cache(&mut self) {
        self.template_cache.rendered_cache.clear();
        self.template_cache.cache_stats = CacheStats::default();
    }
}

impl OptimizationPipeline {
    pub fn new() -> Self {
        OptimizationPipeline {
            consolidator: StatementConsolidator::default(),
            alias_manager: AliasManager::default(),
            dependency_orderer: DependencyOrderer::default(),
            bundler: ImportBundler::default(),
            config: PipelineConfig::default(),
        }
    }
    
    pub fn optimize_imports(&mut self, imports: &[ResolvedImport]) -> Result<Vec<ResolvedImport>, TreeShakeError> {
        let mut optimized = imports.to_vec();
        
        // Apply consolidation if enabled
        if self.config.enable_consolidation {
            optimized = self.consolidator.consolidate_imports(optimized)?;
        }
        
        // Apply alias management if enabled
        if self.config.enable_alias_management {
            optimized = self.alias_manager.resolve_aliases(optimized)?;
        }
        
        // Apply dependency ordering if enabled
        if self.config.enable_dependency_ordering {
            optimized = self.dependency_orderer.order_imports(optimized)?;
        }
        
        Ok(optimized)
    }
}

impl StatementConsolidator {
    pub fn consolidate_imports(&mut self, imports: Vec<ResolvedImport>) -> Result<Vec<ResolvedImport>, TreeShakeError> {
        let imports_len = imports.len();
        // Group imports by module
        let mut module_groups: HashMap<String, Vec<ResolvedImport>> = HashMap::new();
        
        for import in imports.into_iter() {
            module_groups.entry(import.source_module.clone())
                .or_insert_with(Vec::new)
                .push(import);
        }
        
        let mut consolidated = Vec::new();
        
        // Consolidate imports from same module
        for (module, imports) in module_groups {
            if imports.len() == 1 {
                consolidated.extend(imports);
            } else {
                // Merge selective imports from same module
                let mut all_functions = Vec::new();
                
                for import in &imports {
                    if let ResolvedImportType::Selective { functions } = &import.import_type {
                        all_functions.extend(functions.clone());
                    }
                }
                
                if !all_functions.is_empty() {
                    // Remove duplicates and sort
                    let unique_functions: BTreeSet<String> = all_functions.into_iter().collect();
                    let functions: Vec<String> = unique_functions.into_iter().collect();
                    
                    consolidated.push(ResolvedImport {
                        import_type: ResolvedImportType::Selective { functions },
                        source_module: module,
                        imported_functions: imports[0].imported_functions.clone(),
                        import_statement: String::new(), // Will be generated later
                        metadata: imports[0].metadata.clone(),
                        performance_impact: imports[0].performance_impact.clone(),
                    });
                } else {
                    consolidated.extend(imports);
                }
            }
        }
        
        // Update statistics
        self.stats.total_imports = imports_len;
        self.stats.consolidated_imports = consolidated.len();
        self.stats.consolidation_ratio = if imports.len() > 0 {
            consolidated.len() as f64 / imports.len() as f64
        } else {
            1.0
        };
        
        Ok(consolidated)
    }
}

impl AliasManager {
    pub fn resolve_aliases(&mut self, imports: Vec<ResolvedImport>) -> Result<Vec<ResolvedImport>, TreeShakeError> {
        // For now, just return the imports unchanged
        // In a real implementation, this would handle alias conflicts
        Ok(imports)
    }
}

impl DependencyOrderer {
    pub fn order_imports(&mut self, imports: Vec<ResolvedImport>) -> Result<Vec<ResolvedImport>, TreeShakeError> {
        // For now, just sort alphabetically by module name
        let mut ordered = imports;
        ordered.sort_by(|a, b| a.source_module.cmp(&b.source_module));
        Ok(ordered)
    }
}

/// Results of import statement generation
#[derive(Debug, Clone)]
pub struct ImportGenerationResults {
    /// Generated import code
    pub generated_code: String,
    
    /// Format used for generation
    pub format_used: OutputFormat,
    
    /// Number of statements generated
    pub statements_generated: usize,
    
    /// Whether optimization was applied
    pub optimization_applied: bool,
    
    /// Validation result (if validation was enabled)
    pub validation_result: Option<ValidationResult>,
    
    /// Performance metrics
    pub performance_metrics: ImportGenerationPerformance,
    
    /// Generation metadata
    pub metadata: ImportGenerationMetadata,
}

/// Performance metrics for import generation
#[derive(Debug, Clone)]
pub struct ImportGenerationPerformance {
    /// Total generation time
    pub generation_time: Duration,
    
    /// Statements generated per second
    pub statements_per_second: f64,
    
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
    
    /// Memory usage
    pub memory_usage: usize,
}

/// Import generation metadata
#[derive(Debug, Clone)]
pub struct ImportGenerationMetadata {
    /// Generator version
    pub generator_version: String,
    
    /// Generation timestamp
    pub generated_at: SystemTime,
    
    /// Configuration hash
    pub config_hash: u64,
    
    /// Optimization effectiveness
    pub optimization_effectiveness: f64,
}

impl Default for ImportStatementGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ContextMetadata {
    fn default() -> Self {
        ContextMetadata {
            generated_at: SystemTime::now(),
            generator_version: "1.0.0".to_string(),
            config_hash: 0,
        }
    }
}

impl Default for GenerationContext {
    fn default() -> Self {
        GenerationContext {
            current_module: None,
            import_cache: HashMap::new(),
            options: GenerationOptions::default(),
            metadata: ContextMetadata::default(),
        }
    }
}

impl std::fmt::Display for GenerationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.error_type, self.message)
    }
}

impl std::fmt::Display for GenerationErrorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenerationErrorType::TemplateError => write!(f, "Template Error"),
            GenerationErrorType::ValidationError => write!(f, "Validation Error"),
            GenerationErrorType::FormatNotSupported => write!(f, "Format Not Supported"),
            GenerationErrorType::InvalidConfiguration => write!(f, "Invalid Configuration"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_creation() {
        let generator = ImportStatementGenerator::new();
        assert_eq!(generator.config.default_format, OutputFormat::ES6Module);
        assert!(generator.config.enable_consolidation);
    }

    #[test]
    fn test_es6_generator() {
        let generator = ES6ModuleGenerator::new();
        let import = ResolvedImport {
            import_type: ResolvedImportType::Selective { 
                functions: vec!["Sin".to_string(), "Cos".to_string()] 
            },
            source_module: "std::math".to_string(),
            imported_functions: Vec::new(),
            import_statement: String::new(),
            metadata: Default::default(),
            performance_impact: Default::default(),
        };
        
        let context = GenerationContext::default();
        let result = generator.generate_import(&import, &context).unwrap();
        assert!(result.contains("import { Sin, Cos } from 'std::math'"));
    }

    #[test]
    fn test_commonjs_generator() {
        let generator = CommonJSGenerator::new();
        let import = ResolvedImport {
            import_type: ResolvedImportType::Selective { 
                functions: vec!["Sin".to_string()] 
            },
            source_module: "std::math".to_string(),
            imported_functions: Vec::new(),
            import_statement: String::new(),
            metadata: Default::default(),
            performance_impact: Default::default(),
        };
        
        let context = GenerationContext::default();
        let result = generator.generate_import(&import, &context).unwrap();
        assert!(result.contains("const { Sin } = require('std::math')"));
    }

    #[test]
    fn test_rust_generator() {
        let generator = RustGenerator::new();
        let import = ResolvedImport {
            import_type: ResolvedImportType::Selective { 
                functions: vec!["Sin".to_string(), "Cos".to_string()] 
            },
            source_module: "std::math".to_string(),
            imported_functions: Vec::new(),
            import_statement: String::new(),
            metadata: Default::default(),
            performance_impact: Default::default(),
        };
        
        let context = GenerationContext::default();
        let result = generator.generate_import(&import, &context).unwrap();
        assert!(result.contains("use std::math::{Sin, Cos}"));
    }

    #[test]
    fn test_statement_consolidation() {
        let mut consolidator = StatementConsolidator::default();
        
        let imports = vec![
            ResolvedImport {
                import_type: ResolvedImportType::Selective { 
                    functions: vec!["Sin".to_string()] 
                },
                source_module: "std::math".to_string(),
                imported_functions: Vec::new(),
                import_statement: String::new(),
                metadata: Default::default(),
                performance_impact: Default::default(),
            },
            ResolvedImport {
                import_type: ResolvedImportType::Selective { 
                    functions: vec!["Cos".to_string()] 
                },
                source_module: "std::math".to_string(),
                imported_functions: Vec::new(),
                import_statement: String::new(),
                metadata: Default::default(),
                performance_impact: Default::default(),
            },
        ];
        
        let result = consolidator.consolidate_imports(imports).unwrap();
        assert_eq!(result.len(), 1);
        
        if let ResolvedImportType::Selective { functions } = &result[0].import_type {
            assert_eq!(functions.len(), 2);
            assert!(functions.contains(&"Sin".to_string()));
            assert!(functions.contains(&"Cos".to_string()));
        } else {
            panic!("Expected selective import type");
        }
    }

    #[test]
    fn test_output_formats() {
        assert_eq!(OutputFormat::ES6Module, OutputFormat::ES6Module);
        assert_ne!(OutputFormat::ES6Module, OutputFormat::CommonJS);
        
        let custom1 = OutputFormat::Custom { name: "test".to_string() };
        let custom2 = OutputFormat::Custom { name: "test".to_string() };
        assert_eq!(custom1, custom2);
    }

    #[test]
    fn test_format_metadata() {
        let es6_gen = ES6ModuleGenerator::new();
        let metadata = es6_gen.get_metadata();
        assert_eq!(metadata.name, "ES6 Modules");
        assert!(metadata.features.contains(&FormatFeature::TreeShaking));
    }

    #[test]
    fn test_template_engine() {
        let engine = TemplateEngine::new();
        assert_eq!(engine.templates.len(), 0);
        assert_eq!(engine.template_cache.cache_stats.hits, 0);
    }

    #[test]
    fn test_optimization_pipeline() {
        let pipeline = OptimizationPipeline::new();
        assert!(pipeline.config.enable_consolidation);
        assert!(pipeline.config.enable_dependency_ordering);
    }

    #[test]
    fn test_validation_result() {
        let result = ValidationResult {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            metadata: ValidationMetadata::default(),
        };
        
        assert!(result.valid);
        assert_eq!(result.errors.len(), 0);
    }

    #[test]
    fn test_generation_error() {
        let error = GenerationError {
            error_type: GenerationErrorType::TemplateError,
            message: "Test error".to_string(),
            context: None,
            line: Some(1),
            column: Some(5),
        };
        
        assert_eq!(error.error_type, GenerationErrorType::TemplateError);
        assert_eq!(error.line, Some(1));
    }
}