#![allow(unused_variables, unused_imports)]
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
use crate::vm::Value;
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
            // Analytics: Time Series
            "TimeSeriesDecompose" => "TimeSeriesDecompose[series_, model_, period_]".to_string(),
            "SeasonalDecompose" => "SeasonalDecompose[series_, period_, method_?]".to_string(),
            "ARIMAAdvanced" | "ARIMA" => "ARIMAAdvanced[series_, order_, seasonalOrder_?]".to_string(),
            "Forecast" => "Forecast[model_, periods_, confidenceLevel_?]".to_string(),
            "AutoCorrelation" => "AutoCorrelation[series_, lags_]".to_string(),
            "PartialAutoCorrelation" => "PartialAutoCorrelation[series_, lags_]".to_string(),
            "TrendAnalysis" => "TrendAnalysis[series_, method_?]".to_string(),
            "ChangePointDetection" => "ChangePointDetection[series_, method_?, sensitivity_?]".to_string(),
            "AnomalyDetection" => "AnomalyDetection[series_, method_?, threshold_?]".to_string(),
            "StationarityTest" => "StationarityTest[series_, testType_?]".to_string(),
            "CrossCorrelation" => "CrossCorrelation[series1_, series2_, lags_]".to_string(),
            "SpectralDensity" => "SpectralDensity[series_, method_?]".to_string(),
            // Analytics: Business Intelligence
            "KPI" => "KPI[data_, metricDefinition_, target_?, period_]".to_string(),
            "CohortAnalysis" => "CohortAnalysis[data_, cohortCriteria_, metrics_]".to_string(),
            "FunnelAnalysis" => "FunnelAnalysis[data_, stages_, conversionEvents_]".to_string(),
            "RetentionAnalysis" => "RetentionAnalysis[data_, userIdCol_, eventDateCol_, periods_]".to_string(),
            "LTV" => "LTV[customerData_, revenueEvents_, timeHorizon_]".to_string(),
            "Churn" => "Churn[data_, features_, predictionHorizon_?]".to_string(),
            "Segmentation" => "Segmentation[data_, features_, method_, k_?]".to_string(),
            "ABTestAnalysis" => "ABTestAnalysis[control_, treatment_, metric_, alpha_?]".to_string(),
            "AttributionModel" => "AttributionModel[touchpoints_, conversions_, modelType_]".to_string(),
            "Dashboard" => "Dashboard[metrics_, filters_?, visualizations_?]".to_string(),
            // Signal: FFT and filtering
            // Vision: Edge detection
            "CannyEdges" => "CannyEdges[image_, opts_?]".to_string(),
            "SobelEdges" => "SobelEdges[image_, opts_?]".to_string(),
            "LaplacianEdges" => "LaplacianEdges[image_, opts_?]".to_string(),
            "PrewittEdges" => "PrewittEdges[image_, opts_?]".to_string(),
            "RobertsEdges" => "RobertsEdges[image_, opts_?]".to_string(),
            "ScharrEdges" => "ScharrEdges[image_, opts_?]".to_string(),
            // Vision: Features
            "HarrisCorners" => "HarrisCorners[image_, opts_?]".to_string(),
            "SIFTFeatures" => "SIFTFeatures[image_, opts_?]".to_string(),
            "ORBFeatures" => "ORBFeatures[image_, opts_?]".to_string(),
            "MatchFeatures" => "MatchFeatures[features1_, features2_, opts_?]".to_string(),
            "FFT" => "FFT[signal_]".to_string(),
            "IFFT" => "IFFT[spectrum_]".to_string(),
            "RealFFT" => "RealFFT[signal_]".to_string(),
            "Periodogram" => "Periodogram[signal_]".to_string(),
            "WelchPSD" => "WelchPSD[signal_, segmentLength_?, overlap_?]".to_string(),
            "FIRFilter" => "FIRFilter[signal_, coefficients_]".to_string(),
            "IIRFilter" => "IIRFilter[signal_, bCoeffs_, aCoeffs_]".to_string(),
            "LowPassFilter" => "LowPassFilter[signalData_, cutoff_]".to_string(),
            "HighPassFilter" => "HighPassFilter[signalData_, cutoff_]".to_string(),
            "BandPassFilter" => "BandPassFilter[signalData_, low_, high_]".to_string(),
            "MedianFilter" => "MedianFilter[signal_, window_]".to_string(),
            "ApplyWindow" => "ApplyWindow[signalData_, window_]".to_string(),
            // AI/RAG/Vector
            "VectorSearch" => "VectorSearch[store_, vector_, k_, filter_?]".to_string(),
            "VectorCluster" => "VectorCluster[store_, algorithm_, params_]".to_string(),
            "EmbeddingCluster" => "EmbeddingCluster[embeddings_, k_]".to_string(),
            "ContextRetrieval" => "ContextRetrieval[store_, query_, k_, filter_?]".to_string(),
            "RAGQuery" => "RAGQuery[question_, contexts_, model_, template_]".to_string(),
            // Graph
            "AdvancedPageRank" => "AdvancedPageRank[graph_, damping_?, maxIter_?]".to_string(),
            "LouvainCommunityDetection" => "LouvainCommunityDetection[graph_, opts_?]".to_string(),
            "BetweennessCentrality" => "BetweennessCentrality[graph_]".to_string(),
            "ClosenessCentrality" => "ClosenessCentrality[graph_]".to_string(),
            // ML Quick starts and evaluation
            "AutoMLQuickStart" => "AutoMLQuickStart[dataset_]".to_string(),
            "AutoMLQuickStartTable" => "AutoMLQuickStartTable[table_, featureCols_, targetCol_]".to_string(),
            "CrossValidate" => "CrossValidate[dataset_, k_, metric_, opts_?]".to_string(),
            "CrossValidateTable" => "CrossValidateTable[table_, featureCols_, targetCol_, k_, metric_, opts_?]".to_string(),
            // Number Theory: Crypto/Modular
            "HashFunction" => "HashFunction[data_, algorithm_]".to_string(),
            "DiscreteLog" => "DiscreteLog[a_, b_, m_]".to_string(),
            // Time Series moving averages
            "MovingAverage" => "MovingAverage[timeseries_, window_]".to_string(),
            "ExponentialMovingAverage" => "ExponentialMovingAverage[timeseries_, alpha_]".to_string(),
            // Image: Core transforms and analysis
            "AffineTransform" => "AffineTransform[image_, transformMatrix_, interpolation_?]".to_string(),
            "PerspectiveTransform" => "PerspectiveTransform[image_, transformMatrix_, interpolation_?]".to_string(),
            "ContourDetection" => "ContourDetection[image_, threshold_?, method_?]".to_string(),
            "FeatureDetection" => "FeatureDetection[image_, method_?, threshold_?]".to_string(),
            "TemplateMatching" => "TemplateMatching[image_, template_, method_?]".to_string(),
            "ImageSegmentation" => "ImageSegmentation[image_, method_?, parameters_?]".to_string(),
            // Streaming constructors
            "WindowAggregate" => "WindowAggregate[windowType_, windowSizeMs_, aggregation_]".to_string(),
            "StreamJoin" => "StreamJoin[joinKey_, windowSizeMs_]".to_string(),
            "ComplexEventProcessing" => "ComplexEventProcessing[]".to_string(),
            "BackpressureControl" => "BackpressureControl[strategy_, thresholds_]".to_string(),
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
            // Analytics: Statistics
            "Regression" => "Fits a regression model over data using method 'linear' | 'polynomial' | 'logistic'. Supports options such as method and degree (for polynomial). Returns an association with coefficients, rSquared/pseudoRSquared, errors, and more.".to_string(),
            "TTest" => "Performs one- or two-sample Student's t-test. Accepts samples and options (mu, confidenceLevel). Returns an association with statistic, pValue, df, effectSize, and confidenceInterval.".to_string(),
            "CorrelationMatrix" => "Computes a correlation matrix for variables using 'pearson' | 'spearman' | 'kendall'. Returns an association with matrix, columns, and method.".to_string(),
            "ANOVA" => "Runs analysis of variance. Returns an association with fStatistic, pValue, df1, df2, and effectSize (eta-squared).".to_string(),
            "ConfidenceInterval" => "Computes a confidence interval for the mean. Accepts options confidenceLevel and method ('t' | 'z'). Returns an association with estimate, standardError, confidenceInterval, method, and n.".to_string(),
            "EffectSize" => "Computes effect size between two groups. Supports method ('cohens_d' | 'eta_squared'). Returns an association with method and effectSize.".to_string(),
            "ChiSquareTest" => "Performs chi-square goodness-of-fit test. Returns an association with test, statistic, pValue, and df.".to_string(),
            "NormalityTest" => "Tests normality using 'shapiro' or 'ks'. Returns an association with test, statistic, and pValue.".to_string(),
            "BootstrapSample" => "Performs bootstrap resampling of the mean. Returns an association with originalStatistic, bootstrapStatistics, standardError, bias, and confidenceInterval.".to_string(),
            "PowerAnalysis" => "Computes simple power analysis parameters. Returns an association with effectSize, alpha, power, testType, and sampleSize (approximate).".to_string(),
            // Analytics: Time Series
            "TimeSeriesDecompose" => "Decomposes a time series into trend, seasonal, and residual components (additive or multiplicative). Returns an association.".to_string(),
            "SeasonalDecompose" => "Convenience decomposition by period and method. Returns trend, seasonal, residual.".to_string(),
            "ARIMAAdvanced" | "ARIMA" => "Fits a simple ARIMA model and returns an association with order, seasonalOrder, coefficients, fittedValues, residuals, aic, bic.".to_string(),
            "Forecast" => "Generates a naive forecast from a model association. Returns forecasts and confidence bounds.".to_string(),
            "AutoCorrelation" => "Computes the autocorrelation function up to specified lags.".to_string(),
            "PartialAutoCorrelation" => "Computes an approximate partial autocorrelation function.".to_string(),
            "TrendAnalysis" => "Extracts a trend component (linear, polynomial, moving_average).".to_string(),
            "ChangePointDetection" => "Detects change points using methods like 'cusum' or 'variance'.".to_string(),
            "AnomalyDetection" => "Detects anomalies in a time series with 'zscore' method and threshold.".to_string(),
            "StationarityTest" => "Runs a simple stationarity test (adf, kpss). Returns an association of test results.".to_string(),
            "CrossCorrelation" => "Computes cross-correlation between two series across lags.".to_string(),
            "SpectralDensity" => "Estimates spectral density via 'periodogram' or 'welch'. Returns frequencies and power.".to_string(),
            // Analytics: Business Intelligence
            "KPI" => "Computes a KPI metric summary. Returns an association with value, target, status, performance, etc.".to_string(),
            "CohortAnalysis" => "Builds retention cohorts and sizes. Returns an association with cohorts, periods, cohortSizes.".to_string(),
            "FunnelAnalysis" => "Analyzes conversion funnel counts and rates. Returns an association with stages, counts, conversionRates, dropOffRates.".to_string(),
            "RetentionAnalysis" => "Computes retention rates over periods. Returns a list of percentages.".to_string(),
            "LTV" => "Estimates customer lifetime value. Returns an association (averageLTV, totalLTV, individualLTV).".to_string(),
            "Churn" => "Performs a simple churn analysis. Returns an association with churnRate, riskFactors, predictionAccuracy.".to_string(),
            "Segmentation" => "Performs customer segmentation (kmeans, rfm). Returns an association describing segments.".to_string(),
            "ABTestAnalysis" => "Runs a basic A/B test analysis. Returns an association with lift, pValue, confidenceInterval.".to_string(),
            "AttributionModel" => "Computes marketing attribution by model type (first_touch, last_touch, linear). Returns an association.".to_string(),
            "Dashboard" => "Creates a summarized dashboard association from metrics, filters, and visualizations.".to_string(),
            // Vision: Edge detection
            "CannyEdges" | "SobelEdges" | "LaplacianEdges" | "PrewittEdges" | "RobertsEdges" | "ScharrEdges" =>
                "Edge detection. Returns <|edges, width, height, algorithm, thresholdLow, thresholdHigh|>.".to_string(),
            // Vision: Features
            "HarrisCorners" | "SIFTFeatures" | "ORBFeatures" =>
                "Returns <|featureType, keypoints: List[Assoc], descriptors: List[List[Real]]|>.".to_string(),
            "MatchFeatures" =>
                "Returns <|matchType, count, matches: List[<|queryIndex, trainIndex, distance, confidence|>]|>.".to_string(),
            // AI/RAG/Vector
            "VectorSearch" => "Search vector store. Returns List of <|id, score, distance, vector, metadata|>. Optional filter is an association.".to_string(),
            "VectorCluster" => "Cluster vectors in store. Returns <|algorithm, k, clusters: List[<|clusterId, members, score, centroid|>]|>.".to_string(),
            "EmbeddingCluster" => "Cluster raw embeddings. Returns <|k, assignments: List[Int], method|>.".to_string(),
            "ContextRetrieval" => "Retrieve relevant contexts. Returns List of <|id, content, score|>.".to_string(),
            "RAGQuery" => "Run a RAG query. Returns <|question, answer|>.".to_string(),
            // Numerical: Differentiation/Roots/Integration schemas
            "FiniteDifference" => "Numerical derivative. Returns <|value, errorEstimate, step, method, order|>.".to_string(),
            "RichardsonExtrapolation" => "Improved derivative via Richardson. Returns <|value, errorEstimate, step, method, order|>.".to_string(),
            "Bisection" | "NewtonRaphson" | "Secant" | "Brent" | "FixedPoint" =>
                "Root finding. Returns <|root, iterations, functionValue, converged, errorEstimate, method|>.".to_string(),
            "Trapezoidal" | "Simpson" | "Romberg" | "GaussQuadrature" | "MonteCarloIntegrate" =>
                "Numerical integration. Returns <|value, errorEstimate, evaluations, method, converged|>.".to_string(),
            // Signal: FFT and filtering
            "FFT" => "Computes the FFT. Returns <|frequencies, magnitudes, phases, sampleRate, method -> \"FFT\"|>.".to_string(),
            "IFFT" => "Reconstructs a time-domain signal from a spectrum. Accepts association with magnitudes and phases. Returns SignalData.".to_string(),
            "RealFFT" => "Computes one-sided FFT for real signals. Returns spectral association with method -> \"RealFFT\".".to_string(),
            "Periodogram" => "Estimates PSD via periodogram. Returns spectral association with method -> \"Periodogram\".".to_string(),
            "WelchPSD" => "Estimates PSD via Welch's method. Returns spectral association with method -> \"WelchPSD\".".to_string(),
            "FIRFilter" => "Applies FIR filter. Returns <|filterType, parameters, success, message, filteredSignal|>.".to_string(),
            "IIRFilter" => "Applies IIR filter. Returns <|filterType, parameters, success, message, filteredSignal|>.".to_string(),
            "LowPassFilter" => "Applies low-pass filter to SignalData. Returns filter association.".to_string(),
            "HighPassFilter" => "Applies high-pass filter to SignalData. Returns filter association.".to_string(),
            "BandPassFilter" => "Applies band-pass filter to SignalData. Returns filter association.".to_string(),
            "MedianFilter" => "Applies median filter to a numeric list. Returns filter association.".to_string(),
            "ApplyWindow" => "Applies a window to SignalData. Returns SignalData (Foreign).".to_string(),
            // Image: transforms and analysis
            "AffineTransform" => "Applies an affine transform to an Image. Returns Image (Foreign).".to_string(),
            "PerspectiveTransform" => "Applies a perspective transform to an Image. Returns Image (Foreign).".to_string(),
            "ContourDetection" => "Detects contours. Returns a list of associations: <|points, area, perimeter, isClosed|>.".to_string(),
            "FeatureDetection" => "Detects feature points. Returns a list of associations: <|x, y, confidence, scale, angle|>.".to_string(),
            "TemplateMatching" => "Performs normalized cross-correlation template matching. Returns an Image (correlation map).".to_string(),
            "ImageSegmentation" => "Segments an image. Returns <|labels: Image, numRegions, regionStats: List[Assoc]|>.".to_string(),
            // Streaming
            "WindowAggregate" => "Creates a windowed aggregation processor (Foreign). Method process returns List[Assoc] <|windowStart, windowEnd, result, count|>.".to_string(),
            "StreamJoin" => "Creates a stream join processor (Foreign). Methods processLeft/processRight return List[Assoc] <|left, right, joinKey, timestamp|>.".to_string(),
            "ComplexEventProcessing" => "Creates a CEP engine (Foreign). processEvent returns List of action results.".to_string(),
            "BackpressureControl" => "Creates a backpressure controller (Foreign). getMetrics returns <|bufferUtilization, throughputRate, latencyP95, droppedMessages, blockedDurationMs|>.".to_string(),
            // Integrations: Optimization results
            "CostAnalysis" => "Returns <|pricingModel, allocationTags, totalCost, costData: Assoc, costTrends: List[Assoc]|>. costData maps service -> <|service, currentCost, projectedCost, costByCategory, optimizationPotential, recommendations|>.".to_string(),
            "RightSizing" => "Returns <|services, targetUtilization, recommendations: Assoc, potentialSavings|>. recommendations maps service -> <|currentSize, recommendedSize, costSaving, performanceImpact, confidenceLevel, implementationEffort|>.".to_string(),
            "CostAlerts" => "Returns <|thresholds: Assoc, recipients: List[String], actions: List[String], activeAlerts: List[Assoc], alertHistory: List[Assoc]|>.".to_string(),
            "BudgetManagement" => "Returns <|budgets: Assoc, allocations: Assoc, tracking: Assoc, forecasts: List[Assoc]|>. tracking maps budget -> <|allocatedAmount, spentAmount, remainingAmount, utilizationPercentage, variance, forecastAccuracy|>.".to_string(),
            // Graph: standard schemas
            "AdvancedPageRank" => "PageRank scores. Returns <|scores: Assoc[String->Real], iterations, error, method|>.".to_string(),
            "LouvainCommunityDetection" => "Louvain communities. Returns <|communities: Assoc[String->Int], modularity, numCommunities, method|>.".to_string(),
            "BetweennessCentrality" => "Betweenness centrality. Returns <|scores: Assoc[String->Real], measureType -> \"Betweenness\"|>.".to_string(),
            "ClosenessCentrality" => "Closeness centrality. Returns <|scores: Assoc[String->Real], measureType -> \"Closeness\"|>.".to_string(),
            // Mathematics: Linear Algebra (decompositions/solvers)
            "SVD" => "Singular value decomposition. Returns <|factor1: U, factor2: S, factor3: Vt, values: singularValues, decompType: \"SVD\", success, condition, info|>.".to_string(),
            "QRDecomposition" => "QR decomposition. Returns <|factor1: Q, factor2: R, decompType: \"QR\", success, condition, info|>.".to_string(),
            "LUDecomposition" => "LU decomposition. Returns <|factor1: L, factor2: U, factor3: P?, decompType: \"LU\", success, condition, info|>.".to_string(),
            "CholeskyDecomposition" => "Cholesky decomposition. Returns <|factor1: L, decompType: \"Cholesky\", success, condition, info|>.".to_string(),
            "EigenDecomposition" => "Eigen decomposition. Returns <|values: eigenvalues, factor1: eigenvectors, decompType: \"Eigen\", success, condition, info|>.".to_string(),
            "SchurDecomposition" => "Schur decomposition. Returns <|factor1: Q, factor2: T, decompType: \"Schur\", success, condition, info|>.".to_string(),
            "LinearSolve" => "Solve linear system Ax=b. Returns <|solution, residual, method, success, condition, rank, info|>.".to_string(),
            "LeastSquares" => "Least squares solution. Returns <|solution, residual, method, success, condition, rank, info|>.".to_string(),
            // ML Evaluation (schemas)
            "CrossValidate" | "CrossValidateTable" => "Cross-validation. Returns <|foldScores, meanScore, stdDev, bestFoldIndex, scoringMetric, foldReports|>. foldReports items are Classification or Regression report associations.".to_string(),
            "ClassificationReport" => "Returns <|accuracy, precision, recall, f1Score, support, confusionMatrix|>.".to_string(),
            "RegressionReport" => "Returns <|meanSquaredError, meanAbsoluteError, rootMeanSquaredError, rSquared, sampleCount|>.".to_string(),
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
            // Analytics examples
            "Regression" => vec![
                "data = {{1.0, 2.0}, {2.0, 3.9}, {3.0, 6.1}};".to_string(),
                "Regression[data, \"y ~ x\", \"linear\"]".to_string(),
                "Regression[data, \"y ~ x + x^2\", <|method -> \"polynomial\", degree -> 2|>]".to_string(),
            ],
            // NLP examples
            "TFIDFVectorize" => vec![
                "TFIDFVectorize[{\"doc one\", \"doc two\"}]".to_string(),
            ],
            "WordFrequency" => vec![
                "WordFrequency[\"to be or not to be\"]".to_string(),
            ],
            "SentimentAnalysis" => vec![
                "SentimentAnalysis[\"I love this product!\"]".to_string(),
            ],
            "NamedEntityRecognition" => vec![
                "NamedEntityRecognition[\"Alice went to Paris\"]".to_string(),
            ],
            "POSTagging" => vec![
                "POSTagging[\"Time flies like an arrow\"]".to_string(),
            ],
            "LanguageDetection" => vec![
                "LanguageDetection[\"Bonjour le monde\"]".to_string(),
            ],
            "TextClassification" => vec![
                "TextClassification[\"This is great\", \"rule_based\"]".to_string(),
            ],
            "SpellCheck" => vec![
                "SpellCheck[\"Ths sentnce has erors\"]".to_string(),
            ],
            "TTest" => vec![
                "TTest[{1.1, 0.9, 1.3, 1.0}, <|mu -> 1.0, confidenceLevel -> 0.95|>]".to_string(),
                "TTest[{1.2, 0.8, 1.1}, {0.9, 1.0, 1.05}, <|confidenceLevel -> 0.99|>]".to_string(),
            ],
            // ML evaluation examples
            "CrossValidate" => vec![
                "(* Assume `dataset` is a ForeignDataset *)".to_string(),
                "CrossValidate[dataset, 5, \"Accuracy\", <|builder -> {\"Linear[32]\", \"ReLU\", \"Linear[1]\"}|>]".to_string(),
            ],
            "CrossValidateTable" => vec![
                "CrossValidateTable[table, {\"x1\", \"x2\"}, \"y\", 5, \"MSE\"]".to_string(),
            ],
            "ClassificationReport" => vec![
                "ClassificationReport[{1,0,1,1},{1,0,0,1}]".to_string(),
            ],
            "RegressionReport" => vec![
                "RegressionReport[{2.3, 3.1, 4.7}, {2.2, 3.0, 4.9}]".to_string(),
            ],
            "CorrelationMatrix" => vec![
                "CorrelationMatrix[{{1.0,2.0,3.0},{2.0,3.5,5.0},{3.0,5.1,7.9}}, \"pearson\"]".to_string(),
                "CorrelationMatrix[{{1.0,1.1,0.9},{2.0,1.9,2.1},{3.0,3.1,2.9}}, <|method -> \"spearman\"|>]".to_string(),
            ],
            "ANOVA" => vec![
                "ANOVA[groups, \"y\", {\"factor1\", \"factor2\"}]".to_string(),
            ],
            "ConfidenceInterval" => vec![
                "ConfidenceInterval[{1.1, 0.9, 1.3, 1.0}, <|confidenceLevel -> 0.95, method -> \"t\"|>]".to_string(),
                "ConfidenceInterval[{1.1, 0.9, 1.3, 1.0}, 0.99, \"z\"]".to_string(),
            ],
            "EffectSize" => vec![
                "EffectSize[{1.1, 1.0, 0.9}, {0.8, 1.0, 1.1}, \"cohens_d\"]".to_string(),
                "EffectSize[{1.1, 1.0, 0.9}, {0.8, 1.0, 1.1}, <|method -> \"cohens_d\"|>]".to_string(),
            ],
            "ChiSquareTest" => vec![
                "ChiSquareTest[{10, 20, 30}, {15, 25, 20}]".to_string(),
            ],
            "NormalityTest" => vec![
                "NormalityTest[{1.1, 0.9, 1.0, 1.2}, \"shapiro\"]".to_string(),
                "NormalityTest[{1.1, 0.9, 1.0, 1.2}, \"ks\"]".to_string(),
            ],
            "BootstrapSample" => vec![
                "BootstrapSample[{1.0, 1.1, 0.9, 1.2}, 4, 1000]".to_string(),
            ],
            "PowerAnalysis" => vec![
                "PowerAnalysis[0.5, 0.05, 0.8, \"t-test\"]".to_string(),
            ],
            // Time Series examples
            "TimeSeriesDecompose" => vec![
                "TimeSeriesDecompose[{1,2,3,4,5,6}, \"additive\", 2]".to_string(),
            ],
            "SeasonalDecompose" => vec![
                "SeasonalDecompose[data, 12, \"multiplicative\"]".to_string(),
            ],
            "ARIMAAdvanced" => vec![
                "ARIMAAdvanced[{1.0,1.1,0.9,1.2}, {1,0,1}]".to_string(),
            ],
            "Forecast" => vec![
                "m = ARIMAAdvanced[data, {1,0,1}]; Forecast[m, 5, 0.95]".to_string(),
            ],
            "AutoCorrelation" => vec!["AutoCorrelation[data, 20]".to_string()],
            "PartialAutoCorrelation" => vec!["PartialAutoCorrelation[data, 20]".to_string()],
            "TrendAnalysis" => vec!["TrendAnalysis[data, \"linear\"]".to_string()],
            "ChangePointDetection" => vec!["ChangePointDetection[data, \"cusum\", 1.5]".to_string()],
            "AnomalyDetection" => vec!["AnomalyDetection[data, \"zscore\", 2.0]".to_string()],
            "StationarityTest" => vec!["StationarityTest[data, \"adf\"]".to_string()],
            "CrossCorrelation" => vec!["CrossCorrelation[x, y, 10]".to_string()],
            "SpectralDensity" => vec!["SpectralDensity[data, \"periodogram\"]".to_string()],
            // BI examples
            "KPI" => vec!["KPI[data, \"revenue\", 10000., \"Q1\"]".to_string()],
            "CohortAnalysis" => vec!["CohortAnalysis[events, \"signup_month\", {\"retention\"}]".to_string()],
            "FunnelAnalysis" => vec!["FunnelAnalysis[journeys, {\"Visit\",\"Add to Cart\",\"Purchase\"}, events]".to_string()],
            "RetentionAnalysis" => vec!["RetentionAnalysis[events, \"user_id\", \"date\", {0,30,60,90}]".to_string()],
            "LTV" => vec!["LTV[customers, revenues, 365]".to_string()],
            "Churn" => vec!["Churn[data, {\"engagement\", \"support_tickets\"}, 30]".to_string()],
            "Segmentation" => vec!["Segmentation[data, features, \"kmeans\", 3]".to_string()],
            "ABTestAnalysis" => vec!["ABTestAnalysis[control, treatment, \"mean\", 0.05]".to_string()],
            "AttributionModel" => vec!["AttributionModel[touchpoints, conversions, \"first_touch\"]".to_string()],
            "Dashboard" => vec!["Dashboard[metrics, <||>, {\"table\", \"chart\"}]".to_string()],
            // Signal examples
            "FFT" => vec!["FFT[{1.0,0.0,-1.0,0.0}]".to_string()],
            "RealFFT" => vec!["RealFFT[Table[Sin[2*Pi*10*t], {t,0,1,0.01}]]".to_string()],
            "IFFT" => vec!["IFFT[<|magnitudes -> m, phases -> p|>]".to_string()],
            "Periodogram" => vec!["Periodogram[data]".to_string()],
            "WelchPSD" => vec!["WelchPSD[data, 256, 0.5]".to_string()],
            "FIRFilter" => vec!["FIRFilter[data, coeffs]".to_string()],
            "IIRFilter" => vec!["IIRFilter[data, b, a]".to_string()],
            "LowPassFilter" => vec!["LowPassFilter[signalData, 100.0]".to_string()],
            "HighPassFilter" => vec!["HighPassFilter[signalData, 5.0]".to_string()],
            "BandPassFilter" => vec!["BandPassFilter[signalData, 5.0, 50.0]".to_string()],
            "MedianFilter" => vec!["MedianFilter[{1,100,2,3,2,1}, 3]".to_string()],
            "ApplyWindow" => vec!["ApplyWindow[signalData, HammingWindow[Length[signalData]]]".to_string()],
            // Image examples
            "AffineTransform" => vec!["AffineTransform[img, {{1,0,10},{0,1,5}}]".to_string()],
            "PerspectiveTransform" => vec!["PerspectiveTransform[img, {{1,0,0},{0,1,0},{0.001,0.0005,1}}]".to_string()],
            "ContourDetection" => vec!["ContourDetection[img, 0.5]".to_string()],
            "FeatureDetection" => vec!["FeatureDetection[img, \"harris\", 0.01]".to_string()],
            "TemplateMatching" => vec!["TemplateMatching[img, tpl]".to_string()],
            "ImageSegmentation" => vec!["ImageSegmentation[img, \"Threshold\"]".to_string()],
            // Vision: Edge detection examples
            "CannyEdges" => vec!["CannyEdges[img]".to_string()],
            "SobelEdges" => vec!["SobelEdges[img]".to_string()],
            "LaplacianEdges" => vec!["LaplacianEdges[img]".to_string()],
            "PrewittEdges" => vec!["PrewittEdges[img]".to_string()],
            "RobertsEdges" => vec!["RobertsEdges[img]".to_string()],
            "ScharrEdges" => vec!["ScharrEdges[img]".to_string()],
            // Streaming examples
            "WindowAggregate" => vec![
                "wa = WindowAggregate[\"tumbling\", 60000, \"sum\"]; wa (* then call via method in REPL: wa@process[value, timestamp] *)".to_string(),
            ],
            "StreamJoin" => vec![
                "sj = StreamJoin[\"user_id\", 300000]; sj (* then use sj@processLeft[value, \"user_id\", t] *)".to_string(),
            ],
            "ComplexEventProcessing" => vec![
                "cep = ComplexEventProcessing[]; cep@processEvent[\"Login\", data, t]".to_string(),
            ],
            "BackpressureControl" => vec![
                "bc = BackpressureControl[\"Block\", <|warning -> 0.7, critical -> 0.9, recovery -> 0.5|>]; bc@getMetrics[]".to_string(),
            ],
            // Optimization examples
            "CostAnalysis" => vec!["CostAnalysis[\"res\", \"on_demand\", {\"env\", \"prod\"}]".to_string()],
            "RightSizing" => vec!["RightSizing[{\"svcA\", \"svcB\"}, \"usage_data\", {\"target_utilization\", 70}]".to_string()],
            "CostAlerts" => vec!["CostAlerts[{\"monthly\", 5000}, {\"ops@co\"}, {\"email\"}]".to_string()],
            "BudgetManagement" => vec!["BudgetManagement[{\"marketing\", 10000}, {\"marketing\", \"teamA\"}, {}]".to_string()],
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
            "Regression" => vec![
                ParameterInfo { name: "data".to_string(), type_hint: "Matrix[Real]".to_string(), description: "Rows of observations and variables".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "formula".to_string(), type_hint: "String".to_string(), description: "Model formula (e.g., 'y ~ x + x^2')".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "methodOrOpts".to_string(), type_hint: "String | Assoc".to_string(), description: "Method or options association (method, degree)".to_string(), optional: true, default_value: Some("\"linear\"".to_string()) },
                ParameterInfo { name: "opts".to_string(), type_hint: "Assoc".to_string(), description: "Options association (method, degree)".to_string(), optional: true, default_value: None },
            ],
            "TTest" => vec![
                ParameterInfo { name: "sample1".to_string(), type_hint: "Vector[Real]".to_string(), description: "First sample".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "sample2OrOpts".to_string(), type_hint: "Vector[Real] | Assoc".to_string(), description: "Second sample or options association".to_string(), optional: true, default_value: None },
                ParameterInfo { name: "opts".to_string(), type_hint: "Assoc".to_string(), description: "Options association (mu, confidenceLevel)".to_string(), optional: true, default_value: Some("<|confidenceLevel -> 0.95|>".to_string()) },
            ],
            "CorrelationMatrix" => vec![
                ParameterInfo { name: "data".to_string(), type_hint: "Matrix[Real]".to_string(), description: "Variables as columns or rows".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "methodOrOpts".to_string(), type_hint: "String | Assoc".to_string(), description: "Correlation method or options association (method)".to_string(), optional: true, default_value: Some("\"pearson\"".to_string()) },
            ],
            "ANOVA" => vec![
                ParameterInfo { name: "groups".to_string(), type_hint: "Assoc | Any".to_string(), description: "Grouped data by factors".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "dependent".to_string(), type_hint: "String".to_string(), description: "Dependent variable name".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "factors".to_string(), type_hint: "List[String]".to_string(), description: "Factor names".to_string(), optional: false, default_value: None },
            ],
            "ConfidenceInterval" => vec![
                ParameterInfo { name: "data".to_string(), type_hint: "Vector[Real]".to_string(), description: "Sample data".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "opts".to_string(), type_hint: "Assoc | Number | String".to_string(), description: "Options or legacy positional args (confidenceLevel, method)".to_string(), optional: true, default_value: Some("<|confidenceLevel -> 0.95, method -> \"t\"|>".to_string()) },
            ],
            "EffectSize" => vec![
                ParameterInfo { name: "group1".to_string(), type_hint: "Vector[Real]".to_string(), description: "First group".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "group2".to_string(), type_hint: "Vector[Real]".to_string(), description: "Second group".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "methodOrOpts".to_string(), type_hint: "String | Assoc".to_string(), description: "Method or options association (method)".to_string(), optional: true, default_value: Some("\"cohens_d\"".to_string()) },
            ],
            "ChiSquareTest" => vec![
                ParameterInfo { name: "observed".to_string(), type_hint: "Vector[Real]".to_string(), description: "Observed frequencies".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "expected".to_string(), type_hint: "Vector[Real]".to_string(), description: "Expected frequencies".to_string(), optional: false, default_value: None },
            ],
            // Time Series
            "TimeSeriesDecompose" => vec![
                ParameterInfo { name: "series".to_string(), type_hint: "Vector[Real]".to_string(), description: "Time series data".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "model".to_string(), type_hint: "String".to_string(), description: "'additive' or 'multiplicative'".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "period".to_string(), type_hint: "Integer".to_string(), description: "Seasonal period".to_string(), optional: false, default_value: None },
            ],
            "SeasonalDecompose" => vec![
                ParameterInfo { name: "series".to_string(), type_hint: "Vector[Real]".to_string(), description: "Time series data".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "period".to_string(), type_hint: "Integer".to_string(), description: "Seasonal period".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "method".to_string(), type_hint: "String".to_string(), description: "'additive' or 'multiplicative'".to_string(), optional: true, default_value: Some("\"additive\"".to_string()) },
            ],
            "ARIMAAdvanced" | "ARIMA" => vec![
                ParameterInfo { name: "series".to_string(), type_hint: "Vector[Real]".to_string(), description: "Time series data".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "order".to_string(), type_hint: "List[Int]".to_string(), description: "{p, d, q}".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "seasonalOrder".to_string(), type_hint: "List[Int]".to_string(), description: "{P, D, Q, s}".to_string(), optional: true, default_value: None },
            ],
            "Forecast" => vec![
                ParameterInfo { name: "model".to_string(), type_hint: "Association".to_string(), description: "Model association with fittedValues".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "periods".to_string(), type_hint: "Integer".to_string(), description: "Forecast horizon".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "confidenceLevel".to_string(), type_hint: "Real".to_string(), description: "Confidence level (e.g., 0.95)".to_string(), optional: true, default_value: Some("0.95".to_string()) },
            ],
            "AutoCorrelation" | "PartialAutoCorrelation" => vec![
                ParameterInfo { name: "series".to_string(), type_hint: "Vector[Real]".to_string(), description: "Time series data".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "lags".to_string(), type_hint: "Integer".to_string(), description: "Maximum lags".to_string(), optional: false, default_value: None },
            ],
            "TrendAnalysis" => vec![
                ParameterInfo { name: "series".to_string(), type_hint: "Vector[Real]".to_string(), description: "Time series data".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "method".to_string(), type_hint: "String".to_string(), description: "'linear' | 'polynomial' | 'moving_average'".to_string(), optional: true, default_value: Some("\"linear\"".to_string()) },
            ],
            "ChangePointDetection" => vec![
                ParameterInfo { name: "series".to_string(), type_hint: "Vector[Real]".to_string(), description: "Time series data".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "method".to_string(), type_hint: "String".to_string(), description: "'cusum' | 'variance'".to_string(), optional: true, default_value: Some("\"cusum\"".to_string()) },
                ParameterInfo { name: "sensitivity".to_string(), type_hint: "Real".to_string(), description: "Threshold/sensitivity parameter".to_string(), optional: true, default_value: Some("1.0".to_string()) },
            ],
            "AnomalyDetection" => vec![
                ParameterInfo { name: "series".to_string(), type_hint: "Vector[Real]".to_string(), description: "Time series data".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "method".to_string(), type_hint: "String".to_string(), description: "'zscore'".to_string(), optional: true, default_value: Some("\"zscore\"".to_string()) },
                ParameterInfo { name: "threshold".to_string(), type_hint: "Real".to_string(), description: "Anomaly threshold".to_string(), optional: true, default_value: Some("2.0".to_string()) },
            ],
            "StationarityTest" => vec![
                ParameterInfo { name: "series".to_string(), type_hint: "Vector[Real]".to_string(), description: "Time series data".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "testType".to_string(), type_hint: "String".to_string(), description: "'adf' | 'kpss'".to_string(), optional: true, default_value: Some("\"adf\"".to_string()) },
            ],
            "CrossCorrelation" => vec![
                ParameterInfo { name: "series1".to_string(), type_hint: "Vector[Real]".to_string(), description: "First series".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "series2".to_string(), type_hint: "Vector[Real]".to_string(), description: "Second series".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "lags".to_string(), type_hint: "Integer".to_string(), description: "Maximum lags".to_string(), optional: false, default_value: None },
            ],
            "SpectralDensity" => vec![
                ParameterInfo { name: "series".to_string(), type_hint: "Vector[Real]".to_string(), description: "Time series data".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "method".to_string(), type_hint: "String".to_string(), description: "'periodogram' | 'welch'".to_string(), optional: true, default_value: Some("\"periodogram\"".to_string()) },
            ],
            // Business Intelligence
            "KPI" => vec![
                ParameterInfo { name: "data".to_string(), type_hint: "Assoc | Any".to_string(), description: "Metric inputs (time series or aggregates)".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "metricDefinition".to_string(), type_hint: "String".to_string(), description: "e.g., 'revenue', 'conversion_rate'".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "target".to_string(), type_hint: "Real".to_string(), description: "Target value".to_string(), optional: true, default_value: None },
                ParameterInfo { name: "period".to_string(), type_hint: "String".to_string(), description: "Period label (e.g., 'Q1')".to_string(), optional: false, default_value: None },
            ],
            "CohortAnalysis" => vec![
                ParameterInfo { name: "data".to_string(), type_hint: "List[Assoc] | Any".to_string(), description: "Event or user dataset".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "cohortCriteria".to_string(), type_hint: "String".to_string(), description: "Grouping key, e.g., 'signup_month'".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "metrics".to_string(), type_hint: "List[String]".to_string(), description: "Metrics to compute (e.g., 'retention')".to_string(), optional: false, default_value: None },
            ],
            "FunnelAnalysis" => vec![
                ParameterInfo { name: "data".to_string(), type_hint: "List[Assoc] | Any".to_string(), description: "User journey events".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "stages".to_string(), type_hint: "List[String]".to_string(), description: "Funnel stages".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "conversionEvents".to_string(), type_hint: "List[String]".to_string(), description: "Events indicating conversion between stages".to_string(), optional: false, default_value: None },
            ],
            "RetentionAnalysis" => vec![
                ParameterInfo { name: "data".to_string(), type_hint: "List[Assoc] | Any".to_string(), description: "Activity events".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "userIdCol".to_string(), type_hint: "String".to_string(), description: "User ID column/key".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "eventDateCol".to_string(), type_hint: "String".to_string(), description: "Event date column/key".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "periods".to_string(), type_hint: "List[Real]".to_string(), description: "Periods offsets".to_string(), optional: false, default_value: None },
            ],
            "LTV" => vec![
                ParameterInfo { name: "customerData".to_string(), type_hint: "List[Assoc] | Any".to_string(), description: "Customer features".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "revenueEvents".to_string(), type_hint: "List[Assoc] | Any".to_string(), description: "Revenue events".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "timeHorizon".to_string(), type_hint: "Real".to_string(), description: "Days to project".to_string(), optional: false, default_value: None },
            ],
            "Churn" => vec![
                ParameterInfo { name: "data".to_string(), type_hint: "List[Assoc] | Any".to_string(), description: "Customer dataset".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "features".to_string(), type_hint: "List[String]".to_string(), description: "Feature names".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "predictionHorizon".to_string(), type_hint: "Real".to_string(), description: "Days ahead".to_string(), optional: true, default_value: Some("30".to_string()) },
            ],
            "Segmentation" => vec![
                ParameterInfo { name: "data".to_string(), type_hint: "List[Assoc] | Any".to_string(), description: "Customer dataset".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "features".to_string(), type_hint: "List[String]".to_string(), description: "Features to segment on".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "method".to_string(), type_hint: "String".to_string(), description: "'kmeans' | 'rfm'".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "k".to_string(), type_hint: "Integer".to_string(), description: "Clusters for kmeans".to_string(), optional: true, default_value: Some("3".to_string()) },
            ],
            "ABTestAnalysis" => vec![
                ParameterInfo { name: "control".to_string(), type_hint: "Vector[Real]".to_string(), description: "Control sample".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "treatment".to_string(), type_hint: "Vector[Real]".to_string(), description: "Treatment sample".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "metric".to_string(), type_hint: "String".to_string(), description: "Metric, e.g., 'mean'".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "alpha".to_string(), type_hint: "Real".to_string(), description: "Significance level".to_string(), optional: true, default_value: Some("0.05".to_string()) },
            ],
            "AttributionModel" => vec![
                ParameterInfo { name: "touchpoints".to_string(), type_hint: "List[Assoc] | Any".to_string(), description: "Marketing touchpoints".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "conversions".to_string(), type_hint: "List[Assoc] | Any".to_string(), description: "Conversion events".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "modelType".to_string(), type_hint: "String".to_string(), description: "'first_touch' | 'last_touch' | 'linear'".to_string(), optional: false, default_value: None },
            ],
            "Dashboard" => vec![
                ParameterInfo { name: "metrics".to_string(), type_hint: "Any".to_string(), description: "Metrics specification".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "filters".to_string(), type_hint: "Assoc".to_string(), description: "Filters association".to_string(), optional: true, default_value: Some("<||>".to_string()) },
                ParameterInfo { name: "visualizations".to_string(), type_hint: "List[String]".to_string(), description: "Visualization types".to_string(), optional: true, default_value: Some("{\"table\"}".to_string()) },
            ],
            // Signal parameter hints
            "FFT" | "RealFFT" | "Periodogram" => vec![
                ParameterInfo { name: "signal".to_string(), type_hint: "List[Real] | SignalData".to_string(), description: "Input signal".to_string(), optional: false, default_value: None },
            ],
            "IFFT" => vec![
                ParameterInfo { name: "spectrum".to_string(), type_hint: "Assoc | SpectralResult".to_string(), description: "Spectral data with magnitudes, phases".to_string(), optional: false, default_value: None },
            ],
            "WelchPSD" => vec![
                ParameterInfo { name: "signal".to_string(), type_hint: "List[Real]".to_string(), description: "Input signal".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "segmentLength".to_string(), type_hint: "Integer".to_string(), description: "Segment/window length".to_string(), optional: true, default_value: Some("256".to_string()) },
                ParameterInfo { name: "overlap".to_string(), type_hint: "Real".to_string(), description: "Overlap fraction 0..1".to_string(), optional: true, default_value: Some("0.5".to_string()) },
            ],
            "FIRFilter" => vec![
                ParameterInfo { name: "signal".to_string(), type_hint: "List[Real]".to_string(), description: "Input signal".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "coefficients".to_string(), type_hint: "List[Real]".to_string(), description: "FIR coefficients".to_string(), optional: false, default_value: None },
            ],
            "IIRFilter" => vec![
                ParameterInfo { name: "signal".to_string(), type_hint: "List[Real]".to_string(), description: "Input signal".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "bCoeffs".to_string(), type_hint: "List[Real]".to_string(), description: "Numerator coefficients".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "aCoeffs".to_string(), type_hint: "List[Real]".to_string(), description: "Denominator coefficients".to_string(), optional: false, default_value: None },
            ],
            "LowPassFilter" | "HighPassFilter" => vec![
                ParameterInfo { name: "signalData".to_string(), type_hint: "SignalData".to_string(), description: "Signal data object".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "cutoff".to_string(), type_hint: "Real".to_string(), description: "Cutoff frequency (Hz)".to_string(), optional: false, default_value: None },
            ],
            "BandPassFilter" => vec![
                ParameterInfo { name: "signalData".to_string(), type_hint: "SignalData".to_string(), description: "Signal data object".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "low".to_string(), type_hint: "Real".to_string(), description: "Low cutoff (Hz)".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "high".to_string(), type_hint: "Real".to_string(), description: "High cutoff (Hz)".to_string(), optional: false, default_value: None },
            ],
            "MedianFilter" => vec![
                ParameterInfo { name: "signal".to_string(), type_hint: "List[Real]".to_string(), description: "Input signal".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "window".to_string(), type_hint: "Integer".to_string(), description: "Window size".to_string(), optional: false, default_value: None },
            ],
            "ApplyWindow" => vec![
                ParameterInfo { name: "signalData".to_string(), type_hint: "SignalData".to_string(), description: "Signal data object".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "window".to_string(), type_hint: "List[Real]".to_string(), description: "Window coefficients".to_string(), optional: false, default_value: None },
            ],
            // Image parameters
            "AffineTransform" => vec![
                ParameterInfo { name: "image".to_string(), type_hint: "Image".to_string(), description: "Input image (Foreign)".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "transformMatrix".to_string(), type_hint: "{{a,b,tx},{c,d,ty}}".to_string(), description: "2x3 affine matrix".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "interpolation".to_string(), type_hint: "String".to_string(), description: "NearestNeighbor | Bilinear | Bicubic | Lanczos".to_string(), optional: true, default_value: Some("Bilinear".to_string()) },
            ],
            "PerspectiveTransform" => vec![
                ParameterInfo { name: "image".to_string(), type_hint: "Image".to_string(), description: "Input image (Foreign)".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "transformMatrix".to_string(), type_hint: "{{...},{...},{...}}".to_string(), description: "3x3 perspective matrix".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "interpolation".to_string(), type_hint: "String".to_string(), description: "NearestNeighbor | Bilinear | Bicubic | Lanczos".to_string(), optional: true, default_value: Some("Bilinear".to_string()) },
            ],
            "ContourDetection" => vec![
                ParameterInfo { name: "image".to_string(), type_hint: "Image".to_string(), description: "Input image (Foreign)".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "threshold".to_string(), type_hint: "Real".to_string(), description: "Binarization threshold".to_string(), optional: true, default_value: Some("0.5".to_string()) },
                ParameterInfo { name: "method".to_string(), type_hint: "String".to_string(), description: "Method (reserved)".to_string(), optional: true, default_value: None },
            ],
            "FeatureDetection" => vec![
                ParameterInfo { name: "image".to_string(), type_hint: "Image".to_string(), description: "Input image (Foreign)".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "method".to_string(), type_hint: "String".to_string(), description: "Detector (e.g., 'harris')".to_string(), optional: true, default_value: None },
                ParameterInfo { name: "threshold".to_string(), type_hint: "Real".to_string(), description: "Response threshold".to_string(), optional: true, default_value: Some("0.01".to_string()) },
            ],
            "TemplateMatching" => vec![
                ParameterInfo { name: "image".to_string(), type_hint: "Image".to_string(), description: "Input image".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "template".to_string(), type_hint: "Image".to_string(), description: "Template image".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "method".to_string(), type_hint: "String".to_string(), description: "Matching method (reserved)".to_string(), optional: true, default_value: None },
            ],
            "ImageSegmentation" => vec![
                ParameterInfo { name: "image".to_string(), type_hint: "Image".to_string(), description: "Input image".to_string(), optional: false, default_value: None },
                ParameterInfo { name: "method".to_string(), type_hint: "String".to_string(), description: "'Threshold' | 'Watershed'".to_string(), optional: true, default_value: Some("Threshold".to_string()) },
                ParameterInfo { name: "parameters".to_string(), type_hint: "Assoc".to_string(), description: "Algorithm parameters".to_string(), optional: true, default_value: None },
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
            // Analytics
            "Regression" | "TTest" | "CorrelationMatrix" | "ANOVA" | "ConfidenceInterval" | "EffectSize" | "ChiSquareTest" | "NormalityTest" | "BootstrapSample" | "PowerAnalysis" => "Association".to_string(),
            // NLP
            "TFIDFVectorize" | "WordFrequency" | "SentimentAnalysis" | "NamedEntityRecognition" | "POSTagging" | "TextClassification" | "RuleBasedSentiment" | "StatisticalSentiment" | "EmotionDetection" => "Association".to_string(),
            "LanguageDetection" => "String".to_string(),
            // Analytics: Timeseries & BI
            "TimeSeriesDecompose" | "SeasonalDecompose" | "ARIMAAdvanced" | "ARIMA" | "Forecast" | "StationarityTest" | "SpectralDensity" | "KPI" | "CohortAnalysis" | "FunnelAnalysis" | "LTV" | "Churn" | "Segmentation" | "ABTestAnalysis" | "AttributionModel" | "Dashboard" => "Association".to_string(),
            // Number Theory and Time Series (normalized Associations)
            "HashFunction" => "Association".to_string(),
            "DiscreteLog" => "Association".to_string(),
            "MovingAverage" => "Association".to_string(),
            "ExponentialMovingAverage" => "Association".to_string(),
            // Signal return types
            "FFT" | "RealFFT" | "Periodogram" | "WelchPSD" | "FIRFilter" | "IIRFilter" | "LowPassFilter" | "HighPassFilter" | "BandPassFilter" | "MedianFilter" => "Association".to_string(),
            "IFFT" | "ApplyWindow" => "SignalData (Foreign)".to_string(),
            // Image return types
            "AffineTransform" | "PerspectiveTransform" | "TemplateMatching" => "Image (Foreign)".to_string(),
            "ContourDetection" | "FeatureDetection" => "List[Association]".to_string(),
            "ImageSegmentation" => "Association".to_string(),
            // Streaming constructors (return Foreign objects)
            "WindowAggregate" | "StreamJoin" | "ComplexEventProcessing" | "BackpressureControl" => "Foreign Object".to_string(),
            // Optimization returns Associations
            "CostAnalysis" | "RightSizing" | "CostAlerts" | "BudgetManagement" => "Association".to_string(),
            // AI/RAG/Vector returns
            "VectorSearch" => "List[Association]".to_string(),
            "VectorCluster" => "Association".to_string(),
            "EmbeddingCluster" => "Association".to_string(),
            "ContextRetrieval" => "List[Association]".to_string(),
            "RAGQuery" => "Association".to_string(),
            // ML quick starts and evaluation
            "AutoMLQuickStart" | "AutoMLQuickStartTable" | "CrossValidate" | "CrossValidateTable" => "Association".to_string(),
            // Numerical: return types
            "FiniteDifference" | "RichardsonExtrapolation" => "Association".to_string(),
            "Bisection" | "NewtonRaphson" | "Secant" | "Brent" | "FixedPoint" => "Association".to_string(),
            "Trapezoidal" | "Simpson" | "Romberg" | "GaussQuadrature" | "MonteCarloIntegrate" => "Association".to_string(),
            // Graph
            "AdvancedPageRank" | "LouvainCommunityDetection" | "BetweennessCentrality" | "ClosenessCentrality" => "Association".to_string(),
            // Vision: Edge detection
            "CannyEdges" | "SobelEdges" | "LaplacianEdges" | "PrewittEdges" | "RobertsEdges" | "ScharrEdges" => "Association".to_string(),
            "AutoCorrelation" | "PartialAutoCorrelation" | "CrossCorrelation" | "TrendAnalysis" | "RetentionAnalysis" | "ChangePointDetection" => "List".to_string(),
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
    pub fn get_smart_suggestions(&self, _context: &str) -> Vec<String> {
        // This could analyze the input context and suggest commonly used functions
        // For now, return most frequently used functions
        self.get_usage_stats()
            .into_iter()
            .take(5)
            .map(|(name, count)| format!("{} (used {} times)", name, count))
            .collect()
    }

    /// Return a programmatic description as an Association for Describe["Function"]
    pub fn describe_association(&self, function_name: &str) -> ReplResult<Value> {
        let info = if let Some(i) = self.function_database.get_function(function_name) {
            i.clone()
        } else if let Some(real) = self.aliases.get(&function_name.to_lowercase()) {
            self.function_database
                .get_function(real)
                .cloned()
                .ok_or_else(|| ReplError::Other { message: format!("Function '{}' not found", function_name) })?
        } else {
            return Err(ReplError::Other { message: format!("Function '{}' not found", function_name) });
        };

        // Build association
        let mut m = std::collections::HashMap::new();
        m.insert("name".to_string(), Value::String(info.name));
        m.insert("signature".to_string(), Value::String(info.signature));
        m.insert("summary".to_string(), Value::String(info.description));
        m.insert("module".to_string(), Value::String(info.module));
        m.insert("category".to_string(), Value::String(info.category));

        // parameters as list of associations
        let params: Vec<Value> = info.parameters.into_iter().map(|p| {
            let mut pm = std::collections::HashMap::new();
            pm.insert("name".to_string(), Value::String(p.name));
            pm.insert("type".to_string(), Value::String(p.type_hint));
            pm.insert("description".to_string(), Value::String(p.description));
            pm.insert("optional".to_string(), Value::Boolean(p.optional));
            if let Some(def) = p.default_value { pm.insert("default".to_string(), Value::String(def)); }
            Value::Object(pm)
        }).collect();
        m.insert("parameters".to_string(), Value::List(params));
        m.insert("returnType".to_string(), Value::String(info.return_type));

        // examples as list of strings
        m.insert("examples".to_string(), Value::List(info.examples.into_iter().map(Value::String).collect()));

        Ok(Value::Object(m))
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
