//! Standard Library for Lyra
//!
//! This module provides the core standard library functions that are built into
//! the Lyra symbolic computation engine. Functions are organized by category:
//! - List operations
//! - String operations  
//! - Math functions
//! - Pattern matching and rules
//! - Graph theory and network analysis

use crate::vm::{Value, VmResult};
use std::collections::HashMap;

pub mod async_ops;  // Re-enabling for Week 8 Day 5
pub mod autodiff;
pub mod secure_wrapper;
pub mod clustering;
pub mod data;
pub mod finance;
pub mod graph;
pub mod image;
pub mod vision;
pub mod io;
pub mod list;
pub mod mathematics;
pub mod ml;
pub mod result;
pub mod rules;
pub mod sparse;
pub mod spatial;
pub mod analytics;
// pub mod statistics; // CONSOLIDATED into analytics/statistics.rs
pub mod string;
pub mod table;
pub mod tensor;
pub mod timeseries;
pub mod numerical;
pub mod network;
pub mod number_theory;
pub mod combinatorics;
pub mod geometry;
pub mod topology;
pub mod data_processing;
pub mod game_theory;
pub mod temporal;
pub mod developer_tools;
pub mod system;
pub mod collections;
pub mod bioinformatics;
pub mod signal;
pub mod quantum;
pub mod nlp;
pub mod language;
// Phase A lightweight utility modules (avoid pulling full utilities tree)
#[path = "utilities/serialization.rs"]
pub mod util_serialization;
#[path = "utilities/config.rs"]
pub mod util_config;
#[path = "utilities/cache.rs"]
pub mod util_cache;

// New infrastructure and algorithm modules
pub mod common;
pub mod algorithms;
pub mod data_structures;

// Alias: prefer analytics::timeseries as consolidated entry under std::analytics
pub use analytics::timeseries as timeseries_analytics;

/// Standard library function signature
pub type StdlibFunction = fn(&[Value]) -> VmResult<Value>;

/// Registry of all standard library functions
#[derive(Debug)]
pub struct StandardLibrary {
    functions: HashMap<String, StdlibFunction>,
}

impl StandardLibrary {
    /// Create a new standard library with all built-in functions registered
    pub fn new() -> Self {
        let mut stdlib = StandardLibrary {
            functions: HashMap::new(),
        };

        // Register all function categories
        stdlib.register_list_functions();
        stdlib.register_string_functions();
        // Consolidated mathematics registration via per-module registry
        stdlib.register_mathematics_group();
        // Analytics statistics remain consolidated under analytics::statistics
        stdlib.register_statistics_functions();
        stdlib.register_rule_functions();
        stdlib.register_table_functions();
        stdlib.register_tensor_functions();
        stdlib.register_ml_functions();
        stdlib.register_io_functions();
        // Consolidated mathematics subdomains handled by register_mathematics_group()
        stdlib.register_graph_functions();
        stdlib.register_signal_functions();
        // Image consolidated under image::register_image_functions()
        stdlib.register_image_functions();
        stdlib.register_vision_functions();
        // Timeseries consolidated under analytics::timeseries registry
        stdlib.register_timeseries_functions();
        stdlib.register_clustering_functions();
        stdlib.register_numerical_functions();
        stdlib.register_sparse_functions();
        stdlib.register_spatial_functions();
        stdlib.register_result_functions();
        stdlib.register_async_functions();
        stdlib.register_network_functions();
        stdlib.register_number_theory_functions();
        stdlib.register_combinatorics_functions();
        stdlib.register_geometry_functions();
        stdlib.register_topology_functions();
        stdlib.register_data_processing_functions();
        stdlib.register_game_theory_functions();
        stdlib.register_temporal_functions();
        stdlib.register_developer_tools_functions();
        stdlib.register_system_functions();
        stdlib.register_collections_functions();
        stdlib.register_bioinformatics_functions();
        stdlib.register_finance_functions();
        stdlib.register_quantum_functions();
        stdlib.register_nlp_functions();
        stdlib.register_algorithm_functions();
        stdlib.register_data_structure_functions();
        stdlib.register_language_functions();
        // Phase A additions
        stdlib.register_serialization_functions();
        stdlib.register_config_functions();
        stdlib.register_cache_functions();
        stdlib.register_object_store_functions();

        stdlib
    }

    /// Look up a function by name
    pub fn get_function(&self, name: &str) -> Option<StdlibFunction> {
        self.functions.get(name).copied()
    }

    /// Register a function with the given name
    pub fn register(&mut self, name: impl Into<String>, func: StdlibFunction) {
        self.functions.insert(name.into(), func);
    }

    /// Get all registered function names in deterministic sorted order
    pub fn function_names(&self) -> Vec<&String> {
        let mut names: Vec<&String> = self.functions.keys().collect();
        names.sort();
        names
    }

    // Registration functions for each category
    fn register_list_functions(&mut self) {
        self.register("Length", list::length);
        self.register("Head", list::head);
        self.register("Tail", list::tail);
        self.register("Append", list::append);
        self.register("Flatten", list::flatten);
        self.register("Map", list::map);
        self.register("Apply", list::apply);
    }

    fn register_string_functions(&mut self) {
        // Basic string operations
        self.register("StringJoin", string::basic::string_join);
        self.register("StringLength", string::basic::string_length);
        self.register("StringTake", string::basic::string_take);
        self.register("StringDrop", string::basic::string_drop);
        
        // Advanced string operations
        // String template and regex operations
        self.register("StringTemplate", string::advanced::string_template);
        self.register("RegularExpression", string::advanced::regular_expression);
        self.register("StringMatch", string::advanced::string_match);
        self.register("StringExtract", string::advanced::string_extract);
        self.register("StringReplace", string::advanced::string_replace);
        
        // Core string utilities
        self.register("StringSplit", string::advanced::string_split);
        self.register("StringTrim", string::advanced::string_trim);
        self.register("StringContains", string::advanced::string_contains);
        self.register("StringStartsWith", string::advanced::string_starts_with);
        self.register("StringEndsWith", string::advanced::string_ends_with);
        self.register("StringReverse", string::advanced::string_reverse);
        self.register("StringRepeat", string::advanced::string_repeat);
        
        // Case operations
        self.register("ToUpperCase", string::advanced::to_upper_case);
        self.register("ToLowerCase", string::advanced::to_lower_case);
        self.register("TitleCase", string::advanced::title_case);
        self.register("CamelCase", string::advanced::camel_case);
        self.register("SnakeCase", string::advanced::snake_case);
        
        // Encoding/Decoding functions
        self.register("Base64Encode", string::advanced::base64_encode);
        self.register("Base64Decode", string::advanced::base64_decode);
        self.register("URLEncode", string::advanced::url_encode);
        self.register("URLDecode", string::advanced::url_decode);
        self.register("HTMLEscape", string::advanced::html_escape);
        self.register("HTMLUnescape", string::advanced::html_unescape);
        self.register("JSONEscape", string::advanced::json_escape);
        
        // String formatting
        self.register("StringFormat", string::advanced::string_format);

        // Basic output
        self.register("Print", Self::print_value);
    }

    /// Print[arg1, arg2, ...] → prints each argument; returns Missing
    fn print_value(args: &[crate::vm::Value]) -> crate::vm::VmResult<crate::vm::Value> {
        use crate::vm::Value;
        fn fmt(v: &Value) -> String {
            match v {
                Value::Integer(n) => n.to_string(),
                Value::Real(f) => if f.fract() == 0.0 { format!("{:.1}", f) } else { f.to_string() },
                Value::String(s) | Value::Symbol(s) => s.clone(),
                Value::Boolean(b) => if *b { "True".into() } else { "False".into() },
                Value::List(items) => {
                    let inner: Vec<String> = items.iter().map(fmt).collect();
                    format!("{{{}}}", inner.join(", "))
                }
                Value::Object(_) => "Object[]".into(),
                Value::LyObj(o) => format!("{}[...]", o.type_name()),
                Value::Function(name) => format!("Function[{}]", name),
                Value::Quote(_) => "Hold[...]".into(),
                Value::Pattern(p) => format!("{}", p),
                Value::Rule { lhs, rhs } => format!("{} -> {}", fmt(lhs), fmt(rhs)),
                Value::PureFunction { .. } => "PureFunction[...]".into(),
                Value::Slot { .. } => "Slot[...]".into(),
                Value::Missing => "Missing[]".into(),
            }
        }
        for a in args { println!("{}", fmt(a)); }
        Ok(Value::Missing)
    }

    /// Consolidated mathematics registration via per-module registry
    fn register_mathematics_group(&mut self) {
        let functions = mathematics::register_mathematics_functions();
        for (name, function) in functions {
            self.register(name, function);
        }
    }


    fn register_math_functions(&mut self) {
        // Basic arithmetic functions (for Listable attribute support)
        self.register("Plus", mathematics::basic::plus);
        self.register("Times", mathematics::basic::times);
        self.register("Divide", mathematics::basic::divide);
        self.register("Power", mathematics::basic::power);
        self.register("Minus", mathematics::basic::minus);
        
        // Trigonometric and other math functions
        self.register("Sin", mathematics::basic::sin);
        self.register("Cos", mathematics::basic::cos);
        self.register("Tan", mathematics::basic::tan);
        self.register("Exp", mathematics::basic::exp);
        self.register("Log", mathematics::basic::log);
        self.register("Sqrt", mathematics::basic::sqrt);
        
        // Test functions for Hold attribute support
        self.register("TestHold", mathematics::basic::test_hold);
        self.register("TestHoldMultiple", mathematics::basic::test_hold_multiple);
    }

    fn register_calculus_functions(&mut self) {
        // Symbolic differentiation
        self.register("D", mathematics::calculus::d);
        
        // Symbolic integration
        self.register("Integrate", mathematics::calculus::integrate);
        self.register("IntegrateDefinite", mathematics::calculus::integrate_definite);
    }

    fn register_statistics_functions(&mut self) {
        // Basic Descriptive statistics (consolidated into analytics/statistics.rs)
        self.register("Mean", analytics::statistics::mean);
        self.register("Variance", analytics::statistics::variance);
        self.register("StandardDeviation", analytics::statistics::standard_deviation);
        self.register("Median", analytics::statistics::median);
        self.register("Mode", analytics::statistics::mode);
        self.register("Quantile", analytics::statistics::quantile);
        
        // Min/Max and aggregation (consolidated into analytics/statistics.rs)
        self.register("Min", analytics::statistics::min);
        self.register("Max", analytics::statistics::max);
        self.register("Total", analytics::statistics::total);
        
        // Random number generation (consolidated into analytics/statistics.rs)
        self.register("RandomReal", analytics::statistics::random_real);
        self.register("RandomInteger", analytics::statistics::random_integer);
        
        // Correlation and covariance (consolidated into analytics/statistics.rs)
        self.register("Correlation", analytics::statistics::correlation);
        self.register("Covariance", analytics::statistics::covariance);
        
        // Advanced Statistical Analysis Functions
        self.register("Regression", analytics::statistics::regression);
        self.register("ANOVA", analytics::statistics::anova);
        self.register("TTest", analytics::statistics::t_test);
        self.register("ChiSquareTest", analytics::statistics::chi_square_test);
        self.register("CorrelationMatrix", analytics::statistics::correlation_matrix);
        self.register("PCA", analytics::statistics::pca);
        self.register("HypothesisTest", analytics::statistics::hypothesis_test);
        self.register("ConfidenceInterval", analytics::statistics::confidence_interval);
        self.register("BootstrapSample", analytics::statistics::bootstrap_sample);
        self.register("StatisticalSummary", analytics::statistics::statistical_summary);
        self.register("OutlierDetection", analytics::statistics::outlier_detection);
        self.register("NormalityTest", analytics::statistics::normality_test);
        self.register("PowerAnalysis", analytics::statistics::power_analysis);
        self.register("EffectSize", analytics::statistics::effect_size);
        self.register("MultipleComparison", analytics::statistics::multiple_comparison);
    }

    fn register_rule_functions(&mut self) {
        self.register("MatchQ", rules::match_q);
        self.register("Cases", rules::cases);
        self.register("CountPattern", rules::count_pattern);
        self.register("Position", rules::position);
        self.register("ReplaceAll", rules::replace_all);
        self.register("ReplaceRepeated", rules::replace_repeated);
        self.register("Rule", rules::rule);
        self.register("RuleDelayed", rules::rule_delayed);
    }

    fn register_table_functions(&mut self) {
        // Consolidated registration from table module
        let fns = crate::stdlib::table::register_table_functions();
        for (n, f) in fns { self.register(n, f); }
    }

    fn register_tensor_functions(&mut self) {
        // Basic tensor operations
        self.register("Array", tensor::array);
        self.register("ArrayDimensions", tensor::array_dimensions);
        self.register("ArrayRank", tensor::array_rank);
        self.register("ArrayReshape", tensor::array_reshape);
        self.register("ArrayFlatten", tensor::array_flatten);
        
        // Linear algebra operations
        self.register("Dot", tensor::dot);
        self.register("Transpose", tensor::transpose);
        self.register("Maximum", tensor::maximum);
        
        // Neural network activation functions
        self.register("Sigmoid", tensor::sigmoid);
        self.register("Tanh", tensor::tanh);
        self.register("Softmax", tensor::softmax);
    }

    fn register_ml_functions(&mut self) {
        let fns = crate::stdlib::ml::register_ml_functions();
        for (n, f) in fns { self.register(n, f); }
    }

    fn register_io_functions(&mut self) {
        // Import/Export operations
        self.register("Import", io::import);
        self.register("Export", io::export);
        
        // File operations
        self.register("FileRead", io::file_read);
        self.register("FileReadLines", io::file_read_lines);
        self.register("FileWrite", io::file_write);
        self.register("FileAppend", io::file_append);
        self.register("FileExists", io::file_exists);
        self.register("FileSize", io::file_size);
        self.register("FileDelete", io::file_delete);
        self.register("FileCopy", io::file_copy);
        
        // Directory operations
        self.register("DirectoryCreate", io::directory_create);
        self.register("DirectoryDelete", io::directory_delete);
        self.register("DirectoryExists", io::directory_exists);
        self.register("DirectoryList", io::directory_list);
        self.register("DirectorySize", io::directory_size);
        self.register("DirectoryWatch", io::directory_watch);
        
        // Path operations
        self.register("PathJoin", io::path_join);
        self.register("PathSplit", io::path_split);
        self.register("PathParent", io::path_parent);
        self.register("PathFilename", io::path_filename);
        self.register("PathExtension", io::path_extension);
        self.register("PathAbsolute", io::path_absolute);
    }

    fn register_linalg_functions(&mut self) {
        // Matrix decompositions
        self.register("SVD", mathematics::linear_algebra::svd);
        self.register("QRDecomposition", mathematics::linear_algebra::qr_decomposition);
        self.register("LUDecomposition", mathematics::linear_algebra::lu_decomposition);
        self.register("CholeskyDecomposition", mathematics::linear_algebra::cholesky_decomposition);
        
        // Eigenvalue computations
        self.register("EigenDecomposition", mathematics::linear_algebra::eigen_decomposition);
        self.register("SchurDecomposition", mathematics::linear_algebra::schur_decomposition);
        
        // Linear systems
        self.register("LinearSolve", mathematics::linear_algebra::linear_solve);
        self.register("LeastSquares", mathematics::linear_algebra::least_squares);
        self.register("PseudoInverse", mathematics::linear_algebra::pseudo_inverse);
        
        // Matrix functions
        self.register("MatrixPower", mathematics::linear_algebra::matrix_power);
        self.register("MatrixFunction", mathematics::linear_algebra::matrix_function);
        self.register("MatrixTrace", mathematics::linear_algebra::matrix_trace);
        
        // Matrix analysis
        self.register("MatrixRank", mathematics::linear_algebra::matrix_rank);
        self.register("MatrixCondition", mathematics::linear_algebra::matrix_condition);
        self.register("MatrixNorm", mathematics::linear_algebra::matrix_norm);
        self.register("Determinant", mathematics::linear_algebra::determinant);
    }

    fn register_diffeq_functions(&mut self) {
        // Ordinary Differential Equations
        self.register("NDSolve", mathematics::differential::nd_solve);
        self.register("DSolve", mathematics::differential::d_solve);
        self.register("DEigensystem", mathematics::differential::d_eigensystem);
        
        // Partial Differential Equations
        self.register("PDSolve", mathematics::differential::pd_solve);
        self.register("LaplacianFilter", mathematics::differential::laplacian_filter);
        self.register("WaveEquation", mathematics::differential::wave_equation);
        
        // Vector Calculus
        self.register("VectorCalculus", mathematics::differential::vector_calculus);
        self.register("Gradient", mathematics::differential::gradient);
        self.register("Divergence", mathematics::differential::divergence);
        self.register("Curl", mathematics::differential::curl);
        
        // Numerical Methods
        self.register("RungeKutta", mathematics::differential::runge_kutta);
        self.register("AdamsBashforth", mathematics::differential::adams_bashforth);
        self.register("BDF", mathematics::differential::bdf);
        
        // Special Functions
        self.register("BesselJ", mathematics::differential::bessel_j);
        self.register("HermiteH", mathematics::differential::hermite_h);
        self.register("LegendreP", mathematics::differential::legendre_p);
        
        // Transform Methods
        self.register("LaplaceTransform", mathematics::differential::laplace_transform);
        self.register("ZTransform", mathematics::differential::z_transform);
        self.register("HankelTransform", mathematics::differential::hankel_transform);
    }

    fn register_interp_functions(&mut self) {
        // Interpolation Methods
        self.register("Interpolation", mathematics::interpolation::interpolation);
        self.register("SplineInterpolation", mathematics::interpolation::spline_interpolation);
        self.register("PolynomialInterpolation", mathematics::interpolation::polynomial_interpolation);
        
        // Numerical Integration
        self.register("NIntegrateAdvanced", mathematics::interpolation::n_integrate_advanced);
        self.register("GaussLegendre", mathematics::interpolation::gauss_legendre);
        self.register("AdaptiveQuadrature", mathematics::interpolation::adaptive_quadrature_wrapper);
        
        // Root Finding
        self.register("FindRootAdvanced", mathematics::interpolation::find_root_advanced);
        self.register("BrentMethod", mathematics::interpolation::brent_method_wrapper);
        self.register("NewtonRaphson", mathematics::interpolation::newton_raphson_wrapper);
        
        // Curve Fitting
        self.register("NonlinearFit", mathematics::interpolation::nonlinear_fit);
        self.register("LeastSquaresFit", mathematics::interpolation::least_squares_fit);
        self.register("SplineFit", mathematics::interpolation::spline_fit);
        
        // Numerical Differentiation
        self.register("NDerivative", mathematics::interpolation::n_derivative);
        self.register("FiniteDifference", mathematics::interpolation::finite_difference_wrapper);
        self.register("RichardsonExtrapolation", mathematics::interpolation::richardson_extrapolation);
        
        // Error Analysis
        self.register("ErrorEstimate", mathematics::interpolation::error_estimate);
        self.register("RichardsonExtrapolationError", mathematics::interpolation::richardson_extrapolation_error);
        self.register("AdaptiveMethod", mathematics::interpolation::adaptive_method);
    }

    fn register_special_functions(&mut self) {
        // Mathematical Constants
        self.register("Pi", mathematics::special::pi_constant);
        self.register("E", mathematics::special::e_constant);
        self.register("EulerGamma", mathematics::special::euler_gamma);
        self.register("GoldenRatio", mathematics::special::golden_ratio);
        
        // Gamma Functions
        self.register("Gamma", mathematics::special::gamma_function);
        self.register("LogGamma", mathematics::special::log_gamma);
        self.register("Digamma", mathematics::special::digamma);
        self.register("Polygamma", mathematics::special::polygamma);
        
        // Hypergeometric Functions
        self.register("Hypergeometric0F1", mathematics::special::hypergeometric_0f1);
        self.register("Hypergeometric1F1", mathematics::special::hypergeometric_1f1);
        
        // Elliptic Functions
        self.register("EllipticK", mathematics::special::elliptic_k);
        self.register("EllipticE", mathematics::special::elliptic_e);
        self.register("EllipticTheta", mathematics::special::elliptic_theta);
        
        // Orthogonal Polynomials
        self.register("ChebyshevT", mathematics::special::chebyshev_t);
        self.register("ChebyshevU", mathematics::special::chebyshev_u);
        self.register("GegenbauerC", mathematics::special::gegenbauer_c);
        
        // Error Functions
        self.register("Erf", mathematics::special::erf_function);
        self.register("Erfc", mathematics::special::erfc_function);
        self.register("InverseErf", mathematics::special::inverse_erf);
        self.register("FresnelC", mathematics::special::fresnel_c);
        self.register("FresnelS", mathematics::special::fresnel_s);
    }

    fn register_graph_functions(&mut self) {
        // Use the centralized graph function registry
        let graph_functions = graph::register_graph_functions();
        for (name, function) in graph_functions {
            self.register(&name, function);
        }
    }

    fn register_optimization_functions(&mut self) {
        // Root finding algorithms
        self.register("FindRoot", mathematics::optimization::find_root);
        self.register("Newton", mathematics::optimization::newton_method_wrapper);
        self.register("Bisection", mathematics::optimization::bisection_method_wrapper);
        self.register("Secant", mathematics::optimization::secant_method_wrapper);
        
        // Optimization algorithms
        self.register("Minimize", mathematics::optimization::minimize);
        self.register("Maximize", mathematics::optimization::maximize);
        
        // Numerical integration
        self.register("NIntegrate", mathematics::optimization::n_integrate);
        self.register("GaussianQuadrature", mathematics::optimization::gaussian_quadrature);
        self.register("MonteCarlo", mathematics::optimization::monte_carlo_integration);
    }

    fn register_signal_functions(&mut self) {
        // Phase 1.2: Signal Processing & FFT Fundamentals
        // Register all signal processing functions from the dedicated signal module
        let signal_functions = signal::register_signal_functions();
        for (name, function) in signal_functions {
            self.register(&name, function);
        }
    }

    fn register_image_functions(&mut self) {
        let image_functions = image::register_image_functions();
        for (name, function) in image_functions {
            self.register(name, function);
        }
    }

    fn register_vision_functions(&mut self) {
        let vision_functions = vision::register_vision_functions();
        for (name, function) in vision_functions {
            self.register(name, function);
        }
    }

    fn register_timeseries_functions(&mut self) {
        let ts_functions = analytics::timeseries::register_timeseries_functions();
        for (name, function) in ts_functions {
            self.register(name, function);
        }
    }

    fn register_clustering_functions(&mut self) {
        let clust_functions = clustering::register_clustering_functions();
        for (name, function) in clust_functions {
            self.register(name, function);
        }
    }

    fn register_sparse_functions(&mut self) {
        // Core matrix creation functions
        self.register("CSRMatrix", sparse::csr::csr_matrix);
        self.register("CSCMatrix", sparse::csc::csc_matrix);
        self.register("COOMatrix", sparse::coo::coo_matrix);
        
        // Sparse matrix operations
        self.register("SparseAdd", sparse::operations::sparse_add);
        self.register("SparseMultiply", sparse::operations::sparse_multiply);
        self.register("SparseTranspose", sparse::operations::sparse_transpose);
        
        // Matrix information functions
        self.register("SparseShape", sparse::operations::sparse_shape);
        self.register("SparseNNZ", sparse::operations::sparse_nnz);
        self.register("SparseDensity", sparse::operations::sparse_density);
        
        // Conversion functions
        self.register("SparseToDense", sparse::operations::sparse_to_dense);
    }

    fn register_spatial_functions(&mut self) {
        // Note: Spatial module has compilation issues that need to be fixed
        // For now, registering minimal functions that are closer to working
        // TODO: Fix error type conversions and f64 ordering issues in spatial module
        
        // Core spatial tree functions (commented out due to compilation issues)
        // self.register("KDTree", spatial::kdtree::kdtree);
        // self.register("BallTree", spatial::balltree::balltree);
        // self.register("RTree", spatial::rtree::rtree);
    }

    fn register_numerical_functions(&mut self) {
        let num_functions = numerical::register_numerical_functions();
        for (name, function) in num_functions {
            self.register(name, function);
        }
    }

    fn register_result_functions(&mut self) {
        // Result constructors
        self.register("Ok", result::ok_constructor);
        self.register("Error", result::error_constructor);
        
        // Option constructors  
        self.register("Some", result::some_constructor);
        self.register("None", result::none_constructor);
        
        // Result methods
        self.register("ResultIsOk", result::result_is_ok);
        self.register("ResultIsError", result::result_is_error);
        self.register("ResultUnwrap", result::result_unwrap);
        self.register("ResultUnwrapOr", result::result_unwrap_or);
        self.register("ResultMap", result::result_map);
        self.register("ResultAndThen", result::result_and_then);
        
        // Option methods
        self.register("OptionIsSome", result::option_is_some);
        self.register("OptionIsNone", result::option_is_none);
        self.register("OptionUnwrap", result::option_unwrap);
        self.register("OptionUnwrapOr", result::option_unwrap_or);
        self.register("OptionMap", result::option_map);
        self.register("OptionAndThen", result::option_and_then);

        // Schema inspection
        self.register("Schema", common::schema::schema_function);
    }
    
    fn register_async_functions(&mut self) {
        // Register async operations using the concurrency system
        let async_functions = async_ops::get_async_functions();
        for (name, function) in async_functions {
            self.register(name, function);
        }
    }

    fn register_network_functions(&mut self) {
        let fns = crate::stdlib::network::register_network_functions();
        for (n, f) in fns { self.register(n, f); }
    }

    fn register_serialization_functions(&mut self) {
        let fns = crate::stdlib::util_serialization::register_serialization_functions();
        for (n, f) in fns { self.register(n, f); }
    }

    fn register_config_functions(&mut self) {
        let fns = crate::stdlib::util_config::register_config_functions();
        for (n, f) in fns { self.register(n, f); }
    }

    fn register_cache_functions(&mut self) {
        let fns = crate::stdlib::util_cache::register_cache_functions();
        for (n, f) in fns { self.register(n, f); }
    }

    fn register_object_store_functions(&mut self) {
        let fns = crate::stdlib::io::object_store::register_object_store_functions();
        for (n, f) in fns { self.register(n, f); }
    }

    fn register_number_theory_functions(&mut self) {
        let nt_functions = number_theory::register_number_theory_functions();
        for (name, function) in nt_functions {
            self.register(name, function);
        }
    }

    fn register_combinatorics_functions(&mut self) {
        let comb_functions = combinatorics::register_combinatorics_functions();
        for (name, function) in comb_functions {
            self.register(name, function);
        }
    }

    fn register_geometry_functions(&mut self) {
        let geo_functions = geometry::register_geometry_functions();
        for (name, function) in geo_functions {
            self.register(name, function);
        }
    }

    fn register_topology_functions(&mut self) {
        let fns = crate::stdlib::topology::register_topology_functions();
        for (n, f) in fns { self.register(n, f); }
    }
    
    fn register_data_processing_functions(&mut self) {
        let fns = crate::stdlib::data_processing::register_data_processing_functions();
        for (n, f) in fns { self.register(n, f); }
    }

    fn register_temporal_functions(&mut self) {
        // Agent 3: Date/Time & Temporal Operations System (30+ functions)
        
        // Core Date/Time Types
        self.register("Date", temporal::date);
        self.register("DateTime", temporal::datetime);
        self.register("Duration", temporal::duration);
        self.register("TimeZone", temporal::timezone);
        
        // Current Time Functions
        self.register("Now", temporal::now);
        self.register("Today", temporal::today);
        self.register("UTCNow", temporal::utc_now);
        self.register("UnixTimestamp", temporal::unix_timestamp);
        
        // Date Construction
        self.register("DateParse", temporal::date_parse);
        self.register("DateFromUnix", temporal::date_from_unix);
        self.register("DateFromDays", temporal::date_from_days);
        self.register("DateFromISOWeek", temporal::date_from_iso_week);
        
        // Date Manipulation
        self.register("DateAdd", temporal::date_add);
        self.register("DateSubtract", temporal::date_subtract);
        self.register("DateDifference", temporal::date_difference);
        self.register("DateTruncate", temporal::date_truncate);
        self.register("DateRound", temporal::date_round);
        self.register("DateRange", temporal::date_range);
        
        // Date Formatting
        self.register("DateFormat", temporal::date_format);
        self.register("DateFormatISO", temporal::date_format_iso);
        self.register("DateFormatLocal", temporal::date_format_local);
        self.register("DateToString", temporal::date_to_string);
        
        // Date Components
        self.register("Year", temporal::year);
        self.register("Month", temporal::month);
        self.register("Day", temporal::day);
        self.register("Hour", temporal::hour);
        self.register("Minute", temporal::minute);
        self.register("Second", temporal::second);
        self.register("DayOfWeek", temporal::day_of_week);
        self.register("DayOfYear", temporal::day_of_year);
        self.register("WeekOfYear", temporal::week_of_year);
        self.register("Quarter", temporal::quarter);
        
        // Time Zone Operations
        self.register("TimeZoneConvert", temporal::timezone_convert);
        self.register("TimeZoneList", temporal::timezone_list);
        self.register("LocalTimeZone", temporal::local_timezone);
        self.register("IsDST", temporal::is_dst);
        
        // Calendar Operations
        self.register("BusinessDays", temporal::business_days);
        self.register("AddBusinessDays", temporal::add_business_days);
        self.register("IsBusinessDay", temporal::is_business_day);
        self.register("IsWeekend", temporal::is_weekend);
        self.register("IsLeapYear", temporal::is_leap_year_fn);
        
        // Duration Operations
        self.register("DurationToSeconds", temporal::duration_to_seconds);
        self.register("DurationToMinutes", temporal::duration_to_minutes);
        self.register("DurationToHours", temporal::duration_to_hours);
        self.register("DurationToDays", temporal::duration_to_days);
        self.register("DurationAdd", temporal::duration_add);
        self.register("DurationSubtract", temporal::duration_subtract);
    }

    fn register_developer_tools_functions(&mut self) {
        // Agent 4: Developer Experience & Debugging System (25+ functions)
        
        // Debugging Tools (5 functions)
        self.register("Inspect", developer_tools::inspect);
        self.register("Debug", developer_tools::debug);
        self.register("Trace", developer_tools::trace_execution);
        self.register("DebugBreak", developer_tools::debug_break);
        self.register("StackTrace", developer_tools::stack_trace);
        
        // Performance Tools (5 functions)
        self.register("Timing", developer_tools::timing);
        self.register("MemoryUsage", developer_tools::memory_usage);
        self.register("ProfileFunction", developer_tools::profile_function);
        self.register("Benchmark", developer_tools::benchmark);
        self.register("BenchmarkCompare", developer_tools::benchmark_compare);
        
        // Error Handling (5 functions)
        self.register("Try", developer_tools::try_catch);
        self.register("Assert", developer_tools::assert);
        self.register("Validate", developer_tools::validate);
        self.register("ErrorMessage", developer_tools::error_message);
        self.register("ThrowError", developer_tools::throw_error);
        
        // Testing Framework (5 functions)
        self.register("Test", developer_tools::test);
        self.register("TestSuite", developer_tools::test_suite);
        self.register("MockData", developer_tools::mock_data);
        self.register("BenchmarkSuite", developer_tools::benchmark_suite);
        self.register("TestReport", developer_tools::test_report);
        
        // Logging System (5 functions)
        self.register("Log", developer_tools::log);
        self.register("LogLevel", developer_tools::log_level);
        self.register("LogToFile", developer_tools::log_to_file);
        self.register("LogFilter", developer_tools::log_filter);
        self.register("LogHistory", developer_tools::log_history);
        
        // Introspection & Reflection (6 functions)
        self.register("FunctionInfo", developer_tools::function_info);
        self.register("FunctionList", developer_tools::function_list);
        self.register("Help", developer_tools::help);
        self.register("TypeOf", developer_tools::type_of);
        self.register("SizeOf", developer_tools::size_of);
        self.register("Dependencies", developer_tools::dependencies);
    }

    fn register_system_functions(&mut self) {
        // Agent 5: System Integration & Environment System (39+ functions)
        
        // Environment Variables (5 functions)
        self.register("Environment", system::environment);
        self.register("SetEnvironment", system::set_environment);
        self.register("UnsetEnvironment", system::unset_environment);
        self.register("EnvironmentList", system::environment_list);
        self.register("SystemInfo", system::system_info);
        
        // File System Operations (8 functions)
        self.register("FileExists", system::file_exists);
        self.register("DirectoryExists", system::directory_exists);
        self.register("DirectoryList", system::directory_list);
        self.register("CreateDirectory", system::create_directory);
        self.register("DeleteFile", system::delete_file);
        self.register("DeleteDirectory", system::delete_directory);
        self.register("CopyFile", system::copy_file);
        self.register("MoveFile", system::move_file);
        
        // File Information (7 functions)
        self.register("FileSize", system::file_size);
        self.register("FileModificationTime", system::file_modification_time);
        self.register("FilePermissions", system::file_permissions);
        self.register("SetFilePermissions", system::set_file_permissions);
        self.register("IsFile", system::is_file);
        self.register("IsDirectory", system::is_directory);
        self.register("IsSymbolicLink", system::is_symbolic_link);
        
        // Process Management (6 functions)
        self.register("RunCommand", system::run_command);
        self.register("ProcessStart", system::process_start);
        self.register("ProcessList", system::process_list);
        self.register("ProcessKill", system::process_kill);
        self.register("CurrentPID", system::current_pid);
        self.register("ProcessExists", system::process_exists);
        
        // Path Operations (7 functions)
        self.register("AbsolutePath", system::absolute_path);
        self.register("RelativePath", system::relative_path);
        self.register("PathJoin", system::path_join);
        self.register("PathSplit", system::path_split);
        self.register("FileName", system::file_name);
        self.register("FileExtension", system::file_extension);
        self.register("DirectoryName", system::directory_name);
        
        // System Information (6 functions)
        self.register("CurrentDirectory", system::current_directory);
        self.register("SetCurrentDirectory", system::set_current_directory);
        self.register("HomeDirectory", system::home_directory);
        self.register("TempDirectory", system::temp_directory);
        self.register("CurrentUser", system::current_user);
        self.register("SystemArchitecture", system::system_architecture);
    }

    fn register_language_functions(&mut self) {
        // Core language constructors and evaluator
        self.register("Expr", language::expr_constructor);
        self.register("Eval", language::eval_function);
    }

    fn register_collections_functions(&mut self) {
        // Set operations (8 functions)
        self.register("SetCreate", collections::set_create);
        self.register("SetUnion", collections::set_union);
        self.register("SetIntersection", collections::set_intersection);
        self.register("SetDifference", collections::set_difference);
        self.register("SetContains", collections::set_contains);
        self.register("SetAdd", collections::set_add);
        self.register("SetRemove", collections::set_remove);
        self.register("SetSize", collections::set_size);
        
        // Dictionary operations (9 functions)
        self.register("DictCreate", collections::dict_create);
        self.register("DictGet", collections::dict_get);
        self.register("DictSet", collections::dict_set);
        self.register("DictDelete", collections::dict_delete);
        self.register("DictKeys", collections::dict_keys);
        self.register("DictValues", collections::dict_values);
        self.register("DictContains", collections::dict_contains);
        self.register("DictMerge", collections::dict_merge);
        self.register("DictSize", collections::dict_size);
        
        // Queue operations (4 functions)
        self.register("QueueCreate", collections::queue_create);
        self.register("QueueEnqueue", collections::queue_enqueue);
        self.register("QueueDequeue", collections::queue_dequeue);
        self.register("QueueSize", collections::queue_size);
        
        // Stack operations (4 functions)
        self.register("StackCreate", collections::stack_create);
        self.register("StackPush", collections::stack_push);
        self.register("StackPop", collections::stack_pop);
        self.register("StackSize", collections::stack_size);
    }

    fn register_bioinformatics_functions(&mut self) {
        // Phase 2.1: Bioinformatics Algorithms
        
        // Sequence Alignment Functions (4 functions)
        self.register("GlobalAlignment", bioinformatics::alignment::global_alignment);
        self.register("LocalAlignment", bioinformatics::alignment::local_alignment);
        self.register("MultipleAlignment", bioinformatics::alignment::multiple_alignment);
        self.register("BlastSearch", bioinformatics::alignment::blast_search);
        
        // Phylogenetic Functions (4 functions)
        self.register("PhylogeneticTree", bioinformatics::phylogenetics::phylogenetic_tree);
        self.register("NeighborJoining", bioinformatics::phylogenetics::neighbor_joining);
        self.register("MaximumLikelihood", bioinformatics::phylogenetics::maximum_likelihood);
        self.register("PairwiseDistance", bioinformatics::phylogenetics::pairwise_distance);
        
        // Genomics Functions (8 functions)
        self.register("BiologicalSequence", bioinformatics::genomics::biological_sequence);
        self.register("ReverseComplement", bioinformatics::genomics::reverse_complement);
        self.register("Translate", bioinformatics::genomics::translate);
        self.register("Transcribe", bioinformatics::genomics::transcribe);
        self.register("GCContent", bioinformatics::genomics::gc_content);
        self.register("FindORFs", bioinformatics::genomics::find_orfs);
        self.register("FindMotifs", bioinformatics::genomics::find_motifs);
        self.register("CodonUsage", bioinformatics::genomics::codon_usage);
    }

    fn register_finance_functions(&mut self) {
        // Phase 2.3: Financial Mathematics Algorithms
        
        // Option Pricing Models (6 functions)
        self.register("BlackScholes", finance::pricing::black_scholes);
        self.register("BinomialTree", finance::pricing::binomial_tree);
        self.register("MonteCarloOption", finance::pricing::monte_carlo_option);
        self.register("GreeksCalculation", finance::pricing::greeks_calculation);
        self.register("CreateOption", finance::pricing::create_option);
        
        // Risk Metrics (8 functions)
        self.register("ValueAtRisk", finance::risk::value_at_risk);
        self.register("ConditionalVaR", finance::risk::conditional_var_fn);
        self.register("SharpeRatio", finance::risk::sharpe_ratio);
        self.register("BetaCalculation", finance::risk::beta_calculation);
        self.register("CorrelationCalculation", finance::risk::correlation_calculation);
        self.register("TreynorRatio", finance::risk::treynor_ratio);
        self.register("InformationRatio", finance::risk::information_ratio);
        self.register("MaxDrawdown", finance::risk::max_drawdown);
        
        // Portfolio Optimization (8 functions)
        self.register("MarkowitzOptimization", finance::portfolio::markowitz_optimization);
        self.register("CAPMModel", finance::portfolio::capm_model);
        self.register("BondPrice", finance::portfolio::bond_price);
        self.register("BondDuration", finance::portfolio::bond_duration_fn);
        self.register("BondConvexity", finance::portfolio::bond_convexity_fn);
        self.register("EfficientFrontier", finance::portfolio::efficient_frontier);
        self.register("PortfolioPerformance", finance::portfolio::portfolio_performance);
        self.register("YieldCurveInterpolation", finance::portfolio::yield_curve_interpolation);
    }

    fn register_quantum_functions(&mut self) {
        // Phase 2.2: Quantum Computing Simulation Framework
        
        // Quantum Gates (12 functions)
        self.register("PauliXGate", quantum::pauli_x_gate);
        self.register("PauliYGate", quantum::pauli_y_gate);
        self.register("PauliZGate", quantum::pauli_z_gate);
        self.register("HadamardGate", quantum::hadamard_gate);
        self.register("PhaseGate", quantum::phase_gate);
        self.register("TGate", quantum::t_gate);
        self.register("RotationXGate", quantum::rotation_x_gate);
        self.register("RotationYGate", quantum::rotation_y_gate);
        self.register("RotationZGate", quantum::rotation_z_gate);
        self.register("CNOTGate", quantum::cnot_gate);
        self.register("CZGate", quantum::cz_gate);
        self.register("SWAPGate", quantum::swap_gate);
        self.register("ToffoliGate", quantum::toffoli_gate);
        self.register("FredkinGate", quantum::fredkin_gate);
        self.register("ControlledGate", quantum::controlled_gate);
        self.register("MultiControlledGate", quantum::multi_controlled_gate);
        self.register("CustomGate", quantum::custom_gate);
        
        // Quantum Circuits (8 functions)
        self.register("QuantumCircuit", quantum::quantum_circuit);
        self.register("QubitRegister", quantum::qubit_register);
        self.register("CircuitAddGate", quantum::circuit_add_gate);
        self.register("ExecuteCircuit", quantum::execute_circuit);
        self.register("MeasureQubit", quantum::measure_qubit);
        self.register("CreateQubitState", quantum::create_qubit_state);
        self.register("CreateSuperposition", quantum::create_superposition_state);
        self.register("CreateBellState", quantum::create_bell_state);
        self.register("StateProbabilities", quantum::state_probabilities);
        self.register("NormalizeState", quantum::normalize_state);
        self.register("PartialTrace", quantum::partial_trace);
        
        // Quantum Algorithms (10 functions)
        self.register("QuantumFourierTransform", quantum::quantum_fourier_transform);
        self.register("InverseQuantumFourierTransform", quantum::inverse_quantum_fourier_transform);
        self.register("GroverSearch", quantum::grovers_search);
        self.register("GroverOracle", quantum::grover_oracle);
        self.register("QuantumPhaseEstimation", quantum::quantum_phase_estimation);
        self.register("QuantumTeleportation", quantum::quantum_teleportation);
        self.register("ThreeQubitEncode", quantum::three_qubit_encode);
        self.register("ApplyBitFlipError", quantum::apply_bit_flip_error);
        self.register("ThreeQubitCorrect", quantum::three_qubit_correct);
        self.register("ThreeQubitDecode", quantum::three_qubit_decode);
        self.register("EntanglementMeasure", quantum::entanglement_measure);
        self.register("MeasureBellState", quantum::measure_bell_state);
    }

    fn register_nlp_functions(&mut self) {
        // Phase 3.2: Natural Language Processing Algorithms
        
        // Text Processing Functions (8 functions)
        self.register("TokenizeText", nlp::tokenize_text);
        self.register("StemText", nlp::stem_text);
        self.register("GenerateNGrams", nlp::generate_ngrams);
        self.register("TFIDFVectorize", nlp::tfidf_vectorize);
        self.register("WordFrequency", nlp::word_frequency);
        self.register("TextSimilarity", nlp::text_similarity);
        self.register("NormalizeText", nlp::normalize_text);
        self.register("RemoveStopWords", nlp::remove_stop_words);
        
        // Text Analysis Functions (6 functions)
        self.register("SentimentAnalysis", nlp::sentiment_analysis);
        self.register("NamedEntityRecognition", nlp::named_entity_recognition);
        self.register("POSTagging", nlp::pos_tagging);
        self.register("LanguageDetection", nlp::language_detection);
        self.register("TextClassification", nlp::text_classification);
        self.register("KeywordExtraction", nlp::keyword_extraction);
        
        // Language Modeling Functions (4 functions)
        self.register("LanguageModel", nlp::language_model);
        self.register("TextSummarization", nlp::text_summarization);
        self.register("SpellCheck", nlp::spell_check);
        self.register("SpellCorrect", nlp::spell_correct);
        
        // Advanced Sentiment Analysis Functions (3 functions)
        self.register("RuleBasedSentiment", nlp::rule_based_sentiment);
        self.register("StatisticalSentiment", nlp::statistical_sentiment);
        self.register("EmotionDetection", nlp::emotion_detection);
    }

    fn register_game_theory_functions(&mut self) {
        // Phase 3.1: Game Theory & Mechanism Design Algorithms
        
        // Equilibrium Concepts (4 functions)
        self.register("NashEquilibrium", game_theory::nash_equilibrium);
        self.register("CorrelatedEquilibrium", game_theory::correlated_equilibrium);
        self.register("EvolutionaryStableStrategy", game_theory::evolutionary_stable_strategy);
        self.register("EliminateDominatedStrategies", game_theory::eliminate_dominated_strategies);
        
        // Auction Mechanisms (6 functions)
        self.register("FirstPriceAuction", game_theory::first_price_auction);
        self.register("SecondPriceAuction", game_theory::second_price_auction);
        self.register("VickreyAuction", game_theory::vickrey_auction);
        self.register("CombinatorialAuction", game_theory::combinatorial_auction);
        self.register("EnglishAuction", game_theory::english_auction);
        self.register("DutchAuction", game_theory::dutch_auction);
        
        // Mechanism Design (6 functions)
        self.register("VCGMechanism", game_theory::vcg_mechanism);
        self.register("OptimalAuction", game_theory::optimal_auction);
        self.register("RevenueMaximization", game_theory::revenue_maximization);
        self.register("StableMarriage", game_theory::stable_marriage);
        self.register("AssignmentProblem", game_theory::assignment_problem);
        self.register("StableAssignment", game_theory::stable_assignment);
    }

    fn register_algorithm_functions(&mut self) {
        // Core Computer Science Algorithms
        
        // Sorting Algorithms (6 functions)
        self.register("Sort", algorithms::sort_list);
        self.register("QuickSort", algorithms::quicksort_list);
        self.register("MergeSort", algorithms::mergesort_list);
        self.register("HeapSort", algorithms::heapsort_list);
        self.register("InsertionSort", algorithms::insertion_sort_list);
        self.register("IsSorted", algorithms::is_sorted);
        
        // Search Algorithms (5 functions)
        self.register("BinarySearch", algorithms::binary_search);
        self.register("BinarySearchFirst", algorithms::binary_search_first);
        self.register("BinarySearchLast", algorithms::binary_search_last);
        self.register("LinearSearch", algorithms::linear_search);
        self.register("InterpolationSearch", algorithms::interpolation_search);
        
        // String Algorithms (9 functions)
        self.register("KMPSearch", algorithms::kmp_search);
        self.register("EditDistance", algorithms::edit_distance);
        self.register("BoyerMooreSearch", algorithms::boyer_moore_search);
        self.register("RabinKarpSearch", algorithms::rabin_karp_search);
        self.register("HammingDistance", algorithms::hamming_distance);
        self.register("JaroWinklerDistance", algorithms::jaro_winkler_distance);
        self.register("LongestCommonSubstring", algorithms::longest_common_substring);
        self.register("SuffixArray", algorithms::suffix_array);
        self.register("ZAlgorithm", algorithms::z_algorithm);
        
        // Compression Algorithms (2 functions - more to be implemented)
        self.register("RunLengthEncode", algorithms::run_length_encode);
        self.register("RunLengthDecode", algorithms::run_length_decode);
    }

    fn register_data_structure_functions(&mut self) {
        // Data Structure Algorithms
        
        // Binary Heap Operations (9 functions)
        self.register("BinaryHeap", data_structures::heap::binary_heap);
        self.register("BinaryMaxHeap", data_structures::heap::binary_max_heap);
        self.register("HeapInsert", data_structures::heap::heap_insert);
        self.register("HeapExtractMin", data_structures::heap::heap_extract_min);
        self.register("HeapExtractMax", data_structures::heap::heap_extract_max);
        self.register("HeapPeek", data_structures::heap::heap_peek);
        self.register("HeapSize", data_structures::heap::heap_size);
        self.register("HeapIsEmpty", data_structures::heap::heap_is_empty);
        self.register("HeapMerge", data_structures::heap::heap_merge);
        
        // Queue Operations (4 functions)
        self.register("Queue", data_structures::queue::queue);
        self.register("Enqueue", data_structures::queue::enqueue);
        self.register("Dequeue", data_structures::queue::dequeue);
        self.register("QueueFront", data_structures::queue::queue_front);
        
        // Stack Operations (4 functions)
        self.register("Stack", data_structures::stack::stack);
        self.register("Push", data_structures::stack::push);
        self.register("Pop", data_structures::stack::pop);
        self.register("StackTop", data_structures::stack::stack_top);
        
        // Priority Queue Operations (10 functions)
        self.register("PriorityQueue", data_structures::priority_queue::priority_queue);
        self.register("MinPriorityQueue", data_structures::priority_queue::min_priority_queue);
        self.register("PQInsert", data_structures::priority_queue::pq_insert);
        self.register("PQExtract", data_structures::priority_queue::pq_extract);
        self.register("PQPeek", data_structures::priority_queue::pq_peek);
        self.register("PQPeekPriority", data_structures::priority_queue::pq_peek_priority);
        self.register("PQSize", data_structures::priority_queue::pq_size);
        self.register("PQIsEmpty", data_structures::priority_queue::pq_is_empty);
        self.register("PQContains", data_structures::priority_queue::pq_contains);
        self.register("PQMerge", data_structures::priority_queue::pq_merge);
        
        // Binary Search Tree Operations (11 functions)
        self.register("BST", data_structures::trees::bst::bst_new);
        self.register("BSTInsert", data_structures::trees::bst::bst_insert);
        self.register("BSTSearch", data_structures::trees::bst::bst_search);
        self.register("BSTDelete", data_structures::trees::bst::bst_delete);
        self.register("BSTMin", data_structures::trees::bst::bst_min);
        self.register("BSTMax", data_structures::trees::bst::bst_max);
        self.register("BSTInOrder", data_structures::trees::bst::bst_inorder);
        self.register("BSTPreOrder", data_structures::trees::bst::bst_preorder);
        self.register("BSTPostOrder", data_structures::trees::bst::bst_postorder);
        self.register("BSTHeight", data_structures::trees::bst::bst_height);
        self.register("BSTBalance", data_structures::trees::bst::bst_is_balanced);
        
        // Trie (Prefix Tree) Operations (12 functions)
        self.register("Trie", data_structures::trees::trie::trie_new);
        self.register("TrieInsert", data_structures::trees::trie::trie_insert);
        self.register("TrieInsertWithValue", data_structures::trees::trie::trie_insert_with_value);
        self.register("TrieSearch", data_structures::trees::trie::trie_search);
        self.register("TrieStartsWith", data_structures::trees::trie::trie_starts_with);
        self.register("TrieGet", data_structures::trees::trie::trie_get);
        self.register("TrieDelete", data_structures::trees::trie::trie_delete);
        self.register("TrieWordsWithPrefix", data_structures::trees::trie::trie_get_words_with_prefix);
        self.register("TrieAllWords", data_structures::trees::trie::trie_get_all_words);
        self.register("TrieSize", data_structures::trees::trie::trie_size);
        self.register("TrieIsEmpty", data_structures::trees::trie::trie_is_empty);
        self.register("TrieLongestPrefix", data_structures::trees::trie::trie_longest_common_prefix);
        
        // Disjoint Set (Union-Find) Operations (11 functions)
        self.register("DisjointSet", data_structures::disjoint_set::disjoint_set_new);
        self.register("DSFind", data_structures::disjoint_set::disjoint_set_find);
        self.register("DSUnion", data_structures::disjoint_set::disjoint_set_union);
        self.register("DSConnected", data_structures::disjoint_set::disjoint_set_connected);
        self.register("DSSetSize", data_structures::disjoint_set::disjoint_set_set_size);
        self.register("DSCount", data_structures::disjoint_set::disjoint_set_count);
        self.register("DSTotalElements", data_structures::disjoint_set::disjoint_set_total_elements);
        self.register("DSGetMembers", data_structures::disjoint_set::disjoint_set_get_members);
        self.register("DSGetAllSets", data_structures::disjoint_set::disjoint_set_get_all_sets);
        self.register("DSGetRoots", data_structures::disjoint_set::disjoint_set_get_roots);
        self.register("DSReset", data_structures::disjoint_set::disjoint_set_reset);
    }
}

impl Default for StandardLibrary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stdlib_creation() {
        let stdlib = StandardLibrary::new();

        // Verify key functions are registered
        assert!(stdlib.get_function("Length").is_some());
        assert!(stdlib.get_function("Head").is_some());
        assert!(stdlib.get_function("StringJoin").is_some());
        assert!(stdlib.get_function("Sin").is_some());
        assert!(stdlib.get_function("ReplaceAll").is_some());
        assert!(stdlib.get_function("Array").is_some());
        assert!(stdlib.get_function("ArrayDimensions").is_some());
        
        // Verify linear algebra functions are registered
        assert!(stdlib.get_function("Dot").is_some());
        assert!(stdlib.get_function("Transpose").is_some());
        assert!(stdlib.get_function("Maximum").is_some());
        
        // Verify activation functions are registered
        assert!(stdlib.get_function("Sigmoid").is_some());
        assert!(stdlib.get_function("Tanh").is_some());
        assert!(stdlib.get_function("Softmax").is_some());
    }

    #[test]
    fn test_stdlib_function_count() {
        let stdlib = StandardLibrary::new();
        let function_count = stdlib.function_names().len();

        // Should have at least the core functions we're implementing
        assert!(function_count >= 39); // 7 list + 4 string + 6 math + 3 rules + 11 tensor + 8 graph = 39 minimum
    }

    #[test]
    fn test_function_lookup() {
        let stdlib = StandardLibrary::new();

        // Test valid lookup
        assert!(stdlib.get_function("Length").is_some());

        // Test invalid lookup
        assert!(stdlib.get_function("NonexistentFunction").is_none());
    }

    #[test]
    fn test_ml_functions_registered() {
        let stdlib = StandardLibrary::new();
        
        // Test spatial layer functions
        assert!(stdlib.get_function("FlattenLayer").is_some());
        assert!(stdlib.get_function("ReshapeLayer").is_some());
        assert!(stdlib.get_function("PermuteLayer").is_some());
        assert!(stdlib.get_function("TransposeLayer").is_some());
        
        // Test layer composition functions
        assert!(stdlib.get_function("Sequential").is_some());
        assert!(stdlib.get_function("Identity").is_some());
        
        // Test tensor utility functions
        assert!(stdlib.get_function("TensorShape").is_some());
        assert!(stdlib.get_function("TensorRank").is_some());
        assert!(stdlib.get_function("TensorSize").is_some());
    }

    #[test]
    fn test_graph_functions_registered() {
        let stdlib = StandardLibrary::new();
        
        // Test graph creation functions
        assert!(stdlib.get_function("Graph").is_some());
        assert!(stdlib.get_function("DirectedGraph").is_some());
        assert!(stdlib.get_function("AdjacencyMatrix").is_some());
        
        // Test graph algorithms
        assert!(stdlib.get_function("DepthFirstSearch").is_some());
        assert!(stdlib.get_function("BreadthFirstSearch").is_some());
        assert!(stdlib.get_function("Dijkstra").is_some());
        assert!(stdlib.get_function("ConnectedComponents").is_some());
        
        // Test graph properties
        assert!(stdlib.get_function("GraphProperties").is_some());
    }

    #[test]
    fn test_linalg_functions_registered() {
        let stdlib = StandardLibrary::new();
        
        // Test matrix decomposition functions
        assert!(stdlib.get_function("SVD").is_some());
        assert!(stdlib.get_function("QRDecomposition").is_some());
        assert!(stdlib.get_function("LUDecomposition").is_some());
        assert!(stdlib.get_function("CholeskyDecomposition").is_some());
        
        // Test eigenvalue computation functions
        assert!(stdlib.get_function("EigenDecomposition").is_some());
        assert!(stdlib.get_function("SchurDecomposition").is_some());
        
        // Test linear system functions
        assert!(stdlib.get_function("LinearSolve").is_some());
        assert!(stdlib.get_function("LeastSquares").is_some());
        assert!(stdlib.get_function("PseudoInverse").is_some());
        
        // Test matrix function operations
        assert!(stdlib.get_function("MatrixPower").is_some());
        assert!(stdlib.get_function("MatrixFunction").is_some());
        assert!(stdlib.get_function("MatrixTrace").is_some());
        
        // Test matrix analysis functions
        assert!(stdlib.get_function("MatrixRank").is_some());
        assert!(stdlib.get_function("MatrixCondition").is_some());
        assert!(stdlib.get_function("MatrixNorm").is_some());
        assert!(stdlib.get_function("Determinant").is_some());
    }

    #[test]
    fn test_optimization_functions_registered() {
        let stdlib = StandardLibrary::new();
        
        // Test root finding functions
        assert!(stdlib.get_function("FindRoot").is_some());
        assert!(stdlib.get_function("Newton").is_some());
        assert!(stdlib.get_function("Bisection").is_some());
        assert!(stdlib.get_function("Secant").is_some());
        
        // Test optimization functions
        assert!(stdlib.get_function("Minimize").is_some());
        assert!(stdlib.get_function("Maximize").is_some());
        
        // Test integration functions
        assert!(stdlib.get_function("NIntegrate").is_some());
        assert!(stdlib.get_function("GaussianQuadrature").is_some());
        assert!(stdlib.get_function("MonteCarlo").is_some());
    }

    #[test]
    fn test_interp_functions_registered() {
        let stdlib = StandardLibrary::new();
        
        // Test interpolation functions
        assert!(stdlib.get_function("Interpolation").is_some());
        assert!(stdlib.get_function("SplineInterpolation").is_some());
        assert!(stdlib.get_function("PolynomialInterpolation").is_some());
        
        // Test numerical integration functions
        assert!(stdlib.get_function("NIntegrateAdvanced").is_some());
        assert!(stdlib.get_function("GaussLegendre").is_some());
        assert!(stdlib.get_function("AdaptiveQuadrature").is_some());
        
        // Test root finding functions
        assert!(stdlib.get_function("FindRootAdvanced").is_some());
        assert!(stdlib.get_function("BrentMethod").is_some());
        assert!(stdlib.get_function("NewtonRaphson").is_some());
        
        // Test curve fitting functions
        assert!(stdlib.get_function("NonlinearFit").is_some());
        assert!(stdlib.get_function("LeastSquaresFit").is_some());
        assert!(stdlib.get_function("SplineFit").is_some());
        
        // Test numerical differentiation functions
        assert!(stdlib.get_function("NDerivative").is_some());
        assert!(stdlib.get_function("FiniteDifference").is_some());
        assert!(stdlib.get_function("RichardsonExtrapolation").is_some());
        
        // Test error analysis functions
        assert!(stdlib.get_function("ErrorEstimate").is_some());
        assert!(stdlib.get_function("RichardsonExtrapolationError").is_some());
        assert!(stdlib.get_function("AdaptiveMethod").is_some());
    }

    #[test]
    fn test_signal_functions_registered() {
        let stdlib = StandardLibrary::new();
        
        // Test Fourier transform functions
        assert!(stdlib.get_function("FFT").is_some());
        assert!(stdlib.get_function("IFFT").is_some());
        assert!(stdlib.get_function("DCT").is_some());
        assert!(stdlib.get_function("PowerSpectrum").is_some());
        
        // Test spectral analysis functions
        assert!(stdlib.get_function("Periodogram").is_some());
        assert!(stdlib.get_function("Spectrogram").is_some());
        assert!(stdlib.get_function("PSDEstimate").is_some());
        
        // Test windowing functions
        assert!(stdlib.get_function("HammingWindow").is_some());
        assert!(stdlib.get_function("HanningWindow").is_some());
        assert!(stdlib.get_function("BlackmanWindow").is_some());
        assert!(stdlib.get_function("ApplyWindow").is_some());
        
        // Test convolution functions
        assert!(stdlib.get_function("Convolve").is_some());
        assert!(stdlib.get_function("CrossCorrelation").is_some());
        assert!(stdlib.get_function("AutoCorrelation").is_some());
        
        // Test filtering functions
        assert!(stdlib.get_function("LowPassFilter").is_some());
        assert!(stdlib.get_function("HighPassFilter").is_some());
        assert!(stdlib.get_function("MedianFilter").is_some());
        
        // Test advanced processing functions
        assert!(stdlib.get_function("Hilbert").is_some());
        assert!(stdlib.get_function("ZeroPadding").is_some());
        assert!(stdlib.get_function("PhaseUnwrap").is_some());
    }

    #[test]
    fn test_special_functions_registered() {
        let stdlib = StandardLibrary::new();
        
        // Test mathematical constants
        assert!(stdlib.get_function("Pi").is_some());
        assert!(stdlib.get_function("E").is_some());
        assert!(stdlib.get_function("EulerGamma").is_some());
        assert!(stdlib.get_function("GoldenRatio").is_some());
        
        // Test gamma functions
        assert!(stdlib.get_function("Gamma").is_some());
        assert!(stdlib.get_function("LogGamma").is_some());
        assert!(stdlib.get_function("Digamma").is_some());
        assert!(stdlib.get_function("Polygamma").is_some());
        
        // Test hypergeometric functions
        assert!(stdlib.get_function("Hypergeometric0F1").is_some());
        assert!(stdlib.get_function("Hypergeometric1F1").is_some());
        
        // Test elliptic functions
        assert!(stdlib.get_function("EllipticK").is_some());
        assert!(stdlib.get_function("EllipticE").is_some());
        assert!(stdlib.get_function("EllipticTheta").is_some());
        
        // Test orthogonal polynomials
        assert!(stdlib.get_function("ChebyshevT").is_some());
        assert!(stdlib.get_function("ChebyshevU").is_some());
        assert!(stdlib.get_function("GegenbauerC").is_some());
        
        // Test error functions
        assert!(stdlib.get_function("Erf").is_some());
        assert!(stdlib.get_function("Erfc").is_some());
        assert!(stdlib.get_function("InverseErf").is_some());
        assert!(stdlib.get_function("FresnelC").is_some());
        assert!(stdlib.get_function("FresnelS").is_some());
    }
}
