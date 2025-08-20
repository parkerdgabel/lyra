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

// pub mod async_ops;  // Removed due to compilation issues
pub mod autodiff;
pub mod secure_wrapper;
pub mod calculus;
pub mod clustering;
pub mod data;
pub mod diffeq;
pub mod graph;
pub mod image;
pub mod interp;
pub mod io;
pub mod linalg;
pub mod list;
pub mod math;
pub mod ml;
pub mod optimization;
pub mod result;
pub mod rules;
pub mod signal;
// pub mod sparse;
// pub mod spatial;
pub mod special;
pub mod statistics;
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
        stdlib.register_math_functions();
        stdlib.register_calculus_functions();
        stdlib.register_statistics_functions();
        stdlib.register_rule_functions();
        stdlib.register_table_functions();
        stdlib.register_tensor_functions();
        stdlib.register_ml_functions();
        stdlib.register_io_functions();
        stdlib.register_linalg_functions();
        stdlib.register_diffeq_functions();
        stdlib.register_interp_functions();
        stdlib.register_special_functions();
        stdlib.register_graph_functions();
        stdlib.register_optimization_functions();
        stdlib.register_signal_functions();
        stdlib.register_image_functions();
        stdlib.register_timeseries_functions();
        stdlib.register_clustering_functions();
        stdlib.register_numerical_functions();
        // stdlib.register_sparse_functions();
        // stdlib.register_spatial_functions();
        stdlib.register_result_functions();
        stdlib.register_async_functions();
        stdlib.register_network_functions();
        stdlib.register_number_theory_functions();
        stdlib.register_combinatorics_functions();
        stdlib.register_geometry_functions();
        stdlib.register_topology_functions();

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
        self.register("StringJoin", string::string_join);
        self.register("StringLength", string::string_length);
        self.register("StringTake", string::string_take);
        self.register("StringDrop", string::string_drop);
    }

    fn register_math_functions(&mut self) {
        // Basic arithmetic functions (for Listable attribute support)
        self.register("Plus", math::plus);
        self.register("Times", math::times);
        self.register("Divide", math::divide);
        self.register("Power", math::power);
        self.register("Minus", math::minus);
        
        // Trigonometric and other math functions
        self.register("Sin", math::sin);
        self.register("Cos", math::cos);
        self.register("Tan", math::tan);
        self.register("Exp", math::exp);
        self.register("Log", math::log);
        self.register("Sqrt", math::sqrt);
        
        // Test functions for Hold attribute support
        self.register("TestHold", math::test_hold);
        self.register("TestHoldMultiple", math::test_hold_multiple);
    }

    fn register_calculus_functions(&mut self) {
        // Symbolic differentiation
        self.register("D", calculus::d);
        
        // Symbolic integration
        self.register("Integrate", calculus::integrate);
        self.register("IntegrateDefinite", calculus::integrate_definite);
    }

    fn register_statistics_functions(&mut self) {
        // Descriptive statistics
        self.register("Mean", statistics::mean);
        self.register("Variance", statistics::variance);
        self.register("StandardDeviation", statistics::standard_deviation);
        self.register("Median", statistics::median);
        self.register("Mode", statistics::mode);
        self.register("Quantile", statistics::quantile);
        
        // Min/Max and aggregation
        self.register("Min", statistics::min);
        self.register("Max", statistics::max);
        self.register("Total", statistics::total);
        
        // Random number generation
        self.register("RandomReal", statistics::random_real);
        self.register("RandomInteger", statistics::random_integer);
        
        // Correlation and covariance
        self.register("Correlation", statistics::correlation);
        self.register("Covariance", statistics::covariance);
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
        // Legacy table functions  
        self.register("GroupBy", table::group_by);
        self.register("Aggregate", table::aggregate);
        self.register("Count", table::count);
        
        // Foreign table constructors
        self.register("Table", table::table);
        self.register("TableFromRows", table::table_from_rows);
        self.register("EmptyTable", table::empty_table);
        
        // Foreign series constructors
        self.register("Series", table::series);
        self.register("Range", table::range);
        self.register("Zeros", table::zeros);
        self.register("Ones", table::ones);
        self.register("ConstantSeries", table::constant_series);
        
        // Foreign tensor constructors
        self.register("Tensor", table::tensor);
        self.register("ZerosTensor", table::zeros_tensor);
        self.register("OnesTensor", table::ones_tensor);
        self.register("EyeTensor", table::eye_tensor);
        self.register("RandomTensor", table::random_tensor);
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
        // Spatial layer functions for tree-shaking optimized ML framework
        self.register("FlattenLayer", ml::wrapper::flatten_layer);
        self.register("ReshapeLayer", ml::wrapper::reshape_layer);
        self.register("PermuteLayer", ml::wrapper::permute_layer);
        self.register("TransposeLayer", ml::wrapper::transpose_layer);
        
        // Layer composition functions
        self.register("Sequential", ml::wrapper::sequential_layer);
        self.register("Identity", ml::wrapper::identity_layer);
        
        // Tensor utility functions
        self.register("TensorShape", ml::wrapper::tensor_shape);
        self.register("TensorRank", ml::wrapper::tensor_rank);
        self.register("TensorSize", ml::wrapper::tensor_size);
    }

    fn register_io_functions(&mut self) {
        // I/O operations
        self.register("Import", io::import);
        self.register("Export", io::export);
    }

    fn register_linalg_functions(&mut self) {
        // Matrix decompositions
        self.register("SVD", linalg::svd);
        self.register("QRDecomposition", linalg::qr_decomposition);
        self.register("LUDecomposition", linalg::lu_decomposition);
        self.register("CholeskyDecomposition", linalg::cholesky_decomposition);
        
        // Eigenvalue computations
        self.register("EigenDecomposition", linalg::eigen_decomposition);
        self.register("SchurDecomposition", linalg::schur_decomposition);
        
        // Linear systems
        self.register("LinearSolve", linalg::linear_solve);
        self.register("LeastSquares", linalg::least_squares);
        self.register("PseudoInverse", linalg::pseudo_inverse);
        
        // Matrix functions
        self.register("MatrixPower", linalg::matrix_power);
        self.register("MatrixFunction", linalg::matrix_function);
        self.register("MatrixTrace", linalg::matrix_trace);
        
        // Matrix analysis
        self.register("MatrixRank", linalg::matrix_rank);
        self.register("MatrixCondition", linalg::matrix_condition);
        self.register("MatrixNorm", linalg::matrix_norm);
        self.register("Determinant", linalg::determinant);
    }

    fn register_diffeq_functions(&mut self) {
        // Ordinary Differential Equations
        self.register("NDSolve", diffeq::nd_solve);
        self.register("DSolve", diffeq::d_solve);
        self.register("DEigensystem", diffeq::d_eigensystem);
        
        // Partial Differential Equations
        self.register("PDSolve", diffeq::pd_solve);
        self.register("LaplacianFilter", diffeq::laplacian_filter);
        self.register("WaveEquation", diffeq::wave_equation);
        
        // Vector Calculus
        self.register("VectorCalculus", diffeq::vector_calculus);
        self.register("Gradient", diffeq::gradient);
        self.register("Divergence", diffeq::divergence);
        self.register("Curl", diffeq::curl);
        
        // Numerical Methods
        self.register("RungeKutta", diffeq::runge_kutta);
        self.register("AdamsBashforth", diffeq::adams_bashforth);
        self.register("BDF", diffeq::bdf);
        
        // Special Functions
        self.register("BesselJ", diffeq::bessel_j);
        self.register("HermiteH", diffeq::hermite_h);
        self.register("LegendreP", diffeq::legendre_p);
        
        // Transform Methods
        self.register("LaplaceTransform", diffeq::laplace_transform);
        self.register("ZTransform", diffeq::z_transform);
        self.register("HankelTransform", diffeq::hankel_transform);
    }

    fn register_interp_functions(&mut self) {
        // Interpolation Methods
        self.register("Interpolation", interp::interpolation);
        self.register("SplineInterpolation", interp::spline_interpolation);
        self.register("PolynomialInterpolation", interp::polynomial_interpolation);
        
        // Numerical Integration
        self.register("NIntegrateAdvanced", interp::n_integrate_advanced);
        self.register("GaussLegendre", interp::gauss_legendre);
        self.register("AdaptiveQuadrature", interp::adaptive_quadrature_wrapper);
        
        // Root Finding
        self.register("FindRootAdvanced", interp::find_root_advanced);
        self.register("BrentMethod", interp::brent_method_wrapper);
        self.register("NewtonRaphson", interp::newton_raphson_wrapper);
        
        // Curve Fitting
        self.register("NonlinearFit", interp::nonlinear_fit);
        self.register("LeastSquaresFit", interp::least_squares_fit);
        self.register("SplineFit", interp::spline_fit);
        
        // Numerical Differentiation
        self.register("NDerivative", interp::n_derivative);
        self.register("FiniteDifference", interp::finite_difference_wrapper);
        self.register("RichardsonExtrapolation", interp::richardson_extrapolation);
        
        // Error Analysis
        self.register("ErrorEstimate", interp::error_estimate);
        self.register("RichardsonExtrapolationError", interp::richardson_extrapolation_error);
        self.register("AdaptiveMethod", interp::adaptive_method);
    }

    fn register_special_functions(&mut self) {
        // Mathematical Constants
        self.register("Pi", special::pi_constant);
        self.register("E", special::e_constant);
        self.register("EulerGamma", special::euler_gamma);
        self.register("GoldenRatio", special::golden_ratio);
        
        // Gamma Functions
        self.register("Gamma", special::gamma_function);
        self.register("LogGamma", special::log_gamma);
        self.register("Digamma", special::digamma);
        self.register("Polygamma", special::polygamma);
        
        // Hypergeometric Functions
        self.register("Hypergeometric0F1", special::hypergeometric_0f1);
        self.register("Hypergeometric1F1", special::hypergeometric_1f1);
        
        // Elliptic Functions
        self.register("EllipticK", special::elliptic_k);
        self.register("EllipticE", special::elliptic_e);
        self.register("EllipticTheta", special::elliptic_theta);
        
        // Orthogonal Polynomials
        self.register("ChebyshevT", special::chebyshev_t);
        self.register("ChebyshevU", special::chebyshev_u);
        self.register("GegenbauerC", special::gegenbauer_c);
        
        // Error Functions
        self.register("Erf", special::erf_function);
        self.register("Erfc", special::erfc_function);
        self.register("InverseErf", special::inverse_erf);
        self.register("FresnelC", special::fresnel_c);
        self.register("FresnelS", special::fresnel_s);
    }

    fn register_graph_functions(&mut self) {
        // Graph creation functions
        self.register("Graph", graph::graph);
        self.register("DirectedGraph", graph::directed_graph);
        self.register("AdjacencyMatrix", graph::adjacency_matrix);
        
        // Graph traversal algorithms
        self.register("DepthFirstSearch", graph::depth_first_search);
        self.register("BreadthFirstSearch", graph::breadth_first_search);
        
        // Shortest path algorithms
        self.register("Dijkstra", graph::dijkstra);
        
        // Connectivity analysis
        self.register("ConnectedComponents", graph::connected_components);
        
        // Graph properties
        self.register("GraphProperties", graph::graph_properties);
    }

    fn register_optimization_functions(&mut self) {
        // Root finding algorithms
        self.register("FindRoot", optimization::find_root);
        self.register("Newton", optimization::newton_method_wrapper);
        self.register("Bisection", optimization::bisection_method_wrapper);
        self.register("Secant", optimization::secant_method_wrapper);
        
        // Optimization algorithms
        self.register("Minimize", optimization::minimize);
        self.register("Maximize", optimization::maximize);
        
        // Numerical integration
        self.register("NIntegrate", optimization::n_integrate);
        self.register("GaussianQuadrature", optimization::gaussian_quadrature);
        self.register("MonteCarlo", optimization::monte_carlo_integration);
    }

    fn register_signal_functions(&mut self) {
        // Fourier transform functions
        self.register("FFT", signal::fft);
        self.register("IFFT", signal::ifft);
        self.register("DCT", signal::dct);
        self.register("PowerSpectrum", signal::power_spectrum);
        
        // Spectral analysis functions
        self.register("Periodogram", signal::periodogram);
        self.register("Spectrogram", signal::spectrogram);
        self.register("PSDEstimate", signal::psd_estimate);
        
        // Windowing functions
        self.register("HammingWindow", signal::hamming_window);
        self.register("HanningWindow", signal::hanning_window);
        self.register("BlackmanWindow", signal::blackman_window);
        self.register("ApplyWindow", signal::apply_window);
        
        // Convolution and correlation
        self.register("Convolve", signal::convolve);
        self.register("CrossCorrelation", signal::cross_correlation);
        self.register("AutoCorrelation", signal::auto_correlation);
        
        // Digital filtering
        self.register("LowPassFilter", signal::low_pass_filter);
        self.register("HighPassFilter", signal::high_pass_filter);
        self.register("MedianFilter", signal::median_filter);
        
        // Advanced processing
        self.register("Hilbert", signal::hilbert_transform);
        self.register("ZeroPadding", signal::zero_padding);
        self.register("PhaseUnwrap", signal::phase_unwrap);
    }

    fn register_image_functions(&mut self) {
        // Phase 6A: Core Infrastructure  
        self.register("ImageImport", image::core::image_import);
        self.register("ImageExport", image::core::image_export);
        self.register("ImageInfo", image::core::image_info);
        self.register("ImageResize", image::core::image_resize);
        self.register("ImageHistogram", image::core::image_histogram);
        
        // Phase 6B: Filtering & Enhancement
        self.register("GaussianFilter", image::filters::gaussian_filter);
        self.register("MedianFilter", image::filters::median_filter);
        self.register("SobelFilter", image::filters::sobel_filter);
        self.register("CannyEdgeDetection", image::filters::canny_edge_detection);
        self.register("ImageRotate", image::filters::image_rotate);
        
        // Phase 6C: Morphological Operations
        self.register("Erosion", image::morphology::erosion);
        self.register("Dilation", image::morphology::dilation);
        self.register("Opening", image::morphology::opening);
        self.register("Closing", image::morphology::closing);
        
        // Phase 6D: Advanced Analysis
        self.register("AffineTransform", image::analysis::affine_transform);
        self.register("PerspectiveTransform", image::analysis::perspective_transform);
        self.register("ContourDetection", image::analysis::contour_detection);
        self.register("FeatureDetection", image::analysis::feature_detection);
        self.register("TemplateMatching", image::analysis::template_matching);
        self.register("ImageSegmentation", image::analysis::image_segmentation);
    }

    fn register_timeseries_functions(&mut self) {
        // Core time series functions
        self.register("TimeSeries", timeseries::core::timeseries);
        self.register("TimeSeriesWithIndex", timeseries::core::timeseries_with_index);
        
        // ARIMA/SARIMA models
        self.register("ARIMA", timeseries::arima::arima);
        self.register("SARIMA", timeseries::arima::sarima);
        
        // Forecasting methods
        self.register("ExponentialSmoothing", timeseries::forecasting::exponential_smoothing);
        self.register("MovingAverage", timeseries::forecasting::moving_average);
        self.register("ExponentialMovingAverage", timeseries::forecasting::exponential_moving_average);
    }

    fn register_clustering_functions(&mut self) {
        // Core clustering infrastructure
        self.register("ClusterData", clustering::core::cluster_data);
        self.register("DistanceMatrix", clustering::core::distance_matrix);
        
        // K-means family algorithms
        self.register("KMeans", clustering::kmeans::kmeans);
        self.register("MiniBatchKMeans", clustering::kmeans::mini_batch_kmeans);
    }

    // fn register_sparse_functions(&mut self) {
    //     // CSR format functions
    //     self.register("CSRMatrix", sparse::csr::csr_matrix);
    //     self.register("CSRFromTriplets", sparse::csr::csr_from_triplets);
    //     self.register("CSRFromDense", sparse::csr::csr_from_dense);
    //     
    //     // CSC format functions
    //     self.register("CSCMatrix", sparse::csc::csc_matrix);
    //     self.register("CSCFromTriplets", sparse::csc::csc_from_triplets);
    //     self.register("CSCFromDense", sparse::csc::csc_from_dense);
    //     
    //     // COO format functions
    //     self.register("COOMatrix", sparse::coo::coo_matrix);
    //     self.register("COOFromTriplets", sparse::coo::coo_from_triplets);
    //     self.register("COOFromDense", sparse::coo::coo_from_dense);
    // }

    // fn register_spatial_functions(&mut self) {
    //     // Core spatial tree functions
    //     self.register("KDTree", spatial::kdtree::kdtree);
    //     self.register("BallTree", spatial::balltree::balltree);
    //     self.register("RTree", spatial::rtree::rtree);
    // }

    fn register_numerical_functions(&mut self) {
        // Root finding and equation solving
        self.register("Bisection", numerical::roots::bisection);
        self.register("NewtonRaphson", numerical::roots::newton_raphson);
        self.register("Secant", numerical::roots::secant);
        self.register("Brent", numerical::roots::brent);
        self.register("FixedPoint", numerical::roots::fixed_point);
        
        // Numerical integration
        self.register("Trapezoidal", numerical::integration::trapezoidal);
        self.register("Simpson", numerical::integration::simpson);
        self.register("Romberg", numerical::integration::romberg);
        self.register("GaussQuadrature", numerical::integration::gauss_quadrature_fn);
        self.register("MonteCarlo", numerical::integration::monte_carlo);
        
        // Numerical differentiation
        self.register("FiniteDifference", numerical::differentiation::finite_difference);
        self.register("RichardsonExtrapolation", numerical::differentiation::richardson_extrapolation_fn);
        
        // Mesh generation (placeholders)
        self.register("DelaunayMesh", numerical::mesh::delaunay_mesh);
        self.register("VoronoiMesh", numerical::mesh::voronoi_mesh);
        self.register("UniformMesh", numerical::mesh::uniform_mesh);
        
        // Finite element components (placeholders)
        self.register("StiffnessMatrix", numerical::fem::stiffness_matrix);
        self.register("MassMatrix", numerical::fem::mass_matrix);
        self.register("LoadVector", numerical::fem::load_vector);
        
        // ODE/PDE solvers (placeholders)
        self.register("RungeKutta4", numerical::solvers::runge_kutta4);
        self.register("AdaptiveRK", numerical::solvers::adaptive_rk);
        self.register("CrankNicolson", numerical::solvers::crank_nicolson);
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
    }
    
    fn register_async_functions(&mut self) {
        // Async operations temporarily disabled due to compilation issues
        // TODO: Re-enable after fixing thread safety issues in async_ops module
    }

    fn register_network_functions(&mut self) {
        // Phase 12A: Core Network Primitives
        self.register("NetworkEndpoint", network::network_endpoint);
        self.register("NetworkRequest", network::network_request);
        self.register("NetworkAuth", network::network_auth);
        self.register("URLRead", network::url_read);
        self.register("URLWrite", network::url_write);
        self.register("URLStream", network::url_stream);
        self.register("NetworkPing", network::network_ping);
        self.register("DNSResolve", network::dns_resolve);
        self.register("HttpClient", network::http_client);
        
        // WebSocket operations
        self.register("WebSocket", network::websocket);
        self.register("WebSocketConnect", network::websocket_connect);
        self.register("WebSocketSend", network::websocket_send);
        self.register("WebSocketReceive", network::websocket_receive);
        self.register("WebSocketClose", network::websocket_close);
        self.register("WebSocketPing", network::websocket_ping);
        
        // Phase 12B: Event-Driven Architecture (placeholders)
        self.register("EventStream", network::event_stream);
        self.register("EventSubscribe", network::event_subscribe);
        self.register("EventPublish", network::event_publish);
        self.register("MessageQueue", network::message_queue);
        self.register("NetworkChannel", network::network_channel);
        
        // Phase 12C: Distributed Computing (placeholders)
        self.register("RemoteFunction", network::remote_function);
        self.register("RemoteFunctionCall", network::remote_function_call);
        self.register("DistributedMap", network::distributed_map);
        self.register("DistributedMapExecute", network::distributed_map_execute);
        self.register("DistributedReduce", network::distributed_reduce);
        self.register("DistributedReduceExecute", network::distributed_reduce_execute);
        self.register("ServiceRegistry", network::service_registry);
        self.register("ServiceDiscover", network::service_discover);
        self.register("ServiceHealthCheck", network::service_health_check);
        self.register("LoadBalancer", network::load_balancer);
        self.register("LoadBalancerRequest", network::load_balancer_request);
        self.register("ComputeCluster", network::compute_cluster);
        self.register("ClusterAddNode", network::cluster_add_node);
        self.register("ClusterSubmitTask", network::cluster_submit_task);
        self.register("ClusterGetStats", network::cluster_get_stats);
        
        // Phase 12D: Network Analysis (production implementations)
        self.register("NetworkGraph", network::network_graph);
        self.register("GraphAddNode", network::graph_add_node);
        self.register("GraphAddEdge", network::graph_add_edge);
        self.register("GraphShortestPath", network::graph_shortest_path);
        self.register("GraphMST", network::graph_mst);
        self.register("GraphComponents", network::graph_components);
        self.register("GraphMetrics", network::graph_metrics);
        
        // Centrality and community analysis
        self.register("NetworkCentrality", network::network_centrality);
        self.register("CommunityDetection", network::community_detection);
        self.register("GraphDiameter", network::graph_diameter);
        self.register("GraphDensity", network::graph_density);
        self.register("ClusteringCoefficient", network::clustering_coefficient);
        
        // Network flow algorithms
        self.register("NetworkFlow", network::network_flow);
        self.register("MinimumCut", network::minimum_cut);
        self.register("FlowDecomposition", network::flow_decomposition);
        self.register("FlowBottlenecks", network::flow_bottlenecks);
        self.register("MaxFlowValue", network::max_flow_value);
        
        // Network monitoring and diagnostics
        self.register("NetworkMonitor", network::network_monitor);
        self.register("MonitorStart", network::monitor_start);
        self.register("MonitorStop", network::monitor_stop);
        self.register("MonitorGetMetrics", network::monitor_get_metrics);
        self.register("MonitorSetAlerts", network::monitor_set_alerts);
        self.register("MonitorPing", network::monitor_ping);
        self.register("NetworkBottlenecks", network::network_bottlenecks);
        self.register("OptimizeTopology", network::optimize_topology);
        
        // Phase 12E: Cloud Integration
        self.register("CloudFunction", network::cloud_function);
        self.register("CloudFunctionDeploy", network::cloud_function_deploy);
        self.register("CloudFunctionInvoke", network::cloud_function_invoke);
        self.register("CloudFunctionUpdate", network::cloud_function_update);
        self.register("CloudFunctionLogs", network::cloud_function_logs);
        self.register("CloudFunctionMetrics", network::cloud_function_metrics);
        self.register("CloudFunctionDelete", network::cloud_function_delete);
        self.register("CloudStorage", network::cloud_storage);
        self.register("ContainerRun", network::container_run);
        self.register("KubernetesService", network::kubernetes_service);
        self.register("KubernetesDeploy", network::kubernetes_deploy);
        self.register("DeploymentScale", network::deployment_scale);
        self.register("RollingUpdate", network::rolling_update);
        self.register("ConfigMapCreate", network::configmap_create);
        self.register("ServiceExpose", network::service_expose);
        self.register("PodLogs", network::pod_logs);
        self.register("ResourceGet", network::resource_get);
        self.register("ResourceDelete", network::resource_delete);
        self.register("CloudDeploy", network::cloud_deploy);
        self.register("CloudMonitor", network::cloud_monitor);
        
        // Cloud Storage API Functions
        self.register("CloudUpload", network::cloud_upload);
        self.register("CloudDownload", network::cloud_download);
        self.register("CloudList", network::cloud_list);
        self.register("CloudDelete", network::cloud_delete);
        self.register("CloudMetadata", network::cloud_metadata);
        self.register("CloudPresignedURL", network::cloud_presigned_url);
        
        // Docker Container API Functions
        self.register("ContainerStop", network::container_stop);
        self.register("ContainerLogs", network::container_logs);
        self.register("ContainerInspect", network::container_inspect);
        self.register("ContainerExec", network::container_exec);
        self.register("ContainerList", network::container_list);
        self.register("ContainerPull", network::container_pull);
    }

    fn register_number_theory_functions(&mut self) {
        // Phase 13A: Advanced Number Theory (25 functions)
        
        // Prime Number Algorithms (8 functions)
        self.register("PrimeQ", number_theory::prime_q);
        self.register("NextPrime", number_theory::next_prime);
        self.register("PreviousPrime", number_theory::previous_prime);
        self.register("PrimePi", number_theory::prime_pi);
        self.register("PrimeFactorization", number_theory::prime_factorization);
        self.register("EulerPhi", number_theory::euler_phi_fn);
        self.register("MoebiusMu", number_theory::moebius_mu_fn);
        self.register("DivisorSigma", number_theory::divisor_sigma_fn);
        
        // Algebraic Number Theory (7 functions)
        self.register("GCD", number_theory::gcd_fn);
        self.register("LCM", number_theory::lcm_fn);
        self.register("ChineseRemainder", number_theory::chinese_remainder);
        self.register("JacobiSymbol", number_theory::jacobi_symbol_fn);
        self.register("ContinuedFraction", number_theory::continued_fraction_fn);
        self.register("AlgebraicNumber", number_theory::algebraic_number);
        self.register("MinimalPolynomial", number_theory::minimal_polynomial);
        
        // Modular Arithmetic (6 functions)
        self.register("PowerMod", number_theory::power_mod_fn);
        self.register("ModularInverse", number_theory::modular_inverse_fn);
        self.register("DiscreteLog", number_theory::discrete_log_fn);
        self.register("QuadraticResidue", number_theory::quadratic_residue_fn);
        self.register("PrimitiveRoot", number_theory::primitive_root_fn);
        self.register("MultOrder", number_theory::mult_order_fn);
        
        // Cryptographic Primitives (4 functions)
        self.register("RSAGenerate", number_theory::rsa_generate);
        self.register("ECPoint", number_theory::ec_point);
        self.register("HashFunction", number_theory::hash_function);
        self.register("RandomPrime", number_theory::random_prime);
    }

    fn register_combinatorics_functions(&mut self) {
        // Phase 13B: Combinatorics and Advanced Graph Algorithms
        
        // Basic Combinatorial Functions (4 functions)
        self.register("Binomial", combinatorics::binomial_fn);
        self.register("Multinomial", combinatorics::multinomial_fn);
        self.register("Permutations", combinatorics::permutations_fn);
        self.register("Combinations", combinatorics::combinations_fn);
        
        // Advanced Combinatorial Functions (4 functions)
        self.register("StirlingNumber", combinatorics::stirling_number_fn);
        self.register("BellNumber", combinatorics::bell_number_fn);
        self.register("CatalanNumber", combinatorics::catalan_number_fn);
        self.register("Partitions", combinatorics::partitions_fn);
        
        // Combinatorial Sequences (5 functions)
        self.register("FibonacciNumber", combinatorics::fibonacci_number_fn);
        self.register("LucasNumber", combinatorics::lucas_number_fn);
        self.register("TribonacciNumber", combinatorics::tribonacci_number_fn);
        self.register("PellNumber", combinatorics::pell_number_fn);
        self.register("JacobsthalNumber", combinatorics::jacobsthal_number_fn);
        
        // Advanced Graph Algorithms (12 functions)
        // Connectivity algorithms
        self.register("MinimumSpanningTree", graph::minimum_spanning_tree);
        self.register("MaximumFlow", graph::maximum_flow);
        self.register("ArticulationPoints", graph::articulation_points);
        self.register("Bridges", graph::bridges);
        
        // Optimization algorithms
        self.register("GraphColoring", graph::graph_coloring);
        self.register("HamiltonianPath", graph::hamiltonian_path);
        self.register("EulerianPath", graph::eulerian_path);
        self.register("VertexCover", graph::vertex_cover);
        self.register("IndependentSet", graph::independent_set);
        
        // Analysis algorithms
        self.register("BetweennessCentrality", graph::betweenness_centrality);
        self.register("ClosenessCentrality", graph::closeness_centrality);
        self.register("PageRank", graph::pagerank);
        self.register("HITS", graph::hits);
        self.register("CommunityDetection", graph::community_detection);
        self.register("GraphIsomorphism", graph::graph_isomorphism);
    }

    fn register_geometry_functions(&mut self) {
        // Phase 13C.1: Computational Geometry (10 functions)
        
        // Core Geometric Primitives (4 functions)
        self.register("ConvexHull", geometry::convex_hull::convex_hull_fn);
        self.register("VoronoiDiagram", geometry::triangulation::voronoi_diagram_fn);
        self.register("DelaunayTriangulation", geometry::triangulation::delaunay_triangulation_fn);
        self.register("MinkowskiSum", geometry::operations::minkowski_sum_fn);
        
        // Geometric Queries & Analysis (3 functions)
        self.register("PointInPolygon", geometry::queries::point_in_polygon_fn);
        self.register("PolygonIntersection", geometry::queries::polygon_intersection_fn);
        self.register("ClosestPair", geometry::queries::closest_pair_fn);
        
        // Advanced Geometric Operations (3 functions)
        self.register("GeometricMedian", geometry::operations::geometric_median_fn);
        self.register("ShapeMatching", geometry::operations::shape_matching_fn);
        self.register("PolygonDecomposition", geometry::operations::polygon_decomposition_fn);
    }

    fn register_topology_functions(&mut self) {
        // Phase 13C.2: Topological Data Analysis (8 functions)
        
        // Persistent Homology (3 functions)
        self.register("PersistentHomology", topology::homology::persistent_homology_fn);
        self.register("BettiNumbers", topology::homology::betti_numbers_fn);
        self.register("PersistenceDiagram", topology::homology::persistence_diagram_fn);
        
        // Simplicial Complexes (3 functions)
        self.register("SimplicialComplex", topology::complexes::simplicial_complex_fn);
        self.register("VietorisRips", topology::complexes::vietoris_rips_fn);
        self.register("CechComplex", topology::complexes::cech_complex_fn);
        
        // Topological Analysis (2 functions)
        self.register("TopologicalFeatures", topology::analysis::topological_features_fn);
        self.register("MapperAlgorithm", topology::analysis::mapper_algorithm_fn);
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
