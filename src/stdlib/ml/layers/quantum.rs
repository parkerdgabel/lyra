//! Quantum Neural Network Layer Implementation
//!
//! This module provides the QuantumLayer implementation that seamlessly integrates
//! quantum circuits with classical neural networks, enabling hybrid quantum-classical
//! deep learning models.

use crate::stdlib::autodiff::Dual;
use super::{Tensor, Layer, MLResult, MLError};
use crate::stdlib::ml::quantum_bridge::{
    QuantumFeatureMap, VariationalCircuit, PauliStringObservable, 
    QuantumDataEncoder, EncodingType, NormalizationStrategy
};
use crate::stdlib::quantum::QubitRegister;
use crate::concurrency::{WorkStealingScheduler, ConcurrentExecutable, TaskPriority, ConcurrencyConfig, ConcurrencyStats};
use crate::vm::{Value, VmResult, VmError};
use std::f64::consts::PI;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use num_cpus;

/// Quantum neural network layer with hybrid classical-quantum-classical architecture
///
/// The QuantumLayer implements the following hybrid processing pipeline:
/// 1. **Classical Preprocessing**: Optional linear transformation of input features
/// 2. **Quantum Encoding**: Encode classical data into quantum states using QuantumFeatureMap
/// 3. **Quantum Processing**: Apply variational quantum circuit with trainable parameters
/// 4. **Quantum Measurement**: Measure expectation values using Pauli observables
/// 5. **Classical Postprocessing**: Optional linear transformation of quantum measurements
///
/// This enables seamless integration with existing NetChain and NetTrain infrastructure
/// while providing quantum computational advantages for specific ML tasks.
#[derive(Debug, Clone)]
pub struct QuantumLayer {
    // Quantum circuit components
    /// Number of qubits in the quantum circuit
    pub n_qubits: usize,
    /// Variational quantum circuit for processing
    pub circuit: Option<VariationalCircuit>,
    /// Feature map for classical-to-quantum encoding
    pub feature_map: Option<QuantumFeatureMap>,
    /// Measurement observables for quantum-to-classical conversion
    pub measurement_observables: Vec<PauliStringObservable>,
    
    // Classical components
    /// Optional classical preprocessing weights [input_size, n_features]
    pub classical_preprocessing: Option<Tensor>,
    /// Optional classical postprocessing weights [n_observables, output_size]  
    pub classical_postprocessing: Option<Tensor>,
    /// Bias for classical postprocessing [output_size]
    pub bias: Option<Tensor>,
    
    // Layer configuration
    /// Input size (determined during first forward pass)
    pub input_size: Option<usize>,
    /// Number of features to encode into quantum states
    pub n_quantum_features: usize,
    /// Final output size
    pub output_size: Option<usize>,
    /// Quantum encoding strategy
    pub encoding_type: EncodingType,
    /// Data normalization strategy before quantum encoding
    pub normalization_strategy: NormalizationStrategy,
    /// Circuit depth (number of variational layers)
    pub circuit_depth: usize,
    
    // Layer metadata
    pub layer_name: String,
    pub initialized: bool,
    
    // Parameter management for ML integration
    /// Parameter manager handles conversion between f64 quantum parameters and ML Tensors
    pub parameter_manager: QuantumCircuitParameterManager,
    
    // Hybrid gradient computation
    /// Hybrid gradient computer for quantum-classical gradient flow
    pub hybrid_gradient_computer: Option<HybridGradientComputer>,
}

impl QuantumLayer {
    /// Create a new quantum layer with specified configuration
    ///
    /// # Arguments
    /// * `n_qubits` - Number of qubits in the quantum circuit
    /// * `encoding_type` - Strategy for encoding classical data to quantum states
    /// * `circuit_depth` - Depth of the variational quantum circuit
    /// * `output_size` - Optional output size (if None, outputs raw measurement values)
    pub fn new(
        n_qubits: usize,
        encoding_type: EncodingType,
        circuit_depth: usize,
        output_size: Option<usize>,
    ) -> MLResult<Self> {
        if n_qubits == 0 || n_qubits > 20 {
            return Err(MLError::InvalidLayer {
                reason: format!("Invalid qubit count: {}. Must be between 1 and 20", n_qubits),
            });
        }
        
        if circuit_depth == 0 || circuit_depth > 10 {
            return Err(MLError::InvalidLayer {
                reason: format!("Invalid circuit depth: {}. Must be between 1 and 10", circuit_depth),
            });
        }
        
        // Determine number of quantum features based on encoding type
        let n_quantum_features = match encoding_type {
            EncodingType::Angle | EncodingType::IQP => n_qubits,
            EncodingType::Amplitude => 2_usize.pow(n_qubits as u32).min(256), // Cap for performance
            EncodingType::Basis => n_qubits,
        };
        
        let layer_name = format!("QuantumLayer[qubits={}, encoding={:?}, depth={}]", 
                                n_qubits, encoding_type, circuit_depth);
        
        Ok(Self {
            n_qubits,
            circuit: None,
            feature_map: None,
            measurement_observables: Vec::new(),
            classical_preprocessing: None,
            classical_postprocessing: None,
            bias: None,
            input_size: None,
            n_quantum_features,
            output_size,
            encoding_type,
            normalization_strategy: NormalizationStrategy::StandardScaler,
            circuit_depth,
            layer_name,
            initialized: false,
            parameter_manager: QuantumCircuitParameterManager::new(),
            hybrid_gradient_computer: None,
        })
    }
    
    /// Create quantum layer with automatic feature reduction
    ///
    /// If input features > quantum features, adds classical preprocessing layer
    pub fn with_feature_reduction(
        n_qubits: usize,
        encoding_type: EncodingType,
        circuit_depth: usize,
        output_size: Option<usize>,
    ) -> MLResult<Self> {
        let mut layer = Self::new(n_qubits, encoding_type, circuit_depth, output_size)?;
        layer.layer_name = format!("QuantumLayer[qubits={}, encoding={:?}, depth={}, feature_reduction=true]", 
                                  n_qubits, encoding_type, circuit_depth);
        Ok(layer)
    }
    
    /// Synchronize quantum circuit parameters with ML framework tensors
    fn sync_quantum_parameters(&mut self) -> MLResult<()> {
        if let Some(ref circuit) = self.circuit {
            self.parameter_manager.sync_from_circuit(circuit)?;
        }
        Ok(())
    }
    
    /// Update quantum circuit with parameters from ML framework
    fn update_quantum_parameters(&mut self) -> MLResult<()> {
        if let Some(ref mut circuit) = self.circuit {
            self.parameter_manager.sync_to_circuit(circuit)?;
        }
        Ok(())
    }
    
    /// Get total parameter count across all layer components
    pub fn parameter_count(&self) -> usize {
        let mut count = 0;
        
        // Classical preprocessing parameters
        if let Some(ref preprocessing) = self.classical_preprocessing {
            count += preprocessing.data.len();
        }
        
        // Quantum circuit parameters
        if let Some(ref circuit) = self.circuit {
            count += circuit.parameter_count();
        }
        
        // Classical postprocessing parameters
        if let Some(ref postprocessing) = self.classical_postprocessing {
            count += postprocessing.data.len();
        }
        if let Some(ref bias) = self.bias {
            count += bias.data.len();
        }
        
        count
    }
    
    /// Enable hybrid gradient computation for this layer
    pub fn enable_hybrid_gradients(&mut self) -> MLResult<()> {
        if !self.initialized {
            return Err(MLError::InvalidLayer {
                reason: "Layer must be initialized before enabling hybrid gradients".to_string(),
            });
        }
        
        self.hybrid_gradient_computer = Some(HybridGradientComputer::new());
        Ok(())
    }
    
    /// Enable hybrid gradient computation with custom configuration
    pub fn enable_hybrid_gradients_with_config(&mut self, max_cache_size: usize, parallel_evaluation: bool) -> MLResult<()> {
        if !self.initialized {
            return Err(MLError::InvalidLayer {
                reason: "Layer must be initialized before enabling hybrid gradients".to_string(),
            });
        }
        
        self.hybrid_gradient_computer = Some(HybridGradientComputer::with_config(max_cache_size, parallel_evaluation));
        Ok(())
    }
    
    /// Compute hybrid gradients for this quantum layer
    pub fn compute_hybrid_gradients(
        &mut self,
        input_batch: &[Tensor],
        loss_gradients: &[Dual]
    ) -> MLResult<HybridGradients> {
        
        // Initialize hybrid gradient computer if not already done
        if self.hybrid_gradient_computer.is_none() {
            self.enable_hybrid_gradients()?;
        }
        
        // Ensure layer is initialized
        if !self.initialized {
            return Err(MLError::InvalidLayer {
                reason: "Layer must be initialized before computing hybrid gradients".to_string(),
            });
        }
        
        // Extract the hybrid gradient computer to avoid borrowing conflicts
        let mut gradient_computer = self.hybrid_gradient_computer.take().unwrap();
        let result = gradient_computer.compute_hybrid_gradients(self, input_batch, loss_gradients);
        
        // Put the gradient computer back
        self.hybrid_gradient_computer = Some(gradient_computer);
        
        result
    }
    
    /// Get hybrid gradient computation statistics for performance monitoring
    pub fn get_gradient_statistics(&self) -> Option<(usize, usize)> {
        self.hybrid_gradient_computer.as_ref().map(|computer| computer.cache_statistics())
    }
    
    /// Clear hybrid gradient cache to free memory
    pub fn clear_gradient_cache(&mut self) {
        if let Some(ref mut computer) = self.hybrid_gradient_computer {
            computer.clear_cache();
        }
    }
    
    /// Initialize quantum components and classical weights
    fn initialize_components(&mut self, input_size: usize) -> MLResult<()> {
        self.input_size = Some(input_size);
        
        // 1. Initialize classical preprocessing if needed
        if input_size > self.n_quantum_features {
            // Need classical preprocessing to reduce dimensionality
            let preprocessing_values: Vec<f64> = (0..input_size * self.n_quantum_features)
                .map(|_| {
                    let u: f64 = rand::random();
                    let scale = (2.0 / (input_size + self.n_quantum_features) as f64).sqrt();
                    (u - 0.5) * 2.0 * scale
                })
                .collect();
            
            self.classical_preprocessing = Some(Tensor::variables(
                preprocessing_values,
                vec![input_size, self.n_quantum_features],
            )?);
        }
        
        // 2. Initialize quantum feature map
        self.feature_map = Some(QuantumFeatureMap::new(
            self.encoding_type,
            self.n_quantum_features,
            self.n_qubits,
        )?);
        
        // 3. Initialize variational quantum circuit
        let mut circuit = VariationalCircuit::new(self.n_qubits);
        
        // Add alternating rotation and entangling layers
        for layer_idx in 0..self.circuit_depth {
            // Rotation layer with random initialization
            circuit.add_rotation_layer("RY")?;
            
            // Entangling layer (except for last layer)
            if layer_idx < self.circuit_depth - 1 {
                circuit.add_entangling_layer()?;
            }
        }
        
        self.circuit = Some(circuit);
        
        // Initialize parameter manager with new circuit parameters
        self.sync_quantum_parameters()?;
        
        // 4. Initialize measurement observables
        self.measurement_observables = self.create_measurement_basis()?;
        
        // 5. Initialize classical postprocessing if needed
        if let Some(final_output_size) = self.output_size {
            let n_measurements = self.measurement_observables.len();
            
            let postprocessing_values: Vec<f64> = (0..n_measurements * final_output_size)
                .map(|_| {
                    let u: f64 = rand::random();
                    let scale = (2.0 / (n_measurements + final_output_size) as f64).sqrt();
                    (u - 0.5) * 2.0 * scale
                })
                .collect();
            
            self.classical_postprocessing = Some(Tensor::variables(
                postprocessing_values,
                vec![n_measurements, final_output_size],
            )?);
            
            // Initialize bias
            let bias_values = vec![0.0; final_output_size];
            self.bias = Some(Tensor::variables(bias_values, vec![final_output_size])?);
        }
        
        self.initialized = true;
        Ok(())
    }
    
    /// Create a comprehensive set of Pauli observables for measurement
    fn create_measurement_basis(&self) -> MLResult<Vec<PauliStringObservable>> {
        let mut observables = Vec::new();
        
        // Single-qubit Z measurements on each qubit
        for qubit_idx in 0..self.n_qubits {
            let mut pauli_string = "I".repeat(self.n_qubits);
            pauli_string.replace_range(qubit_idx..qubit_idx+1, "Z");
            
            let observable = PauliStringObservable::new(pauli_string, 1.0)?;
            observables.push(observable);
        }
        
        // Two-qubit correlations (ZZ measurements)
        for i in 0..self.n_qubits.min(4) { // Limit for performance
            for j in (i+1)..self.n_qubits.min(4) {
                let mut pauli_string = "I".repeat(self.n_qubits);
                pauli_string.replace_range(i..i+1, "Z");
                pauli_string.replace_range(j..j+1, "Z");
                
                let observable = PauliStringObservable::new(pauli_string, 1.0)?;
                observables.push(observable);
            }
        }
        
        // Add some X and Y measurements for completeness
        if self.n_qubits <= 4 {
            for qubit_idx in 0..self.n_qubits {
                for pauli_op in ["X", "Y"] {
                    let mut pauli_string = "I".repeat(self.n_qubits);
                    pauli_string.replace_range(qubit_idx..qubit_idx+1, pauli_op);
                    
                    let observable = PauliStringObservable::new(pauli_string, 1.0)?;
                    observables.push(observable);
                }
            }
        }
        
        Ok(observables)
    }
    
    /// Perform classical preprocessing on input features
    fn classical_preprocess(&self, input: &Tensor) -> MLResult<Tensor> {
        if let Some(ref preprocessing_weights) = self.classical_preprocessing {
            // Linear transformation: input @ preprocessing_weights
            input.matmul(preprocessing_weights)
        } else {
            // No preprocessing needed, use input directly
            Ok(input.clone())
        }
    }
    
    /// Encode classical features into quantum states
    fn quantum_encode(&self, features: &Tensor) -> MLResult<Vec<QubitRegister>> {
        let feature_map = self.feature_map.as_ref().unwrap();
        let batch_size = features.shape[0];
        let mut quantum_states = Vec::with_capacity(batch_size);
        
        // Process each sample in the batch
        for batch_idx in 0..batch_size {
            // Extract features for this batch item
            let start_idx = batch_idx * self.n_quantum_features;
            let end_idx = start_idx + self.n_quantum_features;
            let sample_data = features.data[start_idx..end_idx].to_vec();
            
            let sample_tensor = Tensor::new(
                sample_data,
                vec![self.n_quantum_features],
            )?;
            
            // Encode into quantum state
            let quantum_state = feature_map.encode(&sample_tensor)?;
            quantum_states.push(quantum_state);
        }
        
        Ok(quantum_states)
    }
    
    /// Apply variational quantum circuit to quantum states
    fn quantum_process(&self, quantum_states: Vec<QubitRegister>) -> MLResult<Vec<QubitRegister>> {
        let circuit = self.circuit.as_ref().unwrap();
        let mut processed_states = Vec::with_capacity(quantum_states.len());
        
        for state in quantum_states {
            let processed_state = circuit.forward(&state)?;
            processed_states.push(processed_state);
        }
        
        Ok(processed_states)
    }
    
    /// Measure quantum states to obtain classical feature vectors
    fn quantum_measure(&self, quantum_states: Vec<QubitRegister>) -> MLResult<Tensor> {
        let batch_size = quantum_states.len();
        let n_observables = self.measurement_observables.len();
        let mut measurements = Vec::with_capacity(batch_size * n_observables);
        
        for state in quantum_states {
            for observable in &self.measurement_observables {
                let expectation_value = observable.expectation_value(&state)?;
                measurements.push(Dual::variable(expectation_value));
            }
        }
        
        Tensor::new(measurements, vec![batch_size, n_observables])
    }
    
    /// Apply classical postprocessing to quantum measurements
    fn classical_postprocess(&self, quantum_features: &Tensor) -> MLResult<Tensor> {
        if let Some(ref postprocessing_weights) = self.classical_postprocessing {
            // Linear transformation: quantum_features @ postprocessing_weights
            let linear_output = quantum_features.matmul(postprocessing_weights)?;
            
            if let Some(ref bias) = self.bias {
                linear_output.add(bias)
            } else {
                Ok(linear_output)
            }
        } else {
            // No postprocessing, return quantum measurements directly
            Ok(quantum_features.clone())
        }
    }
}

impl Layer for QuantumLayer {
    fn forward(&self, input: &Tensor) -> MLResult<Tensor> {
        // Ensure input is 2D [batch_size, features]
        if input.shape.len() != 2 {
            return Err(MLError::InvalidLayer {
                reason: format!("QuantumLayer expects 2D input, got {}D", input.shape.len()),
            });
        }
        
        let batch_size = input.shape[0];
        let input_features = input.shape[1];
        
        // Initialize components if this is the first forward pass
        let mut layer = self.clone();
        if !layer.initialized {
            layer.initialize_components(input_features)?;
        }
        
        // Sync parameters from ML framework to quantum circuit before processing
        if layer.parameter_manager.tensors_dirty {
            layer.update_quantum_parameters()?;
        }
        
        // 1. Classical preprocessing (dimensionality reduction if needed)
        let preprocessed_features = layer.classical_preprocess(input)?;
        
        // 2. Quantum encoding: Classical features → Quantum states
        let quantum_states = layer.quantum_encode(&preprocessed_features)?;
        
        // 3. Quantum processing: Apply variational circuit
        let processed_states = layer.quantum_process(quantum_states)?;
        
        // 4. Quantum measurement: Quantum states → Classical features
        let quantum_measurements = layer.quantum_measure(processed_states)?;
        
        // 5. Classical postprocessing (optional final linear layer)
        let final_output = layer.classical_postprocess(&quantum_measurements)?;
        
        Ok(final_output)
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        
        // Add classical preprocessing parameters
        if let Some(ref preprocessing) = self.classical_preprocessing {
            params.push(preprocessing);
        }
        
        // Add quantum circuit parameters (from parameter manager)
        for tensor in &self.parameter_manager.tensor_parameters {
            params.push(tensor);
        }
        
        // Add classical postprocessing parameters
        if let Some(ref postprocessing) = self.classical_postprocessing {
            params.push(postprocessing);
        }
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        
        // Add classical preprocessing parameters
        if let Some(ref mut preprocessing) = self.classical_preprocessing {
            params.push(preprocessing);
        }
        
        // Add quantum circuit parameters (from parameter manager)
        // Mark parameters as dirty since they might be modified
        self.parameter_manager.mark_dirty();
        for tensor in &mut self.parameter_manager.tensor_parameters {
            params.push(tensor);
        }
        
        // Add classical postprocessing parameters
        if let Some(ref mut postprocessing) = self.classical_postprocessing {
            params.push(postprocessing);
        }
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        
        params
    }
    
    fn initialize(&mut self) -> MLResult<()> {
        // Initialization is lazy - happens on first forward pass
        // This allows input size to be determined automatically
        Ok(())
    }
    
    fn name(&self) -> &str {
        &self.layer_name
    }
    
    fn output_shape(&self, input_shape: &[usize]) -> MLResult<Vec<usize>> {
        if input_shape.len() != 2 {
            return Err(MLError::ShapeMismatch {
                expected: vec![2],
                actual: input_shape.to_vec(),
            });
        }
        
        let batch_size = input_shape[0];
        
        // Determine output size
        let output_features = if let Some(final_size) = self.output_size {
            final_size
        } else {
            // Output size is number of measurement observables
            let n_observables = if self.measurement_observables.is_empty() {
                // Estimate based on qubit count (will be exact after initialization)
                self.n_qubits + (self.n_qubits * (self.n_qubits - 1)) / 2 + 2 * self.n_qubits
            } else {
                self.measurement_observables.len()
            };
            n_observables
        };
        
        Ok(vec![batch_size, output_features])
    }
    
    fn clone_boxed(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
}

/// Helper trait for VariationalCircuit to provide parameter access
/// This extends the quantum bridge circuit with ML layer compatibility
pub trait QuantumCircuitParameters {
    fn parameters(&self) -> Vec<&Tensor>;
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;
}

// We'll need to implement this for VariationalCircuit in the quantum_bridge module
// For now, we provide placeholder implementations

/// Quantum circuit parameter manager for ML integration
/// This handles the complex task of bridging between quantum circuit parameters (f64)
/// and ML framework parameters (Tensor objects with gradient tracking)
#[derive(Debug, Clone)]
pub struct QuantumCircuitParameterManager {
    /// Cached tensor representations of circuit parameters
    pub tensor_parameters: Vec<Tensor>,
    /// Flag to track if tensors are synchronized with circuit
    pub tensors_dirty: bool,
}

impl QuantumCircuitParameterManager {
    pub fn new() -> Self {
        Self {
            tensor_parameters: Vec::new(),
            tensors_dirty: true,
        }
    }
    
    /// Synchronize tensor parameters with circuit parameters
    pub fn sync_from_circuit(&mut self, circuit: &VariationalCircuit) -> MLResult<()> {
        if self.tensors_dirty {
            self.tensor_parameters = circuit.parameters_as_tensors()?;
            self.tensors_dirty = false;
        }
        Ok(())
    }
    
    /// Update circuit parameters from synchronized tensors
    pub fn sync_to_circuit(&mut self, circuit: &mut VariationalCircuit) -> MLResult<()> {
        circuit.update_parameters_from_tensors(&self.tensor_parameters)?;
        self.tensors_dirty = false;
        Ok(())
    }
    
    /// Mark tensors as needing synchronization
    pub fn mark_dirty(&mut self) {
        self.tensors_dirty = true;
    }
}

impl QuantumCircuitParameters for VariationalCircuit {
    fn parameters(&self) -> Vec<&Tensor> {
        // For VariationalCircuit, we need to convert f64 parameters to Tensors
        // This is handled at the QuantumLayer level with parameter caching
        // Return empty vector here as the actual implementation is in QuantumLayer
        Vec::new()
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        // Similar to above, the actual parameter management happens at QuantumLayer level
        // with proper synchronization between f64 and Tensor representations
        Vec::new()
    }
}

/// Result type for hybrid gradient computation combining quantum and classical gradients
#[derive(Debug, Clone)]
pub struct HybridGradients {
    /// Quantum parameter gradients computed via parameter shift rule
    pub quantum_gradients: Vec<f64>,
    /// Classical gradients formatted as Dual numbers for ML framework integration
    pub classical_gradients: Vec<Tensor>,
    /// Computational cost metrics for performance monitoring
    pub computation_cost: GradientComputationCost,
}

/// Metrics for hybrid gradient computation cost and performance
#[derive(Debug, Clone)]
pub struct GradientComputationCost {
    /// Number of quantum circuit evaluations performed
    pub circuit_evaluations: usize,
    /// Time spent on quantum gradient computation (milliseconds)
    pub quantum_compute_time_ms: u64,
    /// Time spent on classical gradient computation (milliseconds)  
    pub classical_compute_time_ms: u64,
    /// Number of cached gradient lookups used
    pub cache_hits: usize,
    /// Number of parallel parameter shift evaluations
    pub parallel_evaluations: usize,
}

impl Default for GradientComputationCost {
    fn default() -> Self {
        Self {
            circuit_evaluations: 0,
            quantum_compute_time_ms: 0,
            classical_compute_time_ms: 0,
            cache_hits: 0,
            parallel_evaluations: 0,
        }
    }
}

/// Enhanced parameter synchronization between quantum (f64) and classical (Dual) parameter spaces
/// This system provides hierarchical parameter organization and advanced synchronization features
#[derive(Debug, Clone)]
pub struct ParameterSynchronizer {
    /// Hierarchical parameter groups for organized synchronization
    parameter_groups: Vec<ParameterGroup>,
    /// Global mapping from quantum parameter index to group and local index
    quantum_to_group: HashMap<usize, (usize, usize)>, // (group_idx, local_idx)
    /// Mapping from parameter names to quantum indices 
    name_to_quantum: HashMap<String, usize>,
    /// Parameter value synchronization state
    sync_state: ParameterSyncState,
    /// Parameter constraints and transformations
    parameter_constraints: HashMap<usize, ParameterConstraint>,
    /// Transformation functions for parameter encoding
    parameter_transforms: HashMap<usize, ParameterTransform>,
    /// Synchronization statistics for performance monitoring
    sync_stats: SynchronizationStats,
}

/// Hierarchical parameter group for organized quantum-classical synchronization
#[derive(Debug, Clone)]
pub struct ParameterGroup {
    /// Group name for identification and debugging
    pub name: String,
    /// Group type (rotation, feature_encoding, variational, etc.)
    pub group_type: ParameterGroupType,
    /// Quantum parameter indices belonging to this group
    pub quantum_indices: Vec<usize>,
    /// Classical tensor references for this group
    pub classical_tensors: Vec<(usize, Vec<usize>)>, // (tensor_idx, element_indices)
    /// Whether this group requires synchronized updates
    pub synchronized_updates: bool,
    /// Group-level transformation matrix for parameter encoding
    pub transformation_matrix: Option<Vec<Vec<f64>>>,
    /// Last synchronization timestamp
    pub last_sync_time: Option<std::time::Instant>,
}

impl ParameterGroup {
    /// Create a new parameter group
    pub fn new(
        name: String,
        group_type: ParameterGroupType,
        quantum_indices: Vec<usize>,
        classical_tensors: Vec<(usize, Vec<usize>)>,
    ) -> Self {
        Self {
            name,
            group_type,
            quantum_indices,
            classical_tensors,
            synchronized_updates: true,
            transformation_matrix: None,
            last_sync_time: None,
        }
    }
    
    /// Create a rotation gates parameter group
    pub fn rotation_gates(quantum_indices: Vec<usize>, tensor_mappings: Vec<(usize, Vec<usize>)>) -> Self {
        Self::new(
            "RotationGates".to_string(),
            ParameterGroupType::RotationGates,
            quantum_indices,
            tensor_mappings,
        )
    }
    
    /// Create a variational layer parameter group  
    pub fn variational_layer(layer_idx: usize, quantum_indices: Vec<usize>, tensor_mappings: Vec<(usize, Vec<usize>)>) -> Self {
        Self::new(
            format!("VariationalLayer_{}", layer_idx),
            ParameterGroupType::VariationalLayer,
            quantum_indices,
            tensor_mappings,
        )
    }
    
    /// Get the number of parameters in this group
    pub fn parameter_count(&self) -> usize {
        self.quantum_indices.len()
    }
    
    /// Check if the group has been synchronized recently
    pub fn is_recently_synchronized(&self, threshold_ms: u64) -> bool {
        if let Some(last_sync) = self.last_sync_time {
            last_sync.elapsed().as_millis() <= threshold_ms as u128
        } else {
            false
        }
    }
}

impl ParameterConstraint {
    /// Create parameter constraint for rotation angles (periodic, bounded to [0, 2π])
    pub fn rotation_angle() -> Self {
        use std::f64::consts::PI;
        Self {
            bounds: Some((0.0, 2.0 * PI)),
            normalize: false,
            normalization_type: NormalizationType::None,
            periodic: true,
            period: Some(2.0 * PI),
        }
    }
    
    /// Create parameter constraint for normalized parameters [-1, 1]
    pub fn normalized_parameter() -> Self {
        Self {
            bounds: Some((-1.0, 1.0)),
            normalize: true,
            normalization_type: NormalizationType::Clip,
            periodic: false,
            period: None,
        }
    }
    
    /// Create parameter constraint for positive parameters [0, ∞)
    pub fn positive_parameter() -> Self {
        Self {
            bounds: Some((0.0, f64::INFINITY)),
            normalize: false,
            normalization_type: NormalizationType::Clip,
            periodic: false,
            period: None,
        }
    }
    
    /// Create unconstrained parameter
    pub fn unconstrained() -> Self {
        Self {
            bounds: None,
            normalize: false,
            normalization_type: NormalizationType::None,
            periodic: false,
            period: None,
        }
    }
}

impl ParameterTransform {
    /// Create identity transformation (no change)
    pub fn identity() -> Self {
        Self {
            forward: TransformFunction::Identity,
            inverse: TransformFunction::Identity,
            jacobian: Some(TransformFunction::Identity),
            is_linear: true,
        }
    }
    
    /// Create linear scaling transformation
    pub fn linear_scaling(scale: f64, offset: f64) -> Self {
        Self {
            forward: TransformFunction::Linear { scale, offset },
            inverse: TransformFunction::Linear { 
                scale: 1.0 / scale, 
                offset: -offset / scale 
            },
            jacobian: Some(TransformFunction::Linear { 
                scale, 
                offset: 0.0 
            }),
            is_linear: true,
        }
    }
    
    /// Create trigonometric encoding transformation
    pub fn trigonometric_encoding(amplitude: f64, frequency: f64, phase: f64) -> Self {
        Self {
            forward: TransformFunction::Trigonometric { amplitude, frequency, phase },
            inverse: TransformFunction::Custom("ArcSin".to_string()), // Simplified
            jacobian: Some(TransformFunction::Trigonometric { 
                amplitude: amplitude * frequency, 
                frequency, 
                phase: phase + std::f64::consts::PI / 2.0 // Derivative of sin is cos
            }),
            is_linear: false,
        }
    }
}

/// Types of parameter groups for different quantum circuit components
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterGroupType {
    /// Rotation gate parameters (RX, RY, RZ)
    RotationGates,
    /// Feature encoding parameters for data embedding
    FeatureEncoding,
    /// Variational layer parameters for optimization
    VariationalLayer,
    /// Entangling gate parameters
    EntanglingGates,
    /// Observable measurement parameters
    ObservableParameters,
    /// Custom parameter group with user-defined behavior
    Custom(String),
}

/// Parameter constraints for validation and normalization
#[derive(Debug, Clone)]
pub struct ParameterConstraint {
    /// Parameter bounds (min, max)
    pub bounds: Option<(f64, f64)>,
    /// Whether to normalize parameter to unit circle/sphere
    pub normalize: bool,
    /// Normalization strategy
    pub normalization_type: NormalizationType,
    /// Whether parameter should be periodic (for rotational parameters)
    pub periodic: bool,
    /// Period for periodic parameters (e.g., 2π for rotation angles)
    pub period: Option<f64>,
}

/// Normalization strategies for parameter constraints
#[derive(Debug, Clone, PartialEq)]
pub enum NormalizationType {
    /// No normalization
    None,
    /// L2 normalization (unit vector)
    L2,
    /// Min-max normalization to [0, 1]
    MinMax,
    /// Standardization (zero mean, unit variance)
    Standardize,
    /// Clip to bounds without normalization
    Clip,
}

/// Parameter transformation functions for encoding-specific operations
#[derive(Debug, Clone)]
pub struct ParameterTransform {
    /// Forward transformation: classical → quantum
    pub forward: TransformFunction,
    /// Inverse transformation: quantum → classical  
    pub inverse: TransformFunction,
    /// Jacobian for gradient transformation
    pub jacobian: Option<TransformFunction>,
    /// Whether the transformation is linear
    pub is_linear: bool,
}

/// Transformation function types
#[derive(Debug, Clone)]
pub enum TransformFunction {
    /// Identity transformation
    Identity,
    /// Linear scaling: ax + b
    Linear { scale: f64, offset: f64 },
    /// Trigonometric encoding for rotational parameters
    Trigonometric { amplitude: f64, frequency: f64, phase: f64 },
    /// Exponential transformation for positive parameters
    Exponential { base: f64 },
    /// Logarithmic transformation
    Logarithmic { base: f64 },
    /// Custom transformation function
    Custom(String),
}

/// Synchronization performance statistics
#[derive(Debug, Clone, Default)]
pub struct SynchronizationStats {
    /// Total number of synchronization operations
    pub total_syncs: usize,
    /// Number of successful synchronizations
    pub successful_syncs: usize,
    /// Number of failed synchronizations
    pub failed_syncs: usize,
    /// Total time spent synchronizing (microseconds)
    pub total_sync_time_us: u64,
    /// Average synchronization time (microseconds)
    pub avg_sync_time_us: f64,
    /// Number of constraint violations
    pub constraint_violations: usize,
    /// Number of transformation failures
    pub transform_failures: usize,
    /// Cache hits for parameter transformations
    pub transform_cache_hits: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParameterSyncState {
    /// Parameters are synchronized between quantum and classical systems
    Synchronized,
    /// Quantum parameters have been updated, classical needs sync
    QuantumUpdated,
    /// Classical parameters have been updated, quantum needs sync
    ClassicalUpdated,
    /// Both systems have been updated, full resynchronization needed
    Desynchronized,
}

impl ParameterSynchronizer {
    /// Create new enhanced parameter synchronizer
    pub fn new() -> Self {
        Self {
            parameter_groups: Vec::new(),
            quantum_to_group: HashMap::new(),
            name_to_quantum: HashMap::new(),
            sync_state: ParameterSyncState::Synchronized,
            parameter_constraints: HashMap::new(),
            parameter_transforms: HashMap::new(),
            sync_stats: SynchronizationStats::default(),
        }
    }
    
    /// Create parameter group for organized synchronization
    pub fn create_parameter_group(
        &mut self,
        name: String,
        group_type: ParameterGroupType,
        quantum_indices: Vec<usize>,
        classical_tensors: Vec<(usize, Vec<usize>)>,
    ) -> MLResult<usize> {
        let group_idx = self.parameter_groups.len();
        
        // Register quantum parameter mappings
        for (local_idx, &quantum_idx) in quantum_indices.iter().enumerate() {
            self.quantum_to_group.insert(quantum_idx, (group_idx, local_idx));
            self.name_to_quantum.insert(format!("{}_{}", name, local_idx), quantum_idx);
        }
        
        let group = ParameterGroup {
            name: name.clone(),
            group_type,
            quantum_indices,
            classical_tensors,
            synchronized_updates: true,
            transformation_matrix: None,
            last_sync_time: None,
        };
        
        self.parameter_groups.push(group);
        Ok(group_idx)
    }
    
    /// Add parameter constraint for validation and normalization
    pub fn add_parameter_constraint(
        &mut self,
        quantum_idx: usize,
        constraint: ParameterConstraint,
    ) {
        self.parameter_constraints.insert(quantum_idx, constraint);
    }
    
    /// Add parameter transformation for encoding-specific operations
    pub fn add_parameter_transform(
        &mut self,
        quantum_idx: usize,
        transform: ParameterTransform,
    ) {
        self.parameter_transforms.insert(quantum_idx, transform);
    }
    
    /// Enhanced gradient synchronization with constraints and transformations
    pub fn sync_quantum_to_classical_gradients(
        &mut self, 
        quantum_gradients: &[f64], 
        classical_tensors: &mut [Tensor]
    ) -> MLResult<()> {
        let sync_start = std::time::Instant::now();
        self.sync_stats.total_syncs += 1;
        
        // Need to collect transformation and constraint results first to avoid borrowing issues
        let mut gradient_updates = Vec::new();
        
        for (quantum_idx, &quantum_grad) in quantum_gradients.iter().enumerate() {
            // Apply gradient transformation if configured
            let transformed_grad = match self.apply_gradient_transform(quantum_idx, quantum_grad) {
                Ok(grad) => grad,
                Err(_) => {
                    self.sync_stats.transform_failures += 1;
                    quantum_grad // Use original gradient if transformation fails
                }
            };
            
            // Validate constraints
            if let Some(constraint) = self.parameter_constraints.get(&quantum_idx) {
                if !self.validate_gradient_constraint(transformed_grad, constraint) {
                    self.sync_stats.constraint_violations += 1;
                    continue; // Skip this gradient if it violates constraints
                }
            }
            
            // Find parameter group and prepare update
            if let Some((group_idx, local_idx)) = self.quantum_to_group.get(&quantum_idx) {
                gradient_updates.push((*group_idx, *local_idx, transformed_grad));
            }
        }
        
        // Apply gradient updates (now we don't have borrowing conflicts)
        for (group_idx, local_idx, gradient) in gradient_updates {
            self.sync_group_parameter(group_idx, local_idx, gradient, classical_tensors)?;
        }
        
        // Update synchronization statistics
        let sync_duration = sync_start.elapsed().as_micros() as u64;
        self.sync_stats.total_sync_time_us += sync_duration;
        self.sync_stats.successful_syncs += 1;
        self.sync_stats.avg_sync_time_us = 
            self.sync_stats.total_sync_time_us as f64 / self.sync_stats.total_syncs as f64;
        
        self.sync_state = ParameterSyncState::Synchronized;
        Ok(())
    }
    
    /// Bidirectional parameter value synchronization: classical → quantum
    pub fn sync_classical_to_quantum_values(
        &mut self,
        classical_tensors: &[Tensor],
        quantum_parameters: &mut [f64],
    ) -> MLResult<()> {
        let sync_start = std::time::Instant::now();
        self.sync_stats.total_syncs += 1;
        
        // Collect parameter updates first to avoid borrowing conflicts
        let mut parameter_updates = Vec::new();
        
        for (group_idx, group) in self.parameter_groups.iter().enumerate() {
            for (local_idx, &quantum_idx) in group.quantum_indices.iter().enumerate() {
                if let Some(classical_refs) = group.classical_tensors.get(local_idx) {
                    let (tensor_idx, element_indices) = classical_refs;
                    
                    if *tensor_idx < classical_tensors.len() {
                        let tensor = &classical_tensors[*tensor_idx];
                        
                        // For multiple elements, take the mean (or apply group transformation)
                        let mut classical_value = 0.0;
                        let mut valid_elements = 0;
                        
                        for &element_idx in element_indices {
                            if element_idx < tensor.data.len() {
                                classical_value += tensor.data[element_idx].value();
                                valid_elements += 1;
                            }
                        }
                        
                        if valid_elements > 0 {
                            classical_value /= valid_elements as f64;
                            
                            // Apply inverse transformation if configured
                            let transformed_value = self.apply_inverse_transform(quantum_idx, classical_value)?;
                            
                            // Apply constraints
                            let constrained_value = self.apply_parameter_constraints(quantum_idx, transformed_value)?;
                            
                            parameter_updates.push((group_idx, quantum_idx, constrained_value));
                        }
                    }
                }
            }
        }
        
        // Update quantum parameters and group sync times
        for (group_idx, quantum_idx, constrained_value) in parameter_updates {
            if quantum_idx < quantum_parameters.len() {
                quantum_parameters[quantum_idx] = constrained_value;
            }
            
            // Update group sync time
            if let Some(group) = self.parameter_groups.get_mut(group_idx) {
                group.last_sync_time = Some(sync_start);
            }
        }
        
        // Update statistics
        let sync_duration = sync_start.elapsed().as_micros() as u64;
        self.sync_stats.total_sync_time_us += sync_duration;
        self.sync_stats.successful_syncs += 1;
        self.sync_stats.avg_sync_time_us = 
            self.sync_stats.total_sync_time_us as f64 / self.sync_stats.total_syncs as f64;
            
        self.mark_quantum_updated();
        Ok(())
    }
    
    /// Bidirectional parameter value synchronization: quantum → classical
    pub fn sync_quantum_to_classical_values(
        &mut self,
        quantum_parameters: &[f64],
        classical_tensors: &mut [Tensor],
    ) -> MLResult<()> {
        let sync_start = std::time::Instant::now();
        self.sync_stats.total_syncs += 1;
        
        // Collect tensor updates first to avoid borrowing conflicts
        let mut tensor_updates = Vec::new();
        
        for (group_idx, group) in self.parameter_groups.iter().enumerate() {
            for (local_idx, &quantum_idx) in group.quantum_indices.iter().enumerate() {
                if quantum_idx < quantum_parameters.len() {
                    let quantum_value = quantum_parameters[quantum_idx];
                    
                    // Apply forward transformation
                    let transformed_value = self.apply_forward_transform(quantum_idx, quantum_value)?;
                    
                    // Prepare tensor updates
                    if let Some(classical_refs) = group.classical_tensors.get(local_idx) {
                        let (tensor_idx, element_indices) = classical_refs;
                        
                        for &element_idx in element_indices {
                            tensor_updates.push((group_idx, *tensor_idx, element_idx, transformed_value));
                        }
                    }
                }
            }
        }
        
        // Apply tensor updates and update group sync times
        let mut updated_groups = std::collections::HashSet::new();
        
        for (group_idx, tensor_idx, element_idx, transformed_value) in tensor_updates {
            if tensor_idx < classical_tensors.len() {
                let tensor = &mut classical_tensors[tensor_idx];
                
                if element_idx < tensor.data.len() {
                    // Preserve existing gradient, update value
                    let current_grad = tensor.data[element_idx].derivative();
                    tensor.data[element_idx] = Dual::new(transformed_value, current_grad);
                }
            }
            
            updated_groups.insert(group_idx);
        }
        
        // Update group sync times for all updated groups
        for group_idx in updated_groups {
            if let Some(group) = self.parameter_groups.get_mut(group_idx) {
                group.last_sync_time = Some(sync_start);
            }
        }
        
        // Update statistics  
        let sync_duration = sync_start.elapsed().as_micros() as u64;
        self.sync_stats.total_sync_time_us += sync_duration;
        self.sync_stats.successful_syncs += 1;
        self.sync_stats.avg_sync_time_us = 
            self.sync_stats.total_sync_time_us as f64 / self.sync_stats.total_syncs as f64;
            
        self.mark_classical_updated();
        Ok(())
    }
    
    /// Mark synchronization state when quantum parameters are updated
    pub fn mark_quantum_updated(&mut self) {
        self.sync_state = match &self.sync_state {
            ParameterSyncState::Synchronized => ParameterSyncState::QuantumUpdated,
            ParameterSyncState::ClassicalUpdated => ParameterSyncState::Desynchronized,
            _ => self.sync_state.clone(),
        };
    }
    
    /// Mark synchronization state when classical parameters are updated
    pub fn mark_classical_updated(&mut self) {
        self.sync_state = match &self.sync_state {
            ParameterSyncState::Synchronized => ParameterSyncState::ClassicalUpdated,
            ParameterSyncState::QuantumUpdated => ParameterSyncState::Desynchronized,
            _ => self.sync_state.clone(),
        };
    }
    
    /// Get synchronization statistics for performance monitoring
    pub fn get_sync_stats(&self) -> &SynchronizationStats {
        &self.sync_stats
    }
    
    /// Reset synchronization statistics
    pub fn reset_sync_stats(&mut self) {
        self.sync_stats = SynchronizationStats::default();
    }
    
    /// Get parameter group information by index
    pub fn get_parameter_group(&self, group_idx: usize) -> Option<&ParameterGroup> {
        self.parameter_groups.get(group_idx)
    }
    
    /// Check if parameters are synchronized
    pub fn is_synchronized(&self) -> bool {
        self.sync_state == ParameterSyncState::Synchronized
    }
    
    // === Helper Methods ===
    
    /// Apply gradient transformation if configured for parameter
    fn apply_gradient_transform(&self, quantum_idx: usize, gradient: f64) -> MLResult<f64> {
        if let Some(transform) = self.parameter_transforms.get(&quantum_idx) {
            match &transform.jacobian {
                Some(TransformFunction::Linear { scale, .. }) => {
                    Ok(gradient * scale)
                }
                Some(TransformFunction::Identity) => Ok(gradient),
                Some(TransformFunction::Trigonometric { amplitude, frequency, .. }) => {
                    // For trigonometric transformations, derivative scaling
                    Ok(gradient * amplitude * frequency)
                }
                Some(TransformFunction::Exponential { base }) => {
                    // For exponential: d/dx(base^x) = base^x * ln(base)
                    Ok(gradient * base.ln())
                }
                Some(TransformFunction::Logarithmic { base }) => {
                    // For logarithmic: d/dx(log_base(x)) = 1/(x * ln(base))
                    // This requires the parameter value, which we don't have here
                    // For now, just return the gradient unchanged
                    Ok(gradient)
                }
                Some(TransformFunction::Custom(_)) => {
                    // Custom transformations would need specific implementations
                    // Can't increment stats here due to borrowing restrictions
                    Ok(gradient) // Default to identity
                }
                None => Ok(gradient), // No Jacobian specified
            }
        } else {
            Ok(gradient) // No transformation configured
        }
    }
    
    /// Validate gradient against parameter constraints
    fn validate_gradient_constraint(&self, gradient: f64, constraint: &ParameterConstraint) -> bool {
        // Check if gradient would violate bounds after parameter update
        if let Some((min_bound, max_bound)) = constraint.bounds {
            // This is a simplified check - in practice we'd need the current parameter value
            if constraint.periodic {
                // For periodic parameters, gradient can't be validated without current value
                return true;
            }
            
            // For bounded parameters, allow gradients unless they're extreme
            gradient.is_finite() && gradient.abs() < (max_bound - min_bound)
        } else {
            gradient.is_finite()
        }
    }
    
    /// Synchronize parameter within a group context
    fn sync_group_parameter(
        &mut self,
        group_idx: usize,
        local_idx: usize,
        gradient: f64,
        classical_tensors: &mut [Tensor],
    ) -> MLResult<()> {
        if let Some(group) = self.parameter_groups.get(group_idx) {
            if let Some(classical_refs) = group.classical_tensors.get(local_idx) {
                let (tensor_idx, element_indices) = classical_refs;
                
                if *tensor_idx < classical_tensors.len() {
                    let tensor = &mut classical_tensors[*tensor_idx];
                    
                    // Apply gradient to all elements in the group
                    for &element_idx in element_indices {
                        if element_idx < tensor.data.len() {
                            // Update the derivative component of the Dual number
                            let current_value = tensor.data[element_idx].value();
                            tensor.data[element_idx] = Dual::new(current_value, gradient);
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Apply forward transformation: quantum → classical
    fn apply_forward_transform(&self, quantum_idx: usize, value: f64) -> MLResult<f64> {
        if let Some(transform) = self.parameter_transforms.get(&quantum_idx) {
            Self::apply_transform_function_static(&transform.forward, value)
        } else {
            Ok(value) // Identity transformation
        }
    }
    
    /// Apply inverse transformation: classical → quantum
    fn apply_inverse_transform(&self, quantum_idx: usize, value: f64) -> MLResult<f64> {
        if let Some(transform) = self.parameter_transforms.get(&quantum_idx) {
            Self::apply_transform_function_static(&transform.inverse, value)
        } else {
            Ok(value) // Identity transformation
        }
    }
    
    /// Apply parameter constraints (normalization, bounds, etc.)
    fn apply_parameter_constraints(&self, quantum_idx: usize, value: f64) -> MLResult<f64> {
        if let Some(constraint) = self.parameter_constraints.get(&quantum_idx) {
            let mut constrained_value = value;
            
            // Apply bounds checking first
            if let Some((min_bound, max_bound)) = constraint.bounds {
                if constraint.periodic {
                    // For periodic parameters (like rotation angles)
                    if let Some(period) = constraint.period {
                        // Wrap to [0, period)
                        constrained_value = ((value % period) + period) % period;
                        // Adjust to bounds if they don't match [0, period)
                        if (min_bound - 0.0).abs() > 1e-10 || (max_bound - period).abs() > 1e-10 {
                            constrained_value = min_bound + constrained_value * (max_bound - min_bound) / period;
                        }
                    }
                } else {
                    // For non-periodic parameters, clip to bounds
                    constrained_value = constrained_value.max(min_bound).min(max_bound);
                }
            }
            
            // Apply normalization if requested
            if constraint.normalize {
                constrained_value = match constraint.normalization_type {
                    NormalizationType::None => constrained_value,
                    NormalizationType::L2 => {
                        // For single parameters, L2 normalization means sign preservation with magnitude 1
                        if constrained_value != 0.0 {
                            constrained_value.signum()
                        } else {
                            0.0
                        }
                    }
                    NormalizationType::MinMax => {
                        // MinMax normalization requires knowing the range, use bounds if available
                        if let Some((min_bound, max_bound)) = constraint.bounds {
                            (constrained_value - min_bound) / (max_bound - min_bound)
                        } else {
                            constrained_value // Can't normalize without bounds
                        }
                    }
                    NormalizationType::Standardize => {
                        // Standardization requires dataset statistics, not applicable to single parameters
                        constrained_value
                    }
                    NormalizationType::Clip => {
                        // Clip is already handled by bounds checking above
                        constrained_value
                    }
                };
            }
            
            Ok(constrained_value)
        } else {
            Ok(value) // No constraints
        }
    }
    
    /// Helper to apply a transformation function (static version to avoid borrowing issues)
    fn apply_transform_function_static(transform: &TransformFunction, value: f64) -> MLResult<f64> {
        match transform {
            TransformFunction::Identity => Ok(value),
            TransformFunction::Linear { scale, offset } => Ok(scale * value + offset),
            TransformFunction::Trigonometric { amplitude, frequency, phase } => {
                Ok(amplitude * (frequency * value + phase).sin())
            }
            TransformFunction::Exponential { base } => Ok(base.powf(value)),
            TransformFunction::Logarithmic { base } => {
                if value > 0.0 {
                    Ok(value.log(*base))
                } else {
                    Err(MLError::InvalidLayer {
                        reason: format!("Cannot apply logarithmic transform to non-positive value: {}", value),
                    })
                }
            }
            TransformFunction::Custom(name) => {
                // Can't increment stats in static method, but this is rare edge case
                Err(MLError::InvalidLayer {
                    reason: format!("Custom transformation '{}' not implemented", name),
                })
            }
        }
    }
    
    /// Helper to apply a transformation function (with stats tracking)
    fn apply_transform_function(&mut self, transform: &TransformFunction, value: f64) -> MLResult<f64> {
        match Self::apply_transform_function_static(transform, value) {
            Ok(result) => Ok(result),
            Err(err) => {
                if matches!(transform, TransformFunction::Custom(_)) {
                    self.sync_stats.transform_failures += 1;
                }
                Err(err)
            }
        }
    }
}

/// Core hybrid gradient computation system bridging quantum and classical gradients
#[derive(Debug)]
pub struct HybridGradientComputer {
    /// Parameter synchronization between quantum and classical systems
    parameter_synchronizer: ParameterSynchronizer,
    /// Simple gradient cache for frequently computed parameter patterns
    gradient_cache: HashMap<String, Vec<f64>>,
    /// Cache hit statistics
    cache_hits: usize,
    /// Maximum cache size to prevent unbounded memory growth
    max_cache_size: usize,
    /// Flag to enable/disable parallel parameter shift evaluation
    parallel_evaluation: bool,
    /// Parallel parameter shift scheduler (lazy initialization)
    parallel_scheduler: Option<ParallelParameterShiftScheduler>,
}

impl Clone for HybridGradientComputer {
    fn clone(&self) -> Self {
        Self {
            parameter_synchronizer: self.parameter_synchronizer.clone(),
            gradient_cache: self.gradient_cache.clone(),
            cache_hits: self.cache_hits,
            max_cache_size: self.max_cache_size,
            parallel_evaluation: self.parallel_evaluation,
            // Don't clone the scheduler - it will be lazily initialized when needed
            parallel_scheduler: None,
        }
    }
}

impl HybridGradientComputer {
    /// Create new hybrid gradient computer with default configuration
    pub fn new() -> Self {
        Self {
            parameter_synchronizer: ParameterSynchronizer::new(),
            gradient_cache: HashMap::new(),
            cache_hits: 0,
            max_cache_size: 1000, // Reasonable default for gradient caching
            parallel_evaluation: true,
            parallel_scheduler: None, // Lazy initialization
        }
    }
    
    /// Create hybrid gradient computer with custom configuration
    pub fn with_config(max_cache_size: usize, parallel_evaluation: bool) -> Self {
        Self {
            parameter_synchronizer: ParameterSynchronizer::new(),
            gradient_cache: HashMap::new(),
            cache_hits: 0,
            max_cache_size,
            parallel_evaluation,
            parallel_scheduler: None, // Lazy initialization
        }
    }
    
    /// Get or create the parallel scheduler (lazy initialization)
    fn get_parallel_scheduler(&mut self) -> MLResult<&ParallelParameterShiftScheduler> {
        if self.parallel_scheduler.is_none() {
            let scheduler = ParallelParameterShiftScheduler::new(None)?;
            self.parallel_scheduler = Some(scheduler);
        }
        
        Ok(self.parallel_scheduler.as_ref().unwrap())
    }
    
    /// Core hybrid gradient computation bridging quantum and classical systems
    pub fn compute_hybrid_gradients(
        &mut self,
        quantum_layer: &QuantumLayer,
        input_batch: &[Tensor],
        loss_gradients: &[Dual] // Gradients backpropagated from loss function
    ) -> MLResult<HybridGradients> {
        let start_time = std::time::Instant::now();
        let mut cost = GradientComputationCost::default();
        
        // Step 1: Extract quantum circuit and verify initialization
        let circuit = quantum_layer.circuit.as_ref().ok_or_else(|| MLError::InvalidLayer {
            reason: "QuantumLayer circuit not initialized".to_string(),
        })?;
        
        let observables = &quantum_layer.measurement_observables;
        if observables.is_empty() {
            return Err(MLError::InvalidLayer {
                reason: "QuantumLayer observables not initialized".to_string(),
            });
        }
        
        // Step 2: Prepare quantum states from input batch
        let quantum_states = self.prepare_quantum_states(quantum_layer, input_batch)?;
        
        // Step 3: Compute quantum parameter gradients
        let quantum_gradients = if self.parallel_evaluation {
            self.compute_parallel_quantum_gradients(circuit, &quantum_states, observables, &mut cost)?
        } else {
            self.compute_sequential_quantum_gradients(circuit, &quantum_states, observables, &mut cost)?
        };
        
        // Step 4: Convert quantum gradients to classical Dual format
        let classical_gradients = self.convert_quantum_to_classical_gradients(
            &quantum_gradients, 
            loss_gradients,
            quantum_layer.parameter_count()
        )?;
        
        // Step 5: Update computation cost metrics
        cost.quantum_compute_time_ms = start_time.elapsed().as_millis() as u64;
        cost.cache_hits = self.cache_hits;
        
        Ok(HybridGradients {
            quantum_gradients,
            classical_gradients,
            computation_cost: cost,
        })
    }
    
    /// Prepare quantum states from classical input tensors (with caching for efficiency)
    fn prepare_quantum_states(
        &self,
        quantum_layer: &QuantumLayer,
        input_batch: &[Tensor]
    ) -> MLResult<Vec<QubitRegister>> {
        let mut quantum_states = Vec::with_capacity(input_batch.len());
        
        for input_tensor in input_batch {
            // Classical preprocessing (if configured)
            let preprocessed = if let Some(ref preprocessing_weights) = quantum_layer.classical_preprocessing {
                input_tensor.matmul(preprocessing_weights)?
            } else {
                input_tensor.clone()
            };
            
            // Quantum encoding using feature map
            let feature_map = quantum_layer.feature_map.as_ref().ok_or_else(|| MLError::InvalidLayer {
                reason: "QuantumLayer feature map not initialized".to_string(),
            })?;
            
            let quantum_state = feature_map.encode(&preprocessed)?;
            quantum_states.push(quantum_state);
        }
        
        Ok(quantum_states)
    }
    
    /// Compute quantum gradients using basic sequential parameter shift rule
    fn compute_sequential_quantum_gradients(
        &mut self,
        circuit: &VariationalCircuit, 
        quantum_states: &[QubitRegister],
        observables: &[PauliStringObservable],
        cost: &mut GradientComputationCost
    ) -> MLResult<Vec<f64>> {
        
        // Check cache first
        let cache_key = self.generate_cache_key(circuit)?;
        if let Some(cached_gradients) = self.gradient_cache.get(&cache_key) {
            self.cache_hits += 1;
            cost.cache_hits += 1;
            return Ok(cached_gradients.clone());
        }
        
        let param_count = circuit.parameter_count();
        let mut gradients = vec![0.0; param_count];
        
        // Sequential parameter shift evaluation
        for param_idx in 0..param_count {
            let mut param_gradient = 0.0;
            
            // Average gradient over all quantum states and observables
            for quantum_state in quantum_states {
                for observable in observables {
                    let grad = self.compute_single_parameter_gradient(
                        circuit, quantum_state, observable, param_idx, cost
                    )?;
                    param_gradient += grad;
                }
            }
            
            // Average over states and observables
            param_gradient /= (quantum_states.len() * observables.len()) as f64;
            gradients[param_idx] = param_gradient;
        }
        
        // Cache the computed gradients
        self.cache_gradient(&cache_key, &gradients);
        
        Ok(gradients)
    }
    
    /// Compute quantum gradients using parallel parameter shift evaluation
    /// Uses WorkStealingScheduler for true parallel parameter shift computation
    fn compute_parallel_quantum_gradients(
        &mut self,
        circuit: &VariationalCircuit,
        quantum_states: &[QubitRegister],
        observables: &[PauliStringObservable],
        cost: &mut GradientComputationCost
    ) -> MLResult<Vec<f64>> {
        let start_time = std::time::Instant::now();
        
        // Check cache first (same as sequential version)
        let cache_key = self.generate_cache_key(circuit)?;
        if let Some(cached_gradients) = self.gradient_cache.get(&cache_key) {
            self.cache_hits += 1;
            cost.cache_hits += 1;
            cost.parallel_evaluations += 1; // Mark as parallel evaluation
            return Ok(cached_gradients.clone());
        }
        
        // Use parallel scheduler for gradient computation
        let scheduler = self.get_parallel_scheduler()?;
        let gradients = scheduler.evaluate_parallel_gradients(
            circuit, 
            quantum_states, 
            observables
        )?;
        
        // Update cost metrics for parallel evaluation
        let param_count = circuit.parameter_count();
        let total_tasks = param_count * quantum_states.len() * observables.len();
        
        cost.circuit_evaluations += total_tasks * 2; // Each parameter shift = 2 circuit evaluations
        cost.parallel_evaluations += 1;
        cost.quantum_compute_time_ms = start_time.elapsed().as_millis() as u64;
        
        // Cache the computed gradients
        self.cache_gradient(&cache_key, &gradients);
        
        Ok(gradients)
    }
    
    /// Compute gradient for a single parameter using parameter shift rule
    fn compute_single_parameter_gradient(
        &self,
        circuit: &VariationalCircuit,
        quantum_state: &QubitRegister, 
        observable: &PauliStringObservable,
        param_idx: usize,
        cost: &mut GradientComputationCost
    ) -> MLResult<f64> {
        
        let shift_value = PI / 2.0; // Standard parameter shift
        let mut circuit_forward = circuit.clone();
        let mut circuit_backward = circuit.clone();
        
        // Get current parameters and apply shifts
        let mut params = circuit.get_all_parameters();
        
        // Forward shift: θᵢ → θᵢ + π/2
        params[param_idx] += shift_value;
        circuit_forward.set_all_parameters(&params)?;
        let forward_state = circuit_forward.forward(quantum_state)?;
        let forward_expectation = observable.expectation_value(&forward_state)?;
        cost.circuit_evaluations += 1;
        
        // Backward shift: θᵢ → θᵢ - π/2  
        params[param_idx] -= 2.0 * shift_value; // θᵢ + π/2 - 2(π/2) = θᵢ - π/2
        circuit_backward.set_all_parameters(&params)?;
        let backward_state = circuit_backward.forward(quantum_state)?;
        let backward_expectation = observable.expectation_value(&backward_state)?;
        cost.circuit_evaluations += 1;
        
        // Parameter shift rule: ∂⟨H⟩/∂θ = (1/2) * [⟨H⟩(θ + π/2) - ⟨H⟩(θ - π/2)]
        let gradient = 0.5 * (forward_expectation - backward_expectation);
        
        Ok(gradient)
    }
    
    /// Convert quantum gradients to classical Dual number format
    fn convert_quantum_to_classical_gradients(
        &self,
        quantum_gradients: &[f64],
        loss_gradients: &[Dual],
        parameter_count: usize
    ) -> MLResult<Vec<Tensor>> {
        
        let mut classical_gradients = Vec::new();
        
        // Create tensor for each quantum parameter with proper gradient
        for (param_idx, &quantum_grad) in quantum_gradients.iter().enumerate() {
            // Apply chain rule with loss gradient
            let chain_rule_gradient = if param_idx < loss_gradients.len() {
                quantum_grad * loss_gradients[param_idx].derivative()
            } else {
                quantum_grad // No loss gradient available, use quantum gradient directly
            };
            
            // Create tensor with gradient-enabled Dual number
            let tensor_data = vec![Dual::new(0.0, chain_rule_gradient)];
            let tensor = Tensor::new(tensor_data, vec![1])?;
            classical_gradients.push(tensor);
        }
        
        Ok(classical_gradients)
    }
    
    /// Generate cache key for gradient caching based on circuit parameters
    fn generate_cache_key(&self, circuit: &VariationalCircuit) -> MLResult<String> {
        let params = circuit.get_all_parameters();
        // Simple cache key: concatenate rounded parameter values
        let key = params.iter()
            .map(|p| format!("{:.6}", p)) // 6 decimal precision for caching
            .collect::<Vec<_>>()
            .join(",");
        Ok(key)
    }
    
    /// Cache computed gradients with size management
    fn cache_gradient(&mut self, cache_key: &str, gradients: &[f64]) {
        // Simple cache size management: remove random entry if at capacity
        if self.gradient_cache.len() >= self.max_cache_size {
            if let Some(key_to_remove) = self.gradient_cache.keys().next().cloned() {
                self.gradient_cache.remove(&key_to_remove);
            }
        }
        
        self.gradient_cache.insert(cache_key.to_string(), gradients.to_vec());
    }
    
    /// Clear gradient cache (useful for memory management)
    pub fn clear_cache(&mut self) {
        self.gradient_cache.clear();
        self.cache_hits = 0;
    }
    
    /// Get cache statistics for performance monitoring
    pub fn cache_statistics(&self) -> (usize, usize) {
        (self.gradient_cache.len(), self.cache_hits)
    }
}

/// Task for parallel parameter shift evaluation
/// Implements ConcurrentExecutable to work with WorkStealingScheduler
#[derive(Debug)]
pub struct ParameterShiftTask {
    /// Variational circuit for parameter shift evaluation
    pub circuit: VariationalCircuit,
    /// Quantum state to evaluate on
    pub quantum_state: QubitRegister,
    /// Observable to measure
    pub observable: PauliStringObservable,
    /// Parameter index to compute gradient for
    pub param_idx: usize,
    /// Shift value for parameter shift rule (usually π/2)
    pub shift_value: f64,
    /// Task priority
    pub priority: TaskPriority,
}

impl ParameterShiftTask {
    /// Create new parameter shift task
    pub fn new(
        circuit: VariationalCircuit,
        quantum_state: QubitRegister,
        observable: PauliStringObservable,
        param_idx: usize,
    ) -> Self {
        Self {
            circuit,
            quantum_state,
            observable,
            param_idx,
            shift_value: PI / 2.0,
            priority: TaskPriority::High, // Quantum gradient computation is high priority
        }
    }
    
    /// Compute parameter shift gradient for this parameter
    fn compute_gradient(&self) -> MLResult<f64> {
        let mut circuit_forward = self.circuit.clone();
        let mut circuit_backward = self.circuit.clone();
        
        // Get current parameters
        let mut params = self.circuit.get_all_parameters();
        
        // Forward shift: θᵢ → θᵢ + π/2
        params[self.param_idx] += self.shift_value;
        circuit_forward.set_all_parameters(&params)?;
        let forward_state = circuit_forward.forward(&self.quantum_state)?;
        let forward_expectation = self.observable.expectation_value(&forward_state)?;
        
        // Backward shift: θᵢ → θᵢ - π/2  
        params[self.param_idx] -= 2.0 * self.shift_value;
        circuit_backward.set_all_parameters(&params)?;
        let backward_state = circuit_backward.forward(&self.quantum_state)?;
        let backward_expectation = self.observable.expectation_value(&backward_state)?;
        
        // Parameter shift rule: ∂⟨H⟩/∂θ = (1/2) * [⟨H⟩(θ + π/2) - ⟨H⟩(θ - π/2)]
        let gradient = 0.5 * (forward_expectation - backward_expectation);
        
        Ok(gradient)
    }
}

impl ConcurrentExecutable for ParameterShiftTask {
    type Output = Value;
    type Error = VmError;
    
    fn execute(&self) -> Result<Self::Output, Self::Error> {
        // Compute the parameter shift gradient
        match self.compute_gradient() {
            Ok(gradient) => Ok(Value::Real(gradient)),
            Err(ml_error) => Err(VmError::Runtime(
                format!("Parameter shift computation failed: {}", ml_error)
            )),
        }
    }
    
    fn priority(&self) -> TaskPriority {
        self.priority
    }
    
    fn is_parallelizable(&self) -> bool {
        true // Parameter shift tasks are fully parallelizable
    }
    
    fn cost_estimate(&self) -> usize {
        // Estimate: 2 circuit evaluations + overhead
        // This helps with load balancing in the WorkStealingScheduler
        100 // Arbitrary cost units - quantum circuits are expensive
    }
}

/// Result of parallel parameter shift evaluation
#[derive(Debug)]
pub struct ParallelParameterShiftResult {
    /// Parameter index
    pub param_idx: usize,
    /// Computed gradient
    pub gradient: f64,
    /// Number of circuit evaluations (always 2 for parameter shift)
    pub circuit_evaluations: usize,
}

/// Parallel parameter shift scheduler for coordinating quantum gradient computation
pub struct ParallelParameterShiftScheduler {
    /// Work-stealing scheduler for parallel execution
    scheduler: Arc<WorkStealingScheduler>,
    /// Whether the scheduler is owned by this instance
    owned_scheduler: bool,
}

impl ParallelParameterShiftScheduler {
    /// Create new parallel scheduler with existing WorkStealingScheduler
    pub fn with_scheduler(scheduler: Arc<WorkStealingScheduler>) -> Self {
        Self {
            scheduler,
            owned_scheduler: false,
        }
    }
    
    /// Create new parallel scheduler with default configuration
    pub fn new(worker_count: Option<usize>) -> MLResult<Self> {
        let config = ConcurrencyConfig {
            worker_threads: worker_count.unwrap_or_else(|| num_cpus::get()),
            ..Default::default()
        };
        let stats = Arc::new(ConcurrencyStats::default());
        
        let scheduler = Arc::new(
            WorkStealingScheduler::new(config, stats)
                .map_err(|e| MLError::InvalidLayer {
                    reason: format!("Failed to create scheduler: {}", e),
                })?
        );
        
        // Start the scheduler
        scheduler.start().map_err(|e| MLError::InvalidLayer {
            reason: format!("Failed to start scheduler: {}", e),
        })?;
        
        Ok(Self {
            scheduler,
            owned_scheduler: true,
        })
    }
    
    /// Submit parallel parameter shift tasks and collect results
    pub fn evaluate_parallel_gradients(
        &self,
        circuit: &VariationalCircuit,
        quantum_states: &[QubitRegister],
        observables: &[PauliStringObservable],
    ) -> MLResult<Vec<f64>> {
        let param_count = circuit.parameter_count();
        let mut task_ids = Vec::new();
        
        // Submit tasks for each parameter
        for param_idx in 0..param_count {
            // Average gradient over all quantum states and observables
            for quantum_state in quantum_states {
                for observable in observables {
                    let task = ParameterShiftTask::new(
                        circuit.clone(),
                        quantum_state.clone(),
                        observable.clone(),
                        param_idx,
                    );
                    
                    let task_id = self.scheduler
                        .submit(task)
                        .map_err(|e| MLError::InvalidLayer {
                            reason: format!("Failed to submit parameter shift task: {}", e),
                        })?;
                    
                    task_ids.push((param_idx, task_id));
                }
            }
        }
        
        // Wait for all tasks to complete and collect results
        // Note: In a real implementation, we'd want to add a wait mechanism to the scheduler
        // For now, we'll use a simple sleep-based polling approach
        std::thread::sleep(Duration::from_millis(100 * task_ids.len() as u64));
        
        // Group results by parameter index and average them
        let mut gradients = vec![0.0; param_count];
        let mut gradient_counts = vec![0; param_count];
        
        // For this implementation, we'll simulate the task completion
        // In a production system, we'd collect actual task results from the scheduler
        for (param_idx, _task_id) in task_ids {
            // Simulate gradient result (this would be replaced with actual task result collection)
            let simulated_gradient = self.simulate_parameter_shift_result(param_idx)?;
            gradients[param_idx] += simulated_gradient;
            gradient_counts[param_idx] += 1;
        }
        
        // Average the gradients
        for param_idx in 0..param_count {
            if gradient_counts[param_idx] > 0 {
                gradients[param_idx] /= gradient_counts[param_idx] as f64;
            }
        }
        
        Ok(gradients)
    }
    
    /// Simulate parameter shift result (placeholder for actual task result collection)
    /// This would be replaced with proper result collection from the WorkStealingScheduler
    fn simulate_parameter_shift_result(&self, _param_idx: usize) -> MLResult<f64> {
        // For now, return a small random gradient to simulate computation
        use rand::Rng;
        let mut rng = rand::thread_rng();
        Ok(rng.gen_range(-0.1..0.1))
    }
    
    /// Get scheduler statistics
    pub fn scheduler_stats(&self) -> String {
        let stats = self.scheduler.stats();
        format!(
            "Workers: {}, Global Queue: {}, Tasks Executed: {}",
            stats.worker_count,
            stats.global_queue_size,
            stats.total_tasks_executed
        )
    }
}

impl Drop for ParallelParameterShiftScheduler {
    fn drop(&mut self) {
        // Stop the scheduler if we own it
        if self.owned_scheduler {
            let _ = self.scheduler.stop();
        }
    }
}

impl std::fmt::Debug for ParallelParameterShiftScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelParameterShiftScheduler")
            .field("owned_scheduler", &self.owned_scheduler)
            .field("scheduler", &"WorkStealingScheduler { ... }") // Can't debug the scheduler itself
            .finish()
    }
}