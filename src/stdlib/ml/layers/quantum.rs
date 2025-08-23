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
use std::f64::consts::PI;

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