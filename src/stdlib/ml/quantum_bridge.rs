//! Quantum-ML Bridge Implementation
//!
//! This module provides the bridge between Lyra's quantum computing framework
//! and machine learning infrastructure, enabling hybrid quantum-classical ML.
//!
//! Key components:
//! - QuantumFeatureMap: Encode classical data into quantum states
//! - VariationalCircuit: Parameterized quantum circuits for ML
//! - QuantumLayer: Quantum neural network layer compatible with NetChain

use crate::vm::{Value, VmResult, VmError};
use crate::foreign::{LyObj, Foreign, ForeignError};
use crate::stdlib::quantum::{QuantumMatrix, QuantumCircuit, QuantumState, QuantumGate, QubitRegister};
use crate::stdlib::common::Complex;
use crate::stdlib::ml::{MLResult, MLError, layers::Tensor};
use crate::stdlib::autodiff::Dual;
use std::f64::consts::PI;
use rand;

/// Convert ForeignError to MLError
impl From<ForeignError> for MLError {
    fn from(err: ForeignError) -> Self {
        MLError::DataError {
            reason: format!("Foreign object error: {:?}", err),
        }
    }
}

/// Convert VmError to MLError
impl From<VmError> for MLError {
    fn from(err: VmError) -> Self {
        MLError::DataError {
            reason: format!("VM error: {:?}", err),
        }
    }
}

/// Feature encoding strategies for classical-to-quantum data conversion
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EncodingType {
    /// Encode features as rotation angles on quantum gates
    Angle,
    /// Encode features directly as quantum state amplitudes
    Amplitude, 
    /// Encode binary features as computational basis states
    Basis,
    /// Encode features using IQP (Instantaneous Quantum Polynomial) circuits
    IQP,
}

/// Normalization strategies for classical data before quantum encoding
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormalizationStrategy {
    /// No normalization
    None,
    /// Normalize to [0, 1] range
    MinMax,
    /// Standardize to zero mean, unit variance
    StandardScaler,
    /// Normalize to unit L2 norm
    L2Norm,
}

/// Quantum Feature Map for encoding classical data into quantum states
#[derive(Debug, Clone)]
pub struct QuantumFeatureMap {
    pub encoding_type: EncodingType,
    pub n_features: usize,
    pub n_qubits: usize,
    pub circuit_template: QuantumCircuit,
    pub feature_scaling: Vec<f64>,
    pub name: String,
}

impl QuantumFeatureMap {
    /// Create new feature map with specified encoding
    pub fn new(
        encoding_type: EncodingType,
        n_features: usize,
        n_qubits: usize,
    ) -> MLResult<Self> {
        // Validate parameters
        match encoding_type {
            EncodingType::Angle | EncodingType::IQP => {
                if n_features > n_qubits {
                    return Err(MLError::DataError {
                        reason: format!("Angle encoding requires n_features ({}) <= n_qubits ({})", 
                                       n_features, n_qubits),
                    });
                }
            },
            EncodingType::Amplitude => {
                let required_qubits = (n_features as f64).log2().ceil() as usize;
                if n_qubits < required_qubits {
                    return Err(MLError::DataError {
                        reason: format!("Amplitude encoding requires at least {} qubits for {} features", 
                                       required_qubits, n_features),
                    });
                }
            },
            EncodingType::Basis => {
                if n_qubits < n_features {
                    return Err(MLError::DataError {
                        reason: format!("Basis encoding requires n_qubits ({}) >= n_features ({})", 
                                       n_qubits, n_features),
                    });
                }
            },
        }

        // Create circuit template
        let circuit_template = QuantumCircuit::new(n_qubits);
        let feature_scaling = vec![1.0; n_features];
        let name = format!("QuantumFeatureMap[{:?}, {}, {}]", encoding_type, n_features, n_qubits);

        Ok(Self {
            encoding_type,
            n_features,
            n_qubits,
            circuit_template,
            feature_scaling,
            name,
        })
    }

    /// Encode classical ML tensor into quantum state
    pub fn encode(&self, features: &Tensor) -> MLResult<QubitRegister> {
        // Validate input tensor
        if features.data.len() != self.n_features {
            return Err(MLError::ShapeMismatch {
                expected: vec![self.n_features],
                actual: features.shape.clone(),
            });
        }

        // Extract feature values
        let feature_values: Vec<f64> = features.data.iter()
            .map(|dual| dual.value())
            .collect();

        match self.encoding_type {
            EncodingType::Angle => self.angle_encoding(&feature_values),
            EncodingType::Amplitude => self.amplitude_encoding(&feature_values),
            EncodingType::Basis => self.basis_encoding(&feature_values),
            EncodingType::IQP => self.iqp_encoding(&feature_values),
        }
    }

    /// Angle encoding: Features as rotation angles
    fn angle_encoding(&self, features: &[f64]) -> MLResult<QubitRegister> {
        let initial_state = QuantumState::new(self.n_qubits);
        let mut circuit = self.circuit_template.clone();

        // Apply rotation gates with feature values as angles
        for (qubit_idx, &feature_value) in features.iter().enumerate() {
            if qubit_idx >= self.n_qubits {
                break;
            }
            
            // Scale feature to [0, 2π] range
            let scaled_angle = feature_value * self.feature_scaling[qubit_idx] * 2.0 * PI;
            
            // Apply RY rotation (commonly used in quantum ML)
            let ry_gate = self.create_ry_gate(scaled_angle)?;
            circuit.add_gate(ry_gate, vec![qubit_idx])
                .map_err(|e| MLError::DataError {
                    reason: format!("Gate addition failed: {:?}", e),
                })?;
        }

        // Execute circuit to prepare quantum state
        let (final_state, _) = circuit.execute(&initial_state)
            .map_err(|e| MLError::DataError {
                reason: format!("Circuit execution failed: {:?}", e),
            })?;
        Ok(QubitRegister::from_state(final_state))
    }

    /// Amplitude encoding: Features as quantum amplitudes
    fn amplitude_encoding(&self, features: &[f64]) -> MLResult<QubitRegister> {
        // Normalize features to valid probability amplitudes
        let norm: f64 = features.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm == 0.0 {
            return Err(MLError::DataError {
                reason: "Cannot encode zero vector as quantum amplitudes".to_string(),
            });
        }

        let normalized_features: Vec<f64> = features.iter()
            .map(|x| x / norm)
            .collect();

        // Pad with zeros if needed to reach 2^n_qubits amplitudes
        let target_size = 2_usize.pow(self.n_qubits as u32);
        let mut amplitudes = normalized_features;
        amplitudes.resize(target_size, 0.0);

        // Create quantum state directly from amplitudes
        let complex_amplitudes: Vec<Complex> = amplitudes.iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        let quantum_state = QuantumState::from_amplitudes(complex_amplitudes)
            .map_err(|e| MLError::DataError {
                reason: format!("Failed to create quantum state from amplitudes: {:?}", e),
            })?;
        Ok(QubitRegister::from_state(quantum_state))
    }

    /// Basis encoding: Features as computational basis states
    fn basis_encoding(&self, features: &[f64]) -> MLResult<QubitRegister> {
        let initial_state = QuantumState::new(self.n_qubits);
        let mut circuit = self.circuit_template.clone();

        // Apply X gates where features indicate |1⟩ state
        for (qubit_idx, &feature_value) in features.iter().enumerate() {
            if qubit_idx >= self.n_qubits {
                break;
            }
            
            // Threshold for binary classification (can be adjusted)
            if feature_value > 0.5 {
                let x_gate = self.create_pauli_x_gate()?;
                circuit.add_gate(x_gate, vec![qubit_idx])
                    .map_err(|e| MLError::DataError {
                        reason: format!("Gate addition failed: {:?}", e),
                    })?;
            }
        }

        let (final_state, _) = circuit.execute(&initial_state)
            .map_err(|e| MLError::DataError {
                reason: format!("Circuit execution failed: {:?}", e),
            })?;
        Ok(QubitRegister::from_state(final_state))
    }

    /// IQP encoding: Instantaneous Quantum Polynomial circuits
    fn iqp_encoding(&self, features: &[f64]) -> MLResult<QubitRegister> {
        let initial_state = QuantumState::new(self.n_qubits);
        let mut circuit = self.circuit_template.clone();

        // Apply Hadamard gates to all qubits (create superposition)
        for qubit_idx in 0..self.n_qubits {
            let h_gate = self.create_hadamard_gate()?;
            circuit.add_gate(h_gate, vec![qubit_idx])
                .map_err(|e| MLError::DataError {
                    reason: format!("Gate addition failed: {:?}", e),
                })?;
        }

        // Apply diagonal gates with feature-dependent phases
        for (qubit_idx, &feature_value) in features.iter().enumerate() {
            if qubit_idx >= self.n_qubits {
                break;
            }
            
            let scaled_phase = feature_value * self.feature_scaling[qubit_idx] * PI;
            let rz_gate = self.create_rz_gate(scaled_phase)?;
            circuit.add_gate(rz_gate, vec![qubit_idx])
                .map_err(|e| MLError::DataError {
                    reason: format!("Gate addition failed: {:?}", e),
                })?;
        }

        // Add entangling layers (optional, can be configured)
        for qubit_idx in 0..(self.n_qubits - 1) {
            // Create RZZ entangling gate
            let entangling_angle = features.iter().sum::<f64>() * 0.1; // Simple entangling strategy
            let rzz_gate = self.create_rzz_gate(entangling_angle)?;
            circuit.add_gate(rzz_gate, vec![qubit_idx, qubit_idx + 1])
                .map_err(|e| MLError::DataError {
                    reason: format!("Gate addition failed: {:?}", e),
                })?;
        }

        let (final_state, _) = circuit.execute(&initial_state)
            .map_err(|e| MLError::DataError {
                reason: format!("Circuit execution failed: {:?}", e),
            })?;
        Ok(QubitRegister::from_state(final_state))
    }

    /// Helper: Create RY rotation gate
    fn create_ry_gate(&self, angle: f64) -> MLResult<QuantumGate> {
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();
        
        let matrix = QuantumMatrix::from_data(vec![
            vec![Complex::new(cos_half, 0.0), Complex::new(-sin_half, 0.0)],
            vec![Complex::new(sin_half, 0.0), Complex::new(cos_half, 0.0)],
        ]).map_err(|e| MLError::DataError {
            reason: format!("Failed to create RY gate matrix: {:?}", e),
        })?;

        Ok(QuantumGate::new(matrix, format!("RY[{}]", angle)))
    }

    /// Helper: Create RZ rotation gate
    fn create_rz_gate(&self, angle: f64) -> MLResult<QuantumGate> {
        let exp_neg_i = Complex::new((angle / 2.0).cos(), -(angle / 2.0).sin());
        let exp_pos_i = Complex::new((angle / 2.0).cos(), (angle / 2.0).sin());
        
        let matrix = QuantumMatrix::from_data(vec![
            vec![exp_neg_i, Complex::new(0.0, 0.0)],
            vec![Complex::new(0.0, 0.0), exp_pos_i],
        ]).map_err(|e| MLError::DataError {
            reason: format!("Failed to create RZ gate matrix: {:?}", e),
        })?;

        Ok(QuantumGate::new(matrix, format!("RZ[{}]", angle)))
    }

    /// Helper: Create RZZ entangling gate  
    fn create_rzz_gate(&self, angle: f64) -> MLResult<QuantumGate> {
        let exp_neg_i = Complex::new((angle / 2.0).cos(), -(angle / 2.0).sin());
        let exp_pos_i = Complex::new((angle / 2.0).cos(), (angle / 2.0).sin());
        let zero = Complex::new(0.0, 0.0);
        
        let matrix = QuantumMatrix::from_data(vec![
            vec![exp_neg_i, zero, zero, zero],
            vec![zero, exp_pos_i, zero, zero],
            vec![zero, zero, exp_pos_i, zero],
            vec![zero, zero, zero, exp_neg_i],
        ]).map_err(|e| MLError::DataError {
            reason: format!("Failed to create RZZ gate matrix: {:?}", e),
        })?;

        Ok(QuantumGate::new(matrix, format!("RZZ[{}]", angle)))
    }

    /// Helper: Create Pauli-X gate
    fn create_pauli_x_gate(&self) -> MLResult<QuantumGate> {
        let matrix = QuantumMatrix::from_data(vec![
            vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
            vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
        ]).map_err(|e| MLError::DataError {
            reason: format!("Failed to create Pauli-X gate matrix: {:?}", e),
        })?;

        Ok(QuantumGate::new(matrix, "X".to_string()))
    }

    /// Helper: Create Hadamard gate
    fn create_hadamard_gate(&self) -> MLResult<QuantumGate> {
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let matrix = QuantumMatrix::from_data(vec![
            vec![Complex::new(inv_sqrt2, 0.0), Complex::new(inv_sqrt2, 0.0)],
            vec![Complex::new(inv_sqrt2, 0.0), Complex::new(-inv_sqrt2, 0.0)],
        ]).map_err(|e| MLError::DataError {
            reason: format!("Failed to create Hadamard gate matrix: {:?}", e),
        })?;

        Ok(QuantumGate::new(matrix, "H".to_string()))
    }

    /// Convert to VM Value for method return
    fn to_value(&self) -> Value {
        Value::LyObj(LyObj::new(Box::new(self.clone())))
    }
}

impl Foreign for QuantumFeatureMap {
    fn type_name(&self) -> &'static str {
        "QuantumFeatureMap"
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "encodingType" => Ok(Value::String(format!("{:?}", self.encoding_type))),
            "nFeatures" => Ok(Value::Integer(self.n_features as i64)),
            "nQubits" => Ok(Value::Integer(self.n_qubits as i64)),
            "name" => Ok(Value::String(self.name.clone())),
            "encode" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                // Extract tensor from LyObj
                let tensor = match &args[0] {
                    Value::LyObj(obj) => {
                        obj.downcast_ref::<Tensor>()
                            .ok_or_else(|| ForeignError::InvalidArgumentType {
                                method: method.to_string(),
                                expected: "Tensor".to_string(),
                                actual: "Unknown".to_string(),
                            })?
                    },
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Tensor".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };

                // Encode tensor into quantum state
                let quantum_state = self.encode(tensor)
                    .map_err(|e| ForeignError::RuntimeError {
                        message: format!("Feature encoding failed: {:?}", e),
                    })?;

                Ok(Value::LyObj(LyObj::new(Box::new(quantum_state))))
            },
            _ => Err(ForeignError::UnknownMethod {
                method: method.to_string(),
                type_name: self.type_name().to_string(),
            }),
        }
    }
}

/// Quantum Data Encoder for ML tensor preprocessing
#[derive(Debug, Clone)]
pub struct QuantumDataEncoder {
    pub feature_map: QuantumFeatureMap,
    pub normalization_strategy: NormalizationStrategy,
    pub name: String,
}

impl QuantumDataEncoder {
    /// Create new quantum data encoder
    pub fn new(feature_map: QuantumFeatureMap) -> Self {
        let name = format!("QuantumDataEncoder[{}]", feature_map.name);
        Self {
            feature_map,
            normalization_strategy: NormalizationStrategy::StandardScaler,
            name,
        }
    }

    /// Encode ML tensor into quantum state with preprocessing
    pub fn encode_tensor(&self, tensor: &Tensor) -> MLResult<QubitRegister> {
        // Apply normalization
        let normalized_tensor = self.normalize_tensor(tensor)?;
        
        // Encode using feature map
        self.feature_map.encode(&normalized_tensor)
    }

    /// Apply normalization strategy to tensor
    fn normalize_tensor(&self, tensor: &Tensor) -> MLResult<Tensor> {
        match self.normalization_strategy {
            NormalizationStrategy::None => Ok(tensor.clone()),
            NormalizationStrategy::L2Norm => {
                let norm = tensor.data.iter()
                    .map(|dual| dual.value().powi(2))
                    .sum::<f64>()
                    .sqrt();
                
                if norm == 0.0 {
                    return Err(MLError::DataError {
                        reason: "Cannot L2 normalize zero tensor".to_string(),
                    });
                }

                let normalized_data: Vec<Dual> = tensor.data.iter()
                    .map(|dual| Dual::constant(dual.value() / norm))
                    .collect();

                Tensor::new(normalized_data, tensor.shape.clone())
                    .map_err(|e| MLError::DataError {
                        reason: format!("Tensor creation failed: {:?}", e),
                    })
            },
            NormalizationStrategy::MinMax => {
                let values: Vec<f64> = tensor.data.iter().map(|d| d.value()).collect();
                let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                
                if (max_val - min_val).abs() < 1e-10 {
                    return Ok(tensor.clone()); // No normalization needed for constant tensor
                }

                let normalized_data: Vec<Dual> = tensor.data.iter()
                    .map(|dual| Dual::constant((dual.value() - min_val) / (max_val - min_val)))
                    .collect();

                Tensor::new(normalized_data, tensor.shape.clone())
                    .map_err(|e| MLError::DataError {
                        reason: format!("Tensor creation failed: {:?}", e),
                    })
            },
            NormalizationStrategy::StandardScaler => {
                let values: Vec<f64> = tensor.data.iter().map(|d| d.value()).collect();
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance = values.iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>() / values.len() as f64;
                let std_dev = variance.sqrt();
                
                if std_dev < 1e-10 {
                    return Ok(tensor.clone()); // No standardization needed for constant tensor
                }

                let normalized_data: Vec<Dual> = tensor.data.iter()
                    .map(|dual| Dual::constant((dual.value() - mean) / std_dev))
                    .collect();

                Tensor::new(normalized_data, tensor.shape.clone())
                    .map_err(|e| MLError::DataError {
                        reason: format!("Tensor creation failed: {:?}", e),
                    })
            },
        }
    }
}

impl Foreign for QuantumDataEncoder {
    fn type_name(&self) -> &'static str {
        "QuantumDataEncoder"
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "encode" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                let tensor = match &args[0] {
                    Value::LyObj(obj) => {
                        obj.downcast_ref::<Tensor>()
                            .ok_or_else(|| ForeignError::InvalidArgumentType {
                                method: method.to_string(),
                                expected: "Tensor".to_string(),
                                actual: "Unknown".to_string(),
                            })?
                    },
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Tensor".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };

                let quantum_state = self.encode_tensor(tensor)
                    .map_err(|e| ForeignError::RuntimeError {
                        message: format!("Quantum encoding failed: {:?}", e),
                    })?;

                Ok(Value::LyObj(LyObj::new(Box::new(quantum_state))))
            },
            "featureMap" => Ok(self.feature_map.to_value()),
            "normalization" => Ok(Value::String(format!("{:?}", self.normalization_strategy))),
            "name" => Ok(Value::String(self.name.clone())),
            _ => Err(ForeignError::UnknownMethod {
                method: method.to_string(),
                type_name: self.type_name().to_string(),
            }),
        }
    }
}

/// Parameterized quantum gate with trainable parameters
#[derive(Debug, Clone)]
pub struct ParameterizedGate {
    pub gate_type: String,
    pub parameters: Vec<f64>,
    pub qubits: Vec<usize>,
    pub parameter_names: Vec<String>,
    pub name: String,
}

impl ParameterizedGate {
    pub fn new(gate_type: String, qubits: Vec<usize>) -> Self {
        let parameter_names = match gate_type.as_str() {
            "RY" | "RX" | "RZ" => vec!["angle".to_string()],
            "RZZ" => vec!["angle".to_string()],
            "U3" => vec!["theta".to_string(), "phi".to_string(), "lambda".to_string()],
            _ => vec![],
        };
        
        let parameters = vec![0.0; parameter_names.len()];
        let name = format!("Param{}[{}]", gate_type, qubits.len());
        
        Self {
            gate_type,
            parameters,
            qubits,
            parameter_names,
            name,
        }
    }
    
    pub fn set_parameters(&mut self, params: Vec<f64>) -> MLResult<()> {
        if params.len() != self.parameters.len() {
            return Err(MLError::DataError {
                reason: format!("Expected {} parameters, got {}", self.parameters.len(), params.len()),
            });
        }
        self.parameters = params;
        Ok(())
    }
    
    pub fn get_parameters(&self) -> &[f64] {
        &self.parameters
    }
    
    pub fn instantiate_gate(&self) -> MLResult<QuantumGate> {
        match self.gate_type.as_str() {
            "RY" => {
                if self.parameters.len() != 1 {
                    return Err(MLError::DataError {
                        reason: "RY gate requires exactly 1 parameter".to_string(),
                    });
                }
                let angle = self.parameters[0];
                let cos_half = (angle / 2.0).cos();
                let sin_half = (angle / 2.0).sin();
                
                let matrix = QuantumMatrix::from_data(vec![
                    vec![Complex::new(cos_half, 0.0), Complex::new(-sin_half, 0.0)],
                    vec![Complex::new(sin_half, 0.0), Complex::new(cos_half, 0.0)],
                ]).map_err(|e| MLError::DataError {
                    reason: format!("Failed to create RY gate: {:?}", e),
                })?;
                
                Ok(QuantumGate::new(matrix, format!("RY[{:.6}]", angle)))
            },
            "RX" => {
                if self.parameters.len() != 1 {
                    return Err(MLError::DataError {
                        reason: "RX gate requires exactly 1 parameter".to_string(),
                    });
                }
                let angle = self.parameters[0];
                let cos_half = (angle / 2.0).cos();
                let sin_half = (angle / 2.0).sin();
                
                let matrix = QuantumMatrix::from_data(vec![
                    vec![Complex::new(cos_half, 0.0), Complex::new(0.0, -sin_half)],
                    vec![Complex::new(0.0, -sin_half), Complex::new(cos_half, 0.0)],
                ]).map_err(|e| MLError::DataError {
                    reason: format!("Failed to create RX gate: {:?}", e),
                })?;
                
                Ok(QuantumGate::new(matrix, format!("RX[{:.6}]", angle)))
            },
            "RZ" => {
                if self.parameters.len() != 1 {
                    return Err(MLError::DataError {
                        reason: "RZ gate requires exactly 1 parameter".to_string(),
                    });
                }
                let angle = self.parameters[0];
                let exp_neg_i = Complex::new((angle / 2.0).cos(), -(angle / 2.0).sin());
                let exp_pos_i = Complex::new((angle / 2.0).cos(), (angle / 2.0).sin());
                
                let matrix = QuantumMatrix::from_data(vec![
                    vec![exp_neg_i, Complex::new(0.0, 0.0)],
                    vec![Complex::new(0.0, 0.0), exp_pos_i],
                ]).map_err(|e| MLError::DataError {
                    reason: format!("Failed to create RZ gate: {:?}", e),
                })?;
                
                Ok(QuantumGate::new(matrix, format!("RZ[{:.6}]", angle)))
            },
            "RZZ" => {
                if self.parameters.len() != 1 {
                    return Err(MLError::DataError {
                        reason: "RZZ gate requires exactly 1 parameter".to_string(),
                    });
                }
                let angle = self.parameters[0];
                let exp_neg_i = Complex::new((angle / 2.0).cos(), -(angle / 2.0).sin());
                let exp_pos_i = Complex::new((angle / 2.0).cos(), (angle / 2.0).sin());
                let zero = Complex::new(0.0, 0.0);
                
                let matrix = QuantumMatrix::from_data(vec![
                    vec![exp_neg_i, zero, zero, zero],
                    vec![zero, exp_pos_i, zero, zero],
                    vec![zero, zero, exp_pos_i, zero],
                    vec![zero, zero, zero, exp_neg_i],
                ]).map_err(|e| MLError::DataError {
                    reason: format!("Failed to create RZZ gate: {:?}", e),
                })?;
                
                Ok(QuantumGate::new(matrix, format!("RZZ[{:.6}]", angle)))
            },
            _ => Err(MLError::DataError {
                reason: format!("Unsupported parameterized gate type: {}", self.gate_type),
            }),
        }
    }
}

impl Foreign for ParameterizedGate {
    fn type_name(&self) -> &'static str {
        "ParameterizedGate"
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "gateType" => Ok(Value::String(self.gate_type.clone())),
            "parameters" => {
                let param_values: Vec<Value> = self.parameters.iter()
                    .map(|&p| Value::Real(p))
                    .collect();
                Ok(Value::List(param_values))
            },
            "setParameters" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: "setParameters".to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let new_params = match &args[0] {
                    Value::List(param_list) => {
                        let mut params = Vec::new();
                        for param_val in param_list {
                            match param_val {
                                Value::Real(p) => params.push(*p),
                                _ => return Err(ForeignError::InvalidArgumentType {
                                    method: "setParameters".to_string(),
                                    expected: "List of numbers".to_string(),
                                    actual: "Non-numeric value".to_string(),
                                }),
                            }
                        }
                        params
                    },
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: "setParameters".to_string(),
                        expected: "List".to_string(),
                        actual: "Non-list".to_string(),
                    }),
                };
                
                let mut gate = self.clone();
                gate.set_parameters(new_params)
                    .map_err(|e| ForeignError::RuntimeError {
                        message: format!("Parameter setting failed: {:?}", e),
                    })?;
                
                Ok(Value::LyObj(LyObj::new(Box::new(gate))))
            },
            "instantiate" => {
                let quantum_gate = self.instantiate_gate()
                    .map_err(|e| ForeignError::RuntimeError {
                        message: format!("Gate instantiation failed: {:?}", e),
                    })?;
                Ok(Value::LyObj(LyObj::new(Box::new(quantum_gate))))
            },
            "name" => Ok(Value::String(self.name.clone())),
            "qubits" => {
                let qubit_values: Vec<Value> = self.qubits.iter()
                    .map(|&q| Value::Integer(q as i64))
                    .collect();
                Ok(Value::List(qubit_values))
            },
            _ => Err(ForeignError::UnknownMethod {
                method: method.to_string(),
                type_name: self.type_name().to_string(),
            }),
        }
    }
}

/// Variational quantum circuit with trainable parameters
#[derive(Debug, Clone)]
pub struct VariationalCircuit {
    pub n_qubits: usize,
    pub parameterized_gates: Vec<ParameterizedGate>,
    pub circuit_template: QuantumCircuit,
    pub total_parameters: usize,
    pub parameter_bounds: Vec<(f64, f64)>,
    pub name: String,
}

impl VariationalCircuit {
    pub fn new(n_qubits: usize) -> Self {
        let circuit_template = QuantumCircuit::new(n_qubits);
        let name = format!("VariationalCircuit[{}]", n_qubits);
        
        Self {
            n_qubits,
            parameterized_gates: Vec::new(),
            circuit_template,
            total_parameters: 0,
            parameter_bounds: Vec::new(),
            name,
        }
    }
    
    pub fn add_parameterized_gate(&mut self, mut gate: ParameterizedGate) -> MLResult<()> {
        // Validate qubit indices
        for &qubit in &gate.qubits {
            if qubit >= self.n_qubits {
                return Err(MLError::DataError {
                    reason: format!("Qubit index {} out of range for {}-qubit circuit", 
                                   qubit, self.n_qubits),
                });
            }
        }
        
        // Add parameter bounds (default: [0, 2π] for rotation gates)
        for _ in 0..gate.parameters.len() {
            self.parameter_bounds.push((0.0, 2.0 * PI));
        }
        
        self.total_parameters += gate.parameters.len();
        self.parameterized_gates.push(gate);
        Ok(())
    }
    
    pub fn set_all_parameters(&mut self, params: &[f64]) -> MLResult<()> {
        if params.len() != self.total_parameters {
            return Err(MLError::DataError {
                reason: format!("Expected {} parameters, got {}", self.total_parameters, params.len()),
            });
        }
        
        let mut param_index = 0;
        for gate in &mut self.parameterized_gates {
            let gate_param_count = gate.parameters.len();
            let gate_params = params[param_index..param_index + gate_param_count].to_vec();
            gate.set_parameters(gate_params)?;
            param_index += gate_param_count;
        }
        
        Ok(())
    }
    
    pub fn get_all_parameters(&self) -> Vec<f64> {
        let mut all_params = Vec::new();
        for gate in &self.parameterized_gates {
            all_params.extend_from_slice(gate.get_parameters());
        }
        all_params
    }
    
    pub fn build_circuit(&self) -> MLResult<QuantumCircuit> {
        let mut circuit = self.circuit_template.clone();
        
        for param_gate in &self.parameterized_gates {
            let quantum_gate = param_gate.instantiate_gate()?;
            circuit.add_gate(quantum_gate, param_gate.qubits.clone())
                .map_err(|e| MLError::DataError {
                    reason: format!("Failed to add gate to circuit: {:?}", e),
                })?;
        }
        
        Ok(circuit)
    }
    
    pub fn forward(&self, input_state: &QubitRegister) -> MLResult<QubitRegister> {
        let circuit = self.build_circuit()?;
        let (final_state, _) = circuit.execute(&input_state.state)
            .map_err(|e| MLError::DataError {
                reason: format!("Circuit execution failed: {:?}", e),
            })?;
        Ok(QubitRegister::from_state(final_state))
    }
    
    pub fn add_rotation_layer(&mut self, rotation_type: &str) -> MLResult<()> {
        for qubit in 0..self.n_qubits {
            let mut gate = ParameterizedGate::new(rotation_type.to_string(), vec![qubit]);
            gate.set_parameters(vec![rand::random::<f64>() * 2.0 * PI])?;
            self.add_parameterized_gate(gate)?;
        }
        Ok(())
    }
    
    pub fn add_entangling_layer(&mut self) -> MLResult<()> {
        for qubit in 0..(self.n_qubits - 1) {
            let mut gate = ParameterizedGate::new("RZZ".to_string(), vec![qubit, qubit + 1]);
            gate.set_parameters(vec![rand::random::<f64>() * 2.0 * PI])?;
            self.add_parameterized_gate(gate)?;
        }
        Ok(())
    }
}

impl Foreign for VariationalCircuit {
    fn type_name(&self) -> &'static str {
        "VariationalCircuit"
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "nQubits" => Ok(Value::Integer(self.n_qubits as i64)),
            "totalParameters" => Ok(Value::Integer(self.total_parameters as i64)),
            "parameters" => {
                let param_values: Vec<Value> = self.get_all_parameters().iter()
                    .map(|&p| Value::Real(p))
                    .collect();
                Ok(Value::List(param_values))
            },
            "setParameters" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: "setParameters".to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let new_params = match &args[0] {
                    Value::List(param_list) => {
                        let mut params = Vec::new();
                        for param_val in param_list {
                            match param_val {
                                Value::Real(p) => params.push(*p),
                                _ => return Err(ForeignError::InvalidArgumentType {
                                    method: "setParameters".to_string(),
                                    expected: "List of numbers".to_string(),
                                    actual: "Non-numeric value".to_string(),
                                }),
                            }
                        }
                        params
                    },
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: "setParameters".to_string(),
                        expected: "List".to_string(),
                        actual: "Non-list".to_string(),
                    }),
                };
                
                let mut circuit = self.clone();
                circuit.set_all_parameters(&new_params)
                    .map_err(|e| ForeignError::RuntimeError {
                        message: format!("Parameter setting failed: {:?}", e),
                    })?;
                
                Ok(Value::LyObj(LyObj::new(Box::new(circuit))))
            },
            "forward" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: "forward".to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let input_state = match &args[0] {
                    Value::LyObj(obj) => {
                        obj.downcast_ref::<QubitRegister>()
                            .ok_or_else(|| ForeignError::InvalidArgumentType {
                                method: "forward".to_string(),
                                expected: "QubitRegister".to_string(),
                                actual: "Unknown".to_string(),
                            })?
                    },
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: "forward".to_string(),
                        expected: "QubitRegister".to_string(),
                        actual: "Non-QubitRegister".to_string(),
                    }),
                };
                
                let output_state = self.forward(input_state)
                    .map_err(|e| ForeignError::RuntimeError {
                        message: format!("Forward pass failed: {:?}", e),
                    })?;
                
                Ok(Value::LyObj(LyObj::new(Box::new(output_state))))
            },
            "addRotationLayer" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: "addRotationLayer".to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let rotation_type = match &args[0] {
                    Value::String(s) => s,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: "addRotationLayer".to_string(),
                        expected: "String".to_string(),
                        actual: "Non-string".to_string(),
                    }),
                };
                
                let mut circuit = self.clone();
                circuit.add_rotation_layer(rotation_type)
                    .map_err(|e| ForeignError::RuntimeError {
                        message: format!("Failed to add rotation layer: {:?}", e),
                    })?;
                
                Ok(Value::LyObj(LyObj::new(Box::new(circuit))))
            },
            "addEntanglingLayer" => {
                let mut circuit = self.clone();
                circuit.add_entangling_layer()
                    .map_err(|e| ForeignError::RuntimeError {
                        message: format!("Failed to add entangling layer: {:?}", e),
                    })?;
                
                Ok(Value::LyObj(LyObj::new(Box::new(circuit))))
            },
            "name" => Ok(Value::String(self.name.clone())),
            _ => Err(ForeignError::UnknownMethod {
                method: method.to_string(),
                type_name: self.type_name().to_string(),
            }),
        }
    }
}

/// Pauli string observable for quantum measurement
#[derive(Debug, Clone)]
pub struct PauliStringObservable {
    pub pauli_string: String,
    pub coefficient: f64,
    pub n_qubits: usize,
    pub pauli_matrices: Vec<QuantumMatrix>,
    pub name: String,
}

impl PauliStringObservable {
    pub fn new(pauli_string: String, coefficient: f64) -> MLResult<Self> {
        // Validate Pauli string (should contain only I, X, Y, Z)
        for c in pauli_string.chars() {
            if !matches!(c, 'I' | 'X' | 'Y' | 'Z') {
                return Err(MLError::DataError {
                    reason: format!("Invalid Pauli character '{}' in string '{}'", c, pauli_string),
                });
            }
        }
        
        let n_qubits = pauli_string.len();
        if n_qubits == 0 {
            return Err(MLError::DataError {
                reason: "Pauli string cannot be empty".to_string(),
            });
        }
        
        // Create Pauli matrices for each character
        let mut pauli_matrices = Vec::new();
        for c in pauli_string.chars() {
            let matrix = match c {
                'I' => QuantumMatrix::from_data(vec![
                    vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
                    vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
                ])?,
                'X' => QuantumMatrix::from_data(vec![
                    vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
                    vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
                ])?,
                'Y' => QuantumMatrix::from_data(vec![
                    vec![Complex::new(0.0, 0.0), Complex::new(0.0, -1.0)],
                    vec![Complex::new(0.0, 1.0), Complex::new(0.0, 0.0)],
                ])?,
                'Z' => QuantumMatrix::from_data(vec![
                    vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
                    vec![Complex::new(0.0, 0.0), Complex::new(-1.0, 0.0)],
                ])?,
                _ => unreachable!(),
            };
            pauli_matrices.push(matrix);
        }
        
        let name = format!("PauliObservable[{}, {}]", pauli_string, coefficient);
        
        Ok(Self {
            pauli_string,
            coefficient,
            n_qubits,
            pauli_matrices,
            name,
        })
    }
    
    pub fn expectation_value(&self, quantum_state: &QubitRegister) -> MLResult<f64> {
        // Build the full Pauli operator as tensor product
        let full_operator = self.build_full_operator()?;
        
        // Get the quantum state vector
        let state_vector = &quantum_state.state.amplitudes;
        
        // Compute ⟨ψ|P|ψ⟩ = ψ†·P·ψ
        let n_states = state_vector.len();
        if full_operator.rows != n_states || full_operator.cols != n_states {
            return Err(MLError::DataError {
                reason: format!("Operator size {}×{} doesn't match state size {}", 
                               full_operator.rows, full_operator.cols, n_states),
            });
        }
        
        // Multiply P|ψ⟩
        let mut p_psi = vec![Complex::new(0.0, 0.0); n_states];
        for i in 0..n_states {
            for j in 0..n_states {
                p_psi[i] = p_psi[i] + full_operator.data[i][j] * state_vector[j];
            }
        }
        
        // Compute ⟨ψ|P|ψ⟩ = Σᵢ ψᵢ* · (P|ψ⟩)ᵢ
        let mut expectation = Complex::new(0.0, 0.0);
        for i in 0..n_states {
            expectation = expectation + state_vector[i].conjugate() * p_psi[i];
        }
        
        // Expectation value should be real for Hermitian operators
        if expectation.imag.abs() > 1e-10 {
            return Err(MLError::DataError {
                reason: format!("Expectation value has significant imaginary part: {}", expectation.imag),
            });
        }
        
        Ok(self.coefficient * expectation.real)
    }
    
    fn build_full_operator(&self) -> MLResult<QuantumMatrix> {
        if self.pauli_matrices.is_empty() {
            return Err(MLError::DataError {
                reason: "No Pauli matrices to build operator from".to_string(),
            });
        }
        
        // Start with the first Pauli matrix
        let mut result = self.pauli_matrices[0].clone();
        
        // Tensor product with each subsequent matrix
        for matrix in &self.pauli_matrices[1..] {
            result = result.tensor_product(matrix);
        }
        
        Ok(result)
    }
    
    pub fn measure_state(&self, quantum_state: &QubitRegister, shots: usize) -> MLResult<f64> {
        let expectation = self.expectation_value(quantum_state)?;
        
        // For realistic quantum simulation, add measurement noise
        if shots > 0 {
            // Simulate shot noise using binomial distribution approximation
            let variance = (1.0 - expectation.powi(2)) / shots as f64;
            let noise = (rand::random::<f64>() - 0.5) * 2.0 * variance.sqrt();
            Ok(expectation + noise)
        } else {
            Ok(expectation)
        }
    }
}

impl Foreign for PauliStringObservable {
    fn type_name(&self) -> &'static str {
        "PauliStringObservable"
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "pauliString" => Ok(Value::String(self.pauli_string.clone())),
            "coefficient" => Ok(Value::Real(self.coefficient)),
            "nQubits" => Ok(Value::Integer(self.n_qubits as i64)),
            "name" => Ok(Value::String(self.name.clone())),
            "expectationValue" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: "expectationValue".to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let quantum_state = match &args[0] {
                    Value::LyObj(obj) => {
                        obj.downcast_ref::<QubitRegister>()
                            .ok_or_else(|| ForeignError::InvalidArgumentType {
                                method: "expectationValue".to_string(),
                                expected: "QubitRegister".to_string(),
                                actual: "Unknown".to_string(),
                            })?
                    },
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: "expectationValue".to_string(),
                        expected: "QubitRegister".to_string(),
                        actual: "Non-QubitRegister".to_string(),
                    }),
                };
                
                let expectation = self.expectation_value(quantum_state)
                    .map_err(|e| ForeignError::RuntimeError {
                        message: format!("Expectation value computation failed: {:?}", e),
                    })?;
                
                Ok(Value::Real(expectation))
            },
            "measure" => {
                if args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: "measure".to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }
                
                let quantum_state = match &args[0] {
                    Value::LyObj(obj) => {
                        obj.downcast_ref::<QubitRegister>()
                            .ok_or_else(|| ForeignError::InvalidArgumentType {
                                method: "measure".to_string(),
                                expected: "QubitRegister".to_string(),
                                actual: "Unknown".to_string(),
                            })?
                    },
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: "measure".to_string(),
                        expected: "QubitRegister".to_string(),
                        actual: "Non-QubitRegister".to_string(),
                    }),
                };
                
                let shots = match &args[1] {
                    Value::Integer(s) => {
                        if *s < 0 {
                            return Err(ForeignError::InvalidArgumentType {
                                method: "measure".to_string(),
                                expected: "Non-negative integer".to_string(),
                                actual: s.to_string(),
                            });
                        }
                        *s as usize
                    },
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: "measure".to_string(),
                        expected: "Integer".to_string(),
                        actual: "Non-integer".to_string(),
                    }),
                };
                
                let measurement = self.measure_state(quantum_state, shots)
                    .map_err(|e| ForeignError::RuntimeError {
                        message: format!("Measurement failed: {:?}", e),
                    })?;
                
                Ok(Value::Real(measurement))
            },
            _ => Err(ForeignError::UnknownMethod {
                method: method.to_string(),
                type_name: self.type_name().to_string(),
            }),
        }
    }
}

/// Quantum gradient computer using parameter shift rule
#[derive(Debug, Clone)]
pub struct QuantumGradientComputer {
    pub circuit: VariationalCircuit,
    pub observable: PauliStringObservable,
    pub shift_value: f64,
    pub name: String,
}

impl QuantumGradientComputer {
    pub fn new(circuit: VariationalCircuit, observable: PauliStringObservable) -> Self {
        let name = format!("QuantumGradients[{}, {}]", circuit.name, observable.name);
        Self {
            circuit,
            observable,
            shift_value: PI / 2.0, // Standard parameter shift
            name,
        }
    }
    
    pub fn compute_gradients(&self, input_state: &QubitRegister) -> MLResult<Vec<f64>> {
        let mut gradients = Vec::new();
        let current_params = self.circuit.get_all_parameters();
        
        for param_idx in 0..current_params.len() {
            let gradient = self.compute_parameter_gradient(input_state, param_idx)?;
            gradients.push(gradient);
        }
        
        Ok(gradients)
    }
    
    fn compute_parameter_gradient(&self, input_state: &QubitRegister, param_idx: usize) -> MLResult<f64> {
        let current_params = self.circuit.get_all_parameters();
        
        if param_idx >= current_params.len() {
            return Err(MLError::DataError {
                reason: format!("Parameter index {} out of range (total: {})", 
                               param_idx, current_params.len()),
            });
        }
        
        // Forward shift: θᵢ + π/2
        let mut forward_params = current_params.clone();
        forward_params[param_idx] += self.shift_value;
        
        let mut forward_circuit = self.circuit.clone();
        forward_circuit.set_all_parameters(&forward_params)?;
        let forward_state = forward_circuit.forward(input_state)?;
        let forward_expectation = self.observable.expectation_value(&forward_state)?;
        
        // Backward shift: θᵢ - π/2
        let mut backward_params = current_params.clone();
        backward_params[param_idx] -= self.shift_value;
        
        let mut backward_circuit = self.circuit.clone();
        backward_circuit.set_all_parameters(&backward_params)?;
        let backward_state = backward_circuit.forward(input_state)?;
        let backward_expectation = self.observable.expectation_value(&backward_state)?;
        
        // Parameter shift rule: ∂⟨H⟩/∂θ = (1/2) * [⟨H⟩(θ + π/2) - ⟨H⟩(θ - π/2)]
        let gradient = 0.5 * (forward_expectation - backward_expectation);
        
        Ok(gradient)
    }
    
    pub fn compute_gradient_vector(&self, input_states: &[QubitRegister]) -> MLResult<Vec<f64>> {
        if input_states.is_empty() {
            return Err(MLError::DataError {
                reason: "Cannot compute gradients with empty input states".to_string(),
            });
        }
        
        let n_params = self.circuit.total_parameters;
        let mut total_gradients = vec![0.0; n_params];
        
        // Average gradients over all input states
        for state in input_states {
            let state_gradients = self.compute_gradients(state)?;
            for (i, &grad) in state_gradients.iter().enumerate() {
                total_gradients[i] += grad;
            }
        }
        
        // Average the gradients
        let n_states = input_states.len() as f64;
        for grad in &mut total_gradients {
            *grad /= n_states;
        }
        
        Ok(total_gradients)
    }
    
    pub fn set_shift_value(&mut self, shift: f64) -> MLResult<()> {
        if shift <= 0.0 || shift > PI {
            return Err(MLError::DataError {
                reason: format!("Shift value {} must be in range (0, π]", shift),
            });
        }
        self.shift_value = shift;
        Ok(())
    }
}

impl Foreign for QuantumGradientComputer {
    fn type_name(&self) -> &'static str {
        "QuantumGradientComputer"
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "computeGradients" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: "computeGradients".to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let input_state = match &args[0] {
                    Value::LyObj(obj) => {
                        obj.downcast_ref::<QubitRegister>()
                            .ok_or_else(|| ForeignError::InvalidArgumentType {
                                method: "computeGradients".to_string(),
                                expected: "QubitRegister".to_string(),
                                actual: "Unknown".to_string(),
                            })?
                    },
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: "computeGradients".to_string(),
                        expected: "QubitRegister".to_string(),
                        actual: "Non-QubitRegister".to_string(),
                    }),
                };
                
                let gradients = self.compute_gradients(input_state)
                    .map_err(|e| ForeignError::RuntimeError {
                        message: format!("Gradient computation failed: {:?}", e),
                    })?;
                
                let gradient_values: Vec<Value> = gradients.iter()
                    .map(|&g| Value::Real(g))
                    .collect();
                
                Ok(Value::List(gradient_values))
            },
            "computeGradientVector" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: "computeGradientVector".to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let state_list = match &args[0] {
                    Value::List(states) => {
                        let mut quantum_states = Vec::new();
                        for state_val in states {
                            match state_val {
                                Value::LyObj(obj) => {
                                    if let Some(state) = obj.downcast_ref::<QubitRegister>() {
                                        quantum_states.push(state.clone());
                                    } else {
                                        return Err(ForeignError::InvalidArgumentType {
                                            method: "computeGradientVector".to_string(),
                                            expected: "List of QubitRegister".to_string(),
                                            actual: "Non-QubitRegister in list".to_string(),
                                        });
                                    }
                                },
                                _ => return Err(ForeignError::InvalidArgumentType {
                                    method: "computeGradientVector".to_string(),
                                    expected: "List of QubitRegister".to_string(),
                                    actual: "Non-QubitRegister in list".to_string(),
                                }),
                            }
                        }
                        quantum_states
                    },
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: "computeGradientVector".to_string(),
                        expected: "List".to_string(),
                        actual: "Non-list".to_string(),
                    }),
                };
                
                let gradients = self.compute_gradient_vector(&state_list)
                    .map_err(|e| ForeignError::RuntimeError {
                        message: format!("Gradient vector computation failed: {:?}", e),
                    })?;
                
                let gradient_values: Vec<Value> = gradients.iter()
                    .map(|&g| Value::Real(g))
                    .collect();
                
                Ok(Value::List(gradient_values))
            },
            "setShiftValue" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: "setShiftValue".to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let shift = match &args[0] {
                    Value::Real(s) => *s,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: "setShiftValue".to_string(),
                        expected: "Number".to_string(),
                        actual: "Non-number".to_string(),
                    }),
                };
                
                let mut computer = self.clone();
                computer.set_shift_value(shift)
                    .map_err(|e| ForeignError::RuntimeError {
                        message: format!("Shift value setting failed: {:?}", e),
                    })?;
                
                Ok(Value::LyObj(LyObj::new(Box::new(computer))))
            },
            "shiftValue" => Ok(Value::Real(self.shift_value)),
            "circuit" => Ok(Value::LyObj(LyObj::new(Box::new(self.circuit.clone())))),
            "observable" => Ok(Value::LyObj(LyObj::new(Box::new(self.observable.clone())))),
            "name" => Ok(Value::String(self.name.clone())),
            _ => Err(ForeignError::UnknownMethod {
                method: method.to_string(),
                type_name: self.type_name().to_string(),
            }),
        }
    }
}

// ============================================================================
// STDLIB FUNCTION IMPLEMENTATIONS FOR VM REGISTRATION
// ============================================================================

/// Create quantum feature map with specified encoding type
pub fn quantum_feature_map(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::ArityError { 
            function_name: "QuantumFeatureMap".to_string(), 
            expected: 3, 
            actual: args.len() 
        });
    }

    // Parse encoding type
    let encoding_type = match &args[0] {
        Value::String(s) => match s.as_str() {
            "Angle" => EncodingType::Angle,
            "Amplitude" => EncodingType::Amplitude,
            "Basis" => EncodingType::Basis,
            "IQP" => EncodingType::IQP,
            _ => return Err(VmError::ArgumentTypeError {
                function_name: "QuantumFeatureMap".to_string(),
                param_index: 0,
                expected: "Encoding type (Angle|Amplitude|Basis|IQP)".to_string(),
                actual: s.clone(),
            }),
        },
        _ => return Err(VmError::ArgumentTypeError {
            function_name: "QuantumFeatureMap".to_string(),
            param_index: 0,
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    // Parse feature count
    let n_features = match &args[1] {
        Value::Integer(n) => {
            if *n <= 0 {
                return Err(VmError::ArgumentTypeError {
                    function_name: "QuantumFeatureMap".to_string(),
                    param_index: 1,
                    expected: "Positive integer".to_string(),
                    actual: n.to_string(),
                });
            }
            *n as usize
        },
        _ => return Err(VmError::ArgumentTypeError {
            function_name: "QuantumFeatureMap".to_string(),
            param_index: 1,
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    // Parse qubit count
    let n_qubits = match &args[2] {
        Value::Integer(n) => {
            if *n <= 0 {
                return Err(VmError::ArgumentTypeError {
                    function_name: "QuantumFeatureMap".to_string(),
                    param_index: 2,
                    expected: "Positive integer".to_string(),
                    actual: n.to_string(),
                });
            }
            *n as usize
        },
        _ => return Err(VmError::ArgumentTypeError {
            function_name: "QuantumFeatureMap".to_string(),
            param_index: 2,
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };

    // Create feature map
    let feature_map = QuantumFeatureMap::new(encoding_type, n_features, n_qubits)
        .map_err(|e| VmError::Runtime(format!("Failed to create quantum feature map: {:?}", e)))?;

    Ok(Value::LyObj(LyObj::new(Box::new(feature_map))))
}

/// Create quantum data encoder with feature map
pub fn quantum_data_encoder(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::ArityError { 
            function_name: "QuantumDataEncoder".to_string(), 
            expected: 1, 
            actual: args.len() 
        });
    }

    // Extract feature map from argument
    let feature_map = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<QuantumFeatureMap>()
                .ok_or_else(|| VmError::ArgumentTypeError {
                    function_name: "QuantumDataEncoder".to_string(),
                    param_index: 0,
                    expected: "QuantumFeatureMap".to_string(),
                    actual: "Unknown".to_string(),
                })?
                .clone()
        },
        _ => return Err(VmError::ArgumentTypeError {
            function_name: "QuantumDataEncoder".to_string(),
            param_index: 0,
            expected: "QuantumFeatureMap".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let encoder = QuantumDataEncoder::new(feature_map);
    Ok(Value::LyObj(LyObj::new(Box::new(encoder))))
}

/// Encode classical tensor into quantum state using feature map
pub fn encode_to_quantum_state(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::ArityError { 
            function_name: "EncodeToQuantumState".to_string(), 
            expected: 2, 
            actual: args.len() 
        });
    }

    // Extract encoder
    let encoder = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<QuantumDataEncoder>()
                .ok_or_else(|| VmError::ArgumentTypeError {
                    function_name: "EncodeToQuantumState".to_string(),
                    param_index: 0,
                    expected: "QuantumDataEncoder".to_string(),
                    actual: "Unknown".to_string(),
                })?
        },
        _ => return Err(VmError::ArgumentTypeError {
            function_name: "EncodeToQuantumState".to_string(),
            param_index: 0,
            expected: "QuantumDataEncoder".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    // Extract tensor
    let tensor = match &args[1] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<Tensor>()
                .ok_or_else(|| VmError::ArgumentTypeError {
                    function_name: "EncodeToQuantumState".to_string(),
                    param_index: 1,
                    expected: "Tensor".to_string(),
                    actual: "Unknown".to_string(),
                })?
        },
        _ => return Err(VmError::ArgumentTypeError {
            function_name: "EncodeToQuantumState".to_string(),
            param_index: 1,
            expected: "Tensor".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    // Encode tensor
    let quantum_state = encoder.encode_tensor(tensor)
        .map_err(|e| VmError::Runtime(format!("Quantum encoding failed: {:?}", e)))?;

    Ok(Value::LyObj(LyObj::new(Box::new(quantum_state))))
}

/// Create parameterized quantum gate
pub fn parameterized_gate(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::ArityError { 
            function_name: "ParameterizedGate".to_string(), 
            expected: 2, 
            actual: args.len() 
        });
    }

    let gate_type = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::ArgumentTypeError {
            function_name: "ParameterizedGate".to_string(),
            param_index: 0,
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let qubits = match &args[1] {
        Value::List(qubit_list) => {
            let mut qubits = Vec::new();
            for (i, qubit_val) in qubit_list.iter().enumerate() {
                match qubit_val {
                    Value::Integer(q) => {
                        if *q < 0 {
                            return Err(VmError::ArgumentTypeError {
                                function_name: "ParameterizedGate".to_string(),
                                param_index: 1,
                                expected: "Non-negative integer".to_string(),
                                actual: q.to_string(),
                            });
                        }
                        qubits.push(*q as usize);
                    },
                    _ => return Err(VmError::ArgumentTypeError {
                        function_name: "ParameterizedGate".to_string(),
                        param_index: 1,
                        expected: "List of integers".to_string(),
                        actual: format!("Element {} is not an integer", i),
                    }),
                }
            }
            qubits
        },
        Value::Integer(q) => {
            if *q < 0 {
                return Err(VmError::ArgumentTypeError {
                    function_name: "ParameterizedGate".to_string(),
                    param_index: 1,
                    expected: "Non-negative integer".to_string(),
                    actual: q.to_string(),
                });
            }
            vec![*q as usize]
        },
        _ => return Err(VmError::ArgumentTypeError {
            function_name: "ParameterizedGate".to_string(),
            param_index: 1,
            expected: "Integer or list of integers".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let param_gate = ParameterizedGate::new(gate_type, qubits);
    Ok(Value::LyObj(LyObj::new(Box::new(param_gate))))
}

/// Create variational quantum circuit
pub fn variational_circuit(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::ArityError { 
            function_name: "VariationalCircuit".to_string(), 
            expected: 1, 
            actual: args.len() 
        });
    }

    let n_qubits = match &args[0] {
        Value::Integer(n) => {
            if *n <= 0 || *n > 20 {
                return Err(VmError::ArgumentTypeError {
                    function_name: "VariationalCircuit".to_string(),
                    param_index: 0,
                    expected: "Positive integer ≤ 20".to_string(),
                    actual: n.to_string(),
                });
            }
            *n as usize
        },
        _ => return Err(VmError::ArgumentTypeError {
            function_name: "VariationalCircuit".to_string(),
            param_index: 0,
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let circuit = VariationalCircuit::new(n_qubits);
    Ok(Value::LyObj(LyObj::new(Box::new(circuit))))
}

/// Create Pauli string observable
pub fn pauli_observable(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::ArityError { 
            function_name: "PauliObservable".to_string(), 
            expected: 2, 
            actual: args.len() 
        });
    }

    let pauli_string = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::ArgumentTypeError {
            function_name: "PauliObservable".to_string(),
            param_index: 0,
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let coefficient = if args.len() > 1 {
        match &args[1] {
            Value::Real(c) => *c,
            Value::Integer(c) => *c as f64,
            _ => return Err(VmError::ArgumentTypeError {
                function_name: "PauliObservable".to_string(),
                param_index: 1,
                expected: "Number".to_string(),
                actual: format!("{:?}", args[1]),
            }),
        }
    } else {
        1.0
    };

    let observable = PauliStringObservable::new(pauli_string, coefficient)
        .map_err(|e| VmError::Runtime(format!("Failed to create Pauli observable: {:?}", e)))?;

    Ok(Value::LyObj(LyObj::new(Box::new(observable))))
}

/// Create quantum gradient computer using parameter shift rule
pub fn quantum_gradient_computer(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::ArityError { 
            function_name: "QuantumGradientComputer".to_string(), 
            expected: 2, 
            actual: args.len() 
        });
    }

    let circuit = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<VariationalCircuit>()
                .ok_or_else(|| VmError::ArgumentTypeError {
                    function_name: "QuantumGradientComputer".to_string(),
                    param_index: 0,
                    expected: "VariationalCircuit".to_string(),
                    actual: "Unknown".to_string(),
                })?
                .clone()
        },
        _ => return Err(VmError::ArgumentTypeError {
            function_name: "QuantumGradientComputer".to_string(),
            param_index: 0,
            expected: "VariationalCircuit".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let observable = match &args[1] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<PauliStringObservable>()
                .ok_or_else(|| VmError::ArgumentTypeError {
                    function_name: "QuantumGradientComputer".to_string(),
                    param_index: 1,
                    expected: "PauliStringObservable".to_string(),
                    actual: "Unknown".to_string(),
                })?
                .clone()
        },
        _ => return Err(VmError::ArgumentTypeError {
            function_name: "QuantumGradientComputer".to_string(),
            param_index: 1,
            expected: "PauliStringObservable".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let gradient_computer = QuantumGradientComputer::new(circuit, observable);
    Ok(Value::LyObj(LyObj::new(Box::new(gradient_computer))))
}