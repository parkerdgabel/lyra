//! Quantum Computing Simulation Framework
//! 
//! Comprehensive quantum computing framework for the Lyra programming language.
//! Implements quantum gates, circuits, algorithms, and advanced quantum operations.
//! 
//! All quantum objects are implemented as Foreign objects using the LyObj wrapper
//! to maintain VM simplicity while providing powerful quantum simulation capabilities.

use crate::vm::VmError;
use crate::foreign::ForeignError;
use crate::stdlib::common::Complex;

pub mod gates;
pub mod circuits;
pub mod algorithms;

// Re-export all public functions
pub use gates::*;
pub use circuits::*;
pub use algorithms::*;

/// Convert ForeignError to VmError for stdlib functions
pub fn foreign_to_vm_error(err: ForeignError) -> VmError {
    match err {
        ForeignError::UnknownMethod { method, type_name } => {
            VmError::Runtime(format!("Unknown method '{}' for type '{}'", method, type_name))
        }
        ForeignError::InvalidArity { method, expected, actual } => {
            VmError::Runtime(format!("Invalid arity for method '{}': expected {}, got {}", method, expected, actual))
        }
        ForeignError::InvalidArgumentType { method, expected, actual } => {
            VmError::Runtime(format!("Invalid argument type for method '{}': expected {}, got {}", method, expected, actual))
        }
        ForeignError::IndexOutOfBounds { index, bounds } => {
            VmError::Runtime(format!("Index out of bounds: {} not in range {}", index, bounds))
        }
        ForeignError::RuntimeError { message } => {
            VmError::Runtime(message)
        }
        ForeignError::ArgumentError { expected, actual } => {
            VmError::Runtime(format!("Argument error: expected {}, got {}", expected, actual))
        }
        ForeignError::TypeError { expected, actual } => {
            VmError::TypeError { expected, actual }
        }
        ForeignError::InvalidArgument(message) => {
            VmError::Runtime(message)
        }
    }
}


/// Quantum matrix representation for gates and operations
#[derive(Debug, Clone)]
pub struct QuantumMatrix {
    pub data: Vec<Vec<Complex>>,
    pub rows: usize,
    pub cols: usize,
}

impl QuantumMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![vec![Complex::zero(); cols]; rows],
            rows,
            cols,
        }
    }
    
    pub fn identity(size: usize) -> Self {
        let mut matrix = Self::new(size, size);
        for i in 0..size {
            matrix.data[i][i] = Complex::one();
        }
        matrix
    }
    
    pub fn from_data(data: Vec<Vec<Complex>>) -> Result<Self, VmError> {
        if data.is_empty() {
            return Err(VmError::Runtime("Empty matrix data".to_string()));
        }
        
        let rows = data.len();
        let cols = data[0].len();
        
        // Verify all rows have the same length
        for row in &data {
            if row.len() != cols {
                return Err(VmError::Runtime("Inconsistent matrix dimensions".to_string()));
            }
        }
        
        Ok(Self { data, rows, cols })
    }
    
    pub fn multiply(&self, other: &QuantumMatrix) -> Result<QuantumMatrix, VmError> {
        if self.cols != other.rows {
            return Err(VmError::Runtime(
                format!("Matrix dimension mismatch: {}x{} * {}x{}", 
                       self.rows, self.cols, other.rows, other.cols)
            ));
        }
        
        let mut result = QuantumMatrix::new(self.rows, other.cols);
        
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = Complex::zero();
                for k in 0..self.cols {
                    sum = sum + self.data[i][k].clone() * other.data[k][j].clone();
                }
                result.data[i][j] = sum;
            }
        }
        
        Ok(result)
    }
    
    pub fn apply_to_state(&self, state: &[Complex]) -> Result<Vec<Complex>, VmError> {
        if self.cols != state.len() {
            return Err(VmError::Runtime(
                format!("State dimension mismatch: matrix {}x{}, state length {}", 
                       self.rows, self.cols, state.len())
            ));
        }
        
        let mut result = vec![Complex::zero(); self.rows];
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i] = result[i].clone() + self.data[i][j].clone() * state[j].clone();
            }
        }
        
        Ok(result)
    }
    
    pub fn tensor_product(&self, other: &QuantumMatrix) -> QuantumMatrix {
        let new_rows = self.rows * other.rows;
        let new_cols = self.cols * other.cols;
        let mut result = QuantumMatrix::new(new_rows, new_cols);
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                for k in 0..other.rows {
                    for l in 0..other.cols {
                        let result_row = i * other.rows + k;
                        let result_col = j * other.cols + l;
                        result.data[result_row][result_col] = 
                            self.data[i][j].clone() * other.data[k][l].clone();
                    }
                }
            }
        }
        
        result
    }
    
    pub fn dagger(&self) -> QuantumMatrix {
        let mut result = QuantumMatrix::new(self.cols, self.rows);
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j][i] = self.data[i][j].conjugate();
            }
        }
        
        result
    }
    
    pub fn is_unitary(&self, tolerance: f64) -> bool {
        if self.rows != self.cols {
            return false;
        }
        
        let dagger = self.dagger();
        match self.multiply(&dagger) {
            Ok(product) => {
                let identity = QuantumMatrix::identity(self.rows);
                
                // Check if product is close to identity
                for i in 0..self.rows {
                    for j in 0..self.cols {
                        let diff = product.data[i][j].clone() - identity.data[i][j].clone();
                        if diff.magnitude() > tolerance {
                            return false;
                        }
                    }
                }
                true
            }
            Err(_) => false,
        }
    }
}

/// Quantum state representation
#[derive(Debug, Clone)]
pub struct QuantumState {
    pub amplitudes: Vec<Complex>,
    pub n_qubits: usize,
}

impl QuantumState {
    pub fn new(n_qubits: usize) -> Self {
        let size = 1 << n_qubits; // 2^n_qubits
        let mut amplitudes = vec![Complex::zero(); size];
        amplitudes[0] = Complex::one(); // Initialize to |00...0⟩
        
        Self { amplitudes, n_qubits }
    }
    
    pub fn from_amplitudes(amplitudes: Vec<Complex>) -> Result<Self, VmError> {
        // Check if length is a power of 2
        let size = amplitudes.len();
        if size == 0 || (size & (size - 1)) != 0 {
            return Err(VmError::Runtime(
                "State vector length must be a power of 2".to_string()
            ));
        }
        
        let n_qubits = (size as f64).log2() as usize;
        
        // Check normalization
        let norm_squared: f64 = amplitudes.iter()
            .map(|amp| amp.magnitude_squared())
            .sum();
        
        if (norm_squared - 1.0).abs() > 1e-10 {
            return Err(VmError::Runtime(
                format!("State vector is not normalized: norm² = {}", norm_squared)
            ));
        }
        
        Ok(Self { amplitudes, n_qubits })
    }
    
    pub fn normalize(&mut self) {
        let norm_squared: f64 = self.amplitudes.iter()
            .map(|amp| amp.magnitude_squared())
            .sum();
        let norm = norm_squared.sqrt();
        
        if norm > 1e-10 {
            for amp in &mut self.amplitudes {
                *amp = amp.clone() * (1.0 / norm);
            }
        }
    }
    
    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes.iter()
            .map(|amp| amp.magnitude_squared())
            .collect()
    }
    
    pub fn measure_qubit(&mut self, qubit_index: usize) -> Result<usize, VmError> {
        if qubit_index >= self.n_qubits {
            return Err(VmError::Runtime(
                format!("Qubit index {} out of range for {}-qubit system", 
                       qubit_index, self.n_qubits)
            ));
        }
        
        // Calculate probabilities for measuring 0 and 1
        let mut prob_0 = 0.0;
        let mut prob_1 = 0.0;
        
        for i in 0..self.amplitudes.len() {
            let bit = (i >> qubit_index) & 1;
            let prob = self.amplitudes[i].magnitude_squared();
            
            if bit == 0 {
                prob_0 += prob;
            } else {
                prob_1 += prob;
            }
        }
        
        // Perform measurement (random outcome based on probabilities)
        let random_value: f64 = rand::random();
        let outcome = if random_value < prob_0 { 0 } else { 1 };
        
        // Collapse the state
        let normalization = if outcome == 0 { prob_0.sqrt() } else { prob_1.sqrt() };
        
        for i in 0..self.amplitudes.len() {
            let bit = (i >> qubit_index) & 1;
            if bit != outcome {
                self.amplitudes[i] = Complex::zero();
            } else {
                self.amplitudes[i] = self.amplitudes[i].clone() * (1.0 / normalization);
            }
        }
        
        Ok(outcome)
    }
    
    pub fn apply_gate(&mut self, gate: &QuantumMatrix, qubits: &[usize]) -> Result<(), VmError> {
        // Verify gate dimensions match number of qubits
        let gate_qubits = (gate.rows as f64).log2() as usize;
        if qubits.len() != gate_qubits {
            return Err(VmError::Runtime(
                format!("Gate operates on {} qubits, but {} qubits specified", 
                       gate_qubits, qubits.len())
            ));
        }
        
        // Check if qubits are valid
        for &qubit in qubits {
            if qubit >= self.n_qubits {
                return Err(VmError::Runtime(
                    format!("Qubit index {} out of range", qubit)
                ));
            }
        }
        
        // For single-qubit gates, apply directly
        if gate_qubits == 1 {
            self.apply_single_qubit_gate(gate, qubits[0])?;
        } else {
            // For multi-qubit gates, construct full system gate and apply
            self.apply_multi_qubit_gate(gate, qubits)?;
        }
        
        Ok(())
    }
    
    fn apply_single_qubit_gate(&mut self, gate: &QuantumMatrix, qubit: usize) -> Result<(), VmError> {
        let mut new_amplitudes = vec![Complex::zero(); self.amplitudes.len()];
        
        for i in 0..self.amplitudes.len() {
            let bit = (i >> qubit) & 1;
            let other_bits = i & !(1 << qubit);
            
            // Apply gate to this basis state
            let state_0 = other_bits;
            let state_1 = other_bits | (1 << qubit);
            
            if bit == 0 {
                new_amplitudes[state_0] = new_amplitudes[state_0].clone() + 
                    gate.data[0][0].clone() * self.amplitudes[i].clone();
                new_amplitudes[state_1] = new_amplitudes[state_1].clone() + 
                    gate.data[1][0].clone() * self.amplitudes[i].clone();
            } else {
                new_amplitudes[state_0] = new_amplitudes[state_0].clone() + 
                    gate.data[0][1].clone() * self.amplitudes[i].clone();
                new_amplitudes[state_1] = new_amplitudes[state_1].clone() + 
                    gate.data[1][1].clone() * self.amplitudes[i].clone();
            }
        }
        
        self.amplitudes = new_amplitudes;
        Ok(())
    }
    
    fn apply_multi_qubit_gate(&mut self, gate: &QuantumMatrix, qubits: &[usize]) -> Result<(), VmError> {
        // This is a simplified implementation - a full implementation would be more efficient
        let gate_size = gate.rows;
        let mut new_amplitudes = vec![Complex::zero(); self.amplitudes.len()];
        
        for i in 0..self.amplitudes.len() {
            // Extract the bits for the target qubits
            let mut target_state = 0;
            for (pos, &qubit) in qubits.iter().enumerate() {
                if (i >> qubit) & 1 == 1 {
                    target_state |= 1 << pos;
                }
            }
            
            // Apply gate to all possible target states
            for j in 0..gate_size {
                let gate_amp = gate.data[target_state][j].clone();
                if gate_amp.magnitude() > 1e-15 {
                    // Construct the new basis state
                    let mut new_i = i;
                    for (pos, &qubit) in qubits.iter().enumerate() {
                        if (j >> pos) & 1 == 1 {
                            new_i |= 1 << qubit;
                        } else {
                            new_i &= !(1 << qubit);
                        }
                    }
                    
                    new_amplitudes[new_i] = new_amplitudes[new_i].clone() + 
                        gate_amp * self.amplitudes[i].clone();
                }
            }
        }
        
        self.amplitudes = new_amplitudes;
        Ok(())
    }
}

/// Utility functions for quantum operations
pub fn validate_qubit_count(n_qubits: f64) -> Result<usize, VmError> {
    if n_qubits < 1.0 || n_qubits > 20.0 || n_qubits.fract() != 0.0 {
        return Err(VmError::Runtime(
            format!("Invalid qubit count: {}. Must be integer between 1 and 20", n_qubits)
        ));
    }
    Ok(n_qubits as usize)
}

pub fn validate_qubit_index(index: f64, n_qubits: usize) -> Result<usize, VmError> {
    if index < 0.0 || index >= n_qubits as f64 || index.fract() != 0.0 {
        return Err(VmError::Runtime(
            format!("Invalid qubit index: {}. Must be integer between 0 and {}", 
                   index, n_qubits - 1)
        ));
    }
    Ok(index as usize)
}

pub fn create_rotation_matrix(axis: char, angle: f64) -> QuantumMatrix {
    let cos_half = (angle / 2.0).cos();
    let sin_half = (angle / 2.0).sin();
    
    match axis {
        'X' | 'x' => {
            QuantumMatrix::from_data(vec![
                vec![Complex::new(cos_half, 0.0), Complex::new(0.0, -sin_half)],
                vec![Complex::new(0.0, -sin_half), Complex::new(cos_half, 0.0)],
            ]).unwrap()
        }
        'Y' | 'y' => {
            QuantumMatrix::from_data(vec![
                vec![Complex::new(cos_half, 0.0), Complex::new(-sin_half, 0.0)],
                vec![Complex::new(sin_half, 0.0), Complex::new(cos_half, 0.0)],
            ]).unwrap()
        }
        'Z' | 'z' => {
            QuantumMatrix::from_data(vec![
                vec![Complex::new(cos_half, -sin_half), Complex::zero()],
                vec![Complex::zero(), Complex::new(cos_half, sin_half)],
            ]).unwrap()
        }
        _ => QuantumMatrix::identity(2), // Default to identity for invalid axis
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_arithmetic() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        
        let sum = a.clone() + b.clone();
        assert_eq!(sum.real, 4.0);
        assert_eq!(sum.imag, 6.0);
        
        let product = a.clone() * b.clone();
        assert_eq!(product.real, -5.0); // (1*3 - 2*4)
        assert_eq!(product.imag, 10.0); // (1*4 + 2*3)
        
        assert!((a.magnitude() - (5.0_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_quantum_matrix_multiplication() {
        let pauli_x = QuantumMatrix::from_data(vec![
            vec![Complex::zero(), Complex::one()],
            vec![Complex::one(), Complex::zero()],
        ]).unwrap();
        
        let pauli_y = QuantumMatrix::from_data(vec![
            vec![Complex::zero(), Complex::new(0.0, -1.0)],
            vec![Complex::new(0.0, 1.0), Complex::zero()],
        ]).unwrap();
        
        let product = pauli_x.multiply(&pauli_y).unwrap();
        
        // X * Y = iZ
        let expected = QuantumMatrix::from_data(vec![
            vec![Complex::new(0.0, 1.0), Complex::zero()],
            vec![Complex::zero(), Complex::new(0.0, -1.0)],
        ]).unwrap();
        
        for i in 0..2 {
            for j in 0..2 {
                let diff = product.data[i][j].clone() - expected.data[i][j].clone();
                assert!(diff.magnitude() < 1e-10);
            }
        }
    }

    #[test]
    fn test_quantum_state_creation() {
        let state = QuantumState::new(2);
        assert_eq!(state.n_qubits, 2);
        assert_eq!(state.amplitudes.len(), 4);
        assert_eq!(state.amplitudes[0].real, 1.0);
        
        for i in 1..4 {
            assert_eq!(state.amplitudes[i].magnitude(), 0.0);
        }
    }

    #[test]
    fn test_quantum_state_normalization() {
        let mut state = QuantumState::from_amplitudes(vec![
            Complex::new(2.0, 0.0),
            Complex::new(0.0, 2.0),
            Complex::zero(),
            Complex::zero(),
        ]).unwrap_or_else(|_| {
            let mut state = QuantumState::new(2);
            state.amplitudes[0] = Complex::new(2.0, 0.0);
            state.amplitudes[1] = Complex::new(0.0, 2.0);
            state
        });
        
        state.normalize();
        
        let norm_squared: f64 = state.amplitudes.iter()
            .map(|amp| amp.magnitude_squared())
            .sum();
        
        assert!((norm_squared - 1.0).abs() < 1e-10);
    }
}