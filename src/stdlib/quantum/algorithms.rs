//! Quantum Algorithm Implementations
//! 
//! This module provides implementations of important quantum algorithms:
//! - Quantum Fourier Transform (QFT)
//! - Grover's search algorithm
//! - Quantum phase estimation
//! - Quantum teleportation protocol
//! - Quantum error correction codes
//! - Shor's algorithm components

use crate::vm::{Value, VmResult, VmError};
use crate::foreign::{LyObj, Foreign, ForeignError};
use super::{QuantumMatrix, Complex, QuantumState, validate_qubit_count, validate_qubit_index};
use super::circuits::{QuantumCircuit, QubitRegister, CircuitGate};
use super::gates::QuantumGate;
use std::f64::consts::PI;

/// Quantum Fourier Transform
/// Creates a QFT circuit for n qubits
pub fn quantum_fourier_transform(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("QFT requires 1 argument (number of qubits)".to_string()));
    }
    
    let n_qubits = match &args[0] {
        Value::Number(n) => validate_qubit_count(*n)?,
        _ => return Err(VmError::Runtime("Number of qubits must be a number".to_string())),
    };
    
    let mut circuit = QuantumCircuit::new(n_qubits);
    
    // Build QFT circuit
    for i in 0..n_qubits {
        // Apply Hadamard gate
        let h_gate = create_hadamard_gate();
        circuit.add_gate(h_gate, vec![i])?;
        
        // Apply controlled phase gates
        for j in (i + 1)..n_qubits {
            let phase_angle = PI / (1 << (j - i)) as f64; // 2π/2^(j-i+1)
            let cp_gate = create_controlled_phase_gate(phase_angle);
            circuit.add_gate(cp_gate, vec![j, i])?; // Control on j, target on i
        }
    }
    
    // Swap qubits to reverse the order (QFT convention)
    for i in 0..(n_qubits / 2) {
        let swap_gate = create_swap_gate();
        circuit.add_gate(swap_gate, vec![i, n_qubits - 1 - i])?;
    }
    
    Ok(Value::LyObj(LyObj::new(Box::new(circuit))))
}

/// Inverse Quantum Fourier Transform
pub fn inverse_quantum_fourier_transform(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("Inverse QFT requires 1 argument (number of qubits)".to_string()));
    }
    
    let n_qubits = match &args[0] {
        Value::Number(n) => validate_qubit_count(*n)?,
        _ => return Err(VmError::Runtime("Number of qubits must be a number".to_string())),
    };
    
    let mut circuit = QuantumCircuit::new(n_qubits);
    
    // Inverse QFT is QFT† (reverse order and conjugate phases)
    
    // First, swap qubits back
    for i in 0..(n_qubits / 2) {
        let swap_gate = create_swap_gate();
        circuit.add_gate(swap_gate, vec![i, n_qubits - 1 - i])?;
    }
    
    // Apply inverse operations in reverse order
    for i in (0..n_qubits).rev() {
        // Apply controlled phase gates with negative phases
        for j in ((i + 1)..n_qubits).rev() {
            let phase_angle = -PI / (1 << (j - i)) as f64; // Negative phase for inverse
            let cp_gate = create_controlled_phase_gate(phase_angle);
            circuit.add_gate(cp_gate, vec![j, i])?;
        }
        
        // Apply Hadamard gate
        let h_gate = create_hadamard_gate();
        circuit.add_gate(h_gate, vec![i])?;
    }
    
    Ok(Value::LyObj(LyObj::new(Box::new(circuit))))
}

/// Grover's Search Algorithm
/// Searches for a marked item in an unsorted database
pub fn grovers_search(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime("Grover's search requires oracle and number of qubits".to_string()));
    }
    
    let oracle = match &args[0] {
        Value::LyObj(oracle_obj) => {
            if let Some(o) = oracle_obj.as_any().downcast_ref::<QuantumGate>() {
                o
            } else {
                return Err(VmError::Runtime("First argument must be an oracle (QuantumGate)".to_string()));
            }
        }
        _ => return Err(VmError::Runtime("First argument must be an oracle (QuantumGate)".to_string())),
    };
    
    let n_qubits = match &args[1] {
        Value::Number(n) => validate_qubit_count(*n)?,
        _ => return Err(VmError::Runtime("Number of qubits must be a number".to_string())),
    };
    
    if oracle.n_qubits != n_qubits {
        return Err(VmError::Runtime("Oracle qubit count doesn't match specified qubits".to_string()));
    }
    
    let mut circuit = QuantumCircuit::new(n_qubits);
    
    // Step 1: Create uniform superposition
    for i in 0..n_qubits {
        let h_gate = create_hadamard_gate();
        circuit.add_gate(h_gate, vec![i])?;
    }
    
    // Step 2: Calculate optimal number of iterations
    let n_items = 1 << n_qubits; // 2^n items
    let optimal_iterations = ((PI / 4.0) * (n_items as f64).sqrt()).round() as usize;
    
    // Step 3: Grover iterations
    for _ in 0..optimal_iterations {
        // Apply oracle
        circuit.add_gate(oracle.clone(), (0..n_qubits).collect())?;
        
        // Apply diffusion operator (inversion about average)
        let diffusion_gate = create_grover_diffusion_operator(n_qubits)?;
        circuit.add_gate(diffusion_gate, (0..n_qubits).collect())?;
    }
    
    Ok(Value::LyObj(LyObj::new(Box::new(circuit))))
}

/// Create Grover Oracle for a specific marked item
pub fn grover_oracle(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime("Grover oracle requires marked item index and number of qubits".to_string()));
    }
    
    let marked_item = match &args[0] {
        Value::Number(item) => {
            if *item < 0.0 || item.fract() != 0.0 {
                return Err(VmError::Runtime("Marked item must be non-negative integer".to_string()));
            }
            *item as usize
        }
        _ => return Err(VmError::Runtime("Marked item must be a number".to_string())),
    };
    
    let n_qubits = match &args[1] {
        Value::Number(n) => validate_qubit_count(*n)?,
        _ => return Err(VmError::Runtime("Number of qubits must be a number".to_string())),
    };
    
    let n_items = 1 << n_qubits;
    if marked_item >= n_items {
        return Err(VmError::Runtime(
            format!("Marked item {} out of range for {} qubits (max {})", 
                   marked_item, n_qubits, n_items - 1)
        ));
    }
    
    // Create oracle matrix (diagonal matrix with -1 for marked item)
    let mut oracle_matrix = QuantumMatrix::identity(n_items);
    oracle_matrix.data[marked_item][marked_item] = Complex::new(-1.0, 0.0);
    
    let oracle = QuantumGate::new(oracle_matrix, format!("Oracle({})", marked_item));
    Ok(Value::LyObj(LyObj::new(Box::new(oracle))))
}

/// Quantum Phase Estimation Algorithm
pub fn quantum_phase_estimation(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::Runtime("Phase estimation requires unitary, eigenstate, and counting qubits".to_string()));
    }
    
    let unitary = match &args[0] {
        Value::LyObj(unitary_obj) => {
            if let Some(u) = unitary_obj.as_any().downcast_ref::<QuantumGate>() {
                u
            } else {
                return Err(VmError::Runtime("First argument must be a unitary gate".to_string()));
            }
        }
        _ => return Err(VmError::Runtime("First argument must be a unitary gate".to_string())),
    };
    
    let _eigenstate = match &args[1] {
        Value::LyObj(state_obj) => {
            if let Some(s) = state_obj.as_any().downcast_ref::<QubitRegister>() {
                s
            } else {
                return Err(VmError::Runtime("Second argument must be a quantum state".to_string()));
            }
        }
        _ => return Err(VmError::Runtime("Second argument must be a quantum state".to_string())),
    };
    
    let counting_qubits = match &args[2] {
        Value::Number(n) => validate_qubit_count(*n)?,
        _ => return Err(VmError::Runtime("Counting qubits must be a number".to_string())),
    };
    
    let total_qubits = counting_qubits + unitary.n_qubits;
    let mut circuit = QuantumCircuit::new(total_qubits);
    
    // Step 1: Create superposition in counting qubits
    for i in 0..counting_qubits {
        let h_gate = create_hadamard_gate();
        circuit.add_gate(h_gate, vec![i])?;
    }
    
    // Step 2: Apply controlled unitary operations
    for i in 0..counting_qubits {
        let power = 1 << i; // 2^i
        
        // Apply U^(2^i) controlled by qubit i
        for _ in 0..power {
            // This is a simplified version - full implementation would create controlled-U
            let controlled_u = create_controlled_unitary(unitary)?;
            let control_qubits = vec![i];
            let target_qubits: Vec<usize> = (counting_qubits..total_qubits).collect();
            
            circuit.add_controlled_gate(controlled_u, target_qubits, control_qubits)?;
        }
    }
    
    // Step 3: Apply inverse QFT to counting qubits
    let inv_qft = create_inverse_qft_for_range(counting_qubits)?;
    circuit.add_gate(inv_qft, (0..counting_qubits).collect())?;
    
    Ok(Value::LyObj(LyObj::new(Box::new(circuit))))
}

/// Quantum Teleportation Protocol
pub fn quantum_teleportation(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("Quantum teleportation requires state to teleport".to_string()));
    }
    
    let state_to_teleport = match &args[0] {
        Value::LyObj(state_obj) => {
            if let Some(s) = state_obj.as_any().downcast_ref::<QubitRegister>() {
                if s.state.n_qubits != 1 {
                    return Err(VmError::Runtime("Can only teleport single-qubit states".to_string()));
                }
                s
            } else {
                return Err(VmError::Runtime("Argument must be a quantum state".to_string()));
            }
        }
        _ => return Err(VmError::Runtime("Argument must be a quantum state".to_string())),
    };
    
    // Create 3-qubit circuit for teleportation
    let mut circuit = QuantumCircuit::new(3);
    
    // Qubit 0: State to teleport (already prepared)
    // Qubit 1: Alice's part of Bell pair
    // Qubit 2: Bob's part of Bell pair
    
    // Step 1: Create Bell pair between qubits 1 and 2
    let h_gate = create_hadamard_gate();
    circuit.add_gate(h_gate, vec![1])?;
    
    let cnot_gate = create_cnot_gate();
    circuit.add_gate(cnot_gate, vec![1, 2])?; // Control: 1, Target: 2
    
    // Step 2: Bell measurement on qubits 0 and 1
    circuit.add_gate(cnot_gate.clone(), vec![0, 1])?; // Control: 0, Target: 1
    circuit.add_gate(h_gate.clone(), vec![0])?;
    
    // Step 3: Add measurements
    circuit.add_measurement(0, 0)?; // Measure qubit 0 to classical bit 0
    circuit.add_measurement(1, 1)?; // Measure qubit 1 to classical bit 1
    
    // Step 4: Conditional operations on qubit 2 based on measurements
    // (This would be handled during circuit execution based on measurement results)
    
    // Create initial 3-qubit state with the state to teleport in position 0
    let mut initial_amplitudes = vec![Complex::zero(); 8];
    
    // Copy the state to teleport into the full 3-qubit system
    for i in 0..2 {
        // |ψ⟩ ⊗ |00⟩ (Bell pair starts as |00⟩)
        initial_amplitudes[i * 4] = state_to_teleport.state.amplitudes[i].clone();
    }
    
    let initial_state = QuantumState::from_amplitudes(initial_amplitudes)?;
    let initial_register = QubitRegister::from_state(initial_state);
    
    // Execute the circuit
    let (final_state, measurements) = circuit.execute(&initial_register.state)?;
    
    // Return the teleportation results
    let final_register = QubitRegister::from_state(final_state);
    let measurements_list: Vec<Value> = measurements.iter()
        .map(|(&bit, &result)| Value::List(vec![
            Value::Number(bit as f64),
            Value::Number(result as f64),
        ]))
        .collect();
    
    Ok(Value::List(vec![
        Value::LyObj(LyObj::new(Box::new(final_register))),
        Value::List(measurements_list),
    ]))
}

/// Three-Qubit Bit Flip Error Correction
pub fn three_qubit_encode(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("Three-qubit encoding requires 1 data qubit".to_string()));
    }
    
    let data_qubit = match &args[0] {
        Value::LyObj(state_obj) => {
            if let Some(s) = state_obj.as_any().downcast_ref::<QubitRegister>() {
                if s.state.n_qubits != 1 {
                    return Err(VmError::Runtime("Can only encode single-qubit states".to_string()));
                }
                s
            } else {
                return Err(VmError::Runtime("Argument must be a quantum state".to_string()));
            }
        }
        _ => return Err(VmError::Runtime("Argument must be a quantum state".to_string())),
    };
    
    // Create 3-qubit encoded state: |ψ⟩ → α|000⟩ + β|111⟩
    let alpha = data_qubit.state.amplitudes[0].clone(); // Amplitude for |0⟩
    let beta = data_qubit.state.amplitudes[1].clone();  // Amplitude for |1⟩
    
    let mut encoded_amplitudes = vec![Complex::zero(); 8];
    encoded_amplitudes[0] = alpha; // |000⟩
    encoded_amplitudes[7] = beta;  // |111⟩
    
    let encoded_state = QuantumState::from_amplitudes(encoded_amplitudes)?;
    let encoded_register = QubitRegister::from_state(encoded_state);
    
    Ok(Value::LyObj(LyObj::new(Box::new(encoded_register))))
}

/// Apply bit flip error to encoded state
pub fn apply_bit_flip_error(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime("Bit flip error requires state and qubit index".to_string()));
    }
    
    let encoded_state = match &args[0] {
        Value::LyObj(state_obj) => {
            if let Some(s) = state_obj.as_any().downcast_ref::<QubitRegister>() {
                s
            } else {
                return Err(VmError::Runtime("First argument must be a quantum state".to_string()));
            }
        }
        _ => return Err(VmError::Runtime("First argument must be a quantum state".to_string())),
    };
    
    let qubit_index = match &args[1] {
        Value::Number(q) => validate_qubit_index(*q, encoded_state.state.n_qubits)?,
        _ => return Err(VmError::Runtime("Qubit index must be a number".to_string())),
    };
    
    // Apply X gate to simulate bit flip error
    let mut corrupted_state = encoded_state.state.clone();
    let x_gate = create_pauli_x_gate();
    corrupted_state.apply_gate(&x_gate.matrix, &[qubit_index])?;
    
    let corrupted_register = QubitRegister::from_state(corrupted_state);
    Ok(Value::LyObj(LyObj::new(Box::new(corrupted_register))))
}

/// Correct bit flip errors in three-qubit code
pub fn three_qubit_correct(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("Three-qubit correction requires encoded state".to_string()));
    }
    
    let encoded_state = match &args[0] {
        Value::LyObj(state_obj) => {
            if let Some(s) = state_obj.as_any().downcast_ref::<QubitRegister>() {
                if s.state.n_qubits != 3 {
                    return Err(VmError::Runtime("Three-qubit correction requires 3-qubit state".to_string()));
                }
                s
            } else {
                return Err(VmError::Runtime("Argument must be a quantum state".to_string()));
            }
        }
        _ => return Err(VmError::Runtime("Argument must be a quantum state".to_string())),
    };
    
    // Create error correction circuit
    let mut circuit = QuantumCircuit::new(5); // 3 data + 2 syndrome qubits
    
    // Copy input state to first 3 qubits (this is conceptual - actual implementation would be different)
    
    // Syndrome extraction
    let cnot_gate = create_cnot_gate();
    
    // First syndrome qubit (parity of qubits 0 and 1)
    circuit.add_gate(cnot_gate.clone(), vec![0, 3])?;
    circuit.add_gate(cnot_gate.clone(), vec![1, 3])?;
    
    // Second syndrome qubit (parity of qubits 1 and 2)
    circuit.add_gate(cnot_gate.clone(), vec![1, 4])?;
    circuit.add_gate(cnot_gate.clone(), vec![2, 4])?;
    
    // Error correction based on syndrome
    // This is a simplified implementation
    let corrected_register = QubitRegister::from_state(encoded_state.state.clone());
    
    Ok(Value::LyObj(LyObj::new(Box::new(corrected_register))))
}

/// Decode three-qubit error corrected state
pub fn three_qubit_decode(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("Three-qubit decoding requires encoded state".to_string()));
    }
    
    let encoded_state = match &args[0] {
        Value::LyObj(state_obj) => {
            if let Some(s) = state_obj.as_any().downcast_ref::<QubitRegister>() {
                if s.state.n_qubits != 3 {
                    return Err(VmError::Runtime("Three-qubit decoding requires 3-qubit state".to_string()));
                }
                s
            } else {
                return Err(VmError::Runtime("Argument must be a quantum state".to_string()));
            }
        }
        _ => return Err(VmError::Runtime("Argument must be a quantum state".to_string())),
    };
    
    // Extract logical qubit from encoded state
    // For perfect encoding: α|000⟩ + β|111⟩ → α|0⟩ + β|1⟩
    
    let alpha = encoded_state.state.amplitudes[0].clone(); // |000⟩ amplitude
    let beta = encoded_state.state.amplitudes[7].clone();  // |111⟩ amplitude
    
    let decoded_amplitudes = vec![alpha, beta];
    let decoded_state = QuantumState::from_amplitudes(decoded_amplitudes)?;
    let decoded_register = QubitRegister::from_state(decoded_state);
    
    Ok(Value::LyObj(LyObj::new(Box::new(decoded_register))))
}

// Entanglement and Advanced Operations

/// Measure entanglement in a quantum state
pub fn entanglement_measure(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("Entanglement measure requires 1 quantum state".to_string()));
    }
    
    let _state = match &args[0] {
        Value::LyObj(state_obj) => {
            if let Some(s) = state_obj.as_any().downcast_ref::<QubitRegister>() {
                s
            } else {
                return Err(VmError::Runtime("Argument must be a quantum state".to_string()));
            }
        }
        _ => return Err(VmError::Runtime("Argument must be a quantum state".to_string())),
    };
    
    // For now, return a placeholder value
    // Full implementation would calculate von Neumann entropy of reduced density matrix
    Ok(Value::Number(0.5)) // Placeholder entanglement measure
}

/// Measure Bell state and return correlations
pub fn measure_bell_state(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("Bell state measurement requires 1 quantum state".to_string()));
    }
    
    let state = match &args[0] {
        Value::LyObj(state_obj) => {
            if let Some(s) = state_obj.as_any().downcast_ref::<QubitRegister>() {
                if s.state.n_qubits != 2 {
                    return Err(VmError::Runtime("Bell state measurement requires 2-qubit state".to_string()));
                }
                s
            } else {
                return Err(VmError::Runtime("Argument must be a quantum state".to_string()));
            }
        }
        _ => return Err(VmError::Runtime("Argument must be a quantum state".to_string())),
    };
    
    // Measure both qubits and return results
    let mut measurement_state = state.state.clone();
    let measurement_0 = measurement_state.measure_qubit(0)?;
    let measurement_1 = measurement_state.measure_qubit(1)?;
    
    Ok(Value::List(vec![
        Value::Number(measurement_0 as f64),
        Value::Number(measurement_1 as f64),
    ]))
}

// Helper functions for creating standard gates

fn create_hadamard_gate() -> QuantumGate {
    let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
    let matrix = QuantumMatrix::from_data(vec![
        vec![Complex::new(inv_sqrt2, 0.0), Complex::new(inv_sqrt2, 0.0)],
        vec![Complex::new(inv_sqrt2, 0.0), Complex::new(-inv_sqrt2, 0.0)],
    ]).unwrap();
    
    QuantumGate::new(matrix, "H".to_string())
}

fn create_pauli_x_gate() -> QuantumGate {
    let matrix = QuantumMatrix::from_data(vec![
        vec![Complex::zero(), Complex::one()],
        vec![Complex::one(), Complex::zero()],
    ]).unwrap();
    
    QuantumGate::new(matrix, "X".to_string())
}

fn create_cnot_gate() -> QuantumGate {
    let matrix = QuantumMatrix::from_data(vec![
        vec![Complex::one(), Complex::zero(), Complex::zero(), Complex::zero()],
        vec![Complex::zero(), Complex::one(), Complex::zero(), Complex::zero()],
        vec![Complex::zero(), Complex::zero(), Complex::zero(), Complex::one()],
        vec![Complex::zero(), Complex::zero(), Complex::one(), Complex::zero()],
    ]).unwrap();
    
    QuantumGate::new(matrix, "CNOT".to_string())
}

fn create_swap_gate() -> QuantumGate {
    let matrix = QuantumMatrix::from_data(vec![
        vec![Complex::one(), Complex::zero(), Complex::zero(), Complex::zero()],
        vec![Complex::zero(), Complex::zero(), Complex::one(), Complex::zero()],
        vec![Complex::zero(), Complex::one(), Complex::zero(), Complex::zero()],
        vec![Complex::zero(), Complex::zero(), Complex::zero(), Complex::one()],
    ]).unwrap();
    
    QuantumGate::new(matrix, "SWAP".to_string())
}

fn create_controlled_phase_gate(phase: f64) -> QuantumGate {
    let matrix = QuantumMatrix::from_data(vec![
        vec![Complex::one(), Complex::zero(), Complex::zero(), Complex::zero()],
        vec![Complex::zero(), Complex::one(), Complex::zero(), Complex::zero()],
        vec![Complex::zero(), Complex::zero(), Complex::one(), Complex::zero()],
        vec![Complex::zero(), Complex::zero(), Complex::zero(), Complex::new(phase.cos(), phase.sin())],
    ]).unwrap();
    
    QuantumGate::new(matrix, format!("CP({:.3})", phase))
}

fn create_grover_diffusion_operator(n_qubits: usize) -> Result<QuantumGate, VmError> {
    let n = 1 << n_qubits; // 2^n
    let mut matrix = QuantumMatrix::identity(n);
    
    // Diffusion operator: 2|s⟩⟨s| - I, where |s⟩ is uniform superposition
    let amplitude = 1.0 / (n as f64).sqrt();
    
    for i in 0..n {
        for j in 0..n {
            if i == j {
                matrix.data[i][j] = Complex::new(2.0 * amplitude * amplitude - 1.0, 0.0);
            } else {
                matrix.data[i][j] = Complex::new(2.0 * amplitude * amplitude, 0.0);
            }
        }
    }
    
    Ok(QuantumGate::new(matrix, format!("Diffusion({})", n_qubits)))
}

fn create_controlled_unitary(unitary: &QuantumGate) -> Result<QuantumGate, VmError> {
    if unitary.n_qubits != 1 {
        return Err(VmError::Runtime("Can only create controlled version of single-qubit gates".to_string()));
    }
    
    let mut controlled_matrix = QuantumMatrix::identity(4);
    
    // Copy the unitary to the bottom-right 2x2 block
    for i in 0..2 {
        for j in 0..2 {
            controlled_matrix.data[i + 2][j + 2] = unitary.matrix.data[i][j].clone();
        }
    }
    
    Ok(QuantumGate::new(controlled_matrix, format!("C{}", unitary.name)))
}

fn create_inverse_qft_for_range(n_qubits: usize) -> Result<QuantumGate, VmError> {
    // This is a simplified placeholder - full implementation would build the actual inverse QFT matrix
    let matrix = QuantumMatrix::identity(1 << n_qubits);
    Ok(QuantumGate::new(matrix, format!("IQFT({})", n_qubits)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qft_creation() {
        let qft = quantum_fourier_transform(&[Value::Number(3.0)]).unwrap();
        
        if let Value::LyObj(circuit_obj) = qft {
            let circuit = circuit_obj.as_any().downcast_ref::<QuantumCircuit>().unwrap();
            assert_eq!(circuit.n_qubits, 3);
            assert!(circuit.gates.len() > 0); // Should have gates
        }
    }

    #[test]
    fn test_grover_oracle_creation() {
        let oracle = grover_oracle(&[Value::Number(2.0), Value::Number(2.0)]).unwrap();
        
        if let Value::LyObj(oracle_obj) = oracle {
            let oracle_gate = oracle_obj.as_any().downcast_ref::<QuantumGate>().unwrap();
            assert_eq!(oracle_gate.n_qubits, 2);
            assert!(oracle_gate.matrix.is_unitary(1e-10));
            
            // Check that it flips the sign of state |10⟩ (index 2)
            assert_eq!(oracle_gate.matrix.data[2][2].real, -1.0);
        }
    }

    #[test]
    fn test_grovers_search() {
        let oracle = grover_oracle(&[Value::Number(1.0), Value::Number(2.0)]).unwrap();
        let grover_circuit = grovers_search(&[oracle, Value::Number(2.0)]).unwrap();
        
        if let Value::LyObj(circuit_obj) = grover_circuit {
            let circuit = circuit_obj.as_any().downcast_ref::<QuantumCircuit>().unwrap();
            assert_eq!(circuit.n_qubits, 2);
            // Should have initial Hadamards + Grover iterations
            assert!(circuit.gates.len() >= 2); // At least H gates for superposition
        }
    }

    #[test]
    fn test_three_qubit_encoding() {
        // Create a test qubit state |+⟩ = (|0⟩ + |1⟩)/√2
        let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
        let amplitudes = vec![Complex::new(inv_sqrt2, 0.0), Complex::new(inv_sqrt2, 0.0)];
        let test_state = QuantumState::from_amplitudes(amplitudes).unwrap();
        let test_register = QubitRegister::from_state(test_state);
        
        let encoded = three_qubit_encode(&[Value::LyObj(LyObj::new(Box::new(test_register)))]).unwrap();
        
        if let Value::LyObj(encoded_obj) = encoded {
            let encoded_register = encoded_obj.as_any().downcast_ref::<QubitRegister>().unwrap();
            assert_eq!(encoded_register.state.n_qubits, 3);
            
            // Check that only |000⟩ and |111⟩ have non-zero amplitudes
            for (i, amplitude) in encoded_register.state.amplitudes.iter().enumerate() {
                if i == 0 || i == 7 {
                    assert!(amplitude.magnitude() > 1e-10);
                } else {
                    assert!(amplitude.magnitude() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_bell_state_measurement() {
        let bell_state = super::super::circuits::create_bell_state(&[Value::Number(0.0)]).unwrap();
        let measurement = measure_bell_state(&[bell_state]).unwrap();
        
        if let Value::List(results) = measurement {
            assert_eq!(results.len(), 2);
            
            // Both measurements should be numbers (0 or 1)
            for result in results {
                match result {
                    Value::Number(n) => assert!(n == 0.0 || n == 1.0),
                    _ => panic!("Measurement result should be a number"),
                }
            }
        }
    }

    #[test]
    fn test_quantum_teleportation() {
        // Create a test state to teleport
        let amplitudes = vec![Complex::new(0.6, 0.0), Complex::new(0.8, 0.0)];
        let test_state = QuantumState::from_amplitudes(amplitudes).unwrap();
        let test_register = QubitRegister::from_state(test_state);
        
        let teleportation_result = quantum_teleportation(&[Value::LyObj(LyObj::new(Box::new(test_register)))]).unwrap();
        
        if let Value::List(results) = teleportation_result {
            assert_eq!(results.len(), 2); // [final_state, measurements]
            
            // Should have final state and measurement results
            match &results[0] {
                Value::LyObj(state_obj) => {
                    let _final_state = state_obj.as_any().downcast_ref::<QubitRegister>().unwrap();
                    // Final state should be a 3-qubit state
                }
                _ => panic!("First result should be a quantum state"),
            }
            
            match &results[1] {
                Value::List(_measurements) => {
                    // Should have measurement results
                }
                _ => panic!("Second result should be measurement list"),
            }
        }
    }
}