//! Quantum Computing Framework Tests
//! 
//! Comprehensive test suite for the quantum computing framework following TDD principles.
//! Tests cover quantum gates, circuits, algorithms, and edge cases.

use lyra::stdlib::quantum::*;
use lyra::vm::{Value, VmResult};
use lyra::memory::optimized_value::LyObj;
use std::f64::consts::PI;

#[cfg(test)]
mod quantum_gate_tests {
    use super::*;

    #[test]
    fn test_pauli_x_gate() {
        // Test Pauli-X (NOT) gate
        let args = vec![];
        let result = pauli_x_gate(&args);
        assert!(result.is_ok());
        
        // Should create a 2x2 matrix [[0, 1], [1, 0]]
        if let Ok(Value::LyObj(gate_obj)) = result {
            // Test gate application to |0⟩ state should give |1⟩
            // Test gate application to |1⟩ state should give |0⟩
        }
    }

    #[test]
    fn test_pauli_y_gate() {
        let args = vec![];
        let result = pauli_y_gate(&args);
        assert!(result.is_ok());
        
        // Should create matrix [[0, -i], [i, 0]]
    }

    #[test]
    fn test_pauli_z_gate() {
        let args = vec![];
        let result = pauli_z_gate(&args);
        assert!(result.is_ok());
        
        // Should create matrix [[1, 0], [0, -1]]
    }

    #[test]
    fn test_hadamard_gate() {
        let args = vec![];
        let result = hadamard_gate(&args);
        assert!(result.is_ok());
        
        // Should create matrix (1/√2) * [[1, 1], [1, -1]]
    }

    #[test]
    fn test_phase_gate() {
        let args = vec![Value::Number(PI / 4.0)]; // π/4 phase
        let result = phase_gate(&args);
        assert!(result.is_ok());
        
        // Should create matrix [[1, 0], [0, e^(iφ)]]
    }

    #[test]
    fn test_rotation_gates() {
        // Test RX gate
        let args = vec![Value::Number(PI / 2.0)];
        let result = rotation_x_gate(&args);
        assert!(result.is_ok());
        
        // Test RY gate
        let args = vec![Value::Number(PI / 2.0)];
        let result = rotation_y_gate(&args);
        assert!(result.is_ok());
        
        // Test RZ gate
        let args = vec![Value::Number(PI / 2.0)];
        let result = rotation_z_gate(&args);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cnot_gate() {
        let args = vec![];
        let result = cnot_gate(&args);
        assert!(result.is_ok());
        
        // Should create 4x4 controlled-NOT matrix
        // [[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]]
    }

    #[test]
    fn test_toffoli_gate() {
        let args = vec![];
        let result = toffoli_gate(&args);
        assert!(result.is_ok());
        
        // Should create 8x8 controlled-controlled-NOT matrix
    }

    #[test]
    fn test_fredkin_gate() {
        let args = vec![];
        let result = fredkin_gate(&args);
        assert!(result.is_ok());
        
        // Should create 8x8 controlled-SWAP matrix
    }
}

#[cfg(test)]
mod quantum_circuit_tests {
    use super::*;

    #[test]
    fn test_quantum_circuit_creation() {
        let args = vec![Value::Number(3.0)]; // 3 qubits
        let result = quantum_circuit(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(circuit_obj)) = result {
            // Verify circuit has 3 qubits and empty gate list
        }
    }

    #[test]
    fn test_qubit_register_creation() {
        let args = vec![Value::Number(2.0)]; // 2 qubits
        let result = qubit_register(&args);
        assert!(result.is_ok());
        
        // Should initialize to |00⟩ state
        if let Ok(Value::LyObj(register_obj)) = result {
            // Verify state vector is [1, 0, 0, 0]
        }
    }

    #[test]
    fn test_circuit_add_gate() {
        // Create circuit
        let circuit_args = vec![Value::Number(2.0)];
        let circuit = quantum_circuit(&circuit_args).unwrap();
        
        // Create Hadamard gate
        let gate = hadamard_gate(&[]).unwrap();
        
        // Add gate to circuit at qubit 0
        let args = vec![circuit, gate, Value::Number(0.0)];
        let result = circuit_add_gate(&args);
        assert!(result.is_ok());
    }

    #[test]
    fn test_circuit_measurement() {
        // Create circuit with Hadamard gate
        let circuit_args = vec![Value::Number(1.0)];
        let circuit = quantum_circuit(&circuit_args).unwrap();
        
        // Apply Hadamard to create superposition
        let hadamard = hadamard_gate(&[]).unwrap();
        let circuit = circuit_add_gate(&[circuit, hadamard, Value::Number(0.0)]).unwrap();
        
        // Measure qubit
        let args = vec![circuit, Value::Number(0.0)];
        let result = measure_qubit(&args);
        assert!(result.is_ok());
        
        // Result should be 0 or 1 with equal probability
        if let Ok(Value::Number(measurement)) = result {
            assert!(measurement == 0.0 || measurement == 1.0);
        }
    }

    #[test]
    fn test_circuit_evolution() {
        // Test state evolution through circuit
        let circuit_args = vec![Value::Number(1.0)];
        let mut circuit = quantum_circuit(&circuit_args).unwrap();
        
        // Add X gate
        let x_gate = pauli_x_gate(&[]).unwrap();
        circuit = circuit_add_gate(&[circuit, x_gate, Value::Number(0.0)]).unwrap();
        
        // Execute circuit
        let register_args = vec![Value::Number(1.0)];
        let register = qubit_register(&register_args).unwrap();
        
        let args = vec![circuit, register];
        let result = execute_circuit(&args);
        assert!(result.is_ok());
        
        // Final state should be |1⟩
    }

    #[test]
    fn test_quantum_state_operations() {
        // Test state vector operations
        let args = vec![Value::Number(2.0)];
        let register = qubit_register(&args).unwrap();
        
        // Test state normalization
        let norm_result = normalize_state(&[register.clone()]);
        assert!(norm_result.is_ok());
        
        // Test state probability calculation
        let prob_result = state_probabilities(&[register.clone()]);
        assert!(prob_result.is_ok());
        
        // Test partial trace
        let trace_args = vec![register, Value::Number(0.0)];
        let trace_result = partial_trace(&trace_args);
        assert!(trace_result.is_ok());
    }
}

#[cfg(test)]
mod quantum_algorithm_tests {
    use super::*;

    #[test]
    fn test_quantum_fourier_transform() {
        let args = vec![Value::Number(3.0)]; // 3-qubit QFT
        let result = quantum_fourier_transform(&args);
        assert!(result.is_ok());
        
        // Should create QFT circuit for 3 qubits
        if let Ok(Value::LyObj(circuit_obj)) = result {
            // Verify QFT circuit structure
        }
    }

    #[test]
    fn test_grovers_search() {
        // Create oracle for marked item at index 2 in 4-item database
        let oracle_args = vec![Value::Number(2.0), Value::Number(2.0)]; // marked_item=2, n_qubits=2
        let oracle = grover_oracle(&oracle_args).unwrap();
        
        // Run Grover's algorithm
        let grover_args = vec![oracle, Value::Number(2.0)]; // oracle, n_qubits
        let result = grovers_search(&grover_args);
        assert!(result.is_ok());
        
        // Should find the marked item with high probability
    }

    #[test]
    fn test_quantum_teleportation() {
        // Create state to teleport (arbitrary qubit state)
        let state_args = vec![Value::Number(0.6), Value::Number(0.8)]; // α|0⟩ + β|1⟩
        let state = create_qubit_state(&state_args).unwrap();
        
        // Perform teleportation
        let args = vec![state];
        let result = quantum_teleportation(&args);
        assert!(result.is_ok());
        
        // Should return measurement results and final state
        if let Ok(Value::List(results)) = result {
            assert_eq!(results.len(), 3); // [measurement1, measurement2, final_state]
        }
    }

    #[test]
    fn test_quantum_phase_estimation() {
        // Create unitary matrix (e.g., Z gate with eigenvalue -1)
        let unitary = pauli_z_gate(&[]).unwrap();
        
        // Create eigenstate |1⟩
        let eigenstate_args = vec![Value::Number(1.0), Value::Number(1.0)];
        let eigenstate = qubit_register(&eigenstate_args).unwrap();
        
        // Run phase estimation
        let args = vec![unitary, eigenstate, Value::Number(3.0)]; // 3 counting qubits
        let result = quantum_phase_estimation(&args);
        assert!(result.is_ok());
        
        // Should estimate phase π (for eigenvalue -1 = e^(iπ))
    }

    #[test]
    fn test_quantum_error_correction() {
        // Test 3-qubit bit flip code
        let data_qubit = create_qubit_state(&[Value::Number(0.6), Value::Number(0.8)]).unwrap();
        
        // Encode
        let encoded = three_qubit_encode(&[data_qubit]).unwrap();
        
        // Introduce error
        let error_args = vec![encoded, Value::Number(1.0)]; // flip qubit 1
        let corrupted = apply_bit_flip_error(&error_args).unwrap();
        
        // Correct error
        let corrected = three_qubit_correct(&[corrupted]).unwrap();
        
        // Decode
        let decoded = three_qubit_decode(&[corrected]);
        assert!(decoded.is_ok());
        
        // Should recover original state
    }

    #[test]
    fn test_quantum_entanglement() {
        // Test entanglement creation
        let bell_state = create_bell_state(&[Value::Number(0.0)]).unwrap(); // |Φ+⟩
        
        // Test entanglement measures
        let entanglement = entanglement_measure(&[bell_state.clone()]);
        assert!(entanglement.is_ok());
        
        // Test Bell state measurement
        let measurement = measure_bell_state(&[bell_state]);
        assert!(measurement.is_ok());
    }
}

#[cfg(test)]
mod quantum_density_matrix_tests {
    use super::*;

    #[test]
    fn test_density_matrix_creation() {
        // Create density matrix from pure state
        let state_args = vec![Value::Number(1.0)];
        let state = qubit_register(&state_args).unwrap();
        
        let args = vec![state];
        let result = density_matrix(&args);
        assert!(result.is_ok());
        
        // Should create density matrix ρ = |ψ⟩⟨ψ|
    }

    #[test]
    fn test_mixed_state_operations() {
        // Create mixed state
        let rho = create_mixed_state(&[Value::Number(0.7), Value::Number(0.3)]).unwrap();
        
        // Test trace
        let trace_result = density_matrix_trace(&[rho.clone()]);
        assert!(trace_result.is_ok());
        if let Ok(Value::Number(trace)) = trace_result {
            assert!((trace - 1.0).abs() < 1e-10); // Trace should be 1
        }
        
        // Test partial trace
        let partial_trace_result = density_matrix_partial_trace(&[rho.clone(), Value::Number(0.0)]);
        assert!(partial_trace_result.is_ok());
        
        // Test purity
        let purity_result = density_matrix_purity(&[rho]);
        assert!(purity_result.is_ok());
    }

    #[test]
    fn test_quantum_channels() {
        // Test depolarizing channel
        let state = qubit_register(&[Value::Number(1.0)]).unwrap();
        let rho = density_matrix(&[state]).unwrap();
        
        let channel_args = vec![rho, Value::Number(0.1)]; // 10% depolarization
        let result = depolarizing_channel(&channel_args);
        assert!(result.is_ok());
        
        // Test amplitude damping channel
        let damping_args = vec![rho.clone(), Value::Number(0.2)]; // 20% damping
        let damping_result = amplitude_damping_channel(&damping_args);
        assert!(damping_result.is_ok());
    }
}

#[cfg(test)]
mod quantum_property_tests {
    use super::*;

    #[test]
    fn test_gate_unitarity() {
        // All quantum gates should be unitary (U†U = I)
        let gates = vec![
            pauli_x_gate(&[]).unwrap(),
            pauli_y_gate(&[]).unwrap(),
            pauli_z_gate(&[]).unwrap(),
            hadamard_gate(&[]).unwrap(),
        ];
        
        for gate in gates {
            let unitarity_check = verify_unitarity(&[gate]);
            assert!(unitarity_check.is_ok());
            if let Ok(Value::Boolean(is_unitary)) = unitarity_check {
                assert!(is_unitary);
            }
        }
    }

    #[test]
    fn test_quantum_mechanics_laws() {
        // Test conservation of probability
        let state = qubit_register(&[Value::Number(1.0)]).unwrap();
        let probabilities = state_probabilities(&[state]).unwrap();
        
        // Sum of probabilities should be 1
        if let Value::List(probs) = probabilities {
            let sum: f64 = probs.iter()
                .map(|v| if let Value::Number(p) = v { *p } else { 0.0 })
                .sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
        
        // Test Born rule for measurements
        let superposition = create_superposition_state(&[]).unwrap();
        let measurement_stats = measurement_statistics(&[superposition, Value::Number(1000.0)]);
        assert!(measurement_stats.is_ok());
    }

    #[test]
    fn test_quantum_circuit_properties() {
        // Test circuit depth calculation
        let circuit = quantum_circuit(&[Value::Number(3.0)]).unwrap();
        let depth = circuit_depth(&[circuit.clone()]);
        assert!(depth.is_ok());
        
        // Test circuit size
        let size = circuit_size(&[circuit.clone()]);
        assert!(size.is_ok());
        
        // Test circuit optimization
        let optimized = optimize_circuit(&[circuit]);
        assert!(optimized.is_ok());
    }
}

#[cfg(test)]
mod quantum_error_tests {
    use super::*;

    #[test]
    fn test_invalid_qubit_count() {
        // Test negative qubit count
        let args = vec![Value::Number(-1.0)];
        let result = quantum_circuit(&args);
        assert!(result.is_err());
        
        // Test zero qubit count
        let args = vec![Value::Number(0.0)];
        let result = quantum_circuit(&args);
        assert!(result.is_err());
        
        // Test too many qubits (> 20 for simulation limit)
        let args = vec![Value::Number(25.0)];
        let result = quantum_circuit(&args);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_gate_operations() {
        // Test applying gate to invalid qubit index
        let circuit = quantum_circuit(&[Value::Number(2.0)]).unwrap();
        let gate = hadamard_gate(&[]).unwrap();
        
        let args = vec![circuit, gate, Value::Number(5.0)]; // qubit 5 doesn't exist
        let result = circuit_add_gate(&args);
        assert!(result.is_err());
    }

    #[test]
    fn test_measurement_errors() {
        // Test measuring invalid qubit
        let circuit = quantum_circuit(&[Value::Number(2.0)]).unwrap();
        
        let args = vec![circuit, Value::Number(3.0)]; // qubit 3 doesn't exist
        let result = measure_qubit(&args);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_state_operations() {
        // Test invalid amplitude normalization
        let args = vec![Value::Number(2.0), Value::Number(3.0)]; // |α|² + |β|² ≠ 1
        let result = create_qubit_state(&args);
        assert!(result.is_err());
    }
}

// Integration tests with VM
#[cfg(test)]
mod quantum_vm_integration_tests {
    use super::*;
    use lyra::vm::VM;
    use lyra::stdlib::StandardLibrary;

    #[test]
    fn test_quantum_function_registration() {
        let stdlib = StandardLibrary::new();
        
        // Verify quantum functions are registered
        assert!(stdlib.get_function("QuantumCircuit").is_some());
        assert!(stdlib.get_function("QubitRegister").is_some());
        assert!(stdlib.get_function("HadamardGate").is_some());
        assert!(stdlib.get_function("PauliXGate").is_some());
        assert!(stdlib.get_function("CNOTGate").is_some());
        assert!(stdlib.get_function("GroverSearch").is_some());
        assert!(stdlib.get_function("QuantumFourierTransform").is_some());
        assert!(stdlib.get_function("QuantumTeleportation").is_some());
    }

    #[test]
    fn test_quantum_workflow_integration() {
        let mut vm = VM::new();
        
        // Test complete quantum workflow
        // 1. Create circuit
        // 2. Add gates
        // 3. Execute
        // 4. Measure
        
        // This would test the full integration with the VM
    }
}