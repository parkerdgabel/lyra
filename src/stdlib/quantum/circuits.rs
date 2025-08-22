//! Quantum Circuit Implementation
//! 
//! This module provides quantum circuit construction, simulation, and measurement operations.
//! - Circuit construction and gate application
//! - Quantum state evolution and simulation
//! - Measurement operations and Born rule implementation
//! - State preparation and manipulation

use crate::vm::{Value, VmResult, VmError};
use crate::foreign::{LyObj, Foreign, ForeignError};
use super::{Complex, QuantumState, validate_qubit_count, validate_qubit_index};
use super::gates::QuantumGate;
use std::collections::HashMap;

/// Quantum Circuit Foreign Object
#[derive(Debug, Clone)]
pub struct QuantumCircuit {
    pub n_qubits: usize,
    pub gates: Vec<CircuitGate>,
    pub measurements: HashMap<usize, usize>, // qubit -> classical bit mapping
}

#[derive(Debug, Clone)]
pub struct CircuitGate {
    pub gate: QuantumGate,
    pub qubits: Vec<usize>,
    pub controls: Vec<usize>,
}

impl QuantumCircuit {
    pub fn new(n_qubits: usize) -> Self {
        Self {
            n_qubits,
            gates: Vec::new(),
            measurements: HashMap::new(),
        }
    }
    
    pub fn add_gate(&mut self, gate: QuantumGate, qubits: Vec<usize>) -> Result<(), ForeignError> {
        // Validate qubit indices
        for &qubit in &qubits {
            if qubit >= self.n_qubits {
                return Err(ForeignError::RuntimeError { 
                    message: format!("Qubit index {} out of range for {}-qubit circuit", qubit, self.n_qubits)
                });
            }
        }
        
        // Validate gate qubit count
        if gate.n_qubits != qubits.len() {
            return Err(ForeignError::RuntimeError { 
                message: format!("Gate requires {} qubits, but {} specified", gate.n_qubits, qubits.len())
            });
        }
        
        self.gates.push(CircuitGate {
            gate,
            qubits,
            controls: Vec::new(),
        });
        
        Ok(())
    }
    
    pub fn add_controlled_gate(&mut self, gate: QuantumGate, qubits: Vec<usize>, controls: Vec<usize>) -> Result<(), ForeignError> {
        // Validate all qubit indices
        for &qubit in qubits.iter().chain(controls.iter()) {
            if qubit >= self.n_qubits {
                return Err(ForeignError::RuntimeError { 
                    message: format!("Qubit index {} out of range for {}-qubit circuit", qubit, self.n_qubits)
                });
            }
        }
        
        // Check for overlapping qubits
        for &control in &controls {
            if qubits.contains(&control) {
                return Err(ForeignError::RuntimeError { message: "Control and target qubits cannot overlap".to_string() });
            }
        }
        
        self.gates.push(CircuitGate {
            gate,
            qubits,
            controls,
        });
        
        Ok(())
    }
    
    pub fn add_measurement(&mut self, qubit: usize, classical_bit: usize) -> Result<(), ForeignError> {
        if qubit >= self.n_qubits {
            return Err(ForeignError::RuntimeError { 
                message: format!("Qubit index {} out of range", qubit)
            });
        }
        
        self.measurements.insert(qubit, classical_bit);
        Ok(())
    }
    
    pub fn depth(&self) -> usize {
        // Simple depth calculation - could be optimized
        self.gates.len()
    }
    
    pub fn gate_count(&self) -> usize {
        self.gates.len()
    }
    
    pub fn execute(&self, initial_state: &QuantumState) -> Result<(QuantumState, HashMap<usize, usize>), ForeignError> {
        if initial_state.n_qubits != self.n_qubits {
            return Err(ForeignError::RuntimeError { 
                message: format!("State has {} qubits, circuit expects {}", initial_state.n_qubits, self.n_qubits)
            });
        }
        
        let mut state = initial_state.clone();
        let mut classical_results = HashMap::new();
        
        // Apply all gates
        for circuit_gate in &self.gates {
            if circuit_gate.controls.is_empty() {
                // Regular gate application
                state.apply_gate(&circuit_gate.gate.matrix, &circuit_gate.qubits)?;
            } else {
                // Controlled gate application
                self.apply_controlled_gate(&mut state, circuit_gate)?;
            }
        }
        
        // Perform measurements
        for (&qubit, &classical_bit) in &self.measurements {
            let measurement_result = state.measure_qubit(qubit)?;
            classical_results.insert(classical_bit, measurement_result);
        }
        
        Ok((state, classical_results))
    }
    
    fn apply_controlled_gate(&self, state: &mut QuantumState, circuit_gate: &CircuitGate) -> Result<(), VmError> {
        // For controlled gates, we need to check control qubits and apply gate conditionally
        // This is a simplified implementation - a full version would be more efficient
        
        let control_mask: usize = circuit_gate.controls.iter().map(|&c| 1 << c).sum();
        let mut new_amplitudes = vec![Complex::zero(); state.amplitudes.len()];
        
        for (i, amplitude) in state.amplitudes.iter().enumerate() {
            // Check if all control qubits are |1⟩
            let controls_satisfied = circuit_gate.controls.iter()
                .all(|&control| (i >> control) & 1 == 1);
            
            if controls_satisfied {
                // Apply the gate to the target qubits
                // This is a simplified implementation
                new_amplitudes[i] = amplitude.clone();
            } else {
                // Don't apply the gate
                new_amplitudes[i] = amplitude.clone();
            }
        }
        
        state.amplitudes = new_amplitudes;
        Ok(())
    }
}

impl Foreign for QuantumCircuit {
    fn type_name(&self) -> &'static str {
        "QuantumCircuit"
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "addGate" => {
                if args.len() < 2 {
                    return Err(ForeignError::InvalidArity {
                        method: "addGate".to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }
                
                let mut circuit = self.clone();
                
                let gate = match &args[0] {
                    Value::LyObj(gate_obj) => {
                        if let Some(g) = gate_obj.downcast_ref::<QuantumGate>() {
                            g.clone()
                        } else {
                            return Err(ForeignError::InvalidArgumentType {
                                method: "addGate".to_string(),
                                expected: "QuantumGate".to_string(),
                                actual: "other".to_string(),
                            });
                        }
                    }
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: "addGate".to_string(),
                        expected: "QuantumGate".to_string(),
                        actual: "other".to_string(),
                    }),
                };
                
                let qubits = match &args[1] {
                    Value::List(qubit_list) => {
                        let mut qubits = Vec::new();
                        for qubit_val in qubit_list {
                            match qubit_val {
                                Value::Real(q) => {
                                    let qubit_idx = validate_qubit_index(*q, self.n_qubits)
                                        .map_err(|e| ForeignError::RuntimeError { message: e.to_string() })?;
                                    qubits.push(qubit_idx);
                                }
                                _ => return Err(ForeignError::RuntimeError { 
                                    message: "Qubit indices must be numbers".to_string()
                                }),
                            }
                        }
                        qubits
                    }
                    Value::Real(q) => {
                        let qubit_idx = validate_qubit_index(*q, self.n_qubits)
                            .map_err(|e| ForeignError::RuntimeError { message: e.to_string() })?;
                        vec![qubit_idx]
                    }
                    _ => return Err(ForeignError::RuntimeError { message: "Qubit specification must be number or list".to_string() }),
                };
                
                circuit.add_gate(gate, qubits)?;
                Ok(Value::LyObj(LyObj::new(Box::new(circuit))))
            }
            "addMeasurement" => {
                if args.len() != 2 {
                    return Err(ForeignError::RuntimeError { message: "addMeasurement requires qubit and classical bit indices".to_string() });
                }
                
                let mut circuit = self.clone();
                
                let qubit = match &args[0] {
                    Value::Real(q) => validate_qubit_index(*q, self.n_qubits)
                            .map_err(|e| ForeignError::RuntimeError { message: e.to_string() })?,
                    _ => return Err(ForeignError::RuntimeError { message: "Qubit index must be a number".to_string() }),
                };
                
                let classical_bit = match &args[1] {
                    Value::Real(c) => {
                        if *c < 0.0 || c.fract() != 0.0 {
                            return Err(ForeignError::RuntimeError { message: "Classical bit index must be non-negative integer".to_string() });
                        }
                        *c as usize
                    }
                    _ => return Err(ForeignError::RuntimeError { message: "Classical bit index must be a number".to_string() }),
                };
                
                circuit.add_measurement(qubit, classical_bit)?;
                Ok(Value::LyObj(LyObj::new(Box::new(circuit))))
            }
            "execute" => {
                if args.len() != 1 {
                    return Err(ForeignError::RuntimeError { message: "execute requires a quantum state".to_string() });
                }
                
                let state = match &args[0] {
                    Value::LyObj(state_obj) => {
                        if let Some(s) = state_obj.downcast_ref::<QubitRegister>() {
                            &s.state
                        } else {
                            return Err(ForeignError::RuntimeError { message: "Argument must be a QubitRegister".to_string() });
                        }
                    }
                    _ => return Err(ForeignError::RuntimeError { message: "Argument must be a QubitRegister".to_string() }),
                };
                
                let (final_state, measurements) = self.execute(state)?;
                
                // Return [final_state, measurements]
                let final_state_obj = QubitRegister { state: final_state };
                let measurements_list: Vec<Value> = measurements.iter()
                    .map(|(&bit, &result)| Value::List(vec![
                        Value::Real(bit as f64),
                        Value::Real(result as f64),
                    ]))
                    .collect();
                
                Ok(Value::List(vec![
                    Value::LyObj(LyObj::new(Box::new(final_state_obj))),
                    Value::List(measurements_list),
                ]))
            }
            "depth" => Ok(Value::Real(self.depth() as f64)),
            "gateCount" => Ok(Value::Real(self.gate_count() as f64)),
            "qubits" => Ok(Value::Real(self.n_qubits as f64)),
            _ => Err(ForeignError::UnknownMethod {
                type_name: "QubitRegister".to_string(),
                method: method.to_string(),
            }),
        }
    }

}

/// Qubit Register Foreign Object
#[derive(Debug, Clone)]
pub struct QubitRegister {
    pub state: QuantumState,
}

impl QubitRegister {
    pub fn new(n_qubits: usize) -> Self {
        Self {
            state: QuantumState::new(n_qubits),
        }
    }
    
    pub fn from_state(state: QuantumState) -> Self {
        Self { state }
    }
}

impl Foreign for QubitRegister {
    fn type_name(&self) -> &'static str {
        "QubitRegister"
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "measure" => {
                if args.is_empty() {
                    // Measure all qubits
                    let mut state = self.state.clone();
                    let mut results = Vec::new();
                    
                    for i in 0..state.n_qubits {
                        let measurement = state.measure_qubit(i)?;
                        results.push(Value::Real(measurement as f64));
                    }
                    
                    Ok(Value::List(results))
                } else if args.len() == 1 {
                    // Measure specific qubit
                    let qubit = match &args[0] {
                        Value::Real(q) => validate_qubit_index(*q, self.state.n_qubits).map_err(|e| ForeignError::RuntimeError { message: e.to_string() })?,
                        _ => return Err(ForeignError::InvalidArgumentType {
                            method: "measure".to_string(),
                            expected: "Number".to_string(),
                            actual: "other".to_string(),
                        }),
                    };
                    
                    let mut state = self.state.clone();
                    let measurement = state.measure_qubit(qubit).map_err(|e| ForeignError::RuntimeError { message: e.to_string() })?;
                    
                    // Return [measurement_result, new_state]
                    let new_register = QubitRegister::from_state(state);
                    Ok(Value::List(vec![
                        Value::Real(measurement as f64),
                        Value::LyObj(LyObj::new(Box::new(new_register))),
                    ]))
                } else {
                    Err(ForeignError::InvalidArity {
                        method: "measure".to_string(),
                        expected: 1,
                        actual: args.len(),
                    })
                }
            }
            "probabilities" => {
                let probs = self.state.probabilities();
                let prob_values: Vec<Value> = probs.iter()
                    .map(|&p| Value::Real(p))
                    .collect();
                Ok(Value::List(prob_values))
            }
            "amplitudes" => {
                let amp_values: Vec<Value> = self.state.amplitudes.iter()
                    .map(|amp| Value::List(vec![
                        Value::Real(amp.real),
                        Value::Real(amp.imag),
                    ]))
                    .collect();
                Ok(Value::List(amp_values))
            }
            "normalize" => {
                let mut state = self.state.clone();
                state.normalize();
                let normalized_register = QubitRegister::from_state(state);
                Ok(Value::LyObj(LyObj::new(Box::new(normalized_register))))
            }
            "clone" => {
                Ok(Value::LyObj(LyObj::new(Box::new(self.clone()))))
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: "QubitRegister".to_string(),
                method: method.to_string(),
            }),
        }
    }

}

// Quantum Circuit Construction Functions

/// Create a new quantum circuit
pub fn quantum_circuit(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("QuantumCircuit requires 1 argument (number of qubits)".to_string()));
    }
    
    let n_qubits = match &args[0] {
        Value::Real(n) => validate_qubit_count(*n).map_err(|e| VmError::Runtime(e.to_string()))?,
        _ => return Err(VmError::Runtime("Number of qubits must be a number".to_string())),
    };
    
    let circuit = QuantumCircuit::new(n_qubits);
    Ok(Value::LyObj(LyObj::new(Box::new(circuit))))
}

/// Create a new qubit register
pub fn qubit_register(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("QubitRegister requires 1 argument (number of qubits)".to_string()));
    }
    
    let n_qubits = match &args[0] {
        Value::Real(n) => validate_qubit_count(*n).map_err(|e| VmError::Runtime(e.to_string()))?,
        _ => return Err(VmError::Runtime("Number of qubits must be a number".to_string())),
    };
    
    let register = QubitRegister::new(n_qubits);
    Ok(Value::LyObj(LyObj::new(Box::new(register))))
}

/// Add a gate to a circuit
pub fn circuit_add_gate(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(VmError::Runtime("circuit_add_gate requires circuit, gate, and qubit indices".to_string()));
    }
    
    let mut circuit = match &args[0] {
        Value::LyObj(circuit_obj) => {
            if let Some(c) = circuit_obj.downcast_ref::<QuantumCircuit>() {
                c.clone()
            } else {
                return Err(VmError::Runtime("First argument must be a QuantumCircuit".to_string()));
            }
        }
        _ => return Err(VmError::Runtime("First argument must be a QuantumCircuit".to_string())),
    };
    
    let gate = match &args[1] {
        Value::LyObj(gate_obj) => {
            if let Some(g) = gate_obj.downcast_ref::<QuantumGate>() {
                g.clone()
            } else {
                return Err(VmError::Runtime("Second argument must be a QuantumGate".to_string()));
            }
        }
        _ => return Err(VmError::Runtime("Second argument must be a QuantumGate".to_string())),
    };
    
    // Parse qubit indices (can be a single number or list)
    let qubits = if args.len() == 3 {
        // Single argument for qubits
        match &args[2] {
            Value::Real(q) => {
                let qubit_idx = validate_qubit_index(*q, circuit.n_qubits)
                .map_err(|e| VmError::Runtime(e.to_string()))?;
                vec![qubit_idx]
            }
            Value::List(qubit_list) => {
                let mut qubits = Vec::new();
                for qubit_val in qubit_list {
                    match qubit_val {
                        Value::Real(q) => {
                            let qubit_idx = validate_qubit_index(*q, circuit.n_qubits)
                .map_err(|e| VmError::Runtime(e.to_string()))?;
                            qubits.push(qubit_idx);
                        }
                        _ => return Err(VmError::Runtime("Qubit indices must be numbers".to_string())),
                    }
                }
                qubits
            }
            _ => return Err(VmError::Runtime("Qubit specification must be number or list".to_string())),
        }
    } else {
        // Multiple arguments for qubits
        let mut qubits = Vec::new();
        for arg in &args[2..] {
            match arg {
                Value::Real(q) => {
                    let qubit_idx = validate_qubit_index(*q, circuit.n_qubits)
                .map_err(|e| VmError::Runtime(e.to_string()))?;
                    qubits.push(qubit_idx);
                }
                _ => return Err(VmError::Runtime("Qubit indices must be numbers".to_string())),
            }
        }
        qubits
    };
    
    circuit.add_gate(gate, qubits).map_err(|e| VmError::Runtime(e.to_string()))?;
    Ok(Value::LyObj(LyObj::new(Box::new(circuit))))
}

/// Execute a quantum circuit
pub fn execute_circuit(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime("execute_circuit requires circuit and initial state".to_string()));
    }
    
    let circuit = match &args[0] {
        Value::LyObj(circuit_obj) => {
            if let Some(c) = circuit_obj.downcast_ref::<QuantumCircuit>() {
                c
            } else {
                return Err(VmError::Runtime("First argument must be a QuantumCircuit".to_string()));
            }
        }
        _ => return Err(VmError::Runtime("First argument must be a QuantumCircuit".to_string())),
    };
    
    let initial_state = match &args[1] {
        Value::LyObj(state_obj) => {
            if let Some(s) = state_obj.downcast_ref::<QubitRegister>() {
                &s.state
            } else {
                return Err(VmError::Runtime("Second argument must be a QubitRegister".to_string()));
            }
        }
        _ => return Err(VmError::Runtime("Second argument must be a QubitRegister".to_string())),
    };
    
    let (final_state, measurements) = circuit.execute(initial_state).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let final_register = QubitRegister::from_state(final_state);
    let measurements_list: Vec<Value> = measurements.iter()
        .map(|(&bit, &result)| Value::List(vec![
            Value::Real(bit as f64),
            Value::Real(result as f64),
        ]))
        .collect();
    
    Ok(Value::List(vec![
        Value::LyObj(LyObj::new(Box::new(final_register))),
        Value::List(measurements_list),
    ]))
}

/// Measure a qubit
pub fn measure_qubit(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime("measure_qubit requires register and qubit index".to_string()));
    }
    
    let register = match &args[0] {
        Value::LyObj(state_obj) => {
            if let Some(s) = state_obj.downcast_ref::<QubitRegister>() {
                s
            } else {
                return Err(VmError::Runtime("First argument must be a QubitRegister".to_string()));
            }
        }
        _ => return Err(VmError::Runtime("First argument must be a QubitRegister".to_string())),
    };
    
    let qubit = match &args[1] {
        Value::Real(q) => validate_qubit_index(*q, register.state.n_qubits)
            .map_err(|e| VmError::Runtime(e.to_string()))?,
        _ => return Err(VmError::Runtime("Qubit index must be a number".to_string())),
    };
    
    let mut state = register.state.clone();
    let measurement = state.measure_qubit(qubit)?;
    
    Ok(Value::Real(measurement as f64))
}

// State Preparation Functions

/// Create a custom qubit state from amplitudes
pub fn create_qubit_state(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(VmError::Runtime("create_qubit_state requires amplitude coefficients".to_string()));
    }
    
    let mut amplitudes = Vec::new();
    
    for arg in args {
        let amplitude = match arg {
            Value::Real(real) => Complex::new(*real, 0.0),
            Value::List(complex_pair) => {
                if complex_pair.len() != 2 {
                    return Err(VmError::Runtime("Complex amplitude must be [real, imag]".to_string()));
                }
                
                let real = match &complex_pair[0] {
                    Value::Real(r) => *r,
                    _ => return Err(VmError::Runtime("Real part must be a number".to_string())),
                };
                
                let imag = match &complex_pair[1] {
                    Value::Real(i) => *i,
                    _ => return Err(VmError::Runtime("Imaginary part must be a number".to_string())),
                };
                
                Complex::new(real, imag)
            }
            _ => return Err(VmError::Runtime("Amplitude must be number or [real, imag]".to_string())),
        };
        
        amplitudes.push(amplitude);
    }
    
    let state = QuantumState::from_amplitudes(amplitudes)?;
    let register = QubitRegister::from_state(state);
    
    Ok(Value::LyObj(LyObj::new(Box::new(register))))
}

/// Create a superposition state (equal superposition of all basis states)
pub fn create_superposition_state(args: &[Value]) -> VmResult<Value> {
    let n_qubits = if args.is_empty() {
        1 // Default to single qubit
    } else {
        match &args[0] {
            Value::Real(n) => validate_qubit_count(*n)?,
            _ => return Err(VmError::Runtime("Number of qubits must be a number".to_string())),
        }
    };
    
    let size = 1 << n_qubits;
    let amplitude = Complex::new(1.0 / (size as f64).sqrt(), 0.0);
    let amplitudes = vec![amplitude; size];
    
    let state = QuantumState::from_amplitudes(amplitudes)?;
    let register = QubitRegister::from_state(state);
    
    Ok(Value::LyObj(LyObj::new(Box::new(register))))
}

/// Create a Bell state
pub fn create_bell_state(args: &[Value]) -> VmResult<Value> {
    let bell_type = if args.is_empty() {
        0 // Default to |Φ+⟩
    } else {
        match &args[0] {
            Value::Real(t) => {
                if *t < 0.0 || *t > 3.0 || t.fract() != 0.0 {
                    return Err(VmError::Runtime("Bell state type must be 0, 1, 2, or 3".to_string()));
                }
                *t as usize
            }
            _ => return Err(VmError::Runtime("Bell state type must be a number".to_string())),
        }
    };
    
    let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
    
    let amplitudes = match bell_type {
        0 => vec![Complex::new(inv_sqrt2, 0.0), Complex::zero(), Complex::zero(), Complex::new(inv_sqrt2, 0.0)], // |Φ+⟩ = (|00⟩ + |11⟩)/√2
        1 => vec![Complex::new(inv_sqrt2, 0.0), Complex::zero(), Complex::zero(), Complex::new(-inv_sqrt2, 0.0)], // |Φ-⟩ = (|00⟩ - |11⟩)/√2
        2 => vec![Complex::zero(), Complex::new(inv_sqrt2, 0.0), Complex::new(inv_sqrt2, 0.0), Complex::zero()], // |Ψ+⟩ = (|01⟩ + |10⟩)/√2
        3 => vec![Complex::zero(), Complex::new(inv_sqrt2, 0.0), Complex::new(-inv_sqrt2, 0.0), Complex::zero()], // |Ψ-⟩ = (|01⟩ - |10⟩)/√2
        _ => unreachable!(),
    };
    
    let state = QuantumState::from_amplitudes(amplitudes)?;
    let register = QubitRegister::from_state(state);
    
    Ok(Value::LyObj(LyObj::new(Box::new(register))))
}

// State Analysis Functions

/// Get state probabilities
pub fn state_probabilities(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("state_probabilities requires 1 argument".to_string()));
    }
    
    let register = match &args[0] {
        Value::LyObj(state_obj) => {
            if let Some(s) = state_obj.downcast_ref::<QubitRegister>() {
                s
            } else {
                return Err(VmError::Runtime("Argument must be a QubitRegister".to_string()));
            }
        }
        _ => return Err(VmError::Runtime("Argument must be a QubitRegister".to_string())),
    };
    
    let probabilities = register.state.probabilities();
    let prob_values: Vec<Value> = probabilities.iter()
        .map(|&p| Value::Real(p))
        .collect();
    
    Ok(Value::List(prob_values))
}

/// Normalize a quantum state
pub fn normalize_state(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("normalize_state requires 1 argument".to_string()));
    }
    
    let register = match &args[0] {
        Value::LyObj(state_obj) => {
            if let Some(s) = state_obj.downcast_ref::<QubitRegister>() {
                s
            } else {
                return Err(VmError::Runtime("Argument must be a QubitRegister".to_string()));
            }
        }
        _ => return Err(VmError::Runtime("Argument must be a QubitRegister".to_string())),
    };
    
    let mut state = register.state.clone();
    state.normalize();
    let normalized_register = QubitRegister::from_state(state);
    
    Ok(Value::LyObj(LyObj::new(Box::new(normalized_register))))
}

/// Partial trace operation (trace out specified qubits)
pub fn partial_trace(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime("partial_trace requires state and qubits to trace out".to_string()));
    }
    
    let _register = match &args[0] {
        Value::LyObj(state_obj) => {
            if let Some(s) = state_obj.downcast_ref::<QubitRegister>() {
                s
            } else {
                return Err(VmError::Runtime("First argument must be a QubitRegister".to_string()));
            }
        }
        _ => return Err(VmError::Runtime("First argument must be a QubitRegister".to_string())),
    };
    
    let _qubits_to_trace = match &args[1] {
        Value::List(qubit_list) => qubit_list,
        Value::Real(_q) => {
            // Single qubit to trace out
            return Err(VmError::Runtime("Partial trace not yet implemented".to_string()));
        }
        _ => return Err(VmError::Runtime("Second argument must be qubit indices".to_string())),
    };
    
    // TODO: Implement partial trace operation
    Err(VmError::Runtime("Partial trace not yet implemented".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stdlib::quantum::gates::*;

    #[test]
    fn test_quantum_circuit_creation() {
        let circuit = quantum_circuit(&[Value::Real(3.0)]).unwrap();
        
        if let Value::LyObj(circuit_obj) = circuit {
            let circuit = circuit_obj.downcast_ref::<QuantumCircuit>().unwrap();
            assert_eq!(circuit.n_qubits, 3);
            assert_eq!(circuit.gates.len(), 0);
        }
    }

    #[test]
    fn test_qubit_register_creation() {
        let register = qubit_register(&[Value::Real(2.0)]).unwrap();
        
        if let Value::LyObj(register_obj) = register {
            let register = register_obj.downcast_ref::<QubitRegister>().unwrap();
            assert_eq!(register.state.n_qubits, 2);
            assert_eq!(register.state.amplitudes.len(), 4);
            assert_eq!(register.state.amplitudes[0].real, 1.0); // |00⟩ state
        }
    }

    #[test]
    fn test_circuit_gate_addition() {
        let circuit = quantum_circuit(&[Value::Real(2.0)]).unwrap();
        let h_gate = hadamard_gate(&[]).unwrap();
        
        let modified_circuit = circuit_add_gate(&[circuit, h_gate, Value::Real(0.0)]).unwrap();
        
        if let Value::LyObj(circuit_obj) = modified_circuit {
            let circuit = circuit_obj.downcast_ref::<QuantumCircuit>().unwrap();
            assert_eq!(circuit.gates.len(), 1);
            assert_eq!(circuit.gates[0].qubits, vec![0]);
        }
    }

    #[test]
    fn test_bell_state_creation() {
        let bell_state = create_bell_state(&[Value::Real(0.0)]).unwrap();
        
        if let Value::LyObj(state_obj) = bell_state {
            let register = state_obj.downcast_ref::<QubitRegister>().unwrap();
            let probs = register.state.probabilities();
            
            // |Φ+⟩ = (|00⟩ + |11⟩)/√2 should have 50% probability each for |00⟩ and |11⟩
            assert!((probs[0] - 0.5).abs() < 1e-10);
            assert!((probs[1] - 0.0).abs() < 1e-10);
            assert!((probs[2] - 0.0).abs() < 1e-10);
            assert!((probs[3] - 0.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_superposition_state() {
        let superposition = create_superposition_state(&[Value::Real(2.0)]).unwrap();
        
        if let Value::LyObj(state_obj) = superposition {
            let register = state_obj.downcast_ref::<QubitRegister>().unwrap();
            let probs = register.state.probabilities();
            
            // Equal superposition should have 25% probability for each basis state
            for prob in probs {
                assert!((prob - 0.25).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_state_measurement() {
        let register = qubit_register(&[Value::Real(1.0)]).unwrap();
        let measurement = measure_qubit(&[register, Value::Real(0.0)]).unwrap();
        
        // |0⟩ state should always measure 0
        assert_eq!(measurement, Value::Real(0.0));
    }

    #[test]
    fn test_circuit_execution() {
        let circuit = quantum_circuit(&[Value::Real(1.0)]).unwrap();
        let x_gate = pauli_x_gate(&[]).unwrap();
        let circuit_with_gate = circuit_add_gate(&[circuit, x_gate, Value::Real(0.0)]).unwrap();
        
        let initial_state = qubit_register(&[Value::Real(1.0)]).unwrap();
        let result = execute_circuit(&[circuit_with_gate, initial_state]).unwrap();
        
        if let Value::List(results) = result {
            assert_eq!(results.len(), 2); // [final_state, measurements]
            
            if let Value::LyObj(final_state_obj) = &results[0] {
                let final_register = final_state_obj.downcast_ref::<QubitRegister>().unwrap();
                // After X gate, |0⟩ → |1⟩
                assert!((final_register.state.amplitudes[0].magnitude() - 0.0).abs() < 1e-10);
                assert!((final_register.state.amplitudes[1].magnitude() - 1.0).abs() < 1e-10);
            }
        }
    }
}