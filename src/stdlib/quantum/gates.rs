//! Quantum Gate Implementations
//! 
//! This module provides implementations of all quantum gates:
//! - Single-qubit gates: Pauli gates (X, Y, Z), Hadamard, Phase, Rotation gates
//! - Multi-qubit gates: CNOT, Toffoli, Fredkin, controlled gates
//! 
//! All gates are implemented as Foreign objects and return quantum matrices.

use crate::vm::{Value, VmResult, VmError};
use crate::foreign::{LyObj, Foreign, ForeignError};
use super::{QuantumMatrix, Complex};
use std::f64::consts::{PI, SQRT_2};

/// Quantum Gate Foreign Object
#[derive(Debug, Clone)]
pub struct QuantumGate {
    pub matrix: QuantumMatrix,
    pub name: String,
    pub n_qubits: usize,
}

impl QuantumGate {
    pub fn new(matrix: QuantumMatrix, name: String) -> Self {
        let n_qubits = (matrix.rows as f64).log2() as usize;
        Self { matrix, name, n_qubits }
    }
}

impl Foreign for QuantumGate {
    fn type_name(&self) -> &'static str {
        "QuantumGate"
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "matrix" => Ok(self.matrix_to_value()),
            "name" => Ok(Value::String(self.name.clone())),
            "qubits" => Ok(Value::Integer(self.n_qubits as i64)),
            "isUnitary" => Ok(Value::Boolean(self.matrix.is_unitary(1e-10))),
            "dagger" => {
                let dagger_matrix = self.matrix.dagger();
                let dagger_gate = QuantumGate::new(dagger_matrix, format!("{}†", self.name));
                Ok(Value::LyObj(LyObj::new(Box::new(dagger_gate))))
            }
            "tensorProduct" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: "tensorProduct".to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                if let Value::LyObj(other_obj) = &args[0] {
                    if let Some(other_gate) = other_obj.downcast_ref::<QuantumGate>() {
                        let product_matrix = self.matrix.tensor_product(&other_gate.matrix);
                        let product_gate = QuantumGate::new(
                            product_matrix, 
                            format!("{} ⊗ {}", self.name, other_gate.name)
                        );
                        return Ok(Value::LyObj(LyObj::new(Box::new(product_gate))));
                    }
                }
                Err(ForeignError::InvalidArgumentType {
                    method: "tensorProduct".to_string(),
                    expected: "QuantumGate".to_string(),
                    actual: "other".to_string(),
                })
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: "QuantumGate".to_string(),
                method: method.to_string(),
            }),
        }
    }

}

impl QuantumGate {
    fn matrix_to_value(&self) -> Value {
        let mut rows = Vec::new();
        
        for row in &self.matrix.data {
            let mut row_values = Vec::new();
            for complex in row {
                // Represent complex number as [real, imaginary]
                row_values.push(Value::List(vec![
                    Value::Real(complex.real),
                    Value::Real(complex.imag),
                ]));
            }
            rows.push(Value::List(row_values));
        }
        
        Value::List(rows)
    }
}

// Single-Qubit Gates

/// Pauli-X Gate (NOT Gate)
/// Matrix: [[0, 1], [1, 0]]
pub fn pauli_x_gate(_args: &[Value]) -> VmResult<Value> {
    let matrix = QuantumMatrix::from_data(vec![
        vec![Complex::zero(), Complex::one()],
        vec![Complex::one(), Complex::zero()],
    ])?;
    
    let gate = QuantumGate::new(matrix, "X".to_string());
    Ok(Value::LyObj(LyObj::new(Box::new(gate))))
}

/// Pauli-Y Gate
/// Matrix: [[0, -i], [i, 0]]
pub fn pauli_y_gate(_args: &[Value]) -> VmResult<Value> {
    let matrix = QuantumMatrix::from_data(vec![
        vec![Complex::zero(), Complex::new(0.0, -1.0)],
        vec![Complex::new(0.0, 1.0), Complex::zero()],
    ])?;
    
    let gate = QuantumGate::new(matrix, "Y".to_string());
    Ok(Value::LyObj(LyObj::new(Box::new(gate))))
}

/// Pauli-Z Gate
/// Matrix: [[1, 0], [0, -1]]
pub fn pauli_z_gate(_args: &[Value]) -> VmResult<Value> {
    let matrix = QuantumMatrix::from_data(vec![
        vec![Complex::one(), Complex::zero()],
        vec![Complex::zero(), Complex::new(-1.0, 0.0)],
    ])?;
    
    let gate = QuantumGate::new(matrix, "Z".to_string());
    Ok(Value::LyObj(LyObj::new(Box::new(gate))))
}

/// Hadamard Gate
/// Matrix: (1/√2) * [[1, 1], [1, -1]]
pub fn hadamard_gate(_args: &[Value]) -> VmResult<Value> {
    let inv_sqrt2 = 1.0 / SQRT_2;
    let matrix = QuantumMatrix::from_data(vec![
        vec![Complex::new(inv_sqrt2, 0.0), Complex::new(inv_sqrt2, 0.0)],
        vec![Complex::new(inv_sqrt2, 0.0), Complex::new(-inv_sqrt2, 0.0)],
    ])?;
    
    let gate = QuantumGate::new(matrix, "H".to_string());
    Ok(Value::LyObj(LyObj::new(Box::new(gate))))
}

/// Phase Gate (S Gate)
/// Matrix: [[1, 0], [0, i]]
pub fn phase_gate(args: &[Value]) -> VmResult<Value> {
    let phase = if args.is_empty() {
        PI / 2.0 // Default S gate
    } else {
        match &args[0] {
            Value::Real(p) => *p,
            _ => return Err(VmError::Runtime("Phase must be a number".to_string())),
        }
    };
    
    let matrix = QuantumMatrix::from_data(vec![
        vec![Complex::one(), Complex::zero()],
        vec![Complex::zero(), Complex::new(phase.cos(), phase.sin())],
    ])?;
    
    let name = if (phase - PI / 2.0).abs() < 1e-10 {
        "S".to_string()
    } else if (phase - PI / 4.0).abs() < 1e-10 {
        "T".to_string()
    } else {
        format!("P({:.3})", phase)
    };
    
    let gate = QuantumGate::new(matrix, name);
    Ok(Value::LyObj(LyObj::new(Box::new(gate))))
}

/// T Gate (π/8 Gate)
/// Matrix: [[1, 0], [0, e^(iπ/4)]]
pub fn t_gate(_args: &[Value]) -> VmResult<Value> {
    phase_gate(&[Value::Real(PI / 4.0)])
}

/// Rotation-X Gate
/// Matrix: [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]]
pub fn rotation_x_gate(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("RX gate requires 1 angle parameter".to_string()));
    }
    
    let angle = match &args[0] {
        Value::Real(a) => *a,
        _ => return Err(VmError::Runtime("Angle must be a number".to_string())),
    };
    
    let cos_half = (angle / 2.0).cos();
    let sin_half = (angle / 2.0).sin();
    
    let matrix = QuantumMatrix::from_data(vec![
        vec![Complex::new(cos_half, 0.0), Complex::new(0.0, -sin_half)],
        vec![Complex::new(0.0, -sin_half), Complex::new(cos_half, 0.0)],
    ])?;
    
    let gate = QuantumGate::new(matrix, format!("RX({:.3})", angle));
    Ok(Value::LyObj(LyObj::new(Box::new(gate))))
}

/// Rotation-Y Gate
/// Matrix: [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
pub fn rotation_y_gate(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("RY gate requires 1 angle parameter".to_string()));
    }
    
    let angle = match &args[0] {
        Value::Real(a) => *a,
        _ => return Err(VmError::Runtime("Angle must be a number".to_string())),
    };
    
    let cos_half = (angle / 2.0).cos();
    let sin_half = (angle / 2.0).sin();
    
    let matrix = QuantumMatrix::from_data(vec![
        vec![Complex::new(cos_half, 0.0), Complex::new(-sin_half, 0.0)],
        vec![Complex::new(sin_half, 0.0), Complex::new(cos_half, 0.0)],
    ])?;
    
    let gate = QuantumGate::new(matrix, format!("RY({:.3})", angle));
    Ok(Value::LyObj(LyObj::new(Box::new(gate))))
}

/// Rotation-Z Gate
/// Matrix: [[e^(-iθ/2), 0], [0, e^(iθ/2)]]
pub fn rotation_z_gate(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("RZ gate requires 1 angle parameter".to_string()));
    }
    
    let angle = match &args[0] {
        Value::Real(a) => *a,
        _ => return Err(VmError::Runtime("Angle must be a number".to_string())),
    };
    
    let half_angle = angle / 2.0;
    
    let matrix = QuantumMatrix::from_data(vec![
        vec![Complex::new(half_angle.cos(), -half_angle.sin()), Complex::zero()],
        vec![Complex::zero(), Complex::new(half_angle.cos(), half_angle.sin())],
    ])?;
    
    let gate = QuantumGate::new(matrix, format!("RZ({:.3})", angle));
    Ok(Value::LyObj(LyObj::new(Box::new(gate))))
}

// Multi-Qubit Gates

/// CNOT Gate (Controlled-X)
/// 4x4 Matrix: [[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]]
pub fn cnot_gate(_args: &[Value]) -> VmResult<Value> {
    let matrix = QuantumMatrix::from_data(vec![
        vec![Complex::one(), Complex::zero(), Complex::zero(), Complex::zero()],
        vec![Complex::zero(), Complex::one(), Complex::zero(), Complex::zero()],
        vec![Complex::zero(), Complex::zero(), Complex::zero(), Complex::one()],
        vec![Complex::zero(), Complex::zero(), Complex::one(), Complex::zero()],
    ])?;
    
    let gate = QuantumGate::new(matrix, "CNOT".to_string());
    Ok(Value::LyObj(LyObj::new(Box::new(gate))))
}

/// CZ Gate (Controlled-Z)
/// 4x4 Matrix: [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,-1]]
pub fn cz_gate(_args: &[Value]) -> VmResult<Value> {
    let matrix = QuantumMatrix::from_data(vec![
        vec![Complex::one(), Complex::zero(), Complex::zero(), Complex::zero()],
        vec![Complex::zero(), Complex::one(), Complex::zero(), Complex::zero()],
        vec![Complex::zero(), Complex::zero(), Complex::one(), Complex::zero()],
        vec![Complex::zero(), Complex::zero(), Complex::zero(), Complex::new(-1.0, 0.0)],
    ])?;
    
    let gate = QuantumGate::new(matrix, "CZ".to_string());
    Ok(Value::LyObj(LyObj::new(Box::new(gate))))
}

/// SWAP Gate
/// 4x4 Matrix: [[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]]
pub fn swap_gate(_args: &[Value]) -> VmResult<Value> {
    let matrix = QuantumMatrix::from_data(vec![
        vec![Complex::one(), Complex::zero(), Complex::zero(), Complex::zero()],
        vec![Complex::zero(), Complex::zero(), Complex::one(), Complex::zero()],
        vec![Complex::zero(), Complex::one(), Complex::zero(), Complex::zero()],
        vec![Complex::zero(), Complex::zero(), Complex::zero(), Complex::one()],
    ])?;
    
    let gate = QuantumGate::new(matrix, "SWAP".to_string());
    Ok(Value::LyObj(LyObj::new(Box::new(gate))))
}

/// Toffoli Gate (Controlled-Controlled-X, CCX)
/// 8x8 Matrix with only the last two rows swapped
pub fn toffoli_gate(_args: &[Value]) -> VmResult<Value> {
    let mut matrix = QuantumMatrix::identity(8);
    
    // Swap the last two diagonal elements (|110⟩ ↔ |111⟩)
    matrix.data[6][6] = Complex::zero();
    matrix.data[7][7] = Complex::zero();
    matrix.data[6][7] = Complex::one();
    matrix.data[7][6] = Complex::one();
    
    let gate = QuantumGate::new(matrix, "Toffoli".to_string());
    Ok(Value::LyObj(LyObj::new(Box::new(gate))))
}

/// Fredkin Gate (Controlled-SWAP, CSWAP)
/// 8x8 Matrix with swapped rows for controlled swap
pub fn fredkin_gate(_args: &[Value]) -> VmResult<Value> {
    let mut matrix = QuantumMatrix::identity(8);
    
    // Swap |101⟩ ↔ |110⟩ when control is |1⟩
    matrix.data[5][5] = Complex::zero();
    matrix.data[6][6] = Complex::zero();
    matrix.data[5][6] = Complex::one();
    matrix.data[6][5] = Complex::one();
    
    let gate = QuantumGate::new(matrix, "Fredkin".to_string());
    Ok(Value::LyObj(LyObj::new(Box::new(gate))))
}

/// Controlled Gate Constructor
/// Creates a controlled version of any single-qubit gate
pub fn controlled_gate(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("Controlled gate requires 1 gate argument".to_string()));
    }
    
    let base_gate = match &args[0] {
        Value::LyObj(gate_obj) => {
            if let Some(gate) = gate_obj.downcast_ref::<QuantumGate>() {
                gate
            } else {
                return Err(VmError::Runtime("Argument must be a QuantumGate".to_string()));
            }
        }
        _ => return Err(VmError::Runtime("Argument must be a QuantumGate".to_string())),
    };
    
    if base_gate.n_qubits != 1 {
        return Err(VmError::Runtime("Can only create controlled version of single-qubit gates".to_string()));
    }
    
    // Create 4x4 controlled gate matrix
    let mut controlled_matrix = QuantumMatrix::identity(4);
    
    // Copy the base gate to the bottom-right 2x2 block
    for i in 0..2 {
        for j in 0..2 {
            controlled_matrix.data[i + 2][j + 2] = base_gate.matrix.data[i][j].clone();
        }
    }
    
    let gate = QuantumGate::new(controlled_matrix, format!("C{}", base_gate.name));
    Ok(Value::LyObj(LyObj::new(Box::new(gate))))
}

/// Multi-Controlled Gate Constructor
/// Creates a gate controlled by multiple qubits
pub fn multi_controlled_gate(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime("Multi-controlled gate requires gate and control count".to_string()));
    }
    
    let base_gate = match &args[0] {
        Value::LyObj(gate_obj) => {
            if let Some(gate) = gate_obj.downcast_ref::<QuantumGate>() {
                gate
            } else {
                return Err(VmError::Runtime("First argument must be a QuantumGate".to_string()));
            }
        }
        _ => return Err(VmError::Runtime("First argument must be a QuantumGate".to_string())),
    };
    
    let control_count = match &args[1] {
        Value::Real(n) => {
            if *n < 1.0 || *n > 10.0 || n.fract() != 0.0 {
                return Err(VmError::Runtime("Control count must be integer between 1 and 10".to_string()));
            }
            *n as usize
        }
        _ => return Err(VmError::Runtime("Control count must be a number".to_string())),
    };
    
    if base_gate.n_qubits != 1 {
        return Err(VmError::Runtime("Can only create controlled version of single-qubit gates".to_string()));
    }
    
    let total_qubits = control_count + 1;
    let matrix_size = 1 << total_qubits;
    let mut controlled_matrix = QuantumMatrix::identity(matrix_size);
    
    // The gate acts only when all control qubits are |1⟩
    // This corresponds to the last 2 rows/columns of the matrix
    let control_state = (1 << control_count) - 1; // All 1s for control qubits
    let base_start = control_state << 1; // Position in the full matrix
    
    // Clear the identity in the relevant positions
    for i in 0..2 {
        for j in 0..2 {
            controlled_matrix.data[base_start + i][base_start + j] = base_gate.matrix.data[i][j].clone();
        }
    }
    
    let gate = QuantumGate::new(
        controlled_matrix, 
        format!("C^{}({})", control_count, base_gate.name)
    );
    Ok(Value::LyObj(LyObj::new(Box::new(gate))))
}

/// Custom Gate Constructor
/// Creates a gate from a matrix specification
pub fn custom_gate(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(VmError::Runtime("Custom gate requires matrix data and name".to_string()));
    }
    
    let matrix_data = match &args[0] {
        Value::List(rows) => rows,
        _ => return Err(VmError::Runtime("Matrix data must be a list of rows".to_string())),
    };
    
    let name = match &args[1] {
        Value::String(n) => n.clone(),
        _ => return Err(VmError::Runtime("Gate name must be a string".to_string())),
    };
    
    // Parse matrix data
    let mut parsed_matrix = Vec::new();
    
    for row in matrix_data {
        let row_data = match row {
            Value::List(row_elements) => row_elements,
            _ => return Err(VmError::Runtime("Each row must be a list".to_string())),
        };
        
        let mut parsed_row = Vec::new();
        
        for element in row_data {
            let complex_val = match element {
                Value::Real(real) => Complex::new(*real, 0.0),
                Value::List(complex_pair) => {
                    if complex_pair.len() != 2 {
                        return Err(VmError::Runtime("Complex number must be [real, imag]".to_string()));
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
                _ => return Err(VmError::Runtime("Matrix element must be number or [real, imag]".to_string())),
            };
            
            parsed_row.push(complex_val);
        }
        
        parsed_matrix.push(parsed_row);
    }
    
    let matrix = QuantumMatrix::from_data(parsed_matrix)?;
    
    // Verify the matrix is unitary (optional warning)
    if !matrix.is_unitary(1e-10) {
        eprintln!("Warning: Custom gate '{}' is not unitary", name);
    }
    
    let gate = QuantumGate::new(matrix, name);
    Ok(Value::LyObj(LyObj::new(Box::new(gate))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pauli_gates() {
        // Test Pauli-X
        let x_gate = pauli_x_gate(&[]).unwrap();
        if let Value::LyObj(gate_obj) = x_gate {
            let gate = gate_obj.downcast_ref::<QuantumGate>().unwrap();
            assert_eq!(gate.name, "X");
            assert_eq!(gate.n_qubits, 1);
            assert!(gate.matrix.is_unitary(1e-10));
        }
        
        // Test Pauli-Y
        let y_gate = pauli_y_gate(&[]).unwrap();
        if let Value::LyObj(gate_obj) = y_gate {
            let gate = gate_obj.downcast_ref::<QuantumGate>().unwrap();
            assert_eq!(gate.name, "Y");
        }
        
        // Test Pauli-Z
        let z_gate = pauli_z_gate(&[]).unwrap();
        if let Value::LyObj(gate_obj) = z_gate {
            let gate = gate_obj.downcast_ref::<QuantumGate>().unwrap();
            assert_eq!(gate.name, "Z");
        }
    }

    #[test]
    fn test_hadamard_gate() {
        let h_gate = hadamard_gate(&[]).unwrap();
        if let Value::LyObj(gate_obj) = h_gate {
            let gate = gate_obj.downcast_ref::<QuantumGate>().unwrap();
            assert_eq!(gate.name, "H");
            assert!(gate.matrix.is_unitary(1e-10));
            
            // Test that H² = I
            let h_squared = gate.matrix.multiply(&gate.matrix).unwrap();
            let identity = QuantumMatrix::identity(2);
            
            for i in 0..2 {
                for j in 0..2 {
                    let diff = h_squared.data[i][j].clone() - identity.data[i][j].clone();
                    assert!(diff.magnitude() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_rotation_gates() {
        let angle = PI / 4.0;
        
        let rx_gate = rotation_x_gate(&[Value::Real(angle)]).unwrap();
        let ry_gate = rotation_y_gate(&[Value::Real(angle)]).unwrap();
        let rz_gate = rotation_z_gate(&[Value::Real(angle)]).unwrap();
        
        // All rotation gates should be unitary
        for gate_value in [rx_gate, ry_gate, rz_gate] {
            if let Value::LyObj(gate_obj) = gate_value {
                let gate = gate_obj.downcast_ref::<QuantumGate>().unwrap();
                assert!(gate.matrix.is_unitary(1e-10));
            }
        }
    }

    #[test]
    fn test_cnot_gate() {
        let cnot = cnot_gate(&[]).unwrap();
        if let Value::LyObj(gate_obj) = cnot {
            let gate = gate_obj.downcast_ref::<QuantumGate>().unwrap();
            assert_eq!(gate.name, "CNOT");
            assert_eq!(gate.n_qubits, 2);
            assert!(gate.matrix.is_unitary(1e-10));
            
            // Test CNOT matrix structure
            assert_eq!(gate.matrix.data[0][0].real, 1.0);
            assert_eq!(gate.matrix.data[1][1].real, 1.0);
            assert_eq!(gate.matrix.data[2][3].real, 1.0);
            assert_eq!(gate.matrix.data[3][2].real, 1.0);
        }
    }

    #[test]
    fn test_toffoli_gate() {
        let toffoli = toffoli_gate(&[]).unwrap();
        if let Value::LyObj(gate_obj) = toffoli {
            let gate = gate_obj.downcast_ref::<QuantumGate>().unwrap();
            assert_eq!(gate.name, "Toffoli");
            assert_eq!(gate.n_qubits, 3);
            assert!(gate.matrix.is_unitary(1e-10));
        }
    }

    #[test]
    fn test_controlled_gate() {
        let x_gate = pauli_x_gate(&[]).unwrap();
        let cx_gate = controlled_gate(&[x_gate]).unwrap();
        
        if let Value::LyObj(gate_obj) = cx_gate {
            let gate = gate_obj.downcast_ref::<QuantumGate>().unwrap();
            assert_eq!(gate.name, "CX");
            assert_eq!(gate.n_qubits, 2);
            assert!(gate.matrix.is_unitary(1e-10));
        }
    }

    #[test]
    fn test_gate_methods() {
        let h_gate = hadamard_gate(&[]).unwrap();
        
        if let Value::LyObj(gate_obj) = h_gate {
            let gate = gate_obj.downcast_ref::<QuantumGate>().unwrap();
            
            // Test method calls
            let name_result = gate.call_method("name", &[]).unwrap();
            assert_eq!(name_result, Value::String("H".to_string()));
            
            let qubits_result = gate.call_method("qubits", &[]).unwrap();
            assert_eq!(qubits_result, Value::Real(1.0));
            
            let unitary_result = gate.call_method("isUnitary", &[]).unwrap();
            assert_eq!(unitary_result, Value::Boolean(true));
        }
    }
}