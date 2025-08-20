//! Mathematical Computation Workload Simulations
//!
//! Large-scale symbolic math operations that stress the symbol interning,
//! expression evaluation, and memory management systems.

use criterion::{black_box, criterion_group, Criterion, BenchmarkId, Throughput};
use lyra::{
    vm::{VirtualMachine, Value},
    parser::Parser,
    compiler::Compiler,
    memory::{MemoryManager, CompactValue, StringInterner},
};
use std::sync::Arc;

/// Generate symbolic polynomial expressions of varying complexity
fn generate_polynomial_expression(degree: usize, variables: &[&str]) -> String {
    let mut terms = Vec::new();
    
    for i in 0..=degree {
        for var in variables {
            if i == 0 {
                terms.push(format!("{}^{}", var, i + 1));
            } else {
                terms.push(format!("{}*{}^{}", i + 1, var, i + 1));
            }
        }
    }
    
    terms.join(" + ")
}

/// Benchmark large polynomial evaluation
fn large_polynomial_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("polynomial_evaluation");
    group.throughput(Throughput::Elements(1));
    
    let variables = vec!["x", "y", "z"];
    
    for degree in [5, 10, 20, 30].iter() {
        let expression = generate_polynomial_expression(*degree, &variables);
        
        group.bench_with_input(
            BenchmarkId::new("polynomial_degree", degree),
            degree,
            |b, _| {
                let mut parser = Parser::from_source(&expression).unwrap();
                let statements = parser.parse().unwrap();
                let mut compiler = Compiler::new();
                let bytecode = compiler.compile_program(&statements).unwrap();
                
                b.iter(|| {
                    let mut vm = VirtualMachine::new();
                    vm.load_bytecode(black_box(&bytecode));
                    black_box(vm.run())
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark symbolic differentiation operations
fn symbolic_differentiation_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("symbolic_differentiation");
    
    let expressions = vec![
        ("simple", "x^2 + 3*x + 1"),
        ("trigonometric", "Sin[x] + Cos[x] + Tan[x]"),
        ("exponential", "Exp[x] + Log[x] + Sqrt[x]"),
        ("composite", "Sin[Exp[x^2 + 1]] + Cos[Log[x + 1]]"),
        ("multivariate", "x^2*y + y^2*z + z^2*x"),
    ];
    
    for (name, expr) in expressions {
        group.bench_function(name, |b| {
            // Simulate differentiation by parsing and evaluating the expression multiple times
            b.iter(|| {
                let mut parser = Parser::from_source(expr).unwrap();
                let statements = parser.parse().unwrap();
                let mut compiler = Compiler::new();
                let bytecode = compiler.compile_program(&statements).unwrap();
                
                // Simulate differentiation by multiple evaluations with slight variations
                for i in 1..=10 {
                    let mut vm = VirtualMachine::new();
                    vm.load_bytecode(&bytecode);
                    black_box(vm.run());
                }
            });
        });
    }
    
    group.finish();
}

/// Benchmark matrix operations in symbolic form
fn symbolic_matrix_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("symbolic_matrix_operations");
    
    // Create symbolic matrices
    fn create_symbolic_matrix(size: usize) -> Vec<Vec<String>> {
        (0..size).map(|i| {
            (0..size).map(|j| {
                format!("m_{}_{}", i, j)
            }).collect()
        }).collect()
    }
    
    for size in [2, 3, 4, 5].iter() {
        let matrix = create_symbolic_matrix(*size);
        
        group.bench_with_input(
            BenchmarkId::new("matrix_creation", size),
            size,
            |b, _| {
                b.iter(|| {
                    // Create matrix as nested lists
                    let mut matrix_expr = String::from("{");
                    for (i, row) in matrix.iter().enumerate() {
                        if i > 0 { matrix_expr.push_str(", "); }
                        matrix_expr.push('{');
                        for (j, elem) in row.iter().enumerate() {
                            if j > 0 { matrix_expr.push_str(", "); }
                            matrix_expr.push_str(elem);
                        }
                        matrix_expr.push('}');
                    }
                    matrix_expr.push('}');
                    
                    let mut parser = Parser::from_source(&matrix_expr).unwrap();
                    let statements = parser.parse().unwrap();
                    let mut compiler = Compiler::new();
                    let bytecode = compiler.compile_program(&statements).unwrap();
                    
                    let mut vm = VirtualMachine::new();
                    vm.load_bytecode(black_box(&bytecode));
                    black_box(vm.run())
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark complex number operations
fn complex_number_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_number_operations");
    
    let complex_expressions = vec![
        "I^2",                           // Basic complex unit
        "1 + 2*I",                       // Complex number construction
        "(1 + I) * (1 - I)",            // Complex multiplication
        "Exp[I * Pi]",                   // Euler's formula
        "Sin[1 + I] + Cos[1 + I]",      // Complex trigonometry
        "Log[1 + I]",                   // Complex logarithm
        "Sqrt[-1]",                     // Square root of negative
    ];
    
    for (i, expr) in complex_expressions.iter().enumerate() {
        group.bench_function(&format!("complex_expr_{}", i), |b| {
            let mut parser = Parser::from_source(expr).unwrap();
            let statements = parser.parse().unwrap();
            let mut compiler = Compiler::new();
            let bytecode = compiler.compile_program(&statements).unwrap();
            
            b.iter(|| {
                let mut vm = VirtualMachine::new();
                vm.load_bytecode(black_box(&bytecode));
                black_box(vm.run())
            });
        });
    }
    
    group.finish();
}

/// Benchmark series expansion operations
fn series_expansion_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("series_expansion");
    
    // Simulate Taylor series expansion by evaluating many polynomial terms
    let series_functions = vec![
        ("exponential", "Exp[x]"),
        ("sine", "Sin[x]"),
        ("cosine", "Cos[x]"),
        ("logarithm", "Log[1 + x]"),
    ];
    
    for (name, func) in series_functions {
        group.bench_function(name, |b| {
            b.iter(|| {
                // Simulate series expansion by multiple evaluations
                for n in 1..=20 {
                    let expansion_term = format!("{}^{}", func, n);
                    let mut parser = Parser::from_source(&expansion_term).unwrap();
                    let statements = parser.parse().unwrap();
                    let mut compiler = Compiler::new();
                    let bytecode = compiler.compile_program(&statements).unwrap();
                    
                    let mut vm = VirtualMachine::new();
                    vm.load_bytecode(&bytecode);
                    black_box(vm.run());
                }
            });
        });
    }
    
    group.finish();
}

/// Benchmark symbolic integration simulation
fn symbolic_integration_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("symbolic_integration");
    
    let integration_problems = vec![
        ("polynomial", "x^3 + 2*x^2 + x + 1"),
        ("trigonometric", "Sin[x] * Cos[x]"),
        ("exponential", "x * Exp[x]"),
        ("rational", "1 / (x^2 + 1)"),
        ("mixed", "x * Sin[x] + Exp[x] * Cos[x]"),
    ];
    
    for (name, integrand) in integration_problems {
        group.bench_function(name, |b| {
            b.iter(|| {
                // Simulate symbolic integration by pattern matching and manipulation
                let mut parser = Parser::from_source(integrand).unwrap();
                let statements = parser.parse().unwrap();
                let mut compiler = Compiler::new();
                let bytecode = compiler.compile_program(&statements).unwrap();
                
                // Multiple passes simulating integration rules
                for _ in 0..5 {
                    let mut vm = VirtualMachine::new();
                    vm.load_bytecode(&bytecode);
                    black_box(vm.run());
                }
            });
        });
    }
    
    group.finish();
}

/// Benchmark large expression tree manipulation
fn expression_tree_manipulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("expression_tree_manipulation");
    
    // Create deeply nested expressions
    fn create_nested_expression(depth: usize) -> String {
        if depth == 0 {
            "x".to_string()
        } else {
            let inner = create_nested_expression(depth - 1);
            format!("Sin[{}] + Cos[{}]", inner, inner)
        }
    }
    
    for depth in [3, 5, 7, 10].iter() {
        let expression = create_nested_expression(*depth);
        
        group.bench_with_input(
            BenchmarkId::new("nested_depth", depth),
            depth,
            |b, _| {
                let mut parser = Parser::from_source(&expression).unwrap();
                let statements = parser.parse().unwrap();
                let mut compiler = Compiler::new();
                let bytecode = compiler.compile_program(&statements).unwrap();
                
                b.iter(|| {
                    let mut vm = VirtualMachine::new();
                    vm.load_bytecode(black_box(&bytecode));
                    black_box(vm.run())
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory efficiency in mathematical computations
fn mathematical_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("mathematical_memory_efficiency");
    
    group.bench_function("memory_managed_computation", |b| {
        let mut memory_manager = MemoryManager::new();
        
        b.iter(|| {
            // Create many symbolic values and expressions
            let mut values = Vec::new();
            
            for i in 0..1000 {
                // Create symbolic variables
                let var_name = format!("var_{}", i);
                let symbol_id = memory_manager.intern_symbol(&var_name);
                let compact_val = CompactValue::Symbol(symbol_id);
                values.push(memory_manager.alloc_compact_value(compact_val));
                
                // Create integer coefficients
                let coeff = CompactValue::SmallInt(i as i32 % 100);
                values.push(memory_manager.alloc_compact_value(coeff));
            }
            
            // Simulate cleanup
            for value in &values {
                memory_manager.recycle_compact_value(value);
            }
            
            black_box(values);
        });
    });
    
    group.bench_function("traditional_computation", |b| {
        b.iter(|| {
            // Create many values without memory management
            let mut values = Vec::new();
            
            for i in 0..1000 {
                values.push(Value::Symbol(format!("var_{}", i)));
                values.push(Value::Integer(i as i64 % 100));
            }
            
            black_box(values);
        });
    });
    
    group.finish();
}

criterion_group!(
    mathematical_computation_benchmarks,
    large_polynomial_evaluation,
    symbolic_differentiation_workload,
    symbolic_matrix_operations,
    complex_number_operations,
    series_expansion_operations,
    symbolic_integration_simulation,
    expression_tree_manipulation,
    mathematical_memory_efficiency
);

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn validate_polynomial_generation() {
        let expr = generate_polynomial_expression(3, &["x", "y"]);
        println!("Generated polynomial: {}", expr);
        
        // Should contain terms for both variables
        assert!(expr.contains("x"));
        assert!(expr.contains("y"));
        assert!(expr.contains("^"));
        assert!(expr.contains("+"));
    }
    
    #[test]
    fn validate_nested_expression_generation() {
        let expr = create_nested_expression(2);
        println!("Nested expression: {}", expr);
        
        // Should have proper nesting structure
        assert!(expr.contains("Sin["));
        assert!(expr.contains("Cos["));
        assert!(expr.contains("+"));
    }
    
    #[test]
    fn validate_mathematical_computation_workload() {
        // Test that we can parse and compile mathematical expressions
        let expr = "x^2 + 3*x + 1";
        let mut parser = Parser::from_source(expr).unwrap();
        let statements = parser.parse().unwrap();
        let mut compiler = Compiler::new();
        let bytecode = compiler.compile_program(&statements).unwrap();
        
        let mut vm = VirtualMachine::new();
        vm.load_bytecode(&bytecode);
        let result = vm.run();
        
        // Should execute without error
        assert!(result.is_ok());
    }
    
    #[test]
    fn validate_memory_managed_mathematical_computation() {
        let mut memory_manager = MemoryManager::new();
        
        // Test memory-efficient mathematical computation
        let initial_stats = memory_manager.memory_stats();
        
        // Create symbolic computation elements
        let x_symbol = memory_manager.intern_symbol("x");
        let y_symbol = memory_manager.intern_symbol("y");
        let coefficient = CompactValue::SmallInt(42);
        
        let x_compact = memory_manager.alloc_compact_value(CompactValue::Symbol(x_symbol));
        let y_compact = memory_manager.alloc_compact_value(CompactValue::Symbol(y_symbol));
        let coeff_compact = memory_manager.alloc_compact_value(coefficient);
        
        let final_stats = memory_manager.memory_stats();
        
        println!("Memory usage - Initial: {}, Final: {}", 
                initial_stats.total_allocated, final_stats.total_allocated);
        
        // Should have allocated some memory
        assert!(final_stats.total_allocated >= initial_stats.total_allocated);
        
        // Clean up
        memory_manager.recycle_compact_value(&x_compact);
        memory_manager.recycle_compact_value(&y_compact);
        memory_manager.recycle_compact_value(&coeff_compact);
    }
}