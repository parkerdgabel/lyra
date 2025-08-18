use insta::assert_yaml_snapshot;
use lyra::{compiler::Compiler, parser::Parser};

/// Helper function to evaluate an expression and return the formatted result
fn evaluate_expression(source: &str) -> String {
    let mut parser = Parser::from_source(source).unwrap();
    let statements = parser.parse().unwrap();
    let expr = &statements[0];

    match Compiler::eval(expr) {
        Ok(value) => format_value(&value),
        Err(e) => format!("Error: {}", e),
    }
}

/// Format a value for consistent output
fn format_value(value: &lyra::vm::Value) -> String {
    match value {
        lyra::vm::Value::Integer(n) => n.to_string(),
        lyra::vm::Value::Real(f) => {
            if f.fract() == 0.0 {
                format!("{:.1}", f)
            } else {
                format!("{:.6}", f) // Fixed precision for consistent snapshots
            }
        }
        lyra::vm::Value::String(s) => format!("\"{}\"", s),
        lyra::vm::Value::Symbol(s) => s.clone(),
        lyra::vm::Value::List(items) => {
            let formatted_items: Vec<String> = items.iter().map(format_value).collect();
            format!("{{{}}}", formatted_items.join(", "))
        }
        lyra::vm::Value::Function(name) => format!("Function[{}]", name),
        lyra::vm::Value::Boolean(b) => if *b { "True" } else { "False" }.to_string(),
        lyra::vm::Value::Tensor(tensor) => {
            format!("Tensor[shape: {:?}, elements: {}]", 
                    tensor.shape(), 
                    tensor.len())
        }
        lyra::vm::Value::Missing => "Missing[]".to_string(),
        // Note: Series, Table, Dataset, Schema are now handled by LyObj case below
        lyra::vm::Value::LyObj(obj) => {
            format!("{}[...]", obj.type_name())
        }
    }
}

#[test]
fn test_arithmetic_expressions_snapshot() {
    let expressions = vec![
        "2 + 3",
        "10 - 4",
        "3 * 7",
        "15 / 3",
        "2^8",
        "2 + 3 * 4",
        "(2 + 3) * 4",
        "2^3 * 4 + 1",
        "2^(3 + 1)",
        "(5 + 3) / (2 * 2)",
    ];

    let results: Vec<String> = expressions
        .iter()
        .map(|expr| evaluate_expression(expr))
        .collect();

    assert_yaml_snapshot!("arithmetic_expressions", results);
}

#[test]
fn test_mathematical_functions_snapshot() {
    let expressions = vec![
        "Sin[0]", "Cos[0]", "Tan[0]", "Exp[0]", "Exp[1]", "Log[1]", "Sqrt[4]", "Sqrt[16]", "2^10",
        "10^3",
    ];

    let results: Vec<String> = expressions
        .iter()
        .map(|expr| evaluate_expression(expr))
        .collect();

    assert_yaml_snapshot!("mathematical_functions", results);
}

#[test]
fn test_list_operations_snapshot() {
    let expressions = vec![
        "{1, 2, 3, 4, 5}",
        "Length[{1, 2, 3, 4, 5}]",
        "Head[{10, 20, 30}]",
        "Tail[{10, 20, 30}]",
        "Append[{1, 2, 3}, 4]",
        "Length[{}]",
        "Append[{}, \"first\"]",
        "{\"apple\", \"banana\", \"cherry\"}",
    ];

    let results: Vec<String> = expressions
        .iter()
        .map(|expr| evaluate_expression(expr))
        .collect();

    assert_yaml_snapshot!("list_operations", results);
}

#[test]
fn test_string_operations_snapshot() {
    let expressions = vec![
        "\"Hello, World!\"",
        "StringLength[\"Hello\"]",
        "StringLength[\"\"]",
        "StringJoin[\"Hello\", \" \", \"World\"]",
        "StringTake[\"Programming\", 4]",
        "StringDrop[\"Programming\", 4]",
        "StringLength[StringJoin[\"Hello\", \" \", \"World\"]]",
    ];

    let results: Vec<String> = expressions
        .iter()
        .map(|expr| evaluate_expression(expr))
        .collect();

    assert_yaml_snapshot!("string_operations", results);
}

#[test]
fn test_complex_expressions_snapshot() {
    let expressions = vec![
        "Sin[3.14159 / 2] + Cos[0]",
        "Sqrt[3^2 + 4^2]",
        "Length[Append[{1, 2, 3}, Sqrt[16]]]",
        "StringLength[StringJoin[\"Result: \", \"42\"]]",
        "((2 + 3) * (4 + 5)) / ((6 + 7) - (8 - 9))",
    ];

    let results: Vec<String> = expressions
        .iter()
        .map(|expr| evaluate_expression(expr))
        .collect();

    assert_yaml_snapshot!("complex_expressions", results);
}

#[test]
fn test_edge_cases_snapshot() {
    let expressions = vec![
        "0 + 0",
        "1 * 1",
        "0 / 1",
        "Sqrt[0]",
        "Log[1]",
        "Sin[0]",
        "Cos[0]",
        "Exp[0]",
        "Length[{}]",
        "StringLength[\"\"]",
    ];

    let results: Vec<String> = expressions
        .iter()
        .map(|expr| evaluate_expression(expr))
        .collect();

    assert_yaml_snapshot!("edge_cases", results);
}
