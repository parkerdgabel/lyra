use lyra::{
    ast::Expr,
    compiler::Compiler,
    lexer::Lexer,
    parser::Parser,
    vm::{Value, VirtualMachine},
};

fn parse_statements(src: &str) -> Vec<Expr> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().expect("tokenize");
    let mut parser = Parser::new(tokens);
    parser.parse().expect("parse")
}

#[test]
fn test_delayed_simple_value_evaluates_on_read() {
    // y := 100
    let stmts = parse_statements("y := 100");
    let assign = &stmts[0];

    let mut c = Compiler::new();
    c.compile_expr(assign).expect("compile :=");

    let mut vm = VirtualMachine::new();
    vm.load(c.context.code, c.context.constants);

    // Execute the assignment
    while vm.ip < vm.code.len() {
        vm.step().expect("vm step");
    }

    // Now evaluate symbol y
    let y_stmt = parse_statements("y");
    let mut c2 = Compiler::new();
    c2.compile_expr(&y_stmt[0]).expect("compile y");
    let mut vm2 = VirtualMachine::new();
    vm2.global_symbols = vm.global_symbols.clone();
    vm2.delayed_definitions = vm.delayed_definitions.clone();
    vm2.load(c2.context.code, c2.context.constants);

    let result = vm2.run().expect("run y");
    assert_eq!(result, Value::Integer(100));
}

#[test]
fn test_delayed_depends_on_immediate_value() {
    // x = 2
    let x_stmts = parse_statements("x = 2");
    let mut cx = Compiler::new();
    cx.compile_expr(&x_stmts[0]).expect("compile =");
    let mut vm = VirtualMachine::new();
    vm.load(cx.context.code, cx.context.constants);
    while vm.ip < vm.code.len() {
        vm.step().expect("vm step");
    }

    // y := x + 1
    let y_stmts = parse_statements("y := x + 1");
    let mut cy = Compiler::new();
    cy.compile_expr(&y_stmts[0]).expect("compile :=");
    let mut vm_y = VirtualMachine::new();
    vm_y.global_symbols = vm.global_symbols.clone();
    vm_y.load(cy.context.code, cy.context.constants);
    while vm_y.ip < vm_y.code.len() {
        vm_y.step().expect("vm step");
    }

    // Evaluate y should yield 3
    let y_eval = parse_statements("y");
    let mut ce = Compiler::new();
    ce.compile_expr(&y_eval[0]).expect("compile y");
    let mut vm_eval = VirtualMachine::new();
    vm_eval.global_symbols = vm_y.global_symbols.clone();
    vm_eval.delayed_definitions = vm_y.delayed_definitions.clone();
    vm_eval.load(ce.context.code, ce.context.constants);
    let result = vm_eval.run().expect("run y");
    assert_eq!(result, Value::Integer(3));
}

#[test]
fn test_delayed_self_recursion_errors() {
    // a := a
    let stmts = parse_statements("a := a");
    let mut c = Compiler::new();
    c.compile_expr(&stmts[0]).expect("compile :=");
    let mut vm = VirtualMachine::new();
    vm.load(c.context.code, c.context.constants);
    while vm.ip < vm.code.len() {
        vm.step().expect("vm step");
    }

    // Evaluate a -> should error due to recursive SetDelayed
    let eval = parse_statements("a");
    let mut ce = Compiler::new();
    ce.compile_expr(&eval[0]).expect("compile a");
    let mut vm2 = VirtualMachine::new();
    vm2.delayed_definitions = vm.delayed_definitions.clone();
    vm2.load(ce.context.code, ce.context.constants);
    let res = vm2.run();
    assert!(res.is_err(), "expected recursion error");
}

