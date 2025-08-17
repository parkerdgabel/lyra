use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lyra::{compiler::Compiler, parser::Parser};

fn parse_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("parsing");

    group.bench_function("simple_arithmetic", |b| {
        b.iter(|| {
            let source = black_box("2 + 3 * 4");
            let mut parser = Parser::from_source(source).unwrap();
            parser.parse().unwrap()
        });
    });

    group.bench_function("complex_expression", |b| {
        b.iter(|| {
            let source = black_box("Sin[Sqrt[2] / 2] + Cos[3.14159 / 4]^2");
            let mut parser = Parser::from_source(source).unwrap();
            parser.parse().unwrap()
        });
    });

    group.bench_function("nested_lists", |b| {
        b.iter(|| {
            let source = black_box("Length[{1, 2, 3, 4, 5}]");
            let mut parser = Parser::from_source(source).unwrap();
            parser.parse().unwrap()
        });
    });

    group.finish();
}

fn compilation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("compilation");

    group.bench_function("arithmetic_compilation", |b| {
        let source = "2 + 3 * 4 - 1";
        let mut parser = Parser::from_source(source).unwrap();
        let statements = parser.parse().unwrap();
        let expr = &statements[0];

        b.iter(|| {
            let mut compiler = Compiler::new();
            compiler.compile_expr(black_box(expr)).unwrap()
        });
    });

    group.bench_function("function_call_compilation", |b| {
        let source = "Sin[3.14159 / 2]";
        let mut parser = Parser::from_source(source).unwrap();
        let statements = parser.parse().unwrap();
        let expr = &statements[0];

        b.iter(|| {
            let mut compiler = Compiler::new();
            compiler.compile_expr(black_box(expr)).unwrap()
        });
    });

    group.finish();
}

fn evaluation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluation");

    group.bench_function("simple_math", |b| {
        b.iter(|| {
            let source = black_box("2 + 3 * 4");
            let mut parser = Parser::from_source(source).unwrap();
            let statements = parser.parse().unwrap();
            let expr = &statements[0];
            Compiler::eval(expr).unwrap()
        });
    });

    group.bench_function("trigonometric", |b| {
        b.iter(|| {
            let source = black_box("Sin[3.14159 / 2]");
            let mut parser = Parser::from_source(source).unwrap();
            let statements = parser.parse().unwrap();
            let expr = &statements[0];
            Compiler::eval(expr).unwrap()
        });
    });

    group.bench_function("list_operations", |b| {
        b.iter(|| {
            let source = black_box("Length[{1, 2, 3, 4, 5}]");
            let mut parser = Parser::from_source(source).unwrap();
            let statements = parser.parse().unwrap();
            let expr = &statements[0];
            Compiler::eval(expr).unwrap()
        });
    });

    group.bench_function("nested_function_calls", |b| {
        b.iter(|| {
            let source = black_box("Sqrt[Sin[3.14159/4]^2 + Cos[3.14159/4]^2]");
            let mut parser = Parser::from_source(source).unwrap();
            let statements = parser.parse().unwrap();
            let expr = &statements[0];
            Compiler::eval(expr).unwrap()
        });
    });

    group.bench_function("complex_arithmetic", |b| {
        b.iter(|| {
            let source = black_box("((2 + 3) * (4 + 5)) / ((6 + 7) - (8 - 9))");
            let mut parser = Parser::from_source(source).unwrap();
            let statements = parser.parse().unwrap();
            let expr = &statements[0];
            Compiler::eval(expr).unwrap()
        });
    });

    group.finish();
}

fn end_to_end_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end");

    group.bench_function("full_pipeline_simple", |b| {
        b.iter(|| {
            let source = black_box("2 + 3 * 4");
            let mut parser = Parser::from_source(source).unwrap();
            let statements = parser.parse().unwrap();
            let expr = &statements[0];
            Compiler::eval(expr).unwrap()
        });
    });

    group.bench_function("full_pipeline_complex", |b| {
        b.iter(|| {
            let source = black_box("Sin[Sqrt[2] / 2] + Length[{1, 2, 3}] * Cos[0]");
            let mut parser = Parser::from_source(source).unwrap();
            let statements = parser.parse().unwrap();
            let expr = &statements[0];
            Compiler::eval(expr).unwrap()
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    parse_benchmark,
    compilation_benchmark,
    evaluation_benchmark,
    end_to_end_benchmark
);
criterion_main!(benches);
