use lyra::{parser::Parser, Result as LyraResult};

fn parses(expr: &str) -> bool {
    match Parser::from_source(expr).and_then(|mut p| p.parse_expression()) {
        Ok(_) => true,
        Err(e) => {
            eprintln!("Failed to parse `{}`: {:?}", expr, e);
            false
        }
    }
}

fn main() -> LyraResult<()> {
    let samples = vec![
        // literals
        "42", "3.14", "\"hello\"", "True", "False",
        // list and association
        "{1, 2, 3}", "<|\"a\" -> 1, \"b\" -> {2}|>",
        // calls and pure function
        "f[x, y]", "(x) &",
        // patterns and rules
        "x_", "_Integer", "x_Integer", "x -> x^2", "x :> RandomReal[]",
        // method-like calls and indexing
        "obj.method[]", "{1,2,3}[[2]]",
        // pipelines
        "data |> Map[f] |> Total[]",
        // grouping
        "(1 + 2) * 3",
        // associations with nested content
        "<|\"nested\" -> <|\"inner\" -> 42|>|>",
    ];

    let mut failures = 0;
    for s in samples {
        if !parses(s) { failures += 1; }
    }

    if failures == 0 {
        println!("parser_conformance: OK ({} samples)", samples.len());
        Ok(())
    } else {
        eprintln!("parser_conformance: {} failures", failures);
        std::process::exit(1);
    }
}

