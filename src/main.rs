use clap::{Parser, Subcommand};
use lyra::{compiler::Compiler, parser::Parser as LyraParser, Result};
use rustyline::{error::ReadlineError, DefaultEditor};
use std::{fs, path::PathBuf, process::{Command, Stdio}};
use walkdir::WalkDir;

#[derive(Parser)]
#[command(name = "lyra")]
#[command(about = "A symbolic computation engine")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start interactive REPL
    Repl,
    /// Run a script file
    Run { file: PathBuf },
    /// Build a script without running
    Build { file: PathBuf },
    /// Dump intermediate representation
    DumpIr { file: PathBuf },

    /// Format Lyra source files and Rust (optional)
    Fmt {
        /// Check mode: exit non-zero if any file would be reformatted
        #[arg(long)]
        check: bool,
        /// Format Rust and common docs/configs too
        #[arg(long)]
        all: bool,
        /// Limit to a path or file (defaults to repo root)
        #[arg(long)]
        path: Option<PathBuf>,
    },

    /// Run linters (clippy; extend later)
    Lint {
        /// Emit JSON report (placeholder wiring)
        #[arg(long)]
        json: bool,
    },

    /// Run format check + lint (no writes)
    Check,

    /// Attempt autofixes where possible (clippy --fix)
    Fix,

    /// CI entrypoint: fmt --check, lint, tests
    Ci,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Repl => run_repl(),
        Commands::Run { file } => run_file(&file),
        Commands::Build { file } => build_file(&file),
        Commands::DumpIr { file } => dump_ir(&file),
        Commands::Fmt { check, all, path } => fmt_cmd(check, all, path),
        Commands::Lint { json } => lint_cmd(json),
        Commands::Check => {
            fmt_cmd(true, true, None)?;
            lint_cmd(false)
        }
        Commands::Fix => fix_cmd(),
        Commands::Ci => ci_cmd(),
    }
}

/// Run the interactive REPL
fn run_repl() -> Result<()> {
    println!(
        "Lyra Symbolic Computation Engine v{}",
        env!("CARGO_PKG_VERSION")
    );
    println!("Type expressions to evaluate, 'help' for assistance, or 'exit' to quit.");
    println!();

    let mut rl = DefaultEditor::new().map_err(|e| lyra::Error::Runtime {
        message: e.to_string(),
    })?;
    let mut line_number = 1;

    loop {
        let prompt = format!("lyra[{}]> ", line_number);

        match rl.readline(&prompt) {
            Ok(line) => {
                let line = line.trim();

                // Handle special commands
                if line.is_empty() {
                    continue;
                }
                if line == "exit" || line == "quit" {
                    println!("Goodbye!");
                    break;
                }
                if line == "help" || line == "?" {
                    show_repl_help();
                    continue;
                }
                if line == "functions" {
                    show_available_functions();
                    continue;
                }
                if line == "examples" {
                    show_examples();
                    continue;
                }

                // Add to history
                let _ = rl.add_history_entry(line);

                // Evaluate the expression
                match eval_expression(line) {
                    Ok(result) => {
                        println!("Out[{}]= {}", line_number, format_value(&result));
                    }
                    Err(e) => {
                        eprintln!("{}", lyra::error::format_error_with_context(&e, line));
                    }
                }

                line_number += 1;
            }
            Err(ReadlineError::Interrupted) => {
                println!("^C");
                continue;
            }
            Err(ReadlineError::Eof) => {
                println!("^D");
                break;
            }
            Err(err) => {
                eprintln!("Error reading line: {}", err);
                break;
            }
        }
    }

    Ok(())
}

/// Run a script file
fn run_file(file_path: &PathBuf) -> Result<()> {
    let source = fs::read_to_string(file_path)?;

    match eval_expression(&source) {
        Ok(result) => {
            println!("{}", format_value(&result));
            Ok(())
        }
        Err(e) => {
            eprintln!("Error executing {}:", file_path.display());
            eprintln!("{}", lyra::error::format_error_with_context(&e, &source));
            Err(e)
        }
    }
}

/// Build (compile) a file without running it
fn build_file(file_path: &PathBuf) -> Result<()> {
    let source = fs::read_to_string(file_path)?;

    match compile_source(&source) {
        Ok(_) => {
            println!("✓ {} compiled successfully", file_path.display());
            Ok(())
        }
        Err(e) => {
            eprintln!("✗ Compilation failed for {}:", file_path.display());
            eprintln!("{}", lyra::error::format_error_with_context(&e, &source));
            Err(e)
        }
    }
}

/// Dump intermediate representation for a file
fn dump_ir(file_path: &PathBuf) -> Result<()> {
    let source = fs::read_to_string(file_path)?;

    println!("=== Source ===");
    println!("{}", source);
    println!();

    // Parse to AST
    let mut parser = LyraParser::from_source(&source)?;
    let statements = parser.parse()?;

    println!("=== AST ===");
    for (i, stmt) in statements.iter().enumerate() {
        println!("Statement {}: {:#?}", i + 1, stmt);
    }
    println!();

    // Compile to bytecode
    let mut compiler = Compiler::new();
    for stmt in &statements {
        compiler.compile_expr(stmt)?;
    }
    // No halt instruction needed in minimal opcode set

    println!("=== Bytecode ===");
    for (i, instruction) in compiler.context.code.iter().enumerate() {
        println!("{:04}: {:?}", i, instruction);
    }
    println!();

    println!("=== Constants ===");
    for (i, constant) in compiler.context.constants.iter().enumerate() {
        println!("{}: {:?}", i, constant);
    }
    println!();

    println!("=== Symbols ===");
    for (name, index) in &compiler.context.symbols {
        println!("{}: {}", index, name);
    }

    Ok(())
}

/// Evaluate a single expression and return the result
fn eval_expression(source: &str) -> Result<lyra::vm::Value> {
    // Parse source code to AST
    let mut parser = LyraParser::from_source(source)?;
    let statements = parser.parse()?;

    // Take the last statement as the expression to evaluate
    let expr = statements.last().ok_or_else(|| lyra::Error::Runtime {
        message: "No expressions to evaluate".to_string(),
    })?;

    // Compile and execute
    Compiler::eval(expr).map_err(|e| lyra::Error::Compilation {
        message: e.to_string(),
    })
}

/// Compile source code without executing
fn compile_source(source: &str) -> Result<Compiler> {
    let mut parser = LyraParser::from_source(source)?;
    let statements = parser.parse()?;

    let mut compiler = Compiler::new();
    for stmt in &statements {
        compiler.compile_expr(stmt)?;
    }
    // No halt instruction needed in minimal opcode set

    Ok(compiler)
}

/// Format a value for display in the REPL
fn format_value(value: &lyra::vm::Value) -> String {
    match value {
        lyra::vm::Value::Integer(n) => n.to_string(),
        lyra::vm::Value::Real(f) => {
            // Format floats nicely - remove trailing zeros
            if f.fract() == 0.0 {
                format!("{:.1}", f)
            } else {
                f.to_string()
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
        lyra::vm::Value::LyObj(obj) => {
            format!("{}[...]", obj.type_name())
        }
    }
}

/// Show REPL help information
fn show_repl_help() {
    println!("Lyra REPL Help");
    println!("===============");
    println!();
    println!("Commands:");
    println!("  help, ?         - Show this help message");
    println!("  functions       - List all available functions");
    println!("  examples        - Show example expressions");
    println!("  exit, quit      - Exit the REPL");
    println!();
    println!("Syntax:");
    println!("  Function calls: f[x, y]");
    println!("  Lists:          {{1, 2, 3}}");
    println!("  Arithmetic:     2 + 3 * 4");
    println!("  Strings:        \"Hello, World!\"");
    println!("  Rules:          x -> x^2");
    println!("  Replacement:    expr /. rule");
    println!();
    println!("Try: Sin[Pi/2], Length[{{1,2,3}}], 2^10");
    println!();
}

/// Show available functions
fn show_available_functions() {
    println!("Available Functions");
    println!("===================");
    println!();
    println!("Math Functions:");
    println!("  Sin[x], Cos[x], Tan[x]    - Trigonometric functions");
    println!("  Exp[x], Log[x], Sqrt[x]   - Exponential and logarithmic");
    println!("  +, -, *, /, ^             - Arithmetic operators");
    println!();
    println!("List Functions:");
    println!("  Length[list]              - Get length of a list");
    println!("  Head[list]                - Get first element");
    println!("  Tail[list]                - Get all but first element");
    println!("  Append[list, elem]        - Add element to list");
    println!("  Flatten[list]             - Flatten nested lists");
    println!();
    println!("String Functions:");
    println!("  StringLength[str]         - Get string length");
    println!("  StringJoin[str1, str2]    - Concatenate strings");
    println!("  StringTake[str, n]        - Take first n characters");
    println!("  StringDrop[str, n]        - Drop first n characters");
    println!();
    println!("Rule Functions:");
    println!("  Rule[x, y]                - Create replacement rule x -> y");
    println!("  RuleDelayed[x, y]         - Create delayed rule x :> y");
    println!();
}

/// Show example expressions
fn show_examples() {
    println!("Example Expressions");
    println!("===================");
    println!();
    println!("Basic Arithmetic:");
    println!("  2 + 3 * 4                 => 14");
    println!("  2^10                      => 1024");
    println!("  Sqrt[16]                  => 4.0");
    println!();
    println!("Lists:");
    println!("  {{1, 2, 3, 4, 5}}");
    println!("  Length[{{1, 2, 3}}]        => 3");
    println!("  Head[{{a, b, c}}]          => a");
    println!("  Append[{{1, 2}}, 3]        => {{1, 2, 3}}");
    println!();
    println!("Mathematical Functions:");
    println!("  Sin[Pi/2]                 => 1.0");
    println!("  Cos[0]                    => 1.0");
    println!("  Log[Exp[1]]               => 1.0");
    println!();
    println!("Strings:");
    println!("  StringLength[\"Hello\"]     => 5");
    println!("  StringJoin[\"Hello\", \" \", \"World!\"]");
    println!();
    println!("Complex Expressions:");
    println!("  Sin[Pi/4] + Cos[Pi/4]");
    println!("  Flatten[{{{{1, 2}}, 3}}, 4}}]");
    println!();
}

/// Helper function to run subprocesses
fn run_cmd(name: &str, mut cmd: Command) -> Result<()> {
    let status = cmd
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|e| lyra::Error::Runtime {
            message: format!("failed to spawn {name}: {e}"),
        })?;
    if !status.success() {
        return Err(lyra::Error::Runtime {
            message: format!("{name} failed with {status}"),
        });
    }
    Ok(())
}

/// Format Lyra source files and optionally Rust files
fn fmt_cmd(check: bool, all: bool, path: Option<PathBuf>) -> Result<()> {
    use std::io::Read;

    // 1) Lyra files formatting (.ly, .lyra) — recurse from path or repo root
    let root = path.unwrap_or_else(|| PathBuf::from("."));
    let mut changed = 0usize;

    for entry in WalkDir::new(&root).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        let p = entry.path();
        let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("");
        if matches!(ext, "ly" | "lyra") {
            let mut src = String::new();
            fs::File::open(p)
                .and_then(|mut f| f.read_to_string(&mut src))
                .map_err(|e| lyra::Error::Runtime {
                    message: format!("read {}: {e}", p.display()),
                })?;
            let out = lyra::format::format_str(&src, &lyra::format::FormatterConfig::default())
                .map_err(|e| lyra::Error::Runtime {
                    message: format!("format {}: {e}", p.display()),
                })?;

            if check {
                if normalize_newlines(&out) != normalize_newlines(&src) {
                    eprintln!("would reformat: {}", p.display());
                    changed += 1;
                }
            } else if out != src {
                fs::write(p, out).map_err(|e| lyra::Error::Runtime {
                    message: format!("write {}: {e}", p.display()),
                })?;
            }
        }
    }

    if check && changed > 0 {
        return Err(lyra::Error::Runtime {
            message: format!("{} file(s) would be reformatted", changed),
        });
    }

    // 2) cargo fmt for Rust sources
    let mut c = Command::new("cargo");
    c.arg("fmt");
    if check {
        c.arg("--").arg("--check");
    }
    run_cmd("cargo fmt", c)?;

    // 3) Optional doc/config formatters when --all (placeholder; wire later)
    if all {
        // e.g., taplo fmt, markdownlint, shfmt, prettier/biome, etc.
    }

    Ok(())
}

/// Run linters (clippy)
fn lint_cmd(_json: bool) -> Result<()> {
    // clippy over all targets/features, fail on warnings
    let mut c = Command::new("cargo");
    c.args(["clippy", "--all-targets", "--all-features", "--", "-D", "warnings"]);
    run_cmd("cargo clippy", c)
}

/// Attempt autofixes where possible
fn fix_cmd() -> Result<()> {
    let mut c = Command::new("cargo");
    c.args([
        "clippy",
        "--fix",
        "--allow-dirty",
        "--allow-staged",
        "--",
        "-D",
        "warnings",
    ]);
    run_cmd("cargo clippy --fix", c)
}

/// CI entrypoint: format check, lint, tests
fn ci_cmd() -> Result<()> {
    fmt_cmd(true, true, None)?;
    lint_cmd(false)?;
    // run tests
    let mut cmd = Command::new("cargo");
    cmd.arg("test");
    run_cmd("cargo test", cmd)
}

/// Normalize newlines for comparison
fn normalize_newlines(s: &str) -> String {
    s.replace("\r\n", "\n")
}
