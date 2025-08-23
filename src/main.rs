use clap::{Parser, Subcommand};
use lyra::{compiler::Compiler, parser::Parser as LyraParser, Result, repl::ReplEngine, stdlib::StandardLibrary};
use lyra::repl::enhanced_helper::EnhancedLyraHelper;
use rustyline::{error::ReadlineError, Editor, Config};
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

    /// Package management commands
    Pkg {
        #[command(subcommand)]
        command: PkgCommands,
    },

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

#[derive(Subcommand)]
enum PkgCommands {
    /// Initialize a new package
    Init {
        /// Package name
        name: Option<String>,
        /// Package directory
        #[arg(short, long)]
        path: Option<PathBuf>,
    },
    
    /// Build the current package
    Build {
        /// Build in release mode
        #[arg(short, long)]
        release: bool,
    },
    
    /// Run package tests
    Test {
        /// Run only specific test
        #[arg(short, long)]
        test: Option<String>,
    },
    
    /// Install a package
    Install {
        /// Package name
        package: String,
        /// Version constraint
        #[arg(short, long)]
        version: Option<String>,
        /// Install as development dependency
        #[arg(long)]
        dev: bool,
    },
    
    /// Update packages
    Update {
        /// Specific package to update
        package: Option<String>,
    },
    
    /// Remove a package
    Remove {
        /// Package name
        package: String,
    },
    
    /// Search for packages
    Search {
        /// Search query
        query: String,
        /// Limit number of results
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },
    
    /// Show package information
    Info {
        /// Package name
        package: String,
        /// Show specific version
        #[arg(short, long)]
        version: Option<String>,
    },
    
    /// List installed packages
    List {
        /// Show only outdated packages
        #[arg(long)]
        outdated: bool,
    },
    
    /// Show dependency tree
    Tree {
        /// Package to show tree for
        package: Option<String>,
        /// Maximum depth
        #[arg(short, long)]
        depth: Option<usize>,
    },
    
    /// Check package health
    Check,
    
    /// Publish package to registry
    Publish {
        /// Registry to publish to
        #[arg(short, long)]
        registry: Option<String>,
        /// Authentication token
        #[arg(short, long)]
        token: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Repl => run_repl(),
        Commands::Run { file } => run_file(&file),
        Commands::Build { file } => build_file(&file),
        Commands::DumpIr { file } => dump_ir(&file),
        Commands::Pkg { command } => run_pkg_command(command),
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

/// Handle package management commands
fn run_pkg_command(pkg_cmd: PkgCommands) -> Result<()> {
    use lyra::modules::cli::PackageCommand;
    use lyra::modules::simple_cli::SimplePackageCli;
    
    // Convert clap command to our package command enum
    let command = match pkg_cmd {
        PkgCommands::Init { name, path } => PackageCommand::Init { name, path },
        PkgCommands::Build { release } => PackageCommand::Build { release },
        PkgCommands::Test { test } => PackageCommand::Test { test_name: test },
        PkgCommands::Install { package, version, dev } => PackageCommand::Install { package, version, dev },
        PkgCommands::Update { package } => PackageCommand::Update { package },
        PkgCommands::Remove { package } => PackageCommand::Remove { package },
        PkgCommands::Search { query, limit } => PackageCommand::Search { query, limit },
        PkgCommands::Info { package, version } => PackageCommand::Info { package, version },
        PkgCommands::List { outdated } => PackageCommand::List { outdated },
        PkgCommands::Tree { package, depth } => PackageCommand::Tree { package, depth },
        PkgCommands::Check => PackageCommand::Check,
        PkgCommands::Publish { registry, token } => PackageCommand::Publish { registry, token },
    };
    
    // Create and run simplified CLI (bypasses compilation issues)
    let rt = tokio::runtime::Runtime::new().map_err(|e| lyra::Error::Runtime {
        message: format!("Failed to create async runtime: {}", e),
    })?;
    
    let mut cli = SimplePackageCli::new();
    
    rt.block_on(async {
        cli.execute(command).await.map_err(|e| lyra::Error::Runtime {
            message: e,
        })
    })
}

/// Run the interactive REPL
fn run_repl() -> Result<()> {
    use colored::*;
    
    println!("{}", "Lyra Interactive Symbolic Computation".bright_cyan().bold());
    println!("{} {}", "Version".dimmed(), env!("CARGO_PKG_VERSION").bright_green());
    println!("Type {} to evaluate, {} for help, or {} to quit.", 
        "expressions".bright_yellow(), 
        "%help".bright_blue(), 
        "exit".bright_red()
    );
    println!();

    // Initialize the REPL engine with configuration system
    let mut repl_engine = ReplEngine::new().map_err(|e| lyra::Error::Runtime {
        message: format!("Failed to initialize REPL: {}", e),
    })?;
    
    // Show configuration status
    let config = repl_engine.get_config();
    if config.repl.show_timing {
        println!("{}", "Timing enabled".bright_green());
    }
    if config.repl.show_performance {
        println!("{}", "Performance monitoring enabled".bright_green());
    }
    
    // Show configuration paths (for debugging/first run)
    match lyra::repl::config::ReplConfig::get_config_file_path() {
        Ok(path) => {
            if !path.exists() {
                println!("{} Configuration file created at: {}", 
                    "Info:".bright_blue(), 
                    path.display().to_string().dimmed()
                );
            }
        }
        Err(_) => {} // Ignore error for now
    }

    // Create rustyline editor with enhanced helper
    let editor_config = Config::builder()
        .completion_type(rustyline::CompletionType::List)
        .build();
    
    // Create standard library for hint system
    let stdlib = StandardLibrary::new();
    
    // Create enhanced helper with all quality-of-life features
    let enhanced_helper = EnhancedLyraHelper::new(
        repl_engine.get_config().clone(),
        repl_engine.create_shared_completer(),
        &stdlib
    ).map_err(|e| lyra::Error::Runtime {
        message: format!("Failed to create enhanced helper: {}", e),
    })?;
    
    let mut rl = Editor::with_config(editor_config).map_err(|e| lyra::Error::Runtime {
        message: e.to_string(),
    })?;
    rl.set_helper(Some(enhanced_helper));
    let mut line_number = 1;

    loop {
        use colored::*;
        
        // Determine prompt based on multiline state
        let prompt = if repl_engine.has_multiline_input() {
            "   ...> ".to_string()
        } else {
            format!("{}[{}]> ", "lyra".bright_cyan(), line_number.to_string().bright_white())
        };

        match rl.readline(&prompt) {
            Ok(line) => {
                let line = line.trim();

                // Handle empty lines
                if line.is_empty() {
                    if repl_engine.has_multiline_input() {
                        // Empty line in multiline mode - try to complete
                        if repl_engine.add_multiline_input("") {
                            // Expression is complete, evaluate it
                            match repl_engine.evaluate_multiline() {
                                Ok(result) => {
                                    println!("{} {}", 
                                        format!("Out[{}]=", line_number).bright_blue(), 
                                        colorize_output(&result.result)
                                    );
                                    
                                    if let Some(perf_info) = result.performance_info {
                                        if !perf_info.is_empty() && result.execution_time.as_millis() > 10 {
                                            println!("{} {}", "Performance:".bright_black(), perf_info.dimmed());
                                        }
                                    }
                                    line_number += 1;
                                }
                                Err(e) => {
                                    println!("{} {}", "✗ Error:".bright_red().bold(), e.to_string().red());
                                    repl_engine.clear_multiline_buffer();
                                }
                            }
                        }
                        // Otherwise continue multiline input
                    }
                    continue;
                }

                // Handle exit commands
                if line == "exit" || line == "quit" {
                    println!("{}", "Goodbye! 👋".bright_green());
                    break;
                }

                // Handle help commands
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

                // Add to history only if not in multiline mode or completing
                if !repl_engine.has_multiline_input() {
                    let _ = rl.add_history_entry(line);
                }

                // Handle multiline input
                if repl_engine.has_multiline_input() {
                    // Already in multiline mode, add this line
                    if repl_engine.add_multiline_input(line) {
                        // Expression is complete, evaluate it
                        let full_input = repl_engine.get_multiline_input();
                        let _ = rl.add_history_entry(&full_input); // Add complete expression to history
                        
                        match repl_engine.evaluate_multiline() {
                            Ok(result) => {
                                println!("{} {}", 
                                    format!("Out[{}]=", line_number).bright_blue(), 
                                    colorize_output(&result.result)
                                );
                                
                                if let Some(perf_info) = result.performance_info {
                                    if !perf_info.is_empty() && result.execution_time.as_millis() > 10 {
                                        println!("{} {}", "Performance:".bright_black(), perf_info.dimmed());
                                    }
                                }
                                line_number += 1;
                            }
                            Err(e) => {
                                println!("{} {}", "✗ Error:".bright_red().bold(), e.to_string().red());
                                repl_engine.clear_multiline_buffer();
                            }
                        }
                    } else {
                        // Show hint if available
                        if let Some(hint) = repl_engine.get_multiline_hint() {
                            println!("{} {}", "Hint:".bright_yellow(), hint.dimmed());
                        }
                    }
                } else {
                    // Not in multiline mode, try to start multiline or evaluate immediately
                    if repl_engine.add_multiline_input(line) {
                        // Complete expression, evaluate immediately
                        match repl_engine.evaluate_multiline() {
                            Ok(result) => {
                                println!("{} {}", 
                                    format!("Out[{}]=", line_number).bright_blue(), 
                                    colorize_output(&result.result)
                                );
                                
                                if let Some(perf_info) = result.performance_info {
                                    if !perf_info.is_empty() && result.execution_time.as_millis() > 10 {
                                        println!("{} {}", "Performance:".bright_black(), perf_info.dimmed());
                                    }
                                }
                                line_number += 1;
                            }
                            Err(e) => {
                                println!("{} {}", "✗ Error:".bright_red().bold(), e.to_string().red());
                            }
                        }
                    } else {
                        // Incomplete expression, entered multiline mode
                        if let Some(hint) = repl_engine.get_multiline_hint() {
                            println!("{} {}", "Hint:".bright_yellow(), hint.dimmed());
                        }
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                if repl_engine.has_multiline_input() {
                    println!("^C");
                    repl_engine.clear_multiline_buffer();
                    println!("{}", "Multiline input cancelled".dimmed());
                } else {
                    println!("^C");
                }
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

    // Save history and configuration on exit
    if let Err(e) = repl_engine.shutdown() {
        eprintln!("Warning: Failed to save session data: {}", e);
    }

    Ok(())
}

/// Run a script file
fn run_file(file_path: &PathBuf) -> Result<()> {
    let source = fs::read_to_string(file_path)?;

    // Parse source code to AST
    let mut parser = LyraParser::from_source(&source)?;
    let statements = parser.parse()?;

    if statements.is_empty() {
        eprintln!("No statements to execute in {}", file_path.display());
        return Ok(());
    }

    // Execute each statement and print results for non-assignments
    for (i, statement) in statements.iter().enumerate() {
        match Compiler::eval(statement) {
            Ok(result) => {
                // Print the result (in a real implementation, we might want to 
                // suppress output for assignment statements, but for now print all results)
                println!("{}", format_value(&result));
            }
            Err(e) => {
                let error = lyra::Error::Compilation {
                    message: format!("Error in statement {}: {}", i + 1, e),
                };
                eprintln!("Error executing statement {} in {}:", i + 1, file_path.display());
                eprintln!("{}", lyra::error::format_error_with_context(&error, &source));
                return Err(error);
            }
        }
    }

    Ok(())
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

/// Colorize output for pretty display in REPL
fn colorize_output(output: &str) -> String {
    use colored::*;
    
    // Try to detect the type of output and colorize appropriately
    if output.starts_with('"') && output.ends_with('"') {
        // String values
        output.bright_green().to_string()
    } else if output.parse::<i64>().is_ok() {
        // Integer values
        output.bright_blue().to_string()
    } else if output.parse::<f64>().is_ok() {
        // Float values
        output.bright_cyan().to_string()
    } else if output.starts_with('{') && output.ends_with('}') {
        // List/Array values
        output.bright_magenta().to_string()
    } else if output == "True" || output == "False" {
        // Boolean values
        if output == "True" {
            output.bright_green().to_string()
        } else {
            output.bright_red().to_string()
        }
    } else if output.contains('[') || output.starts_with("Function") {
        // Function calls or function definitions
        output.bright_yellow().to_string()
    } else {
        // Default: symbols and other values
        output.normal().to_string()
    }
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
        lyra::vm::Value::Missing => "Missing[]".to_string(),
        lyra::vm::Value::LyObj(obj) => {
            // Enhanced display for tensor foreign objects
            if obj.type_name() == "Tensor" {
                // Try to get tensor info via method calls
                if let Ok(dims_result) = obj.call_method("Dimensions", &[]) {
                    if let lyra::vm::Value::List(dims) = dims_result {
                        let shape: Vec<String> = dims.iter().map(|v| format!("{}", format_value(v))).collect();
                        if let Ok(len_result) = obj.call_method("Length", &[]) {
                            if let lyra::vm::Value::Integer(len) = len_result {
                                return format!("Tensor[shape: [{}], elements: {}]", shape.join(", "), len);
                            }
                        }
                    }
                }
            }
            format!("{}[...]", obj.type_name())
        }
        lyra::vm::Value::Quote(expr) => {
            format!("Hold[{:?}]", expr)
        }
        lyra::vm::Value::Pattern(pattern) => {
            format!("{}", pattern)
        }
        lyra::vm::Value::Rule { lhs, rhs } => {
            format!("{} -> {}", format_value(lhs), format_value(rhs))
        }
        lyra::vm::Value::Object(_) => {
            "Object[]".to_string()
        }
        lyra::vm::Value::PureFunction { .. } => {
            "PureFunction[...]".to_string()
        }
        lyra::vm::Value::Slot { .. } => {
            "Slot[...]".to_string()
        }
    }
}

/// Show REPL help information
fn show_repl_help() {
    use colored::*;
    
    println!("{}", "Lyra Interactive REPL Help".bright_cyan().bold());
    println!("{}", "═══════════════════════════".bright_cyan());
    println!();
    
    println!("{}", "Built-in Commands:".bright_yellow().bold());
    println!("  {} - Show this help message", "help, ?".bright_blue());
    println!("  {} - List all available functions", "functions".bright_blue());
    println!("  {} - Show example expressions", "examples".bright_blue());
    println!("  {} - Exit the REPL", "exit, quit".bright_red());
    println!();
    
    println!("{}", "Meta Commands:".bright_yellow().bold());
    println!("  {} - Show detailed REPL help", "%help".bright_magenta());
    println!("  {} - Show command history", "%history".bright_magenta());
    println!("  {} - Show performance statistics", "%perf".bright_magenta());
    println!("  {} - Clear session (variables, history)", "%clear".bright_magenta());
    println!("  {} - Show defined variables", "%vars".bright_magenta());
    println!("  {} - Enable/disable execution timing", "%timing on/off".bright_magenta());
    println!("  {} - Enable/disable performance info", "%perf on/off".bright_magenta());
    println!("  {} - Show current configuration", "%config".bright_magenta());
    println!("  {} - Show history settings", "%history-settings".bright_magenta());
    println!();
    
    println!("{}", "Syntax:".bright_yellow().bold());
    println!("  Variables:      {}", "x = 5, name = \"Alice\"".green());
    println!("  Functions:      {}", "f[x_] := x^2".green());
    println!("  Function calls: {}", "f[x, y]".green());
    println!("  Lists:          {}", "{1, 2, 3}".green());
    println!("  Arithmetic:     {}", "2 + 3 * 4".green());
    println!("  Strings:        {}", "\"Hello, World!\"".green());
    println!("  Rules:          {}", "x -> x^2".green());
    println!("  Replacement:    {}", "expr /. rule".green());
    println!();
    
    println!("{}", "Quick Start:".bright_yellow().bold());
    println!("  {}", "x = 5".bright_white());
    println!("  {}", "Sin[Pi/2]".bright_white());
    println!("  {}", "Length[{1, 2, 3, 4}]".bright_white());
    println!();
}

/// Show available functions
fn show_available_functions() {
    use colored::*;
    
    println!("{}", "Available Functions".bright_cyan().bold());
    println!("{}", "═══════════════════".bright_cyan());
    println!();
    
    println!("{}", "Mathematical Functions:".bright_yellow().bold());
    println!("  {} - Trigonometric functions", "Sin[x], Cos[x], Tan[x]".bright_blue());
    println!("  {} - Exponential and logarithmic", "Exp[x], Log[x], Sqrt[x]".bright_blue());
    println!("  {} - Power and modular arithmetic", "Power[x, y], Modulo[x, y]".bright_blue());
    println!("  {} - Basic arithmetic", "+, -, *, /, ^".bright_blue());
    println!();
    
    println!("{}", "List Operations:".bright_yellow().bold());
    println!("  {} - Get length of a list", "Length[list]".bright_green());
    println!("  {} - Get first element", "Head[list]".bright_green());
    println!("  {} - Get all but first element", "Tail[list]".bright_green());
    println!("  {} - Add element to list", "Append[list, elem]".bright_green());
    println!("  {} - Flatten nested lists", "Flatten[list]".bright_green());
    println!("  {} - Apply function to each element", "Map[func, list]".bright_green());
    println!();
    
    println!("{}", "String Operations:".bright_yellow().bold());
    println!("  {} - Get string length", "StringLength[str]".bright_magenta());
    println!("  {} - Concatenate strings", "StringJoin[str1, str2, ...]".bright_magenta());
    println!("  {} - Take first n characters", "StringTake[str, n]".bright_magenta());
    println!("  {} - Drop first n characters", "StringDrop[str, n]".bright_magenta());
    println!();
    
    println!("{}", "Constants:".bright_yellow().bold());
    println!("  {} - Mathematical constants", "Pi, E".bright_red());
    println!("  {} - Boolean values", "True, False".bright_red());
    println!("  {} - Special values", "Missing, Undefined, Infinity".bright_red());
    println!();
    
    println!("Type {} for usage examples", "examples".bright_white().bold());
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
