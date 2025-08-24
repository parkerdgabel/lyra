use anyhow::Result;
use colored::Colorize;
use lyra_core::format_value;
use lyra_runtime::Evaluator;
use lyra_parser::Parser;
use rustyline::{Editor, error::ReadlineError};
use rustyline::history::DefaultHistory;

fn main() -> Result<()> {
    let mut rl = Editor::<(), DefaultHistory>::new()?;
    println!("{}", "Lyra REPL (prototype)".bright_yellow().bold());
    let mut ev = Evaluator::new();
    loop {
        match rl.readline("> ") {
            Ok(line) => {
                let _ = rl.add_history_entry(line.as_str());
                let trimmed = line.trim();
                if trimmed.is_empty() { continue; }
                // Simple help system: ?help or ?Symbol
                if trimmed.starts_with('?') {
                    handle_help(trimmed);
                    continue;
                }
                let mut p = Parser::from_source(&line);
                match p.parse_all() {
                    Ok(values) => {
                        for v in values {
                            let out = ev.eval(v);
                            println!("{}", format_value(&out));
                        }
                    }
                    Err(e) => eprintln!("{} {}", "Error:".red().bold(), e),
                }
            }
            Err(ReadlineError::Interrupted) => { println!("^C"); continue; }
            Err(ReadlineError::Eof) => { println!("^D"); break; }
            Err(e) => { eprintln!("readline error: {e}"); break; }
        }
    }
    Ok(())
}

fn handle_help(q: &str) {
    let topic = q.trim_start_matches('?').trim();
    if topic.is_empty() || topic.eq_ignore_ascii_case("help") {
        let header = format!("{}", "Lyra REPL help".bright_green().bold());
        let body = "  - ?help: show this help\n  - ?Symbol: show a short description (e.g., ?Plus)\n  - Expressions use f[x, y], {a, b}, <|k->v|>\n  - Try: Explain[Plus[1, 2]] or Schema[<|\"a\"->1|>]";
        println!("{}\n{}", header, body);
        return;
    }
    let desc = builtin_help(topic);
    println!("{} {} — {}", "Help:".cyan().bold(), topic, desc);
}

fn builtin_help(sym: &str) -> &'static str {
    match sym {
        "Plus" => "Add numbers; Listable, Flat, Orderless.",
        "Times" => "Multiply numbers; Listable, Flat, Orderless.",
        "Minus" => "Subtract or unary negate.",
        "Divide" => "Divide two numbers.",
        "Power" => "Exponentiation (right-associative in parser).",
        "Map" => "Map a function over a list: Map[f, {..}].",
        "Replace" => "Replace first match by rule(s).",
        "ReplaceAll" => "Replace all matches by rule(s).",
        "ReplaceFirst" => "Replace first element(s) matching pattern.",
        "Set" => "Assignment: Set[symbol, value].",
        "With" => "Lexically bind symbols within a body.",
        "Schema" => "Return a minimal schema for a value/association.",
        "Explain" => "Explain evaluation; stub returns minimal trace info.",
        _ => "No documentation yet.",
    }
}
