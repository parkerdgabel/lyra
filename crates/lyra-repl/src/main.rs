use anyhow::Result;
use colored::Colorize;
use lyra_core::{format_value, Value};
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
                if line.trim().is_empty() { continue; }
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
