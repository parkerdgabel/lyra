use clap::{Parser, Subcommand};
use lyra::Result;
use std::path::PathBuf;

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
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Repl => {
            println!("Lyra REPL - not yet implemented");
            Ok(())
        }
        Commands::Run { file } => {
            println!("Running file: {:?} - not yet implemented", file);
            Ok(())
        }
        Commands::Build { file } => {
            println!("Building file: {:?} - not yet implemented", file);
            Ok(())
        }
        Commands::DumpIr { file } => {
            println!("Dumping IR for file: {:?} - not yet implemented", file);
            Ok(())
        }
    }
}