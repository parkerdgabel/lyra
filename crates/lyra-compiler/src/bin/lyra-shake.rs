use std::{path::PathBuf, fs};
use anyhow::{Result, anyhow};
use lyra_compiler::{Analyzer};
use lyra_compiler::manifest::{Manifest, Capability};
use lyra_compiler::registry::{features_for, capabilities_for};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() || matches!(args[0].as_str(), "-h"|"--help") {
        print_help();
        return Ok(());
    }
    let cmd = args.remove(0);
    match cmd.as_str() {
        "analyze" => cmd_analyze(args),
        _ => { eprintln!("Unknown command: {}", cmd); print_help(); Ok(()) }
    }
}

fn cmd_analyze(mut args: Vec<String>) -> Result<()> {
    let mut out_path: Option<PathBuf> = None;
    let mut files: Vec<PathBuf> = vec![];
    while !args.is_empty() {
        let a = args.remove(0);
        if a=="-o" || a=="--output" { out_path = Some(PathBuf::from(args.remove(0))); continue; }
        if a.starts_with('-') { return Err(anyhow!("Unknown flag: {}", a)); }
        files.push(PathBuf::from(a));
    }
    if files.is_empty() { return Err(anyhow!("No input files provided")); }

    let analyzer = Analyzer::new();
    let res = analyzer.analyze_files(&files)?;
    let features = features_for(&res.heads);
    let capabilities = capabilities_for(&res.heads);
    let mut manifest = Manifest::new();
    manifest.symbols = res.heads.into_iter().collect();
    manifest.symbols.sort();
    manifest.features = features.into_iter().collect();
    manifest.features.sort();
    manifest.capabilities = capabilities.into_iter().map(|c| match c {
        "net"=>Capability::Net, "fs"=>Capability::Fs, "db"=>Capability::Db, "gpu"=>Capability::Gpu, "process"=>Capability::Process, _=>Capability::Time
    }).collect();
    manifest.capabilities.sort_by(|a,b| format!("{:?}", a).cmp(&format!("{:?}", b)));

    let json = serde_json::to_string_pretty(&manifest)?;
    if let Some(p) = out_path {
        if let Some(parent) = p.parent() { if !parent.as_os_str().is_empty() { fs::create_dir_all(parent)?; } }
        fs::write(p, json)?;
    } else {
        println!("{}", json);
    }
    Ok(())
}

fn print_help() {
    eprintln!("lyra-shake — tree shaking utilities\n\nUSAGE:\n  lyra-shake analyze [files...] [-o manifest.json]\n\nCommands:\n  analyze   Analyze entry files and emit symbols/features/capabilities manifest\n");
}
