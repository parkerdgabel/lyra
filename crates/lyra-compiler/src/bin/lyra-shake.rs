use anyhow::{anyhow, Result};
use lyra_compiler::manifest::{Capability, Manifest};
use lyra_compiler::registry::{capabilities_for, features_for};
use lyra_compiler::Analyzer;
use std::{fs, path::PathBuf};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() || matches!(args[0].as_str(), "-h" | "--help") {
        print_help();
        return Ok(());
    }
    let cmd = args.remove(0);
    match cmd.as_str() {
        "analyze" => cmd_analyze(args),
        "build" => cmd_build(args),
        _ => {
            eprintln!("Unknown command: {}", cmd);
            print_help();
            Ok(())
        }
    }
}

fn cmd_analyze(mut args: Vec<String>) -> Result<()> {
    let mut out_path: Option<PathBuf> = None;
    let mut files: Vec<PathBuf> = vec![];
    while !args.is_empty() {
        let a = args.remove(0);
        if a == "-o" || a == "--output" {
            out_path = Some(PathBuf::from(args.remove(0)));
            continue;
        }
        if a.starts_with('-') {
            return Err(anyhow!("Unknown flag: {}", a));
        }
        files.push(PathBuf::from(a));
    }
    if files.is_empty() {
        return Err(anyhow!("No input files provided"));
    }

    let analyzer = Analyzer::new();
    let res = analyzer.analyze_files(&files)?;
    let features = features_for(&res.heads);
    let capabilities = capabilities_for(&res.heads);
    let mut manifest = Manifest::new();
    manifest.symbols = res.heads.into_iter().collect();
    manifest.symbols.sort();
    manifest.features = features.into_iter().collect();
    manifest.features.sort();
    manifest.capabilities = capabilities
        .into_iter()
        .map(|c| match c {
            "net" => Capability::Net,
            "fs" => Capability::Fs,
            "db" => Capability::Db,
            "gpu" => Capability::Gpu,
            "process" => Capability::Process,
            _ => Capability::Time,
        })
        .collect();
    manifest.capabilities.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));

    let json = serde_json::to_string_pretty(&manifest)?;
    if let Some(p) = out_path {
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(p, json)?;
    } else {
        println!("{}", json);
    }
    Ok(())
}

fn print_help() {
    eprintln!("lyra-shake — tree shaking utilities\n\nUSAGE:\n  lyra-shake analyze [files...] [-o manifest.json]\n  lyra-shake build [files...] [--features-extra a,b] [--release] [--no-size-opt]\n\nCommands:\n  analyze   Analyze entry files and emit symbols/features/capabilities manifest\n  build     Analyze and build a minimal lyra-runner with only required features\n\nFlags:\n  --features-extra   Comma-separated extra cargo features to include in build\n  --release          Build in release mode\n  --no-size-opt      Disable size optimization flags (LTO/strip/opt-level=z)\n");
}

fn cmd_build(mut args: Vec<String>) -> Result<()> {
    let mut files: Vec<PathBuf> = vec![];
    let mut features_extra: Vec<String> = vec![];
    let mut release = false;
    let mut size_opt = true;
    while !args.is_empty() {
        let a = args.remove(0);
        if a == "--features-extra" {
            if !args.is_empty() {
                features_extra = args
                    .remove(0)
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .collect();
            }
            continue;
        }
        if a == "--release" {
            release = true;
            continue;
        }
        if a == "--no-size-opt" {
            size_opt = false;
            continue;
        }
        if a.starts_with('-') {
            return Err(anyhow!("Unknown flag: {}", a));
        }
        files.push(PathBuf::from(a));
    }
    if files.is_empty() {
        return Err(anyhow!("No input files provided"));
    }
    let analyzer = Analyzer::new();
    let res = analyzer.analyze_files(&files)?;
    // Write keep_symbols.in.rs
    let keep_path = PathBuf::from("crates/lyra-runner/src/keep_symbols.in.rs");
    let mut out = String::new();
    out.push_str(
        "pub static KEEP: &[&str] = &[
",
    );
    let mut syms: Vec<_> = res.heads.iter().cloned().collect::<Vec<_>>();
    syms.sort();
    for s in syms {
        out.push_str("    \"");
        out.push_str(&s);
        out.push_str("\",\n");
    }
    out.push_str(
        "];
",
    );
    std::fs::write(&keep_path, out)?;
    // Compute features
    let mut feats: Vec<String> = features_for(&res.heads).into_iter().collect();
    feats.sort();
    feats.extend(features_extra);
    feats.sort();
    feats.dedup();
    // Build lyra-runner
    let mut cmd = std::process::Command::new("cargo");
    cmd.arg("build").arg("-p").arg("lyra-runner").arg("--no-default-features");
    if release {
        cmd.arg("--release");
    }
    if !feats.is_empty() {
        cmd.arg("--features").arg(feats.join(","));
    }
    if size_opt {
        // Favor minimal binary size: LTO, fewer codegen units, strip symbols, and size-focused opt level.
        // Merge with existing RUSTFLAGS if set.
        let mut flags = std::env::var("RUSTFLAGS").unwrap_or_default();
        let extra = " -C codegen-units=1 -C strip=symbols -C opt-level=z";
        flags.push_str(extra);
        cmd.env("RUSTFLAGS", flags.trim());
    }
    let status = cmd.status()?;
    if !status.success() {
        return Err(anyhow!("cargo build failed"));
    }
    println!(
        "Built lyra-runner. Features: {}",
        if feats.is_empty() { String::from("(none)") } else { feats.join(",") }
    );
    Ok(())
}
