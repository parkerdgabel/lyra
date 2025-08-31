use lyra_runtime::{set_default_registrar, Evaluator};
use lyra_stdlib as stdlib;
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

// This file is generated once; the KEEP list is injected by lyra-compiler build command
// It should define: pub static KEEP: &[&str]
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/keep_symbols.in.rs"));

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let keep_only = args.iter().any(|a| a == "--keep-only");

    // Default: full stdlib. Opt-out with --keep-only for tree-shaken builds.
    if keep_only {
        set_default_registrar(|ev: &mut Evaluator| {
            let mut set: HashSet<&str> = HashSet::new();
            for &s in KEEP.iter() {
                set.insert(s);
            }
            stdlib::register_selected(ev, &set);
        });
    } else {
        set_default_registrar(stdlib::register_all);
    }
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!("lyra-runner
USAGE:
  lyra-runner [--keep-only] [--models mock|auto] --eval <expr>
  lyra-runner [--keep-only] [--models mock|auto] --file <file.lyra>

OPTIONS:
  --keep-only    Register only symbols listed in keep_symbols.in.rs (tree-shaken mode).\n    Default is full stdlib registration.
  --models       Model providers mode. Use 'mock' (default) for offline validation.
");
        return;
    }
    let mut eval_src: Option<String> = None;
    let mut file: Option<PathBuf> = None;
    let mut i = 0;
    let mut models_mode: Option<String> = None;
    while i < args.len() {
        match args[i].as_str() {
            "--keep-only" => { /* handled above; skip */ }
            "--eval" => {
                i += 1;
                if i < args.len() {
                    eval_src = Some(args[i].clone());
                }
            }
            "--file" => {
                i += 1;
                if i < args.len() {
                    file = Some(PathBuf::from(&args[i]));
                }
            }
            "--models" => {
                i += 1;
                if i < args.len() {
                    models_mode = Some(args[i].clone());
                }
            }
            _ => {}
        }
        i += 1;
    }

    let mut ev = Evaluator::new();
    // Ensure model mode is visible to stdlib (defaults to mock for offline safety)
    ev.set_env(
        "ModelsMode",
        lyra_core::value::Value::String(models_mode.unwrap_or_else(|| "mock".into())),
    );
    if let Some(src) = eval_src {
        run_src(&mut ev, &src, None);
    } else if let Some(p) = file {
        match fs::read_to_string(&p) {
            Ok(src) => run_src(&mut ev, &src, Some(&p)),
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        eprintln!("No input provided. Use --eval or --file.");
        std::process::exit(2);
    }
}

fn find_project_root(start: &std::path::Path) -> Option<std::path::PathBuf> {
    let mut p = Some(start);
    while let Some(cur) = p {
        let cand1 = cur.join("project.lyra");
        if std::path::Path::new(&cand1).exists() { return Some(cur.to_path_buf()); }
        p = cur.parent();
    }
    None
}

fn run_src(ev: &mut Evaluator, src: &str, file_path: Option<&std::path::Path>) {
    if let Some(p) = file_path {
        let abs = if p.is_absolute() {
            p.to_path_buf()
        } else {
            std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")).join(p)
        };
        let dir = abs.parent().unwrap_or_else(|| std::path::Path::new(".")).to_path_buf();
        ev.set_env(
            "CurrentFile",
            lyra_core::value::Value::String(abs.to_string_lossy().to_string()),
        );
        ev.set_env(
            "CurrentDir",
            lyra_core::value::Value::String(dir.to_string_lossy().to_string()),
        );
        let root = find_project_root(&dir).unwrap_or(dir);
        ev.set_env(
            "ProjectRoot",
            lyra_core::value::Value::String(root.to_string_lossy().to_string()),
        );
    } else {
        let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        ev.set_env(
            "CurrentDir",
            lyra_core::value::Value::String(cwd.to_string_lossy().to_string()),
        );
    }
    let mut parser = lyra_parser::Parser::from_source(src);
    match parser.parse_all() {
        Ok(exprs) => {
            for e in exprs {
                let out = ev.eval(e);
                println!("{}", lyra_core::pretty::format_value(&out));
            }
        }
        Err(e) => {
            eprintln!("parse error: {:?}", e);
            std::process::exit(2);
        }
    }
}
