use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use lyra_runtime::{Evaluator, set_default_registrar};
use lyra_stdlib as stdlib;

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
            for &s in KEEP.iter() { set.insert(s); }
            stdlib::register_selected(ev, &set);
        });
    } else {
        set_default_registrar(stdlib::register_all);
    }
    if args.is_empty() || args.iter().any(|a| a=="-h"||a=="--help") {
        eprintln!("lyra-runner
USAGE:
  lyra-runner [--keep-only] --eval <expr>
  lyra-runner [--keep-only] --file <file.lyra>

OPTIONS:
  --keep-only    Register only symbols listed in keep_symbols.in.rs (tree-shaken mode).\n    Default is full stdlib registration.
");
        return;
    }
    let mut eval_src: Option<String> = None;
    let mut file: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--keep-only" => { /* handled above; skip */ },
            "--eval" => { i+=1; if i<args.len() { eval_src = Some(args[i].clone()); } },
            "--file" => { i+=1; if i<args.len() { file = Some(PathBuf::from(&args[i])); } },
            _ => {}
        }
        i+=1;
    }

    let mut ev = Evaluator::new();
    if let Some(src) = eval_src {
        run_src(&mut ev, &src);
    } else if let Some(p) = file {
        match fs::read_to_string(&p) {
            Ok(src) => run_src(&mut ev, &src),
            Err(e) => { eprintln!("error: {}", e); std::process::exit(1); }
        }
    } else {
        eprintln!("No input provided. Use --eval or --file.");
        std::process::exit(2);
    }
}

fn run_src(ev: &mut Evaluator, src: &str) {
    let mut parser = lyra_parser::Parser::from_source(src);
    match parser.parse_all() {
        Ok(exprs) => {
            for e in exprs { let out = ev.eval(e); println!("{}", lyra_core::pretty::format_value(&out)); }
        }
        Err(e) => { eprintln!("parse error: {:?}", e); std::process::exit(2); }
    }
}
