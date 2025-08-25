use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use lyra_runtime::{Evaluator, set_default_registrar};

// This file is generated once; the KEEP list is injected by lyra-compiler build command
// It should define: pub static KEEP: &[&str]
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/keep_symbols.in.rs"));

fn main() {
    // Install default registrar that registers only selected symbols
    set_default_registrar(|ev: &mut Evaluator| {
        let mut set: HashSet<&str> = HashSet::new();
        for &s in KEEP.iter() { set.insert(s); }
        lyra_stdlib::register_selected(ev, &set);
    });

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a=="-h"||a=="--help") {
        eprintln!("lyra-runner minimal
USAGE:
  lyra-runner --eval <expr>
  lyra-runner --file <file.lyra>
");
        return;
    }
    let mut eval_src: Option<String> = None;
    let mut file: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
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
