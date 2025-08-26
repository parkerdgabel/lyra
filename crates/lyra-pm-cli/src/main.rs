use lyra_core::pretty::format_value;
use lyra_core::value::Value;
use lyra_runtime::Evaluator;

#[derive(Default, Clone, Copy)]
struct Opts { json: bool, pretty: bool }

fn main() {
    let mut ev = Evaluator::new();
    lyra_stdlib::register_all(&mut ev);

    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut opts = Opts::default();
    // Extract global flags
    let mut i = 0;
    let mut filtered: Vec<String> = Vec::new();
    while i < args.len() {
        match args[i].as_str() {
            "--json" => opts.json = true,
            "--pretty" => opts.pretty = true,
            _ => filtered.push(args[i].clone()),
        }
        i+=1;
    }
    args = filtered;
    if args.is_empty() || args.iter().any(|a| a=="-h"||a=="--help") {
        print_help();
        return;
    }

    let cmd = args[0].as_str();
    match cmd {
        "new" => {
            if args.len() < 2 { eprintln!("usage: lyra-pm new <name-or-path>"); std::process::exit(2); }
            let expr = format!("NewPackage[\"{}\"]", args[1]);
            run_print(&mut ev, &expr, opts);
        }
        "new-module" => {
            if args.len() < 3 { eprintln!("usage: lyra-pm new-module <pkg-path> <name>"); std::process::exit(2); }
            let expr = format!("NewModule[\"{}\", \"{}\"]", args[1], args[2]);
            run_print(&mut ev, &expr, opts);
        }
        "list" => run_print(&mut ev, "ListInstalledPackages[]", opts),
        "info" => {
            if args.len()<2 { eprintln!("usage: lyra-pm info <name>"); std::process::exit(2); }
            run_print(&mut ev, &format!("PackageInfo[\"{}\"]", args[1]), opts);
        }
        "path" => run_print(&mut ev, "ModulePath[]", opts),
        "loaded" => run_print(&mut ev, "LoadedPackages[]", opts),
        "imports" => {
            if args.len()<2 { eprintln!("usage: lyra-pm imports <name>"); std::process::exit(2); }
            run_print(&mut ev, &format!("ImportedSymbols[\"{}\"]", args[1]), opts);
        }
        "exports" => {
            if args.len()<2 { eprintln!("usage: lyra-pm exports <name>"); std::process::exit(2); }
            run_print(&mut ev, &format!("PackageExports[\"{}\"]", args[1]), opts);
        }
        "register-exports" => {
            if args.len()<3 { eprintln!("usage: lyra-pm register-exports <name> <sym1[,sym2,...]>"); std::process::exit(2); }
            let name = &args[1];
            let items: Vec<String> = args[2]
                .split(',')
                .filter(|s| !s.trim().is_empty())
                .map(|s| format!("\"{}\"", s.trim()))
                .collect();
            let expr = format!(
                "RegisterExports[<|name->\"{}\", exports->{{{}}}|>]",
                name,
                items.join(",")
            );
            run_print(&mut ev, &expr, opts);
        }
        "set-path" => {
            if args.len()<2 { eprintln!("usage: lyra-pm set-path <path1>[,<path2>...]"); std::process::exit(2); }
            let parts: Vec<String> = args[1].split(',').map(|s| format!("\"{}\"", s)).collect();
            run_print(&mut ev, &format!("SetModulePath[{{{}}}]", parts.join(",")), opts);
        }
        "using" => {
            if args.len()<2 { eprintln!("usage: lyra-pm using <name> [--all] [--import a,b] [--except x,y]"); std::process::exit(2); }
            let name = &args[1];
            let mut import: Option<String> = None;
            let mut except: Option<String> = None;
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--import" => { i+=1; if i<args.len() { import = Some(args[i].clone()); } }
                    "--except" => { i+=1; if i<args.len() { except = Some(args[i].clone()); } }
                    _ => {}
                }
                i+=1;
            }
            let import_opts_str = build_import_assoc(import.as_deref(), except.as_deref());
            let expr = if import_opts_str.is_empty() {
                format!("Using[\"{}\"]", name)
            } else {
                format!("Using[\"{}\", {}]", name, import_opts_str)
            };
            run_print(&mut ev, &expr, opts);
            // Show effective imports
            let q = format!("ImportedSymbols[\"{}\"]", name);
            let out = eval(&mut ev, &q);
            eprintln!("imports: {}", format_value(&out));
        }
        // Stubs pass-through
        "build" => run_print(&mut ev, "BuildPackage[]", opts),
        "test" => run_print(&mut ev, "TestPackage[]", opts),
        "lint" => run_print(&mut ev, "LintPackage[]", opts),
        "pack" => run_print(&mut ev, "PackPackage[]", opts),
        "sbom" => run_print(&mut ev, "GenerateSBOM[]", opts),
        "sign" => run_print(&mut ev, "SignPackage[]", opts),
        "publish" => run_print(&mut ev, "PublishPackage[]", opts),
        "install" => run_print(&mut ev, "InstallPackage[]", opts),
        "update" => run_print(&mut ev, "UpdatePackage[]", opts),
        "remove" => run_print(&mut ev, "RemovePackage[]", opts),
        "login" => run_print(&mut ev, "LoginRegistry[]", opts),
        "logout" => run_print(&mut ev, "LogoutRegistry[]", opts),
        "whoami" => run_print(&mut ev, "WhoAmI[]", opts),
        "audit" => run_print(&mut ev, "PackageAudit[]", opts),
        "verify" => run_print(&mut ev, "PackageVerify[]", opts),
        _ => { eprintln!("Unknown command: {}", cmd); print_help(); std::process::exit(2); }
    }
}

fn run_print(ev: &mut Evaluator, expr_src: &str, opts: Opts) {
    let mut p = lyra_parser::Parser::from_source(expr_src);
    match p.parse_all() {
        Ok(mut es) => {
            if let Some(e) = es.pop() {
                let out = ev.eval(e);
                if opts.json {
                    // Use ToJson with Pretty option
                    let pretty = if opts.pretty { "True" } else { "False" };
                    let expr = format!("ToJson[#, <|Pretty->{}|>]", pretty).replace("#", &format_value(&out));
                    let json = eval(ev, &expr);
                    println!("{}", format_value(&json));
                } else {
                    println!("{}", format_value(&out));
                }
            }
        }
        Err(e) => { eprintln!("parse error: {:?}", e); std::process::exit(2); }
    }
}

fn eval(ev: &mut Evaluator, expr_src: &str) -> Value {
    let mut p = lyra_parser::Parser::from_source(expr_src);
    match p.parse_all() {
        Ok(mut es) => {
            if let Some(e) = es.pop() { ev.eval(e) } else { Value::Symbol("Null".into()) }
        }
        Err(_) => Value::Symbol("Null".into()),
    }
}

fn build_import_assoc(import: Option<&str>, except: Option<&str>) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(s) = import {
        if s.eq_ignore_ascii_case("all") { parts.push("Import->All".into()); }
        else {
            let syms: Vec<String> = s.split(',').filter(|t| !t.is_empty()).map(|t| format!("\"{}\"", t)).collect();
            if !syms.is_empty() { parts.push(format!("Import->{{{}}}", syms.join(","))); }
        }
    }
    if let Some(s) = except {
        let syms: Vec<String> = s.split(',').filter(|t| !t.is_empty()).map(|t| format!("\"{}\"", t)).collect();
        if !syms.is_empty() { parts.push(format!("Except->{{{}}}", syms.join(","))); }
    }
    if parts.is_empty() { String::new() } else { format!("<|{}|>", parts.join(",")) }
}

fn print_help() {
    eprintln!("lyra-pm (skeleton)\nUSAGE:\n  lyra-pm [--json] [--pretty] <command> [args]\n\n  Commands:\n    new <name-or-path>\n    new-module <pkg-path> <name>\n    list | info <name>\n    path | set-path <paths-comma-separated>\n    loaded | imports <name> | exports <name> | register-exports <name> <sym1[,sym2,...]>\n    using <name> [--all] [--import a,b] [--except x,y]\n    build|test|lint|pack|sbom|sign|publish|install|update|remove|login|logout|whoami|audit|verify\n");
}
