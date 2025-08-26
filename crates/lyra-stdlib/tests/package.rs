use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;

fn eval_str(ev: &mut Evaluator, src: &str) -> Value {
    let mut p = lyra_parser::Parser::from_source(src);
    let vals = p.parse_all().expect("parse");
    ev.eval(vals.into_iter().last().unwrap())
}

#[test]
fn new_package_and_using_path() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);

    // Create a temp dir
    let tmpdir = std::env::temp_dir().join(format!("lyra-pkg-test-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmpdir);

    // Set $PackagePath to the temp dir
    let set_cmd = format!("Set[$PackagePath, {{\"{}\"}}]", tmpdir.to_string_lossy());
    let _ = eval_str(&mut ev, &set_cmd);

    // Scaffold a package in tmpdir
    let pkg_path = tmpdir.join("demo_pkg");
    let new_cmd = format!("NewPackage[\"{}\"]", pkg_path.to_string_lossy());
    let res = eval_str(&mut ev, &new_cmd);
    match res {
        Value::Assoc(m) => {
            assert!(m.get("path").is_some());
        }
        _ => panic!("NewPackage did not return assoc: {:?}", res),
    }

    // Using should find it on $PackagePath
    let using_res = eval_str(&mut ev, "Using[\"demo_pkg\", <|Import->All|>]");
    assert!(matches!(using_res, Value::Boolean(true)), "Using failed: {:?}", using_res);

    // PackageInfo should reflect name and path
    let info = eval_str(&mut ev, "PackageInfo[\"demo_pkg\"]");
    match info {
        Value::Assoc(m) => {
            assert_eq!(m.get("name"), Some(&Value::String("demo_pkg".into())));
            assert!(matches!(m.get("path"), Some(Value::String(_))));
        }
        other => panic!("PackageInfo unexpected: {:?}", other),
    }
}

#[test]
fn module_path_set_get() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let tmp = std::env::temp_dir().join("lyra-pkg-path");
    let _ = std::fs::create_dir_all(&tmp);
    let cmd = format!("SetModulePath[{{\"{}\"}}]", tmp.to_string_lossy());
    let res = eval_str(&mut ev, &cmd);
    assert!(matches!(res, Value::Boolean(true)));
    let mp = eval_str(&mut ev, "ModulePath[]");
    match mp {
        Value::List(vs) => {
            assert!(!vs.is_empty());
        }
        other => panic!("ModulePath unexpected: {:?}", other),
    }
}

#[test]
fn pm_stubs_return_failure() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    for cmd in [
        "BuildPackage[]",
        "TestPackage[]",
        "LintPackage[]",
        "PackPackage[]",
        "GenerateSBOM[]",
        "SignPackage[]",
        "PublishPackage[]",
        "InstallPackage[]",
        "UpdatePackage[]",
        "RemovePackage[]",
        "LoginRegistry[]",
        "LogoutRegistry[]",
        "WhoAmI[]",
        "PackageAudit[]",
        "PackageVerify[]",
    ] {
        let out = eval_str(&mut ev, cmd);
        match out {
            Value::Assoc(m) => {
                assert_eq!(m.get("tag"), Some(&Value::String("PackageManagerNotAvailable".into())));
            }
            other => panic!("stub did not return failure: {} -> {:?}", cmd, other),
        }
    }
}

#[test]
fn import_and_unuse_soft_hide() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);

    // Prepare a temp package on path
    let tmpdir = std::env::temp_dir().join(format!("lyra-pkg-test-{}-unuse", std::process::id()));
    let _ = std::fs::create_dir_all(&tmpdir);
    let set_cmd = format!("Set[$PackagePath, {{\"{}\"}}]", tmpdir.to_string_lossy());
    let _ = eval_str(&mut ev, &set_cmd);
    let pkg_path = tmpdir.join("dummy_pkg");
    let new_cmd = format!("NewPackage[\"{}\"]", pkg_path.to_string_lossy());
    let _ = eval_str(&mut ev, &new_cmd);

    // Declare exports to resolve imports against
    let _ = eval_str(&mut ev, "RegisterExports[<|name->\"dummy_pkg\", exports->{\"A\", \"B\"}|>]");
    // Options are recorded and resolved against exports
    let _ = eval_str(&mut ev, "Using[\"dummy_pkg\", <|Import->{\"A\", \"B\"}, Except->{\"B\"}|>]");
    let imp = eval_str(&mut ev, "ImportedSymbols[\"dummy_pkg\"]");
    match imp {
        Value::List(vs) => assert_eq!(vs, vec![Value::String("A".into())]),
        other => panic!("unexpected imported: {:?}", other),
    }
    let _ = eval_str(&mut ev, "Unuse[\"dummy_pkg\"]");
    let after = eval_str(&mut ev, "ImportedSymbols[\"dummy_pkg\"]");
    match after {
        Value::List(vs) => assert!(vs.is_empty()),
        other => panic!("unexpected after unuse: {:?}", other),
    }
}
