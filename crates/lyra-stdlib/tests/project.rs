use lyra_runtime::Evaluator;
use lyra_core::value::Value;
use lyra_stdlib as stdlib;

fn eval_str(ev: &mut Evaluator, src: &str) -> Value {
    let mut p = lyra_parser::Parser::from_source(src);
    let vals = p.parse_all().expect("parse");
    ev.eval(vals.into_iter().last().unwrap())
}

#[test]
fn project_init_and_validate_ok() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // temp dir
    let root = std::env::temp_dir().join(format!("lyra-proj-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("mkdir");
    // Init
    let cmd = format!("ProjectInit[<|Name->\"demo\", Dir->\"{}\"|>]", root.to_string_lossy());
    let out = eval_str(&mut ev, &cmd);
    match out { Value::Assoc(m) => assert_eq!(m.get("ok"), Some(&Value::Boolean(true))), other => panic!("init unexpected: {:?}", other) }
    // Validate
    let cmd = format!("ProjectValidate[\"{}\"]", root.to_string_lossy());
    let v = eval_str(&mut ev, &cmd);
    match v { Value::Assoc(m) => {
        assert_eq!(m.get("ok"), Some(&Value::Boolean(true)));
    }, other => panic!("validate unexpected: {:?}", other) }
}

#[test]
fn project_validate_missing_module() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // temp dir with bad module path
    let root = std::env::temp_dir().join(format!("lyra-proj-bad-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("mkdir");
    let manifest = format!("Exported[{{}}];\n<|\n  \"Project\"-><|\"Name\"->\"bad\", \"Version\"->\"0.1.0\"|>,\n  \"Modules\"-><|\"main\"->ResolveRelative[\"src/missing.lyra\"]|>\n|>\n");
    std::fs::write(root.join("project.lyra"), manifest).expect("write manifest");
    let cmd = format!("ProjectValidate[\"{}\"]", root.to_string_lossy());
    let v = eval_str(&mut ev, &cmd);
    match v { Value::Assoc(m) => {
        assert_eq!(m.get("ok"), Some(&Value::Boolean(false)));
    }, other => panic!("validate unexpected: {:?}", other) }
}

