use lyra_runtime::Evaluator;

fn eval_str(code: &str) -> lyra_core::value::Value {
    let mut ev = Evaluator::new();
    lyra_stdlib::register_all(&mut ev);
    let mut p = lyra_parser::Parser::from_source(code);
    let exprs = p.parse_all().unwrap();
    let mut out = lyra_core::value::Value::Symbol("Null".into());
    for e in exprs { out = ev.eval(e); }
    out
}

#[test]
fn run_tasks_respects_dependencies() {
    let code = r#"
build = Task[<|"name"->"build","run"->"build"|>];
test = Task[<|"name"->"test","run"->"test","dependsOn"->{"build"}|>];
res = RunTasks[{build, test}];
Part[res, "results"]
"#;
    let v = eval_str(code);
    match v {
        lyra_core::value::Value::List(items) => {
            assert_eq!(items.len(), 2);
            // Ensure execution order is build then test
            let first_name = match &items[0] { lyra_core::value::Value::Assoc(m) => match m.get("name").unwrap() { lyra_core::value::Value::String(s) => s.clone(), lyra_core::value::Value::Symbol(s) => s.clone(), _ => String::new() }, _ => String::new() };
            let second_name = match &items[1] { lyra_core::value::Value::Assoc(m) => match m.get("name").unwrap() { lyra_core::value::Value::String(s) => s.clone(), lyra_core::value::Value::Symbol(s) => s.clone(), _ => String::new() }, _ => String::new() };
            assert_eq!(first_name, "build");
            assert_eq!(second_name, "test");
        }
        other => panic!("unexpected: {}", lyra_core::pretty::format_value(&other)),
    }
}

#[test]
fn explain_tasks_basic() {
    let code = r#"
a = Task[<|"name"->"a","run"->1|>];
b = Task[<|"name"->"b","run"->2,"dependsOn"->{"a"}|>];
e = ExplainTasks[{a,b}];
Part[e, "order"]
"#;
    let v = eval_str(code);
    match v { lyra_core::value::Value::List(order) => { assert_eq!(order.len(), 2); }, other => panic!("unexpected: {}", lyra_core::pretty::format_value(&other)) }
}

#[test]
fn hooks_are_invoked() {
    let code = r#"
Set[x, 0]; Set[y, 0]; Set[z, 0]; Set[w, 0];
a = Task[<|"name"->"a","run"->1|>];
b = Task[<|"name"->"b","run"->2|>];
RunTasks[{a,b}, <|
  "beforeAll"->Set[x, 1],
  "beforeEach"->Set[y, Plus[y, 1]],
  "afterEach"->Set[z, Plus[z, 1]],
  "afterAll"->Set[w, 1]
|>];
<|"x"->x, "y"->y, "z"->z, "w"->w|>
"#;
    let v = eval_str(code);
    match v { lyra_core::value::Value::Assoc(m) => {
        let gv = |k: &str| m.get(k).cloned().unwrap();
        assert_eq!(format!("{}", lyra_core::pretty::format_value(&gv("x"))), "1");
        assert_eq!(format!("{}", lyra_core::pretty::format_value(&gv("w"))), "1");
    }, other => panic!("unexpected: {}", lyra_core::pretty::format_value(&other)) }
}

// Note: retry/timeout behavior is validated manually for now; end-to-end tests to be added once we have test helpers for side-effect flows.
