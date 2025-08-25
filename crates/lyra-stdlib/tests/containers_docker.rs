#![cfg(feature = "containers_docker")]
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;
use lyra_core::value::Value;

fn eval_str(ev: &mut Evaluator, src: &str) -> Value {
    let mut p = lyra_parser::Parser::from_source(src);
    let exprs = p.parse_all().unwrap();
    let mut last = Value::Symbol("Null".into());
    for e in exprs { last = ev.eval(e); }
    last
}

#[test]
fn docker_pull_run_list_inspect() {
    // This test expects a local Docker daemon (DOCKER_HOST or unix socket)
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);

    // Connect to Docker
    let rt = eval_str(&mut ev, "ConnectContainers[\"docker://\"]");

    // Pull alpine image
    let ok = eval_str(&mut ev, &format!("PullImage[{}, \"alpine:3.19\"]", lyra_core::pretty::format_value(&rt)));
    assert!(matches!(ok, Value::Boolean(true)));

    // Run echo container
    let run = eval_str(&mut ev, &format!(
        "RunContainer[{}, <|Image->\"alpine:3.19\", Cmd->{\"sh\",\"-c\",\"echo hi\"}|>]",
        lyra_core::pretty::format_value(&rt)));
    // Ensure we got a Container handle
    let run_s = lyra_core::pretty::format_value(&run);
    assert!(run_s.contains("\"__type\" -> \"Container\""));

    // Inspect the container (should exist, even if exited)
    let info = eval_str(&mut ev, &format!("InspectContainer[{}, {}]", lyra_core::pretty::format_value(&rt), lyra_core::pretty::format_value(&run)));
    let info_s = lyra_core::pretty::format_value(&info);
    assert!(info_s.contains("\"image\" -> \"alpine:3.19\"") || info_s.contains("\"image\" -> \"alpine\""));

    // List containers dataset
    let ds = eval_str(&mut ev, &format!("ListContainers[{}]", lyra_core::pretty::format_value(&rt)));
    let ds_s = lyra_core::pretty::format_value(&ds);
    assert!(ds_s.contains("DatasetFromRows"));
}

