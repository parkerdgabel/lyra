use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_runtime::attrs::Attributes;
use std::collections::HashMap;
use std::sync::{OnceLock, Mutex};
use crate::register_if;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

// Backend-agnostic runtime handle and registries (mock-only behavior for now)
#[derive(Clone)]
enum RuntimeKind { Mock, Docker, Podman, Containerd, K8s }

#[derive(Clone)]
struct RuntimeState {
    dsn: String,
    kind: RuntimeKind,
    // minimal in-memory state for mocking
    images: Vec<String>,
    containers: HashMap<i64, HashMap<String, Value>>, // id -> metadata assoc
    next_container_id: i64,
}

static RT_REG: OnceLock<Mutex<HashMap<i64, RuntimeState>>> = OnceLock::new();
static NEXT_RT_ID: OnceLock<std::sync::atomic::AtomicI64> = OnceLock::new();
fn rt_reg() -> &'static Mutex<HashMap<i64, RuntimeState>> { RT_REG.get_or_init(|| Mutex::new(HashMap::new())) }
fn next_rt_id() -> i64 { let a = NEXT_RT_ID.get_or_init(|| std::sync::atomic::AtomicI64::new(1)); a.fetch_add(1, std::sync::atomic::Ordering::Relaxed) }

fn runtime_handle(id: i64) -> Value {
    Value::Assoc(HashMap::from([
        ("__type".to_string(), Value::String("ContainersRuntime".into())),
        ("id".to_string(), Value::Integer(id)),
    ]))
}

fn get_runtime(v: &Value) -> Option<i64> {
    if let Value::Assoc(m) = v {
        if matches!(m.get("__type"), Some(Value::String(s)) if s=="ContainersRuntime") {
            if let Some(Value::Integer(id)) = m.get("id") { return Some(*id); }
        }
    }
    None
}

fn container_handle(rt_id: i64, id: i64) -> Value {
    Value::Assoc(HashMap::from([
        ("__type".to_string(), Value::String("Container".into())),
        ("runtime_id".to_string(), Value::Integer(rt_id)),
        ("id".to_string(), Value::Integer(id)),
    ]))
}

fn get_container(v: &Value) -> Option<(i64,i64)> {
    if let Value::Assoc(m) = v {
        if matches!(m.get("__type"), Some(Value::String(s)) if s=="Container") {
            if let (Some(Value::Integer(rt)), Some(Value::Integer(id))) = (m.get("runtime_id"), m.get("id")) { return Some((*rt,*id)); }
        }
    }
    None
}

// ---------- Connect / Runtime ----------
fn connect_containers(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("ConnectContainers".into())), args } }
    let dsn = match ev.eval(args[0].clone()) { Value::String(s)|Value::Symbol(s)=>s, other => return Value::Expr { head: Box::new(Value::Symbol("ConnectContainers".into())), args: vec![other] } };
    let kind = if dsn.starts_with("docker:")||dsn.starts_with("docker://") { RuntimeKind::Docker }
        else if dsn.starts_with("podman:")||dsn.starts_with("podman://") { RuntimeKind::Podman }
        else if dsn.starts_with("containerd:")||dsn.starts_with("containerd://") { RuntimeKind::Containerd }
        else if dsn.starts_with("k8s:")||dsn.starts_with("k8s://")||dsn.starts_with("kubernetes:") { RuntimeKind::K8s }
        else { RuntimeKind::Mock };
    let id = next_rt_id();
    rt_reg().lock().unwrap().insert(id, RuntimeState { dsn, kind, images: Vec::new(), containers: HashMap::new(), next_container_id: 1 });
    runtime_handle(id)
}

fn disconnect_containers(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("DisconnectContainers".into())), args } }
    if let Some(id) = get_runtime(&args[0]) { rt_reg().lock().unwrap().remove(&id); Value::Boolean(true) } else { Value::Expr { head: Box::new(Value::Symbol("DisconnectContainers".into())), args } }
}

fn ping_containers(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("PingContainers".into())), args } }
    Value::Boolean(get_runtime(&args[0]).is_some())
}

fn runtime_info(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("RuntimeInfo".into())), args } }
    match get_runtime(&args[0]) {
        Some(id) => {
            let reg = rt_reg().lock().unwrap();
            if let Some(st) = reg.get(&id) {
                let kind = match st.kind { RuntimeKind::Mock=>"mock", RuntimeKind::Docker=>"docker", RuntimeKind::Podman=>"podman", RuntimeKind::Containerd=>"containerd", RuntimeKind::K8s=>"k8s" };
                Value::Assoc(HashMap::from([
                    ("id".to_string(), Value::Integer(id)),
                    ("dsn".to_string(), Value::String(st.dsn.clone())),
                    ("backend".to_string(), Value::String(kind.into())),
                ]))
            } else { Value::Assoc(HashMap::new()) }
        }
        None => Value::Expr { head: Box::new(Value::Symbol("RuntimeInfo".into())), args }
    }
}

fn runtime_capabilities(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("RuntimeCapabilities".into())), args } }
    match get_runtime(&args[0]) {
        Some(id) => {
            let reg = rt_reg().lock().unwrap();
            if let Some(st) = reg.get(&id) {
                let caps = match st.kind { RuntimeKind::Mock => vec!["Images","Containers","Exec","Logs","Stats","Events","Volumes","Networks"],
                    RuntimeKind::Docker|RuntimeKind::Podman => vec!["Images","Containers","Exec","Logs","Stats","Events","Volumes","Networks"],
                    RuntimeKind::Containerd => vec!["Images","Containers","Exec","Logs","Stats","Events"],
                    RuntimeKind::K8s => vec!["Images","Containers","Exec","Logs","Events","Networks"], };
                Value::Assoc(HashMap::from([(String::from("capabilities"), Value::List(caps.into_iter().map(|s| Value::String(s.into())).collect()))]))
            } else { Value::Assoc(HashMap::new()) }
        }
        None => Value::Expr { head: Box::new(Value::Symbol("RuntimeCapabilities".into())), args }
    }
}

// ---------- Images ----------
fn pull_image(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("PullImage".into())), args } }
    let rt = match get_runtime(&args[0]) { Some(id)=> id, None => return Value::Expr { head: Box::new(Value::Symbol("PullImage".into())), args } };
    let name = match ev.eval(args[1].clone()) { Value::String(s)|Value::Symbol(s)=>s, other=> return Value::Expr { head: Box::new(Value::Symbol("PullImage".into())), args: vec![args[0].clone(), other] } };
    // If Docker backend is enabled and runtime is Docker, try real pull
    #[cfg(feature = "containers_docker")]
    {
        let backend_is_docker = { let reg = rt_reg().lock().unwrap(); reg.get(&rt).map(|s| matches!(s.kind, RuntimeKind::Docker)).unwrap_or(false) };
        if backend_is_docker { return containers_docker::docker_pull_image(&get_rt_dsn(rt), &name) }
    }
    let mut reg = rt_reg().lock().unwrap();
    if let Some(st) = reg.get_mut(&rt) { if !st.images.contains(&name) { st.images.push(name); } }
    Value::Boolean(true)
}

fn build_image(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("BuildImage".into())), args } }
    let _rt = match get_runtime(&args[0]) { Some(id)=> id, None => return Value::Expr { head: Box::new(Value::Symbol("BuildImage".into())), args } };
    #[cfg(feature = "containers_docker")]
    {
        let is_docker = { let reg = rt_reg().lock().unwrap(); reg.get(&rt).map(|s| matches!(s.kind, RuntimeKind::Docker)).unwrap_or(false) };
        if is_docker {
            // Accept either a string Dockerfile or an assoc with Dockerfile key
            let (dockerfile, tag, buildargs, pull) = match args.get(1).cloned() {
                Some(Value::Assoc(m)) => {
                    let df = m.get("Dockerfile").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None }).unwrap_or_default();
                    let opts = if args.len()>=3 { ev.eval(args[2].clone()) } else { Value::Assoc(HashMap::new()) };
                    let (tag, buildargs, pull) = match opts {
                        Value::Assoc(mm) => {
                            let tag = mm.get("Tags").and_then(|v| match v { Value::List(xs) if !xs.is_empty()=> match &xs[0] { Value::String(s)|Value::Symbol(s)=> Some(s.clone()), other=> Some(lyra_core::pretty::format_value(other)) }, Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=> None });
                            let buildargs = mm.get("BuildArgs").and_then(|v| match v { Value::Assoc(am)=> Some(am.clone()), _=> None });
                            let pull = mm.get("Pull").and_then(|v| match v { Value::Boolean(b)=> Some(*b), _=> None }).unwrap_or(false);
                            (tag, buildargs, pull)
                        }
                        _ => (None, None, false)
                    };
                    (df, tag, buildargs, pull)
                }
                Some(Value::String(s))|Some(Value::Symbol(s)) => {
                    let opts = if args.len()>=3 { ev.eval(args[2].clone()) } else { Value::Assoc(HashMap::new()) };
                    let (tag, buildargs, pull) = match opts { Value::Assoc(mm) => {
                        let tag = mm.get("Tags").and_then(|v| match v { Value::List(xs) if !xs.is_empty()=> match &xs[0] { Value::String(s)|Value::Symbol(s)=> Some(s.clone()), other=> Some(lyra_core::pretty::format_value(other)) }, Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=> None });
                        let buildargs = mm.get("BuildArgs").and_then(|v| match v { Value::Assoc(am)=> Some(am.clone()), _=> None });
                        let pull = mm.get("Pull").and_then(|v| match v { Value::Boolean(b)=> Some(*b), _=> None }).unwrap_or(false);
                        (tag, buildargs, pull)
                    }, _ => (None, None, false) };
                    (s, tag, buildargs, pull)
                }
                _ => (String::new(), None, None, false)
            };
            return containers_docker::docker_build_image(&get_rt_dsn(rt), dockerfile, tag, buildargs, pull);
        }
    }
    Value::Boolean(true)
}

fn tag_image(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=3 { return Value::Expr { head: Box::new(Value::Symbol("TagImage".into())), args } }
    let rt = match get_runtime(&args[0]) { Some(id)=> id, None => return Value::Expr { head: Box::new(Value::Symbol("TagImage".into())), args } };
    let dst = match ev.eval(args[2].clone()) { Value::String(s)|Value::Symbol(s)=>s, other=> return Value::Expr { head: Box::new(Value::Symbol("TagImage".into())), args: vec![args[0].clone(), args[1].clone(), other] } };
    #[cfg(feature = "containers_docker")]
    {
        let is_docker = { let reg = rt_reg().lock().unwrap(); reg.get(&rt).map(|s| matches!(s.kind, RuntimeKind::Docker)).unwrap_or(false) };
        if is_docker {
            let src = match ev.eval(args[1].clone()) { Value::String(s)|Value::Symbol(s)=>s, _=> String::new() };
            return containers_docker::docker_tag_image(&get_rt_dsn(rt), src, dst);
        }
    }
    let mut reg = rt_reg().lock().unwrap();
    if let Some(st) = reg.get_mut(&rt) { if !st.images.contains(&dst) { st.images.push(dst); } }
    Value::Boolean(true)
}

fn push_image(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("PushImage".into())), args } }
    #[cfg(feature = "containers_docker")]
    {
        if let Some(rt_id) = get_runtime(&args[0]) {
            let is_docker = { let reg = rt_reg().lock().unwrap(); reg.get(&rt_id).map(|s| matches!(s.kind, RuntimeKind::Docker)).unwrap_or(false) };
            if is_docker {
                let image = match &args[1] { Value::String(s)|Value::Symbol(s)=>s.clone(), other=> lyra_core::pretty::format_value(other) };
                let opts = args.get(2).cloned();
                return containers_docker::docker_push_image(&get_rt_dsn(rt_id), image, opts);
            }
        }
    }
    Value::Boolean(true)
}

fn save_image(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("SaveImage".into())), args } }
    #[cfg(feature = "containers_docker")]
    {
        if let Some(rt_id) = get_runtime(&args[0]) {
            let is_docker = { let reg = rt_reg().lock().unwrap(); reg.get(&rt_id).map(|s| matches!(s.kind, RuntimeKind::Docker)).unwrap_or(false) };
            if is_docker { 
                let image = match &args[1] { Value::String(s)|Value::Symbol(s)=>s.clone(), other=> lyra_core::pretty::format_value(other) };
                return containers_docker::docker_save_image(&get_rt_dsn(rt_id), &image);
            }
        }
    }
    // return opaque bytes (base64) for mock
    Value::String(String::from(""))
}

fn load_image(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("LoadImage".into())), args } }
    #[cfg(feature = "containers_docker")]
    {
        if let Some(rt_id) = get_runtime(&args[0]) {
            let is_docker = { let reg = rt_reg().lock().unwrap(); reg.get(&rt_id).map(|s| matches!(s.kind, RuntimeKind::Docker)).unwrap_or(false) };
            if is_docker { 
                let data_b64 = match &args[1] { Value::String(s)|Value::Symbol(s)=>s.clone(), _=> String::new() };
                return containers_docker::docker_load_image(&get_rt_dsn(rt_id), &data_b64);
            }
        }
    }
    Value::Boolean(true)
}

fn list_images(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("ListImages".into())), args } }
    // Docker-backed list when available
    #[cfg(feature = "containers_docker")]
    {
        if let Some(rt_id) = get_runtime(&args[0]) {
            let is_docker = { let reg = rt_reg().lock().unwrap(); reg.get(&rt_id).map(|s| matches!(s.kind, RuntimeKind::Docker)).unwrap_or(false) };
            if is_docker { return containers_docker::docker_list_images(&get_rt_dsn(rt_id)); }
        }
    }
    Value::Expr { head: Box::new(Value::Symbol("DatasetFromRows".into())), args: vec![Value::List(Vec::new())] }
}

fn inspect_image(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("InspectImage".into())), args } }
    #[cfg(feature = "containers_docker")]
    {
        if let Some(rt_id) = get_runtime(&args[0]) {
            let is_docker = { let reg = rt_reg().lock().unwrap(); reg.get(&rt_id).map(|s| matches!(s.kind, RuntimeKind::Docker)).unwrap_or(false) };
            if is_docker { 
                let image = match &args[1] { Value::String(s)|Value::Symbol(s)=>s.clone(), other=> lyra_core::pretty::format_value(other) };
                return containers_docker::docker_inspect_image(&get_rt_dsn(rt_id), &image);
            }
        }
    }
    Value::Assoc(HashMap::new())
}

fn remove_image(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("RemoveImage".into())), args } }
    #[cfg(feature = "containers_docker")]
    {
        if let Some(rt_id) = get_runtime(&args[0]) {
            let is_docker = { let reg = rt_reg().lock().unwrap(); reg.get(&rt_id).map(|s| matches!(s.kind, RuntimeKind::Docker)).unwrap_or(false) };
            if is_docker {
                let image = match ev.eval(args[1].clone()) { Value::String(s)|Value::Symbol(s)=>s, other=> lyra_core::pretty::format_value(&other) };
                let opts = if args.len()>=3 { ev.eval(args[2].clone()) } else { Value::Assoc(HashMap::new()) };
                return containers_docker::docker_remove_image(&get_rt_dsn(rt_id), &image, opts);
            }
        }
    }
    Value::Boolean(true)
}

fn prune_images(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<1 { return Value::Expr { head: Box::new(Value::Symbol("PruneImages".into())), args } }
    #[cfg(feature = "containers_docker")]
    {
        if let Some(rt_id) = get_runtime(&args[0]) {
            let is_docker = { let reg = rt_reg().lock().unwrap(); reg.get(&rt_id).map(|s| matches!(s.kind, RuntimeKind::Docker)).unwrap_or(false) };
            if is_docker {
                let opts = if args.len()>=2 { ev.eval(args[1].clone()) } else { Value::Assoc(HashMap::new()) };
                return containers_docker::docker_prune_images(&get_rt_dsn(rt_id), opts);
            }
        }
    }
    Value::Assoc(HashMap::new())
}

fn search_images(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("SearchImages".into())), args } }
    #[cfg(feature = "containers_docker")]
    {
        if let Some(rt_id) = get_runtime(&args[0]) {
            let is_docker = { let reg = rt_reg().lock().unwrap(); reg.get(&rt_id).map(|s| matches!(s.kind, RuntimeKind::Docker)).unwrap_or(false) };
            if is_docker {
                let term = match ev.eval(args[1].clone()) { Value::String(s)|Value::Symbol(s)=>s, other=> lyra_core::pretty::format_value(&other) };
                let opts = if args.len()>=3 { ev.eval(args[2].clone()) } else { Value::Assoc(HashMap::new()) };
                return containers_docker::docker_search_images(&get_rt_dsn(rt_id), term, opts);
            }
        }
    }
    Value::Expr { head: Box::new(Value::Symbol("DatasetFromRows".into())), args: vec![Value::List(Vec::new())] }
}

fn image_history(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("ImageHistory".into())), args } }
    #[cfg(feature = "containers_docker")]
    {
        if let Some(rt_id) = get_runtime(&args[0]) {
            let is_docker = { let reg = rt_reg().lock().unwrap(); reg.get(&rt_id).map(|s| matches!(s.kind, RuntimeKind::Docker)).unwrap_or(false) };
            if is_docker { 
                let image = match ev.eval(args[1].clone()) { Value::String(s)|Value::Symbol(s)=>s, other=> lyra_core::pretty::format_value(&other) };
                return containers_docker::docker_image_history(&get_rt_dsn(rt_id), &image);
            }
        }
    }
    Value::Expr { head: Box::new(Value::Symbol("DatasetFromRows".into())), args: vec![Value::List(Vec::new())] }
}

fn inspect_registry_image(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("InspectRegistryImage".into())), args } }
    #[cfg(feature = "containers_docker")]
    {
        if let Some(rt_id) = get_runtime(&args[0]) {
            let is_docker = { let reg = rt_reg().lock().unwrap(); reg.get(&rt_id).map(|s| matches!(s.kind, RuntimeKind::Docker)).unwrap_or(false) };
            if is_docker { 
                let image = match ev.eval(args[1].clone()) { Value::String(s)|Value::Symbol(s)=>s, other=> lyra_core::pretty::format_value(&other) };
                let auth = if args.len()>=3 { ev.eval(args[2].clone()) } else { Value::Assoc(HashMap::new()) };
                return containers_docker::docker_inspect_registry_image(&get_rt_dsn(rt_id), &image, auth);
            }
        }
    }
    Value::Assoc(HashMap::new())
}

fn export_images(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("ExportImages".into())), args } }
    #[cfg(feature = "containers_docker")]
    {
        if let Some(rt_id) = get_runtime(&args[0]) {
            let is_docker = { let reg = rt_reg().lock().unwrap(); reg.get(&rt_id).map(|s| matches!(s.kind, RuntimeKind::Docker)).unwrap_or(false) };
            if is_docker { 
                let images: Vec<String> = match ev.eval(args[1].clone()) { Value::List(xs)=> xs.into_iter().map(|v| match v { Value::String(s)|Value::Symbol(s)=>s, other=> lyra_core::pretty::format_value(&other) }).collect(), other=> vec![lyra_core::pretty::format_value(&other)] };
                return containers_docker::docker_export_images(&get_rt_dsn(rt_id), &images);
            }
        }
    }
    Value::String(String::new())
}

// ---------- Containers ----------
fn run_container(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("RunContainer".into())), args } }
    match get_runtime(&args[0]) { Some(rt_id)=> {
        #[cfg(feature = "containers_docker")]
        {
            let is_docker = { let reg = rt_reg().lock().unwrap(); reg.get(&rt_id).map(|s| matches!(s.kind, RuntimeKind::Docker)).unwrap_or(false) };
            if is_docker { return containers_docker::docker_run_container(rt_id, &get_rt_dsn(rt_id), args[1].clone()) }
        }
        let mut reg = rt_reg().lock().unwrap();
        if let Some(st) = reg.get_mut(&rt_id) { let id = st.next_container_id; st.next_container_id += 1; st.containers.insert(id, HashMap::new()); return container_handle(rt_id, id); }
        Value::Expr { head: Box::new(Value::Symbol("RunContainer".into())), args }
    }, None => Value::Expr { head: Box::new(Value::Symbol("RunContainer".into())), args } }
}

fn create_container(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("CreateContainer".into())), args } }
    match get_runtime(&args[0]) { Some(rt_id)=> {
        let mut reg = rt_reg().lock().unwrap();
        if let Some(st) = reg.get_mut(&rt_id) { let id = st.next_container_id; st.next_container_id += 1; st.containers.insert(id, HashMap::new()); return container_handle(rt_id, id); }
        Value::Expr { head: Box::new(Value::Symbol("CreateContainer".into())), args }
    }, None => Value::Expr { head: Box::new(Value::Symbol("CreateContainer".into())), args } }
}

fn start_container(_ev: &mut Evaluator, args: Vec<Value>) -> Value { if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("StartContainer".into())), args } } Value::Boolean(true) }
fn stop_container(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("StopContainer".into())), args } }
    if let Some((_rt_id, _cid)) = get_container(&args[1]) {
        #[cfg(feature = "containers_docker")]
        {
            let is_docker = { let reg = rt_reg().lock().unwrap(); reg.get(&rt_id).map(|s| matches!(s.kind, RuntimeKind::Docker)).unwrap_or(false) };
            if is_docker { return containers_docker::docker_stop_container(&get_rt_dsn(rt_id), cid) }
        }
        return Value::Boolean(true)
    }
    Value::Expr { head: Box::new(Value::Symbol("StopContainer".into())), args }
}
fn restart_container(_ev: &mut Evaluator, args: Vec<Value>) -> Value { if args.len()<1 { return Value::Expr { head: Box::new(Value::Symbol("RestartContainer".into())), args } } Value::Boolean(true) }
fn remove_container(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("RemoveContainer".into())), args } }
    if let Some((_rt_id, _cid)) = get_container(&args[1]) {
        #[cfg(feature = "containers_docker")]
        {
            let is_docker = { let reg = rt_reg().lock().unwrap(); reg.get(&rt_id).map(|s| matches!(s.kind, RuntimeKind::Docker)).unwrap_or(false) };
            if is_docker { return containers_docker::docker_remove_container(&get_rt_dsn(rt_id), cid) }
        }
        return Value::Boolean(true)
    }
    Value::Expr { head: Box::new(Value::Symbol("RemoveContainer".into())), args }
}
fn pause_container(_ev: &mut Evaluator, args: Vec<Value>) -> Value { if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("PauseContainer".into())), args } } Value::Boolean(true) }
fn unpause_container(_ev: &mut Evaluator, args: Vec<Value>) -> Value { if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("UnpauseContainer".into())), args } } Value::Boolean(true) }
fn rename_container(_ev: &mut Evaluator, args: Vec<Value>) -> Value { if args.len()!=3 { return Value::Expr { head: Box::new(Value::Symbol("RenameContainer".into())), args } } Value::Boolean(true) }

fn exec_in_container(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<3 { return Value::Expr { head: Box::new(Value::Symbol("ExecInContainer".into())), args } }
    // Docker-backed exec when available
    #[cfg(feature = "containers_docker")]
    {
        if let Some((rt_id, cid)) = get_container(&args[1]) {
            let is_docker = { let reg = rt_reg().lock().unwrap(); reg.get(&rt_id).map(|s| matches!(s.kind, RuntimeKind::Docker)).unwrap_or(false) };
            if is_docker { return containers_docker::docker_exec_in_container(&get_rt_dsn(rt_id), cid, args[2].clone()); }
        }
    }
    // mock fallback
    Value::Assoc(HashMap::from([
        ("exit_code".into(), Value::Integer(0)),
        ("stdout".into(), Value::String(String::new())),
        ("stderr".into(), Value::String(String::new())),
    ]))
}

fn copy_to_container(_ev: &mut Evaluator, args: Vec<Value>) -> Value { if args.len()<4 { return Value::Expr { head: Box::new(Value::Symbol("CopyToContainer".into())), args } } Value::Boolean(true) }
fn copy_from_container(_ev: &mut Evaluator, args: Vec<Value>) -> Value { if args.len()!=3 { return Value::Expr { head: Box::new(Value::Symbol("CopyFromContainer".into())), args } } Value::String(String::new()) }

fn inspect_container(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("InspectContainer".into())), args } }
    if let Some((_rt_id, _cid)) = get_container(&args[1]) {
        #[cfg(feature = "containers_docker")]
        {
            let is_docker = { let reg = rt_reg().lock().unwrap(); reg.get(&rt_id).map(|s| matches!(s.kind, RuntimeKind::Docker)).unwrap_or(false) };
            if is_docker { return containers_docker::docker_inspect_container(&get_rt_dsn(rt_id), cid) }
        }
    }
    Value::Assoc(HashMap::new())
}
fn wait_container(_ev: &mut Evaluator, args: Vec<Value>) -> Value { if args.len()<1 { return Value::Expr { head: Box::new(Value::Symbol("WaitContainer".into())), args } } Value::Assoc(HashMap::from([("status".into(), Value::String("exited".into()))])) }

fn list_containers(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<1 { return Value::Expr { head: Box::new(Value::Symbol("ListContainers".into())), args } }
    if let Some(_rt_id) = get_runtime(&args[0]) {
        #[cfg(feature = "containers_docker")]
        {
            let is_docker = { let reg = rt_reg().lock().unwrap(); reg.get(&rt_id).map(|s| matches!(s.kind, RuntimeKind::Docker)).unwrap_or(false) };
            if is_docker { return containers_docker::docker_list_containers(&get_rt_dsn(rt_id)) }
        }
    }
    Value::Expr { head: Box::new(Value::Symbol("DatasetFromRows".into())), args: vec![Value::List(Vec::new())] }
}

// ---------- Logs / Stats / Events (cursor-like) ----------
#[allow(dead_code)]
#[derive(Clone)]
struct CCurState { kind: String, rt_dsn: String, target: Value, opts: HashMap<String, Value>, offset: i64, buffer: Vec<Value> }
static CCUR_REG: OnceLock<Mutex<HashMap<i64, CCurState>>> = OnceLock::new();
static NEXT_CCUR_ID: OnceLock<std::sync::atomic::AtomicI64> = OnceLock::new();
fn ccur_reg() -> &'static Mutex<HashMap<i64, CCurState>> { CCUR_REG.get_or_init(|| Mutex::new(HashMap::new())) }
fn next_ccur_id() -> i64 { let a = NEXT_CCUR_ID.get_or_init(|| std::sync::atomic::AtomicI64::new(1)); a.fetch_add(1, std::sync::atomic::Ordering::Relaxed) }

fn containers_cursor_handle(id: i64) -> Value {
    Value::Assoc(HashMap::from([
        ("__type".to_string(), Value::String("ContainersCursor".into())),
        ("id".to_string(), Value::Integer(id)),
    ]))
}

fn containers_logs(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("Logs".into())), args } }
    let rt_id = match get_runtime(&args[0]) { Some(id)=>id, None=> return Value::Expr { head: Box::new(Value::Symbol("Logs".into())), args } };
    let reg = rt_reg().lock().unwrap();
    let dsn = reg.get(&rt_id).map(|s| s.dsn.clone()).unwrap_or_default();
    drop(reg);
    let id = next_ccur_id();
    // For Docker, prefetch bounded logs into buffer
    let buffer: Vec<Value> = Vec::new();
    #[cfg(feature = "containers_docker")]
    {
        if let Some((rtid, cid)) = get_container(&args[1]) {
            let is_docker = { let reg = rt_reg().lock().unwrap(); reg.get(&rtid).map(|s| matches!(s.kind, RuntimeKind::Docker)).unwrap_or(false) };
            if is_docker { buffer = containers_docker::docker_fetch_logs(&dsn, cid, false); }
        }
    }
    ccur_reg().lock().unwrap().insert(id, CCurState { kind: "logs".into(), rt_dsn: dsn, target: args[1].clone(), opts: HashMap::new(), offset: 0, buffer });
    containers_cursor_handle(id)
}

fn containers_stats(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("Stats".into())), args } }
    let rt_id = match get_runtime(&args[0]) { Some(id)=>id, None=> return Value::Expr { head: Box::new(Value::Symbol("Stats".into())), args } };
    let reg = rt_reg().lock().unwrap();
    let dsn = reg.get(&rt_id).map(|s| s.dsn.clone()).unwrap_or_default();
    drop(reg);
    let id = next_ccur_id();
    ccur_reg().lock().unwrap().insert(id, CCurState { kind: "stats".into(), rt_dsn: dsn, target: args[1].clone(), opts: HashMap::new(), offset: 0, buffer: Vec::new() });
    containers_cursor_handle(id)
}

fn containers_events(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<1 { return Value::Expr { head: Box::new(Value::Symbol("Events".into())), args } }
    let rt_id = match get_runtime(&args[0]) { Some(id)=>id, None=> return Value::Expr { head: Box::new(Value::Symbol("Events".into())), args } };
    let reg = rt_reg().lock().unwrap();
    let dsn = reg.get(&rt_id).map(|s| s.dsn.clone()).unwrap_or_default();
    drop(reg);
    let id = next_ccur_id();
    ccur_reg().lock().unwrap().insert(id, CCurState { kind: "events".into(), rt_dsn: dsn, target: Value::Symbol("All".into()), opts: HashMap::new(), offset: 0, buffer: Vec::new() });
    containers_cursor_handle(id)
}

fn containers_fetch(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<1 { return Value::Expr { head: Box::new(Value::Symbol("ContainersFetch".into())), args } }
    let n = if args.len()>=2 { if let Value::Integer(k) = args[1] { k.max(0) as usize } else { 100 } } else { 100 };
    // Return next batch from buffer for now
    if let Value::Assoc(m) = &args[0] { if let Some(Value::Integer(id)) = m.get("id") {
        let mut reg = ccur_reg().lock().unwrap();
        if let Some(st) = reg.get_mut(id) {
            let start = st.offset as usize;
            let end = (start + n).min(st.buffer.len());
            let batch = st.buffer[start..end].to_vec();
            st.offset = end as i64;
            return Value::List(batch);
        }
    }}
    Value::List(Vec::new())
}

fn containers_close(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("ContainersClose".into())), args } }
    if let Value::Assoc(m) = &args[0] { if let Some(Value::Integer(id)) = m.get("id") { ccur_reg().lock().unwrap().remove(id); return Value::Boolean(true); } }
    Value::Boolean(false)
}

// ---------- Volumes / Networks / Registry ----------
fn list_volumes(_ev: &mut Evaluator, args: Vec<Value>) -> Value { if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("ListVolumes".into())), args } } Value::Expr { head: Box::new(Value::Symbol("DatasetFromRows".into())), args: vec![Value::List(Vec::new())] } }
fn create_volume(_ev: &mut Evaluator, args: Vec<Value>) -> Value { if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("CreateVolume".into())), args } } Value::Boolean(true) }
fn remove_volume(_ev: &mut Evaluator, args: Vec<Value>) -> Value { if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("RemoveVolume".into())), args } } Value::Boolean(true) }
fn inspect_volume(_ev: &mut Evaluator, args: Vec<Value>) -> Value { if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("InspectVolume".into())), args } } Value::Assoc(HashMap::new()) }

fn list_networks(_ev: &mut Evaluator, args: Vec<Value>) -> Value { if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("ListNetworks".into())), args } } Value::Expr { head: Box::new(Value::Symbol("DatasetFromRows".into())), args: vec![Value::List(Vec::new())] } }
fn create_network(_ev: &mut Evaluator, args: Vec<Value>) -> Value { if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("CreateNetwork".into())), args } } Value::Boolean(true) }
fn remove_network(_ev: &mut Evaluator, args: Vec<Value>) -> Value { if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("RemoveNetwork".into())), args } } Value::Boolean(true) }
fn inspect_network(_ev: &mut Evaluator, args: Vec<Value>) -> Value { if args.len()<2 { return Value::Expr { head: Box::new(Value::Symbol("InspectNetwork".into())), args } } Value::Assoc(HashMap::new()) }

fn add_registry_auth(_ev: &mut Evaluator, args: Vec<Value>) -> Value { if args.len()<3 { return Value::Expr { head: Box::new(Value::Symbol("AddRegistryAuth".into())), args } } Value::Boolean(true) }
fn list_registry_auth(_ev: &mut Evaluator, args: Vec<Value>) -> Value { if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("ListRegistryAuth".into())), args } } Value::Expr { head: Box::new(Value::Symbol("DatasetFromRows".into())), args: vec![Value::List(Vec::new())] } }

// ---------- Explain / Describe ----------
fn explain_containers(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<1 { return Value::Expr { head: Box::new(Value::Symbol("ExplainContainers".into())), args } }
    Value::String("ExplainContainers not yet implemented".into())
}

fn describe_containers(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("DescribeContainers".into())), args } }
    Value::String("DescribeContainers not yet implemented".into())
}

pub fn register_containers(ev: &mut Evaluator) {
    // Runtime
    ev.register("ConnectContainers", connect_containers as NativeFn, Attributes::empty());
    ev.register("DisconnectContainers", disconnect_containers as NativeFn, Attributes::empty());
    ev.register("PingContainers", ping_containers as NativeFn, Attributes::empty());
    ev.register("RuntimeInfo", runtime_info as NativeFn, Attributes::empty());
    ev.register("RuntimeCapabilities", runtime_capabilities as NativeFn, Attributes::empty());

    // Images
    ev.register("PullImage", pull_image as NativeFn, Attributes::empty());
    ev.register("BuildImage", build_image as NativeFn, Attributes::empty());
    ev.register("TagImage", tag_image as NativeFn, Attributes::empty());
    ev.register("PushImage", push_image as NativeFn, Attributes::empty());
    ev.register("SaveImage", save_image as NativeFn, Attributes::empty());
    ev.register("LoadImage", load_image as NativeFn, Attributes::empty());
    ev.register("ListImages", list_images as NativeFn, Attributes::empty());
    ev.register("InspectImage", inspect_image as NativeFn, Attributes::empty());

    // Containers
    ev.register("RunContainer", run_container as NativeFn, Attributes::empty());
    ev.register("CreateContainer", create_container as NativeFn, Attributes::empty());
    ev.register("StartContainer", start_container as NativeFn, Attributes::empty());
    ev.register("StopContainer", stop_container as NativeFn, Attributes::empty());
    ev.register("RestartContainer", restart_container as NativeFn, Attributes::empty());
    ev.register("RemoveContainer", remove_container as NativeFn, Attributes::empty());
    ev.register("PauseContainer", pause_container as NativeFn, Attributes::empty());
    ev.register("UnpauseContainer", unpause_container as NativeFn, Attributes::empty());
    ev.register("RenameContainer", rename_container as NativeFn, Attributes::empty());
    ev.register("ExecInContainer", exec_in_container as NativeFn, Attributes::empty());
    ev.register("CopyToContainer", copy_to_container as NativeFn, Attributes::empty());
    ev.register("CopyFromContainer", copy_from_container as NativeFn, Attributes::empty());
    ev.register("InspectContainer", inspect_container as NativeFn, Attributes::empty());
    ev.register("WaitContainer", wait_container as NativeFn, Attributes::empty());
    ev.register("ListContainers", list_containers as NativeFn, Attributes::empty());

    // Streams
    ev.register("Logs", containers_logs as NativeFn, Attributes::empty());
    ev.register("Stats", containers_stats as NativeFn, Attributes::empty());
    ev.register("Events", containers_events as NativeFn, Attributes::empty());
    ev.register("ContainersFetch", containers_fetch as NativeFn, Attributes::empty());
    ev.register("ContainersClose", containers_close as NativeFn, Attributes::empty());

    // Volumes / Networks / Registry
    ev.register("ListVolumes", list_volumes as NativeFn, Attributes::empty());
    ev.register("CreateVolume", create_volume as NativeFn, Attributes::empty());
    ev.register("RemoveVolume", remove_volume as NativeFn, Attributes::empty());
    ev.register("InspectVolume", inspect_volume as NativeFn, Attributes::empty());

    ev.register("ListNetworks", list_networks as NativeFn, Attributes::empty());
    ev.register("CreateNetwork", create_network as NativeFn, Attributes::empty());
    ev.register("RemoveNetwork", remove_network as NativeFn, Attributes::empty());
    ev.register("InspectNetwork", inspect_network as NativeFn, Attributes::empty());

    ev.register("AddRegistryAuth", add_registry_auth as NativeFn, Attributes::empty());
    ev.register("ListRegistryAuth", list_registry_auth as NativeFn, Attributes::empty());

    // Explain / Describe
    ev.register("ExplainContainers", explain_containers as NativeFn, Attributes::empty());
    ev.register("DescribeContainers", describe_containers as NativeFn, Attributes::empty());

    // Image maintenance
    ev.register("RemoveImage", remove_image as NativeFn, Attributes::empty());
    ev.register("PruneImages", prune_images as NativeFn, Attributes::empty());
    ev.register("SearchImages", search_images as NativeFn, Attributes::empty());
    ev.register("ImageHistory", image_history as NativeFn, Attributes::empty());
    ev.register("InspectRegistryImage", inspect_registry_image as NativeFn, Attributes::empty());
    ev.register("ExportImages", export_images as NativeFn, Attributes::empty());
}

// Helpers to access runtime info
#[allow(dead_code)]
fn get_rt_dsn(rt_id: i64) -> String { let reg = rt_reg().lock().unwrap(); reg.get(&rt_id).map(|s| s.dsn.clone()).unwrap_or_default() }

// Docker backend implementation
#[cfg(feature = "containers_docker")]
    pub mod containers_docker {
        use super::*;
        use base64::Engine;
        use bollard::Docker;
    use bollard::container::{ListContainersOptions, CreateContainerOptions, StartContainerOptions, RemoveContainerOptions, Config as ContainerConfig, StopContainerOptions, LogsOptions};
    use bollard::image::{CreateImageOptions, ListImagesOptions, TagImageOptions, PushImageOptions};
    use bollard::auth::DockerCredentials;
    use bollard::exec::{CreateExecOptions, StartExecOptions};
    use futures_util::StreamExt;

    fn docker_client(_dsn: &str) -> Docker { Docker::connect_with_local_defaults().expect("Docker socket not available") }

    pub fn docker_pull_image(_dsn: &str, image: &str) -> Value {
        let docker = docker_client(_dsn);
        let img = image.to_string();
        let rt = tokio::runtime::Runtime::new(); if rt.is_err() { return Value::Boolean(false) } let rt = rt.unwrap();
        let res = rt.block_on(async move {
            let mut stream = docker.create_image(Some(CreateImageOptions { from_image: img.as_str(), ..Default::default() }), None, None);
            while let Some(_progress) = stream.next().await {}
            Ok::<(), ()>(())
        });
        Value::Boolean(res.is_ok())
    }

    pub fn docker_run_container(rt_id: i64, dsn: &str, spec_val: Value) -> Value {
        let docker = docker_client(dsn);
        let mut image = String::from("alpine:latest");
        let mut name: Option<String> = None;
        let mut cmd: Option<Vec<String>> = None;
        if let Value::Assoc(m) = spec_val {
            if let Some(Value::String(s))|Some(Value::Symbol(s)) = m.get("Image") { image = s.clone(); }
            if let Some(Value::String(s))|Some(Value::Symbol(s)) = m.get("Name") { name = Some(s.clone()); }
            if let Some(Value::List(xs)) = m.get("Cmd") { cmd = Some(xs.iter().map(|v| match v { Value::String(s)=>s.clone(), Value::Symbol(s)=>s.clone(), other=> lyra_core::pretty::format_value(other) }).collect()); }
        }
        let rt = tokio::runtime::Runtime::new(); if rt.is_err() { return Value::Expr { head: Box::new(Value::Symbol("RunContainer".into())), args: vec![runtime_handle(rt_id)] } } let rt = rt.unwrap();
        let res = rt.block_on(async move {
            let mut stream = docker.create_image(Some(CreateImageOptions { from_image: image.as_str(), ..Default::default() }), None, None);
            while let Some(_progress) = stream.next().await {}
            let mut cfg = ContainerConfig::<String> { image: Some(image.clone()), ..Default::default() };
            if let Some(c) = cmd { cfg.cmd = Some(c); }
            let create = docker.create_container(name.as_deref().map(|n| CreateContainerOptions { name: n, platform: None }), cfg).await;
            let id = match create { Ok(res) => res.id, Err(_) => { return Err(()); } };
            if docker.start_container(&id, None::<StartContainerOptions<String>>).await.is_err() { return Err(()); }
            Ok(id)
        });
        match res { Ok(id) => {
            let mut reg = rt_reg().lock().unwrap();
            if let Some(st) = reg.get_mut(&rt_id) {
                let nid = st.next_container_id; st.next_container_id += 1;
                st.containers.insert(nid, HashMap::from([(String::from("docker_id"), Value::String(id))]));
                return container_handle(rt_id, nid);
            }
            Value::Expr { head: Box::new(Value::Symbol("RunContainer".into())), args: vec![runtime_handle(rt_id)] }
        }, Err(_) => Value::Expr { head: Box::new(Value::Symbol("RunContainer".into())), args: vec![runtime_handle(rt_id)] } }
    }

    pub fn docker_list_containers(dsn: &str) -> Value {
        let docker = docker_client(dsn);
        let rt = tokio::runtime::Runtime::new(); if rt.is_err() { return Value::Expr { head: Box::new(Value::Symbol("DatasetFromRows".into())), args: vec![Value::List(Vec::new())] } } let rt = rt.unwrap();
        let res = rt.block_on(async move { docker.list_containers(Some(ListContainersOptions::<String> { all: true, ..Default::default() })).await });
        match res {
            Ok(list) => {
                let rows: Vec<Value> = list.into_iter().map(|c| {
                    let mut m = HashMap::new();
                    m.insert("id".into(), Value::String(c.id.unwrap_or_default()));
                    if let Some(names) = c.names { if let Some(n) = names.first() { m.insert("name".into(), Value::String(n.trim_start_matches('/').to_string())); } }
                    m.insert("image".into(), Value::String(c.image.unwrap_or_default()));
                    m.insert("state".into(), Value::String(c.state.unwrap_or_default()));
                    m.insert("status".into(), Value::String(c.status.unwrap_or_default()));
                    m.insert("created".into(), Value::Integer(c.created.unwrap_or_default() as i64));
                    Value::Assoc(m)
                }).collect();
                Value::Expr { head: Box::new(Value::Symbol("DatasetFromRows".into())), args: vec![Value::List(rows)] }
            }
            Err(_) => Value::Expr { head: Box::new(Value::Symbol("DatasetFromRows".into())), args: vec![Value::List(Vec::new())] }
        }
    }

    pub fn docker_inspect_container(dsn: &str, cid: i64) -> Value {
        let docker_id = {
            let reg = rt_reg().lock().unwrap();
            reg.values().find_map(|st| st.containers.get(&cid).and_then(|m| m.get("docker_id")).and_then(|v| match v { Value::String(s)=>Some(s.clone()), _=>None }))
        };
        let Some(docker_id) = docker_id else { return Value::Assoc(HashMap::new()) };
        let docker = docker_client(dsn);
        let rt = tokio::runtime::Runtime::new(); if rt.is_err() { return Value::Assoc(HashMap::new()) } let rt = rt.unwrap();
        let res = rt.block_on(async move { docker.inspect_container(&docker_id, None).await });
        match res { Ok(info) => {
            let mut m = HashMap::new();
            m.insert("id".into(), Value::String(info.id.unwrap_or_default()));
            if let Some(state) = info.state { if let Some(running) = state.running { m.insert("running".into(), Value::Boolean(running)); } }
            if let Some(name) = info.name { m.insert("name".into(), Value::String(name.trim_start_matches('/').to_string())); }
            if let Some(cfg) = info.config { if let Some(img) = cfg.image { m.insert("image".into(), Value::String(img)); } }
            Value::Assoc(m)
        }, Err(_) => Value::Assoc(HashMap::new()) }
    }

    pub fn docker_stop_container(dsn: &str, cid: i64) -> Value {
        let docker_id = {
            let reg = rt_reg().lock().unwrap();
            reg.values().find_map(|st| st.containers.get(&cid).and_then(|m| m.get("docker_id")).and_then(|v| match v { Value::String(s)=>Some(s.clone()), _=>None }))
        };
        let Some(docker_id) = docker_id else { return Value::Boolean(false) };
        let docker = docker_client(dsn);
        let rt = tokio::runtime::Runtime::new(); if rt.is_err() { return Value::Boolean(false) } let rt = rt.unwrap();
        let res = rt.block_on(async move { docker.stop_container(&docker_id, Some(StopContainerOptions { t: 10 })).await });
        Value::Boolean(res.is_ok())
    }

    pub fn docker_remove_container(dsn: &str, cid: i64) -> Value {
        let docker_id = {
            let reg = rt_reg().lock().unwrap();
            reg.values().find_map(|st| st.containers.get(&cid).and_then(|m| m.get("docker_id")).and_then(|v| match v { Value::String(s)=>Some(s.clone()), _=>None }))
        };
        let Some(docker_id) = docker_id else { return Value::Boolean(false) };
        let docker = docker_client(dsn);
        let rt = tokio::runtime::Runtime::new(); if rt.is_err() { return Value::Boolean(false) } let rt = rt.unwrap();
        let res = rt.block_on(async move { docker.remove_container(&docker_id, Some(RemoveContainerOptions { force: true, ..Default::default() })).await });
        Value::Boolean(res.is_ok())
    }

    pub fn docker_fetch_logs(dsn: &str, cid: i64, follow: bool) -> Vec<Value> {
        let docker_id = {
            let reg = rt_reg().lock().unwrap();
            reg.values().find_map(|st| st.containers.get(&cid).and_then(|m| m.get("docker_id")).and_then(|v| match v { Value::String(s)=>Some(s.clone()), _=>None }))
        };
        let Some(docker_id) = docker_id else { return Vec::new() };
        let docker = docker_client(dsn);
        let rt = tokio::runtime::Runtime::new(); if rt.is_err() { return Vec::new() } let rt = rt.unwrap();
        let res = rt.block_on(async move {
            let options = LogsOptions::<String> { stdout: true, stderr: true, follow, timestamps: true, tail: "all".into(), ..Default::default() };
            let mut stream = docker.logs(&docker_id, Some(options));
            let mut out = Vec::new();
            while let Some(item) = stream.next().await { if let Ok(chunk) = item {
                use bollard::container::LogOutput;
                match chunk { LogOutput::StdOut { message } => {
                        let msg = String::from_utf8_lossy(&message).to_string();
                        out.push(Value::Assoc(HashMap::from([(String::from("stream"), Value::String(String::from("stdout"))), (String::from("message"), Value::String(msg))])));
                    },
                    LogOutput::StdErr { message } => {
                        let msg = String::from_utf8_lossy(&message).to_string();
                        out.push(Value::Assoc(HashMap::from([(String::from("stream"), Value::String(String::from("stderr"))), (String::from("message"), Value::String(msg))])));
                    },
                    LogOutput::Console { message } => {
                        let msg = String::from_utf8_lossy(&message).to_string();
                        out.push(Value::Assoc(HashMap::from([(String::from("stream"), Value::String(String::from("console"))), (String::from("message"), Value::String(msg))])));
                    },
                    _ => {}
                }
            }}
            out
        });
        res
    }

    pub fn docker_exec_in_container(dsn: &str, cid: i64, spec: Value) -> Value {
        let docker_id = {
            let reg = rt_reg().lock().unwrap();
            reg.values().find_map(|st| st.containers.get(&cid).and_then(|m| m.get("docker_id")).and_then(|v| match v { Value::String(s)=>Some(s.clone()), _=>None }))
        };
        let Some(docker_id) = docker_id else { return Value::Assoc(HashMap::new()) };
        let mut cmd: Vec<String> = Vec::new();
        if let Value::Assoc(m) = spec { if let Some(Value::List(xs)) = m.get("Cmd") { cmd = xs.iter().map(|v| match v { Value::String(s)=>s.clone(), Value::Symbol(s)=>s.clone(), other=> lyra_core::pretty::format_value(other) }).collect(); } }
        if cmd.is_empty() { return Value::Assoc(HashMap::from([(String::from("exit_code"), Value::Integer(127))])) }
        let docker = docker_client(dsn);
        let rt = tokio::runtime::Runtime::new(); if rt.is_err() { return Value::Assoc(HashMap::new()) } let rt = rt.unwrap();
        let res = rt.block_on(async move {
            let create = docker.create_exec(&docker_id, CreateExecOptions { attach_stdout: Some(true), attach_stderr: Some(true), cmd: Some(cmd.iter().map(|s| s.as_str()).collect()), ..Default::default() }).await;
            let id = match create { Ok(x) => x.id, Err(_) => return Err(()) };
            let mut stdout = String::new();
            let mut stderr = String::new();
            let mut stream = docker.start_exec(&id, None::<StartExecOptions>).await.unwrap();
            use bollard::exec::StartExecResults;
            use bollard::container::LogOutput;
            match &mut stream {
                StartExecResults::Attached { output, .. } => {
                    let mut s = output.take(usize::MAX);
                    while let Some(item) = s.next().await { if let Ok(chunk) = item {
                        match chunk { LogOutput::StdOut { message } => { stdout.push_str(&String::from_utf8_lossy(&message)); }, LogOutput::StdErr { message } => { stderr.push_str(&String::from_utf8_lossy(&message)); }, _ => {} }
                    } }
                }
                _ => {}
            }
            let inspect = docker.inspect_exec(&id).await;
            let code = inspect.ok().and_then(|i| i.exit_code).unwrap_or(0);
            Ok((code, stdout, stderr))
        });
        match res { Ok((code, out, err)) => Value::Assoc(HashMap::from([(String::from("exit_code"), Value::Integer(code as i64)), (String::from("stdout"), Value::String(out)), (String::from("stderr"), Value::String(err))])), Err(_) => Value::Assoc(HashMap::new()) }
    }

    pub fn docker_list_images(dsn: &str) -> Value {
        let docker = docker_client(dsn);
        let rt = tokio::runtime::Runtime::new(); if rt.is_err() { return Value::Expr { head: Box::new(Value::Symbol("DatasetFromRows".into())), args: vec![Value::List(Vec::new())] } } let rt = rt.unwrap();
        let res = rt.block_on(async move { docker.list_images(Some(ListImagesOptions::<String> { all: true, ..Default::default() })).await });
        match res { Ok(imgs) => {
            let mut rows: Vec<Value> = Vec::new();
            for img in imgs {
                let tags = img.repo_tags;
                for t in tags {
                    let (repo, tag) = if let Some((r, tg)) = t.rsplit_once(':') { (r.to_string(), tg.to_string()) } else { (t.clone(), "latest".into()) };
                    let mut m = HashMap::new();
                    m.insert("id".into(), Value::String(img.id.clone()));
                    m.insert("repo".into(), Value::String(repo));
                    m.insert("tag".into(), Value::String(tag));
                    m.insert("size".into(), Value::Integer(img.size as i64));
                    rows.push(Value::Assoc(m));
                }
            }
            Value::Expr { head: Box::new(Value::Symbol("DatasetFromRows".into())), args: vec![Value::List(rows)] }
        }, Err(_) => Value::Expr { head: Box::new(Value::Symbol("DatasetFromRows".into())), args: vec![Value::List(Vec::new())] } }
    }

    pub fn docker_tag_image(dsn: &str, src: String, dst: String) -> Value {
        let (repo, tag) = if let Some((r, t)) = dst.rsplit_once(':') { (r.to_string(), t.to_string()) } else { (dst, String::from("latest")) };
        let docker = docker_client(dsn);
        let rt = tokio::runtime::Runtime::new(); if rt.is_err() { return Value::Boolean(false) } let rt = rt.unwrap();
        let res = rt.block_on(async move { docker.tag_image(&src, Some(TagImageOptions { repo, tag })).await });
        Value::Boolean(res.is_ok())
    }

    pub fn docker_push_image(dsn: &str, image: String, opts: Option<Value>) -> Value {
        let docker = docker_client(dsn);
        let mut creds: Option<DockerCredentials> = None;
        if let Some(Value::Assoc(m)) = opts {
            let username = m.get("Username").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None });
            let password = m.get("Password").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None });
            let serveraddress = m.get("Registry").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None });
            if username.is_some() || password.is_some() || serveraddress.is_some() { creds = Some(DockerCredentials { username, password, serveraddress, ..Default::default() }); }
        }
        let (repo, tag) = if let Some((r, t)) = image.rsplit_once(':') { (r.to_string(), t.to_string()) } else { (image, String::from("latest")) };
        let rt = tokio::runtime::Runtime::new(); if rt.is_err() { return Value::Boolean(false) } let rt = rt.unwrap();
        let res = rt.block_on(async move {
            let mut stream = docker.push_image(&repo, Some(PushImageOptions { tag: tag.as_str() }), creds);
            while let Some(_progress) = stream.next().await {}
            Ok::<(), ()>(())
        });
        Value::Boolean(res.is_ok())
    }

    pub fn docker_build_image(
        dsn: &str,
        dockerfile: String,
        tag: Option<String>,
        buildargs: Option<HashMap<String, Value>>,
        pull: bool,
    ) -> Value {
        let docker = docker_client(dsn);
        // Build tar archive with a single Dockerfile
        let mut header = tar::Header::new_gnu();
        header.set_path("Dockerfile").ok();
        header.set_size(dockerfile.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        let mut builder = tar::Builder::new(Vec::new());
        if builder.append(&header, dockerfile.as_bytes()).is_err() { return Value::Boolean(false) }
        let uncompressed = match builder.into_inner() { Ok(v)=>v, Err(_)=> return Value::Boolean(false) };
        let rt = tokio::runtime::Runtime::new(); if rt.is_err() { return Value::Boolean(false) } let rt = rt.unwrap();
        use bollard::image::BuildImageOptions;
        use futures_util::StreamExt;
        let tag_val = tag.unwrap_or_else(|| String::from("lyra-build:latest"));
        let buildargs_map: std::collections::HashMap<String, String> = buildargs.unwrap_or_default().into_iter().map(|(k,v)| (k, match v { Value::String(s)|Value::Symbol(s)=>s, Value::Integer(i)=>i.to_string(), Value::Real(f)=>f.to_string(), Value::Boolean(b)=> if b {"true".into()} else {"false".into()}, other=> lyra_core::pretty::format_value(&other) })).collect();
        let res = rt.block_on(async move {
            let mut stream = docker.build_image(
                BuildImageOptions::<String> { dockerfile: "Dockerfile".into(), t: tag_val, pull, rm: true, buildargs: buildargs_map, ..Default::default() },
                None,
                Some(uncompressed.into()),
            );
            while let Some(_progress) = stream.next().await {}
            Ok::<(), ()>(())
        });
        Value::Boolean(res.is_ok())
    }

    pub fn docker_inspect_image(dsn: &str, image: &str) -> Value {
        let docker = docker_client(dsn);
        let img = image.to_string();
        let rt = tokio::runtime::Runtime::new(); if rt.is_err() { return Value::Assoc(HashMap::new()) } let rt = rt.unwrap();
        let res = rt.block_on(async move { docker.inspect_image(&img).await });
        match res {
            Ok(info) => {
                let mut m = HashMap::new();
                m.insert("id".into(), Value::String(info.id.unwrap_or_default()));
                if let Some(size) = info.size { m.insert("size".into(), Value::Integer(size as i64)); }
                if let Some(v) = info.os { m.insert("os".into(), Value::String(v)); }
                if let Some(v) = info.architecture { m.insert("arch".into(), Value::String(v)); }
                if let Some(v) = info.created { m.insert("created".into(), Value::String(v)); }
                if let Some(config) = info.config { if let Some(env) = config.env { m.insert("env".into(), Value::List(env.into_iter().map(Value::String).collect())); } }
                if let Some(tags) = info.repo_tags { m.insert("tags".into(), Value::List(tags.into_iter().map(Value::String).collect())); }
                Value::Assoc(m)
            }
            Err(_) => Value::Assoc(HashMap::new())
        }
    }

    pub fn docker_save_image(dsn: &str, image: &str) -> Value {
        let docker = docker_client(dsn);
        let name = image.to_string();
        let rt = tokio::runtime::Runtime::new(); if rt.is_err() { return Value::String(String::new()) } let rt = rt.unwrap();
        let res = rt.block_on(async move {
            use futures_util::TryStreamExt;
            let mut stream = docker.export_images(&[name.as_str()]);
            let mut bytes: Vec<u8> = Vec::new();
            while let Some(chunk) = stream.next().await { let b = chunk.map_err(|_| ())?; bytes.extend_from_slice(&b); }
            Ok::<Vec<u8>, ()>(bytes)
        });
        match res { Ok(data) => Value::String(base64::engine::general_purpose::STANDARD.encode(&data)), Err(_) => Value::String(String::new()) }
    }

    pub fn docker_load_image(dsn: &str, b64: &str) -> Value {
        let docker = docker_client(dsn);
        let data = match base64::engine::general_purpose::STANDARD.decode(b64) { Ok(d)=>d, Err(_)=> return Value::Boolean(false) };
        let rt = tokio::runtime::Runtime::new(); if rt.is_err() { return Value::Boolean(false) } let rt = rt.unwrap();
        let res = rt.block_on(async move {
            use bollard::image::ImportImageOptions;
            use bytes::Bytes;
            use futures_util::stream::StreamExt;
            let bytes = Bytes::from(data);
            let mut stream = docker.import_image(ImportImageOptions { quiet: true }, bytes, None);
            while let Some(_chunk) = stream.next().await {}
            Ok::<(), ()>(())
        });
        Value::Boolean(res.is_ok())
    }

    pub fn docker_remove_image(dsn: &str, image: &str, opts: Value) -> Value {
        use bollard::image::RemoveImageOptions;
        let docker = docker_client(dsn);
        let (force, noprune, creds) = match opts {
            Value::Assoc(m) => {
                let force = m.get("Force").and_then(|v| match v { Value::Boolean(b)=>Some(*b), _=>None }).unwrap_or(false);
                let noprune = m.get("NoPrune").and_then(|v| match v { Value::Boolean(b)=>Some(*b), _=>None }).unwrap_or(false);
                let username = m.get("Username").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None });
                let password = m.get("Password").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None });
                let serveraddress = m.get("Registry").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None });
                let creds = if username.is_some() || password.is_some() || serveraddress.is_some() { Some(DockerCredentials { username, password, serveraddress, ..Default::default() }) } else { None };
                (force, noprune, creds)
            }
            _ => (false, false, None)
        };
        let rt = tokio::runtime::Runtime::new(); if rt.is_err() { return Value::Boolean(false) } let rt = rt.unwrap();
        let res = rt.block_on(async move { docker.remove_image(image, Some(RemoveImageOptions { force, noprune }), creds).await });
        Value::Boolean(res.is_ok())
    }

    pub fn docker_prune_images(dsn: &str, opts: Value) -> Value {
        use bollard::image::PruneImagesOptions;
        let docker = docker_client(dsn);
        let filters: std::collections::HashMap<String, Vec<String>> = match opts {
            Value::Assoc(m) => {
                // pass through the provided filters if caller specifies
                if let Some(Value::Assoc(fm)) = m.get("Filters") {
                    fm.iter().map(|(k,v)| (k.clone(), match v { Value::List(xs)=> xs.iter().map(|x| match x { Value::String(s)|Value::Symbol(s)=> s.clone(), other=> lyra_core::pretty::format_value(other) }).collect(), other=> vec![lyra_core::pretty::format_value(other)] })).collect()
                } else { std::collections::HashMap::new() }
            }
            _ => std::collections::HashMap::new()
        };
        let rt = tokio::runtime::Runtime::new(); if rt.is_err() { return Value::Assoc(HashMap::new()) } let rt = rt.unwrap();
        let res = rt.block_on(async move { docker.prune_images::<String>(Some(PruneImagesOptions { filters })).await });
        match res { Ok(resp) => {
            let mut out = HashMap::new();
            let reclaimed = resp.space_reclaimed.unwrap_or(0);
            out.insert("space_reclaimed".into(), Value::Integer(reclaimed as i64));
            if let Some(deleted) = resp.images_deleted {
                let rows: Vec<Value> = deleted.into_iter().map(|d| {
                    let mut m = HashMap::new();
                    if let Some(u) = d.untagged { m.insert("untagged".into(), Value::String(u)); }
                    if let Some(u) = d.deleted { m.insert("deleted".into(), Value::String(u)); }
                    Value::Assoc(m)
                }).collect();
                out.insert("deleted".into(), Value::List(rows));
            }
            Value::Assoc(out)
        }, Err(_) => Value::Assoc(HashMap::new()) }
    }

    pub fn docker_search_images(dsn: &str, term: String, opts: Value) -> Value {
        use bollard::image::SearchImagesOptions;
        let docker = docker_client(dsn);
        let (limit, filters) = match opts {
            Value::Assoc(m) => {
                let limit = m.get("Limit").and_then(|v| match v { Value::Integer(n)=> Some(*n as u64), _=> None });
                let filters = if let Some(Value::Assoc(fm)) = m.get("Filters") { fm.iter().map(|(k,v)| (k.clone(), match v { Value::List(xs)=> xs.iter().map(|x| match x { Value::String(s)|Value::Symbol(s)=> s.clone(), other=> lyra_core::pretty::format_value(other) }).collect(), other=> vec![lyra_core::pretty::format_value(other)] })).collect() } else { std::collections::HashMap::new() };
                (limit, filters)
            }
            _ => (None, std::collections::HashMap::new())
        };
        let rt = tokio::runtime::Runtime::new(); if rt.is_err() { return Value::Expr { head: Box::new(Value::Symbol("DatasetFromRows".into())), args: vec![Value::List(Vec::new())] } } let rt = rt.unwrap();
        let res = rt.block_on(async move { docker.search_images(SearchImagesOptions { term, limit, filters }).await });
        match res { Ok(items) => {
            let rows: Vec<Value> = items.into_iter().map(|it| {
                let mut m = HashMap::new();
                if let Some(n) = it.name { m.insert("name".into(), Value::String(n)); }
                if let Some(d) = it.description { m.insert("description".into(), Value::String(d)); }
                if let Some(s) = it.star_count { m.insert("stars".into(), Value::Integer(s as i64)); }
                if let Some(o) = it.is_official { m.insert("official".into(), Value::Boolean(o)); }
                if let Some(a) = it.is_automated { m.insert("automated".into(), Value::Boolean(a)); }
                Value::Assoc(m)
            }).collect();
            Value::Expr { head: Box::new(Value::Symbol("DatasetFromRows".into())), args: vec![Value::List(rows)] }
        }, Err(_) => Value::Expr { head: Box::new(Value::Symbol("DatasetFromRows".into())), args: vec![Value::List(Vec::new())] } }
    }

    pub fn docker_image_history(dsn: &str, image: &str) -> Value {
        let docker = docker_client(dsn);
        let name = image.to_string();
        let rt = tokio::runtime::Runtime::new(); if rt.is_err() { return Value::Expr { head: Box::new(Value::Symbol("DatasetFromRows".into())), args: vec![Value::List(Vec::new())] } } let rt = rt.unwrap();
        let res = rt.block_on(async move { docker.image_history(&name).await });
        match res { Ok(items) => {
            let rows: Vec<Value> = items.into_iter().map(|h| {
                let mut m = HashMap::new();
                m.insert("id".into(), Value::String(h.id));
                m.insert("created".into(), Value::Integer(h.created as i64));
                m.insert("size".into(), Value::Integer(h.size as i64));
                m.insert("created_by".into(), Value::String(h.created_by));
                m.insert("tags".into(), Value::List(h.tags.into_iter().map(Value::String).collect()));
                m.insert("comment".into(), Value::String(h.comment));
                Value::Assoc(m)
            }).collect();
            Value::Expr { head: Box::new(Value::Symbol("DatasetFromRows".into())), args: vec![Value::List(rows)] }
        }, Err(_) => Value::Expr { head: Box::new(Value::Symbol("DatasetFromRows".into())), args: vec![Value::List(Vec::new())] } }
    }

    pub fn docker_inspect_registry_image(dsn: &str, image: &str, auth: Value) -> Value {
        let docker = docker_client(dsn);
        let creds = match auth { Value::Assoc(m) => {
            let username = m.get("Username").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None });
            let password = m.get("Password").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None });
            let serveraddress = m.get("Registry").and_then(|v| match v { Value::String(s)|Value::Symbol(s)=>Some(s.clone()), _=>None });
            Some(DockerCredentials { username, password, serveraddress, ..Default::default() })
        }, _ => None };
        let name = image.to_string();
        let rt = tokio::runtime::Runtime::new(); if rt.is_err() { return Value::Assoc(HashMap::new()) } let rt = rt.unwrap();
        let res = rt.block_on(async move { docker.inspect_registry_image(&name, creds).await });
        match res { Ok(info) => {
            let mut m = HashMap::new();
            if let Some(d) = info.descriptor.digest { m.insert("digest".into(), Value::String(d)); }
            {
                let rows: Vec<Value> = info.platforms.into_iter().map(|p| {
                    let mut mm = HashMap::new();
                    if let Some(os) = p.os { mm.insert("os".into(), Value::String(os)); }
                    if let Some(arch) = p.architecture { mm.insert("arch".into(), Value::String(arch)); }
                    Value::Assoc(mm)
                }).collect();
                m.insert("platforms".into(), Value::List(rows));
            }
            Value::Assoc(m)
        }, Err(_) => Value::Assoc(HashMap::new()) }
    }

    pub fn docker_export_images(dsn: &str, images: &[String]) -> Value {
        let docker = docker_client(dsn);
        let names: Vec<String> = images.iter().cloned().collect();
        let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let rt = tokio::runtime::Runtime::new(); if rt.is_err() { return Value::String(String::new()) } let rt = rt.unwrap();
        let res = rt.block_on(async move {
            let mut stream = docker.export_images(&name_refs);
            let mut bytes: Vec<u8> = Vec::new();
            while let Some(chunk) = stream.next().await { let b = chunk.map_err(|_| ())?; bytes.extend_from_slice(&b); }
            Ok::<Vec<u8>, ()>(bytes)
        });
        match res { Ok(data) => Value::String(base64::engine::general_purpose::STANDARD.encode(&data)), Err(_) => Value::String(String::new()) }
    }
}


pub fn register_containers_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str)->bool) {
    register_if(ev, pred, "RuntimeInfo", runtime_info as NativeFn, Attributes::empty());
    register_if(ev, pred, "RuntimeCapabilities", runtime_capabilities as NativeFn, Attributes::empty());
    register_if(ev, pred, "PingContainers", ping_containers as NativeFn, Attributes::empty());
    register_if(ev, pred, "ListContainers", list_containers as NativeFn, Attributes::empty());
    register_if(ev, pred, "DescribeContainers", describe_containers as NativeFn, Attributes::empty());
    register_if(ev, pred, "InspectContainer", inspect_container as NativeFn, Attributes::empty());
    register_if(ev, pred, "StartContainer", start_container as NativeFn, Attributes::empty());
    register_if(ev, pred, "StopContainer", stop_container as NativeFn, Attributes::empty());
    register_if(ev, pred, "RestartContainer", restart_container as NativeFn, Attributes::empty());
    register_if(ev, pred, "PauseContainer", pause_container as NativeFn, Attributes::empty());
    register_if(ev, pred, "UnpauseContainer", unpause_container as NativeFn, Attributes::empty());
    register_if(ev, pred, "RemoveContainer", remove_container as NativeFn, Attributes::empty());
    register_if(ev, pred, "WaitContainer", wait_container as NativeFn, Attributes::empty());
    register_if(ev, pred, "Logs", containers_logs as NativeFn, Attributes::empty());
    register_if(ev, pred, "ExecInContainer", exec_in_container as NativeFn, Attributes::empty());
    register_if(ev, pred, "CreateContainer", create_container as NativeFn, Attributes::empty());
    register_if(ev, pred, "RunContainer", run_container as NativeFn, Attributes::empty());
    register_if(ev, pred, "ConnectContainers", connect_containers as NativeFn, Attributes::empty());
    register_if(ev, pred, "DisconnectContainers", disconnect_containers as NativeFn, Attributes::empty());
    register_if(ev, pred, "ListNetworks", list_networks as NativeFn, Attributes::empty());
    register_if(ev, pred, "CreateNetwork", create_network as NativeFn, Attributes::empty());
    register_if(ev, pred, "RemoveNetwork", remove_network as NativeFn, Attributes::empty());
    register_if(ev, pred, "ListVolumes", list_volumes as NativeFn, Attributes::empty());
    register_if(ev, pred, "CreateVolume", create_volume as NativeFn, Attributes::empty());
    register_if(ev, pred, "RemoveVolume", remove_volume as NativeFn, Attributes::empty());
    register_if(ev, pred, "CopyToContainer", copy_to_container as NativeFn, Attributes::empty());
    register_if(ev, pred, "CopyFromContainer", copy_from_container as NativeFn, Attributes::empty());
    register_if(ev, pred, "SearchImages", search_images as NativeFn, Attributes::empty());
    register_if(ev, pred, "ListImages", list_images as NativeFn, Attributes::empty());
    register_if(ev, pred, "InspectImage", inspect_image as NativeFn, Attributes::empty());
    register_if(ev, pred, "PullImage", pull_image as NativeFn, Attributes::empty());
    register_if(ev, pred, "PushImage", push_image as NativeFn, Attributes::empty());
    register_if(ev, pred, "SaveImage", save_image as NativeFn, Attributes::empty());
    register_if(ev, pred, "LoadImage", load_image as NativeFn, Attributes::empty());
    register_if(ev, pred, "RemoveImage", remove_image as NativeFn, Attributes::empty());
    register_if(ev, pred, "PruneImages", prune_images as NativeFn, Attributes::empty());
    register_if(ev, pred, "ImageHistory", image_history as NativeFn, Attributes::empty());
    register_if(ev, pred, "ExportImages", export_images as NativeFn, Attributes::empty());
    register_if(ev, pred, "TagImage", tag_image as NativeFn, Attributes::empty());
    register_if(ev, pred, "AddRegistryAuth", add_registry_auth as NativeFn, Attributes::empty());
    register_if(ev, pred, "ListRegistryAuth", list_registry_auth as NativeFn, Attributes::empty());
    register_if(ev, pred, "InspectRegistryImage", inspect_registry_image as NativeFn, Attributes::empty());
    register_if(ev, pred, "Events", containers_events as NativeFn, Attributes::empty());
    register_if(ev, pred, "Stats", containers_stats as NativeFn, Attributes::empty());
}
