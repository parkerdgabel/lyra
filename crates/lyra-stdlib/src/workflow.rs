use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
use std::collections::HashMap;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

/// Register minimal sequential workflows via `Workflow[{...}]`.
pub fn register_workflow(ev: &mut Evaluator) {
    ev.register("Workflow", workflow as NativeFn, Attributes::HOLD_ALL);
    ev.register("Task", task as NativeFn, Attributes::HOLD_ALL);
    ev.register("RunTasks", run_tasks as NativeFn, Attributes::HOLD_ALL);
    ev.register("ExplainTasks", explain_tasks as NativeFn, Attributes::empty());
    ev.register("__RunTask", run_task_internal as NativeFn, Attributes::HOLD_ALL);
}

/// Conditionally register workflow helpers based on `pred`.
pub fn register_workflow_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    super::register_if(ev, pred, "Workflow", workflow as NativeFn, Attributes::HOLD_ALL);
    super::register_if(ev, pred, "Task", task as NativeFn, Attributes::HOLD_ALL);
    super::register_if(ev, pred, "RunTasks", run_tasks as NativeFn, Attributes::HOLD_ALL);
    super::register_if(ev, pred, "ExplainTasks", explain_tasks as NativeFn, Attributes::empty());
    super::register_if(ev, pred, "__RunTask", run_task_internal as NativeFn, Attributes::HOLD_ALL);
}

// Minimal sequential workflow runner: Workflow[{step1, step2, ...}] or Workflow[{<|"name"->..., "run"->expr|>, ...}]
fn workflow(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("Workflow".into())), args };
    }
    let steps = match &args[0] {
        Value::List(items) => items.clone(),
        _ => return Value::List(vec![]),
    };
    let mut results: Vec<Value> = Vec::new();
    for step in steps {
        let (name, run_expr) = match &step {
            Value::Assoc(m) => {
                let nm = m
                    .get("name")
                    .and_then(|v| match v {
                        Value::String(s) | Value::Symbol(s) => Some(s.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| "step".into());
                let run = m.get("run").cloned().unwrap_or(Value::Symbol("Null".into()));
                (nm, run)
            }
            other => ("step".into(), other.clone()),
        };
        // Optional trace span per step
        let _ = ev.eval(Value::Expr {
            head: Box::new(Value::Symbol("Span".into())),
            args: vec![
                Value::String("workflow:step".into()),
                Value::Assoc(HashMap::from([(String::from("Name"), Value::String(name))])),
            ],
        });
        let res = ev.eval(run_expr);
        let _ =
            ev.eval(Value::Expr { head: Box::new(Value::Symbol("SpanEnd".into())), args: vec![] });
        results.push(res);
    }
    Value::List(results)
}

// ---- Phase 1: Tasks DAG (normalize, explain, run sequential respecting deps) ----

#[derive(Clone)]
struct TaskRec {
    name: String,
    run: Value,
    depends: Vec<String>,
    when: Option<Value>,
    timeout_ms: Option<u64>,
    retry: Option<RetrySpec>,
}

#[derive(Clone)]
struct RetrySpec { max: i64, backoff_ms: u64, exponential: bool, jitter: bool }

fn as_string(v: &Value) -> Option<String> {
    match v { Value::String(s) | Value::Symbol(s) => Some(s.clone()), _ => None }
}

fn normalize_task(ev: &mut Evaluator, v: Value, idx: usize) -> Option<TaskRec> {
    match ev.eval(v) {
        Value::Assoc(m) => {
            let name = m.get("name").and_then(as_string).unwrap_or_else(|| format!("task{}", idx+1));
            let run = m.get("run").cloned().unwrap_or(Value::Symbol("Null".into()));
            let when = m.get("when").cloned();
            let depends: Vec<String> = match m.get("dependsOn").or_else(|| m.get("depends")) {
                Some(Value::List(vs)) => vs.iter().filter_map(|x| as_string(x)).collect(),
                Some(Value::String(s)) | Some(Value::Symbol(s)) => vec![s.clone()],
                _ => Vec::new(),
            };
            // timeoutMs
            let timeout_ms = m.get("timeoutMs").and_then(|v| if let Value::Integer(n)=v { Some((*n).max(0) as u64) } else { None });
            // retries: <|max, backoffMs, exponential, jitter|> or integer
            let retry = match m.get("retries") {
                Some(Value::Integer(n)) => Some(RetrySpec { max: (*n).max(0), backoff_ms: 200, exponential: true, jitter: true }),
                Some(Value::Assoc(rm)) => {
                    let max = rm.get("max").and_then(|v| if let Value::Integer(n)=v { Some(*n) } else { None }).unwrap_or(0).max(0);
                    let backoff_ms = rm.get("backoffMs").and_then(|v| if let Value::Integer(n)=v { Some((*n).max(0) as u64) } else { None }).unwrap_or(200);
                    let exponential = rm.get("exponential").and_then(|v| if let Value::Boolean(b)=v { Some(*b) } else { None }).unwrap_or(true);
                    let jitter = rm.get("jitter").and_then(|v| if let Value::Boolean(b)=v { Some(*b) } else { None }).unwrap_or(true);
                    Some(RetrySpec { max, backoff_ms, exponential, jitter })
                }
                _ => None,
            };
            Some(TaskRec { name, run, depends, when, timeout_ms, retry })
        }
        other => Some(TaskRec { name: format!("task{}", idx+1), run: other, depends: Vec::new(), when: None, timeout_ms: None, retry: None }),
    }
}

fn task(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::Expr { head: Box::new(Value::Symbol("Task".into())), args }; }
    if let Some(t) = normalize_task(ev, args[0].clone(), 0) {
        return Value::Assoc(HashMap::from([
            (String::from("name"), Value::String(t.name)),
            (String::from("run"), t.run),
            (String::from("dependsOn"), Value::List(t.depends.into_iter().map(Value::String).collect())),
        ]));
    }
    Value::Expr { head: Box::new(Value::Symbol("Task".into())), args }
}

fn build_graph(ev: &mut Evaluator, list_v: Value) -> Result<(Vec<TaskRec>, Vec<(usize,usize)>, Vec<usize>), String> {
    let items = match ev.eval(list_v) { Value::List(vs) => vs, _ => return Err("RunTasks expects a list of tasks".into()) };
    let mut tasks: Vec<TaskRec> = Vec::new();
    for (i, it) in items.into_iter().enumerate() {
        if let Some(t) = normalize_task(ev, it, i) { tasks.push(t); }
    }
    // map name -> index
    let mut index: HashMap<String, usize> = HashMap::new();
    for (i, t) in tasks.iter().enumerate() { index.insert(t.name.clone(), i); }
    // edges: dep -> task
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for (i, t) in tasks.iter().enumerate() {
        for d in &t.depends { if let Some(&j) = index.get(d) { edges.push((j, i)); } else { return Err(format!("unknown dependency '{}' for task '{}'", d, t.name)); } }
    }
    // Kahn topo sort
    let n = tasks.len();
    let mut indeg = vec![0usize; n];
    for &(_u, v) in &edges { indeg[v] += 1; }
    let mut q: std::collections::VecDeque<usize> = indeg.iter().enumerate().filter_map(|(i,&d)| if d==0 { Some(i) } else { None }).collect();
    let mut order: Vec<usize> = Vec::new();
    let mut out_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(u,v) in &edges { out_edges[u].push(v); }
    while let Some(u) = q.pop_front() { order.push(u); for &v in &out_edges[u] { indeg[v] -= 1; if indeg[v]==0 { q.push_back(v); } } }
    if order.len() != n { return Err("cycle detected in task dependencies".into()); }
    Ok((tasks, edges, order))
}

fn explain_tasks(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("ExplainTasks".into())), args }; }
    match build_graph(ev, args[0].clone()) {
        Ok((tasks, edges, order)) => {
            let nodes = Value::List(tasks.iter().map(|t| Value::String(t.name.clone())).collect());
            let edges_v = Value::List(edges.into_iter().map(|(u,v)| Value::List(vec![Value::Integer(u as i64), Value::Integer(v as i64)])).collect());
            let order_v = Value::List(order.into_iter().map(|i| Value::Integer(i as i64)).collect());
            Value::Assoc(HashMap::from([
                (String::from("nodes"), nodes),
                (String::from("edges"), edges_v),
                (String::from("order"), order_v),
            ]))
        }
        Err(e) => Value::Assoc(HashMap::from([(String::from("message"), Value::String(e)), (String::from("tag"), Value::String(String::from("Workflow::explain")))]))
    }
}

fn run_tasks(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("RunTasks".into())), args }; }
    let opts = if args.len()>=2 { ev.eval(args[1].clone()) } else { Value::Assoc(HashMap::new()) };
    let continue_on_error = matches!(&opts, Value::Assoc(m) if matches!(m.get("onError"), Some(Value::String(s)) if s.eq_ignore_ascii_case("continue")) || matches!(m.get("onError"), Some(Value::Symbol(s)) if s.eq_ignore_ascii_case("continue")));
    let max_conc: usize = match &opts { Value::Assoc(m) => m.get("maxConcurrency").and_then(|v| if let Value::Integer(n)=v { Some((*n).max(1) as usize) } else { None }).unwrap_or(1), _ => 1 };
    let want_stream: bool = matches!(&opts, Value::Assoc(m) if matches!(m.get("stream"), Some(Value::Boolean(true))));
    let on_event_cb: Option<Value> = match &opts { Value::Assoc(m) => m.get("onEvent").cloned(), _ => None };
    match build_graph(ev, args[0].clone()) {
        Ok((tasks, edges, _order)) => {
            // Hooks from opts
            let (before_all, after_all, before_each, after_each) = match &opts {
                Value::Assoc(m) => (
                    m.get("beforeAll").cloned(),
                    m.get("afterAll").cloned(),
                    m.get("beforeEach").cloned(),
                    m.get("afterEach").cloned(),
                ),
                _ => (None, None, None, None),
            };
            // Build indegree and out_edges
            let n = tasks.len();
            let mut indeg = vec![0usize; n];
            let mut out_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
            for &(u,v) in &edges { indeg[v]+=1; out_edges[u].push(v); }
            let mut ready: std::collections::VecDeque<usize> = indeg.iter().enumerate().filter_map(|(i,&d)| if d==0 { Some(i) } else { None }).collect();
            // Create event channel if requested
            let mut events_val: Option<Value> = None;
            if want_stream {
                let ch = ev.eval(Value::Expr { head: Box::new(Value::Symbol("BoundedChannel".into())), args: vec![Value::Integer(256)] });
                events_val = Some(ch);
            }
            let mut results: Vec<(String, Value)> = Vec::new();
            let mut status: HashMap<String, String> = HashMap::new();
            // Run beforeAll hook
            if let Some(h) = before_all.clone() { let _ = ev.eval(h); }
            // Scheduling loop
            while !ready.is_empty() {
                let mut batch: Vec<usize> = Vec::new();
                for _ in 0..max_conc { if let Some(i) = ready.pop_front() { batch.push(i); } }
                if batch.is_empty() { break; }
                // Build __RunTask calls
                let task_vals: Vec<Value> = batch.iter().map(|&i| {
                    let t = &tasks[i];
                    // when check eager in parent; skip if false
                    Value::Expr { head: Box::new(Value::Symbol("__RunTask".into())), args: vec![Value::Assoc({
                        let mut mm = HashMap::new();
                        mm.insert(String::from("name"), Value::String(t.name.clone()));
                        mm.insert(String::from("run"), t.run.clone());
                        if let Some(ms)=t.timeout_ms { mm.insert(String::from("timeoutMs"), Value::Integer(ms as i64)); }
                        if let Some(r)=t.retry.as_ref() { mm.insert(String::from("retries"), Value::Assoc(HashMap::from([(String::from("max"), Value::Integer(r.max)), (String::from("backoffMs"), Value::Integer(r.backoff_ms as i64)), (String::from("exponential"), Value::Boolean(r.exponential)), (String::from("jitter"), Value::Boolean(r.jitter))]))); }
                        if let Some(ch)=events_val.clone() { mm.insert(String::from("events"), ch); }
                        if let Some(cb)=on_event_cb.clone() { mm.insert(String::from("onEvent"), cb); }
                        if let Some(cond)=t.when.clone() { mm.insert(String::from("when"), cond); }
                        if let Some(h)=before_each.clone() { mm.insert(String::from("beforeEach"), h); }
                        if let Some(h)=after_each.clone() { mm.insert(String::from("afterEach"), h); }
                        mm
                    })] }
                }).collect();
                let pv_opts = Value::Assoc(HashMap::from([(String::from("MaxThreads"), Value::Integer(max_conc as i64))]));
                let out_v = if max_conc>1 { ev.eval(Value::Expr { head: Box::new(Value::Symbol("ParallelEvaluate".into())), args: vec![Value::List(task_vals.clone()), pv_opts] }) } else { // sequential
                    let mut outs: Vec<Value> = Vec::new();
                    for call in task_vals { outs.push(ev.eval(call)); }
                    Value::List(outs)
                };
                // Process results
                let mut batch_failed = false;
                if let Value::List(outs) = out_v {
                    for (j, ov) in outs.into_iter().enumerate() {
                        let i = batch[j];
                        match ov {
                            Value::Assoc(m) => {
                                let name = m.get("name").and_then(|v| as_string(v)).unwrap_or(tasks[i].name.clone());
                                let val = m.get("value").cloned().unwrap_or(Value::Symbol("Null".into()));
                                let st = m.get("status").and_then(|v| as_string(v)).unwrap_or("success".into());
                                results.push((name.clone(), val));
                                status.insert(name.clone(), st.clone());
                                // unlock successors
                                for &v in &out_edges[i] { indeg[v]-=1; if indeg[v]==0 { ready.push_back(v); } }
                                if st=="failed" { batch_failed = true; }
                            }
                            other => {
                                let name = tasks[i].name.clone();
                                results.push((name.clone(), other));
                                status.insert(name.clone(), "success".into());
                                for &v in &out_edges[i] { indeg[v]-=1; if indeg[v]==0 { ready.push_back(v); } }
                            }
                        }
                    }
                }
                if batch_failed && !continue_on_error { break; }
            }
            // afterAll hook with summary
            let summary = Value::Assoc(HashMap::from([
                (String::from("total"), Value::Integer(status.len() as i64)),
                (String::from("success"), Value::Integer(status.values().filter(|s| s.as_str()=="success").count() as i64)),
                (String::from("failed"), Value::Integer(status.values().filter(|s| s.as_str()=="failed").count() as i64)),
                (String::from("skipped"), Value::Integer(status.values().filter(|s| s.as_str()=="skipped").count() as i64)),
            ]));
            if let Some(h) = after_all.clone() { let _ = ev.eval(Value::Expr { head: Box::new(Value::Symbol("With".into())), args: vec![Value::Assoc(HashMap::from([(String::from("SUMMARY"), summary.clone())])), h] }); }
            // Close events channel if any
            if let Some(ch) = events_val.clone() { let _ = ev.eval(Value::Expr { head: Box::new(Value::Symbol("CloseChannel".into())), args: vec![ch] }); }
            Value::Assoc(HashMap::from([
                (String::from("results"), Value::List(results.into_iter().map(|(n,v)| Value::Assoc(HashMap::from([(String::from("name"), Value::String(n)), (String::from("value"), v)]))).collect())),
                (String::from("status"), Value::Assoc(status.into_iter().map(|(k,v)| (k, Value::String(v))).collect())),
                (String::from("events"), events_val.unwrap_or(Value::Symbol("Null".into()))),
            ]))
        }
        Err(e) => Value::Assoc(HashMap::from([(String::from("message"), Value::String(e)), (String::from("tag"), Value::String(String::from("Workflow::run")))]))
    }
}

fn truthy(v: &Value) -> bool { match v { Value::Boolean(b) => *b, Value::Integer(n) => *n != 0, Value::Real(f) => *f != 0.0, Value::String(s) => !s.is_empty(), Value::Symbol(s) => !s.is_empty() && s != "Null" && s != "False", Value::Assoc(m) => !m.is_empty(), Value::List(vs) => !vs.is_empty(), _ => true } }

fn rand_jitter() -> u64 {
    // Simple, deterministic-ish jitter without pulling in extra deps beyond optional rand feature (not always enabled)
    // Use a time-based hash
    use std::time::{SystemTime, UNIX_EPOCH};
    let ns = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.subsec_nanos() as u64).unwrap_or(0);
    ns ^ (ns.rotate_left(13)) ^ 0x9e3779b97f4a7c15
}

// Internal helper: run a single task with retries/timeout; returns <|name, value, status|> and emits events to channel if provided
fn run_task_internal(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("__RunTask".into())), args }; }
    let spec = ev.eval(args[0].clone());
    let mut name = String::from("task");
    let mut run = Value::Symbol("Null".into());
    let mut timeout_ms: Option<u64> = None;
    let mut retry: Option<RetrySpec> = None;
    let mut events: Option<Value> = None;
    let mut when_cond: Option<Value> = None;
    let mut before_each: Option<Value> = None;
    let mut after_each: Option<Value> = None;
    let mut on_event: Option<Value> = None;
    if let Value::Assoc(m) = spec {
        if let Some(s) = m.get("name").and_then(|v| as_string(v)) { name = s; }
        if let Some(v) = m.get("run") { run = v.clone(); }
        if let Some(Value::Integer(n)) = m.get("timeoutMs") { timeout_ms = Some((*n).max(0) as u64); }
        if let Some(Value::Assoc(rm)) = m.get("retries") {
            let max = rm.get("max").and_then(|v| if let Value::Integer(n)=v { Some(*n) } else { None }).unwrap_or(0).max(0);
            let backoff_ms = rm.get("backoffMs").and_then(|v| if let Value::Integer(n)=v { Some((*n).max(0) as u64) } else { None }).unwrap_or(200);
            let exponential = rm.get("exponential").and_then(|v| if let Value::Boolean(b)=v { Some(*b) } else { None }).unwrap_or(true);
            let jitter = rm.get("jitter").and_then(|v| if let Value::Boolean(b)=v { Some(*b) } else { None }).unwrap_or(true);
            retry = Some(RetrySpec { max, backoff_ms, exponential, jitter });
        }
        if let Some(v) = m.get("events") { events = Some(v.clone()); }
        if let Some(v) = m.get("when") { when_cond = Some(v.clone()); }
        if let Some(v) = m.get("beforeEach") { before_each = Some(v.clone()); }
        if let Some(v) = m.get("afterEach") { after_each = Some(v.clone()); }
        if let Some(v) = m.get("onEvent") { on_event = Some(v.clone()); }
    }
    // when condition
    if let Some(cond) = when_cond { if !truthy(&ev.eval(cond)) { return Value::Assoc(HashMap::from([(String::from("name"), Value::String(name)), (String::from("value"), Value::Symbol("Null".into())), (String::from("status"), Value::String(String::from("skipped")))])); } }
    let mut attempt: i64 = 0;
    let max_attempts: i64 = retry.as_ref().map(|r| r.max + 1).unwrap_or(1);
    let mut final_res: Value = Value::Symbol("Null".into());
    let mut final_failed: bool = false;
    while attempt < max_attempts {
        attempt += 1;
        // Send start + beforeEach hook
        if let Some(ch) = events.clone() { let _ = ev.eval(Value::Expr { head: Box::new(Value::Symbol("Send".into())), args: vec![ch, Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("taskStart"))), (String::from("task"), Value::String(name.clone())), (String::from("attempt"), Value::Integer(attempt))]))] }); }
        if let Some(cb) = on_event.clone() { let _ = ev.eval(Value::Expr { head: Box::new(cb), args: vec![Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("taskStart"))), (String::from("task"), Value::String(name.clone())), (String::from("attempt"), Value::Integer(attempt))]))] }); }
        if let Some(h)=before_each.clone() { let _ = ev.eval(h); }
        // Span and timeout
        let _ = ev.eval(Value::Expr { head: Box::new(Value::Symbol("Span".into())), args: vec![Value::String("task".into()), Value::Assoc(HashMap::from([(String::from("Name"), Value::String(name.clone())), (String::from("Attempt"), Value::Integer(attempt))]))] });
        let start = std::time::Instant::now();
        let res = if let Some(ms) = timeout_ms { let opts = Value::Assoc(HashMap::from([(String::from("timeBudgetMs"), Value::Integer(ms as i64))])); ev.eval(Value::Expr { head: Box::new(Value::Symbol("Scope".into())), args: vec![opts, run.clone()] }) } else { ev.eval(run.clone()) };
        let dur = start.elapsed();
        let _ = ev.eval(Value::Expr { head: Box::new(Value::Symbol("SpanEnd".into())), args: vec![] });
        // Determine failure
        let is_error_tag = matches!(&res, Value::Assoc(m) if m.get("tag").and_then(|v| as_string(v)).map(|s| s.to_ascii_lowercase()).unwrap_or_default().contains("error"));
        let soft_timeout = timeout_ms.map(|ms| dur.as_millis() as u64 > ms).unwrap_or(false);
        if !is_error_tag && !soft_timeout { final_res = res; final_failed = false; }
        else {
            final_res = if soft_timeout { Value::Assoc(HashMap::from([(String::from("tag"), Value::String(String::from("Workflow::timeout"))), (String::from("message"), Value::String(format!("task '{}' exceeded timeout {}ms", name, timeout_ms.unwrap())))])) } else { res };
            final_failed = true;
        }
        // Send end + afterEach hook
        if let Some(ch) = events.clone() {
            let _ = ev.eval(Value::Expr { head: Box::new(Value::Symbol("Send".into())), args: vec![ch, Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("taskEnd"))), (String::from("task"), Value::String(name.clone())), (String::from("attempt"), Value::Integer(attempt)), (String::from("status"), Value::String(if final_failed { "failed".into() } else { "success".into() }))]))] });
        }
        if let Some(cb) = on_event.clone() { let _ = ev.eval(Value::Expr { head: Box::new(cb), args: vec![Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("taskEnd"))), (String::from("task"), Value::String(name.clone())), (String::from("attempt"), Value::Integer(attempt)), (String::from("status"), Value::String(if final_failed { String::from("failed") } else { String::from("success") }))]))] }); }
        if let Some(h)=after_each.clone() { let _ = ev.eval(h); }
        if !final_failed { break; }
        if attempt < max_attempts {
            if let Some(r) = &retry { let mut delay = r.backoff_ms; if r.exponential { delay = delay.saturating_mul(2u64.pow((attempt as u32)-1)); } if r.jitter { let j = (delay / 2).max(1); let extra = (rand_jitter() % j) as u64; delay = delay.saturating_sub(j) + extra; } std::thread::sleep(std::time::Duration::from_millis(delay)); }
        }
    }
    Value::Assoc(HashMap::from([
        (String::from("name"), Value::String(name)),
        (String::from("value"), final_res),
        (String::from("status"), Value::String(if final_failed { String::from("failed") } else { String::from("success") })),
    ]))
}
