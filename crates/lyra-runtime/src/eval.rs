use lyra_core::value::Value;
use std::collections::HashMap;
use std::thread::{self, JoinHandle};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use crate::attrs::Attributes;
use lyra_core::schema::schema_of;
use std::sync::OnceLock;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

struct TaskInfo {
    handle: JoinHandle<Value>,
    cancel: Arc<AtomicBool>,
}

pub struct Evaluator {
    builtins: HashMap<String, (NativeFn, Attributes)>,
    env: HashMap<String, Value>,
    tasks: HashMap<i64, TaskInfo>, // minimal future registry
    next_task_id: i64,
    cancel_token: Option<Arc<AtomicBool>>, // cooperative cancellation
    trace_enabled: bool,
    trace_steps: Vec<Value>,
}

static DEFAULT_REGISTRAR: OnceLock<fn(&mut Evaluator)> = OnceLock::new();

pub fn set_default_registrar(f: fn(&mut Evaluator)) {
    let _ = DEFAULT_REGISTRAR.set(f);
}

impl Evaluator {
    pub fn new() -> Self {
        let mut ev = Self { builtins: HashMap::new(), env: HashMap::new(), tasks: HashMap::new(), next_task_id: 1, cancel_token: None, trace_enabled: false, trace_steps: Vec::new() };
        register_compat_prelude(&mut ev);
        if let Some(f) = DEFAULT_REGISTRAR.get().copied() { f(&mut ev); }
        ev
    }

    pub fn with_env(env: HashMap<String, Value>) -> Self { let mut ev = Self::new(); ev.env = env; ev }

    pub fn with_env_and_token(env: HashMap<String, Value>, token: Arc<AtomicBool>) -> Self { let mut ev = Self::new(); ev.env = env; ev.cancel_token = Some(token); ev }

    pub fn register(&mut self, name: &str, f: NativeFn, attrs: Attributes) {
        self.builtins.insert(name.to_string(), (f, attrs));
    }

    pub fn eval(&mut self, v: Value) -> Value {
        if let Some(tok) = &self.cancel_token { if tok.load(Ordering::Relaxed) { return cancelled_failure(); } }
        match v {
            Value::Expr { head, args } => {
                let head_eval = self.eval(*head);
                // Determine function name
                if let Value::PureFunction { params, body } = head_eval.clone() {
                    let eval_args: Vec<Value> = args.into_iter().map(|a| self.eval(a)).collect();
                    let applied = apply_pure_function(*body, params.as_ref(), &eval_args);
                    return self.eval(applied);
                }
                let fname = match &head_eval { Value::Symbol(s) => s.clone(), _ => return Value::Expr { head: Box::new(head_eval), args } };
                // Listable threading
                let (fun, attrs) = match self.builtins.get(&fname) { Some(t) => (t.0, t.1), None => return Value::Expr { head: Box::new(Value::Symbol(fname)), args } };
                if attrs.contains(Attributes::LISTABLE) {
                    if args.iter().any(|a| matches!(a, Value::List(_))) {
                        if self.trace_enabled {
                            let count = args.iter().filter_map(|a| if let Value::List(v)=a { Some(v.len() as i64) } else { None }).max().unwrap_or(0);
                            let arg_lens: Vec<Value> = args.iter().map(|a| match a { Value::List(v)=>Value::Integer(v.len() as i64), _=>Value::Integer(0) }).collect();
                            self.trace_steps.push(Value::Assoc(vec![
                                ("action".to_string(), Value::String("ListableThread".into())),
                                ("head".to_string(), head_eval.clone()),
                                ("count".to_string(), Value::Integer(count)),
                                ("argLens".to_string(), Value::List(arg_lens)),
                            ].into_iter().collect()));
                        }
                        return listable_thread(self, fun, args);
                    }
                }
                // Evaluate arguments respecting Hold* attributes
                if self.trace_enabled {
                    let held: Vec<i64> = if attrs.contains(Attributes::HOLD_ALL) {
                        (1..=args.len()).map(|i| i as i64).collect()
                    } else if attrs.contains(Attributes::HOLD_FIRST) {
                        if args.is_empty() { vec![] } else { vec![1] }
                    } else if attrs.contains(Attributes::HOLD_REST) {
                        if args.len()<=1 { vec![] } else { (2..=args.len()).map(|i| i as i64).collect() }
                    } else { vec![] };
                    if !held.is_empty() {
                        self.trace_steps.push(Value::Assoc(vec![
                            ("action".to_string(), Value::String("Hold".into())),
                            ("head".to_string(), head_eval.clone()),
                            ("held".to_string(), Value::List(held.into_iter().map(Value::Integer).collect())),
                        ].into_iter().collect()));
                    }
                }
                let mut eval_args: Vec<Value> = if attrs.contains(Attributes::HOLD_ALL) {
                    args
                } else if attrs.contains(Attributes::HOLD_FIRST) {
                    let mut out = Vec::with_capacity(args.len());
                    let mut it = args.into_iter();
                    if let Some(first) = it.next() { out.push(first); }
                    for a in it { out.push(self.eval(a)); }
                    out
                } else if attrs.contains(Attributes::HOLD_REST) {
                    let mut out = Vec::with_capacity(args.len());
                    let mut it = args.into_iter();
                    if let Some(first) = it.next() { out.push(self.eval(first)); }
                    for a in it { out.push(a); }
                    out
                } else {
                    args.into_iter().map(|a| self.eval(a)).collect()
                };
                // Sequence splicing at top-level arguments
                eval_args = splice_sequences(eval_args);
                // Flat: flatten same-head nested calls in arguments
                if attrs.contains(Attributes::FLAT) {
                    let mut flat: Vec<Value> = Vec::with_capacity(eval_args.len());
                    for a in eval_args.into_iter() {
                        if let Value::Expr { head: h2, args: a2 } = &a {
                            if matches!(&**h2, Value::Symbol(s) if s == &fname) {
                                if self.trace_enabled {
                                    self.trace_steps.push(Value::Assoc(vec![
                                        ("action".to_string(), Value::String("FlatFlatten".into())),
                                        ("head".to_string(), head_eval.clone()),
                                        ("added".to_string(), Value::Integer(a2.len() as i64)),
                                    ].into_iter().collect()));
                                }
                                flat.extend(a2.clone());
                                continue;
                            }
                        }
                        flat.push(a);
                    }
                    eval_args = flat;
                }
                // Orderless: canonical sort of args
                if attrs.contains(Attributes::ORDERLESS) {
                    eval_args.sort_by(|x,y| value_order(x).cmp(&value_order(y)));
                    if self.trace_enabled {
                        self.trace_steps.push(Value::Assoc(vec![
                            ("action".to_string(), Value::String("OrderlessSort".into())),
                            ("head".to_string(), head_eval.clone()),
                            ("finalOrder".to_string(), Value::List(eval_args.clone())),
                        ].into_iter().collect()));
                    }
                }
                fun(self, eval_args)
            }
            Value::List(items) => Value::List(items.into_iter().map(|x| self.eval(x)).collect()),
            Value::Assoc(m) => Value::Assoc(m.into_iter().map(|(k,v)|(k,self.eval(v))).collect()),
            Value::Symbol(s) => self.env.get(&s).cloned().unwrap_or(Value::Symbol(s)),
            other => other,
        }
    }
}

// Temporary: compatibility registration (to be removed once stdlib migration completes)
pub fn register_compat_prelude(ev: &mut Evaluator) {
    // Minimal math kept for internal threads (Future/ParallelMap) until full migration
    // No math builtins by default; stdlib should register
    ev.register("Map", map as NativeFn, Attributes::empty());
    ev.register("If", iff as NativeFn, Attributes::HOLD_REST);
    ev.register("Equal", equal as NativeFn, Attributes::LISTABLE);
    ev.register("Less", less as NativeFn, Attributes::LISTABLE);
    ev.register("LessEqual", less_equal as NativeFn, Attributes::LISTABLE);
    ev.register("Greater", greater as NativeFn, Attributes::LISTABLE);
    ev.register("GreaterEqual", greater_equal as NativeFn, Attributes::LISTABLE);
    ev.register("And", and_fn as NativeFn, Attributes::empty());
    ev.register("Or", or_fn as NativeFn, Attributes::empty());
    ev.register("Not", not_fn as NativeFn, Attributes::empty());
    // List/Assoc utilities
    ev.register("Apply", apply as NativeFn, Attributes::empty());
    ev.register("Total", total as NativeFn, Attributes::empty());
    ev.register("Fold", fold as NativeFn, Attributes::empty());
    ev.register("Select", select as NativeFn, Attributes::empty());
    ev.register("Filter", select as NativeFn, Attributes::empty());
    ev.register("Keys", keys as NativeFn, Attributes::empty());
    ev.register("Values", values as NativeFn, Attributes::empty());
    ev.register("Lookup", lookup as NativeFn, Attributes::empty());
    ev.register("AssociationMap", association_map as NativeFn, Attributes::empty());
    ev.register("AssociationMapKeys", association_map_keys as NativeFn, Attributes::empty());
    ev.register("AssociationMapKV", association_map_kv as NativeFn, Attributes::empty());
    ev.register("AssociationMapPairs", association_map_pairs as NativeFn, Attributes::empty());
    ev.register("Merge", merge as NativeFn, Attributes::empty());
    ev.register("KeySort", key_sort as NativeFn, Attributes::empty());
    ev.register("SortBy", sort_by as NativeFn, Attributes::empty());
    ev.register("GroupBy", group_by as NativeFn, Attributes::empty());
    ev.register("Flatten", flatten as NativeFn, Attributes::empty());
    ev.register("Thread", thread as NativeFn, Attributes::HOLD_ALL);
    ev.register("Partition", partition as NativeFn, Attributes::empty());
    ev.register("Transpose", transpose as NativeFn, Attributes::empty());
    ev.register("Replace", replace as NativeFn, Attributes::HOLD_ALL);
    ev.register("ReplaceAll", replace_all_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("ReplaceFirst", replace_first as NativeFn, Attributes::HOLD_ALL);
    ev.register("Set", set_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("With", with_fn as NativeFn, Attributes::HOLD_ALL);
    // Concurrency primitives (minimal Phase 1)
    ev.register("Future", future_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Await", await_fn as NativeFn, Attributes::empty());
    ev.register("ParallelMap", parallel_map as NativeFn, Attributes::empty());
    ev.register("MapAsync", map_async as NativeFn, Attributes::empty());
    ev.register("Gather", gather as NativeFn, Attributes::empty());
    ev.register("BusyWait", busy_wait as NativeFn, Attributes::empty());
    ev.register("Cancel", cancel_fn as NativeFn, Attributes::empty());
    ev.register("Fail", fail_fn as NativeFn, Attributes::empty());
    // Phase 0 additions
    ev.register("Schema", schema_fn as NativeFn, Attributes::empty());
    ev.register("Explain", explain_fn as NativeFn, Attributes::HOLD_ALL);
    // Stdlib v0 extras
    ev.register("Length", length as NativeFn, Attributes::empty());
    ev.register("Range", range as NativeFn, Attributes::empty());
    ev.register("Join", join as NativeFn, Attributes::empty());
    ev.register("Reverse", reverse as NativeFn, Attributes::empty());
    // Small stdlib v0 helpers
    ev.register("EvenQ", even_q as NativeFn, Attributes::LISTABLE);
    ev.register("OddQ", odd_q as NativeFn, Attributes::LISTABLE);
    // String builtins moved to lyra-stdlib
    // Math builtins moved to lyra-stdlib
    // String builtins moved to lyra-stdlib
    // Echo helpers for attribute tests
    ev.register("OrderlessEcho", orderless_echo as NativeFn, Attributes::ORDERLESS | Attributes::HOLD_ALL);
    ev.register("FlatEcho", flat_echo as NativeFn, Attributes::FLAT | Attributes::HOLD_ALL);
    ev.register("FlatOrderlessEcho", flat_orderless_echo as NativeFn, Attributes::FLAT | Attributes::ORDERLESS | Attributes::HOLD_ALL);
}

pub fn evaluate(v: Value) -> Value { Evaluator::new().eval(v) }

// Public registration helpers for stdlib wrappers
pub fn register_concurrency(ev: &mut Evaluator) {
    ev.register("Future", future_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Await", await_fn as NativeFn, Attributes::empty());
    ev.register("ParallelMap", parallel_map as NativeFn, Attributes::empty());
    ev.register("MapAsync", map_async as NativeFn, Attributes::empty());
    ev.register("Gather", gather as NativeFn, Attributes::empty());
    ev.register("BusyWait", busy_wait as NativeFn, Attributes::empty());
    ev.register("Cancel", cancel_fn as NativeFn, Attributes::empty());
    ev.register("Fail", fail_fn as NativeFn, Attributes::empty());
}

pub fn register_explain(ev: &mut Evaluator) {
    ev.register("Explain", explain_fn as NativeFn, Attributes::HOLD_ALL);
}

pub fn register_schema(ev: &mut Evaluator) {
    ev.register("Schema", schema_fn as NativeFn, Attributes::empty());
}

fn listable_thread(ev: &mut Evaluator, f: NativeFn, args: Vec<Value>) -> Value {
    // Determine length: max length of list args (scalars broadcast)
    let len = args.iter().filter_map(|a| if let Value::List(v)=a { Some(v.len()) } else { None }).max().unwrap_or(0);
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let mut elem_args = Vec::with_capacity(args.len());
        for a in &args {
            match a {
                Value::List(vs) => {
                    let idx = if i < vs.len() { i } else { vs.len()-1 };
                    elem_args.push(vs[idx].clone());
                }
                other => elem_args.push(other.clone()),
            }
        }
        let evald: Vec<Value> = elem_args.into_iter().map(|x| ev.eval(x)).collect();
        out.push(f(ev, evald));
    }
    Value::List(out)
}

fn splice_sequences(args: Vec<Value>) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    for a in args.into_iter() {
        if let Value::Expr { head, args } = &a {
            if matches!(&**head, Value::Symbol(s) if s=="Sequence") {
                out.extend(args.clone());
                continue;
            }
        }
        out.push(a);
    }
    out
}

fn plus(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Numeric-only for now; fold
    let mut acc_i: Option<i64> = Some(0);
    let mut acc_f: Option<f64> = Some(0.0);
    for a in args {
        match a {
            Value::Integer(n) => { if let Some(i)=acc_i { acc_i=Some(i+n); } else if let Some(f)=acc_f { acc_f=Some(f + n as f64); } },
            Value::Real(x) => { acc_i=None; if let Some(f)=acc_f { acc_f=Some(f + x); } },
            other => return Value::Expr { head: Box::new(Value::Symbol("Plus".into())), args: vec![other] },
        }
    }
    if let Some(i)=acc_i { Value::Integer(i) } else { Value::Real(acc_f.unwrap_or(0.0)) }
}

fn map(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("Map".into())), args } }
    let f = args[0].clone();
    match &args[1] {
        Value::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                let call = Value::Expr { head: Box::new(f.clone()), args: vec![it.clone()] };
                out.push(ev.eval(call));
            }
            Value::List(out)
        }
        other => Value::Expr { head: Box::new(Value::Symbol("Map".into())), args: vec![f, other.clone()] },
    }
}

fn future_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Future".into())), args } }
    let expr = args[0].clone();
    let env_snapshot = ev.env.clone();
    let id = ev.next_task_id;
    ev.next_task_id += 1;
    let token = Arc::new(AtomicBool::new(false));
    let tok2 = token.clone();
    let handle = thread::spawn(move || {
        let mut ev2 = Evaluator::with_env_and_token(env_snapshot, tok2);
        ev2.eval(expr)
    });
    ev.tasks.insert(id, TaskInfo { handle, cancel: token });
    Value::Expr { head: Box::new(Value::Symbol("FutureId".into())), args: vec![Value::Integer(id)] }
}

fn await_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Await".into())), args } }
    let id_opt = match &args[0] {
        Value::Expr { head, args } if matches!(&**head, Value::Symbol(s) if s=="FutureId") => {
            if let Some(Value::Integer(i)) = args.get(0) { Some(*i) } else { None }
        }
        Value::Integer(i) => Some(*i),
        other => match ev.eval(other.clone()) {
            Value::Expr { head, args } if matches!(&*head, Value::Symbol(s) if s=="FutureId") => {
                if let Some(Value::Integer(i)) = args.get(0) { Some(*i) } else { None }
            }
            Value::Integer(i) => Some(i),
            _ => None,
        }
    };
    if let Some(id) = id_opt {
        if let Some(task) = ev.tasks.remove(&id) {
            return task.handle.join().unwrap_or(Value::Symbol("Null".into()));
        }
    }
    // Failure stub
    Value::Assoc(vec![
        ("message".to_string(), Value::String("Await: invalid or unknown future".into())),
        ("tag".to_string(), Value::String("Await::invfuture".into())),
    ].into_iter().collect())
}

fn cancel_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Cancel".into())), args } }
    // Accept FutureId[...] or integer id
    let id_opt = match &args[0] {
        Value::Expr { head, args } if matches!(&**head, Value::Symbol(s) if s=="FutureId") => args.get(0).and_then(|v| if let Value::Integer(i)=v { Some(*i) } else { None }),
        Value::Integer(i) => Some(*i),
        other => match ev.eval(other.clone()) { Value::Integer(i)=>Some(i), _=>None },
    };
    if let Some(id) = id_opt {
        if let Some(task) = ev.tasks.get(&id) {
            task.cancel.store(true, Ordering::Relaxed);
            return Value::Boolean(true);
        }
    }
    Value::Boolean(false)
}

fn cancelled_failure() -> Value {
    Value::Assoc(vec![
        ("message".to_string(), Value::String("Computation cancelled".into())),
        ("tag".to_string(), Value::String("Cancel::abort".into())),
    ].into_iter().collect())
}

fn parallel_map(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("ParallelMap".into())), args } }
    let f = args[0].clone();
    match &args[1] {
        Value::List(items) => {
            // naive parallelism: spawn per item
            let mut handles: Vec<JoinHandle<Value>> = Vec::with_capacity(items.len());
            for it in items.iter().cloned() {
                let f_cl = f.clone();
                // no shared env; spawn clean evaluator for each element
                handles.push(thread::spawn(move || {
                    let mut ev2 = Evaluator::new();
                    let call = Value::Expr { head: Box::new(f_cl), args: vec![it] };
                    ev2.eval(call)
                }));
            }
            let mut out = Vec::with_capacity(handles.len());
            for h in handles { out.push(h.join().unwrap_or(Value::Symbol("Null".into()))); }
            Value::List(out)
        }
        other => Value::Expr { head: Box::new(Value::Symbol("ParallelMap".into())), args: vec![f, other.clone()] },
    }
}

fn map_async(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("MapAsync".into())), args } }
    let f = args[0].clone();
    let v = ev.eval(args[1].clone());
    Value::List(map_async_rec(ev, f, v))
}

fn gather(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Gather".into())), args } }
    let v = ev.eval(args[0].clone());
    Value::List(gather_rec(ev, v))
}

fn busy_wait(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let cycles = match args.as_slice() {
        [Value::Integer(n)] => *n as usize,
        [n] => match ev.eval(n.clone()) { Value::Integer(i)=> i as usize, _=> 10 },
        _ => 10,
    };
    for _ in 0..cycles {
        if let Some(tok) = &ev.cancel_token { if tok.load(Ordering::Relaxed) { return cancelled_failure(); } }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    Value::Symbol("Done".into())
}

fn fail_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let tag = match args.get(0) { Some(Value::String(s)) => s.clone(), Some(Value::Symbol(s)) => s.clone(), _ => "Fail".into() };
    Value::Assoc(vec![
        ("message".to_string(), Value::String("Failure".into())),
        ("tag".to_string(), Value::String(tag)),
    ].into_iter().collect())
}

fn even_q(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] => Value::Boolean(n % 2 == 0),
        [other] => Value::Boolean(matches!(other, Value::List(_)) == false && false),
        _ => Value::Expr { head: Box::new(Value::Symbol("EvenQ".into())), args },
    }
}

fn odd_q(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] => Value::Boolean(n % 2 != 0),
        [other] => Value::Boolean(matches!(other, Value::List(_)) == false && false),
        _ => Value::Expr { head: Box::new(Value::Symbol("OddQ".into())), args },
    }
}

fn string_trim(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s)] => Value::String(s.trim().to_string()),
        [other] => match ev.eval(other.clone()) { Value::String(s)=>Value::String(s.trim().to_string()), v=> Value::Expr { head: Box::new(Value::Symbol("StringTrim".into())), args: vec![v] } },
        _ => Value::Expr { head: Box::new(Value::Symbol("StringTrim".into())), args },
    }
}

fn string_contains(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::String(s), Value::String(sub)] => Value::Boolean(s.contains(sub)),
        [a, b] => {
            let aa = ev.eval(a.clone());
            let bb = ev.eval(b.clone());
            match (aa, bb) {
                (Value::String(s), Value::String(sub)) => Value::Boolean(s.contains(&sub)),
                (aa, bb) => Value::Expr { head: Box::new(Value::Symbol("StringContains".into())), args: vec![aa, bb] }
            }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("StringContains".into())), args },
    }
}

fn abs_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] => Value::Integer(n.abs()),
        [Value::Real(x)] => Value::Real(x.abs()),
        _ => Value::Expr { head: Box::new(Value::Symbol("Abs".into())), args },
    }
}

fn min_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Support Min[list] and Min[a,b,...]; numeric only for now
    if args.len()==1 {
        match ev.eval(args[0].clone()) {
            Value::List(items) => return min_over_iter(items.into_iter()),
            other => return Value::Expr { head: Box::new(Value::Symbol("Min".into())), args: vec![other] },
        }
    }
    let evald: Vec<Value> = args.into_iter().map(|a| ev.eval(a)).collect();
    min_over_iter(evald.into_iter())
}

fn max_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()==1 {
        match ev.eval(args[0].clone()) {
            Value::List(items) => return max_over_iter(items.into_iter()),
            other => return Value::Expr { head: Box::new(Value::Symbol("Max".into())), args: vec![other] },
        }
    }
    let evald: Vec<Value> = args.into_iter().map(|a| ev.eval(a)).collect();
    max_over_iter(evald.into_iter())
}

fn min_over_iter<I: Iterator<Item=Value>>(iter: I) -> Value {
    let mut have = false;
    let mut use_real = false;
    let mut cur_i: i64 = 0;
    let mut cur_f: f64 = 0.0;
    for v in iter {
        match v {
            Value::Integer(n) => {
                if !have { have=true; cur_i=n; cur_f=n as f64; }
                else {
                    if use_real { if (n as f64) < cur_f { cur_f = n as f64; } }
                    else if n < cur_i { cur_i = n; }
                }
            }
            Value::Real(x) => {
                if !have { have=true; cur_f=x; cur_i=x as i64; use_real=true; }
                else { use_real=true; if x < cur_f { cur_f = x; } }
            }
            other => return Value::Expr { head: Box::new(Value::Symbol("Min".into())), args: vec![other] },
        }
    }
    if !have { Value::Expr { head: Box::new(Value::Symbol("Min".into())), args: vec![] } }
    else if use_real { Value::Real(cur_f) } else { Value::Integer(cur_i) }
}

fn max_over_iter<I: Iterator<Item=Value>>(iter: I) -> Value {
    let mut have = false;
    let mut use_real = false;
    let mut cur_i: i64 = 0;
    let mut cur_f: f64 = 0.0;
    for v in iter {
        match v {
            Value::Integer(n) => {
                if !have { have=true; cur_i=n; cur_f=n as f64; }
                else {
                    if use_real { if (n as f64) > cur_f { cur_f = n as f64; } }
                    else if n > cur_i { cur_i = n; }
                }
            }
            Value::Real(x) => {
                if !have { have=true; cur_f=x; cur_i=x as i64; use_real=true; }
                else { use_real=true; if x > cur_f { cur_f = x; } }
            }
            other => return Value::Expr { head: Box::new(Value::Symbol("Max".into())), args: vec![other] },
        }
    }
    if !have { Value::Expr { head: Box::new(Value::Symbol("Max".into())), args: vec![] } }
    else if use_real { Value::Real(cur_f) } else { Value::Integer(cur_i) }
}

fn length(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Length".into())), args } }
    match ev.eval(args[0].clone()) {
        Value::List(v) => Value::Integer(v.len() as i64),
        Value::String(s) => Value::Integer(s.chars().count() as i64),
        _ => Value::Integer(0),
    }
}

fn range(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] => Value::List((1..=*n).map(Value::Integer).collect()),
        [Value::Integer(a), Value::Integer(b)] => {
            let (start, end) = (*a, *b);
            let mut out = Vec::new();
            if start <= end { for i in start..=end { out.push(Value::Integer(i)); } }
            else { for i in (end..=start).rev() { out.push(Value::Integer(i)); } }
            Value::List(out)
        }
        [a, b] => { let a1 = ev.eval(a.clone()); let b1 = ev.eval(b.clone()); range(ev, vec![a1, b1]) },
        _ => Value::Expr { head: Box::new(Value::Symbol("Range".into())), args },
    }
}

fn join(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::List(a), Value::List(b)] => {
            let mut out = a.clone();
            out.extend(b.clone());
            Value::List(out)
        }
        [a, b] => { let a1 = ev.eval(a.clone()); let b1 = ev.eval(b.clone()); join(ev, vec![a1, b1]) },
        _ => Value::Expr { head: Box::new(Value::Symbol("Join".into())), args },
    }
}

fn reverse(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Reverse".into())), args } }
    match ev.eval(args[0].clone()) {
        Value::List(mut v) => { v.reverse(); Value::List(v) }
        other => other,
    }
}

fn flat_orderless_echo(_ev: &mut Evaluator, args: Vec<Value>) -> Value { Value::List(args) }

fn step_assoc(action: &str, head: &Value) -> Value {
    Value::Assoc(vec![
        ("action".to_string(), Value::String(action.to_string())),
        ("head".to_string(), head.clone()),
    ].into_iter().collect())
}

fn map_async_rec(ev: &mut Evaluator, f: Value, v: Value) -> Vec<Value> {
    match v {
        Value::List(items) => items.into_iter().map(|it| {
            let mapped = map_async_rec(ev, f.clone(), it);
            if mapped.len()==1 { mapped.into_iter().next().unwrap() } else { Value::List(mapped) }
        }).collect(),
        other => {
            let call = Value::Expr { head: Box::new(f), args: vec![other] };
            vec![future_fn(ev, vec![call])]
        }
    }
}

fn gather_rec(ev: &mut Evaluator, v: Value) -> Vec<Value> {
    match v {
        Value::List(items) => items.into_iter().map(|it| {
            let g = gather_rec(ev, it);
            if g.len()==1 { g.into_iter().next().unwrap() } else { Value::List(g) }
        }).collect(),
        other => vec![await_fn(ev, vec![other])],
    }
}

fn orderless_echo(_ev: &mut Evaluator, args: Vec<Value>) -> Value { Value::List(args) }
fn flat_echo(_ev: &mut Evaluator, args: Vec<Value>) -> Value { Value::List(args) }

fn schema_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::Expr { head: Box::new(Value::Symbol("Schema".into())), args } }
    let v = ev.eval(args[0].clone());
    schema_of(&v)
}

fn explain_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::Expr { head: Box::new(Value::Symbol("Explain".into())), args } }
    let expr = args[0].clone();
    // Evaluate under tracing using same env
    let env_snapshot = ev.env.clone();
    let mut ev2 = Evaluator::with_env(env_snapshot);
    ev2.trace_enabled = true;
    let _ = ev2.eval(expr);
    let steps = Value::List(ev2.trace_steps);
    Value::Assoc(vec![
        ("steps".to_string(), steps),
        ("algorithm".to_string(), Value::String("stub".into())),
        ("provider".to_string(), Value::String("cpu".into())),
        ("estCost".to_string(), Value::Assoc(Default::default())),
    ].into_iter().collect())
}

fn apply_pure_function(body: Value, params: Option<&Vec<String>>, args: &Vec<Value>) -> Value {
    if let Some(ps) = params {
        // Replace symbols matching params with args[i]
        fn subst(v: &Value, names: &Vec<String>, args: &Vec<Value>) -> Value {
            match v {
                Value::Symbol(s) => {
                    if let Some((i, _)) = names.iter().enumerate().find(|(_, n)| *n == s) {
                        args.get(i).cloned().unwrap_or(Value::Symbol("Null".into()))
                    } else { v.clone() }
                }
                Value::List(items) => Value::List(items.iter().map(|x| subst(x, names, args)).collect()),
                Value::Assoc(m) => Value::Assoc(m.iter().map(|(k,v)| (k.clone(), subst(v, names, args))).collect()),
                Value::Expr { head, args: a } => Value::Expr { head: Box::new(subst(head, names, args)), args: a.iter().map(|x| subst(x, names, args)).collect() },
                other => other.clone(),
            }
        }
        return subst(&body, ps, args);
    }
    // Slot-based substitution: # -> args[0], #n -> args[n-1]
    fn subst_slot(v: &Value, args: &Vec<Value>) -> Value {
        match v {
            Value::Slot(None) => args.get(0).cloned().unwrap_or(Value::Symbol("Null".into())),
            Value::Slot(Some(n)) => args.get(n.saturating_sub(1)).cloned().unwrap_or(Value::Symbol("Null".into())),
            Value::List(items) => Value::List(items.iter().map(|x| subst_slot(x, args)).collect()),
            Value::Assoc(m) => Value::Assoc(m.iter().map(|(k,v)| (k.clone(), subst_slot(v, args))).collect()),
            Value::Expr { head, args: a } => Value::Expr { head: Box::new(subst_slot(head, args)), args: a.iter().map(|x| subst_slot(x, args)).collect() },
            other => other.clone(),
        }
    }
    subst_slot(&body, args)
}

// Pattern matching engine (minimal): Blank, NamedBlank, PatternTest, Condition
fn match_pattern(ev: &mut Evaluator, pat: &Value, expr: &Value, binds: &mut HashMap<String, Value>) -> bool {
    match pat {
        Value::Expr { head, args } => match &**head {
            Value::Symbol(s) if s=="Alternative" => {
                for alt in args { let mut b = binds.clone(); if match_pattern(ev, alt, expr, &mut b) { *binds = b; return true; } }
                false
            }
            Value::Symbol(s) if s=="Blank" => {
                // _ or _Type
                if args.is_empty() { return true; }
                return type_matches(&args[0], expr);
            }
            Value::Symbol(s) if s=="NamedBlank" => {
                // NamedBlank[name, type?]
                if args.is_empty() { return true; }
                let name = match &args[0] { Value::Symbol(n)=>n.clone(), _=> return false };
                let ok = if args.len()>1 { type_matches(&args[1], expr) } else { true };
                if !ok { return false; }
                if let Some(existing)=binds.get(&name) { if existing != expr { return false; } }
                binds.insert(name, expr.clone());
                true
            }
            Value::Symbol(s) if s=="PatternTest" => {
                if args.len()!=2 { return false; }
                if !match_pattern(ev, &args[0], expr, binds) { return false; }
                test_predicate(ev, &args[1], expr)
            }
            Value::Symbol(s) if s=="Condition" => {
                if args.len()!=2 { return false; }
                let mut local = binds.clone();
                if !match_pattern(ev, &args[0], expr, &mut local) { return false; }
                let cond_sub = substitute_named(&args[1], &local);
                matches!(ev.eval(cond_sub), Value::Boolean(true))
            }
            // Structural head match with argument matching and sequence patterns
            Value::Symbol(pat_head) => {
                if let Value::Expr { head: expr_head, args: expr_args } = expr {
                    // Accept if heads equal symbolically
                    if matches!(&**expr_head, Value::Symbol(ref s) if s==pat_head) {
                        return match_args(ev, args, expr_args, binds);
                    }
                }
                // Fall through: not a head match, try direct equality
                expr == pat
            }
            // Head is itself a pattern (e.g., Alternative, Blank, NamedBlank)
            hpat @ Value::Expr { .. } => {
                if let Value::Expr { head: expr_head, args: expr_args } = expr {
                    // Try matching head pattern against expr head
                    let mut local = binds.clone();
                    if match_pattern(ev, hpat, expr_head, &mut local) {
                        if match_args(ev, args, expr_args, &mut local) { *binds = local; return true; }
                    }
                }
                false
            }
            _ => expr == pat,
        },
        _ => expr == pat,
    }
}

fn match_args(ev: &mut Evaluator, pats: &Vec<Value>, exprs: &Vec<Value>, binds: &mut HashMap<String, Value>) -> bool {
    fn min_required(pats: &[Value]) -> usize {
        let mut min = 0;
        for p in pats {
            if let Value::Expr { head, .. } = p {
                if let Value::Symbol(s) = &**head { if s=="BlankNullSequence" || s=="NamedBlankNullSequence" { continue; } }
            }
            min += 1;
        }
        min
    }
    fn go(ev: &mut Evaluator, pats: &[Value], exprs: &[Value], binds: &mut HashMap<String, Value>) -> bool {
        if pats.is_empty() { return exprs.is_empty(); }
        let p0 = &pats[0];
        // Sequence patterns
        if let Value::Expr { head, args } = p0 {
            if let Value::Symbol(hs) = &**head {
                let (min_take, named, ty_opt, _null_ok) = match hs.as_str() {
                    "BlankSequence" => (1usize, None, args.get(0), false),
                    "BlankNullSequence" => (0usize, None, args.get(0), true),
                    "NamedBlankSequence" => {
                        if args.is_empty() { (1usize, Some(String::new()), None, false) } else {
                            let name = if let Value::Symbol(n) = &args[0] { n.clone() } else { String::new() };
                            let ty = if args.len()>1 { Some(&args[1]) } else { None };
                            (1usize, Some(name), ty, false)
                        }
                    }
                    "NamedBlankNullSequence" => {
                        if args.is_empty() { (0usize, Some(String::new()), None, true) } else {
                            let name = if let Value::Symbol(n) = &args[0] { n.clone() } else { String::new() };
                            let ty = if args.len()>1 { Some(&args[1]) } else { None };
                            (0usize, Some(name), ty, true)
                        }
                    }
                    _ => (usize::MAX, None, None, false),
                };
                if min_take != usize::MAX {
                    // choose k elements to consume
                    let rem_min = min_required(&pats[1..]);
                    let max_take = if exprs.len()>=rem_min { exprs.len()-rem_min } else { 0 };
                    let start = min_take.min(exprs.len());
                    for k in start..=max_take {
                        // type check
                        let slice = &exprs[..k];
                        if let Some(ty) = ty_opt { if !slice.iter().all(|e| type_matches(ty, e)) { continue; } }
                        // bind named sequence to Sequence[...] for splicing
                        let mut local = binds.clone();
                        if let Some(n) = named.as_ref() {
                            if !n.is_empty() {
                                let seq = Value::Expr { head: Box::new(Value::Symbol("Sequence".into())), args: slice.to_vec() };
                                if let Some(prev) = local.get(n) { if prev != &seq { continue; } }
                                local.insert(n.clone(), seq);
                            }
                        }
                        if go(ev, &pats[1..], &exprs[k..], &mut local) { *binds = local; return true; }
                    }
                    return false;
                }
            }
        }
        // Non-sequence: need at least one expr
        if exprs.is_empty() { return false; }
        let mut local = binds.clone();
        if match_pattern(ev, &pats[0], &exprs[0], &mut local) { if go(ev, &pats[1..], &exprs[1..], &mut local) { *binds = local; return true; } }
        false
    }
    go(ev, pats.as_slice(), exprs.as_slice(), binds)
}

fn type_matches(ty: &Value, expr: &Value) -> bool {
    match ty {
        Value::Symbol(s) => match s.as_str() {
            "Integer" => matches!(expr, Value::Integer(_)),
            "Real" => matches!(expr, Value::Real(_)),
            "String" => matches!(expr, Value::String(_)),
            "Symbol" => matches!(expr, Value::Symbol(_)),
            "List" => matches!(expr, Value::List(_)),
            "Assoc" => matches!(expr, Value::Assoc(_)),
            _ => true,
        },
        _ => true,
    }
}

fn test_predicate(ev: &mut Evaluator, pred: &Value, expr: &Value) -> bool {
    let call = Value::Expr { head: Box::new(pred.clone()), args: vec![expr.clone()] };
    matches!(ev.eval(call), Value::Boolean(true))
}

fn substitute_named(v: &Value, binds: &HashMap<String, Value>) -> Value {
    match v {
        Value::Symbol(s) => binds.get(s).cloned().unwrap_or(v.clone()),
        Value::List(items) => Value::List(items.iter().map(|x| substitute_named(x, binds)).collect()),
        Value::Assoc(m) => Value::Assoc(m.iter().map(|(k,x)| (k.clone(), substitute_named(x, binds))).collect()),
        Value::Expr { head, args } => Value::Expr { head: Box::new(substitute_named(head, binds)), args: args.iter().map(|x| substitute_named(x, binds)).collect() },
        other => other.clone(),
    }
}

fn iff(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [cond, then_v] => {
            let c = ev.eval(cond.clone());
            match c { Value::Boolean(true) => ev.eval(then_v.clone()), _ => Value::Symbol("Null".into()) }
        }
        [cond, then_v, else_v] => {
            let c = ev.eval(cond.clone());
            match c { Value::Boolean(true) => ev.eval(then_v.clone()), Value::Boolean(false) => ev.eval(else_v.clone()), _ => Value::Symbol("Null".into()) }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("If".into())), args },
    }
}

fn equal(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("Equal".into())), args } }
    Value::Boolean(args[0] == args[1])
}

fn less(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("Less".into())), args } }
    Value::Boolean(value_order(&args[0]) < value_order(&args[1]))
}

fn less_equal(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("LessEqual".into())), args } }
    Value::Boolean(value_order(&args[0]) <= value_order(&args[1]))
}

fn greater(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("Greater".into())), args } }
    Value::Boolean(value_order(&args[0]) > value_order(&args[1]))
}

fn greater_equal(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("GreaterEqual".into())), args } }
    Value::Boolean(value_order(&args[0]) >= value_order(&args[1]))
}

fn and_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    for a in args {
        let v = ev.eval(a);
        if matches!(v, Value::Boolean(false)) { return Value::Boolean(false); }
        if !matches!(v, Value::Boolean(true)) { return Value::Boolean(false); }
    }
    Value::Boolean(true)
}

fn or_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    for a in args {
        let v = ev.eval(a);
        if matches!(v, Value::Boolean(true)) { return Value::Boolean(true); }
    }
    Value::Boolean(false)
}

fn not_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Not".into())), args } }
    match ev.eval(args[0].clone()) { Value::Boolean(b) => Value::Boolean(!b), _ => Value::Boolean(false) }
}

pub fn value_order_key(v: &Value) -> String { value_order(v) }

fn value_order(v: &Value) -> String {
    // Simple canonical string for ordering
    match v {
        Value::Integer(n) => format!("0:{n:020}"),
        Value::Real(f) => format!("1:{:.*}", 16, f),
        Value::String(s) => format!("2:{s}"),
        Value::Symbol(s) => format!("3:{s}"),
        Value::Boolean(b) => format!("4:{}", if *b {1}else{0}),
        Value::List(items) => format!("5:[{}]", items.iter().map(|x| value_order(x)).collect::<Vec<_>>().join(";")),
        Value::Assoc(m) => {
            let mut keys: Vec<_> = m.keys().collect(); keys.sort();
            let parts: Vec<_> = keys.into_iter().map(|k| format!("{}=>{}", k, value_order(m.get(k).unwrap()))).collect();
            format!("6:<|{}|>", parts.join(","))
        }
        Value::Expr { head, args } => format!("7:{}[{}]", value_order(head), args.iter().map(|x| value_order(x)).collect::<Vec<_>>().join(",")),
        Value::Slot(n) => format!("8:#{}", n.unwrap_or(1)),
        Value::PureFunction { .. } => "9:PureFunction".into(),
    }
}

// List/Assoc functions
fn apply(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("Apply".into())), args } }
    let f = ev.eval(args[0].clone());
    match ev.eval(args[1].clone()) {
        Value::List(items) => {
            let call = Value::Expr { head: Box::new(f), args: items }; ev.eval(call)
        }
        other => Value::Expr { head: Box::new(Value::Symbol("Apply".into())), args: vec![f, other] },
    }
}

fn total(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v] => { let vv = ev.eval(v.clone()); sum_list(ev, vv) },
        [v, Value::Symbol(s)] if s == "Infinity" => { let vv = ev.eval(v.clone()); sum_all(ev, vv) },
        [v, Value::Integer(n)] => {
            let mut val = ev.eval(v.clone());
            // reduce n levels by summing at each level
            let mut lvl = *n;
            while lvl > 0 { val = sum_list(ev, val); lvl -= 1; }
            val
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("Total".into())), args },
    }
}

fn sum_list(ev: &mut Evaluator, v: Value) -> Value {
    match v {
        Value::List(items) => {
            let mut acc_i: Option<i64> = Some(0);
            let mut acc_f: Option<f64> = Some(0.0);
            for it in items { match ev.eval(it) {
                Value::Integer(n) => { if let Some(i)=acc_i { acc_i=Some(i+n); } else if let Some(f)=acc_f { acc_f=Some(f+n as f64);} }
                Value::Real(x) => { acc_i=None; if let Some(f)=acc_f { acc_f=Some(f+x);} }
                other => { // sum nested by evaluating Total[other]
                    let inner = total(ev, vec![other]);
                    match inner {
                        Value::Integer(n) => { if let Some(i)=acc_i { acc_i=Some(i+n); } else if let Some(f)=acc_f { acc_f=Some(f+n as f64);} }
                        Value::Real(x) => { acc_i=None; if let Some(f)=acc_f { acc_f=Some(f+x);} }
                        _ => {}
                    }
                }
            }}
            if let Some(i)=acc_i { Value::Integer(i) } else { Value::Real(acc_f.unwrap_or(0.0)) }
        }
        other => other,
    }
}

fn sum_all(ev: &mut Evaluator, v: Value) -> Value {
    match v {
        Value::List(items) => {
            let mut acc = Value::Integer(0);
            for it in items { let itv = ev.eval(it); let s = sum_all(ev, itv); acc = plus(ev, vec![acc, s]); }
            acc
        }
        Value::Integer(_) | Value::Real(_) => v,
        other => other,
    }
}

fn fold(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [f, Value::List(items)] => {
            if items.is_empty() { return Value::Symbol("Null".into()); }
            let mut acc = ev.eval(items[0].clone());
            for it in &items[1..] {
                let call = Value::Expr { head: Box::new(ev.eval(f.clone())), args: vec![acc, ev.eval(it.clone())] };
                acc = ev.eval(call);
            }
            acc
        }
        [f, init, Value::List(items)] => {
            let mut acc = ev.eval(init.clone());
            for it in items { let call = Value::Expr { head: Box::new(ev.eval(f.clone())), args: vec![acc, ev.eval(it.clone())] }; acc = ev.eval(call); }
            acc
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("Fold".into())), args },
    }
}

fn select(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        // Select[list, pred]
        [list, pred] => match ev.eval(list.clone()) {
            Value::List(items) => {
                let mut out = Vec::new();
                for it in items {
                    let call_args = match &it { Value::List(inner) => inner.clone(), _ => vec![it.clone()] };
                    let pred_call = Value::Expr { head: Box::new(pred.clone()), args: call_args };
                    if matches!(ev.eval(pred_call), Value::Boolean(true)) { out.push(ev.eval(it)); }
                }
                Value::List(out)
            }
            other => other,
        }
        // Select[list1, list2, pred] → return tuples {a,b} that satisfy pred[a,b]
        [l1, l2, pred] => {
            match (ev.eval(l1.clone()), ev.eval(l2.clone())) {
                (Value::List(a), Value::List(b)) => {
                    let len = a.len().min(b.len());
                    let mut out = Vec::new();
                    for i in 0..len {
                        let pred_call = Value::Expr { head: Box::new(pred.clone()), args: vec![a[i].clone(), b[i].clone()] };
                        if matches!(ev.eval(pred_call), Value::Boolean(true)) { out.push(Value::List(vec![ev.eval(a[i].clone()), ev.eval(b[i].clone())])); }
                    }
                    Value::List(out)
                }
                (x, _) => x,
            }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("Select".into())), args },
    }
}

fn keys(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Keys".into())), args } }
    match ev.eval(args[0].clone()) {
        Value::Assoc(m) => Value::List(m.keys().cloned().map(Value::String).collect()),
        other => other,
    }
}

fn values(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Values".into())), args } }
    match ev.eval(args[0].clone()) {
        Value::Assoc(m) => Value::List(m.values().cloned().collect()),
        other => other,
    }
}

fn lookup(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [assoc, Value::String(k)] => match ev.eval(assoc.clone()) { Value::Assoc(m) => m.get(k).cloned().unwrap_or(Value::Symbol("Null".into())), other => other },
        [assoc, Value::String(k), default] => match ev.eval(assoc.clone()) { Value::Assoc(m) => m.get(k).cloned().unwrap_or(ev.eval(default.clone())), other => other },
        _ => Value::Expr { head: Box::new(Value::Symbol("Lookup".into())), args },
    }
}

fn association_map(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("AssociationMap".into())), args } }
    match ev.eval(args[1].clone()) {
        Value::Assoc(m) => {
            let mm = m.into_iter().map(|(k,v)| (k, ev.eval(Value::Expr { head: Box::new(args[0].clone()), args: vec![v] }))).collect();
            Value::Assoc(mm)
        }
        other => other,
    }
}

fn association_map_keys(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("AssociationMapKeys".into())), args } }
    match ev.eval(args[1].clone()) {
        Value::Assoc(m) => {
            let mut out = HashMap::new();
            for (k, v) in m.into_iter() {
                let newk_v = ev.eval(Value::Expr { head: Box::new(args[0].clone()), args: vec![Value::String(k.clone())] });
                let newk = match newk_v { Value::String(s)=>s, Value::Symbol(s)=>s, other=> lyra_core::pretty::format_value(&other) };
                out.insert(newk, ev.eval(v));
            }
            Value::Assoc(out)
        }
        other => other,
    }
}

fn association_map_kv(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("AssociationMapKV".into())), args } }
    match ev.eval(args[1].clone()) {
        Value::Assoc(m) => {
            let mut out = HashMap::new();
            for (k, v) in m.into_iter() {
                let newv = ev.eval(Value::Expr { head: Box::new(args[0].clone()), args: vec![Value::String(k.clone()), v] });
                out.insert(k, newv);
            }
            Value::Assoc(out)
        }
        other => other,
    }
}

fn association_map_pairs(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("AssociationMapPairs".into())), args } }
    match ev.eval(args[1].clone()) {
        Value::Assoc(m) => {
            let mut out = HashMap::new();
            for (k, v) in m.into_iter() {
                let res = ev.eval(Value::Expr { head: Box::new(args[0].clone()), args: vec![Value::String(k.clone()), v] });
                match res {
                    Value::List(mut pair) if pair.len()==2 => {
                        let newk = match pair.remove(0) { Value::String(s)=>s, Value::Symbol(s)=>s, other=> lyra_core::pretty::format_value(&other) };
                        let newv = pair.remove(0);
                        out.insert(newk, newv);
                    }
                    Value::Assoc(am) => {
                        let nk = am.get("key").cloned().unwrap_or(Value::String(k.clone()));
                        let nv = am.get("value").cloned().unwrap_or(Value::Symbol("Null".into()));
                        let newk = match nk { Value::String(s)=>s, Value::Symbol(s)=>s, other=> lyra_core::pretty::format_value(&other) };
                        out.insert(newk, nv);
                    }
                    other => { out.insert(k, other); }
                }
            }
            Value::Assoc(out)
        }
        other => other,
    }
}

fn merge(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Assoc(HashMap::new()) }
    // forms: Merge[listOfAssocs], Merge[a1, a2, ...], Merge[listOrArgs, combiner]
    let (vals, combiner) = if args.len()>=2 {
        if let Value::Expr { .. } | Value::Symbol(_) | Value::PureFunction { .. } = args[args.len()-1] {
            (args[..args.len()-1].to_vec(), Some(args[args.len()-1].clone()))
        } else { (args, None) }
    } else { (args, None) };
    let list: Vec<Value> = if vals.len()==1 { match ev.eval(vals[0].clone()) { Value::List(v)=>v, x=>vec![x] } } else { vals.into_iter().map(|a| ev.eval(a)).collect() };
    let mut groups: HashMap<String, Vec<Value>> = HashMap::new();
    for v in list {
        if let Value::Assoc(m) = v { for (k,val) in m { groups.entry(k).or_default().push(val); } }
    }
    let out = if let Some(f) = combiner {
        groups.into_iter().map(|(k, xs)| {
            let res = if xs.len()==1 { xs[0].clone() } else { ev.eval(Value::Expr { head: Box::new(f.clone()), args: vec![Value::List(xs)] }) };
            (k, res)
        }).collect()
    } else {
        groups.into_iter().map(|(k, mut xs)| (k, xs.pop().unwrap())).collect()
    };
    Value::Assoc(out)
}

fn key_sort(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("KeySort".into())), args } }
    match ev.eval(args[0].clone()) {
        Value::Assoc(m) => {
            let mut keys: Vec<String> = m.keys().cloned().collect();
            keys.sort();
            let mut out = HashMap::new();
            for k in keys { out.insert(k.clone(), m.get(&k).cloned().unwrap()); }
            Value::Assoc(out)
        }
        other => other,
    }
}

fn sort_by(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("SortBy".into())), args } }
    match ev.eval(args[1].clone()) {
        Value::Assoc(m) => {
            let mut items: Vec<(String, Value)> = m.into_iter().collect();
            items.sort_by(|(ka, va), (kb, vb)| {
                let va_key = ev.eval(Value::Expr { head: Box::new(args[0].clone()), args: vec![Value::String(ka.clone()), va.clone()] });
                let vb_key = ev.eval(Value::Expr { head: Box::new(args[0].clone()), args: vec![Value::String(kb.clone()), vb.clone()] });
                value_order(&va_key).cmp(&value_order(&vb_key))
            });
            let mut out = HashMap::new();
            for (k,v) in items { out.insert(k,v); }
            Value::Assoc(out)
        }
        other => other,
    }
}

fn group_by(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("GroupBy".into())), args } }
    match ev.eval(args[0].clone()) {
        Value::List(items) => {
            let mut groups: HashMap<String, Vec<Value>> = HashMap::new();
            for it in items {
                let key_v = ev.eval(Value::Expr { head: Box::new(args[1].clone()), args: vec![it.clone()] });
                let key = match key_v { Value::String(s)=>s, Value::Symbol(s)=>s, other=> format!("{}", lyra_core::pretty::format_value(&other)) };
                groups.entry(key).or_default().push(ev.eval(it));
            }
            Value::Assoc(groups.into_iter().map(|(k,vs)|(k, Value::List(vs))).collect())
        }
        other => other,
    }
}

fn flatten(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v] => flatten_once(ev.eval(v.clone())),
        [v, Value::Integer(n)] => {
            let mut res = ev.eval(v.clone());
            for _ in 0..*n { res = flatten_once(res); }
            res
        }
        [v, Value::Symbol(s)] if s=="Infinity" => {
            let mut res = ev.eval(v.clone());
            loop { let next = flatten_once(res.clone()); if next==res { break; } res = next; }
            res
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("Flatten".into())), args },
    }
}

fn flatten_once(v: Value) -> Value {
    match v {
        Value::List(items) => Value::List(items.into_iter().flat_map(|x| match x { Value::List(inner)=>inner, other=>vec![other] }).collect()),
        other => other,
    }
}

fn thread(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Forms: Thread[f, list1, ...] or Thread[expr]
    match args.as_slice() {
        [Value::Expr { head, args: a }] => {
            // thread over list arguments of expr
            let lists: Vec<_> = a.iter().map(|x| ev.eval(x.clone())).collect();
            let lens: Vec<usize> = lists.iter().map(|v| if let Value::List(l)=v { l.len() } else { usize::MAX }).collect();
            let len = lens.into_iter().min().unwrap_or(0);
            let mut out = Vec::with_capacity(len);
            for i in 0..len {
                let mut call_args = Vec::with_capacity(lists.len());
                for lst in &lists { match lst { Value::List(v)=>call_args.push(v[i].clone()), other=>call_args.push(other.clone()) } }
                out.push(ev.eval(Value::Expr { head: head.clone(), args: call_args }));
            }
            Value::List(out)
        }
        [f, rest @ ..] if rest.len()>=1 => {
            let lists: Vec<Value> = rest.iter().map(|a| ev.eval(a.clone())).collect();
            let lens: Vec<usize> = lists.iter().map(|v| if let Value::List(l)=v { l.len() } else { 0 }).collect();
            let len = lens.iter().copied().min().unwrap_or(0);
            let mut out = Vec::with_capacity(len);
            for i in 0..len {
                let mut call_args = Vec::with_capacity(lists.len());
                for lst in &lists { if let Value::List(v)=lst { call_args.push(v[i].clone()); } }
                let fh = ev.eval(f.clone());
                out.push(ev.eval(Value::Expr { head: Box::new(fh), args: call_args }));
            }
            Value::List(out)
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("Thread".into())), args },
    }
}

// Partition[list, n], Partition[list, n, step]
fn partition(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v, Value::Integer(n)] => do_partition(ev.eval(v.clone()), *n as usize, *n as usize),
        [v, Value::Integer(n), Value::Integer(step)] => do_partition(ev.eval(v.clone()), *n as usize, *step as usize),
        _ => Value::Expr { head: Box::new(Value::Symbol("Partition".into())), args },
    }
}

fn do_partition(v: Value, n: usize, step: usize) -> Value {
    match v {
        Value::List(items) => {
            let mut out = Vec::new();
            let mut i = 0;
            while i + n <= items.len() {
                out.push(Value::List(items[i..i+n].to_vec()));
                i += step;
            }
            Value::List(out)
        }
        other => other,
    }
}

// Transpose[matrix]
fn transpose(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return Value::Expr { head: Box::new(Value::Symbol("Transpose".into())), args } }
    match ev.eval(args[0].clone()) {
        Value::List(rows) => {
            if rows.is_empty() { return Value::List(vec![]); }
            let cols = match &rows[0] { Value::List(r)=>r.len(), _=> return Value::List(rows) };
            let mut out = Vec::with_capacity(cols);
            for c in 0..cols { let mut col = Vec::with_capacity(rows.len()); for r in &rows { if let Value::List(rr)=r { if c<rr.len() { col.push(rr[c].clone()); } } } out.push(Value::List(col)); }
            Value::List(out)
        }
        other => other,
    }
}

fn collect_rules(ev: &mut Evaluator, rules_v: Value) -> Vec<(Value, Value)> {
    let mut out = Vec::new();
    let val = ev.eval(rules_v);
    match val {
        Value::List(rs) => {
            for r in rs { if let Value::Expr { head, args } = r { if matches!(*head, Value::Symbol(ref s) if s=="Rule") && args.len()==2 { out.push((args[0].clone(), args[1].clone())); } } }
        }
        Value::Expr { head, args } if matches!(*head, Value::Symbol(ref s) if s=="Rule") && args.len()==2 => out.push((args[0].clone(), args[1].clone())),
        _ => {}
    }
    out
}

fn apply_rules_once(ev: &mut Evaluator, v: Value, rules: &[(Value, Value)]) -> Value {
    for (lhs, rhs) in rules {
        let mut binds = HashMap::new();
        if match_pattern(ev, lhs, &v, &mut binds) {
            let replaced = substitute_named(rhs, &binds);
            return ev.eval(replaced);
        }
    }
    match v {
        Value::List(items) => {
            let mut out = Vec::new();
            for it in items {
                let r = apply_rules_once(ev, it, rules);
                if let Value::Expr { head, args } = &r { if matches!(**head, Value::Symbol(ref s) if s=="Sequence") { out.extend(args.clone()); continue; } }
                out.push(r);
            }
            Value::List(out)
        }
        Value::Assoc(m) => Value::Assoc(m.into_iter().map(|(k,x)| (k, apply_rules_once(ev, x, rules))).collect()),
        Value::Expr { head, args } => {
            let new_head = apply_rules_once(ev, *head, rules);
            let mut new_args: Vec<Value> = Vec::new();
            for a in args { let r = apply_rules_once(ev, a, rules); if let Value::Expr { head, args } = &r { if matches!(**head, Value::Symbol(ref s) if s=="Sequence") { new_args.extend(args.clone()); continue; } } new_args.push(r); }
            // If top-level Sequence, convert to List
            if let Value::Symbol(ref s) = new_head { if s=="Sequence" { return Value::List(new_args); } }
            Value::Expr { head: Box::new(new_head), args: new_args }
        }
        other => other,
    }
}

fn apply_rules_all(ev: &mut Evaluator, v: Value, rules: &[(Value, Value)]) -> Value {
    // Keep applying until fixed point
    let mut curr = v;
    loop {
        let next = apply_rules_once(ev, curr.clone(), rules);
        if next == curr { return next; }
        curr = next;
    }
}

fn apply_rules_with_limit(ev: &mut Evaluator, v: Value, rules: &[(Value, Value)], limit: &mut usize) -> Value {
    if *limit == 0 { return v; }
    for (lhs, rhs) in rules {
        let mut binds = HashMap::new();
        if match_pattern(ev, lhs, &v, &mut binds) {
            let replaced = substitute_named(rhs, &binds);
            *limit = limit.saturating_sub(1);
            return ev.eval(replaced);
        }
    }
    match v {
        Value::List(items) => {
            let mut out = Vec::new();
            for it in items {
                if *limit == 0 { out.push(it); continue; }
                let r = apply_rules_with_limit(ev, it, rules, limit);
                if let Value::Expr { head, args } = &r { if matches!(**head, Value::Symbol(ref s) if s=="Sequence") { out.extend(args.clone()); continue; } }
                out.push(r);
            }
            Value::List(out)
        }
        Value::Assoc(m) => Value::Assoc(m.into_iter().map(|(k,x)| (k, apply_rules_with_limit(ev, x, rules, limit))).collect()),
        Value::Expr { head, args } => {
            let new_head = apply_rules_with_limit(ev, *head, rules, limit);
            let mut new_args: Vec<Value> = Vec::new();
            for a in args { if *limit == 0 { new_args.push(a); continue; } let r = apply_rules_with_limit(ev, a, rules, limit); if let Value::Expr { head, args } = &r { if matches!(**head, Value::Symbol(ref s) if s=="Sequence") { new_args.extend(args.clone()); continue; } } new_args.push(r); }
            if let Value::Symbol(ref s) = new_head { if s=="Sequence" { return Value::List(new_args); } }
            Value::Expr { head: Box::new(new_head), args: new_args }
        }
        other => other,
    }
}

fn replace(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [expr, rules_v] => {
            let rules = collect_rules(ev, rules_v.clone());
            let target = ev.eval(expr.clone());
            apply_rules_once(ev, target, &rules)
        }
        [expr, rules_v, Value::Integer(n)] => {
            let mut limit = (*n as isize).max(0) as usize;
            let rules = collect_rules(ev, rules_v.clone());
            let target = ev.eval(expr.clone());
            apply_rules_with_limit(ev, target, &rules, &mut limit)
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("Replace".into())), args },
    }
}

fn replace_all_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("ReplaceAll".into())), args } }
    let rules = collect_rules(ev, args[1].clone());
    let target = ev.eval(args[0].clone());
    apply_rules_all(ev, target, &rules)
}

fn replace_first(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("ReplaceFirst".into())), args } }
    let rules = collect_rules(ev, args[1].clone());
    let target = ev.eval(args[0].clone());
    let mut limit = 1usize;
    apply_rules_with_limit(ev, target, &rules, &mut limit)
}

fn set_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("Set".into())), args } }
    match &args[0] {
        Value::Symbol(name) => { let v = ev.eval(args[1].clone()); ev.env.insert(name.clone(), v.clone()); v }
        _ => Value::Expr { head: Box::new(Value::Symbol("Set".into())), args },
    }
}

fn with_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=2 { return Value::Expr { head: Box::new(Value::Symbol("With".into())), args } }
    let assoc = ev.eval(args[0].clone());
    let body = args[1].clone();
    let mut saved: Vec<(String, Option<Value>)> = Vec::new();
    if let Value::Assoc(m) = assoc {
        for (k,v) in m {
            let old = ev.env.get(&k).cloned();
            let newv = ev.eval(v);
            saved.push((k.clone(), old));
            ev.env.insert(k, newv);
        }
        let result = ev.eval(body);
        // restore
        for (k,ov) in saved.into_iter() {
            if let Some(val)=ov { ev.env.insert(k, val); } else { ev.env.remove(&k); }
        }
        result
    } else { Value::Expr { head: Box::new(Value::Symbol("With".into())), args } }
}
