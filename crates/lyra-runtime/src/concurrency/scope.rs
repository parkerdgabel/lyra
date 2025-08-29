use crate::attrs::Attributes;
use crate::concurrency::pool::{spawn_task, ThreadLimiter};
use crate::eval::{Evaluator, NativeFn};
use lyra_core::value::Value;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::{
    atomic::{AtomicBool, AtomicI64, Ordering},
    Arc, Mutex,
};
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

#[derive(Clone)]
struct ScopeCtx {
    cancel: Arc<AtomicBool>,
    limiter: Option<Arc<ThreadLimiter>>,
    deadline: Option<Instant>,
}

static SCOPE_REG: OnceLock<Mutex<HashMap<i64, ScopeCtx>>> = OnceLock::new();
static NEXT_SCOPE_ID: OnceLock<AtomicI64> = OnceLock::new();

fn scope_reg() -> &'static Mutex<HashMap<i64, ScopeCtx>> {
    SCOPE_REG.get_or_init(|| Mutex::new(HashMap::new()))
}
fn next_scope_id() -> i64 {
    let a = NEXT_SCOPE_ID.get_or_init(|| AtomicI64::new(1));
    a.fetch_add(1, Ordering::Relaxed)
}

pub(crate) fn register_scopes(ev: &mut Evaluator) {
    ev.register("Scope", scope_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("StartScope", start_scope_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("InScope", in_scope_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("CancelScope", cancel_scope_fn as NativeFn, Attributes::empty());
    ev.register("EndScope", end_scope_fn as NativeFn, Attributes::empty());
    ev.register("ParallelEvaluate", parallel_evaluate as NativeFn, Attributes::HOLD_ALL);
}

fn parallel_evaluate(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("ParallelEvaluate".into())), args }; }
    let target = args[0].clone();
    let mut opt_limiter: Option<Arc<ThreadLimiter>> = None;
    let mut opt_deadline: Option<Instant> = None;
    if args.len() >= 2 {
        if let Value::Assoc(m) = ev.eval(args[1].clone()) {
            if let Some(Value::Integer(n)) = m.get("MaxThreads").or_else(|| m.get("maxThreads")) { if *n > 0 { opt_limiter = Some(Arc::new(ThreadLimiter::new(*n as usize))); } }
            if let Some(Value::Integer(ms)) = m.get("TimeBudgetMs").or_else(|| m.get("timeBudgetMs")) { if *ms > 0 { opt_deadline = Some(Instant::now() + Duration::from_millis(*ms as u64)); } }
        }
    }
    let env_snapshot = ev.env.clone();
    match ev.eval(target) {
        Value::List(items) => {
            let token = ev.cancel_token.clone();
            let limiter = opt_limiter.or_else(|| ev.thread_limiter.clone());
            let deadline = opt_deadline.or(ev.deadline);
            if ev.trace_enabled {
                let data = Value::Assoc(vec![
                    ("items".to_string(), Value::Integer(items.len() as i64)),
                    ("maxThreads".to_string(), Value::Integer(limiter.as_ref().map(|l| l.max_permits() as i64).unwrap_or(-1))),
                    ("hasDeadline".to_string(), Value::Boolean(deadline.is_some())),
                ].into_iter().collect());
                ev.trace_steps.push(Value::Assoc(vec![
                    ("action".to_string(), Value::String("ParallelDispatch".into())),
                    ("head".to_string(), Value::Symbol("ParallelEvaluate".into())),
                    ("data".to_string(), data),
                ].into_iter().collect()));
            }
            let mut rxs: Vec<Receiver<Value>> = Vec::with_capacity(items.len());
            for it in items.into_iter() {
                let token_cl = token.clone();
                let limiter_cl = limiter.clone();
                let deadline_cl = deadline;
                let env_cl = env_snapshot.clone();
                rxs.push(spawn_task(move || {
                    if let Some(l) = limiter_cl.as_ref() { l.acquire(); }
                    let mut ev2 = if let Some(tok) = token_cl { Evaluator::with_env_and_token(env_cl.clone(), tok) } else { Evaluator::with_env(env_cl.clone()) };
                    ev2.thread_limiter = limiter_cl;
                    ev2.deadline = deadline_cl;
                    let out = ev2.eval(it);
                    if let Some(l) = ev2.thread_limiter.as_ref() { l.release(); }
                    out
                }));
            }
            let mut out = Vec::with_capacity(rxs.len());
            for rx in rxs { out.push(rx.recv().unwrap_or(Value::Symbol("Null".into()))); }
            Value::List(out)
        }
        other => Value::Expr { head: Box::new(Value::Symbol("ParallelEvaluate".into())), args: vec![other] },
    }
}

fn scope_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("Scope".into())), args }; }
    let opts = ev.eval(args[0].clone());
    let body = args[1].clone();
    let mut ev2 = Evaluator::with_env(ev.env.clone());
    if let Value::Assoc(m) = opts {
        if let Some(Value::Integer(n)) = m.get("MaxThreads").or_else(|| m.get("maxThreads")) { if *n > 0 { ev2.thread_limiter = Some(Arc::new(ThreadLimiter::new(*n as usize))); } }
        if let Some(Value::Integer(ms)) = m.get("TimeBudgetMs").or_else(|| m.get("timeBudgetMs")) { if *ms > 0 { ev2.deadline = Some(Instant::now() + Duration::from_millis(*ms as u64)); } }
    }
    if ev.trace_enabled {
        let data = Value::Assoc(vec![
            ("maxThreads".to_string(), Value::Integer(ev2.thread_limiter.as_ref().map(|l| l.max_permits() as i64).unwrap_or(-1))),
            ("hasDeadline".to_string(), Value::Boolean(ev2.deadline.is_some())),
        ].into_iter().collect());
        ev.trace_steps.push(Value::Assoc(vec![
            ("action".to_string(), Value::String("ScopeApply".into())),
            ("head".to_string(), Value::Symbol("Scope".into())),
            ("data".to_string(), data),
        ].into_iter().collect()));
    }
    ev2.eval(body)
}

fn start_scope_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::Expr { head: Box::new(Value::Symbol("StartScope".into())), args }; }
    let opts = ev.eval(args[0].clone());
    let id = next_scope_id();
    let mut ctx = ScopeCtx { cancel: Arc::new(AtomicBool::new(false)), limiter: None, deadline: None };
    if let Value::Assoc(m) = opts {
        if let Some(Value::Integer(n)) = m.get("MaxThreads").or_else(|| m.get("maxThreads")) { if *n > 0 { ctx.limiter = Some(Arc::new(ThreadLimiter::new(*n as usize))); } }
        if let Some(Value::Integer(ms)) = m.get("TimeBudgetMs").or_else(|| m.get("timeBudgetMs")) { if *ms > 0 { ctx.deadline = Some(Instant::now() + Duration::from_millis(*ms as u64)); } }
    }
    scope_reg().lock().unwrap().insert(id, ctx);
    Value::Expr { head: Box::new(Value::Symbol("ScopeId".into())), args: vec![Value::Integer(id)] }
}

fn in_scope_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("InScope".into())), args }; }
    let sid_val = ev.eval(args[0].clone());
    let sid = match &sid_val {
        Value::Expr { head, args } if matches!(&**head, Value::Symbol(s) if s=="ScopeId") => args.get(0).and_then(|v| if let Value::Integer(i) = v { Some(*i) } else { None }),
        _ => None,
    };
    if let Some(id) = sid {
        if let Some(ctx) = scope_reg().lock().unwrap().get(&id).cloned() {
            let old_tok = ev.cancel_token.clone();
            let old_lim = ev.thread_limiter.clone();
            let old_dead = ev.deadline;
            ev.cancel_token = Some(ctx.cancel.clone());
            ev.thread_limiter = ctx.limiter.clone();
            ev.deadline = ctx.deadline;
            let out = ev.eval(args[1].clone());
            ev.cancel_token = old_tok;
            ev.thread_limiter = old_lim;
            ev.deadline = old_dead;
            return out;
        }
    }
    Value::Assoc(vec![
        ("message".to_string(), Value::String("InScope: invalid scope id".into())),
        ("tag".to_string(), Value::String("InScope::invscope".into())),
    ].into_iter().collect())
}

fn cancel_scope_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::Expr { head: Box::new(Value::Symbol("CancelScope".into())), args }; }
    let sid_val = ev.eval(args[0].clone());
    let sid = match &sid_val {
        Value::Expr { head, args } if matches!(&**head, Value::Symbol(s) if s=="ScopeId") => args.get(0).and_then(|v| if let Value::Integer(i) = v { Some(*i) } else { None }),
        _ => None,
    };
    if let Some(id) = sid { if let Some(ctx) = scope_reg().lock().unwrap().get(&id) { ctx.cancel.store(true, Ordering::Relaxed); return Value::Boolean(true); } }
    Value::Boolean(false)
}

fn end_scope_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::Expr { head: Box::new(Value::Symbol("EndScope".into())), args }; }
    let sid_val = ev.eval(args[0].clone());
    let sid = match &sid_val {
        Value::Expr { head, args } if matches!(&**head, Value::Symbol(s) if s=="ScopeId") => args.get(0).and_then(|v| if let Value::Integer(i) = v { Some(*i) } else { None }),
        _ => None,
    };
    if let Some(id) = sid { return Value::Boolean(scope_reg().lock().unwrap().remove(&id).is_some()); }
    Value::Boolean(false)
}

