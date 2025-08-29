use crate::concurrency::pool::{spawn_task, ThreadLimiter};
use crate::eval::Evaluator;
use crate::eval::NativeFn; // use same type alias from eval
use crate::attrs::Attributes;
use lyra_core::value::Value;
use std::sync::mpsc::Receiver;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

pub(crate) struct TaskInfo {
    pub(crate) rx: Receiver<Value>,
    pub(crate) cancel: Arc<AtomicBool>,
}

pub(crate) fn register_futures(ev: &mut Evaluator) {
    ev.register("Future", future_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Await", await_fn as NativeFn, Attributes::empty());
    ev.register("Cancel", cancel_fn as NativeFn, Attributes::empty());
}

pub(crate) fn future_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("Future".into())), args };
    }
    let expr = args[0].clone();
    let mut opt_limiter: Option<Arc<ThreadLimiter>> = None;
    let mut opt_deadline: Option<Instant> = None;
    if args.len() >= 2 {
        if let Value::Assoc(m) = ev.eval(args[1].clone()) {
            if let Some(Value::Integer(n)) = m.get("MaxThreads").or_else(|| m.get("maxThreads")) {
                if *n > 0 {
                    opt_limiter = Some(Arc::new(ThreadLimiter::new(*n as usize)));
                }
            }
            if let Some(Value::Integer(ms)) = m.get("TimeBudgetMs").or_else(|| m.get("timeBudgetMs")) {
                if *ms > 0 {
                    opt_deadline = Some(Instant::now() + Duration::from_millis(*ms as u64));
                }
            }
        }
    }
    let env_snapshot = ev.env.clone();
    let id = ev.next_task_id;
    ev.next_task_id += 1;
    let token = ev.cancel_token.clone().unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
    let token_for_task = token.clone();
    let limiter = opt_limiter.or_else(|| ev.thread_limiter.clone());
    let deadline = opt_deadline.or(ev.deadline);
    let rx = spawn_task(move || {
        if let Some(l) = limiter.as_ref() {
            l.acquire();
        }
        let mut ev2 = Evaluator::with_env_and_token(env_snapshot, token_for_task);
        ev2.thread_limiter = limiter;
        ev2.deadline = deadline;
        let out = ev2.eval(expr);
        if let Some(l) = ev2.thread_limiter.as_ref() {
            l.release();
        }
        out
    });
    ev.tasks.insert(id, TaskInfo { rx, cancel: token });
    Value::Expr { head: Box::new(Value::Symbol("FutureId".into())), args: vec![Value::Integer(id)] }
}

pub(crate) fn await_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("Await".into())), args };
    }
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
        },
    };
    if let Some(id) = id_opt {
        if let Some(task) = ev.tasks.remove(&id) {
            return task.rx.recv().unwrap_or(Value::Symbol("Null".into()));
        }
    }
    ev.make_error("Await: invalid or unknown future", "Await::invfuture")
}

pub(crate) fn cancel_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("Cancel".into())), args };
    }
    // Allow Cancel[FSWatch] to route to stdlib's CancelWatch
    if let Value::Assoc(m) = ev.eval(args[0].clone()) {
        if let Some(Value::String(t)) = m.get("__type") {
            if t == "FSWatch" {
                return ev.eval(Value::Expr { head: Box::new(Value::Symbol("CancelWatch".into())), args: vec![Value::Assoc(m)] });
            }
        }
    }
    // Accept FutureId[...] or integer id
    let id_opt = match &args[0] {
        Value::Expr { head, args } if matches!(&**head, Value::Symbol(s) if s=="FutureId") => {
            args.get(0).and_then(|v| if let Value::Integer(i) = v { Some(*i) } else { None })
        }
        Value::Integer(i) => Some(*i),
        other => match ev.eval(other.clone()) {
            Value::Integer(i) => Some(i),
            _ => None,
        },
    };
    if let Some(id) = id_opt {
        if let Some(task) = ev.tasks.get(&id) {
            task.cancel.store(true, Ordering::Relaxed);
            return Value::Boolean(true);
        }
    }
    Value::Boolean(false)
}
