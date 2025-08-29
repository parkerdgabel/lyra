use crate::attrs::Attributes;
use crate::concurrency::pool::ThreadLimiter; // not used directly but keeps module grouping consistent
use crate::eval::{Evaluator, NativeFn};
use lyra_core::value::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::OnceLock;
use std::sync::{
    atomic::{AtomicBool, AtomicI64, Ordering},
    Arc, Condvar, Mutex,
};
use std::time::{Duration, Instant};

pub(crate) struct ChannelQueue {
    cap: usize,
    inner: Mutex<ChannelInner>,
    not_full: Condvar,
    not_empty: Condvar,
}

struct ChannelInner {
    q: VecDeque<Value>,
    closed: bool,
}

impl ChannelQueue {
    pub(crate) fn new(cap: usize) -> Self {
        Self { cap, inner: Mutex::new(ChannelInner { q: VecDeque::new(), closed: false }), not_full: Condvar::new(), not_empty: Condvar::new() }
    }
    pub(crate) fn send(&self, v: Value, cancel: Option<Arc<AtomicBool>>, deadline: Option<Instant>) -> bool {
        let mut guard = self.inner.lock().unwrap();
        loop {
            if guard.closed { return false; }
            if guard.q.len() < self.cap {
                guard.q.push_back(v);
                self.not_empty.notify_one();
                return true;
            }
            if let Some(tok) = &cancel { if tok.load(Ordering::Relaxed) { return false; } }
            if let Some(dl) = deadline { if Instant::now() > dl { return false; } }
            if let Some(dl) = deadline {
                let now = Instant::now();
                if dl <= now { return false; }
                let to = dl - now;
                let (g, _timeout_res) = self.not_full.wait_timeout(guard, to).unwrap();
                guard = g;
            } else {
                let (g, _timeout) = self.not_full.wait_timeout(guard, Duration::from_millis(5)).unwrap();
                guard = g;
            }
        }
    }
    pub(crate) fn recv(&self, cancel: Option<Arc<AtomicBool>>, deadline: Option<Instant>) -> Option<Value> {
        let mut guard = self.inner.lock().unwrap();
        loop {
            if let Some(v) = guard.q.pop_front() { self.not_full.notify_one(); return Some(v); }
            if guard.closed { return None; }
            if let Some(tok) = &cancel { if tok.load(Ordering::Relaxed) { return None; } }
            if let Some(dl) = deadline { if Instant::now() > dl { return None; } }
            if let Some(dl) = deadline {
                let now = Instant::now();
                if dl <= now { return None; }
                let to = dl - now;
                let (g, _) = self.not_empty.wait_timeout(guard, to).unwrap();
                guard = g;
            } else {
                let (g, _) = self.not_empty.wait_timeout(guard, Duration::from_millis(5)).unwrap();
                guard = g;
            }
        }
    }
    pub(crate) fn close(&self) {
        let mut guard = self.inner.lock().unwrap();
        guard.closed = true;
        self.not_full.notify_all();
        self.not_empty.notify_all();
    }
}

static CH_REG: OnceLock<Mutex<HashMap<i64, Arc<ChannelQueue>>>> = OnceLock::new();
static NEXT_CH_ID: OnceLock<AtomicI64> = OnceLock::new();

fn ch_reg() -> &'static Mutex<HashMap<i64, Arc<ChannelQueue>>> {
    CH_REG.get_or_init(|| Mutex::new(HashMap::new()))
}
fn next_ch_id() -> i64 {
    let a = NEXT_CH_ID.get_or_init(|| AtomicI64::new(1));
    a.fetch_add(1, Ordering::Relaxed)
}

pub(crate) fn new_channel(cap: usize) -> i64 {
    let id = next_ch_id();
    ch_reg().lock().unwrap().insert(id, Arc::new(ChannelQueue::new(cap)));
    id
}

pub(crate) fn get_channel(id: i64) -> Option<Arc<ChannelQueue>> {
    ch_reg().lock().unwrap().get(&id).cloned()
}

pub(crate) fn register_channels(ev: &mut Evaluator) {
    ev.register("BoundedChannel", bounded_channel_fn as NativeFn, Attributes::empty());
    ev.register("Send", send_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Receive", receive_fn as NativeFn, Attributes::empty());
    ev.register("CloseChannel", close_channel_fn as NativeFn, Attributes::empty());
    ev.register("TrySend", try_send_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("TryReceive", try_receive_fn as NativeFn, Attributes::empty());
}

fn bounded_channel_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let cap = match args.get(0) { Some(Value::Integer(n)) if *n > 0 => *n as usize, _ => 16 };
    let id = new_channel(cap);
    Value::Expr { head: Box::new(Value::Symbol("ChannelId".into())), args: vec![Value::Integer(id)] }
}

fn send_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("Send".into())), args }; }
    let chv = ev.eval(args[0].clone());
    let val = ev.eval(args[1].clone());
    let mut call_deadline: Option<Instant> = None;
    if args.len() >= 3 {
        if let Value::Assoc(m) = ev.eval(args[2].clone()) {
            if let Some(Value::Integer(ms)) = m.get("TimeoutMs").or_else(|| m.get("timeoutMs")) {
                if *ms > 0 { call_deadline = Some(Instant::now() + Duration::from_millis(*ms as u64)); }
            }
        }
    }
    let cid = match chv {
        Value::Expr { head, args } if matches!(&*head, Value::Symbol(s) if s=="ChannelId") => args.get(0).and_then(|v| if let Value::Integer(i) = v { Some(*i) } else { None }),
        _ => None,
    };
    if let Some(id) = cid {
        if let Some(ch) = get_channel(id) {
            let ok = ch.send(val, ev.cancel_token.clone(), call_deadline.or(ev.deadline));
            return Value::Boolean(ok);
        }
    }
    Value::Boolean(false)
}

fn receive_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Receive".into())), args }; }
    let chv = ev.eval(args[0].clone());
    let mut call_deadline: Option<Instant> = None;
    if args.len() >= 2 {
        if let Value::Assoc(m) = ev.eval(args[1].clone()) {
            if let Some(Value::Integer(ms)) = m.get("TimeoutMs").or_else(|| m.get("timeoutMs")) {
                if *ms > 0 { call_deadline = Some(Instant::now() + Duration::from_millis(*ms as u64)); }
            }
        }
    }
    let cid = match chv {
        Value::Expr { head, args } if matches!(&*head, Value::Symbol(s) if s=="ChannelId") => args.get(0).and_then(|v| if let Value::Integer(i) = v { Some(*i) } else { None }),
        _ => None,
    };
    if let Some(id) = cid {
        if let Some(ch) = get_channel(id) {
            return ch.recv(ev.cancel_token.clone(), call_deadline.or(ev.deadline)).unwrap_or(Value::Symbol("Null".into()));
        }
    }
    Value::Symbol("Null".into())
}

fn close_channel_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::Expr { head: Box::new(Value::Symbol("CloseChannel".into())), args }; }
    let chv = ev.eval(args[0].clone());
    let cid = match chv {
        Value::Expr { head, args } if matches!(&*head, Value::Symbol(s) if s=="ChannelId") => args.get(0).and_then(|v| if let Value::Integer(i) = v { Some(*i) } else { None }),
        _ => None,
    };
    if let Some(id) = cid { if let Some(ch) = get_channel(id) { ch.close(); return Value::Boolean(true); } }
    Value::Boolean(false)
}

fn try_send_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("TrySend".into())), args }; }
    let chv = ev.eval(args[0].clone());
    let val = ev.eval(args[1].clone());
    let cid = match chv {
        Value::Expr { head, args } if matches!(&*head, Value::Symbol(s) if s=="ChannelId") => args.get(0).and_then(|v| if let Value::Integer(i) = v { Some(*i) } else { None }),
        _ => None,
    };
    if let Some(id) = cid { if let Some(ch) = get_channel(id) { let ok = ch.send(val, ev.cancel_token.clone(), Some(Instant::now())); return Value::Boolean(ok); } }
    Value::Boolean(false)
}

fn try_receive_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::Expr { head: Box::new(Value::Symbol("TryReceive".into())), args }; }
    let chv = ev.eval(args[0].clone());
    let cid = match chv {
        Value::Expr { head, args } if matches!(&*head, Value::Symbol(s) if s=="ChannelId") => args.get(0).and_then(|v| if let Value::Integer(i) = v { Some(*i) } else { None }),
        _ => None,
    };
    if let Some(id) = cid { if let Some(ch) = get_channel(id) { return ch.recv(ev.cancel_token.clone(), Some(Instant::now())).unwrap_or(Value::Symbol("Null".into())); } }
    Value::Symbol("Null".into())
}

