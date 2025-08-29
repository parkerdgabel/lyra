use crate::concurrency::channels;
use crate::concurrency::futures;
use crate::concurrency::pool::spawn_task;
use crate::eval::{Evaluator, NativeFn};
use crate::attrs::Attributes;
use lyra_core::value::Value;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::{
    atomic::{AtomicI64, Ordering},
    Arc, Mutex,
};

#[derive(Clone)]
struct ActorInfo {
    chan_id: i64,
}

static ACT_REG: OnceLock<Mutex<HashMap<i64, ActorInfo>>> = OnceLock::new();
static NEXT_ACT_ID: OnceLock<AtomicI64> = OnceLock::new();

fn act_reg() -> &'static Mutex<HashMap<i64, ActorInfo>> {
    ACT_REG.get_or_init(|| Mutex::new(HashMap::new()))
}
fn next_act_id() -> i64 {
    let a = NEXT_ACT_ID.get_or_init(|| AtomicI64::new(1));
    a.fetch_add(1, Ordering::Relaxed)
}

pub(crate) fn register_actors(ev: &mut Evaluator) {
    ev.register("Actor", actor_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Tell", tell_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Ask", ask_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("StopActor", stop_actor_fn as NativeFn, Attributes::empty());
}

fn actor_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("Actor".into())), args };
    }
    let handler = args[0].clone();
    let cap = 64usize;
    let chan_id = channels::new_channel(cap);
    let chan = channels::get_channel(chan_id).unwrap();
    let token = ev.cancel_token.clone();
    let env_snapshot = ev.env.clone();
    let limiter = ev.thread_limiter.clone();
    let deadline = ev.deadline;
    let _ = spawn_task(move || {
        loop {
            let msg_opt = chan.recv(token.clone(), deadline);
            match msg_opt {
                Some(msg) => {
                    if let Some(l) = limiter.as_ref() { l.acquire(); }
                    let mut ev2 = if let Some(tok) = token.clone() { Evaluator::with_env_and_token(env_snapshot.clone(), tok) } else { Evaluator::with_env(env_snapshot.clone()) };
                    ev2.thread_limiter = limiter.clone();
                    ev2.deadline = deadline;
                    let call = Value::Expr { head: Box::new(handler.clone()), args: vec![msg] };
                    let _ = ev2.eval(call);
                    if let Some(l) = limiter.as_ref() { l.release(); }
                }
                None => break,
            }
        }
        Value::Symbol("Done".into())
    });
    let aid = next_act_id();
    act_reg().lock().unwrap().insert(aid, ActorInfo { chan_id });
    Value::Expr { head: Box::new(Value::Symbol("ActorId".into())), args: vec![Value::Integer(aid)] }
}

fn tell_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("Tell".into())), args };
    }
    let actv = ev.eval(args[0].clone());
    let msg = args[1].clone();
    let aid = match actv {
        Value::Expr { head, args } if matches!(&*head, Value::Symbol(s) if s=="ActorId") => args.get(0).and_then(|v| if let Value::Integer(i) = v { Some(*i) } else { None }),
        _ => None,
    };
    if let Some(id) = aid {
        if let Some(info) = act_reg().lock().unwrap().get(&id).cloned() {
            if let Some(ch) = channels::get_channel(info.chan_id) {
                let ok = ch.send(msg, ev.cancel_token.clone(), ev.deadline);
                return Value::Boolean(ok);
            }
        }
    }
    Value::Boolean(false)
}

fn stop_actor_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("StopActor".into())), args };
    }
    let actv = ev.eval(args[0].clone());
    let aid = match actv {
        Value::Expr { head, args } if matches!(&*head, Value::Symbol(s) if s=="ActorId") => args.get(0).and_then(|v| if let Value::Integer(i) = v { Some(*i) } else { None }),
        _ => None,
    };
    if let Some(id) = aid {
        if let Some(info) = act_reg().lock().unwrap().remove(&id) {
            if let Some(ch) = channels::get_channel(info.chan_id) { ch.close(); }
            return Value::Boolean(true);
        }
    }
    Value::Boolean(false)
}

fn ask_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("Ask".into())), args };
    }
    let actv = ev.eval(args[0].clone());
    let msg = args[1].clone();
    let mut recv_opts: Option<Value> = None;
    if args.len() >= 3 { recv_opts = Some(ev.eval(args[2].clone())); }
    let aid = match actv {
        Value::Expr { head, args } if matches!(&*head, Value::Symbol(s) if s=="ActorId") => args.get(0).and_then(|v| if let Value::Integer(i) = v { Some(*i) } else { None }),
        _ => None,
    };
    if let Some(id) = aid {
        let ch_id = channels::new_channel(1);
        let reply = Value::Expr { head: Box::new(Value::Symbol("ChannelId".into())), args: vec![Value::Integer(ch_id)] };
        let m = Value::Assoc(vec![("msg".to_string(), msg), ("replyTo".to_string(), reply.clone())].into_iter().collect());
        let _ = tell_fn(ev, vec![Value::Expr { head: Box::new(Value::Symbol("ActorId".into())), args: vec![Value::Integer(id)] }, m]);
        let recv_expr = if let Some(o) = recv_opts { Value::Expr { head: Box::new(Value::Symbol("Receive".into())), args: vec![reply, o] } } else { Value::Expr { head: Box::new(Value::Symbol("Receive".into())), args: vec![reply] } };
        return futures::future_fn(ev, vec![recv_expr]);
    }
    Value::Expr { head: Box::new(Value::Symbol("Ask".into())), args }
}

