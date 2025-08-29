use crate::register_if;
use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Mutex, OnceLock};

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

// -------- Set (handle-backed, stores key->value to preserve originals) --------
#[derive(Clone)]
struct SetState {
    elems: HashMap<String, Value>,
}

static SET_REG: OnceLock<Mutex<HashMap<i64, SetState>>> = OnceLock::new();
static NEXT_SET_ID: OnceLock<std::sync::atomic::AtomicI64> = OnceLock::new();
fn set_reg() -> &'static Mutex<HashMap<i64, SetState>> {
    SET_REG.get_or_init(|| Mutex::new(HashMap::new()))
}
fn next_set_id() -> i64 {
    let a = NEXT_SET_ID.get_or_init(|| std::sync::atomic::AtomicI64::new(1));
    a.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

fn set_handle(id: i64) -> Value {
    Value::Assoc(HashMap::from([
        (String::from("__type"), Value::String(String::from("Set"))),
        (String::from("id"), Value::Integer(id)),
    ]))
}
fn get_set(v: &Value) -> Option<i64> {
    if let Value::Assoc(m) = v {
        if matches!(m.get("__type"), Some(Value::String(s)) if s=="Set") {
            if let Some(Value::Integer(id)) = m.get("id") {
                return Some(*id);
            }
        }
    }
    None
}

fn key_of(v: &Value) -> String {
    lyra_runtime::eval::value_order_key(v)
}

fn set_create(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let mut elems: HashMap<String, Value> = HashMap::new();
    if let Some(arg0) = args.get(0) {
        match ev.eval(arg0.clone()) {
            Value::List(items) => {
                for it in items {
                    elems.insert(key_of(&it), it);
                }
            }
            other => {
                elems.insert(key_of(&other), other);
            }
        }
    }
    let id = next_set_id();
    set_reg().lock().unwrap().insert(id, SetState { elems });
    set_handle(id)
}

fn set_from_list(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [list] => match ev.eval(list.clone()) {
            Value::List(items) => set_create(ev, vec![Value::List(items)]),
            other => set_create(ev, vec![other]),
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("SetFromList".into())), args },
    }
}

fn set_to_list(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("SetToList".into())), args };
    }
    match get_set(&args[0]) {
        Some(id) => {
            let reg = set_reg().lock().unwrap();
            if let Some(st) = reg.get(&id) {
                Value::List(st.elems.values().cloned().collect())
            } else {
                Value::List(vec![])
            }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("SetToList".into())), args },
    }
}

fn set_insert(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("SetInsert".into())), args };
    }
    if let Some(id) = get_set(&args[0]) {
        let v = ev.eval(args[1].clone());
        let mut reg = set_reg().lock().unwrap();
        if let Some(st) = reg.get_mut(&id) {
            st.elems.insert(key_of(&v), v);
            return set_handle(id);
        }
    }
    Value::Expr { head: Box::new(Value::Symbol("SetInsert".into())), args }
}

fn set_remove(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("SetRemove".into())), args };
    }
    if let Some(id) = get_set(&args[0]) {
        let v = ev.eval(args[1].clone());
        let mut reg = set_reg().lock().unwrap();
        if let Some(st) = reg.get_mut(&id) {
            st.elems.remove(&key_of(&v));
            return set_handle(id);
        }
    }
    Value::Expr { head: Box::new(Value::Symbol("SetRemove".into())), args }
}

fn set_member_q(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("SetMemberQ".into())), args };
    }
    match get_set(&args[0]) {
        Some(id) => {
            let v = ev.eval(args[1].clone());
            let reg = set_reg().lock().unwrap();
            Value::Boolean(reg.get(&id).map(|s| s.elems.contains_key(&key_of(&v))).unwrap_or(false))
        }
        None => Value::Expr { head: Box::new(Value::Symbol("SetMemberQ".into())), args },
    }
}

// Legacy SetSize/SetEmptyQ removed; use Length/EmptyQ

fn set_union(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("SetUnion".into())), args };
    }
    let mut out: HashMap<String, Value> = HashMap::new();
    for a in args {
        if let Some(id) = get_set(&a) {
            if let Some(st) = set_reg().lock().unwrap().get(&id) {
                for (k, v) in &st.elems {
                    out.insert(k.clone(), v.clone());
                }
            }
        }
    }
    let id = next_set_id();
    set_reg().lock().unwrap().insert(id, SetState { elems: out });
    set_handle(id)
}

fn set_intersection(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("SetIntersection".into())), args };
    }
    let mut its: Option<HashSet<String>> = None;
    for a in &args {
        if let Some(id) = get_set(a) {
            if let Some(st) = set_reg().lock().unwrap().get(&id) {
                let keys: HashSet<String> = st.elems.keys().cloned().collect();
                its = Some(match its {
                    Some(cur) => cur.intersection(&keys).cloned().collect(),
                    None => keys,
                });
            }
        }
    }
    let mut elems = HashMap::new();
    if let Some(keys) = its {
        for a in &args {
            if let Some(id) = get_set(a) {
                if let Some(st) = set_reg().lock().unwrap().get(&id) {
                    for k in keys.iter() {
                        if let Some(v) = st.elems.get(k) {
                            elems.insert(k.clone(), v.clone());
                        }
                    }
                    break;
                }
            }
        }
    }
    let id = next_set_id();
    set_reg().lock().unwrap().insert(id, SetState { elems });
    set_handle(id)
}

fn set_difference(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("SetDifference".into())), args };
    }
    let a_id = match get_set(&args[0]) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("SetDifference".into())), args },
    };
    let mut out = {
        let reg = set_reg().lock().unwrap();
        reg.get(&a_id).map(|s| s.elems.clone()).unwrap_or_default()
    };
    let mut remove: HashSet<String> = HashSet::new();
    for b in args.into_iter().skip(1) {
        if let Some(id) = get_set(&b) {
            if let Some(st) = set_reg().lock().unwrap().get(&id) {
                for k in st.elems.keys() {
                    remove.insert(k.clone());
                }
            }
        }
    }
    for k in remove {
        out.remove(&k);
    }
    let id = next_set_id();
    set_reg().lock().unwrap().insert(id, SetState { elems: out });
    set_handle(id)
}

fn set_subset_q(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("SetSubsetQ".into())), args };
    }
    match (get_set(&args[0]), get_set(&args[1])) {
        (Some(a), Some(b)) => {
            let reg = set_reg().lock().unwrap();
            let aa = reg
                .get(&a)
                .map(|s| s.elems.keys().cloned().collect::<HashSet<_>>())
                .unwrap_or_default();
            let bb = reg
                .get(&b)
                .map(|s| s.elems.keys().cloned().collect::<HashSet<_>>())
                .unwrap_or_default();
            Value::Boolean(aa.is_subset(&bb))
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("SetSubsetQ".into())), args },
    }
}

fn set_equal_q(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("SetEqualQ".into())), args };
    }
    match (get_set(&args[0]), get_set(&args[1])) {
        (Some(a), Some(b)) => {
            let reg = set_reg().lock().unwrap();
            let aa = reg
                .get(&a)
                .map(|s| s.elems.keys().cloned().collect::<HashSet<_>>())
                .unwrap_or_default();
            let bb = reg
                .get(&b)
                .map(|s| s.elems.keys().cloned().collect::<HashSet<_>>())
                .unwrap_or_default();
            Value::Boolean(aa == bb)
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("SetEqualQ".into())), args },
    }
}

// List set ops (pure, order-stable by first list)
fn list_union(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("ListUnion".into())), args };
    }
    let lists: Vec<Value> = args.into_iter().map(|a| ev.eval(a)).collect();
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<Value> = Vec::new();
    for v in lists {
        if let Value::List(xs) = v {
            for x in xs {
                let k = key_of(&x);
                if seen.insert(k) {
                    out.push(x);
                }
            }
        }
    }
    Value::List(out)
}

fn list_intersection(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("ListIntersection".into())), args };
    }
    let a = ev.eval(args[0].clone());
    let b = ev.eval(args[1].clone());
    match (a, b) {
        (Value::List(la), Value::List(lb)) => {
            let sb: HashSet<String> = lb.iter().map(|x| key_of(x)).collect();
            let mut seen: HashSet<String> = HashSet::new();
            let mut out = Vec::new();
            for x in la {
                let k = key_of(&x);
                if sb.contains(&k) && seen.insert(k) {
                    out.push(x);
                }
            }
            Value::List(out)
        }
        (x, y) => Value::Expr {
            head: Box::new(Value::Symbol("ListIntersection".into())),
            args: vec![x, y],
        },
    }
}

fn list_difference(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("ListDifference".into())), args };
    }
    let a = ev.eval(args[0].clone());
    let b = ev.eval(args[1].clone());
    match (a, b) {
        (Value::List(la), Value::List(lb)) => {
            let sb: HashSet<String> = lb.iter().map(|x| key_of(x)).collect();
            let mut out = Vec::new();
            for x in la {
                if !sb.contains(&key_of(&x)) {
                    out.push(x);
                }
            }
            Value::List(out)
        }
        (x, y) => {
            Value::Expr { head: Box::new(Value::Symbol("ListDifference".into())), args: vec![x, y] }
        }
    }
}

// -------- Multiset / Bag --------
#[derive(Clone)]
struct BagState {
    counts: HashMap<String, (Value, i64)>,
}

static BAG_REG: OnceLock<Mutex<HashMap<i64, BagState>>> = OnceLock::new();
static NEXT_BAG_ID: OnceLock<std::sync::atomic::AtomicI64> = OnceLock::new();
fn bag_reg() -> &'static Mutex<HashMap<i64, BagState>> {
    BAG_REG.get_or_init(|| Mutex::new(HashMap::new()))
}
fn next_bag_id() -> i64 {
    let a = NEXT_BAG_ID.get_or_init(|| std::sync::atomic::AtomicI64::new(1));
    a.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}
fn bag_handle(id: i64) -> Value {
    Value::Assoc(HashMap::from([
        (String::from("__type"), Value::String(String::from("Bag"))),
        (String::from("id"), Value::Integer(id)),
    ]))
}
fn get_bag(v: &Value) -> Option<i64> {
    if let Value::Assoc(m) = v {
        if matches!(m.get("__type"), Some(Value::String(s)) if s=="Bag") {
            if let Some(Value::Integer(id)) = m.get("id") {
                return Some(*id);
            }
        }
    }
    None
}

fn bag_create(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let mut counts: HashMap<String, (Value, i64)> = HashMap::new();
    if let Some(arg0) = args.get(0) {
        match ev.eval(arg0.clone()) {
            Value::List(xs) => {
                for x in xs {
                    let k = key_of(&x);
                    let e = counts.entry(k).or_insert((x.clone(), 0));
                    e.1 += 1;
                }
            }
            Value::Assoc(m) => {
                for (k0, v0) in m {
                    let n = match v0 {
                        Value::Integer(n) => n.max(0),
                        _ => 0,
                    };
                    counts.insert(k0.clone(), (Value::String(k0), n));
                }
            }
            other => {
                let k = key_of(&other);
                counts.insert(k.clone(), (other, 1));
            }
        }
    }
    let id = next_bag_id();
    bag_reg().lock().unwrap().insert(id, BagState { counts });
    bag_handle(id)
}

fn bag_add(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("BagAdd".into())), args };
    }
    let id = match get_bag(&args[0]) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("BagAdd".into())), args },
    };
    let x = ev.eval(args[1].clone());
    let n = if args.len() >= 3 {
        match ev.eval(args[2].clone()) {
            Value::Integer(k) => k,
            _ => 1,
        }
    } else {
        1
    };
    let mut reg = bag_reg().lock().unwrap();
    if let Some(st) = reg.get_mut(&id) {
        let k = key_of(&x);
        let e = st.counts.entry(k).or_insert((x.clone(), 0));
        e.1 += n.max(0);
    }
    bag_handle(id)
}

fn bag_remove(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("BagRemove".into())), args };
    }
    let id = match get_bag(&args[0]) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("BagRemove".into())), args },
    };
    let x = ev.eval(args[1].clone());
    let n = if args.len() >= 3 {
        match ev.eval(args[2].clone()) {
            Value::Integer(k) => k,
            _ => 1,
        }
    } else {
        1
    };
    let mut reg = bag_reg().lock().unwrap();
    if let Some(st) = reg.get_mut(&id) {
        let k = key_of(&x);
        if let Some(e) = st.counts.get_mut(&k) {
            e.1 -= n.max(0);
            if e.1 <= 0 {
                st.counts.remove(&k);
            }
        }
    }
    bag_handle(id)
}

fn bag_count(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("BagCount".into())), args };
    }
    match get_bag(&args[0]) {
        Some(id) => {
            let x = ev.eval(args[1].clone());
            let reg = bag_reg().lock().unwrap();
            let n = reg.get(&id).and_then(|s| s.counts.get(&key_of(&x))).map(|e| e.1).unwrap_or(0);
            Value::Integer(n)
        }
        None => Value::Expr { head: Box::new(Value::Symbol("BagCount".into())), args },
    }
}

fn bag_size(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("BagSize".into())), args };
    }
    match get_bag(&args[0]) {
        Some(id) => {
            let reg = bag_reg().lock().unwrap();
            Value::Integer(reg.get(&id).map(|s| s.counts.values().map(|e| e.1).sum()).unwrap_or(0))
        }
        None => Value::Expr { head: Box::new(Value::Symbol("BagSize".into())), args },
    }
}

fn bag_union(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("BagUnion".into())), args };
    }
    let mut counts: HashMap<String, (Value, i64)> = HashMap::new();
    for a in args {
        if let Some(id) = get_bag(&a) {
            if let Some(st) = bag_reg().lock().unwrap().get(&id) {
                for (k, (v, n)) in &st.counts {
                    let e = counts.entry(k.clone()).or_insert((v.clone(), 0));
                    e.1 += *n;
                }
            }
        }
    }
    let id = next_bag_id();
    bag_reg().lock().unwrap().insert(id, BagState { counts });
    bag_handle(id)
}

fn bag_intersection(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("BagIntersection".into())), args };
    }
    let (a, b) = (get_bag(&args[0]), get_bag(&args[1]));
    match (a, b) {
        (Some(aid), Some(bid)) => {
            let reg = bag_reg().lock().unwrap();
            let (as_, bs_) = (
                reg.get(&aid).cloned().unwrap_or(BagState { counts: HashMap::new() }),
                reg.get(&bid).cloned().unwrap_or(BagState { counts: HashMap::new() }),
            );
            let mut out = HashMap::new();
            for (k, (v, na)) in as_.counts {
                if let Some((_vb, nb)) = bs_.counts.get(&k) {
                    out.insert(k, (v, na.min(*nb)));
                }
            }
            drop(reg);
            let id = next_bag_id();
            bag_reg().lock().unwrap().insert(id, BagState { counts: out });
            bag_handle(id)
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("BagIntersection".into())), args },
    }
}

fn bag_difference(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("BagDifference".into())), args };
    }
    let (a, b) = (get_bag(&args[0]), get_bag(&args[1]));
    match (a, b) {
        (Some(aid), Some(bid)) => {
            let reg = bag_reg().lock().unwrap();
            let (mut out, bs_) = (
                reg.get(&aid).cloned().unwrap_or(BagState { counts: HashMap::new() }),
                reg.get(&bid).cloned().unwrap_or(BagState { counts: HashMap::new() }),
            );
            for (k, (_, nb)) in bs_.counts {
                if let Some(e) = out.counts.get_mut(&k) {
                    e.1 -= nb;
                    if e.1 <= 0 {
                        out.counts.remove(&k);
                    }
                }
            }
            drop(reg);
            let id = next_bag_id();
            bag_reg().lock().unwrap().insert(id, out);
            bag_handle(id)
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("BagDifference".into())), args },
    }
}

// -------- Queue / Deque / Stack --------
#[derive(Clone)]
struct QueueState {
    q: VecDeque<Value>,
}
static QUEUE_REG: OnceLock<Mutex<HashMap<i64, QueueState>>> = OnceLock::new();
static NEXT_Q_ID: OnceLock<std::sync::atomic::AtomicI64> = OnceLock::new();
fn queue_reg() -> &'static Mutex<HashMap<i64, QueueState>> {
    QUEUE_REG.get_or_init(|| Mutex::new(HashMap::new()))
}
fn next_q_id() -> i64 {
    let a = NEXT_Q_ID.get_or_init(|| std::sync::atomic::AtomicI64::new(1));
    a.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}
fn queue_handle(id: i64) -> Value {
    Value::Assoc(HashMap::from([
        (String::from("__type"), Value::String(String::from("Queue"))),
        (String::from("id"), Value::Integer(id)),
    ]))
}
fn get_queue(v: &Value) -> Option<i64> {
    if let Value::Assoc(m) = v {
        if matches!(m.get("__type"), Some(Value::String(s)) if s=="Queue") {
            if let Some(Value::Integer(id)) = m.get("id") {
                return Some(*id);
            }
        }
    }
    None
}

fn queue_create(_ev: &mut Evaluator, _args: Vec<Value>) -> Value {
    let id = next_q_id();
    queue_reg().lock().unwrap().insert(id, QueueState { q: VecDeque::new() });
    queue_handle(id)
}
fn enqueue(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("Enqueue".into())), args };
    }
    let id = match get_queue(&args[0]) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("Enqueue".into())), args },
    };
    let v = ev.eval(args[1].clone());
    let mut reg = queue_reg().lock().unwrap();
    if let Some(st) = reg.get_mut(&id) {
        st.q.push_back(v);
    }
    queue_handle(id)
}
fn dequeue(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("Dequeue".into())), args };
    }
    match get_queue(&args[0]) {
        Some(id) => {
            let mut reg = queue_reg().lock().unwrap();
            if let Some(st) = reg.get_mut(&id) {
                st.q.pop_front().unwrap_or(Value::Symbol("Null".into()))
            } else {
                Value::Symbol("Null".into())
            }
        }
        None => Value::Expr { head: Box::new(Value::Symbol("Dequeue".into())), args },
    }
}
fn queue_peek(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("Peek".into())), args };
    }
    match get_queue(&args[0]) {
        Some(id) => {
            let reg = queue_reg().lock().unwrap();
            if let Some(st) = reg.get(&id) {
                st.q.front().cloned().unwrap_or(Value::Symbol("Null".into()))
            } else {
                Value::Symbol("Null".into())
            }
        }
        None => Value::Expr { head: Box::new(Value::Symbol("Peek".into())), args },
    }
}
// Legacy QueueSize/QueueEmptyQ removed; use Length/EmptyQ

// Stack
#[derive(Clone)]
struct StackState {
    s: Vec<Value>,
}
static STACK_REG: OnceLock<Mutex<HashMap<i64, StackState>>> = OnceLock::new();
static NEXT_S_ID: OnceLock<std::sync::atomic::AtomicI64> = OnceLock::new();
fn stack_reg() -> &'static Mutex<HashMap<i64, StackState>> {
    STACK_REG.get_or_init(|| Mutex::new(HashMap::new()))
}
fn next_s_id() -> i64 {
    let a = NEXT_S_ID.get_or_init(|| std::sync::atomic::AtomicI64::new(1));
    a.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}
fn stack_handle(id: i64) -> Value {
    Value::Assoc(HashMap::from([
        (String::from("__type"), Value::String(String::from("Stack"))),
        (String::from("id"), Value::Integer(id)),
    ]))
}
fn get_stack(v: &Value) -> Option<i64> {
    if let Value::Assoc(m) = v {
        if matches!(m.get("__type"), Some(Value::String(s)) if s=="Stack") {
            if let Some(Value::Integer(id)) = m.get("id") {
                return Some(*id);
            }
        }
    }
    None
}

fn stack_create(_ev: &mut Evaluator, _args: Vec<Value>) -> Value {
    let id = next_s_id();
    stack_reg().lock().unwrap().insert(id, StackState { s: Vec::new() });
    stack_handle(id)
}
fn push(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("Push".into())), args };
    }
    let id = match get_stack(&args[0]) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("Push".into())), args },
    };
    let v = ev.eval(args[1].clone());
    let mut reg = stack_reg().lock().unwrap();
    if let Some(st) = reg.get_mut(&id) {
        st.s.push(v);
    }
    stack_handle(id)
}
fn pop(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("Pop".into())), args };
    }
    match get_stack(&args[0]) {
        Some(id) => {
            let mut reg = stack_reg().lock().unwrap();
            if let Some(st) = reg.get_mut(&id) {
                st.s.pop().unwrap_or(Value::Symbol("Null".into()))
            } else {
                Value::Symbol("Null".into())
            }
        }
        None => Value::Expr { head: Box::new(Value::Symbol("Pop".into())), args },
    }
}
fn top(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("Top".into())), args };
    }
    match get_stack(&args[0]) {
        Some(id) => {
            let reg = stack_reg().lock().unwrap();
            if let Some(st) = reg.get(&id) {
                st.s.last().cloned().unwrap_or(Value::Symbol("Null".into()))
            } else {
                Value::Symbol("Null".into())
            }
        }
        None => Value::Expr { head: Box::new(Value::Symbol("Top".into())), args },
    }
}
// Legacy StackSize/StackEmptyQ removed; use Length/EmptyQ

// -------- Priority Queue (min/max by value_order_key) --------
#[derive(Clone)]
struct PQState {
    order: String,
    items: Vec<(String, i64, Value)>,
    next_seq: i64,
}
static PQ_REG: OnceLock<Mutex<HashMap<i64, PQState>>> = OnceLock::new();
static NEXT_PQ_ID: OnceLock<std::sync::atomic::AtomicI64> = OnceLock::new();
fn pq_reg() -> &'static Mutex<HashMap<i64, PQState>> {
    PQ_REG.get_or_init(|| Mutex::new(HashMap::new()))
}
fn next_pq_id() -> i64 {
    let a = NEXT_PQ_ID.get_or_init(|| std::sync::atomic::AtomicI64::new(1));
    a.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}
fn pq_handle(id: i64) -> Value {
    Value::Assoc(HashMap::from([
        (String::from("__type"), Value::String(String::from("PriorityQueue"))),
        (String::from("id"), Value::Integer(id)),
    ]))
}
fn get_pq(v: &Value) -> Option<i64> {
    if let Value::Assoc(m) = v {
        if matches!(m.get("__type"), Some(Value::String(s)) if s=="PriorityQueue") {
            if let Some(Value::Integer(id)) = m.get("id") {
                return Some(*id);
            }
        }
    }
    None
}

fn pq_create(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let order = if let Some(Value::Assoc(m)) = args.get(0).map(|x| ev.eval(x.clone())) {
        match m.get("Order") {
            Some(Value::String(s)) | Some(Value::Symbol(s)) if s == "max" => "max".to_string(),
            _ => "min".to_string(),
        }
    } else {
        "min".into()
    };
    let id = next_pq_id();
    pq_reg().lock().unwrap().insert(id, PQState { order, items: Vec::new(), next_seq: 1 });
    pq_handle(id)
}

fn pq_insert(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("PQInsert".into())), args };
    }
    let id = match get_pq(&args[0]) {
        Some(id) => id,
        None => return Value::Expr { head: Box::new(Value::Symbol("PQInsert".into())), args },
    };
    let v = ev.eval(args[1].clone());
    let key = key_of(&v);
    let mut reg = pq_reg().lock().unwrap();
    if let Some(st) = reg.get_mut(&id) {
        let seq = st.next_seq;
        st.next_seq += 1;
        st.items.push((key, seq, v));
        st.items.sort_by(|a, b| {
            let ord = a.0.cmp(&b.0);
            if st.order == "min" {
                ord.then_with(|| a.1.cmp(&b.1))
            } else {
                ord.reverse().then_with(|| a.1.cmp(&b.1))
            }
        });
    }
    pq_handle(id)
}
fn pq_pop(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("PQPop".into())), args };
    }
    match get_pq(&args[0]) {
        Some(id) => {
            let mut reg = pq_reg().lock().unwrap();
            if let Some(st) = reg.get_mut(&id) {
                if st.items.is_empty() {
                    Value::Symbol("Null".into())
                } else {
                    st.items.remove(0).2.clone()
                }
            } else {
                Value::Symbol("Null".into())
            }
        }
        None => Value::Expr { head: Box::new(Value::Symbol("PQPop".into())), args },
    }
}
fn pq_peek(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("PQPeek".into())), args };
    }
    match get_pq(&args[0]) {
        Some(id) => {
            let reg = pq_reg().lock().unwrap();
            if let Some(st) = reg.get(&id) {
                st.items.get(0).map(|t| t.2.clone()).unwrap_or(Value::Symbol("Null".into()))
            } else {
                Value::Symbol("Null".into())
            }
        }
        None => Value::Expr { head: Box::new(Value::Symbol("PQPeek".into())), args },
    }
}
// Legacy PQSize/PQEmptyQ removed; use Length/EmptyQ

/// Register collection types (Set, Queue, Deque, Bag, Counter) and
/// operations for creation, mutation, and conversion.
pub fn register_collections(ev: &mut Evaluator) {
    // Set
    ev.register("HashSet", set_create as NativeFn, Attributes::empty());
    ev.register("SetFromList", set_from_list as NativeFn, Attributes::empty());
    ev.register("SetToList", set_to_list as NativeFn, Attributes::empty());
    ev.register("SetInsert", set_insert as NativeFn, Attributes::empty());
    ev.register("SetRemove", set_remove as NativeFn, Attributes::empty());
    ev.register("SetMemberQ", set_member_q as NativeFn, Attributes::empty());
    // Legacy size/empty functions removed in favor of Length/EmptyQ
    ev.register("__SetUnion", set_union as NativeFn, Attributes::empty());
    ev.register("__SetIntersection", set_intersection as NativeFn, Attributes::empty());
    ev.register("__SetDifference", set_difference as NativeFn, Attributes::empty());
    ev.register("SetSubsetQ", set_subset_q as NativeFn, Attributes::empty());
    ev.register("SetEqualQ", set_equal_q as NativeFn, Attributes::empty());
    // List set ops
    // List-based set ops are handled in dispatch; internal entry points kept private if needed
    // Bag
    ev.register("Bag", bag_create as NativeFn, Attributes::empty());
    ev.register("BagAdd", bag_add as NativeFn, Attributes::empty());
    ev.register("BagRemove", bag_remove as NativeFn, Attributes::empty());
    ev.register("BagCount", bag_count as NativeFn, Attributes::empty());
    ev.register("BagSize", bag_size as NativeFn, Attributes::empty());
    ev.register("BagUnion", bag_union as NativeFn, Attributes::empty());
    ev.register("BagIntersection", bag_intersection as NativeFn, Attributes::empty());
    ev.register("BagDifference", bag_difference as NativeFn, Attributes::empty());
    // Queue
    ev.register("Queue", queue_create as NativeFn, Attributes::empty());
    ev.register("Enqueue", enqueue as NativeFn, Attributes::empty());
    ev.register("Dequeue", dequeue as NativeFn, Attributes::empty());
    ev.register("Peek", queue_peek as NativeFn, Attributes::empty());
    // Legacy size/empty functions removed in favor of Length/EmptyQ
    // Stack
    ev.register("Stack", stack_create as NativeFn, Attributes::empty());
    ev.register("Push", push as NativeFn, Attributes::empty());
    ev.register("Pop", pop as NativeFn, Attributes::empty());
    ev.register("Top", top as NativeFn, Attributes::empty());
    // Legacy size/empty functions removed in favor of Length/EmptyQ
    // Priority Queue
    ev.register("PriorityQueue", pq_create as NativeFn, Attributes::empty());
    ev.register("PQInsert", pq_insert as NativeFn, Attributes::empty());
    ev.register("PQPop", pq_pop as NativeFn, Attributes::empty());
    ev.register("PQPeek", pq_peek as NativeFn, Attributes::empty());
    // Legacy size/empty functions removed in favor of Length/EmptyQ
}

/// Conditionally register collection types and operations based on `pred`.
pub fn register_collections_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    register_if(ev, pred, "HashSet", set_create as NativeFn, Attributes::empty());
    register_if(ev, pred, "SetFromList", set_from_list as NativeFn, Attributes::empty());
    register_if(ev, pred, "SetToList", set_to_list as NativeFn, Attributes::empty());
    register_if(ev, pred, "SetInsert", set_insert as NativeFn, Attributes::empty());
    register_if(ev, pred, "SetRemove", set_remove as NativeFn, Attributes::empty());
    register_if(ev, pred, "SetMemberQ", set_member_q as NativeFn, Attributes::empty());
    // Legacy size/empty functions removed
    register_if(ev, pred, "__SetUnion", set_union as NativeFn, Attributes::empty());
    register_if(ev, pred, "__SetIntersection", set_intersection as NativeFn, Attributes::empty());
    register_if(ev, pred, "__SetDifference", set_difference as NativeFn, Attributes::empty());
    register_if(ev, pred, "SetSubsetQ", set_subset_q as NativeFn, Attributes::empty());
    register_if(ev, pred, "SetEqualQ", set_equal_q as NativeFn, Attributes::empty());
    // List-based set ops are handled in dispatch
    register_if(ev, pred, "Bag", bag_create as NativeFn, Attributes::empty());
    register_if(ev, pred, "BagAdd", bag_add as NativeFn, Attributes::empty());
    register_if(ev, pred, "BagRemove", bag_remove as NativeFn, Attributes::empty());
    register_if(ev, pred, "BagCount", bag_count as NativeFn, Attributes::empty());
    register_if(ev, pred, "BagSize", bag_size as NativeFn, Attributes::empty());
    register_if(ev, pred, "BagUnion", bag_union as NativeFn, Attributes::empty());
    register_if(ev, pred, "BagIntersection", bag_intersection as NativeFn, Attributes::empty());
    register_if(ev, pred, "BagDifference", bag_difference as NativeFn, Attributes::empty());
    register_if(ev, pred, "Queue", queue_create as NativeFn, Attributes::empty());
    register_if(ev, pred, "Enqueue", enqueue as NativeFn, Attributes::empty());
    register_if(ev, pred, "Dequeue", dequeue as NativeFn, Attributes::empty());
    register_if(ev, pred, "Peek", queue_peek as NativeFn, Attributes::empty());
    // Legacy size/empty functions removed
    register_if(ev, pred, "Stack", stack_create as NativeFn, Attributes::empty());
    register_if(ev, pred, "Push", push as NativeFn, Attributes::empty());
    register_if(ev, pred, "Pop", pop as NativeFn, Attributes::empty());
    register_if(ev, pred, "Top", top as NativeFn, Attributes::empty());
    // Legacy size/empty functions removed
    register_if(ev, pred, "PriorityQueue", pq_create as NativeFn, Attributes::empty());
    register_if(ev, pred, "PQInsert", pq_insert as NativeFn, Attributes::empty());
    register_if(ev, pred, "PQPop", pq_pop as NativeFn, Attributes::empty());
    register_if(ev, pred, "PQPeek", pq_peek as NativeFn, Attributes::empty());
    // Legacy size/empty functions removed
}
