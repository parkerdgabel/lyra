use lyra_core::value::Value;
use lyra_runtime::{Evaluator};
use lyra_runtime::attrs::Attributes;

pub fn register_dispatch(ev: &mut Evaluator) {
    // Register after list and dataset so we override conflicting names with a dispatcher.
    ev.register("Join", join_dispatch as NativeFn, Attributes::empty());
    ev.register("Union", union_dispatch as NativeFn, Attributes::empty());
}

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

fn is_dataset_handle(v: &Value) -> bool {
    if let Value::Assoc(m) = v {
        if let Some(Value::String(t)) = m.get("__type") { return t == "Dataset"; }
    }
    false
}

fn join_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [a, b] => {
            let av = ev.eval(a.clone());
            let bv = ev.eval(b.clone());
            match (&av, &bv) {
                (Value::List(la), Value::List(lb)) => {
                    let mut out = la.clone();
                    out.extend(lb.clone());
                    Value::List(out)
                }
                _ => {
                    // If any side is a dataset, delegate to dataset Join
                    if is_dataset_handle(&av) || is_dataset_handle(&bv) {
                        #[cfg(feature = "dataset")] { return crate::dataset::join_ds(ev, vec![av, bv, Value::List(vec![])]); }
                        #[cfg(not(feature = "dataset"))] { return Value::expr(Value::symbol("Join"), vec![av, bv]); }
                    }
                    // Fallback: keep unevaluated
                    Value::expr(Value::symbol("Join"), vec![av, bv])
                }
            }
        }
        // 3+ args: if any side is dataset, delegate; else leave unevaluated or try list semantics if both lists and third is not list.
        [a, b, rest @ ..] => {
            let av = ev.eval(a.clone());
            let bv = ev.eval(b.clone());
            if is_dataset_handle(&av) || is_dataset_handle(&bv) {
                let mut v = vec![av, bv];
                v.extend(rest.iter().cloned().map(|x| ev.eval(x)));
                #[cfg(feature = "dataset")] { return crate::dataset::join_ds(ev, v); }
                #[cfg(not(feature = "dataset"))] { return Value::expr(Value::symbol("Join"), v); }
            }
            Value::expr(Value::symbol("Join"), {
                let mut v = vec![av, bv];
                v.extend(rest.iter().cloned().map(|x| ev.eval(x)));
                v
            })
        }
        _ => Value::expr(Value::symbol("Join"), args),
    }
}

fn is_set_handle(v: &Value) -> bool {
    if let Value::Assoc(m) = v { if let Some(Value::String(t)) = m.get("__type") { return t == "Set"; } }
    false
}

fn all_lists(vs: &[Value]) -> bool { vs.iter().all(|v| matches!(v, Value::List(_))) }
fn list_is_assoc_rows(v: &Value) -> bool { if let Value::List(xs) = v { return xs.iter().all(|x| matches!(x, Value::Assoc(_))); } false }

fn union_dispatch(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::expr(Value::symbol("Union"), args); }
    let evald: Vec<Value> = args.into_iter().map(|a| ev.eval(a)).collect();
    if evald.iter().any(|v| is_dataset_handle(v)) {
        #[cfg(feature = "dataset")] { return crate::dataset::union_general(ev, evald); }
        #[cfg(not(feature = "dataset"))] { return Value::expr(Value::symbol("Union"), evald); }
    }
    // If all args are lists and appear to be rows (assoc), delegate to dataset union_general for row-wise union-by-columns
    if !evald.is_empty() && all_lists(&evald) && evald.iter().any(|v| list_is_assoc_rows(v)) {
        #[cfg(feature = "dataset")] { return crate::dataset::union_general(ev, evald); }
        #[cfg(not(feature = "dataset"))] { return Value::expr(Value::symbol("Union"), evald); }
    }
    // If any are Set handles, use SetUnion
    if evald.iter().any(|v| is_set_handle(v)) {
        return ev.eval(Value::expr(Value::symbol("SetUnion"), evald));
    }
    // Default for lists: ListUnion (order-stable)
    if all_lists(&evald) {
        return ev.eval(Value::expr(Value::symbol("ListUnion"), evald));
    }
    // Fallback: leave unevaluated
    Value::expr(Value::symbol("Union"), evald)
}
