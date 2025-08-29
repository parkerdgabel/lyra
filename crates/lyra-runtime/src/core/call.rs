//! Call pipeline helpers.
//!
//! This module centralizes mechanics that shape how a call’s arguments are
//! prepared before invoking a builtin implementation, independent from
//! rewrite semantics:
//! - Sequence splicing: expand top‑level `Sequence[...]` in argument lists.
//! - Hold semantics: respect `HoldAll`, `HoldFirst`, `HoldRest` when
//!   evaluating arguments.
//! - Flat: flatten same‑head nested calls into a single argument list.
//! - Orderless: canonical sort of arguments (and traced variant for Explain).
//!
//! The evaluator delegates here after it resolves the callee head and before
//! invoking the builtin function pointer.
//! Rewrite‑driven behaviors (Sub/Up/Down/Own values, Replace*, Thread, etc.)
//! remain in `core::rewrite`.

use crate::attrs::Attributes;
use crate::eval::Evaluator;
// tracing utils are not directly used; the traced helpers push steps manually
use lyra_core::value::Value;

/// Expand any top-level `Sequence[...]` arguments into positional arguments.
pub(crate) fn splice_sequences(args: Vec<Value>) -> Vec<Value> {
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

/// Flatten same-head nested calls for functions with `FLAT` attribute.
/// Example: `Plus[1, Plus[2, 3]] -> Plus[1, 2, 3]`.
pub(crate) fn flat_flatten(fname: &str, args: Vec<Value>) -> Vec<Value> {
    let mut flat: Vec<Value> = Vec::with_capacity(args.len());
    for a in args.into_iter() {
        if let Value::Expr { head: h2, args: a2 } = &a {
            if matches!(&**h2, Value::Symbol(s) if s == fname) {
                flat.extend(a2.clone());
                continue;
            }
        }
        flat.push(a);
    }
    flat
}

/// Traced variant of `flat_flatten` that records a `FlatFlatten` step.
pub(crate) fn flat_flatten_traced(ev: &mut Evaluator, head_for_trace: &Value, fname: &str, args: Vec<Value>) -> Vec<Value> {
    let mut flat: Vec<Value> = Vec::with_capacity(args.len());
    for a in args.into_iter() {
        if let Value::Expr { head: h2, args: a2 } = &a {
            if matches!(&**h2, Value::Symbol(s) if s == fname) {
                let data = Value::Assoc(
                    vec![("added".to_string(), Value::Integer(a2.len() as i64))]
                        .into_iter()
                        .collect(),
                );
                ev.trace_steps.push(Value::Assoc(
                    vec![
                        ("action".to_string(), Value::String("FlatFlatten".into())),
                        ("head".to_string(), head_for_trace.clone()),
                        ("data".to_string(), data),
                    ]
                    .into_iter()
                    .collect(),
                ));
                flat.extend(a2.clone());
                continue;
            }
        }
        flat.push(a);
    }
    flat
}

/// Sort arguments canonically for functions with `ORDERLESS` attribute.
pub(crate) fn orderless_sort(mut args: Vec<Value>) -> Vec<Value> {
    args.sort_by(|x, y| crate::core::rewrite::value_order(x).cmp(&crate::core::rewrite::value_order(y)));
    args
}

/// Traced variant of `orderless_sort` that records final order in `Explain`.
pub(crate) fn orderless_sort_traced(ev: &mut Evaluator, head_for_trace: &Value, mut args: Vec<Value>) -> Vec<Value> {
    args.sort_by(|x, y| crate::core::rewrite::value_order(x).cmp(&crate::core::rewrite::value_order(y)));
    let data = Value::Assoc(
        vec![("finalOrder".to_string(), Value::List(args.clone()))]
            .into_iter()
            .collect(),
    );
    ev.trace_steps.push(Value::Assoc(
        vec![
            ("action".to_string(), Value::String("OrderlessSort".into())),
            ("head".to_string(), head_for_trace.clone()),
            ("data".to_string(), data),
        ]
        .into_iter()
        .collect(),
    ));
    args
}

/// Compute 1-based positions of held arguments given attributes and arity.
pub(crate) fn held_positions(attrs: Attributes, arg_len: usize) -> Vec<i64> {
    if attrs.contains(Attributes::HOLD_ALL) {
        (1..=arg_len).map(|i| i as i64).collect()
    } else if attrs.contains(Attributes::HOLD_FIRST) {
        if arg_len == 0 { vec![] } else { vec![1] }
    } else if attrs.contains(Attributes::HOLD_REST) {
        if arg_len <= 1 { vec![] } else { (2..=arg_len).map(|i| i as i64).collect() }
    } else {
        vec![]
    }
}

/// Evaluate arguments respecting `HoldAll`/`HoldFirst`/`HoldRest` attributes.
pub(crate) fn eval_with_hold(ev: &mut Evaluator, args: Vec<Value>, attrs: Attributes) -> Vec<Value> {
    if attrs.contains(Attributes::HOLD_ALL) {
        return args;
    }
    if attrs.contains(Attributes::HOLD_FIRST) {
        if args.is_empty() {
            return vec![];
        }
        let mut out = Vec::with_capacity(args.len());
        out.push(args[0].clone());
        for a in args.into_iter().skip(1) {
            out.push(ev.eval(a));
        }
        return out;
    }
    if attrs.contains(Attributes::HOLD_REST) {
        if args.len() <= 1 {
            return args.into_iter().map(|a| ev.eval(a)).collect();
        }
        let mut it = args.into_iter();
        let mut out = Vec::new();
        if let Some(first) = it.next() { out.push(ev.eval(first)); }
        out.extend(it);
        return out;
    }
    args.into_iter().map(|a| ev.eval(a)).collect()
}

/// Thread list arguments elementwise for functions with `LISTABLE` attribute.
/// Preserves argument order and length, returning a list of results.
pub(crate) fn listable_thread(ev: &mut Evaluator, f: crate::eval::NativeFn, args: Vec<Value>) -> Value {
    let mut target_len: Option<usize> = None;
    let mut saw_list = false;
    let mut arg_lens: Vec<i64> = Vec::with_capacity(args.len());
    for a in &args {
        if let Value::List(v) = a {
            saw_list = true;
            let l = v.len();
            arg_lens.push(l as i64);
            if l > 1 {
                match target_len {
                    None => target_len = Some(l),
                    Some(t) if t == l => {}
                    Some(_) => {
                        let mut m = std::collections::HashMap::new();
                        m.insert("message".into(), Value::String("Failure".into()));
                        m.insert("tag".into(), Value::String("Listable::lengthMismatch".into()));
                        m.insert(
                            "argLens".into(),
                            Value::List(arg_lens.iter().map(|n| Value::Integer(*n)).collect()),
                        );
                        return Value::Assoc(m);
                    }
                }
            }
        } else {
            arg_lens.push(0);
        }
    }
    if !saw_list {
        let evald: Vec<Value> = args.into_iter().map(|x| ev.eval(x)).collect();
        return f(ev, evald);
    }
    let len = target_len.unwrap_or(1);
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let mut elem_args = Vec::with_capacity(args.len());
        for a in &args {
            match a {
                Value::List(vs) => {
                    let l = vs.len();
                    if l == 0 {
                        elem_args.push(Value::List(vec![]));
                    } else if l == 1 {
                        elem_args.push(vs[0].clone());
                    } else {
                        elem_args.push(vs[i].clone());
                    }
                }
                other => elem_args.push(other.clone()),
            }
        }
        let evald: Vec<Value> = elem_args.into_iter().map(|x| ev.eval(x)).collect();
        out.push(f(ev, evald));
    }
    Value::List(out)
}
