use lyra_core::value::Value;
use crate::matcher::{Bindings, MatcherCtx, match_rule_with, substitute_named};

pub fn rewrite_once_with_ctx(ctx: &MatcherCtx, v: Value, rules: &[(Value, Value)]) -> Value {
    for (lhs, rhs) in rules {
        if let Some(b) = match_rule_with(ctx, lhs, &v) {
            let replaced = substitute_named(rhs, &b);
            return replaced;
        }
    }
    match v {
        Value::List(items) => {
            let mut out = Vec::new();
            for it in items { let r = rewrite_once_with_ctx(ctx, it, rules); if let Value::Expr { head, args } = &r { if matches!(**head, Value::Symbol(ref s) if s=="Sequence") { out.extend(args.clone()); continue; } } out.push(r); }
            Value::List(out)
        }
        Value::Assoc(m) => Value::Assoc(m.into_iter().map(|(k,x)| (k, rewrite_once_with_ctx(ctx, x, rules))).collect()),
        Value::Expr { head, args } => {
            let new_head = rewrite_once_with_ctx(ctx, *head, rules);
            let mut new_args: Vec<Value> = Vec::new();
            for a in args { let r = rewrite_once_with_ctx(ctx, a, rules); if let Value::Expr { head, args } = &r { if matches!(**head, Value::Symbol(ref s) if s=="Sequence") { new_args.extend(args.clone()); continue; } } new_args.push(r); }
            if let Value::Symbol(ref s) = new_head { if s=="Sequence" { return Value::List(new_args); } }
            Value::Expr { head: Box::new(new_head), args: new_args }
        }
        other => other,
    }
}

pub fn rewrite_once(v: Value, rules: &[(Value, Value)]) -> Value { rewrite_once_with_ctx(&MatcherCtx::default(), v, rules) }

pub fn rewrite_all_with_ctx(ctx: &MatcherCtx, v: Value, rules: &[(Value, Value)]) -> Value {
    let mut curr = v;
    loop {
        let next = rewrite_once_with_ctx(ctx, curr.clone(), rules);
        if next == curr { return next; }
        curr = next;
    }
}

pub fn rewrite_all(v: Value, rules: &[(Value, Value)]) -> Value { rewrite_all_with_ctx(&MatcherCtx::default(), v, rules) }

pub fn rewrite_with_limit_with_ctx(ctx: &MatcherCtx, v: Value, rules: &[(Value, Value)], mut limit: usize) -> Value {
    fn go(ctx: &MatcherCtx, v: Value, rules: &[(Value, Value)], limit: &mut usize) -> Value {
        if *limit == 0 { return v; }
        for (lhs, rhs) in rules {
            if let Some(b) = match_rule_with(ctx, lhs, &v) {
                *limit = limit.saturating_sub(1);
                return substitute_named(rhs, &b);
            }
        }
        match v {
            Value::List(items) => {
                let mut out = Vec::new();
                for it in items { let r = go(ctx, it, rules, limit); if let Value::Expr { head, args } = &r { if matches!(**head, Value::Symbol(ref s) if s=="Sequence") { out.extend(args.clone()); continue; } } out.push(r); }
                Value::List(out)
            }
            Value::Assoc(m) => Value::Assoc(m.into_iter().map(|(k,x)| (k, go(ctx, x, rules, limit))).collect()),
            Value::Expr { head, args } => {
                let new_head = go(ctx, *head, rules, limit);
                let mut new_args: Vec<Value> = Vec::new();
                for a in args { if *limit == 0 { new_args.push(a); continue; } let r = go(ctx, a, rules, limit); if let Value::Expr { head, args } = &r { if matches!(**head, Value::Symbol(ref s) if s=="Sequence") { new_args.extend(args.clone()); continue; } } new_args.push(r); }
                if let Value::Symbol(ref s) = new_head { if s=="Sequence" { return Value::List(new_args); } }
                Value::Expr { head: Box::new(new_head), args: new_args }
            }
            other => other,
        }
    }
    go(ctx, v, rules, &mut limit)
}

pub fn rewrite_with_limit(v: Value, rules: &[(Value, Value)], limit: usize) -> Value { rewrite_with_limit_with_ctx(&MatcherCtx::default(), v, rules, limit) }
