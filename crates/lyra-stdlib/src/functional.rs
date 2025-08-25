use lyra_core::value::Value;
use lyra_runtime::{Evaluator};
use lyra_runtime::attrs::Attributes;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

pub fn register_functional(ev: &mut Evaluator) {
    ev.register("Apply", apply_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Compose", compose_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("RightCompose", right_compose_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Nest", nest_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("NestList", nest_list_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("FoldList", fold_list_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("FixedPoint", fixed_point_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("FixedPointList", fixed_point_list_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Through", through_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Identity", identity_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("ConstantFunction", constant_function_fn as NativeFn, Attributes::HOLD_ALL);
}

fn uneval(head: &str, args: Vec<Value>) -> Value {
    Value::Expr { head: Box::new(Value::Symbol(head.into())), args }
}

// Apply[f, list] or Apply[f, expr]
fn apply_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        // Apply at level 1: Apply[f, expr, 1] or Apply[f, expr, {1}]
        [f, expr, lvl] => {
            let lvl_ok = match lvl {
                Value::Integer(n) if *n == 1 => true,
                Value::List(vs) if vs.len()==1 && matches!(vs[0], Value::Integer(1)) => true,
                _ => false,
            };
            if !lvl_ok { return uneval("Apply", args); }
            match ev.eval(expr.clone()) {
                Value::List(items) => {
                    let mut out: Vec<Value> = Vec::with_capacity(items.len());
                    for it in items {
                        // Apply to each element
                        out.push(apply_fn(ev, vec![f.clone(), it]));
                    }
                    Value::List(out)
                }
                other => uneval("Apply", vec![f.clone(), other, lvl.clone()]),
            }
        }
        // Apply[f, list] or Apply[f, expr]
        [f, expr] => {
            let fcl = f.clone();
            match ev.eval(expr.clone()) {
                Value::List(items) => {
                    let call = Value::Expr { head: Box::new(fcl), args: items.into_iter().map(|x| ev.eval(x)).collect() };
                    ev.eval(call)
                }
                Value::Expr { args: expr_args, .. } => {
                    let call = Value::Expr { head: Box::new(fcl), args: expr_args.into_iter().map(|x| ev.eval(x)).collect() };
                    ev.eval(call)
                }
                other => uneval("Apply", vec![f.clone(), other]),
            }
        }
        _ => uneval("Apply", args),
    }
}

// Compose[f, g, h] => PureFunction composing left-to-right: f[g[h[#]]]&
fn compose_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let mut fns: Vec<Value> = match args.as_slice() {
        [Value::List(fs)] => fs.clone(),
        _ => args.clone(),
    };
    if fns.is_empty() { return Value::pure_function(None, Value::Slot(None)); }
    // Compose normalizes by evaluating function heads minimally (do not apply)
    for f in &mut fns { *f = f.clone(); }
    // Build nested call: f1[f2[...fn[#]...]]
    let mut body = Value::Slot(None);
    for f in fns.into_iter().rev() {
        body = Value::Expr { head: Box::new(f), args: vec![body] };
    }
    Value::pure_function(None, body)
}

// RightCompose[f, g, h] => h[g[f[#]]]&
fn right_compose_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let mut fns: Vec<Value> = match args.as_slice() {
        [Value::List(fs)] => fs.clone(),
        _ => args.clone(),
    };
    if fns.is_empty() { return Value::pure_function(None, Value::Slot(None)); }
    // Build nested call: h[g[f[#]]]
    let mut body = Value::Slot(None);
    for f in fns.into_iter() {
        body = Value::Expr { head: Box::new(f), args: vec![body] };
    }
    Value::pure_function(None, body)
}

// Nest[f, x, n]
fn nest_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=3 { return uneval("Nest", args); }
    let f = args[0].clone();
    let mut x = ev.eval(args[1].clone());
    let n = match ev.eval(args[2].clone()) { Value::Integer(k) if k>=0 => k as usize, other => return uneval("Nest", vec![f, x, other]) };
    for _ in 0..n { x = ev.eval(Value::Expr { head: Box::new(f.clone()), args: vec![x] }); }
    x
}

// NestList[f, x, n]
fn nest_list_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=3 { return uneval("NestList", args); }
    let f = args[0].clone();
    let mut x = ev.eval(args[1].clone());
    let n = match ev.eval(args[2].clone()) { Value::Integer(k) if k>=0 => k as usize, other => return uneval("NestList", vec![f, x, other]) };
    let mut out = Vec::with_capacity(n+1);
    out.push(x.clone());
    for _ in 0..n { x = ev.eval(Value::Expr { head: Box::new(f.clone()), args: vec![x] }); out.push(x.clone()); }
    Value::List(out)
}

// FoldList[f, list] or FoldList[f, init, list]
fn fold_list_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [f, list] => match ev.eval(list.clone()) {
            Value::List(items) => {
                let mut out: Vec<Value> = Vec::new();
                let mut it = items.into_iter();
                if let Some(first) = it.next() {
                    let mut acc = ev.eval(first);
                    out.push(acc.clone());
                    for v in it {
                        let vv = ev.eval(v);
                        acc = ev.eval(Value::Expr { head: Box::new(f.clone()), args: vec![acc, vv] });
                        out.push(acc.clone());
                    }
                }
                Value::List(out)
            }
            other => other,
        },
        [f, init, list] => match ev.eval(list.clone()) {
            Value::List(items) => {
                let mut out: Vec<Value> = Vec::with_capacity(items.len()+1);
                let mut acc = ev.eval(init.clone());
                out.push(acc.clone());
                for v in items {
                    let vv = ev.eval(v);
                    acc = ev.eval(Value::Expr { head: Box::new(f.clone()), args: vec![acc, vv] });
                    out.push(acc.clone());
                }
                Value::List(out)
            }
            other => other,
        },
        _ => uneval("FoldList", args),
    }
}

fn same_value(a: &Value, b: &Value) -> bool { a == b }

// FixedPoint[f, x, opts?] with MaxIterations
fn fixed_point_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<2 { return uneval("FixedPoint", args); }
    let f = args[0].clone();
    let mut cur = ev.eval(args[1].clone());
    let mut max_iter: usize = 100;
    if args.len()>=3 {
        if let Value::Assoc(m) = ev.eval(args[2].clone()) {
            if let Some(Value::Integer(n)) = m.get("MaxIterations") { if *n>0 { max_iter = *n as usize; } }
        }
    }
    for _ in 0..max_iter {
        let next = ev.eval(Value::Expr { head: Box::new(f.clone()), args: vec![cur.clone()] });
        if same_value(&next, &cur) { return next; }
        cur = next;
    }
    cur
}

// FixedPointList[f, x, opts?]
fn fixed_point_list_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()<2 { return uneval("FixedPointList", args); }
    let f = args[0].clone();
    let mut cur = ev.eval(args[1].clone());
    let mut max_iter: usize = 100;
    if args.len()>=3 {
        if let Value::Assoc(m) = ev.eval(args[2].clone()) {
            if let Some(Value::Integer(n)) = m.get("MaxIterations") { if *n>0 { max_iter = *n as usize; } }
        }
    }
    let mut out = vec![cur.clone()];
    for _ in 0..max_iter {
        let next = ev.eval(Value::Expr { head: Box::new(f.clone()), args: vec![cur.clone()] });
        out.push(next.clone());
        if same_value(&next, &cur) { break; }
        cur = next;
    }
    Value::List(out)
}

// Through[{f,g}, x] or Through[{f,g}][x]
fn through_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::List(funcs)] => {
            // Operator form: PureFunction that calls Through[funcs, #]
            let body = Value::Expr { head: Box::new(Value::Symbol("Through".into())), args: vec![Value::List(funcs.clone()), Value::Slot(None)] };
            Value::pure_function(None, body)
        }
        [Value::List(funcs), x] => {
            let xv = ev.eval(x.clone());
            let mut out: Vec<Value> = Vec::with_capacity(funcs.len());
            for f in funcs { out.push(ev.eval(Value::Expr { head: Box::new(f.clone()), args: vec![xv.clone()] })); }
            Value::List(out)
        }
        _ => uneval("Through", args),
    }
}

// Identity[] => PureFunction[#&]; Identity[x] => x
fn identity_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [] => Value::pure_function(None, Value::Slot(None)),
        [x] => ev.eval(x.clone()),
        _ => uneval("Identity", args),
    }
}

// ConstantFunction[c] => PureFunction[ c ] (ignores inputs)
fn constant_function_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len()!=1 { return uneval("ConstantFunction", args); }
    let c = ev.eval(args[0].clone());
    Value::pure_function(None, c)
}
