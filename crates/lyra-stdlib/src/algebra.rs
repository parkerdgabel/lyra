use crate::register_if;
use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;

pub fn register_algebra(ev: &mut Evaluator) {
    ev.register("Simplify", simplify as NativeFn, Attributes::empty());
    ev.register("Expand", expand as NativeFn, Attributes::empty());
    ev.register("ExpandAll", expand_all as NativeFn, Attributes::empty());
    ev.register("CollectTerms", collect as NativeFn, Attributes::empty());
    ev.register("CollectTermsBy", collect_by as NativeFn, Attributes::empty());
    ev.register("Factor", factor as NativeFn, Attributes::empty());
    ev.register("D", diff as NativeFn, Attributes::empty());
    // Named CancelRational to avoid conflict with concurrency Cancel
    ev.register("CancelRational", cancel as NativeFn, Attributes::empty());
    ev.register("Apart", apart as NativeFn, Attributes::empty());
    ev.register("Solve", solve as NativeFn, Attributes::empty());
    ev.register("Roots", roots as NativeFn, Attributes::empty());
}

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

// ----- Simplify -----

fn simplify(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::expr(Value::symbol("Simplify"), args);
    }
    simp(args[0].clone())
}

fn simp(v: Value) -> Value {
    match v {
        Value::Expr { head, args } => match *head {
            Value::Symbol(ref s) if s == "Plus" => {
                let mut flat: Vec<Value> = Vec::new();
                let mut acc_num: Option<Value> = None;
                for a in args.into_iter().map(simp) {
                    if let Value::Expr { head: h2, args: a2 } = &a {
                        if let Value::Symbol(ref t) = **h2 {
                            if t == "Plus" {
                                flat.extend(a2.clone());
                                continue;
                            }
                        }
                    }
                    if let Some(n) = try_numeric(&a) {
                        acc_num = match (acc_num, n) {
                            (None, x) => Some(x),
                            (Some(x), y) => add_num(x, y),
                        };
                    } else {
                        flat.push(a);
                    }
                }
                if let Some(n) = acc_num {
                    if !is_zero(&n) {
                        flat.insert(0, n);
                    }
                }
                match flat.len() {
                    0 => Value::Integer(0),
                    1 => flat.pop().unwrap(),
                    _ => Value::expr(Value::symbol("Plus"), flat),
                }
            }
            Value::Symbol(ref s) if s == "Times" => {
                let mut flat: Vec<Value> = Vec::new();
                let mut acc_num: Option<Value> = None;
                for a in args.into_iter().map(simp) {
                    if let Value::Expr { head: h2, args: a2 } = &a {
                        if let Value::Symbol(ref t) = **h2 {
                            if t == "Times" {
                                flat.extend(a2.clone());
                                continue;
                            }
                        }
                    }
                    if let Some(n) = try_numeric(&a) {
                        acc_num = match (acc_num, n) {
                            (None, x) => Some(x),
                            (Some(x), y) => mul_num(x, y),
                        };
                    } else {
                        flat.push(a);
                    }
                }
                if let Some(n) = acc_num {
                    if is_zero(&n) {
                        return Value::Integer(0);
                    }
                    if !is_one(&n) {
                        flat.insert(0, n);
                    }
                }
                match flat.len() {
                    0 => Value::Integer(1),
                    1 => flat.pop().unwrap(),
                    _ => Value::expr(Value::symbol("Times"), flat),
                }
            }
            Value::Symbol(ref s) if s == "Power" => match args.as_slice() {
                [a, b] => {
                    let a = simp(a.clone());
                    let b = simp(b.clone());
                    match (&a, &b) {
                        (_, Value::Integer(0)) => Value::Integer(1),
                        (x, Value::Integer(1)) => x.clone(),
                        (Value::Integer(0), _) => Value::Integer(0),
                        (Value::Integer(1), _) => Value::Integer(1),
                        _ => Value::expr(Value::symbol("Power"), vec![a, b]),
                    }
                }
                _ => Value::expr(Value::symbol("Power"), args.into_iter().map(simp).collect()),
            },
            _ => Value::expr(*head, args.into_iter().map(simp).collect()),
        },
        other => other,
    }
}

// ----- Expand -----

fn expand(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::expr(Value::symbol("Expand"), args);
    }
    expd(args[0].clone())
}

fn expd(v: Value) -> Value {
    match v {
        Value::Expr { head, args } => match *head {
            Value::Symbol(ref s) if s == "Times" => {
                // distributive expansion Times over Plus
                let mut terms: Vec<Value> = vec![Value::Integer(1)];
                for a in args.into_iter().map(expd) {
                    let mut new_terms = Vec::new();
                    match a {
                        Value::Expr { head: h2, args: a2 } if matches!(*h2, Value::Symbol(ref t) if t=="Plus") => {
                            for t in terms.iter() {
                                for addend in &a2 {
                                    new_terms.push(simp(Value::expr(
                                        Value::symbol("Times"),
                                        vec![t.clone(), addend.clone()],
                                    )));
                                }
                            }
                        }
                        other => {
                            for t in terms.iter() {
                                new_terms.push(simp(Value::expr(
                                    Value::symbol("Times"),
                                    vec![t.clone(), other.clone()],
                                )));
                            }
                        }
                    }
                    terms = new_terms;
                }
                if terms.len() == 1 {
                    terms.pop().unwrap()
                } else {
                    Value::expr(Value::symbol("Plus"), terms)
                }
            }
            Value::Symbol(ref s) if s == "Power" => {
                match args.as_slice() {
                    [base, Value::Integer(n)] if *n >= 0 && *n <= 8 => {
                        // simple repeated multiplication then expand
                        let mut res = Value::Integer(1);
                        for _ in 0..*n {
                            res =
                                expd(Value::expr(Value::symbol("Times"), vec![res, base.clone()]));
                        }
                        res
                    }
                    _ => Value::expr(*head, args.into_iter().map(expd).collect()),
                }
            }
            Value::Symbol(ref s) if s == "Plus" => {
                Value::expr(Value::symbol("Plus"), args.into_iter().map(expd).collect())
            }
            _ => Value::expr(*head, args.into_iter().map(expd).collect()),
        },
        other => other,
    }
}

// ----- ExpandAll -----

fn expand_all(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::expr(Value::symbol("ExpandAll"), args);
    }
    expd_recursive(args[0].clone())
}

fn expd_recursive(v: Value) -> Value {
    match v {
        Value::Expr { head, args } => {
            let ex_head = *head;
            let ex_args: Vec<Value> = args.into_iter().map(expd_recursive).collect();
            expd(Value::expr(ex_head, ex_args))
        }
        other => other,
    }
}

// ----- Collect (single variable) -----

fn collect(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [expr] => collect_in(expr.clone(), None),
        [expr, var] => collect_in(expr.clone(), Some(var.clone())),
        _ => Value::expr(Value::symbol("Collect"), args),
    }
}

fn collect_in(expr: Value, var: Option<Value>) -> Value {
    let v = expd(simp(expr));
    let var_sym = var.and_then(|v| as_symbol_name(&v));
    if let Some(poly) = Poly::from_expr(&v, var_sym.as_deref()) {
        poly.to_expr()
    } else {
        v
    }
}

// ----- CollectTermsBy: multi-variable collection over provided vars -----

fn collect_by(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // CollectTermsBy[expr] or CollectTermsBy[expr, {vars..}] or CollectTermsBy[expr, {vars..}, <|OrderBy->"lex"|"gradedLex"|>]
    if args.is_empty() {
        return Value::expr(Value::symbol("CollectTermsBy"), args);
    }
    let expr = args[0].clone();
    let mut vars: Option<Vec<String>> = None;
    let mut order_by: String = "lex".into();
    if args.len() >= 2 {
        if let Value::List(vs) = &args[1] {
            vars = Some(vs.iter().filter_map(|v| as_symbol_name(v)).collect());
        }
    }
    if args.len() >= 3 {
        if let Value::Assoc(m) = &args[2] {
            if let Some(Value::String(s)) | Some(Value::Symbol(s)) = m.get("OrderBy") {
                order_by = s.clone();
            }
        }
    }
    let expr_n = expd(simp(expr));
    let var_list = match vars {
        Some(v) if !v.is_empty() => v,
        _ => {
            // auto-detect symbols in expr
            let mut set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
            collect_symbols(&expr_n, &mut set);
            set.into_iter().collect()
        }
    };
    collect_mono_with_order(expr_n, &var_list, &order_by)
}

fn collect_mono_with_order(expr: Value, vars: &[String], order_by: &str) -> Value {
    use std::collections::BTreeMap;
    let mut map: BTreeMap<Vec<usize>, i64> = BTreeMap::new();
    fn add_term(
        map: &mut std::collections::BTreeMap<Vec<usize>, i64>,
        coeff: i64,
        deg: Vec<usize>,
    ) {
        if coeff == 0 {
            return;
        }
        let entry = map.entry(deg).or_insert(0);
        *entry = entry.saturating_add(coeff);
    }
    fn accum(
        vars: &[String],
        v: &Value,
        map: &mut std::collections::BTreeMap<Vec<usize>, i64>,
    ) -> bool {
        match v {
            Value::Integer(n) => {
                add_term(map, *n, vec![0; vars.len()]);
                true
            }
            Value::Expr { head, args } if matches!(&**head, Value::Symbol(s) if s=="Plus") => {
                for a in args {
                    if !accum(vars, a, map) {
                        return false;
                    }
                }
                true
            }
            _ => {
                // parse product of integer and variable powers
                let mut coeff: i64 = 1;
                let mut deg = vec![0usize; vars.len()];
                let mut rest: Vec<Value> = Vec::new();
                let factors: Vec<Value> = match v {
                    Value::Expr { head, args } if matches!(&**head, Value::Symbol(s) if s=="Times") => {
                        args.clone()
                    }
                    other => vec![other.clone()],
                };
                for f in factors {
                    match f {
                        Value::Integer(n) => {
                            coeff = coeff.saturating_mul(n);
                        }
                        Value::Symbol(s) => {
                            if let Some(idx) = vars.iter().position(|x| x == &s) {
                                deg[idx] = deg[idx].saturating_add(1);
                            } else {
                                rest.push(Value::Symbol(s));
                            }
                        }
                        Value::Expr { head, args } => {
                            if let Value::Symbol(h) = *head.clone() {
                                if h == "Power" {
                                    if let [base, exp] = args.as_slice() {
                                        if let Value::Symbol(s) = base {
                                            if let Some(idx) = vars.iter().position(|x| x == s) {
                                                if let Value::Integer(n) = exp {
                                                    if *n >= 0 {
                                                        deg[idx] =
                                                            deg[idx].saturating_add(*n as usize);
                                                        continue;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            rest.push(Value::Expr { head, args });
                        }
                        _ => {
                            rest.push(f);
                        }
                    }
                }
                if rest.is_empty() {
                    add_term(map, coeff, deg);
                    true
                } else {
                    false
                }
            }
        }
    }
    if !accum(vars, &expr, &mut map) {
        return expr;
    }
    // choose order: lex (by degree vector), or gradedLex (by total degree then lex)
    let mut items: Vec<(Vec<usize>, i64)> = map.into_iter().collect();
    if order_by == "gradedLex" {
        items.sort_by(|(a, _), (b, _)| {
            let sa: usize = a.iter().sum();
            let sb: usize = b.iter().sum();
            sa.cmp(&sb).then_with(|| a.cmp(b))
        });
    }
    let mut terms: Vec<Value> = Vec::new();
    for (deg, c) in items.into_iter() {
        if c == 0 {
            continue;
        }
        let mut factors: Vec<Value> = vec![Value::Integer(c)];
        for (i, d) in deg.into_iter().enumerate() {
            if d == 0 {
                continue;
            }
            if d == 1 {
                factors.push(Value::symbol(vars[i].clone()));
            } else {
                factors.push(Value::expr(
                    Value::symbol("Power"),
                    vec![Value::symbol(vars[i].clone()), Value::Integer(d as i64)],
                ));
            }
        }
        terms.push(mul_list(factors));
    }
    if terms.is_empty() {
        Value::Integer(0)
    } else if terms.len() == 1 {
        terms.pop().unwrap()
    } else {
        Value::expr(Value::symbol("Plus"), terms)
    }
}

fn collect_symbols(v: &Value, out: &mut std::collections::BTreeSet<String>) {
    match v {
        Value::Symbol(s) => {
            out.insert(s.clone());
        }
        Value::Expr { head, args } => {
            collect_symbols(head, out);
            for a in args {
                collect_symbols(a, out);
            }
        }
        _ => {}
    }
}

// ----- Cancel (basic) -----

fn cancel(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Cancel[(num)/(den), x?]
    if args.is_empty() {
        return Value::expr(Value::symbol("Cancel"), args);
    }
    let (num, den) = match &args[0] {
        Value::Expr { head, args: ab }
            if matches!(**head, Value::Symbol(ref s) if s=="Divide") && ab.len() == 2 =>
        {
            (ab[0].clone(), ab[1].clone())
        }
        other => return other.clone(),
    };
    let var_sym = args.get(1).and_then(|v| as_symbol_name(v));
    let nf = factor_list(num, var_sym.as_deref());
    let df = factor_list(den, var_sym.as_deref());
    let (mut nf2, mut df2) = (Vec::new(), Vec::new());
    let mut used = vec![false; df.len()];
    'outer: for n in nf.iter() {
        for (j, d) in df.iter().enumerate() {
            if !used[j] && n == d {
                used[j] = true;
                continue 'outer;
            }
        }
        nf2.push(n.clone());
    }
    for (j, d) in df.iter().enumerate() {
        if !used[j] {
            df2.push(d.clone());
        }
    }
    let num_r = mul_list(nf2);
    let den_r = mul_list(df2);
    if is_one(&den_r) {
        simp(num_r)
    } else {
        Value::expr(Value::symbol("Divide"), vec![simp(num_r), simp(den_r)])
    }
}

fn factor_list(v: Value, var: Option<&str>) -> Vec<Value> {
    // Use our basic Poly factoring (content and quadratic) and flatten Times
    let fv = match var {
        Some(_) => v.clone(),
        None => v.clone(),
    };
    let mut out: Vec<Value> = Vec::new();
    let fv = match var {
        Some(x) => factor_basic_with_var(fv, Some(x)),
        None => factor_basic(fv),
    };
    flatten_times(&fv, &mut out);
    out
}

fn flatten_times(v: &Value, out: &mut Vec<Value>) {
    match v {
        Value::Expr { head, args } if matches!(**head, Value::Symbol(ref s) if s=="Times") => {
            for a in args {
                flatten_times(a, out);
            }
        }
        other => out.push(other.clone()),
    }
}

fn factor_basic_with_var(v: Value, var: Option<&str>) -> Value {
    let v = to_add_mul(v);
    if let Some(mut p) = Poly::from_expr(&v, var) {
        p.factor_content();
        if let Some((p1, p2)) = p.factor_quadratic() {
            return mul_list(vec![p1.to_expr(), p2.to_expr()]);
        }
        return p.to_expr();
    }
    v
}

// ----- Apart (very limited) -----

fn apart(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Apart[1/((a1 x + b1)(a2 x + b2)), x] -> A/(a1 x + b1) + B/(a2 x + b2) for constant numerator
    if args.len() < 1 {
        return Value::expr(Value::symbol("Apart"), args);
    }
    let var = args.get(1).and_then(|v| as_symbol_name(v)).unwrap_or_else(|| "x".into());
    let (num, den) = match &args[0] {
        Value::Expr { head, args: ab }
            if matches!(**head, Value::Symbol(ref s) if s=="Divide") && ab.len() == 2 =>
        {
            (ab[0].clone(), ab[1].clone())
        }
        other => return other.clone(),
    };
    let cnum = match num {
        Value::Integer(n) => n,
        _ => return Value::expr(Value::symbol("Apart"), args),
    };
    let mut den_factors = Vec::new();
    flatten_times(&factor_basic_with_var(den, Some(&var)), &mut den_factors);
    if den_factors.len() != 2 {
        return Value::expr(Value::symbol("Apart"), args);
    }
    let (a1, b1) = match linear_coeffs(&den_factors[0], &var) {
        Some(t) => t,
        None => return Value::expr(Value::symbol("Apart"), args),
    };
    let (a2, b2) = match linear_coeffs(&den_factors[1], &var) {
        Some(t) => t,
        None => return Value::expr(Value::symbol("Apart"), args),
    };
    // Solve cnum = A*(a2 x + b2) + B*(a1 x + b1) -> match x and constant terms
    // Coef of x: A*a2 + B*a1 = 0; Constant: A*b2 + B*b1 = cnum
    let det = a1 * b2 - a2 * b1;
    if det == 0 {
        return Value::expr(Value::symbol("Apart"), args);
    }
    let a = Value::Integer((cnum * a1) / det);
    let b = Value::Integer((cnum * (-a2)) / det);
    let t1 = Value::expr(Value::symbol("Divide"), vec![a, den_factors[0].clone()]);
    let t2 = Value::expr(Value::symbol("Divide"), vec![b, den_factors[1].clone()]);
    simp(Value::expr(Value::symbol("Plus"), vec![t1, t2]))
}

fn linear_coeffs(v: &Value, var: &str) -> Option<(i64, i64)> {
    // Parse a1 x + b1 in normalized forms
    match v {
        Value::Symbol(s) if s == var => Some((1, 0)),
        Value::Expr { head, args } if matches!(**head, Value::Symbol(ref s) if s=="Plus") => {
            if args.len() != 2 {
                return None;
            }
            match (&args[0], &args[1]) {
                (Value::Expr { head: h2, args: a2 }, b)
                | (b, Value::Expr { head: h2, args: a2 })
                    if matches!(**h2, Value::Symbol(ref t) if t=="Times") =>
                {
                    let (coef, ok) = match a2.as_slice() {
                        [Value::Integer(n), Value::Symbol(s)] if s == var => (*n, true),
                        [Value::Symbol(s), Value::Integer(n)] if s == var => (*n, true),
                        _ => (0, false),
                    };
                    if !ok {
                        return None;
                    }
                    let bconst = match b {
                        Value::Integer(n) => *n,
                        _ => return None,
                    };
                    Some((coef, bconst))
                }
                (Value::Symbol(s), Value::Integer(b)) if s == var => Some((1, *b)),
                (Value::Integer(b), Value::Symbol(s)) if s == var => Some((1, *b)),
                _ => None,
            }
        }
        Value::Expr { head, args } if matches!(**head, Value::Symbol(ref s) if s=="Times") => {
            match args.as_slice() {
                [Value::Integer(n), Value::Symbol(s)] if s == var => Some((*n, 0)),
                [Value::Symbol(s), Value::Integer(n)] if s == var => Some((*n, 0)),
                _ => None,
            }
        }
        _ => None,
    }
}

// ----- Solve (linear systems) -----

fn solve(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Solve[{eqns..}, {vars..}] -> list of Rule[var, value]
    if args.len() != 2 {
        return Value::expr(Value::symbol("Solve"), args);
    }
    let (eqs, vars) = match (&args[0], &args[1]) {
        (Value::List(es), Value::List(vs)) => (es.clone(), vs.clone()),
        _ => return Value::expr(Value::symbol("Solve"), args),
    };
    let var_names: Vec<String> = vars.iter().filter_map(|v| as_symbol_name(v)).collect();
    if var_names.len() != vars.len() {
        return Value::expr(Value::symbol("Solve"), args);
    }
    let n = var_names.len();
    let m = eqs.len();
    if m == 0 || n == 0 {
        return Value::List(vec![]);
    }
    // Build Ax = b using f64 arithmetic
    let mut a = vec![vec![0.0f64; n]; m];
    let mut b = vec![0.0f64; m];
    for (i, e) in eqs.iter().enumerate() {
        if let Value::Expr { head, args } = e {
            if matches!(**head, Value::Symbol(ref s) if s=="Equal") && args.len() == 2 {
                let lhs = args[0].clone();
                let rhs = args[1].clone();
                // Move all to LHS: lhs - rhs = 0
                let diff = Value::expr(Value::symbol("Minus"), vec![lhs, rhs]);
                let expd = to_add_mul(expd(simp(diff)));
                // Collect per var
                let mut coeffs = vec![0.0f64; n];
                let mut const_term = 0.0f64;
                if decompose_linear(&expd, &var_names, &mut coeffs, &mut const_term).is_none() {
                    return Value::expr(Value::symbol("Solve"), args.to_vec());
                }
                for j in 0..n {
                    a[i][j] = coeffs[j];
                }
                b[i] = -const_term;
                continue;
            }
        }
        return Value::expr(Value::symbol("Solve"), args);
    }
    if let Some(xs) = gaussian_elimination(a, b) {
        let mut rules: Vec<Value> = Vec::with_capacity(n);
        for (j, name) in var_names.iter().enumerate() {
            rules.push(Value::expr(
                Value::symbol("Rule"),
                vec![Value::symbol(name.clone()), Value::Real(xs[j])],
            ))
        }
        Value::List(rules)
    } else {
        Value::expr(Value::symbol("Solve"), args)
    }
}

fn decompose_linear(v: &Value, vars: &[String], out: &mut [f64], c: &mut f64) -> Option<()> {
    match v {
        Value::Integer(n) => {
            *c += *n as f64;
            Some(())
        }
        Value::Real(x) => {
            *c += *x;
            Some(())
        }
        Value::Expr { head, args } if matches!(**head, Value::Symbol(ref s) if s=="Minus") => {
            // a - b => a + (-1)*b
            match args.as_slice() {
                [a, b] => {
                    decompose_linear(a, vars, out, c)?;
                    decompose_linear(
                        &Value::expr(Value::symbol("Times"), vec![Value::Integer(-1), b.clone()]),
                        vars,
                        out,
                        c,
                    )
                }
                [a] => decompose_linear(
                    &Value::expr(Value::symbol("Times"), vec![Value::Integer(-1), a.clone()]),
                    vars,
                    out,
                    c,
                ),
                _ => None,
            }
        }
        Value::Expr { head, args } if matches!(**head, Value::Symbol(ref s) if s=="Plus") => {
            for a in args {
                decompose_linear(a, vars, out, c)?;
            }
            Some(())
        }
        Value::Expr { head, args } if matches!(**head, Value::Symbol(ref s) if s=="Times") => {
            // Expect numeric * var
            match args.as_slice() {
                [Value::Integer(n), Value::Symbol(s)] => {
                    if let Some(idx) = vars.iter().position(|x| x == s) {
                        out[idx] += *n as f64;
                        return Some(());
                    }
                }
                [Value::Real(x), Value::Symbol(s)] => {
                    if let Some(idx) = vars.iter().position(|x| x == s) {
                        out[idx] += *x;
                        return Some(());
                    }
                }
                [Value::Symbol(s), Value::Integer(n)] => {
                    if let Some(idx) = vars.iter().position(|x| x == s) {
                        out[idx] += *n as f64;
                        return Some(());
                    }
                }
                [Value::Symbol(s), Value::Real(x)] => {
                    if let Some(idx) = vars.iter().position(|x| x == s) {
                        out[idx] += *x;
                        return Some(());
                    }
                }
                _ => {}
            }
            None
        }
        Value::Symbol(s) => {
            if let Some(idx) = vars.iter().position(|x| x == s) {
                out[idx] += 1.0;
                Some(())
            } else {
                None
            }
        }
        _ => None,
    }
}

fn to_add_mul(v: Value) -> Value {
    match v.clone() {
        Value::Expr { head, args } => match *head {
            Value::Symbol(ref s) if s == "Minus" => match args.as_slice() {
                [a, b] => Value::expr(
                    Value::symbol("Plus"),
                    vec![
                        to_add_mul(a.clone()),
                        Value::expr(
                            Value::symbol("Times"),
                            vec![Value::Integer(-1), to_add_mul(b.clone())],
                        ),
                    ],
                ),
                [a] => Value::expr(
                    Value::symbol("Times"),
                    vec![Value::Integer(-1), to_add_mul(a.clone())],
                ),
                _ => Value::expr(*head, args.into_iter().map(to_add_mul).collect()),
            },
            _ => Value::expr(*head, args.into_iter().map(to_add_mul).collect()),
        },
        other => other,
    }
}

fn gaussian_elimination(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = a[0].len();
    let m = a.len();
    if m < n {
        return None;
    }
    let mut col = 0usize;
    for row in 0..n {
        // find pivot
        let mut pivot = row;
        while pivot < m && a[pivot][col].abs() < 1e-12 {
            pivot += 1;
        }
        if pivot == m {
            return None;
        }
        a.swap(row, pivot);
        b.swap(row, pivot);
        // normalize
        let div = a[row][col];
        if div.abs() < 1e-12 {
            return None;
        }
        for j in col..n {
            a[row][j] /= div;
        }
        b[row] /= div;
        // eliminate others
        for i in 0..m {
            if i != row {
                let f = a[i][col];
                if f.abs() > 1e-12 {
                    for j in col..n {
                        a[i][j] -= f * a[row][j];
                    }
                    b[i] -= f * b[row];
                }
            }
        }
        if col + 1 < n {
            col += 1;
        } else {
            break;
        }
    }
    let mut x = vec![0.0f64; n];
    for i in 0..n {
        x[i] = b[i];
    }
    Some(x)
}

// ----- Roots (degree <= 2) -----

fn roots(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Roots[poly, x]
    if args.len() != 2 {
        return Value::expr(Value::symbol("Roots"), args);
    }
    let var = match as_symbol_name(&args[1]) {
        Some(s) => s,
        None => return Value::expr(Value::symbol("Roots"), args),
    };
    let v = to_add_mul(expd(simp(args[0].clone())));
    let Some(p) = Poly::from_expr(&v, Some(&var)) else {
        return Value::expr(Value::symbol("Roots"), args);
    };
    let deg = p.coeffs.len().saturating_sub(1);
    match deg {
        0 => Value::List(vec![]),
        1 => {
            let b = *p.coeffs.get(1).unwrap_or(&0) as f64;
            let c = *p.coeffs.get(0).unwrap_or(&0) as f64;
            if b.abs() < 1e-12 {
                return Value::List(vec![]);
            }
            Value::List(vec![Value::Real(-c / b)])
        }
        2 => {
            let a = *p.coeffs.get(2).unwrap_or(&0) as f64;
            let b = *p.coeffs.get(1).unwrap_or(&0) as f64;
            let c = *p.coeffs.get(0).unwrap_or(&0) as f64;
            let disc = b * b - 4.0 * a * c;
            if disc >= 0.0 {
                let s = disc.sqrt();
                Value::List(vec![
                    Value::Real((-b - s) / (2.0 * a)),
                    Value::Real((-b + s) / (2.0 * a)),
                ])
            } else {
                // complex roots
                let s = (-disc).sqrt();
                let re = -b / (2.0 * a);
                let im = s / (2.0 * a);
                Value::List(vec![
                    Value::Complex {
                        re: Box::new(Value::Real(re)),
                        im: Box::new(Value::Real(-im)),
                    },
                    Value::Complex { re: Box::new(Value::Real(re)), im: Box::new(Value::Real(im)) },
                ])
            }
        }
        _ => Value::expr(Value::symbol("Roots"), args),
    }
}

// ----- Factor (basic) -----

fn factor(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [expr] => factor_basic(expd(simp(expr.clone()))),
        [expr, var] => {
            let v = expd(simp(expr.clone()));
            let var_sym = as_symbol_name(var);
            if let Some(mut p) = Poly::from_expr(&v, var_sym.as_deref()) {
                p.factor_content();
                if let Some((p1, p2)) = p.factor_quadratic() {
                    return mul_list(vec![p1.to_expr(), p2.to_expr()]);
                }
                return p.to_expr();
            }
            v
        }
        _ => Value::expr(Value::symbol("Factor"), args),
    }
}

fn factor_basic(v: Value) -> Value {
    if let Some(mut p) = Poly::from_expr(&v, None) {
        p.factor_content();
        if let Some((p1, p2)) = p.factor_quadratic() {
            return mul_list(vec![p1.to_expr(), p2.to_expr()]);
        }
        p.to_expr()
    } else {
        v
    }
}

// ----- Differentiation (D) -----

fn diff(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // D[expr, x]
    if args.len() != 2 {
        return Value::expr(Value::symbol("D"), args);
    }
    let expr = args[0].clone();
    let var = args[1].clone();
    let var_name = match as_symbol_name(&var) {
        Some(s) => s,
        None => return Value::expr(Value::symbol("D"), vec![expr, var]),
    };
    d(expr, &var_name)
}

fn d(expr: Value, var: &str) -> Value {
    match expr {
        Value::Integer(_) | Value::Real(_) | Value::Rational { .. } | Value::BigReal(_) => {
            Value::Integer(0)
        }
        Value::Symbol(ref s) if s == var => Value::Integer(1),
        Value::Symbol(_) => Value::Integer(0),
        Value::Expr { head, args } => match *head {
            Value::Symbol(ref s) if s == "Plus" => {
                Value::expr(Value::symbol("Plus"), args.into_iter().map(|a| d(a, var)).collect())
            }
            Value::Symbol(ref s) if s == "Times" => {
                // product rule for n-ary: sum_i (a_i' * prod_{j!=i} a_j)
                let n = args.len();
                let mut terms: Vec<Value> = Vec::with_capacity(n);
                for i in 0..n {
                    let mut prod: Vec<Value> = Vec::with_capacity(n);
                    for (j, a) in args.iter().enumerate() {
                        if i == j {
                            prod.push(d(a.clone(), var));
                        } else {
                            prod.push(a.clone());
                        }
                    }
                    terms.push(mul_list(prod));
                }
                Value::expr(Value::symbol("Plus"), terms)
            }
            Value::Symbol(ref s) if s == "Power" => {
                match args.as_slice() {
                    [base, exp] => match exp {
                        Value::Integer(n) if *n >= 0 => {
                            // n * base^(n-1) * base'
                            let nval = Value::Integer(*n);
                            let n1 = Value::Integer(*n - 1);
                            mul_list(vec![
                                nval,
                                Value::expr(Value::symbol("Power"), vec![base.clone(), n1]),
                                d(base.clone(), var),
                            ])
                        }
                        _ => Value::expr(
                            Value::symbol("Times"),
                            vec![
                                expr_of("Log", vec![base.clone()]),
                                d(base.clone(), var),
                                Value::expr(
                                    Value::symbol("Power"),
                                    vec![base.clone(), exp.clone()],
                                ),
                            ],
                        ),
                    },
                    _ => Value::expr(
                        Value::symbol("D"),
                        vec![Value::expr(*head, args), Value::symbol(var.to_string())],
                    ),
                }
            }
            _ => Value::expr(
                Value::symbol("D"),
                vec![Value::expr(*head, args), Value::symbol(var.to_string())],
            ),
        },
        other => Value::expr(Value::symbol("D"), vec![other, Value::symbol(var.to_string())]),
    }
}

// ----- Helpers -----

fn expr_of(name: &str, args: Vec<Value>) -> Value {
    Value::expr(Value::symbol(name.to_string()), args)
}
fn mul_list(mut args: Vec<Value>) -> Value {
    // drop neutral 1s
    args.retain(|a| !is_one(a));
    if args.is_empty() {
        Value::Integer(1)
    } else if args.len() == 1 {
        args.pop().unwrap()
    } else {
        Value::expr(Value::symbol("Times"), args)
    }
}

fn as_symbol_name(v: &Value) -> Option<String> {
    if let Value::Symbol(s) = v {
        Some(s.clone())
    } else {
        None
    }
}

fn is_zero(v: &Value) -> bool {
    match v {
        Value::Integer(0) => true,
        Value::Real(x) if *x == 0.0 => true,
        Value::Rational { num, .. } if *num == 0 => true,
        _ => false,
    }
}
fn is_one(v: &Value) -> bool {
    match v {
        Value::Integer(1) => true,
        Value::Real(x) if *x == 1.0 => true,
        Value::Rational { num, den } if *num == *den => true,
        _ => false,
    }
}

fn try_numeric(v: &Value) -> Option<Value> {
    match v {
        Value::Integer(_) | Value::Real(_) | Value::Rational { .. } | Value::BigReal(_) => {
            Some(v.clone())
        }
        _ => None,
    }
}

fn add_num(a: Value, b: Value) -> Option<Value> {
    match (a, b) {
        (Value::Integer(x), Value::Integer(y)) => Some(Value::Integer(x + y)),
        (Value::Real(x), Value::Real(y)) => Some(Value::Real(x + y)),
        (Value::Integer(x), Value::Real(y)) => Some(Value::Real((x as f64) + y)),
        (Value::Real(x), Value::Integer(y)) => Some(Value::Real(x + (y as f64))),
        (Value::Rational { num: n1, den: d1 }, Value::Rational { num: n2, den: d2 }) => {
            Some(Value::Rational { num: n1 * d2 + n2 * d1, den: d1 * d2 })
        }
        (Value::Integer(x), Value::Rational { num, den })
        | (Value::Rational { num, den }, Value::Integer(x)) => {
            Some(Value::Rational { num: num + x * den, den })
        }
        (Value::Real(x), Value::Rational { num, den })
        | (Value::Rational { num, den }, Value::Real(x)) => {
            Some(Value::Real(x + (num as f64) / (den as f64)))
        }
        (a, b) => Some(Value::expr(Value::symbol("Plus"), vec![a, b])),
    }
}

fn mul_num(a: Value, b: Value) -> Option<Value> {
    match (a, b) {
        (Value::Integer(x), Value::Integer(y)) => Some(Value::Integer(x * y)),
        (Value::Real(x), Value::Real(y)) => Some(Value::Real(x * y)),
        (Value::Integer(x), Value::Real(y)) => Some(Value::Real((x as f64) * y)),
        (Value::Real(x), Value::Integer(y)) => Some(Value::Real(x * (y as f64))),
        (Value::Rational { num: n1, den: d1 }, Value::Rational { num: n2, den: d2 }) => {
            Some(Value::Rational { num: n1 * n2, den: d1 * d2 })
        }
        (Value::Integer(x), Value::Rational { num, den })
        | (Value::Rational { num, den }, Value::Integer(x)) => {
            Some(Value::Rational { num: num * x, den })
        }
        (Value::Real(x), Value::Rational { num, den })
        | (Value::Rational { num, den }, Value::Real(x)) => {
            Some(Value::Real(x * (num as f64) / (den as f64)))
        }
        (a, b) => Some(Value::expr(Value::symbol("Times"), vec![a, b])),
    }
}

// Simple single-variable polynomial representation with integer coefficients
#[derive(Clone, Debug, PartialEq, Eq)]
struct Poly {
    // sum_{k} c_k x^k
    var: String,
    coeffs: Vec<i64>, // lowest degree first
    content: i64,     // extracted gcd of integer coefficients (sign preserved in content)
}

impl Poly {
    fn from_expr(expr: &Value, var: Option<&str>) -> Option<Poly> {
        // discover var if not provided
        let vname = match var {
            Some(s) => s.to_string(),
            None => first_symbol(expr)?,
        };
        let mut map: std::collections::BTreeMap<usize, i64> = std::collections::BTreeMap::new();
        if !Self::accumulate(expr, &vname, 1, &mut map) {
            return None;
        }
        let max_deg = map.keys().copied().max().unwrap_or(0);
        let mut coeffs = vec![0i64; max_deg + 1];
        for (k, c) in map {
            coeffs[k] = c;
        }
        let mut p = Poly { var: vname, coeffs, content: 1 };
        p.factor_content();
        Some(p)
    }

    fn accumulate(
        e: &Value,
        var: &str,
        sign: i64,
        map: &mut std::collections::BTreeMap<usize, i64>,
    ) -> bool {
        match e {
            Value::Integer(n) => {
                *map.entry(0).or_default() += sign * *n;
                true
            }
            Value::Rational { .. } | Value::Real(_) | Value::BigReal(_) => false,
            Value::Symbol(s) => {
                if s == var {
                    *map.entry(1).or_default() += sign;
                    true
                } else {
                    false
                }
            }
            Value::Expr { head, args } => match &**head {
                Value::Symbol(h) if h == "Plus" => {
                    for a in args {
                        if !Self::accumulate(a, var, sign, map) {
                            return false;
                        }
                    }
                    true
                }
                Value::Symbol(h) if h == "Times" => {
                    // require a numeric integer coefficient times a pure power of var
                    let mut coeff = 1i64 * sign;
                    let mut deg = 0usize;
                    for a in args {
                        match a {
                            Value::Integer(n) => {
                                coeff = coeff.saturating_mul(*n);
                            }
                            Value::Symbol(s) if s == var => {
                                deg = deg.saturating_add(1);
                            }
                            Value::Expr { head: h2, args: a2 } => {
                                if let Value::Symbol(pw) = &**h2 {
                                    if pw == "Power" {
                                        if let [base, exp] = a2.as_slice() {
                                            if let Value::Symbol(s) = base {
                                                if s == var {
                                                    if let Value::Integer(n) = exp {
                                                        if *n >= 0 {
                                                            deg = deg.saturating_add(*n as usize);
                                                            continue;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        return false;
                                    }
                                }
                                return false;
                            }
                            _ => return false,
                        }
                    }
                    *map.entry(deg).or_default() += coeff;
                    true
                }
                Value::Symbol(h) if h == "Power" => {
                    if let [base, exp] = args.as_slice() {
                        if let Value::Symbol(s) = base {
                            if s == var {
                                if let Value::Integer(n) = exp {
                                    if *n >= 0 {
                                        *map.entry(*n as usize).or_default() += sign;
                                        return true;
                                    }
                                }
                            }
                        }
                    }
                    false
                }
                _ => false,
            },
            _ => false,
        }
    }

    fn to_expr(&self) -> Value {
        let mut terms: Vec<Value> = Vec::new();
        for (k, c) in self.coeffs.iter().enumerate() {
            if *c == 0 {
                continue;
            }
            let mut term = vec![Value::Integer(*c * self.content)];
            if k > 0 {
                if k == 1 {
                    term.push(Value::symbol(self.var.clone()));
                } else {
                    term.push(Value::expr(
                        Value::symbol("Power"),
                        vec![Value::symbol(self.var.clone()), Value::Integer(k as i64)],
                    ));
                }
            }
            terms.push(mul_list(term));
        }
        if terms.is_empty() {
            Value::Integer(0)
        } else if terms.len() == 1 {
            terms[0].clone()
        } else {
            Value::expr(Value::symbol("Plus"), terms)
        }
    }

    fn factor_content(&mut self) {
        // extract gcd of coefficients and normalize sign to content
        let mut g: i64 = 0;
        for c in &self.coeffs {
            g = gcd(g, *c);
        }
        if g == 0 {
            self.content = 1;
            return;
        }
        self.content = g;
        for c in &mut self.coeffs {
            *c /= g;
        }
        // keep content sign positive by moving sign to content
        if self.coeffs.iter().all(|&c| c == 0) {
            self.content = 1;
        }
    }

    fn factor_quadratic(&self) -> Option<(Poly, Poly)> {
        // ax^2 + bx + c with a!=0; factor over integers if discriminant square
        if self.coeffs.len() < 3 {
            return None;
        }
        let a = *self.coeffs.get(2).unwrap_or(&0);
        let b = *self.coeffs.get(1).unwrap_or(&0);
        let c = *self.coeffs.get(0).unwrap_or(&0);
        if a == 0 {
            return None;
        }
        let sd = is_perfect_square_i64(b * b - 4 * a * c)?;
        let r1n = -b + sd;
        let r2n = -b - sd;
        if r1n % (2 * a) != 0 || r2n % (2 * a) != 0 {
            return None;
        }
        let r1 = r1n / (2 * a);
        let r2 = r2n / (2 * a);
        // a(x - r1)(x - r2) -> move 'a' into content and keep monic
        let mut p1 = Poly { var: self.var.clone(), coeffs: vec![-r1, 1], content: 1 };
        let mut p2 =
            Poly { var: self.var.clone(), coeffs: vec![-r2, 1], content: self.content * a };
        p1.factor_content();
        p2.factor_content();
        Some((p1, p2))
    }
}

fn first_symbol(v: &Value) -> Option<String> {
    match v {
        Value::Symbol(s) => Some(s.clone()),
        Value::Expr { head, args } => {
            if let Some(s) = first_symbol(head) {
                return Some(s);
            }
            for a in args {
                if let Some(s) = first_symbol(a) {
                    return Some(s);
                }
            }
            None
        }
        _ => None,
    }
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.abs()
}

fn is_perfect_square_i64(n: i64) -> Option<i64> {
    if n < 0 {
        return None;
    }
    let r = (n as f64).sqrt().round() as i64;
    if r * r == n {
        Some(r)
    } else {
        None
    }
}

pub fn register_algebra_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    register_if(ev, pred, "Simplify", simplify as NativeFn, Attributes::empty());
    register_if(ev, pred, "Expand", expand as NativeFn, Attributes::empty());
    register_if(ev, pred, "ExpandAll", expand_all as NativeFn, Attributes::empty());
    register_if(ev, pred, "CollectTerms", collect as NativeFn, Attributes::empty());
    register_if(ev, pred, "CollectTermsBy", collect_by as NativeFn, Attributes::empty());
    register_if(ev, pred, "Factor", factor as NativeFn, Attributes::empty());
    register_if(ev, pred, "D", diff as NativeFn, Attributes::empty());
    register_if(ev, pred, "CancelRational", cancel as NativeFn, Attributes::empty());
    register_if(ev, pred, "Apart", apart as NativeFn, Attributes::empty());
    register_if(ev, pred, "Solve", solve as NativeFn, Attributes::empty());
    register_if(ev, pred, "Roots", roots as NativeFn, Attributes::empty());
}
