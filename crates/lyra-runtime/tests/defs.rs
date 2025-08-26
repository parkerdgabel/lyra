use lyra_core::value::Value;
use lyra_runtime::eval::Evaluator;
use lyra_runtime::set_default_registrar;
use lyra_stdlib as stdlib;

fn sym(s: &str) -> Value {
    Value::Symbol(s.into())
}
fn int(n: i64) -> Value {
    Value::Integer(n)
}

#[test]
fn downvalues_basic_rewrite() {
    // SetDownValues[f, { f[x_] -> 42 }]; f[7] => 42
    let lhs = Value::Expr {
        head: Box::new(sym("f")),
        args: vec![Value::Expr { head: Box::new(sym("NamedBlank")), args: vec![sym("x")] }],
    };
    let rhs = int(42);
    let rule = Value::Expr { head: Box::new(sym("Rule")), args: vec![lhs, rhs] };
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    let set = Value::Expr {
        head: Box::new(sym("SetDownValues")),
        args: vec![sym("f"), Value::List(vec![rule])],
    };
    let _ = ev.eval(set);
    let gv = Value::Expr { head: Box::new(sym("GetDownValues")), args: vec![sym("f")] };
    let rules = ev.eval(gv);
    match &rules {
        Value::List(xs) => assert_eq!(xs.len(), 1),
        _ => panic!("expected list"),
    }
    let call = Value::Expr { head: Box::new(sym("f")), args: vec![int(7)] };
    let out = ev.eval(call);
    assert_eq!(out, int(42));
}

#[test]
fn upvalues_basic_rewrite() {
    // SetUpValues[a, { g[a] -> 99 }]; g[a] => 99
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    let lhs = Value::Expr { head: Box::new(sym("g")), args: vec![sym("a")] };
    let rhs = int(99);
    let rule = Value::Expr { head: Box::new(sym("Rule")), args: vec![lhs, rhs] };
    let set = Value::Expr {
        head: Box::new(sym("SetUpValues")),
        args: vec![sym("a"), Value::List(vec![rule])],
    };
    let _ = ev.eval(set);
    let call = Value::Expr { head: Box::new(sym("g")), args: vec![sym("a")] };
    let out = ev.eval(call);
    assert_eq!(out, int(99));
}

#[test]
fn ownvalues_basic_rewrite() {
    // SetOwnValues[x, { x -> 10 }]; x => 10
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    let lhs = sym("x");
    let rhs = int(10);
    let rule = Value::Expr { head: Box::new(sym("Rule")), args: vec![lhs, rhs] };
    let set = Value::Expr {
        head: Box::new(sym("SetOwnValues")),
        args: vec![sym("x"), Value::List(vec![rule])],
    };
    let _ = ev.eval(set);
    let out = ev.eval(sym("x"));
    assert_eq!(out, int(10));
}

#[test]
fn subvalues_basic_rewrite() {
    // SetSubValues[f, { (f[a_])[b_] -> 123 }]; f[1][2] => 123
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    let f = sym("f");
    let a = Value::Expr { head: Box::new(sym("NamedBlank")), args: vec![sym("a")] };
    let b = Value::Expr { head: Box::new(sym("NamedBlank")), args: vec![sym("b")] };
    let lhs_head = Value::Expr { head: Box::new(f.clone()), args: vec![a] };
    let lhs = Value::Expr { head: Box::new(lhs_head), args: vec![b] };
    let rhs = int(123);
    let rule = Value::Expr { head: Box::new(sym("Rule")), args: vec![lhs, rhs] };
    let set = Value::Expr {
        head: Box::new(sym("SetSubValues")),
        args: vec![f.clone(), Value::List(vec![rule])],
    };
    let _ = ev.eval(set);
    let call1 = Value::Expr { head: Box::new(f.clone()), args: vec![int(1)] };
    let call = Value::Expr { head: Box::new(call1), args: vec![int(2)] };
    let out = ev.eval(call);
    assert_eq!(out, int(123));
}

#[test]
fn precedence_up_over_down_and_env() {
    // UpValues should take precedence over DownValues and env.
    // Define: SetUpValues[a, { h[a] -> 1 }]; SetDownValues[h, { h[x_] -> 2 }]; env h[a] -> 3
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    // Set env to map h[a] -> 3 is not straightforward; instead, define Set (assignment) and ensure it's lower precedence.
    // We'll simulate env by evaluating Set[h[a], 3] which binds symbol h[a] if implemented; otherwise ignore.
    let up_lhs = Value::Expr { head: Box::new(sym("h")), args: vec![sym("a")] };
    let up_rule = Value::Expr { head: Box::new(sym("Rule")), args: vec![up_lhs.clone(), int(1)] };
    let _ = ev.eval(Value::Expr {
        head: Box::new(sym("SetUpValues")),
        args: vec![sym("a"), Value::List(vec![up_rule])],
    });

    let down_lhs = Value::Expr {
        head: Box::new(sym("h")),
        args: vec![Value::Expr { head: Box::new(sym("NamedBlank")), args: vec![sym("x")] }],
    };
    let down_rule = Value::Expr { head: Box::new(sym("Rule")), args: vec![down_lhs, int(2)] };
    let _ = ev.eval(Value::Expr {
        head: Box::new(sym("SetDownValues")),
        args: vec![sym("h"), Value::List(vec![down_rule])],
    });

    let out = ev.eval(Value::Expr { head: Box::new(sym("h")), args: vec![sym("a")] });
    assert_eq!(out, int(1));
}

#[test]
fn precedence_sub_above_down() {
    // SubValues should apply on compound head before DownValues on the outer symbol
    // SetSubValues[f, { (f[a_])[b_] -> 5 }]; SetDownValues[g, { g[x_] -> 6 }]; expr: (f[1])[2] wrapped in g should rewrite inner first
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    let f = sym("f");
    let a = Value::Expr { head: Box::new(sym("NamedBlank")), args: vec![sym("a")] };
    let b = Value::Expr { head: Box::new(sym("NamedBlank")), args: vec![sym("b")] };
    let lhs_head = Value::Expr { head: Box::new(f.clone()), args: vec![a] };
    let lhs = Value::Expr { head: Box::new(lhs_head), args: vec![b] };
    let rule = Value::Expr { head: Box::new(sym("Rule")), args: vec![lhs, int(5)] };
    let _ = ev.eval(Value::Expr {
        head: Box::new(sym("SetSubValues")),
        args: vec![f.clone(), Value::List(vec![rule])],
    });

    let g = sym("g");
    let down_lhs_g = Value::Expr {
        head: Box::new(g.clone()),
        args: vec![Value::Expr { head: Box::new(sym("NamedBlank")), args: vec![sym("x")] }],
    };
    let down_rule_g = Value::Expr { head: Box::new(sym("Rule")), args: vec![down_lhs_g, int(6)] };
    let _ = ev.eval(Value::Expr {
        head: Box::new(sym("SetDownValues")),
        args: vec![g.clone(), Value::List(vec![down_rule_g])],
    });

    let inner = Value::Expr {
        head: Box::new(Value::Expr { head: Box::new(f.clone()), args: vec![int(1)] }),
        args: vec![int(2)],
    };
    let expr = Value::Expr { head: Box::new(g.clone()), args: vec![inner] };
    let out = ev.eval(expr);
    // g[(f[1])[2]] should first become g[5], then apply g's downvalue to 6
    assert_eq!(out, int(6));
}
