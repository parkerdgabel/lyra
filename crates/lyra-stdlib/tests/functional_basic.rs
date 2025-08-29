use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;

fn eval_str(s: &str) -> Value {
    use lyra_parser::Parser;
    let mut p = Parser::from_source(s);
    let exprs = p.parse_all().expect("parse");
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let mut last = Value::Symbol("Null".into());
    for e in exprs {
        last = ev.eval(e);
    }
    last
}

#[test]
fn apply_and_compose() {
    let r = eval_str("Apply[Plus, {1,2,3}]");
    assert_eq!(r, Value::Integer(6));
    let r = eval_str("Plus @@ {1,2,3}");
    assert_eq!(r, Value::Integer(6));
    let r = eval_str("Plus @@@ {{1,2},{3,4}}");
    match r {
        Value::List(v) => {
            assert_eq!(v, vec![3, 7].into_iter().map(Value::Integer).collect::<Vec<_>>())
        }
        _ => panic!(),
    }
    let _r = eval_str("Compose[Times, Plus][2]"); // Times[Plus[2]] -> invalid, but Compose should produce a function; apply to x: f(g(x))
                                                 // We'll compose simple numeric functions: Compose[(#*2)&, (#+3)&][10]
    let r = eval_str("Compose[(#*2)&, (#+3)&][10]");
    assert_eq!(r, Value::Integer(26));
}

#[test]
fn nest_and_foldlist() {
    let r = eval_str("Nest[(#*2)&, 1, 4]");
    assert_eq!(r, Value::Integer(16));
    let r = eval_str("NestList[(#*2)&, 1, 4]");
    match r {
        Value::List(v) => assert_eq!(v.len(), 5),
        _ => panic!(),
    }
    let r = eval_str("FoldList[Plus, {1,2,3,4}]");
    match r {
        Value::List(v) => {
            assert_eq!(v, vec![1, 3, 6, 10].into_iter().map(Value::Integer).collect::<Vec<_>>())
        }
        _ => panic!(),
    }
}

#[test]
fn fixedpoint_and_through() {
    let r = eval_str("FixedPoint[(# - 1)&, 5]");
    // With default MaxIterations=100 and integer arithmetic, it stabilizes at -95 then keeps going, but we stop early when no change; this function never stabilizes; expect last result
    match r {
        Value::Integer(_) => {}
        _ => panic!(),
    }
    let r = eval_str("Through[{(#&), (#+1)&}, 5]");
    match r {
        Value::List(v) => {
            assert_eq!(v.len(), 2);
        }
        _ => panic!(),
    }
}

#[test]
fn operator_forms() {
    let r = eval_str("Map[(#*#)&][{1,2,3}]");
    match r {
        Value::List(v) => {
            assert_eq!(v, vec![1, 4, 9].into_iter().map(Value::Integer).collect::<Vec<_>>())
        }
        _ => panic!(),
    }
    let r = eval_str("Reduce[Plus][{1,2,3}]");
    assert_eq!(r, Value::Integer(6));
    let r = eval_str("(#*#)& /@ {1,2,3}");
    match r {
        Value::List(v) => {
            assert_eq!(v, vec![1, 4, 9].into_iter().map(Value::Integer).collect::<Vec<_>>())
        }
        _ => panic!(),
    }
}

#[test]
fn set_delayed_defs() {
    // Function DownValue with SetDelayed
    let r = eval_str("f[x_] := x*x; f[5]");
    assert_eq!(r, Value::Integer(25));
    // OwnValue with SetDelayed (evaluated on access)
    let r = eval_str("y := Plus[2,3]; y");
    assert_eq!(r, Value::Integer(5));
}
