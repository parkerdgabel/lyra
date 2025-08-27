use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;

#[test]
fn set_basics_and_ops() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let v = ev.eval(Value::expr(
        Value::Symbol("SetFromList".into()),
        vec![Value::List(vec![
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(2),
            Value::Integer(3),
        ])],
    ));
    let sz = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("SetSize".into())),
        args: vec![v.clone()],
    });
    assert_eq!(sz, Value::Integer(3));
    let mq = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("SetMemberQ".into())),
        args: vec![v.clone(), Value::Integer(2)],
    });
    assert_eq!(mq, Value::Boolean(true));
    let s1 = ev.eval(Value::expr(
        Value::Symbol("SetFromList".into()),
        vec![Value::List(vec![Value::Integer(1), Value::Integer(2)])],
    ));
    let s2 = ev.eval(Value::expr(
        Value::Symbol("SetFromList".into()),
        vec![Value::List(vec![Value::Integer(2), Value::Integer(4)])],
    ));
    let u = ev.eval(Value::expr(Value::Symbol("SetUnion".into()), vec![s1, s2]));
    let us =
        ev.eval(Value::Expr { head: Box::new(Value::Symbol("SetSize".into())), args: vec![u] });
    assert_eq!(us, Value::Integer(3));
    let s3 = ev.eval(Value::expr(
        Value::Symbol("SetFromList".into()),
        vec![Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)])],
    ));
    let s4 = ev.eval(Value::expr(
        Value::Symbol("SetFromList".into()),
        vec![Value::List(vec![Value::Integer(2), Value::Integer(3), Value::Integer(4)])],
    ));
    let inter = ev.eval(Value::expr(Value::Symbol("SetIntersection".into()), vec![s3, s4]));
    let isz =
        ev.eval(Value::Expr { head: Box::new(Value::Symbol("SetSize".into())), args: vec![inter] });
    assert_eq!(isz, Value::Integer(2));
    let s5 = ev.eval(Value::expr(
        Value::Symbol("SetFromList".into()),
        vec![Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)])],
    ));
    let s6 = ev.eval(Value::expr(
        Value::Symbol("SetFromList".into()),
        vec![Value::List(vec![Value::Integer(2), Value::Integer(5)])],
    ));
    let diff = ev.eval(Value::expr(Value::Symbol("SetDifference".into()), vec![s5, s6]));
    let dsz =
        ev.eval(Value::Expr { head: Box::new(Value::Symbol("SetSize".into())), args: vec![diff] });
    assert_eq!(dsz, Value::Integer(2));
}

#[test]
fn list_set_ops() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let u = ev.eval(Value::expr(
        Value::Symbol("ListUnion".into()),
        vec![
            Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(2)]),
            Value::List(vec![Value::Integer(2), Value::Integer(3)]),
        ],
    ));
    assert!(matches!(u, Value::List(v) if v.len()==3));
    let i = ev.eval(Value::expr(
        Value::Symbol("ListIntersection".into()),
        vec![
            Value::List(vec![
                Value::Integer(1),
                Value::Integer(2),
                Value::Integer(2),
                Value::Integer(3),
            ]),
            Value::List(vec![Value::Integer(2), Value::Integer(4), Value::Integer(2)]),
        ],
    ));
    assert!(matches!(i, Value::List(v) if v.len()==1));
    let d = ev.eval(Value::expr(
        Value::Symbol("ListDifference".into()),
        vec![
            Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)]),
            Value::List(vec![Value::Integer(2)]),
        ],
    ));
    assert!(matches!(d, Value::List(v) if v.len()==2));
}

#[test]
fn queue_stack_pq() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // Queue
    let q = ev.eval(Value::expr(Value::Symbol("Queue".into()), vec![]));
    let q = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("Enqueue".into())),
        args: vec![q.clone(), Value::Integer(1)],
    });
    let q = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("Enqueue".into())),
        args: vec![q.clone(), Value::Integer(2)],
    });
    let x1 = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("Dequeue".into())),
        args: vec![q.clone()],
    });
    assert_eq!(x1, Value::Integer(1));
    // Stack
    let s = ev.eval(Value::expr(Value::Symbol("Stack".into()), vec![]));
    let s = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("Push".into())),
        args: vec![s.clone(), Value::Integer(5)],
    });
    let s = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("Push".into())),
        args: vec![s.clone(), Value::Integer(7)],
    });
    let t =
        ev.eval(Value::Expr { head: Box::new(Value::Symbol("Top".into())), args: vec![s.clone()] });
    assert_eq!(t, Value::Integer(7));
    let p =
        ev.eval(Value::Expr { head: Box::new(Value::Symbol("Pop".into())), args: vec![s.clone()] });
    assert_eq!(p, Value::Integer(7));
    // PQ min
    let pq = ev.eval(Value::expr(Value::Symbol("PriorityQueue".into()), vec![]));
    let pq = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("PQInsert".into())),
        args: vec![pq.clone(), Value::Integer(10)],
    });
    let pq = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("PQInsert".into())),
        args: vec![pq.clone(), Value::Integer(3)],
    });
    let pq = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("PQInsert".into())),
        args: vec![pq.clone(), Value::Integer(7)],
    });
    let a = ev.eval(Value::Expr {
        head: Box::new(Value::Symbol("PQPop".into())),
        args: vec![pq.clone()],
    });
    assert_eq!(a, Value::Integer(3));
}
