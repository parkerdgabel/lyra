use lyra_stdlib as stdlib;
use lyra_runtime::Evaluator;
use lyra_core::value::Value;

fn eval_str(ev: &mut Evaluator, s: &str) -> Value {
    let mut p = lyra_parser::Parser::from_source(s);
    let parsed = p.parse_all().unwrap().remove(0);
    ev.eval(parsed)
}

fn as_f64(v: &Value) -> f64 { match v { Value::Real(x) => *x, Value::Integer(n) => *n as f64, _ => f64::NAN } }

#[test]
fn matrix_norm_variants() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let a = "{{1,2},{3,4}}";
    let n1 = eval_str(&mut ev, &format!("MatrixNorm[{}, 1]", a));
    let ninf = eval_str(&mut ev, &format!("MatrixNorm[{}, Infinity]", a));
    let nf = eval_str(&mut ev, &format!("MatrixNorm[{}, \"Frobenius\"]", a));
    let n2 = eval_str(&mut ev, &format!("MatrixNorm[{}, 2]", a));
    assert!((as_f64(&n1) - 6.0).abs() < 1e-9);
    assert!((as_f64(&ninf) - 7.0).abs() < 1e-9);
    assert!((as_f64(&nf) - (30.0f64).sqrt()).abs() < 1e-9);
    // spectral norm of [[1,2],[3,4]] ~= 5.464985704
    assert!((as_f64(&n2) - 5.464985704).abs() < 1e-4);
}

#[test]
fn norm_matrix_consistency() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let a = "{{1,2},{3,4}}";
    let n2a = eval_str(&mut ev, &format!("Norm[{}, 2]", a));
    let n2b = eval_str(&mut ev, &format!("MatrixNorm[{}, 2]", a));
    assert!((as_f64(&n2a) - as_f64(&n2b)).abs() < 1e-9);
    let nf1 = eval_str(&mut ev, &format!("Norm[{}]", a));
    let nf2 = eval_str(&mut ev, &format!("Norm[{}, \"Frobenius\"]", a));
    let nf3 = eval_str(&mut ev, &format!("MatrixNorm[{}, \"Frobenius\"]", a));
    assert!((as_f64(&nf1) - as_f64(&nf2)).abs() < 1e-12);
    assert!((as_f64(&nf1) - as_f64(&nf3)).abs() < 1e-12);
}

#[test]
fn pseudoinverse_tolerance_and_shape() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // Nearly singular diagonal
    let a_val = Value::List(vec![
        Value::List(vec![Value::Integer(1), Value::Integer(0)]),
        Value::List(vec![Value::Integer(0), Value::Real(1e-8)])
    ]);
    let pinv_default = ev.eval(Value::Expr { head: Box::new(Value::Symbol("PseudoInverse".into())), args: vec![a_val.clone()] });
    let norm_default = ev.eval(Value::Expr { head: Box::new(Value::Symbol("MatrixNorm".into())), args: vec![pinv_default.clone(), Value::Integer(2)] });
    assert!(as_f64(&norm_default) > 1e6); // large due to 1/1e-8
    let pinv_tol = ev.eval(Value::Expr { head: Box::new(Value::Symbol("PseudoInverse".into())), args: vec![a_val.clone(), Value::Real(1e-6)] });
    let norm_tol = ev.eval(Value::Expr { head: Box::new(Value::Symbol("MatrixNorm".into())), args: vec![pinv_tol.clone(), Value::Integer(2)] });
    assert!(as_f64(&norm_tol) < 10.0);

    // Rectangular: A is 2x3; pseudoinverse is 3x2
    let a_rect = Value::List(vec![
        Value::List(vec![Value::Integer(1), Value::Integer(0), Value::Integer(0)]),
        Value::List(vec![Value::Integer(0), Value::Integer(1), Value::Integer(0)]),
    ]);
    let pinv = ev.eval(Value::Expr { head: Box::new(Value::Symbol("PseudoInverse".into())), args: vec![a_rect.clone()] });
    let shp = ev.eval(Value::Expr { head: Box::new(Value::Symbol("Shape".into())), args: vec![pinv.clone()] });
    assert_eq!(shp, Value::List(vec![Value::Integer(3), Value::Integer(2)]));
    // Sanity: Dot[A, A^+] is 2x2
    let prod = ev.eval(Value::Expr { head: Box::new(Value::Symbol("Dot".into())), args: vec![a_rect.clone(), pinv.clone()] });
    let shp2 = ev.eval(Value::Expr { head: Box::new(Value::Symbol("Shape".into())), args: vec![prod] });
    assert_eq!(shp2, Value::List(vec![Value::Integer(2), Value::Integer(2)]));
}
