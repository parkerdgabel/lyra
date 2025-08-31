use lyra_stdlib as stdlib;
use lyra_runtime::Evaluator;
use lyra_core::value::Value;

fn eval_str(ev: &mut Evaluator, s: &str) -> Value {
    let mut p = lyra_parser::Parser::from_source(s);
    let mut exprs = p.parse_all().unwrap();
    let expr = exprs.remove(0);
    ev.eval(expr)
}

fn as_f64(v: &Value) -> f64 { match v { Value::Real(x) => *x, Value::Integer(n) => *n as f64, _ => f64::NAN } }

#[test]
fn bernoulli_edges() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // PDF at 0/1 and others
    let v0 = eval_str(&mut ev, "PDF[Bernoulli[0.3], 0]");
    let v1 = eval_str(&mut ev, "PDF[Bernoulli[0.3], 1]");
    let v2 = eval_str(&mut ev, "PDF[Bernoulli[0.3], 2]");
    assert!((as_f64(&v0) - 0.7).abs() < 1e-9);
    assert!((as_f64(&v1) - 0.3).abs() < 1e-9);
    assert!((as_f64(&v2) - 0.0).abs() < 1e-12);
    // CDF steps
    let c_neg = eval_str(&mut ev, "CDF[Bernoulli[0.3], -1]");
    let c_0 = eval_str(&mut ev, "CDF[Bernoulli[0.3], 0.5]");
    let c_1 = eval_str(&mut ev, "CDF[Bernoulli[0.3], 1]");
    assert!(as_f64(&c_neg) == 0.0);
    assert!((as_f64(&c_0) - 0.7).abs() < 1e-9);
    assert!(as_f64(&c_1) == 1.0);
}

#[test]
fn binomial_basic() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // PMF at some k for n=10, p=0.5
    let pmf_0 = eval_str(&mut ev, "PDF[BinomialDistribution[10, 0.5], 0]");
    let pmf_5 = eval_str(&mut ev, "PDF[BinomialDistribution[10, 0.5], 5]");
    assert!((as_f64(&pmf_0) - 0.0009765625).abs() < 1e-12);
    assert!((as_f64(&pmf_5) - 0.24609375).abs() < 1e-12);
    // CDF should be increasing and bounded
    let c_3 = eval_str(&mut ev, "CDF[BinomialDistribution[10, 0.5], 3]");
    let c_7 = eval_str(&mut ev, "CDF[BinomialDistribution[10, 0.5], 7]");
    assert!(as_f64(&c_3) < as_f64(&c_7));
    assert!(as_f64(&c_7) <= 1.0);
}

#[test]
fn poisson_exponential_gamma_sanity() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // Poisson pmf sum approx
    let mut sum = 0.0;
    for k in 0..20 {
        let expr = format!("PDF[Poisson[3.0], {}]", k);
        sum += as_f64(&eval_str(&mut ev, &expr));
    }
    assert!((sum - 1.0).abs() < 1e-6);
    // Exponential PDF/CDF
    let pdf_e = eval_str(&mut ev, "PDF[Exponential[2.0], 1.0]");
    let cdf_e = eval_str(&mut ev, "CDF[Exponential[2.0], 1.0]");
    assert!((as_f64(&pdf_e) - (2.0 * (-2.0f64).exp())).abs() < 1e-12);
    assert!((as_f64(&cdf_e) - (1.0 - (-2.0f64).exp())).abs() < 1e-12);
    // Gamma(k=1,theta) reduces to exponential with lambda=1/theta
    let pdf_g = eval_str(&mut ev, "PDF[Gamma[1.0, 0.5], 1.0]");
    let pdf_e2 = eval_str(&mut ev, "PDF[Exponential[2.0], 1.0]");
    assert!((as_f64(&pdf_g) - as_f64(&pdf_e2)).abs() < 1e-8);
    // CDF monotonic sanity for Gamma
    let cdf_g1 = eval_str(&mut ev, "CDF[Gamma[2.0, 1.0], 1.0]");
    let cdf_g2 = eval_str(&mut ev, "CDF[Gamma[2.0, 1.0], 3.0]");
    assert!(as_f64(&cdf_g1) < as_f64(&cdf_g2));
}

#[test]
fn poisson_cdf_spot() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // λ=2, CDF at x=3 equals sum_{k=0..3} e^{-2} 2^k/k!
    let c = eval_str(&mut ev, "CDF[Poisson[2.0], 3.0]");
    let expected = (-2.0f64).exp() * (1.0 + 2.0 + (2.0f64).powi(2)/2.0 + (2.0f64).powi(3)/6.0);
    assert!((as_f64(&c) - expected).abs() < 1e-10);
}

#[test]
fn gamma_cdf_erlang_closed_forms() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // k=2, θ=1, CDF(x) = 1 - e^{-x}(1+x)
    let c2 = eval_str(&mut ev, "CDF[Gamma[2.0, 1.0], 3.0]");
    let e2 = 1.0 - (-3.0f64).exp() * (1.0 + 3.0);
    assert!((as_f64(&c2) - e2).abs() < 1e-9);
    // k=3, θ=1, CDF(x) = 1 - e^{-x}(1 + x + x^2/2)
    let c3 = eval_str(&mut ev, "CDF[Gamma[3.0, 1.0], 2.0]");
    let e3 = 1.0 - (-2.0f64).exp() * (1.0 + 2.0 + (2.0f64).powi(2)/2.0);
    assert!((as_f64(&c3) - e3).abs() < 1e-9);
}
