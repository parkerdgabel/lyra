use lyra_core::numeric::{BigReal, Complex, Rational};

#[test]
#[ignore]
fn rational_constructs_and_normalizes() {
    let r = Rational::new(2, 4);
    // TODO: assert reduced to 1/2 once implemented
    assert_eq!(r.num, 2);
    assert_eq!(r.den, 4);
}

#[test]
#[ignore]
fn complex_basic() {
    let z = Complex::new(1.0f64, -2.0);
    assert_eq!(z.re, 1.0);
    assert_eq!(z.im, -2.0);
}

#[test]
#[ignore]
fn bigreal_placeholder() {
    let b = BigReal::from_str("3.14159");
    assert!(b.digits.starts_with("3.14"));
}
