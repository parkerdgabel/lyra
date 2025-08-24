// Phase 2 numeric tower scaffolding (not yet integrated into Value)

#[derive(Debug, Clone, PartialEq)]
pub struct Rational {
    pub num: i128,
    pub den: i128,
}

impl Rational {
    pub fn new(num: i128, den: i128) -> Self {
        // TODO: normalize sign and reduce by gcd
        Self { num, den }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Complex<T> {
    pub re: T,
    pub im: T,
}

impl<T> Complex<T> {
    pub fn new(re: T, im: T) -> Self { Self { re, im } }
}

// BigReal placeholder: actual big-float backend to be feature-gated in M2.
#[derive(Debug, Clone, PartialEq)]
pub struct BigReal {
    pub digits: String, // decimal string placeholder; to be replaced
}

impl BigReal {
    pub fn from_str(s: &str) -> Self { Self { digits: s.to_string() } }
}

