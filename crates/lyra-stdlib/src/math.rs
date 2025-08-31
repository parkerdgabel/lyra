// (moved to top of file)

use crate::register_if;
#[cfg(feature = "tools")]
use crate::tool_spec;
#[cfg(feature = "tools")]
use crate::tools::add_specs;
use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
#[cfg(feature = "big-real-rug")]
use rug::Float;
#[cfg(feature = "tools")]
use std::collections::HashMap;

/// Register core math: arithmetic, statistics, trig, number theory,
/// combinatorics, random, and numeric utilities.
pub fn register_math(ev: &mut Evaluator) {
    ev.register(
        "Plus",
        plus as NativeFn,
        Attributes::LISTABLE | Attributes::FLAT | Attributes::ORDERLESS | Attributes::ONE_IDENTITY,
    );
    ev.register(
        "Times",
        times as NativeFn,
        Attributes::LISTABLE | Attributes::FLAT | Attributes::ORDERLESS | Attributes::ONE_IDENTITY,
    );
    ev.register("Minus", minus as NativeFn, Attributes::LISTABLE);
    ev.register("Divide", divide as NativeFn, Attributes::LISTABLE);
    ev.register("Power", power as NativeFn, Attributes::empty());
    ev.register("Abs", abs_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Min", min_fn as NativeFn, Attributes::FLAT | Attributes::ORDERLESS);
    ev.register("Max", max_fn as NativeFn, Attributes::FLAT | Attributes::ORDERLESS);

    // New math functions
    ev.register("Floor", floor_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Ceiling", ceiling_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Round", round_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Trunc", trunc_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Mod", mod_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Quotient", quotient_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Remainder", remainder_fn as NativeFn, Attributes::LISTABLE);
    ev.register("DivMod", divmod_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Sqrt", sqrt_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Exp", exp_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Log", log_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Sin", sin_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Cos", cos_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Tan", tan_fn as NativeFn, Attributes::LISTABLE);
    ev.register("ASin", asin_fn as NativeFn, Attributes::LISTABLE);
    ev.register("ACos", acos_fn as NativeFn, Attributes::LISTABLE);
    ev.register("ATan", atan_fn as NativeFn, Attributes::LISTABLE);
    ev.register("ATan2", atan2_fn as NativeFn, Attributes::empty());
    ev.register("NthRoot", nthroot_fn as NativeFn, Attributes::empty());
    ev.register("Total", total_fn as NativeFn, Attributes::empty());
    ev.register("Mean", mean_fn as NativeFn, Attributes::empty());
    ev.register("Median", median_fn as NativeFn, Attributes::empty());
    ev.register("Variance", variance_fn as NativeFn, Attributes::empty());
    ev.register("StandardDeviation", stddev_fn as NativeFn, Attributes::empty());
    ev.register("Quantile", quantile_fn as NativeFn, Attributes::empty());
    ev.register("Percentile", percentile_fn as NativeFn, Attributes::empty());
    ev.register("Mode", mode_fn as NativeFn, Attributes::empty());
    ev.register("Correlation", correlation_fn as NativeFn, Attributes::empty());
    ev.register("Covariance", covariance_fn as NativeFn, Attributes::empty());
    ev.register("Skewness", skewness_fn as NativeFn, Attributes::empty());
    ev.register("Kurtosis", kurtosis_fn as NativeFn, Attributes::empty());
    ev.register("GCD", gcd_fn as NativeFn, Attributes::FLAT | Attributes::ORDERLESS);
    ev.register("LCM", lcm_fn as NativeFn, Attributes::FLAT | Attributes::ORDERLESS);
    ev.register("Factorial", factorial_fn as NativeFn, Attributes::empty());
    ev.register("Binomial", binomial_fn as NativeFn, Attributes::empty());
    // Number theory extensions
    ev.register("PrimeFactors", prime_factors_fn as NativeFn, Attributes::empty());
    ev.register("EulerPhi", euler_phi_fn as NativeFn, Attributes::LISTABLE);
    ev.register("MobiusMu", mobius_mu_fn as NativeFn, Attributes::LISTABLE);
    ev.register("PowerMod", power_mod_fn as NativeFn, Attributes::empty());
    // Number theory
    ev.register("ExtendedGCD", extended_gcd_fn as NativeFn, Attributes::empty());
    ev.register("ModInverse", mod_inverse_fn as NativeFn, Attributes::empty());
    ev.register("ChineseRemainder", chinese_remainder_fn as NativeFn, Attributes::empty());
    ev.register("DividesQ", divides_q_fn as NativeFn, Attributes::LISTABLE);
    ev.register("CoprimeQ", coprime_q_fn as NativeFn, Attributes::LISTABLE);
    ev.register("PrimeQ", prime_q_fn as NativeFn, Attributes::LISTABLE);
    ev.register("NextPrime", next_prime_fn as NativeFn, Attributes::LISTABLE);
    ev.register("FactorInteger", factor_integer_fn as NativeFn, Attributes::empty());
    // Combinatorics
    ev.register("Permutations", permutations_fn as NativeFn, Attributes::empty());
    ev.register("Combinations", combinations_fn as NativeFn, Attributes::empty());
    ev.register("PermutationsCount", permutations_count_fn as NativeFn, Attributes::empty());
    ev.register("CombinationsCount", combinations_count_fn as NativeFn, Attributes::empty());
    ev.register("ToDegrees", to_degrees_fn as NativeFn, Attributes::LISTABLE);
    ev.register("ToRadians", to_radians_fn as NativeFn, Attributes::LISTABLE);
    ev.register("Clip", clip_fn as NativeFn, Attributes::empty());
    // Internal aliases for tensor-aware dispatch fallbacks
    ev.register("__MathExp", exp_fn as NativeFn, Attributes::LISTABLE);
    ev.register("__MathLog", log_fn as NativeFn, Attributes::LISTABLE);
    ev.register("__MathSqrt", sqrt_fn as NativeFn, Attributes::LISTABLE);
    ev.register("__MathSin", sin_fn as NativeFn, Attributes::LISTABLE);
    ev.register("__MathCos", cos_fn as NativeFn, Attributes::LISTABLE);
    ev.register("__MathTanh", tanh_fn as NativeFn, Attributes::LISTABLE);
    ev.register("__MathPower", power as NativeFn, Attributes::empty());
    ev.register("__MathClip", clip_fn as NativeFn, Attributes::empty());
    ev.register("Signum", signum_fn as NativeFn, Attributes::LISTABLE);
    // Randomness (simple deterministic RNG)
    ev.register("SeedRandom", seed_random_fn as NativeFn, Attributes::empty());
    ev.register("RandomInteger", random_integer_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("RandomReal", random_real_fn as NativeFn, Attributes::HOLD_ALL);
    // Stats helpers
    ev.register("DescriptiveStats", descriptive_stats_fn as NativeFn, Attributes::empty());
    ev.register("Quantiles", quantiles_fn as NativeFn, Attributes::empty());
    ev.register("RollingStats", rolling_stats_fn as NativeFn, Attributes::empty());
    ev.register("RandomSample", random_sample_fn as NativeFn, Attributes::HOLD_ALL);

    // Tier1: Distributions and probability functions (stubs)
    ev.register("Normal", dist_normal as NativeFn, Attributes::empty());
    ev.register("Bernoulli", dist_bernoulli as NativeFn, Attributes::empty());
    ev.register("BinomialDistribution", dist_binomial as NativeFn, Attributes::empty());
    ev.register("Poisson", dist_poisson as NativeFn, Attributes::empty());
    ev.register("Exponential", dist_exponential as NativeFn, Attributes::empty());
    ev.register("Gamma", dist_gamma as NativeFn, Attributes::empty());
    ev.register("PDF", pdf_fn as NativeFn, Attributes::empty());
    ev.register("CDF", cdf_fn as NativeFn, Attributes::empty());
    ev.register("RandomVariate", random_variate_fn as NativeFn, Attributes::HOLD_ALL);

    #[cfg(feature = "tools")]
    add_specs(vec![
        tool_spec!("Plus", summary: "Sum numbers (variadic)", params: ["args"], tags: ["math","sum"], input_schema: Value::Assoc(HashMap::from([
            ("type".to_string(), Value::String("object".into())),
            ("properties".to_string(), Value::Assoc(HashMap::from([
                ("args".to_string(), Value::Assoc(HashMap::from([
                    ("type".to_string(), Value::String("array".into())),
                    ("items".to_string(), Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("number")))]))),
                ]))),
            ]))),
        ])), output_schema: Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("number")))])), examples: [
            Value::Assoc(HashMap::from([
                ("args".to_string(), Value::Assoc(HashMap::from([("args".to_string(), Value::List(vec![Value::Integer(1), Value::Integer(2)]))]))),
                ("result".to_string(), Value::Integer(3)),
            ]))
        ]),
        tool_spec!("Times", summary: "Multiply numbers (variadic)", params: ["args"], tags: ["math","product"], input_schema: Value::Assoc(HashMap::from([
            ("type".to_string(), Value::String("object".into())),
            ("properties".to_string(), Value::Assoc(HashMap::from([
                ("args".to_string(), Value::Assoc(HashMap::from([
                    ("type".to_string(), Value::String("array".into())),
                    ("items".to_string(), Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("number")))]))),
                ]))),
            ]))),
        ])), output_schema: Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("number")))])), examples: [
            Value::Assoc(HashMap::from([
                ("args".to_string(), Value::Assoc(HashMap::from([("args".to_string(), Value::List(vec![Value::Integer(2), Value::Integer(3)]))]))),
                ("result".to_string(), Value::Integer(6)),
            ]))
        ]),
        tool_spec!("Quantile", summary: "Quantile(s) of numeric data (R7)", params: ["data","q"], tags: ["math","stats"], examples: [Value::String("Quantile[{1,2,3,4}, 0.25]  ==> 1.75".into())]),
        tool_spec!("Percentile", summary: "Percentiles of numeric data (R7)", params: ["data","p"], tags: ["math","stats"], examples: [Value::String("Percentile[{1,2,3,4}, 25]  ==> 1.75".into())]),
        tool_spec!("Mode", summary: "Most frequent element (first tie)", params: ["data"], tags: ["math","stats"], examples: [Value::String("Mode[{1,2,2,3}]  ==> 2".into())]),
        tool_spec!("Correlation", summary: "Pearson correlation of two lists", params: ["a","b"], tags: ["math","stats"], examples: [Value::String("Correlation[{1,2,3},{2,4,6}]  ==> 1.0".into())]),
        tool_spec!("Covariance", summary: "Covariance of two lists (population)", params: ["a","b"], tags: ["math","stats"], examples: [Value::String("Covariance[{1,2,3},{2,4,6}]  ==> 2.0".into())]),
        tool_spec!("Skewness", summary: "Skewness (population moment)", params: ["data"], tags: ["math","stats"], examples: [Value::String("Skewness[{1,2,3}]".into())]),
        tool_spec!("Kurtosis", summary: "Kurtosis (population moment)", params: ["data"], tags: ["math","stats"], examples: [Value::String("Kurtosis[{1,2,3}]".into())]),
        tool_spec!("SeedRandom", summary: "Seed deterministic RNG for this evaluator", params: ["seed?"], tags: ["random"], examples: [Value::String("SeedRandom[1]  ==> True".into())]),
        tool_spec!("RandomInteger", summary: "Random integer; support {min,max}", params: ["spec?"], tags: ["random"], examples: [Value::String("SeedRandom[1]; RandomInteger[{1,10}]".into())]),
        tool_spec!("RandomReal", summary: "Random real; support {min,max}", params: ["spec?"], tags: ["random"], examples: [Value::String("SeedRandom[1]; RandomReal[{0.0,1.0}]".into())]),
        tool_spec!("Normal", summary: "Normal distribution head", params: ["mu","sigma"], tags: ["stats","dist"]),
        tool_spec!("Bernoulli", summary: "Bernoulli distribution head", params: ["p"], tags: ["stats","dist"]),
        tool_spec!("BinomialDistribution", summary: "Binomial distribution head", params: ["n","p"], tags: ["stats","dist"]),
        tool_spec!("Poisson", summary: "Poisson distribution head", params: ["lambda"], tags: ["stats","dist"]),
        tool_spec!("Exponential", summary: "Exponential distribution head", params: ["lambda"], tags: ["stats","dist"]),
        tool_spec!("Gamma", summary: "Gamma distribution head (shape k, scale θ)", params: ["k","theta"], tags: ["stats","dist"]),
        tool_spec!("PDF", summary: "Probability density/mass function", params: ["dist","x"], tags: ["stats","dist"]),
        tool_spec!("CDF", summary: "Cumulative distribution function", params: ["dist","x"], tags: ["stats","dist"]),
        tool_spec!("RandomVariate", summary: "Sample from distribution", params: ["dist","n?"], tags: ["stats","random"], examples: [Value::String("RandomVariate[Normal[0,1], 3]".into())]),
        tool_spec!("Abs", summary: "Absolute value", params: ["x"], tags: ["math"], input_schema: Value::Assoc(HashMap::from([
            ("type".to_string(), Value::String("object".into())),
            ("properties".to_string(), Value::Assoc(HashMap::from([(String::from("x"), Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("number")))])))]))),
            ("required".to_string(), Value::List(vec![Value::String("x".into())])),
        ])), output_schema: Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("number")))])), examples: [
            Value::Assoc(HashMap::from([
                ("args".to_string(), Value::Assoc(HashMap::from([("x".to_string(), Value::Integer(-2))]))),
                ("result".to_string(), Value::Integer(2)),
            ]))
        ]),
        tool_spec!("Min", summary: "Minimum of values or list", params: ["args"], tags: ["math"], input_schema: Value::Assoc(HashMap::from([
            ("type".to_string(), Value::String("object".into())),
            ("properties".to_string(), Value::Assoc(HashMap::from([
                ("args".to_string(), Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("array")))]))),
            ]))),
        ]))),
        tool_spec!("Max", summary: "Maximum of values or list", params: ["args"], tags: ["math"], input_schema: Value::Assoc(HashMap::from([
            ("type".to_string(), Value::String("object".into())),
            ("properties".to_string(), Value::Assoc(HashMap::from([
                ("args".to_string(), Value::Assoc(HashMap::from([(String::from("type"), Value::String(String::from("array")))]))),
            ]))),
        ]))),
    ]);
}

/// Conditionally register math functions based on `pred`.
pub fn register_math_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    register_if(
        ev,
        pred,
        "Plus",
        plus as NativeFn,
        Attributes::LISTABLE | Attributes::FLAT | Attributes::ORDERLESS | Attributes::ONE_IDENTITY,
    );
    register_if(
        ev,
        pred,
        "Times",
        times as NativeFn,
        Attributes::LISTABLE | Attributes::FLAT | Attributes::ORDERLESS | Attributes::ONE_IDENTITY,
    );
    register_if(ev, pred, "Minus", minus as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Divide", divide as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Power", power as NativeFn, Attributes::empty());
    register_if(ev, pred, "Abs", abs_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Min", min_fn as NativeFn, Attributes::FLAT | Attributes::ORDERLESS);
    register_if(ev, pred, "Max", max_fn as NativeFn, Attributes::FLAT | Attributes::ORDERLESS);
    register_if(ev, pred, "Floor", floor_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Ceiling", ceiling_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Round", round_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Trunc", trunc_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Mod", mod_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Quotient", quotient_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Remainder", remainder_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "DivMod", divmod_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Sqrt", sqrt_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Exp", exp_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Log", log_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Sin", sin_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Cos", cos_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Tanh", tanh_fn as NativeFn, Attributes::LISTABLE);
    // Internal aliases for dispatch fallbacks
    register_if(ev, pred, "__MathExp", exp_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "__MathLog", log_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "__MathSqrt", sqrt_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "__MathSin", sin_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "__MathCos", cos_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "__MathTanh", tanh_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "__MathPower", power as NativeFn, Attributes::empty());
    register_if(ev, pred, "__MathClip", clip_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Tan", tan_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "ASin", asin_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "ACos", acos_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "ATan", atan_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "ATan2", atan2_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "NthRoot", nthroot_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Total", total_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Mean", mean_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Median", median_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Variance", variance_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "StandardDeviation", stddev_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Quantile", quantile_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Percentile", percentile_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Mode", mode_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Correlation", correlation_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Covariance", covariance_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Skewness", skewness_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Kurtosis", kurtosis_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "GCD", gcd_fn as NativeFn, Attributes::FLAT | Attributes::ORDERLESS);
    register_if(ev, pred, "LCM", lcm_fn as NativeFn, Attributes::FLAT | Attributes::ORDERLESS);
    register_if(ev, pred, "Factorial", factorial_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Binomial", binomial_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "PrimeFactors", prime_factors_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "EulerPhi", euler_phi_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "MobiusMu", mobius_mu_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "PowerMod", power_mod_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "ExtendedGCD", extended_gcd_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "ModInverse", mod_inverse_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "ChineseRemainder", chinese_remainder_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "DividesQ", divides_q_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "CoprimeQ", coprime_q_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "PrimeQ", prime_q_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "NextPrime", next_prime_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "FactorInteger", factor_integer_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Permutations", permutations_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Combinations", combinations_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "PermutationsCount", permutations_count_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "CombinationsCount", combinations_count_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "ToDegrees", to_degrees_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "ToRadians", to_radians_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Clip", clip_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Signum", signum_fn as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "SeedRandom", seed_random_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "RandomInteger", random_integer_fn as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "RandomReal", random_real_fn as NativeFn, Attributes::HOLD_ALL);
}

// ----- Simple RNG backed by evaluator env -----
fn rng_next_u64(ev: &mut Evaluator) -> u64 {
    let state_key = "__rng_state";
    let mut x: u64 = match ev.get_env(state_key) { Some(lyra_core::value::Value::Integer(n)) => n as u64, _ => {
        let seed = (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(1) as u64) | 1;
        ev.set_env(state_key, lyra_core::value::Value::Integer(seed as i64)); seed }
    };
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27; let r = x.wrapping_mul(2685821657736338717);
    ev.set_env(state_key, lyra_core::value::Value::Integer(x as i64));
    r
}
fn rng_uniform01(ev: &mut Evaluator) -> f64 { (rng_next_u64(ev) as f64) / (u64::MAX as f64 + 1.0) }

fn seed_random_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let state_key = "__rng_state";
    let seed: u64 = match args.as_slice() {
        [] => 1,
        [Value::Integer(n)] => (*n as u64) | 1,
        [other] => return Value::Expr { head: Box::new(Value::Symbol("SeedRandom".into())), args: vec![ev.eval(other.clone())] },
        _ => return Value::Expr { head: Box::new(Value::Symbol("SeedRandom".into())), args },
    };
    ev.set_env(state_key, Value::Integer(seed as i64));
    Value::Boolean(true)
}

fn random_integer_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [] => {
            let n = (rng_next_u64(ev) % (i64::MAX as u64)) as i64;
            Value::Integer(n)
        }
        [Value::Integer(max)] => {
            let m = *max; if m <= 0 { return Value::Integer(0); }
            let r = (rng_uniform01(ev) * ((m + 1) as f64)) as i64; Value::Integer(r.min(m))
        }
        [Value::List(spec)] if spec.len() == 2 => {
            let a = ev.eval(spec[0].clone()); let b = ev.eval(spec[1].clone());
            if let (Value::Integer(min), Value::Integer(max)) = (a, b) {
                if max < min { return Value::Integer(min); }
                let span = (max - min + 1) as f64; let r = (rng_uniform01(ev) * span) as i64; Value::Integer(min + r)
            } else {
                Value::Expr { head: Box::new(Value::Symbol("RandomInteger".into())), args }
            }
        }
        [other] => Value::Expr { head: Box::new(Value::Symbol("RandomInteger".into())), args: vec![ev.eval(other.clone())] },
        _ => Value::Expr { head: Box::new(Value::Symbol("RandomInteger".into())), args },
    }
}

fn random_real_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [] => Value::Real(rng_uniform01(ev)),
        [Value::List(spec)] if spec.len() == 2 => {
            let a = ev.eval(spec[0].clone()); let b = ev.eval(spec[1].clone());
            match (a, b) {
                (Value::Integer(min), Value::Integer(max)) => {
                    let u = rng_uniform01(ev);
                    Value::Real((min as f64) + u * ((max - min) as f64))
                }
                (Value::Real(min), Value::Real(max)) => {
                    let u = rng_uniform01(ev);
                    Value::Real(min + u * (max - min))
                }
                (Value::Integer(min), Value::Real(max)) => { let u = rng_uniform01(ev); Value::Real((min as f64) + u * (max - (min as f64))) }
                (Value::Real(min), Value::Integer(max)) => { let u = rng_uniform01(ev); Value::Real(min + u * ((max as f64) - min)) }
                _ => Value::Expr { head: Box::new(Value::Symbol("RandomReal".into())), args },
            }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("RandomReal".into())), args },
    }
}

// -------- Tier1 Distributions (stubs) --------
fn dist_head(name: &str, args: Vec<Value>) -> Value { Value::Expr { head: Box::new(Value::Symbol(name.into())), args } }

fn dist_normal(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Canonicalize when possible: evaluate args, coerce to reals, validate sigma>0
    match args.as_slice() {
        [a, b] => {
            let mu_v = ev.eval(a.clone());
            let sigma_v = ev.eval(b.clone());
            match (super_num_to_f64(&mu_v), super_num_to_f64(&sigma_v)) {
                (Some(mu), Some(sigma)) if sigma > 0.0 => {
                    return Value::Expr {
                        head: Box::new(Value::Symbol("Normal".into())),
                        args: vec![Value::Real(mu), Value::Real(sigma)],
                    }
                }
                _ => {}
            }
            // Fallback to inert head with evaluated args
            Value::Expr { head: Box::new(Value::Symbol("Normal".into())), args: vec![mu_v, sigma_v] }
        }
        _ => dist_head("Normal", args),
    }
}
fn dist_bernoulli(_ev: &mut Evaluator, args: Vec<Value>) -> Value { dist_head("Bernoulli", args) }
fn dist_binomial(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Canonical head name is BinomialDistribution; evaluate and normalize when possible
    match args.as_slice() {
        [n_v, p_v] => {
            let n_e = ev.eval(n_v.clone());
            let p_e = ev.eval(p_v.clone());
            let n_ok = matches!(n_e, Value::Integer(k) if k >= 0);
            let p_ok = super_num_to_f64(&p_e).map(|p| (0.0..=1.0).contains(&p)).unwrap_or(false);
            let args_canon = if n_ok && p_ok {
                let n = if let Value::Integer(k) = n_e { k } else { 0 } as i64;
                let p = super_num_to_f64(&p_e).unwrap();
                vec![Value::Integer(n), Value::Real(p)]
            } else {
                vec![n_e, p_e]
            };
            Value::Expr { head: Box::new(Value::Symbol("BinomialDistribution".into())), args: args_canon }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("BinomialDistribution".into())), args },
    }
}
fn dist_poisson(_ev: &mut Evaluator, args: Vec<Value>) -> Value { dist_head("Poisson", args) }
fn dist_exponential(_ev: &mut Evaluator, args: Vec<Value>) -> Value { dist_head("Exponential", args) }
fn dist_gamma(_ev: &mut Evaluator, args: Vec<Value>) -> Value { dist_head("Gamma", args) }

fn parse_normal(ev: &mut Evaluator, v: Value) -> Option<(f64, f64)> {
    match ev.eval(v) {
        Value::Expr { head, args } => {
            if let Value::Symbol(h) = *head {
                if h == "Normal" && (args.len() == 2) {
                    let mu = match ev.eval(args[0].clone()) { x => super_num_to_f64(&x)? };
                    let sigma = match ev.eval(args[1].clone()) { x => super_num_to_f64(&x)? };
                    if sigma > 0.0 { return Some((mu, sigma)); }
                }
            }
            None
        }
        _ => None,
    }
}

fn parse_bernoulli(ev: &mut Evaluator, v: Value) -> Option<f64> {
    match ev.eval(v) {
        Value::Expr { head, args } => {
            if let Value::Symbol(h) = *head {
                if h == "Bernoulli" && args.len() == 1 {
                    let p = super_num_to_f64(&ev.eval(args[0].clone()))?;
                    if (0.0..=1.0).contains(&p) { return Some(p); }
                }
            }
            None
        }
        _ => None,
    }
}

fn parse_binomial(ev: &mut Evaluator, v: Value) -> Option<(usize, f64)> {
    match ev.eval(v) {
        Value::Expr { head, args } => {
            if let Value::Symbol(h) = *head {
                if (h == "BinomialDistribution" || h == "Binomial") && args.len() == 2 {
                    let n = match ev.eval(args[0].clone()) { Value::Integer(k) if k >= 0 => k as usize, _ => return None };
                    let p = super_num_to_f64(&ev.eval(args[1].clone()))?;
                    if (0.0..=1.0).contains(&p) { return Some((n, p)); }
                }
            }
            None
        }
        _ => None,
    }
}

fn parse_poisson(ev: &mut Evaluator, v: Value) -> Option<f64> {
    match ev.eval(v) {
        Value::Expr { head, args } => {
            if let Value::Symbol(h) = *head {
                if h == "Poisson" && args.len() == 1 {
                    let lambda = super_num_to_f64(&ev.eval(args[0].clone()))?;
                    if lambda > 0.0 { return Some(lambda); }
                }
            }
            None
        }
        _ => None,
    }
}

fn parse_exponential(ev: &mut Evaluator, v: Value) -> Option<f64> {
    match ev.eval(v) {
        Value::Expr { head, args } => {
            if let Value::Symbol(h) = *head {
                if h == "Exponential" && args.len() == 1 {
                    let lambda = super_num_to_f64(&ev.eval(args[0].clone()))?;
                    if lambda > 0.0 { return Some(lambda); }
                }
            }
            None
        }
        _ => None,
    }
}

fn parse_gamma(ev: &mut Evaluator, v: Value) -> Option<(f64, f64)> {
    match ev.eval(v) {
        Value::Expr { head, args } => {
            if let Value::Symbol(h) = *head {
                if h == "Gamma" && args.len() == 2 {
                    let k = super_num_to_f64(&ev.eval(args[0].clone()))?; // shape
                    let theta = super_num_to_f64(&ev.eval(args[1].clone()))?; // scale
                    if k > 0.0 && theta > 0.0 { return Some((k, theta)); }
                }
            }
            None
        }
        _ => None,
    }
}

fn super_num_to_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Integer(n) => Some(*n as f64),
        Value::Real(x) => Some(*x),
        Value::Rational { num, den } => if *den != 0 { Some((*num as f64)/(*den as f64)) } else { None },
        Value::BigReal(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

fn normal_pdf(mu: f64, sigma: f64, x: f64) -> f64 {
    let z = (x - mu) / sigma;
    (1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt())) * (-0.5 * z * z).exp()
}

fn erf_approx(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1) * t * (-x*x).exp();
    sign * y
}

fn normal_cdf(mu: f64, sigma: f64, x: f64) -> f64 {
    let z = (x - mu) / (sigma * std::f64::consts::SQRT_2);
    0.5 * (1.0 + erf_approx(z))
}

fn pdf_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return dist_head("PDF", args); }
    // Normal
    if let Some((mu, sigma)) = parse_normal(ev, args[0].clone()) {
        match ev.eval(args[1].clone()) {
            v => match super_num_to_f64(&v) {
                Some(x) => return Value::Real(normal_pdf(mu, sigma, x)),
                None => return dist_head("PDF", vec![Value::Expr { head: Box::new(Value::Symbol("Normal".into())), args: vec![Value::Real(mu), Value::Real(sigma)] }, v]),
            },
        }
    }
    // Bernoulli
    if let Some(p) = parse_bernoulli(ev, args[0].clone()) {
        let xv = ev.eval(args[1].clone());
        let x = match xv { Value::Integer(k) => k, Value::Real(r) => r.round() as i64, _ => -1 };
        let pmf = if x == 0 { 1.0 - p } else if x == 1 { p } else { 0.0 };
        return Value::Real(pmf);
    }
    // Binomial
    if let Some((n, p)) = parse_binomial(ev, args[0].clone()) {
        let xr = match super_num_to_f64(&ev.eval(args[1].clone())) { Some(x) => x, None => return dist_head("PDF", args) };
        let k = xr.round() as i64;
        if k < 0 || (k as usize) > n { return Value::Real(0.0); }
        let k = k as usize;
        let pmf = binomial_pmf(n, k, p);
        return Value::Real(pmf);
    }
    dist_head("PDF", args)
}

fn cdf_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return dist_head("CDF", args); }
    // Normal
    if let Some((mu, sigma)) = parse_normal(ev, args[0].clone()) {
        match ev.eval(args[1].clone()) {
            v => match super_num_to_f64(&v) {
                Some(x) => return Value::Real(normal_cdf(mu, sigma, x)),
                None => return dist_head("CDF", vec![Value::Expr { head: Box::new(Value::Symbol("Normal".into())), args: vec![Value::Real(mu), Value::Real(sigma)] }, v]),
            },
        }
    }
    // Bernoulli
    if let Some(p) = parse_bernoulli(ev, args[0].clone()) {
        let x = match super_num_to_f64(&ev.eval(args[1].clone())) { Some(x) => x, None => return dist_head("CDF", args) };
        if x < 0.0 { return Value::Real(0.0); }
        if x < 1.0 { return Value::Real(1.0 - p); }
        return Value::Real(1.0);
    }
    // Binomial
    if let Some((n, p)) = parse_binomial(ev, args[0].clone()) {
        let x = match super_num_to_f64(&ev.eval(args[1].clone())) { Some(x) => x, None => return dist_head("CDF", args) };
        if x < 0.0 { return Value::Real(0.0); }
        if x >= n as f64 { return Value::Real(1.0); }
        let kmax = x.floor() as usize;
        let mut c = 0.0f64;
        for k in 0..=kmax { c += binomial_pmf(n, k, p); }
        return Value::Real(c.min(1.0));
    }
    // Poisson
    if let Some(lambda) = parse_poisson(ev, args[0].clone()) {
        let x = match super_num_to_f64(&ev.eval(args[1].clone())) { Some(x) => x, None => return dist_head("CDF", args) };
        if x < 0.0 { return Value::Real(0.0); }
        let kmax = x.floor() as usize;
        let mut c = 0.0f64;
        for k in 0..=kmax { c += poisson_pmf(lambda, k); }
        return Value::Real(c.min(1.0));
    }
    // Exponential
    if let Some(lambda) = parse_exponential(ev, args[0].clone()) {
        let x = match super_num_to_f64(&ev.eval(args[1].clone())) { Some(x) => x, None => return dist_head("CDF", args) };
        if x < 0.0 { return Value::Real(0.0); }
        return Value::Real(1.0 - (-lambda * x).exp());
    }
    // Gamma
    if let Some((k, theta)) = parse_gamma(ev, args[0].clone()) {
        let x = match super_num_to_f64(&ev.eval(args[1].clone())) { Some(x) => x, None => return dist_head("CDF", args) };
        if x <= 0.0 { return Value::Real(0.0); }
        let a = k; let z = x / theta;
        let p = lower_incomplete_gamma_regularized(a, z);
        return Value::Real(p);
    }
    dist_head("CDF", args)
}

fn random_variate_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [dist] => {
            if let Some((mu, sigma)) = parse_normal(ev, dist.clone()) {
                return Value::Real(normal_sample(ev, mu, sigma));
            }
            if let Some(p) = parse_bernoulli(ev, dist.clone()) {
                let u = rng_uniform01(ev); return Value::Integer(if u < p { 1 } else { 0 });
            }
            if let Some((n, p)) = parse_binomial(ev, dist.clone()) {
                return Value::Integer(binomial_sample(ev, n, p) as i64);
            }
            if let Some(lambda) = parse_poisson(ev, dist.clone()) {
                return Value::Integer(poisson_sample(ev, lambda) as i64);
            }
            if let Some(lambda) = parse_exponential(ev, dist.clone()) {
                return Value::Real(exponential_sample(ev, lambda));
            }
            if let Some((k, theta)) = parse_gamma(ev, dist.clone()) {
                return Value::Real(gamma_sample(ev, k, theta));
            }
            dist_head("RandomVariate", args)
        }
        [dist, Value::Integer(nn)] => {
            let k = (*nn).max(0) as usize;
            if let Some((mu, sigma)) = parse_normal(ev, dist.clone()) {
                let mut out = Vec::with_capacity(k);
                for _ in 0..k { out.push(Value::Real(normal_sample(ev, mu, sigma))); }
                return Value::List(out);
            }
            if let Some(p) = parse_bernoulli(ev, dist.clone()) {
                let mut out: Vec<Value> = Vec::with_capacity(k);
                for _ in 0..k { let u = rng_uniform01(ev); out.push(Value::Integer(if u < p { 1 } else { 0 })); }
                return Value::List(out);
            }
            if let Some((n, p)) = parse_binomial(ev, dist.clone()) {
                let mut out: Vec<Value> = Vec::with_capacity(k);
                for _ in 0..k { out.push(Value::Integer(binomial_sample(ev, n, p) as i64)); }
                return Value::List(out);
            }
            if let Some(lambda) = parse_poisson(ev, dist.clone()) {
                let mut out: Vec<Value> = Vec::with_capacity(k);
                for _ in 0..k { out.push(Value::Integer(poisson_sample(ev, lambda) as i64)); }
                return Value::List(out);
            }
            if let Some(lambda) = parse_exponential(ev, dist.clone()) {
                let mut out: Vec<Value> = Vec::with_capacity(k);
                for _ in 0..k { out.push(Value::Real(exponential_sample(ev, lambda))); }
                return Value::List(out);
            }
            if let Some((kk, theta)) = parse_gamma(ev, dist.clone()) {
                let mut out: Vec<Value> = Vec::with_capacity(k);
                for _ in 0..k { out.push(Value::Real(gamma_sample(ev, kk, theta))); }
                return Value::List(out);
            }
            dist_head("RandomVariate", args)
        }
        _ => dist_head("RandomVariate", args),
    }
}

fn normal_sample(ev: &mut Evaluator, mu: f64, sigma: f64) -> f64 {
    // Box-Muller
    let mut u1 = rng_uniform01(ev); let u2 = rng_uniform01(ev);
    // Avoid log(0)
    if u1 <= 1e-12 { u1 = 1e-12; }
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    let z = r * theta.cos();
    mu + sigma * z
}

fn binomial_coeff(n: usize, k: usize) -> f64 {
    if k > n { return 0.0; }
    let k = k.min(n - k);
    let mut c = 1.0f64;
    for i in 1..=k { c *= (n - k + i) as f64; c /= i as f64; }
    c
}

fn binomial_pmf(n: usize, k: usize, p: f64) -> f64 {
    if p < 0.0 || p > 1.0 { return 0.0; }
    if k > n { return 0.0; }
    let c = binomial_coeff(n, k);
    c * p.powi(k as i32) * (1.0 - p).powi((n - k) as i32)
}

fn binomial_sample(ev: &mut Evaluator, n: usize, p: f64) -> usize {
    let mut cnt = 0usize;
    for _ in 0..n { if rng_uniform01(ev) < p { cnt += 1; } }
    cnt
}

fn factorial_u64(k: usize) -> f64 { (1..=k).fold(1.0, |acc, i| acc * (i as f64)) }

fn poisson_pmf(lambda: f64, k: usize) -> f64 {
    if lambda <= 0.0 { return 0.0; }
    let kf = factorial_u64(k);
    ((-lambda).exp()) * lambda.powi(k as i32) / kf
}

fn poisson_sample(ev: &mut Evaluator, lambda: f64) -> usize {
    // Knuth's algorithm
    let l = (-lambda).exp();
    let mut p = 1.0; let mut k = 0usize;
    loop {
        k += 1;
        p *= rng_uniform01(ev);
        if p <= l { break; }
    }
    k - 1
}

fn exponential_sample(ev: &mut Evaluator, lambda: f64) -> f64 {
    let mut u = rng_uniform01(ev);
    if u <= 1e-12 { u = 1e-12; }
    -u.ln() / lambda
}

// ---- Gamma helpers ----
fn ln_gamma(z: f64) -> f64 {
    // Lanczos approximation, g=7, n=9 coefficients
    let p: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if z < 0.5 {
        return std::f64::consts::PI.ln() - (std::f64::consts::PI * z).sin().ln() - ln_gamma(1.0 - z);
    }
    let z = z - 1.0;
    let x0 = p[0];
    let mut x = x0;
    for i in 1..9 { x += p[i] / (z + i as f64); }
    let t = z + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
}

fn lower_incomplete_gamma_series(a: f64, x: f64) -> f64 {
    // Series expansion for P(a,x) * Gamma(a)
    let mut sum = 1.0 / a;
    let mut term = sum;
    let mut n = 1.0;
    while term.abs() > 1e-12 && n < 10000.0 {
        term *= x / (a + n);
        sum += term;
        n += 1.0;
    }
    sum * x.powf(a) * (-x).exp()
}

fn lower_incomplete_gamma_cf(a: f64, x: f64) -> f64 {
    // Continued fraction for Q(a,x) = Gamma(a,x)/Gamma(a)
    // We compute Q and then return Gamma(a) * (1-Q)
    let max_iter = 200;
    let eps = 1e-12;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / 1e-30;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=max_iter {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b; if d.abs() < 1e-30 { d = 1e-30; }
        c = b + an / c; if c.abs() < 1e-30 { c = 1e-30; }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < eps { break; }
    }
    ((-x).exp()) * h
}

fn lower_incomplete_gamma_regularized(a: f64, x: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x < a + 1.0 {
        let s = lower_incomplete_gamma_series(a, x);
        let p = s / (ln_gamma(a).exp());
        p.max(0.0).min(1.0)
    } else {
        let q = lower_incomplete_gamma_cf(a, x);
        (1.0 - q).max(0.0).min(1.0)
    }
}


fn gamma_sample(ev: &mut Evaluator, k: f64, theta: f64) -> f64 {
    let a = k;
    if a < 1.0 {
        let u = rng_uniform01(ev);
        return gamma_sample(ev, a + 1.0, theta) * u.powf(1.0 / a);
    }
    // Marsaglia and Tsang's method for a>=1
    let d = a - 1.0 / 3.0;
    let c = (1.0 / (9.0 * d)).sqrt();
    loop {
        let x: f64;
        let v: f64;
        // Standard normal via Box-Muller
        let z = {
            let u1 = rng_uniform01(ev).max(1e-12);
            let u2 = rng_uniform01(ev);
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            r * theta.cos()
        };
        x = 1.0 + c * z;
        if x <= 0.0 { continue; }
        v = x * x * x;
        let u = rng_uniform01(ev);
        if u < 1.0 - 0.0331 * z * z * z * z { return (d * v) * theta; }
        if (u.ln()) < 0.5 * z * z + d * (1.0 - v + v.ln()) { return (d * v) * theta; }
    }
}


type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.abs()
}

fn reduce_rat(num: i64, den: i64) -> (i64, i64) {
    if den == 0 {
        return (num, den);
    }
    let mut n = num;
    let mut d = den;
    if d < 0 {
        n = -n;
        d = -d;
    }
    let g = gcd(n, d);
    (n / g, d / g)
}

fn rat_value(num: i64, den: i64) -> Value {
    let (n, d) = reduce_rat(num, den);
    if d == 1 {
        Value::Integer(n)
    } else {
        Value::Rational { num: n, den: d }
    }
}

fn format_float(v: f64) -> String {
    // Trim trailing zeros and decimal point
    let s = format!("{:.12}", v);
    let s = s.trim_end_matches('0').trim_end_matches('.').to_string();
    if s.is_empty() {
        "0".into()
    } else {
        s
    }
}

#[cfg(feature = "big-real-rug")]
fn bigreal_binop(ax: &str, by: &str, op: fn(Float, Float) -> Float) -> Option<String> {
    let af = ax.parse::<f64>().ok()?;
    let bf = by.parse::<f64>().ok()?;
    let r = op(Float::with_val(128, af), Float::with_val(128, bf));
    Some(r.to_string())
}

#[cfg(not(feature = "big-real-rug"))]
fn bigreal_binop(ax: &str, by: &str, op: fn(f64, f64) -> f64) -> Option<String> {
    let a = ax.parse::<f64>().ok()?;
    let b = by.parse::<f64>().ok()?;
    Some(format_float(op(a, b)))
}

fn add_numeric(a: Value, b: Value) -> Option<Value> {
    match (a, b) {
        (Value::Integer(x), Value::Integer(y)) => Some(Value::Integer(x + y)),
        (Value::Real(x), Value::Real(y)) => Some(Value::Real(x + y)),
        (Value::Integer(x), Value::Real(y)) => Some(Value::Real((x as f64) + y)),
        (Value::Real(x), Value::Integer(y)) => Some(Value::Real(x + (y as f64))),
        (Value::BigReal(ax), Value::BigReal(by)) => {
            let s = bigreal_binop(&ax, &by, |a, b| a + b)?;
            Some(Value::BigReal(s))
        }
        (Value::BigReal(ax), other) | (other, Value::BigReal(ax)) => {
            let xf = ax.parse::<f64>().ok()?;
            let yf = match other {
                Value::Integer(n) => n as f64,
                Value::Real(r) => r,
                Value::Rational { num, den } => (num as f64) / (den as f64),
                _ => return None,
            };
            Some(Value::BigReal(format_float(xf + yf)))
        }
        (
            Value::PackedArray { shape: s1, data: d1 },
            Value::PackedArray { shape: s2, data: d2 },
        ) => {
            if s1 == s2 && d1.len() == d2.len() {
                let data: Vec<f64> = d1.iter().zip(d2.iter()).map(|(x, y)| x + y).collect();
                Some(Value::PackedArray { shape: s1, data })
            } else {
                None
            }
        }
        (Value::PackedArray { shape, data }, other)
        | (other, Value::PackedArray { shape, data }) => {
            fn to_f64_scalar(v: &Value) -> Option<f64> {
                match v {
                    Value::Integer(n) => Some(*n as f64),
                    Value::Real(x) => Some(*x),
                    Value::Rational { num, den } => {
                        if *den != 0 {
                            Some((*num as f64) / (*den as f64))
                        } else {
                            None
                        }
                    }
                    Value::BigReal(s) => s.parse::<f64>().ok(),
                    _ => None,
                }
            }
            if let Some(s) = to_f64_scalar(&other) {
                let mut out = data.clone();
                for x in &mut out {
                    *x += s;
                }
                Some(Value::PackedArray { shape, data: out })
            } else {
                None
            }
        }
        (Value::Rational { num: n1, den: d1 }, Value::Rational { num: n2, den: d2 }) => {
            Some(rat_value(n1 * d2 + n2 * d1, d1 * d2))
        }
        (Value::Integer(x), Value::Rational { num, den })
        | (Value::Rational { num, den }, Value::Integer(x)) => Some(rat_value(num + x * den, den)),
        (Value::Real(x), Value::Rational { num, den })
        | (Value::Rational { num, den }, Value::Real(x)) => {
            Some(Value::Real(x + (num as f64) / (den as f64)))
        }
        (Value::Complex { re: ar, im: ai }, Value::Complex { re: br, im: bi }) => {
            let rr = add_numeric((*ar).clone(), (*br).clone())?;
            let ri = add_numeric((*ai).clone(), (*bi).clone())?;
            Some(Value::Complex { re: Box::new(rr), im: Box::new(ri) })
        }
        (Value::Complex { re, im }, other) | (other, Value::Complex { re, im }) => {
            let rr = add_numeric((*re).clone(), other)?;
            Some(Value::Complex { re: Box::new(rr), im })
        }
        _ => None,
    }
}

fn mul_numeric(a: Value, b: Value) -> Option<Value> {
    match (a, b) {
        (Value::Integer(x), Value::Integer(y)) => Some(Value::Integer(x * y)),
        (Value::Real(x), Value::Real(y)) => Some(Value::Real(x * y)),
        (Value::Integer(x), Value::Real(y)) => Some(Value::Real((x as f64) * y)),
        (Value::Real(x), Value::Integer(y)) => Some(Value::Real(x * (y as f64))),
        (Value::BigReal(ax), Value::BigReal(by)) => {
            let s = bigreal_binop(&ax, &by, |a, b| a * b)?;
            Some(Value::BigReal(s))
        }
        (Value::BigReal(ax), other) | (other, Value::BigReal(ax)) => {
            let xf = ax.parse::<f64>().ok()?;
            let yf = match other {
                Value::Integer(n) => n as f64,
                Value::Real(r) => r,
                Value::Rational { num, den } => (num as f64) / (den as f64),
                _ => return None,
            };
            Some(Value::BigReal(format_float(xf * yf)))
        }
        (
            Value::PackedArray { shape: s1, data: d1 },
            Value::PackedArray { shape: s2, data: d2 },
        ) => {
            if s1 == s2 && d1.len() == d2.len() {
                let data: Vec<f64> = d1.iter().zip(d2.iter()).map(|(x, y)| x * y).collect();
                Some(Value::PackedArray { shape: s1, data })
            } else {
                None
            }
        }
        (Value::PackedArray { shape, data }, other)
        | (other, Value::PackedArray { shape, data }) => {
            fn to_f64_scalar(v: &Value) -> Option<f64> {
                match v {
                    Value::Integer(n) => Some(*n as f64),
                    Value::Real(x) => Some(*x),
                    Value::Rational { num, den } => {
                        if *den != 0 {
                            Some((*num as f64) / (*den as f64))
                        } else {
                            None
                        }
                    }
                    Value::BigReal(s) => s.parse::<f64>().ok(),
                    _ => None,
                }
            }
            if let Some(s) = to_f64_scalar(&other) {
                let mut out = data.clone();
                for x in &mut out {
                    *x *= s;
                }
                Some(Value::PackedArray { shape, data: out })
            } else {
                None
            }
        }
        (Value::Rational { num: n1, den: d1 }, Value::Rational { num: n2, den: d2 }) => {
            Some(rat_value(n1 * n2, d1 * d2))
        }
        (Value::Integer(x), Value::Rational { num, den })
        | (Value::Rational { num, den }, Value::Integer(x)) => Some(rat_value(num * x, den)),
        (Value::Real(x), Value::Rational { num, den })
        | (Value::Rational { num, den }, Value::Real(x)) => {
            Some(Value::Real(x * (num as f64) / (den as f64)))
        }
        (Value::Complex { re: ar, im: ai }, Value::Complex { re: br, im: bi }) => {
            // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            let ac = mul_numeric((*ar).clone(), (*br).clone())?;
            let bd = mul_numeric((*ai).clone(), (*bi).clone())?;
            let ad = mul_numeric((*ar).clone(), (*bi).clone())?;
            let bc = mul_numeric((*ai).clone(), (*br).clone())?;
            let real = add_numeric(
                ac,
                Value::Expr { head: Box::new(Value::Symbol("Minus".into())), args: vec![bd] },
            )?; // fallback unevaluated minus
            let imag = add_numeric(ad, bc)?;
            Some(Value::Complex { re: Box::new(real), im: Box::new(imag) })
        }
        (Value::Complex { re, im }, other) | (other, Value::Complex { re, im }) => {
            let ar = (*re).clone();
            let ai = (*im).clone();
            let br = other.clone();
            let bi = Value::Integer(0);
            let real = add_numeric(
                mul_numeric(ar.clone(), br.clone())?,
                Value::Expr {
                    head: Box::new(Value::Symbol("Minus".into())),
                    args: vec![mul_numeric(ai.clone(), bi.clone())?],
                },
            )?;
            let imag = add_numeric(mul_numeric(ar, bi)?, mul_numeric(ai, br)?)?;
            Some(Value::Complex { re: Box::new(real), im: Box::new(imag) })
        }
        _ => None,
    }
}

fn sub_numeric(a: Value, b: Value) -> Option<Value> {
    match (a, b) {
        (Value::Integer(x), Value::Integer(y)) => Some(Value::Integer(x - y)),
        (Value::Real(x), Value::Real(y)) => Some(Value::Real(x - y)),
        (Value::Integer(x), Value::Real(y)) => Some(Value::Real((x as f64) - y)),
        (Value::Real(x), Value::Integer(y)) => Some(Value::Real(x - (y as f64))),
        (Value::BigReal(ax), Value::BigReal(by)) => {
            let s = bigreal_binop(&ax, &by, |a, b| a - b)?;
            Some(Value::BigReal(s))
        }
        (Value::BigReal(ax), other) => {
            let xf = ax.parse::<f64>().ok()?;
            let yf = match other {
                Value::Integer(n) => n as f64,
                Value::Real(r) => r,
                Value::Rational { num, den } => (num as f64) / (den as f64),
                _ => return None,
            };
            Some(Value::BigReal(format_float(xf - yf)))
        }
        (other, Value::BigReal(by)) => {
            let xf = match other {
                Value::Integer(n) => n as f64,
                Value::Real(r) => r,
                Value::Rational { num, den } => (num as f64) / (den as f64),
                _ => return None,
            };
            let yf = by.parse::<f64>().ok()?;
            Some(Value::BigReal(format_float(xf - yf)))
        }
        (
            Value::PackedArray { shape: s1, data: d1 },
            Value::PackedArray { shape: s2, data: d2 },
        ) => {
            // broadcast-aware elementwise subtraction
            broadcast_elementwise(&s1, &d1, &s2, &d2, |x, y| x - y)
        }
        (Value::PackedArray { shape, data }, other) => {
            if let Some(s) = to_f64_scalar(&other) {
                let mut out = data.clone();
                for x in &mut out {
                    *x -= s;
                }
                Some(Value::PackedArray { shape, data: out })
            } else {
                None
            }
        }
        (other, Value::PackedArray { shape, data }) => {
            if let Some(s) = to_f64_scalar(&other) {
                let mut out = data.clone();
                for x in &mut out {
                    *x = s - *x;
                }
                Some(Value::PackedArray { shape, data: out })
            } else {
                None
            }
        }
        (Value::Rational { num: n1, den: d1 }, Value::Rational { num: n2, den: d2 }) => {
            Some(rat_value(n1 * d2 - n2 * d1, d1 * d2))
        }
        (Value::Integer(x), Value::Rational { num, den }) => Some(rat_value(x * den - num, den)),
        (Value::Rational { num, den }, Value::Integer(x)) => Some(rat_value(num - x * den, den)),
        (Value::Real(x), Value::Rational { num, den }) => {
            Some(Value::Real(x - (num as f64) / (den as f64)))
        }
        (Value::Rational { num, den }, Value::Real(x)) => {
            Some(Value::Real((num as f64) / (den as f64) - x))
        }
        (Value::Complex { re: ar, im: ai }, Value::Complex { re: br, im: bi }) => {
            let rr = sub_numeric((*ar).clone(), (*br).clone())?;
            let ri = sub_numeric((*ai).clone(), (*bi).clone())?;
            Some(Value::Complex { re: Box::new(rr), im: Box::new(ri) })
        }
        (Value::Complex { re, im }, other) => {
            let rr = sub_numeric((*re).clone(), other)?;
            Some(Value::Complex { re: Box::new(rr), im })
        }
        (other, Value::Complex { re, im }) => {
            let rr = sub_numeric(other, (*re).clone())?;
            Some(Value::Complex {
                re: Box::new(Value::Expr {
                    head: Box::new(Value::Symbol("Minus".into())),
                    args: vec![rr],
                }),
                im: Box::new(Value::Expr {
                    head: Box::new(Value::Symbol("Minus".into())),
                    args: vec![*im],
                }),
            })
        }
        _ => None,
    }
}

fn div_numeric(a: Value, b: Value) -> Option<Value> {
    match (a, b) {
        (
            Value::PackedArray { shape: s1, data: d1 },
            Value::PackedArray { shape: s2, data: d2 },
        ) => broadcast_elementwise(&s1, &d1, &s2, &d2, |x, y| x / y),
        (Value::PackedArray { shape, data }, other) => {
            if let Some(s) = to_f64_scalar(&other) {
                let mut out = data.clone();
                for x in &mut out {
                    *x /= s;
                }
                Some(Value::PackedArray { shape, data: out })
            } else {
                None
            }
        }
        (other, Value::PackedArray { shape, data }) => {
            if let Some(s) = to_f64_scalar(&other) {
                let mut out = data.clone();
                for x in &mut out {
                    *x = s / *x;
                }
                Some(Value::PackedArray { shape, data: out })
            } else {
                None
            }
        }
        (Value::Integer(x), Value::Integer(y)) => {
            if y == 0 {
                None
            } else {
                Some(rat_value(x, y))
            }
        }
        (Value::Real(x), Value::Real(y)) => Some(Value::Real(x / y)),
        (Value::Integer(x), Value::Real(y)) => Some(Value::Real((x as f64) / y)),
        (Value::Real(x), Value::Integer(y)) => Some(Value::Real(x / (y as f64))),
        (Value::BigReal(ax), Value::BigReal(by)) => {
            let s = bigreal_binop(&ax, &by, |a, b| a / b)?;
            Some(Value::BigReal(s))
        }
        (Value::BigReal(ax), other) => {
            let xf = ax.parse::<f64>().ok()?;
            let yf = match other {
                Value::Integer(n) => n as f64,
                Value::Real(r) => r,
                Value::Rational { num, den } => (num as f64) / (den as f64),
                _ => return None,
            };
            Some(Value::BigReal(format_float(xf / yf)))
        }
        (other, Value::BigReal(by)) => {
            let xf = match other {
                Value::Integer(n) => n as f64,
                Value::Real(r) => r,
                Value::Rational { num, den } => (num as f64) / (den as f64),
                _ => return None,
            };
            let yf = by.parse::<f64>().ok()?;
            Some(Value::BigReal(format_float(xf / yf)))
        }
        (Value::Rational { num: n1, den: d1 }, Value::Rational { num: n2, den: d2 }) => {
            if n2 == 0 {
                None
            } else {
                Some(rat_value(n1 * d2, d1 * n2))
            }
        }
        (Value::Integer(x), Value::Rational { num, den }) => {
            if num == 0 {
                return None;
            }
            Some(rat_value(x * den, num))
        }
        (Value::Rational { num, den }, Value::Integer(x)) => {
            if x == 0 {
                return None;
            }
            Some(rat_value(num, den * x))
        }
        (Value::Real(x), Value::Rational { num, den }) => {
            if num == 0 {
                None
            } else {
                Some(Value::Real(x / ((num as f64) / (den as f64))))
            }
        }
        (Value::Rational { num, den }, Value::Real(x)) => {
            Some(Value::Real(((num as f64) / (den as f64)) / x))
        }
        (Value::Complex { re: ar, im: ai }, Value::Integer(y)) => {
            if y == 0 {
                return None;
            }
            let denom = Value::Integer(y);
            let rr = div_numeric((*ar).clone(), denom.clone())?;
            let ri = div_numeric((*ai).clone(), denom)?;
            Some(Value::Complex { re: Box::new(rr), im: Box::new(ri) })
        }
        (Value::Complex { re: ar, im: ai }, Value::Real(y)) => {
            let denom = Value::Real(y);
            let rr = div_numeric((*ar).clone(), denom.clone())?;
            let ri = div_numeric((*ai).clone(), denom)?;
            Some(Value::Complex { re: Box::new(rr), im: Box::new(ri) })
        }
        (Value::Complex { re: ar, im: ai }, Value::Complex { re: br, im: bi }) => {
            // (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c^2 + d^2)
            let c2 = mul_numeric((*br).clone(), (*br).clone())?;
            let d2 = mul_numeric((*bi).clone(), (*bi).clone())?;
            let denom = add_numeric(c2, d2)?;
            // real numerator: ac + bd
            let ac = mul_numeric((*ar).clone(), (*br).clone())?;
            let bd = mul_numeric((*ai).clone(), (*bi).clone())?;
            let num_r = add_numeric(ac, bd)?;
            // imag numerator: bc - ad
            let bc = mul_numeric((*ai).clone(), (*br).clone())?;
            let ad = mul_numeric((*ar).clone(), (*bi).clone())?;
            let num_i = sub_numeric(bc, ad)?;
            let rr = div_numeric(num_r, denom.clone())?;
            let ri = div_numeric(num_i, denom)?;
            Some(Value::Complex { re: Box::new(rr), im: Box::new(ri) })
        }
        _ => None,
    }
}

fn to_f64_scalar(v: &Value) -> Option<f64> {
    match v {
        Value::Integer(n) => Some(*n as f64),
        Value::Real(x) => Some(*x),
        Value::Rational { num, den } => {
            if *den != 0 {
                Some((*num as f64) / (*den as f64))
            } else {
                None
            }
        }
        Value::BigReal(s) => s.parse::<f64>().ok(),
        Value::Symbol(s) => match s.as_str() {
            "Pi" => Some(std::f64::consts::PI),
            "E" => Some(std::f64::consts::E),
            "Tau" => Some(std::f64::consts::TAU),
            "Degree" => Some(std::f64::consts::PI / 180.0),
            _ => None,
        },
        Value::Expr { head, args } => {
            if let Value::Symbol(name) = &**head {
                match (name.as_str(), args.as_slice()) {
                    ("Plus", list) => {
                        let mut acc = 0.0;
                        for a in list {
                            acc += to_f64_scalar(a)?;
                        }
                        Some(acc)
                    }
                    ("Times", list) => {
                        let mut acc = 1.0;
                        for a in list {
                            acc *= to_f64_scalar(a)?;
                        }
                        Some(acc)
                    }
                    ("Minus", [a]) => Some(-to_f64_scalar(a)?),
                    ("Minus", [a, b]) => Some(to_f64_scalar(a)? - to_f64_scalar(b)?),
                    ("Divide", [a, b]) => Some(to_f64_scalar(a)? / to_f64_scalar(b)?),
                    ("Power", [a, b]) => Some(to_f64_scalar(a)?.powf(to_f64_scalar(b)?)),
                    _ => None,
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

fn broadcast_elementwise(
    s1: &Vec<usize>,
    d1: &Vec<f64>,
    s2: &Vec<usize>,
    d2: &Vec<f64>,
    op: fn(f64, f64) -> f64,
) -> Option<Value> {
    let ndim = std::cmp::max(s1.len(), s2.len());
    let mut sh1 = vec![1; ndim];
    let mut sh2 = vec![1; ndim];
    sh1[ndim - s1.len()..].clone_from_slice(&s1);
    sh2[ndim - s2.len()..].clone_from_slice(&s2);
    let mut out_shape: Vec<usize> = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let a = sh1[i];
        let b = sh2[i];
        if a == b {
            out_shape.push(a);
        } else if a == 1 {
            out_shape.push(b);
        } else if b == 1 {
            out_shape.push(a);
        } else {
            return None;
        }
    }
    let total: usize = out_shape.iter().product();
    let strides = |shape: &Vec<usize>| -> Vec<usize> {
        let mut st = vec![0; ndim];
        let mut acc = 1usize;
        for i in (0..ndim).rev() {
            st[i] = acc;
            acc *= shape[i];
        }
        st
    };
    let st1 = strides(&sh1);
    let st2 = strides(&sh2);
    let mut out = Vec::with_capacity(total);
    for idx in 0..total {
        // convert idx to multi-index
        let mut rem = idx;
        let mut off1 = 0usize;
        let mut off2 = 0usize;
        for i in 0..ndim {
            let _dim = out_shape[i];
            let coord = rem / (out_shape[i + 1..].iter().product::<usize>().max(1));
            rem %= out_shape[i + 1..].iter().product::<usize>().max(1);
            let c1 = if sh1[i] == 1 { 0 } else { coord };
            let c2 = if sh2[i] == 1 { 0 } else { coord };
            off1 += c1 * st1[i];
            off2 += c2 * st2[i];
        }
        out.push(op(d1[off1], d2[off2]));
    }
    Some(Value::PackedArray { shape: out_shape, data: out })
}

fn pow_numeric(base: Value, exp: Value) -> Option<Value> {
    match (base.clone(), exp) {
        (Value::Integer(a), Value::Integer(e)) => {
            if e >= 0 {
                Some(Value::Integer(a.pow(e as u32)))
            } else {
                let (rn, rd) = reduce_rat(1, a.pow((-e) as u32));
                Some(Value::Rational { num: rn, den: rd })
            }
        }
        (Value::Real(a), Value::Integer(e)) => Some(Value::Real(a.powi(e as i32))),
        (Value::BigReal(ax), Value::Integer(e)) => {
            let xf = ax.parse::<f64>().ok()?;
            Some(Value::BigReal(format_float(xf.powi(e as i32))))
        }
        (Value::Rational { num, den }, Value::Integer(e)) => {
            if e >= 0 {
                Some(rat_value(num.pow(e as u32), den.pow(e as u32)))
            } else {
                let p = (-e) as u32;
                Some(rat_value(den.pow(p), num.pow(p)))
            }
        }
        (Value::Complex { .. }, Value::Integer(e)) => {
            let mut k = e.abs();
            if k == 0 {
                return Some(Value::Integer(1));
            }
            // repeated multiplication (simple)
            let mut acc: Option<Value> = None;
            let mut base_opt: Option<Value> = None;
            if let (Value::Complex { .. }, _) = (base.clone(), e) {
                base_opt = Some(base.clone());
            }
            while k > 0 {
                if acc.is_none() {
                    acc = base_opt.clone();
                    k -= 1;
                    continue;
                }
                acc = Some(mul_numeric(acc.unwrap(), base_opt.clone().unwrap())?);
                k -= 1;
            }
            if e < 0 {
                div_numeric(Value::Integer(1), acc.unwrap())
            } else {
                acc
            }
        }
        _ => None,
    }
}

fn plus(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Integer(0);
    }
    let mut acc = args[0].clone();
    for i in 1..args.len() {
        let a = args[i].clone();
        match add_numeric(acc.clone(), a.clone()) {
            Some(v) => acc = v,
            None => {
                // Fall back to symbolic form preserving remaining args (with folded prefix)
                let mut rest = Vec::with_capacity(args.len() - i + 1);
                rest.push(acc);
                rest.push(a);
                rest.extend_from_slice(&args[i + 1..]);
                return Value::Expr { head: Box::new(Value::Symbol("Plus".into())), args: rest };
            }
        }
    }
    acc
}

fn times(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Integer(1);
    }
    let mut acc = args[0].clone();
    for i in 1..args.len() {
        let a = args[i].clone();
        match mul_numeric(acc.clone(), a.clone()) {
            Some(v) => acc = v,
            None => {
                // Fall back to symbolic form preserving remaining args (with folded prefix)
                let mut rest = Vec::with_capacity(args.len() - i + 1);
                rest.push(acc);
                rest.push(a);
                rest.extend_from_slice(&args[i + 1..]);
                return Value::Expr { head: Box::new(Value::Symbol("Times".into())), args: rest };
            }
        }
    }
    acc
}

fn minus(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [a, b] => sub_numeric(a.clone(), b.clone()).unwrap_or(Value::Expr {
            head: Box::new(Value::Symbol("Minus".into())),
            args: vec![a.clone(), b.clone()],
        }),
        [a] => sub_numeric(Value::Integer(0), a.clone()).unwrap_or(Value::Expr {
            head: Box::new(Value::Symbol("Minus".into())),
            args: vec![a.clone()],
        }),
        other => {
            Value::Expr { head: Box::new(Value::Symbol("Minus".into())), args: other.to_vec() }
        }
    }
}

fn divide(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [a, b] => div_numeric(a.clone(), b.clone()).unwrap_or(Value::Expr {
            head: Box::new(Value::Symbol("Divide".into())),
            args: vec![a.clone(), b.clone()],
        }),
        other => {
            Value::Expr { head: Box::new(Value::Symbol("Divide".into())), args: other.to_vec() }
        }
    }
}

fn power(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [a, b] => {
            // Tensor-aware: if either side evaluates to a PackedArray, delegate to NDPow
            let av = ev.eval(a.clone());
            let bv = ev.eval(b.clone());
            if matches!(av, Value::PackedArray { .. }) || matches!(bv, Value::PackedArray { .. }) {
                return ev.eval(Value::Expr { head: Box::new(Value::Symbol("NDPow".into())), args: vec![av, bv] });
            }
            pow_numeric(av, bv).unwrap_or(Value::Expr {
                head: Box::new(Value::Symbol("Power".into())),
                args: vec![a.clone(), b.clone()],
            })
        }
        other => Value::Expr { head: Box::new(Value::Symbol("Power".into())), args: other.to_vec() },
    }
}

fn abs_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] => Value::Integer(n.abs()),
        [Value::Real(x)] => Value::Real(x.abs()),
        [Value::Rational { num, den }] => {
            let (n, d) = reduce_rat(num.abs(), den.abs());
            Value::Rational { num: n, den: d }
        }
        [Value::Complex { re, im }] => {
            // If parts are integers, try perfect square; else compute f64 sqrt(a^2+b^2)
            match (&**re, &**im) {
                (Value::Integer(a), Value::Integer(b)) => {
                    let aa = (*a).saturating_mul(*a);
                    let bb = (*b).saturating_mul(*b);
                    let sum = aa.saturating_add(bb);
                    let rt = (sum as f64).sqrt().round() as i64;
                    if rt.saturating_mul(rt) == sum {
                        Value::Integer(rt)
                    } else {
                        Value::Real((sum as f64).sqrt())
                    }
                }
                _ => {
                    fn to_f64(v: &Value) -> Option<f64> {
                        match v {
                            Value::Integer(n) => Some(*n as f64),
                            Value::Real(x) => Some(*x),
                            Value::Rational { num, den } => {
                                if *den != 0 {
                                    Some((*num as f64) / (*den as f64))
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        }
                    }
                    if let (Some(ar), Some(ai)) = (to_f64(&re), to_f64(&im)) {
                        Value::Real(((ar * ar) + (ai * ai)).sqrt())
                    } else {
                        Value::Expr {
                            head: Box::new(Value::Symbol("Abs".into())),
                            args: args.to_vec(),
                        }
                    }
                }
            }
        }
        other => Value::Expr { head: Box::new(Value::Symbol("Abs".into())), args: other.to_vec() },
    }
}

fn tanh_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] => Value::Real((*n as f64).tanh()),
        [Value::Real(x)] => Value::Real(x.tanh()),
        [Value::Rational { num, den }] => {
            let x = (*num as f64) / (*den as f64);
            Value::Real(x.tanh())
        }
        other => Value::Expr { head: Box::new(Value::Symbol("Tanh".into())), args: other.to_vec() },
    }
}

fn min_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() == 1 {
        match ev.eval(args[0].clone()) {
            Value::List(items) => return min_over_iter(items.into_iter()),
            other => {
                return Value::Expr {
                    head: Box::new(Value::Symbol("Min".into())),
                    args: vec![other],
                }
            }
        }
    }
    let evald: Vec<Value> = args.into_iter().map(|a| ev.eval(a)).collect();
    min_over_iter(evald.into_iter())
}

fn max_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() == 1 {
        match ev.eval(args[0].clone()) {
            Value::List(items) => return max_over_iter(items.into_iter()),
            other => {
                return Value::Expr {
                    head: Box::new(Value::Symbol("Max".into())),
                    args: vec![other],
                }
            }
        }
    }
    let evald: Vec<Value> = args.into_iter().map(|a| ev.eval(a)).collect();
    max_over_iter(evald.into_iter())
}

fn min_over_iter<I: Iterator<Item = Value>>(iter: I) -> Value {
    let mut have = false;
    let mut use_real = false;
    let mut cur_i: i64 = 0;
    let mut cur_f: f64 = 0.0;
    for v in iter {
        match v {
            Value::Integer(n) => {
                if !have {
                    have = true;
                    cur_i = n;
                    cur_f = n as f64;
                } else {
                    if use_real {
                        if (n as f64) < cur_f {
                            cur_f = n as f64;
                        }
                    } else if n < cur_i {
                        cur_i = n;
                    }
                }
            }
            Value::Real(x) => {
                if !have {
                    have = true;
                    cur_f = x;
                    cur_i = x as i64;
                    use_real = true;
                } else {
                    use_real = true;
                    if x < cur_f {
                        cur_f = x;
                    }
                }
            }
            other => {
                return Value::Expr {
                    head: Box::new(Value::Symbol("Min".into())),
                    args: vec![other],
                }
            }
        }
    }
    if !have {
        Value::Expr { head: Box::new(Value::Symbol("Min".into())), args: vec![] }
    } else if use_real {
        Value::Real(cur_f)
    } else {
        Value::Integer(cur_i)
    }
}

fn max_over_iter<I: Iterator<Item = Value>>(iter: I) -> Value {
    let mut have = false;
    let mut use_real = false;
    let mut cur_i: i64 = 0;
    let mut cur_f: f64 = 0.0;
    for v in iter {
        match v {
            Value::Integer(n) => {
                if !have {
                    have = true;
                    cur_i = n;
                    cur_f = n as f64;
                } else {
                    if use_real {
                        if (n as f64) > cur_f {
                            cur_f = n as f64;
                        }
                    } else if n > cur_i {
                        cur_i = n;
                    }
                }
            }
            Value::Real(x) => {
                if !have {
                    have = true;
                    cur_f = x;
                    cur_i = x as i64;
                    use_real = true;
                } else {
                    use_real = true;
                    if x > cur_f {
                        cur_f = x;
                    }
                }
            }
            other => {
                return Value::Expr {
                    head: Box::new(Value::Symbol("Max".into())),
                    args: vec![other],
                }
            }
        }
    }
    if !have {
        Value::Expr { head: Box::new(Value::Symbol("Max".into())), args: vec![] }
    } else if use_real {
        Value::Real(cur_f)
    } else {
        Value::Integer(cur_i)
    }
}

// ---------- New math ops implementations ----------

fn floor_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] => Value::Integer(*n),
        [v] => match to_f64_scalar(v) {
            Some(x) => Value::Integer(x.floor() as i64),
            None => map_unary_packed("Floor", v.clone(), |x| x.floor()),
        },
        other => {
            Value::Expr { head: Box::new(Value::Symbol("Floor".into())), args: other.to_vec() }
        }
    }
}

fn ceiling_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] => Value::Integer(*n),
        [v] => match to_f64_scalar(v) {
            Some(x) => Value::Integer(x.ceil() as i64),
            None => map_unary_packed("Ceiling", v.clone(), |x| x.ceil()),
        },
        other => {
            Value::Expr { head: Box::new(Value::Symbol("Ceiling".into())), args: other.to_vec() }
        }
    }
}

fn round_half_to_even(x: f64) -> f64 {
    let f = x.floor();
    let frac = x - f;
    if frac < 0.5 {
        f
    } else if frac > 0.5 {
        f + 1.0
    } else {
        // exactly half; choose even
        if (f as i64) % 2 == 0 {
            f
        } else {
            f + 1.0
        }
    }
}

fn round_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] => Value::Integer(*n),
        [v] => match to_f64_scalar(v) {
            Some(x) => Value::Integer(round_half_to_even(x) as i64),
            None => map_unary_packed("Round", v.clone(), |x| round_half_to_even(x)),
        },
        other => {
            Value::Expr { head: Box::new(Value::Symbol("Round".into())), args: other.to_vec() }
        }
    }
}

fn trunc_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] => Value::Integer(*n),
        [v] => match to_f64_scalar(v) {
            Some(x) => Value::Integer(x.trunc() as i64),
            None => map_unary_packed("Trunc", v.clone(), |x| x.trunc()),
        },
        other => {
            Value::Expr { head: Box::new(Value::Symbol("Trunc".into())), args: other.to_vec() }
        }
    }
}

fn floor_div(a: i64, b: i64) -> i64 {
    let mut q = a / b;
    let r = a % b;
    if (r != 0) && ((r > 0) != (b > 0)) {
        q -= 1;
    }
    q
}

fn mod_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(a), Value::Integer(b)] => {
            if *b == 0 {
                Value::Expr {
                    head: Box::new(Value::Symbol("Mod".into())),
                    args: vec![Value::Integer(*a), Value::Integer(*b)],
                }
            } else {
                Value::Integer(a.rem_euclid(*b))
            }
        }
        [a, b] => match (to_f64_scalar(a), to_f64_scalar(b)) {
            (Some(x), Some(y)) if y != 0.0 => Value::Real(x.rem_euclid(y)),
            _ => Value::Expr {
                head: Box::new(Value::Symbol("Mod".into())),
                args: vec![a.clone(), b.clone()],
            },
        },
        other => Value::Expr { head: Box::new(Value::Symbol("Mod".into())), args: other.to_vec() },
    }
}

fn quotient_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(a), Value::Integer(b)] => {
            if *b == 0 {
                Value::Expr {
                    head: Box::new(Value::Symbol("Quotient".into())),
                    args: vec![Value::Integer(*a), Value::Integer(*b)],
                }
            } else {
                Value::Integer(floor_div(*a, *b))
            }
        }
        [a, b] => match (to_f64_scalar(a), to_f64_scalar(b)) {
            (Some(x), Some(y)) if y != 0.0 => Value::Integer((x / y).floor() as i64),
            _ => Value::Expr {
                head: Box::new(Value::Symbol("Quotient".into())),
                args: vec![a.clone(), b.clone()],
            },
        },
        other => {
            Value::Expr { head: Box::new(Value::Symbol("Quotient".into())), args: other.to_vec() }
        }
    }
}

fn remainder_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(a), Value::Integer(b)] => {
            if *b == 0 {
                Value::Expr {
                    head: Box::new(Value::Symbol("Remainder".into())),
                    args: vec![Value::Integer(*a), Value::Integer(*b)],
                }
            } else {
                let q = floor_div(*a, *b);
                Value::Integer(a - b * q)
            }
        }
        [a, b] => match (to_f64_scalar(a), to_f64_scalar(b)) {
            (Some(x), Some(y)) if y != 0.0 => {
                let q = (x / y).floor();
                Value::Real(x - y * q)
            }
            _ => Value::Expr {
                head: Box::new(Value::Symbol("Remainder".into())),
                args: vec![a.clone(), b.clone()],
            },
        },
        other => {
            Value::Expr { head: Box::new(Value::Symbol("Remainder".into())), args: other.to_vec() }
        }
    }
}

fn divmod_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(a), Value::Integer(b)] => {
            if *b == 0 {
                Value::Expr {
                    head: Box::new(Value::Symbol("DivMod".into())),
                    args: vec![Value::Integer(*a), Value::Integer(*b)],
                }
            } else {
                let q = floor_div(*a, *b);
                let r = a - b * q;
                Value::List(vec![Value::Integer(q), Value::Integer(r)])
            }
        }
        [a, b] => match (to_f64_scalar(a), to_f64_scalar(b)) {
            (Some(x), Some(y)) if y != 0.0 => {
                let q = (x / y).floor();
                let r = x - y * q;
                Value::List(vec![Value::Integer(q as i64), Value::Real(r)])
            }
            _ => Value::Expr {
                head: Box::new(Value::Symbol("DivMod".into())),
                args: vec![a.clone(), b.clone()],
            },
        },
        other => {
            Value::Expr { head: Box::new(Value::Symbol("DivMod".into())), args: other.to_vec() }
        }
    }
}

fn sqrt_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    fn isqrt(n: i64) -> Option<i64> {
        if n < 0 {
            return None;
        }
        let r = (n as f64).sqrt() as i64;
        if r * r == n {
            Some(r)
        } else if (r + 1) * (r + 1) == n {
            Some(r + 1)
        } else {
            None
        }
    }
    match args.as_slice() {
        [Value::Integer(n)] => {
            if *n < 0 {
                Value::Expr {
                    head: Box::new(Value::Symbol("Sqrt".into())),
                    args: vec![Value::Integer(*n)],
                }
            } else if let Some(r) = isqrt(*n) {
                Value::Integer(r)
            } else {
                Value::Real((*n as f64).sqrt())
            }
        }
        [Value::Rational { num, den }] => {
            if *num < 0 {
                Value::Expr {
                    head: Box::new(Value::Symbol("Sqrt".into())),
                    args: vec![Value::Rational { num: *num, den: *den }],
                }
            } else {
                let nr = isqrt(*num);
                let dr = isqrt(*den);
                match (nr, dr) {
                    (Some(a), Some(b)) => rat_value(a, b),
                    _ => Value::Real(((*num as f64) / (*den as f64)).sqrt()),
                }
            }
        }
        [v] => match to_f64_scalar(v) {
            Some(x) if x >= 0.0 => Value::Real(x.sqrt()),
            _ => map_unary_packed("Sqrt", v.clone(), |x| x.sqrt()),
        },
        other => Value::Expr { head: Box::new(Value::Symbol("Sqrt".into())), args: other.to_vec() },
    }
}

fn exp_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v] => match to_f64_scalar(v) {
            Some(x) => Value::Real(x.exp()),
            _ => map_unary_packed("Exp", v.clone(), |x| x.exp()),
        },
        other => Value::Expr { head: Box::new(Value::Symbol("Exp".into())), args: other.to_vec() },
    }
}

fn log_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v] => match to_f64_scalar(v) {
            Some(x) if x > 0.0 => Value::Real(x.ln()),
            _ => Value::Expr { head: Box::new(Value::Symbol("Log".into())), args: vec![v.clone()] },
        },
        [v, base] => match (to_f64_scalar(v), to_f64_scalar(base)) {
            (Some(x), Some(b)) if x > 0.0 && b > 0.0 && b != 1.0 => Value::Real(x.log(b)),
            _ => Value::Expr {
                head: Box::new(Value::Symbol("Log".into())),
                args: vec![v.clone(), base.clone()],
            },
        },
        other => Value::Expr { head: Box::new(Value::Symbol("Log".into())), args: other.to_vec() },
    }
}

fn sin_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v] => match to_f64_scalar(v) {
            Some(x) => Value::Real(x.sin()),
            _ => map_unary_packed("Sin", v.clone(), |x| x.sin()),
        },
        other => Value::Expr { head: Box::new(Value::Symbol("Sin".into())), args: other.to_vec() },
    }
}

fn cos_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v] => match to_f64_scalar(v) {
            Some(x) => Value::Real(x.cos()),
            _ => map_unary_packed("Cos", v.clone(), |x| x.cos()),
        },
        other => Value::Expr { head: Box::new(Value::Symbol("Cos".into())), args: other.to_vec() },
    }
}

fn tan_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v] => match to_f64_scalar(v) {
            Some(x) => Value::Real(x.tan()),
            _ => map_unary_packed("Tan", v.clone(), |x| x.tan()),
        },
        other => Value::Expr { head: Box::new(Value::Symbol("Tan".into())), args: other.to_vec() },
    }
}

fn asin_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v] => match to_f64_scalar(v) {
            Some(x) => Value::Real(x.asin()),
            _ => map_unary_packed("ASin", v.clone(), |x| x.asin()),
        },
        other => Value::Expr { head: Box::new(Value::Symbol("ASin".into())), args: other.to_vec() },
    }
}

fn acos_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v] => match to_f64_scalar(v) {
            Some(x) => Value::Real(x.acos()),
            _ => map_unary_packed("ACos", v.clone(), |x| x.acos()),
        },
        other => Value::Expr { head: Box::new(Value::Symbol("ACos".into())), args: other.to_vec() },
    }
}

fn atan_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v] => match to_f64_scalar(v) {
            Some(x) => Value::Real(x.atan()),
            _ => map_unary_packed("ATan", v.clone(), |x| x.atan()),
        },
        other => Value::Expr { head: Box::new(Value::Symbol("ATan".into())), args: other.to_vec() },
    }
}

fn atan2_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [y, x] => match (to_f64_scalar(y), to_f64_scalar(x)) {
            (Some(a), Some(b)) => Value::Real(a.atan2(b)),
            _ => Value::Expr {
                head: Box::new(Value::Symbol("ATan2".into())),
                args: vec![y.clone(), x.clone()],
            },
        },
        other => {
            Value::Expr { head: Box::new(Value::Symbol("ATan2".into())), args: other.to_vec() }
        }
    }
}

fn nthroot_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(x), Value::Integer(n)] if *n > 0 => {
            let n_u = *n as u32;
            if *x == 0 {
                return Value::Integer(0);
            }
            if *x > 0 {
                // exact check for perfect power
                let xr = (*x as f64).powf(1.0 / (*n as f64));
                let r = xr.round() as i64;
                let mut acc: i128 = 1;
                for _ in 0..n_u {
                    acc = acc.saturating_mul(r as i128);
                }
                if acc == *x as i128 {
                    Value::Integer(r)
                } else {
                    Value::Real((*x as f64).powf(1.0 / (*n as f64)))
                }
            } else {
                if n % 2 != 0 {
                    // odd root of negative is negative root of abs
                    let xr = (-*x as f64).powf(1.0 / (*n as f64));
                    Value::Real(-xr)
                } else {
                    Value::Expr {
                        head: Box::new(Value::Symbol("NthRoot".into())),
                        args: vec![Value::Integer(*x), Value::Integer(*n)],
                    }
                }
            }
        }
        [a, b] => match (to_f64_scalar(a), to_f64_scalar(b)) {
            (Some(x), Some(n)) if n != 0.0 => Value::Real(x.powf(1.0 / n)),
            _ => Value::Expr {
                head: Box::new(Value::Symbol("NthRoot".into())),
                args: vec![a.clone(), b.clone()],
            },
        },
        other => {
            Value::Expr { head: Box::new(Value::Symbol("NthRoot".into())), args: other.to_vec() }
        }
    }
}

fn total_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() == 1 {
        match ev.eval(args[0].clone()) {
            Value::List(items) => {
                if items.is_empty() {
                    return Value::Integer(0);
                }
                let mut acc = items[0].clone();
                for a in items.into_iter().skip(1) {
                    match add_numeric(acc.clone(), a.clone()) {
                        Some(v) => acc = v,
                        None => {
                            return Value::Expr {
                                head: Box::new(Value::Symbol("Total".into())),
                                args: vec![Value::List(vec![])],
                            }
                        }
                    }
                }
                acc
            }
            other => other,
        }
    } else if !args.is_empty() {
        plus(ev, args)
    } else {
        Value::Integer(0)
    }
}

fn mean_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let (items, n) = if args.len() == 1 {
        match ev.eval(args[0].clone()) {
            Value::List(xs) => {
                let n = xs.len();
                (xs, n)
            }
            other => (vec![other], 1),
        }
    } else {
        let n = args.len();
        let items = args.into_iter().map(|a| ev.eval(a)).collect::<Vec<_>>();
        (items, n)
    };
    if n == 0 {
        return Value::Expr {
            head: Box::new(Value::Symbol("Mean".into())),
            args: vec![Value::List(vec![])],
        };
    }
    let sum = total_fn(ev, vec![Value::List(items)]);
    div_numeric(sum, Value::Integer(n as i64))
        .unwrap_or(Value::Expr { head: Box::new(Value::Symbol("Mean".into())), args: vec![] })
}

fn median_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let items = match args.as_slice() {
        [v] => match ev.eval(v.clone()) {
            Value::List(xs) => xs,
            other => vec![other],
        },
        _ => args.into_iter().map(|a| ev.eval(a)).collect(),
    };
    if items.is_empty() {
        return Value::Expr {
            head: Box::new(Value::Symbol("Median".into())),
            args: vec![Value::List(vec![])],
        };
    }
    let mut nums: Vec<(bool, f64, Value)> = Vec::with_capacity(items.len());
    for v in items {
        if let Some(x) = to_f64_scalar(&v) {
            nums.push((matches!(v, Value::Integer(_) | Value::Rational { .. }), x, v));
        } else {
            return Value::Expr { head: Box::new(Value::Symbol("Median".into())), args: vec![] };
        }
    }
    nums.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let n = nums.len();
    if n % 2 == 1 {
        nums[n / 2].2.clone()
    } else {
        // average of two middles; try exact if both are exact ints/rats
        let a = &nums[n / 2 - 1].2;
        let b = &nums[n / 2].2;
        match add_numeric(a.clone(), b.clone()).and_then(|s| div_numeric(s, Value::Integer(2))) {
            Some(v) => v,
            None => Value::Real((nums[n / 2 - 1].1 + nums[n / 2].1) / 2.0),
        }
    }
}

fn variance_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let items = match args.as_slice() {
        [v] => match ev.eval(v.clone()) {
            Value::List(xs) => xs,
            other => vec![other],
        },
        _ => args.into_iter().map(|a| ev.eval(a)).collect(),
    };
    if items.is_empty() {
        return Value::Expr {
            head: Box::new(Value::Symbol("Variance".into())),
            args: vec![Value::List(vec![])],
        };
    }
    let mut vals: Vec<f64> = Vec::with_capacity(items.len());
    for v in items {
        if let Some(x) = to_f64_scalar(&v) {
            vals.push(x);
        } else {
            return Value::Expr { head: Box::new(Value::Symbol("Variance".into())), args: vec![] };
        }
    }
    let n = vals.len() as f64;
    let mean = vals.iter().copied().sum::<f64>() / n;
    let var = vals
        .iter()
        .map(|x| {
            let d = *x - mean;
            d * d
        })
        .sum::<f64>()
        / n; // population variance
    Value::Real(var)
}

fn stddev_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match variance_fn(ev, args) {
        Value::Real(v) => Value::Real(v.sqrt()),
        other => other,
    }
}

// ----- Quantiles -----
fn collect_f64_list(ev: &mut Evaluator, v: Value) -> Option<Vec<f64>> {
    match ev.eval(v) {
        Value::List(xs) => {
            let mut out = Vec::with_capacity(xs.len());
            for it in xs {
                if let Some(x) = to_f64_scalar(&ev.eval(it)) { out.push(x); } else { return None; }
            }
            Some(out)
        }
        Value::PackedArray { data, .. } => Some(data),
        other => to_f64_scalar(&other).map(|x| vec![x]),
    }
}

fn quantile_single(mut vals: Vec<f64>, q: f64) -> Value {
    if vals.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Quantile".into())), args: vec![Value::List(vec![]), Value::Real(q)] }; }
    if q.is_nan() { return Value::Symbol("Null".into()); }
    let n = vals.len();
    vals.sort_by(|a,b| a.partial_cmp(b).unwrap());
    // Use linear interpolation between closest ranks (R type 7)
    let h = (q.clamp(0.0,1.0)) * ((n - 1) as f64);
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    if lo == hi { Value::Real(vals[lo]) } else { let w = h - (lo as f64); Value::Real(vals[lo] * (1.0 - w) + vals[hi] * w) }
}

fn quantile_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v, q] => {
            let vals = match collect_f64_list(ev, v.clone()) { Some(v) => v, None => return Value::Expr { head: Box::new(Value::Symbol("Quantile".into())), args } };
            match ev.eval(q.clone()) {
                Value::Real(p) => quantile_single(vals, p),
                Value::Integer(i) => quantile_single(vals, i as f64),
                Value::List(qs) => Value::List(qs.into_iter().map(|qq| match ev.eval(qq) { Value::Real(p) => quantile_single(vals.clone(), p), Value::Integer(i) => quantile_single(vals.clone(), i as f64), other => other }).collect()),
                other => Value::Expr { head: Box::new(Value::Symbol("Quantile".into())), args: vec![Value::List(vals.into_iter().map(Value::Real).collect()), other] },
            }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("Quantile".into())), args },
    }
}

fn percentile_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v, p] => {
            let vals = match collect_f64_list(ev, v.clone()) { Some(v) => v, None => return Value::Expr { head: Box::new(Value::Symbol("Percentile".into())), args } };
            match ev.eval(p.clone()) {
                Value::Real(pp) => quantile_single(vals, pp / 100.0),
                Value::Integer(i) => quantile_single(vals, (i as f64) / 100.0),
                Value::List(ps) => Value::List(ps.into_iter().map(|qq| match ev.eval(qq) { Value::Real(pp) => quantile_single(vals.clone(), pp / 100.0), Value::Integer(ii) => quantile_single(vals.clone(), (ii as f64) / 100.0), other => other }).collect()),
                other => Value::Expr { head: Box::new(Value::Symbol("Percentile".into())), args: vec![Value::List(vals.into_iter().map(Value::Real).collect()), other] },
            }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("Percentile".into())), args },
    }
}

fn mode_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let items: Vec<Value> = match args.as_slice() {
        [v] => match ev.eval(v.clone()) { Value::List(xs) => xs, other => vec![other] },
        _ => args.into_iter().map(|a| ev.eval(a)).collect(),
    };
    if items.is_empty() { return Value::Symbol("Null".into()); }
    use std::collections::HashMap;
    let mut counts: HashMap<String, (i64, Value, usize)> = HashMap::new();
    for (i, it) in items.into_iter().enumerate() {
        let v = ev.eval(it);
        let k = lyra_runtime::eval::value_order_key(&v);
        let e = counts.entry(k).or_insert((0, v.clone(), i));
        e.0 += 1;
    }
    let mut best: Option<(i64, usize, Value)> = None;
    for (_k, (cnt, v, idx)) in counts.into_iter() {
        match &best { Some((bc, bi, _)) if cnt < *bc || (cnt == *bc && idx >= *bi) => {} _ => best = Some((cnt, idx, v)) }
    }
    best.map(|(_, _, v)| v).unwrap_or(Value::Symbol("Null".into()))
}

fn covariance_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("Covariance".into())), args }; }
    let a = match collect_f64_list(ev, args[0].clone()) { Some(v) => v, None => return Value::Expr { head: Box::new(Value::Symbol("Covariance".into())), args } };
    let b = match collect_f64_list(ev, args[1].clone()) { Some(v) => v, None => return Value::Expr { head: Box::new(Value::Symbol("Covariance".into())), args } };
    if a.len() != b.len() || a.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Covariance".into())), args: vec![Value::List(a.into_iter().map(Value::Real).collect()), Value::List(b.into_iter().map(Value::Real).collect())] }; }
    let n = a.len() as f64;
    let ma = a.iter().copied().sum::<f64>() / n;
    let mb = b.iter().copied().sum::<f64>() / n;
    let mut acc = 0.0; for i in 0..a.len() { acc += (a[i]-ma)*(b[i]-mb); }
    Value::Real(acc / n)
}

fn correlation_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("Correlation".into())), args }; }
    match (covariance_fn(ev, args.clone()), variance_fn(ev, vec![args[0].clone()]), variance_fn(ev, vec![args[1].clone()])) {
        (Value::Real(cov), Value::Real(va), Value::Real(vb)) if va > 0.0 && vb > 0.0 => Value::Real(cov / (va.sqrt() * vb.sqrt())),
        _ => Value::Expr { head: Box::new(Value::Symbol("Correlation".into())), args },
    }
}

fn skewness_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let vals = match args.as_slice() { [v] => collect_f64_list(ev, v.clone()), _ => Some(args.into_iter().filter_map(|a| to_f64_scalar(&ev.eval(a))).collect()) };
    let mut x = match vals { Some(v) => v, None => return Value::Expr { head: Box::new(Value::Symbol("Skewness".into())), args: vec![] } };
    if x.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Skewness".into())), args: vec![Value::List(vec![])] }; }
    let n = x.len() as f64; let mean = x.iter().sum::<f64>()/n; let mut m2=0.0; let mut m3=0.0;
    for v in x.drain(..) { let d = v-mean; m2 += d*d; m3 += d*d*d; }
    if m2 == 0.0 { return Value::Real(0.0); }
    Value::Real((m3/n) / (m2/n).powf(1.5))
}

fn kurtosis_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let vals = match args.as_slice() { [v] => collect_f64_list(ev, v.clone()), _ => Some(args.into_iter().filter_map(|a| to_f64_scalar(&ev.eval(a))).collect()) };
    let mut x = match vals { Some(v) => v, None => return Value::Expr { head: Box::new(Value::Symbol("Kurtosis".into())), args: vec![] } };
    if x.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("Kurtosis".into())), args: vec![Value::List(vec![])] }; }
    let n = x.len() as f64; let mean = x.iter().sum::<f64>()/n; let mut m2=0.0; let mut m4=0.0;
    for v in x.drain(..) { let d = v-mean; let d2=d*d; m2 += d2; m4 += d2*d2; }
    if m2 == 0.0 { return Value::Real(0.0); }
    Value::Real((m4/n) / ((m2/n).powi(2)))
}

// ----- DescriptiveStats -----
fn descriptive_stats_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Accept: list or variadic numbers
    let vals: Vec<f64> = match args.as_slice() {
        [v] => match collect_f64_list(ev, v.clone()) { Some(vs) => vs, None => vec![] },
        _ => args.into_iter().filter_map(|a| to_f64_scalar(&ev.eval(a))).collect(),
    };
    if vals.is_empty() { return Value::Assoc(std::collections::HashMap::new()); }
    let n = vals.len() as f64;
    let mut s = 0.0; let mut s2 = 0.0; let mut mn = f64::INFINITY; let mut mx = f64::NEG_INFINITY;
    for &x in &vals { s += x; s2 += x*x; if x<mn {mn=x}; if x>mx {mx=x}; }
    let mean = s / n;
    let var = (s2 / n) - mean*mean;
    let mut sorted = vals.clone(); sorted.sort_by(|a,b| a.partial_cmp(b).unwrap());
    let median = if sorted.len()%2==1 { sorted[sorted.len()/2] } else { (sorted[sorted.len()/2-1] + sorted[sorted.len()/2]) / 2.0 };
    let p25 = quantile_single(sorted.clone(), 0.25);
    let p75 = quantile_single(sorted.clone(), 0.75);
    let mut out = std::collections::HashMap::new();
    out.insert("count".into(), Value::Integer(vals.len() as i64));
    out.insert("sum".into(), Value::Real(s));
    out.insert("mean".into(), Value::Real(mean));
    out.insert("variance".into(), Value::Real(var.max(0.0)));
    out.insert("stddev".into(), Value::Real(var.max(0.0).sqrt()));
    out.insert("min".into(), Value::Real(mn));
    out.insert("max".into(), Value::Real(mx));
    out.insert("median".into(), Value::Real(median));
    // flatten Quantile results
    if let Value::Real(q1) = p25 { out.insert("p25".into(), Value::Real(q1)); }
    if let Value::Real(q3) = p75 { out.insert("p75".into(), Value::Real(q3)); }
    Value::Assoc(out)
}

// ----- Quantiles (plural wrapper) -----
fn quantiles_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v] => {
            // convenience: return common percentiles
            let q0 = quantile_fn(ev, vec![v.clone(), Value::Real(0.0)]);
            let q25 = quantile_fn(ev, vec![v.clone(), Value::Real(0.25)]);
            let q50 = quantile_fn(ev, vec![v.clone(), Value::Real(0.5)]);
            let q75 = quantile_fn(ev, vec![v.clone(), Value::Real(0.75)]);
            let q100 = quantile_fn(ev, vec![v.clone(), Value::Real(1.0)]);
            Value::Assoc(std::collections::HashMap::from([
                ("p0".into(), q0),
                ("p25".into(), q25),
                ("p50".into(), q50),
                ("p75".into(), q75),
                ("p100".into(), q100),
            ]))
        }
        [v, qs] => quantile_fn(ev, vec![v.clone(), qs.clone()]),
        _ => Value::Expr { head: Box::new(Value::Symbol("Quantiles".into())), args },
    }
}

// ----- RollingStats -----
fn rolling_stats_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("RollingStats".into())), args }; }
    let vals = match collect_f64_list(ev, args[0].clone()) { Some(v) => v, None => vec![] };
    let w = match ev.eval(args[1].clone()) { Value::Integer(n) if n>0 => n as usize, _ => 0 };
    if w == 0 || vals.len() < w { return Value::List(vec![]); }
    let mut out: Vec<Value> = Vec::new();
    for i in 0..=vals.len()-w {
        let win = &vals[i..i+w];
        let n = w as f64;
        let sum: f64 = win.iter().sum();
        let mean = sum / n;
        let var = win.iter().map(|x| { let d=x-mean; d*d }).sum::<f64>()/n;
        let mut mn = f64::INFINITY; let mut mx = f64::NEG_INFINITY;
        for &x in win { if x<mn {mn=x}; if x>mx {mx=x}; }
        out.push(Value::Assoc(std::collections::HashMap::from([
            ("sum".into(), Value::Real(sum)),
            ("mean".into(), Value::Real(mean)),
            ("variance".into(), Value::Real(var)),
            ("stddev".into(), Value::Real(var.sqrt())),
            ("min".into(), Value::Real(mn)),
            ("max".into(), Value::Real(mx)),
        ])));
    }
    Value::List(out)
}

// ----- RandomSample (wrapper around list::Sample) -----
fn random_sample_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Forms: RandomSample[list, k, opts? <|seed->i64|>]
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("RandomSample".into())), args }; }
    if let Some(Value::Assoc(m)) = args.get(2).map(|v| ev.eval(v.clone())) {
        if let Some(Value::Integer(seed)) = m.get("seed") { let _ = seed_random_fn(ev, vec![Value::Integer(*seed)]); }
    }
    // Delegate to Sample
    ev.eval(Value::expr(Value::Symbol("Sample".into()), vec![args[0].clone(), args[1].clone()]))
}


fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
    if a == 0 {
        return b.abs();
    }
    if b == 0 {
        return a.abs();
    }
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}
fn lcm_i64(a: i64, b: i64) -> i64 {
    if a == 0 || b == 0 {
        0
    } else {
        (a / gcd_i64(a, b)).saturating_mul(b).abs()
    }
}

fn gcd_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Integer(0);
    }
    let mut acc: Option<i64> = None;
    for v in args {
        match v {
            Value::Integer(n) => {
                acc = Some(match acc {
                    None => n.abs(),
                    Some(a) => gcd_i64(a, n),
                });
            }
            _ => return Value::Expr { head: Box::new(Value::Symbol("GCD".into())), args: vec![] },
        }
    }
    Value::Integer(acc.unwrap_or(0))
}

fn lcm_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Integer(1);
    }
    let mut acc: Option<i64> = None;
    for v in args {
        match v {
            Value::Integer(n) => {
                acc = Some(match acc {
                    None => n.abs(),
                    Some(a) => lcm_i64(a, n),
                });
            }
            _ => return Value::Expr { head: Box::new(Value::Symbol("LCM".into())), args: vec![] },
        }
    }
    Value::Integer(acc.unwrap_or(1))
}

fn factorial_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] if *n >= 0 => {
            let mut acc: i128 = 1;
            for k in 2..=(*n as i128) {
                acc = acc.saturating_mul(k);
                if acc > i64::MAX as i128 {
                    return Value::Expr {
                        head: Box::new(Value::Symbol("Factorial".into())),
                        args: vec![Value::Integer(*n)],
                    };
                }
            }
            Value::Integer(acc as i64)
        }
        [v] => {
            Value::Expr { head: Box::new(Value::Symbol("Factorial".into())), args: vec![v.clone()] }
        }
        other => {
            Value::Expr { head: Box::new(Value::Symbol("Factorial".into())), args: other.to_vec() }
        }
    }
}

fn binomial_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n), Value::Integer(k)] if *n >= 0 && *k >= 0 && *k <= *n => {
            let n0 = *n;
            let k0 = *k;
            let k = std::cmp::min(k0 as i128, (n0 as i128) - (k0 as i128));
            let mut num: i128 = 1;
            let mut den: i128 = 1;
            for i in 1..=k {
                num = num.saturating_mul((*n as i128) - k + i);
                den = den.saturating_mul(i);
                let g = gcd_i128(num, den);
                num /= g;
                den /= g;
                if num > i64::MAX as i128 {
                    return Value::Expr {
                        head: Box::new(Value::Symbol("Binomial".into())),
                        args: vec![Value::Integer(n0), Value::Integer(k0)],
                    };
                }
            }
            if den != 1 {
                let g = gcd_i128(num, den);
                return rat_value((num / g) as i64, (den / g) as i64);
            }
            Value::Integer(num as i64)
        }
        [a, b] => Value::Expr {
            head: Box::new(Value::Symbol("Binomial".into())),
            args: vec![a.clone(), b.clone()],
        },
        other => {
            Value::Expr { head: Box::new(Value::Symbol("Binomial".into())), args: other.to_vec() }
        }
    }
}

fn gcd_i128(mut a: i128, mut b: i128) -> i128 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

// ----- Combinatorics -----

// (removed unused helpers value_list_to_vec and range_list)

fn combinations_indices(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut res: Vec<Vec<usize>> = Vec::new();
    if k == 0 { res.push(vec![]); return res; }
    if k > n { return res; }
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        res.push(idx.clone());
        // Generate next combination in lex order
        let mut i = k;
        while i > 0 {
            i -= 1;
            if idx[i] != i + n - k { break; }
        }
        if idx[i] == i + n - k { break; }
        idx[i] += 1;
        for j in i+1..k { idx[j] = idx[j-1] + 1; }
    }
    res
}

fn permutations_backtrack(items: &[Value]) -> Vec<Vec<Value>> {
    let n = items.len();
    let mut res: Vec<Vec<Value>> = Vec::new();
    if n == 0 { return vec![vec![]]; }
    let mut used = vec![false; n];
    let mut cur: Vec<Value> = Vec::with_capacity(n);
    fn dfs(items: &[Value], used: &mut [bool], cur: &mut Vec<Value>, res: &mut Vec<Vec<Value>>) {
        if cur.len() == items.len() {
            res.push(cur.clone());
            return;
        }
        for i in 0..items.len() {
            if !used[i] {
                used[i] = true;
                cur.push(items[i].clone());
                dfs(items, used, cur, res);
                cur.pop();
                used[i] = false;
            }
        }
    }
    dfs(items, &mut used, &mut cur, &mut res);
    res
}

fn k_permutations_backtrack(items: &[Value], k: usize) -> Vec<Vec<Value>> {
    let n = items.len();
    let mut res: Vec<Vec<Value>> = Vec::new();
    if k == 0 { return vec![vec![]]; }
    if k > n { return res; }
    let mut used = vec![false; n];
    let mut cur: Vec<Value> = Vec::with_capacity(k);
    fn dfs_k(items: &[Value], k: usize, used: &mut [bool], cur: &mut Vec<Value>, res: &mut Vec<Vec<Value>>) {
        if cur.len() == k {
            res.push(cur.clone());
            return;
        }
        for i in 0..items.len() {
            if !used[i] {
                used[i] = true;
                cur.push(items[i].clone());
                dfs_k(items, k, used, cur, res);
                cur.pop();
                used[i] = false;
            }
        }
    }
    dfs_k(items, k, &mut used, &mut cur, &mut res);
    res
}

fn permutations_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::List(xs)] => {
            let perms = permutations_backtrack(xs);
            Value::List(perms.into_iter().map(Value::List).collect())
        }
        [Value::List(xs), Value::Integer(k)] if *k >= 0 => {
            let k = *k as usize;
            let perms = k_permutations_backtrack(xs, k);
            Value::List(perms.into_iter().map(Value::List).collect())
        }
        [Value::Integer(n)] if *n >= 0 => {
            let xs: Vec<Value> = (1..=(*n as i64)).map(Value::Integer).collect();
            let perms = permutations_backtrack(&xs);
            Value::List(perms.into_iter().map(Value::List).collect())
        }
        [Value::Integer(n), Value::Integer(k)] if *n >= 0 && *k >= 0 => {
            let n = *n as usize;
            let k = *k as usize;
            let xs: Vec<Value> = (1..=n as i64).map(Value::Integer).collect();
            let perms = k_permutations_backtrack(&xs, k);
            Value::List(perms.into_iter().map(Value::List).collect())
        }
        other => Value::Expr { head: Box::new(Value::Symbol("Permutations".into())), args: other.to_vec() },
    }
}

fn combinations_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::List(xs), Value::Integer(k)] if *k >= 0 => {
            let k = *k as usize;
            let n = xs.len();
            let idxs = combinations_indices(n, k);
            let mut out: Vec<Value> = Vec::with_capacity(idxs.len());
            for idx in idxs {
                let mut combo: Vec<Value> = Vec::with_capacity(k);
                for &i in &idx { combo.push(xs[i].clone()); }
                out.push(Value::List(combo));
            }
            Value::List(out)
        }
        [Value::Integer(n), Value::Integer(k)] if *n >= 0 && *k >= 0 => {
            let n = *n as usize;
            let k = *k as usize;
            let idxs = combinations_indices(n, k);
            let mut out: Vec<Value> = Vec::with_capacity(idxs.len());
            for idx in idxs {
                let mut combo: Vec<Value> = Vec::with_capacity(k);
                for &i in &idx { combo.push(Value::Integer((i + 1) as i64)); }
                out.push(Value::List(combo));
            }
            Value::List(out)
        }
        other => Value::Expr { head: Box::new(Value::Symbol("Combinations".into())), args: other.to_vec() },
    }
}

fn permutations_count_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] if *n >= 0 => {
            // n!
            let mut acc: i128 = 1;
            for k in 2..=(*n as i128) {
                acc = acc.saturating_mul(k);
                if acc > i64::MAX as i128 {
                    return Value::Expr { head: Box::new(Value::Symbol("PermutationsCount".into())), args: vec![Value::Integer(*n)] };
                }
            }
            Value::Integer(acc as i64)
        }
        [Value::Integer(n), Value::Integer(k)] if *n >= 0 && *k >= 0 && *k <= *n => {
            // nPk = n! / (n-k)!
            let n_i = *n;
            let k_i = *k;
            let n = n_i as i128;
            let k = k_i as i128;
            let mut acc: i128 = 1;
            for x in (n - k + 1)..=n {
                acc = acc.saturating_mul(x);
                if acc > i64::MAX as i128 {
                    return Value::Expr { head: Box::new(Value::Symbol("PermutationsCount".into())), args: vec![Value::Integer(n_i), Value::Integer(k_i)] };
                }
            }
            Value::Integer(acc as i64)
        }
        [Value::Integer(n), Value::Integer(k)] if *k > *n || *k < 0 => Value::Integer(0),
        other => Value::Expr { head: Box::new(Value::Symbol("PermutationsCount".into())), args: other.to_vec() },
    }
}

fn combinations_count_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n), Value::Integer(k)] => binomial_fn(ev, vec![Value::Integer(*n), Value::Integer(*k)]),
        other => Value::Expr { head: Box::new(Value::Symbol("CombinationsCount".into())), args: other.to_vec() },
    }
}

fn to_degrees_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v] => match to_f64_scalar(v) {
            Some(x) => Value::Real(x * 180.0 / std::f64::consts::PI),
            _ => map_unary_packed("ToDegrees", v.clone(), |x| x * 180.0 / std::f64::consts::PI),
        },
        other => {
            Value::Expr { head: Box::new(Value::Symbol("ToDegrees".into())), args: other.to_vec() }
        }
    }
}

// ----- Number theory -----

fn gcd_ext_i64(a: i64, b: i64) -> (i64, i64, i64) {
    // returns (g, x, y) s.t. ax + by = g
    let (mut old_r, mut r) = (a, b);
    let (mut old_s, mut s) = (1i64, 0i64);
    let (mut old_t, mut t) = (0i64, 1i64);
    while r != 0 {
        let q = old_r / r;
        let tmp_r = old_r - q * r; old_r = r; r = tmp_r;
        let tmp_s = old_s - q * s; old_s = s; s = tmp_s;
        let tmp_t = old_t - q * t; old_t = t; t = tmp_t;
    }
    (old_r.abs(), old_s, old_t)
}

fn is_prime_i64(n: i64) -> bool {
    if n < 2 { return false; }
    if n % 2 == 0 { return n == 2; }
    if n % 3 == 0 { return n == 3; }
    let mut i: i64 = 5;
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 { return false; }
        i += 6;
    }
    true
}

fn next_prime_from(n: i64) -> i64 {
    let mut k = if n < 2 { 2 } else { n + 1 };
    while !is_prime_i64(k) { k += 1; }
    k
}

fn extended_gcd_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(a), Value::Integer(b)] => {
            let (g, x, y) = gcd_ext_i64(*a, *b);
            Value::List(vec![Value::Integer(g), Value::Integer(x), Value::Integer(y)])
        }
        other => Value::Expr { head: Box::new(Value::Symbol("ExtendedGCD".into())), args: other.to_vec() },
    }
}

fn mod_inverse_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(a), Value::Integer(m)] if *m != 0 => {
            let (g, x, _) = gcd_ext_i64(a.rem_euclid(*m), *m);
            if g == 1 {
                let inv = x.rem_euclid(*m);
                Value::Integer(inv)
            } else {
                Value::Expr { head: Box::new(Value::Symbol("ModInverse".into())), args: vec![Value::Integer(*a), Value::Integer(*m)] }
            }
        }
        other => Value::Expr { head: Box::new(Value::Symbol("ModInverse".into())), args: other.to_vec() },
    }
}

fn chinese_remainder_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // ChineseRemainder[{r1,...},{m1,...}]
    match args.as_slice() {
        [Value::List(rs), Value::List(ms)] if rs.len() == ms.len() && !rs.is_empty() => {
            let mut residues: Vec<i128> = Vec::with_capacity(rs.len());
            let mut moduli: Vec<i128> = Vec::with_capacity(ms.len());
            for r in rs {
                if let Value::Integer(v) = r { residues.push(*v as i128); } else { return Value::Expr { head: Box::new(Value::Symbol("ChineseRemainder".into())), args: vec![Value::List(rs.clone()), Value::List(ms.clone())] }; }
            }
            for m in ms {
                if let Value::Integer(v) = m { if *v <= 0 { return Value::Expr { head: Box::new(Value::Symbol("ChineseRemainder".into())), args: vec![Value::List(rs.clone()), Value::List(ms.clone())] }; } moduli.push(*v as i128); } else { return Value::Expr { head: Box::new(Value::Symbol("ChineseRemainder".into())), args: vec![Value::List(rs.clone()), Value::List(ms.clone())] }; }
            }
            // require pairwise coprime moduli
            for i in 0..moduli.len() {
                for j in (i+1)..moduli.len() {
                    if gcd_i128(moduli[i], moduli[j]) != 1 { return Value::Expr { head: Box::new(Value::Symbol("ChineseRemainder".into())), args: vec![Value::List(rs.clone()), Value::List(ms.clone())] }; }
                }
            }
            let m_prod: i128 = moduli.iter().product();
            let mut x: i128 = 0;
            for i in 0..moduli.len() {
                let mi = moduli[i];
                let ai = residues[i].rem_euclid(mi);
                let m_i = m_prod / mi;
                // compute inverse of Mi mod mi
                let (g, inv, _) = {
                    let a = (m_i % mi) as i64; let b = mi as i64; gcd_ext_i64(a, b)
                };
                if g != 1 { return Value::Expr { head: Box::new(Value::Symbol("ChineseRemainder".into())), args: vec![Value::List(rs.clone()), Value::List(ms.clone())] }; }
                let inv_mi = (inv.rem_euclid(mi as i64)) as i128;
                x = (x + ai * m_i * inv_mi).rem_euclid(m_prod);
            }
            if x <= i64::MAX as i128 && x >= i64::MIN as i128 { Value::Integer(x as i64) } else { Value::Expr { head: Box::new(Value::Symbol("ChineseRemainder".into())), args: vec![Value::List(rs.clone()), Value::List(ms.clone())] } }
        }
        other => Value::Expr { head: Box::new(Value::Symbol("ChineseRemainder".into())), args: other.to_vec() },
    }
}

fn divides_q_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(a), Value::Integer(b)] => Value::Boolean(b % a == 0),
        other => Value::Expr { head: Box::new(Value::Symbol("DividesQ".into())), args: other.to_vec() },
    }
}

fn coprime_q_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(a), Value::Integer(b)] => Value::Boolean(gcd_i64(*a, *b) == 1),
        other => Value::Expr { head: Box::new(Value::Symbol("CoprimeQ".into())), args: other.to_vec() },
    }
}

fn prime_q_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] => Value::Boolean(is_prime_i64(*n)),
        other => Value::Expr { head: Box::new(Value::Symbol("PrimeQ".into())), args: other.to_vec() },
    }
}

fn next_prime_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] => Value::Integer(next_prime_from(*n)),
        other => Value::Expr { head: Box::new(Value::Symbol("NextPrime".into())), args: other.to_vec() },
    }
}

fn factor_integer_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] => {
            let mut m = (*n).abs() as i128;
            if m == 0 { return Value::Expr { head: Box::new(Value::Symbol("FactorInteger".into())), args: vec![Value::Integer(*n)] }; }
            let mut res: Vec<Value> = Vec::new();
            let mut count = 0i64;
            while m % 2 == 0 { m /= 2; count += 1; }
            if count > 0 { res.push(Value::List(vec![Value::Integer(2), Value::Integer(count)])); }
            let mut f: i128 = 3;
            while f * f <= m {
                let mut c = 0i64;
                while m % f == 0 { m /= f; c += 1; }
                if c > 0 { if f <= i64::MAX as i128 { res.push(Value::List(vec![Value::Integer(f as i64), Value::Integer(c)])); } else { return Value::Expr { head: Box::new(Value::Symbol("FactorInteger".into())), args: vec![Value::Integer(*n)] }; } }
                f += 2;
            }
            if m > 1 { if m <= i64::MAX as i128 { res.push(Value::List(vec![Value::Integer(m as i64), Value::Integer(1)])); } else { return Value::Expr { head: Box::new(Value::Symbol("FactorInteger".into())), args: vec![Value::Integer(*n)] }; } }
            Value::List(res)
        }
        other => Value::Expr { head: Box::new(Value::Symbol("FactorInteger".into())), args: other.to_vec() },
    }
}

fn factor_pairs_i64(n: i64) -> Option<Vec<(i64, i64)>> {
    if n == 0 { return None; }
    let mut m = n.abs() as i128;
    let mut res: Vec<(i64, i64)> = Vec::new();
    let mut c = 0i64;
    while m % 2 == 0 { m /= 2; c += 1; }
    if c > 0 { res.push((2, c)); }
    let mut f: i128 = 3;
    while f * f <= m {
        let mut e = 0i64;
        while m % f == 0 { m /= f; e += 1; }
        if e > 0 {
            if f > i64::MAX as i128 { return None; }
            res.push((f as i64, e));
        }
        f += 2;
    }
    if m > 1 {
        if m > i64::MAX as i128 { return None; }
        res.push((m as i64, 1));
    }
    Some(res)
}

fn prime_factors_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] => {
            if *n <= 1 { return Value::List(vec![]); }
            match factor_pairs_i64(*n) {
                Some(pairs) => {
                    let mut out: Vec<Value> = Vec::new();
                    for (p, e) in pairs { for _ in 0..e { out.push(Value::Integer(p)); } }
                    Value::List(out)
                }
                None => Value::Expr { head: Box::new(Value::Symbol("PrimeFactors".into())), args: vec![Value::Integer(*n)] },
            }
        }
        other => Value::Expr { head: Box::new(Value::Symbol("PrimeFactors".into())), args: other.to_vec() },
    }
}

fn euler_phi_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] if *n >= 1 => {
            if *n == 1 { return Value::Integer(1); }
            if let Some(pairs) = factor_pairs_i64(*n) {
                let mut phi: i128 = *n as i128;
                for (p, _) in pairs {
                    phi = phi / (p as i128) * ((p as i128) - 1);
                }
                if phi <= i64::MAX as i128 { Value::Integer(phi as i64) } else { Value::Expr { head: Box::new(Value::Symbol("EulerPhi".into())), args: vec![Value::Integer(*n)] } }
            } else {
                Value::Expr { head: Box::new(Value::Symbol("EulerPhi".into())), args: vec![Value::Integer(*n)] }
            }
        }
        [v] => Value::Expr { head: Box::new(Value::Symbol("EulerPhi".into())), args: vec![v.clone()] },
        other => Value::Expr { head: Box::new(Value::Symbol("EulerPhi".into())), args: other.to_vec() },
    }
}

fn mobius_mu_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] => {
            let m = *n;
            if m == 0 { return Value::Integer(0); }
            if m == 1 || m == -1 { return Value::Integer(1); }
            if let Some(pairs) = factor_pairs_i64(m) {
                for (_, e) in &pairs { if *e > 1 { return Value::Integer(0); } }
                let k = pairs.len() as i64;
                if k % 2 == 0 { Value::Integer(1) } else { Value::Integer(-1) }
            } else {
                Value::Expr { head: Box::new(Value::Symbol("MobiusMu".into())), args: vec![Value::Integer(m)] }
            }
        }
        other => Value::Expr { head: Box::new(Value::Symbol("MobiusMu".into())), args: other.to_vec() },
    }
}

fn mod_mul_i64(a: i64, b: i64, m: i64) -> i64 {
    let m128 = m as i128;
    let res = ((a as i128).rem_euclid(m128) * (b as i128).rem_euclid(m128)).rem_euclid(m128);
    res as i64
}

fn mod_pow_i64(mut base: i64, mut exp: i64, m: i64) -> Option<i64> {
    if m == 0 { return None; }
    let mut result: i64 = 1 % m;
    if exp < 0 {
        // require inverse of base modulo m
        let (g, x, _) = gcd_ext_i64(base.rem_euclid(m), m);
        if g != 1 { return None; }
        base = x.rem_euclid(m);
        exp = -exp;
    }
    base = base.rem_euclid(m);
    let mut e = exp as u64;
    while e > 0 {
        if (e & 1) == 1 { result = mod_mul_i64(result, base, m); }
        e >>= 1;
        if e > 0 { base = mod_mul_i64(base, base, m); }
    }
    Some(result)
}

fn power_mod_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(a), Value::Integer(b), Value::Integer(m)] if *m != 0 => {
            match mod_pow_i64(*a, *b, *m) {
                Some(r) => Value::Integer(r),
                None => Value::Expr { head: Box::new(Value::Symbol("PowerMod".into())), args: vec![Value::Integer(*a), Value::Integer(*b), Value::Integer(*m)] },
            }
        }
        other => Value::Expr { head: Box::new(Value::Symbol("PowerMod".into())), args: other.to_vec() },
    }
}

fn to_radians_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [v] => match to_f64_scalar(v) {
            Some(x) => Value::Real(x * std::f64::consts::PI / 180.0),
            _ => map_unary_packed("ToRadians", v.clone(), |x| x * std::f64::consts::PI / 180.0),
        },
        other => {
            Value::Expr { head: Box::new(Value::Symbol("ToRadians".into())), args: other.to_vec() }
        }
    }
}

fn clip_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [x, Value::List(bounds)] if bounds.len() == 2 => {
            let min_v = to_f64_scalar(&bounds[0]);
            let max_v = to_f64_scalar(&bounds[1]);
            match (to_f64_scalar(x), min_v, max_v) {
                (Some(v), Some(lo), Some(hi)) => Value::Real(v.max(lo).min(hi)),
                _ => Value::Expr {
                    head: Box::new(Value::Symbol("Clip".into())),
                    args: vec![x.clone(), Value::List(bounds.clone())],
                },
            }
        }
        [v] => map_unary_packed("Clip", v.clone(), |x| x),
        other => Value::Expr { head: Box::new(Value::Symbol("Clip".into())), args: other.to_vec() },
    }
}

fn signum_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] => Value::Integer(n.signum()),
        [Value::Rational { num, .. }] => Value::Integer(num.signum()),
        [v] => match to_f64_scalar(v) {
            Some(x) => Value::Integer(if x > 0.0 {
                1
            } else if x < 0.0 {
                -1
            } else {
                0
            }),
            None => map_unary_packed("Signum", v.clone(), |x| {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }),
        },
        other => {
            Value::Expr { head: Box::new(Value::Symbol("Signum".into())), args: other.to_vec() }
        }
    }
}

fn map_unary_packed(head: &str, v: Value, f: fn(f64) -> f64) -> Value {
    match v {
        Value::PackedArray { shape, data } => {
            let out: Vec<f64> = data.into_iter().map(|x| f(x)).collect();
            Value::PackedArray { shape, data: out }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol(head.into())), args: vec![v] },
    }
}
