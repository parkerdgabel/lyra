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
fn fft_ifft_roundtrip_impulse() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let x = "{1,0,0,0}";
    let X = eval_str(&mut ev, &format!("FFT[{}]", x));
    // All ones (1+0i)
    match X {
        Value::List(xs) => {
            assert_eq!(xs.len(), 4);
        }
        _ => panic!("FFT did not return a list"),
    }
    let x_rec = eval_str(&mut ev, &format!("IFFT[FFT[{}]]", x));
    match x_rec {
        Value::List(xs) => {
            let want = vec![1.0, 0.0, 0.0, 0.0];
            for (i, v) in xs.iter().enumerate() { assert!((as_f64(v) - want[i]).abs() < 1e-9); }
        }
        _ => panic!("IFFT did not return a real list"),
    }
}

#[test]
fn window_and_convolve() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let w = eval_str(&mut ev, "Window[\"Hann\", 4]");
    match w { Value::List(xs) => assert_eq!(xs.len(), 4), _ => panic!("bad window") }
    let y = eval_str(&mut ev, "Convolve[{1,2,1}, {1,1,1}]");
    match y {
        Value::List(xs) => {
            let want = vec![1.0, 3.0, 4.0, 3.0, 1.0];
            for (i, v) in xs.iter().enumerate() { assert!((as_f64(v) - want[i]).abs() < 1e-9); }
        }
        _ => panic!("bad convolve")
    }
}

#[test]
fn stft_basic_frames() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // simple ramp, size 8, hop 4 => floor((32-8)/4)+1 = 7 frames
    let s = "Range[0, 31]";
    let S = eval_str(&mut ev, &format!("STFT[{}, 8, 4]", s));
    match S {
        Value::List(frames) => {
            assert_eq!(frames.len(), 7);
            match &frames[0] { Value::List(spec0) => assert_eq!(spec0.len(), 8/2+1), _ => panic!("frame0 not list") }
        }
        _ => panic!("STFT not a list"),
    }
}
