#![cfg(feature = "nn")]

use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;

fn v_to_f64(v: &Value) -> f64 { match v { Value::Real(r) => *r, Value::Integer(n) => *n as f64, _ => 0.0 } }

#[test]
fn batchnorm2d_channelwise_mean_zero() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);

    // Two channels, H=1, W=2; channelwise BN should zero-mean each channel
    let bn = Value::expr(
        Value::Symbol("BatchNorm".into()),
        vec![Value::Assoc([
            ("Epsilon".into(), Value::Real(1e-5)),
            ("InputChannels".into(), Value::Integer(2)),
            ("Height".into(), Value::Integer(1)),
            ("Width".into(), Value::Integer(2)),
            ("Gamma".into(), Value::List(vec![Value::Integer(1), Value::Integer(1)])),
            ("Beta".into(), Value::List(vec![Value::Integer(0), Value::Integer(0)])),
        ].into_iter().collect())],
    );
    let net = ev.eval(Value::expr(Value::Symbol("Sequential".into()), vec![Value::List(vec![bn]) ]));
    // Flattened input: c0:{1,3}, c1:{2,6}
    let x = Value::List(vec![Value::Integer(1), Value::Integer(3), Value::Integer(2), Value::Integer(6)]);
    let y = ev.eval(Value::expr(Value::Symbol("Predict".into()), vec![net, x]));
    match y {
        Value::List(xs) => {
            let m0 = (v_to_f64(&xs[0]) + v_to_f64(&xs[1])) / 2.0;
            let m1 = (v_to_f64(&xs[2]) + v_to_f64(&xs[3])) / 2.0;
            assert!(m0.abs() < 1e-6);
            assert!(m1.abs() < 1e-6);
        }
        Value::PackedArray { shape: _, data } => {
            let m0 = (data[0] + data[1]) / 2.0;
            let m1 = (data[2] + data[3]) / 2.0;
            assert!(m0.abs() < 1e-6);
            assert!(m1.abs() < 1e-6);
        }
        _ => panic!("unexpected output type"),
    }
}

#[test]
fn layernorm2d_across_channels_mean_zero() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let ln = Value::expr(
        Value::Symbol("LayerNorm".into()),
        vec![Value::Assoc([
            ("Epsilon".into(), Value::Real(1e-5)),
            ("InputChannels".into(), Value::Integer(2)),
            ("Height".into(), Value::Integer(1)),
            ("Width".into(), Value::Integer(2)),
        ].into_iter().collect())],
    );
    let net = ev.eval(Value::expr(Value::Symbol("Sequential".into()), vec![Value::List(vec![ln]) ]));
    // c0:{1,3}, c1:{5,7}
    let x = Value::List(vec![Value::Integer(1), Value::Integer(3), Value::Integer(5), Value::Integer(7)]);
    let y = ev.eval(Value::expr(Value::Symbol("Predict".into()), vec![net, x]));
    match y {
        Value::List(xs) => {
            let pos0_mean = (v_to_f64(&xs[0]) + v_to_f64(&xs[2])) / 2.0; // (h=0,w=0)
            let pos1_mean = (v_to_f64(&xs[1]) + v_to_f64(&xs[3])) / 2.0; // (h=0,w=1)
            assert!(pos0_mean.abs() < 1e-6);
            assert!(pos1_mean.abs() < 1e-6);
        }
        Value::PackedArray { shape: _, data } => {
            // channel-first: c0 pos0 = idx0, c1 pos0 = idx2; c0 pos1 = idx1, c1 pos1 = idx3
            let pos0_mean = (data[0] + data[2]) / 2.0;
            let pos1_mean = (data[1] + data[3]) / 2.0;
            assert!(pos0_mean.abs() < 1e-6);
            assert!(pos1_mean.abs() < 1e-6);
        }
        _ => panic!("unexpected output type"),
    }
}
