#![cfg(feature = "media")]

use lyra_core::value::Value;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;

fn gen_sine_raw(sr: u32, ch: u16, secs: f32, freq: f32) -> Value {
    let frames = (sr as f32 * secs).round() as usize;
    let mut samples: Vec<u8> = Vec::with_capacity(frames * ch as usize * 2);
    for n in 0..frames {
        let t = n as f32 / sr as f32;
        let val = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.25; // -6 dBFS
        let s16 = (val * i16::MAX as f32) as i16;
        for _ in 0..ch {
            samples.extend_from_slice(&s16.to_le_bytes());
        }
    }
    Value::Assoc(
        [
            ("sampleRate".into(), Value::Integer(sr as i64)),
            ("channels".into(), Value::Integer(ch as i64)),
            ("encoding".into(), Value::String("s16le".into())),
            (
                "pcm".into(),
                Value::String({
                    use base64::Engine as _;
                    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&samples)
                }),
            ),
        ]
        .into_iter()
        .collect(),
    )
}

#[test]
fn media_pipeline_filter_graph_audio_resample() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);

    // Prepare a small WAV via AudioEncode
    let raw = gen_sine_raw(8000, 1, 0.1, 440.0);
    let wav_b64 = ev.eval(Value::expr(Value::Symbol("AudioEncode".into()), vec![raw]));

    // Build a filter_complex graph to resample audio to 12000 Hz
    let out = ev.eval(Value::expr(
        Value::Symbol("MediaPipeline".into()),
        vec![Value::Assoc(
            [
                ("inputs".into(), Value::List(vec![wav_b64.clone()])),
                (
                    "filterGraph".into(),
                    Value::Assoc(
                        [(
                            "chains".into(),
                            Value::List(vec![Value::Assoc(
                                [
                                    ("in".into(), Value::List(vec![Value::String("0:a".into())])),
                                    (
                                        "filters".into(),
                                        Value::List(vec![Value::Assoc(
                                            [
                                                ("op".into(), Value::String("aresample".into())),
                                                (
                                                    "args".into(),
                                                    Value::String("sample_rate=12000".into()),
                                                ),
                                            ]
                                            .into_iter()
                                            .collect(),
                                        )]),
                                    ),
                                    ("out".into(), Value::String("a0".into())),
                                ]
                                .into_iter()
                                .collect(),
                            )]),
                        )]
                        .into_iter()
                        .collect(),
                    ),
                ),
                ("maps".into(), Value::List(vec![Value::String("[a0]".into())])),
                (
                    "output".into(),
                    Value::Assoc(
                        [("format".into(), Value::String("wav".into()))].into_iter().collect(),
                    ),
                ),
            ]
            .into_iter()
            .collect(),
        )],
    ));

    // Confirm AudioInfo sees the new sample rate
    let info = ev.eval(Value::expr(Value::Symbol("AudioInfo".into()), vec![out]));
    if let Value::Assoc(m) = info {
        assert!(matches!(m.get("sampleRate"), Some(Value::Integer(n)) if *n==12000));
    } else {
        panic!("AudioInfo invalid");
    }
}
