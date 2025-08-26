#![cfg(feature = "audio")]

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
        for _c in 0..ch {
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
fn audio_encode_info_decode_trim_gain() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);

    // Raw sine -> WAV
    let raw = gen_sine_raw(8000, 1, 0.1, 440.0);
    let wav = ev.eval(Value::expr(Value::Symbol("AudioEncode".into()), vec![raw]));

    // AudioInfo
    let info = ev.eval(Value::expr(Value::Symbol("AudioInfo".into()), vec![wav.clone()]));
    if let Value::Assoc(m) = info {
        assert!(matches!(m.get("sampleRate"), Some(Value::Integer(n)) if *n==8000));
        assert!(matches!(m.get("channels"), Some(Value::Integer(n)) if *n==1));
    } else {
        panic!("AudioInfo invalid");
    }

    // Decode to raw
    let raw2 = ev.eval(Value::expr(
        Value::Symbol("AudioDecode".into()),
        vec![
            wav.clone(),
            Value::Assoc([("format".into(), Value::String("raw".into()))].into_iter().collect()),
        ],
    ));
    if let Value::Assoc(m) = raw2 {
        assert!(matches!(m.get("encoding"), Some(Value::String(s)) if s=="s16le"));
    } else {
        panic!("raw2 invalid");
    }

    // Trim first 20ms and take 50ms
    let trimmed = ev.eval(Value::expr(
        Value::Symbol("AudioTrim".into()),
        vec![
            wav.clone(),
            Value::Assoc(
                [("startSec".into(), Value::Real(0.02)), ("durationSec".into(), Value::Real(0.05))]
                    .into_iter()
                    .collect(),
            ),
        ],
    ));
    let tinfo = ev.eval(Value::expr(Value::Symbol("AudioInfo".into()), vec![trimmed]));
    if let Value::Assoc(m) = tinfo {
        let dur = match m.get("durationSec") {
            Some(Value::Real(f)) => *f,
            _ => 0.0,
        };
        assert!(dur > 0.045 && dur < 0.055);
    } else {
        panic!("trim info invalid");
    }

    // Gain -6 dB (linear ~0.5)
    let _gained = ev.eval(Value::expr(
        Value::Symbol("AudioGain".into()),
        vec![wav, Value::Assoc([("db".into(), Value::Real(-6.0))].into_iter().collect())],
    ));
}

#[test]
fn audio_concat_resample_fade_mix() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);

    // Generate two raw sines with different rates/channels
    let raw1 = gen_sine_raw(8000, 1, 0.05, 440.0);
    let wav1 = ev.eval(Value::expr(Value::Symbol("AudioEncode".into()), vec![raw1]));
    let raw2 = gen_sine_raw(16000, 2, 0.05, 660.0);
    let wav2 = ev.eval(Value::expr(Value::Symbol("AudioEncode".into()), vec![raw2]));

    // Concat to 16000 mono
    let concat = ev.eval(Value::expr(
        Value::Symbol("AudioConcat".into()),
        vec![
            Value::List(vec![wav1.clone(), wav2.clone()]),
            Value::Assoc(
                [
                    ("sampleRate".into(), Value::Integer(16000)),
                    ("channels".into(), Value::Integer(1)),
                ]
                .into_iter()
                .collect(),
            ),
        ],
    ));
    let ci = ev.eval(Value::expr(Value::Symbol("AudioInfo".into()), vec![concat]));
    if let Value::Assoc(m) = ci {
        assert!(matches!(m.get("sampleRate"), Some(Value::Integer(n)) if *n==16000));
        assert!(matches!(m.get("channels"), Some(Value::Integer(n)) if *n==1));
        // ~0.1s total
        if let Some(Value::Real(d)) = m.get("durationSec") {
            assert!(*d > 0.09 && *d < 0.11);
        }
    } else {
        panic!("concat info");
    }

    // Resample
    let rs = ev.eval(Value::expr(
        Value::Symbol("AudioResample".into()),
        vec![
            wav1.clone(),
            Value::Assoc([("sampleRate".into(), Value::Integer(12000))].into_iter().collect()),
        ],
    ));
    let rsi = ev.eval(Value::expr(Value::Symbol("AudioInfo".into()), vec![rs]));
    if let Value::Assoc(m) = rsi {
        assert!(matches!(m.get("sampleRate"), Some(Value::Integer(n)) if *n==12000));
    } else {
        panic!("resample info");
    }

    // Fade
    let fad = ev.eval(Value::expr(
        Value::Symbol("AudioFade".into()),
        vec![
            wav2.clone(),
            Value::Assoc(
                [("inSec".into(), Value::Real(0.01)), ("outSec".into(), Value::Real(0.02))]
                    .into_iter()
                    .collect(),
            ),
        ],
    ));
    let _ = ev.eval(Value::expr(Value::Symbol("AudioInfo".into()), vec![fad]));

    // Channel mix: stereo -> mono
    let mono = ev.eval(Value::expr(
        Value::Symbol("AudioChannelMix".into()),
        vec![
            wav2.clone(),
            Value::Assoc([("to".into(), Value::String("mono".into()))].into_iter().collect()),
        ],
    ));
    let mi = ev.eval(Value::expr(Value::Symbol("AudioInfo".into()), vec![mono]));
    if let Value::Assoc(m) = mi {
        assert!(matches!(m.get("channels"), Some(Value::Integer(n)) if *n==1));
    } else {
        panic!("mix info");
    }

    // Channel mix: mono -> stereo
    let stereo = ev.eval(Value::expr(
        Value::Symbol("AudioChannelMix".into()),
        vec![
            wav1,
            Value::Assoc([("to".into(), Value::String("stereo".into()))].into_iter().collect()),
        ],
    ));
    let si = ev.eval(Value::expr(Value::Symbol("AudioInfo".into()), vec![stereo]));
    if let Value::Assoc(m) = si {
        assert!(matches!(m.get("channels"), Some(Value::Integer(n)) if *n==2));
    } else {
        panic!("mix2 info");
    }
}

#[test]
fn audio_save_and_convert() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);

    // Generate raw and encode to wav (base64)
    let raw = gen_sine_raw(8000, 1, 0.05, 330.0);
    let wav_b64 = ev.eval(Value::expr(Value::Symbol("AudioEncode".into()), vec![raw]));

    // Save to disk with mkdirs
    let out_path = format!("{}/target/test_audio/sine.wav", env!("CARGO_MANIFEST_DIR"));
    let _ = ev.eval(Value::expr(
        Value::Symbol("AudioSave".into()),
        vec![
            wav_b64.clone(),
            Value::Assoc(
                [
                    ("path".into(), Value::String(out_path.clone())),
                    ("mkdirs".into(), Value::Boolean(true)),
                ]
                .into_iter()
                .collect(),
            ),
        ],
    ));
    assert!(std::path::Path::new(&out_path).exists());

    // Convert base64 wav to wav again (noop) via AudioConvert accepting string format
    let wav2 = ev.eval(Value::expr(
        Value::Symbol("AudioConvert".into()),
        vec![wav_b64, Value::String("wav".into())],
    ));
    // Probe info
    let info = ev.eval(Value::expr(Value::Symbol("AudioInfo".into()), vec![wav2]));
    if let Value::Assoc(m) = info {
        assert!(matches!(m.get("sampleRate"), Some(Value::Integer(n)) if *n==8000));
    } else {
        panic!("info after convert");
    }
}
