use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;
use crate::register_if;
use std::io::Cursor;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

#[cfg(feature = "tools")] use crate::tools::add_specs;
#[cfg(feature = "tools")] use crate::tool_spec;
#[cfg(feature = "tools")] use crate::{schema_str, schema_bool};
#[cfg(feature = "tools")] use std::collections::HashMap;

pub fn register_audio(ev: &mut Evaluator) {
    ev.register("AudioInfo", audio_info as NativeFn, Attributes::empty());
    ev.register("AudioDecode", audio_decode as NativeFn, Attributes::empty());
    ev.register("AudioEncode", audio_encode as NativeFn, Attributes::empty());
    ev.register("AudioConvert", audio_convert as NativeFn, Attributes::empty());
    ev.register("AudioTrim", audio_trim as NativeFn, Attributes::empty());
    ev.register("AudioGain", audio_gain as NativeFn, Attributes::empty());
    ev.register("AudioResample", audio_resample as NativeFn, Attributes::empty());
    ev.register("AudioConcat", audio_concat as NativeFn, Attributes::empty());
    ev.register("AudioFade", audio_fade as NativeFn, Attributes::empty());
    ev.register("AudioChannelMix", audio_channel_mix as NativeFn, Attributes::empty());
    ev.register("AudioSave", audio_save as NativeFn, Attributes::empty());

    #[cfg(feature = "tools")]
    add_specs(vec![
        tool_spec!("AudioInfo", summary: "Probe audio metadata", params: ["input"], tags: ["audio"], output_schema: lyra_core::value::Value::Assoc(HashMap::from([(String::from("type"), lyra_core::value::Value::String(String::from("object")))]))),
        tool_spec!("AudioDecode", summary: "Decode audio to raw (s16le) or WAV", params: ["input","opts"], tags: ["audio","decode"], output_schema: schema_str!()),
        tool_spec!("AudioEncode", summary: "Encode raw PCM to WAV", params: ["raw","opts"], tags: ["audio","encode"], output_schema: schema_str!()),
        tool_spec!("AudioConvert", summary: "Convert audio to WAV", params: ["input","format","opts"], tags: ["audio"], output_schema: schema_str!()),
        tool_spec!("AudioTrim", summary: "Trim audio by time range", params: ["input","opts"], tags: ["audio","edit"], output_schema: schema_str!()),
        tool_spec!("AudioGain", summary: "Apply gain in dB or linear", params: ["input","opts"], tags: ["audio","edit"], output_schema: schema_str!()),
        tool_spec!("AudioResample", summary: "Resample to new sample rate", params: ["input","opts"], tags: ["audio","edit"], output_schema: schema_str!()),
        tool_spec!("AudioConcat", summary: "Concatenate multiple inputs", params: ["inputs","opts"], tags: ["audio","edit"], output_schema: schema_str!()),
        tool_spec!("AudioFade", summary: "Fade in/out", params: ["input","opts"], tags: ["audio","edit"], output_schema: schema_str!()),
        tool_spec!("AudioChannelMix", summary: "Convert channel count (mono/stereo)", params: ["input","opts"], tags: ["audio","edit"], output_schema: schema_str!()),
        tool_spec!("AudioSave", summary: "Encode and write audio to path (WAV)", params: ["input","output","encoding"], tags: ["audio","io"], output_schema: lyra_core::value::Value::Assoc(HashMap::from([(String::from("type"), lyra_core::value::Value::String(String::from("object")))]))),
    ]);
}

fn failure(tag: &str, msg: &str) -> Value {
    Value::Assoc(vec![
        ("message".to_string(), Value::String(msg.to_string())),
        ("tag".to_string(), Value::String(tag.to_string())),
    ].into_iter().collect())
}

fn base64url_encode(data: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(data)
}

fn base64url_decode(s: &str) -> Result<Vec<u8>, String> {
    use base64::Engine as _;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(s).map_err(|e| e.to_string())
}

fn read_input_bytes(ev: &mut Evaluator, v: Value) -> Result<Vec<u8>, String> {
    match ev.eval(v) {
        Value::Assoc(m) => {
            if let Some(Value::String(p)) = m.get("Path").or(m.get("path")).cloned() {
                std::fs::read(&p).map_err(|e| e.to_string())
            } else if let Some(Value::String(b)) = m.get("Bytes").or(m.get("bytes")).cloned() {
                let enc = m.get("Encoding").or(m.get("encoding")).and_then(|v| if let Value::String(s)=v { Some(s.clone()) } else { None }).unwrap_or_else(|| "base64url".into());
                match enc.as_str() { "base64url" | "base64" | "b64" => base64url_decode(&b), _ => Err("Unsupported encoding".into()) }
            } else { Err("Missing Path or Bytes".into()) }
        }
        Value::String(s) | Value::Symbol(s) => {
            match base64url_decode(&s) {
                Ok(b) => Ok(b),
                Err(_) => if std::path::Path::new(&s).exists() { std::fs::read(&s).map_err(|e| e.to_string()) } else { Err("Invalid input: provide {Path} or base64 bytes".into()) }
            }
        }
        _ => Err("Invalid input".into()),
    }
}

struct DecodedAudio {
    sample_rate: u32,
    channels: u16,
    samples: Vec<f32>, // interleaved, normalized [-1,1]
}

fn decode_with_symphonia(bytes: &[u8]) -> Result<DecodedAudio, String> {
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error as SymphErr;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;
    use symphonia::core::audio::{AudioBufferRef, SignalSpec, SampleBuffer};

    let cursor = Cursor::new(bytes.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
    let mut hint = Hint::new();
    let probed = symphonia::default::get_probe().format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| e.to_string())?;
    let mut format = probed.format;
    let track = format.default_track().ok_or_else(|| "No default audio track".to_string())?;
    let track_id = track.id;
    let params = track.codec_params.clone();
    let mut decoder = symphonia::default::get_codecs().make(&params, &DecoderOptions::default()).map_err(|e| e.to_string())?;

    let sr = params.sample_rate.ok_or_else(|| "Missing sample rate".to_string())?;
    let chs = params.channels.ok_or_else(|| "Missing channels".to_string())?;
    let spec = SignalSpec::new(sr, chs);
    let channels = spec.channels.count() as u16;
    let sample_rate = spec.rate;
    let mut sample_buf: Option<SampleBuffer<f32>> = None;
    let mut out: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() { Ok(p) => p, Err(SymphErr::ResetRequired) => { decoder.reset(); continue }, Err(_) => break };
        if packet.track_id() != track_id { continue; }
        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                match audio_buf {
                    AudioBufferRef::F32(buf) => {
                        if sample_buf.is_none() { sample_buf = Some(SampleBuffer::<f32>::new(buf.capacity() as u64, *buf.spec())) }
                        if let Some(s) = &mut sample_buf {
                            s.copy_interleaved_ref(AudioBufferRef::F32(buf));
                            out.extend_from_slice(s.samples());
                        }
                    }
                    other => {
                        // Convert to f32
                        if sample_buf.is_none() { sample_buf = Some(SampleBuffer::<f32>::new(other.capacity() as u64, *other.spec())) }
                        if let Some(s) = &mut sample_buf {
                            s.copy_interleaved_ref(other);
                            out.extend_from_slice(s.samples());
                        }
                    }
                }
            }
            Err(SymphErr::DecodeError(_)) => continue,
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(DecodedAudio { sample_rate, channels, samples: out })
}

fn encode_wav_s16(sr: u32, ch: u16, samples: &[f32]) -> Result<Vec<u8>, String> {
    let mut buf: Vec<u8> = Vec::new();
    let cursor = Cursor::new(&mut buf);
    let spec = hound::WavSpec { channels: ch, sample_rate: sr, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut writer = hound::WavWriter::new(cursor, spec).map_err(|e| e.to_string())?;
    for s in samples.iter() {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(v).map_err(|e| e.to_string())?;
    }
    writer.finalize().map_err(|e| e.to_string())?;
    Ok(buf)
}

#[cfg(any(feature = "audio_mp3", feature = "audio_ogg_vorbis", feature = "audio_flac"))]
fn ffmpeg_path() -> Option<String> {
    // Try env override first
    if let Ok(p) = std::env::var("FFMPEG") { if !p.is_empty() { return Some(p); } }
    // Default to "ffmpeg" in PATH
    Some("ffmpeg".into())
}

#[cfg(any(feature = "audio_mp3", feature = "audio_ogg_vorbis", feature = "audio_flac"))]
fn write_temp_file(prefix: &str, ext: &str, bytes: &[u8]) -> Result<std::path::PathBuf, String> {
    let mut p = std::env::temp_dir();
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_err(|e| e.to_string())?.as_nanos();
    p.push(format!("{}_{}.{}", prefix, ts, ext));
    std::fs::write(&p, bytes).map_err(|e| e.to_string())?;
    Ok(p)
}

#[cfg(any(feature = "audio_mp3", feature = "audio_ogg_vorbis", feature = "audio_flac"))]
fn ffmpeg_encode_to_bytes(mut dec: DecodedAudio, fmt: &str, opts: Option<std::collections::HashMap<String, Value>>) -> Result<Vec<u8>, String> {
    // Write input WAV to temp file
    let wav = encode_wav_s16(dec.sample_rate, dec.channels, &dec.samples)?;
    let in_path = write_temp_file("lyra_in", "wav", &wav)?;
    let out_ext = match fmt { "mp3" => "mp3", "ogg" => "ogg", "flac" => "flac", _ => return Err("Unsupported ffmpeg fmt".into()) };
    let mut out_path = in_path.clone();
    out_path.set_file_name(format!("{}_enc.{}", in_path.file_stem().and_then(|s| s.to_str()).unwrap_or("out"), out_ext));

    let ff = ffmpeg_path().ok_or_else(|| "ffmpeg binary not found".to_string())?;
    let mut cmd = std::process::Command::new(ff);
    cmd.arg("-y").arg("-hide_banner").arg("-loglevel").arg("error").arg("-i").arg(&in_path);
    if let Some(m) = opts {
        match fmt {
            "mp3" => {
                cmd.arg("-c:a").arg("libmp3lame");
                if let Some(Value::Integer(b)) = m.get("bitrateKbps").or(m.get("BitrateKbps")) { cmd.arg("-b:a").arg(format!("{}k", b)); }
                if let Some(Value::Boolean(vbr)) = m.get("vbr").or(m.get("Vbr")) { if *vbr { if let Some(Value::Integer(q)) = m.get("quality").or(m.get("Quality")) { cmd.arg("-q:a").arg(q.to_string()); } } }
            }
            "ogg" => {
                cmd.arg("-c:a").arg("libvorbis");
                if let Some(Value::Real(q)) = m.get("quality").or(m.get("Quality")) { cmd.arg("-q:a").arg(format!("{}", q)); }
                if let Some(Value::Integer(qi)) = m.get("quality").or(m.get("Quality")) { cmd.arg("-q:a").arg(qi.to_string()); }
            }
            "flac" => {
                cmd.arg("-c:a").arg("flac");
                if let Some(Value::Integer(l)) = m.get("compressionLevel").or(m.get("CompressionLevel")) { cmd.arg("-compression_level").arg(l.to_string()); }
            }
            _ => {}
        }
    } else {
        match fmt { "mp3" => { cmd.arg("-c:a").arg("libmp3lame"); }, "ogg" => { cmd.arg("-c:a").arg("libvorbis"); }, "flac" => { cmd.arg("-c:a").arg("flac"); }, _=>{} }
    }
    cmd.arg(&out_path);
    let status = cmd.status().map_err(|e| e.to_string())?;
    // Clean up input regardless
    let _ = std::fs::remove_file(&in_path);
    if !status.success() { return Err("ffmpeg failed to encode".into()); }
    let out_bytes = std::fs::read(&out_path).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&out_path);
    Ok(out_bytes)
}

#[cfg(any(feature = "audio_mp3", feature = "audio_ogg_vorbis", feature = "audio_flac"))]
fn ffmpeg_encode_and_return(dec: DecodedAudio, fmt: &str, opts: Option<std::collections::HashMap<String, Value>>) -> Value {
    match ffmpeg_encode_to_bytes(dec, fmt, opts) {
        Ok(bytes) => Value::String(base64url_encode(&bytes)),
        Err(e) => failure("Audio::encode", &e),
    }
}

fn audio_info(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 1 { return Value::Expr { head: Box::new(Value::Symbol("AudioInfo".into())), args } }
    let bytes = match read_input_bytes(ev, args[0].clone()) { Ok(b)=>b, Err(e)=> return failure("Audio::input", &format!("AudioInfo: {}", e)) };
    match decode_with_symphonia(&bytes) {
        Ok(dec) => {
            let mut m = std::collections::HashMap::new();
            m.insert("sampleRate".into(), Value::Integer(dec.sample_rate as i64));
            m.insert("channels".into(), Value::Integer(dec.channels as i64));
            let frames = (dec.samples.len() as u64) / dec.channels as u64;
            m.insert("frames".into(), Value::Integer(frames as i64));
            let dur = frames as f64 / dec.sample_rate as f64;
            m.insert("durationSec".into(), Value::Real(dur));
            Value::Assoc(m)
        }
        Err(e) => failure("Audio::decode", &format!("AudioInfo: {}", e)),
    }
}

fn audio_decode(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // AudioDecode(input, { format?: "raw"|"wav" }) -> raw assoc or base64 wav
    if args.len() < 1 { return Value::Expr { head: Box::new(Value::Symbol("AudioDecode".into())), args } }
    let bytes = match read_input_bytes(ev, args[0].clone()) { Ok(b)=>b, Err(e)=> return failure("Audio::input", &format!("AudioDecode: {}", e)) };
    let dec = match decode_with_symphonia(&bytes) { Ok(d)=>d, Err(e)=> return failure("Audio::decode", &format!("AudioDecode: {}", e)) };
    let opts = if args.len() > 1 { match ev.eval(args[1].clone()) { Value::Assoc(m)=>m, _=>std::collections::HashMap::new() } } else { std::collections::HashMap::new() };
    let fmt = opts.get("format").or(opts.get("Format")).and_then(|v| if let Value::String(s)=v { Some(s.to_lowercase()) } else { None }).unwrap_or_else(|| "raw".into());
    if fmt == "raw" {
        // Return s16le as bytes for compactness
        let mut pcm_s16: Vec<u8> = Vec::with_capacity(dec.samples.len()*2);
        for s in dec.samples.iter() { let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16; pcm_s16.extend_from_slice(&v.to_le_bytes()); }
        let mut m = std::collections::HashMap::new();
        m.insert("sampleRate".into(), Value::Integer(dec.sample_rate as i64));
        m.insert("channels".into(), Value::Integer(dec.channels as i64));
        m.insert("encoding".into(), Value::String("s16le".into()));
        m.insert("pcm".into(), Value::String(base64url_encode(&pcm_s16)));
        Value::Assoc(m)
    } else {
        match encode_wav_s16(dec.sample_rate, dec.channels, &dec.samples) {
            Ok(wav) => Value::String(base64url_encode(&wav)),
            Err(e) => failure("Audio::encode", &format!("AudioDecode: {}", e))
        }
    }
}

fn audio_encode(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // AudioEncode({ pcm, sampleRate, channels, encoding? }, opts?) -> base64 wav
    if args.len() < 1 { return Value::Expr { head: Box::new(Value::Symbol("AudioEncode".into())), args } }
    let raw = match ev.eval(args[0].clone()) { Value::Assoc(m)=>m, _=> return failure("Audio::encode", "Expected raw assoc") };
    let sr = raw.get("sampleRate").and_then(|v| if let Value::Integer(i)=v { Some(*i as u32) } else { None }).unwrap_or(44100);
    let ch = raw.get("channels").and_then(|v| if let Value::Integer(i)=v { Some(*i as u16) } else { None }).unwrap_or(2);
    let enc = raw.get("encoding").and_then(|v| if let Value::String(s)=v { Some(s.to_lowercase()) } else { None }).unwrap_or_else(|| "s16le".into());
    let pcm_b64 = raw.get("pcm").and_then(|v| if let Value::String(s)=v { Some(s.clone()) } else { None }).ok_or_else(|| "Missing pcm".to_string());
    let pcm_bytes = match pcm_b64 { Ok(s)=> base64url_decode(&s), Err(e)=> return failure("Audio::encode", &e) };
    let (samples_f32, ok) = match (pcm_bytes, enc.as_str()) {
        (Ok(bytes), "s16le") => {
            let mut out = Vec::with_capacity(bytes.len()/2);
            for chunk in bytes.chunks_exact(2) { let v = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / i16::MAX as f32; out.push(v); }
            (out, true)
        }
        (Ok(bytes), "f32le") => {
            let mut out = Vec::with_capacity(bytes.len()/4);
            for chunk in bytes.chunks_exact(4) { let v = f32::from_le_bytes([chunk[0],chunk[1],chunk[2],chunk[3]]); out.push(v); }
            (out, true)
        }
        (Err(e), _) => { return failure("Audio::encode", &e) }
        _ => { return failure("Audio::encode", "Unsupported encoding") }
    };
    if !ok { return failure("Audio::encode", "Invalid pcm") }
    match encode_wav_s16(sr, ch, &samples_f32) {
        Ok(wav) => Value::String(base64url_encode(&wav)),
        Err(e) => failure("Audio::encode", &e),
    }
}

fn audio_convert(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // AudioConvert(input, formatOrOpts)
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("AudioConvert".into())), args } }
    let bytes = match read_input_bytes(ev, args[0].clone()) { Ok(b)=>b, Err(e)=> return failure("Audio::input", &format!("AudioConvert: {}", e)) };
    let dec = match decode_with_symphonia(&bytes) { Ok(d)=>d, Err(e)=> return failure("Audio::decode", &format!("AudioConvert: {}", e)) };
    let (fmt, _opts) = match ev.eval(args[1].clone()) {
        Value::String(s) | Value::Symbol(s) => (s.to_lowercase(), std::collections::HashMap::new()),
        Value::Assoc(m) => {
            let f = m.get("format").or(m.get("Format")).and_then(|v| if let Value::String(s)=v { Some(s.to_lowercase()) } else { None }).unwrap_or_else(|| "wav".into());
            (f, m)
        }
        _ => ("wav".into(), std::collections::HashMap::new()),
    };
    match fmt.as_str() {
        "wav" => match encode_wav_s16(dec.sample_rate, dec.channels, &dec.samples) {
            Ok(wav) => Value::String(base64url_encode(&wav)),
            Err(e) => failure("Audio::encode", &e)
        },
        "mp3" => {
            #[cfg(feature = "audio_mp3")]
            { return ffmpeg_encode_and_return(dec, "mp3", Some(_opts)); }
            #[cfg(not(feature = "audio_mp3"))]
            { return failure("Audio::convert", "MP3 encoding not enabled (feature audio_mp3)"); }
        }
        "ogg" | "vorbis" => {
            #[cfg(feature = "audio_ogg_vorbis")]
            { return ffmpeg_encode_and_return(dec, "ogg", Some(_opts)); }
            #[cfg(not(feature = "audio_ogg_vorbis"))]
            { return failure("Audio::convert", "Ogg/Vorbis encoding not enabled (feature audio_ogg_vorbis)"); }
        }
        "flac" => {
            #[cfg(feature = "audio_flac")]
            { return ffmpeg_encode_and_return(dec, "flac", Some(_opts)); }
            #[cfg(not(feature = "audio_flac"))]
            { return failure("Audio::convert", "FLAC encoding not enabled (feature audio_flac)"); }
        }
        other => failure("Audio::convert", &format!("Unsupported target format: {}", other))
    }
}

fn audio_trim(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // AudioTrim(input, { startSec, durationSec? }) -> base64 wav
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("AudioTrim".into())), args } }
    let bytes = match read_input_bytes(ev, args[0].clone()) { Ok(b)=>b, Err(e)=> return failure("Audio::input", &format!("AudioTrim: {}", e)) };
    let dec = match decode_with_symphonia(&bytes) { Ok(d)=>d, Err(e)=> return failure("Audio::decode", &format!("AudioTrim: {}", e)) };
    let opts = match ev.eval(args[1].clone()) { Value::Assoc(m)=>m, _=> std::collections::HashMap::new() };
    let start = opts.get("startSec").or(opts.get("StartSec")).and_then(|v| match v { Value::Real(f)=>Some(*f), Value::Integer(i)=>Some(*i as f64), _=>None }).unwrap_or(0.0);
    let dur_opt = opts.get("durationSec").or(opts.get("DurationSec")).and_then(|v| match v { Value::Real(f)=>Some(*f), Value::Integer(i)=>Some(*i as f64), _=>None });
    let frame_start = (start * dec.sample_rate as f64).round().max(0.0) as usize;
    let total_frames = dec.samples.len() / dec.channels as usize;
    let max_frames = if let Some(d) = dur_opt { (d * dec.sample_rate as f64).round() as usize } else { total_frames.saturating_sub(frame_start) };
    let frame_end = (frame_start + max_frames).min(total_frames);
    let ch = dec.channels as usize;
    let start_i = frame_start * ch; let end_i = frame_end * ch;
    let slice = if start_i < end_i && end_i <= dec.samples.len() { &dec.samples[start_i..end_i] } else { &[] };
    match encode_wav_s16(dec.sample_rate, dec.channels, slice) {
        Ok(wav) => Value::String(base64url_encode(&wav)),
        Err(e) => failure("Audio::encode", &e)
    }
}

fn audio_gain(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // AudioGain(input, { db?: f64, linear?: f64 }) -> base64 wav
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("AudioGain".into())), args } }
    let bytes = match read_input_bytes(ev, args[0].clone()) { Ok(b)=>b, Err(e)=> return failure("Audio::input", &format!("AudioGain: {}", e)) };
    let mut dec = match decode_with_symphonia(&bytes) { Ok(d)=>d, Err(e)=> return failure("Audio::decode", &format!("AudioGain: {}", e)) };
    let opts = match ev.eval(args[1].clone()) { Value::Assoc(m)=>m, _=> std::collections::HashMap::new() };
    let linear = if let Some(Value::Real(db)) = opts.get("db").or(opts.get("dB")) { 10f32.powf((*db as f32)/20.0) }
                 else if let Some(Value::Integer(db)) = opts.get("db").or(opts.get("dB")) { 10f32.powf((*db as f32)/20.0) }
                 else if let Some(Value::Real(lin)) = opts.get("linear") { *lin as f32 }
                 else if let Some(Value::Integer(lin)) = opts.get("linear") { *lin as f32 }
                 else { 1.0 };
    for s in dec.samples.iter_mut() { *s = (*s * linear).clamp(-1.0, 1.0); }
    match encode_wav_s16(dec.sample_rate, dec.channels, &dec.samples) {
        Ok(wav) => Value::String(base64url_encode(&wav)),
        Err(e) => failure("Audio::encode", &e)
    }
}

fn resample_linear(samples: &[f32], channels: u16, in_sr: u32, out_sr: u32) -> Vec<f32> {
    if in_sr == out_sr { return samples.to_vec(); }
    let ch = channels as usize;
    let in_frames = samples.len() / ch;
    if in_frames == 0 { return Vec::new(); }
    let out_frames = ((in_frames as u64) * (out_sr as u64) + in_sr as u64 - 1) / in_sr as u64; // ceil
    let out_frames = out_frames as usize;
    let mut out = vec![0.0f32; out_frames * ch];
    let ratio = in_sr as f64 / out_sr as f64;
    for of in 0..out_frames {
        let pos = of as f64 * ratio;
        let i0 = pos.floor() as isize;
        let frac = (pos - i0 as f64) as f32;
        let i1 = (i0 + 1) as isize;
        for c in 0..ch {
            let s0 = if i0 < 0 { samples[c] } else if (i0 as usize) < in_frames { samples[i0 as usize * ch + c] } else { samples[(in_frames - 1) * ch + c] };
            let s1 = if i1 < 0 { samples[c] } else if (i1 as usize) < in_frames { samples[i1 as usize * ch + c] } else { samples[(in_frames - 1) * ch + c] };
            out[of * ch + c] = s0 + (s1 - s0) * frac;
        }
    }
    out
}

fn mix_channels(samples: &[f32], in_ch: u16, out_ch: u16) -> Vec<f32> {
    if in_ch == out_ch { return samples.to_vec(); }
    let in_ch_usize = in_ch as usize;
    let out_ch_usize = out_ch as usize;
    let frames = samples.len() / in_ch_usize;
    let mut out = vec![0.0f32; frames * out_ch_usize];
    for f in 0..frames {
        // Average of input channels for downmix
        let mut sum = 0.0f32;
        for c in 0..in_ch_usize { sum += samples[f * in_ch_usize + c]; }
        let avg = sum / in_ch_usize as f32;
        for c in 0..out_ch_usize {
            // If increasing channels, duplicate avg; if decreasing, write avg to all out channels
            out[f * out_ch_usize + c] = if out_ch_usize >= in_ch_usize { samples[f * in_ch_usize + (c % in_ch_usize)] } else { avg };
        }
    }
    out
}

fn audio_resample(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // AudioResample(input, { sampleRate })
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("AudioResample".into())), args } }
    let bytes = match read_input_bytes(ev, args[0].clone()) { Ok(b)=>b, Err(e)=> return failure("Audio::input", &format!("AudioResample: {}", e)) };
    let dec = match decode_with_symphonia(&bytes) { Ok(d)=>d, Err(e)=> return failure("Audio::decode", &format!("AudioResample: {}", e)) };
    let opts = match ev.eval(args[1].clone()) { Value::Assoc(m)=>m, _=> std::collections::HashMap::new() };
    let target_sr = opts.get("sampleRate").or(opts.get("SampleRate")).and_then(|v| if let Value::Integer(i)=v { Some(*i as u32) } else { None }).unwrap_or(dec.sample_rate);
    let out_samples = resample_linear(&dec.samples, dec.channels, dec.sample_rate, target_sr);
    match encode_wav_s16(target_sr, dec.channels, &out_samples) { Ok(wav)=> Value::String(base64url_encode(&wav)), Err(e)=> failure("Audio::encode", &e) }
}

fn audio_concat(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // AudioConcat([inputs], { sampleRate?, channels? }) -> base64 wav
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("AudioConcat".into())), args } }
    let list = match ev.eval(args[0].clone()) { Value::List(xs)=>xs, _=> return failure("Audio::concat", "Expected list of inputs") };
    let opts = if args.len()>1 { match ev.eval(args[1].clone()) { Value::Assoc(m)=>m, _=> std::collections::HashMap::new() } } else { std::collections::HashMap::new() };
    let mut decoded: Vec<DecodedAudio> = Vec::new();
    for v in list { let b = match read_input_bytes(ev, v) { Ok(bb)=>bb, Err(e)=> return failure("Audio::input", &e) }; match decode_with_symphonia(&b) { Ok(d)=> decoded.push(d), Err(e)=> return failure("Audio::decode", &e) } }
    if decoded.is_empty() { return failure("Audio::concat", "No inputs") }
    let target_sr = opts.get("sampleRate").or(opts.get("SampleRate")).and_then(|v| if let Value::Integer(i)=v { Some(*i as u32) } else { None }).unwrap_or(decoded[0].sample_rate);
    let target_ch = opts.get("channels").or(opts.get("Channels")).and_then(|v| if let Value::Integer(i)=v { Some(*i as u16) } else { None }).unwrap_or(decoded[0].channels);
    let mut out: Vec<f32> = Vec::new();
    for mut d in decoded {
        if d.sample_rate != target_sr { d.samples = resample_linear(&d.samples, d.channels, d.sample_rate, target_sr); d.sample_rate = target_sr; }
        if d.channels != target_ch { d.samples = mix_channels(&d.samples, d.channels, target_ch); d.channels = target_ch; }
        out.extend_from_slice(&d.samples);
    }
    match encode_wav_s16(target_sr, target_ch, &out) { Ok(wav)=> Value::String(base64url_encode(&wav)), Err(e)=> failure("Audio::encode", &e) }
}

fn audio_fade(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // AudioFade(input, { inSec?: f64, outSec?: f64 })
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("AudioFade".into())), args } }
    let bytes = match read_input_bytes(ev, args[0].clone()) { Ok(b)=>b, Err(e)=> return failure("Audio::input", &format!("AudioFade: {}", e)) };
    let mut dec = match decode_with_symphonia(&bytes) { Ok(d)=>d, Err(e)=> return failure("Audio::decode", &format!("AudioFade: {}", e)) };
    let opts = match ev.eval(args[1].clone()) { Value::Assoc(m)=>m, _=> std::collections::HashMap::new() };
    let fade_in = opts.get("inSec").or(opts.get("InSec")).and_then(|v| match v { Value::Real(f)=>Some(*f), Value::Integer(i)=>Some(*i as f64), _=>None }).unwrap_or(0.0);
    let fade_out = opts.get("outSec").or(opts.get("OutSec")).and_then(|v| match v { Value::Real(f)=>Some(*f), Value::Integer(i)=>Some(*i as f64), _=>None }).unwrap_or(0.0);
    let ch = dec.channels as usize;
    let frames = dec.samples.len()/ch;
    let fi = (fade_in * dec.sample_rate as f64).round() as usize;
    let fo = (fade_out * dec.sample_rate as f64).round() as usize;
    // Fade in
    for n in 0..fi.min(frames) {
        let g = n as f32 / fi.max(1) as f32;
        for c in 0..ch { dec.samples[n*ch + c] *= g; }
    }
    // Fade out
    for i in 0..fo.min(frames) {
        let n = frames.saturating_sub(fo) + i;
        if n < frames {
            let g = (fo.saturating_sub(i)) as f32 / fo.max(1) as f32;
            for c in 0..ch { dec.samples[n*ch + c] *= g; }
        }
    }
    match encode_wav_s16(dec.sample_rate, dec.channels, &dec.samples) { Ok(wav)=> Value::String(base64url_encode(&wav)), Err(e)=> failure("Audio::encode", &e) }
}

fn audio_channel_mix(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // AudioChannelMix(input, { to?: "mono"|"stereo"|"channels", channels?: int })
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("AudioChannelMix".into())), args } }
    let bytes = match read_input_bytes(ev, args[0].clone()) { Ok(b)=>b, Err(e)=> return failure("Audio::input", &format!("AudioChannelMix: {}", e)) };
    let mut dec = match decode_with_symphonia(&bytes) { Ok(d)=>d, Err(e)=> return failure("Audio::decode", &format!("AudioChannelMix: {}", e)) };
    let opts = match ev.eval(args[1].clone()) { Value::Assoc(m)=>m, _=> std::collections::HashMap::new() };
    let to = opts.get("to").or(opts.get("To")).and_then(|v| if let Value::String(s)=v { Some(s.to_lowercase()) } else { None });
    let target_ch: u16 = match to.as_deref() {
        Some("mono") => 1,
        Some("stereo") => 2,
        _ => opts.get("channels").or(opts.get("Channels")).and_then(|v| if let Value::Integer(i)=v { Some(*i as u16) } else { None }).unwrap_or(dec.channels),
    };
    if target_ch != dec.channels { dec.samples = mix_channels(&dec.samples, dec.channels, target_ch); dec.channels = target_ch; }
    match encode_wav_s16(dec.sample_rate, dec.channels, &dec.samples) { Ok(wav)=> Value::String(base64url_encode(&wav)), Err(e)=> failure("Audio::encode", &e) }
}

fn infer_audio_format_from_ext(path: &str) -> Option<String> {
    std::path::Path::new(path).extension().and_then(|e| e.to_str()).map(|s| s.to_lowercase()).and_then(|ext| {
        match ext.as_str() { "wav" => Some("wav".into()), _ => None }
    })
}

fn audio_save(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // AudioSave(input, output, encoding?) -> { path }
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("AudioSave".into())), args } }
    // Decode input to f32 samples, supporting raw assoc too.
    let maybe_assoc = matches!(ev.eval(args[0].clone()), Value::Assoc(_));
    let mut dec: DecodedAudio;
    if maybe_assoc {
        // Reuse AudioEncode path to get a WAV, then write bytes directly.
        let wav_b64 = audio_encode(ev, vec![args[0].clone()]);
        let wav_bytes = match wav_b64 { Value::String(s) => base64url_decode(&s), _ => Err("AudioSave: invalid raw input".into()) };
        let (out_path, mkdirs) = match ev.eval(args[1].clone()) {
            Value::String(s) | Value::Symbol(s) => (s, false),
            Value::Assoc(m) => {
                let p = m.get("path").or(m.get("Path")).and_then(|v| if let Value::String(s)=v { Some(s.clone()) } else { None });
                let mk = m.get("mkdirs").or(m.get("Mkdirs")).and_then(|v| if let Value::Boolean(b)=v { Some(*b) } else { None }).unwrap_or(false);
                match p { Some(pp)=> (pp, mk), None => return failure("Audio::save", "Missing output path") }
            }
            _ => return failure("Audio::save", "Invalid output"),
        };
        if let Some(parent) = std::path::Path::new(&out_path).parent() { if mkdirs { let _ = std::fs::create_dir_all(parent); } }
        return match wav_bytes { Ok(b)=> { match std::fs::write(&out_path, &b) { Ok(_)=> Value::Assoc(std::iter::once(("path".into(), Value::String(out_path))).collect()), Err(e)=> failure("Audio::save", &e.to_string()) } }, Err(e)=> failure("Audio::save", &e) };
    } else {
        let bytes = match read_input_bytes(ev, args[0].clone()) { Ok(b)=>b, Err(e)=> return failure("Audio::input", &format!("AudioSave: {}", e)) };
        dec = match decode_with_symphonia(&bytes) { Ok(d)=>d, Err(e)=> return failure("Audio::decode", &format!("AudioSave: {}", e)) };
    }
    let (out_path, mkdirs) = match ev.eval(args[1].clone()) {
        Value::String(s) | Value::Symbol(s) => (s, false),
        Value::Assoc(m) => {
            let p = m.get("path").or(m.get("Path")).and_then(|v| if let Value::String(s)=v { Some(s.clone()) } else { None });
            let mk = m.get("mkdirs").or(m.get("Mkdirs")).and_then(|v| if let Value::Boolean(b)=v { Some(*b) } else { None }).unwrap_or(false);
            match p { Some(pp)=> (pp, mk), None => return failure("Audio::save", "Missing output path") }
        }
        _ => return failure("Audio::save", "Invalid output"),
    };
    let mut enc_opts = if args.len() > 2 { match ev.eval(args[2].clone()) { Value::Assoc(m)=>m, _=> std::collections::HashMap::new() } } else { std::collections::HashMap::new() };
    if !enc_opts.contains_key("format") && !enc_opts.contains_key("Format") {
        if let Some(fmt) = infer_audio_format_from_ext(&out_path) { enc_opts.insert("format".into(), Value::String(fmt)); }
    }
    let fmt = enc_opts.get("format").or(enc_opts.get("Format")).and_then(|v| if let Value::String(s)=v { Some(s.to_lowercase()) } else { None }).unwrap_or_else(|| "wav".into());
    let bytes = match fmt.as_str() {
        "wav" => match encode_wav_s16(dec.sample_rate, dec.channels, &dec.samples) { Ok(x)=>x, Err(e)=> return failure("Audio::encode", &e) },
        "mp3" => {
            #[cfg(feature = "audio_mp3")]
            { match ffmpeg_encode_to_bytes(dec, "mp3", Some(enc_opts)) { Ok(b)=>b, Err(e)=> return failure("Audio::save", &e) } }
            #[cfg(not(feature = "audio_mp3"))]
            { return failure("Audio::save", "MP3 encoding not enabled (feature audio_mp3)"); }
        }
        "ogg" | "vorbis" => {
            #[cfg(feature = "audio_ogg_vorbis")]
            { match ffmpeg_encode_to_bytes(dec, "ogg", Some(enc_opts)) { Ok(b)=>b, Err(e)=> return failure("Audio::save", &e) } }
            #[cfg(not(feature = "audio_ogg_vorbis"))]
            { return failure("Audio::save", "Ogg/Vorbis encoding not enabled (feature audio_ogg_vorbis)"); }
        }
        "flac" => {
            #[cfg(feature = "audio_flac")]
            { match ffmpeg_encode_to_bytes(dec, "flac", Some(enc_opts)) { Ok(b)=>b, Err(e)=> return failure("Audio::save", &e) } }
            #[cfg(not(feature = "audio_flac"))]
            { return failure("Audio::save", "FLAC encoding not enabled (feature audio_flac)"); }
        }
        other => return failure("Audio::save", &format!("Unsupported format: {}", other))
    };
    if let Some(parent) = std::path::Path::new(&out_path).parent() { if mkdirs { let _ = std::fs::create_dir_all(parent); } }
    match std::fs::write(&out_path, &bytes) {
        Ok(_) => Value::Assoc(std::iter::once(("path".into(), Value::String(out_path))).collect()),
        Err(e) => failure("Audio::save", &e.to_string()),
    }
}


pub fn register_audio_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str)->bool) {
    register_if(ev, pred, "AudioInfo", audio_info as NativeFn, Attributes::empty());
    register_if(ev, pred, "AudioDecode", audio_decode as NativeFn, Attributes::empty());
    register_if(ev, pred, "AudioEncode", audio_encode as NativeFn, Attributes::empty());
    register_if(ev, pred, "AudioConvert", audio_convert as NativeFn, Attributes::empty());
    register_if(ev, pred, "AudioTrim", audio_trim as NativeFn, Attributes::empty());
    register_if(ev, pred, "AudioGain", audio_gain as NativeFn, Attributes::empty());
    register_if(ev, pred, "AudioResample", audio_resample as NativeFn, Attributes::empty());
    register_if(ev, pred, "AudioConcat", audio_concat as NativeFn, Attributes::empty());
    register_if(ev, pred, "AudioFade", audio_fade as NativeFn, Attributes::empty());
    register_if(ev, pred, "AudioChannelMix", audio_channel_mix as NativeFn, Attributes::empty());
    register_if(ev, pred, "AudioSave", audio_save as NativeFn, Attributes::empty());
}
