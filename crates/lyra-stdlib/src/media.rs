use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

use crate::register_if;
#[cfg(feature = "tools")]
use crate::schema_str;
#[cfg(feature = "tools")]
use crate::tool_spec;
#[cfg(feature = "tools")]
use crate::tools::add_specs;
#[cfg(feature = "tools")]
use std::collections::HashMap;

pub fn register_media(ev: &mut Evaluator) {
    ev.register("MediaProbe", media_probe as NativeFn, Attributes::empty());
    ev.register("MediaTranscode", media_transcode as NativeFn, Attributes::empty());
    ev.register("MediaThumbnail", media_thumbnail as NativeFn, Attributes::empty());
    ev.register("MediaConcat", media_concat as NativeFn, Attributes::empty());
    ev.register("MediaPipeline", media_pipeline as NativeFn, Attributes::empty());
    ev.register("MediaExtractAudio", media_extract_audio as NativeFn, Attributes::empty());
    ev.register("MediaMux", media_mux as NativeFn, Attributes::empty());

    #[cfg(feature = "tools")]
    add_specs(vec![
        tool_spec!("MediaProbe", summary: "Probe media via ffprobe", params: ["input"], tags: ["media","ffmpeg"], output_schema: lyra_core::value::Value::Assoc(HashMap::from([(String::from("type"), lyra_core::value::Value::String(String::from("object")))]))),
        tool_spec!("MediaTranscode", summary: "Transcode media with options", params: ["input","opts"], tags: ["media","ffmpeg"], output_schema: schema_str!()),
        tool_spec!("MediaThumbnail", summary: "Extract video frame as image", params: ["input","opts"], tags: ["media","ffmpeg","image"], output_schema: schema_str!()),
        tool_spec!("MediaConcat", summary: "Concatenate media files", params: ["inputs","opts"], tags: ["media","ffmpeg"], output_schema: schema_str!()),
        tool_spec!("MediaPipeline", summary: "Run arbitrary ffmpeg pipeline (builder)", params: ["desc"], tags: ["media","ffmpeg","pipeline"], output_schema: schema_str!()),
        tool_spec!("MediaExtractAudio", summary: "Extract audio track to format", params: ["input","opts"], tags: ["media","ffmpeg","audio"], output_schema: schema_str!()),
        tool_spec!("MediaMux", summary: "Mux separate video+audio into container", params: ["video","audio","opts"], tags: ["media","ffmpeg"], output_schema: schema_str!()),
    ]);
}

fn failure(tag: &str, msg: &str) -> Value {
    Value::Assoc(
        vec![
            ("message".to_string(), Value::String(msg.to_string())),
            ("tag".to_string(), Value::String(tag.to_string())),
        ]
        .into_iter()
        .collect(),
    )
}

fn base64url_encode(data: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(data)
}

fn base64url_decode(s: &str) -> Result<Vec<u8>, String> {
    use base64::Engine as _;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(s).map_err(|e| e.to_string())
}

fn ffmpeg_bin() -> String {
    std::env::var("FFMPEG").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| "ffmpeg".into())
}
fn ffprobe_bin() -> String {
    std::env::var("FFPROBE").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| "ffprobe".into())
}

fn write_temp(prefix: &str, ext: &str, bytes: &[u8]) -> Result<std::path::PathBuf, String> {
    let mut p = std::env::temp_dir();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    p.push(format!("{}_{}.{}", prefix, ts, ext));
    std::fs::write(&p, bytes).map_err(|e| e.to_string())?;
    Ok(p)
}

fn read_input_bytes(ev: &mut Evaluator, v: Value) -> Result<Vec<u8>, String> {
    match ev.eval(v) {
        Value::Assoc(m) => {
            if let Some(Value::String(p)) = m.get("Path").or(m.get("path")).cloned() {
                std::fs::read(&p).map_err(|e| e.to_string())
            } else if let Some(Value::String(b)) = m.get("Bytes").or(m.get("bytes")).cloned() {
                let enc = m
                    .get("Encoding")
                    .or(m.get("encoding"))
                    .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
                    .unwrap_or_else(|| "base64url".into());
                match enc.as_str() {
                    "base64url" | "base64" | "b64" => base64url_decode(&b),
                    _ => Err("Unsupported encoding".into()),
                }
            } else {
                Err("Missing Path or Bytes".into())
            }
        }
        Value::String(s) | Value::Symbol(s) => match base64url_decode(&s) {
            Ok(b) => Ok(b),
            Err(_) => {
                if std::path::Path::new(&s).exists() {
                    std::fs::read(&s).map_err(|e| e.to_string())
                } else {
                    Err("Invalid input: provide {Path} or base64 bytes".into())
                }
            }
        },
        _ => Err("Invalid input".into()),
    }
}

fn json_to_value(j: serde_json::Value) -> Value {
    match j {
        serde_json::Value::Null => Value::Symbol("Null".into()),
        serde_json::Value::Bool(b) => Value::Boolean(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Integer(i)
            } else if let Some(f) = n.as_f64() {
                Value::Real(f)
            } else {
                Value::String(n.to_string())
            }
        }
        serde_json::Value::String(s) => Value::String(s),
        serde_json::Value::Array(xs) => Value::List(xs.into_iter().map(json_to_value).collect()),
        serde_json::Value::Object(m) => {
            Value::Assoc(m.into_iter().map(|(k, v)| (k, json_to_value(v))).collect())
        }
    }
}

fn media_probe(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("MediaProbe".into())), args };
    }
    let bytes = match read_input_bytes(ev, args[0].clone()) {
        Ok(b) => b,
        Err(e) => return failure("Media::input", &format!("MediaProbe: {}", e)),
    };
    let in_path = match write_temp("lyra_media_in", "bin", &bytes) {
        Ok(p) => p,
        Err(e) => return failure("Media::temp", &e),
    };
    let out = std::process::Command::new(ffprobe_bin())
        .arg("-v")
        .arg("error")
        .arg("-print_format")
        .arg("json")
        .arg("-show_format")
        .arg("-show_streams")
        .arg(&in_path)
        .output();
    let _ = std::fs::remove_file(&in_path);
    match out {
        Ok(o) if o.status.success() => {
            match serde_json::from_slice::<serde_json::Value>(&o.stdout) {
                Ok(j) => json_to_value(j),
                Err(e) => failure("Media::probe", &e.to_string()),
            }
        }
        Ok(o) => failure("Media::probe", &String::from_utf8_lossy(&o.stderr).to_string()),
        Err(e) => failure("Media::probe", &e.to_string()),
    }
}

fn media_transcode(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // MediaTranscode(input, { format?: mp4|mp3|ogg|webm|flac|wav, videoCodec?, audioCodec?, crf?, bitrateVideoKbps?, bitrateAudioKbps?, scale?: {width,height}, fps?, startSec?, durationSec?, audioChannels?, audioSampleRate?, copyVideo?: bool, copyAudio?: bool, output?: {path, mkdirs?} })
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("MediaTranscode".into())), args };
    }
    let bytes = match read_input_bytes(ev, args[0].clone()) {
        Ok(b) => b,
        Err(e) => return failure("Media::input", &format!("MediaTranscode: {}", e)),
    };
    let opts = match ev.eval(args[1].clone()) {
        Value::Assoc(m) => m,
        _ => std::collections::HashMap::new(),
    };
    // Prepare input temp
    let in_path = match write_temp("lyra_media_in", "bin", &bytes) {
        Ok(p) => p,
        Err(e) => return failure("Media::temp", &e),
    };
    // Output handling
    let out_format = opts
        .get("format")
        .or(opts.get("Format"))
        .and_then(|v| if let Value::String(s) = v { Some(s.to_lowercase()) } else { None })
        .unwrap_or_else(|| "mp4".into());
    let out_path = if let Some(Value::Assoc(m)) = opts.get("output").or(opts.get("Output")) {
        if let Some(Value::String(p)) = m.get("path").or(m.get("Path")) {
            p.clone()
        } else {
            format!("{}.{}", in_path.display(), out_format)
        }
    } else {
        format!("{}.{}", in_path.display(), out_format)
    };
    // Build command
    let mut cmd = std::process::Command::new(ffmpeg_bin());
    cmd.arg("-y").arg("-hide_banner").arg("-loglevel").arg("error").arg("-i").arg(&in_path);
    // Trim
    if let Some(ss) = opts.get("startSec").or(opts.get("StartSec")).and_then(|v| match v {
        Value::Real(f) => Some(*f),
        Value::Integer(i) => Some(*i as f64),
        _ => None,
    }) {
        cmd.arg("-ss").arg(format!("{}", ss));
    }
    if let Some(t) = opts.get("durationSec").or(opts.get("DurationSec")).and_then(|v| match v {
        Value::Real(f) => Some(*f),
        Value::Integer(i) => Some(*i as f64),
        _ => None,
    }) {
        cmd.arg("-t").arg(format!("{}", t));
    }
    // Video
    if let Some(Value::Boolean(copy)) = opts.get("copyVideo").or(opts.get("CopyVideo")) {
        if *copy {
            cmd.arg("-c:v").arg("copy");
        }
    }
    if let Some(Value::String(vc)) = opts.get("videoCodec").or(opts.get("VideoCodec")) {
        cmd.arg("-c:v").arg(vc);
    }
    if let Some(Value::Integer(crf)) = opts.get("crf").or(opts.get("Crf")) {
        cmd.arg("-crf").arg(crf.to_string());
    }
    if let Some(Value::Integer(bv)) = opts.get("bitrateVideoKbps").or(opts.get("BitrateVideoKbps"))
    {
        cmd.arg("-b:v").arg(format!("{}k", bv));
    }
    if let Some(Value::Integer(fps)) = opts.get("fps").or(opts.get("Fps")) {
        cmd.arg("-r").arg(fps.to_string());
    }
    if let Some(Value::Assoc(sc)) = opts.get("scale").or(opts.get("Scale")) {
        let w = sc
            .get("width")
            .or(sc.get("Width"))
            .and_then(|v| if let Value::Integer(i) = v { Some(*i as i64) } else { None })
            .unwrap_or(-1);
        let h = sc
            .get("height")
            .or(sc.get("Height"))
            .and_then(|v| if let Value::Integer(i) = v { Some(*i as i64) } else { None })
            .unwrap_or(-1);
        cmd.arg("-vf").arg(format!("scale={}:{},flags=bicubic", w, h));
    }
    // Audio
    if let Some(Value::Boolean(copy)) = opts.get("copyAudio").or(opts.get("CopyAudio")) {
        if *copy {
            cmd.arg("-c:a").arg("copy");
        }
    }
    if let Some(Value::String(ac)) = opts.get("audioCodec").or(opts.get("AudioCodec")) {
        cmd.arg("-c:a").arg(ac);
    }
    if let Some(Value::Integer(ba)) = opts.get("bitrateAudioKbps").or(opts.get("BitrateAudioKbps"))
    {
        cmd.arg("-b:a").arg(format!("{}k", ba));
    }
    if let Some(Value::Integer(ch)) = opts.get("audioChannels").or(opts.get("AudioChannels")) {
        cmd.arg("-ac").arg(ch.to_string());
    }
    if let Some(Value::Integer(sr)) = opts.get("audioSampleRate").or(opts.get("AudioSampleRate")) {
        cmd.arg("-ar").arg(sr.to_string());
    }
    // Format/output
    cmd.arg(&out_path);
    let status = cmd.status();
    let _ = std::fs::remove_file(&in_path);
    match status {
        Ok(s) if s.success() => {
            let out_bytes = std::fs::read(&out_path).map_err(|e| e.to_string());
            let _ = std::fs::remove_file(&out_path);
            match out_bytes {
                Ok(b) => Value::String(base64url_encode(&b)),
                Err(e) => failure("Media::read", &e),
            }
        }
        Ok(s) => failure("Media::ffmpeg", &format!("Exit code {}", s)),
        Err(e) => failure("Media::ffmpeg", &e.to_string()),
    }
}

fn media_thumbnail(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // MediaThumbnail(input, { atSec?: f64, width?, height?, format?: png|jpg|webp }) -> base64 image
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("MediaThumbnail".into())), args };
    }
    let bytes = match read_input_bytes(ev, args[0].clone()) {
        Ok(b) => b,
        Err(e) => return failure("Media::input", &format!("MediaThumbnail: {}", e)),
    };
    let opts = match ev.eval(args[1].clone()) {
        Value::Assoc(m) => m,
        _ => std::collections::HashMap::new(),
    };
    let in_path = match write_temp("lyra_media_in", "bin", &bytes) {
        Ok(p) => p,
        Err(e) => return failure("Media::temp", &e),
    };
    let fmt = opts
        .get("format")
        .or(opts.get("Format"))
        .and_then(|v| if let Value::String(s) = v { Some(s.to_lowercase()) } else { None })
        .unwrap_or_else(|| "png".into());
    let mut out_path = in_path.clone();
    out_path.set_file_name(format!(
        "{}_thumb.{}",
        in_path.file_stem().and_then(|s| s.to_str()).unwrap_or("out"),
        fmt
    ));
    let mut cmd = std::process::Command::new(ffmpeg_bin());
    cmd.arg("-y").arg("-hide_banner").arg("-loglevel").arg("error");
    if let Some(ts) = opts.get("atSec").or(opts.get("AtSec")).and_then(|v| match v {
        Value::Real(f) => Some(*f),
        Value::Integer(i) => Some(*i as f64),
        _ => None,
    }) {
        cmd.arg("-ss").arg(format!("{}", ts));
    }
    cmd.arg("-i").arg(&in_path).arg("-frames:v").arg("1");
    if let Some(Value::Integer(w)) = opts.get("width").or(opts.get("Width")) {
        let h = opts
            .get("height")
            .or(opts.get("Height"))
            .and_then(|v| if let Value::Integer(i) = v { Some(*i) } else { None })
            .unwrap_or(-1);
        cmd.arg("-vf").arg(format!("scale={}:{}", w, h));
    }
    cmd.arg(&out_path);
    let status = cmd.status();
    let _ = std::fs::remove_file(&in_path);
    match status {
        Ok(s) if s.success() => {
            let out_bytes = std::fs::read(&out_path).map_err(|e| e.to_string());
            let _ = std::fs::remove_file(&out_path);
            match out_bytes {
                Ok(b) => Value::String(base64url_encode(&b)),
                Err(e) => failure("Media::read", &e),
            }
        }
        Ok(_) => failure("Media::ffmpeg", "Failed to generate thumbnail"),
        Err(e) => failure("Media::ffmpeg", &e.to_string()),
    }
}

fn media_concat(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // MediaConcat([inputs], { format?: mp4|mp3|..., copy?: bool, reencode?: bool }) -> base64
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("MediaConcat".into())), args };
    }
    let list = match ev.eval(args[0].clone()) {
        Value::List(xs) => xs,
        _ => return failure("Media::concat", "Expected list of inputs"),
    };
    let opts = if args.len() > 1 {
        match ev.eval(args[1].clone()) {
            Value::Assoc(m) => m,
            _ => std::collections::HashMap::new(),
        }
    } else {
        std::collections::HashMap::new()
    };
    let mut file_paths: Vec<std::path::PathBuf> = Vec::new();
    for v in list {
        let b = match read_input_bytes(ev, v) {
            Ok(bb) => bb,
            Err(e) => return failure("Media::input", &e),
        };
        let p = match write_temp("lyra_media_in", "bin", &b) {
            Ok(pp) => pp,
            Err(e) => return failure("Media::temp", &e),
        };
        file_paths.push(p);
    }
    // Build concat list file
    let mut list_path = std::env::temp_dir();
    list_path.push(format!(
        "lyra_concat_{}.txt",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    let mut lf = String::new();
    for p in &file_paths {
        lf.push_str(&format!("file '{}", p.display()));
        lf.push_str("'\n");
    }
    if let Err(e) = std::fs::write(&list_path, lf) {
        return failure("Media::temp", &e.to_string());
    }
    let format = opts
        .get("format")
        .or(opts.get("Format"))
        .and_then(|v| if let Value::String(s) = v { Some(s.to_lowercase()) } else { None })
        .unwrap_or_else(|| "mp4".into());
    let mut out_path = list_path.clone();
    out_path.set_file_name(format!("concat_out.{}", format));
    let mut cmd = std::process::Command::new(ffmpeg_bin());
    cmd.arg("-y")
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-f")
        .arg("concat")
        .arg("-safe")
        .arg("0")
        .arg("-i")
        .arg(&list_path);
    let reencode = opts
        .get("reencode")
        .or(opts.get("Reencode"))
        .and_then(|v| if let Value::Boolean(b) = v { Some(*b) } else { None })
        .unwrap_or(false);
    if !reencode {
        cmd.arg("-c").arg("copy");
    }
    cmd.arg(&out_path);
    let status = cmd.status();
    let _ = std::fs::remove_file(&list_path);
    for p in &file_paths {
        let _ = std::fs::remove_file(p);
    }
    match status {
        Ok(s) if s.success() => {
            let out_bytes = std::fs::read(&out_path).map_err(|e| e.to_string());
            let _ = std::fs::remove_file(&out_path);
            match out_bytes {
                Ok(b) => Value::String(base64url_encode(&b)),
                Err(e) => failure("Media::read", &e),
            }
        }
        Ok(_) => failure("Media::ffmpeg", "Concat failed"),
        Err(e) => failure("Media::ffmpeg", &e.to_string()),
    }
}

fn media_pipeline(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // MediaPipeline({ inputs:[path|bytes...], args?: ["-vf","scale=...",...], videoFilters?: [..], audioFilters?: [..], maps?: [..], output?: { path?, format? } }) -> bytes or {path}
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("MediaPipeline".into())), args };
    }
    let desc = match ev.eval(args[0].clone()) {
        Value::Assoc(m) => m,
        _ => return failure("Media::pipeline", "Expected assoc desc"),
    };
    // Collect temp inputs
    let mut input_paths: Vec<std::path::PathBuf> = Vec::new();
    if let Some(Value::List(xs)) = desc.get("inputs").or(desc.get("Inputs")) {
        for v in xs {
            let b = match read_input_bytes(ev, v.clone()) {
                Ok(bb) => bb,
                Err(e) => return failure("Media::input", &e),
            };
            let p = match write_temp("lyra_media_in", "bin", &b) {
                Ok(pp) => pp,
                Err(e) => return failure("Media::temp", &e),
            };
            input_paths.push(p);
        }
    } else {
        return failure("Media::pipeline", "Missing inputs");
    }
    let mut args_list: Vec<String> =
        if let Some(Value::List(xs)) = desc.get("args").or(desc.get("Args")) {
            xs.iter()
                .filter_map(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
                .collect()
        } else {
            Vec::new()
        };
    // Structured filters: either full filterGraph or simple -vf/-af
    let mut _used_filter_complex = false;
    if let Some(fg) = desc.get("filterGraph").or(desc.get("FilterGraph")) {
        if let Some(fc) = build_filter_complex(fg.clone()) {
            args_list.push("-filter_complex".into());
            args_list.push(fc);
            _used_filter_complex = true;
        }
    } else {
        if let Some(Value::List(vfs)) = desc.get("videoFilters").or(desc.get("VideoFilters")) {
            if let Some(vf_str) = build_video_filters(vfs) {
                args_list.push("-vf".into());
                args_list.push(vf_str);
            }
        }
        if let Some(Value::List(afs)) = desc.get("audioFilters").or(desc.get("AudioFilters")) {
            if let Some(af_str) = build_audio_filters(afs) {
                args_list.push("-af".into());
                args_list.push(af_str);
            }
        }
    }
    if let Some(Value::List(maps)) = desc.get("maps").or(desc.get("Maps")) {
        for m in maps {
            if let Value::String(s) = m {
                args_list.push("-map".into());
                args_list.push(s.clone());
            }
        }
    }
    // Output target
    let output_assoc = desc.get("output").or(desc.get("Output")).and_then(|v| {
        if let Value::Assoc(m) = v {
            Some(m.clone())
        } else {
            None
        }
    });
    // Build command
    let mut cmd = std::process::Command::new(ffmpeg_bin());
    cmd.arg("-y").arg("-hide_banner").arg("-loglevel").arg("error");
    for p in &input_paths {
        cmd.arg("-i").arg(p);
    }
    for a in &args_list {
        cmd.arg(a);
    }
    let mut out_temp: Option<std::path::PathBuf> = None;
    if let Some(m) = output_assoc {
        if let Some(Value::String(path)) = m.get("path").or(m.get("Path")) {
            cmd.arg(path);
        } else {
            let fmt = m
                .get("format")
                .or(m.get("Format"))
                .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
                .unwrap_or_else(|| "mp4".into());
            let mut op = std::env::temp_dir();
            op.push(format!(
                "lyra_media_out_{}.{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                fmt
            ));
            cmd.arg(&op);
            out_temp = Some(op);
        }
    } else {
        let mut op = std::env::temp_dir();
        op.push(format!(
            "lyra_media_out_{}.mp4",
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        cmd.arg(&op);
        out_temp = Some(op);
    }
    let status = cmd.status();
    for p in &input_paths {
        let _ = std::fs::remove_file(p);
    }
    match status {
        Ok(s) if s.success() => {
            if let Some(p) = out_temp {
                match std::fs::read(&p) {
                    Ok(b) => {
                        let _ = std::fs::remove_file(&p);
                        Value::String(base64url_encode(&b))
                    }
                    Err(e) => failure("Media::read", &e.to_string()),
                }
            } else {
                Value::Boolean(true)
            }
        }
        Ok(_) => failure("Media::ffmpeg", "Pipeline failed"),
        Err(e) => failure("Media::ffmpeg", &e.to_string()),
    }
}

fn build_filter_complex(graph_v: Value) -> Option<String> {
    match graph_v {
        Value::Assoc(m) => {
            if let Some(Value::List(chains)) = m.get("chains").or(m.get("Chains")) {
                let mut out_parts: Vec<String> = Vec::new();
                for ch in chains {
                    if let Value::Assoc(cm) = ch {
                        if let Some(s) = render_chain(cm) {
                            out_parts.push(s);
                        }
                    }
                }
                if out_parts.is_empty() {
                    None
                } else {
                    Some(out_parts.join(";"))
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

fn render_chain(cm: &std::collections::HashMap<String, Value>) -> Option<String> {
    // Chain: { in:["0:v"|"v0",...], filters:[ {op, args?}... ] | {op, args?}, out?:"v1" }
    let inputs: Vec<String> = if let Some(Value::List(ins)) =
        cm.get("in").or(cm.get("In")).or(cm.get("inputs")).or(cm.get("Inputs"))
    {
        ins.iter()
            .filter_map(|v| if let Value::String(s) = v { Some(format!("[{}]", s)) } else { None })
            .collect()
    } else {
        Vec::new()
    };
    let out_lbl = cm
        .get("out")
        .or(cm.get("Out"))
        .or(cm.get("label"))
        .or(cm.get("Label"))
        .and_then(|v| if let Value::String(s) = v { Some(format!("[{}]", s)) } else { None });
    // filters can be a list or a single
    let filters_str = if let Some(Value::List(fs)) = cm.get("filters").or(cm.get("Filters")) {
        let mut ops: Vec<String> = Vec::new();
        for f in fs {
            if let Some(s) = render_filter(f) {
                ops.push(s);
            }
        }
        ops.join(",")
    } else if let Some(f) = cm.get("op").or(cm.get("Op")) {
        render_filter(f).unwrap_or_default()
    } else {
        String::new()
    };
    if filters_str.is_empty() {
        return None;
    }
    let mut part = String::new();
    for i in inputs {
        part.push_str(&i);
    }
    part.push_str(&filters_str);
    if let Some(lbl) = out_lbl {
        part.push_str(&lbl);
    }
    Some(part)
}

fn render_filter(fv: &Value) -> Option<String> {
    match fv {
        Value::Assoc(m) => {
            let op = m.get("op").or(m.get("Op")).and_then(|v| {
                if let Value::String(s) = v {
                    Some(s.clone())
                } else {
                    None
                }
            })?;
            let args = m.get("args").or(m.get("Args"));
            let arg_str = render_args(args);
            if arg_str.is_empty() {
                Some(op)
            } else {
                Some(format!("{}={}", op, arg_str))
            }
        }
        Value::String(s) => Some(s.clone()),
        _ => None,
    }
}

fn render_args(args: Option<&Value>) -> String {
    match args {
        Some(Value::Assoc(m)) => {
            let mut kvs: Vec<String> = Vec::new();
            for (k, v) in m.iter() {
                let val = match v {
                    Value::Integer(i) => i.to_string(),
                    Value::Real(f) => f.to_string(),
                    Value::String(s) => s.clone(),
                    Value::Boolean(b) => {
                        if *b {
                            "1".into()
                        } else {
                            "0".into()
                        }
                    }
                    _ => lyra_core::pretty::format_value(v),
                };
                kvs.push(format!("{}={}", k, val));
            }
            kvs.join(":")
        }
        Some(Value::String(s)) => s.clone(),
        _ => String::new(),
    }
}

fn build_video_filters(vfs: &Vec<Value>) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    for v in vfs {
        if let Value::Assoc(m) = v {
            let op = m
                .get("op")
                .or(m.get("Op"))
                .and_then(|v| if let Value::String(s) = v { Some(s.to_lowercase()) } else { None })
                .unwrap_or_default();
            match op.as_str() {
                "scale" => {
                    let w = m
                        .get("width")
                        .or(m.get("w"))
                        .or(m.get("Width"))
                        .and_then(|v| match v {
                            Value::Integer(i) => Some(i.to_string()),
                            Value::String(s) => Some(s.clone()),
                            _ => None,
                        })
                        .unwrap_or("-1".into());
                    let h = m
                        .get("height")
                        .or(m.get("h"))
                        .or(m.get("Height"))
                        .and_then(|v| match v {
                            Value::Integer(i) => Some(i.to_string()),
                            Value::String(s) => Some(s.clone()),
                            _ => None,
                        })
                        .unwrap_or("-1".into());
                    let flags = m
                        .get("flags")
                        .and_then(|v| {
                            if let Value::String(s) = v {
                                Some(format!(":flags={}", s))
                            } else {
                                None
                            }
                        })
                        .unwrap_or_default();
                    parts.push(format!("scale={}{}{}", w, ":", h) + &flags);
                }
                "fps" => {
                    if let Some(Value::Integer(f)) =
                        m.get("fps").or(m.get("Fps")).or(m.get("n")).or(m.get("N"))
                    {
                        parts.push(format!("fps={}", f));
                    }
                }
                "crop" => {
                    let w = get_int_or(&m, &["w", "width", "Width"], -1);
                    let h = get_int_or(&m, &["h", "height", "Height"], -1);
                    let x = get_int_or(&m, &["x", "X"], 0);
                    let y = get_int_or(&m, &["y", "Y"], 0);
                    parts.push(format!("crop={}:{}:{}:{}", w, h, x, y));
                }
                "pad" => {
                    let w = get_int_or(&m, &["w", "width", "Width"], -1);
                    let h = get_int_or(&m, &["h", "height", "Height"], -1);
                    let x = get_int_or(&m, &["x", "X"], 0);
                    let y = get_int_or(&m, &["y", "Y"], 0);
                    let color = m
                        .get("color")
                        .or(m.get("Color"))
                        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
                        .unwrap_or("black".into());
                    parts.push(format!("pad={}:{}:{}:{}:color={}", w, h, x, y, color));
                }
                "rotate" => {
                    if let Some(Value::Real(a)) = m.get("angle").or(m.get("Angle")) {
                        parts.push(format!("rotate={}", a));
                    } else if let Some(Value::Integer(a)) = m.get("angle").or(m.get("Angle")) {
                        parts.push(format!("rotate={}", *a as f64));
                    }
                }
                _ => {}
            }
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(","))
    }
}

fn get_int_or(m: &std::collections::HashMap<String, Value>, keys: &[&str], default: i64) -> i64 {
    for k in keys {
        if let Some(Value::Integer(i)) = m.get(*k) {
            return *i;
        }
    }
    default
}

fn build_audio_filters(afs: &Vec<Value>) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    for v in afs {
        if let Value::Assoc(m) = v {
            let op = m
                .get("op")
                .or(m.get("Op"))
                .and_then(|v| if let Value::String(s) = v { Some(s.to_lowercase()) } else { None })
                .unwrap_or_default();
            match op.as_str() {
                "aresample" => {
                    if let Some(Value::Integer(sr)) =
                        m.get("sampleRate").or(m.get("SampleRate")).or(m.get("sr"))
                    {
                        parts.push(format!("aresample={}", sr));
                    }
                }
                "volume" => {
                    if let Some(Value::Real(db)) = m.get("db").or(m.get("dB")) {
                        parts.push(format!("volume={}dB", db));
                    } else if let Some(Value::Integer(db)) = m.get("db").or(m.get("dB")) {
                        parts.push(format!("volume={}dB", db));
                    } else if let Some(Value::Real(l)) = m.get("linear").or(m.get("Linear")) {
                        parts.push(format!("volume={}", l));
                    }
                }
                "afade" => {
                    let t = m
                        .get("type")
                        .or(m.get("Type"))
                        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
                        .unwrap_or("in".into());
                    let start = m
                        .get("startSec")
                        .or(m.get("StartSec"))
                        .and_then(|v| match v {
                            Value::Real(f) => Some(*f),
                            Value::Integer(i) => Some(*i as f64),
                            _ => None,
                        })
                        .unwrap_or(0.0);
                    let dur = m
                        .get("durationSec")
                        .or(m.get("DurationSec"))
                        .and_then(|v| match v {
                            Value::Real(f) => Some(*f),
                            Value::Integer(i) => Some(*i as f64),
                            _ => None,
                        })
                        .unwrap_or(0.0);
                    parts.push(format!("afade=t={}:st={}:d={}", t, start, dur));
                }
                "atrim" => {
                    let start = m.get("startSec").or(m.get("StartSec")).and_then(|v| match v {
                        Value::Real(f) => Some(*f),
                        Value::Integer(i) => Some(*i as f64),
                        _ => None,
                    });
                    let dur = m.get("durationSec").or(m.get("DurationSec")).and_then(|v| match v {
                        Value::Real(f) => Some(*f),
                        Value::Integer(i) => Some(*i as f64),
                        _ => None,
                    });
                    match (start, dur) {
                        (Some(s), Some(d)) => {
                            parts.push(format!("atrim=start={}:duration={}", s, d))
                        }
                        (Some(s), None) => parts.push(format!("atrim=start={}", s)),
                        (None, Some(d)) => parts.push(format!("atrim=duration={}", d)),
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(","))
    }
}

fn media_extract_audio(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // MediaExtractAudio(input, { format?: mp3|ogg|flac|wav, audioCodec?, bitrateAudioKbps?, sampleRate?, channels? })
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("MediaExtractAudio".into())), args };
    }
    let bytes = match read_input_bytes(ev, args[0].clone()) {
        Ok(b) => b,
        Err(e) => return failure("Media::input", &format!("MediaExtractAudio: {}", e)),
    };
    let opts = match ev.eval(args[1].clone()) {
        Value::Assoc(m) => m,
        _ => std::collections::HashMap::new(),
    };
    let in_path = match write_temp("lyra_media_in", "bin", &bytes) {
        Ok(p) => p,
        Err(e) => return failure("Media::temp", &e),
    };
    let fmt = opts
        .get("format")
        .or(opts.get("Format"))
        .and_then(|v| if let Value::String(s) = v { Some(s.to_lowercase()) } else { None })
        .unwrap_or_else(|| "wav".into());
    let mut out_path = in_path.clone();
    out_path.set_file_name(format!(
        "{}_ea.{}",
        in_path.file_stem().and_then(|s| s.to_str()).unwrap_or("out"),
        fmt
    ));
    let mut cmd = std::process::Command::new(ffmpeg_bin());
    cmd.arg("-y")
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-i")
        .arg(&in_path)
        .arg("-vn");
    if let Some(Value::String(ac)) = opts.get("audioCodec").or(opts.get("AudioCodec")) {
        cmd.arg("-c:a").arg(ac);
    }
    if let Some(Value::Integer(b)) = opts.get("bitrateAudioKbps").or(opts.get("BitrateAudioKbps")) {
        cmd.arg("-b:a").arg(format!("{}k", b));
    }
    if let Some(Value::Integer(sr)) = opts
        .get("sampleRate")
        .or(opts.get("SampleRate"))
        .or(opts.get("audioSampleRate"))
        .or(opts.get("AudioSampleRate"))
    {
        cmd.arg("-ar").arg(sr.to_string());
    }
    if let Some(Value::Integer(ch)) = opts
        .get("channels")
        .or(opts.get("Channels"))
        .or(opts.get("audioChannels"))
        .or(opts.get("AudioChannels"))
    {
        cmd.arg("-ac").arg(ch.to_string());
    }
    cmd.arg(&out_path);
    let status = cmd.status();
    let _ = std::fs::remove_file(&in_path);
    match status {
        Ok(s) if s.success() => {
            let b = std::fs::read(&out_path).map_err(|e| e.to_string());
            let _ = std::fs::remove_file(&out_path);
            match b {
                Ok(bb) => Value::String(base64url_encode(&bb)),
                Err(e) => failure("Media::read", &e),
            }
        }
        Ok(_) => failure("Media::ffmpeg", "Extract audio failed"),
        Err(e) => failure("Media::ffmpeg", &e.to_string()),
    }
}

fn media_mux(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // MediaMux(video, audio, { format?: mp4|mkv|mov, videoCodec?, audioCodec? })
    if args.len() < 3 {
        return Value::Expr { head: Box::new(Value::Symbol("MediaMux".into())), args };
    }
    let vbytes = match read_input_bytes(ev, args[0].clone()) {
        Ok(b) => b,
        Err(e) => return failure("Media::input", &format!("MediaMux(video): {}", e)),
    };
    let abytes = match read_input_bytes(ev, args[1].clone()) {
        Ok(b) => b,
        Err(e) => return failure("Media::input", &format!("MediaMux(audio): {}", e)),
    };
    let opts = match ev.eval(args[2].clone()) {
        Value::Assoc(m) => m,
        _ => std::collections::HashMap::new(),
    };
    let vpath = match write_temp("lyra_media_v", "bin", &vbytes) {
        Ok(p) => p,
        Err(e) => return failure("Media::temp", &e),
    };
    let apath = match write_temp("lyra_media_a", "bin", &abytes) {
        Ok(p) => p,
        Err(e) => {
            let _ = std::fs::remove_file(&vpath);
            return failure("Media::temp", &e);
        }
    };
    let fmt = opts
        .get("format")
        .or(opts.get("Format"))
        .and_then(|v| if let Value::String(s) = v { Some(s.to_lowercase()) } else { None })
        .unwrap_or_else(|| "mp4".into());
    let mut out_path = vpath.clone();
    out_path.set_file_name(format!(
        "{}_mux.{}",
        vpath.file_stem().and_then(|s| s.to_str()).unwrap_or("out"),
        fmt
    ));
    let mut cmd = std::process::Command::new(ffmpeg_bin());
    cmd.arg("-y")
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-i")
        .arg(&vpath)
        .arg("-i")
        .arg(&apath);
    if let Some(Value::String(vc)) = opts.get("videoCodec").or(opts.get("VideoCodec")) {
        cmd.arg("-c:v").arg(vc);
    } else {
        cmd.arg("-c:v").arg("copy");
    }
    if let Some(Value::String(ac)) = opts.get("audioCodec").or(opts.get("AudioCodec")) {
        cmd.arg("-c:a").arg(ac);
    } else {
        cmd.arg("-c:a").arg("copy");
    }
    cmd.arg(&out_path);
    let status = cmd.status();
    let _ = std::fs::remove_file(&vpath);
    let _ = std::fs::remove_file(&apath);
    match status {
        Ok(s) if s.success() => {
            let b = std::fs::read(&out_path).map_err(|e| e.to_string());
            let _ = std::fs::remove_file(&out_path);
            match b {
                Ok(bb) => Value::String(base64url_encode(&bb)),
                Err(e) => failure("Media::read", &e),
            }
        }
        Ok(_) => failure("Media::ffmpeg", "Mux failed"),
        Err(e) => failure("Media::ffmpeg", &e.to_string()),
    }
}

pub fn register_media_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    register_if(ev, pred, "MediaProbe", media_probe as NativeFn, Attributes::empty());
    register_if(ev, pred, "MediaTranscode", media_transcode as NativeFn, Attributes::empty());
    register_if(ev, pred, "MediaThumbnail", media_thumbnail as NativeFn, Attributes::empty());
    register_if(ev, pred, "MediaConcat", media_concat as NativeFn, Attributes::empty());
    register_if(ev, pred, "MediaPipeline", media_pipeline as NativeFn, Attributes::empty());
    register_if(
        ev,
        pred,
        "MediaExtractAudio",
        media_extract_audio as NativeFn,
        Attributes::empty(),
    );
    register_if(ev, pred, "MediaMux", media_mux as NativeFn, Attributes::empty());
}
