use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use image::GenericImageView;
use lyra_runtime::Evaluator;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

#[cfg(feature = "tools")] use crate::tools::add_specs;
#[cfg(feature = "tools")] use crate::tool_spec;
#[cfg(feature = "tools")] use crate::{schema_str};
#[cfg(feature = "tools")] use std::collections::HashMap;

pub fn register_image(ev: &mut Evaluator) {
    ev.register("ImageInfo", image_info as NativeFn, Attributes::empty());
    ev.register("ImageCanvas", image_canvas as NativeFn, Attributes::empty());
    ev.register("ImageDecode", image_decode as NativeFn, Attributes::empty());
    ev.register("ImageEncode", image_encode as NativeFn, Attributes::empty());
    ev.register("ImageResize", image_resize as NativeFn, Attributes::empty());
    ev.register("ImageConvert", image_convert as NativeFn, Attributes::empty());
    ev.register("ImageCrop", image_crop as NativeFn, Attributes::empty());
    ev.register("ImagePad", image_pad as NativeFn, Attributes::empty());
    ev.register("ImageThumbnail", image_thumbnail as NativeFn, Attributes::empty());
    ev.register("ImageTransform", image_transform as NativeFn, Attributes::empty());
    ev.register("ImageSave", image_save as NativeFn, Attributes::empty());

    #[cfg(feature = "tools")]
    add_specs(vec![
        tool_spec!("ImageInfo", summary: "Read basic image info", params: ["input","opts"], tags: ["image"], output_schema: lyra_core::value::Value::Assoc(HashMap::from([(String::from("type"), lyra_core::value::Value::String(String::from("object")))]))),
        tool_spec!("ImageCanvas", summary: "Create a blank canvas (PNG)", params: ["opts"], tags: ["image"], output_schema: schema_str!()),
        tool_spec!("ImageDecode", summary: "Decode image to raw or reencoded bytes", params: ["input","opts"], tags: ["image","decode"], output_schema: schema_str!()),
        tool_spec!("ImageEncode", summary: "Encode raw pixels or reencode bytes", params: ["input","encoding"], tags: ["image","encode"], output_schema: schema_str!()),
        tool_spec!("ImageResize", summary: "Resize image (contain/cover)", params: ["input","opts"], tags: ["image","transform"], output_schema: schema_str!()),
        tool_spec!("ImageCrop", summary: "Crop image by rect or gravity", params: ["input","opts"], tags: ["image","transform"], output_schema: schema_str!()),
        tool_spec!("ImagePad", summary: "Pad image to target size", params: ["input","opts"], tags: ["image","transform"], output_schema: schema_str!()),
        tool_spec!("ImageConvert", summary: "Convert image format", params: ["input","format","opts"], tags: ["image"], output_schema: schema_str!()),
        tool_spec!("ImageThumbnail", summary: "Create thumbnail (cover)", params: ["input","opts"], tags: ["image","optimize"], output_schema: schema_str!()),
        tool_spec!("ImageTransform", summary: "Apply pipeline of operations", params: ["input","pipeline"], tags: ["image","pipeline"], output_schema: schema_str!()),
        tool_spec!("ImageSave", summary: "Encode and write image to path", params: ["input","output","encoding"], tags: ["image","io"], output_schema: lyra_core::value::Value::Assoc(HashMap::from([(String::from("type"), lyra_core::value::Value::String(String::from("object")))]))),
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

fn parse_color(v: Option<&Value>) -> image::Rgba<u8> {
    if let Some(Value::String(s)) = v {
        // Accept #RRGGBB or #RRGGBBAA
        let cs = s.trim();
        if let Some(hex) = cs.strip_prefix('#') {
            let bytes = hex.as_bytes();
            if bytes.len() == 6 || bytes.len() == 8 {
                let h = |i| -> u8 {
                    fn val(c: u8) -> u8 { match c { b'0'..=b'9' => c-b'0', b'a'..=b'f' => c-b'a'+10, b'A'..=b'F'=>c-b'A'+10, _=>0 } }
                    (val(bytes[i])<<4) | val(bytes[i+1])
                };
                let r = h(0); let g = h(2); let b = h(4); let a = if bytes.len()==8 { h(6) } else { 255 };
                return image::Rgba([r,g,b,a]);
            }
        }
    }
    image::Rgba([255,255,255,255])
}

fn read_input_bytes(ev: &mut Evaluator, v: Value) -> Result<Vec<u8>, String> {
    match ev.eval(v) {
        Value::Assoc(m) => {
            if let Some(Value::String(p)) = m.get("Path") {
                std::fs::read(p).map_err(|e| e.to_string())
            } else if let Some(Value::String(b)) = m.get("Bytes") {
                let enc = m.get("Encoding").and_then(|v| if let Value::String(s)=v { Some(s.clone()) } else { None }).unwrap_or_else(|| "base64url".into());
                match enc.as_str() {
                    "base64url" | "base64" | "b64" => base64url_decode(b),
                    _ => Err("Unsupported encoding".into()),
                }
            } else { Err("Missing Path or Bytes".into()) }
        }
        Value::String(s) | Value::Symbol(s) => {
            // Try base64url first
            match base64url_decode(&s) {
                Ok(b) => Ok(b),
                Err(_) => {
                    // If looks like a file path, attempt to read
                    if std::path::Path::new(&s).exists() { std::fs::read(&s).map_err(|e| e.to_string()) }
                    else { Err("Invalid input: provide {Path} or base64 bytes".into()) }
                }
            }
        }
        _ => Err("Invalid input".into()),
    }
}

fn decode_image(ev: &mut Evaluator, v: Value) -> Result<image::DynamicImage, String> {
    let bytes = read_input_bytes(ev, v)?;
    image::load_from_memory(&bytes).map_err(|e| e.to_string())
}

fn encode_image(img: &image::DynamicImage, opts: &std::collections::HashMap<String, Value>) -> Result<Vec<u8>, String> {
    let fmt = opts.get("format").or_else(|| opts.get("Format")).and_then(|v| if let Value::String(s)=v { Some(s.clone()) } else { None }).unwrap_or_else(|| "png".into());
    let mut buf = Vec::new();
    match fmt.to_lowercase().as_str() {
        "png" => {
            img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageOutputFormat::Png).map_err(|e| e.to_string())?;
        }
        "jpeg" | "jpg" => {
            let q = opts.get("quality").or_else(|| opts.get("Quality")).and_then(|v| if let Value::Integer(i)=v { Some(*i as u8) } else { None }).unwrap_or(85);
            img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageOutputFormat::Jpeg(q)).map_err(|e| e.to_string())?;
        }
        // WebP encoding support in image may vary; not guaranteed. Return error for now.
        "webp" => { return Err("WebP encode not supported yet".into()); }
        other => { return Err(format!("Unsupported format: {}", other)); }
    }
    Ok(buf)
}

fn wrap_bytes_output(bytes: Vec<u8>, opts: Option<std::collections::HashMap<String, Value>>) -> Value {
    // Currently return base64url string by default; allow { Output: { Encoding: "base64url"|"hex" } }
    let enc = opts.and_then(|m| m.get("Output").and_then(|v| if let Value::Assoc(mm)=v { mm.get("Encoding").cloned() } else { None }))
                  .and_then(|v| if let Value::String(s)=v { Some(s) } else { None })
                  .unwrap_or_else(|| "base64url".into());
    match enc.as_str() {
        "base64url" | "base64" | "b64" => Value::String(base64url_encode(&bytes)),
        _ => Value::String(base64url_encode(&bytes)),
    }
}

fn image_decode(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // ImageDecode(input, { format?: "png"|"jpeg"|"raw" })
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("ImageDecode".into())), args } }
    let img = match decode_image(ev, args[0].clone()) { Ok(i) => i, Err(e) => return failure("Image::decode", &format!("ImageDecode: {}", e)) };
    let opts = if args.len() > 1 { match ev.eval(args[1].clone()) { Value::Assoc(m)=>m, _=>std::collections::HashMap::new() } } else { std::collections::HashMap::new() };
    let fmt = opts.get("format").or_else(|| opts.get("Format")).and_then(|v| if let Value::String(s)=v { Some(s.to_lowercase()) } else { None }).unwrap_or_else(|| "png".into());
    if fmt == "raw" {
        let rgba = img.to_rgba8();
        let (w,h) = (rgba.width(), rgba.height());
        let bytes = rgba.into_raw();
        let mut m = std::collections::HashMap::new();
        m.insert("width".into(), Value::Integer(w as i64));
        m.insert("height".into(), Value::Integer(h as i64));
        m.insert("channels".into(), Value::String("rgba8".into()));
        m.insert("bytes".into(), Value::String(base64url_encode(&bytes)));
        Value::Assoc(m)
    } else {
        match encode_image(&img, &std::collections::HashMap::from([(String::from("format"), Value::String(fmt))])) {
            Ok(bytes) => Value::String(base64url_encode(&bytes)),
            Err(e) => failure("Image::encode", &format!("ImageDecode: {}", e)),
        }
    }
}

fn image_from_raw_assoc(m: &std::collections::HashMap<String, Value>) -> Result<image::DynamicImage, String> {
    let w = m.get("width").or_else(|| m.get("Width")).and_then(|v| if let Value::Integer(i)=v { Some(*i as u32) } else { None }).ok_or_else(|| "Missing width".to_string())?;
    let h = m.get("height").or_else(|| m.get("Height")).and_then(|v| if let Value::Integer(i)=v { Some(*i as u32) } else { None }).ok_or_else(|| "Missing height".to_string())?;
    let ch = m.get("channels").or_else(|| m.get("Channels")).and_then(|v| if let Value::String(s)=v { Some(s.to_lowercase()) } else { None }).unwrap_or_else(|| "rgba8".into());
    if ch != "rgba8" { return Err("Only rgba8 supported".into()); }
    let bytes_s = m.get("bytes").or_else(|| m.get("Bytes")).and_then(|v| if let Value::String(s)=v { Some(s.clone()) } else { None }).ok_or_else(|| "Missing bytes".to_string())?;
    let enc = m.get("encoding").or_else(|| m.get("Encoding")).and_then(|v| if let Value::String(s)=v { Some(s.to_lowercase()) } else { None }).unwrap_or_else(|| "base64url".into());
    let raw = match enc.as_str() { "base64url" | "base64" | "b64" => base64url_decode(&bytes_s), _ => Err("Unsupported raw encoding".into()) }?;
    if raw.len() != (w as usize)*(h as usize)*4 { return Err("Raw length mismatch".into()); }
    let img = image::RgbaImage::from_raw(w, h, raw).ok_or_else(|| "Invalid raw buffer".to_string())?;
    Ok(image::DynamicImage::ImageRgba8(img))
}

fn image_encode(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // ImageEncode(input, encoding)
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("ImageEncode".into())), args } }
    let enc_opts = match ev.eval(args[1].clone()) { Value::Assoc(m) => m, _ => std::collections::HashMap::new() };
    // Input may be raw assoc or another image input
    let img = match ev.eval(args[0].clone()) {
        Value::Assoc(m) => match image_from_raw_assoc(&m) { Ok(i) => i, Err(_) => match decode_image(ev, Value::Assoc(m)) { Ok(i)=>i, Err(e)=> return failure("Image::decode", &format!("ImageEncode: {}", e)) } },
        other => match decode_image(ev, other) { Ok(i)=>i, Err(e)=> return failure("Image::decode", &format!("ImageEncode: {}", e)) },
    };
    match encode_image(&img, &enc_opts) {
        Ok(bytes) => Value::String(base64url_encode(&bytes)),
        Err(e) => failure("Image::encode", &format!("ImageEncode: {}", e)),
    }
}

fn image_info(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("ImageInfo".into())), args } }
    match decode_image(ev, args[0].clone()) {
        Ok(img) => {
            let (w, h) = img.dimensions();
            let c = img.color();
            let has_alpha = matches!(c, image::ColorType::La8 | image::ColorType::La16 | image::ColorType::Rgba8 | image::ColorType::Rgba16);
            let mut m = std::collections::HashMap::new();
            m.insert("width".into(), Value::Integer(w as i64));
            m.insert("height".into(), Value::Integer(h as i64));
            m.insert("hasAlpha".into(), Value::Boolean(has_alpha));
            Value::Assoc(m)
        }
        Err(e) => failure("Image::decode", &format!("ImageInfo: {}", e)),
    }
}

fn image_canvas(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // ImageCanvas({ width, height, background? }) -> base64url PNG
    if args.len() != 1 { return Value::Expr { head: Box::new(Value::Symbol("ImageCanvas".into())), args } }
    let opts = match &args[0] { Value::Assoc(m) => m, _ => return Value::Expr { head: Box::new(Value::Symbol("ImageCanvas".into())), args } };
    let w = opts.get("width").or_else(|| opts.get("Width")).and_then(|v| if let Value::Integer(i)=v { Some(*i as u32) } else { None }).unwrap_or(1);
    let h = opts.get("height").or_else(|| opts.get("Height")).and_then(|v| if let Value::Integer(i)=v { Some(*i as u32) } else { None }).unwrap_or(1);
    let bg = parse_color(opts.get("background").or_else(|| opts.get("Background")));
    let mut canvas = image::RgbaImage::from_pixel(w, h, bg);
    let dynimg = image::DynamicImage::ImageRgba8(std::mem::take(&mut canvas));
    match encode_image(&dynimg, &HashMap::from([(String::from("format"), Value::String(String::from("png")))])) {
        Ok(bytes) => Value::String(base64url_encode(&bytes)),
        Err(e) => failure("Image::encode", &format!("ImageCanvas: {}", e)),
    }
}

fn image_resize(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("ImageResize".into())), args } }
    let opts = match ev.eval(args[1].clone()) { Value::Assoc(m) => m, _ => std::collections::HashMap::new() };
    let fit = opts.get("fit").or_else(|| opts.get("Fit")).and_then(|v| if let Value::String(s)=v { Some(s.to_lowercase()) } else { None }).unwrap_or_else(|| "contain".into());
    let w_opt = opts.get("width").or_else(|| opts.get("Width")).and_then(|v| if let Value::Integer(i)=v { Some(*i as u32) } else { None });
    let h_opt = opts.get("height").or_else(|| opts.get("Height")).and_then(|v| if let Value::Integer(i)=v { Some(*i as u32) } else { None });
    let kernel = opts.get("kernel").or_else(|| opts.get("Kernel")).and_then(|v| if let Value::String(s)=v { Some(s.to_lowercase()) } else { None }).unwrap_or_else(|| "lanczos3".into());
    let filter = match kernel.as_str() { "nearest" => image::imageops::FilterType::Nearest, "mitchell" => image::imageops::FilterType::CatmullRom, _ => image::imageops::FilterType::Lanczos3 };
    let img = match decode_image(ev, args[0].clone()) { Ok(i) => i, Err(e) => return failure("Image::decode", &format!("ImageResize: {}", e)) };
    let (ow, oh) = img.dimensions();
    let target_w = w_opt.unwrap_or_else(|| if let Some(h) = h_opt { ((ow as f32)*(h as f32/oh as f32)) as u32 } else { ow });
    let target_h = h_opt.unwrap_or_else(|| if let Some(w) = w_opt { ((oh as f32)*(w as f32/ow as f32)) as u32 } else { oh });
    let out_img = if fit == "cover" {
        // scale to cover then center-crop
        let scale = f32::max(target_w as f32 / ow as f32, target_h as f32 / oh as f32);
        let rw = ((ow as f32) * scale).round().max(1.0) as u32;
        let rh = ((oh as f32) * scale).round().max(1.0) as u32;
        let resized = img.resize(rw, rh, filter);
        let x = if rw > target_w { (rw - target_w)/2 } else { 0 };
        let y = if rh > target_h { (rh - target_h)/2 } else { 0 };
        image::DynamicImage::ImageRgba8(image::imageops::crop_imm(&resized, x, y, target_w.min(rw), target_h.min(rh)).to_image())
    } else {
        // contain
        let scale = f32::min(target_w as f32 / ow as f32, target_h as f32 / oh as f32);
        let rw = ((ow as f32) * scale).round().max(1.0) as u32;
        let rh = ((oh as f32) * scale).round().max(1.0) as u32;
        img.resize(rw, rh, filter)
    };
    let enc_opts = opts.get("encoding").or_else(|| opts.get("Encoding")).and_then(|v| if let Value::Assoc(m)=v { Some(m.clone()) } else { None }).unwrap_or_default();
    match encode_image(&out_img, &enc_opts) {
        Ok(bytes) => wrap_bytes_output(bytes, Some(opts)),
        Err(e) => failure("Image::encode", &format!("ImageResize: {}", e)),
    }
}

fn image_convert(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("ImageConvert".into())), args } }
    let img = match decode_image(ev, args[0].clone()) { Ok(i) => i, Err(e) => return failure("Image::decode", &format!("ImageConvert: {}", e)) };
    let mut enc_opts = std::collections::HashMap::new();
    let fmt = match &args[1] { Value::String(s) | Value::Symbol(s) => s.clone(), _ => "png".into() };
    enc_opts.insert("format".into(), Value::String(fmt));
    if args.len() > 2 {
        if let Value::Assoc(m) = ev.eval(args[2].clone()) {
            for (k,v) in m { enc_opts.insert(k, v); }
        }
    }
    match encode_image(&img, &enc_opts) {
        Ok(bytes) => wrap_bytes_output(bytes, None),
        Err(e) => failure("Image::encode", &format!("ImageConvert: {}", e)),
    }
}

fn image_thumbnail(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("ImageThumbnail".into())), args } }
    // Accept size as integer (square) or {width,height}
    let (tw, th, mut opts) = match ev.eval(args[1].clone()) {
        Value::Integer(n) => (n as u32, n as u32, std::collections::HashMap::new()),
        Value::Assoc(m) => {
            let w = m.get("size").and_then(|v| if let Value::Integer(i)=v { Some(*i as u32) } else { None })
                .or_else(|| m.get("width").or_else(|| m.get("Width")).and_then(|v| if let Value::Integer(i)=v { Some(*i as u32) } else { None }))
                .unwrap_or(256);
            let h = m.get("size").and_then(|v| if let Value::Integer(i)=v { Some(*i as u32) } else { None })
                .or_else(|| m.get("height").or_else(|| m.get("Height")).and_then(|v| if let Value::Integer(i)=v { Some(*i as u32) } else { None }))
                .unwrap_or(w);
            (w, h, m)
        }
        _ => return Value::Expr { head: Box::new(Value::Symbol("ImageThumbnail".into())), args }
    };
    // Force cover fit
    opts.insert("fit".into(), Value::String("cover".into()));
    opts.insert("width".into(), Value::Integer(tw as i64));
    opts.insert("height".into(), Value::Integer(th as i64));
    image_resize(ev, vec![args[0].clone(), Value::Assoc(opts)])
}

fn image_transform(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("ImageTransform".into())), args } }
    let mut img = match decode_image(ev, args[0].clone()) { Ok(i) => i, Err(e) => return failure("Image::decode", &format!("ImageTransform: {}", e)) };
    let pipeline = match ev.eval(args[1].clone()) { Value::Assoc(m) => m, _ => std::collections::HashMap::new() };
    let steps = pipeline.get("steps").or_else(|| pipeline.get("Steps"));
    if let Some(Value::List(ops)) = steps {
        for opv in ops {
            if let Value::Assoc(op) = opv {
                let name = op.get("op").or_else(|| op.get("Op")).and_then(|v| if let Value::String(s)=v { Some(s.to_lowercase()) } else { None }).unwrap_or_default();
                let args_map = op.get("args").or_else(|| op.get("Args")).and_then(|v| if let Value::Assoc(m)=v { Some(m.clone()) } else { None }).unwrap_or_default();
                match name.as_str() {
                    "resize" => {
                        let kernel = args_map.get("kernel").or_else(|| args_map.get("Kernel")).and_then(|v| if let Value::String(s)=v { Some(s.to_lowercase()) } else { None }).unwrap_or_else(|| "lanczos3".into());
                        let filter = match kernel.as_str() { "nearest" => image::imageops::FilterType::Nearest, "mitchell" => image::imageops::FilterType::CatmullRom, _ => image::imageops::FilterType::Lanczos3 };
                        let fit = args_map.get("fit").or_else(|| args_map.get("Fit")).and_then(|v| if let Value::String(s)=v { Some(s.to_lowercase()) } else { None }).unwrap_or_else(|| "contain".into());
                        let w_opt = args_map.get("width").or_else(|| args_map.get("Width")).and_then(|v| if let Value::Integer(i)=v { Some(*i as u32) } else { None });
                        let h_opt = args_map.get("height").or_else(|| args_map.get("Height")).and_then(|v| if let Value::Integer(i)=v { Some(*i as u32) } else { None });
                        let (ow, oh) = img.dimensions();
                        let target_w = w_opt.unwrap_or_else(|| if let Some(h) = h_opt { ((ow as f32)*(h as f32/oh as f32)) as u32 } else { ow });
                        let target_h = h_opt.unwrap_or_else(|| if let Some(w) = w_opt { ((oh as f32)*(w as f32/ow as f32)) as u32 } else { oh });
                        img = if fit == "cover" {
                            let scale = f32::max(target_w as f32 / ow as f32, target_h as f32 / oh as f32);
                            let rw = ((ow as f32) * scale).round().max(1.0) as u32;
                            let rh = ((oh as f32) * scale).round().max(1.0) as u32;
                            let resized = img.resize(rw, rh, filter);
                            let x = if rw > target_w { (rw - target_w)/2 } else { 0 };
                            let y = if rh > target_h { (rh - target_h)/2 } else { 0 };
                            image::DynamicImage::ImageRgba8(image::imageops::crop_imm(&resized, x, y, target_w.min(rw), target_h.min(rh)).to_image())
                        } else {
                            let scale = f32::min(target_w as f32 / ow as f32, target_h as f32 / oh as f32);
                            let rw = ((ow as f32) * scale).round().max(1.0) as u32;
                            let rh = ((oh as f32) * scale).round().max(1.0) as u32;
                            img.resize(rw, rh, filter)
                        };
                    }
                    "rotate" => {
                        let angle = args_map.get("angle").or_else(|| args_map.get("Angle")).and_then(|v| if let Value::Integer(i)=v { Some(*i as i64) } else { None }).unwrap_or(0);
                        img = match angle.rem_euclid(360) {
                            90 => img.rotate90(),
                            180 => img.rotate180(),
                            270 => img.rotate270(),
                            _ => img,
                        };
                    }
                    "flip" => {
                        let horiz = args_map.get("horizontal").or_else(|| args_map.get("Horizontal")).and_then(|v| if let Value::Boolean(b)=v { Some(*b) } else { None }).unwrap_or(false);
                        let vert = args_map.get("vertical").or_else(|| args_map.get("Vertical")).and_then(|v| if let Value::Boolean(b)=v { Some(*b) } else { None }).unwrap_or(false);
                        if horiz { img = img.fliph(); }
                        if vert { img = img.flipv(); }
                    }
                    _ => {}
                }
            }
        }
    }
    let enc_opts = pipeline.get("encoding").or_else(|| pipeline.get("Encoding")).and_then(|v| if let Value::Assoc(m)=v { Some(m.clone()) } else { None }).unwrap_or_default();
    match encode_image(&img, &enc_opts) {
        Ok(bytes) => wrap_bytes_output(bytes, Some(pipeline)),
        Err(e) => failure("Image::encode", &format!("ImageTransform: {}", e)),
    }
}

fn gravity_offset(container_w: u32, container_h: u32, content_w: u32, content_h: u32, gravity: &str) -> (u32, u32) {
    let mut x = match gravity {
        "west" | "northwest" | "southwest" => 0,
        "east" | "northeast" | "southeast" => container_w.saturating_sub(content_w),
        _ => ((container_w as i64 - content_w as i64) / 2).max(0) as u32,
    };
    let mut y = match gravity {
        "north" | "northwest" | "northeast" => 0,
        "south" | "southwest" | "southeast" => container_h.saturating_sub(content_h),
        _ => ((container_h as i64 - content_h as i64) / 2).max(0) as u32,
    };
    if content_w > container_w { x = 0; }
    if content_h > container_h { y = 0; }
    (x,y)
}

fn image_crop(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // ImageCrop(input, { rect?: {x,y,width,height} | width?, height?, gravity?: string, Encoding? })
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("ImageCrop".into())), args } }
    let img = match decode_image(ev, args[0].clone()) { Ok(i) => i, Err(e) => return failure("Image::decode", &format!("ImageCrop: {}", e)) };
    let opts = match ev.eval(args[1].clone()) { Value::Assoc(m) => m, _ => std::collections::HashMap::new() };
    let (ow, oh) = img.dimensions();
    let out_img = if let Some(Value::Assoc(r)) = opts.get("rect").or_else(|| opts.get("Rect")) {
        let x = r.get("x").or_else(|| r.get("X")).and_then(|v| if let Value::Integer(i)=v { Some(*i as i64) } else { None }).unwrap_or(0).clamp(0, ow as i64) as u32;
        let y = r.get("y").or_else(|| r.get("Y")).and_then(|v| if let Value::Integer(i)=v { Some(*i as i64) } else { None }).unwrap_or(0).clamp(0, oh as i64) as u32;
        let w = r.get("width").or_else(|| r.get("Width")).and_then(|v| if let Value::Integer(i)=v { Some(*i as i64) } else { None }).unwrap_or(ow as i64).clamp(1, ow as i64 - x as i64) as u32;
        let h = r.get("height").or_else(|| r.get("Height")).and_then(|v| if let Value::Integer(i)=v { Some(*i as i64) } else { None }).unwrap_or(oh as i64).clamp(1, oh as i64 - y as i64) as u32;
        image::DynamicImage::ImageRgba8(image::imageops::crop_imm(&img, x, y, w, h).to_image())
    } else {
        // Gravity-based crop to width/height without scaling
        let tw = opts.get("width").or_else(|| opts.get("Width")).and_then(|v| if let Value::Integer(i)=v { Some(*i as u32) } else { None }).unwrap_or(ow);
        let th = opts.get("height").or_else(|| opts.get("Height")).and_then(|v| if let Value::Integer(i)=v { Some(*i as u32) } else { None }).unwrap_or(oh);
        let gravity = opts.get("gravity").or_else(|| opts.get("Gravity")).and_then(|v| if let Value::String(s)=v { Some(s.to_lowercase()) } else { None }).unwrap_or_else(|| "center".into());
        let w = tw.min(ow); let h = th.min(oh);
        let (x,y) = gravity_offset(ow, oh, w, h, &gravity);
        image::DynamicImage::ImageRgba8(image::imageops::crop_imm(&img, x, y, w, h).to_image())
    };
    let enc_opts = opts.get("encoding").or_else(|| opts.get("Encoding")).and_then(|v| if let Value::Assoc(m)=v { Some(m.clone()) } else { None }).unwrap_or_default();
    match encode_image(&out_img, &enc_opts) {
        Ok(bytes) => wrap_bytes_output(bytes, Some(opts)),
        Err(e) => failure("Image::encode", &format!("ImageCrop: {}", e)),
    }
}

fn image_pad(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // ImagePad(input, { width, height, background?: color, gravity?: string, downscale?: bool, Encoding? })
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("ImagePad".into())), args } }
    let img = match decode_image(ev, args[0].clone()) { Ok(i) => i, Err(e) => return failure("Image::decode", &format!("ImagePad: {}", e)) };
    let opts = match ev.eval(args[1].clone()) { Value::Assoc(m) => m, _ => std::collections::HashMap::new() };
    let tw = opts.get("width").or_else(|| opts.get("Width")).and_then(|v| if let Value::Integer(i)=v { Some(*i as u32) } else { None });
    let th = opts.get("height").or_else(|| opts.get("Height")).and_then(|v| if let Value::Integer(i)=v { Some(*i as u32) } else { None });
    let (ow, oh) = img.dimensions();
    let (tw, th) = (tw.unwrap_or(ow), th.unwrap_or(oh));
    let gravity = opts.get("gravity").or_else(|| opts.get("Gravity")).and_then(|v| if let Value::String(s)=v { Some(s.to_lowercase()) } else { None }).unwrap_or_else(|| "center".into());
    let bg = parse_color(opts.get("background").or_else(|| opts.get("Background")));
    let kernel = opts.get("kernel").or_else(|| opts.get("Kernel")).and_then(|v| if let Value::String(s)=v { Some(s.to_lowercase()) } else { None }).unwrap_or_else(|| "lanczos3".into());
    let filter = match kernel.as_str() { "nearest" => image::imageops::FilterType::Nearest, "mitchell" => image::imageops::FilterType::CatmullRom, _ => image::imageops::FilterType::Lanczos3 };
    // If the image is larger than target, scale it down to fit (contain)
    let mut content = if ow > tw || oh > th {
        let scale = f32::min(tw as f32 / ow as f32, th as f32 / oh as f32);
        let rw = ((ow as f32) * scale).round().max(1.0) as u32;
        let rh = ((oh as f32) * scale).round().max(1.0) as u32;
        img.resize(rw, rh, filter)
    } else { img.clone() };
    // Create background canvas and overlay
    let mut canvas = image::RgbaImage::from_pixel(tw, th, bg);
    // Ensure overlay is RGBA8
    let overlay = content.to_rgba8();
    let (cw, ch) = (overlay.width(), overlay.height());
    let (x, y) = gravity_offset(tw, th, cw, ch, &gravity);
    image::imageops::overlay(&mut canvas, &image::DynamicImage::ImageRgba8(overlay), x as i64, y as i64);
    let out_img = image::DynamicImage::ImageRgba8(canvas);
    let enc_opts = opts.get("encoding").or_else(|| opts.get("Encoding")).and_then(|v| if let Value::Assoc(m)=v { Some(m.clone()) } else { None }).unwrap_or_default();
    match encode_image(&out_img, &enc_opts) {
        Ok(bytes) => wrap_bytes_output(bytes, Some(opts)),
        Err(e) => failure("Image::encode", &format!("ImagePad: {}", e)),
    }
}

fn infer_format_from_ext(path: &str) -> Option<String> {
    std::path::Path::new(path).extension().and_then(|e| e.to_str()).map(|s| s.to_lowercase()).and_then(|ext| {
        match ext.as_str() { "png" => Some("png".into()), "jpg" | "jpeg" => Some("jpeg".into()), _ => None }
    })
}

fn image_save(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // ImageSave(input, output, encoding?) -> { path }
    if args.len() < 2 { return Value::Expr { head: Box::new(Value::Symbol("ImageSave".into())), args } }
    let img = match decode_image(ev, args[0].clone()) { Ok(i) => i, Err(e) => return failure("Image::decode", &format!("ImageSave: {}", e)) };
    let (out_path, mkdirs) = match ev.eval(args[1].clone()) {
        Value::String(s) | Value::Symbol(s) => (s, false),
        Value::Assoc(m) => {
            let p = m.get("path").or_else(|| m.get("Path")).and_then(|v| if let Value::String(s)=v { Some(s.clone()) } else { None });
            let mk = m.get("mkdirs").or_else(|| m.get("Mkdirs")).and_then(|v| if let Value::Boolean(b)=v { Some(*b) } else { None }).unwrap_or(false);
            match p { Some(pp)=> (pp, mk), None => return failure("Image::save", "Missing output path") }
        }
        _ => return failure("Image::save", "Invalid output"),
    };
    let mut enc_opts = if args.len() > 2 { match ev.eval(args[2].clone()) { Value::Assoc(m) => m, _ => std::collections::HashMap::new() } } else { std::collections::HashMap::new() };
    if !enc_opts.contains_key("format") && !enc_opts.contains_key("Format") {
        if let Some(fmt) = infer_format_from_ext(&out_path) { enc_opts.insert("format".into(), Value::String(fmt)); }
    }
    let bytes = match encode_image(&img, &enc_opts) { Ok(b) => b, Err(e) => return failure("Image::encode", &format!("ImageSave: {}", e)) };
    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        if mkdirs { let _ = std::fs::create_dir_all(parent); }
    }
    match std::fs::write(&out_path, &bytes) {
        Ok(_) => Value::Assoc(std::iter::once(("path".into(), Value::String(out_path))).collect()),
        Err(e) => failure("Image::save", &format!("{}", e)),
    }
}
