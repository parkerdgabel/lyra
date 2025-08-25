#![cfg(feature = "image")]

use lyra_runtime::Evaluator;
use lyra_core::value::Value;
use lyra_stdlib as stdlib;
use base64::Engine;

fn b64str(v: &Value) -> String { match v { Value::String(s) => s.clone(), _ => lyra_core::pretty::format_value(v) } }

#[test]
fn image_canvas_info_resize_convert_thumbnail_transform() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);

    // Create a small canvas (8x6)
    let png = ev.eval(Value::expr(Value::Symbol("ImageCanvas".into()), vec![
        Value::Assoc([
            ("width".to_string(), Value::Integer(8)),
            ("height".to_string(), Value::Integer(6)),
            ("background".to_string(), Value::String("#ff0000".into())),
        ].into_iter().collect()),
    ]));
    let info = ev.eval(Value::expr(Value::Symbol("ImageInfo".into()), vec![png.clone()]));
    if let Value::Assoc(m) = info {
        assert!(matches!(m.get("width"), Some(Value::Integer(n)) if *n==8));
        assert!(matches!(m.get("height"), Some(Value::Integer(n)) if *n==6));
    } else { panic!("ImageInfo invalid"); }

    // Resize to 4x4 cover
    let r = ev.eval(Value::expr(Value::Symbol("ImageResize".into()), vec![
        png.clone(),
        Value::Assoc([
            ("width".to_string(), Value::Integer(4)),
            ("height".to_string(), Value::Integer(4)),
            ("fit".to_string(), Value::String("cover".into())),
        ].into_iter().collect()),
    ]));
    let r_info = ev.eval(Value::expr(Value::Symbol("ImageInfo".into()), vec![r.clone()]));
    if let Value::Assoc(m) = r_info { assert!(matches!(m.get("width"), Some(Value::Integer(n)) if *n==4)); assert!(matches!(m.get("height"), Some(Value::Integer(n)) if *n==4)); } else { panic!("resize info"); }

    // Convert to JPEG
    let jpg = ev.eval(Value::expr(Value::Symbol("ImageConvert".into()), vec![
        r.clone(), Value::String("jpeg".into())
    ]));
    let _ = b64str(&jpg); // ensure it returns a string
    let j_info = ev.eval(Value::expr(Value::Symbol("ImageInfo".into()), vec![jpg.clone()]));
    if let Value::Assoc(m) = j_info { assert!(matches!(m.get("width"), Some(Value::Integer(n)) if *n==4)); } else { panic!("jpeg info"); }

    // Thumbnail 2x2
    let th = ev.eval(Value::expr(Value::Symbol("ImageThumbnail".into()), vec![
        png.clone(), Value::Integer(2)
    ]));
    let t_info = ev.eval(Value::expr(Value::Symbol("ImageInfo".into()), vec![th]));
    if let Value::Assoc(m) = t_info { assert!(matches!(m.get("width"), Some(Value::Integer(n)) if *n==2)); assert!(matches!(m.get("height"), Some(Value::Integer(n)) if *n==2)); } else { panic!("thumb info"); }

    // Transform pipeline: contain to 3x5
    let tx = ev.eval(Value::expr(Value::Symbol("ImageTransform".into()), vec![
        png,
        Value::Assoc([
            ("steps".into(), Value::List(vec![
                Value::Assoc([
                    ("op".into(), Value::String("resize".into())),
                    ("args".into(), Value::Assoc([
                        ("width".into(), Value::Integer(3)),
                        ("height".into(), Value::Integer(5)),
                        ("fit".into(), Value::String("contain".into())),
                    ].into_iter().collect())),
                ].into_iter().collect())
            ])),
        ].into_iter().collect()),
    ]));
    let txi = ev.eval(Value::expr(Value::Symbol("ImageInfo".into()), vec![tx]));
    if let Value::Assoc(m) = txi { assert!(matches!(m.get("width"), Some(Value::Integer(n)) if *n<=3)); assert!(matches!(m.get("height"), Some(Value::Integer(n)) if *n<=5)); } else { panic!("transform info"); }
}

#[test]
fn image_save_to_disk() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    let png = ev.eval(Value::expr(Value::Symbol("ImageCanvas".into()), vec![
        Value::Assoc([
            ("width".to_string(), Value::Integer(10)),
            ("height".to_string(), Value::Integer(10)),
            ("background".to_string(), Value::String("#00ff00".into())),
        ].into_iter().collect()),
    ]));
    let out_path = format!("{}/target/test_images/save1.png", env!("CARGO_MANIFEST_DIR"));
    let _ = ev.eval(Value::expr(Value::Symbol("ImageSave".into()), vec![
        png,
        Value::Assoc([
            ("path".into(), Value::String(out_path.clone())),
            ("mkdirs".into(), Value::Boolean(true)),
        ].into_iter().collect()),
    ]));
    assert!(std::path::Path::new(&out_path).exists());
    let bytes = std::fs::read(&out_path).expect("read saved");
    assert!(bytes.len() > 0);
}

#[test]
fn image_decode_raw_and_encode_roundtrip() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);

    // Start with a small canvas
    let base = ev.eval(Value::expr(Value::Symbol("ImageCanvas".into()), vec![
        Value::Assoc([
            ("width".to_string(), Value::Integer(3)),
            ("height".to_string(), Value::Integer(2)),
            ("background".to_string(), Value::String("#0000ff".into())),
        ].into_iter().collect()),
    ]));

    // Decode to raw
    let raw = ev.eval(Value::expr(Value::Symbol("ImageDecode".into()), vec![
        base.clone(),
        Value::Assoc([( "format".into(), Value::String("raw".into()) )].into_iter().collect()),
    ]));

    // Validate raw structure
    let (w,h,bytes) = match &raw {
        Value::Assoc(m) => {
            let w = match m.get("width") { Some(Value::Integer(n)) => *n as usize, _ => panic!("missing width") };
            let h = match m.get("height") { Some(Value::Integer(n)) => *n as usize, _ => panic!("missing height") };
            let b = match m.get("bytes") { Some(Value::String(s)) => s.clone(), _ => panic!("missing bytes") };
            (w,h,b)
        }
        _ => panic!("raw not assoc"),
    };
    // Expect RGBA8 length
    let raw_bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(bytes).unwrap();
    assert_eq!(raw_bytes.len(), w*h*4);

    // Encode raw back to PNG
    let png = ev.eval(Value::expr(Value::Symbol("ImageEncode".into()), vec![
        raw.clone(),
        Value::Assoc([( "format".into(), Value::String("png".into()) )].into_iter().collect()),
    ]));
    // Sanity: info still 3x2
    let info = ev.eval(Value::expr(Value::Symbol("ImageInfo".into()), vec![png]));
    if let Value::Assoc(m) = info { assert!(matches!(m.get("width"), Some(Value::Integer(n)) if *n==3)); assert!(matches!(m.get("height"), Some(Value::Integer(n)) if *n==2)); } else { panic!("info invalid"); }

    // Also test ImageEncode re-encoding to jpeg directly from original
    let _jpg = ev.eval(Value::expr(Value::Symbol("ImageEncode".into()), vec![
        base,
        Value::Assoc([( "format".into(), Value::String("jpeg".into()) )].into_iter().collect()),
    ]));
}

#[test]
fn image_crop_and_pad() {
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);

    // Base: 8x6 red
    let base = ev.eval(Value::expr(Value::Symbol("ImageCanvas".into()), vec![
        Value::Assoc([
            ("width".to_string(), Value::Integer(8)),
            ("height".to_string(), Value::Integer(6)),
            ("background".to_string(), Value::String("#ff0000".into())),
        ].into_iter().collect()),
    ]));

    // Rect crop 4x3 at (2,1)
    let crop_rect = ev.eval(Value::expr(Value::Symbol("ImageCrop".into()), vec![
        base.clone(),
        Value::Assoc([
            ("rect".into(), Value::Assoc([
                ("x".into(), Value::Integer(2)),
                ("y".into(), Value::Integer(1)),
                ("width".into(), Value::Integer(4)),
                ("height".into(), Value::Integer(3)),
            ].into_iter().collect())),
        ].into_iter().collect()),
    ]));
    let info = ev.eval(Value::expr(Value::Symbol("ImageInfo".into()), vec![crop_rect]));
    if let Value::Assoc(m) = info { assert!(matches!(m.get("width"), Some(Value::Integer(n)) if *n==4)); assert!(matches!(m.get("height"), Some(Value::Integer(n)) if *n==3)); } else { panic!("crop_rect info"); }

    // Gravity crop to 6x4 northwest
    let crop_g = ev.eval(Value::expr(Value::Symbol("ImageCrop".into()), vec![
        base.clone(),
        Value::Assoc([
            ("width".into(), Value::Integer(6)),
            ("height".into(), Value::Integer(4)),
            ("gravity".into(), Value::String("northwest".into())),
        ].into_iter().collect()),
    ]));
    let info2 = ev.eval(Value::expr(Value::Symbol("ImageInfo".into()), vec![crop_g]));
    if let Value::Assoc(m) = info2 { assert!(matches!(m.get("width"), Some(Value::Integer(n)) if *n==6)); assert!(matches!(m.get("height"), Some(Value::Integer(n)) if *n==4)); } else { panic!("crop_g info"); }

    // Pad to 10x10 center white
    let pad = ev.eval(Value::expr(Value::Symbol("ImagePad".into()), vec![
        base.clone(),
        Value::Assoc([
            ("width".into(), Value::Integer(10)),
            ("height".into(), Value::Integer(10)),
            ("background".into(), Value::String("#ffffff".into())),
            ("gravity".into(), Value::String("center".into())),
        ].into_iter().collect()),
    ]));
    let pinfo = ev.eval(Value::expr(Value::Symbol("ImageInfo".into()), vec![pad]));
    if let Value::Assoc(m) = pinfo { assert!(matches!(m.get("width"), Some(Value::Integer(n)) if *n==10)); assert!(matches!(m.get("height"), Some(Value::Integer(n)) if *n==10)); } else { panic!("pad info"); }

    // Pad with downscale to 4x4
    let pad_small = ev.eval(Value::expr(Value::Symbol("ImagePad".into()), vec![
        base,
        Value::Assoc([
            ("width".into(), Value::Integer(4)),
            ("height".into(), Value::Integer(4)),
            ("background".into(), Value::String("#000000".into())),
        ].into_iter().collect()),
    ]));
    let psinfo = ev.eval(Value::expr(Value::Symbol("ImageInfo".into()), vec![pad_small]));
    if let Value::Assoc(m) = psinfo { assert!(matches!(m.get("width"), Some(Value::Integer(n)) if *n==4)); assert!(matches!(m.get("height"), Some(Value::Integer(n)) if *n==4)); } else { panic!("pad small info"); }
}
