use lyra_core::value::Value;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

// Build a data URL using standard base64 encoding
fn data_url(mime: &str, bytes: &[u8]) -> String {
    use base64::Engine as _;
    let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
    format!("data:{};base64,{}", mime, b64)
}

pub fn display_bytes(mime: &str, bytes: Vec<u8>) -> Value {
    Value::String(data_url(mime, &bytes))
}

pub fn display_text(mime: &str, text: &str) -> Value {
    // Encode as base64 to avoid URL-encoding pitfalls; keeps sanitizer decisions in UI
    display_bytes(mime, text.as_bytes().to_vec())
}

static PREFER_DISPLAY_DEFAULT: OnceLock<AtomicBool> = OnceLock::new();

fn prefer_display_flag() -> &'static AtomicBool {
    PREFER_DISPLAY_DEFAULT.get_or_init(|| AtomicBool::new(false))
}

pub fn set_prefer_display(enabled: bool) {
    prefer_display_flag().store(enabled, Ordering::SeqCst);
}

pub fn prefer_display() -> bool {
    prefer_display_flag().load(Ordering::SeqCst)
}
