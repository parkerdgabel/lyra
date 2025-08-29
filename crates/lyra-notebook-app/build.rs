fn main() {
    // Ensure a minimal placeholder RGBA PNG icon is present for tauri-build
    let icon_path = std::path::Path::new("icons/icon.png");
    if let Some(dir) = icon_path.parent() { let _ = std::fs::create_dir_all(dir); }
    // 1x1 transparent PNG (RGBA)
    const PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABpfZFQAAAAABJRU5ErkJggg==";
    // Use non-deprecated base64 Engine API
    if let Ok(bytes) = {
        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD.decode(PNG_BASE64)
    } {
        // Overwrite to guarantee RGBA-encoded icon
        let _ = std::fs::write(icon_path, &bytes);
    }
    tauri_build::build()
}
