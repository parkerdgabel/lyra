fn main() {
    // Ensure a minimal placeholder RGBA PNG icon is present for tauri-build
    let icon_path = std::path::Path::new("icons/icon.png");
    if let Some(dir) = icon_path.parent() { let _ = std::fs::create_dir_all(dir); }
    // 1x1 transparent PNG (RGBA)
    const PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABpfZFQAAAAABJRU5ErkJggg==";
    if let Ok(bytes) = base64::decode(PNG_BASE64) {
        // Overwrite to guarantee RGBA-encoded icon
        let _ = std::fs::write(icon_path, &bytes);
    }
    tauri_build::build()
}
