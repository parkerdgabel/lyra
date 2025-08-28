fn main() {
    // Ensure a minimal placeholder icon exists to satisfy tauri-build
    let icon_path = std::path::Path::new("icons/icon.png");
    if !icon_path.exists() {
        if let Some(dir) = icon_path.parent() { let _ = std::fs::create_dir_all(dir); }
        const PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y3u7u8AAAAASUVORK5CYII=";
        if let Ok(bytes) = base64::decode(PNG_BASE64) {
            let _ = std::fs::write(icon_path, &bytes);
        }
    }
    tauri_build::build()
}
