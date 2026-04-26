#![cfg_attr(windows, windows_subsystem = "windows")]

#[cfg(not(target_os = "linux"))]
mod wry_backend;
#[cfg(target_os = "linux")]
mod cef_backend;

use std::path::PathBuf;

#[cfg(not(target_os = "linux"))]
pub(crate) const BG: (u8, u8, u8) = (14, 13, 12); // #0e0d0c

#[cfg(windows)]
fn local_app_data() -> PathBuf {
    std::env::var("LOCALAPPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("USERPROFILE").unwrap_or_else(|_| ".".into());
            PathBuf::from(home).join("AppData").join("Local")
        })
}

pub(crate) fn cache_dir(name: &str) -> PathBuf {
    #[cfg(windows)]
    {
        local_app_data().join("shallot").join(name)
    }
    #[cfg(target_os = "macos")]
    {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        PathBuf::from(home)
            .join("Library")
            .join("Application Support")
            .join("shallot")
            .join(name)
    }
    #[cfg(target_os = "linux")]
    {
        let base = std::env::var("XDG_DATA_HOME").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            format!("{}/.local/share", home)
        });
        PathBuf::from(base).join("shallot").join(name)
    }
}

#[cfg(not(debug_assertions))]
pub(crate) fn unpack_to_cache(compressed: &[u8], name: &str, key: u64) -> Option<PathBuf> {
    let dir = cache_dir(name);
    let marker = dir.join(".installed");
    if let Ok(stored) = std::fs::read_to_string(&marker) {
        if let Ok(stored_key) = stored.trim().parse::<u64>() {
            if stored_key == key {
                return Some(dir);
            }
        }
    }
    std::fs::create_dir_all(&dir).ok()?;
    let decompressed = zstd::decode_all(compressed).ok()?;
    let mut archive = tar::Archive::new(decompressed.as_slice());
    archive.unpack(&dir).ok()?;
    std::fs::write(&marker, key.to_string()).ok()?;
    Some(dir)
}

#[cfg(all(not(debug_assertions), any(windows, target_os = "linux")))]
pub(crate) fn extract_payload(exe: &std::path::Path) -> Option<PathBuf> {
    const MAGIC: u32 = 0x544C4853;
    let data = std::fs::read(exe).ok()?;
    if data.len() < 8 {
        return None;
    }
    let footer = &data[data.len() - 8..];
    let payload_size = u32::from_le_bytes([footer[0], footer[1], footer[2], footer[3]]) as usize;
    let magic = u32::from_le_bytes([footer[4], footer[5], footer[6], footer[7]]);
    if magic != MAGIC {
        return None;
    }
    let start = data.len() - 8 - payload_size;
    let name = exe.file_stem()?.to_str()?;
    unpack_to_cache(&data[start..start + payload_size], name, data.len() as u64)
}

#[cfg(all(not(debug_assertions), target_os = "macos"))]
pub(crate) fn extract_bundle_payload(exe: &std::path::Path) -> Option<PathBuf> {
    let payload_path = exe.parent()?.parent()?.join("Resources").join("payload.bin");
    let data = std::fs::read(&payload_path).ok()?;
    let name = exe.file_stem()?.to_str()?;
    unpack_to_cache(&data, name, data.len() as u64)
}

pub(crate) fn asset_dir() -> PathBuf {
    let exe = std::env::current_exe().expect("failed to get exe path");

    #[cfg(not(debug_assertions))]
    {
        #[cfg(target_os = "macos")]
        if let Some(cache) = extract_bundle_payload(&exe) {
            return cache.join("dist");
        }

        #[cfg(any(windows, target_os = "linux"))]
        if let Some(cache) = extract_payload(&exe) {
            return cache.join("dist");
        }
    }

    exe.parent().expect("exe has no parent").join("dist")
}

pub(crate) fn app_name() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.file_stem().map(|s| s.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "shallot".into())
}

pub(crate) fn content_type(path: &str) -> &'static str {
    if path.ends_with(".html") {
        "text/html"
    } else if path.ends_with(".js") {
        "text/javascript"
    } else if path.ends_with(".css") {
        "text/css"
    } else if path.ends_with(".wasm") {
        "application/wasm"
    } else if path.ends_with(".json") {
        "application/json"
    } else if path.ends_with(".svg") {
        "image/svg+xml"
    } else if path.ends_with(".png") {
        "image/png"
    } else if path.ends_with(".jpg") || path.ends_with(".jpeg") {
        "image/jpeg"
    } else if path.ends_with(".ico") {
        "image/x-icon"
    } else if path.ends_with(".woff2") {
        "font/woff2"
    } else if path.ends_with(".woff") {
        "font/woff"
    } else if path.ends_with(".glb") {
        "model/gltf-binary"
    } else {
        "application/octet-stream"
    }
}

fn main() {
    #[cfg(not(target_os = "linux"))]
    wry_backend::run();
    #[cfg(target_os = "linux")]
    cef_backend::run();
}
