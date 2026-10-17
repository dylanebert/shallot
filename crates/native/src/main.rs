#![cfg_attr(windows, windows_subsystem = "windows")]

// Backend by build mode (not by platform). `portable` ships the Chromium runtime (CEF) and runs
// anywhere; the default uses the platform's system webview via wry (WebView2 on Windows, WKWebView
// on macOS, WebKitGTK on Linux). A system-webview build is small but depends on the host — and
// WebKitGTK has no usable WebGPU, so a default Linux build reaches the engine's diagnostic tier at
// launch. macOS WKWebView lacks subgroup support, so a physics app there needs `--portable`.
#[cfg(feature = "portable")]
mod cef_backend;
mod gpu_policy;
#[cfg(all(feature = "portable", target_os = "macos"))]
mod mac;
#[cfg(not(feature = "portable"))]
mod wry_backend;

use std::borrow::Cow;
use std::path::PathBuf;

#[cfg(not(feature = "portable"))]
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

// build.rs supplies ASSETS from SHALLOT_DIST, or an empty list when it is unset.
mod embedded {
    include!(concat!(env!("OUT_DIR"), "/assets.rs"));
}

/// On macOS, assets belong in Contents/Resources/dist: codesign treats files
/// beside the executable in Contents/MacOS as nested code and rejects web assets.
/// Other platforms use dist beside the executable.
fn dist_dir() -> PathBuf {
    let exe = std::env::current_exe().expect("failed to get exe path");
    let dir = exe.parent().expect("exe has no parent");
    #[cfg(target_os = "macos")]
    {
        dir.parent()
            .expect("MacOS has no parent")
            .join("Resources")
            .join("dist")
    }
    #[cfg(not(target_os = "macos"))]
    {
        dir.join("dist")
    }
}

#[derive(Debug)]
pub(crate) enum AssetResolution<'a> {
    Embedded(&'a [u8]),
    Disk,
}

/// An empty asset list selects disk for the prebuilt-shell fallback. A nonempty
/// list is authoritative: a miss returns None rather than masking a stale dist
/// behind fresh embedded assets.
pub(crate) fn resolve_asset<'a>(
    assets: &'a [(&'a str, &'a [u8])],
    path: &str,
) -> Option<AssetResolution<'a>> {
    if assets.is_empty() {
        Some(AssetResolution::Disk)
    } else {
        assets
            .iter()
            .find(|entry| entry.0 == path)
            .map(|entry| AssetResolution::Embedded(entry.1))
    }
}

pub(crate) fn asset(path: &str) -> Option<Cow<'static, [u8]>> {
    match resolve_asset(embedded::ASSETS, path) {
        Some(AssetResolution::Embedded(bytes)) => Some(Cow::Borrowed(bytes)),
        Some(AssetResolution::Disk) => std::fs::read(dist_dir().join(path)).ok().map(Cow::Owned),
        None => None,
    }
}

fn decode_icon(data: Option<&[u8]>) -> image::RgbaImage {
    let bytes = data.unwrap_or(include_bytes!("../../../assets/icon-1024.png"));
    image::load_from_memory_with_format(bytes, image::ImageFormat::Png)
        .unwrap_or_else(|error| panic!("invalid native icon.png: {error}; replace public/icon.png and rebuild the native shell"))
        .into_rgba8()
}

pub(crate) fn window_icon() -> image::RgbaImage {
    let data = asset("icon.png");
    decode_icon(data.as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_icon_default_and_explicit_replacement_are_distinct() {
        let default = decode_icon(None);
        assert_eq!(default.dimensions(), (1024, 1024));
        let custom = image::RgbaImage::from_pixel(32, 32, image::Rgba([255, 0, 42, 255]));
        let mut png = std::io::Cursor::new(Vec::new());
        custom.write_to(&mut png, image::ImageFormat::Png).unwrap();
        let assets: &[(&str, &[u8])] = &[("icon.png", png.get_ref())];
        let Some(AssetResolution::Embedded(bytes)) = resolve_asset(assets, "icon.png") else {
            panic!("embedded icon did not resolve");
        };
        assert_eq!(decode_icon(Some(bytes)), custom);
        assert!(matches!(
            resolve_asset(&[], "icon.png"),
            Some(AssetResolution::Disk)
        ));
        let missing: &[(&str, &[u8])] = &[("index.html", b"page")];
        assert!(resolve_asset(missing, "icon.png").is_none());
        assert_ne!(default.get_pixel(0, 0), custom.get_pixel(0, 0));
    }

    #[test]
    #[should_panic(expected = "invalid native icon.png")]
    fn invalid_native_icon_is_not_silently_replaced() {
        decode_icon(Some(b"not a PNG"));
    }

    #[test]
    fn empty_embedded_list_falls_back_to_disk() {
        let assets: &[(&str, &[u8])] = &[];
        assert!(
            matches!(
                resolve_asset(assets, "index.html"),
                Some(AssetResolution::Disk)
            ),
            "empty embedded list must resolve to disk, not None"
        );
    }

    #[test]
    fn non_empty_list_hit_serves_embedded_bytes() {
        let bytes: &[u8] = b"<html></html>";
        let assets: &[(&str, &[u8])] = &[("index.html", bytes)];
        match resolve_asset(assets, "index.html") {
            Some(AssetResolution::Embedded(b)) => assert_eq!(b, bytes),
            other => panic!("expected Embedded, got {other:?}"),
        }
    }

    #[test]
    fn non_empty_list_miss_returns_none_no_disk_fallback_for_a_baked_binary() {
        // A binary that bakes its assets does not fall through to disk for a path it doesn't carry
        // — that would mask a stale dist/ behind a fresh embed. Miss is a 404.
        let assets: &[(&str, &[u8])] = &[("index.html", b"<html></html>")];
        assert!(matches!(resolve_asset(assets, "missing.js"), None));
    }

    #[test]
    fn dev_url_replaces_the_packaged_asset_url() {
        assert_eq!(
            launch_url(
                Some("http://localhost:5173/".into()),
                "shallot://localhost/".into()
            ),
            "http://localhost:5173/"
        );
        assert_eq!(
            launch_url(None, "shallot://localhost/".into()),
            "shallot://localhost/"
        );
    }
}

pub(crate) fn launch_url(dev_url: Option<String>, fallback: String) -> String {
    dev_url
        .filter(|url| !url.trim().is_empty())
        .unwrap_or(fallback)
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
    #[cfg(feature = "portable")]
    cef_backend::run();
    #[cfg(not(feature = "portable"))]
    wry_backend::run();
}
