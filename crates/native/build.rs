use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    if let Ok(cef_dir) = std::env::var("DEP_CEF_DLL_WRAPPER_CEF_DIR") {
        println!("cargo:rustc-env=SHALLOT_CEF_DIR={cef_dir}");
    }
    gen_assets();

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "windows" {
        return;
    }

    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out = std::env::var("OUT_DIR").unwrap();
    println!("cargo:rerun-if-env-changed=SHALLOT_ICON_ICO");
    let custom = std::env::var("SHALLOT_ICON_ICO")
        .ok()
        .filter(|path| !path.is_empty());
    let icon = custom
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(&manifest).join("icon.ico"));
    assert!(
        icon.is_absolute(),
        "SHALLOT_ICON_ICO must be an absolute invocation-owned path"
    );
    let bytes = std::fs::read(&icon).expect("cannot read Windows ICO resource");
    assert!(
        bytes.len() >= 22 && bytes[..4] == [0, 0, 1, 0] && bytes[4..6] != [0, 0],
        "invalid Windows ICO resource: {}",
        icon.display()
    );
    println!("cargo:rerun-if-changed={}", icon.display());
    let rc = format!("{out}/icon.rc");
    std::fs::write(
        &rc,
        format!("1 ICON \"{}\"\n", escape(&icon.to_string_lossy())),
    )
    .unwrap();

    // The icon is compiled by rc.exe on a native MSVC build (→ .res) and by mingw windres on a
    // cargo-xwin cross build (→ .o); link.exe and lld-link each take their own resource format. Try
    // the MSVC tool first, fall back to windres. An explicit replacement must never silently
    // compile as the default; the RC stays in OUT_DIR and names the invocation-owned ICO.
    let res = format!("{out}/icon.res");
    if try_run("rc.exe", &["/nologo", "/I", &manifest, "/fo", &res, &rc]) {
        println!("cargo:rustc-link-arg={res}");
        return;
    }
    let obj = format!("{out}/icon.o");
    if try_run("x86_64-w64-mingw32-windres", &[&rc, &obj]) {
        println!("cargo:rustc-link-arg={obj}");
        return;
    }
    assert!(custom.is_none(), "cannot compile requested Windows ICO resource {}: rc.exe and windres failed or are unavailable; install the Windows resource toolchain", icon.display());
    println!("cargo:warning=no resource compiler (rc.exe / windres) found — building without an exe icon");
}

fn try_run(prog: &str, args: &[&str]) -> bool {
    matches!(Command::new(prog).args(args).status(), Ok(s) if s.success())
}

// Embed the project's built web assets (set via SHALLOT_DIST) directly into the binary so the
// release exe carries no appended overlay and never extracts itself at runtime — both are dropper
// signatures that trip Windows Defender on unsigned builds. Empty when SHALLOT_DIST is unset;
// debug and explicitly disk-mode releases serve from the sibling dist/ instead.
fn gen_assets() {
    println!("cargo:rerun-if-env-changed=SHALLOT_DIST");
    println!("cargo:rerun-if-env-changed=SHALLOT_ASSETS");
    let disk_mode = std::env::var("SHALLOT_ASSETS").as_deref() == Ok("disk");

    let mut files = Vec::new();
    if let Ok(dist) = std::env::var("SHALLOT_DIST") {
        let dist = PathBuf::from(&dist);
        if dist.is_dir() {
            collect(&dist, &dist, &mut files);
            println!("cargo:rerun-if-changed={}", dist.display());
        }
    }
    files.sort();

    if disk_mode && !files.is_empty() {
        eprintln!("SHALLOT_ASSETS=disk cannot be used when SHALLOT_DIST contains assets");
        std::process::exit(1);
    }

    if files.is_empty() && std::env::var("PROFILE").as_deref() == Ok("release") && !disk_mode {
        println!(
            "cargo:warning=SHALLOT_DIST unset or empty — release binary embeds no web assets; it will serve the sibling dist/ on disk (Contents/Resources/dist on mac) instead"
        );
    }

    let mut src = String::from("pub static ASSETS: &[(&str, &[u8])] = &[\n");
    for (rel, abs) in &files {
        println!("cargo:rerun-if-changed={}", abs.display());
        let abs = escape(&abs.to_string_lossy());
        let rel = escape(rel);
        writeln!(src, "    (\"{rel}\", include_bytes!(\"{abs}\")),").unwrap();
    }
    src.push_str("];\n");

    let dest = Path::new(&std::env::var("OUT_DIR").unwrap()).join("assets.rs");
    std::fs::write(&dest, src).unwrap();
}

fn collect(dir: &Path, base: &Path, out: &mut Vec<(String, PathBuf)>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, base, out);
        } else if let Ok(rel) = path.strip_prefix(base) {
            out.push((rel.to_string_lossy().replace('\\', "/"), path));
        }
    }
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}
