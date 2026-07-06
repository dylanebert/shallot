use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    gen_assets();

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "windows" {
        return;
    }

    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out = std::env::var("OUT_DIR").unwrap();
    let rc = format!("{manifest}/icon.rc");

    println!("cargo:rerun-if-changed=icon.rc");
    println!("cargo:rerun-if-changed=icon.ico");

    // The icon is compiled by rc.exe on a native MSVC build (→ .res) and by mingw windres on a
    // cargo-xwin cross build (→ .o); link.exe and lld-link each take their own resource format. Try
    // the MSVC tool first, fall back to windres, and warn rather than fail if neither is present —
    // the icon is cosmetic, so a missing resource compiler shouldn't break the build. `/I` points rc
    // at the manifest dir so it finds icon.ico when the build runs from a different cwd.
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
    println!("cargo:warning=no resource compiler (rc.exe / windres) found — building without an exe icon");
}

fn try_run(prog: &str, args: &[&str]) -> bool {
    matches!(Command::new(prog).args(args).status(), Ok(s) if s.success())
}

// Embed the project's built web assets (set via SHALLOT_DIST) directly into the binary so the
// release exe carries no appended overlay and never extracts itself at runtime — both are dropper
// signatures that trip Windows Defender on unsigned builds. Empty when SHALLOT_DIST is unset
// (debug serves from the sibling dist/ instead, so iteration doesn't pay a recompile).
fn gen_assets() {
    println!("cargo:rerun-if-env-changed=SHALLOT_DIST");

    let mut files = Vec::new();
    if let Ok(dist) = std::env::var("SHALLOT_DIST") {
        let dist = PathBuf::from(&dist);
        if dist.is_dir() {
            collect(&dist, &dist, &mut files);
            println!("cargo:rerun-if-changed={}", dist.display());
        }
    }
    files.sort();

    if files.is_empty() && std::env::var("PROFILE").as_deref() == Ok("release") {
        println!(
            "cargo:warning=SHALLOT_DIST unset or empty — release binary embeds no web assets and will render blank"
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
