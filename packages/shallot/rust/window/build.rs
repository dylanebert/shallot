use std::process::Command;

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "windows" {
        return;
    }

    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out = std::env::var("OUT_DIR").unwrap();
    let rc = format!("{manifest}/icon.rc");
    let obj = format!("{out}/icon.o");

    let status = Command::new("x86_64-w64-mingw32-windres")
        .args([&rc, &obj])
        .status()
        .expect("failed to run windres");

    if !status.success() {
        panic!("windres failed");
    }

    println!("cargo:rustc-link-arg={obj}");
    println!("cargo:rerun-if-changed=icon.rc");
    println!("cargo:rerun-if-changed=icon.ico");
}
