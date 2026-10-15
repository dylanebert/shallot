// Only the macOS portable (CEF) build uses separate helper .app bundles;
// Windows/Linux CEF re-exec the main binary.
#[cfg(all(target_os = "macos", feature = "portable"))]
fn main() {
    use cef::args::Args;
    use cef::library_loader::LibraryLoader;
    use cef::{api_hash, execute_process, sys, App};

    let args = Args::new();

    let loader = LibraryLoader::new(&std::env::current_exe().expect("no exe path"), true);
    assert!(loader.load(), "failed to load CEF framework in helper");

    let _ = api_hash(sys::CEF_API_VERSION_LAST, 0);

    execute_process(
        Some(args.as_main_args()),
        None::<&mut App>,
        std::ptr::null_mut(),
    );
}

#[cfg(not(all(target_os = "macos", feature = "portable")))]
fn main() {}
