use cef::*;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::OnceLock;

use crate::{app_name, asset, cache_dir, content_type};

#[cfg(target_os = "linux")]
const VK_F11: i32 = 0x7A;
#[cfg(target_os = "linux")]
const VK_RETURN: i32 = 0x0D;
#[cfg(target_os = "linux")]
const VK_ESCAPE: i32 = 0x1B;
#[cfg(target_os = "linux")]
const EVENTFLAG_ALT: u32 = 8;

static ORIGIN: OnceLock<String> = OnceLock::new();

fn start_asset_server() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("failed to bind asset server");
    let port = listener.local_addr().unwrap().port();
    let origin = format!("http://127.0.0.1:{}", port);
    eprintln!("[shallot] asset server on {}", origin);

    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut buf = [0u8; 4096];
            let n = match stream.read(&mut buf) {
                Ok(n) if n > 0 => n,
                _ => continue,
            };
            let request = String::from_utf8_lossy(&buf[..n]);
            let path = request
                .lines()
                .next()
                .and_then(|line| line.split_whitespace().nth(1))
                .unwrap_or("/");
            let path = if path == "/" { "/index.html" } else { path };

            match asset(&path[1..]) {
                Some(data) => {
                    let mime = content_type(path);
                    // COOP/COEP: cross-origin isolation so tumble physics can multithread
                    // (mirrors the JS serve surfaces' CROSS_ORIGIN_ISOLATION in project/vite.ts)
                    let header = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nCross-Origin-Opener-Policy: same-origin\r\nCross-Origin-Embedder-Policy: require-corp\r\nConnection: close\r\n\r\n",
                        mime, data.len()
                    );
                    let _ = stream.write_all(header.as_bytes());
                    let _ = stream.write_all(&data);
                }
                None => {
                    let _ = stream.write_all(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    );
                }
            }
        }
    });

    origin
}

wrap_app! {
    pub struct ShallotApp;

    impl App {
        fn on_before_command_line_processing(
            &self,
            _process_type: Option<&CefString>,
            command_line: Option<&mut CommandLine>,
        ) {
            if let Some(cmd) = command_line {
                // enable-unsafe-webgpu lifts the WebGPU adapter blocklist; ignore-gpu-blocklist lifts
                // the broader GPU software-rendering list — without both, requestAdapter() returns
                // null on otherwise-capable GPUs. allow_unsafe_apis exposes the unsafe Dawn features
                // an engine plugin may require (timestamp-query, which ProfilePlugin declares).
                cmd.append_switch(Some(&CefString::from("enable-unsafe-webgpu")));
                cmd.append_switch(Some(&CefString::from("ignore-gpu-blocklist")));
                cmd.append_switch_with_value(
                    Some(&CefString::from("enable-dawn-features")),
                    Some(&CefString::from("allow_unsafe_apis")),
                );
                // Linux WebGPU runs on the Vulkan backend; Windows uses D3D12 and macOS Metal (no flag).
                #[cfg(target_os = "linux")]
                cmd.append_switch(Some(&CefString::from("enable-features=Vulkan")));
                // Windows: run the GPU in-process. The re-exec'd GPU subprocess can't initialize GL in
                // this single-exe CEF build — gl::init::InitializeStaticGLBindingsOneOff fails, the GPU
                // process exits, Chromium falls back to use-gl=disabled, and requestAdapter() returns
                // null ("no compatible GPU"). In-process the discrete GPU is detected and hardware
                // WebGPU works (verified rendering on real hardware).
                #[cfg(target_os = "windows")]
                cmd.append_switch(Some(&CefString::from("in-process-gpu")));
                #[cfg(debug_assertions)]
                cmd.append_switch_with_value(
                    Some(&CefString::from("remote-debugging-port")),
                    Some(&CefString::from("9222")),
                );
            }
        }

        fn browser_process_handler(&self) -> Option<BrowserProcessHandler> {
            Some(ShallotBrowserProcessHandler::new())
        }
    }
}

wrap_browser_process_handler! {
    struct ShallotBrowserProcessHandler;

    impl BrowserProcessHandler {
        fn on_context_initialized(&self) {
            let origin = ORIGIN.get().expect("asset server not started");
            let url = CefString::from(format!("{}/", origin).as_str());
            let settings = BrowserSettings::default();
            let window_info = WindowInfo::default();
            let mut client = ShallotClient::new();

            browser_host_create_browser(
                Some(&window_info),
                Some(&mut client),
                Some(&url),
                Some(&settings),
                None,
                None,
            );
        }
    }
}

#[cfg(target_os = "linux")]
fn set_x11_fullscreen(xid: std::os::raw::c_ulong, action: std::os::raw::c_long) {
    #[link(name = "X11")]
    extern "C" {
        fn XOpenDisplay(name: *const i8) -> *mut u8;
        fn XDefaultRootWindow(display: *mut u8) -> std::os::raw::c_ulong;
        fn XInternAtom(
            display: *mut u8,
            name: *const i8,
            only_if_exists: i32,
        ) -> std::os::raw::c_ulong;
        fn XSendEvent(
            display: *mut u8,
            window: std::os::raw::c_ulong,
            propagate: i32,
            mask: i64,
            event: *mut XClientMessageEvent,
        ) -> i32;
        fn XFlush(display: *mut u8) -> i32;
        fn XCloseDisplay(display: *mut u8) -> i32;
    }

    #[repr(C)]
    struct XClientMessageEvent {
        type_: i32,
        serial: std::os::raw::c_ulong,
        send_event: i32,
        display: *mut u8,
        window: std::os::raw::c_ulong,
        message_type: std::os::raw::c_ulong,
        format: i32,
        data: [std::os::raw::c_long; 5],
    }

    unsafe {
        let display = XOpenDisplay(std::ptr::null());
        if display.is_null() {
            return;
        }
        let root = XDefaultRootWindow(display);
        let wm_state = XInternAtom(display, b"_NET_WM_STATE\0".as_ptr() as _, 0);
        let wm_fullscreen = XInternAtom(display, b"_NET_WM_STATE_FULLSCREEN\0".as_ptr() as _, 0);

        let mut event = XClientMessageEvent {
            type_: 33, // ClientMessage
            serial: 0,
            send_event: 1,
            display,
            window: xid,
            message_type: wm_state,
            format: 32,
            data: [
                action,
                wm_fullscreen as _,
                0,
                1, // source: application
                0,
            ],
        };

        // SubstructureRedirectMask | SubstructureNotifyMask
        XSendEvent(display, root, 0, 1 << 20 | 1 << 19, &mut event);
        XFlush(display);
        XCloseDisplay(display);
    }
}

// X11 fullscreen toggle on key. macOS gets native fullscreen for free (Cmd+Ctrl+F / the green button
// on the CEF-created NSWindow), so no keyboard handler is installed there.
#[cfg(target_os = "linux")]
wrap_keyboard_handler! {
    struct ShallotKeyboardHandler;

    impl KeyboardHandler {
        fn on_pre_key_event(
            &self,
            browser: Option<&mut Browser>,
            event: Option<&KeyEvent>,
            _os_event: Option<&mut sys::XEvent>,
            _is_keyboard_shortcut: Option<&mut ::std::os::raw::c_int>,
        ) -> ::std::os::raw::c_int {
            let Some(event) = event else { return 0 };
            if event.type_ != KeyEventType::RAWKEYDOWN {
                return 0;
            }

            let key = event.windows_key_code;
            let alt = event.modifiers & EVENTFLAG_ALT != 0;

            let toggle = key == VK_F11 || (key == VK_RETURN && alt);
            let escape = key == VK_ESCAPE;

            if toggle || escape {
                if let Some(browser) = browser {
                    if let Some(host) = browser.host() {
                        let xid = host.window_handle();
                        if xid != 0 {
                            let action = if escape { 0 } else { 2 }; // remove vs toggle
                            set_x11_fullscreen(xid, action);
                        }
                    }
                }
                if toggle {
                    return 1;
                }
            }

            0
        }
    }
}

wrap_client! {
    struct ShallotClient;

    impl Client {
        fn life_span_handler(&self) -> Option<LifeSpanHandler> {
            Some(ShallotLifeSpanHandler::new())
        }

        fn keyboard_handler(&self) -> Option<KeyboardHandler> {
            #[cfg(target_os = "linux")]
            {
                Some(ShallotKeyboardHandler::new())
            }
            #[cfg(not(target_os = "linux"))]
            {
                None
            }
        }
    }
}

wrap_life_span_handler! {
    struct ShallotLifeSpanHandler;

    impl LifeSpanHandler {
        fn on_before_close(&self, _browser: Option<&mut Browser>) {
            quit_message_loop();
        }
    }
}

#[cfg(target_os = "linux")]
fn find_cef_dir() -> Option<std::path::PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let exe_dir = exe.parent()?;
    let cef_dir = exe_dir.join("cef");
    if cef_dir.join("libcef.so").exists() {
        return Some(cef_dir);
    }
    if exe_dir.join("libcef.so").exists() {
        return Some(exe_dir.to_path_buf());
    }
    None
}

#[cfg(target_os = "linux")]
pub fn run() {
    let _ = api_hash(sys::CEF_API_VERSION_LAST, 0);

    let args = args::Args::new();
    let main_args = args.as_main_args();

    let mut app = ShallotApp::new();

    // Single-binary subprocess model: CEF re-execs this exe for renderer/GPU/etc. (browser_subprocess_path
    // below). Pass the app so each subprocess also runs on_before_command_line_processing — the
    // WebGPU/GPU switches must reach the GPU process, not just the browser, or its adapter stays
    // blocklisted. execute_process returns >= 0 in those subprocesses; the browser process gets -1.
    let ret = execute_process(Some(main_args), Some(&mut app), std::ptr::null_mut());
    if ret >= 0 {
        std::process::exit(ret);
    }

    let origin = start_asset_server();
    ORIGIN.set(origin).expect("origin already set");

    let exe = std::env::current_exe().expect("failed to get exe path");
    let exe_str = exe.to_string_lossy().to_string();

    let name = app_name();
    let cef_cache = cache_dir(&name).join("cef");
    let cef_cache_str = cef_cache.to_string_lossy().to_string();

    let mut settings = Settings {
        no_sandbox: 1,
        cache_path: CefString::from(cef_cache_str.as_str()),
        browser_subprocess_path: CefString::from(exe_str.as_str()),
        ..Default::default()
    };

    if let Some(cef_dir) = find_cef_dir() {
        let dir_str = cef_dir.to_string_lossy().to_string();
        settings.resources_dir_path = CefString::from(dir_str.as_str());
        let locales = cef_dir.join("locales");
        let locales_str = locales.to_string_lossy().to_string();
        settings.locales_dir_path = CefString::from(locales_str.as_str());
    }

    assert_eq!(
        initialize(
            Some(main_args),
            Some(&settings),
            Some(&mut app),
            std::ptr::null_mut()
        ),
        1,
        "CEF initialization failed"
    );

    run_message_loop();
    shutdown();
}

#[cfg(target_os = "windows")]
fn find_cef_dir() -> Option<std::path::PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let exe_dir = exe.parent()?;
    if exe_dir.join("libcef.dll").exists() {
        return Some(exe_dir.to_path_buf());
    }
    let cef_dir = exe_dir.join("cef");
    if cef_dir.join("libcef.dll").exists() {
        return Some(cef_dir);
    }
    None
}

// Windows portable CEF: same single-binary subprocess model as Linux (the exe re-execs itself for
// renderer/GPU/etc.), reading libcef.dll + the resource paks beside it. WebGPU runs on the default
// Dawn backend (D3D12), so only `enable-unsafe-webgpu` is needed (set in the command-line handler).
// No fullscreen key handler here yet — the system-webview build owns that path.
#[cfg(target_os = "windows")]
pub fn run() {
    let _ = api_hash(sys::CEF_API_VERSION_LAST, 0);

    let args = args::Args::new();
    let main_args = args.as_main_args();

    let mut app = ShallotApp::new();

    // Pass the app so each re-exec'd subprocess (GPU, renderer) also runs
    // on_before_command_line_processing — the WebGPU/GPU switches must reach the GPU process, not
    // just the browser, or its adapter stays blocklisted. Subprocesses return >= 0 and exit here.
    let ret = execute_process(Some(main_args), Some(&mut app), std::ptr::null_mut());
    if ret >= 0 {
        std::process::exit(ret);
    }

    let origin = start_asset_server();
    ORIGIN.set(origin).expect("origin already set");

    let exe = std::env::current_exe().expect("failed to get exe path");
    let exe_str = exe.to_string_lossy().to_string();

    let name = app_name();
    let cef_cache = cache_dir(&name).join("cef");
    let cef_cache_str = cef_cache.to_string_lossy().to_string();

    let mut settings = Settings {
        no_sandbox: 1,
        cache_path: CefString::from(cef_cache_str.as_str()),
        browser_subprocess_path: CefString::from(exe_str.as_str()),
        ..Default::default()
    };

    if let Some(cef_dir) = find_cef_dir() {
        let dir_str = cef_dir.to_string_lossy().to_string();
        settings.resources_dir_path = CefString::from(dir_str.as_str());
        let locales = cef_dir.join("locales");
        let locales_str = locales.to_string_lossy().to_string();
        settings.locales_dir_path = CefString::from(locales_str.as_str());
    }

    assert_eq!(
        initialize(
            Some(main_args),
            Some(&settings),
            Some(&mut app),
            std::ptr::null_mut()
        ),
        1,
        "CEF initialization failed"
    );

    run_message_loop();
    shutdown();
}

#[cfg(target_os = "macos")]
pub fn run() {
    use cef::library_loader::LibraryLoader;

    // Load the framework from Contents/Frameworks before any CEF call, then install the
    // CEF-compatible NSApplication. Both must happen before initialize.
    let loader = LibraryLoader::new(
        &std::env::current_exe().expect("failed to get exe path"),
        false,
    );
    assert!(loader.load(), "failed to load CEF framework");

    let _ = api_hash(sys::CEF_API_VERSION_LAST, 0);

    crate::mac::setup_application();

    let args = args::Args::new();
    let main_args = args.as_main_args();

    // Subprocesses run the separate helper bundles, so the main bundle executable is always the
    // browser process here (-1). Helpers never re-enter this code.
    let ret = execute_process(Some(main_args), None, std::ptr::null_mut());
    assert_eq!(ret, -1, "unexpected subprocess in main bundle executable");

    let origin = start_asset_server();
    ORIGIN.set(origin).expect("origin already set");

    let name = app_name();
    let cef_cache = cache_dir(&name).join("cef");
    let cef_cache_str = cef_cache.to_string_lossy().to_string();

    // framework_dir_path, main_bundle_path, and the helper subprocess paths are auto-discovered from
    // the loaded framework and the standard .app layout (Contents/Frameworks/<name> Helper (*).app).
    let settings = Settings {
        no_sandbox: 1,
        cache_path: CefString::from(cef_cache_str.as_str()),
        ..Default::default()
    };

    let mut app = ShallotApp::new();

    assert_eq!(
        initialize(
            Some(main_args),
            Some(&settings),
            Some(&mut app),
            std::ptr::null_mut()
        ),
        1,
        "CEF initialization failed"
    );

    run_message_loop();
    shutdown();
    drop(loader);
}
