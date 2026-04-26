use cef::*;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::OnceLock;

use crate::{app_name, asset_dir, cache_dir, content_type};

const VK_F11: i32 = 0x7A;
const VK_RETURN: i32 = 0x0D;
const VK_ESCAPE: i32 = 0x1B;
const EVENTFLAG_ALT: u32 = 8;

static ORIGIN: OnceLock<String> = OnceLock::new();

fn start_asset_server() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("failed to bind asset server");
    let port = listener.local_addr().unwrap().port();
    let origin = format!("http://127.0.0.1:{}", port);
    eprintln!("[shallot] asset server on {}", origin);

    std::thread::spawn(move || {
        let dist = asset_dir();
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

            let file_path = dist.join(&path[1..]);
            match std::fs::read(&file_path) {
                Ok(data) => {
                    let mime = content_type(path);
                    let header = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
                        mime, data.len()
                    );
                    let _ = stream.write_all(header.as_bytes());
                    let _ = stream.write_all(&data);
                }
                Err(_) => {
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
                cmd.append_switch(Some(&CefString::from("enable-unsafe-webgpu")));
                cmd.append_switch(Some(&CefString::from("enable-features=Vulkan")));
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
            Some(ShallotKeyboardHandler::new())
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

pub fn run() {
    let _ = api_hash(sys::CEF_API_VERSION_LAST, 0);

    let args = args::Args::new();
    let main_args = args.as_main_args();

    let ret = execute_process(Some(main_args), None, std::ptr::null_mut());
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
}
