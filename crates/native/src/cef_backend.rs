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

mod policy;

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
                    // Cross-origin isolation permits shared-memory physics workers.
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

#[cfg(target_os = "linux")]
fn uses_wayland(ozone_platform: Option<&str>, session_type: Option<&str>) -> bool {
    // Chromium resolves an absent ozone-platform from XDG_SESSION_TYPE after this callback.
    ozone_platform.or(session_type) == Some("wayland")
}

wrap_app! {
    pub struct ShallotApp;

    impl App {
        fn render_process_handler(&self) -> Option<RenderProcessHandler> {
            Some(crate::gpu_policy::PolicyHandler::new())
        }

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
                // ANGLE Vulkan avoids Wayland stalls. On composited X11 its root-visual configs
                // can fail Chromium's ARGB visual selection.
                #[cfg(target_os = "linux")]
                {
                    let ozone = CefString::from("ozone-platform");
                    let platform = (cmd.has_switch(Some(&ozone)) != 0)
                        .then(|| CefString::from(&cmd.switch_value(Some(&ozone))).to_string());
                    let session_type = std::env::var("XDG_SESSION_TYPE").ok();
                    if uses_wayland(platform.as_deref(), session_type.as_deref()) {
                        cmd.append_switch_with_value(
                            Some(&CefString::from("use-angle")),
                            Some(&CefString::from("vulkan")),
                        );
                    }
                }
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
                    Some(&CefString::from("0")),
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
            let fallback = format!("{}/", origin);
            let launch_url = crate::launch_url(std::env::var("SHALLOT_DEV_URL").ok(), fallback);
            let url = CefString::from(launch_url.as_str());
            let settings = BrowserSettings::default();
            let mut client = ShallotClient::new();
            let mut delegate = ShallotBrowserViewDelegate::new();
            let view = browser_view_create(
                Some(&mut client), Some(&url), Some(&settings), None, None,
                Some(&mut delegate),
            ).expect("failed to create app browser view");
            let icons = window_icons(&crate::window_icon(), || image_create().expect("CEF could not create native icon image"));
            let mut window_delegate = ShallotWindowDelegate::new(std::cell::RefCell::new(Some(view)), Some(icons));
            window_create_top_level(Some(&mut window_delegate)).expect("failed to create app window");
        }
    }
}

wrap_browser_view_delegate! {
    struct ShallotBrowserViewDelegate {}
    impl ViewDelegate {}
    impl BrowserViewDelegate {
        fn browser_runtime_style(&self) -> RuntimeStyle { RuntimeStyle::CHROME }
        fn chrome_toolbar_type(&self, _view: Option<&mut BrowserView>) -> ChromeToolbarType {
            ChromeToolbarType::NONE
        }
    }
}

fn window_icons(png: &image::RgbaImage, mut create: impl FnMut() -> Image) -> (Image, Image) {
    let mut make = |size: u32| {
        let image = create();
        fill_icon(&image, png, size);
        image
    };
    (make(16), make(32))
}

fn fill_icon(image: &Image, png: &image::RgbaImage, size: u32) {
    for scale in [1_u32, 2] {
        let pixels = image::imageops::resize(
            png,
            size * scale,
            size * scale,
            image::imageops::FilterType::Lanczos3,
        );
        assert_ne!(
            image.add_bitmap(
                scale as f32,
                (size * scale) as i32,
                (size * scale) as i32,
                ColorType::RGBA_8888,
                AlphaType::POSTMULTIPLIED,
                Some(pixels.as_raw())
            ),
            0,
            "CEF refused native icon bitmap"
        );
    }
}

fn set_window_icons(window: &Window, mut small: Image, mut app: Image) {
    window.set_window_icon(Some(&mut small));
    window.set_window_app_icon(Some(&mut app));
}

wrap_window_delegate! {
    struct ShallotWindowDelegate {
        view: std::cell::RefCell<Option<BrowserView>>,
        icons: Option<(Image, Image)>,
    }
    impl ViewDelegate {
        fn preferred_size(&self, _view: Option<&mut View>) -> Size {
            Size { width: 800, height: 600 }
        }
    }
    impl PanelDelegate {}
    impl WindowDelegate {
        fn window_runtime_style(&self) -> RuntimeStyle { RuntimeStyle::CHROME }
        // Rust delegates default boolean callbacks to false, unlike CEF's C++ delegates.
        fn with_standard_window_buttons(&self, _window: Option<&mut Window>) -> i32 { 1 }
        fn can_resize(&self, _window: Option<&mut Window>) -> i32 { 1 }
        fn can_maximize(&self, _window: Option<&mut Window>) -> i32 { 1 }
        fn can_minimize(&self, _window: Option<&mut Window>) -> i32 { 1 }
        fn on_window_created(&self, window: Option<&mut Window>) {
            if let (Some(window), Some(view)) = (window, self.view.borrow().as_ref()) {
                window.set_title(Some(&CefString::from(app_name().as_str())));
                if let Some((small, app)) = &self.icons {
                    set_window_icons(window, small.clone(), app.clone());
                }
                window.set_to_fill_layout();
                window.add_child_view(Some(&mut view.into()));
                window.show();
                view.request_focus();
            }
        }
        fn can_close(&self, _window: Option<&mut Window>) -> i32 {
            self.view.borrow().as_ref().and_then(|view| view.browser())
                .and_then(|browser| browser.host()).map_or(1, |host| host.try_close_browser())
        }
        fn on_window_destroyed(&self, _window: Option<&mut Window>) {
            self.view.borrow_mut().take();
        }
    }
}

wrap_display_handler! {
    struct ShallotDisplayHandler;
    impl DisplayHandler {
        fn on_title_change(&self, browser: Option<&mut Browser>, title: Option<&CefString>) {
            if let Some(window) = browser_view_get_for_browser(browser).and_then(|view| view.window()) {
                let fallback = CefString::from(app_name().as_str());
                window.set_title(Some(title.filter(|title| !title.to_string().is_empty()).unwrap_or(&fallback)));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn main_app_installs_context_policy_handler() {
        assert!(ShallotApp::new().render_process_handler().is_some());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cef_sets_separate_title_bar_and_app_icons_with_decoded_scaled_rgba() {
        use cef::rc::ConvertReturnValue;
        use std::cell::RefCell;
        thread_local! { static EVENTS: RefCell<Vec<(f32, i32, i32)>> = const { RefCell::new(Vec::new()) }; }
        unsafe extern "C" fn bitmap(
            _image: *mut sys::cef_image_t,
            scale: f32,
            width: i32,
            height: i32,
            color: sys::cef_color_type_t,
            alpha: sys::cef_alpha_type_t,
            pixels: *const std::ffi::c_void,
            len: usize,
        ) -> i32 {
            assert_eq!(color, sys::cef_color_type_t::CEF_COLOR_TYPE_RGBA_8888);
            assert_eq!(alpha, sys::cef_alpha_type_t::CEF_ALPHA_TYPE_POSTMULTIPLIED);
            assert_eq!(len, (width * height * 4) as usize);
            assert_eq!(
                std::slice::from_raw_parts(pixels.cast::<u8>(), 4),
                [255, 0, 42, 128]
            );
            EVENTS.with(|events| events.borrow_mut().push((scale, width, height)));
            1
        }
        unsafe extern "C" fn small(_window: *mut sys::cef_window_t, image: *mut sys::cef_image_t) {
            assert!(!image.is_null());
            assert_eq!((*image).get_width.unwrap()(image), 16);
            EVENTS.with(|events| events.borrow_mut().push((0.0, 16, 16)));
        }
        unsafe extern "C" fn app(_window: *mut sys::cef_window_t, image: *mut sys::cef_image_t) {
            assert!(!image.is_null());
            assert_eq!((*image).get_width.unwrap()(image), 32);
            EVENTS.with(|events| events.borrow_mut().push((0.0, 32, 32)));
        }
        let png = image::RgbaImage::from_pixel(7, 7, image::Rgba([255, 0, 42, 128]));
        let mut raw_small: sys::cef_image_t = unsafe { std::mem::zeroed() };
        let mut raw_app: sys::cef_image_t = unsafe { std::mem::zeroed() };
        unsafe extern "C" fn small_width(_image: *mut sys::cef_image_t) -> usize {
            16
        }
        unsafe extern "C" fn app_width(_image: *mut sys::cef_image_t) -> usize {
            32
        }
        raw_small.add_bitmap = Some(bitmap);
        raw_app.add_bitmap = Some(bitmap);
        raw_small.get_width = Some(small_width);
        raw_app.get_width = Some(app_width);
        let small_image = (&mut raw_small as *mut sys::cef_image_t).wrap_result();
        let app_image = (&mut raw_app as *mut sys::cef_image_t).wrap_result();
        let mut images = [small_image, app_image].into_iter();
        let (small_image, app_image) = window_icons(&png, || images.next().unwrap());
        let mut raw_window: sys::cef_window_t = unsafe { std::mem::zeroed() };
        raw_window.set_window_icon = Some(small);
        raw_window.set_window_app_icon = Some(app);
        let mut window = (&mut raw_window as *mut sys::cef_window_t).wrap_result();
        let mut raw_view: sys::cef_browser_view_t = unsafe { std::mem::zeroed() };
        let view = (&mut raw_view as *mut sys::cef_browser_view_t).wrap_result();
        let delegate =
            ShallotWindowDelegate::new(RefCell::new(Some(view)), Some((small_image, app_image)));
        delegate.on_window_created(Some(&mut window));
        EVENTS.with(|events| {
            assert_eq!(
                *events.borrow(),
                [
                    (1.0, 16, 16),
                    (2.0, 32, 32),
                    (1.0, 32, 32),
                    (2.0, 64, 64),
                    (0.0, 16, 16),
                    (0.0, 32, 32)
                ]
            )
        });
    }

    #[test]
    fn app_window_uses_chrome_without_browser_controls() {
        let delegate = ShallotBrowserViewDelegate::new();
        assert_eq!(delegate.browser_runtime_style(), RuntimeStyle::CHROME);
        assert_eq!(delegate.chrome_toolbar_type(None), ChromeToolbarType::NONE);
        let window = ShallotWindowDelegate::new(std::cell::RefCell::new(None), None);
        assert_eq!(window.window_runtime_style(), RuntimeStyle::CHROME);
        assert_eq!(window.is_frameless(None), 0);
        assert_eq!(window.with_standard_window_buttons(None), 1);
        let size = window.preferred_size(None);
        assert_eq!((size.width, size.height), (800, 600));
        assert_eq!(window.can_resize(None), 1);
        assert_eq!(window.can_maximize(None), 1);
        assert_eq!(window.can_minimize(None), 1);
        assert_eq!(window.can_close(None), 1);
        window.on_window_destroyed(None);
        assert_eq!(window.can_close(None), 1);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn window_attaches_only_the_browser_shows_focuses_and_releases_it() {
        use cef::rc::ConvertReturnValue;
        use std::cell::RefCell;
        thread_local! { static EVENTS: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) }; }
        unsafe extern "C" fn fill(_panel: *mut sys::cef_panel_t) -> *mut sys::cef_fill_layout_t {
            EVENTS.with(|events| events.borrow_mut().push("fill"));
            std::ptr::null_mut()
        }
        unsafe extern "C" fn attach(_panel: *mut sys::cef_panel_t, view: *mut sys::cef_view_t) {
            assert!(!view.is_null());
            EVENTS.with(|events| events.borrow_mut().push("browser"));
        }
        unsafe extern "C" fn show(_window: *mut sys::cef_window_t) {
            EVENTS.with(|events| events.borrow_mut().push("show"));
        }
        unsafe extern "C" fn focus(_view: *mut sys::cef_view_t) {
            EVENTS.with(|events| events.borrow_mut().push("focus"));
        }
        unsafe extern "C" fn release(_base: *mut sys::cef_base_ref_counted_t) -> i32 {
            EVENTS.with(|events| events.borrow_mut().push("release"));
            0
        }
        // CEF interface stubs stay alive until every wrapper has released its reference.
        let mut view: sys::cef_browser_view_t = unsafe { std::mem::zeroed() };
        view.base.request_focus = Some(focus);
        view.base.base.release = Some(release);
        let mut native_window: sys::cef_window_t = unsafe { std::mem::zeroed() };
        native_window.base.set_to_fill_layout = Some(fill);
        native_window.base.add_child_view = Some(attach);
        native_window.show = Some(show);
        let view = (&mut view as *mut sys::cef_browser_view_t).wrap_result();
        let mut window = (&mut native_window as *mut sys::cef_window_t).wrap_result();
        let delegate = ShallotWindowDelegate::new(RefCell::new(Some(view)), None);
        delegate.on_window_created(Some(&mut window));
        EVENTS.with(|events| {
            assert_eq!(
                *events.borrow(),
                ["fill", "browser", "release", "show", "focus", "release"]
            )
        });
        delegate.on_window_destroyed(Some(&mut window));
        EVENTS.with(|events| assert_eq!(events.borrow().last(), Some(&"release")));
        assert_eq!(delegate.can_close(Some(&mut window)), 1);
        delegate.on_window_created(Some(&mut window));
        EVENTS.with(|events| assert_eq!(events.borrow().len(), 7));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn window_close_waits_for_the_browser() {
        use cef::rc::ConvertReturnValue;
        use std::cell::Cell;
        thread_local! {
            static HOST: Cell<*mut sys::cef_browser_host_t> = const { Cell::new(std::ptr::null_mut()) };
            static BROWSER: Cell<*mut sys::cef_browser_t> = const { Cell::new(std::ptr::null_mut()) };
            static READY: Cell<i32> = const { Cell::new(0) };
        }
        unsafe extern "C" fn browser(
            _view: *mut sys::cef_browser_view_t,
        ) -> *mut sys::cef_browser_t {
            BROWSER.with(Cell::get)
        }
        unsafe extern "C" fn host(
            _browser: *mut sys::cef_browser_t,
        ) -> *mut sys::cef_browser_host_t {
            HOST.with(Cell::get)
        }
        unsafe extern "C" fn close(_host: *mut sys::cef_browser_host_t) -> i32 {
            READY.with(Cell::get)
        }
        let mut native_host: sys::cef_browser_host_t = unsafe { std::mem::zeroed() };
        native_host.try_close_browser = Some(close);
        let mut native_browser: sys::cef_browser_t = unsafe { std::mem::zeroed() };
        native_browser.get_host = Some(host);
        let mut native_view: sys::cef_browser_view_t = unsafe { std::mem::zeroed() };
        native_view.get_browser = Some(browser);
        HOST.with(|value| value.set(&mut native_host));
        BROWSER.with(|value| value.set(&mut native_browser));
        let view = (&mut native_view as *mut sys::cef_browser_view_t).wrap_result();
        let delegate = ShallotWindowDelegate::new(std::cell::RefCell::new(Some(view)), None);
        assert_eq!(delegate.can_close(None), 0);
        READY.with(|value| value.set(1));
        assert_eq!(delegate.can_close(None), 1);
        delegate.on_window_destroyed(None);
        READY.with(|value| value.set(0));
        assert_eq!(delegate.can_close(None), 1);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn fullscreen_shortcuts_consume_toggles_but_leave_escape_to_the_page() {
        let handler = ShallotKeyboardHandler::new();
        let mut event = KeyEvent {
            type_: KeyEventType::RAWKEYDOWN,
            windows_key_code: VK_F11,
            ..Default::default()
        };
        assert_eq!(handler.on_pre_key_event(None, Some(&event), None, None), 1);
        event.windows_key_code = VK_RETURN;
        assert_eq!(handler.on_pre_key_event(None, Some(&event), None, None), 0);
        event.modifiers = EVENTFLAG_ALT;
        assert_eq!(handler.on_pre_key_event(None, Some(&event), None, None), 1);
        event.windows_key_code = VK_ESCAPE;
        assert_eq!(handler.on_pre_key_event(None, Some(&event), None, None), 0);
        event.windows_key_code = VK_F11;
        event.type_ = KeyEventType::KEYUP;
        assert_eq!(handler.on_pre_key_event(None, Some(&event), None, None), 0);
    }

    #[test]
    fn remote_debugging_uses_an_ephemeral_port_only_in_debug_builds() {
        let _ = api_hash(sys::CEF_API_VERSION_LAST, 0);
        let app = ShallotApp::new();
        for process in [None, Some(CefString::from("gpu-process"))] {
            let mut switches = command_line_create().unwrap();
            app.on_before_command_line_processing(process.as_ref(), Some(&mut switches));
            assert_eq!(
                switches.has_switch(Some(&CefString::from("remote-debugging-port"))),
                i32::from(cfg!(debug_assertions))
            );
            assert_eq!(
                CefString::from(
                    &switches.switch_value(Some(&CefString::from("remote-debugging-port")))
                )
                .to_string(),
                if cfg!(debug_assertions) { "0" } else { "" }
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn wayland_selection_matches_chromium() {
        for session in [None, Some(""), Some("tty"), Some("x11"), Some("wayland")] {
            assert_eq!(uses_wayland(None, session), session == Some("wayland"));
            assert!(uses_wayland(Some("wayland"), session));
            assert!(!uses_wayland(Some("x11"), session));
            assert!(!uses_wayland(Some(""), session));
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_scopes_angle_vulkan_to_wayland_without_forcing_the_vulkan_compositor() {
        let _ = api_hash(sys::CEF_API_VERSION_LAST, 0);
        let app = ShallotApp::new();
        for platform in ["x11", "wayland"] {
            for process in [None, Some(CefString::from("gpu-process"))] {
                let mut switches = command_line_create().unwrap();
                switches.append_switch_with_value(
                    Some(&CefString::from("ozone-platform")),
                    Some(&CefString::from(platform)),
                );
                app.on_before_command_line_processing(process.as_ref(), Some(&mut switches));
                assert_eq!(
                    CefString::from(&switches.switch_value(Some(&CefString::from("use-angle"))))
                        .to_string(),
                    if platform == "wayland" { "vulkan" } else { "" }
                );
                assert_eq!(
                    CefString::from(
                        &switches.switch_value(Some(&CefString::from("ozone-platform")))
                    )
                    .to_string(),
                    platform
                );
                for name in ["enable-unsafe-webgpu", "ignore-gpu-blocklist"] {
                    assert_eq!(switches.has_switch(Some(&CefString::from(name))), 1);
                }
                assert_eq!(
                    CefString::from(
                        &switches.switch_value(Some(&CefString::from("enable-dawn-features")))
                    )
                    .to_string(),
                    "allow_unsafe_apis"
                );
                let enabled_features = CefString::from(
                    &switches.switch_value(Some(&CefString::from("enable-features"))),
                )
                .to_string();
                assert!(enabled_features
                    .split(',')
                    .all(|feature| feature != "Vulkan"));
            }
        }
    }
}

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
                    if let Some(window) = browser_view_get_for_browser(Some(browser)).and_then(|view| view.window()) {
                        window.set_fullscreen(if escape { 0 } else { 1 - window.is_fullscreen() });
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
        fn command_handler(&self) -> Option<CommandHandler> {
            Some(policy::AppCommandHandler::new())
        }

        fn context_menu_handler(&self) -> Option<ContextMenuHandler> {
            Some(policy::AppContextMenuHandler::new())
        }

        fn request_handler(&self) -> Option<RequestHandler> {
            Some(policy::AppRequestHandler::new())
        }

        fn display_handler(&self) -> Option<DisplayHandler> {
            Some(ShallotDisplayHandler::new())
        }

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
        fn on_before_popup(
            &self,
            _browser: Option<&mut Browser>,
            _frame: Option<&mut Frame>,
            _popup_id: i32,
            _target_url: Option<&CefString>,
            _target_frame_name: Option<&CefString>,
            _target_disposition: WindowOpenDisposition,
            _user_gesture: i32,
            _popup_features: Option<&PopupFeatures>,
            _window_info: Option<&mut WindowInfo>,
            _client: Option<&mut Option<Client>>,
            _settings: Option<&mut BrowserSettings>,
            _extra_info: Option<&mut Option<DictionaryValue>>,
            _no_javascript_access: Option<&mut i32>,
        ) -> i32 { 1 }

        fn on_before_dev_tools_popup(
            &self,
            _browser: Option<&mut Browser>,
            _window_info: Option<&mut WindowInfo>,
            client: Option<&mut Option<Client>>,
            _settings: Option<&mut BrowserSettings>,
            _extra_info: Option<&mut Option<DictionaryValue>>,
            use_default_window: Option<&mut i32>,
        ) {
            // DevTools must not inherit app command filtering or app exit lifetime.
            if let Some(client) = client { *client = Some(DevToolsClient::new()); }
            if let Some(use_default_window) = use_default_window { *use_default_window = 1; }
        }

        fn on_before_close(&self, _browser: Option<&mut Browser>) {
            quit_message_loop();
        }
    }
}

wrap_client! {
    struct DevToolsClient;
    impl Client {
        fn life_span_handler(&self) -> Option<LifeSpanHandler> {
            Some(DevToolsLifeSpanHandler::new())
        }
    }
}

wrap_life_span_handler! {
    struct DevToolsLifeSpanHandler;
    impl LifeSpanHandler {
        fn on_before_close(&self, _browser: Option<&mut Browser>) {}
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
