use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{CursorGrabMode, Fullscreen, Icon, Window, WindowAttributes, WindowId};
use wry::http::Response;
use wry::{Rect, WebContext, WebViewBuilder};

#[cfg(windows)]
use wry::WebViewBuilderExtWindows;

use crate::{app_name, asset, cache_dir, content_type, BG};

const FULLSCREEN_JS: &str = r#"
document.addEventListener('keydown', e => {
    if (e.key === 'F11' ||
        (e.key === 'Enter' && e.altKey) ||
        (e.key === 'f' && e.metaKey && e.ctrlKey)) {
        e.preventDefault();
        window.ipc.postMessage('fullscreen:toggle');
    } else if (e.key === 'Escape') {
        window.ipc.postMessage('fullscreen:exit');
    }
});
"#;

// WKWebView doesn't grant the Pointer Lock API (Chromium does) — wry's UI delegate has no
// pointer-lock method and there's no preference that enables it — so the web engine's
// `requestPointerLock()` silently fails and a pointer-lock controller never engages. Tauri hits the
// same wall and exposes the winit cursor grab for the app to drive look itself; we do that under the
// hood instead, so engine code stays standard. The shim polyfills the API over the IPC bridge below:
// requestPointerLock grabs the OS cursor (winit `CursorGrabMode::Locked`, the `pointer:lock` handler),
// and native mouse-motion deltas come back through `__shallot_mouse_delta` as synthetic locked
// `mousemove` events — the native-event-to-injected-JS-MouseEvent pattern is wry's own
// (`wkwebview/synthetic_mouse_events.rs`).
//
// macOS only. WebView2 (Windows) and CEF are full Chromium with working native pointer lock. The
// polyfill actively breaks them: winit's `CursorGrabMode::Locked` is unsupported on Windows (the grab
// errors, swallowed by `let _`), and raw `DeviceEvent::MouseMotion` reaches the focused WebView2 child
// HWND, not the winit parent, so the cursor never locks and no deltas arrive. The `pointer:lock` IPC
// arms and the `device_event` injection below stay uniform but inert off macOS: nothing posts the
// messages that arm them.
#[cfg(target_os = "macos")]
const POINTERLOCK_JS: &str = r#"
(function () {
    let el = null;
    Object.defineProperty(document, 'pointerLockElement', { configurable: true, get: () => el });
    Element.prototype.requestPointerLock = function () {
        el = this;
        window.ipc.postMessage('pointer:lock');
        document.dispatchEvent(new Event('pointerlockchange'));
        return Promise.resolve();
    };
    document.exitPointerLock = function () {
        if (!el) return;
        el = null;
        window.ipc.postMessage('pointer:unlock');
        document.dispatchEvent(new Event('pointerlockchange'));
    };
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape' && el) document.exitPointerLock();
    });
    window.__shallot_mouse_delta = function (dx, dy) {
        if (!el) return;
        const e = new MouseEvent('mousemove', { bubbles: true });
        Object.defineProperty(e, 'movementX', { value: dx });
        Object.defineProperty(e, 'movementY', { value: dy });
        document.dispatchEvent(e);
    };
})();
"#;

fn toggle_fullscreen(window: &Window) {
    let fs = if window.fullscreen().is_some() {
        None
    } else {
        Some(Fullscreen::Borderless(None))
    };
    window.set_fullscreen(fs);
}

#[cfg(windows)]
fn style_window(hwnd: isize, r: u8, g: u8, b: u8) {
    #[link(name = "dwmapi")]
    extern "system" {
        fn DwmSetWindowAttribute(hwnd: isize, attr: u32, data: *const u32, size: u32) -> i32;
    }
    let colorref: u32 = r as u32 | (g as u32) << 8 | (b as u32) << 16;
    let text: u32 = 0x00E8ECF0;
    unsafe {
        DwmSetWindowAttribute(hwnd, 34, &colorref, 4);
        DwmSetWindowAttribute(hwnd, 35, &colorref, 4);
        DwmSetWindowAttribute(hwnd, 36, &text, 4);
    }
}

fn load_icon() -> Option<Icon> {
    let data = asset("icon.png")?;
    let img = image::load_from_memory(&data).ok()?.into_rgba8();
    let (w, h) = img.dimensions();
    Icon::from_rgba(img.into_raw(), w, h).ok()
}

struct App {
    title: String,
    ctx: WebContext,
    window: Rc<RefCell<Option<Window>>>,
    webview: Option<wry::WebView>,
    pointer_locked: Rc<Cell<bool>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.borrow().is_some() {
            return;
        }

        let mut attrs = WindowAttributes::default()
            .with_title(&self.title)
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
            .with_theme(Some(winit::window::Theme::Dark));
        if let Some(icon) = load_icon() {
            attrs = attrs.with_window_icon(Some(icon));
        }

        let window = event_loop
            .create_window(attrs)
            .expect("failed to create window");

        #[cfg(windows)]
        {
            use winit::raw_window_handle::HasWindowHandle;
            if let Ok(handle) = window.window_handle() {
                if let winit::raw_window_handle::RawWindowHandle::Win32(h) = handle.as_raw() {
                    style_window(h.hwnd.get() as isize, BG.0, BG.1, BG.2);
                }
            }
        }

        let win_ref = self.window.clone();
        let lock_ref = self.pointer_locked.clone();
        let builder = WebViewBuilder::with_web_context(&mut self.ctx)
            .with_background_color((BG.0, BG.1, BG.2, 255))
            .with_initialization_script(FULLSCREEN_JS)
            .with_ipc_handler(move |request| {
                let msg = request.body();
                let win = win_ref.borrow();
                let Some(window) = win.as_ref() else { return };
                match msg.as_str() {
                    "fullscreen:toggle" => toggle_fullscreen(window),
                    "fullscreen:exit" => {
                        if window.fullscreen().is_some() {
                            window.set_fullscreen(None);
                        }
                    }
                    "pointer:lock" => {
                        let _ = window.set_cursor_grab(CursorGrabMode::Locked);
                        window.set_cursor_visible(false);
                        lock_ref.set(true);
                    }
                    "pointer:unlock" => {
                        let _ = window.set_cursor_grab(CursorGrabMode::None);
                        window.set_cursor_visible(true);
                        lock_ref.set(false);
                    }
                    _ => {}
                }
            })
            .with_custom_protocol("shallot".into(), move |_id, request| {
                let path = request.uri().path();
                let path = if path == "/" { "/index.html" } else { path };

                match asset(&path[1..]) {
                    Some(data) => Response::builder()
                        .header("Content-Type", content_type(path))
                        .header("Access-Control-Allow-Origin", "*")
                        // COOP/COEP: cross-origin isolation so tumble physics can multithread
                        // (mirrors the JS serve surfaces' CROSS_ORIGIN_ISOLATION in project/vite.ts)
                        .header("Cross-Origin-Opener-Policy", "same-origin")
                        .header("Cross-Origin-Embedder-Policy", "require-corp")
                        .body(data)
                        .unwrap(),
                    None => Response::builder()
                        .status(404)
                        .body(Cow::Borrowed(&[] as &[u8]))
                        .unwrap(),
                }
            })
            .with_url("shallot://localhost/");

        #[cfg(target_os = "macos")]
        let builder = builder.with_initialization_script(POINTERLOCK_JS);

        #[cfg(windows)]
        let builder = builder.with_theme(wry::Theme::Dark);

        #[cfg(debug_assertions)]
        let builder = builder.with_devtools(true);

        let size = window.inner_size();
        let webview = builder
            .with_bounds(Rect {
                position: wry::dpi::Position::Logical(wry::dpi::LogicalPosition::new(0.0, 0.0)),
                size: wry::dpi::Size::Physical(wry::dpi::PhysicalSize::new(
                    size.width.max(1),
                    size.height.max(1),
                )),
            })
            .build_as_child(&window)
            .expect("failed to create webview");

        *self.window.borrow_mut() = Some(window);
        self.webview = Some(webview);
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.pointer_locked.get() {
                if let Some(webview) = &self.webview {
                    let _ = webview.evaluate_script(&format!(
                        "window.__shallot_mouse_delta({},{})",
                        delta.0, delta.1
                    ));
                }
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::Resized(size) => {
                if let Some(webview) = &self.webview {
                    let _ = webview.set_bounds(Rect {
                        position: wry::dpi::Position::Logical(wry::dpi::LogicalPosition::new(
                            0.0, 0.0,
                        )),
                        size: wry::dpi::Size::Physical(wry::dpi::PhysicalSize::new(
                            size.width,
                            size.height,
                        )),
                    });
                }
            }
            WindowEvent::CloseRequested => {
                self.webview.take();
                *self.window.borrow_mut() = None;
                event_loop.exit();
            }
            _ => {}
        }
    }
}

pub fn run() {
    let title = std::env::args().nth(1).unwrap_or_else(|| "Shallot".into());
    let name = app_name();
    let data_dir = cache_dir(&name).join("webview");
    let ctx = WebContext::new(Some(data_dir));
    let event_loop = EventLoop::new().expect("failed to create event loop");
    let window = Rc::new(RefCell::new(None));
    let pointer_locked = Rc::new(Cell::new(false));
    let mut app = App {
        title,
        ctx,
        window,
        webview: None,
        pointer_locked,
    };
    event_loop.run_app(&mut app).unwrap();
}
