fn parse_require_hardware(value: Option<&str>) -> bool {
    value != Some("1")
}

pub fn require_hardware() -> bool {
    parse_require_hardware(std::env::var("SHALLOT_ALLOW_SOFTWARE_GPU").ok().as_deref())
}

pub fn initialization_script() -> &'static str {
    script(require_hardware())
}

fn script(require_hardware: bool) -> &'static str {
    if require_hardware {
        "Object.defineProperty(globalThis, '__shallotRequireHardwareGpu', { value: true });"
    } else {
        "Object.defineProperty(globalThis, '__shallotRequireHardwareGpu', { value: false });"
    }
}

#[cfg(all(target_os = "linux", feature = "portable"))]
pub fn configure_linux(
    cmd: &mut cef::CommandLine,
    platform: Option<&str>,
    session: Option<&str>,
    hardware: bool,
) {
    use cef::{CefString, ImplCommandLine};
    // Chromium resolves an absent Ozone platform from the session, defaulting to X11.
    let platform = platform.unwrap_or(if session == Some("wayland") {
        "wayland"
    } else {
        "x11"
    });
    match platform {
        "wayland" => {
            // ANGLE Vulkan avoids Wayland stalls, but its X11 configs can lack Chromium's ARGB visual.
            cmd.append_switch_with_value(
                Some(&CefString::from("use-angle")),
                Some(&CefString::from("vulkan")),
            );
        }
        "x11" if hardware => {
            // Vulkan composition supplies the provider needed for hardware WebGPU without
            // ANGLE Vulkan's X11 visual mismatch.
            let key = CefString::from("enable-features");
            let value = CefString::from(&cmd.switch_value(Some(&key))).to_string();
            if !value.split(',').any(|entry| {
                let entry = entry.trim_ascii();
                let entry = entry.strip_prefix('*').unwrap_or(entry);
                entry.split(['<', ':']).next() == Some("Vulkan")
            }) {
                let value = if value.is_empty() {
                    "Vulkan".to_string()
                } else {
                    format!("{value},Vulkan")
                };
                cmd.append_switch_with_value(Some(&key), Some(&CefString::from(value.as_str())));
            }
        }
        _ => {}
    }
}

#[cfg(feature = "portable")]
mod cef_context {
    use cef::*;

    wrap_app! {
        pub struct PolicyApp;

        impl App {
            fn render_process_handler(&self) -> Option<RenderProcessHandler> {
                Some(PolicyHandler::new())
            }
        }
    }

    wrap_render_process_handler! {
        pub struct PolicyHandler;

        impl RenderProcessHandler {
            fn on_context_created(
                &self,
                _browser: Option<&mut Browser>,
                _frame: Option<&mut Frame>,
                context: Option<&mut V8Context>,
            ) {
                if let Some(context) = context {
                    let mut value = None;
                    let mut exception = None;
                    assert_ne!(context.eval(
                        Some(&CefString::from(super::initialization_script())),
                        None, 0, Some(&mut value), Some(&mut exception),
                    ), 0, "failed to publish native GPU policy");
                }
            }
        }
    }
}

#[cfg(all(target_os = "macos", feature = "portable"))]
pub use cef_context::PolicyApp;
#[cfg(feature = "portable")]
pub use cef_context::PolicyHandler;

#[cfg(test)]
mod tests {
    #[cfg(all(target_os = "linux", feature = "portable"))]
    #[test]
    fn linux_launch_preserves_features_and_platform_across_callbacks() {
        use cef::*;
        let _ = api_hash(sys::CEF_API_VERSION_LAST, 0);
        for platform in [
            None,
            Some("x11"),
            Some("wayland"),
            Some("headless"),
            Some(""),
        ] {
            for session in [None, Some("tty"), Some("x11"), Some("wayland")] {
                for hardware in [false, true] {
                    let mut cmd = command_line_create().unwrap();
                    for (key, value) in [
                        ("enable-features", "Other:param/value"),
                        ("disable-features", "Unrelated<Trial,Vulkan:param/value"),
                    ] {
                        cmd.append_switch_with_value(
                            Some(&CefString::from(key)),
                            Some(&CefString::from(value)),
                        );
                    }
                    let effective = platform.unwrap_or(if session == Some("wayland") {
                        "wayland"
                    } else {
                        "x11"
                    });
                    for _ in 0..2 {
                        super::configure_linux(&mut cmd, platform, session, hardware);
                        let get = |key: &str| {
                            CefString::from(&cmd.switch_value(Some(&CefString::from(key))))
                                .to_string()
                        };
                        assert_eq!(
                            get("use-angle"),
                            if effective == "wayland" { "vulkan" } else { "" }
                        );
                        let composition = effective == "x11" && hardware;
                        assert_eq!(
                            get("enable-features"),
                            if composition {
                                "Other:param/value,Vulkan"
                            } else {
                                "Other:param/value"
                            }
                        );
                        assert_eq!(
                            get("disable-features"),
                            "Unrelated<Trial,Vulkan:param/value"
                        );
                        assert_eq!(
                            CefString::from(&cmd.command_line_string())
                                .to_string()
                                .matches("--disable-features=")
                                .count(),
                            1
                        );
                    }
                }
            }
        }
    }

    #[cfg(all(target_os = "linux", feature = "portable"))]
    #[test]
    fn existing_vulkan_feature_parameters_survive() {
        use cef::*;
        let _ = api_hash(sys::CEF_API_VERSION_LAST, 0);
        let mut cmd = command_line_create().unwrap();
        let key = CefString::from("enable-features");
        for value in [
            "Vulkan<Trial:param/value,Other",
            "Vulkan:param/value,Other",
            " *Vulkan<Trial,Other ",
        ] {
            cmd.append_switch_with_value(Some(&key), Some(&CefString::from(value)));
            for _ in 0..2 {
                super::configure_linux(&mut cmd, Some("x11"), None, true);
                assert_eq!(
                    CefString::from(&cmd.switch_value(Some(&key))).to_string(),
                    value
                );
            }
        }
    }

    #[test]
    fn contexts_receive_readonly_policy_data() {
        assert_eq!(
            super::script(true),
            "Object.defineProperty(globalThis, '__shallotRequireHardwareGpu', { value: true });"
        );
        assert_eq!(
            super::script(false),
            "Object.defineProperty(globalThis, '__shallotRequireHardwareGpu', { value: false });"
        );
    }

    #[cfg(feature = "portable")]
    #[test]
    fn portable_helper_app_installs_context_handler() {
        use cef::ImplApp;
        assert!(super::cef_context::PolicyApp::new()
            .render_process_handler()
            .is_some());
    }

    #[test]
    fn software_requires_exact_launch_opt_in() {
        for value in [None, Some(""), Some("0"), Some("true")] {
            assert!(super::parse_require_hardware(value));
        }
        assert!(!super::parse_require_hardware(Some("1")));
    }
}
