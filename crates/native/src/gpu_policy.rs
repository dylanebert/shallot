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
