use cef::application_mac::{CefAppProtocol, CrAppControlProtocol, CrAppProtocol};
use cef::quit_message_loop;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, Bool, NSObjectProtocol};
use objc2::{define_class, extern_methods, msg_send, ClassType, DefinedClass, MainThreadMarker};
use objc2_app_kit::{NSApp, NSApplication, NSEvent};
use std::cell::Cell;

#[derive(Default)]
struct Ivars {
    handling_send_event: Cell<Bool>,
}

define_class!(
    // CEF on macOS requires the shared NSApplication to conform to CefAppProtocol — it tracks whether
    // the app is inside `sendEvent:` so the framework can dispatch events correctly. Without this the
    // framework aborts on startup.
    #[unsafe(super(NSApplication))]
    #[ivars = Ivars]
    struct ShallotApplication;

    impl ShallotApplication {
        #[unsafe(method(sendEvent:))]
        unsafe fn send_event(&self, event: &NSEvent) {
            let was_handling = self.is_handling_send_event();
            self.set_handling_send_event(true);
            let _: () = msg_send![super(self), sendEvent: event];
            if !was_handling {
                self.set_handling_send_event(false);
            }
        }

        // Route Cmd+Q through CEF's orderly shutdown. The default `terminate:` calls exit() and never
        // returns to the run loop, but CEF needs to leave run_message_loop to shut down cleanly.
        #[unsafe(method(terminate:))]
        unsafe fn terminate(&self, _sender: Option<&AnyObject>) {
            quit_message_loop();
        }
    }

    unsafe impl CrAppControlProtocol for ShallotApplication {
        #[unsafe(method(setHandlingSendEvent:))]
        unsafe fn set_handling_send_event_proto(&self, handling: Bool) {
            self.ivars().handling_send_event.set(handling);
        }
    }

    unsafe impl CrAppProtocol for ShallotApplication {
        #[unsafe(method(isHandlingSendEvent))]
        unsafe fn is_handling_send_event_proto(&self) -> Bool {
            self.ivars().handling_send_event.get()
        }
    }

    unsafe impl CefAppProtocol for ShallotApplication {}
);

impl ShallotApplication {
    extern_methods!(
        #[unsafe(method(sharedApplication))]
        fn shared_application() -> Retained<Self>;

        #[unsafe(method(setHandlingSendEvent:))]
        fn set_handling_send_event(&self, handling: bool);

        #[unsafe(method(isHandlingSendEvent))]
        fn is_handling_send_event(&self) -> bool;
    );
}

/// Install the CEF-compatible NSApplication. The first `sharedApplication` call fixes the `NSApp`
/// class, so this must run before `cef::initialize` and before anything else touches `NSApp`.
pub fn setup_application() {
    let _ = ShallotApplication::shared_application();
    let mtm = MainThreadMarker::new().expect("not on the main thread");
    assert!(
        NSApp(mtm).isKindOfClass(ShallotApplication::class()),
        "NSApp is not ShallotApplication; something touched NSApp before setup"
    );
}
