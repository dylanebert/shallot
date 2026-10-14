use cef::*;
use std::ffi::CString;
use std::sync::OnceLock;

// cefclient's version-resolved IDC allowlist, narrowed to local app operations.
// This is shell UX policy, not a restriction on the page's web capabilities.
struct Commands {
    local: Vec<i32>,
    navigation: Vec<i32>,
    spelling: i32,
    languages: i32,
}

fn resolve(name: &str) -> i32 {
    let name = CString::new(name).unwrap();
    unsafe { sys::cef_id_for_command_id_name(name.as_ptr()) }
}

fn resolve_all(names: &[&str]) -> Vec<i32> {
    names
        .iter()
        .map(|name| resolve(name))
        .filter(|id| *id >= 0)
        .collect()
}

fn commands() -> &'static Commands {
    static COMMANDS: OnceLock<Commands> = OnceLock::new();
    COMMANDS.get_or_init(|| {
        let mut local = resolve_all(&[
            "IDC_EDIT_MENU",
            "IDC_CUT",
            "IDC_COPY",
            "IDC_PASTE",
            "IDC_CONTENT_CONTEXT_CUT",
            "IDC_CONTENT_CONTEXT_COPY",
            "IDC_CONTENT_CONTEXT_PASTE",
            "IDC_CONTENT_CONTEXT_PASTE_AND_MATCH_STYLE",
            "IDC_CONTENT_CONTEXT_DELETE",
            "IDC_CONTENT_CONTEXT_SELECTALL",
            "IDC_CONTENT_CONTEXT_UNDO",
            "IDC_CONTENT_CONTEXT_REDO",
            "IDC_CONTENT_CONTEXT_COPYLINKLOCATION",
            "IDC_CONTENT_CONTEXT_COPYLINKTEXT",
            "IDC_CONTENT_CONTEXT_COPYIMAGE",
            "IDC_CONTENT_CONTEXT_COPYIMAGELOCATION",
            "IDC_CONTENT_CONTEXT_COPYAVLOCATION",
            "IDC_CONTENT_CONTEXT_COPYVIDEOFRAME",
            "IDC_CONTENT_CONTEXT_EMOJI",
            "IDC_SPELLCHECK_MENU",
            "IDC_CHECK_SPELLING_WHILE_TYPING",
            "IDC_SPELLCHECK_MULTI_LINGUAL",
            "IDC_WRITING_DIRECTION_MENU",
            "IDC_WRITING_DIRECTION_DEFAULT",
            "IDC_WRITING_DIRECTION_LTR",
            "IDC_WRITING_DIRECTION_RTL",
            "IDC_SPELLCHECK_ADD_TO_DICTIONARY",
            "IDC_SPELLCHECK_REMOVE_FROM_DICTIONARY",
            "IDC_CONTENT_CONTEXT_SPELLING_TOGGLE",
            "IDC_CONTENT_CONTEXT_NO_SPELLING_SUGGESTIONS",
            "IDC_FIND",
            "IDC_FIND_NEXT",
            "IDC_FIND_PREVIOUS",
            "IDC_CLOSE_FIND_OR_STOP",
            "IDC_ZOOM_MENU",
            "IDC_ZOOM_PLUS",
            "IDC_ZOOM_NORMAL",
            "IDC_ZOOM_MINUS",
            "IDC_FULLSCREEN",
            "IDC_CONTENT_CONTEXT_EXIT_FULLSCREEN",
            "IDC_CLOSE_TAB",
            "IDC_CLOSE_WINDOW",
            "IDC_EXIT",
        ]);
        if cfg!(debug_assertions) {
            local.extend(resolve_all(&[
                "IDC_MORE_TOOLS_MENU",
                "IDC_DEV_TOOLS",
                "IDC_DEV_TOOLS_CONSOLE",
                "IDC_DEV_TOOLS_TOGGLE",
                "IDC_DEV_TOOLS_INSPECT",
                "IDC_CONTENT_CONTEXT_INSPECTELEMENT",
                "IDC_CONTENT_CONTEXT_INSPECTELEMENT_WITH_DEVTOOLS",
            ]));
        }
        Commands {
            local,
            navigation: if cfg!(debug_assertions) {
                resolve_all(&[
                    "IDC_RELOAD",
                    "IDC_RELOAD_BYPASSING_CACHE",
                    "IDC_RELOAD_CLEARING_CACHE",
                    "IDC_CONTENT_CONTEXT_RELOADFRAME",
                ])
            } else {
                Vec::new()
            },
            spelling: resolve("IDC_SPELLCHECK_SUGGESTION_0"),
            languages: resolve("IDC_SPELLCHECK_LANGUAGES_FIRST"),
        }
    })
}

fn is_local(id: i32) -> bool {
    let commands = commands();
    id >= 0
        && (commands.local.contains(&id)
        // Chromium's SpellingMenuObserver reserves five suggestions from this IDC.
        || (commands.spelling >= 0 && (0..5).contains(&(id - commands.spelling)))
        // SpellingOptionsSubMenuObserver reserves 100 dictionary language slots.
        || (commands.languages >= 0 && (0..100).contains(&(id - commands.languages))))
}

fn allowed(id: i32) -> bool {
    is_local(id) || (id >= 0 && commands().navigation.contains(&id))
}

fn allowed_target(id: i32, disposition: WindowOpenDisposition) -> bool {
    // Reload uses disposition to duplicate the tab. Close/exit and local commands
    // do not navigate, and window-system commands can arrive with UNKNOWN.
    allowed(id)
        && (disposition == WindowOpenDisposition::CURRENT_TAB
            || (disposition == WindowOpenDisposition::UNKNOWN && is_local(id)))
}

fn filter_menu(model: &mut MenuModel) {
    for index in (0..model.count()).rev() {
        match model.type_at(index) {
            MenuItemType::SUBMENU => {
                if let Some(mut submenu) = model.sub_menu_at(index) {
                    filter_menu(&mut submenu);
                    if submenu.count() > 0 {
                        continue;
                    }
                }
                model.remove_at(index);
            }
            MenuItemType::SEPARATOR => {}
            _ => {
                if !allowed(model.command_id_at(index)) {
                    model.remove_at(index);
                }
            }
        }
    }
    // Prune after removals: a removed first item can expose a leading separator.
    let mut index = 0;
    while index < model.count() {
        if model.type_at(index) == MenuItemType::SEPARATOR
            && (index == 0
                || index == model.count() - 1
                || model.type_at(index + 1) == MenuItemType::SEPARATOR)
        {
            model.remove_at(index);
        } else {
            index += 1;
        }
    }
}

wrap_command_handler! {
    pub(super) struct AppCommandHandler;
    impl CommandHandler {
        fn on_chrome_command(&self, _browser: Option<&mut Browser>, command_id: i32, disposition: WindowOpenDisposition) -> i32 {
            i32::from(!allowed_target(command_id, disposition))
        }
        fn is_chrome_app_menu_item_visible(&self, _browser: Option<&mut Browser>, command_id: i32) -> i32 {
            i32::from(allowed(command_id))
        }
        fn is_chrome_app_menu_item_enabled(&self, _browser: Option<&mut Browser>, command_id: i32) -> i32 {
            i32::from(allowed(command_id))
        }
        fn is_chrome_page_action_icon_visible(&self, icon_type: ChromePageActionIconType) -> i32 {
            i32::from(icon_type == ChromePageActionIconType::FIND || icon_type == ChromePageActionIconType::ZOOM)
        }
        fn is_chrome_toolbar_button_visible(&self, _button_type: ChromeToolbarButtonType) -> i32 { 0 }
    }
}

wrap_context_menu_handler! {
    pub(super) struct AppContextMenuHandler;
    impl ContextMenuHandler {
        fn on_before_context_menu(&self, _browser: Option<&mut Browser>, _frame: Option<&mut Frame>, _params: Option<&mut ContextMenuParams>, model: Option<&mut MenuModel>) {
            if let Some(model) = model { filter_menu(model); }
        }
        fn on_context_menu_command(&self, _browser: Option<&mut Browser>, _frame: Option<&mut Frame>, _params: Option<&mut ContextMenuParams>, command_id: i32, _event_flags: EventFlags) -> i32 {
            i32::from(!allowed(command_id))
        }
    }
}

wrap_request_handler! {
    pub(super) struct AppRequestHandler;
    impl RequestHandler {
        fn on_open_urlfrom_tab(&self, _browser: Option<&mut Browser>, _frame: Option<&mut Frame>, _target_url: Option<&CefString>, target_disposition: WindowOpenDisposition, _user_gesture: i32) -> i32 {
            i32::from(target_disposition != WindowOpenDisposition::CURRENT_TAB)
        }
    }
}

#[cfg(test)]
mod tests;
