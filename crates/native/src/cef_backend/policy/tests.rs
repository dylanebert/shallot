use super::*;
use crate::cef_backend::{ShallotClient, ShallotLifeSpanHandler};
use cef::rc::ConvertReturnValue;
use std::cell::RefCell;

fn id(name: &str) -> i32 {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let _ = api_hash(sys::CEF_API_VERSION_LAST, 0);
        // CEF populates its IDC map on first use on the UI thread. Serialize
        // that first call when portable tests run without a browser UI thread.
        assert!(resolve("IDC_BACK") >= 0);
    });
    let id = resolve(name);
    assert!(id >= 0, "fixture IDC not present in loaded CEF: {name}");
    id
}

#[test]
fn app_client_wires_retained_commands_and_explicit_menu_predicates() {
    let client = ShallotClient::new();
    let handler = client.command_handler().expect("app command handler");
    let menu = client
        .context_menu_handler()
        .expect("app context menu handler");
    for name in [
        "IDC_CUT",
        "IDC_COPY",
        "IDC_PASTE",
        "IDC_FIND",
        "IDC_FIND_NEXT",
        "IDC_FIND_PREVIOUS",
        "IDC_ZOOM_PLUS",
        "IDC_ZOOM_NORMAL",
        "IDC_ZOOM_MINUS",
        "IDC_FULLSCREEN",
        "IDC_CLOSE_TAB",
        "IDC_CLOSE_WINDOW",
        "IDC_EXIT",
        "IDC_CONTENT_CONTEXT_UNDO",
        "IDC_CONTENT_CONTEXT_REDO",
        "IDC_CONTENT_CONTEXT_CUT",
        "IDC_CONTENT_CONTEXT_COPY",
        "IDC_CONTENT_CONTEXT_PASTE",
        "IDC_CONTENT_CONTEXT_PASTE_AND_MATCH_STYLE",
        "IDC_CONTENT_CONTEXT_DELETE",
        "IDC_CONTENT_CONTEXT_SELECTALL",
        "IDC_CONTENT_CONTEXT_COPYLINKLOCATION",
        "IDC_CONTENT_CONTEXT_COPYIMAGE",
        "IDC_SPELLCHECK_ADD_TO_DICTIONARY",
        "IDC_SPELLCHECK_MULTI_LINGUAL",
        "IDC_WRITING_DIRECTION_DEFAULT",
        "IDC_WRITING_DIRECTION_LTR",
        "IDC_WRITING_DIRECTION_RTL",
    ] {
        let id = id(name);
        assert_eq!(
            handler.on_chrome_command(None, id, WindowOpenDisposition::CURRENT_TAB),
            0,
            "{name}"
        );
        assert_eq!(
            handler.on_chrome_command(None, id, WindowOpenDisposition::UNKNOWN),
            0,
            "{name} is not navigation"
        );
        assert_eq!(
            handler.is_chrome_app_menu_item_visible(None, id),
            1,
            "{name}"
        );
        assert_eq!(
            handler.is_chrome_app_menu_item_enabled(None, id),
            1,
            "{name}"
        );
        assert_eq!(
            menu.on_context_menu_command(None, None, None, id, EventFlags::default()),
            0,
            "{name}"
        );
        for disposition in [
            WindowOpenDisposition::NEW_BACKGROUND_TAB,
            WindowOpenDisposition::NEW_FOREGROUND_TAB,
            WindowOpenDisposition::NEW_WINDOW,
            WindowOpenDisposition::OFF_THE_RECORD,
        ] {
            assert_eq!(
                handler.on_chrome_command(None, id, disposition),
                1,
                "{name}"
            );
        }
    }
    assert_eq!(
        handler.is_chrome_page_action_icon_visible(ChromePageActionIconType::FIND),
        1
    );
    assert_eq!(
        handler.is_chrome_page_action_icon_visible(ChromePageActionIconType::ZOOM),
        1
    );
    assert_eq!(
        handler.is_chrome_page_action_icon_visible(ChromePageActionIconType::default()),
        0
    );
    assert_eq!(
        handler.is_chrome_toolbar_button_visible(ChromeToolbarButtonType::default()),
        0
    );
}

#[test]
fn browser_commands_and_invalid_ids_never_reach_default_execution() {
    let client = ShallotClient::new();
    let handler = client.command_handler().unwrap();
    let menu = client.context_menu_handler().unwrap();
    let _ = id("IDC_BACK");
    let mut rejected = vec![-1, -2, i32::MAX, resolve("IDC_SHALLOT_NONEXISTENT")];
    assert_eq!(rejected[3], -1);
    assert!(resolve_all(&["IDC_SHALLOT_NONEXISTENT"]).is_empty());
    rejected.extend(
        [
            "IDC_NEW_TAB",
            "IDC_NEW_WINDOW",
            "IDC_NEW_INCOGNITO_WINDOW",
            "IDC_RESTORE_TAB",
            "IDC_BACK",
            "IDC_FORWARD",
            "IDC_HOME",
            "IDC_FOCUS_LOCATION",
            "IDC_OPTIONS",
            "IDC_PRINT",
            "IDC_SAVE_PAGE",
            "IDC_VIEW_SOURCE",
            "IDC_PASTE_AND_GO",
            "IDC_CONTENT_CONTEXT_OPENLINKNEWTAB",
            "IDC_CONTENT_CONTEXT_OPENLINKNEWWINDOW",
            "IDC_CONTENT_CONTEXT_SEARCHWEBFOR",
            "IDC_CONTENT_CONTEXT_LANGUAGE_SETTINGS",
            "IDC_CONTENT_CONTEXT_SAVELINKAS",
            "IDC_CONTENT_CONTEXT_SAVEIMAGEAS",
        ]
        .map(id),
    );
    for id in rejected {
        for disposition in [
            WindowOpenDisposition::CURRENT_TAB,
            WindowOpenDisposition::UNKNOWN,
            WindowOpenDisposition::NEW_WINDOW,
        ] {
            assert_eq!(handler.on_chrome_command(None, id, disposition), 1, "{id}");
        }
        assert_eq!(handler.is_chrome_app_menu_item_visible(None, id), 0, "{id}");
        assert_eq!(handler.is_chrome_app_menu_item_enabled(None, id), 0, "{id}");
        assert_eq!(
            menu.on_context_menu_command(None, None, None, id, EventFlags::default()),
            1,
            "{id}"
        );
    }
}

#[test]
fn reload_and_tools_are_debug_only_and_reload_cannot_duplicate_the_app() {
    let client = ShallotClient::new();
    let handler = client.command_handler().unwrap();
    let menu = client.context_menu_handler().unwrap();
    for name in [
        "IDC_RELOAD",
        "IDC_RELOAD_BYPASSING_CACHE",
        "IDC_RELOAD_CLEARING_CACHE",
        "IDC_CONTENT_CONTEXT_RELOADFRAME",
        "IDC_DEV_TOOLS",
        "IDC_DEV_TOOLS_CONSOLE",
        "IDC_DEV_TOOLS_TOGGLE",
        "IDC_DEV_TOOLS_INSPECT",
        "IDC_CONTENT_CONTEXT_INSPECTELEMENT",
    ] {
        let id = id(name);
        let blocked = i32::from(!cfg!(debug_assertions));
        assert_eq!(
            handler.on_chrome_command(None, id, WindowOpenDisposition::CURRENT_TAB),
            blocked,
            "{name}"
        );
        assert_eq!(
            handler.is_chrome_app_menu_item_visible(None, id),
            1 - blocked,
            "{name}"
        );
        assert_eq!(
            handler.is_chrome_app_menu_item_enabled(None, id),
            1 - blocked,
            "{name}"
        );
        assert_eq!(
            menu.on_context_menu_command(None, None, None, id, EventFlags::default()),
            blocked,
            "{name}"
        );
        assert_eq!(
            handler.on_chrome_command(None, id, WindowOpenDisposition::NEW_BACKGROUND_TAB),
            1,
            "{name}"
        );
        assert_eq!(
            handler.on_chrome_command(None, id, WindowOpenDisposition::NEW_WINDOW),
            1,
            "{name}"
        );
    }
    for name in [
        "IDC_RELOAD",
        "IDC_RELOAD_BYPASSING_CACHE",
        "IDC_RELOAD_CLEARING_CACHE",
    ] {
        assert_eq!(
            handler.on_chrome_command(None, id(name), WindowOpenDisposition::UNKNOWN),
            1,
            "{name}"
        );
    }
}

#[test]
fn app_client_rejects_popups_and_alternate_navigation_without_replacing_the_page() {
    let client = ShallotClient::new();
    let life = client.life_span_handler().expect("app lifetime handler");
    let request = client.request_handler().expect("app request handler");
    for disposition in [
        WindowOpenDisposition::CURRENT_TAB,
        WindowOpenDisposition::UNKNOWN,
        WindowOpenDisposition::NEW_BACKGROUND_TAB,
        WindowOpenDisposition::NEW_FOREGROUND_TAB,
        WindowOpenDisposition::NEW_WINDOW,
        WindowOpenDisposition::NEW_POPUP,
        WindowOpenDisposition::OFF_THE_RECORD,
        WindowOpenDisposition::SAVE_TO_DISK,
    ] {
        for gesture in [0, 1] {
            assert_eq!(
                life.on_before_popup(
                    None,
                    None,
                    7,
                    None,
                    None,
                    disposition,
                    gesture,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None
                ),
                1
            );
            assert_eq!(
                request.on_open_urlfrom_tab(None, None, None, disposition, gesture),
                i32::from(disposition != WindowOpenDisposition::CURRENT_TAB)
            );
        }
    }
    assert_eq!(request.on_before_browse(None, None, None, 1, 0), 0);
    assert_eq!(request.on_before_browse(None, None, None, 0, 1), 0);
}

#[test]
fn devtools_gets_its_own_unfiltered_client_and_non_quitting_lifetime() {
    let app = ShallotClient::new();
    let mut client = Some(app);
    let mut default_window = 0;
    ShallotLifeSpanHandler::new().on_before_dev_tools_popup(
        None,
        None,
        Some(&mut client),
        None,
        None,
        Some(&mut default_window),
    );
    let client = client.unwrap();
    assert_eq!(default_window, 1);
    assert!(client.command_handler().is_none());
    assert!(client.context_menu_handler().is_none());
    assert!(client.request_handler().is_none());
    assert!(client.keyboard_handler().is_none());
    let life = client.life_span_handler().expect("separate tool lifetime");
    assert_eq!(life.do_close(None), 0);
    // Calling this actual callback must not require a running CEF message loop.
    life.on_before_close(None);
}

// Independent menu fixtures exercise the real MenuModel C interface. The boxes
// outlive their wrappers; CEF's reference operations are no-ops in this fixture.
#[repr(C)]
struct Fixture {
    native: sys::cef_menu_model_t,
    items: RefCell<Vec<Item>>,
}
enum Item {
    Command(i32),
    Separator,
    Submenu(Box<Fixture>),
}
impl Fixture {
    fn new(items: Vec<Item>) -> Box<Self> {
        let mut native: sys::cef_menu_model_t = unsafe { std::mem::zeroed() };
        native.get_count = Some(count);
        native.get_type_at = Some(kind);
        native.get_command_id_at = Some(command);
        native.get_sub_menu_at = Some(submenu);
        native.remove_at = Some(remove);
        Box::new(Self {
            native,
            items: RefCell::new(items),
        })
    }
    fn model(&mut self) -> MenuModel {
        (&mut self.native as *mut sys::cef_menu_model_t).wrap_result()
    }
    fn snapshot(&self) -> Vec<String> {
        self.items
            .borrow()
            .iter()
            .map(|item| match item {
                Item::Command(id) => id.to_string(),
                Item::Separator => "|".into(),
                Item::Submenu(menu) => format!("{:?}", menu.snapshot()),
            })
            .collect()
    }
}
unsafe extern "C" fn count(model: *mut sys::cef_menu_model_t) -> usize {
    (*(model as *mut Fixture)).items.borrow().len()
}
unsafe extern "C" fn kind(
    model: *mut sys::cef_menu_model_t,
    index: usize,
) -> sys::cef_menu_item_type_t {
    match (*(model as *mut Fixture)).items.borrow()[index] {
        Item::Command(_) => sys::cef_menu_item_type_t::MENUITEMTYPE_COMMAND,
        Item::Separator => sys::cef_menu_item_type_t::MENUITEMTYPE_SEPARATOR,
        Item::Submenu(_) => sys::cef_menu_item_type_t::MENUITEMTYPE_SUBMENU,
    }
}
unsafe extern "C" fn command(model: *mut sys::cef_menu_model_t, index: usize) -> i32 {
    match (*(model as *mut Fixture)).items.borrow()[index] {
        Item::Command(id) => id,
        _ => -1,
    }
}
unsafe extern "C" fn submenu(
    model: *mut sys::cef_menu_model_t,
    index: usize,
) -> *mut sys::cef_menu_model_t {
    match &mut (*(model as *mut Fixture)).items.borrow_mut()[index] {
        Item::Submenu(menu) => &mut menu.native,
        _ => std::ptr::null_mut(),
    }
}
unsafe extern "C" fn remove(model: *mut sys::cef_menu_model_t, index: usize) -> i32 {
    (*(model as *mut Fixture)).items.borrow_mut().remove(index);
    1
}

#[test]
fn context_callback_prunes_mixed_recursive_menus_and_exposed_separators() {
    use Item::*;
    let copy = id("IDC_CONTENT_CONTEXT_COPY");
    let paste = id("IDC_CONTENT_CONTEXT_PASTE");
    let search = id("IDC_CONTENT_CONTEXT_SEARCHWEBFOR");
    let reload = id("IDC_RELOAD");
    let mut fixture = Fixture::new(vec![
        Command(search),
        Separator,
        Separator,
        Submenu(Fixture::new(vec![Command(search), Separator])),
        Submenu(Fixture::new(vec![
            Separator,
            Command(copy),
            Separator,
            Command(search),
            Separator,
            Submenu(Fixture::new(vec![Separator, Command(paste), Separator])),
            Separator,
        ])),
        Separator,
        Separator,
        Command(reload),
        Separator,
        Command(-1),
    ]);
    let mut model = fixture.model();
    ShallotClient::new()
        .context_menu_handler()
        .unwrap()
        .on_before_context_menu(None, None, None, Some(&mut model));
    let nested = format!(
        "{:?}",
        vec![
            copy.to_string(),
            "|".into(),
            format!("{:?}", vec![paste.to_string()])
        ]
    );
    let expected = if cfg!(debug_assertions) {
        vec![nested, "|".into(), reload.to_string()]
    } else {
        vec![nested]
    };
    assert_eq!(fixture.snapshot(), expected);
}

#[test]
fn empty_canvas_has_no_release_menu_and_local_spelling_is_retained() {
    use Item::*;
    let handler = ShallotClient::new().context_menu_handler().unwrap();
    let mut canvas = Fixture::new(vec![
        Separator,
        Command(id("IDC_BACK")),
        Separator,
        Command(id("IDC_RELOAD")),
        Separator,
        Command(id("IDC_VIEW_SOURCE")),
        Command(id("IDC_CONTENT_CONTEXT_INSPECTELEMENT")),
        Separator,
    ]);
    handler.on_before_context_menu(None, None, None, Some(&mut canvas.model()));
    let expected = if cfg!(debug_assertions) {
        vec![
            id("IDC_RELOAD").to_string(),
            "|".into(),
            id("IDC_CONTENT_CONTEXT_INSPECTELEMENT").to_string(),
        ]
    } else {
        vec![]
    };
    assert_eq!(canvas.snapshot(), expected);
    let suggestion = id("IDC_SPELLCHECK_SUGGESTION_0");
    let copy = id("IDC_CONTENT_CONTEXT_COPYLINKLOCATION");
    let language = id("IDC_SPELLCHECK_LANGUAGES_FIRST");
    let mut edit = Fixture::new(vec![
        Command(suggestion),
        Command(suggestion + 4),
        Command(suggestion + 5),
        Command(language),
        Command(language + 99),
        Command(language + 100),
        Separator,
        Command(copy),
        Command(id("IDC_CONTENT_CONTEXT_SEARCHWEBFOR")),
    ]);
    handler.on_before_context_menu(None, None, None, Some(&mut edit.model()));
    assert_eq!(
        edit.snapshot(),
        vec![
            suggestion.to_string(),
            (suggestion + 4).to_string(),
            language.to_string(),
            (language + 99).to_string(),
            "|".into(),
            copy.to_string()
        ]
    );
}
