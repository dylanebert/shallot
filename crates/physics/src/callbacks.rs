//! User collision callbacks run only on the orchestrator, after callback-dependent task work joins.
use crate::regions::MAX_WORLDS;

static mut FILTER: [bool; MAX_WORLDS] = [false; MAX_WORLDS];

#[export_name = "worldSetCustomFilterCallback"]
pub unsafe extern "C" fn set_filter(world: usize, enabled: bool) {
    FILTER[world] = enabled;
}

pub unsafe fn filter_enabled(world: usize) -> bool {
    FILTER[world]
}

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "env")]
extern "C" {
    fn collisionCallback(kind: u32, a: usize, b: usize, data: *const f32) -> bool;
}

pub unsafe fn filter(world: usize, a: usize, b: usize) -> bool {
    if !FILTER[world] {
        return true;
    }
    let shapes = crate::shapes::col_slice(world);
    if (shapes[a * crate::shapes::SHAPE_STRIDE + crate::shapes::S_FLAGS]
        | shapes[b * crate::shapes::SHAPE_STRIDE + crate::shapes::S_FLAGS])
        & (4 << 16)
        == 0
    {
        return true;
    }
    #[cfg(target_arch = "wasm32")]
    return collisionCallback(0, a, b, core::ptr::null());
    #[cfg(not(target_arch = "wasm32"))]
    true
}
