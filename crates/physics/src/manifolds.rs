//! Each World's contact directory and retained manifold pool own their allocations.
use crate::col::Col;
use crate::manifold_abi::{DIR_STRIDE, MANIFOLD_STRIDE};
use crate::regions::{self, Columns, MAX_WORLDS};
static mut COLUMNS: [Columns<2>; MAX_WORLDS] = [Columns::EMPTY; MAX_WORLDS];
static mut CAPS: [[usize; 2]; MAX_WORLDS] = [[0; 2]; MAX_WORLDS];
pub fn dir_col() -> Col<'static, u32> {
    unsafe {
        let id = regions::active();
        Col::new(COLUMNS[id].layout[0] as *mut u32, CAPS[id][0] * DIR_STRIDE)
    }
}
pub fn pool_col() -> Col<'static, f32> {
    unsafe {
        let id = regions::active();
        Col::new(
            COLUMNS[id].layout[1] as *mut f32,
            CAPS[id][1] * MANIFOLD_STRIDE,
        )
    }
}
#[export_name = "manifoldLayoutPtr"]
pub extern "C" fn manifold_layout_ptr() -> *const u32 {
    unsafe { COLUMNS[regions::active()].layout.as_ptr() }
}
#[export_name = "reserveManifolds"]
pub extern "C" fn reserve_manifolds(contact_cap: usize, manifold_cap: usize) {
    unsafe {
        let id = regions::active();
        for (column, cap, stride) in [
            (0, contact_cap, DIR_STRIDE),
            (1, manifold_cap, MANIFOLD_STRIDE),
        ] {
            let cap = cap.max(CAPS[id][column]);
            COLUMNS[id].reserve(column, cap * stride * 4);
            CAPS[id][column] = cap;
        }
    }
}
pub unsafe fn reset(id: usize) {
    COLUMNS[id].release();
    CAPS[id] = [0; 2];
}
pub unsafe fn restore_id(from: usize, to: usize) {
    reset(to);
    COLUMNS[to] = COLUMNS[from];
    CAPS[to] = CAPS[from];
    COLUMNS[from] = Columns::EMPTY;
    CAPS[from] = [0; 2];
}
