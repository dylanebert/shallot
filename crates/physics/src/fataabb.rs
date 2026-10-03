//! One allocator-owned fat-AABB column per World.
use crate::regions::{self, Columns, MAX_WORLDS};
pub const AABB_STRIDE: usize = 6;
static mut COLUMNS: [Columns<1>; MAX_WORLDS] = [Columns::EMPTY; MAX_WORLDS];
static mut CAPS: [usize; MAX_WORLDS] = [0; MAX_WORLDS];
pub fn col_slice() -> &'static [f32] {
    unsafe {
        core::slice::from_raw_parts(
            COLUMNS[regions::active()].layout[0] as *const f32,
            fat_aabb_cap() * AABB_STRIDE,
        )
    }
}
pub fn col() -> crate::col::Col<'static, f32> {
    unsafe {
        crate::col::Col::new(
            COLUMNS[regions::active()].layout[0] as *mut f32,
            fat_aabb_cap() * AABB_STRIDE,
        )
    }
}
#[export_name = "fatAabbLayoutPtr"]
pub extern "C" fn fat_aabb_layout_ptr() -> *const u32 {
    unsafe { COLUMNS[regions::active()].layout.as_ptr() }
}
#[export_name = "fatAabbCap"]
pub extern "C" fn fat_aabb_cap() -> usize {
    unsafe { CAPS[regions::active()] }
}
#[export_name = "reserveFatAabb"]
pub extern "C" fn reserve_fat_aabb(cap: usize) -> u32 {
    unsafe {
        let id = regions::active();
        if cap <= CAPS[id] {
            return 0;
        }
        COLUMNS[id].reserve(0, cap * AABB_STRIDE * 4);
        CAPS[id] = cap;
        1
    }
}
pub unsafe fn reset(id: usize) {
    COLUMNS[id].release();
    CAPS[id] = 0;
}
pub unsafe fn snapshot(id: usize, out: &mut Vec<u8>) {
    regions::write_word(out, CAPS[id]);
    COLUMNS[id].snapshot(out);
}
pub unsafe fn restore(id: usize, input: &mut &[u8]) {
    CAPS[id] = regions::read_word(input);
    COLUMNS[id].restore(input);
}
