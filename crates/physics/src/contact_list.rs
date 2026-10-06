//! The awake contact traversal, including its swap-remove index.
use crate::{manifold_abi::*, manifolds, regions};
static mut LISTS: [Vec<u32>; regions::MAX_WORLDS] = [const { Vec::new() }; regions::MAX_WORLDS];
#[export_name = "awakeContactCount"]
pub unsafe extern "C" fn count() -> usize {
    LISTS[regions::active()].len()
}
#[export_name = "awakeContactGet"]
pub unsafe extern "C" fn get(index: usize) -> u32 {
    LISTS[regions::active()][index]
}
#[export_name = "awakeContactCopy"]
pub unsafe extern "C" fn copy(ptr: *mut u32) {
    let list = &LISTS[regions::active()];
    core::ptr::copy_nonoverlapping(list.as_ptr(), ptr, list.len());
}
#[export_name = "awakeContactRemove"]
pub unsafe extern "C" fn remove(id: usize) {
    let d = manifolds::dir_col();
    let o = id * DIR_STRIDE;
    let index = d.get(o + DIR_COLLIDE_INDEX);
    if index == u32::MAX {
        return;
    }
    let list = &mut LISTS[regions::active()];
    list.swap_remove(index as usize);
    if let Some(&moved) = list.get(index as usize) {
        d.set(moved as usize * DIR_STRIDE + DIR_COLLIDE_INDEX, index);
    }
    d.set(o + DIR_COLLIDE_INDEX, u32::MAX);
}
#[export_name = "awakeContactUpdate"]
pub unsafe extern "C" fn update(id: usize) {
    let d = manifolds::dir_col();
    let o = id * DIR_STRIDE;
    if d.get(o + DIR_SET_INDEX) != 2 {
        remove(id);
        return;
    }
    if d.get(o + DIR_COLLIDE_INDEX) != u32::MAX {
        return;
    }
    let list = &mut LISTS[regions::active()];
    d.set(o + DIR_COLLIDE_INDEX, list.len() as u32);
    list.push(id as u32);
}
pub unsafe fn reset(world: usize) {
    LISTS[world] = Vec::new();
}
pub unsafe fn snapshot(world: usize, out: &mut Vec<u8>) {
    regions::write_word(out, LISTS[world].len());
    for &id in &LISTS[world] {
        regions::write_word(out, id as usize);
    }
}
pub unsafe fn restore(world: usize, input: &mut &[u8]) {
    reset(world);
    let count = regions::read_word(input);
    LISTS[world].reserve(count);
    for _ in 0..count {
        LISTS[world].push(regions::read_word(input) as u32);
    }
}
