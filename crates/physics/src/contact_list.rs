//! The awake contact traversal, including its swap-remove index.
use crate::{manifold_abi::*, manifolds, regions};
static mut LISTS: [Vec<u32>; regions::MAX_WORLDS] = [const { Vec::new() }; regions::MAX_WORLDS];
#[export_name = "awakeContactCount"]
pub unsafe extern "C" fn count() -> usize {
    unsafe { count_in_world(crate::regions::active()) }
}

pub unsafe extern "C" fn count_in_world(world_index: usize) -> usize {
    LISTS[world_index].len()
}
#[export_name = "awakeContactGet"]
pub unsafe extern "C" fn get(index: usize) -> u32 {
    unsafe { get_in_world(crate::regions::active(), index) }
}

pub unsafe extern "C" fn get_in_world(world_index: usize, index: usize) -> u32 {
    LISTS[world_index][index]
}
#[export_name = "awakeContactCopy"]
pub unsafe extern "C" fn copy(ptr: *mut u32) {
    unsafe { copy_in_world(crate::regions::active(), ptr) }
}

pub unsafe extern "C" fn copy_in_world(world_index: usize, ptr: *mut u32) {
    let list = &LISTS[world_index];
    core::ptr::copy_nonoverlapping(list.as_ptr(), ptr, list.len());
}
#[export_name = "awakeContactRemove"]
pub unsafe extern "C" fn remove(id: usize) {
    unsafe { remove_in_world(crate::regions::active(), id) }
}

pub unsafe extern "C" fn remove_in_world(world_index: usize, id: usize) {
    let d = manifolds::dir_col(world_index);
    let o = id * DIR_STRIDE;
    let index = d.get(o + DIR_COLLIDE_INDEX);
    if index == u32::MAX {
        return;
    }
    let list = &mut LISTS[world_index];
    list.swap_remove(index as usize);
    if let Some(&moved) = list.get(index as usize) {
        d.set(moved as usize * DIR_STRIDE + DIR_COLLIDE_INDEX, index);
    }
    d.set(o + DIR_COLLIDE_INDEX, u32::MAX);
}
#[export_name = "awakeContactUpdate"]
pub unsafe extern "C" fn update(id: usize) {
    unsafe { update_in_world(crate::regions::active(), id) }
}

pub unsafe extern "C" fn update_in_world(world_index: usize, id: usize) {
    let d = manifolds::dir_col(world_index);
    let o = id * DIR_STRIDE;
    if d.get(o + DIR_SET_INDEX) != 2 {
        remove_in_world(world_index, id);
        return;
    }
    if d.get(o + DIR_COLLIDE_INDEX) != u32::MAX {
        return;
    }
    let list = &mut LISTS[world_index];
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
