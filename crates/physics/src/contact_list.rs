//! Box3D's collide traversal, gathered from graph colors and the awake solver set.
use crate::{constraint_graph as graph, solver_set};
#[export_name = "awakeContactCount"]
pub unsafe extern "C" fn count() -> usize {
    count_in_world(crate::regions::active())
}
pub unsafe fn count_in_world(world: usize) -> usize {
    (0..graph::COLORS)
        .map(|color| {
            graph::count_in_world(world, color, false) + graph::count_in_world(world, color, true)
        })
        .sum::<usize>()
        + solver_set::array_count_in_world(world, 2, 0)
}
#[export_name = "awakeContactGet"]
pub unsafe extern "C" fn get(index: usize) -> u32 {
    get_in_world(crate::regions::active(), index)
}
pub unsafe fn get_in_world(world: usize, mut index: usize) -> u32 {
    for color in 0..graph::COLORS {
        for scalar in [false, true] {
            let count = graph::count_in_world(world, color, scalar);
            if index < count {
                let ptr = graph::pointer_in_world(world, color, scalar) as *const u32;
                return if scalar {
                    (*(ptr.cast::<crate::contact_spans::ContactSpec>().add(index))).contact_id
                        as u32
                } else {
                    *ptr.add(index)
                };
            }
            index -= count;
        }
    }
    solver_set::array_get_in_world(world, 2, 0, index) as u32
}
#[export_name = "awakeContactCopy"]
pub unsafe extern "C" fn copy(ptr: *mut u32) {
    copy_in_world(crate::regions::active(), ptr)
}
pub unsafe fn copy_in_world(world: usize, mut ptr: *mut u32) {
    for color in 0..graph::COLORS {
        let count = graph::count_in_world(world, color, false);
        core::ptr::copy_nonoverlapping(
            graph::pointer_in_world(world, color, false) as *const u32,
            ptr,
            count,
        );
        ptr = ptr.add(count);
        let count = graph::count_in_world(world, color, true);
        let records =
            graph::pointer_in_world(world, color, true) as *const crate::contact_spans::ContactSpec;
        for index in 0..count {
            *ptr = (*records.add(index)).contact_id as u32;
            ptr = ptr.add(1);
        }
    }
    for index in 0..solver_set::array_count_in_world(world, 2, 0) {
        *ptr = solver_set::array_get_in_world(world, 2, 0, index) as u32;
        ptr = ptr.add(1);
    }
}
