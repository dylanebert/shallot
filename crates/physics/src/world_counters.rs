//! Box3D physics_world.c b3World_GetCounters. Memory and tasks measure this kernel's host.
use crate::{
    arena, bodies, broad, constraint_graph, island, joint_record, joints, manifolds, regions,
    shapes, solver_set,
};

static mut COUNTERS: [[i32; 50]; regions::MAX_WORLDS] = [[0; 50]; regions::MAX_WORLDS];
static mut TASK_COUNTS: [i32; regions::MAX_WORLDS] = [0; regions::MAX_WORLDS];

pub(crate) unsafe fn reset(world: usize) {
    TASK_COUNTS[world] = 0;
}
pub(crate) unsafe fn finish_tasks(world: usize) {
    TASK_COUNTS[world] = crate::solve::SCHEDULER.task_count() as i32;
}

fn tree_height(world: usize, kind: usize) -> i32 {
    let (root, _) = broad::tree_root(world, kind);
    if root == -1 {
        return 0;
    }
    unsafe {
        (*broad::tree_ptr(world, kind).add(root as usize * crate::tree::STRIDE + 11) >> 16) as i32
    }
}

#[export_name = "worldCounters"]
pub unsafe extern "C" fn read(world: usize) -> *const i32 {
    let out = &mut COUNTERS[world];
    out.fill(0);
    out[0] = bodies::body_count(world as u32) as i32;
    out[1] = shapes::shape_count(world as u32) as i32;
    out[2] = manifolds::contact_count(world) as i32;
    out[3] = joint_record::count_in_world(world) as i32;
    out[4] = island::count_in_world(world) as i32;
    out[7] = tree_height(world, 0);
    out[8] = tree_height(world, 1).max(tree_height(world, 2));
    out[11] = crate::allocation::live_bytes() as i32;
    out[12] = TASK_COUNTS[world];
    for color in 0..constraint_graph::COLORS {
        let contacts = constraint_graph::count_in_world(world, color, false)
            + constraint_graph::count_in_world(world, color, true);
        out[13 + color] = (contacts + joints::count_in_world(world, color)) as i32;
        out[45] += contacts as i32;
    }
    out[45] += solver_set::array_count_in_world(
        world,
        solver_set::AWAKE_SET as usize,
        solver_set::CONTACT_IDS,
    ) as i32;
    arena::diagnostics(world, out);
    out.as_ptr()
}
