//! solver_set.c's set array and id pool. Body columns retain the solver's column layout.
use crate::body::{SIM_STRIDE, STATE_STRIDE};
use crate::regions::{self, Columns, MAX_WORLDS};
const AWAKE: usize = 2;
const STRIDES: [usize; 6] = [STATE_STRIDE, SIM_STRIDE, 0, 0, 0, 0];
struct SolverSet {
    columns: Columns<6>,
    body_count: usize,
    indices: [Vec<i32>; 2],
    index: i32,
    joint_sims: crate::joints::JointArray,
}
impl SolverSet {
    fn empty() -> Self {
        Self {
            columns: Columns::EMPTY,
            body_count: 0,
            indices: [Vec::new(), Vec::new()],
            index: -1,
            joint_sims: crate::joints::JointArray::EMPTY,
        }
    }
}
struct Sets {
    sets: Vec<SolverSet>,
    free: Vec<usize>,
}
static mut WORLDS: [Sets; MAX_WORLDS] = [const {
    Sets {
        sets: Vec::new(),
        free: Vec::new(),
    }
}; MAX_WORLDS];
unsafe fn world() -> &'static mut Sets {
    &mut WORLDS[regions::active()]
}
unsafe fn set(id: usize) -> &'static mut SolverSet {
    &mut world().sets[id]
}
#[export_name = "solverSetCreate"]
pub unsafe extern "C" fn create() -> usize {
    let w = world();
    let id = if let Some(id) = w.free.pop() {
        id
    } else {
        w.sets.push(SolverSet::empty());
        w.sets.len() - 1
    };
    w.sets[id].index = id as i32;
    id
}
#[export_name = "solverSetCount"]
pub unsafe extern "C" fn count() -> usize {
    world().sets.len()
}
#[export_name = "solverSetIndex"]
pub unsafe extern "C" fn index(id: usize) -> i32 {
    set(id).index
}
#[export_name = "solverSetDestroy"]
pub unsafe extern "C" fn destroy(id: usize) {
    let s = set(id);
    s.columns.release();
    s.joint_sims.records.release();
    *s = SolverSet::empty();
    world().free.push(id);
}
#[export_name = "solverSetBodyCount"]
pub unsafe extern "C" fn body_count(id: usize) -> usize {
    set(id).body_count
}
#[export_name = "solverSetBodyAppend"]
pub unsafe extern "C" fn body_append(id: usize) -> usize {
    let s = set(id);
    let i = s.body_count;
    s.body_count += 1;
    if id != AWAKE {
        for c in [1] {
            s.columns.reserve(c, s.body_count * STRIDES[c] * 4);
        }
    }
    i
}
#[export_name = "solverSetBodyPop"]
pub unsafe extern "C" fn body_pop(id: usize) {
    set(id).body_count -= 1;
}
#[export_name = "solverSetLayout"]
pub unsafe extern "C" fn layout(id: usize) -> *const u32 {
    let s = set(id);
    s.columns.layout[2] = s.columns.layout[1];
    s.columns.layout[5] = s.columns.layout[1];
    s.columns.layout[4] = s.columns.layout[0];
    s.columns.layout.as_ptr()
}
#[export_name = "solverSetArrayCount"]
pub unsafe extern "C" fn array_count(id: usize, kind: usize) -> usize {
    set(id).indices[kind].len()
}
#[export_name = "solverSetArrayGet"]
pub unsafe extern "C" fn array_get(id: usize, kind: usize, i: usize) -> i32 {
    set(id).indices[kind][i]
}
#[export_name = "solverSetArrayPush"]
pub unsafe extern "C" fn array_push(id: usize, kind: usize, value: i32) -> usize {
    let v = &mut set(id).indices[kind];
    let i = v.len();
    v.push(value);
    i
}
#[export_name = "solverSetArrayRemove"]
pub unsafe extern "C" fn array_remove(id: usize, kind: usize, i: usize) -> i32 {
    let v = &mut set(id).indices[kind];
    let last = v.len() - 1;
    v.swap_remove(i);
    if i == last {
        -1
    } else {
        last as i32
    }
}
#[export_name = "solverSetArrayWrite"]
pub unsafe extern "C" fn array_write(id: usize, kind: usize, i: usize, value: i32) {
    set(id).indices[kind][i] = value;
}
#[export_name = "solverSetArrayPop"]
pub unsafe extern "C" fn array_pop(id: usize, kind: usize) {
    set(id).indices[kind].pop();
}
pub unsafe fn reset(id: usize) {
    for s in &mut WORLDS[id].sets {
        s.columns.release();
        s.joint_sims.records.release();
    }
    WORLDS[id] = Sets {
        sets: Vec::new(),
        free: Vec::new(),
    };
}
pub unsafe fn snapshot(id: usize, out: &mut Vec<u8>) {
    let w = &WORLDS[id];
    regions::write_word(out, w.sets.len());
    for s in &w.sets {
        regions::write_word(out, s.index as usize);
        regions::write_word(out, s.body_count);
        s.columns.snapshot(out);
        regions::write_word(out, s.joint_sims.count);
        s.joint_sims.records.snapshot(out);
        for v in &s.indices {
            regions::write_word(out, v.len());
            for &x in v {
                regions::write_word(out, x as usize);
            }
        }
    }
    regions::write_word(out, w.free.len());
    for &x in &w.free {
        regions::write_word(out, x);
    }
}
pub unsafe fn restore(id: usize, input: &mut &[u8]) {
    reset(id);
    let w = &mut WORLDS[id];
    let count = regions::read_word(input);
    for _ in 0..count {
        let mut s = SolverSet::empty();
        s.index = regions::read_word(input) as i32;
        s.body_count = regions::read_word(input);
        s.columns.restore(input);
        s.joint_sims.count = regions::read_word(input);
        s.joint_sims.records.restore(input);
        for v in &mut s.indices {
            let n = regions::read_word(input);
            for _ in 0..n {
                v.push(regions::read_word(input) as i32);
            }
        }
        w.sets.push(s);
    }
    let n = regions::read_word(input);
    for _ in 0..n {
        w.free.push(regions::read_word(input));
    }
}

pub unsafe fn awake_base(column: usize) -> usize {
    set(AWAKE).columns.layout[if column == 2 || column == 5 {
        1
    } else if column == 4 {
        0
    } else {
        column
    }] as usize
}
pub unsafe fn reserve_awake(cap: usize) {
    let s = set(AWAKE);
    for c in 0..6 {
        s.columns
            .reserve(c, (cap + crate::bodies::IDENT_RECORDS) * STRIDES[c] * 4);
    }
}
pub unsafe fn joint_array(id: usize) -> &'static mut crate::joints::JointArray {
    &mut set(id).joint_sims
}
pub(crate) unsafe fn body_ptr_world(
    world: usize,
    id: usize,
    index: usize,
    column: usize,
) -> *mut u32 {
    let column = if column == 2 || column == 5 {
        1
    } else {
        column
    };
    if column == 4 {
        (WORLDS[world].sets[id].columns.layout[0] as *mut u32)
            .add(index * STATE_STRIDE + crate::body::STATE_FLAGS)
    } else {
        (WORLDS[world].sets[id].columns.layout[column] as *mut u32).add(index * STRIDES[column])
    }
}

#[export_name = "simColumnPtr"]
pub unsafe extern "C" fn sim_column_ptr(
    world: usize,
    set: usize,
    index: usize,
    column: usize,
) -> usize {
    body_ptr_world(world, set, index, column) as usize
}

pub(crate) unsafe fn body_ptr(id: usize, index: usize, column: usize) -> *mut u32 {
    body_ptr_world(regions::active(), id, index, column)
}
#[export_name = "solverSetBodyId"]
pub unsafe extern "C" fn body_id(set: usize, index: usize) -> u32 {
    *body_ptr(set, index, 5).add(crate::body::S2_BODY_ID)
}

unsafe fn copy_body(source: usize, index: usize, target: usize, destination: usize) {
    for c in [1] {
        core::ptr::copy(
            body_ptr(source, index, c),
            body_ptr(target, destination, c),
            STRIDES[c],
        );
    }
}
unsafe fn remove_body(source: usize, index: usize) -> u32 {
    let last = body_count(source) - 1;
    let mut moved = u32::MAX;
    if index != last {
        copy_body(source, last, source, index);
        moved = *body_ptr(source, index, 5).add(crate::body::S2_BODY_ID);
        crate::bodies::set_location(moved as usize, source, index);
        if source == AWAKE {
            for c in [0] {
                core::ptr::copy(
                    body_ptr(source, last, c),
                    body_ptr(source, index, c),
                    STRIDES[c],
                );
            }
        }
    }
    body_pop(source);
    moved
}
unsafe fn wake_state(index: usize, flags: u32, head: i32) {
    body_ptr(AWAKE, index, 0).write_bytes(0, STATE_STRIDE);
    *body_ptr(AWAKE, index, 0).add(12) = 1.0f32.to_bits();
    *body_ptr(AWAKE, index, 4) = flags;
    let _ = head;
}
static mut BODY_RESULT: [u32; 2] = [0; 2];
#[export_name = "solverSetTransferBody"]
pub unsafe extern "C" fn transfer_body(
    source: usize,
    index: usize,
    target: usize,
    flags: u32,
    head: i32,
    clear_transient: bool,
) -> usize {
    let destination = body_append(target);
    copy_body(source, index, target, destination);
    if clear_transient {
        *body_ptr(target, destination, 5).add(crate::body::S2_FLAGS) &=
            !(crate::body::flags::IS_FAST
                | crate::body::flags::IS_SPEED_CAPPED
                | crate::body::flags::HAD_TIME_OF_IMPACT);
    }
    if target == AWAKE {
        wake_state(destination, flags, head);
    }
    let id = *body_ptr(target, destination, 5).add(crate::body::S2_BODY_ID);
    if source == AWAKE && target >= 3 {
        let record = crate::bodies::record_mut(regions::active(), id as usize);
        if record.body_move_index != -1 {
            crate::bodies::mark_move_asleep(record.body_move_index as usize);
            record.body_move_index = -1;
        }
    }
    crate::bodies::set_location(id as usize, target, destination);
    let moved = remove_body(source, index);
    BODY_RESULT = [destination as u32, moved];
    core::ptr::addr_of!(BODY_RESULT) as usize
}
#[export_name = "solverSetWakeBody"]
pub unsafe extern "C" fn wake_body(source: usize, index: usize, flags: u32, head: i32) -> usize {
    let destination = body_append(AWAKE);
    copy_body(source, index, AWAKE, destination);
    wake_state(destination, flags, head);
    let id = *body_ptr(AWAKE, destination, 5).add(crate::body::S2_BODY_ID);
    crate::bodies::set_location(id as usize, AWAKE, destination);
    crate::bodies::record_mut(regions::active(), id as usize).sleep_time = 0.0;
    destination
}
#[export_name = "solverSetRemoveBody"]
pub unsafe extern "C" fn destroy_body(source: usize, index: usize) -> u32 {
    remove_body(source, index)
}
#[export_name = "solverSetCopyBody"]
pub unsafe extern "C" fn copy_body_row(
    source: usize,
    index: usize,
    target: usize,
    destination: usize,
) {
    copy_body(source, index, target, destination)
}
#[export_name = "solverSetMoveContact"]
pub unsafe extern "C" fn move_contact(source: usize, index: usize, target: usize) -> usize {
    use crate::manifold_abi::*;
    let id = array_get(source, 0, index);
    let destination = array_push(target, 0, id);
    let moved = array_remove(source, 0, index);
    let d = crate::manifolds::dir_col();
    if moved != -1 {
        d.set(
            array_get(source, 0, index) as usize * DIR_STRIDE + DIR_LOCAL_INDEX,
            index as u32,
        );
    }
    d.set(id as usize * DIR_STRIDE + DIR_SET_INDEX, target as u32);
    d.set(
        id as usize * DIR_STRIDE + DIR_LOCAL_INDEX,
        destination as u32,
    );
    destination
}
#[export_name = "solverSetSleepContact"]
pub unsafe extern "C" fn sleep_contact(id: usize, target: usize) {
    use crate::manifold_abi::*;
    let d = crate::manifolds::dir_col();
    let o = id * DIR_STRIDE;
    let destination = array_push(target, 0, id as i32);
    crate::constraint_graph::remove_contact(
        d.get(o + DIR_EDGE_A) as usize,
        d.get(o + DIR_EDGE_B) as usize,
        d.get(o + DIR_COLOR_INDEX) as usize,
        d.get(o + DIR_LOCAL_INDEX) as usize,
        d.get(o + 6) & 0x00400000 != 0,
    );
    d.set(o + DIR_SET_INDEX, target as u32);
    d.set(o + DIR_COLOR_INDEX, u32::MAX);
    d.set(o + DIR_LOCAL_INDEX, destination as u32);
}
static mut ISLAND_RESULT: [u32; 2] = [0; 2];
#[export_name = "solverSetMoveIsland"]
pub unsafe extern "C" fn move_island(source: usize, index: usize, target: usize) -> usize {
    let id = array_get(source, 1, index);
    let destination = array_push(target, 1, id);
    let old = array_remove(source, 1, index);
    let moved = if old == -1 {
        u32::MAX
    } else {
        array_get(source, 1, index) as u32
    };
    ISLAND_RESULT = [destination as u32, moved];
    core::ptr::addr_of!(ISLAND_RESULT) as usize
}
pub unsafe fn merge(mut target: usize, mut source: usize) {
    use crate::manifold_abi::*;
    assert!(target >= 3 && source >= 3 && target != source);
    if body_count(target) < body_count(source) {
        core::mem::swap(&mut target, &mut source);
    }
    for i in 0..body_count(source) {
        let id = *body_ptr(source, i, 5).add(crate::body::S2_BODY_ID);
        let destination = body_append(target);
        copy_body(source, i, target, destination);
        crate::bodies::set_location(id as usize, target, destination);
    }
    let d = crate::manifolds::dir_col();
    for i in 0..array_count(source, 0) {
        let id = array_get(source, 0, i);
        let destination = array_push(target, 0, id);
        d.set(id as usize * DIR_STRIDE + DIR_SET_INDEX, target as u32);
        d.set(
            id as usize * DIR_STRIDE + DIR_LOCAL_INDEX,
            destination as u32,
        );
    }
    let source_key = crate::constraint_graph::COLORS + source;
    let target_key = crate::constraint_graph::COLORS + target;
    for i in 0..crate::joints::count(source_key) {
        let destination = crate::joints::append(target_key);
        let src = (crate::joints::pointer(source_key) as *const u32)
            .add(i * crate::joint_abi::JOINT_STRIDE);
        let dst = (crate::joints::pointer(target_key) as *mut u32)
            .add(destination * crate::joint_abi::JOINT_STRIDE);
        core::ptr::copy_nonoverlapping(src, dst, crate::joint_abi::JOINT_STRIDE);
        crate::joint_record::set_location(
            *src.add(crate::joint_abi::J_JOINT_ID) as usize,
            target_key,
            destination,
        );
    }
    for i in 0..array_count(source, 1) {
        let id = array_get(source, 1, i);
        let destination = array_push(target, 1, id);
        crate::island::set_field(id as usize, 0, target as i32);
        crate::island::set_field(id as usize, 1, destination as i32);
    }
    destroy(source);
}

pub unsafe fn transfer_joint(
    source: usize,
    color: usize,
    index: usize,
    target: usize,
    a: usize,
    b: usize,
) {
    let source_key = if source == AWAKE {
        color
    } else {
        crate::constraint_graph::COLORS + source
    };
    if target == AWAKE {
        crate::constraint_graph::add_joint(source_key, index, a, b);
    } else {
        if source == AWAKE {
            crate::constraint_graph::clear(color, a, b);
        }
        crate::joints::move_record(source_key, index, crate::constraint_graph::COLORS + target);
    }
}

pub unsafe fn wake(set: usize) {
    use crate::{
        bodies, constraint_graph as graph, island, joint_abi::J_JOINT_ID, joint_record as records,
        joints, manifold_abi::*, manifolds,
    };
    if set < 3 {
        return;
    }
    let world = regions::active();
    let count = body_count(set);
    for i in 0..count {
        let id = body_id(set, i) as usize;
        let body = *bodies::record(world, id);
        wake_body(set, i, body.flags, body.head_shape_id);
        crate::shape_lifecycle::sync_body(world, id);
        let mut key = body.head_contact_key;
        while key != -1 {
            let id = (key >> 1) as usize;
            let o = id * DIR_STRIDE;
            let d = manifolds::dir_col();
            key = d.get(o + DIR_EDGE_A + 2 + 3 * (key & 1) as usize) as i32;
            if d.get(o + DIR_SET_INDEX) == 1 {
                move_contact(1, d.get(o + DIR_LOCAL_INDEX) as usize, 2);
            }
        }
    }
    for i in 0..array_count(set, 0) {
        let id = array_get(set, 0, i) as usize;
        let d = manifolds::dir_col();
        let o = id * DIR_STRIDE;
        let a = d.get(o + DIR_EDGE_A) as usize;
        let b = d.get(o + DIR_EDGE_B) as usize;
        graph::add_contact(
            id,
            bodies::record(world, a).local_index as u32,
            bodies::record(world, b).local_index as u32,
        );
        d.set(o + DIR_SET_INDEX, 2);
    }
    let key = graph::COLORS + set;
    let count_joints = joints::count(key);
    for i in 0..count_joints {
        let index = i.min(count_joints - 1 - i);
        let id = joints::read_word(key, index, J_JOINT_ID) as usize;
        let r = *records::record(id);
        graph::add_joint(
            key,
            index,
            r.edges[0].body_id as usize,
            r.edges[1].body_id as usize,
        );
    }
    for i in 0..array_count(set, 1) {
        let id = array_get(set, 1, i);
        let index = array_push(2, 1, id);
        island::set_field(id as usize, 0, 2);
        island::set_field(id as usize, 1, index as i32);
    }
    // Classification needs the final graph placement of both endpoints.
    for i in 0..count {
        let id = body_id(set, i) as usize;
        bodies::sync_contacts(id);
        let mut key = bodies::record(world, id).head_contact_key;
        while key != -1 {
            let id = (key >> 1) as usize;
            let d = manifolds::dir_col();
            key = d.get(id * DIR_STRIDE + DIR_EDGE_A + 2 + 3 * (key & 1) as usize) as i32;
            crate::contact_list::update(id);
        }
    }
    destroy(set);
}
#[export_name = "solverSetWake"]
pub unsafe extern "C" fn wake_set(set: usize) {
    wake(set);
}
