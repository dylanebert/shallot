//! solver_set.c's set array and id pool. Body columns retain the solver's column layout.
use crate::body::{FIN_OUT_STRIDE, FIN_STRIDE, SIM2_STRIDE, SIM_STRIDE, STATE_STRIDE};
use crate::regions::{self, Columns, MAX_WORLDS};
const AWAKE: usize = 2;
const STRIDES: [usize; 6] = [
    STATE_STRIDE,
    SIM_STRIDE,
    FIN_STRIDE,
    FIN_OUT_STRIDE,
    1,
    SIM2_STRIDE,
];
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
    merge_result: Vec<u32>,
}
static mut WORLDS: [Sets; MAX_WORLDS] = [const {
    Sets {
        sets: Vec::new(),
        free: Vec::new(),
        merge_result: Vec::new(),
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
        for c in [1, 2, 5] {
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
    set(id).columns.layout.as_ptr()
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
        merge_result: Vec::new(),
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
    set(AWAKE).columns.layout[column] as usize
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
    (WORLDS[world].sets[id].columns.layout[column] as *mut u32).add(index * STRIDES[column])
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
    (set(id).columns.layout[column] as *mut u32).add(index * STRIDES[column])
}
#[export_name = "solverSetBodyId"]
pub unsafe extern "C" fn body_id(set: usize, index: usize) -> u32 {
    *body_ptr(set, index, 5).add(crate::body::S2_BODY_ID)
}

unsafe fn copy_body(source: usize, index: usize, target: usize, destination: usize) {
    for c in [1, 2, 5] {
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
            for c in [0, 4] {
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
    *body_ptr(AWAKE, index, 5).add(crate::body::S2_HEAD_SHAPE) = head as u32;
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
            *(crate::bodies::move_base() as *mut u32)
                .add(record.body_move_index as usize * crate::bodies::MOVE_STRIDE + 2) = 1;
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
// Header: survivor, retired set, move count; moves: record kind, id, local index.
#[export_name = "solverSetMerge"]
pub unsafe extern "C" fn merge(mut target: usize, mut source: usize) -> usize {
    use crate::manifold_abi::*;
    assert!(target >= 3 && source >= 3 && target != source);
    if body_count(target) < body_count(source) {
        core::mem::swap(&mut target, &mut source);
    }
    let mut result = core::mem::take(&mut world().merge_result);
    result.clear();
    result.extend_from_slice(&[target as u32, source as u32, 0]);
    for i in 0..body_count(source) {
        let id = *body_ptr(source, i, 5).add(crate::body::S2_BODY_ID);
        let destination = body_append(target);
        copy_body(source, i, target, destination);
        crate::bodies::set_location(id as usize, target, destination);
        result.extend_from_slice(&[0, id, destination as u32]);
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
        result.extend_from_slice(&[
            1,
            *src.add(crate::joint_abi::J_JOINT_ID),
            destination as u32,
        ]);
    }
    for i in 0..array_count(source, 1) {
        let id = array_get(source, 1, i);
        let destination = array_push(target, 1, id);
        result.extend_from_slice(&[2, id as u32, destination as u32]);
    }
    destroy(source);
    result[2] = ((result.len() - 3) / 3) as u32;
    world().merge_result = result;
    world().merge_result.as_ptr() as usize
}

static mut JOINT_RESULT: [u32; 3] = [0; 3];
#[export_name = "solverSetTransferJoint"]
pub unsafe extern "C" fn transfer_joint(
    source: usize,
    color: usize,
    index: usize,
    target: usize,
    a: usize,
    b: usize,
) -> usize {
    let source_key = if source == AWAKE {
        color
    } else {
        crate::constraint_graph::COLORS + source
    };
    let target_color;
    let destination;
    let moved;
    if target == AWAKE {
        let ptr = crate::constraint_graph::add_joint(source_key, index, a, b) as *const u32;
        target_color = *ptr;
        destination = *ptr.add(1);
        moved = *ptr.add(2);
    } else {
        let key = crate::constraint_graph::COLORS + target;
        destination = crate::joints::count(key) as u32;
        target_color = u32::MAX;
        if source == AWAKE {
            crate::constraint_graph::clear(color, a, b);
        }
        moved = crate::joints::move_record(source_key, index, key);
    }
    JOINT_RESULT = [target_color, destination, moved];
    core::ptr::addr_of!(JOINT_RESULT) as usize
}
