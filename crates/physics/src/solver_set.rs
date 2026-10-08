//! solver_set.c's body sim/state arrays and set id pool.
use crate::body::{BodySim, BodyState, STATE_STRIDE};
use crate::regions::{self, MAX_WORLDS};
use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};
const AWAKE: usize = 2;

#[repr(C)]
struct BodyArray<T> {
    data: *mut T,
    count: i32,
    capacity: i32,
    #[cfg(feature = "box3d-oracle")]
    allocations: usize,
}
impl<T> BodyArray<T> {
    const EMPTY: Self = Self {
        data: 16 as *mut T,
        count: 0,
        capacity: 0,
        #[cfg(feature = "box3d-oracle")]
        allocations: 0,
    };
    unsafe fn reserve(&mut self, capacity: usize) {
        let capacity =
            i32::try_from(capacity).expect("body array capacity exceeds Box3D's int range");
        if capacity <= self.capacity {
            return;
        }
        let bytes = capacity as usize * core::mem::size_of::<T>();
        let layout = Layout::from_size_align(bytes, 16).unwrap();
        let data = alloc(layout) as *mut T;
        #[cfg(feature = "box3d-oracle")]
        {
            self.allocations += 1;
        }
        if data.is_null() {
            handle_alloc_error(layout);
        }
        if self.capacity != 0 {
            core::ptr::copy_nonoverlapping(self.data, data, self.capacity as usize);
            dealloc(
                self.data.cast(),
                Layout::from_size_align(self.capacity as usize * core::mem::size_of::<T>(), 16)
                    .unwrap(),
            );
        }
        self.data = data;
        self.capacity = capacity;
        regions::invalidate_views();
    }
    unsafe fn emplace(&mut self) -> usize {
        if self.count == self.capacity {
            let capacity = if self.capacity == 0 {
                16
            } else {
                self.capacity
                    .checked_mul(2)
                    .expect("body array capacity exceeds Box3D's int range")
            };
            self.reserve(capacity as usize);
        }
        let index = self.count as usize;
        self.count += 1;
        index
    }
    unsafe fn remove_swap(&mut self, index: usize) -> i32 {
        debug_assert!(index < self.count as usize);
        self.count -= 1;
        if index != self.count as usize {
            core::ptr::copy_nonoverlapping(
                self.data.add(self.count as usize),
                self.data.add(index),
                1,
            );
            self.count
        } else {
            -1
        }
    }
    unsafe fn release(&mut self) {
        if self.capacity != 0 {
            dealloc(
                self.data.cast(),
                Layout::from_size_align(self.capacity as usize * core::mem::size_of::<T>(), 16)
                    .unwrap(),
            );
            regions::invalidate_views();
        }
        *self = Self::EMPTY;
    }
    unsafe fn snapshot(&self, out: &mut Vec<u8>) {
        regions::write_word(out, self.count as usize);
        out.extend_from_slice(core::slice::from_raw_parts(
            self.data.cast::<u8>(),
            self.count as usize * core::mem::size_of::<T>(),
        ));
    }
    unsafe fn restore(&mut self, input: &mut &[u8]) {
        self.count = i32::try_from(regions::read_word(input)).unwrap();
        self.reserve(self.count as usize);
        assert!(0 <= self.count && self.count <= self.capacity);
        let bytes = self.count as usize * core::mem::size_of::<T>();
        let (data, rest) = input.split_at(bytes);
        core::ptr::copy_nonoverlapping(data.as_ptr(), self.data.cast::<u8>(), bytes);
        *input = rest;
    }
}
struct SolverSet {
    body_sims: BodyArray<BodySim>,
    body_states: BodyArray<BodyState>,
    body_layout: [u32; 6],
    indices: [Vec<i32>; 2],
    index: i32,
    joint_sims: crate::joints::JointArray,
}
impl SolverSet {
    fn empty() -> Self {
        Self {
            body_sims: BodyArray::EMPTY,
            body_states: BodyArray::EMPTY,
            body_layout: [16; 6],
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
unsafe fn world(world_index: usize) -> &'static mut Sets {
    &mut WORLDS[world_index]
}
unsafe fn set(world_index: usize, id: usize) -> &'static mut SolverSet {
    &mut world(world_index).sets[id]
}
#[export_name = "solverSetCreate"]
pub unsafe extern "C" fn create() -> usize {
    unsafe { create_in_world(crate::regions::active()) }
}

pub unsafe extern "C" fn create_in_world(world_index: usize) -> usize {
    let w = world(world_index);
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
    unsafe { count_in_world(crate::regions::active()) }
}

pub unsafe extern "C" fn count_in_world(world_index: usize) -> usize {
    world(world_index).sets.len()
}
#[export_name = "solverSetIndex"]
pub unsafe extern "C" fn index(id: usize) -> i32 {
    unsafe { index_in_world(crate::regions::active(), id) }
}

pub unsafe extern "C" fn index_in_world(world_index: usize, id: usize) -> i32 {
    set(world_index, id).index
}
#[export_name = "solverSetDestroy"]
pub unsafe extern "C" fn destroy(id: usize) {
    unsafe { destroy_in_world(crate::regions::active(), id) }
}

pub unsafe extern "C" fn destroy_in_world(world_index: usize, id: usize) {
    let s = set(world_index, id);
    s.body_sims.release();
    s.body_states.release();
    s.joint_sims.release();
    *s = SolverSet::empty();
    world(world_index).free.push(id);
}
#[export_name = "solverSetBodyCount"]
pub unsafe extern "C" fn body_count(id: usize) -> usize {
    unsafe { body_count_in_world(crate::regions::active(), id) }
}

pub unsafe extern "C" fn body_count_in_world(world_index: usize, id: usize) -> usize {
    set(world_index, id).body_sims.count as usize
}
#[export_name = "solverSetBodyAppend"]
pub unsafe extern "C" fn body_append(id: usize) -> usize {
    unsafe { body_append_in_world(crate::regions::active(), id) }
}

pub unsafe extern "C" fn body_append_in_world(world_index: usize, id: usize) -> usize {
    let s = set(world_index, id);
    let i = s.body_sims.emplace();
    if id == AWAKE {
        let state_index = s.body_states.emplace();
        debug_assert_eq!(i, state_index);
    }
    i
}
#[export_name = "solverSetBodyPop"]
pub unsafe extern "C" fn body_pop(id: usize) {
    unsafe { body_pop_in_world(crate::regions::active(), id) }
}

pub unsafe extern "C" fn body_pop_in_world(world_index: usize, id: usize) {
    let s = set(world_index, id);
    s.body_sims.count -= 1;
    if id == AWAKE {
        s.body_states.count -= 1;
    }
}
#[export_name = "solverSetLayout"]
pub unsafe extern "C" fn layout(id: usize) -> *const u32 {
    unsafe { layout_in_world(crate::regions::active(), id) }
}

pub unsafe extern "C" fn layout_in_world(world_index: usize, id: usize) -> *const u32 {
    let s = set(world_index, id);
    let sim = s.body_sims.data as usize as u32;
    let state = s.body_states.data as usize as u32;
    s.body_layout = [state, sim, sim, 16, state, sim];
    s.body_layout.as_ptr()
}
#[export_name = "solverSetArrayCount"]
pub unsafe extern "C" fn array_count(id: usize, kind: usize) -> usize {
    unsafe { array_count_in_world(crate::regions::active(), id, kind) }
}

pub unsafe extern "C" fn array_count_in_world(world_index: usize, id: usize, kind: usize) -> usize {
    set(world_index, id).indices[kind].len()
}
#[export_name = "solverSetArrayGet"]
pub unsafe extern "C" fn array_get(id: usize, kind: usize, i: usize) -> i32 {
    unsafe { array_get_in_world(crate::regions::active(), id, kind, i) }
}

pub unsafe extern "C" fn array_get_in_world(
    world_index: usize,
    id: usize,
    kind: usize,
    i: usize,
) -> i32 {
    set(world_index, id).indices[kind][i]
}
#[export_name = "solverSetArrayPush"]
pub unsafe extern "C" fn array_push(id: usize, kind: usize, value: i32) -> usize {
    unsafe { array_push_in_world(crate::regions::active(), id, kind, value) }
}

pub unsafe extern "C" fn array_push_in_world(
    world_index: usize,
    id: usize,
    kind: usize,
    value: i32,
) -> usize {
    let v = &mut set(world_index, id).indices[kind];
    let i = v.len();
    v.push(value);
    i
}
#[export_name = "solverSetArrayRemove"]
pub unsafe extern "C" fn array_remove(id: usize, kind: usize, i: usize) -> i32 {
    unsafe { array_remove_in_world(crate::regions::active(), id, kind, i) }
}

pub unsafe extern "C" fn array_remove_in_world(
    world_index: usize,
    id: usize,
    kind: usize,
    i: usize,
) -> i32 {
    let v = &mut set(world_index, id).indices[kind];
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
    unsafe { array_write_in_world(crate::regions::active(), id, kind, i, value) }
}

pub unsafe extern "C" fn array_write_in_world(
    world_index: usize,
    id: usize,
    kind: usize,
    i: usize,
    value: i32,
) {
    set(world_index, id).indices[kind][i] = value;
}
#[export_name = "solverSetArrayPop"]
pub unsafe extern "C" fn array_pop(id: usize, kind: usize) {
    unsafe { array_pop_in_world(crate::regions::active(), id, kind) }
}

pub unsafe extern "C" fn array_pop_in_world(world_index: usize, id: usize, kind: usize) {
    set(world_index, id).indices[kind].pop();
}
pub unsafe fn reset(id: usize) {
    for s in &mut WORLDS[id].sets {
        s.body_sims.release();
        s.body_states.release();
        s.joint_sims.release();
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
        s.body_sims.snapshot(out);
        s.body_states.snapshot(out);
        s.joint_sims.snapshot(out);
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
        s.body_sims.restore(input);
        s.body_states.restore(input);
        s.joint_sims.restore(input);
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

#[cfg(feature = "box3d-oracle")]
#[export_name = "box3dSleepingBodyCapacity"]
pub unsafe extern "C" fn sleeping_body_capacity(world: usize, body: usize) -> i32 {
    set(world, crate::bodies::record(world, body).set_index as usize)
        .body_sims
        .capacity
}
#[cfg(feature = "box3d-oracle")]
#[export_name = "box3dSleepingBodyAllocations"]
pub unsafe extern "C" fn sleeping_body_allocations(world: usize, body: usize) -> usize {
    set(world, crate::bodies::record(world, body).set_index as usize)
        .body_sims
        .allocations
}

pub unsafe fn awake_base(world_index: usize, column: usize) -> usize {
    let s = set(world_index, AWAKE);
    match column {
        0 | 4 => s.body_states.data as usize,
        1 | 2 | 5 => s.body_sims.data as usize,
        _ => 16,
    }
}
pub unsafe fn reserve_sleeping(world_index: usize, id: usize, bodies: usize, contacts: usize) {
    let s = set(world_index, id);
    s.body_sims.reserve(bodies);
    s.indices[0].reserve(contacts);
}
pub unsafe fn reserve_awake(world_index: usize, cap: usize) {
    let s = set(world_index, AWAKE);
    s.body_sims.reserve(cap);
    s.body_states.reserve(cap);
}
pub unsafe fn joint_array(world_index: usize, id: usize) -> &'static mut crate::joints::JointArray {
    &mut set(world_index, id).joint_sims
}
pub(crate) unsafe fn body_ptr_world(
    world: usize,
    id: usize,
    index: usize,
    column: usize,
) -> *mut u32 {
    let s = &WORLDS[world].sets[id];
    match column {
        0 => s.body_states.data.add(index).cast(),
        4 => s
            .body_states
            .data
            .add(index)
            .cast::<u32>()
            .add(crate::body::STATE_FLAGS),
        1 | 2 | 5 => s.body_sims.data.add(index).cast(),
        _ => 16 as *mut u32,
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

pub(crate) unsafe fn body_ptr(
    world_index: usize,
    id: usize,
    index: usize,
    column: usize,
) -> *mut u32 {
    body_ptr_world(world_index, id, index, column)
}
#[export_name = "solverSetBodyId"]
pub unsafe extern "C" fn body_id(set: usize, index: usize) -> u32 {
    unsafe { body_id_in_world(crate::regions::active(), set, index) }
}

pub unsafe extern "C" fn body_id_in_world(world_index: usize, set: usize, index: usize) -> u32 {
    *body_ptr(world_index, set, index, 5).add(crate::body::S2_BODY_ID)
}

unsafe fn copy_body(
    world_index: usize,
    source: usize,
    index: usize,
    target: usize,
    destination: usize,
) {
    let src = set(world_index, source).body_sims.data.add(index);
    let dst = set(world_index, target).body_sims.data.add(destination);
    core::ptr::copy(src, dst, 1);
}
unsafe fn remove_body(world_index: usize, source: usize, index: usize) -> u32 {
    let s = set(world_index, source);
    let removed = s.body_sims.remove_swap(index);
    let moved = if removed != -1 {
        let moved = (*s.body_sims.data.add(index)).body_id as u32;
        crate::bodies::set_location(world_index, moved as usize, source, index);
        moved
    } else {
        u32::MAX
    };
    if source == AWAKE {
        let state_removed = s.body_states.remove_swap(index);
        debug_assert_eq!(removed, state_removed);
    }
    moved
}
unsafe fn wake_state(world_index: usize, index: usize, flags: u32, head: i32) {
    body_ptr(world_index, AWAKE, index, 0).write_bytes(0, STATE_STRIDE);
    *body_ptr(world_index, AWAKE, index, 0).add(12) = 1.0f32.to_bits();
    *body_ptr(world_index, AWAKE, index, 4) = flags;
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
    unsafe {
        transfer_body_in_world(
            crate::regions::active(),
            source,
            index,
            target,
            flags,
            head,
            clear_transient,
        )
    }
}

pub unsafe extern "C" fn transfer_body_in_world(
    world_index: usize,
    source: usize,
    index: usize,
    target: usize,
    flags: u32,
    head: i32,
    clear_transient: bool,
) -> usize {
    if target == source {
        BODY_RESULT = [index as u32, u32::MAX];
        return core::ptr::addr_of!(BODY_RESULT) as usize;
    }
    let destination = set(world_index, target).body_sims.emplace();
    copy_body(world_index, source, index, target, destination);
    if clear_transient {
        *body_ptr(world_index, target, destination, 5).add(crate::body::S2_FLAGS) &=
            !(crate::body::flags::IS_FAST
                | crate::body::flags::IS_SPEED_CAPPED
                | crate::body::flags::HAD_TIME_OF_IMPACT);
    }
    let id = *body_ptr(world_index, target, destination, 5).add(crate::body::S2_BODY_ID);
    if source == AWAKE && target >= 3 {
        let record = crate::bodies::record_mut(world_index, id as usize);
        if record.body_move_index != -1 {
            crate::bodies::mark_move_asleep(world_index, record.body_move_index as usize);
            record.body_move_index = -1;
        }
    }
    let moved = remove_body(world_index, source, index);
    if target == AWAKE {
        let state_index = set(world_index, target).body_states.emplace();
        debug_assert_eq!(state_index, destination);
        wake_state(world_index, destination, flags, head);
    }
    crate::bodies::set_location(world_index, id as usize, target, destination);
    BODY_RESULT = [destination as u32, moved];
    core::ptr::addr_of!(BODY_RESULT) as usize
}
#[export_name = "solverSetWakeBody"]
pub unsafe extern "C" fn wake_body(source: usize, index: usize, flags: u32, head: i32) -> usize {
    unsafe { wake_body_in_world(crate::regions::active(), source, index, flags, head) }
}

pub unsafe extern "C" fn wake_body_in_world(
    world_index: usize,
    source: usize,
    index: usize,
    flags: u32,
    head: i32,
) -> usize {
    let destination = body_append_in_world(world_index, AWAKE);
    copy_body(world_index, source, index, AWAKE, destination);
    wake_state(world_index, destination, flags, head);
    let id = *body_ptr(world_index, AWAKE, destination, 5).add(crate::body::S2_BODY_ID);
    crate::bodies::set_location(world_index, id as usize, AWAKE, destination);
    crate::bodies::record_mut(world_index, id as usize).sleep_time = 0.0;
    destination
}
#[export_name = "solverSetRemoveBody"]
pub unsafe extern "C" fn destroy_body(source: usize, index: usize) -> u32 {
    unsafe { destroy_body_in_world(crate::regions::active(), source, index) }
}

pub unsafe extern "C" fn destroy_body_in_world(
    world_index: usize,
    source: usize,
    index: usize,
) -> u32 {
    remove_body(world_index, source, index)
}
#[export_name = "solverSetCopyBody"]
pub unsafe extern "C" fn copy_body_row(
    source: usize,
    index: usize,
    target: usize,
    destination: usize,
) {
    unsafe { copy_body_row_in_world(crate::regions::active(), source, index, target, destination) }
}

pub unsafe extern "C" fn copy_body_row_in_world(
    world_index: usize,
    source: usize,
    index: usize,
    target: usize,
    destination: usize,
) {
    copy_body(world_index, source, index, target, destination)
}
#[export_name = "solverSetMoveContact"]
pub unsafe extern "C" fn move_contact(source: usize, index: usize, target: usize) -> usize {
    unsafe { move_contact_in_world(crate::regions::active(), source, index, target) }
}

pub unsafe extern "C" fn move_contact_in_world(
    world_index: usize,
    source: usize,
    index: usize,
    target: usize,
) -> usize {
    use crate::manifold_abi::*;
    let id = array_get_in_world(world_index, source, 0, index);
    let destination = array_push_in_world(world_index, target, 0, id);
    let moved = array_remove_in_world(world_index, source, 0, index);
    let d = crate::manifolds::dir_col(world_index);
    if moved != -1 {
        d.set(
            array_get_in_world(world_index, source, 0, index) as usize * DIR_STRIDE
                + DIR_LOCAL_INDEX,
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
    unsafe { sleep_contact_in_world(crate::regions::active(), id, target) }
}

pub unsafe extern "C" fn sleep_contact_in_world(world_index: usize, id: usize, target: usize) {
    use crate::manifold_abi::*;
    let d = crate::manifolds::dir_col(world_index);
    let o = id * DIR_STRIDE;
    let destination = array_push_in_world(world_index, target, 0, id as i32);
    crate::constraint_graph::remove_contact_in_world(
        world_index,
        d.get(o + DIR_EDGE_A) as usize,
        d.get(o + DIR_EDGE_B) as usize,
        d.get(o + DIR_COLOR_INDEX) as usize,
        d.get(o + DIR_LOCAL_INDEX) as usize,
        d.get(o + DIR_FLAGS) & 0x00400000 != 0,
    );
    d.set(o + DIR_SET_INDEX, target as u32);
    d.set(o + DIR_COLOR_INDEX, u32::MAX);
    d.set(o + DIR_LOCAL_INDEX, destination as u32);
}
static mut ISLAND_RESULT: [u32; 2] = [0; 2];
#[export_name = "solverSetMoveIsland"]
pub unsafe extern "C" fn move_island(source: usize, index: usize, target: usize) -> usize {
    unsafe { move_island_in_world(crate::regions::active(), source, index, target) }
}

pub unsafe extern "C" fn move_island_in_world(
    world_index: usize,
    source: usize,
    index: usize,
    target: usize,
) -> usize {
    let id = array_get_in_world(world_index, source, 1, index);
    let destination = array_push_in_world(world_index, target, 1, id);
    let old = array_remove_in_world(world_index, source, 1, index);
    let moved = if old == -1 {
        u32::MAX
    } else {
        array_get_in_world(world_index, source, 1, index) as u32
    };
    ISLAND_RESULT = [destination as u32, moved];
    core::ptr::addr_of!(ISLAND_RESULT) as usize
}
pub unsafe fn merge(world_index: usize, mut target: usize, mut source: usize) {
    use crate::manifold_abi::*;
    assert!(target >= 3 && source >= 3 && target != source);
    if body_count_in_world(world_index, target) < body_count_in_world(world_index, source) {
        core::mem::swap(&mut target, &mut source);
    }
    for i in 0..body_count_in_world(world_index, source) {
        let id = *body_ptr(world_index, source, i, 5).add(crate::body::S2_BODY_ID);
        let destination = body_append_in_world(world_index, target);
        copy_body(world_index, source, i, target, destination);
        crate::bodies::set_location(world_index, id as usize, target, destination);
    }
    let d = crate::manifolds::dir_col(world_index);
    for i in 0..array_count_in_world(world_index, source, 0) {
        let id = array_get_in_world(world_index, source, 0, i);
        let destination = array_push_in_world(world_index, target, 0, id);
        d.set(id as usize * DIR_STRIDE + DIR_SET_INDEX, target as u32);
        d.set(
            id as usize * DIR_STRIDE + DIR_LOCAL_INDEX,
            destination as u32,
        );
    }
    let source_key = crate::constraint_graph::COLORS + source;
    let target_key = crate::constraint_graph::COLORS + target;
    for i in 0..crate::joints::count_in_world(world_index, source_key) {
        crate::joints::copy_record_in_world(world_index, source_key, i, target_key);
    }
    for i in 0..array_count_in_world(world_index, source, 1) {
        let id = array_get_in_world(world_index, source, 1, i);
        let destination = array_push_in_world(world_index, target, 1, id);
        crate::island::set_field_in_world(world_index, id as usize, 0, target as i32);
        crate::island::set_field_in_world(world_index, id as usize, 1, destination as i32);
    }
    destroy_in_world(world_index, source);
}

pub unsafe fn transfer_joint(
    world_index: usize,
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
        crate::constraint_graph::add_joint(world_index, source_key, index, a, b);
        crate::joints::remove_in_world(world_index, source_key, index);
    } else {
        if source == AWAKE {
            crate::constraint_graph::clear_in_world(world_index, color, a, b);
        }
        crate::joints::move_record_in_world(
            world_index,
            source_key,
            index,
            crate::constraint_graph::COLORS + target,
        );
    }
}

pub unsafe fn wake(world_index: usize, set: usize) {
    use crate::{
        bodies, constraint_graph as graph, island, joint_abi::J_JOINT_ID, joint_record as records,
        joints, manifold_abi::*, manifolds,
    };
    if set < 3 {
        return;
    }
    let world = world_index;
    let count = body_count_in_world(world_index, set);
    for i in 0..count {
        let id = body_id_in_world(world_index, set, i) as usize;
        let body = *bodies::record(world, id);
        wake_body_in_world(world_index, set, i, body.flags, body.head_shape_id);
        let mut key = body.head_contact_key;
        while key != -1 {
            let id = (key >> 1) as usize;
            let o = id * DIR_STRIDE;
            let d = manifolds::dir_col(world_index);
            key = d.get(o + DIR_EDGE_A + 2 + 3 * (key & 1) as usize) as i32;
            if d.get(o + DIR_SET_INDEX) == 1 {
                move_contact_in_world(world_index, 1, d.get(o + DIR_LOCAL_INDEX) as usize, 2);
            }
        }
    }
    for i in 0..array_count_in_world(world_index, set, 0) {
        let id = array_get_in_world(world_index, set, 0, i) as usize;
        let d = manifolds::dir_col(world_index);
        let o = id * DIR_STRIDE;
        let a = d.get(o + DIR_EDGE_A) as usize;
        let b = d.get(o + DIR_EDGE_B) as usize;
        graph::add_contact_in_world(
            world_index,
            id,
            bodies::record(world, a).local_index as u32,
            bodies::record(world, b).local_index as u32,
        );
        d.set(o + DIR_SET_INDEX, 2);
    }
    let key = graph::COLORS + set;
    let count_joints = joints::count_in_world(world_index, key);
    for i in 0..count_joints {
        let index = i;
        let id = joints::read_word_in_world(world_index, key, index, J_JOINT_ID) as usize;
        let r = records::record(world_index, id);
        graph::add_joint(
            world_index,
            key,
            index,
            r.edges[0].body_id as usize,
            r.edges[1].body_id as usize,
        );
    }
    for i in 0..array_count_in_world(world_index, set, 1) {
        let id = array_get_in_world(world_index, set, 1, i);
        let index = array_push_in_world(world_index, 2, 1, id);
        island::set_field_in_world(world_index, id as usize, 0, 2);
        island::set_field_in_world(world_index, id as usize, 1, index as i32);
    }
    destroy_in_world(world_index, set);
}
#[export_name = "solverSetWake"]
pub unsafe extern "C" fn wake_set(set: usize) {
    unsafe { wake_set_in_world(crate::regions::active(), set) }
}

pub unsafe extern "C" fn wake_set_in_world(world_index: usize, set: usize) {
    wake(world_index, set);
}
