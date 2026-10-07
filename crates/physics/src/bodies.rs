//! Each World's awake columns and public body lifecycle records own their allocations.

#[export_name = "bodyShouldBodiesCollide"]
pub unsafe extern "C" fn should_collide(a: u32, b: u32) -> bool {
    let world_id = crate::regions::active();
    let a = record(world_id, a as usize);
    let b = record(world_id, b as usize);
    if a.body_type != 2 && b.body_type != 2 {
        return false;
    }
    let (mut key, other) = if a.joint_count < b.joint_count {
        (a.head_joint_key, b.id)
    } else {
        (b.head_joint_key, a.id)
    };
    while key != -1 {
        let joint = crate::joint_record::record((key >> 1) as usize);
        let edge = (key & 1) as usize;
        if !joint.collide_connected && joint.edges[edge ^ 1].body_id == other {
            return false;
        }
        key = joint.edges[edge].next_key;
    }
    true
}
use crate::body::flags::DYNAMIC;
use crate::body::{FIN_STRIDE, SIM_STRIDE, STATE_STRIDE};
use crate::regions::{self, Columns, MAX_WORLDS};

pub const IDENT_RECORDS: usize = 8;
pub const MOVE_STRIDE: usize = core::mem::size_of::<crate::events::BodyMove>() / 4;
const B_STATE: usize = 0;
const B_SIM: usize = 1;
const B_FIN: usize = 2;
const B_FLAGS: usize = 4;
const B_SIM2: usize = 5;
const B_MOVE: usize = 6;
const B_RECORD_EID: usize = 7;
const B_SYNC_EID: usize = 8;
const B_SYNC_POS: usize = 9;
const B_SYNC_QUAT: usize = 10;
const B_SYNC_VEL: usize = 11;
const B_SYNC_INDEX: usize = 12;
const B_RECORD: usize = 13;
const N_BODY: usize = 14;

pub unsafe fn record(world_id: usize, id: usize) -> &'static crate::body_record::BodyRecord {
    &*(world(world_id).columns.layout[B_RECORD] as *const crate::body_record::BodyRecord).add(id)
}

pub unsafe fn record_mut(
    world_id: usize,
    id: usize,
) -> &'static mut crate::body_record::BodyRecord {
    &mut *(world(world_id).columns.layout[B_RECORD] as *mut crate::body_record::BodyRecord).add(id)
}

struct Bodies {
    columns: Columns<N_BODY>,
    cap: usize,
    next: usize,
    free: Vec<u32>,
}
impl Bodies {
    const EMPTY: Self = Self {
        columns: Columns::EMPTY,
        cap: 0,
        next: 0,
        free: Vec::new(),
    };
}
static mut WORLDS: [Bodies; MAX_WORLDS] = [const { Bodies::EMPTY }; MAX_WORLDS];
unsafe fn world(id: usize) -> &'static Bodies {
    &WORLDS[id]
}
unsafe fn world_mut(id: usize) -> &'static mut Bodies {
    &mut WORLDS[id]
}
fn base(column: usize) -> usize {
    unsafe {
        if column < 6 {
            crate::solver_set::awake_base(column)
        } else {
            world(regions::active()).columns.layout[column] as usize
        }
    }
}
pub unsafe fn column(id: usize, column: usize, stride: usize) -> crate::col::Col<'static, f32> {
    let body = record(regions::active(), id);
    crate::col::Col::new(
        crate::solver_set::body_ptr(body.set_index as usize, body.local_index as usize, column)
            as *mut f32,
        stride,
    )
}

pub unsafe fn geometry(id: usize) -> (crate::math::Transform, crate::body::SimFinalize, u32) {
    let sim = column(id, 1, SIM_STRIDE);
    let fin = column(id, 2, FIN_STRIDE);
    let sim2 = column(id, 5, crate::body::SIM2_STRIDE);
    let pose = crate::math::Transform {
        p: crate::math::Vec3::new(fin.get(0), fin.get(1), fin.get(2)),
        q: crate::math::Quat {
            v: crate::math::Vec3::new(sim.get(3), sim.get(4), sim.get(5)),
            s: sim.get(6),
        },
    };
    (
        pose,
        crate::body::read_fin(fin, 0),
        sim2.get(crate::body::S2_FLAGS).to_bits(),
    )
}

pub unsafe fn set_location(id: usize, set: usize, index: usize) {
    let body = record_mut(regions::active(), id);
    body.set_index = set as i32;
    body.local_index = index as i32;
    sync_contacts(id);
}

pub unsafe fn sync_contacts(id: usize) {
    use crate::manifold_abi::*;
    let body = record(regions::active(), id);
    let index = if body.body_type == 0 {
        u32::MAX
    } else {
        body.local_index as u32
    };
    let d = crate::manifolds::dir_col();
    let mut key = body.head_contact_key;
    while key != -1 {
        let edge = (key & 1) as usize;
        let offset = (key >> 1) as usize * DIR_STRIDE;
        d.set(offset + 9 + edge, index);
        key = d.get(offset + DIR_EDGE_A + 2 + 3 * edge) as i32;
    }
}

pub fn state_base() -> usize {
    base(B_STATE)
}
pub fn flags_base() -> usize {
    base(B_STATE) + crate::body::STATE_FLAGS * 4
}
pub fn sim_base() -> usize {
    base(B_SIM)
}
pub fn fin_base() -> usize {
    base(B_FIN)
}
pub fn sim2_base() -> usize {
    base(B_SIM2)
}
pub unsafe fn get_type(world_id: usize, id: usize) -> u32 {
    record(world_id, id).body_type as u32
}
pub fn move_base() -> usize {
    base(B_MOVE)
}

#[export_name = "bodyCap"]
pub extern "C" fn body_cap() -> usize {
    unsafe { world(regions::active()).cap }
}
#[export_name = "bodyLayoutPtr"]
pub extern "C" fn body_layout_ptr() -> *const u32 {
    unsafe {
        let w = world_mut(regions::active());
        for c in 0..6 {
            w.columns.layout[c] = crate::solver_set::awake_base(c) as u32;
        }
        w.columns.layout.as_ptr()
    }
}
#[export_name = "reserveBodies"]
pub extern "C" fn reserve_bodies(cap: usize) -> u32 {
    unsafe {
        let w = world_mut(regions::active());
        if cap <= w.cap {
            return 0;
        }
        let old = w.cap;
        crate::solver_set::reserve_awake(cap);
        for column in [B_RECORD_EID, B_SYNC_EID, B_SYNC_INDEX] {
            w.columns.reserve(column, cap * 4);
        }
        w.columns.reserve(
            B_RECORD,
            cap * core::mem::size_of::<crate::body_record::BodyRecord>(),
        );
        for id in old..cap {
            *(w.columns.layout[B_RECORD] as *mut crate::body_record::BodyRecord).add(id) =
                crate::body_record::BodyRecord::EMPTY;
        }
        w.columns.reserve(B_MOVE, cap * MOVE_STRIDE * 4);
        for column in [B_SYNC_POS, B_SYNC_QUAT, B_SYNC_VEL] {
            w.columns.reserve(column, cap * 16);
        }
        for id in old..cap {
            *(w.columns.layout[B_RECORD_EID] as *mut u32).add(id) = u32::MAX;
        }
        // Wide null lanes need one write-disjoint identity per worker.
        for worker in 0..IDENT_RECORDS {
            let ptr = (crate::solver_set::awake_base(B_STATE) as *mut f32)
                .add((cap + worker) * STATE_STRIDE);
            ptr.write_bytes(0, STATE_STRIDE);
            *ptr.add(12) = 1.0;
            *(ptr as *mut u32).add(crate::body::STATE_FLAGS) = DYNAMIC;
        }
        w.cap = cap;
        1
    }
}
#[export_name = "bodySetActiveWorld"]
pub extern "C" fn body_set_active_world(world: u32) {
    regions::select(world);
}
#[export_name = "bodySetEntity"]
pub extern "C" fn body_set_entity(id: usize, body: usize, eid: u32) {
    unsafe {
        *(world(id).columns.layout[B_RECORD_EID] as *mut u32).add(body) = eid;
    }
}
#[export_name = "bodySyncMoved"]
pub extern "C" fn body_sync_moved(count: usize) -> usize {
    unsafe {
        let layout = world(regions::active()).columns.layout;
        let mut written = 0;
        for row in 0..count {
            let event = &*(layout[B_MOVE] as *const crate::events::BodyMove).add(row);
            let body = event.body.index1 as usize - 1;
            let eid = *(layout[B_RECORD_EID] as *const u32).add(body);
            let index = (layout[B_SYNC_INDEX] as *mut u32).add(row);
            *index = u32::MAX;
            if eid == u32::MAX {
                continue;
            }
            *index = written as u32;
            *(layout[B_SYNC_EID] as *mut u32).add(written) = eid;
            let pos = (layout[B_SYNC_POS] as *mut f32).add(written * 4);
            let quat = (layout[B_SYNC_QUAT] as *mut f32).add(written * 4);
            let vel = (layout[B_SYNC_VEL] as *mut f32).add(written * 4);
            for lane in 0..3 {
                *pos.add(lane) = *(event.transform.p.as_ptr()).add(lane);
                *vel.add(lane) = *(layout[B_STATE] as *const f32).add(row * STATE_STRIDE + lane);
            }
            *pos.add(3) = 0.0;
            *vel.add(3) = 0.0;
            core::ptr::copy_nonoverlapping(event.transform.q.as_ptr(), quat, 4);
            written += 1;
        }
        written
    }
}
pub(crate) unsafe fn mark_move_asleep(index: usize) {
    let layout = world(regions::active()).columns.layout;
    (*(layout[B_MOVE] as *mut crate::events::BodyMove).add(index)).fell_asleep = true;
    let row = *(layout[B_SYNC_INDEX] as *const u32).add(index);
    if row != u32::MAX {
        (layout[B_SYNC_VEL] as *mut f32)
            .add(row as usize * 4)
            .write_bytes(0, 4);
    }
}

#[export_name = "bodyCreate"]
pub extern "C" fn body_create(id: u32) -> u32 {
    regions::select(id);
    unsafe {
        let p = world(id as usize);
        if p.free.is_empty() && p.next == p.cap {
            reserve_bodies((p.cap * 2).max(16));
        }
        let w = world_mut(id as usize);
        let body = if let Some(body) = w.free.pop() {
            body as usize
        } else {
            let body = w.next;
            w.next += 1;
            body
        };
        let record = (w.columns.layout[B_RECORD] as *mut crate::body_record::BodyRecord).add(body);
        let generation = (*record).generation.wrapping_add(1);
        *record = crate::body_record::BodyRecord {
            id: body as i32,
            generation,
            ..crate::body_record::BodyRecord::EMPTY
        };
        *(w.columns.layout[B_RECORD_EID] as *mut u32).add(body) = u32::MAX;
        body as u32
    }
}
#[export_name = "bodyDestroy"]
pub extern "C" fn body_destroy(id: u32, body: u32) -> u32 {
    unsafe {
        let w = world_mut(id as usize);
        let body = body as usize;
        if body >= w.next || body >= w.cap {
            return u32::MAX;
        }
        if record(id as usize, body).id == -1 {
            return u32::MAX;
        }
        regions::select(id);
        let previous = *record(id as usize, body);
        let moved = if previous.set_index >= 0 {
            let source = previous.set_index as usize;
            let moved = crate::solver_set::destroy_body(source, previous.local_index as usize);
            if source >= 3 && crate::solver_set::body_count(source) == 0 {
                crate::solver_set::destroy(source);
            }
            moved
        } else {
            u32::MAX
        };
        let record = record_mut(id as usize, body);
        record.id = -1;
        record.set_index = -1;
        record.local_index = -1;
        w.free.push(body as u32);
        moved
    }
}
#[export_name = "bodyResetWorld"]
pub extern "C" fn body_reset_world(id: u32) {
    unsafe {
        let w = world_mut(id as usize);
        w.columns.release();
        *w = Bodies::EMPTY;
    }
}
#[export_name = "bodyGeneration"]
pub extern "C" fn body_generation(id: u32, body: u32) -> u32 {
    unsafe {
        let w = world(id as usize);
        if body as usize >= w.cap {
            return 0;
        }
        record(id as usize, body as usize).generation as u32
    }
}
#[export_name = "bodyAlive"]
pub extern "C" fn body_alive(id: u32, body: u32) -> u32 {
    unsafe {
        let w = world(id as usize);
        if body as usize >= w.cap {
            return 0;
        }
        (record(id as usize, body as usize).id != -1) as u32
    }
}
#[export_name = "bodyCount"]
pub extern "C" fn body_count(id: u32) -> usize {
    unsafe {
        let w = world(id as usize);
        w.next - w.free.len()
    }
}
#[export_name = "bodyLength"]
pub extern "C" fn body_length(id: u32) -> usize {
    unsafe { world(id as usize).next }
}
pub fn active_generation(id: u32) -> u32 {
    body_generation(regions::active() as u32, id)
}
pub unsafe fn snapshot(id: usize, out: &mut Vec<u8>) {
    let w = &WORLDS[id];
    for value in [w.cap, w.next, w.free.len()] {
        regions::write_word(out, value);
    }
    for &id in &w.free {
        regions::write_word(out, id as usize);
    }
    w.columns.snapshot(out);
}
pub unsafe fn restore(id: usize, input: &mut &[u8]) {
    let w = &mut WORLDS[id];
    w.cap = regions::read_word(input);
    w.next = regions::read_word(input);
    let count = regions::read_word(input);
    w.free.clear();
    w.free.reserve(count);
    for _ in 0..count {
        w.free.push(regions::read_word(input) as u32);
    }
    w.columns.restore(input);
}
