//! Each World's awake columns and public body lifecycle records own their allocations.
use crate::body::flags::DYNAMIC;
use crate::body::{FIN_OUT_STRIDE, FIN_STRIDE, SIM2_STRIDE, SIM_STRIDE, STATE_STRIDE};
use crate::regions::{self, Columns, MAX_WORLDS};

pub const IDENT_RECORDS: usize = 8;
pub const MOVE_STRIDE: usize = 3;
const B_STATE: usize = 0;
const B_SIM: usize = 1;
const B_FIN: usize = 2;
const B_FIN_OUT: usize = 3;
const B_FLAGS: usize = 4;
const B_SIM2: usize = 5;
const B_RECORD_GENERATION: usize = 6;
const B_RECORD_ALIVE: usize = 7;
const B_RECORD_NEXT: usize = 8;
const B_MOVE: usize = 9;
const B_RECORD_EID: usize = 10;
const B_SYNC_EID: usize = 11;
const B_SYNC_POS: usize = 12;
const B_SYNC_QUAT: usize = 13;
const B_SYNC_VEL: usize = 14;
const B_SYNC_INDEX: usize = 15;
const N_BODY: usize = 16;

#[derive(Clone, Copy)]
struct Bodies {
    columns: Columns<N_BODY>,
    cap: usize,
    next: usize,
    free: i32,
    count: usize,
}
impl Bodies {
    const EMPTY: Self = Self {
        columns: Columns::EMPTY,
        cap: 0,
        next: 0,
        free: -1,
        count: 0,
    };
}
static mut WORLDS: [Bodies; MAX_WORLDS] = [Bodies::EMPTY; MAX_WORLDS];
unsafe fn world(id: usize) -> &'static Bodies {
    &WORLDS[id]
}
unsafe fn world_mut(id: usize) -> &'static mut Bodies {
    &mut WORLDS[id]
}
fn base(column: usize) -> usize {
    unsafe { world(regions::active()).columns.layout[column] as usize }
}
pub fn state_base() -> usize {
    base(B_STATE)
}
pub fn flags_base() -> usize {
    base(B_FLAGS)
}
pub fn sim_base() -> usize {
    base(B_SIM)
}
pub fn fin_base() -> usize {
    base(B_FIN)
}
pub fn fin_out_base() -> usize {
    base(B_FIN_OUT)
}
pub fn sim2_base() -> usize {
    base(B_SIM2)
}
pub fn record_generation_base() -> usize {
    base(B_RECORD_GENERATION)
}
pub fn record_alive_base() -> usize {
    base(B_RECORD_ALIVE)
}
pub fn record_next_base() -> usize {
    base(B_RECORD_NEXT)
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
    unsafe { world(regions::active()).columns.layout.as_ptr() }
}
#[export_name = "reserveBodies"]
pub extern "C" fn reserve_bodies(cap: usize) -> u32 {
    unsafe {
        let w = world_mut(regions::active());
        if cap <= w.cap {
            return 0;
        }
        let old = w.cap;
        for (column, stride) in [
            (B_STATE, STATE_STRIDE),
            (B_SIM, SIM_STRIDE),
            (B_FIN, FIN_STRIDE),
            (B_FIN_OUT, FIN_OUT_STRIDE),
            (B_FLAGS, 1),
            (B_SIM2, SIM2_STRIDE),
        ] {
            w.columns
                .reserve(column, (cap + IDENT_RECORDS) * stride * 4);
        }
        for column in [
            B_RECORD_GENERATION,
            B_RECORD_ALIVE,
            B_RECORD_NEXT,
            B_RECORD_EID,
            B_SYNC_EID,
            B_SYNC_INDEX,
        ] {
            w.columns.reserve(column, cap * 4);
        }
        w.columns.reserve(B_MOVE, cap * MOVE_STRIDE * 4);
        for column in [B_SYNC_POS, B_SYNC_QUAT, B_SYNC_VEL] {
            w.columns.reserve(column, cap * 16);
        }
        for id in old..cap {
            *(w.columns.layout[B_RECORD_GENERATION] as *mut u32).add(id) = 0;
            *(w.columns.layout[B_RECORD_ALIVE] as *mut u32).add(id) = 0;
            *(w.columns.layout[B_RECORD_NEXT] as *mut u32).add(id) = u32::MAX;
            *(w.columns.layout[B_RECORD_EID] as *mut u32).add(id) = u32::MAX;
        }
        // Wide null lanes need one write-disjoint identity per worker.
        for worker in 0..IDENT_RECORDS {
            let ptr = (w.columns.layout[B_STATE] as *mut f32).add((cap + worker) * STATE_STRIDE);
            ptr.write_bytes(0, STATE_STRIDE);
            *ptr.add(12) = 1.0;
            *(w.columns.layout[B_FLAGS] as *mut u32).add(cap + worker) = DYNAMIC;
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
            let body = *(layout[B_MOVE] as *const u32).add(row * MOVE_STRIDE) as usize;
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
                *pos.add(lane) = *(layout[B_FIN] as *const f32).add(row * FIN_STRIDE + 9 + lane);
                *vel.add(lane) = *(layout[B_STATE] as *const f32).add(row * STATE_STRIDE + lane);
            }
            *pos.add(3) = 0.0;
            *vel.add(3) = 0.0;
            core::ptr::copy_nonoverlapping(
                (layout[B_SIM] as *const f32).add(row * SIM_STRIDE + 28),
                quat,
                4,
            );
            written += 1;
        }
        written
    }
}
#[export_name = "bodyCreate"]
pub extern "C" fn body_create(id: u32) -> u32 {
    regions::select(id);
    unsafe {
        let p = world(id as usize);
        if p.free < 0 && p.next == p.cap {
            reserve_bodies((p.cap * 2).max(16));
        }
        let w = world_mut(id as usize);
        let body = if w.free >= 0 {
            let body = w.free as usize;
            w.free = *(w.columns.layout[B_RECORD_NEXT] as *const u32).add(body) as i32;
            body
        } else {
            let body = w.next;
            w.next += 1;
            body
        };
        let generation = (w.columns.layout[B_RECORD_GENERATION] as *mut u32).add(body);
        *generation = (*generation).wrapping_add(1);
        *(w.columns.layout[B_RECORD_ALIVE] as *mut u32).add(body) = 1;
        *(w.columns.layout[B_RECORD_NEXT] as *mut u32).add(body) = u32::MAX;
        *(w.columns.layout[B_RECORD_EID] as *mut u32).add(body) = u32::MAX;
        w.count += 1;
        body as u32
    }
}
#[export_name = "bodyDestroy"]
pub extern "C" fn body_destroy(id: u32, body: u32) {
    unsafe {
        let w = world_mut(id as usize);
        let body = body as usize;
        if body >= w.next || body >= w.cap {
            return;
        }
        let alive = (w.columns.layout[B_RECORD_ALIVE] as *mut u32).add(body);
        if *alive == 0 {
            return;
        }
        *alive = 0;
        *(w.columns.layout[B_RECORD_NEXT] as *mut u32).add(body) = w.free as u32;
        w.free = body as i32;
        w.count -= 1;
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
        *(w.columns.layout[B_RECORD_GENERATION] as *const u32).add(body as usize)
    }
}
#[export_name = "bodyAlive"]
pub extern "C" fn body_alive(id: u32, body: u32) -> u32 {
    unsafe {
        let w = world(id as usize);
        if body as usize >= w.cap {
            return 0;
        }
        *(w.columns.layout[B_RECORD_ALIVE] as *const u32).add(body as usize)
    }
}
#[export_name = "bodyCount"]
pub extern "C" fn body_count(id: u32) -> usize {
    unsafe { world(id as usize).count }
}
pub fn active_generation(id: u32) -> u32 {
    body_generation(regions::active() as u32, id)
}
pub unsafe fn restore_id(from: usize, to: usize) {
    body_reset_world(to as u32);
    WORLDS[to] = WORLDS[from];
    WORLDS[from] = Bodies::EMPTY;
}
