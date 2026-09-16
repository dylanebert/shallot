//! The persistent body-state region — the awake body columns held resident across steps (4a),
//! placed *first* in linear memory so nothing below it ever shifts. Wasm-only (the columns alias
//! linear memory); TS owns the growth policy (`src/bodycolumns.ts`) and, once the columns are
//! consumed (4a.2/4a.3), the offset-backed `BodySim`/`BodyState` views that read them.
//!
//! Layout of linear memory (the body region anchors every persistent region above it):
//! ```text
//! [0, heap_base)              Rust statics (LAYOUT / BODY_LAYOUT / FATAABB_LAYOUT / … headers)
//! [heap_base, body_end)       persistent body columns (this module) — first, so it never shifts
//! [body_end, fataabb_end)     persistent fat-AABB column (fataabb.rs)
//! [fataabb_end, shape_end)    persistent shape column (shapes.rs)
//! [shape_end, manifold_end)   persistent manifold columns (manifolds.rs)
//! [manifold_end, geo_end)     static geometry pools (geo.rs)
//! [geo_end, ...)              per-step solver columns (arena.rs)
//! ```
//! The body region is empty before the first `reserveBodies`, so `persistent_base` returns
//! `heap_base` and the regions above behave exactly as they did before it existed. Its own growth
//! (`createBody` raising the total-body high-water — rare, and between steps in practice) shifts the
//! fat-AABB + shape + manifold + geometry regions above it, so a grow relocates them in place: all shift
//! by the same delta, so one overlapping memmove of the whole `[body_end, geo_end)` span moves them
//! together, then their statics rebase by the delta. Their contents are element-relative, so nothing
//! inside the moved span needs fixing.

use crate::body::flags::DYNAMIC;
use crate::body::{FIN_OUT_STRIDE, FIN_STRIDE, SIM2_STRIDE, SIM_STRIDE, STATE_STRIDE};
use crate::geo::heap_base;

const PAGE: usize = 65536;

/// Identity records the region reserves past its `cap` real bodies — **one per thread**, not one.
///
/// The wide gather remaps every null/static lane onto an identity record, and its whole-vector scatter
/// writes that record back (`contact_wide::scatter_t`). Under the staged solver two blocks running on
/// two threads can each hold a record with a static lane, so a single shared identity record would be a
/// same-address concurrent write — benign by value (a null lane always round-trips zeros) but a data
/// race, and the only element of the state column that is not write-disjoint. Each worker writes its
/// own record instead (`ident_rec(worker)` = `cap + worker`).
///
/// 8 = the thread ceiling the shadow stack affords (main + 7 workers; `src/pool.ts` `maxWorkers`).
pub const IDENT_RECORDS: usize = 8;

/// f32 stride of the flags sidecar (one u32 per body — `BodyState.flags`).
const FLAGS_STRIDE: usize = 1;

// BODY_LAYOUT indices (byte offsets into linear memory), in memory order.
const B_STATE: usize = 0;
const B_SIM: usize = 1;
const B_FIN: usize = 2;
const B_FIN_OUT: usize = 3;
const B_FLAGS: usize = 4;
const B_SIM2: usize = 5;
// Lifecycle records are keyed by public body index, not awake-set index. Move records are keyed by
// the awake-set index and are the retained bridge consumed after finalize.
const B_RECORD_GENERATION: usize = 6;
const B_RECORD_ALIVE: usize = 7;
const B_RECORD_NEXT: usize = 8;
const B_MOVE: usize = 9;
const N_BODY: usize = 10;

pub const MOVE_STRIDE: usize = 3;
const MAX_WORLDS: usize = 128;

/// Public body-slot lifecycle state. These records live beside the solver columns in wasm memory;
/// TypeScript retains only the authoring/handle bridge and never owns index, generation, validity or
/// count decisions. The small world metadata arrays name the free-list heads and high-water marks;
/// the records themselves remain in the kernel's linear memory.
static mut BODY_NEXT_INDEX: [usize; MAX_WORLDS] = [0; MAX_WORLDS];
static mut BODY_FREE_HEAD: [i32; MAX_WORLDS] = [-1; MAX_WORLDS];
static mut BODY_LIVE_COUNT: [usize; MAX_WORLDS] = [0; MAX_WORLDS];
static mut ACTIVE_WORLD: usize = 0;

#[inline]
fn record_index(world: usize, id: usize, cap: usize) -> usize {
    world * cap + id
}

#[inline]
fn record_slots(cap: usize) -> usize {
    cap * MAX_WORLDS
}

/// Per-column byte offsets into linear memory, rewritten by every grow-triggering `reserveBodies`.
/// TS reads this header (`bodyLayoutPtr`) to build its column views (4a.2 on).
static mut BODY_LAYOUT: [u32; N_BODY] = [0; N_BODY];
/// First free byte past the body region — where the fat-AABB region anchors. Zero until the first
/// `reserveBodies`; `persistent_base` treats that as an empty region ending at `heap_base`.
static mut BODY_END: u32 = 0;
/// The record capacity the current layout was sized to. Grow-only: the region tracks the total-body
/// high-water, so it never shrinks with the churny awake set (a mid-step wake can't outgrow it).
static mut BODY_CAP: usize = 0;

#[inline]
fn align16(x: usize) -> usize {
    (x + 15) & !15
}

/// The byte offset the persistent regions above the body region anchor from: past the body region if
/// one was reserved, else the bare heap base.
#[inline]
pub fn persistent_base() -> usize {
    let end = unsafe { BODY_END } as usize;
    if end == 0 {
        heap_base()
    } else {
        end
    }
}

/// Byte offset of the resident `state` column (4a.2): the solver phases read/write awake body
/// velocity/delta state here instead of a marshaled per-step scratch column, so `arena::reserve`
/// points `LAYOUT[STATE]` at this base. Valid once `reserveBodies` has laid out the region.
#[inline]
pub fn state_base() -> usize {
    unsafe { BODY_LAYOUT[B_STATE] as usize }
}

/// Byte offset of the resident `flags` column (4a.2), the `u32` sidecar paired with `state`.
#[inline]
pub fn flags_base() -> usize {
    unsafe { BODY_LAYOUT[B_FLAGS] as usize }
}

/// Byte offset of the resident `sim` column (4a.3): the integrate/finalize sim fields, held resident
/// so the per-step marshal dies (the awake `BodySim` becomes an offset-backed view over this column).
/// `arena::reserve` points `LAYOUT[SIM]` here.
#[inline]
pub fn sim_base() -> usize {
    unsafe { BODY_LAYOUT[B_SIM] as usize }
}

/// Byte offset of the resident `fin` column (4a.3): the pose-finalize geometric fields (center,
/// localCenter, maxExtent, transform.p). `arena::reserve` points `LAYOUT[FIN]` here.
#[inline]
pub fn fin_base() -> usize {
    unsafe { BODY_LAYOUT[B_FIN] as usize }
}

/// Byte offset of the resident `finOut` column (4a.3): the two per-step sleep/continuous decision
/// scalars finalize emits. Transient (recomputed each step), but it lives in the body region — the
/// region already reserves it — so the arena's per-step scratch drops it. `arena::reserve` points
/// `LAYOUT[FIN_OUT]` here.
#[inline]
pub fn fin_out_base() -> usize {
    unsafe { BODY_LAYOUT[B_FIN_OUT] as usize }
}

/// Byte offset of the resident `sim2` column: the `BodySim` fields the solver never gathers, plus the
/// headShapeId lane the finalize refit walks the body's shape list from (`body::S2_HEAD_SHAPE`).
#[inline]
pub fn sim2_base() -> usize {
    unsafe { BODY_LAYOUT[B_SIM2] as usize }
}

/// Byte offset of the generation column keyed by public body index.
#[inline]
pub fn record_generation_base() -> usize {
    unsafe { BODY_LAYOUT[B_RECORD_GENERATION] as usize }
}

/// Byte offset of the alive column keyed by public body index.
#[inline]
pub fn record_alive_base() -> usize {
    unsafe { BODY_LAYOUT[B_RECORD_ALIVE] as usize }
}

/// Byte offset of the free-list next column keyed by public body index.
#[inline]
pub fn record_next_base() -> usize {
    unsafe { BODY_LAYOUT[B_RECORD_NEXT] as usize }
}

/// Byte offset of the retained body-move bridge. Each record is bodyId, generation, fellAsleep.
#[inline]
pub fn move_base() -> usize {
    unsafe { BODY_LAYOUT[B_MOVE] as usize }
}

/// The record capacity the resident region is currently sized to — the single source of truth for the
/// TS body-store's column-view lengths (`src/bodycolumns.ts`). Zero before the first `reserveBodies`.
#[export_name = "bodyCap"]
pub extern "C" fn body_cap() -> usize {
    unsafe { BODY_CAP }
}

unsafe fn ensure_capacity(end_byte: usize) {
    let have = core::arch::wasm32::memory_size(0) * PAGE;
    if end_byte > have {
        let pages = (end_byte - have + PAGE - 1) / PAGE;
        core::arch::wasm32::memory_grow(0, pages);
    }
}

/// Byte offset of the body layout header (`[u32; N_BODY]` of per-column byte offsets). TS reads this
/// to build its column views after every grow-triggering `reserveBodies`.
#[export_name = "bodyLayoutPtr"]
pub extern "C" fn body_layout_ptr() -> *const u32 {
    &raw const BODY_LAYOUT as *const u32
}

/// Lay out the persistent body columns for `cap` bodies (plus [`IDENT_RECORDS`] trailing null-lane
/// records for 4c's gather remap, one per thread), growing memory to fit and relocating every persistent
/// region above it (fat-AABB, shape, manifold, geometry). Each
/// column base is 16-aligned (STATE's records are the v128-aligned targets of 4c's record gather; the
/// rest are padded up). Grow-only: a call with `cap <= BODY_CAP` is a no-op returning 0. @returns 1
/// when the region grew — the caller then refreshes views over the (possibly relocated + detached)
/// regions above it, so `BODY_CAP` is the single source of truth for the region's size.
#[export_name = "reserveBodies"]
pub extern "C" fn reserve_bodies(cap: usize) -> u32 {
    unsafe {
        if cap <= BODY_CAP {
            return 0;
        }

        // Capture the current layout + capacity before recomputing — the live columns are copied from
        // these old offsets into the new (wider-spaced) ones below.
        let old_cap = BODY_CAP;
        let old_layout = BODY_LAYOUT;

        let records = cap + IDENT_RECORDS; // trailing null-lane records, one per thread
        let lifecycle = record_slots(cap);
        let old_lifecycle = record_slots(old_cap);
        let old_base = persistent_base(); // where the fat-AABB region currently anchors

        let base = align16(heap_base());
        let mut off = base;
        BODY_LAYOUT[B_STATE] = off as u32;
        off += records * STATE_STRIDE * 4;
        BODY_LAYOUT[B_SIM] = off as u32;
        off += records * SIM_STRIDE * 4;
        BODY_LAYOUT[B_FIN] = off as u32;
        off += records * FIN_STRIDE * 4;
        BODY_LAYOUT[B_FIN_OUT] = off as u32;
        off += records * FIN_OUT_STRIDE * 4;
        off = align16(off);
        BODY_LAYOUT[B_FLAGS] = off as u32;
        off += records * FLAGS_STRIDE * 4;
        off = align16(off);
        BODY_LAYOUT[B_SIM2] = off as u32;
        off += records * SIM2_STRIDE * 4;
        off = align16(off);
        BODY_LAYOUT[B_RECORD_GENERATION] = off as u32;
        off += lifecycle * 4;
        BODY_LAYOUT[B_RECORD_ALIVE] = off as u32;
        off += lifecycle * 4;
        BODY_LAYOUT[B_RECORD_NEXT] = off as u32;
        off += lifecycle * 4;
        off = align16(off);
        BODY_LAYOUT[B_MOVE] = off as u32;
        off += cap * MOVE_STRIDE * 4;
        let new_end = align16(off);

        // Relocate the live persistent data above the body region up by the growth delta. The fat-AABB,
        // shape, manifold and geometry regions all shift by the same amount, so one overlapping memmove
        // of the whole `[old_base, top)` span moves them together; their statics rebase by the delta.
        let delta = new_end - old_base;
        let geo_end = crate::geo::region_end();
        let broad_end = crate::broad::region_end();
        let manifold_end = crate::manifolds::region_end();
        let shape_end = crate::shapes::region_end();
        let fataabb_end = crate::fataabb::region_end();
        let top = if geo_end != 0 {
            geo_end
        } else if broad_end != 0 {
            broad_end
        } else if manifold_end != 0 {
            manifold_end
        } else if shape_end != 0 {
            shape_end
        } else if fataabb_end != 0 {
            fataabb_end
        } else {
            old_base
        };
        if delta > 0 && top > old_base {
            ensure_capacity(top + delta);
            // `copy` is memmove; dest > src (the region only grows), so the overlap is handled.
            core::ptr::copy(
                old_base as *const u8,
                (old_base + delta) as *mut u8,
                top - old_base,
            );
            crate::fataabb::relocate(delta);
            crate::shapes::relocate(delta);
            crate::manifolds::relocate(delta);
            crate::broad::relocate(delta);
            crate::geo::relocate(delta);
        } else {
            ensure_capacity(new_end);
        }

        // Preserve the body region's own columns. STATE stays at `base` (its offset is
        // capacity-independent), but every column above it shifts up as `records` grows — their live
        // contents (the awake bodies' resident state/flags/sim…) must move too, or the awake set reads
        // garbage after a grow (e.g. the dropped DYNAMIC flag stops the solver writing velocity back).
        // The manifold/geo relocation above vacated `[old_base, new_end)`, so the body region owns
        // `[base, new_end)`; copy each column top-down (highest offset first) so a write never lands on a
        // lower column's not-yet-copied old bytes. STATE is skipped — its bytes are already in place.
        if old_cap > 0 {
            let strides = [
                STATE_STRIDE,
                SIM_STRIDE,
                FIN_STRIDE,
                FIN_OUT_STRIDE,
                FLAGS_STRIDE,
                SIM2_STRIDE,
            ];
            for c in (B_SIM..=B_SIM2).rev() {
                let bytes = old_cap * strides[c] * 4;
                core::ptr::copy(old_layout[c] as *const u8, BODY_LAYOUT[c] as *mut u8, bytes);
            }
            for c in B_RECORD_GENERATION..=B_RECORD_NEXT {
                let bytes = old_lifecycle * 4;
                core::ptr::copy(old_layout[c] as *const u8, BODY_LAYOUT[c] as *mut u8, bytes);
            }
            core::ptr::copy(
                old_layout[B_MOVE] as *const u8,
                BODY_LAYOUT[B_MOVE] as *mut u8,
                old_cap * MOVE_STRIDE * 4,
            );
        }

        // Newly exposed public body slots start empty and point nowhere in the lifecycle record.
        if cap > old_cap {
            let generation = BODY_LAYOUT[B_RECORD_GENERATION] as *mut u32;
            let alive = BODY_LAYOUT[B_RECORD_ALIVE] as *mut u32;
            let next = BODY_LAYOUT[B_RECORD_NEXT] as *mut u32;
            for world in 0..MAX_WORLDS {
                for i in old_cap..cap {
                    let i = record_index(world, i, cap);
                    *generation.add(i) = 0;
                    *alive.add(i) = 0;
                    *next.add(i) = u32::MAX;
                }
            }
            let moves = BODY_LAYOUT[B_MOVE] as *mut u32;
            for i in old_cap..cap {
                for lane in 0..MOVE_STRIDE {
                    *moves.add(i * MOVE_STRIDE + lane) = 0;
                }
            }
        }

        // Initialise the trailing null-lane records (`cap + w`, one per thread) to identity: zero
        // velocity/delta, identity delta-rotation (dq.s=1), DYNAMIC flag. 4c's wide gather remaps every
        // null/static lane onto the running worker's record (`contact_wide::rec_of`) and its
        // whole-vector scatter writes it with the null lane's zero velocity — DYNAMIC lets that fast
        // path fire, and the zero write leaves the record identity for the next gather. Re-init on every
        // grow: the records move with `cap`, and the grow/memmove above may leave stale bytes there.
        for w in 0..IDENT_RECORDS {
            let state_null = (BODY_LAYOUT[B_STATE] as *mut f32).add((cap + w) * STATE_STRIDE);
            for k in 0..STATE_STRIDE {
                *state_null.add(k) = 0.0;
            }
            *state_null.add(12) = 1.0; // delta_rotation.s
            *(BODY_LAYOUT[B_FLAGS] as *mut u32).add(cap + w) = DYNAMIC;
        }

        BODY_END = new_end as u32;
        BODY_CAP = cap;
        1
    }
}

/// Select the world whose resident solver step is about to run. Finalization uses this context to
/// stamp move records with the public body generation without a TypeScript per-body publication pass.
#[export_name = "bodySetActiveWorld"]
pub extern "C" fn body_set_active_world(world: u32) {
    unsafe {
        ACTIVE_WORLD = (world as usize) % MAX_WORLDS;
    }
}

/// Allocate a body index from the kernel-owned world-local LIFO free list and advance its generation.
#[export_name = "bodyCreate"]
pub extern "C" fn body_create(world: u32) -> u32 {
    unsafe {
        let world = (world as usize) % MAX_WORLDS;
        let next = BODY_NEXT_INDEX[world];
        if BODY_FREE_HEAD[world] < 0 && next == BODY_CAP {
            let mut cap = if BODY_CAP == 0 { 16 } else { BODY_CAP };
            while cap <= next {
                cap *= 2;
            }
            reserve_bodies(cap);
        }
        let id = if BODY_FREE_HEAD[world] >= 0 {
            let id = BODY_FREE_HEAD[world] as usize;
            let slot = record_index(world, id, BODY_CAP);
            BODY_FREE_HEAD[world] = *(BODY_LAYOUT[B_RECORD_NEXT] as *const u32).add(slot) as i32;
            id
        } else {
            let id = BODY_NEXT_INDEX[world];
            BODY_NEXT_INDEX[world] += 1;
            id
        };
        let slot = record_index(world, id, BODY_CAP);
        let generation = (BODY_LAYOUT[B_RECORD_GENERATION] as *mut u32).add(slot);
        *generation = (*generation).wrapping_add(1);
        *(BODY_LAYOUT[B_RECORD_ALIVE] as *mut u32).add(slot) = 1;
        *(BODY_LAYOUT[B_RECORD_NEXT] as *mut u32).add(slot) = u32::MAX;
        BODY_LIVE_COUNT[world] += 1;
        id as u32
    }
}

/// Destroy a public body slot and return it to the kernel-owned world-local free list.
#[export_name = "bodyDestroy"]
pub extern "C" fn body_destroy(world: u32, id: u32) {
    unsafe {
        let world = (world as usize) % MAX_WORLDS;
        let id = id as usize;
        if id >= BODY_NEXT_INDEX[world] || id >= BODY_CAP {
            return;
        }
        let slot = record_index(world, id, BODY_CAP);
        let alive = (BODY_LAYOUT[B_RECORD_ALIVE] as *mut u32).add(slot);
        if *alive == 0 {
            return;
        }
        *alive = 0;
        *(BODY_LAYOUT[B_RECORD_NEXT] as *mut u32).add(slot) = BODY_FREE_HEAD[world] as u32;
        BODY_FREE_HEAD[world] = id as i32;
        BODY_LIVE_COUNT[world] -= 1;
    }
}

/// Clear all slots for a world being destroyed. The world generation already invalidates its public
/// handles, so the next incarnation starts with a clean kernel pool.
#[export_name = "bodyResetWorld"]
pub extern "C" fn body_reset_world(world: u32) {
    unsafe {
        let world = (world as usize) % MAX_WORLDS;
        if BODY_CAP != 0 {
            let generation = BODY_LAYOUT[B_RECORD_GENERATION] as *mut u32;
            let alive = BODY_LAYOUT[B_RECORD_ALIVE] as *mut u32;
            let next = BODY_LAYOUT[B_RECORD_NEXT] as *mut u32;
            for id in 0..BODY_NEXT_INDEX[world] {
                let slot = record_index(world, id, BODY_CAP);
                *generation.add(slot) = 0;
                *alive.add(slot) = 0;
                *next.add(slot) = u32::MAX;
            }
        }
        BODY_NEXT_INDEX[world] = 0;
        BODY_FREE_HEAD[world] = -1;
        BODY_LIVE_COUNT[world] = 0;
    }
}

#[export_name = "bodyGeneration"]
pub extern "C" fn body_generation(world: u32, id: u32) -> u32 {
    unsafe {
        let world = (world as usize) % MAX_WORLDS;
        let id = id as usize;
        if id >= BODY_CAP {
            return 0;
        }
        *(BODY_LAYOUT[B_RECORD_GENERATION] as *const u32).add(record_index(world, id, BODY_CAP))
    }
}

#[export_name = "bodyAlive"]
pub extern "C" fn body_alive(world: u32, id: u32) -> u32 {
    unsafe {
        let world = (world as usize) % MAX_WORLDS;
        let id = id as usize;
        if id >= BODY_CAP {
            return 0;
        }
        *(BODY_LAYOUT[B_RECORD_ALIVE] as *const u32).add(record_index(world, id, BODY_CAP))
    }
}

#[export_name = "bodyCount"]
pub extern "C" fn body_count(world: u32) -> usize {
    unsafe { BODY_LIVE_COUNT[(world as usize) % MAX_WORLDS] }
}

/// Read the active world's generation for kernel finalization's move publication.
#[inline]
pub fn active_generation(id: u32) -> u32 {
    unsafe {
        let id = id as usize;
        if id >= BODY_CAP {
            return 0;
        }
        *(BODY_LAYOUT[B_RECORD_GENERATION] as *const u32).add(record_index(
            ACTIVE_WORLD,
            id,
            BODY_CAP,
        ))
    }
}
