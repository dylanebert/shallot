//! The persistent shape region — one record per shapeId (type code, local geometry, nextShapeId), held
//! resident across steps so the in-kernel finalize refit (T4) can walk a body's shape list and compute
//! its fat AABBs without a per-step marshal. Keyed by shapeId (id-space, grow-on-`createShape`), a
//! single column, so — like the fat-AABB region it sits above — it needs no record migration: a
//! shapeId's slot is fixed for the shape's life. Wasm-only (the column aliases linear memory); TS owns
//! the growth policy and writes authored geometry/material payloads at shape create (`src/shapecolumns.ts`).
//!
//! Placement — a third *low* persistent region, directly above the fat-AABB region, so it joins the
//! same relocation chain:
//! ```text
//! [0, heap_base)              Rust statics
//! [heap_base, body_end)       persistent body columns (bodies.rs) — first, never shifts
//! [body_end, fataabb_end)     persistent fat-AABB column (fataabb.rs)
//! [fataabb_end, shape_end)    persistent shape column (this module)
//! [shape_end, manifold_end)   persistent manifold columns (manifolds.rs)
//! [manifold_end, geo_end)     static geometry pools (geo.rs)
//! [geo_end, ...)              per-step solver columns (arena.rs)
//! ```
//! Anchoring `region_top` (shape_end, or the fat-AABB region's `region_top` when empty) is what the
//! manifold region sits on, so inserting this region is transparent to everything above it. A body or
//! fat-AABB grow below shifts this whole region up — `relocate` rebases its offset (the caller memmoves
//! the bytes). Its own growth (`createShape` raising the shape high-water — rare, between steps in
//! practice) shifts the manifold + geometry regions above it, relocated in place exactly like a body
//! grow does. The column is a single base-anchored slab keyed by shapeId, so its own contents never
//! move on its own grow (only the regions above it) — no top-down copy, unlike the multi-column body
//! region.

//! Reachability is the read contract: a grow does not zero the new records, and the singleton reuses
//! memory across worlds, so a slot outside a live chain holds whatever the manifold region left there —
//! bytes that reinterpret as *plausible* type codes and geometry. Every slot reachable from an awake
//! body's head lane through `next` is written before it can be read; index the column that way and
//! never by a bare `0..cap` sweep.

use crate::col::Col;
use crate::fataabb::region_top as shape_base;

const PAGE: usize = 65536;

/// 4-byte stride of one shape record, read through both a u32 and an f32 view of the same bytes:
/// type(1, u32) + nextShapeId(1, u32) + geometry payload(7, f32) + finalize-refit output(7). The
/// geometry payload is sized to the widest convex case — a capsule's center1(3) center2(3) radius(1); a
/// sphere uses center(3) radius(1), a hull its local AABB lower(3) upper(3) (the only hull field the
/// AABB path reads, `src/shape.ts` hull branch of `computeShapeAABBOut`). The refit output is the
/// finalize pass's per-shape write-back: the candidate fat AABB (6, f32) + the escaped flag (1, u32).
/// Mesh/height-field/compound records carry only the type code + list link (the kernel skips them; TS
/// computes their AABB at their list position). Attachment lanes 16 and 17 sit outside the refit
/// output and survive every shape write. No other padding — the record is read scalar, one shape at a
/// time.
pub const SHAPE_STRIDE: usize = 18;

/// Record slots. Type codes are the TS `ShapeType` values verbatim (the same codes the narrowphase
/// dispatch already carries — `finalize::TY_SPHERE` etc), so sphere/capsule/hull dispatch and every other
/// value (compound/height-field/mesh) is the TS-fallback partition.
pub const S_TYPE: usize = 0;
pub const S_NEXT: usize = 1;
pub const S_GEOM: usize = 2;
/// Finalize refit output (written per convex shape by `arena::refit_block`, read by TS `finalizeBodies`):
/// the candidate fat AABB `[lower.xyz, upper.xyz]` (6 f32) + the escaped flag (u32, 0/1).
pub const S_CAND: usize = 9;
pub const S_ESCAPED: usize = 15;
/// Kernel-owned material allocation attached to this shape; lanes 16 and 17 are outside finalize.
pub const S_MATERIAL_HEAD: usize = 16;
pub const S_MATERIAL_COUNT: usize = 17;

/// End-of-list sentinel in the `nextShapeId` slot (and in the body record's headShapeId lane): TS's
/// `NULL_INDEX` (-1) written through a u32 view.
pub const NULL_SHAPE: u32 = u32::MAX;

const B_RECORD_GENERATION: usize = 1;
const B_RECORD_ALIVE: usize = 2;
const B_RECORD_NEXT: usize = 3;
const MATERIAL_BASE: usize = 4;
const N_SHAPE: usize = 5;
const MAX_WORLDS: usize = 128;
/// Material records are kernel-owned and world-local. Twelve u32s hold the six f32 fields, a u64
/// application id, color, free-list link, generation and alive bit.
pub const MATERIAL_STRIDE: usize = 12;
const M_NEXT: usize = 9;
const M_GENERATION: usize = 10;
const M_ALIVE: usize = 11;

// Shape-slot lifecycle state is kernel-owned, just like body-slot lifecycle state. The public
// TypeScript array remains an authoring/handle bridge; index, generation, validity and reuse are
// decided here so a stale Shape cannot resolve to a later occupant.
static mut SHAPE_NEXT_INDEX: [usize; MAX_WORLDS] = [0; MAX_WORLDS];
static mut SHAPE_FREE_HEAD: [i32; MAX_WORLDS] = [-1; MAX_WORLDS];
static mut SHAPE_LIVE_COUNT: [usize; MAX_WORLDS] = [0; MAX_WORLDS];

#[inline]
fn record_index(world: usize, id: usize, cap: usize) -> usize {
    world * cap + id
}

/// Single-column data plus world-local lifecycle columns. TS reads this (`shapeLayoutPtr`) to build
/// its views after every grow-triggering `reserveShapes`.
static mut SHAPE_LAYOUT: [u32; N_SHAPE] = [0; N_SHAPE];
/// First free byte past the shape region — where the manifold region anchors. Zero until the first
/// `reserveShapes`; `region_top` then treats it as an empty region ending at the fat-AABB region's top.
static mut SHAPE_END: u32 = 0;
/// The record capacity the current layout was sized to (shape high-water). Grow-only. The single source
/// of truth for the TS view lengths (`src/shapecolumns.ts`).
static mut SHAPE_CAP: usize = 0;
static mut MATERIAL_CAP: usize = 0;
static mut MATERIAL_NEXT_INDEX: [usize; MAX_WORLDS] = [0; MAX_WORLDS];
static mut MATERIAL_FREE_HEAD: [i32; MAX_WORLDS] = [-1; MAX_WORLDS];
/// The world whose body set is currently being stepped. Shape records are world-local slabs.
static mut ACTIVE_WORLD: usize = 0;

#[inline]
fn align16(x: usize) -> usize {
    (x + 15) & !15
}

/// The byte offset the persistent regions above this one anchor from: past the shape region if one was
/// reserved, else the fat-AABB region's top (where this region would begin).
#[inline]
pub fn region_top() -> usize {
    let end = unsafe { SHAPE_END } as usize;
    if end == 0 {
        shape_base()
    } else {
        end
    }
}

/// First free byte past the shape region (0 if none reserved).
#[inline]
pub fn region_end() -> usize {
    unsafe { SHAPE_END as usize }
}

/// The resident shape column as a shared-mutable u32 handle (`SHAPE_STRIDE` slots per shape). The
/// finalize refit reads the type + list link and writes the escaped flag; it runs per body under the
/// parallel-for, and a shape belongs to one body, so the block writes are disjoint per shape record —
/// the [`Col`] carries that promise (col.rs). Empty (len 0) before the first `reserveShapes`. The
/// geometry + candidate slots are f32 bits — read/write them through [`col_f`].
pub fn col() -> Col<'static, u32> {
    unsafe {
        let base = SHAPE_LAYOUT[0] as usize + ACTIVE_WORLD * SHAPE_CAP * SHAPE_STRIDE * 4;
        Col::new(base as *mut u32, SHAPE_CAP * SHAPE_STRIDE)
    }
}

/// The same bytes as [`col`], as a shared-mutable f32 handle — the geometry payload + candidate AABB.
pub fn col_f() -> Col<'static, f32> {
    unsafe {
        let base = SHAPE_LAYOUT[0] as usize + ACTIVE_WORLD * SHAPE_CAP * SHAPE_STRIDE * 4;
        Col::new(base as *mut f32, SHAPE_CAP * SHAPE_STRIDE)
    }
}

/// The resident shape column as a read-only `[u32]` — the broad-phase pair query (`pairwork.rs`) reads
/// each found shape's type code (`S_TYPE`) to partition compound leaves onto the TS fallback path.
pub fn col_slice() -> &'static [u32] {
    unsafe {
        let base = SHAPE_LAYOUT[0] as usize + ACTIVE_WORLD * SHAPE_CAP * SHAPE_STRIDE * 4;
        core::slice::from_raw_parts(base as *const u32, SHAPE_CAP * SHAPE_STRIDE)
    }
}

/// Select the world-local shape and fat-AABB slabs used by the in-kernel step.
#[export_name = "shapeSetActiveWorld"]
pub extern "C" fn shape_set_active_world(world: u32) {
    unsafe {
        ACTIVE_WORLD = (world as usize) % MAX_WORLDS;
    }
    crate::fataabb::set_active_world(world);
}

/// Shift the region's byte offset up by `delta` after a region below it grew and moved it (the caller
/// memmoves the bytes). No-op if none reserved. The column is shapeId-indexed (element offsets absolute
/// within it), so only the header offset + end marker rebase.
pub fn relocate(delta: usize) {
    unsafe {
        if SHAPE_END == 0 {
            return;
        }
        // All shape-region columns move together when the lower fat-AABB region grows. In
        // particular, lifecycle rows must rebase too; leaving them behind makes the next reserve copy
        // stale generation/alive/free-list bytes.
        for slot in 0..N_SHAPE {
            SHAPE_LAYOUT[slot] += delta as u32;
        }
        SHAPE_END += delta as u32;
    }
}

unsafe fn ensure_capacity(end_byte: usize) {
    let have = core::arch::wasm32::memory_size(0) * PAGE;
    if end_byte > have {
        let pages = (end_byte - have + PAGE - 1) / PAGE;
        core::arch::wasm32::memory_grow(0, pages);
    }
}

/// Byte offset of the layout header. TS reads this to build its column views after every grow-triggering
/// `reserveShapes`.
#[export_name = "shapeLayoutPtr"]
pub extern "C" fn shape_layout_ptr() -> *const u32 {
    &raw const SHAPE_LAYOUT as *const u32
}

/// The record capacity the region is sized to — the single source of truth for the TS views' length.
#[export_name = "shapeCap"]
pub extern "C" fn shape_cap() -> usize {
    unsafe { SHAPE_CAP }
}

/// Lay out the shape column for `cap` shapes, growing memory to fit and relocating the manifold +
/// geometry regions above it in place. Base is the fat-AABB region's top; the column is a single
/// base-anchored slab, so growth only pushes the regions above it up (its own bytes stay put).
/// Grow-only: a call with `cap <= SHAPE_CAP` is a no-op returning 0. @returns 1 when the region grew —
/// the caller then refreshes any views over the (relocated + detached) regions above it.
#[export_name = "reserveShapes"]
pub extern "C" fn reserve_shapes(cap: usize) -> u32 {
    unsafe {
        if cap <= SHAPE_CAP {
            return 0;
        }

        let old_cap = SHAPE_CAP;
        let old_layout = SHAPE_LAYOUT;
        // The column sits at the fat-AABB region's top (which doesn't move during this call), so its
        // shape-data base is unchanged. Lifecycle columns widen after it; the manifold + geometry
        // regions above shift up by the complete growth delta.
        let base = align16(shape_base());
        SHAPE_LAYOUT[0] = base as u32;
        let data_end = align16(base + MAX_WORLDS * cap * SHAPE_STRIDE * 4);
        let lifecycle = cap * MAX_WORLDS;
        let mut off = data_end;
        for slot in [B_RECORD_GENERATION, B_RECORD_ALIVE, B_RECORD_NEXT] {
            SHAPE_LAYOUT[slot] = off as u32;
            off += lifecycle * 4;
        }
        SHAPE_LAYOUT[MATERIAL_BASE] = off as u32;
        off += MATERIAL_CAP * MAX_WORLDS * MATERIAL_STRIDE * 4;
        let new_end = align16(off);
        let old_top = region_top(); // where the manifold region currently anchors
        let delta = new_end - old_top;

        let geo_end = crate::geo::region_end();
        let broad_end = crate::broad::region_end();
        let manifold_end = crate::manifolds::region_end();
        let top = if geo_end != 0 {
            geo_end
        } else if broad_end != 0 {
            broad_end
        } else if manifold_end != 0 {
            manifold_end
        } else {
            old_top
        };
        if delta > 0 && top > old_top {
            ensure_capacity(top + delta);
            // `copy` is memmove; dest > src (the region only grows), so the overlap is handled.
            core::ptr::copy(
                old_top as *const u8,
                (old_top + delta) as *mut u8,
                top - old_top,
            );
            crate::manifolds::relocate(delta);
            crate::broad::relocate(delta);
            crate::geo::relocate(delta);
        } else {
            ensure_capacity(new_end);
        }

        // The widened shape data/lifecycle destinations can overlap the old material slab. Preserve
        // materials first, before either destination copy can overwrite their old rows.
        if MATERIAL_CAP > 0 {
            for world in (0..MAX_WORLDS).rev() {
                let old_row =
                    old_layout[MATERIAL_BASE] as usize + world * MATERIAL_CAP * MATERIAL_STRIDE * 4;
                let new_row = SHAPE_LAYOUT[MATERIAL_BASE] as usize
                    + world * MATERIAL_CAP * MATERIAL_STRIDE * 4;
                core::ptr::copy(
                    old_row as *const u8,
                    new_row as *mut u8,
                    MATERIAL_CAP * MATERIAL_STRIDE * 4,
                );
            }
        }

        // Lifecycle columns move from their old post-data offsets into the widened rows. Copy them
        // before the larger data destination can overwrite the old lifecycle source.
        if old_cap > 0 {
            for slot in [B_RECORD_NEXT, B_RECORD_ALIVE, B_RECORD_GENERATION] {
                for world in (0..MAX_WORLDS).rev() {
                    let old_row = old_layout[slot] as usize + world * old_cap * 4;
                    let new_row = SHAPE_LAYOUT[slot] as usize + world * cap * 4;
                    core::ptr::copy(old_row as *const u8, new_row as *mut u8, old_cap * 4);
                }
            }
        }
        // The shape-data destination is copied last because its widened world slabs cover the old
        // lifecycle and material locations.
        if old_cap > 0 {
            for world in (0..MAX_WORLDS).rev() {
                let old_row = old_layout[0] as usize + world * old_cap * SHAPE_STRIDE * 4;
                let new_row = SHAPE_LAYOUT[0] as usize + world * cap * SHAPE_STRIDE * 4;
                core::ptr::copy(
                    old_row as *const u8,
                    new_row as *mut u8,
                    old_cap * SHAPE_STRIDE * 4,
                );
            }
        }
        // Newly exposed slots begin invalid and unlinked. Generations intentionally survive reuse;
        // shape_create increments them when the slot is handed out.
        for world in 0..MAX_WORLDS {
            for id in old_cap..cap {
                let slot = record_index(world, id, cap);
                *(SHAPE_LAYOUT[B_RECORD_GENERATION] as *mut u32).add(slot) = 0;
                *(SHAPE_LAYOUT[B_RECORD_ALIVE] as *mut u32).add(slot) = 0;
                *(SHAPE_LAYOUT[B_RECORD_NEXT] as *mut u32).add(slot) = u32::MAX;
            }
        }

        SHAPE_END = new_end as u32;
        SHAPE_CAP = cap;
        1
    }
}

/// Grow the kernel-owned material record column, preserving every world's live records.
#[export_name = "reserveMaterials"]
pub extern "C" fn reserve_materials(cap: usize) -> u32 {
    unsafe {
        if cap <= MATERIAL_CAP {
            return 0;
        }
        let old_cap = MATERIAL_CAP;
        let old_layout = SHAPE_LAYOUT;
        let base = if SHAPE_CAP == 0 {
            align16(shape_base())
        } else {
            SHAPE_END as usize - old_cap * MAX_WORLDS * MATERIAL_STRIDE * 4
        };
        let new_end = align16(base + cap * MAX_WORLDS * MATERIAL_STRIDE * 4);
        let old_top = region_top();
        let delta = new_end - old_top;
        let geo_end = crate::geo::region_end();
        let broad_end = crate::broad::region_end();
        let manifold_end = crate::manifolds::region_end();
        let top = if geo_end != 0 {
            geo_end
        } else if broad_end != 0 {
            broad_end
        } else if manifold_end != 0 {
            manifold_end
        } else {
            old_top
        };
        if delta > 0 && top > old_top {
            ensure_capacity(top + delta);
            core::ptr::copy(
                old_top as *const u8,
                (old_top + delta) as *mut u8,
                top - old_top,
            );
            crate::manifolds::relocate(delta);
            crate::broad::relocate(delta);
            crate::geo::relocate(delta);
        } else {
            ensure_capacity(new_end);
        }
        SHAPE_LAYOUT[MATERIAL_BASE] = base as u32;
        if old_cap > 0 {
            for world in (0..MAX_WORLDS).rev() {
                let old_row =
                    old_layout[MATERIAL_BASE] as usize + world * old_cap * MATERIAL_STRIDE * 4;
                let new_row =
                    SHAPE_LAYOUT[MATERIAL_BASE] as usize + world * cap * MATERIAL_STRIDE * 4;
                core::ptr::copy(
                    old_row as *const u8,
                    new_row as *mut u8,
                    old_cap * MATERIAL_STRIDE * 4,
                );
            }
        }
        MATERIAL_CAP = cap;
        SHAPE_END = new_end as u32;
        1
    }
}

/// The material layout header and per-world capacity are consumed by the authoring bridge.
#[export_name = "materialLayoutPtr"]
pub extern "C" fn material_layout_ptr() -> *const u32 {
    unsafe { &raw const SHAPE_LAYOUT[MATERIAL_BASE] }
}

#[export_name = "materialCap"]
pub extern "C" fn material_cap() -> usize {
    unsafe { MATERIAL_CAP }
}

#[inline]
fn material_ptr(world: usize, id: usize) -> *mut u32 {
    unsafe {
        (SHAPE_LAYOUT[MATERIAL_BASE] as usize + (world * MATERIAL_CAP + id) * MATERIAL_STRIDE * 4)
            as *mut u32
    }
}

/// Allocate one kernel material record. The TS material arrays are only the authored input/handle
/// bridge; this pool owns live allocation and reuse.
#[export_name = "materialCreate"]
pub extern "C" fn material_create(world: u32) -> u32 {
    unsafe {
        let world = (world as usize) % MAX_WORLDS;
        let next = MATERIAL_NEXT_INDEX[world];
        if MATERIAL_FREE_HEAD[world] < 0 && next == MATERIAL_CAP {
            let mut cap = if MATERIAL_CAP == 0 { 16 } else { MATERIAL_CAP };
            while cap <= next {
                cap *= 2;
            }
            reserve_materials(cap);
        }
        let id = if MATERIAL_FREE_HEAD[world] >= 0 {
            let id = MATERIAL_FREE_HEAD[world] as usize;
            let p = material_ptr(world, id);
            MATERIAL_FREE_HEAD[world] = *p.add(M_NEXT) as i32;
            id
        } else {
            let id = MATERIAL_NEXT_INDEX[world];
            MATERIAL_NEXT_INDEX[world] += 1;
            id
        };
        let p = material_ptr(world, id);
        *p.add(M_GENERATION) = (*p.add(M_GENERATION)).wrapping_add(1);
        *p.add(M_ALIVE) = 1;
        *p.add(M_NEXT) = u32::MAX;
        id as u32
    }
}

#[export_name = "materialDestroy"]
pub extern "C" fn material_destroy(world: u32, id: u32) {
    unsafe {
        let world = (world as usize) % MAX_WORLDS;
        let id = id as usize;
        if id >= MATERIAL_NEXT_INDEX[world] || id >= MATERIAL_CAP {
            return;
        }
        let p = material_ptr(world, id);
        if *p.add(M_ALIVE) == 0 {
            return;
        }
        *p.add(M_ALIVE) = 0;
        *p.add(M_NEXT) = MATERIAL_FREE_HEAD[world] as u32;
        MATERIAL_FREE_HEAD[world] = id as i32;
    }
}

#[export_name = "materialResetWorld"]
pub extern "C" fn material_reset_world(world: u32) {
    unsafe {
        let world = (world as usize) % MAX_WORLDS;
        MATERIAL_NEXT_INDEX[world] = 0;
        MATERIAL_FREE_HEAD[world] = -1;
    }
}

#[export_name = "materialGeneration"]
pub extern "C" fn material_generation(world: u32, id: u32) -> u32 {
    unsafe {
        let world = (world as usize) % MAX_WORLDS;
        if id as usize >= MATERIAL_CAP {
            return 0;
        }
        *material_ptr(world, id as usize).add(M_GENERATION)
    }
}

#[export_name = "materialAlive"]
pub extern "C" fn material_alive(world: u32, id: u32) -> u32 {
    unsafe {
        let world = (world as usize) % MAX_WORLDS;
        if id as usize >= MATERIAL_CAP {
            return 0;
        }
        *material_ptr(world, id as usize).add(M_ALIVE)
    }
}

#[export_name = "shapeMaterialHead"]
pub extern "C" fn shape_material_head(world: u32, id: u32) -> u32 {
    unsafe {
        let world = (world as usize) % MAX_WORLDS;
        if id as usize >= SHAPE_CAP {
            return u32::MAX;
        }
        let p = (SHAPE_LAYOUT[0] as usize + (world * SHAPE_CAP + id as usize) * SHAPE_STRIDE * 4)
            as *const u32;
        *p.add(S_MATERIAL_HEAD)
    }
}

#[export_name = "materialListCount"]
pub extern "C" fn material_list_count(world: u32, head: u32) -> u32 {
    unsafe {
        let world = (world as usize) % MAX_WORLDS;
        let mut id = head;
        let mut count = 0;
        while id != u32::MAX && (id as usize) < MATERIAL_CAP && count <= MATERIAL_CAP as u32 {
            let p = material_ptr(world, id as usize);
            if *p.add(M_ALIVE) == 0 {
                break;
            }
            count += 1;
            id = *p.add(M_NEXT);
        }
        count
    }
}

#[export_name = "shapeMaterialCount"]
pub extern "C" fn shape_material_count(world: u32, id: u32) -> u32 {
    unsafe {
        let world = (world as usize) % MAX_WORLDS;
        if id as usize >= SHAPE_CAP {
            return 0;
        }
        let p = (SHAPE_LAYOUT[0] as usize + (world * SHAPE_CAP + id as usize) * SHAPE_STRIDE * 4)
            as *const u32;
        *p.add(S_MATERIAL_COUNT)
    }
}

/// Allocate a world-local shape slot and advance its generation. The returned index is stable for the
/// shape's lifetime and is reused LIFO after `shapeDestroy`.
#[export_name = "shapeCreate"]
pub extern "C" fn shape_create(world: u32) -> u32 {
    unsafe {
        let world = (world as usize) % MAX_WORLDS;
        let next = SHAPE_NEXT_INDEX[world];
        if SHAPE_FREE_HEAD[world] < 0 && next == SHAPE_CAP {
            let mut cap = if SHAPE_CAP == 0 { 16 } else { SHAPE_CAP };
            while cap <= next {
                cap *= 2;
            }
            reserve_shapes(cap);
        }
        let id = if SHAPE_FREE_HEAD[world] >= 0 {
            let id = SHAPE_FREE_HEAD[world] as usize;
            let slot = record_index(world, id, SHAPE_CAP);
            SHAPE_FREE_HEAD[world] = *(SHAPE_LAYOUT[B_RECORD_NEXT] as *const u32).add(slot) as i32;
            id
        } else {
            let id = SHAPE_NEXT_INDEX[world];
            SHAPE_NEXT_INDEX[world] += 1;
            id
        };
        let slot = record_index(world, id, SHAPE_CAP);
        let generation = (SHAPE_LAYOUT[B_RECORD_GENERATION] as *mut u32).add(slot);
        *generation = (*generation).wrapping_add(1);
        *(SHAPE_LAYOUT[B_RECORD_ALIVE] as *mut u32).add(slot) = 1;
        *(SHAPE_LAYOUT[B_RECORD_NEXT] as *mut u32).add(slot) = u32::MAX;
        SHAPE_LIVE_COUNT[world] += 1;
        id as u32
    }
}

#[export_name = "shapeDestroy"]
pub extern "C" fn shape_destroy(world: u32, id: u32) {
    unsafe {
        let world = (world as usize) % MAX_WORLDS;
        let id = id as usize;
        if id >= SHAPE_NEXT_INDEX[world] || id >= SHAPE_CAP {
            return;
        }
        let slot = record_index(world, id, SHAPE_CAP);
        let alive = (SHAPE_LAYOUT[B_RECORD_ALIVE] as *mut u32).add(slot);
        if *alive == 0 {
            return;
        }
        *alive = 0;
        *(SHAPE_LAYOUT[B_RECORD_NEXT] as *mut u32).add(slot) = SHAPE_FREE_HEAD[world] as u32;
        SHAPE_FREE_HEAD[world] = id as i32;
        SHAPE_LIVE_COUNT[world] -= 1;
    }
}

#[export_name = "shapeResetWorld"]
pub extern "C" fn shape_reset_world(world: u32) {
    unsafe {
        let world = (world as usize) % MAX_WORLDS;
        SHAPE_NEXT_INDEX[world] = 0;
        SHAPE_FREE_HEAD[world] = -1;
        SHAPE_LIVE_COUNT[world] = 0;
    }
}

#[export_name = "shapeGeneration"]
pub extern "C" fn shape_generation(world: u32, id: u32) -> u32 {
    unsafe {
        let world = (world as usize) % MAX_WORLDS;
        if id as usize >= SHAPE_CAP {
            return 0;
        }
        *(SHAPE_LAYOUT[B_RECORD_GENERATION] as *const u32).add(record_index(
            world,
            id as usize,
            SHAPE_CAP,
        ))
    }
}

#[export_name = "shapeAlive"]
pub extern "C" fn shape_alive(world: u32, id: u32) -> u32 {
    unsafe {
        let world = (world as usize) % MAX_WORLDS;
        if id as usize >= SHAPE_CAP {
            return 0;
        }
        *(SHAPE_LAYOUT[B_RECORD_ALIVE] as *const u32).add(record_index(
            world,
            id as usize,
            SHAPE_CAP,
        ))
    }
}

#[export_name = "shapeCount"]
pub extern "C" fn shape_count(world: u32) -> usize {
    unsafe { SHAPE_LIVE_COUNT[(world as usize) % MAX_WORLDS] }
}
