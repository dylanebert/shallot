//! The persistent shape region — one record per shapeId (type code, local geometry, nextShapeId), held
//! resident across steps so the in-kernel finalize refit (T4) can walk a body's shape list and compute
//! its fat AABBs without a per-step marshal. Keyed by shapeId (id-space, grow-on-`createShape`), a
//! single column, so — like the fat-AABB region it sits above — it needs no record migration: a
//! shapeId's slot is fixed for the shape's life. Wasm-only (the column aliases linear memory); TS owns
//! the growth policy and writes the records at shape create/destroy (`src/shapecolumns.ts`).
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
/// computes their AABB at their list position). 16 slots = one 64-byte cache line per shape: the refit
/// reads type/geom and writes candidate/escaped in one line. No other padding — the record is read
/// scalar, one shape at a time.
pub const SHAPE_STRIDE: usize = 16;

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

/// End-of-list sentinel in the `nextShapeId` slot (and in the body record's headShapeId lane): TS's
/// `NULL_INDEX` (-1) written through a u32 view.
pub const NULL_SHAPE: u32 = u32::MAX;

/// Single-column layout header (byte offset of the shape column). TS reads this (`shapeLayoutPtr`) to
/// build its views after every grow-triggering `reserveShapes`.
static mut SHAPE_LAYOUT: [u32; 1] = [0; 1];
/// First free byte past the shape region — where the manifold region anchors. Zero until the first
/// `reserveShapes`; `region_top` then treats it as an empty region ending at the fat-AABB region's top.
static mut SHAPE_END: u32 = 0;
/// The record capacity the current layout was sized to (shape high-water). Grow-only. The single source
/// of truth for the TS view lengths (`src/shapecolumns.ts`).
static mut SHAPE_CAP: usize = 0;

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
    unsafe { Col::new(SHAPE_LAYOUT[0] as *mut u32, SHAPE_CAP * SHAPE_STRIDE) }
}

/// The same bytes as [`col`], as a shared-mutable f32 handle — the geometry payload + candidate AABB.
pub fn col_f() -> Col<'static, f32> {
    unsafe { Col::new(SHAPE_LAYOUT[0] as *mut f32, SHAPE_CAP * SHAPE_STRIDE) }
}

/// The resident shape column as a read-only `[u32]` — the broad-phase pair query (`pairwork.rs`) reads
/// each found shape's type code (`S_TYPE`) to partition compound leaves onto the TS fallback path.
pub fn col_slice() -> &'static [u32] {
    unsafe { core::slice::from_raw_parts(SHAPE_LAYOUT[0] as *const u32, SHAPE_CAP * SHAPE_STRIDE) }
}

/// Shift the region's byte offset up by `delta` after a region below it grew and moved it (the caller
/// memmoves the bytes). No-op if none reserved. The column is shapeId-indexed (element offsets absolute
/// within it), so only the header offset + end marker rebase.
pub fn relocate(delta: usize) {
    unsafe {
        if SHAPE_END == 0 {
            return;
        }
        SHAPE_LAYOUT[0] += delta as u32;
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

        // The column sits at the fat-AABB region's top (which doesn't move during this call), so its
        // base is unchanged and its live bytes stay in place; only the region grows upward. The manifold
        // + geometry regions above shift up by the growth delta.
        let base = align16(shape_base());
        let new_end = align16(base + cap * SHAPE_STRIDE * 4);
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
            core::ptr::copy(old_top as *const u8, (old_top + delta) as *mut u8, top - old_top);
            crate::manifolds::relocate(delta);
            crate::broad::relocate(delta);
            crate::geo::relocate(delta);
        } else {
            ensure_capacity(new_end);
        }

        SHAPE_LAYOUT[0] = base as u32;
        SHAPE_END = new_end as u32;
        SHAPE_CAP = cap;
        1
    }
}
