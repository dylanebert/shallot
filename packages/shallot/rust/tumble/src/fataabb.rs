//! The persistent fat-AABB region — one enlarged broad-phase AABB per shape, held resident across
//! steps so the in-kernel recycle loop (4b) can test contact overlap without a per-step marshal.
//! Keyed by shapeId (id-space, grow-on-`createShape`), a single column, so unlike the body region it
//! needs no record migration — a shapeId's slot is fixed for the shape's life. Wasm-only (the column
//! aliases linear memory); TS owns the growth policy and writes the AABBs at refit time
//! (`src/fataabbcolumns.ts`).
//!
//! Placement — a second *low* persistent region, directly above the body region (bodies.rs), so it
//! joins the same relocation chain:
//! ```text
//! [0, heap_base)              Rust statics
//! [heap_base, body_end)       persistent body columns (bodies.rs) — first, never shifts
//! [body_end, fataabb_end)     persistent fat-AABB column (this module)
//! [fataabb_end, shape_end)    persistent shape column (shapes.rs)
//! [shape_end, manifold_end)   persistent manifold columns (manifolds.rs)
//! [manifold_end, geo_end)     static geometry pools (geo.rs)
//! [geo_end, ...)              per-step solver columns (arena.rs)
//! ```
//! Anchoring `region_top` (fataabb_end, or the body region's `persistent_base` when empty) is what the
//! shape region sits on, so inserting this region is transparent to everything above it. A body
//! region grow below shifts this whole region up — `relocate` rebases its offset (the caller memmoves
//! the bytes). Its own growth (`createShape` raising the shape high-water — rare, between steps in
//! practice) shifts the shape + manifold + geometry regions above it, relocated in place exactly like a
//! body grow does. The column is a single base-anchored slab keyed by shapeId, so its own contents never
//! move on its own grow (only the regions above it) — no top-down copy, unlike the multi-column body
//! region.

use crate::bodies::persistent_base;

const PAGE: usize = 65536;
/// f32 stride of one shape's fat AABB: lowerBound.xyz + upperBound.xyz (mirrors `src/math.ts` AABB).
pub const AABB_STRIDE: usize = 6;

/// Single-column layout header (byte offset of the fat-AABB column). TS reads this (`fatAabbLayoutPtr`)
/// to build its view after every grow-triggering `reserveFatAabb`.
static mut FATAABB_LAYOUT: [u32; 1] = [0; 1];
/// First free byte past the fat-AABB region — where the manifold region anchors. Zero until the first
/// `reserveFatAabb`; `region_top` then treats it as an empty region ending at `persistent_base`.
static mut FATAABB_END: u32 = 0;
/// The record capacity the current layout was sized to (shape high-water). Grow-only. The single source
/// of truth for the TS view length (`src/fataabbcolumns.ts`).
static mut FATAABB_CAP: usize = 0;

#[inline]
fn align16(x: usize) -> usize {
    (x + 15) & !15
}

/// The byte offset the shape region (and, through it, everything above) anchors from: past the fat-AABB
/// region if one was reserved, else the body region's `persistent_base` (where this region would begin).
#[inline]
pub fn region_top() -> usize {
    let end = unsafe { FATAABB_END } as usize;
    if end == 0 {
        persistent_base()
    } else {
        end
    }
}

/// First free byte past the fat-AABB region (0 if none reserved).
#[inline]
pub fn region_end() -> usize {
    unsafe { FATAABB_END as usize }
}

/// The resident fat-AABB column (`AABB_STRIDE` f32 per shape, `[lower.xyz, upper.xyz]` keyed by
/// shapeId). Empty before the first `reserveFatAabb`. Read by the in-kernel recycle overlap test
/// (arena.rs, 4b.3c); TS writes it at refit time (`src/fataabbcolumns.ts`).
pub fn col_slice() -> &'static [f32] {
    unsafe { core::slice::from_raw_parts(FATAABB_LAYOUT[0] as *const f32, FATAABB_CAP * AABB_STRIDE) }
}

/// Shift the region's byte offset up by `delta` after the body region below it grew and moved it (the
/// caller memmoves the bytes). No-op if none reserved. The column is shapeId-indexed (element offsets
/// absolute within it), so only the header offset + end marker rebase.
pub fn relocate(delta: usize) {
    unsafe {
        if FATAABB_END == 0 {
            return;
        }
        FATAABB_LAYOUT[0] += delta as u32;
        FATAABB_END += delta as u32;
    }
}

unsafe fn ensure_capacity(end_byte: usize) {
    let have = core::arch::wasm32::memory_size(0) * PAGE;
    if end_byte > have {
        let pages = (end_byte - have + PAGE - 1) / PAGE;
        core::arch::wasm32::memory_grow(0, pages);
    }
}

/// Byte offset of the layout header. TS reads this to build its column view after every grow-triggering
/// `reserveFatAabb`.
#[export_name = "fatAabbLayoutPtr"]
pub extern "C" fn fat_aabb_layout_ptr() -> *const u32 {
    &raw const FATAABB_LAYOUT as *const u32
}

/// The record capacity the region is sized to — the single source of truth for the TS view's length.
#[export_name = "fatAabbCap"]
pub extern "C" fn fat_aabb_cap() -> usize {
    unsafe { FATAABB_CAP }
}

/// Lay out the fat-AABB column for `cap` shapes, growing memory to fit and relocating the shape +
/// manifold + geometry regions above it in place. Base is `persistent_base` (past the body region); the
/// column is a single base-anchored slab, so growth only pushes the regions above it up (its own bytes
/// stay put). Grow-only: a call with `cap <= FATAABB_CAP` is a no-op returning 0. @returns 1 when the
/// region grew — the caller then refreshes any views over the (relocated + detached) regions above it.
#[export_name = "reserveFatAabb"]
pub extern "C" fn reserve_fat_aabb(cap: usize) -> u32 {
    unsafe {
        if cap <= FATAABB_CAP {
            return 0;
        }

        // The column sits at `persistent_base` (the body region doesn't move during this call), so its
        // base is unchanged and its live bytes stay in place; only the region grows upward. The shape +
        // manifold + geometry regions above shift up by the growth delta.
        let base = align16(persistent_base());
        let new_end = align16(base + cap * AABB_STRIDE * 4);
        let old_top = region_top(); // where the shape region currently anchors
        let delta = new_end - old_top;

        let geo_end = crate::geo::region_end();
        let broad_end = crate::broad::region_end();
        let manifold_end = crate::manifolds::region_end();
        let shape_end = crate::shapes::region_end();
        let top = if geo_end != 0 {
            geo_end
        } else if broad_end != 0 {
            broad_end
        } else if manifold_end != 0 {
            manifold_end
        } else if shape_end != 0 {
            shape_end
        } else {
            old_top
        };
        if delta > 0 && top > old_top {
            ensure_capacity(top + delta);
            // `copy` is memmove; dest > src (the region only grows), so the overlap is handled.
            core::ptr::copy(old_top as *const u8, (old_top + delta) as *mut u8, top - old_top);
            crate::shapes::relocate(delta);
            crate::manifolds::relocate(delta);
            crate::broad::relocate(delta);
            crate::geo::relocate(delta);
        } else {
            ensure_capacity(new_end);
        }

        FATAABB_LAYOUT[0] = base as u32;
        FATAABB_END = new_end as u32;
        FATAABB_CAP = cap;
        1
    }
}
