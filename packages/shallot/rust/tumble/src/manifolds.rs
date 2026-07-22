//! The persistent contact-manifold columns — the warm-start state that survives across steps,
//! keyed by contactId. Wasm-only (the columns alias linear memory); TS owns the allocator + lifecycle
//! (`src/manifoldstore.ts`) and both the TS mesh narrowphase and the kernel convex narrowphase (3c.3)
//! read/write manifold data at TS-assigned offsets. Ported from box3d's `b3Contact.manifolds` heap
//! array (`contact.h`): a per-contact block of `manifoldCount` manifolds (1 convex / N mesh clusters),
//! each a fixed `b3Manifold` (268 B = 67 f32) with `points[4]` inline.
//!
//! Two columns:
//!   - **directory** — one record per contactId (material row + block descriptor). Indexed directly
//!     by contactId; sized to contact capacity.
//!   - **pool** — the variable manifold records. Each contact owns a contiguous run of `manifoldCount`
//!     records at `manifoldBase`; TS's size-class free lists hand out the runs.
//!
//! Layout of linear memory (the manifold region anchors from `region_top` — past the body, fat-AABB and
//! shape regions below it — so its block offsets move only when a region below it grows):
//! ```text
//! [0, heap_base)              Rust statics (LAYOUT / BODY_LAYOUT / FATAABB_LAYOUT / … headers)
//! [heap_base, body_end)       persistent body columns (bodies.rs)
//! [body_end, fataabb_end)     persistent fat-AABB column (fataabb.rs)
//! [fataabb_end, shape_end)    persistent shape column (shapes.rs)
//! [shape_end, manifold_end)   persistent manifold columns (directory + pool)
//! [manifold_end, geo_end)     static geometry pools (geo.rs)
//! [geo_end, ...)              per-step solver columns (arena.rs)
//! ```
//! Growing the manifold region moves only the geo region after it, which is a full TS re-upload
//! anyway; the directory base is fixed at the anchor below and the pool is memmoved in place on a
//! directory grow, so a live contact's manifold block keeps its element offset across a grow. A body
//! or fat-AABB region grow *below* shifts this whole region up — `relocate` rebases its offsets (the
//! caller memmoves the bytes).

use crate::col::Col;
use crate::shapes::region_top as manifold_anchor;
use crate::manifold_abi::{DIR_STRIDE, MANIFOLD_STRIDE};

const PAGE: usize = 65536;

// MANIFOLD_LAYOUT indices (byte offsets into linear memory), in memory order.
const DIR: usize = 0;
const POOL: usize = 1;
const N_MANIFOLD: usize = 2;

static mut MANIFOLD_LAYOUT: [u32; N_MANIFOLD] = [0; N_MANIFOLD];
/// First free byte past the manifold region — where the geometry region begins. Zero until the first
/// `reserveManifolds`; `geo_base` treats that as an empty region ending at the anchor below it.
static mut MANIFOLD_END: u32 = 0;
/// The pool capacity the current layout was sized to (manifold records). Retained so a grow knows how
/// many bytes of live pool data to preserve when the directory grows and shifts the pool base.
static mut MANIFOLD_CAP: usize = 0;
/// The directory capacity the current layout was sized to (contactId records). Sizes the `dir` column
/// the solver gathers through.
static mut DIR_CAP: usize = 0;

/// The persistent directory column (`DIR_STRIDE` u32/contactId), aliased as u32 (material floats read
/// via `f32::from_bits`). Empty before the first reserve. The solver reads it and writes the hit flag.
pub fn dir_col() -> Col<'static, u32> {
    unsafe { Col::new(MANIFOLD_LAYOUT[DIR] as *mut u32, DIR_CAP * DIR_STRIDE) }
}

/// The persistent manifold pool column (`MANIFOLD_STRIDE` f32/record). The solver reads the manifolds
/// in prepare and writes the solved impulses back in store.
pub fn pool_col() -> Col<'static, f32> {
    unsafe { Col::new(MANIFOLD_LAYOUT[POOL] as *mut f32, MANIFOLD_CAP * MANIFOLD_STRIDE) }
}

/// Byte offset the geometry region (geo.rs) starts from: past the manifold region if one was reserved,
/// else the anchor below (past the body + fat-AABB + shape regions), where the manifold region would begin.
#[inline]
pub fn geo_base() -> usize {
    let end = unsafe { MANIFOLD_END } as usize;
    if end == 0 {
        manifold_anchor()
    } else {
        end
    }
}

/// First free byte past the manifold region (0 if none reserved).
#[inline]
pub fn region_end() -> usize {
    unsafe { MANIFOLD_END as usize }
}

/// Shift the manifold region's byte offsets up by `delta` after a region below it grew and moved it
/// (the caller memmoves the bytes). No-op if no region is reserved. The directory is indexed
/// by contactId and the pool by element offset, so only the header offsets + end marker rebase.
pub fn relocate(delta: usize) {
    unsafe {
        if MANIFOLD_END == 0 {
            return;
        }
        MANIFOLD_LAYOUT[DIR] += delta as u32;
        MANIFOLD_LAYOUT[POOL] += delta as u32;
        MANIFOLD_END += delta as u32;
    }
}

unsafe fn ensure_capacity(end_byte: usize) {
    let have = core::arch::wasm32::memory_size(0) * PAGE;
    if end_byte > have {
        let pages = (end_byte - have + PAGE - 1) / PAGE;
        core::arch::wasm32::memory_grow(0, pages);
    }
}

/// Byte offset of the manifold layout header (`[u32; N_MANIFOLD]` of the directory + pool byte offsets).
/// TS writes the directory + manifold records through views derived from this after every reserve.
#[export_name = "manifoldLayoutPtr"]
pub extern "C" fn manifold_layout_ptr() -> *const u32 {
    &raw const MANIFOLD_LAYOUT as *const u32
}

/// Lay out the persistent manifold columns for `contact_cap` directory records and `manifold_cap` pool
/// records, growing memory to fit and preserving the live pool data. The directory base is the anchor
/// below (past the body + fat-AABB + shape regions), so directory data stays in place across a manifold grow; the pool base moves up
/// only when the directory grows, and the existing pool run is memmoved to the new base so every live block
/// keeps its element offset. The caller re-uploads the geometry region afterward (it sits past this region
/// and shifts on a grow).
#[export_name = "reserveManifolds"]
pub extern "C" fn reserve_manifolds(contact_cap: usize, manifold_cap: usize) {
    unsafe {
        // Grow-only, like every other resident region (bodies/shapes/fataabb/broad). The manifold region
        // is a cross-world singleton, but a fresh world's `ManifoldStore` (`src/manifoldstore.ts`) tracks
        // its caps from 0 and reserves up from small — so world 2+ passes caps BELOW the singleton's
        // high-water. Honoring a shrink moves `MANIFOLD_END` below the broad region anchored at it, and
        // broad's `relocate` saturates its `delta` to 0 (never following the region *down*), desyncing the
        // two so the next manifold grow memmoves from a stale `old_top` and orphans the static tree pool's
        // bytes (its query then returns nothing). Clamp requested caps up to the high-water — a single
        // world always requests >= its own cap, so this is a no-op there (bit-exact).
        let contact_cap = if contact_cap > DIR_CAP { contact_cap } else { DIR_CAP };
        let manifold_cap = if manifold_cap > MANIFOLD_CAP { manifold_cap } else { MANIFOLD_CAP };

        let old_pool_base = MANIFOLD_LAYOUT[POOL] as usize;
        let old_cap = MANIFOLD_CAP;
        let old_top = geo_base(); // where the broad region currently anchors (past this region)

        let base = manifold_anchor();
        let new_pool_base = base + contact_cap * DIR_STRIDE * 4;
        let end = ((new_pool_base + manifold_cap * MANIFOLD_STRIDE * 4) + 3) & !3;

        // Relocate the broad-phase + geometry regions above this one up by the growth delta before the
        // pool expands into the space they vacate. The broad region holds live tree/pair-set data with
        // no re-upload path, so it must move in place (the geometry region also relocates here, though
        // the caller additionally re-uploads it). Both shift by the same delta, so one memmove of the
        // whole `[old_top, top)` span moves them together; their statics rebase.
        let delta = end.saturating_sub(old_top);
        let geo_end = crate::geo::region_end();
        let broad_end = crate::broad::region_end();
        let top = if geo_end != 0 {
            geo_end
        } else if broad_end != 0 {
            broad_end
        } else {
            old_top
        };
        if delta > 0 && top > old_top {
            ensure_capacity(top + delta);
            core::ptr::copy(old_top as *const u8, (old_top + delta) as *mut u8, top - old_top);
            crate::broad::relocate(delta);
            crate::geo::relocate(delta);
        } else {
            ensure_capacity(end);
        }

        // Move the live pool run to its new base before publishing the layout. `copy` is memmove, so
        // the dest >= src overlap (the directory only ever grows) is handled. Nothing to move on the
        // first reservation (old_cap 0) or when the pool base is unchanged (only the pool cap grew).
        // The grow-only clamp above guarantees `old_cap <= manifold_cap` (the region never shrinks, even
        // across a fresh world reusing the singleton), so the `min` is a defensive bound that never binds.
        let move_cap = if old_cap < manifold_cap { old_cap } else { manifold_cap };
        if move_cap != 0 && new_pool_base != old_pool_base {
            core::ptr::copy(
                old_pool_base as *const u8,
                new_pool_base as *mut u8,
                move_cap * MANIFOLD_STRIDE * 4,
            );
        }

        MANIFOLD_LAYOUT[DIR] = base as u32;
        MANIFOLD_LAYOUT[POOL] = new_pool_base as u32;
        MANIFOLD_END = end as u32;
        MANIFOLD_CAP = manifold_cap;
        DIR_CAP = contact_cap;
    }
}
