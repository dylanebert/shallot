//! The persistent broad-phase region — the three dynamic-tree node pools (static / kinematic /
//! dynamic) plus the pair-set membership arrays, held resident across steps so the in-kernel pair
//! query + tree rebuild (3d) run over them without a per-step marshal. Wasm-only (the columns alias
//! linear memory); TS owns the growth policy + the tree/table algorithms that write them
//! (`src/tree.ts`, `src/table.ts`, `src/broadcolumns.ts`).
//!
//! Six sub-columns, in memory order — three tree pools then the pair-set's three parallel u32 arrays:
//!   tree[Static]   `cap_s` nodes, `TREE_STRIDE` u32/f32 slots each (`src/tree.ts` node record)
//!   tree[Kinematic] `cap_k` nodes
//!   tree[Dynamic]  `cap_d` nodes
//!   keyHi / keyLo / hashes  `set_cap` u32 each (`src/table.ts` open-addressing set)
//!
//! Placement — a persistent region between the manifold region and the geometry region:
//! ```text
//! [shape_end, manifold_end)   persistent manifold columns (manifolds.rs)
//! [manifold_end, broad_end)   persistent broad-phase columns (this module)
//! [broad_end, geo_end)        static geometry pools (geo.rs)
//! [geo_end, ...)              per-step solver columns (arena.rs)
//! ```
//! It anchors from `manifolds::geo_base()` (past the manifold region), and the geometry region anchors
//! from this module's `region_top()` — so inserting it is transparent to the arena above (which
//! recomputes from `solver_base` every step). A region below it growing shifts this whole region up:
//! `relocate` rebases its offsets (the caller memmoves the bytes, which cover this region's span too).
//! Its own growth shifts only the geometry region above it, memmoved + relocated in place here, and
//! preserves its own sub-columns top-down exactly like the multi-column body region (bodies.rs).

const PAGE: usize = 65536;

/// f32/u32 slots per dynamic-tree node — mirrors `STRIDE` in `src/tree.ts` (sizeof(b3TreeNode)/4 = 12).
pub const TREE_STRIDE: usize = 12;

// BROAD_LAYOUT indices (byte offsets into linear memory), in memory order.
const TREE_S: usize = 0;
const TREE_K: usize = 1;
const TREE_D: usize = 2;
const KEY_HI: usize = 3;
const KEY_LO: usize = 4;
const HASHES: usize = 5;
const N_BROAD: usize = 6;

/// Per-column byte offsets into linear memory, rewritten by every grow-triggering `reserveBroad`.
/// TS reads this header (`broadLayoutPtr`) to build its column views after every grow.
static mut BROAD_LAYOUT: [u32; N_BROAD] = [0; N_BROAD];
/// First free byte past the broad region — where the geometry region anchors. Zero until the first
/// `reserveBroad`; `region_top` then treats it as an empty region ending at the manifold region's top.
static mut BROAD_END: u32 = 0;
/// The node capacity each tree pool is sized to (grow-only). The single source of truth for the TS
/// tree views' lengths (`src/broadcolumns.ts`).
static mut TREE_CAP: [usize; 3] = [0; 3];
/// The slot capacity the pair-set arrays are sized to (grow-only, power of two).
static mut SET_CAP: usize = 0;
/// Bumped on every layout change — a grow here or a relocation by a region below. TS keys its
/// view-staleness on it, so it catches a relocation that shifted this region's offsets without a
/// `memory.grow` (a lower region growing within already-committed pages).
static mut BROAD_GEN: u32 = 0;

#[inline]
fn align16(x: usize) -> usize {
    (x + 15) & !15
}

/// Byte offset the geometry region (geo.rs) anchors from: past the broad region if one was reserved,
/// else the anchor below (past the manifold region), where the broad region would begin.
#[inline]
pub fn region_top() -> usize {
    let end = unsafe { BROAD_END } as usize;
    if end == 0 {
        crate::manifolds::geo_base()
    } else {
        end
    }
}

/// First free byte past the broad region (0 if none reserved).
#[inline]
pub fn region_end() -> usize {
    unsafe { BROAD_END as usize }
}

/// Shift the broad region's byte offsets up by `delta` after a region below it grew and moved it (the
/// caller memmoves the bytes). No-op if none reserved. The pools are indexed by node/slot within each
/// column (element offsets absolute within them), so only the header offsets + end marker rebase.
pub fn relocate(delta: usize) {
    unsafe {
        if BROAD_END == 0 {
            return;
        }
        for i in 0..N_BROAD {
            BROAD_LAYOUT[i] += delta as u32;
        }
        BROAD_END += delta as u32;
        BROAD_GEN = BROAD_GEN.wrapping_add(1);
    }
}

/// The layout generation — bumped on every grow or relocation. TS re-derives its views when this
/// changes (or when memory grows), catching a relocation that moved this region without a `memory.grow`.
#[export_name = "broadGen"]
pub extern "C" fn broad_gen() -> u32 {
    unsafe { BROAD_GEN }
}

/// Base pointer of tree pool `i` (0 static / 1 kinematic / 2 dynamic) in linear memory. The in-kernel
/// pair query + rebuild (`pairwork.rs`) view the pool here as a flat `[u32]` of `cap * TREE_STRIDE` slots.
#[inline]
pub fn tree_ptr(i: usize) -> *mut u32 {
    unsafe { BROAD_LAYOUT[TREE_S + i] as *mut u32 }
}

/// Node capacity of tree pool `i` (source of truth for the pool slice length).
#[inline]
pub fn tree_cap(i: usize) -> usize {
    unsafe { TREE_CAP[i] }
}

/// Base pointers of the three pair-set arrays (keyHi / keyLo / hashes), each `set_cap()` u32.
#[inline]
pub fn set_ptrs() -> (*const u32, *const u32, *const u32) {
    unsafe {
        (
            BROAD_LAYOUT[KEY_HI] as *const u32,
            BROAD_LAYOUT[KEY_LO] as *const u32,
            BROAD_LAYOUT[HASHES] as *const u32,
        )
    }
}

/// The pair-set slot capacity (power of two; the probe mask is `set_cap() - 1`).
#[inline]
pub fn set_cap() -> usize {
    unsafe { SET_CAP }
}

unsafe fn ensure_capacity(end_byte: usize) {
    let have = core::arch::wasm32::memory_size(0) * PAGE;
    if end_byte > have {
        let pages = (end_byte - have + PAGE - 1) / PAGE;
        core::arch::wasm32::memory_grow(0, pages);
    }
}

/// Byte offset of the broad layout header (`[u32; N_BROAD]` of per-column byte offsets). TS reads this
/// to build its column views after every grow-triggering `reserveBroad`.
#[export_name = "broadLayoutPtr"]
pub extern "C" fn broad_layout_ptr() -> *const u32 {
    &raw const BROAD_LAYOUT as *const u32
}

/// The node capacity tree pool `i` (0 static / 1 kinematic / 2 dynamic) is sized to — the source of
/// truth for the TS tree view's length. Zero before the first `reserveBroad`.
#[export_name = "broadTreeCap"]
pub extern "C" fn broad_tree_cap(i: usize) -> usize {
    unsafe { TREE_CAP[i] }
}

/// The slot capacity the pair-set arrays are sized to — the source of truth for the TS set view lengths.
#[export_name = "broadSetCap"]
pub extern "C" fn broad_set_cap() -> usize {
    unsafe { SET_CAP }
}

/// Byte size of a column with `records` records of `stride` u32/f32 slots (0 records → 0 bytes).
#[inline]
fn col_bytes(records: usize, stride: usize) -> usize {
    records * stride * 4
}

/// Lay out the broad columns for tree pools of `cap_s`/`cap_k`/`cap_d` nodes and pair-set arrays of
/// `set_cap` slots, growing memory to fit. Grow-only per column: each new capacity is `max(requested,
/// current)`, so a caller growing one column passes 0 for the rest to hold them. A call that raises no
/// column is a no-op returning 0. On a grow it preserves every live sub-column (top-down copy, like the
/// body region), relocates the geometry region above it in place, and returns 1 — the caller then
/// refreshes every view a `memory.grow` detached. @returns 1 when the region grew.
#[export_name = "reserveBroad"]
pub extern "C" fn reserve_broad(cap_s: usize, cap_k: usize, cap_d: usize, set_cap: usize) -> u32 {
    unsafe {
        let new_tree = [
            if cap_s > TREE_CAP[0] { cap_s } else { TREE_CAP[0] },
            if cap_k > TREE_CAP[1] { cap_k } else { TREE_CAP[1] },
            if cap_d > TREE_CAP[2] { cap_d } else { TREE_CAP[2] },
        ];
        let new_set = if set_cap > SET_CAP { set_cap } else { SET_CAP };
        if new_tree[0] == TREE_CAP[0]
            && new_tree[1] == TREE_CAP[1]
            && new_tree[2] == TREE_CAP[2]
            && new_set == SET_CAP
        {
            return 0;
        }

        let old_layout = BROAD_LAYOUT;
        let old_tree = TREE_CAP;
        let old_set = SET_CAP;
        let old_top = region_top(); // where the geometry region currently anchors

        // Anchor at the *raw* manifold-region top — deliberately NOT align16'd. That anchor is only
        // 4-aligned and its residue mod 16 varies as the manifold caps grow (DIR_STRIDE*4 ≡ 4,
        // MANIFOLD_STRIDE*4 ≡ 12 mod 16). Padding the base to 16 here would desync the preserve-copy
        // below: a manifold grow relocates this region by a 4-but-not-16-aligned delta (base + memmoved
        // bytes both shift by that delta), but re-aligning to 16 on the next reserve would move TREE_S
        // to a *different* offset than the relocated bytes, orphaning the static tree pool (the copy
        // loop treats TREE_S as anchor-fixed and never copies it). Raw base = TREE_S tracks relocation
        // exactly (every delta 4-aligned), which is what makes TREE_S genuinely anchor-fixed. All broad
        // columns are scalar u32/f32 (no v128 access), so 4-byte alignment suffices for both the kernel
        // reads and the TS typed-array views (which read offsets from this header, not mirrored math).
        let base = crate::manifolds::geo_base();
        let mut off = base;
        let mut new_layout = [0u32; N_BROAD];
        new_layout[TREE_S] = off as u32;
        off += col_bytes(new_tree[0], TREE_STRIDE);
        new_layout[TREE_K] = off as u32;
        off += col_bytes(new_tree[1], TREE_STRIDE);
        new_layout[TREE_D] = off as u32;
        off += col_bytes(new_tree[2], TREE_STRIDE);
        new_layout[KEY_HI] = off as u32;
        off += col_bytes(new_set, 1);
        new_layout[KEY_LO] = off as u32;
        off += col_bytes(new_set, 1);
        new_layout[HASHES] = off as u32;
        off += col_bytes(new_set, 1);
        let new_end = align16(off);

        // Relocate the geometry region above by the growth delta (it holds live hull data a
        // non-hull shape create must not corrupt). `old_top` is where geo anchors now. Monotone by
        // construction with the raw base: the region's offsets track a lower relocation exactly and
        // `new_end` only grows, so `new_end >= old_top` — the assert pins that invariant and the
        // saturating sub guards against an unchecked usize wrap if it were ever violated.
        debug_assert!(new_end >= old_top, "broad region must never shift down");
        let delta = new_end.saturating_sub(old_top);
        let geo_end = crate::geo::region_end();
        if delta > 0 && geo_end > old_top {
            ensure_capacity(geo_end + delta);
            // `copy` is memmove; dest > src (the region only grows), so the overlap is handled.
            core::ptr::copy(old_top as *const u8, (old_top + delta) as *mut u8, geo_end - old_top);
            crate::geo::relocate(delta);
        } else {
            ensure_capacity(new_end);
        }

        // Preserve this region's own live sub-columns. TREE_S is genuinely anchor-fixed now (base is the
        // raw geo anchor, so a lower relocation shifts TREE_S by the same delta as its bytes) — its bytes
        // are already in place; every column above it shifts up. Copy top-down (highest new offset
        // first) so a write never lands on a lower column's not-yet-copied old bytes. Copy each
        // column's old live byte count; a fresh world reusing the singleton carries a larger stale cap,
        // but the copies stay within `new_end` (the region only grows) and dead bytes are overwritten
        // before read (tree nodes reset on alloc; the set window is cleared before use).
        let old_bytes = [
            col_bytes(old_tree[0], TREE_STRIDE),
            col_bytes(old_tree[1], TREE_STRIDE),
            col_bytes(old_tree[2], TREE_STRIDE),
            col_bytes(old_set, 1),
            col_bytes(old_set, 1),
            col_bytes(old_set, 1),
        ];
        for c in (TREE_K..=HASHES).rev() {
            if old_bytes[c] != 0 && new_layout[c] != old_layout[c] {
                core::ptr::copy(
                    old_layout[c] as *const u8,
                    new_layout[c] as *mut u8,
                    old_bytes[c],
                );
            }
        }

        BROAD_LAYOUT = new_layout;
        BROAD_END = new_end as u32;
        TREE_CAP = new_tree;
        SET_CAP = new_set;
        BROAD_GEN = BROAD_GEN.wrapping_add(1);
        1
    }
}
