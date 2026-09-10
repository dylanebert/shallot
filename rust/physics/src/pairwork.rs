//! The in-kernel broad-phase pair-finding + tree-rebuild pass (3d) — box3d's
//! `b3UpdateBroadPhasePairs` phase 1 (query) + phase 2 (rebuild), over the resident tree pools +
//! pair-set (`broad.rs`). Serial, main-thread: it runs at step top before the worker pool wakes.
//!
//! Phase 1 (`queryPairs`) replays the TS enumeration order byte-for-byte — move-buffer order × per-proxy
//! kinematic→static→dynamic × LIFO DFS × per-proxy reverse walk (the reverse is TS's, phase 3) + the
//! lower-key moved-dedup — and rejects any pair already in the pair-set (`table::contains`), so the
//! candidate slab is ≈empty in steady state. A found compound leaf stays on the TS fallback path: the
//! kernel emits a *placeholder* (dedup applied, its inner-tree recursion + per-child membership left to
//! TS). Phase 2 (`rebuildTrees`) median-splits the dynamic then kinematic trees.
//!
//! Wasm-only: it aliases linear memory (the resident pools + a per-step scratch slab at `solver_base`).
//! Native `cargo test` drives `tree::query`/`tree::rebuild` + `table::contains` against gold vectors.

use crate::broad;
use crate::geo::solver_base;
use crate::shapes::{col_slice as shape_col, SHAPE_STRIDE, S_TYPE};
use crate::table;
use crate::tree;

const PAGE: usize = 65536;

/// `ShapeType.Compound` (the TS enum value the shape column stores) — the only found shape the query
/// hands to the TS fallback path rather than emitting as a direct candidate.
const SHAPE_COMPOUND: u32 = 1;

/// Body types (broadphase.ts `BodyType`), packed into a proxy key's low 2 bits.
const KINEMATIC: u32 = 1;
const DYNAMIC: u32 = 2;

/// u32 slots per candidate slab entry: flag (0 direct / 1 compound placeholder), shapeA, shapeB.
const CAND_STRIDE: usize = 3;
/// u32 per input tree-state record: root, nodeCount, freeList, proxyCount.
const STATE_STRIDE: usize = 4;
/// u32 per rebuilt-tree output record: root, nodeCount, freeList.
const REBUILD_OUT_STRIDE: usize = 3;

// Slab pointers (byte offsets), laid out by `reservePairs` at `solver_base`.
static mut STATE_PTR: u32 = 0;
static mut MOVE_PTR: u32 = 0;
static mut MOVED_PTR: u32 = 0;
static mut CANDEND_PTR: u32 = 0;
static mut CAND_PTR: u32 = 0;
static mut REBUILD_OUT_PTR: u32 = 0;
static mut LEAFIDX_PTR: u32 = 0;
static mut LEAFCEN_PTR: u32 = 0;
static mut GATHER_PTR: u32 = 0;
static mut BUILD_PTR: u32 = 0;

static mut MOVE_COUNT: usize = 0;
static mut MOVED_WORDS: usize = 0;
static mut CAND_CAP: usize = 0;
static mut MAX_PROXY: usize = 0;

unsafe fn ensure_capacity(end_byte: usize) {
    let have = core::arch::wasm32::memory_size(0) * PAGE;
    if end_byte > have {
        let pages = (end_byte - have + PAGE - 1) / PAGE;
        core::arch::wasm32::memory_grow(0, pages);
    }
}

/// Lay out the per-step pair-finding slab at `solver_base` (past the persistent + geometry regions,
/// consumed entirely within pair finding before the solver columns reserve over the same base). Sizes:
/// `move_count` moved proxies, `moved_words` u32 of the dynamic moved-bitset, `cand_cap` candidate
/// entries, `max_proxy` rebuild-leaf scratch (≥ the largest rebuilt tree's proxy count). Grows memory to
/// fit — always above every resident region, so it never relocates them.
///
/// # Safety
/// Nothing may hold a stale view over `[solver_base, ...)` across this (it reissues the slab each step).
#[export_name = "reservePairs"]
pub extern "C" fn reserve_pairs(
    move_count: usize,
    moved_words: usize,
    cand_cap: usize,
    max_proxy: usize,
) {
    unsafe {
        MOVE_COUNT = move_count;
        MOVED_WORDS = moved_words;
        CAND_CAP = cand_cap;
        MAX_PROXY = max_proxy;

        let mut off = solver_base();
        STATE_PTR = off as u32;
        off += 3 * STATE_STRIDE * 4;
        MOVE_PTR = off as u32;
        off += move_count * 4;
        MOVED_PTR = off as u32;
        off += moved_words * 4;
        CANDEND_PTR = off as u32;
        off += move_count * 4;
        CAND_PTR = off as u32;
        off += cand_cap * CAND_STRIDE * 4;
        REBUILD_OUT_PTR = off as u32;
        off += 2 * REBUILD_OUT_STRIDE * 4;
        LEAFIDX_PTR = off as u32;
        off += max_proxy * 4;
        LEAFCEN_PTR = off as u32;
        off += max_proxy * 3 * 4;
        GATHER_PTR = off as u32;
        off += tree::STACK_SIZE * 4;
        BUILD_PTR = off as u32;
        off += tree::STACK_SIZE * 5 * 4;
        ensure_capacity(off);
    }
}

#[export_name = "pairsStatePtr"]
pub extern "C" fn pairs_state_ptr() -> *mut u32 {
    unsafe { STATE_PTR as *mut u32 }
}

#[export_name = "pairsMovePtr"]
pub extern "C" fn pairs_move_ptr() -> *mut u32 {
    unsafe { MOVE_PTR as *mut u32 }
}

#[export_name = "pairsMovedPtr"]
pub extern "C" fn pairs_moved_ptr() -> *mut u32 {
    unsafe { MOVED_PTR as *mut u32 }
}

#[export_name = "pairsCandEndPtr"]
pub extern "C" fn pairs_cand_end_ptr() -> *const u32 {
    unsafe { CANDEND_PTR as *const u32 }
}

#[export_name = "pairsCandPtr"]
pub extern "C" fn pairs_cand_ptr() -> *const u32 {
    unsafe { CAND_PTR as *const u32 }
}

#[export_name = "pairsRebuildOutPtr"]
pub extern "C" fn pairs_rebuild_out_ptr() -> *const u32 {
    unsafe { REBUILD_OUT_PTR as *const u32 }
}

/// One tree pool as a `[u32]` of `cap * STRIDE` slots.
#[inline]
unsafe fn pool_slice(tree_index: usize) -> &'static [u32] {
    core::slice::from_raw_parts(broad::tree_ptr(tree_index), broad::tree_cap(tree_index) * tree::STRIDE)
}

#[inline]
unsafe fn pool_slice_mut(tree_index: usize) -> &'static mut [u32] {
    core::slice::from_raw_parts_mut(
        broad::tree_ptr(tree_index),
        broad::tree_cap(tree_index) * tree::STRIDE,
    )
}

/// The per-proxy emit context: the read-only membership inputs + the mutable candidate cursor. Its
/// `record` is the query callback — the port of `pairs.ts`'s `record` + `emit` (dedup + membership +
/// compound partition); the surviving TS filters (self-body / sensor / shouldShapesCollide / joint walk)
/// run over the emitted slab in `src/pairs.ts`.
struct Emitter<'a> {
    shape: &'a [u32],
    moved: &'a [u32],
    key_hi: &'a [u32],
    key_lo: &'a [u32],
    hashes: &'a [u32],
    set_cap: usize,
    cand: &'a mut [u32],
    cand_cap: usize,
    count: usize,
    query_shape: u32,
    query_key: u32,
    query_dynamic: bool,
    tree_type: u32,
}

impl<'a> Emitter<'a> {
    #[inline]
    fn moved_bit(&self, id: i32) -> bool {
        let block = (id >> 5) as usize;
        block < self.moved.len() && (self.moved[block] >> (id & 31)) & 1 != 0
    }

    /// b3PairQueryCallback's moved-proxy dedup: when both proxies moved, only the lower-keyed proxy's
    /// query creates the pair (dynamic case), and a non-dynamic query skips any moved found proxy.
    #[inline]
    fn dedup_reject(&self, other: i32) -> bool {
        if self.query_dynamic {
            self.tree_type == DYNAMIC
                && (((other as u32) << 2) | DYNAMIC) < self.query_key
                && self.moved_bit(other)
        } else {
            self.moved_bit(other)
        }
    }

    #[inline]
    fn emit(&mut self, flag: u32, a: u32, b: u32) {
        if self.count < self.cand_cap {
            let o = self.count * CAND_STRIDE;
            self.cand[o] = flag;
            self.cand[o + 1] = a;
            self.cand[o + 2] = b;
        }
        self.count += 1;
    }

    fn record(&mut self, other: i32, found_shape: u32) -> bool {
        if found_shape == self.query_shape {
            return true;
        }
        let sty = self.shape[found_shape as usize * SHAPE_STRIDE + S_TYPE];
        if sty == SHAPE_COMPOUND {
            // Compound: the dedup depends only on the found *outer* proxy (identical for every inner
            // child), so run it once here and emit a placeholder; TS maps the query bounds into the
            // compound's frame, recurses its inner tree, and applies per-child membership + filters.
            if self.dedup_reject(other) {
                return true;
            }
            self.emit(1, found_shape, self.query_shape);
            return true;
        }
        if self.dedup_reject(other) {
            return true;
        }
        if table::contains(
            self.key_hi,
            self.key_lo,
            self.hashes,
            self.set_cap,
            found_shape,
            self.query_shape,
            0,
        ) {
            return true;
        }
        self.emit(0, found_shape, self.query_shape);
        true
    }
}

/// Phase 1 — find candidate pairs for every moved proxy, in move-buffer order, into the candidate slab.
/// Returns the total entry count (which may exceed `cand_cap` on a cold step; the slab holds only the
/// first `cand_cap`, and TS grows + re-runs — the query mutates neither the trees nor the pair-set, so a
/// re-run is free of side effects). `candEnd[i]` delimits moved proxy `i`'s entries.
///
/// `set_cap` is the pair-set's *logical* capacity (TS `HashSet.capacity`), not `broad::set_cap()` — the
/// resident region is grow-only across the singleton's worlds, so its slot capacity can exceed this
/// world's table (a prior world's high-water). Probing with the region size would use the wrong mask and
/// miss present pairs; the TS logical capacity is the table the membership actually lives in.
///
/// # Safety
/// `reservePairs` must have run this step with the current move buffer + tree state + dynamic moved-bits
/// written into the slab, and no thread may grow memory while this runs.
#[export_name = "queryPairs"]
pub extern "C" fn query_pairs(set_cap: usize) -> u32 {
    unsafe {
        let move_count = MOVE_COUNT;
        let state = core::slice::from_raw_parts(STATE_PTR as *const u32, 3 * STATE_STRIDE);
        let move_buf = core::slice::from_raw_parts(MOVE_PTR as *const u32, move_count);
        let moved = core::slice::from_raw_parts(MOVED_PTR as *const u32, MOVED_WORDS);
        let cand_end = core::slice::from_raw_parts_mut(CANDEND_PTR as *mut u32, move_count);
        let cand = core::slice::from_raw_parts_mut(CAND_PTR as *mut u32, CAND_CAP * CAND_STRIDE);
        let (khi, klo, hp) = broad::set_ptrs();
        let key_hi = core::slice::from_raw_parts(khi, set_cap);
        let key_lo = core::slice::from_raw_parts(klo, set_cap);
        let hashes = core::slice::from_raw_parts(hp, set_cap);
        let shape = shape_col();
        let stack = core::slice::from_raw_parts_mut(GATHER_PTR as *mut i32, tree::STACK_SIZE);

        let pools = [pool_slice(0), pool_slice(1), pool_slice(2)];
        let roots = [
            state[0] as i32,
            state[STATE_STRIDE] as i32,
            state[2 * STATE_STRIDE] as i32,
        ];
        let counts = [
            state[1] as usize,
            state[STATE_STRIDE + 1] as usize,
            state[2 * STATE_STRIDE + 1] as usize,
        ];

        let mut em = Emitter {
            shape,
            moved,
            key_hi,
            key_lo,
            hashes,
            set_cap,
            cand,
            cand_cap: CAND_CAP,
            count: 0,
            query_shape: 0,
            query_key: 0,
            query_dynamic: false,
            tree_type: 0,
        };

        for i in 0..move_count {
            let query_key = move_buf[i];
            let proxy_type = (query_key & 3) as usize;
            let proxy_id = (query_key >> 2) as i32;
            let query_dynamic = proxy_type as u32 == DYNAMIC;

            let base = pools[proxy_type];
            let (lo, hi) = tree::node_aabb(base, proxy_id);
            let query_shape = tree::user_data(base, proxy_id);

            em.query_shape = query_shape;
            em.query_key = query_key;
            em.query_dynamic = query_dynamic;

            // Dynamic proxies test kinematic then static; every proxy tests the dynamic tree.
            if query_dynamic {
                let k = KINEMATIC as usize;
                run_query(pools[k], roots[k], counts[k], lo, hi, stack, &mut em, KINEMATIC);
                run_query(pools[0], roots[0], counts[0], lo, hi, stack, &mut em, 0);
            }
            let d = DYNAMIC as usize;
            run_query(pools[d], roots[d], counts[d], lo, hi, stack, &mut em, DYNAMIC);

            cand_end[i] = em.count as u32;
        }

        em.count as u32
    }
}

#[inline]
fn run_query(
    pool: &[u32],
    root: i32,
    node_count: usize,
    lo: [f32; 3],
    hi: [f32; 3],
    stack: &mut [i32],
    em: &mut Emitter,
    tree_type: u32,
) {
    em.tree_type = tree_type;
    tree::query(
        pool,
        root,
        node_count,
        lo,
        hi,
        tree::QUERY_MASK_HI,
        tree::QUERY_MASK_LO,
        false,
        stack,
        |other, found_shape| em.record(other, found_shape),
    );
}

/// Phase 2 — rebuild the dynamic then kinematic trees (median split, `full == false`), matching the TS
/// order. Writes each rebuilt tree's new `[root, nodeCount, freeList]` into the rebuild-out slab (dynamic
/// first, then kinematic); TS folds them back into its `DynamicTree` structs. Static is never rebuilt.
///
/// # Safety
/// As `queryPairs`; runs after it (the query reads the pre-rebuild trees). Never grows the pool — the
/// resident capacity (`2*proxyCap-1`) always holds the rebuilt tree.
#[export_name = "rebuildTrees"]
pub extern "C" fn rebuild_trees() {
    unsafe {
        let state = core::slice::from_raw_parts(STATE_PTR as *const u32, 3 * STATE_STRIDE);
        let out = core::slice::from_raw_parts_mut(REBUILD_OUT_PTR as *mut u32, 2 * REBUILD_OUT_STRIDE);
        let mut leaf_indices = core::slice::from_raw_parts_mut(LEAFIDX_PTR as *mut i32, MAX_PROXY);
        let mut leaf_centers = core::slice::from_raw_parts_mut(LEAFCEN_PTR as *mut f32, MAX_PROXY * 3);
        let mut gather_stack =
            core::slice::from_raw_parts_mut(GATHER_PTR as *mut i32, tree::STACK_SIZE);
        let mut build_stack =
            core::slice::from_raw_parts_mut(BUILD_PTR as *mut i32, tree::STACK_SIZE * 5);

        // Dynamic (tree 2) then kinematic (tree 1).
        for (slot, ti) in [(0usize, DYNAMIC as usize), (1usize, KINEMATIC as usize)] {
            let so = ti * STATE_STRIDE;
            let root = state[so] as i32;
            let node_count = state[so + 1] as usize;
            let free_list = state[so + 2] as i32;
            let proxy_count = state[so + 3] as usize;

            let mut rb = tree::Rebuild {
                node_count,
                free_list,
                leaf_indices,
                leaf_centers,
                gather_stack,
                build_stack,
            };
            let pool = pool_slice_mut(ti);
            let new_root = tree::rebuild(pool, root, proxy_count, false, &mut rb);

            let oo = slot * REBUILD_OUT_STRIDE;
            out[oo] = new_root as u32;
            out[oo + 1] = rb.node_count as u32;
            out[oo + 2] = rb.free_list as u32;

            // Re-borrow the scratch for the next tree (the Rebuild moved the &mut in).
            leaf_indices = rb.leaf_indices;
            leaf_centers = rb.leaf_centers;
            gather_stack = rb.gather_stack;
            build_stack = rb.build_stack;
        }
    }
}
