//! Box3D broad_phase.c: parallel pair queries and the serial tree rebuild.
//! Each moved proxy owns a prepended survivor list. TypeScript creates contacts after the join,
//! walking proxies in move-buffer order and each list in place. Allocation order across tasks
//! does not affect contact creation order.

use crate::broad;
use core::sync::atomic::{AtomicUsize, Ordering};

static CAND_COUNT: AtomicUsize = AtomicUsize::new(0);
use crate::shapes::{col_slice as shape_col, SHAPE_STRIDE, S_TYPE};
use crate::table;
use crate::tree;

/// `ShapeType.Compound`, as stored in the shape column.
const SHAPE_COMPOUND: u32 = 1;

/// Body types (broadphase.ts `BodyType`), packed into a proxy key's low 2 bits.
const KINEMATIC: u32 = 1;
const DYNAMIC: u32 = 2;

/// u32 slots per survivor: childIndex, shapeA, shapeB, next.
/// Each proxy owns a LIFO list; allocation order across proxies is immaterial.
const CAND_STRIDE: usize = 4;
/// u32 per input tree-state record: root, nodeCount, freeList, proxyCount.
const STATE_STRIDE: usize = 4;
/// u32 per rebuilt-tree output record: root, nodeCount, freeList.
const REBUILD_OUT_STRIDE: usize = 3;

// Slab pointers (byte offsets) in the shared per-step arena.
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
static mut CAND_CAP: usize = 256;
static mut MAX_PROXY: usize = 0;

/// Reserve shared pair-finding scratch, consumed before dispatch, recycle or solve reserves it.
/// Rebuild scratch covers the largest rebuilt tree; survivor capacity grows on overflow.
#[export_name = "reservePairs"]
pub extern "C" fn reserve_pairs() {
    reserve_pairs_in_world(crate::regions::active())
}

pub extern "C" fn reserve_pairs_in_world(world_index: usize) {
    unsafe {
        let cand_cap = CAND_CAP;
        let move_count = broad::move_count(world_index);
        let max_proxy = [1usize, 2]
            .into_iter()
            .map(|i| {
                if broad::tree_cap(world_index, i) == 0 {
                    0
                } else {
                    *broad::tree_state(world_index, i).add(3) as usize
                }
            })
            .max()
            .unwrap()
            .max(1);
        CAND_COUNT.store(0, Ordering::Relaxed);
        MOVE_COUNT = broad::move_count(world_index);
        MOVED_WORDS = broad::bits_words(world_index, DYNAMIC as usize);
        MAX_PROXY = max_proxy;

        let mut off = 0;
        STATE_PTR = off as u32;
        off += 3 * STATE_STRIDE * 4;
        MOVE_PTR = broad::move_ptr(world_index) as u32;
        MOVED_PTR = broad::bits_ptr(world_index, DYNAMIC as usize) as u32;
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
        let base = crate::arena::reserve_scratch(off) as u32;
        STATE_PTR += base;
        CANDEND_PTR += base;
        CAND_PTR += base;
        REBUILD_OUT_PTR += base;
        LEAFIDX_PTR += base;
        LEAFCEN_PTR += base;
        GATHER_PTR += base;
        BUILD_PTR += base;
        for i in 0..3 {
            let target = (STATE_PTR as *mut u32).add(i * STATE_STRIDE);
            if broad::tree_cap(world_index, i) == 0 {
                *target = u32::MAX;
                *target.add(1) = 0;
                *target.add(2) = u32::MAX;
                *target.add(3) = 0;
            } else {
                core::ptr::copy_nonoverlapping(
                    broad::tree_state(world_index, i),
                    target,
                    STATE_STRIDE,
                );
            }
        }
    }
}

#[export_name = "pairsCandEndPtr"]
pub extern "C" fn pairs_cand_end_ptr() -> *const u32 {
    unsafe { CANDEND_PTR as *const u32 }
}

#[export_name = "pairsCandPtr"]
pub extern "C" fn pairs_cand_ptr() -> *const u32 {
    unsafe { CAND_PTR as *const u32 }
}

/// One tree pool as a `[u32]` of `cap * STRIDE` slots.
#[inline]
unsafe fn pool_slice(world_index: usize, tree_index: usize) -> &'static [u32] {
    core::slice::from_raw_parts(
        broad::tree_ptr(world_index, tree_index),
        broad::tree_cap(world_index, tree_index) * tree::STRIDE,
    )
}

#[inline]
unsafe fn pool_slice_mut(world_index: usize, tree_index: usize) -> &'static mut [u32] {
    core::slice::from_raw_parts_mut(
        broad::tree_ptr(world_index, tree_index),
        broad::tree_cap(world_index, tree_index) * tree::STRIDE,
    )
}

/// b3QueryPairContext over the resident columns and this proxy's move-result list.
struct Emitter<'a> {
    world: usize,
    shape: &'a [u32],
    moved: &'a [u32],
    key_hi: &'a [u32],
    key_lo: &'a [u32],
    hashes: &'a [u32],
    set_cap: usize,
    cand: *mut u32,
    cand_cap: usize,
    head: u32,
    lower: crate::math::Vec3,
    upper: crate::math::Vec3,
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
    fn emit(&mut self, child: u32, a: u32, b: u32) {
        let index = CAND_COUNT.fetch_add(1, Ordering::Relaxed);
        if index < self.cand_cap {
            // Allocation order may race; only this proxy owns its links.
            unsafe {
                let entry = self.cand.add(index * CAND_STRIDE);
                *entry = child;
                *entry.add(1) = a;
                *entry.add(2) = b;
                *entry.add(3) = self.head;
                self.head = index as u32;
            }
        }
    }

    fn record(&mut self, other: i32, found_shape: u32) -> bool {
        if found_shape == self.query_shape {
            return true;
        }
        let sty = self.shape[found_shape as usize * SHAPE_STRIDE + S_TYPE];
        if sty == SHAPE_COMPOUND {
            unsafe {
                let (geometry, _) =
                    crate::query_abi::active_shape(self.world, found_shape as usize);
                let crate::query::Shape::Compound(compound) = geometry else {
                    unreachable!()
                };
                let transform = crate::world_query::pose(
                    self.world,
                    found_shape as usize,
                    crate::math::Vec3::ZERO,
                )
                .invert();
                let center = transform.point(self.lower.add(self.upper).scale(0.5));
                let extent = crate::math::Mat3::from_quat(transform.q)
                    .abs()
                    .mul_v(self.upper.sub(self.lower).scale(0.5));
                crate::compound_query::query(
                    compound,
                    center.sub(extent),
                    center.add(extent),
                    |_, child| self.record_child(other, found_shape, child),
                );
            }
            return true;
        }
        self.record_child(other, found_shape, 0)
    }

    fn record_child(&mut self, other: i32, found_shape: u32, child: u32) -> bool {
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
            child,
        ) {
            return true;
        }
        let a = &self.shape[found_shape as usize * SHAPE_STRIDE..][..SHAPE_STRIDE];
        let b = &self.shape[self.query_shape as usize * SHAPE_STRIDE..][..SHAPE_STRIDE];
        if a[1] == b[1] {
            return true;
        }
        if a[4] != u32::MAX || b[4] != u32::MAX {
            return true;
        }
        if !shapes_collide(a, b) {
            return true;
        }
        if !unsafe { crate::bodies::should_collide(a[1], b[1]) } {
            return true;
        }
        self.emit(child, found_shape, self.query_shape);
        true
    }
}

fn shapes_collide(a: &[u32], b: &[u32]) -> bool {
    if a[42] == b[42] && a[42] != 0 {
        return a[42] as i32 > 0;
    }
    ((a[41] & b[39]) | (a[40] & b[38])) != 0 && ((a[39] & b[41]) | (a[38] & b[40])) != 0
}

/// After the join, grow capacity for a read-only retry if the survivor lists overflowed.
#[export_name = "pairsOverflow"]
pub extern "C" fn pairs_overflow() -> u32 {
    let count = CAND_COUNT.load(Ordering::Relaxed);
    unsafe {
        if count <= CAND_CAP {
            return 0;
        }
        CAND_CAP = count + count / 2;
    }
    1
}

/// Each task owns its moved-proxy heads and traversal stack. The shared candidate allocator only
/// assigns storage; TS walks heads in move-buffer order, as Box3D's b3UpdateBroadPhasePairs walks
/// its prepended pair lists.
///
/// # Safety
/// reservePairs must precede the round; trees, shapes and membership stay fixed until its join.
pub unsafe fn query_block(world_index: usize, start: usize, end: usize, set_cap: usize) {
    unsafe {
        let move_count = MOVE_COUNT;
        let state = core::slice::from_raw_parts(STATE_PTR as *const u32, 3 * STATE_STRIDE);
        let move_buf = core::slice::from_raw_parts(MOVE_PTR as *const u32, move_count);
        let moved = core::slice::from_raw_parts(MOVED_PTR as *const u32, MOVED_WORDS);
        let cand_end = CANDEND_PTR as *mut u32;
        let cand = CAND_PTR as *mut u32;
        let (khi, klo, hp) = broad::set_ptrs(world_index);
        let key_hi = core::slice::from_raw_parts(khi, set_cap);
        let key_lo = core::slice::from_raw_parts(klo, set_cap);
        let hashes = core::slice::from_raw_parts(hp, set_cap);
        let shape = shape_col(world_index);
        let stack = &mut [0i32; tree::STACK_SIZE];

        let pools = [
            pool_slice(world_index, 0),
            pool_slice(world_index, 1),
            pool_slice(world_index, 2),
        ];
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
            world: world_index,
            shape,
            moved,
            key_hi,
            key_lo,
            hashes,
            set_cap,
            cand,
            cand_cap: CAND_CAP,
            head: u32::MAX,
            lower: crate::math::Vec3::ZERO,
            upper: crate::math::Vec3::ZERO,
            query_shape: 0,
            query_key: 0,
            query_dynamic: false,
            tree_type: 0,
        };

        for i in start..end {
            em.head = u32::MAX;
            let query_key = move_buf[i];
            let proxy_type = (query_key & 3) as usize;
            let proxy_id = (query_key >> 2) as i32;
            let query_dynamic = proxy_type as u32 == DYNAMIC;

            let base = pools[proxy_type];
            let (lo, hi) = tree::node_aabb(base, proxy_id);
            let query_shape = tree::user_data(base, proxy_id);

            em.lower = crate::math::Vec3::new(lo[0], lo[1], lo[2]);
            em.upper = crate::math::Vec3::new(hi[0], hi[1], hi[2]);
            em.query_shape = query_shape;
            em.query_key = query_key;
            em.query_dynamic = query_dynamic;

            // Dynamic proxies test kinematic then static; every proxy tests the dynamic tree.
            if query_dynamic {
                let k = KINEMATIC as usize;
                run_query(
                    pools[k], roots[k], counts[k], lo, hi, stack, &mut em, KINEMATIC,
                );
                run_query(pools[0], roots[0], counts[0], lo, hi, stack, &mut em, 0);
            }
            let d = DYNAMIC as usize;
            run_query(
                pools[d], roots[d], counts[d], lo, hi, stack, &mut em, DYNAMIC,
            );

            *cand_end.add(i) = em.head;
        }
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
/// first, then kinematic), and updates their resident headers. Static is never rebuilt.
///
/// # Safety
/// Runs after the query join (the query reads the pre-rebuild trees). Never grows the pool — the
/// resident capacity (`2*proxyCap-1`) always holds the rebuilt tree.
#[export_name = "rebuildTrees"]
pub extern "C" fn rebuild_trees() {
    rebuild_trees_in_world(crate::regions::active())
}

pub extern "C" fn rebuild_trees_in_world(world_index: usize) {
    unsafe {
        let state = core::slice::from_raw_parts(STATE_PTR as *const u32, 3 * STATE_STRIDE);
        let out =
            core::slice::from_raw_parts_mut(REBUILD_OUT_PTR as *mut u32, 2 * REBUILD_OUT_STRIDE);
        let mut leaf_indices = core::slice::from_raw_parts_mut(LEAFIDX_PTR as *mut i32, MAX_PROXY);
        let mut leaf_centers =
            core::slice::from_raw_parts_mut(LEAFCEN_PTR as *mut f32, MAX_PROXY * 3);
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
            let pool = pool_slice_mut(world_index, ti);
            let new_root = tree::rebuild(pool, root, proxy_count, false, &mut rb);

            let oo = slot * REBUILD_OUT_STRIDE;
            out[oo] = new_root as u32;
            out[oo + 1] = rb.node_count as u32;
            out[oo + 2] = rb.free_list as u32;
            if broad::tree_cap(world_index, ti) != 0 {
                core::ptr::copy_nonoverlapping(
                    out.as_ptr().add(oo),
                    broad::tree_state(world_index, ti),
                    3,
                );
            }

            // Re-borrow the scratch for the next tree (the Rebuild moved the &mut in).
            leaf_indices = rb.leaf_indices;
            leaf_centers = rb.leaf_centers;
            gather_stack = rb.gather_stack;
            build_stack = rb.build_stack;
        }
    }
}
