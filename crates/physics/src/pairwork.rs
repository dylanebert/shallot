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

// Slab pointers (byte offsets) in the shared per-step arena.
static mut MOVE_PTR: u32 = 0;
static mut MOVED_PTR: u32 = 0;
static mut CANDEND_PTR: u32 = 0;
static mut CAND_PTR: u32 = 0;

static mut MOVE_COUNT: usize = 0;
static mut MOVED_WORDS: usize = 0;
static mut CAND_CAP: usize = 0;
static mut REBUILD_PENDING: bool = false;
static mut DEFER_QUERIES: bool = false;

pub unsafe fn callbacks_deferred() -> bool {
    DEFER_QUERIES
}

pub unsafe fn schedule_rebuild() {
    REBUILD_PENDING = true;
}

pub unsafe fn rebuild_pending() -> bool {
    REBUILD_PENDING
}

/// Reserve shared pair-finding scratch, consumed before dispatch, recycle or solve reserves it.
#[export_name = "reservePairs"]
pub extern "C" fn reserve_pairs() {
    reserve_pairs_in_world(crate::regions::active())
}

pub extern "C" fn reserve_pairs_in_world(world_index: usize) {
    unsafe {
        DEFER_QUERIES = false;
        if crate::callbacks::filter_enabled(world_index) {
            let shapes = shape_col(world_index);
            DEFER_QUERIES = (0..crate::shapes::shape_cap_in_world(world_index)).any(|id| {
                crate::shapes::shape_alive(world_index as u32, id as u32) != 0
                    && shapes[id * SHAPE_STRIDE + crate::shapes::S_FLAGS] & (4 << 16) != 0
                    && shapes[id * SHAPE_STRIDE + crate::shapes::S_SENSOR_INDEX] == u32::MAX
            });
        }
        let move_count = broad::move_count(world_index);
        let cand_cap = 16 * move_count;
        CAND_CAP = cand_cap;
        CAND_COUNT.store(0, Ordering::Relaxed);
        MOVE_COUNT = broad::move_count(world_index);
        MOVED_WORDS = broad::bits_words(world_index, DYNAMIC as usize);

        let mut off = 0;
        MOVE_PTR = broad::move_ptr(world_index) as u32;
        MOVED_PTR = broad::bits_ptr(world_index, DYNAMIC as usize) as u32;
        CANDEND_PTR = off as u32;
        off += move_count * 4;
        CAND_PTR = off as u32;
        off += cand_cap * CAND_STRIDE * 4;
        let base = crate::arena::reserve_scratch(world_index, off) as u32;
        CANDEND_PTR += base;
        CAND_PTR += base;
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

/// b3QueryPairContext over the resident columns and this proxy's move-result list.
struct Emitter<'a> {
    world: usize,
    shape: &'a [u32],
    moved: &'a [u64],
    items: &'a [table::Item],
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
        let block = (id >> 6) as usize;
        block < self.moved.len() && (self.moved[block] >> (id & 63)) & 1 != 0
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
        if table::contains_item(self.items, found_shape, self.query_shape, child) {
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
        if !unsafe {
            crate::callbacks::filter(self.world, found_shape as usize, self.query_shape as usize)
        } {
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

/// Each task owns its moved-proxy heads and traversal stack. The shared candidate allocator only
/// assigns storage; TS walks heads in move-buffer order, as Box3D's b3UpdateBroadPhasePairs walks
/// its prepended pair lists.
///
/// # Safety
/// reservePairs must precede the round; trees, shapes and membership stay fixed until its join.
pub unsafe fn query_block(world_index: usize, start: usize, end: usize, set_cap: usize) {
    unsafe {
        let move_count = MOVE_COUNT;
        let move_buf = core::slice::from_raw_parts(MOVE_PTR as *const u32, move_count);
        let moved = core::slice::from_raw_parts(MOVED_PTR as *const u64, MOVED_WORDS);
        let cand_end = CANDEND_PTR as *mut u32;
        let cand = CAND_PTR as *mut u32;
        let items = core::slice::from_raw_parts(broad::set_items(world_index), set_cap);
        let shape = shape_col(world_index);
        let stack = &mut [0i32; tree::STACK_SIZE];

        let pools = [
            pool_slice(world_index, 0),
            pool_slice(world_index, 1),
            pool_slice(world_index, 2),
        ];
        let roots = core::array::from_fn::<_, 3, _>(|i| {
            if broad::tree_cap(world_index, i) == 0 {
                -1
            } else {
                *broad::tree_state(world_index, i) as i32
            }
        });
        let counts = core::array::from_fn::<_, 3, _>(|i| {
            if broad::tree_cap(world_index, i) == 0 {
                0
            } else {
                *broad::tree_state(world_index, i).add(1) as usize
            }
        });

        let mut em = Emitter {
            world: world_index,
            shape,
            moved,
            items,
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

/// b3UpdateTreesTask: dynamic then kinematic, with no broadphase readers until its join.
#[export_name = "rebuildTrees"]
pub extern "C" fn rebuild_trees() {
    rebuild_trees_in_world(crate::regions::active())
}

pub extern "C" fn rebuild_trees_in_world(world_index: usize) {
    unsafe {
        for ti in [DYNAMIC as usize, KINEMATIC as usize] {
            if broad::tree_cap(world_index, ti) == 0 {
                continue;
            }
            crate::treework::mutate_resident_in_world(
                world_index,
                ti,
                4,
                0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                0,
                0,
                0,
            );
        }
        REBUILD_PENDING = false;
    }
}
