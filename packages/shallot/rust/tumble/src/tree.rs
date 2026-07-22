//! In-kernel dynamic AABB tree — the read + rebuild half of box3d's `dynamic_tree.c`, ported for the
//! resident broad-phase pair query (3d). Mirrors `src/tree.ts`: one flat node pool, 12 four-byte slots
//! per node over a single `[u32]` buffer (aabb f32 bits in slots 0..5, category hi/lo in 6/7,
//! child1/userData + child2 in 8/9, parent/next in 10, height|flags in 11).
//!
//! Only `query` (broad-phase overlap DFS) and `rebuild` (median-split rebuild) live here — the tree
//! *mutations* (create/move/enlarge/destroy) stay in TS (`src/tree.ts`), driven over the same resident
//! pool. Query is integer + f32-compare only (no arithmetic); rebuild does f32 median splits, ported
//! op-for-op and pinned under `kernel/tests/tree_gold.rs`. Native `cargo test` drives both over owned
//! `Vec<u32>` pools; the wasm path (`pairwork.rs`) drives them over the resident region.

/// Four-byte slots per node (`sizeof(b3TreeNode) / 4`).
pub const STRIDE: usize = 12;
/// b3RebuildItem fields per build-stack entry: nodeIndex, childCount, startIndex, splitIndex, endIndex.
const ITEM_STRIDE: usize = 5;
/// B3_TREE_STACK_SIZE — the query traversal + rebuild gather/build stacks.
pub const STACK_SIZE: usize = 1024;

pub const NULL_INDEX: i32 = -1;

// b3TreeNodeFlags (low 16 bits of slot 11).
const ALLOCATED: u32 = 0x0001;
const ENLARGED: u32 = 0x0002;
const LEAF: u32 = 0x0004;

const ALL_BITS: u32 = 0xffff_ffff;

#[inline]
fn minf(a: f32, b: f32) -> f32 {
    // b3MinFloat: the DISABLE_SIMD ternary, not f32::min (which differs on NaN / -0).
    if a < b {
        a
    } else {
        b
    }
}

#[inline]
fn maxf(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else {
        b
    }
}

#[inline]
fn maxi(a: i32, b: i32) -> i32 {
    if a > b {
        a
    } else {
        b
    }
}

#[inline]
fn fget(pool: &[u32], slot: usize) -> f32 {
    f32::from_bits(pool[slot])
}

#[inline]
fn height_of(pool: &[u32], i: i32) -> i32 {
    (pool[i as usize * STRIDE + 11] >> 16) as i32
}

#[inline]
fn set_height(pool: &mut [u32], i: i32, h: i32) {
    let n = i as usize * STRIDE + 11;
    pool[n] = (pool[n] & 0xffff) | ((h as u32) << 16);
}

#[inline]
fn is_leaf(pool: &[u32], i: i32) -> bool {
    pool[i as usize * STRIDE + 11] & LEAF != 0
}

/// aabb.union(node i, node j) → node k's aabb slots (b3AABB_Union): lower = min lowers, upper = max uppers.
fn union_into(pool: &mut [u32], i: i32, j: i32, k: i32) {
    let a = i as usize * STRIDE;
    let b = j as usize * STRIDE;
    let d = k as usize * STRIDE;
    let lx = minf(fget(pool, a), fget(pool, b));
    let ly = minf(fget(pool, a + 1), fget(pool, b + 1));
    let lz = minf(fget(pool, a + 2), fget(pool, b + 2));
    let ux = maxf(fget(pool, a + 3), fget(pool, b + 3));
    let uy = maxf(fget(pool, a + 4), fget(pool, b + 4));
    let uz = maxf(fget(pool, a + 5), fget(pool, b + 5));
    pool[d] = lx.to_bits();
    pool[d + 1] = ly.to_bits();
    pool[d + 2] = lz.to_bits();
    pool[d + 3] = ux.to_bits();
    pool[d + 4] = uy.to_bits();
    pool[d + 5] = uz.to_bits();
}

/// dst.category |= (a.category | b.category) — the OR propagation (both u32 halves).
fn or_category(pool: &mut [u32], dst: i32, a: i32, b: i32) {
    let d = dst as usize * STRIDE;
    let na = a as usize * STRIDE;
    let nb = b as usize * STRIDE;
    pool[d + 6] = pool[na + 6] | pool[nb + 6];
    pool[d + 7] = pool[na + 7] | pool[nb + 7];
}

/// Broad-phase overlap query (b3DynamicTree_Query with an ALL_BITS mask). Visits every leaf whose fat
/// AABB overlaps `box` in LIFO discovery order, calling `cb(leafNodeId, userData)`; a `false` return
/// stops the walk. Returns `(node_visits, leaf_visits)`. `stack` is caller scratch (`STACK_SIZE` ints).
///
/// The mask test + AABB overlap are ported verbatim from `src/tree.ts::query`; both are integer /
/// f32-compare only, so the traversal is bit-exact by replaying enumeration order.
pub fn query<F: FnMut(i32, u32) -> bool>(
    pool: &[u32],
    root: i32,
    node_count: usize,
    lo: [f32; 3],
    hi: [f32; 3],
    mask_hi: u32,
    mask_lo: u32,
    require_all: bool,
    stack: &mut [i32],
    mut cb: F,
) -> (u32, u32) {
    if node_count == 0 {
        return (0, 0);
    }
    let (blx, bly, blz) = (lo[0], lo[1], lo[2]);
    let (bhx, bhy, bhz) = (hi[0], hi[1], hi[2]);

    let mut node_visits = 0u32;
    let mut leaf_visits = 0u32;
    let mut count = 0usize;
    stack[count] = root;
    count += 1;

    while count > 0 {
        count -= 1;
        let node_id = stack[count];
        let n = node_id as usize * STRIDE;
        node_visits += 1;

        let ch = pool[n + 6] & mask_hi;
        let cl = pool[n + 7] & mask_lo;
        let matched = if require_all {
            ch == mask_hi && cl == mask_lo
        } else {
            ch != 0 || cl != 0
        };
        // Conjunction form (equivalent to b3AABB_Overlaps for finite operands; tree AABBs are finite).
        let overlaps = fget(pool, n + 3) >= blx
            && fget(pool, n) <= bhx
            && fget(pool, n + 4) >= bly
            && fget(pool, n + 1) <= bhy
            && fget(pool, n + 5) >= blz
            && fget(pool, n + 2) <= bhz;

        if matched && overlaps {
            if pool[n + 11] & LEAF != 0 {
                leaf_visits += 1;
                if !cb(node_id, pool[n + 8]) {
                    return (node_visits, leaf_visits);
                }
            } else if count < STACK_SIZE - 1 {
                stack[count] = pool[n + 8] as i32;
                count += 1;
                stack[count] = pool[n + 9] as i32;
                count += 1;
            }
        }
    }
    (node_visits, leaf_visits)
}

/// Read node `i`'s fat AABB (`[lo.xyz, hi.xyz]`) — the query box the broad phase runs for a moved proxy.
#[inline]
pub fn node_aabb(pool: &[u32], i: i32) -> ([f32; 3], [f32; 3]) {
    let n = i as usize * STRIDE;
    (
        [fget(pool, n), fget(pool, n + 1), fget(pool, n + 2)],
        [fget(pool, n + 3), fget(pool, n + 4), fget(pool, n + 5)],
    )
}

/// Read node `i`'s userData (leaf shape index) — slot 8 under the leaf guard.
#[inline]
pub fn user_data(pool: &[u32], i: i32) -> u32 {
    pool[i as usize * STRIDE + 8]
}

pub const QUERY_MASK_HI: u32 = ALL_BITS;
pub const QUERY_MASK_LO: u32 = ALL_BITS;

// --- rebuild (median split, B3_TREE_HEURISTIC == 0) -----------------------------------------

/// Rebuild state threaded through the mutation (nodeCount / freeList change as internals free + alloc;
/// root is the new root). proxyCount is invariant across a rebuild.
pub struct Rebuild<'a> {
    pub node_count: usize,
    pub free_list: i32,
    pub leaf_indices: &'a mut [i32],
    pub leaf_centers: &'a mut [f32], // 3 per leaf
    pub gather_stack: &'a mut [i32],
    pub build_stack: &'a mut [i32], // ITEM_STRIDE per item
}

// *node = b3_defaultTreeNode.
fn reset_to_default(pool: &mut [u32], i: i32) {
    let n = i as usize * STRIDE;
    for k in 0..6 {
        pool[n + k] = 0;
    }
    pool[n + 6] = ALL_BITS;
    pool[n + 7] = ALL_BITS;
    pool[n + 8] = NULL_INDEX as u32;
    pool[n + 9] = NULL_INDEX as u32;
    pool[n + 10] = NULL_INDEX as u32;
    pool[n + 11] = ALLOCATED;
}

/// Pop a free node (rebuild never grows: the pool is sized `2*proxyCap-1` ≥ any live tree, so the free
/// list is never empty here — see `src/tree.ts` allocateNode + the rebuild comment).
fn allocate_node(pool: &mut [u32], rb: &mut Rebuild) -> i32 {
    let node_index = rb.free_list;
    rb.free_list = pool[node_index as usize * STRIDE + 10] as i32;
    reset_to_default(pool, node_index);
    rb.node_count += 1;
    node_index
}

fn free_node(pool: &mut [u32], rb: &mut Rebuild, node_id: i32) {
    let n = node_id as usize * STRIDE;
    pool[n + 10] = rb.free_list as u32;
    pool[n + 11] &= !0xffff; // flags = 0 (height untouched)
    rb.free_list = node_id;
    rb.node_count -= 1;
}

/// Median split of leaf centers along the longest axis (Hoare partition). Returns the left count.
/// `centers`/`indices` are offset by `start` (centers stride 3). Ported op-for-op from `partitionMid`.
fn partition_mid(indices: &mut [i32], centers: &mut [f32], start: usize, count: usize) -> usize {
    if count <= 2 {
        return count / 2;
    }

    let mut lx = centers[start * 3];
    let mut ly = centers[start * 3 + 1];
    let mut lz = centers[start * 3 + 2];
    let mut ux = lx;
    let mut uy = ly;
    let mut uz = lz;
    for i in 1..count {
        let c = (start + i) * 3;
        lx = minf(lx, centers[c]);
        ly = minf(ly, centers[c + 1]);
        lz = minf(lz, centers[c + 2]);
        ux = maxf(ux, centers[c]);
        uy = maxf(uy, centers[c + 1]);
        uz = maxf(uz, centers[c + 2]);
    }

    let dx = ux - lx;
    let dy = uy - ly;
    let dz = uz - lz;
    let cx = 0.5f32 * (lx + ux);
    let cy = 0.5f32 * (ly + uy);
    let cz = 0.5f32 * (lz + uz);

    let mut i1 = 0usize;
    let mut i2 = count;

    let axis = if dx >= dy && dx >= dz {
        0
    } else if dy >= dz {
        1
    } else {
        2
    };
    let pivot = match axis {
        0 => cx,
        1 => cy,
        _ => cz,
    };

    while i1 < i2 {
        while i1 < i2 && centers[(start + i1) * 3 + axis] < pivot {
            i1 += 1;
        }
        while i1 < i2 && centers[(start + i2 - 1) * 3 + axis] >= pivot {
            i2 -= 1;
        }
        if i1 < i2 {
            // swap(i1, i2 - 1)
            let ia = start + i1;
            let ib = start + i2 - 1;
            indices.swap(ia, ib);
            for k in 0..3 {
                centers.swap(ia * 3 + k, ib * 3 + k);
            }
            i1 += 1;
            i2 -= 1;
        }
    }

    if i1 > 0 && i1 < count {
        i1
    } else {
        count / 2
    }
}

fn build_tree(pool: &mut [u32], rb: &mut Rebuild, leaf_count: usize) -> i32 {
    if leaf_count == 1 {
        let i0 = rb.leaf_indices[0];
        pool[i0 as usize * STRIDE + 10] = NULL_INDEX as u32; // parent
        return i0;
    }

    let mut top: usize = 0;
    let node0 = allocate_node(pool, rb);
    {
        let bs = &mut rb.build_stack;
        bs[0] = node0;
        bs[1] = -1;
        bs[2] = 0;
        bs[4] = leaf_count as i32;
    }
    let split0 = partition_mid(rb.leaf_indices, rb.leaf_centers, 0, leaf_count);
    rb.build_stack[3] = split0 as i32;

    loop {
        let base = top * ITEM_STRIDE;
        rb.build_stack[base + 1] += 1;

        if rb.build_stack[base + 1] == 2 {
            if top == 0 {
                break;
            }

            let parent_base = (top - 1) * ITEM_STRIDE;
            let parent_node = rb.build_stack[parent_base];
            let node_index = rb.build_stack[base];
            if rb.build_stack[parent_base + 1] == 0 {
                pool[parent_node as usize * STRIDE + 8] = node_index as u32;
            } else {
                pool[parent_node as usize * STRIDE + 9] = node_index as u32;
            }

            pool[node_index as usize * STRIDE + 10] = parent_node as u32; // parent

            let child1 = pool[node_index as usize * STRIDE + 8] as i32;
            let child2 = pool[node_index as usize * STRIDE + 9] as i32;
            union_into(pool, child1, child2, node_index);
            set_height(pool, node_index, 1 + maxi(height_of(pool, child1), height_of(pool, child2)));
            or_category(pool, node_index, child1, child2);

            top -= 1;
        } else {
            let (start_index, end_index) = if rb.build_stack[base + 1] == 0 {
                (rb.build_stack[base + 2], rb.build_stack[base + 3])
            } else {
                (rb.build_stack[base + 3], rb.build_stack[base + 4])
            };

            let count = (end_index - start_index) as usize;

            if count == 1 {
                let child_index = rb.leaf_indices[start_index as usize];
                let node_index = rb.build_stack[base];
                if rb.build_stack[base + 1] == 0 {
                    pool[node_index as usize * STRIDE + 8] = child_index as u32;
                } else {
                    pool[node_index as usize * STRIDE + 9] = child_index as u32;
                }
                pool[child_index as usize * STRIDE + 10] = node_index as u32; // parent
            } else {
                top += 1;
                let split =
                    partition_mid(rb.leaf_indices, rb.leaf_centers, start_index as usize, count);
                let alloc = allocate_node(pool, rb);
                let nb = top * ITEM_STRIDE;
                rb.build_stack[nb] = alloc;
                rb.build_stack[nb + 1] = -1;
                rb.build_stack[nb + 2] = start_index;
                rb.build_stack[nb + 4] = end_index;
                rb.build_stack[nb + 3] = split as i32 + start_index;
            }
        }
    }

    let root_index = rb.build_stack[0];
    let child1 = pool[root_index as usize * STRIDE + 8] as i32;
    let child2 = pool[root_index as usize * STRIDE + 9] as i32;
    union_into(pool, child1, child2, root_index);
    set_height(pool, root_index, 1 + maxi(height_of(pool, child1), height_of(pool, child2)));
    or_category(pool, root_index, child1, child2);

    root_index
}

/// Rebuild the tree (b3DynamicTree_Rebuild). Gathers grown proxies + un-grown internal nodes as rebuild
/// leaves (freeing the grown internals), then median-splits them into a fresh balanced tree. `full`
/// rebuilds every node; otherwise only the enlarged subtrees. Returns the new root; `rb.node_count` /
/// `rb.free_list` are updated in place. `proxy_count` is the tree's current proxy count.
pub fn rebuild(pool: &mut [u32], root: i32, proxy_count: usize, full: bool, rb: &mut Rebuild) -> i32 {
    if proxy_count == 0 {
        return root;
    }

    let mut leaf_count = 0usize;
    let mut gather_count = 0usize;
    let mut node_index = root;

    loop {
        let n = node_index as usize * STRIDE;
        if is_leaf(pool, node_index) || (pool[n + 11] & ENLARGED == 0 && !full) {
            rb.leaf_indices[leaf_count] = node_index;
            let cx = 0.5f32 * (fget(pool, n + 3) + fget(pool, n));
            let cy = 0.5f32 * (fget(pool, n + 4) + fget(pool, n + 1));
            let cz = 0.5f32 * (fget(pool, n + 5) + fget(pool, n + 2));
            rb.leaf_centers[leaf_count * 3] = cx;
            rb.leaf_centers[leaf_count * 3 + 1] = cy;
            rb.leaf_centers[leaf_count * 3 + 2] = cz;
            leaf_count += 1;
            pool[n + 10] = NULL_INDEX as u32; // parent
        } else {
            let doomed = node_index;
            node_index = pool[n + 8] as i32; // child1
            if gather_count < STACK_SIZE {
                rb.gather_stack[gather_count] = pool[n + 9] as i32; // child2
                gather_count += 1;
            }
            free_node(pool, rb, doomed);
            continue;
        }

        if gather_count == 0 {
            break;
        }
        gather_count -= 1;
        node_index = rb.gather_stack[gather_count];
    }

    build_tree(pool, rb, leaf_count)
}
