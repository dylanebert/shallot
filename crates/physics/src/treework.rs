//! Serial tree operations over resident or caller-uploaded columns.
use crate::body::{S2_BODY_ID, S2_FLAGS, SIM2_STRIDE};
use crate::continuous::{ENLARGE_BOUNDS, IS_BULLET, IS_FAST};
use crate::regions::Buffer;
use crate::tree::{self, Rebuild, STACK_SIZE, STRIDE};
use std::slice;
static mut SCRATCH: [Buffer; 64] = [Buffer::EMPTY; 64];
#[link(wasm_import_module = "env")]
extern "C" {
    fn queryCallback(kind: u32, shape: u32, data: u32, count: u32) -> f32;
}
#[export_name = "reserveTreeWork"]
pub extern "C" fn reserve(depth: usize, words: usize) -> *mut u32 {
    unsafe {
        assert!(depth < 64);
        SCRATCH[depth].reserve(words * 4);
        SCRATCH[depth].ptr as *mut u32
    }
}
#[export_name = "treeMutate"]
pub unsafe extern "C" fn mutate(
    ptr: *mut u32,
    cap: usize,
    state: *mut u32,
    op: u32,
    id: i32,
    lx: f32,
    ly: f32,
    lz: f32,
    hx: f32,
    hy: f32,
    hz: f32,
    ch: u32,
    cl: u32,
    ud: u32,
    udh: u32,
) -> i32 {
    let pool = slice::from_raw_parts_mut(ptr, cap * STRIDE);
    let root = &mut *(state as *mut i32);
    let count = &mut *(state.add(1) as *mut usize);
    let free = &mut *(state.add(2) as *mut i32);
    let lo = [lx, ly, lz];
    let hi = [hx, hy, hz];
    let result = match op {
        0 => {
            *state.add(3) += 1;
            tree::create_proxy(
                pool,
                root,
                count,
                free,
                lo,
                hi,
                ch,
                cl,
                (ud as u64) | ((udh as u64) << 32),
            )
        }
        1 => {
            tree::move_proxy(pool, root, count, free, id, lo, hi);
            id
        }
        2 => {
            tree::enlarge_proxy(pool, id, lo, hi);
            id
        }
        3 => {
            tree::destroy_proxy(pool, root, count, free, id);
            *state.add(3) -= 1;
            id
        }
        4 => {
            let n = (*state.add(3) as usize).max(1);
            let scratch = if ptr == state.add(6) {
                ptr.add(cap * STRIDE)
            } else {
                state.add(6)
            };
            let indices = slice::from_raw_parts_mut(scratch as *mut i32, n);
            let centers = slice::from_raw_parts_mut(scratch.add(n) as *mut f32, n * 3);
            let mut gather = [0; STACK_SIZE];
            let mut build = [0; STACK_SIZE * 5];
            let mut rb = Rebuild {
                node_count: count,
                free_list: free,
                leaf_indices: indices,
                leaf_centers: centers,
                gather_stack: &mut gather,
                build_stack: &mut build,
            };
            *root = tree::rebuild(pool, *root, *state.add(3) as usize, id != 0, &mut rb);
            *root
        }
        _ => unreachable!(),
    };
    result
}
#[export_name = "treeMutateResident"]
pub unsafe extern "C" fn mutate_resident(
    index: usize,
    op: u32,
    id: i32,
    lx: f32,
    ly: f32,
    lz: f32,
    hx: f32,
    hy: f32,
    hz: f32,
    ch: u32,
    cl: u32,
    ud: u32,
    udh: u32,
) -> i32 {
    unsafe {
        mutate_resident_in_world(
            crate::regions::active(),
            index,
            op,
            id,
            lx,
            ly,
            lz,
            hx,
            hy,
            hz,
            ch,
            cl,
            ud,
            udh,
        )
    }
}

pub unsafe extern "C" fn mutate_resident_in_world(
    world_index: usize,
    index: usize,
    op: u32,
    id: i32,
    lx: f32,
    ly: f32,
    lz: f32,
    hx: f32,
    hy: f32,
    hz: f32,
    ch: u32,
    cl: u32,
    ud: u32,
    udh: u32,
) -> i32 {
    let state = crate::broad::tree_state(world_index, index);
    let ptr = crate::broad::tree_ptr(world_index, index);
    if op == 4 {
        let count = *state.add(3) as usize;
        if count == 0 { return *state as i32; }
        let (indices, centers, n) = crate::broad::rebuild_scratch(world_index, index, count);
        let mut gather = [0; STACK_SIZE];
        let mut build = [0; STACK_SIZE * 5];
        let mut rb = Rebuild {
            node_count: &mut *(state.add(1) as *mut usize),
            free_list: &mut *(state.add(2) as *mut i32),
            leaf_indices: slice::from_raw_parts_mut(indices, n),
            leaf_centers: slice::from_raw_parts_mut(centers, n * 3),
            gather_stack: &mut gather,
            build_stack: &mut build,
        };
        let pool = slice::from_raw_parts_mut(ptr, crate::broad::tree_cap(world_index, index) * STRIDE);
        let root = tree::rebuild(pool, *state as i32, *state.add(3) as usize, id != 0, &mut rb);
        *state = root as u32;
        return root;
    }
    mutate(
        ptr,
        crate::broad::tree_cap(world_index, index),
        state,
        op,
        id,
        lx,
        ly,
        lz,
        hx,
        hy,
        hz,
        ch,
        cl,
        ud,
        udh,
    )
}

#[export_name = "treeCreateProxy"]
pub unsafe extern "C" fn create_proxy(
    index: usize,
    lx: f32,
    ly: f32,
    lz: f32,
    hx: f32,
    hy: f32,
    hz: f32,
    ch: u32,
    cl: u32,
    user: u32,
) -> i32 {
    unsafe {
        create_proxy_in_world(
            crate::regions::active(),
            index,
            lx,
            ly,
            lz,
            hx,
            hy,
            hz,
            ch,
            cl,
            user,
        )
    }
}

pub unsafe extern "C" fn create_proxy_in_world(
    world_index: usize,
    index: usize,
    lx: f32,
    ly: f32,
    lz: f32,
    hx: f32,
    hy: f32,
    hz: f32,
    ch: u32,
    cl: u32,
    user: u32,
) -> i32 {
    let cap = crate::broad::tree_cap(world_index, index);
    let state = crate::broad::tree_state(world_index, index);
    if cap - (*state.add(1) as usize) < 2 {
        let next = if cap == 0 { 31 } else { cap + (cap >> 1) };
        let mut caps = [
            crate::broad::tree_cap(world_index, 0),
            crate::broad::tree_cap(world_index, 1),
            crate::broad::tree_cap(world_index, 2),
        ];
        caps[index] = next;
        crate::broad::reserve_broad_in_world(
            world_index,
            caps[0],
            caps[1],
            caps[2],
            crate::broad::set_cap(world_index),
        );
        let state = crate::broad::tree_state(world_index, index);
        let ptr = crate::broad::tree_ptr(world_index, index);
        for i in cap..next {
            *ptr.add(i * STRIDE + 10) = if i + 1 == next {
                u32::MAX
            } else {
                (i + 1) as u32
            };
        }
        if *state.add(2) == u32::MAX {
            *state.add(2) = cap as u32;
        } else {
            let mut i = *state.add(2) as usize;
            while *ptr.add(i * STRIDE + 10) != u32::MAX {
                i = *ptr.add(i * STRIDE + 10) as usize;
            }
            *ptr.add(i * STRIDE + 10) = cap as u32;
        }
    }
    mutate_resident_in_world(
        world_index,
        index,
        0,
        0,
        lx,
        ly,
        lz,
        hx,
        hy,
        hz,
        ch,
        cl,
        user,
        0,
    )
}
#[export_name = "treeDestroyProxy"]
pub unsafe extern "C" fn destroy_proxy(index: usize, id: i32) {
    unsafe { destroy_proxy_in_world(crate::regions::active(), index, id) }
}

pub unsafe extern "C" fn destroy_proxy_in_world(world_index: usize, index: usize, id: i32) {
    mutate_resident_in_world(
        world_index,
        index,
        3,
        id,
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
#[export_name = "treeEnlargeProxy"]
pub unsafe extern "C" fn enlarge_proxy(
    index: usize,
    id: i32,
    lx: f32,
    ly: f32,
    lz: f32,
    hx: f32,
    hy: f32,
    hz: f32,
) {
    unsafe { enlarge_proxy_in_world(crate::regions::active(), index, id, lx, ly, lz, hx, hy, hz) }
}

pub unsafe extern "C" fn enlarge_proxy_in_world(
    world_index: usize,
    index: usize,
    id: i32,
    lx: f32,
    ly: f32,
    lz: f32,
    hx: f32,
    hy: f32,
    hz: f32,
) {
    mutate_resident_in_world(
        world_index,
        index,
        2,
        id,
        lx,
        ly,
        lz,
        hx,
        hy,
        hz,
        0,
        0,
        0,
        0,
    );
}
#[export_name = "treeMoveProxy"]
pub unsafe extern "C" fn move_proxy(
    index: usize,
    id: i32,
    lx: f32,
    ly: f32,
    lz: f32,
    hx: f32,
    hy: f32,
    hz: f32,
) {
    unsafe { move_proxy_in_world(crate::regions::active(), index, id, lx, ly, lz, hx, hy, hz) }
}

pub unsafe extern "C" fn move_proxy_in_world(
    world_index: usize,
    index: usize,
    id: i32,
    lx: f32,
    ly: f32,
    lz: f32,
    hx: f32,
    hy: f32,
    hz: f32,
) {
    mutate_resident_in_world(
        world_index,
        index,
        1,
        id,
        lx,
        ly,
        lz,
        hx,
        hy,
        hz,
        0,
        0,
        0,
        0,
    );
}

/// Box3D solver.c: awake sim order, then each body's head-to-next shape order.
#[export_name = "treeEnlargePass"]
pub unsafe extern "C" fn enlarge_pass(count: usize, bullets: u32) {
    unsafe { enlarge_pass_in_world(crate::regions::active(), count, bullets) }
}

pub unsafe extern "C" fn enlarge_pass_in_world(world_index: usize, count: usize, bullets: u32) {
    let sim2 = crate::bodies::sim2_base(world_index) as *mut u32;
    let shapes = crate::shapes::col(world_index);
    let fat = crate::shapes::col_f(world_index);
    let enlarged = crate::arena::enlarged_sims(world_index);
    let blocks = if bullets == 0 { count.div_ceil(64) } else { crate::continuous::bullet_count() };
    for block in 0..blocks {
        let mut mask = if bullets == 0 {
            *enlarged.bits.add(block)
        } else {
            1
        };
        while mask != 0 {
            let i = if bullets == 0 { block * 64 + mask.trailing_zeros() as usize } else { crate::continuous::bullet_body(block) };
            mask &= mask - 1;
            if i >= count {
                break;
            }
            let row = sim2.add(i * SIM2_STRIDE);
            let flags = *row.add(S2_FLAGS);
            let bullet = flags & (IS_FAST | IS_BULLET) == (IS_FAST | IS_BULLET);
            if bullets != 0 && (!bullet || flags & ENLARGE_BOUNDS == 0) {
                continue;
            }
            let body_id = *row.add(S2_BODY_ID) as usize;
            let mut id = crate::bodies::record(world_index, body_id).head_shape_id as u32;
            while id != u32::MAX {
                let o = id as usize * crate::shapes::SHAPE_STRIDE;
                let key = shapes.get(o + crate::shapes::S_PROXY_KEY);
                if bullets == 0 && bullet {
                    crate::broad::buffer_move_in_world(world_index, key);
                } else if shapes.get(o + crate::shapes::S_FLAGS) & crate::shapes::ENLARGED_FLAG != 0
                {
                    let index = (key & 3) as usize;
                    let b = o + crate::shapes::S_FAT_AABB;
                    let pool = slice::from_raw_parts_mut(
                        crate::broad::tree_ptr(world_index, index),
                        crate::broad::tree_cap(world_index, index) * STRIDE,
                    );
                    tree::enlarge_proxy(
                        pool,
                        (key >> 2) as i32,
                        [fat.get(b), fat.get(b + 1), fat.get(b + 2)],
                        [fat.get(b + 3), fat.get(b + 4), fat.get(b + 5)],
                    );
                    let flags = shapes.get(o + crate::shapes::S_FLAGS);
                    shapes.set(
                        o + crate::shapes::S_FLAGS,
                        flags & !crate::shapes::ENLARGED_FLAG,
                    );
                    if bullets == 0 {
                        crate::broad::buffer_move_in_world(world_index, key);
                    }
                }
                id = shapes.get(o + crate::shapes::S_NEXT);
            }
            if bullets != 0 {
                *row.add(S2_FLAGS) &= !ENLARGE_BOUNDS;
            }
        }
    }
}

#[export_name = "treeQuery"]
pub unsafe extern "C" fn query(
    ptr: *const u32,
    cap: usize,
    root: i32,
    count: usize,
    lx: f32,
    ly: f32,
    lz: f32,
    hx: f32,
    hy: f32,
    hz: f32,
    mh: u32,
    ml: u32,
    all: u32,
    state: *mut u32,
) {
    let pool = slice::from_raw_parts(ptr, cap * STRIDE);
    let mut stack = [0; STACK_SIZE];
    let (nv, lv) = tree::query(
        pool,
        root,
        count,
        [lx, ly, lz],
        [hx, hy, hz],
        mh,
        ml,
        all != 0,
        &mut stack,
        |id, ud| queryCallback(9, id as u32, ud, pool[id as usize * STRIDE + 9]) != 0.0,
    );
    *state.add(4) = nv;
    *state.add(5) = lv;
}
