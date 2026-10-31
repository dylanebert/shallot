//! Serial authoring and post-solve tree updates over world-owned trees.
use crate::body::flags::{ENLARGE_BOUNDS, IS_BULLET, IS_FAST};
use crate::body::{BODY_ID, FLAGS, SIM_STRIDE};
use crate::regions::Buffer;
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
    let t = crate::broad::tree_mut(world_index, index);
    let lo = crate::math::Vec3::new(lx, ly, lz);
    let hi = crate::math::Vec3::new(hx, hy, hz);
    let result = match op {
        0 => t.create(
            lo,
            hi,
            cl as u64 | ((ch as u64) << 32),
            ud as u64 | ((udh as u64) << 32),
            false,
        ) as i32,
        1 => {
            t.move_proxy(id as usize, lo, hi, false);
            id
        }
        2 => {
            t.enlarge(id as usize, lo, hi);
            id
        }
        3 => {
            t.destroy(id as usize);
            id
        }
        4 => {
            t.rebuild(id != 0);
            0
        }
        _ => unreachable!(),
    };
    crate::broad::sync_tree(world_index, index);
    result
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

pub unsafe extern "C" fn enlarge_pass_in_world(world_index: usize, count: usize, bullets: u32) {
    let sim = crate::bodies::sim_base(world_index) as *mut u32;
    let shapes = crate::shapes::col(world_index);
    let fat = crate::shapes::fat_col(world_index);
    let enlarged = crate::arena::enlarged_sims(world_index);
    let blocks = if bullets == 0 {
        count.div_ceil(64)
    } else {
        crate::continuous::bullet_count()
    };
    for block in 0..blocks {
        let mut mask = if bullets == 0 {
            *enlarged.bits.add(block)
        } else {
            1
        };
        while mask != 0 {
            let i = if bullets == 0 {
                block * 64 + mask.trailing_zeros() as usize
            } else {
                crate::continuous::bullet_body(block)
            };
            mask &= mask - 1;
            if i >= count {
                break;
            }
            let row = sim.add(i * SIM_STRIDE);
            let flags = *row.add(FLAGS);
            let bullet = flags & (IS_FAST | IS_BULLET) == (IS_FAST | IS_BULLET);
            if bullets != 0 && (!bullet || flags & ENLARGE_BOUNDS == 0) {
                continue;
            }
            let body_id = *row.add(BODY_ID) as usize;
            let mut id = crate::bodies::record(world_index, body_id).head_shape_id as u32;
            while id != u32::MAX {
                let o = id as usize * crate::shapes::SHAPE_STRIDE;
                let key = shapes.get(o + crate::shapes::S_PROXY_KEY);
                if bullets == 0 && bullet {
                    // Bullets acquire their final bounds after the regular tree refit.
                } else if shapes.get(o + crate::shapes::S_FLAGS) & crate::shapes::ENLARGED_FLAG != 0
                {
                    let index = (key & 3) as usize;
                    let b = id as usize * crate::shapes::FAT_STRIDE;
                    crate::broad::tree_mut(world_index, index).enlarge(
                        (key >> 2) as usize,
                        crate::math::Vec3::new(fat.get(b), fat.get(b + 1), fat.get(b + 2)),
                        crate::math::Vec3::new(fat.get(b + 3), fat.get(b + 4), fat.get(b + 5)),
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
                *row.add(FLAGS) &= !ENLARGE_BOUNDS;
            }
        }
    }
    if bullets == 0 {
        for i in [1, 2] {
            crate::broad::tree_mut(world_index, i).refit();
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
    let _ = (ptr, root, count);
    let (nv, lv) = crate::broad::tree(crate::regions::active(), cap).query(
        crate::math::Vec3::new(lx, ly, lz),
        crate::math::Vec3::new(hx, hy, hz),
        ((mh as u64) << 32) | ml as u64,
        all != 0,
        |id, user| queryCallback(9, id as u32, user as u32, (user >> 32) as u32) != 0.0,
    );
    *state.add(4) = nv;
    *state.add(5) = lv;
}
