//! Serial tree operations over resident or caller-uploaded columns.
use crate::tree::{self, Rebuild, STACK_SIZE, STRIDE};
use std::slice;
static mut WORDS: [usize; 64] = [0; 64];
static mut END: usize = 0;
pub unsafe fn record_end(end: usize) {
    END = END.max(end);
}
#[link(wasm_import_module = "env")]
extern "C" {
    fn queryCallback(kind: u32, shape: u32, data: u32, count: u32) -> f32;
}
#[export_name = "reserveTreeWork"]
pub extern "C" fn reserve(depth: usize, words: usize) -> *mut u32 {
    unsafe {
        assert!(depth < 64);
        WORDS[depth] = (words + 1) & !1;
        let mut offset = (END.max(crate::geo::solver_base()) + 7) & !7;
        for i in 0..depth {
            offset += WORDS[i] * 4;
        }
        let end = offset + words * 4;
        let have = core::arch::wasm32::memory_size(0) * 65536;
        if end > have {
            core::arch::wasm32::memory_grow(0, (end - have + 65535) / 65536);
        }
        offset as *mut u32
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
    let s = slice::from_raw_parts_mut(state, 6);
    let mut root = s[0] as i32;
    let mut count = s[1] as usize;
    let mut free = s[2] as i32;
    let lo = [lx, ly, lz];
    let hi = [hx, hy, hz];
    let result = match op {
        0 => {
            s[3] += 1;
            tree::create_proxy(
                pool,
                &mut root,
                &mut count,
                &mut free,
                lo,
                hi,
                ch,
                cl,
                (ud as u64) | ((udh as u64) << 32),
            )
        }
        1 => {
            tree::move_proxy(pool, &mut root, &mut count, &mut free, id, lo, hi);
            id
        }
        2 => {
            tree::enlarge_proxy(pool, id, lo, hi);
            id
        }
        3 => {
            tree::destroy_proxy(pool, &mut root, &mut count, &mut free, id);
            s[3] -= 1;
            id
        }
        4 => {
            let n = (s[3] as usize).max(1);
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
            root = tree::rebuild(pool, root, s[3] as usize, id != 0, &mut rb);
            count = rb.node_count;
            free = rb.free_list;
            root
        }
        _ => unreachable!(),
    };
    s[0] = root as u32;
    s[1] = count as u32;
    s[2] = free as u32;
    result
}
#[export_name = "treeEnlargeBatch"]
pub unsafe extern "C" fn enlarge_batch(commands: *const f64, count: usize) {
    let commands = slice::from_raw_parts(commands, count * 7);
    for command in commands.chunks_exact(7) {
        let key = command[0] as i32;
        let index = (key & 3) as usize;
        let pool = slice::from_raw_parts_mut(
            crate::broad::tree_ptr(index),
            crate::broad::tree_cap(index) * STRIDE,
        );
        tree::enlarge_proxy(
            pool,
            key >> 2,
            [command[1] as f32, command[2] as f32, command[3] as f32],
            [command[4] as f32, command[5] as f32, command[6] as f32],
        );
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
