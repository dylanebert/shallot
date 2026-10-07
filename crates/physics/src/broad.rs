//! World-local trees, pair membership and pending proxy moves.
use crate::regions::{self, Columns, MAX_WORLDS};
pub const TREE_STRIDE: usize = 12;
const TREE_STATE_WORDS: usize = 6;
const ITEMS: usize = 3;
const MOVE: usize = 6;
const BITS: usize = 7;
const N_BROAD: usize = 10;
#[derive(Clone, Copy)]
struct Broad {
    columns: Columns<N_BROAD>,
    tree: [usize; 3],
    set: usize,
    set_count: usize,
    rebuild: [crate::regions::Buffer; 6],
    rebuild_capacity: [usize; 3],
    bit_count: [usize; 3],
    bit_capacity: [usize; 3],
}
impl Broad {
    const EMPTY: Self = Self {
        columns: Columns::EMPTY,
        tree: [0; 3],
        set: 0,
        set_count: 0,
        rebuild: [crate::regions::Buffer::EMPTY; 6],
        rebuild_capacity: [0; 3],
        bit_count: [0; 3],
        bit_capacity: [0; 3],
    };
}
static mut WORLDS: [Broad; MAX_WORLDS] = [Broad::EMPTY; MAX_WORLDS];
unsafe fn world(world_index: usize) -> &'static Broad {
    &WORLDS[world_index]
}
fn base(world_index: usize, column: usize) -> usize {
    unsafe { world(world_index).columns.layout[column] as usize }
}
fn tree_bytes(cap: usize) -> usize {
    if cap == 0 {
        0
    } else {
        (cap * TREE_STRIDE + TREE_STATE_WORDS) * 4
    }
}
pub fn tree_state(world_index: usize, i: usize) -> *mut u32 {
    base(world_index, i) as *mut u32
}
pub fn tree_ptr(world_index: usize, i: usize) -> *mut u32 {
    unsafe { tree_state(world_index, i).add(TREE_STATE_WORDS) }
}
pub fn tree_cap(world_index: usize, i: usize) -> usize {
    unsafe { world(world_index).tree[i] }
}
pub unsafe fn rebuild_scratch(world_index: usize, index: usize, count: usize) -> (*mut i32, *mut f32, usize) {
    let w = &mut WORLDS[world_index];
    if count > w.rebuild_capacity[index] {
        let capacity = count + count / 2;
        for (slot, stride) in [(2 * index, 4), (2 * index + 1, 12)] {
            w.rebuild[slot].release();
            w.rebuild[slot] = crate::regions::Buffer::allocate(capacity * stride);
            (w.rebuild[slot].ptr as *mut u8).write_bytes(0, capacity * stride);
        }
        w.rebuild_capacity[index] = capacity;
    }
    (w.rebuild[2 * index].ptr as *mut i32, w.rebuild[2 * index + 1].ptr as *mut f32, w.rebuild_capacity[index])
}
pub fn set_count(world_index: usize) -> usize {
    unsafe { world(world_index).set_count }
}
pub fn change_set_count(world_index: usize, delta: isize) {
    unsafe {
        WORLDS[world_index].set_count = (world(world_index).set_count as isize + delta) as usize;
    }
}
pub fn set_cap(world_index: usize) -> usize {
    unsafe { world(world_index).set }
}
pub fn set_items(world_index: usize) -> *mut crate::table::Item {
    base(world_index, ITEMS) as *mut crate::table::Item
}

pub unsafe fn grow_set(world_index: usize) {
    let w = &mut WORLDS[world_index];
    let capacity = w.set;
    let mut old = w.columns.replace_zeroed(ITEMS, 2 * capacity * core::mem::size_of::<crate::table::Item>());
    w.set = 2 * capacity;
    crate::table::transfer_items(
        core::slice::from_raw_parts(old.ptr as *const crate::table::Item, capacity),
        core::slice::from_raw_parts_mut(w.columns.layout[ITEMS] as *mut crate::table::Item, w.set),
    );
    old.release();
}
pub fn move_ptr(world_index: usize) -> *mut u32 {
    unsafe { (base(world_index, MOVE) as *mut u32).add(1) }
}
pub fn move_count(world_index: usize) -> usize {
    unsafe { *(base(world_index, MOVE) as *const u32) as usize }
}
pub fn bits_ptr(world_index: usize, i: usize) -> *mut u64 {
    base(world_index, BITS + i) as *mut u64
}
pub fn bits_words(world_index: usize, i: usize) -> usize {
    unsafe { world(world_index).bit_count[i] }
}
#[export_name = "broadBitsCapacity"]
pub extern "C" fn bits_capacity(i: usize) -> usize {
    unsafe { world(crate::regions::active()).bit_capacity[i] }
}
unsafe fn grow_bits(world_index: usize, i: usize, count: usize) {
    let w = &mut WORLDS[world_index];
    if count > w.bit_capacity[i] {
        let capacity = count + count / 2;
        let mut old = w.columns.replace_zeroed(BITS + i, capacity * 8);
        core::ptr::copy_nonoverlapping(old.ptr as *const u64, w.columns.layout[BITS + i] as *mut u64, w.bit_capacity[i]);
        old.release();
        w.bit_capacity[i] = capacity;
    }
    w.bit_count[i] = count;
}
#[export_name = "broadTestOverlap"]
pub unsafe extern "C" fn test_overlap(a: u32, b: u32) -> u32 {
    unsafe { test_overlap_in_world(crate::regions::active(), a, b) }
}

pub unsafe extern "C" fn test_overlap_in_world(world_index: usize, a: u32, b: u32) -> u32 {
    let pool = |key: u32| {
        core::slice::from_raw_parts(
            tree_ptr(world_index, (key & 3) as usize),
            tree_cap(world_index, (key & 3) as usize) * TREE_STRIDE,
        )
    };
    let (al, ah) = crate::tree::node_aabb(pool(a), (a >> 2) as i32);
    let (bl, bh) = crate::tree::node_aabb(pool(b), (b >> 2) as i32);
    (al[0] <= bh[0]
        && bl[0] <= ah[0]
        && al[1] <= bh[1]
        && bl[1] <= ah[1]
        && al[2] <= bh[2]
        && bl[2] <= ah[2]) as u32
}
#[export_name = "broadCreateProxy"]
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
    shape: u32,
    force: u32,
) -> u32 {
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
            shape,
            force,
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
    shape: u32,
    force: u32,
) -> u32 {
    let id = crate::treework::create_proxy_in_world(
        world_index,
        index,
        lx,
        ly,
        lz,
        hx,
        hy,
        hz,
        ch,
        cl,
        shape,
    );
    let key = ((id as u32) << 2) | index as u32;
    if index != 0 || force != 0 {
        buffer_move_in_world(world_index, key);
    }
    key
}
#[export_name = "broadDestroyProxy"]
pub unsafe extern "C" fn destroy_proxy(key: u32) {
    unsafe { destroy_proxy_in_world(crate::regions::active(), key) }
}

pub unsafe extern "C" fn destroy_proxy_in_world(world_index: usize, key: u32) {
    unbuffer_move(world_index, key);
    crate::treework::destroy_proxy_in_world(world_index, (key & 3) as usize, (key >> 2) as i32);
}
#[export_name = "broadMoveProxy"]
pub unsafe extern "C" fn move_proxy(
    key: u32,
    lx: f32,
    ly: f32,
    lz: f32,
    hx: f32,
    hy: f32,
    hz: f32,
) {
    unsafe { move_proxy_in_world(crate::regions::active(), key, lx, ly, lz, hx, hy, hz) }
}

pub unsafe extern "C" fn move_proxy_in_world(
    world_index: usize,
    key: u32,
    lx: f32,
    ly: f32,
    lz: f32,
    hx: f32,
    hy: f32,
    hz: f32,
) {
    crate::treework::move_proxy_in_world(
        world_index,
        (key & 3) as usize,
        (key >> 2) as i32,
        lx,
        ly,
        lz,
        hx,
        hy,
        hz,
    );
    buffer_move_in_world(world_index, key);
}
#[export_name = "broadEnlargeProxy"]
pub unsafe extern "C" fn enlarge_proxy(
    key: u32,
    lx: f32,
    ly: f32,
    lz: f32,
    hx: f32,
    hy: f32,
    hz: f32,
) {
    unsafe { enlarge_proxy_in_world(crate::regions::active(), key, lx, ly, lz, hx, hy, hz) }
}

pub unsafe extern "C" fn enlarge_proxy_in_world(
    world_index: usize,
    key: u32,
    lx: f32,
    ly: f32,
    lz: f32,
    hx: f32,
    hy: f32,
    hz: f32,
) {
    assert_ne!(key & 3, 0);
    crate::treework::enlarge_proxy_in_world(
        world_index,
        (key & 3) as usize,
        (key >> 2) as i32,
        lx,
        ly,
        lz,
        hx,
        hy,
        hz,
    );
    buffer_move_in_world(world_index, key);
}
#[export_name = "broadBufferMove"]
pub unsafe extern "C" fn buffer_move(key: u32) {
    unsafe { buffer_move_in_world(crate::regions::active(), key) }
}

pub unsafe extern "C" fn buffer_move_in_world(world_index: usize, key: u32) {
    let i = (key & 3) as usize;
    let id = (key >> 2) as usize;
    if id / 64 >= bits_words(world_index, i) {
        grow_bits(world_index, i, id / 64 + 1);
    }
    let p = bits_ptr(world_index, i).add(id / 64);
    let mask = 1u64 << (id & 63);
    if *p & mask == 0 {
        *p |= mask;
        let count = base(world_index, MOVE) as *mut u32;
        *move_ptr(world_index).add(*count as usize) = key;
        *count += 1;
    }
}
pub unsafe fn unbuffer_move(world_index: usize, key: u32) {
    let i = (key & 3) as usize;
    let id = (key >> 2) as usize;
    if id / 64 >= bits_words(world_index, i) { return; }
    let p = bits_ptr(world_index, i).add(id / 64);
    let mask = 1u64 << (id & 63);
    if *p & mask == 0 { return; }
    *p &= !mask;
    let count = base(world_index, MOVE) as *mut u32;
    for n in 0..*count as usize {
        if *move_ptr(world_index).add(n) == key {
            *count -= 1;
            *move_ptr(world_index).add(n) = *move_ptr(world_index).add(*count as usize);
            break;
        }
    }
}
#[export_name = "broadClearMoved"]
pub unsafe extern "C" fn clear_moved(index: usize, id: usize) {
    unsafe { clear_moved_in_world(crate::regions::active(), index, id) }
}

pub unsafe extern "C" fn clear_moved_in_world(world_index: usize, index: usize, id: usize) {
    if id / 64 < bits_words(world_index, index) {
        *bits_ptr(world_index, index).add(id / 64) &= !(1u64 << (id & 63));
    }
}
#[export_name = "broadClearMoves"]
pub unsafe extern "C" fn clear_moves() {
    unsafe { clear_moves_in_world(crate::regions::active()) }
}

pub unsafe extern "C" fn clear_moves_in_world(world_index: usize) {
    for n in 0..move_count(world_index) {
        let key = *move_ptr(world_index).add(n);
        let id = (key >> 2) as usize;
        *bits_ptr(world_index, (key & 3) as usize).add(id / 64) &= !(1u64 << (id & 63));
    }
    *(base(world_index, MOVE) as *mut u32) = 0;
}
#[export_name = "broadLayoutPtr"]
pub extern "C" fn broad_layout_ptr() -> *const u32 {
    broad_layout_ptr_in_world(crate::regions::active())
}

pub extern "C" fn broad_layout_ptr_in_world(world_index: usize) -> *const u32 {
    unsafe { world(world_index).columns.layout.as_ptr() }
}
#[export_name = "broadTreeCap"]
pub extern "C" fn broad_tree_cap(i: usize) -> usize {
    broad_tree_cap_in_world(crate::regions::active(), i)
}

pub extern "C" fn broad_tree_cap_in_world(world_index: usize, i: usize) -> usize {
    tree_cap(world_index, i)
}
#[export_name = "broadSetCap"]
pub extern "C" fn broad_set_cap() -> usize {
    broad_set_cap_in_world(crate::regions::active())
}

pub extern "C" fn broad_set_cap_in_world(world_index: usize) -> usize {
    set_cap(world_index)
}
#[export_name = "reserveBroad"]
pub extern "C" fn reserve_broad(cap_s: usize, cap_k: usize, cap_d: usize, set_cap: usize) -> u32 {
    reserve_broad_in_world(crate::regions::active(), cap_s, cap_k, cap_d, set_cap)
}

pub extern "C" fn reserve_broad_in_world(
    world_index: usize,
    cap_s: usize,
    cap_k: usize,
    cap_d: usize,
    set_cap: usize,
) -> u32 {
    unsafe {
        let w = &mut WORLDS[world_index];
        let tree = [
            cap_s.max(w.tree[0]),
            cap_k.max(w.tree[1]),
            cap_d.max(w.tree[2]),
        ];
        let set = set_cap.max(w.set);
        if tree == w.tree && set == w.set {
            return 0;
        }
        for i in 0..3 {
            w.columns.reserve(i, tree_bytes(tree[i]));
            if w.tree[i] == 0 && tree[i] != 0 {
                let state = w.columns.layout[i] as *mut u32;
                *state = u32::MAX;
                *state.add(2) = u32::MAX;
            }
            if w.bit_capacity[i] == 0 && tree[i] != 0 {
                let mut old = w.columns.replace_zeroed(BITS + i, 8);
                old.release();
                w.bit_capacity[i] = 1;
            }
        }
        w.columns.reserve(ITEMS, set * core::mem::size_of::<crate::table::Item>());
        w.columns
            .reserve(MOVE, (1 + tree.iter().sum::<usize>()) * 4);
        w.tree = tree;
        w.set = set;
        1
    }
}
pub unsafe fn reset(id: usize) {
    for buffer in &mut WORLDS[id].rebuild { buffer.release(); }
    WORLDS[id].columns.release();
    WORLDS[id] = Broad::EMPTY;
}
pub unsafe fn snapshot(id: usize, out: &mut Vec<u8>) {
    let w = &WORLDS[id];
    for value in w.tree {
        regions::write_word(out, value);
    }
    regions::write_word(out, w.set);
    regions::write_word(out, w.set_count);
    for value in w.bit_count { regions::write_word(out, value); }
    let mut bytes = [0; N_BROAD];
    for i in 0..3 {
        bytes[i] = tree_bytes(w.tree[i]);
        bytes[BITS + i] = w.bit_count[i] * 8;
    }
    bytes[ITEMS] = w.set * core::mem::size_of::<crate::table::Item>();
    bytes[MOVE] = if w.tree.iter().sum::<usize>() == 0 { 0 } else { (1 + move_count(id)) * 4 };
    w.columns.snapshot_prefix(out, bytes);
}
pub unsafe fn restore(id: usize, input: &mut &[u8]) {
    let w = &mut WORLDS[id];
    for value in &mut w.tree {
        *value = regions::read_word(input);
    }
    w.set = regions::read_word(input);
    w.set_count = regions::read_word(input);
    for value in &mut w.bit_count { *value = regions::read_word(input); }
    w.bit_capacity = w.bit_count;
    w.columns.restore(input);
    w.columns.reserve(MOVE, (1 + w.tree.iter().sum::<usize>()) * 4);
}
