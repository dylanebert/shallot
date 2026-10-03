//! World-local trees, pair membership, body filters and pending proxy moves.
use crate::regions::{self, Columns, MAX_WORLDS};
pub const TREE_STRIDE: usize = 12;
const TREE_STATE_WORDS: usize = 6;
const KEY_HI: usize = 3;
const KEY_LO: usize = 4;
const HASHES: usize = 5;
const BODY_FILTER: usize = 6;
const MOVE: usize = 7;
const BITS: usize = 8;
const N_BROAD: usize = 11;
#[derive(Clone, Copy)]
struct Broad {
    columns: Columns<N_BROAD>,
    tree: [usize; 3],
    set: usize,
    filter: usize,
}
impl Broad {
    const EMPTY: Self = Self {
        columns: Columns::EMPTY,
        tree: [0; 3],
        set: 0,
        filter: 0,
    };
}
static mut WORLDS: [Broad; MAX_WORLDS] = [Broad::EMPTY; MAX_WORLDS];
unsafe fn world() -> &'static Broad {
    &WORLDS[regions::active()]
}
fn base(column: usize) -> usize {
    unsafe { world().columns.layout[column] as usize }
}
fn tree_bytes(cap: usize) -> usize {
    if cap == 0 {
        0
    } else {
        (cap * TREE_STRIDE + TREE_STATE_WORDS) * 4
    }
}
fn bit_bytes(cap: usize) -> usize {
    cap.div_ceil(32) * 4
}
fn filter_bytes(cap: usize) -> usize {
    if cap == 0 {
        0
    } else {
        (1 + 3 * cap) * 4
    }
}
pub fn tree_state(i: usize) -> *mut u32 {
    base(i) as *mut u32
}
pub fn tree_ptr(i: usize) -> *mut u32 {
    unsafe { tree_state(i).add(TREE_STATE_WORDS) }
}
pub fn tree_cap(i: usize) -> usize {
    unsafe { world().tree[i] }
}
pub fn set_cap() -> usize {
    unsafe { world().set }
}
pub fn set_ptrs() -> (*const u32, *const u32, *const u32) {
    (
        base(KEY_HI) as *const u32,
        base(KEY_LO) as *const u32,
        base(HASHES) as *const u32,
    )
}
pub fn move_ptr() -> *mut u32 {
    unsafe { (base(MOVE) as *mut u32).add(1) }
}
pub fn move_count() -> usize {
    unsafe { *(base(MOVE) as *const u32) as usize }
}
pub fn bits_ptr(i: usize) -> *mut u32 {
    base(BITS + i) as *mut u32
}
pub fn bits_words(i: usize) -> usize {
    tree_cap(i).div_ceil(32)
}
#[export_name = "broadBufferMove"]
pub unsafe extern "C" fn buffer_move(key: u32) {
    let i = (key & 3) as usize;
    let id = (key >> 2) as usize;
    let p = bits_ptr(i).add(id / 32);
    let mask = 1 << (id & 31);
    if *p & mask == 0 {
        *p |= mask;
        let count = base(MOVE) as *mut u32;
        *move_ptr().add(*count as usize) = key;
        *count += 1;
    }
}
pub unsafe fn unbuffer_move(key: u32) {
    let i = (key & 3) as usize;
    let id = (key >> 2) as usize;
    *bits_ptr(i).add(id / 32) &= !(1 << (id & 31));
    let count = base(MOVE) as *mut u32;
    for n in 0..*count as usize {
        if *move_ptr().add(n) == key {
            *count -= 1;
            *move_ptr().add(n) = *move_ptr().add(*count as usize);
            break;
        }
    }
}
#[export_name = "broadClearMoves"]
pub unsafe extern "C" fn clear_moves() {
    for n in 0..move_count() {
        let key = *move_ptr().add(n);
        let id = (key >> 2) as usize;
        *bits_ptr((key & 3) as usize).add(id / 32) &= !(1 << (id & 31));
    }
    *(base(MOVE) as *mut u32) = 0;
}
#[export_name = "broadLayoutPtr"]
pub extern "C" fn broad_layout_ptr() -> *const u32 {
    unsafe { world().columns.layout.as_ptr() }
}
#[export_name = "broadTreeCap"]
pub extern "C" fn broad_tree_cap(i: usize) -> usize {
    tree_cap(i)
}
#[export_name = "broadSetCap"]
pub extern "C" fn broad_set_cap() -> usize {
    set_cap()
}
#[export_name = "broadBodiesFiltered"]
pub extern "C" fn bodies_filtered(a: u32, b: u32) -> u32 {
    unsafe {
        if world().filter == 0 {
            return 0;
        }
        let data = base(BODY_FILTER) as *const u32;
        let (a, b) = (a.min(b), a.max(b));
        let mut lo = 0;
        let mut hi = *data as usize;
        while lo < hi {
            let mid = (lo + hi) / 2;
            let p = data.add(1 + 3 * mid);
            if (*p, *p.add(1)) < (a, b) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        (lo < *data as usize && *data.add(1 + 3 * lo) == a && *data.add(2 + 3 * lo) == b) as u32
    }
}
#[export_name = "reserveBroad"]
pub extern "C" fn reserve_broad(
    cap_s: usize,
    cap_k: usize,
    cap_d: usize,
    set_cap: usize,
    filter_cap: usize,
) -> u32 {
    unsafe {
        let w = &mut WORLDS[regions::active()];
        let tree = [
            cap_s.max(w.tree[0]),
            cap_k.max(w.tree[1]),
            cap_d.max(w.tree[2]),
        ];
        let set = set_cap.max(w.set);
        let filter = filter_cap.max(w.filter);
        if tree == w.tree && set == w.set && filter == w.filter {
            return 0;
        }
        for i in 0..3 {
            w.columns.reserve(i, tree_bytes(tree[i]));
            w.columns.reserve(BITS + i, bit_bytes(tree[i]));
        }
        for c in [KEY_HI, KEY_LO, HASHES] {
            w.columns.reserve(c, set * 4);
        }
        w.columns.reserve(BODY_FILTER, filter_bytes(filter));
        w.columns
            .reserve(MOVE, (1 + tree.iter().sum::<usize>()) * 4);
        w.tree = tree;
        w.set = set;
        w.filter = filter;
        1
    }
}
pub unsafe fn reset(id: usize) {
    WORLDS[id].columns.release();
    WORLDS[id] = Broad::EMPTY;
}
pub unsafe fn restore_id(from: usize, to: usize) {
    reset(to);
    WORLDS[to] = WORLDS[from];
    WORLDS[from] = Broad::EMPTY;
}
