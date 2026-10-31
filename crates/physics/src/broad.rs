//! World-local trees and pair membership. Moved flags belong to the tree hierarchy.
use crate::regions::{self, Columns, MAX_WORLDS};
pub const TREE_STRIDE: usize = 12;
const TREE_STATE_WORDS: usize = 6;
const ITEMS: usize = 3;
const N_BROAD: usize = 4;
#[derive(Clone, Copy)]
struct Broad {
    columns: Columns<N_BROAD>,
    tree: [usize; 3],
    owned: [usize; 3],
    set: usize,
    set_count: usize,
}
impl Broad {
    const EMPTY: Self = Self {
        columns: Columns::EMPTY,
        tree: [0; 3],
        owned: [0; 3],
        set: 0,
        set_count: 0,
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
// The authoring/prepare thread initializes trees before any traversal task is scheduled.
pub unsafe fn tree(world_index: usize, i: usize) -> &'static crate::dynamic_tree::Tree {
    if WORLDS[world_index].owned[i] == 0 {
        tree_mut(world_index, i);
    }
    &*(WORLDS[world_index].owned[i] as *const crate::dynamic_tree::Tree)
}
pub unsafe fn tree_mut(world_index: usize, i: usize) -> &'static mut crate::dynamic_tree::Tree {
    let w = &mut WORLDS[world_index];
    if w.owned[i] == 0 {
        w.owned[i] = Box::into_raw(Box::new(crate::dynamic_tree::Tree::new(16))) as usize;
    }
    &mut *(w.owned[i] as *mut crate::dynamic_tree::Tree)
}
pub unsafe fn sync_tree(world_index: usize, i: usize) {
    let t = tree(world_index, i);
    let w = &mut WORLDS[world_index];
    let cap = t.nodes.len();
    w.columns.reserve(i, tree_bytes(cap));
    w.tree[i] = cap;
    let s = w.columns.layout[i] as *mut u32;
    *s = 0;
    *s.add(1) = t.end as u32;
    *s.add(2) = t.pair_free as u32;
    *s.add(3) = t.count as u32;
}
pub fn tree_state(world_index: usize, i: usize) -> *mut u32 {
    unsafe {
        sync_tree(world_index, i);
    }
    base(world_index, i) as *mut u32
}
pub fn tree_ptr(world_index: usize, i: usize) -> *mut u32 {
    unsafe {
        let ptr = tree_state(world_index, i).add(TREE_STATE_WORDS);
        let t = tree(world_index, i);
        for (index, node) in t.nodes.iter().enumerate() {
            let n = ptr.add(index * TREE_STRIDE);
            core::ptr::write_bytes(n, 0, TREE_STRIDE);
            if index >= t.end || node.flags == crate::dynamic_tree::EMPTY {
                continue;
            }
            for (k, value) in [
                node.lo.x, node.lo.y, node.lo.z, node.hi.x, node.hi.y, node.hi.z,
            ]
            .into_iter()
            .enumerate()
            {
                *n.add(k) = value.to_bits();
            }
            *n.add(10) = t.parents[index] as u32;
            *n.add(11) = 1
                | if node.leaf() { 4 } else { 0 }
                | if node.flags & crate::dynamic_tree::MOVED != 0 {
                    2
                } else {
                    0
                }
                | ((node.height() as u32) << 16);
            if node.leaf() {
                let p = t.proxies[node.index()];
                *n.add(6) = (p.category >> 32) as u32;
                *n.add(7) = p.category as u32;
                *n.add(8) = p.user as u32;
                *n.add(9) = (p.user >> 32) as u32;
            } else {
                *n.add(6) = u32::MAX;
                *n.add(7) = u32::MAX;
                *n.add(8) = node.index() as u32;
                *n.add(9) = (node.index() + 1) as u32;
            }
        }
        ptr
    }
}
pub fn tree_cap(world_index: usize, i: usize) -> usize {
    unsafe { world(world_index).tree[i] }
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
    let mut old = w.columns.replace_zeroed(
        ITEMS,
        2 * capacity * core::mem::size_of::<crate::table::Item>(),
    );
    w.set = 2 * capacity;
    crate::table::transfer_items(
        core::slice::from_raw_parts(old.ptr as *const crate::table::Item, capacity),
        core::slice::from_raw_parts_mut(w.columns.layout[ITEMS] as *mut crate::table::Item, w.set),
    );
    old.release();
}
#[export_name = "broadTestOverlap"]
pub unsafe extern "C" fn test_overlap(a: u32, b: u32) -> u32 {
    unsafe { test_overlap_in_world(crate::regions::active(), a, b) }
}

pub unsafe extern "C" fn test_overlap_in_world(world_index: usize, a: u32, b: u32) -> u32 {
    let node = |key: u32| {
        let t = tree(world_index, (key & 3) as usize);
        t.nodes[t.proxies[(key >> 2) as usize].node as usize]
    };
    let a = node(a);
    let b = node(b);
    crate::mesh_query::aabb_overlaps(a.lo, a.hi, b.lo, b.hi) as u32
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
    tree_mut(world_index, (key & 3) as usize).mark_moved((key >> 2) as usize);
}
#[export_name = "broadIsMoved"]
pub unsafe extern "C" fn is_moved(key: u32) -> u32 {
    let t = tree(crate::regions::active(), (key & 3) as usize);
    let node = t.proxies[(key >> 2) as usize].node;
    (node >= 0 && t.nodes[node as usize].flags & crate::dynamic_tree::MOVED != 0) as u32
}
#[export_name = "broadLayoutPtr"]
pub extern "C" fn broad_layout_ptr() -> *const u32 {
    broad_layout_ptr_in_world(crate::regions::active())
}

pub extern "C" fn broad_layout_ptr_in_world(world_index: usize) -> *const u32 {
    unsafe {
        for i in 0..3 {
            if world(world_index).owned[i] != 0 {
                tree_ptr(world_index, i);
            }
        }
        world(world_index).columns.layout.as_ptr()
    }
}
#[export_name = "broadTreeCap"]
pub extern "C" fn broad_tree_cap(i: usize) -> usize {
    broad_tree_cap_in_world(crate::regions::active(), i)
}

pub extern "C" fn broad_tree_cap_in_world(world_index: usize, i: usize) -> usize {
    tree_cap(world_index, i)
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
            if w.owned[i] == 0 && tree[i] != 0 {
                w.owned[i] = Box::into_raw(Box::new(crate::dynamic_tree::Tree::new(
                    tree[i].div_ceil(2),
                ))) as usize;
            }
            w.columns.reserve(i, tree_bytes(tree[i]));
            if w.tree[i] == 0 && tree[i] != 0 {
                let state = w.columns.layout[i] as *mut u32;
                *state = u32::MAX;
                *state.add(2) = u32::MAX;
            }
        }
        w.columns
            .reserve(ITEMS, set * core::mem::size_of::<crate::table::Item>());
        w.tree = tree;
        w.set = set;
        1
    }
}
pub unsafe fn reset(id: usize) {
    for ptr in WORLDS[id].owned {
        if ptr != 0 {
            drop(Box::from_raw(ptr as *mut crate::dynamic_tree::Tree));
        }
    }
    WORLDS[id].columns.release();
    WORLDS[id] = Broad::EMPTY;
}
pub unsafe fn snapshot(id: usize, out: &mut Vec<u8>) {
    for i in 0..3 {
        tree(id, i).snapshot(out);
    }
    let w = &WORLDS[id];
    for value in w.tree {
        regions::write_word(out, value);
    }
    regions::write_word(out, w.set);
    regions::write_word(out, w.set_count);
    // Inspection columns are derived, never a second simulation image.
    let mut bytes = [0; N_BROAD];
    bytes[ITEMS] = w.set * core::mem::size_of::<crate::table::Item>();
    w.columns.snapshot_prefix(out, bytes);
}
pub unsafe fn restore(id: usize, input: &mut &[u8]) {
    for i in 0..3 {
        let t = crate::dynamic_tree::Tree::restore(input);
        *tree_mut(id, i) = t;
    }
    let w = &mut WORLDS[id];
    for value in &mut w.tree {
        *value = regions::read_word(input);
    }
    w.set = regions::read_word(input);
    w.set_count = regions::read_word(input);
    w.columns.restore(input);
}
