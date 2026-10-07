//! Box3D table.c: symmetric pair keys, linear probing and backward-shift deletion.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Item {
    pub key: u64,
    pub hash: u32,
    padding: u32,
}

fn item_slot(items: &[Item], key: u64, hash: u32) -> usize {
    let mask = items.len() - 1;
    let mut index = hash as usize & mask;
    while items[index].hash != 0 && items[index].key != key {
        index = (index + 1) & mask;
    }
    index
}
fn item_key(a: u32, b: u32, child: u32) -> u64 {
    ((pair_key_hi(a, b) as u64) << 32) | pair_key_lo(a, b, child) as u64
}
fn item_insert(items: &mut [Item], key: u64, hash: u32) {
    let index = item_slot(items, key, hash);
    items[index].key = key;
    items[index].hash = hash;
}
pub fn transfer_items(old: &[Item], new: &mut [Item]) {
    for item in old {
        if item.hash != 0 {
            item_insert(new, item.key, item.hash);
        }
    }
}
pub fn contains_item(items: &[Item], a: u32, b: u32, child: u32) -> bool {
    if items.is_empty() {
        return false;
    }
    let key = item_key(a, b, child);
    let hash = key_hash((key >> 32) as u32, key as u32);
    items[item_slot(items, key, hash)].key == key
}

const SHAPE_MASK: u32 = (1 << 22) - 1;
const CHILD_MASK: u32 = (1 << 20) - 1;

pub fn pair_key_hi(a: u32, b: u32) -> u32 {
    ((a.min(b) & SHAPE_MASK) << 10) | ((a.max(b) & SHAPE_MASK) >> 12)
}
pub fn pair_key_lo(a: u32, b: u32, child: u32) -> u32 {
    ((a.max(b) & 0xfff) << 20) | (child & CHILD_MASK)
}
pub fn key_hash(hi: u32, lo: u32) -> u32 {
    let mut h = ((hi as u64) << 32) | lo as u64;
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    h as u32
}
#[cfg(test)]
fn find(hi: &[u32], lo: &[u32], hashes: &[u32], a: u32, b: u32, hash: u32) -> usize {
    let mask = hashes.len() - 1;
    let mut i = hash as usize & mask;
    while hashes[i] != 0 && (hi[i] != a || lo[i] != b) {
        i = (i + 1) & mask;
    }
    i
}
#[cfg(test)]
pub fn contains(
    hi: &[u32],
    lo: &[u32],
    hashes: &[u32],
    cap: usize,
    a: u32,
    b: u32,
    child: u32,
) -> bool {
    if cap == 0 {
        return false;
    }
    let (a, b) = (pair_key_hi(a, b), pair_key_lo(a, b, child));
    let i = find(hi, lo, &hashes[..cap], a, b, key_hash(a, b));
    hi[i] == a && lo[i] == b
}
#[cfg(test)]
fn insert(hi: &mut [u32], lo: &mut [u32], hashes: &mut [u32], a: u32, b: u32, hash: u32) -> bool {
    let i = find(hi, lo, hashes, a, b, hash);
    if hashes[i] != 0 {
        return true;
    }
    hi[i] = a;
    lo[i] = b;
    hashes[i] = hash;
    false
}
#[cfg(test)]
fn remove(hi: &mut [u32], lo: &mut [u32], hashes: &mut [u32], a: u32, b: u32) -> bool {
    let mut i = find(hi, lo, hashes, a, b, key_hash(a, b));
    if hashes[i] == 0 {
        return false;
    }
    hi[i] = 0;
    lo[i] = 0;
    hashes[i] = 0;
    let mask = hashes.len() - 1;
    let mut j = i;
    loop {
        j = (j + 1) & mask;
        if hashes[j] == 0 {
            break;
        }
        let k = hashes[j] as usize & mask;
        if if i <= j {
            i < k && k <= j
        } else {
            i < k || k <= j
        } {
            continue;
        }
        hi[i] = hi[j];
        lo[i] = lo[j];
        hashes[i] = hashes[j];
        hi[j] = 0;
        lo[j] = 0;
        hashes[j] = 0;
        i = j;
    }
    true
}
fn capacity(need: usize) -> usize {
    need.max(16).next_power_of_two()
}
#[cfg(test)]
fn rehash(old: (&[u32], &[u32], &[u32]), new: (&mut [u32], &mut [u32], &mut [u32])) {
    new.0.fill(0);
    new.1.fill(0);
    new.2.fill(0);
    for i in 0..old.2.len() {
        if old.2[i] != 0 {
            insert(new.0, new.1, new.2, old.0[i], old.1[i], old.2[i]);
        }
    }
}
#[cfg(target_arch = "wasm32")]
#[export_name = "broadCreateSet"]
pub extern "C" fn create_set(need: usize) {
    create_set_in_world(crate::regions::active(), need)
}
#[cfg(target_arch = "wasm32")]
pub extern "C" fn create_set_in_world(world_index: usize, need: usize) {
    crate::broad::reserve_broad_in_world(world_index, 0, 0, 0, capacity(need));
}
#[cfg(target_arch = "wasm32")]
#[export_name = "broadEnsureSet"]
pub extern "C" fn ensure_set(need: usize) {
    ensure_set_in_world(crate::regions::active(), need)
}
#[cfg(target_arch = "wasm32")]
pub extern "C" fn ensure_set_in_world(world_index: usize, need: usize) {
    if crate::broad::set_cap(world_index) == 0 {
        create_set_in_world(world_index, need);
    }
}
#[cfg(target_arch = "wasm32")]
unsafe fn resident(world_index: usize) -> &'static mut [Item] {
    core::slice::from_raw_parts_mut(
        crate::broad::set_items(world_index),
        crate::broad::set_cap(world_index),
    )
}
/// # Safety
/// The active world must be selected, and this must run at a serial point, since growth moves the set.
#[cfg(target_arch = "wasm32")]
#[export_name = "broadAddPair"]
pub unsafe extern "C" fn add_pair(a: u32, b: u32, child: u32) -> u32 {
    unsafe { add_pair_in_world(crate::regions::active(), a, b, child) }
}
#[cfg(target_arch = "wasm32")]
/// # Safety
/// `world_index` must identify a live world, exclusively accessed while its set may grow.
pub unsafe extern "C" fn add_pair_in_world(world_index: usize, a: u32, b: u32, child: u32) -> u32 {
    ensure_set_in_world(world_index, 16);
    let key = item_key(a, b, child);
    let hash = key_hash((key >> 32) as u32, key as u32);
    let items = resident(world_index);
    if items[item_slot(items, key, hash)].hash != 0 {
        return 1;
    }
    if 2 * crate::broad::set_count(world_index) >= items.len() {
        crate::broad::grow_set(world_index);
    }
    item_insert(resident(world_index), key, hash);
    crate::broad::change_set_count(world_index, 1);
    0
}
/// # Safety
/// The active world must be selected, and no other thread may touch the set while this runs.
#[cfg(target_arch = "wasm32")]
#[export_name = "broadRemovePair"]
pub unsafe extern "C" fn remove_pair(a: u32, b: u32, child: u32) -> u32 {
    unsafe { remove_pair_in_world(crate::regions::active(), a, b, child) }
}
#[cfg(target_arch = "wasm32")]
/// # Safety
/// `world_index` must identify a live world, with no concurrent access to its set.
pub unsafe extern "C" fn remove_pair_in_world(
    world_index: usize,
    a: u32,
    b: u32,
    child: u32,
) -> u32 {
    if crate::broad::set_cap(world_index) == 0 {
        return 0;
    }
    let items = resident(world_index);
    let key = item_key(a, b, child);
    let hash = key_hash((key >> 32) as u32, key as u32);
    let mut i = item_slot(items, key, hash);
    if items[i].hash == 0 {
        return 0;
    }
    items[i].key = 0;
    items[i].hash = 0;
    crate::broad::change_set_count(world_index, -1);
    let mask = items.len() - 1;
    let mut j = i;
    loop {
        j = (j + 1) & mask;
        if items[j].hash == 0 {
            break;
        }
        let k = items[j].hash as usize & mask;
        if if i <= j {
            i < k && k <= j
        } else {
            i < k || k <= j
        } {
            continue;
        }
        items[i] = items[j];
        items[j].key = 0;
        items[j].hash = 0;
        i = j;
    }
    1
}
#[cfg(test)]
mod tests {
    use super::*;
    struct Set {
        hi: Vec<u32>,
        lo: Vec<u32>,
        hashes: Vec<u32>,
        count: usize,
    }
    impl Set {
        fn new() -> Self {
            Self {
                hi: vec![0; 16],
                lo: vec![0; 16],
                hashes: vec![0; 16],
                count: 0,
            }
        }
        fn add(&mut self, a: u32, b: u32, c: u32) -> bool {
            let (a, b) = (pair_key_hi(a, b), pair_key_lo(a, b, c));
            let hash = key_hash(a, b);
            if self.hashes[find(&self.hi, &self.lo, &self.hashes, a, b, hash)] != 0 {
                return true;
            }
            if self.count * 2 >= self.hashes.len() {
                let mut next = Self {
                    hi: vec![0; self.hi.len() * 2],
                    lo: vec![0; self.lo.len() * 2],
                    hashes: vec![0; self.hashes.len() * 2],
                    count: self.count,
                };
                rehash(
                    (&self.hi, &self.lo, &self.hashes),
                    (&mut next.hi, &mut next.lo, &mut next.hashes),
                );
                *self = next;
            }
            insert(&mut self.hi, &mut self.lo, &mut self.hashes, a, b, hash);
            self.count += 1;
            false
        }
        fn remove(&mut self, a: u32, b: u32, c: u32) -> bool {
            let found = remove(
                &mut self.hi,
                &mut self.lo,
                &mut self.hashes,
                pair_key_hi(a, b),
                pair_key_lo(a, b, c),
            );
            if found {
                self.count -= 1;
            }
            found
        }
        fn contains(&self, a: u32, b: u32, c: u32) -> bool {
            contains(&self.hi, &self.lo, &self.hashes, self.hashes.len(), a, b, c)
        }
    }
    #[test]
    fn capacity_is_a_power_of_two_covering_the_request() {
        assert_eq!(capacity(3008), 4096);
        assert_eq!(capacity(0), 16);
        assert_eq!(capacity(16), 16);
        assert_eq!(capacity(17), 32);
    }
    #[test]
    fn membership_survives_growth_and_backward_shift_deletion() {
        let mut set = Set::new();
        for a in 0..40 {
            for b in a + 1..40 {
                assert!(!set.add(a, b, 0));
            }
        }
        assert_eq!(set.count, 780);
        for a in 0..39 {
            assert!(set.remove(a, a + 1, 0));
        }
        assert_eq!(set.count, 741);
        for a in 0..40 {
            for b in a + 1..40 {
                assert_eq!(set.contains(b, a, 0), b != a + 1);
            }
        }
        for a in 0..40 {
            for b in a + 1..40 {
                set.remove(a, b, 0);
                assert!(!set.contains(a, b, 0));
            }
        }
        assert_eq!(set.count, 0);
    }
    #[test]
    fn reversed_pairs_are_duplicates_but_children_are_distinct() {
        let mut set = Set::new();
        assert!(!set.add(3, 7, 0));
        assert!(set.add(7, 3, 0));
        assert!(!set.add(3, 7, 1));
        assert_eq!(set.count, 2);
    }
    #[test]
    fn box3d_key_layout_and_hash_gold() {
        let s = SHAPE_MASK;
        let c = CHILD_MASK;
        for (a, b, c, hi, lo, hash) in [
            (0, 0, 0, 0, 0, 0),
            (0, 1, 0, 0, 0x100000, 0x7657ae14),
            (1, 0, 0, 0, 0x100000, 0x7657ae14),
            (s, s, c, 0xffffffff, 0xffffffff, 0x4b825f21),
            (0, s, c, 0x3ff, 0xffffffff, 0xdbe0fe82),
            (s, 0, 0, 0x3ff, 0xfff00000, 0x235cfe2e),
            (s - 1, s, c, 0xfffffbff, 0xffffffff, 0xe86c277b),
            (1, 2, c, 0x400, 0x2fffff, 0x15cd2890),
            (0, 0xfff, 0, 0, 0xfff00000, 0x670ea74e),
            (0, 0x1000, 0, 1, 0, 0xa5f1419),
            (0, 0xfff000, 0, 0x3ff, 0, 0xa0cfccd8),
            (
                7,
                0b1010101010_101010101010,
                0xabcde,
                0x1eaa,
                0xaaaabcde,
                0x8f4f233b,
            ),
        ] {
            assert_eq!(pair_key_hi(a, b), hi);
            assert_eq!(pair_key_lo(a, b, c), lo);
            assert_eq!(key_hash(hi, lo), hash);
        }
        for (a, b, c, hash) in [
            (3, 7, 0, 3489682763),
            (0, 10, 0, 2285512313),
            (2, 5, 0, 1184001114),
            (100, 3, 0, 179916708),
        ] {
            assert_eq!(key_hash(pair_key_hi(a, b), pair_key_lo(a, b, c)), hash);
        }
    }
    #[test]
    fn keys_are_symmetric_and_do_not_alias() {
        let mut seen = std::collections::HashSet::new();
        for a in 0..12 {
            for b in a + 1..12 {
                for c in 0..4 {
                    let key = (pair_key_hi(a, b), pair_key_lo(a, b, c));
                    assert_eq!(key, (pair_key_hi(b, a), pair_key_lo(b, a, c)));
                    assert!(seen.insert(key));
                }
            }
        }
        assert_eq!(seen.len(), 264);
    }
}
