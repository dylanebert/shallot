//! Box3D table.c: symmetric pair keys, linear probing and backward-shift deletion.
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
fn find(hi: &[u32], lo: &[u32], hashes: &[u32], a: u32, b: u32, hash: u32) -> usize {
    let mask = hashes.len() - 1;
    let mut i = hash as usize & mask;
    while hashes[i] != 0 && (hi[i] != a || lo[i] != b) {
        i = (i + 1) & mask;
    }
    i
}
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
unsafe fn resident(
    world_index: usize,
) -> (&'static mut [u32], &'static mut [u32], &'static mut [u32]) {
    let cap = crate::broad::set_cap(world_index);
    let (hi, lo, hashes) = crate::broad::set_ptrs(world_index);
    (
        core::slice::from_raw_parts_mut(hi as *mut u32, cap),
        core::slice::from_raw_parts_mut(lo as *mut u32, cap),
        core::slice::from_raw_parts_mut(hashes as *mut u32, cap),
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

pub unsafe extern "C" fn add_pair_in_world(world_index: usize, a: u32, b: u32, child: u32) -> u32 {
    ensure_set_in_world(world_index, 16);
    let (a, b) = (pair_key_hi(a, b), pair_key_lo(a, b, child));
    let hash = key_hash(a, b);
    let (hi, lo, hashes) = resident(world_index);
    if hashes[find(hi, lo, hashes, a, b, hash)] != 0 {
        return 1;
    }
    if 2 * crate::broad::set_count(world_index) >= hashes.len() {
        let old = (hi.to_vec(), lo.to_vec(), hashes.to_vec());
        crate::broad::reserve_broad_in_world(world_index, 0, 0, 0, hashes.len() * 2);
        let (hi, lo, hashes) = resident(world_index);
        rehash((&old.0, &old.1, &old.2), (hi, lo, hashes));
    }
    let (hi, lo, hashes) = resident(world_index);
    insert(hi, lo, hashes, a, b, hash);
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

pub unsafe extern "C" fn remove_pair_in_world(
    world_index: usize,
    a: u32,
    b: u32,
    child: u32,
) -> u32 {
    if crate::broad::set_cap(world_index) == 0 {
        return 0;
    }
    let (hi, lo, hashes) = resident(world_index);
    let found = remove(hi, lo, hashes, pair_key_hi(a, b), pair_key_lo(a, b, child));
    if found {
        crate::broad::change_set_count(world_index, -1);
    }
    found as u32
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
