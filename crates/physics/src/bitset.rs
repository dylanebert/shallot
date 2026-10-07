//! Box3D bitset.c: 64-bit blocks with a logical count and retained capacity.
use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};

#[repr(C)]
pub(crate) struct BitSet {
    pub bits: *mut u64,
    pub block_capacity: u32,
    pub block_count: u32,
}

impl BitSet {
    pub fn new(bit_capacity: u32) -> Self {
        let block_capacity = bit_capacity.div_ceil(64);
        let bits = if block_capacity == 0 {
            core::ptr::null_mut()
        } else {
            let layout = Self::layout(block_capacity);
            let ptr = unsafe { alloc(layout) };
            if ptr.is_null() {
                handle_alloc_error(layout);
            }
            unsafe { ptr.write_bytes(0, block_capacity as usize * 8) };
            ptr.cast()
        };
        Self {
            bits,
            block_capacity,
            block_count: 0,
        }
    }

    fn layout(block_capacity: u32) -> Layout {
        // b3Alloc rounds the byte size and aligns to B3_ALIGNMENT (16).
        Layout::from_size_align((block_capacity as usize * 8).next_multiple_of(16), 16).unwrap()
    }

    pub fn set_count_and_clear(&mut self, bit_count: u32) {
        let block_count = bit_count.div_ceil(64);
        if self.block_capacity < block_count {
            self.destroy();
            *self = Self::new(bit_count + (bit_count >> 1));
        }
        self.block_count = block_count;
        if block_count != 0 {
            unsafe { self.bits.write_bytes(0, block_count as usize) };
        }
    }

    /// The calling worker exclusively owns writes to these blocks until the fork joins.
    pub unsafe fn set(&self, bit: usize) {
        let block = bit / 64;
        assert!(block < self.block_count as usize);
        *self.bits.add(block) |= 1u64 << (bit % 64);
    }

    pub fn get(&self, bit: usize) -> bool {
        let block = bit / 64;
        assert!(block < self.block_count as usize);
        unsafe { *self.bits.add(block) & (1u64 << (bit % 64)) != 0 }
    }

    pub fn union(&mut self, other: &Self) {
        assert_eq!(self.block_count, other.block_count);
        for i in 0..self.block_count as usize {
            unsafe { *self.bits.add(i) |= *other.bits.add(i) };
        }
    }

    fn destroy(&mut self) {
        if self.block_capacity != 0 {
            unsafe { dealloc(self.bits.cast(), Self::layout(self.block_capacity)) };
        }
        self.block_capacity = 0;
        self.block_count = 0;
        self.bits = core::ptr::null_mut();
    }
}

impl Drop for BitSet {
    fn drop(&mut self) {
        self.destroy();
    }
}

#[cfg(test)]
mod tests {
    use super::BitSet;

    #[test]
    fn worker_union_keeps_ids_on_both_sides_of_64_bit_boundaries() {
        let mut sets: Vec<_> = (0..4).map(|_| BitSet::new(1024)).collect();
        for set in &mut sets {
            set.set_count_and_clear(193);
        }
        for (worker, ids) in [[0, 64, 192], [31, 63, 127], [32, 65, 128], [63, 64, 191]]
            .iter()
            .enumerate()
        {
            for &id in ids {
                assert!(!sets[worker].get(id));
                unsafe { sets[worker].set(id) };
                assert!(sets[worker].get(id));
            }
        }
        let (first, rest) = sets.split_first_mut().unwrap();
        for set in rest {
            first.union(set);
        }
        let mut ids = Vec::new();
        for word in 0..first.block_count as usize {
            let mut mask = unsafe { *first.bits.add(word) };
            while mask != 0 {
                ids.push(word * 64 + mask.trailing_zeros() as usize);
                mask &= mask - 1;
            }
        }
        assert_eq!(ids, [0, 31, 32, 63, 64, 65, 127, 128, 191, 192]);
        let ptr = first.bits;
        first.set_count_and_clear(64);
        assert_eq!(first.bits, ptr);
        assert_eq!(first.block_count, 1);
        assert_eq!(unsafe { *first.bits }, 0);
        first.set_count_and_clear(193);
        assert!((0..4).all(|i| unsafe { *first.bits.add(i) } == 0));
    }

    #[test]
    fn growth_uses_native_bit_capacity_and_clears_the_active_prefix() {
        let mut set = BitSet::new(1024);
        assert_eq!((set.block_capacity, set.block_count), (16, 0));
        set.set_count_and_clear(1024);
        unsafe { set.set(1023) };
        set.set_count_and_clear(1025);
        assert_eq!((set.block_capacity, set.block_count), (25, 17));
        assert!((0..17).all(|i| unsafe { *set.bits.add(i) } == 0));
        set.set_count_and_clear(0);
        assert_eq!((set.block_capacity, set.block_count), (25, 0));
    }
}
