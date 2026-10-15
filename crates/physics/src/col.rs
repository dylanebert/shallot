//! Shared-mutable columns use raw pointers: overlapping whole-column `&mut`
//! references are undefined behavior even when workers write disjoint elements.
//! Constructors establish the access invariant for every copy of the handle.
//! Bounds are checked only in debug builds; callers must keep every index in range.
//!
//! WASM's 'static handles require stable storage until workers join. Reserve
//! columns before the fork. Collide may allocate disjoint manifold blocks under
//! a lock because chunks never move; its contact directory stays fixed during tasks.

use core::marker::PhantomData;

/// A shared-mutable view that never forms references to its elements.
pub struct Col<'a, T> {
    ptr: *mut T,
    len: usize,
    _own: PhantomData<&'a mut [T]>,
}

impl<T> Clone for Col<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Col<'_, T> {}

// SAFETY: construction guarantees disjoint writes or atomic access across copies.
// Sharing grants write access, so Sync requires T: Send just as Send does.
unsafe impl<T: Send> Send for Col<'_, T> {}
unsafe impl<T: Send> Sync for Col<'_, T> {}

impl Col<'_, u32> {
    // Shared flag words need atomic access even when readers inspect only invariant bits.
    #[inline]
    pub fn atomic_get(self, i: usize) -> u32 {
        debug_assert!(i < self.len);
        unsafe { core::sync::atomic::AtomicU32::from_ptr(self.ptr.add(i)) }
            .load(core::sync::atomic::Ordering::Relaxed)
    }

    #[inline]
    pub fn atomic_set(self, i: usize, v: u32) {
        debug_assert!(i < self.len);
        unsafe { core::sync::atomic::AtomicU32::from_ptr(self.ptr.add(i)) }
            .store(v, core::sync::atomic::Ordering::Relaxed);
    }

    #[inline]
    pub fn atomic_or(self, i: usize, v: u32) {
        debug_assert!(i < self.len);
        unsafe { core::sync::atomic::AtomicU32::from_ptr(self.ptr.add(i)) }
            .fetch_or(v, core::sync::atomic::Ordering::Relaxed);
    }
}

impl<'a, T: Copy> Col<'a, T> {
    /// A column over `len` elements at `ptr`.
    ///
    /// # Safety
    /// `ptr` must be non-null, aligned, and valid for `len` `T`s for all of `'a`; and every thread
    /// this handle reaches must confine its writes to elements no other thread touches, unless all
    /// concurrent accesses to an element use the atomic u32 methods.
    #[inline]
    pub const unsafe fn new(ptr: *mut T, len: usize) -> Self {
        Col {
            ptr,
            len,
            _own: PhantomData,
        }
    }

    /// Borrows the slice exclusively for 'a; copies of the returned handle can
    /// still share its storage across threads.
    ///
    /// # Safety
    /// As [`new`](Col::new): all copies must obey its disjoint-write or atomic-access contract.
    #[inline]
    pub unsafe fn of(s: &'a mut [T]) -> Self {
        Col {
            ptr: s.as_mut_ptr(),
            len: s.len(),
            _own: PhantomData,
        }
    }

    #[inline]
    pub fn get(self, i: usize) -> T {
        debug_assert!(
            i < self.len,
            "column index {i} out of bounds ({})",
            self.len
        );
        unsafe { *self.ptr.add(i) }
    }

    #[inline]
    pub fn set(self, i: usize, v: T) {
        debug_assert!(
            i < self.len,
            "column index {i} out of bounds ({})",
            self.len
        );
        unsafe { *self.ptr.add(i) = v }
    }

    /// The base pointer, for the wide solver's `v128` record loads.
    #[inline]
    pub fn ptr(self) -> *mut T {
        self.ptr
    }

    #[inline]
    pub fn len(self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.len == 0
    }
}
