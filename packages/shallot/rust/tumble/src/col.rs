//! `Col` — the shared-mutable column handle every solver phase indexes through.
//!
//! A column is one contiguous run of linear memory (a body field, a constraint record, the manifold
//! pool). The blocks of a solver stage are **write-disjoint** — within a graph color no two
//! constraints share a body, and the flat stages touch only their own record (`stages.rs`) — so many
//! threads may work one column at once and never write the same element.
//!
//! That is a property of the *data flow*, not of the reference. A `&mut [f32]` over the whole column
//! is what the phases used to take, and two concurrent blocks would then each hold a live `&mut` to
//! the same bytes. LLVM emits `noalias` for a `&mut`, so the optimizer is entitled to assume no other
//! thread writes through it; disjoint writes do not rescue that — the UB is in the aliasing
//! references, and its failure mode is a silent miscompile under optimization, not a fault. A raw
//! base pointer makes no aliasing claim, so it is the one shape that can express "we alias, and we
//! promise our writes don't overlap".
//!
//! **The handle is the unsafe token.** Both constructors are `unsafe`; `get`/`set` are not. Once a
//! `Col` exists, every phase is asserting nothing about disjointness — the thread that *minted* it
//! did, on behalf of all of them. That is the only place the promise can be checked, since a column
//! is `Copy` and travels into workers by value.
//!
//! Bounds live in `len` and are checked in debug only: the release path is an unchecked load/store,
//! which is the same bounds-check elimination the wide gather already banks on (kex `tumble.md`).
//!
//! The lifetime is real — a `Col<'a, T>` borrows its storage for `'a`, so the native harnesses get
//! use-after-free protection for free. The wasm arena's `'static` columns are the exception, and they
//! rest on the no-`memory.grow`-while-workers-are-active invariant (the Multithreading contract in `.claude/rules/tumble.md`): a region
//! grow relocates the columns above it, so a `Col` held across one dangles. Every arena shim
//! re-derives its columns from `LAYOUT` per call, and reserves run pre-solve on the main thread.

use core::marker::PhantomData;

/// A shared-mutable view of `len` `T`s. Copyable, thread-shareable, and it never forms a reference to
/// the elements — see the module header for why that is the point.
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

// SAFETY: a `Col` is a bare pointer + length; sharing one hands another thread write access, so
// `Sync` needs the same bound as `Send`. The disjoint-write promise made at construction is what
// makes the sharing sound. Both impls are needed for a `StageWork` holding columns to be `Sync`
// (stages.rs) and for the handle to cross into a worker.
unsafe impl<T: Send> Send for Col<'_, T> {}
unsafe impl<T: Send> Sync for Col<'_, T> {}

impl<'a, T: Copy> Col<'a, T> {
    /// A column over `len` elements at `ptr`.
    ///
    /// # Safety
    /// `ptr` must be non-null, aligned, and valid for `len` `T`s for all of `'a`; and every thread
    /// this handle reaches must confine its writes to elements no other thread touches.
    #[inline]
    pub const unsafe fn new(ptr: *mut T, len: usize) -> Self {
        Col {
            ptr,
            len,
            _own: PhantomData,
        }
    }

    /// A column over an owned slice — the native harnesses' constructor. The unique borrow is
    /// consumed for `'a`, so nothing else can reach the storage while the column is alive.
    ///
    /// # Safety
    /// As [`new`](Col::new): the disjoint-write promise. Unsafe *despite* the sound borrow, because
    /// the handle it returns is `Copy + Sync` — two safe copies on two threads writing one element
    /// would be a data race, and this is the seam where that is ruled out.
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
        debug_assert!(i < self.len, "column index {i} out of bounds ({})", self.len);
        unsafe { *self.ptr.add(i) }
    }

    #[inline]
    pub fn set(self, i: usize, v: T) {
        debug_assert!(i < self.len, "column index {i} out of bounds ({})", self.len);
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
