//! Kernel heap usage and feature-gated allocation observations.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
struct Counting;
#[cfg(feature = "count-allocations")]
static OPERATIONS: AtomicUsize = AtomicUsize::new(0);
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
#[global_allocator]
static ALLOCATOR: Counting = Counting;
#[cfg(target_feature = "atomics")]
#[link(wasm_import_module = "env")]
extern "C" {
    fn allocationFailed();
}

unsafe fn checked(ptr: *mut u8) -> *mut u8 {
    #[cfg(target_feature = "atomics")]
    if ptr.is_null() {
        allocationFailed();
    }
    ptr
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        #[cfg(feature = "count-allocations")]
        OPERATIONS.fetch_add(1, Ordering::Relaxed);
        let ptr = checked(System.alloc(layout));
        if !ptr.is_null() {
            LIVE_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        #[cfg(feature = "count-allocations")]
        OPERATIONS.fetch_add(1, Ordering::Relaxed);
        let ptr = checked(System.alloc_zeroed(layout));
        if !ptr.is_null() {
            LIVE_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        #[cfg(feature = "count-allocations")]
        OPERATIONS.fetch_add(1, Ordering::Relaxed);
        let result = checked(System.realloc(ptr, layout, size));
        if !result.is_null() {
            LIVE_BYTES.fetch_add(size, Ordering::Relaxed);
            LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        }
        result
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        System.dealloc(ptr, layout);
    }
}
#[export_name = "allocationLiveBytes"]
pub extern "C" fn live_bytes() -> usize {
    LIVE_BYTES.load(Ordering::Relaxed)
}
#[cfg(feature = "count-allocations")]
#[export_name = "allocationCount"]
pub extern "C" fn count() -> usize {
    OPERATIONS.load(Ordering::Relaxed)
}
#[cfg(feature = "count-allocations")]
#[export_name = "allocationControl"]
pub unsafe extern "C" fn control() {
    let layout = Layout::from_size_align(64, 16).unwrap();
    let ptr = std::alloc::alloc(layout);
    assert!(!ptr.is_null());
    ptr.write_volatile(1);
    std::alloc::dealloc(ptr, layout);
}
