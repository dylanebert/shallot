//! Feature-gated allocator observations for steady-state WASM checks.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
struct Counting;
static OPERATIONS: AtomicUsize = AtomicUsize::new(0);
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
#[global_allocator]
static ALLOCATOR: Counting = Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        OPERATIONS.fetch_add(1, Ordering::Relaxed);
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            LIVE_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        OPERATIONS.fetch_add(1, Ordering::Relaxed);
        let ptr = System.alloc_zeroed(layout);
        if !ptr.is_null() {
            LIVE_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        OPERATIONS.fetch_add(1, Ordering::Relaxed);
        let result = System.realloc(ptr, layout, size);
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
#[export_name = "allocationCount"]
pub extern "C" fn count() -> usize {
    OPERATIONS.load(Ordering::Relaxed)
}
#[export_name = "allocationControl"]
pub unsafe extern "C" fn control() {
    let layout = Layout::from_size_align(64, 16).unwrap();
    let ptr = std::alloc::alloc(layout);
    assert!(!ptr.is_null());
    ptr.write_volatile(1);
    std::alloc::dealloc(ptr, layout);
}
