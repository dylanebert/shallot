//! Report shared linear-memory exhaustion before Rust's allocation abort traps.
use std::alloc::{GlobalAlloc, Layout, System};

struct Shared;
#[global_allocator]
static ALLOCATOR: Shared = Shared;

#[link(wasm_import_module = "env")]
extern "C" {
    fn allocationFailed();
}

unsafe fn checked(ptr: *mut u8) -> *mut u8 {
    if ptr.is_null() {
        allocationFailed();
    }
    ptr
}

unsafe impl GlobalAlloc for Shared {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        checked(System.alloc(layout))
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        checked(System.alloc_zeroed(layout))
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        checked(System.realloc(ptr, layout, size))
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
}
