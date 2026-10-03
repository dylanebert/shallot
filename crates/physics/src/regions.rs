//! Allocator-owned columns. Reserves run on the calling thread before the worker fork.
use std::alloc::{alloc_zeroed, dealloc, handle_alloc_error, realloc, Layout};

pub const MAX_WORLDS: usize = 128;
static mut ACTIVE_WORLD: usize = 0;
static mut VIEW_EPOCH: u32 = 0;

#[export_name = "viewEpochPtr"]
pub extern "C" fn view_epoch_ptr() -> *const u32 {
    &raw const VIEW_EPOCH
}
unsafe fn invalidate_views() {
    VIEW_EPOCH = VIEW_EPOCH.wrapping_add(1);
}

#[export_name = "activeWorld"]
pub extern "C" fn active() -> usize {
    unsafe { ACTIVE_WORLD }
}
pub fn select(world: u32) {
    assert!((world as usize) < MAX_WORLDS);
    unsafe {
        ACTIVE_WORLD = world as usize;
    }
}
#[export_name = "residentResetWorld"]
pub extern "C" fn reset(world: u32) {
    let id = world as usize;
    assert!(id < MAX_WORLDS);
    unsafe {
        crate::fataabb::reset(id);
        crate::manifolds::reset(id);
        crate::broad::reset(id);
        crate::geo::reset(id);
    }
}

#[export_name = "residentRestoreWorldId"]
pub extern "C" fn restore_world_id(from: u32, to: u32) {
    if from == to {
        return;
    }
    assert!((from as usize) < MAX_WORLDS && (to as usize) < MAX_WORLDS);
    unsafe {
        crate::bodies::restore_id(from as usize, to as usize);
        crate::shapes::restore_id(from as usize, to as usize);
        crate::fataabb::restore_id(from as usize, to as usize);
        crate::manifolds::restore_id(from as usize, to as usize);
        crate::broad::restore_id(from as usize, to as usize);
        crate::geo::restore_id(from as usize, to as usize);
    }
    select(to);
}

#[derive(Clone, Copy)]
pub struct Buffer {
    pub ptr: usize,
    bytes: usize,
}
impl Buffer {
    // Empty slices still require a non-null, aligned pointer.
    pub const EMPTY: Self = Self { ptr: 16, bytes: 0 };
    pub unsafe fn reserve(&mut self, bytes: usize) -> bool {
        if bytes <= self.bytes {
            return false;
        }
        let bytes = bytes.next_power_of_two().max(16);
        let layout = Layout::from_size_align_unchecked(bytes, 16);
        let ptr = if self.bytes == 0 {
            alloc_zeroed(layout)
        } else {
            realloc(
                self.ptr as *mut u8,
                Layout::from_size_align_unchecked(self.bytes, 16),
                bytes,
            )
        };
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        if self.bytes != 0 {
            ptr.add(self.bytes).write_bytes(0, bytes - self.bytes);
        }
        self.ptr = ptr as usize;
        self.bytes = bytes;
        invalidate_views();
        true
    }
    pub unsafe fn release(&mut self) {
        if self.bytes != 0 {
            invalidate_views();
            dealloc(
                self.ptr as *mut u8,
                Layout::from_size_align_unchecked(self.bytes, 16),
            );
        }
        *self = Self::EMPTY;
    }
}

#[derive(Clone, Copy)]
pub struct Columns<const N: usize> {
    pub layout: [u32; N],
    buffers: [Buffer; N],
}
impl<const N: usize> Columns<N> {
    pub const EMPTY: Self = Self {
        layout: [16; N],
        buffers: [Buffer::EMPTY; N],
    };
    pub unsafe fn reserve(&mut self, column: usize, bytes: usize) {
        invalidate_views();
        self.buffers[column].reserve(bytes);
        self.layout[column] = self.buffers[column].ptr as u32;
    }
    pub unsafe fn release(&mut self) {
        for buffer in &mut self.buffers {
            buffer.release();
        }
        self.layout.fill(16);
    }
}
