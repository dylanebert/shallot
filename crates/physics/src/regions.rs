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
        crate::joints::reset(id);
        crate::broad::reset(id);
        crate::geo::reset(id);
    }
}

pub fn write_word(out: &mut Vec<u8>, value: usize) {
    out.extend_from_slice(&(value as u32).to_le_bytes());
}
pub fn read_word(input: &mut &[u8]) -> usize {
    let (word, rest) = input.split_at(4);
    *input = rest;
    u32::from_le_bytes(word.try_into().unwrap()) as usize
}
static mut SNAPSHOT: Vec<u8> = Vec::new();
#[export_name = "worldSnapshot"]
pub extern "C" fn snapshot(world: u32) -> usize {
    assert!((world as usize) < MAX_WORLDS);
    unsafe {
        let mut out = Vec::new();
        crate::bodies::snapshot(world as usize, &mut out);
        crate::shapes::snapshot(world as usize, &mut out);
        crate::fataabb::snapshot(world as usize, &mut out);
        crate::manifolds::snapshot(world as usize, &mut out);
        crate::joints::snapshot(world as usize, &mut out);
        crate::broad::snapshot(world as usize, &mut out);
        crate::geo::snapshot(world as usize, &mut out);
        let buffer = &mut *(&raw mut SNAPSHOT);
        *buffer = out;
        buffer.len()
    }
}
#[export_name = "worldSnapshotBuffer"]
pub extern "C" fn snapshot_buffer(bytes: usize) -> *mut u8 {
    unsafe {
        let buffer = &mut *(&raw mut SNAPSHOT);
        buffer.resize(bytes, 0);
        buffer.as_mut_ptr()
    }
}
#[export_name = "worldRestore"]
pub extern "C" fn restore(world: u32) {
    assert!((world as usize) < MAX_WORLDS);
    unsafe {
        let mut input = (&*(&raw const SNAPSHOT)).as_slice();
        crate::bodies::restore(world as usize, &mut input);
        crate::shapes::restore(world as usize, &mut input);
        crate::fataabb::restore(world as usize, &mut input);
        crate::manifolds::restore(world as usize, &mut input);
        crate::joints::restore(world as usize, &mut input);
        crate::broad::restore(world as usize, &mut input);
        crate::geo::restore(world as usize, &mut input);
        assert!(input.is_empty());
        invalidate_views();
    }
    select(world);
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
    pub unsafe fn snapshot(&self, out: &mut Vec<u8>) {
        for buffer in &self.buffers {
            write_word(out, buffer.bytes);
            out.extend_from_slice(core::slice::from_raw_parts(
                buffer.ptr as *const u8,
                buffer.bytes,
            ));
        }
    }
    pub unsafe fn restore(&mut self, input: &mut &[u8]) {
        self.release();
        for column in 0..N {
            let bytes = read_word(input);
            self.reserve(column, bytes);
            let (data, rest) = input.split_at(bytes);
            core::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.buffers[column].ptr as *mut u8,
                bytes,
            );
            *input = rest;
        }
    }
    pub unsafe fn release(&mut self) {
        for buffer in &mut self.buffers {
            buffer.release();
        }
        self.layout.fill(16);
    }
}
