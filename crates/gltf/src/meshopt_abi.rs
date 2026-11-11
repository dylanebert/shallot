use crate::meshopt::View;
use std::{mem::MaybeUninit, slice};

#[export_name = "meshoptAlloc"]
pub extern "C" fn meshopt_alloc(bytes: usize) -> *mut u8 {
    Box::into_raw(Box::<[u64]>::new_uninit_slice(bytes.div_ceil(8))) as *mut u8
}

/// # Safety
/// `ptr` is a live allocation from `meshoptAlloc(bytes)`.
#[export_name = "meshoptFree"]
pub unsafe extern "C" fn meshopt_free(ptr: *mut u8, bytes: usize) {
    drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
        ptr.cast::<MaybeUninit<u64>>(),
        bytes.div_ceil(8),
    )));
}

#[inline(always)]
unsafe fn decode<const MODE: u32>(
    destination: *mut u8,
    count: usize,
    stride: usize,
    source: *const u8,
    source_len: usize,
    filter: u32,
    khr: bool,
) -> i32 {
    let Some(destination_len) = count
        .checked_mul(stride)
        .filter(|&len| len <= isize::MAX as usize)
    else {
        return -1;
    };
    if source_len > isize::MAX as usize {
        return -1;
    }
    let view = View {
        index: 0,
        count,
        stride,
        mode: MODE,
        filter,
        khr,
    };
    match view.decode_inner(
        slice::from_raw_parts(source, source_len),
        slice::from_raw_parts_mut(destination.cast::<MaybeUninit<u8>>(), destination_len),
    ) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

// Per-codec entries use the reference decoder's call shape. Attributes also select
// the filter and extension, without serializing a View into WASM memory.

/// # Safety
/// Source is initialized for source_len bytes; destination is writable for count*stride bytes.
/// Both pointers are non-null and their live ranges disjoint. Returns zero or -1; JS names the view.
#[export_name = "decodeMeshoptAttributes"]
pub unsafe extern "C" fn decode_attributes(
    destination: *mut u8,
    count: usize,
    stride: usize,
    source: *const u8,
    source_len: usize,
    filter: u32,
    khr: bool,
) -> i32 {
    decode::<0>(destination, count, stride, source, source_len, filter, khr)
}

/// # Safety
/// Same initialized-source and writable, non-null, disjoint destination contract as decode_attributes.
#[export_name = "decodeMeshoptTriangles"]
pub unsafe extern "C" fn decode_triangles(
    destination: *mut u8,
    count: usize,
    stride: usize,
    source: *const u8,
    source_len: usize,
) -> i32 {
    decode::<1>(destination, count, stride, source, source_len, 0, true)
}

/// # Safety
/// Same initialized-source and writable, non-null, disjoint destination contract as decode_attributes.
#[export_name = "decodeMeshoptIndices"]
pub unsafe extern "C" fn decode_indices(
    destination: *mut u8,
    count: usize,
    stride: usize,
    source: *const u8,
    source_len: usize,
) -> i32 {
    decode::<2>(destination, count, stride, source, source_len, 0, true)
}
