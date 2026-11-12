//! Owned Draco input and output at the JavaScript boundary.
use crate::{decode as decode_mesh, DataType, Result};

struct Decoded {
    faces: Vec<[u32; 3]>,
    outputs: Vec<(u8, Vec<u8>)>,
    error: &'static str,
}

/// Zero when the memory cannot be allocated, so the caller can refuse by name.
#[no_mangle]
pub extern "C" fn alloc(length: usize) -> *mut u8 {
    // `Box<[u8]>`'s layout, so `decode` and `release` free it as a box.
    match std::alloc::Layout::array::<u8>(length) {
        Ok(_) if length == 0 => std::ptr::NonNull::dangling().as_ptr(),
        Ok(layout) => unsafe { std::alloc::alloc_zeroed(layout) },
        Err(_) => std::ptr::null_mut(),
    }
}

/// Releases an `alloc` the caller could not hand to `decode`.
#[no_mangle]
pub unsafe extern "C" fn release(ptr: *mut u8, length: usize) {
    drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
        ptr, length,
    )));
}

fn run(bytes: &[u8], faces: usize, points: usize, request: &[u32]) -> Result<Decoded> {
    let (ids, types) = request.split_at(request.len() / 2);
    let mesh = decode_mesh(bytes, faces, points, ids)?;
    let mut outputs = Vec::new();
    outputs
        .try_reserve_exact(ids.len())
        .map_err(|_| "Draco allocation failed")?;
    for (&id, &output) in ids.iter().zip(types) {
        let attribute = mesh
            .attribute(id)
            .ok_or("Draco attribute id not in stream")?;
        let data_type = u8::try_from(output)
            .ok()
            .and_then(DataType::from_id)
            .ok_or("invalid Draco output data type")?;
        outputs.push((attribute.components, attribute.points(data_type)?));
    }
    Ok(Decoded {
        faces: mesh.faces,
        outputs,
        error: "",
    })
}

/// Takes ownership of `stream` and `request` (ids, then their output types).
#[no_mangle]
pub unsafe extern "C" fn decode(
    stream: *mut u8,
    length: usize,
    faces: usize,
    points: usize,
    request: *mut u8,
    requested: usize,
) -> usize {
    let bytes = Box::from_raw(std::ptr::slice_from_raw_parts_mut(stream, length));
    let request = Box::from_raw(std::ptr::slice_from_raw_parts_mut(request, requested * 8));
    let words: Vec<u32> = request
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect();
    let decoded = run(&bytes, faces, points, &words).unwrap_or_else(|error| Decoded {
        faces: Vec::new(),
        outputs: Vec::new(),
        error,
    });
    Box::into_raw(Box::new(decoded)) as usize
}

#[no_mangle]
pub unsafe extern "C" fn close(handle: usize) {
    drop(Box::from_raw(handle as *mut Decoded));
}

/// Fields 0-3: error pointer and length, faces pointer and face count; then
/// pointer, byte length and component count of each requested attribute.
#[no_mangle]
pub unsafe extern "C" fn info(handle: usize, field: usize) -> usize {
    let decoded = &*(handle as *const Decoded);
    match field {
        0 => decoded.error.as_ptr() as usize,
        1 => decoded.error.len(),
        2 => decoded.faces.as_ptr() as usize,
        3 => decoded.faces.len(),
        _ => {
            let (components, values) = &decoded.outputs[(field - 4) / 3];
            match (field - 4) % 3 {
                0 => values.as_ptr() as usize,
                1 => values.len(),
                _ => usize::from(*components),
            }
        }
    }
}
