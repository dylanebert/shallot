//! Owned KTX2 input and output at the JavaScript boundary.
use crate::{Channels, Ktx2, Model, Target, Transfer, UastcTarget};
struct Image {
    bytes: Vec<u8>,
    output: Vec<u8>,
    tables: Option<crate::etc1s::Tables>,
    error: &'static str,
}
#[no_mangle]
pub extern "C" fn alloc(length: usize) -> *mut u8 {
    Box::into_raw(vec![0u8; length].into_boxed_slice()) as *mut u8
}
#[no_mangle]
pub unsafe extern "C" fn open(ptr: *mut u8, length: usize) -> usize {
    let bytes = Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, length)).into_vec();
    // ktx2_transcoder::start_transcoding prepares ETC1S codebooks once per
    // image, not again for each mip level. The tables own no input slices.
    let prepared = Ktx2::parse(&bytes).and_then(|ktx| {
        if ktx.model == Model::Etc1s {
            crate::etc1s::Tables::new(&ktx).map(Some)
        } else {
            Ok(None)
        }
    });
    let (tables, error) = match prepared {
        Ok(tables) => (tables, ""),
        Err(error) => (None, error),
    };
    Box::into_raw(Box::new(Image {
        bytes,
        output: Vec::new(),
        tables,
        error,
    })) as usize
}
#[no_mangle]
pub unsafe extern "C" fn close(handle: usize) {
    drop(Box::from_raw(handle as *mut Image));
}
#[no_mangle]
pub unsafe extern "C" fn info(handle: usize, field: usize) -> usize {
    let image = &*(handle as *const Image);
    if field == 4 {
        return image.error.as_ptr() as usize;
    }
    if field == 5 {
        return image.error.len();
    }
    if field == 6 {
        return image.output.as_ptr() as usize;
    }
    if field == 7 {
        return image.output.len();
    }
    let Ok(ktx) = Ktx2::parse(&image.bytes) else {
        return 0;
    };
    match field {
        0 => ktx.width,
        1 => ktx.height,
        2 => ktx.levels.len(),
        3 => usize::from(ktx.model == Model::Uastc),
        8 => usize::from(ktx.transfer == Transfer::Srgb),
        9 => usize::from(ktx.channels == Channels::Rgba),
        _ => 0,
    }
}
#[no_mangle]
pub unsafe extern "C" fn transcode(handle: usize, level: usize, target: usize) -> usize {
    let image = &mut *(handle as *mut Image);
    let result = Ktx2::parse(&image.bytes).and_then(|ktx| match (ktx.model, target) {
        (Model::Etc1s, 0..=3) => image
            .tables
            .as_ref()
            .ok_or("missing ETC1S codebooks")
            .and_then(|tables| {
                ktx.transcode_etc1s_with_tables(
                    level,
                    [Target::Etc1, Target::Etc2Rgba, Target::Bc7, Target::Rgba8][target],
                    tables,
                )
            }),
        (Model::Uastc, 2) => ktx.transcode_uastc(level, UastcTarget::Bc7),
        (Model::Uastc, 3) => ktx.transcode_uastc(level, UastcTarget::Rgba8),
        (Model::Uastc, 4) => ktx.transcode_uastc(level, UastcTarget::Astc),
        _ => Err("invalid Basis Universal target"),
    });
    match result {
        Ok(output) => {
            image.output = output;
            image.error = "";
            0
        }
        Err(error) => {
            image.output.clear();
            image.error = error;
            1
        }
    }
}
