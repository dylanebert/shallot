//! meshoptimizer's SIMD_WASM paths in vertexcodec.cpp and vertexfilter.cpp.
use super::{DecodeResult, Reader};
use core::arch::wasm32::*;

const fn shuffle_table() -> [u8; 2056] {
    let mut table = [0u8; 2056];
    let mut mask = 0;
    while mask < 256 {
        let mut count = 0;
        let mut i = 0;
        while i < 8 {
            table[mask * 8 + i] = if mask & (1 << i) != 0 { count } else { 128 };
            count += ((mask >> i) & 1) as u8;
            i += 1;
        }
        mask += 1;
    }
    table
}
// The extra eight bytes keep the last mask's 16-byte speculative load in bounds.
static SHUFFLE: [u8; 2056] = shuffle_table();
const fn counts() -> [u8; 256] {
    let mut table = [0; 256];
    let mut i = 0;
    while i < 256 {
        table[i] = (i as u32).count_ones() as u8;
        i += 1;
    }
    table
}
static COUNTS: [u8; 256] = counts();

// LLVM SLP turns the reference's scalar mask multiplies into i64x2.mul,
// slower on M4 Max. wasm-opt inlines this helper after LLVM vectorization.
#[inline(never)]
fn mask(value: u64) -> usize {
    (value.wrapping_mul(0x000103070f1f3f80) >> 56) as usize
}
fn masks(v: v128) -> (usize, usize) {
    (
        mask(u64x2_extract_lane::<0>(v)),
        mask(u64x2_extract_lane::<1>(v)),
    )
}
unsafe fn shuffle(mask0: usize, mask1: usize) -> v128 {
    let a = v128_load(SHUFFLE.as_ptr().add(mask0 * 8).cast());
    let b = v128_load(SHUFFLE.as_ptr().add(mask1 * 8).cast());
    let b = i8x16_add(b, i8x16_splat(COUNTS[mask0] as i8));
    i64x2_shuffle::<0, 2>(a, b)
}

#[inline(always)]
unsafe fn packed<const BITS: usize>(data: *const u8) -> (v128, usize) {
    let values = v128_load(data.cast());
    let p = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(
        u16x8_shr(values, 4),
        values,
    );
    let sel = if BITS == 2 {
        let p = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(
            u16x8_shr(p, 2),
            p,
        );
        v128_and(p, u8x16_splat(3))
    } else {
        v128_and(p, u8x16_splat(15))
    };
    let mask = i8x16_eq(sel, u8x16_splat((1 << BITS) - 1));
    let (a, b) = masks(mask);
    let rest = v128_load(data.add(BITS * 2).cast());
    (
        v128_bitselect(u8x16_swizzle(rest, shuffle(a, b)), sel, mask),
        BITS * 2 + COUNTS[a] as usize + COUNTS[b] as usize,
    )
}

#[inline(always)]
unsafe fn group(data: *const u8, output: *mut u8, hbits: usize) -> usize {
    let bytes = std::slice::from_raw_parts(data, 24);
    let (result, consumed) = match hbits {
        0 | 4 => (u8x16_splat(0), 0),
        3 | 8 => (v128_load(data.cast()), 16),
        5 => {
            let a = usize::from(bytes[0]);
            let b = usize::from(bytes[1]);
            (
                u8x16_swizzle(v128_load(data.add(2).cast()), shuffle(a, b)),
                2 + COUNTS[a] as usize + COUNTS[b] as usize,
            )
        }
        1 | 6 => packed::<2>(data),
        2 | 7 => packed::<4>(data),
        _ => unreachable!(),
    };
    v128_store(output.cast(), result);
    consumed
}

pub fn bytes(
    r: &mut Reader<'_>,
    output: &mut [core::mem::MaybeUninit<u8>],
    groups: usize,
    hshift: usize,
) -> DecodeResult<()> {
    // meshoptimizer-rs 0.2.9 bytes_kernel: specialize the three header alphabets.
    match hshift {
        0 => bytes_header::<0>(r, output, groups),
        4 => bytes_header::<4>(r, output, groups),
        5 => bytes_header::<5>(r, output, groups),
        _ => unreachable!(),
    }
}

fn bytes_header<const H: usize>(
    r: &mut Reader<'_>,
    output: &mut [core::mem::MaybeUninit<u8>],
    groups: usize,
) -> DecodeResult<()> {
    let hshift = H;
    let header = r.take(groups.div_ceil(4))?;
    let mut i = 0;
    // SAFETY: a group reads at most 24 bytes and consumes at most 24. The bulk
    // path reserves four groups; the tail path reserves one. Each writes 16 lanes.
    // The caller supplies groups*16 output lanes. The header has ceil(groups/4)
    // bytes, and both loops keep i<groups<=16.
    unsafe {
        let base = r.bytes.as_ptr();
        let mut data = base;
        let end = base.add(r.bytes.len());
        while i + 4 <= groups && end.offset_from(data) >= 96 {
            let code = *header.get_unchecked(i / 4);
            // Same four-group shortcuts as meshoptimizer-rs bytes_header_kernel
            // (and meshoptimizer's SSE decodeBytesSimd), with the shared 96-byte guard.
            if code == 0 && H != 5 {
                std::ptr::write_bytes(output.as_mut_ptr().add(i * 16).cast::<u8>(), 0, 64);
                i += 4;
                continue;
            }
            if code == 255 && H != 4 {
                std::ptr::copy_nonoverlapping(
                    data,
                    output.as_mut_ptr().add(i * 16).cast::<u8>(),
                    64,
                );
                data = data.add(64);
                i += 4;
                continue;
            }
            data = data.add(group(
                data,
                output.as_mut_ptr().add(i * 16).cast(),
                hshift + (code & 3) as usize,
            ));
            data = data.add(group(
                data,
                output.as_mut_ptr().add((i + 1) * 16).cast(),
                hshift + ((code >> 2) & 3) as usize,
            ));
            data = data.add(group(
                data,
                output.as_mut_ptr().add((i + 2) * 16).cast(),
                hshift + ((code >> 4) & 3) as usize,
            ));
            data = data.add(group(
                data,
                output.as_mut_ptr().add((i + 3) * 16).cast(),
                hshift + ((code >> 6) & 3) as usize,
            ));
            i += 4;
        }
        while i < groups {
            if end.offset_from(data) < 24 {
                return Err("truncated SIMD group");
            }
            let code = (*header.get_unchecked(i / 4) >> ((i % 4) * 2)) & 3;
            data = data.add(group(
                data,
                output.as_mut_ptr().add(i * 16).cast(),
                hshift + code as usize,
            ));
            i += 1;
        }
        r.bytes = std::slice::from_raw_parts(data, end.offset_from(data) as usize);
    }
    Ok(())
}

#[inline(always)]
fn transpose(r: [v128; 4]) -> [v128; 4] {
    let a = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(r[0], r[1]);
    let b =
        i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(r[0], r[1]);
    let c = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(r[2], r[3]);
    let d =
        i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(r[2], r[3]);
    [
        i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a, c),
        i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a, c),
        i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(b, d),
        i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(b, d),
    ]
}
#[inline(always)]
fn delta<const CHANNEL: u8>(v: v128, rot: u32) -> v128 {
    match CHANNEL {
        0 => v128_xor(
            i8x16_sub(u8x16_splat(0), v128_and(v, u8x16_splat(1))),
            u8x16_shr(v, 1),
        ),
        1 => v128_xor(
            i16x8_sub(u16x8_splat(0), v128_and(v, u16x8_splat(1))),
            u16x8_shr(v, 1),
        ),
        _ => v128_or(i32x4_shl(v, rot), u32x4_shr(v, 32 - rot)),
    }
}
#[inline(always)]
fn accumulate<const CHANNEL: u8>(p: v128, v: v128) -> v128 {
    match CHANNEL {
        0 => i8x16_add(p, v),
        1 => i16x8_add(p, v),
        _ => v128_xor(p, v),
    }
}
// meshoptimizer-rs 0.2.9 codec/simd/wasm32.rs: delta_prefix. The two lane
// shifts compute the same wrapping-add/XOR prefix as meshoptimizer's FIXD chain.
#[inline(always)]
fn prefix<const CHANNEL: u8>(v: v128, previous: v128, rot: u32) -> v128 {
    let mut r = delta::<CHANNEL>(v, rot);
    let zero = i32x4_splat(0);
    r = accumulate::<CHANNEL>(r, i32x4_shuffle::<4, 0, 1, 2>(r, zero));
    r = accumulate::<CHANNEL>(r, i32x4_shuffle::<4, 4, 0, 1>(r, zero));
    accumulate::<CHANNEL>(r, previous)
}

#[inline(always)]
pub fn deltas<const CHANNEL: u8>(
    planes: &[&[u8]; 4],
    output: &mut [core::mem::MaybeUninit<u8>],
    n: usize,
    stride: usize,
    k: usize,
    last: &[u8],
    rot: u32,
) {
    // SAFETY: each plane has n initialized lanes; n is rounded to 16 and <=256. A block's
    // rounded count times stride fits output; k selects one aligned four-byte channel.
    unsafe {
        let mut p = v128_load32_splat(last.as_ptr().add(k).cast());
        let mut dst = output.as_mut_ptr().cast::<u8>().add(k);
        for j in (0..n).step_by(16) {
            let r = transpose([
                v128_load(planes[0].as_ptr().add(j).cast()),
                v128_load(planes[1].as_ptr().add(j).cast()),
                v128_load(planes[2].as_ptr().add(j).cast()),
                v128_load(planes[3].as_ptr().add(j).cast()),
            ]);
            for v in r {
                let v = prefix::<CHANNEL>(v, p, rot);
                p = i32x4_splat(i32x4_extract_lane::<3>(v));
                if stride == 4 {
                    v128_store(dst.cast(), v);
                } else {
                    v128_store32_lane::<0>(v, dst.cast());
                    v128_store32_lane::<1>(v, dst.add(stride).cast());
                    v128_store32_lane::<2>(v, dst.add(stride * 2).cast());
                    v128_store32_lane::<3>(v, dst.add(stride * 3).cast());
                }
                dst = dst.wrapping_add(stride * 4);
            }
        }
    }
}

fn snap(v: v128) -> v128 {
    f32x4_add(v, f32x4_splat(12582912.))
}
fn unpack8(v: v128, shift: u32) -> v128 {
    i32x4_shr(i32x4_shl(v, shift), 24)
}
fn unpack16(v: v128) -> v128 {
    i32x4_shr(i32x4_shl(v, 16), 16)
}
fn squares(x: v128, y: v128, z: v128) -> v128 {
    f32x4_add(f32x4_mul(x, x), f32x4_add(f32x4_mul(y, y), f32x4_mul(z, z)))
}
fn pack16(a: v128, b: v128, c: v128, d: v128) -> [v128; 2] {
    let ac = v128_or(v128_and(a, u32x4_splat(65535)), i32x4_shl(c, 16));
    let bd = v128_or(v128_and(b, u32x4_splat(65535)), i32x4_shl(d, 16));
    [
        i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(ac, bd),
        i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(ac, bd),
    ]
}
#[inline(always)]
unsafe fn oct(data: *mut u8, stride: usize) {
    let a = v128_load(data.cast());
    let b = if stride == 8 {
        v128_load(data.add(16).cast())
    } else {
        a
    };
    let (xf, yf, zf) = if stride == 4 {
        (unpack8(a, 24), unpack8(a, 16), unpack8(a, 8))
    } else {
        let xy = i32x4_shuffle::<0, 2, 4, 6>(a, b);
        let zw = i32x4_shuffle::<1, 3, 5, 7>(a, b);
        (
            unpack16(xy),
            i32x4_shr(xy, 16),
            v128_and(zw, u32x4_splat(32767)),
        )
    };
    let mut x = f32x4_convert_i32x4(xf);
    let mut y = f32x4_convert_i32x4(yf);
    let z = f32x4_sub(
        f32x4_convert_i32x4(zf),
        f32x4_add(f32x4_abs(x), f32x4_abs(y)),
    );
    let t = i32x4_min(z, i32x4_splat(0));
    x = f32x4_add(x, v128_xor(t, v128_and(x, f32x4_splat(-0.))));
    y = f32x4_add(y, v128_xor(t, v128_and(y, f32x4_splat(-0.))));
    let s = f32x4_div(
        f32x4_splat(if stride == 4 { 127. } else { 32767. }),
        f32x4_sqrt(squares(x, y, z)),
    );
    let x = snap(f32x4_mul(x, s));
    let y = snap(f32x4_mul(y, s));
    let z = snap(f32x4_mul(z, s));
    if stride == 4 {
        let res = v128_or(
            v128_and(a, u32x4_splat(0xff000000)),
            v128_or(
                v128_and(x, u32x4_splat(255)),
                v128_or(
                    i32x4_shl(v128_and(y, u32x4_splat(255)), 8),
                    i32x4_shl(v128_and(z, u32x4_splat(255)), 16),
                ),
            ),
        );
        v128_store(data.cast(), res);
    } else {
        let [r0, r1] = pack16(x, y, z, u32x4_splat(0));
        let mask = u64x2_splat(0xffff000000000000);
        v128_store(data.cast(), v128_or(r0, v128_and(a, mask)));
        v128_store(data.add(16).cast(), v128_or(r1, v128_and(b, mask)));
    }
}
#[inline(always)]
unsafe fn quat(data: *mut u8) {
    let a = v128_load(data.cast());
    let b = v128_load(data.add(16).cast());
    let xy = i32x4_shuffle::<0, 2, 4, 6>(a, b);
    let zc = i32x4_shuffle::<1, 3, 5, 7>(a, b);
    let cf = i32x4_shr(zc, 16);
    let s = f32x4_convert_i32x4(v128_or(cf, i32x4_splat(3)));
    let x = f32x4_convert_i32x4(unpack16(xy));
    let y = f32x4_convert_i32x4(i32x4_shr(xy, 16));
    let z = f32x4_convert_i32x4(unpack16(zc));
    let ws = f32x4_mul(s, s);
    let ww = f32x4_sub(f32x4_add(ws, ws), squares(x, y, z));
    let w = f32x4_sqrt(i32x4_max(ww, i32x4_splat(0)));
    let ss = f32x4_div(f32x4_splat(32767. / core::f32::consts::SQRT_2), s);
    let [r0, r1] = pack16(
        snap(f32x4_mul(w, ss)),
        snap(f32x4_mul(x, ss)),
        snap(f32x4_mul(y, ss)),
        snap(f32x4_mul(z, ss)),
    );
    let words = [
        u64x2_extract_lane::<0>(r0),
        u64x2_extract_lane::<1>(r0),
        u64x2_extract_lane::<0>(r1),
        u64x2_extract_lane::<1>(r1),
    ];
    let rotations = [
        u32x4_extract_lane::<0>(cf),
        u32x4_extract_lane::<1>(cf),
        u32x4_extract_lane::<2>(cf),
        u32x4_extract_lane::<3>(cf),
    ];
    for j in 0..4 {
        data.add(j * 8)
            .cast::<u64>()
            .write_unaligned(words[j].rotate_left(rotations[j] << 4));
    }
}
#[inline(always)]
unsafe fn exp(data: *mut u8) {
    let v = v128_load(data.cast());
    let e = i32x4_shl(i32x4_add(i32x4_shr(v, 24), i32x4_splat(127)), 23);
    let m = f32x4_convert_i32x4(i32x4_shr(i32x4_shl(v, 8), 8));
    v128_store(data.cast(), f32x4_mul(e, m));
}
#[inline(always)]
unsafe fn color(data: *mut u8, stride: usize) {
    let a = v128_load(data.cast());
    let b = if stride == 8 {
        v128_load(data.add(16).cast())
    } else {
        a
    };
    let (y, co, cg, mut alpha) = if stride == 4 {
        (
            v128_and(a, u32x4_splat(255)),
            unpack8(a, 16),
            unpack8(a, 8),
            u32x4_shr(a, 24),
        )
    } else {
        let yco = i32x4_shuffle::<0, 2, 4, 6>(a, b);
        let cga = i32x4_shuffle::<1, 3, 5, 7>(a, b);
        (
            v128_and(yco, u32x4_splat(65535)),
            i32x4_shr(yco, 16),
            unpack16(cga),
            u32x4_shr(cga, 16),
        )
    };
    let mut scale = alpha;
    for shift in [1, 2, 4] {
        scale = v128_or(scale, u32x4_shr(scale, shift));
    }
    if stride == 8 {
        scale = v128_or(scale, u32x4_shr(scale, 8));
    }
    alpha = v128_or(
        v128_and(i32x4_shl(alpha, 1), scale),
        v128_and(alpha, u32x4_splat(1)),
    );
    let ss = f32x4_div(
        f32x4_splat(if stride == 4 { 255. } else { 65535. }),
        f32x4_convert_i32x4(scale),
    );
    let r = snap(f32x4_mul(
        f32x4_convert_i32x4(i32x4_add(y, i32x4_sub(co, cg))),
        ss,
    ));
    let g = snap(f32x4_mul(f32x4_convert_i32x4(i32x4_add(y, cg)), ss));
    let b = snap(f32x4_mul(
        f32x4_convert_i32x4(i32x4_sub(y, i32x4_add(co, cg))),
        ss,
    ));
    let a = snap(f32x4_mul(f32x4_convert_i32x4(alpha), ss));
    if stride == 4 {
        let res = v128_or(
            v128_and(r, u32x4_splat(255)),
            v128_or(
                i32x4_shl(v128_and(g, u32x4_splat(255)), 8),
                v128_or(
                    i32x4_shl(v128_and(b, u32x4_splat(255)), 16),
                    i32x4_shl(a, 24),
                ),
            ),
        );
        v128_store(data.cast(), res);
    } else {
        let [r0, r1] = pack16(r, g, b, a);
        v128_store(data.cast(), r0);
        v128_store(data.add(16).cast(), r1);
    }
}

#[inline(always)]
unsafe fn process<const FILTER: u32, const STRIDE: usize>(data: *mut u8) {
    match FILTER {
        1 => oct(data, STRIDE),
        2 => quat(data),
        3 => exp(data),
        4 => color(data, STRIDE),
        _ => unreachable!(),
    }
}
fn dispatch<const FILTER: u32, const STRIDE: usize>(output: &mut [u8]) {
    let mut chunks = output.chunks_exact_mut(STRIDE * 4);
    // SAFETY: each call sees exactly four complete elements; the tail is padded to four.
    unsafe {
        for chunk in &mut chunks {
            process::<FILTER, STRIDE>(chunk.as_mut_ptr());
        }
        let remainder = chunks.into_remainder();
        if !remainder.is_empty() {
            let mut tail = [0u8; 32];
            tail[..remainder.len()].copy_from_slice(remainder);
            process::<FILTER, STRIDE>(tail.as_mut_ptr());
            remainder.copy_from_slice(&tail[..remainder.len()]);
        }
    }
}
pub fn filter(output: &mut [u8], stride: usize, filter: u32) {
    match (filter, stride) {
        (0, _) => (),
        (1, 4) => dispatch::<1, 4>(output),
        (1, 8) => dispatch::<1, 8>(output),
        (2, 8) => dispatch::<2, 8>(output),
        (3, _) => dispatch::<3, 4>(output),
        (4, 4) => dispatch::<4, 4>(output),
        (4, 8) => dispatch::<4, 8>(output),
        _ => unreachable!(),
    }
}
