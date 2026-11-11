//! meshoptimizer f0a684f1 decoding paths: vertexcodec.cpp, indexcodec.cpp,
//! vertexfilter.cpp. Filters use the reference WebAssembly arithmetic order.
//! Altered source: safe Rust slices and extension validity checks.
//! See ../licenses/LICENSE-meshoptimizer.

// Preserve the reference's deinterleaved delta traversal.
#![allow(clippy::needless_range_loop)]

use std::mem::MaybeUninit;

#[cfg(target_arch = "wasm32")]
mod simd;
#[cfg(target_arch = "wasm32")]
use simd::filter as apply_filter;

/// Compression metadata for one buffer view. Modes and filters follow the extension's numbering.
#[derive(Clone, Copy)]
pub struct View {
    pub index: usize,
    pub count: usize,
    pub stride: usize,
    pub mode: u32,
    pub filter: u32,
    pub khr: bool,
}

impl View {
    /// Decodes into the parent's buffer view; failures name that view.
    pub fn decode(self, source: &[u8], output: &mut [u8]) -> Result<(), String> {
        // SAFETY: initialized bytes can be borrowed as MaybeUninit. Decoders only write
        // initialized bytes, including on failure; they never deinitialize caller storage.
        let output = unsafe {
            std::slice::from_raw_parts_mut(
                output.as_mut_ptr().cast::<MaybeUninit<u8>>(),
                output.len(),
            )
        };
        self.decode_inner(source, output)
            .map_err(|message| format!("bufferView {}: meshopt {message}", self.index))
    }

    #[inline(always)]
    pub(super) fn decode_inner(
        self,
        source: &[u8],
        output: &mut [MaybeUninit<u8>],
    ) -> Result<(), &'static str> {
        let Self {
            count,
            stride,
            mode,
            filter,
            khr,
            ..
        } = self;
        if count == 0 || count.checked_mul(stride) != Some(output.len()) {
            return Err("invalid count or byteLength");
        }
        match mode {
            0 if (4..=256).contains(&stride) && stride % 4 == 0 => (),
            1 if (stride == 2 || stride == 4) && count % 3 == 0 && filter == 0 => (),
            2 if (stride == 2 || stride == 4) && filter == 0 => (),
            _ => return Err("invalid mode, stride or filter"),
        }
        match filter {
            0 => (),
            1 | 4 if (stride == 4 || stride == 8) && (filter != 4 || khr) => (),
            2 if stride == 8 => (),
            3 if stride % 4 == 0 => (),
            _ => return Err("invalid filter or stride"),
        }
        match mode {
            0 => attributes(source, output, count, stride, khr)?,
            1 if stride == 2 => triangles::<2, 6>(source, output, count)?,
            1 => triangles::<4, 12>(source, output, count)?,
            2 if stride == 2 => indices::<2>(source, output)?,
            2 => indices::<4>(source, output)?,
            _ => unreachable!(),
        }
        if filter != 0 {
            // SAFETY: successful decoding initialized every count*stride output byte.
            let initialized = unsafe {
                std::slice::from_raw_parts_mut(output.as_mut_ptr().cast::<u8>(), output.len())
            };
            apply_filter(initialized, stride, filter);
        }
        Ok(())
    }
}

type DecodeResult<T> = Result<T, &'static str>;

struct Reader<'a> {
    bytes: &'a [u8],
}
impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> DecodeResult<&'a [u8]> {
        let (head, tail) = self
            .bytes
            .split_at_checked(count)
            .ok_or("truncated stream")?;
        self.bytes = tail;
        Ok(head)
    }
    fn byte(&mut self) -> DecodeResult<u8> {
        Ok(self.take(1)?[0])
    }
    fn varint(&mut self) -> DecodeResult<u32> {
        // Both index codecs reserve enough tail bytes for a full five-byte varint.
        let groups = self.bytes.get(..5).ok_or("truncated varint")?;
        let mut result = 0;
        for (i, byte) in groups.iter().copied().enumerate() {
            result |= u32::from(byte & 127) << (i * 7);
            if byte < 128 {
                self.bytes = &self.bytes[i + 1..];
                return Ok(result);
            }
        }
        Err("unterminated varint")
    }
    fn index(&mut self, last: &mut u32) -> DecodeResult<u32> {
        *last = last.wrapping_add(unzigzag(self.varint()?));
        Ok(*last)
    }
    fn finish(&self, padding: usize) -> DecodeResult<()> {
        if self.bytes.len() == padding {
            Ok(())
        } else {
            Err("extra stream data")
        }
    }
}
fn unzigzag(v: u32) -> u32 {
    (v >> 1) ^ 0u32.wrapping_sub(v & 1)
}
#[cfg(not(target_arch = "wasm32"))]
fn read_le(bytes: &[u8]) -> u32 {
    bytes
        .iter()
        .enumerate()
        .fold(0, |v, (j, b)| v | (u32::from(*b) << (j * 8)))
}
#[inline(always)]
fn write_le(bytes: &mut [MaybeUninit<u8>], v: u32) {
    let encoded = v.to_le_bytes();
    let source = &encoded[..bytes.len()];
    // SAFETY: equal byte lengths, disjoint storage, and every copied byte is initialized.
    unsafe {
        std::ptr::copy_nonoverlapping(
            source.as_ptr(),
            bytes.as_mut_ptr().cast::<u8>(),
            source.len(),
        );
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn group(
    r: &mut Reader<'_>,
    output: &mut [std::mem::MaybeUninit<u8>],
    bits: usize,
) -> DecodeResult<()> {
    if bits == 0 {
        output.fill(std::mem::MaybeUninit::new(0));
    } else if bits == 8 {
        let bytes = r.take(16)?;
        for (dst, byte) in output.iter_mut().zip(bytes) {
            dst.write(*byte);
        }
    } else {
        let packed = r.take(bits * 2)?;
        let mask = (1 << bits) - 1;
        for (i, value) in output.iter_mut().enumerate() {
            let shift = if bits == 1 {
                i % 8
            } else {
                8 - bits - (i % (8 / bits)) * bits
            };
            let v = (packed[i / (8 / bits)] >> shift) & mask;
            value.write(if v == mask { r.byte()? } else { v });
        }
    }
    Ok(())
}

fn attributes(
    source: &[u8],
    output: &mut [MaybeUninit<u8>],
    count: usize,
    stride: usize,
    khr: bool,
) -> DecodeResult<()> {
    let version = match source.first() {
        Some(0xa0) => 0,
        Some(0xa1) if khr => 1,
        _ => return Err("invalid attribute header"),
    };
    let tail_size = stride + if version == 1 { stride / 4 } else { 0 };
    let padding = tail_size.max(if version == 0 { 32 } else { 24 });
    if source.len() < 1 + padding {
        return Err("truncated tail");
    }
    let tail = &source[source.len() - tail_size..];
    let channels = if version == 1 { &tail[stride..] } else { &[] };
    if channels
        .iter()
        .any(|c| c & 15 > 2 || (c & 15 < 2 && c >> 4 != 0))
    {
        return Err("invalid channel mode");
    }
    let mut baseline = MaybeUninit::<[u8; 256]>::uninit();
    // SAFETY: stride<=256 and tail contains stride initialized baseline bytes.
    // Only that initialized prefix is ever borrowed, matching decodeVertexBlockSimd.
    let last = unsafe {
        std::ptr::copy_nonoverlapping(tail.as_ptr(), baseline.as_mut_ptr().cast::<u8>(), stride);
        std::slice::from_raw_parts_mut(baseline.as_mut_ptr().cast::<u8>(), stride)
    };
    let mut r = Reader {
        bytes: &source[1..],
    };
    let block_size = ((8192 / stride) & !15).min(256);
    let mut deltas = [MaybeUninit::<u8>::uninit(); 1024];
    #[cfg(target_arch = "wasm32")]
    let mut transposed = [std::mem::MaybeUninit::<u8>::uninit(); 8192];
    let mut offset = 0;
    while offset < count {
        let n = (count - offset).min(block_size);
        let groups = n.div_ceil(16);
        let control = r.take(if version == 1 { stride / 4 } else { 0 })?;
        #[cfg(target_arch = "wasm32")]
        let target = if n.is_multiple_of(16) {
            &mut output[offset * stride..(offset + n) * stride]
        } else {
            &mut transposed[..groups * 16 * stride]
        };
        for k in (0..stride).step_by(4) {
            let ctrl_byte = if version == 1 { control[k / 4] } else { 0 };
            // decodeVertexBlockSimd packs four planes at the rounded block stride.
            for j in 0..4 {
                let delta = &mut deltas[j * groups * 16..(j + 1) * groups * 16];
                let ctrl = (ctrl_byte >> (j * 2)) & 3;
                match ctrl {
                    2 => delta.fill(MaybeUninit::new(0)),
                    3 => {
                        let start = r.bytes;
                        // The reference copies rounded lanes but consumes only n literals.
                        // Its stream tail supplies the lookahead for the last plane.
                        let bytes = r.take(groups * 16)?;
                        r.bytes = &start[n..];
                        // SAFETY: equal lengths, disjoint storage, initialized source bytes.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                bytes.as_ptr(),
                                delta.as_mut_ptr().cast::<u8>(),
                                bytes.len(),
                            );
                        }
                    }
                    _ => {
                        #[cfg(target_arch = "wasm32")]
                        simd::bytes(
                            &mut r,
                            delta,
                            groups,
                            if version == 0 { 0 } else { 4 + ctrl as usize },
                        )?;
                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            let header = r.take(groups.div_ceil(4))?;
                            for g in 0..groups {
                                let code = (header[g / 4] >> ((g % 4) * 2)) & 3;
                                let bits = if version == 0 {
                                    [0, 2, 4, 8][code as usize]
                                } else {
                                    [0, 1, 2, 4, 8][(code + ctrl) as usize]
                                };
                                group(&mut r, &mut delta[g * 16..g * 16 + 16], bits)?;
                            }
                        }
                    }
                }
            }
            // SAFETY: all four plane prefixes were initialized, including rounded group padding.
            let planes: [&[u8]; 4] = std::array::from_fn(|j| unsafe {
                std::slice::from_raw_parts(
                    deltas.as_ptr().add(j * groups * 16).cast::<u8>(),
                    groups * 16,
                )
            });
            let channel = if version == 1 { channels[k / 4] } else { 0 };
            #[cfg(target_arch = "wasm32")]
            match channel & 3 {
                0 => simd::deltas::<0>(&planes, target, groups * 16, stride, k, last, 0),
                1 => simd::deltas::<1>(&planes, target, groups * 16, stride, k, last, 0),
                2 => simd::deltas::<2>(
                    &planes,
                    target,
                    groups * 16,
                    stride,
                    k,
                    last,
                    (32 - u32::from(channel >> 4)) & 31,
                ),
                _ => unreachable!(),
            }
            #[cfg(not(target_arch = "wasm32"))]
            for j in (0..4).step_by([1, 2, 4][(channel & 3) as usize]) {
                let width = [1, 2, 4][(channel & 3) as usize];
                let mut p = read_le(&last[k + j..k + j + width]);
                for i in 0..n {
                    let mut v = 0;
                    for b in 0..width {
                        v |= u32::from(planes[j + b][i]) << (b * 8);
                    }
                    p = if width == 4 {
                        p ^ v.rotate_right(u32::from(channel >> 4))
                    } else {
                        p.wrapping_add(unzigzag(v))
                    };
                    let dst = (offset + i) * stride + k + j;
                    write_le(&mut output[dst..dst + width], p);
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        if !n.is_multiple_of(16) {
            output[offset * stride..(offset + n) * stride]
                .copy_from_slice(&transposed[..n * stride]);
        }
        // SAFETY: every channel of the last element was initialized before this read.
        let previous = unsafe {
            std::slice::from_raw_parts(
                output.as_ptr().add((offset + n - 1) * stride).cast::<u8>(),
                stride,
            )
        };
        last[..stride].copy_from_slice(previous);
        offset += n;
    }
    r.finish(padding)
}

struct Fifo<T: Copy> {
    entries: [T; 16],
    offset: usize,
}
impl<T: Copy> Fifo<T> {
    fn new(initial: T) -> Self {
        Self {
            entries: [initial; 16],
            offset: 0,
        }
    }
    fn get(&self, distance: usize) -> T {
        self.entries[self.offset.wrapping_sub(1).wrapping_sub(distance) & 15]
    }
    fn push(&mut self, v: T) {
        self.push_if(v, true);
    }
    fn push_if(&mut self, v: T, push: bool) {
        self.entries[self.offset & 15] = v;
        self.offset = (self.offset + usize::from(push)) & 15;
    }
}
fn next_index(next: &mut u32) -> u32 {
    let v = *next;
    *next = next.wrapping_add(1);
    v
}
fn triangles<const WIDTH: usize, const TRI_BYTES: usize>(
    source: &[u8],
    output: &mut [MaybeUninit<u8>],
    count: usize,
) -> DecodeResult<()> {
    let codes_end = 1 + count / 3;
    if source.len() < codes_end + 16 {
        return Err("truncated triangles");
    }
    if source[0] != 0xe1 {
        return Err("invalid triangle header");
    }
    let tail_start = source.len() - 16;
    let table = &source[tail_start..];
    if table[14..] != [0, 0] || table.iter().any(|v| v & 15 == 15 || v >> 4 == 15) {
        return Err("invalid triangle lookup table");
    }
    let mut r = Reader {
        bytes: &source[codes_end..],
    };
    let mut edges = Fifo::new((u32::MAX, u32::MAX));
    let mut vertices = Fifo::new(u32::MAX);
    let mut next = 0u32;
    let mut last = 0u32;
    for (code, out) in source[1..codes_end]
        .iter()
        .zip(output.as_chunks_mut::<TRI_BYTES>().0)
    {
        let (a, b, c);
        if *code < 0xf0 {
            (a, b) = edges.get((code >> 4) as usize);
            let fec = code & 15;
            if fec < 13 {
                let cached = vertices.get(fec as usize);
                c = if fec == 0 { next } else { cached };
                next = next.wrapping_add(u32::from(fec == 0));
                vertices.push_if(c, fec == 0);
            } else {
                last = if fec == 15 {
                    r.index(&mut last)?
                } else {
                    last.wrapping_add(u32::from(fec) * 2).wrapping_sub(27)
                };
                c = last;
                vertices.push(c);
            }
        } else if *code < 0xfe {
            let aux = table[(code & 15) as usize];
            let feb = aux >> 4;
            let fec = aux & 15;
            a = next_index(&mut next);
            let bf = vertices.get(feb.wrapping_sub(1) as usize);
            b = if feb == 0 { next } else { bf };
            next = next.wrapping_add(u32::from(feb == 0));
            let cf = vertices.get(fec.wrapping_sub(1) as usize);
            c = if fec == 0 { next } else { cf };
            next = next.wrapping_add(u32::from(fec == 0));
            vertices.push(a);
            vertices.push_if(b, feb == 0);
            vertices.push_if(c, fec == 0);
            edges.push((b, a));
        } else {
            let aux = r.byte()?;
            if aux == 0 {
                next = 0;
            }
            let feb = aux >> 4;
            let fec = aux & 15;
            a = if *code == 0xff {
                r.index(&mut last)?
            } else {
                next_index(&mut next)
            };
            b = match feb {
                0 => next_index(&mut next),
                15 => r.index(&mut last)?,
                _ => vertices.get((feb - 1) as usize),
            };
            c = match fec {
                0 => next_index(&mut next),
                15 => r.index(&mut last)?,
                _ => vertices.get((fec - 1) as usize),
            };
            vertices.push(a);
            vertices.push_if(b, feb == 0 || feb == 15);
            vertices.push_if(c, fec == 0 || fec == 15);
            edges.push((b, a));
        }
        edges.push((c, b));
        edges.push((a, c));
        // meshoptimizer-rs index.rs: one checked fixed-size triangle, three constant-width stores.
        write_le(&mut out[..WIDTH], a);
        write_le(&mut out[WIDTH..WIDTH * 2], b);
        write_le(&mut out[WIDTH * 2..], c);
    }
    r.finish(16)
}
fn indices<const WIDTH: usize>(source: &[u8], output: &mut [MaybeUninit<u8>]) -> DecodeResult<()> {
    if source.len() < 5 {
        return Err("truncated indices");
    }
    if source[0] != 0xd1 {
        return Err("invalid index header");
    }
    let mut r = Reader {
        bytes: &source[1..],
    };
    let mut last = [0u32; 2];
    for dst in output.as_chunks_mut::<WIDTH>().0 {
        let v = r.varint()?;
        let baseline = (v & 1) as usize;
        last[baseline] = last[baseline].wrapping_add(unzigzag(v >> 1));
        write_le(dst, last[baseline]);
    }
    r.finish(4)
}

#[cfg(not(target_arch = "wasm32"))]
fn apply_filter(output: &mut [u8], stride: usize, filter: u32) {
    if filter == 0 {
        return;
    }
    if filter == 3 {
        for dst in output.as_chunks_mut::<4>().0 {
            let v = read_le(dst) as i32;
            let m = (v << 8) >> 8;
            let e = v >> 24;
            let value = f32::from_bits(((e + 127) as u32) << 23) * m as f32;
            dst.copy_from_slice(&value.to_le_bytes());
        }
        return;
    }
    let width = stride / 4;
    for dst in output.chunks_exact_mut(stride) {
        let mut u = [0u32; 4];
        let mut s = [0i32; 4];
        for j in 0..4 {
            u[j] = read_le(&dst[j * width..(j + 1) * width]);
            s[j] = if width == 1 {
                u[j] as i8 as i32
            } else {
                u[j] as i16 as i32
            };
        }
        let mut result = [0u32; 4];
        match filter {
            1 => {
                let mut x = s[0] as f32;
                let mut y = s[1] as f32;
                let z = s[2] as f32 - (x.abs() + y.abs());
                let t = z.min(0.);
                x += if x.is_sign_negative() { -t } else { t };
                y += if y.is_sign_negative() { -t } else { t };
                let scale =
                    (if width == 1 { 127. } else { 32767. }) / (x * x + (y * y + z * z)).sqrt();
                result = [snap(x * scale), snap(y * scale), snap(z * scale), u[3]];
            }
            2 => {
                let x = s[0] as f32;
                let y = s[1] as f32;
                let z = s[2] as f32;
                let one = (s[3] | 3) as f32;
                let w = (one * one * 2. - (x * x + (y * y + z * z))).max(0.).sqrt();
                let scale = (32767. / 2f32.sqrt()) / one;
                for (j, value) in [w, x, y, z].into_iter().enumerate() {
                    result[(j + (u[3] as usize & 3)) & 3] = snap(value * scale);
                }
            }
            4 => {
                let mut a_scale = u[3];
                for shift in [1, 2, 4, 8] {
                    a_scale |= a_scale >> shift;
                }
                let y = u[0] as i32;
                let co = s[1];
                let cg = s[2];
                let a = ((u[3] << 1) & a_scale) | (u[3] & 1);
                let scale = (if width == 1 { 255. } else { 65535. }) / a_scale as f32;
                for (j, v) in [y + co - cg, y + cg, y - co - cg, a as i32]
                    .into_iter()
                    .enumerate()
                {
                    result[j] = snap(v as f32 * scale);
                }
            }
            _ => unreachable!(),
        }
        for j in 0..4 {
            dst[j * width..(j + 1) * width].copy_from_slice(&result[j].to_le_bytes()[..width]);
        }
    }
}
// The reference WASM filter's float snap rounds to nearest, ties to even.
#[cfg(not(target_arch = "wasm32"))]
fn snap(v: f32) -> u32 {
    (v + 12582912.).to_bits()
}
