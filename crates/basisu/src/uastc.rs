//! UASTC LDR 4x4 from Basis Universal 073eb6a0, unpack_uastc and its target paths.
//! Altered source: checked Rust blocks; no HDR, BC1/3, ETC or PVRTC targets.
//! Copyright 2016–2026 Binomial LLC. See ../licenses/NOTICE-basis-universal.
mod tables;
use crate::{Ktx2, Model, Result};
use tables::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UastcTarget {
    Astc,
    Bc7,
    Rgba8,
}

#[derive(Default)]
struct Unpacked {
    mode: usize,
    common_pattern: usize,
    solid_color: [u8; 4],
    ccs: usize,
    partition_seed: usize,
    endpoints: [u8; 24],
    weights: [u8; 32],
}
// All field widths depend on the validated mode, not payload values. Prove
// each layout fits its fixed 16-byte block before using read_bits1_to_9_fst.
const _: () = {
    let mut mode = 0;
    while mode < 19 {
        if mode != 8 {
            let subsets = UASTC_MODE_SUBSETS[mode];
            let planes = UASTC_MODE_PLANES[mode];
            let values = UASTC_MODE_COMPS[mode] * 2 * subsets;
            let [n, trits, quints] = ASTC_BISE_RANGE_TABLE[UASTC_MODE_ENDPOINT_RANGES[mode]];
            let group = if trits != 0 {
                5
            } else if quints != 0 {
                3
            } else {
                0
            };
            let mut tq_bits = 0;
            if group != 0 {
                let mut remaining = values;
                while remaining != 0 {
                    let count = if remaining < group { remaining } else { group };
                    tq_bits += if trits != 0 {
                        [0, 2, 4, 5, 7, 8][count]
                    } else {
                        [0, 3, 5, 7][count]
                    };
                    remaining -= count;
                }
            }
            let total = UASTC_MODE_HUFF_CODES[mode][1]
                + UASTC_MODE_TOTAL_HINT_BITS[mode]
                + if subsets > 1 {
                    if mode == 3 {
                        4
                    } else {
                        5
                    }
                } else {
                    0
                }
                + if planes == 2 && mode != 17 { 2 } else { 0 }
                + values * n
                + tq_bits
                + 16 * planes * UASTC_MODE_WEIGHT_BITS[mode]
                - subsets * planes;
            assert!(total <= 128);
        }
        mode += 1;
    }
};
#[inline(always)]
fn read(bytes: &[u8; 16], offset: &mut usize, count: usize) -> usize {
    debug_assert!(*offset + count <= 128);
    if count == 0 {
        return 0;
    }
    let index = *offset >> 3;
    let shift = *offset & 7;
    // read_bits1_to_9_fst's native word path, expressed without an aligned
    // C++ pointer cast: Wasm load16_u permits unaligned addresses. The final
    // byte still takes the bounded single-byte tail path.
    let word = if index < 15 {
        u16::from_le_bytes(bytes[index..index + 2].try_into().unwrap())
    } else {
        u16::from(bytes[index])
    };
    *offset += count;
    (usize::from(word) >> shift) & ((1 << count) - 1)
}
#[inline(always)]
fn pattern(mode: usize, common: usize) -> &'static [u8; 16] {
    match UASTC_MODE_SUBSETS[mode] {
        3 => &ASTC_BC7_PATTERNS3[common],
        2 if mode == 7 => &BC7_3_ASTC2_PATTERNS2[common],
        2 => &ASTC_BC7_PATTERNS2[common],
        _ => &[0; 16],
    }
}
fn unpack_uastc(bytes: &[u8; 16], u: &mut Unpacked, blue_contract_check: bool) -> Result<()> {
    let mode = UASTC_HUFF_MODES[(bytes[0] & 127) as usize];
    if mode >= 19 {
        return Err("invalid UASTC mode");
    }
    u.mode = mode;
    u.common_pattern = 0;
    let mut offset = UASTC_MODE_HUFF_CODES[mode][1];
    if mode == 8 {
        for c in &mut u.solid_color {
            *c = read(bytes, &mut offset, 8) as u8;
        }
        return Ok(());
    }
    offset += UASTC_MODE_TOTAL_HINT_BITS[mode];
    let subsets = UASTC_MODE_SUBSETS[mode];
    if subsets > 1 {
        u.common_pattern = read(bytes, &mut offset, if mode == 3 { 4 } else { 5 });
        let parts = if mode == 3 {
            &ASTC_BC7_COMMON_PARTITIONS3[..]
        } else if mode == 7 {
            &BC7_3_ASTC2_COMMON_PARTITIONS[..]
        } else {
            &ASTC_BC7_COMMON_PARTITIONS2[..]
        };
        u.partition_seed = parts
            .get(u.common_pattern)
            .ok_or("invalid UASTC partition")?[1];
    }
    let planes = UASTC_MODE_PLANES[mode];
    if planes == 2 {
        u.ccs = if mode == 17 {
            3
        } else {
            read(bytes, &mut offset, 2)
        };
    }
    let comps = UASTC_MODE_COMPS[mode];
    let total_values = comps * 2 * subsets;
    let endpoint_range = UASTC_MODE_ENDPOINT_RANGES[mode];
    let [ep_bits, ep_trits, ep_quints] = ASTC_BISE_RANGE_TABLE[endpoint_range];
    let bundle_size = if ep_trits != 0 {
        5
    } else if ep_quints != 0 {
        3
    } else {
        0
    };
    let mut tq_values = [0; 8];
    if bundle_size != 0 {
        let total_tqs = total_values.div_ceil(bundle_size);
        for (i, tq) in tq_values.iter_mut().enumerate().take(total_tqs) {
            let remaining = (total_values - i * bundle_size).min(bundle_size);
            let count = if ep_trits != 0 {
                [0, 2, 4, 5, 7, 8][remaining]
            } else {
                [0, 3, 5, 7][remaining]
            };
            *tq = read(bytes, &mut offset, count);
        }
    }
    let mut accum = 0;
    let mut accum_remaining = 0;
    let mut next_tq = 0;
    for i in 0..total_values {
        let mut value = read(bytes, &mut offset, ep_bits);
        if bundle_size != 0 {
            if accum_remaining == 0 {
                accum = tq_values[next_tq];
                next_tq += 1;
                accum_remaining = bundle_size;
            }
            accum_remaining -= 1;
            let mul = if ep_trits != 0 { 3 } else { 5 };
            value |= (accum % mul) << ep_bits;
            accum /= mul;
        }
        u.endpoints[i] = value as u8;
    }
    let pat = pattern(mode, u.common_pattern);
    let anchors = match subsets {
        3 => ASTC_BC7_PATTERN3_ANCHORS[u.common_pattern],
        2 if mode == 7 => BC7_3_ASTC2_PATTERNS2_ANCHORS[u.common_pattern],
        2 => ASTC_BC7_PATTERN2_ANCHORS[u.common_pattern],
        _ => [0; 3],
    };
    let weight_bits = UASTC_MODE_WEIGHT_BITS[mode];
    if mode == 18 {
        for i in 0..16 {
            u.weights[i] = read(bytes, &mut offset, weight_bits - usize::from(i == 0)) as u8;
        }
    } else {
        // unpack_uastc loads the final two dwords once for <=64 weight bits.
        let high = u64::from_le_bytes(bytes[8..].try_into().unwrap());
        let mut bits = if offset >= 64 {
            high >> (offset - 64)
        } else {
            (high << (64 - offset)) | u64::from(bytes[7] >> (offset - 56))
        };
        let mask = (1u64 << weight_bits) - 1;
        if subsets == 1 && planes == 1 && weight_bits == 4 {
            // unpack_uastc's unrolled, most-common 4-bit weight path.
            u.weights[0] = (bits & 7) as u8;
            u.weights[1] = ((bits >> 3) & 15) as u8;
            u.weights[2] = ((bits >> 7) & 15) as u8;
            u.weights[3] = ((bits >> 11) & 15) as u8;
            u.weights[4] = ((bits >> 15) & 15) as u8;
            u.weights[5] = ((bits >> 19) & 15) as u8;
            u.weights[6] = ((bits >> 23) & 15) as u8;
            u.weights[7] = ((bits >> 27) & 15) as u8;
            u.weights[8] = ((bits >> 31) & 15) as u8;
            u.weights[9] = ((bits >> 35) & 15) as u8;
            u.weights[10] = ((bits >> 39) & 15) as u8;
            u.weights[11] = ((bits >> 43) & 15) as u8;
            u.weights[12] = ((bits >> 47) & 15) as u8;
            u.weights[13] = ((bits >> 51) & 15) as u8;
            u.weights[14] = ((bits >> 55) & 15) as u8;
            u.weights[15] = ((bits >> 59) & 15) as u8;
        } else {
            for i in 0..16 * planes {
                let count = weight_bits
                    - usize::from(if subsets == 1 {
                        i < planes
                    } else {
                        anchors.contains(&i)
                    });
                u.weights[i] = (bits & (mask >> (weight_bits - count))) as u8;
                bits >>= count;
            }
        }
    }
    if blue_contract_check && comps >= 3 {
        let mut invert = [false; 3];
        let mut any = false;
        for (s, inv) in invert.iter_mut().enumerate().take(subsets) {
            let base = s * comps * 2;
            let sum = |side| {
                (0..3)
                    .map(|c| {
                        usize::from(
                            ASTC_UNQUANT[endpoint_range][u.endpoints[base + c * 2 + side] as usize],
                        )
                    })
                    .sum::<usize>()
            };
            if sum(1) < sum(0) {
                for c in 0..comps {
                    u.endpoints.swap(base + c * 2, base + c * 2 + 1);
                }
                *inv = true;
                any = true;
            }
        }
        if any && subsets == 1 {
            // Expose the whole byte span once instead of checking computed
            // pixel/plane indices. The meshopt WASM port's fixed-width slice
            // traversal (meshopt::sequence) uses the same bounds-proof shape.
            let mask = ((1 << weight_bits) - 1) as u8;
            for weight in &mut u.weights[..16 * planes] {
                *weight = mask - *weight;
            }
        } else if any {
            for (i, &s) in pat.iter().enumerate() {
                if invert[usize::from(s)] {
                    for p in 0..planes {
                        u.weights[i * planes + p] =
                            (1 << weight_bits) - 1 - u.weights[i * planes + p];
                    }
                }
            }
        }
    }
    Ok(())
}
// astc_set_bits_1_to_9, also used for the 10/11-bit ASTC header fields.
#[inline(always)]
fn set(dst: &mut [u8; 16], offset: &mut usize, value: usize, count: usize) {
    if count == 0 {
        return;
    }
    let index = *offset >> 3;
    let shift = *offset & 7;
    let value = value << shift;
    dst[index] |= value as u8;
    if count + shift > 8 {
        dst[index + 1] |= (value >> 8) as u8;
    }
    if count + shift > 16 {
        dst[index + 2] |= (value >> 16) as u8;
    }
    *offset += count;
}
fn set_bise(dst: &mut [u8], offset: &mut usize, mut value: usize, mut count: usize) {
    while count != 0 {
        let n = count.min(8 - (*offset & 7));
        dst[*offset >> 3] |= (value << (*offset & 7)) as u8;
        value >>= n;
        count -= n;
        *offset += n;
    }
}
fn pack_astc(u: &Unpacked, dst: &mut [u8; 16]) {
    if u.mode == 8 {
        dst[..8].copy_from_slice(&[0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]);
        for c in 0..4 {
            dst[8 + c * 2] = u.solid_color[c];
            dst[9 + c * 2] = u.solid_color[c];
        }
        return;
    }
    dst.fill(0);
    let mode = u.mode;
    let subsets = UASTC_MODE_SUBSETS[mode];
    let planes = UASTC_MODE_PLANES[mode];
    let weight_bits = UASTC_MODE_WEIGHT_BITS[mode];
    let mut pos = 0;
    set(dst, &mut pos, UASTC_MODE_ASTC_BLOCK_MODE[mode], 11);
    set(dst, &mut pos, subsets - 1, 2);
    let cem = UASTC_MODE_CEM[mode];
    if subsets == 1 {
        set(dst, &mut pos, cem, 4);
    } else {
        set(dst, &mut pos, u.partition_seed, 10);
        set(dst, &mut pos, cem << 2, 6);
    }
    if planes == 2 {
        set(dst, &mut (128 - 32 * weight_bits - 2), u.ccs, 2);
    }
    let [n, trits, quints] = ASTC_BISE_RANGE_TABLE[UASTC_MODE_ENDPOINT_RANGES[mode]];
    let count = UASTC_MODE_COMPS[mode] * 2 * subsets;
    if trits == 0 && quints == 0 {
        for &value in &u.endpoints[..count] {
            set(dst, &mut pos, usize::from(value), n);
        }
    } else {
        let group = if trits != 0 { 5 } else { 3 };
        for start in (0..count).step_by(group) {
            let mut vals = [0; 5];
            let limit = group.min(count - start);
            for (i, value) in vals.iter_mut().enumerate().take(limit) {
                *value = usize::from(u.endpoints[start + i]);
            }
            let mut tq = 0;
            let mut mul = 1;
            for &v in &vals[..group] {
                tq += (v >> n) * mul;
                mul *= if trits != 0 { 3 } else { 5 };
            }
            let bits = vals.map(|v| v & ((1 << n) - 1));
            if trits != 0 {
                let t = ASTC_TRIT_ENCODE[tq];
                // astc_encode_trits interleaves a group into two fields.
                set_bise(
                    dst,
                    &mut pos,
                    bits[0] | ((t & 3) << n) | (bits[1] << (n + 2)),
                    n * 2 + 2,
                );
                let value = ((t >> 2) & 3)
                    | (bits[2] << 2)
                    | (((t >> 4) & 1) << (n + 2))
                    | (bits[3] << (n + 3))
                    | (((t >> 5) & 3) << (n * 2 + 3))
                    | (bits[4] << (n * 2 + 5))
                    | (((t >> 7) & 1) << (n * 3 + 5));
                set_bise(dst, &mut pos, value, n * 3 + 6);
            } else {
                let t = ASTC_QUINT_ENCODE[tq];
                let value = bits[0]
                    | ((t & 7) << n)
                    | (bits[1] << (n + 3))
                    | (((t >> 3) & 3) << (n * 2 + 3))
                    | (bits[2] << (n * 2 + 5))
                    | (((t >> 5) & 3) << (n * 3 + 5));
                set_bise(dst, &mut pos, value, n * 3 + 7);
            }
        }
    }
    match weight_bits {
        1 => pack_weights::<1>(u, dst, 16 * planes),
        2 => pack_weights::<2>(u, dst, 16 * planes),
        3 => pack_weights::<3>(u, dst, 16 * planes),
        4 => pack_weights::<4>(u, dst, 16 * planes),
        5 => pack_weights::<5>(u, dst, 16 * planes),
        _ => unreachable!(),
    }
}
#[inline(always)]
fn pack_weights<const N: usize>(u: &Unpacked, dst: &mut [u8; 16], count: usize) {
    // pack_astc_block specializes each weight width and reverses via tables.
    const REVERSE5: [u8; 32] = [
        0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30, 1, 17, 9, 25, 5, 21, 13, 29, 3,
        19, 11, 27, 7, 23, 15, 31,
    ];
    if N == 4 {
        const REVERSE: [u8; 16] = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15];
        for (byte, weights) in dst
            .iter_mut()
            .rev()
            .zip(u.weights[..count].as_chunks::<2>().0)
        {
            *byte |= (REVERSE[usize::from(weights[0]) & 15] << 4)
                | REVERSE[usize::from(weights[1]) & 15];
        }
    } else if N == 2 {
        const REVERSE: [u8; 4] = [0, 2, 1, 3];
        for (byte, weights) in dst
            .iter_mut()
            .rev()
            .zip(u.weights[..count].as_chunks::<4>().0)
        {
            *byte |= (REVERSE[usize::from(weights[0]) & 3] << 6)
                | (REVERSE[usize::from(weights[1]) & 3] << 4)
                | (REVERSE[usize::from(weights[2]) & 3] << 2)
                | REVERSE[usize::from(weights[3]) & 3];
        }
    } else if N == 3 || N == 5 {
        // Eight weights form exactly three or five bytes. Assemble that
        // complete byte group rather than updating crossing fields twice.
        let bytes = if N == 3 { 3 } else { 5 };
        for (output, weights) in dst
            .rchunks_exact_mut(bytes)
            .zip(u.weights[..count].as_chunks::<8>().0)
        {
            let mut bits = 0u64;
            for &weight in weights {
                let rev = u64::from(REVERSE5[usize::from(weight) & 31]) >> (5 - N);
                bits = (bits << N) | rev;
            }
            for byte in output.iter_mut() {
                *byte |= bits as u8;
                bits >>= 8;
            }
        }
    } else {
        for (i, &weight) in u.weights[..count].iter().enumerate() {
            let rev = usize::from(REVERSE5[usize::from(weight) & 31]) >> (5 - N);
            set(dst, &mut (128 - (i + 1) * N), rev, N);
        }
    }
}
fn rgba(u: &Unpacked) -> [[u8; 4]; 16] {
    if u.mode == 8 {
        return [u.solid_color; 16];
    }
    let mode = u.mode;
    let comps = UASTC_MODE_COMPS[mode];
    let planes = UASTC_MODE_PLANES[mode];
    let range = UASTC_MODE_ENDPOINT_RANGES[mode];
    let weights: &[usize] = match UASTC_MODE_WEIGHT_BITS[mode] {
        1 => &[0, 64],
        2 => &[0, 21, 43, 64],
        3 => &[0, 9, 18, 27, 37, 46, 55, 64],
        4 => &[0, 4, 8, 12, 17, 21, 25, 29, 35, 39, 43, 47, 52, 56, 60, 64],
        _ => &[
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 34, 36, 38, 40, 42, 44, 46,
            48, 50, 52, 54, 56, 58, 60, 62, 64,
        ],
    };
    let mut colors = [[[255u8; 4]; 32]; 3];
    for (s, palette) in colors.iter_mut().enumerate().take(UASTC_MODE_SUBSETS[mode]) {
        for (l, &w) in weights.iter().enumerate() {
            for (c, channel) in palette[l].iter_mut().enumerate() {
                let comp = if comps == 2 { usize::from(c == 3) } else { c };
                if comp < comps {
                    let lo = usize::from(
                        ASTC_UNQUANT[range][u.endpoints[s * comps * 2 + comp * 2] as usize],
                    ) * 257;
                    let hi = usize::from(
                        ASTC_UNQUANT[range][u.endpoints[s * comps * 2 + comp * 2 + 1] as usize],
                    ) * 257;
                    *channel = (((lo * (64 - w) + hi * w + 32) >> 6) >> 8) as u8;
                }
            }
        }
    }
    let mut pixels = [[0; 4]; 16];
    for (i, &s) in pattern(mode, u.common_pattern).iter().enumerate() {
        for c in 0..4 {
            let p = usize::from(planes == 2 && c == u.ccs);
            pixels[i][c] = colors[usize::from(s)][u.weights[i * planes + p] as usize][c];
        }
    }
    pixels
}
#[derive(Default)]
struct Bc7 {
    mode: usize,
    partition: usize,
    rotation: usize,
    low: [[usize; 4]; 3],
    high: [[usize; 4]; 3],
    pbits: [[usize; 2]; 3],
    selectors: [usize; 16],
    alpha_selectors: [usize; 16],
}
fn pbits(
    lo: [usize; 4],
    hi: [usize; 4],
    comps: usize,
    bits: usize,
    shared: bool,
) -> ([[usize; 4]; 2], [usize; 2]) {
    let mut best = [f32::MAX; 2];
    let mut colors = [[0; 4]; 2];
    let mut pb = [0; 2];
    let scale: i32 = (1 << (bits + 1)) - 1;
    for p in 0..2 {
        let mut candidates = [[0; 4]; 2];
        let mut err = [0.0f32; 2];
        let mut deltas = [[0.0f32; 4]; 2];
        for side in 0..2 {
            for c in 0..4 {
                let x = (if side == 0 { lo[c] } else { hi[c] }) as f32 / 255.0;
                let quant = ((((x * scale as f32 - p as f32) / 2.0 + 0.5) as i32) * 2 + p)
                    .clamp(p, scale - 1 + p) as usize;
                candidates[side][c] = quant >> 1;
                let scaled = quant << (7 - bits);
                let scaled = scaled | (scaled >> (bits + 1));
                if c < comps {
                    let delta = if shared {
                        scaled as f32 / 255.0 - x
                    } else {
                        scaled as f32 - x * 255.0
                    };
                    deltas[side][c] = delta * delta;
                    err[side] += delta * delta;
                }
            }
        }
        if shared {
            let mut error = 0.0f32;
            for (lo, hi) in deltas[0].iter().zip(&deltas[1]).take(comps) {
                error += lo + hi;
            }
            if error < best[0] {
                best[0] = error;
                colors = candidates;
                pb = [p as usize; 2];
            }
        } else {
            for side in 0..2 {
                if err[side] < best[side] {
                    best[side] = err[side];
                    colors[side] = candidates[side];
                    pb[side] = p as usize;
                }
            }
        }
    }
    (colors, pb)
}
fn endpoints(u: &Unpacked, subset: usize) -> [[usize; 4]; 2] {
    let comps = UASTC_MODE_COMPS[u.mode];
    let range = UASTC_MODE_ENDPOINT_RANGES[u.mode];
    let mut colors = [[255; 4]; 2];
    for (side, color) in colors.iter_mut().enumerate() {
        for (c, value) in color.iter_mut().enumerate() {
            let comp = if comps == 2 { usize::from(c == 3) } else { c };
            if comp < comps {
                *value = usize::from(
                    ASTC_UNQUANT[range][u.endpoints[subset * comps * 2 + comp * 2 + side] as usize],
                );
            }
        }
    }
    colors
}
fn bc7(u: &Unpacked) -> [u8; 16] {
    let mut b = Bc7::default();
    let mode = u.mode;
    let comps = UASTC_MODE_COMPS[mode];
    for (dst, &src) in b.selectors.iter_mut().zip(&u.weights[..16]) {
        *dst = usize::from(src);
    }
    match mode {
        0 | 5 | 10 | 12 | 14 | 15 | 18 => {
            b.mode = 6;
            let [lo, hi] = endpoints(u, 0);
            let (colors, pb) = pbits(lo, hi, if comps == 2 { 4 } else { comps }, 7, false);
            b.low[0] = colors[0];
            b.high[0] = colors[1];
            b.pbits[0] = pb;
            if comps == 3 {
                b.low[0][3] = 127;
                b.high[0][3] = 127;
            }
            let map: &[usize] = match mode {
                18 => &[
                    0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 7, 8, 9, 9, 9, 10, 10, 11, 11, 12,
                    12, 13, 13, 14, 14, 15, 15,
                ],
                14 => &[0, 5, 10, 15],
                5 | 12 => &[0, 2, 4, 6, 9, 11, 13, 15],
                _ => &[],
            };
            if !map.is_empty() {
                for s in &mut b.selectors {
                    *s = map[*s];
                }
            }
        }
        1 | 2 | 4 | 9 | 16 => {
            b.mode = match mode {
                1 | 4 => 3,
                2 => 1,
                _ => 7,
            };
            let mut invert = false;
            if mode != 1 {
                let part = ASTC_BC7_COMMON_PARTITIONS2[u.common_pattern];
                b.partition = part[0];
                invert = part[2] != 0;
            }
            for s in 0..UASTC_MODE_SUBSETS[mode] {
                let [lo, hi] = endpoints(u, s);
                let (colors, pb) = pbits(
                    lo,
                    hi,
                    if b.mode == 7 { 4 } else { 3 },
                    if b.mode == 7 {
                        5
                    } else if mode == 2 {
                        6
                    } else {
                        7
                    },
                    mode == 2,
                );
                let dest = if invert { 1 - s } else { s };
                b.low[dest] = colors[0];
                b.high[dest] = colors[1];
                b.pbits[dest] = pb;
            }
            if mode == 1 {
                b.low[1] = b.low[0];
                b.high[1] = b.high[0];
                b.pbits[1] = b.pbits[0];
            }
        }
        3 | 7 => {
            b.mode = 2;
            let part = if mode == 3 {
                ASTC_BC7_COMMON_PARTITIONS3[u.common_pattern]
            } else {
                BC7_3_ASTC2_COMMON_PARTITIONS[u.common_pattern]
            };
            b.partition = part[0];
            for (s, &permuted) in ASTC_TO_BC7_PARTITION_INDEX_PERM_TABLES[part[2]]
                .iter()
                .enumerate()
            {
                let (astc, bc7) = if mode == 3 {
                    (s, permuted)
                } else {
                    let p = match part[2] >> 1 {
                        0 => usize::from(s == 2),
                        1 => usize::from(s != 0),
                        _ => usize::from(s == 1),
                    };
                    (if part[2] & 1 != 0 { 1 - p } else { p }, s)
                };
                let [lo, hi] = endpoints(u, astc);
                for c in 0..3 {
                    b.low[bc7][c] = (lo[c] * 31 + 127) / 255;
                    b.high[bc7][c] = (hi[c] * 31 + 127) / 255;
                }
            }
        }
        6 | 11 | 13 | 17 => {
            b.mode = 5;
            b.rotation = (u.ccs + 1) & 3;
            let [lo, hi] = endpoints(u, 0);
            for c in 0..4 {
                let dest = if c == u.ccs {
                    3
                } else if c == 3 {
                    u.ccs
                } else {
                    c
                };
                b.low[0][dest] = if dest < 3 {
                    (lo[c] * 127 + 127) / 255
                } else {
                    lo[c]
                };
                b.high[0][dest] = if dest < 3 {
                    (hi[c] * 127 + 127) / 255
                } else {
                    hi[c]
                };
            }
            for i in 0..16 {
                b.selectors[i] = usize::from(u.weights[i * 2]);
                b.alpha_selectors[i] = usize::from(u.weights[i * 2 + 1]);
                if mode == 13 {
                    b.selectors[i] *= 3;
                    b.alpha_selectors[i] *= 3;
                }
            }
        }
        8 => {
            let errs = [0, 1].map(|p| {
                u.solid_color
                    .iter()
                    .map(|&c| BC7_MODE_6_OPTIMAL_ENDPOINTS[c as usize][p][0])
                    .sum::<usize>()
            });
            if errs[0] > 0 && errs[1] > 0 {
                b.mode = 5;
                for c in 0..3 {
                    let ep = BC7_MODE_5_OPTIMAL_ENDPOINTS[u.solid_color[c] as usize];
                    b.low[0][c] = ep[1];
                    b.high[0][c] = ep[2];
                }
                b.low[0][3] = u.solid_color[3] as usize;
                b.high[0][3] = b.low[0][3];
                b.selectors = [1; 16];
            } else {
                b.mode = 6;
                let p = usize::from(errs[1] < errs[0]);
                for c in 0..4 {
                    let ep = BC7_MODE_6_OPTIMAL_ENDPOINTS[u.solid_color[c] as usize][p];
                    b.low[0][c] = ep[1];
                    b.high[0][c] = ep[2];
                }
                b.pbits[0] = [p; 2];
                b.selectors = [5; 16];
            }
        }
        _ => unreachable!(),
    }
    encode_bc7(b)
}
fn encode_bc7(mut b: Bc7) -> [u8; 16] {
    let mode = b.mode;
    let subsets = BC7_NUM_SUBSETS[mode];
    let pat = if subsets == 1 {
        &[0; 16][..]
    } else if subsets == 2 {
        &BC7_PARTITION2[b.partition * 16..b.partition * 16 + 16]
    } else {
        &BC7_PARTITION3[b.partition * 16..b.partition * 16 + 16]
    };
    let mut anchors = [usize::MAX; 3];
    let index_bits = BC7_COLOR_INDEX_BITCOUNT[mode];
    for (s, anchor) in anchors.iter_mut().enumerate().take(subsets) {
        *anchor = if s == 0 {
            0
        } else if subsets == 2 {
            BC7_TABLE_ANCHOR_INDEX_SECOND_SUBSET[b.partition]
        } else if s == 1 {
            BC7_TABLE_ANCHOR_INDEX_THIRD_SUBSET_1[b.partition]
        } else {
            BC7_TABLE_ANCHOR_INDEX_THIRD_SUBSET_2[b.partition]
        };
        if b.selectors[*anchor] & (1 << (index_bits - 1)) != 0 {
            for (i, &p) in pat.iter().enumerate() {
                if p == s {
                    b.selectors[i] = (1 << index_bits) - 1 - b.selectors[i];
                }
            }
            for c in 0..if mode == 5 { 3 } else { 4 } {
                std::mem::swap(&mut b.low[s][c], &mut b.high[s][c]);
            }
            if BC7_MODE_HAS_SHARED_P_BITS[mode] == 0 {
                b.pbits[s].swap(0, 1);
            }
        }
        if mode == 5 && b.alpha_selectors[*anchor] & 2 != 0 {
            for (i, &p) in pat.iter().enumerate() {
                if p == s {
                    b.alpha_selectors[i] = 3 - b.alpha_selectors[i];
                }
            }
            std::mem::swap(&mut b.low[s][3], &mut b.high[s][3]);
        }
    }
    let mut dst = [0; 16];
    let mut pos = 0;
    set(&mut dst, &mut pos, 1 << mode, mode + 1);
    if mode == 5 {
        set(&mut dst, &mut pos, b.rotation, 2);
    }
    set(&mut dst, &mut pos, b.partition, BC7_PARTITION_BITS[mode]);
    for c in 0..if mode >= 4 { 4 } else { 3 } {
        for s in 0..subsets {
            let bits = if c == 3 {
                BC7_ALPHA_PRECISION_TABLE[mode]
            } else {
                BC7_COLOR_PRECISION_TABLE[mode]
            };
            set(&mut dst, &mut pos, b.low[s][c], bits);
            set(&mut dst, &mut pos, b.high[s][c], bits);
        }
    }
    if BC7_MODE_HAS_P_BITS[mode] != 0 {
        for s in 0..subsets {
            set(&mut dst, &mut pos, b.pbits[s][0], 1);
            if BC7_MODE_HAS_SHARED_P_BITS[mode] == 0 {
                set(&mut dst, &mut pos, b.pbits[s][1], 1);
            }
        }
    }
    for i in 0..16 {
        set(
            &mut dst,
            &mut pos,
            b.selectors[i],
            index_bits - usize::from(anchors.contains(&i)),
        );
    }
    if mode == 5 {
        for i in 0..16 {
            set(
                &mut dst,
                &mut pos,
                b.alpha_selectors[i],
                2 - usize::from(i == 0),
            );
        }
    }
    debug_assert_eq!(pos, 128);
    dst
}
impl Ktx2<'_> {
    pub fn transcode_uastc(&self, level: usize, target: UastcTarget) -> Result<Vec<u8>> {
        if self.model != Model::Uastc {
            return Err("not UASTC");
        }
        let data = self.level(level)?;
        let width = (self.width >> level).max(1);
        let height = (self.height >> level).max(1);
        let bx = width.div_ceil(4);
        if data.len() != bx * height.div_ceil(4) * 16 {
            return Err("invalid UASTC level length");
        }
        let mut output = vec![
            0;
            if target == UastcTarget::Rgba8 {
                width * height * 4
            } else {
                data.len()
            }
        ];
        // unpack_uastc fills the caller's block descriptor. Retain its byte
        // arrays across blocks, as the reference does, rather than zeroing
        // and returning a Result<descriptor> on every block.
        let mut u = Unpacked::default();
        match target {
            UastcTarget::Astc => {
                for (bytes, dst) in data
                    .as_chunks::<16>()
                    .0
                    .iter()
                    .zip(output.as_chunks_mut::<16>().0)
                {
                    unpack_uastc(bytes, &mut u, true)?;
                    pack_astc(&u, dst);
                }
            }
            UastcTarget::Bc7 => {
                for (bytes, dst) in data
                    .as_chunks::<16>()
                    .0
                    .iter()
                    .zip(output.as_chunks_mut::<16>().0)
                {
                    unpack_uastc(bytes, &mut u, false)?;
                    dst.copy_from_slice(&bc7(&u));
                }
            }
            UastcTarget::Rgba8 => {
                for (i, bytes) in data.as_chunks::<16>().0.iter().enumerate() {
                    unpack_uastc(bytes, &mut u, false)?;
                    let pixels = rgba(&u);
                    for y in 0..4 {
                        for x in 0..4 {
                            let px = i % bx * 4 + x;
                            let py = i / bx * 4 + y;
                            if px < width && py < height {
                                let at = (py * width + px) * 4;
                                output[at..at + 4].copy_from_slice(&pixels[y * 4 + x]);
                            }
                        }
                    }
                }
            }
        }
        Ok(output)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn arbitrary_blocks_never_panic_and_all_modes_reach_each_target() {
        let mut state = 0x12345678u32;
        let mut modes = [0; 19];
        let mut refused = 0;
        for _ in 0..10_000 {
            let bytes = std::array::from_fn(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as u8
            });
            let mut u = Unpacked::default();
            match unpack_uastc(&bytes, &mut u, false) {
                Ok(()) => {
                    modes[u.mode] += 1;
                    let _ = rgba(&u);
                    let _ = bc7(&u);
                    unpack_uastc(&bytes, &mut u, true).unwrap();
                    pack_astc(&u, &mut [0; 16]);
                }
                Err(_) => refused += 1,
            }
        }
        assert!(modes.iter().all(|&n| n > 0));
        assert!(refused > 0);
        assert!(unpack_uastc(&[0x45; 16], &mut Unpacked::default(), false).is_err());
    }
}
