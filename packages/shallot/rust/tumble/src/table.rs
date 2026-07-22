//! Open-addressing pair-set membership — the read half of box3d's `table.c`, ported for the in-kernel
//! broad-phase dedup (3d). Mirrors `src/table.ts`: the symmetric shape-pair key (two u32 halves), the
//! Murmur3 `fmix64` hash truncated to 32 bits, and the linear-probe `contains`. Integer-only, so it
//! replays the TS hash exactly. The kernel only ever *reads* membership (contact create/destroy still
//! write the set from TS via `addKey`/`removeKey`), so only `contains` is ported.

const SHAPE_MASK: u32 = (1 << 22) - 1;
const CHILD_MASK: u32 = (1 << 20) - 1;

const K1_HI: u32 = 0xff51_afd7;
const K1_LO: u32 = 0xed55_8ccd;
const K2_HI: u32 = 0xc4ce_b9fe;
const K2_LO: u32 = 0x1a85_ec53;

/// High 32 bits of the symmetric shape-pair key (`b3ShapePairKey` hi half).
#[inline]
pub fn pair_key_hi(s1: u32, s2: u32) -> u32 {
    let (lo, hi) = if s1 < s2 { (s1, s2) } else { (s2, s1) };
    ((lo & SHAPE_MASK) << 10) | ((hi & SHAPE_MASK) >> 12)
}

/// Low 32 bits of the symmetric shape-pair key: the larger shape index's low 12 bits, then child.
#[inline]
pub fn pair_key_lo(s1: u32, s2: u32, c: u32) -> u32 {
    let hi = if s1 < s2 { s2 } else { s1 };
    ((hi & 0xfff) << 20) | (c & CHILD_MASK)
}

/// Murmur3 `fmix64` over the split key, truncated to the low 32 bits (b3KeyHash). Ported op-for-op from
/// `src/table.ts::keyHash`; `Math.imul` maps to `wrapping_mul` on u32. Each `h *= k` round is a 64x64 →
/// low-64 wrapping multiply over 16-bit limbs (`round`).
pub fn key_hash(k_hi: u32, k_lo: u32) -> u32 {
    let mut h_hi = k_hi;
    let mut h_lo = k_lo ^ (k_hi >> 1);

    // Round 1 (K1).
    let (out_lo, out_hi) = round(h_lo, h_hi, K1_LO, K1_HI);
    h_hi = out_hi;
    h_lo = out_lo ^ (out_hi >> 1);

    // Round 2 (K2).
    let (out_lo, out_hi) = round(h_lo, h_hi, K2_LO, K2_HI);

    out_lo ^ (out_hi >> 1)
}

#[inline]
fn round(h_lo: u32, h_hi: u32, k_lo: u32, k_hi: u32) -> (u32, u32) {
    let a0 = h_lo & 0xffff;
    let a1 = h_lo >> 16;
    let p00 = a0.wrapping_mul(k_lo & 0xffff);
    let p01 = a0.wrapping_mul(k_lo >> 16);
    let p10 = a1.wrapping_mul(k_lo & 0xffff);
    let p11 = a1.wrapping_mul(k_lo >> 16);
    let mid = (p00 >> 16)
        .wrapping_add(p01 & 0xffff)
        .wrapping_add(p10 & 0xffff);
    let out_lo = ((mid & 0xffff) << 16) | (p00 & 0xffff);
    let out_hi = p11
        .wrapping_add(p01 >> 16)
        .wrapping_add(p10 >> 16)
        .wrapping_add(mid >> 16)
        .wrapping_add(h_lo.wrapping_mul(k_hi))
        .wrapping_add(h_hi.wrapping_mul(k_lo));
    (out_lo, out_hi)
}

/// Whether the shape pair `(s1, s2, c)` is in the set (b3ContainsKey). Symmetric in `s1`/`s2`.
/// `cap` is a power of two (probe mask `cap - 1`); `key_hi`/`key_lo`/`hashes` are the set's arrays.
pub fn contains(
    key_hi: &[u32],
    key_lo: &[u32],
    hashes: &[u32],
    cap: usize,
    s1: u32,
    s2: u32,
    c: u32,
) -> bool {
    if cap == 0 {
        return false;
    }
    let k_hi = pair_key_hi(s1, s2);
    let k_lo = pair_key_lo(s1, s2, c);
    let hash = key_hash(k_hi, k_lo);
    let mask = (cap - 1) as u32;
    let mut idx = (hash & mask) as usize;
    while hashes[idx] != 0 && (key_hi[idx] != k_hi || key_lo[idx] != k_lo) {
        idx = ((idx as u32 + 1) & mask) as usize;
    }
    key_hi[idx] == k_hi && key_lo[idx] == k_lo
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn key_hash_matches_ts() {
        // (s1, s2, c, expected_hash) from src/table.ts (TS reference).
        let cases = [
            (3u32, 7u32, 0u32, 3489682763u32),
            (0, 10, 0, 2285512313),
            (2, 5, 0, 1184001114),
            (100, 3, 0, 179916708),
            (0, 1, 0, 1985457684),
        ];
        for (a, b, c, want) in cases {
            let hi = pair_key_hi(a, b);
            let lo = pair_key_lo(a, b, c);
            assert_eq!(key_hash(hi, lo), want, "keyHash({a},{b},{c})");
        }
    }
}
