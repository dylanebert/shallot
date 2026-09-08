//! 4-wide f32 lane type, a direct port of box3d's `b3FloatW` (contact_solver.c).
//!
//! Two implementations, cfg-selected: wasm `simd128` intrinsics for the shipping build, and a
//! scalar `[f32; 4]` fallback for native `cargo test`. The fallback is bit-identical to the wasm
//! path because per-lane IEEE f32 is deterministic — the same reason box3d's SIMD and
//! `DISABLE_SIMD` builds emit identical fixtures (verified: 52/52 default-config scenes match).
//!
//! Semantics mirror the SSE2 branch op-for-op (the reference build on x86 is SSE2), which is the
//! contract the fixtures encode:
//!   - `mul_add(a, b, c)` = `a + b*c`, NEVER fused (box3d disables FMA to match its scalar path;
//!     Rust does not auto-contract `a + b*c`).
//!   - `min`/`max` are `(a<b)?a:b` / `(a>b)?a:b` (SSE `_mm_min_ps`/`_mm_max_ps`), NOT wasm's native
//!     `f32x4_min`/`max`, which differ on NaN and signed zero. Implemented via compare→bitselect so
//!     the wasm lanes match SSE bit-for-bit.

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

#[derive(Clone, Copy)]
pub struct FloatW(
    #[cfg(target_arch = "wasm32")] v128,
    #[cfg(not(target_arch = "wasm32"))] [f32; 4],
);

#[cfg(target_arch = "wasm32")]
impl FloatW {
    #[inline]
    pub fn zero() -> Self {
        FloatW(f32x4_splat(0.0))
    }
    #[inline]
    pub fn splat(s: f32) -> Self {
        FloatW(f32x4_splat(s))
    }
    #[inline]
    pub fn set(a: f32, b: f32, c: f32, d: f32) -> Self {
        FloatW(f32x4(a, b, c, d))
    }
    #[inline]
    pub fn neg(self) -> Self {
        FloatW(f32x4_neg(self.0))
    }
    #[inline]
    pub fn add(self, o: Self) -> Self {
        FloatW(f32x4_add(self.0, o.0))
    }
    #[inline]
    pub fn sub(self, o: Self) -> Self {
        FloatW(f32x4_sub(self.0, o.0))
    }
    #[inline]
    pub fn mul(self, o: Self) -> Self {
        FloatW(f32x4_mul(self.0, o.0))
    }
    #[inline]
    pub fn div(self, o: Self) -> Self {
        FloatW(f32x4_div(self.0, o.0))
    }
    #[inline]
    pub fn sqrt(self) -> Self {
        FloatW(f32x4_sqrt(self.0))
    }
    /// `a + b*c`, non-fused (matches box3d's `b3MulAddW`).
    #[inline]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        FloatW(f32x4_add(self.0, f32x4_mul(b.0, c.0)))
    }
    #[inline]
    pub fn min(self, o: Self) -> Self {
        FloatW(v128_bitselect(self.0, o.0, f32x4_lt(self.0, o.0)))
    }
    #[inline]
    pub fn max(self, o: Self) -> Self {
        FloatW(v128_bitselect(self.0, o.0, f32x4_gt(self.0, o.0)))
    }
    #[inline]
    pub fn or(self, o: Self) -> Self {
        FloatW(v128_or(self.0, o.0))
    }
    /// Per-lane `a > b ? all-ones : 0` mask.
    #[inline]
    pub fn greater_than(self, o: Self) -> Self {
        FloatW(f32x4_gt(self.0, o.0))
    }
    /// Per-lane `a == b ? all-ones : 0` mask.
    #[inline]
    pub fn equals(self, o: Self) -> Self {
        FloatW(f32x4_eq(self.0, o.0))
    }
    #[inline]
    pub fn all_zero(self) -> bool {
        i32x4_all_true(f32x4_eq(self.0, f32x4_splat(0.0)))
    }
    /// Component-wise `mask ? b : a` (matches box3d's `b3BlendW(a, b, mask)`).
    #[inline]
    pub fn blend(a: Self, b: Self, mask: Self) -> Self {
        FloatW(v128_bitselect(b.0, a.0, mask.0))
    }
    #[inline]
    pub fn to_array(self) -> [f32; 4] {
        [
            f32x4_extract_lane::<0>(self.0),
            f32x4_extract_lane::<1>(self.0),
            f32x4_extract_lane::<2>(self.0),
            f32x4_extract_lane::<3>(self.0),
        ]
    }
    /// Wrap a raw lane vector (4c's record-transpose gather builds lanes as `v128` directly).
    #[inline]
    pub fn from_v128(v: v128) -> Self {
        FloatW(v)
    }
    /// The raw lane vector (4c's scatter transposes it back into the record).
    #[inline]
    pub fn v128(self) -> v128 {
        self.0
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl FloatW {
    #[inline]
    pub fn zero() -> Self {
        FloatW([0.0; 4])
    }
    #[inline]
    pub fn splat(s: f32) -> Self {
        FloatW([s; 4])
    }
    #[inline]
    pub fn set(a: f32, b: f32, c: f32, d: f32) -> Self {
        FloatW([a, b, c, d])
    }
    #[inline]
    pub fn neg(self) -> Self {
        self.map(|x| -x)
    }
    #[inline]
    pub fn add(self, o: Self) -> Self {
        self.zip(o, |a, b| a + b)
    }
    #[inline]
    pub fn sub(self, o: Self) -> Self {
        self.zip(o, |a, b| a - b)
    }
    #[inline]
    pub fn mul(self, o: Self) -> Self {
        self.zip(o, |a, b| a * b)
    }
    #[inline]
    pub fn div(self, o: Self) -> Self {
        self.zip(o, |a, b| a / b)
    }
    #[inline]
    pub fn sqrt(self) -> Self {
        self.map(|x| x.sqrt())
    }
    #[inline]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        FloatW([
            self.0[0] + b.0[0] * c.0[0],
            self.0[1] + b.0[1] * c.0[1],
            self.0[2] + b.0[2] * c.0[2],
            self.0[3] + b.0[3] * c.0[3],
        ])
    }
    #[inline]
    pub fn min(self, o: Self) -> Self {
        self.zip(o, |a, b| if a < b { a } else { b })
    }
    #[inline]
    pub fn max(self, o: Self) -> Self {
        self.zip(o, |a, b| if a > b { a } else { b })
    }
    #[inline]
    pub fn or(self, o: Self) -> Self {
        self.bits(o, |a, b| a | b)
    }
    #[inline]
    pub fn greater_than(self, o: Self) -> Self {
        self.mask(o, |a, b| a > b)
    }
    #[inline]
    pub fn equals(self, o: Self) -> Self {
        self.mask(o, |a, b| a == b)
    }
    #[inline]
    pub fn all_zero(self) -> bool {
        self.0.iter().all(|&x| x == 0.0)
    }
    #[inline]
    pub fn blend(a: Self, b: Self, mask: Self) -> Self {
        // (mask & b) | (~mask & a)
        FloatW(core::array::from_fn(|i| {
            let m = mask.0[i].to_bits();
            f32::from_bits((m & b.0[i].to_bits()) | (!m & a.0[i].to_bits()))
        }))
    }
    #[inline]
    pub fn to_array(self) -> [f32; 4] {
        self.0
    }

    #[inline]
    fn map(self, f: impl Fn(f32) -> f32) -> Self {
        FloatW(core::array::from_fn(|i| f(self.0[i])))
    }
    #[inline]
    fn zip(self, o: Self, f: impl Fn(f32, f32) -> f32) -> Self {
        FloatW(core::array::from_fn(|i| f(self.0[i], o.0[i])))
    }
    #[inline]
    fn mask(self, o: Self, f: impl Fn(f32, f32) -> bool) -> Self {
        FloatW(core::array::from_fn(|i| {
            if f(self.0[i], o.0[i]) {
                f32::from_bits(0xFFFF_FFFF)
            } else {
                0.0
            }
        }))
    }
    #[inline]
    fn bits(self, o: Self, f: impl Fn(u32, u32) -> u32) -> Self {
        FloatW(core::array::from_fn(|i| {
            f32::from_bits(f(self.0[i].to_bits(), o.0[i].to_bits()))
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::FloatW;

    #[test]
    fn arithmetic_lanes() {
        let a = FloatW::set(1.0, 2.0, 3.0, 4.0);
        let b = FloatW::splat(2.0);
        assert_eq!(a.add(b).to_array(), [3.0, 4.0, 5.0, 6.0]);
        assert_eq!(a.mul(b).to_array(), [2.0, 4.0, 6.0, 8.0]);
        // mul_add is a + b*c, non-fused
        assert_eq!(
            FloatW::splat(1.0).mul_add(a, b).to_array(),
            [3.0, 5.0, 7.0, 9.0]
        );
        assert_eq!(
            FloatW::set(4.0, 9.0, 16.0, 25.0).sqrt().to_array(),
            [2.0, 3.0, 4.0, 5.0]
        );
    }

    #[test]
    fn min_max_pick_ssemantics() {
        let a = FloatW::set(1.0, 5.0, 3.0, 8.0);
        let b = FloatW::set(4.0, 2.0, 3.0, 6.0);
        assert_eq!(a.min(b).to_array(), [1.0, 2.0, 3.0, 6.0]);
        assert_eq!(a.max(b).to_array(), [4.0, 5.0, 3.0, 8.0]);
    }

    #[test]
    fn blend_selects_by_mask() {
        let a = FloatW::splat(10.0);
        let b = FloatW::splat(20.0);
        let mask = FloatW::set(1.0, 5.0, 3.0, 8.0).greater_than(FloatW::splat(2.5));
        // lanes 1,2,3 > 2.5 -> pick b; lane 0 -> pick a
        assert_eq!(
            FloatW::blend(a, b, mask).to_array(),
            [10.0, 20.0, 20.0, 20.0]
        );
    }

    #[test]
    fn all_zero_detects_zero_lanes() {
        assert!(FloatW::zero().all_zero());
        assert!(!FloatW::set(0.0, 0.0, 0.0, 1.0).all_zero());
    }
}
