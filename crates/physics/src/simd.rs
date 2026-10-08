//! 4-wide f32 lane type, a direct port of box3d's `b3FloatW` (contact_solver.c).
//!
//! Two implementations, cfg-selected: wasm `simd128` intrinsics for the shipping build, and a
//! scalar `[f32; 4]` fallback for native `cargo test`. The fallback is bit-identical to the wasm
//! path because per-lane IEEE f32 is deterministic — the same reason box3d's SIMD and
//! `DISABLE_SIMD` builds emit identical fixtures (verified: 52/52 default-config scenes match).
//!
//! `mul_add(a, b, c)` = `a + b*c`, non-fused. `min`/`max` use ordered comparisons and select
//! the second operand on equality or unordered inputs, via compare→bitselect.

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

#[derive(Clone, Copy)]
pub struct FloatW(
    #[cfg(target_arch = "wasm32")] v128,
    #[cfg(not(target_arch = "wasm32"))] [f32; 4],
);

impl FloatW {
    #[inline]
    fn shuffle<const X: usize, const Y: usize, const Z: usize, const W: usize>(self) -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            Self(i32x4_shuffle::<X, Y, Z, W>(self.0, self.0))
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self::set(self.0[X], self.0[Y], self.0[Z], self.0[W])
        }
    }

    #[inline]
    fn abs(self) -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            Self(f32x4_abs(self.0))
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.map(f32::abs)
        }
    }

    #[inline]
    fn any3(self) -> bool {
        #[cfg(target_arch = "wasm32")]
        {
            i32x4_bitmask(self.0) & 7 != 0
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.0[..3].iter().any(|v| v.to_bits() != 0)
        }
    }

    #[inline]
    fn cross3(self, b: Self) -> Self {
        self.shuffle::<1, 2, 0, 3>()
            .mul(b.shuffle::<2, 0, 1, 3>())
            .sub(self.shuffle::<2, 0, 1, 3>().mul(b.shuffle::<1, 2, 0, 3>()))
    }

    #[inline]
    fn modified_cross3(self, b: Self) -> Self {
        self.shuffle::<1, 2, 0, 3>()
            .mul(b.shuffle::<2, 0, 1, 3>())
            .add(self.shuffle::<2, 0, 1, 3>().mul(b.shuffle::<1, 2, 0, 3>()))
    }

    #[inline]
    fn dot3(self, b: Self) -> Self {
        let p = self.mul(b);
        p.shuffle::<0, 0, 0, 0>()
            .add(p.shuffle::<1, 1, 1, 1>())
            .add(p.shuffle::<2, 2, 2, 2>())
    }

    /// Recover the index embedded in the minimum lane (b3MinIndexW).
    #[inline]
    pub fn min_index(self, bit_count: u32) -> usize {
        #[cfg(target_arch = "wasm32")]
        let a = self.min(FloatW(i32x4_shuffle::<1, 0, 3, 2>(self.0, self.0)));
        #[cfg(not(target_arch = "wasm32"))]
        let a = self.min(Self::set(self.0[1], self.0[0], self.0[3], self.0[2]));
        #[cfg(target_arch = "wasm32")]
        let a = a.min(FloatW(i32x4_shuffle::<2, 3, 0, 1>(a.0, a.0)));
        #[cfg(not(target_arch = "wasm32"))]
        let a = a.min(Self::set(a.0[2], a.0[3], a.0[0], a.0[1]));
        (a.to_array()[0].to_bits() & ((1 << bit_count) - 1)) as usize
    }
}

#[inline]
fn vector(v: crate::math::Vec3) -> FloatW {
    FloatW::set(v.x, v.y, v.z, 0.0)
}

// simd.c:95: SAT uses xyz in one SIMD vector, not a scalar axis loop.
pub(crate) fn bounds_triangle_overlap(
    center: crate::math::Vec3,
    extent: crate::math::Vec3,
    vertices: [crate::math::Vec3; 3],
) -> bool {
    let center = vector(center);
    let extent = vector(extent);
    let [v1, v2, v3] = vertices.map(|v| vector(v).sub(center));
    let zero = FloatW::zero();
    let tri_min = v1.min(v2.min(v3));
    let tri_max = v1.max(v2.max(v3));
    if tri_min
        .sub(extent)
        .max(tri_max.add(extent).neg())
        .greater_than(zero)
        .any3()
    {
        return false;
    }
    let e1 = v2.sub(v1);
    let e2 = v3.sub(v2);
    let e3 = v1.sub(v3);
    let normal = e1.cross3(e2);
    if normal
        .dot3(v1)
        .abs()
        .sub(normal.abs().dot3(extent))
        .greater_than(zero)
        .any3()
    {
        return false;
    }
    let separation = |edge: FloatW, sum: FloatW, other: FloatW| {
        edge.cross3(sum)
            .abs()
            .sub(edge.cross3(other).abs())
            .sub(FloatW::splat(2.0).mul(edge.abs().modified_cross3(extent)))
            .greater_than(zero)
            .any3()
    };
    !separation(e1, v1.add(v3), e3)
        && !separation(e2, v1.add(v2), e1)
        && !separation(e3, v2.add(v3), e2)
}

// simd.c:157: transpose the three edge normals to evaluate the three volumes together.
pub(crate) fn intersect_ray_triangle(
    start: crate::math::Vec3,
    delta: crate::math::Vec3,
    vertices: [crate::math::Vec3; 3],
) -> f32 {
    let start = vector(start);
    let delta = vector(delta);
    let [a, b, c] = vertices.map(vector);
    let half = FloatW::splat(0.5);
    let n1 = c.sub(b).cross3(half.mul(b.add(c)).sub(start));
    let n2 = a.sub(c).cross3(half.mul(c.add(a)).sub(start));
    let n3 = b.sub(a).cross3(half.mul(a.add(b)).sub(start));
    #[cfg(target_arch = "wasm32")]
    let (x, y, z) = {
        let xy = i32x4_shuffle::<0, 1, 4, 5>(n1.0, n2.0);
        let zz = i32x4_shuffle::<2, 3, 6, 7>(n1.0, n2.0);
        (
            FloatW(i32x4_shuffle::<0, 2, 4, 4>(xy, n3.0)),
            FloatW(i32x4_shuffle::<1, 3, 5, 5>(xy, n3.0)),
            FloatW(i32x4_shuffle::<0, 2, 6, 6>(zz, n3.0)),
        )
    };
    #[cfg(not(target_arch = "wasm32"))]
    let (x, y, z) = (
        FloatW::set(n1.0[0], n2.0[0], n3.0[0], 0.0),
        FloatW::set(n1.0[1], n2.0[1], n3.0[1], 0.0),
        FloatW::set(n1.0[2], n2.0[2], n3.0[2], 0.0),
    );
    let volumes = x
        .mul(delta.shuffle::<0, 0, 0, 0>())
        .add(y.mul(delta.shuffle::<1, 1, 1, 1>()))
        .add(z.mul(delta.shuffle::<2, 2, 2, 2>()));
    if volumes.less_than(FloatW::zero()).any3() {
        return 1.0;
    }
    let normal = b.sub(a).cross3(c.sub(a));
    let denominator = normal.dot3(delta);
    if FloatW::zero()
        .less_than(denominator)
        .or(denominator.equals(FloatW::zero()))
        .any3()
    {
        return 1.0;
    }
    let lambda = normal.dot3(a.sub(start)).div(denominator);
    if lambda
        .less_than(FloatW::zero())
        .or(lambda.equals(FloatW::zero()))
        .any3()
    {
        return 1.0;
    }
    lambda.min(FloatW::splat(1.0)).to_array()[0]
}

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
    #[inline]
    pub fn load(values: &[f32]) -> Self {
        assert!(values.len() >= 4);
        unsafe { FloatW(v128_load(values.as_ptr() as *const v128)) }
    }
    #[inline]
    pub fn and(self, o: Self) -> Self {
        FloatW(v128_and(self.0, o.0))
    }
    #[inline]
    pub fn less_than(self, o: Self) -> Self {
        FloatW(f32x4_lt(self.0, o.0))
    }
    #[inline]
    pub fn any_true(self) -> bool {
        v128_any_true(self.0)
    }
    #[inline]
    pub fn embed_index(self, index: usize) -> Self {
        let mask = i32x4_splat(!0x7f);
        let indices = i32x4(
            index as i32,
            (index + 1) as i32,
            (index + 2) as i32,
            (index + 3) as i32,
        );
        FloatW(v128_or(v128_and(self.0, mask), indices))
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
    pub fn load(values: &[f32]) -> Self {
        FloatW(values[..4].try_into().unwrap())
    }
    #[inline]
    pub fn and(self, o: Self) -> Self {
        self.bits(o, |a, b| a & b)
    }
    #[inline]
    pub fn less_than(self, o: Self) -> Self {
        self.mask(o, |a, b| a < b)
    }
    #[inline]
    pub fn any_true(self) -> bool {
        self.0.iter().any(|x| x.to_bits() != 0)
    }
    #[inline]
    pub fn embed_index(self, index: usize) -> Self {
        FloatW(core::array::from_fn(|lane| {
            f32::from_bits((self.0[lane].to_bits() & !0x7f) | (index + lane) as u32)
        }))
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
