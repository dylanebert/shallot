//! Wide (4-lane) vector/matrix layer over `FloatW` — a direct port of the `b3Vec3W`/`b3QuatW`/
//! `b3SymMatrix*W` types and their ops in box3d's `contact_solver.c`. The wide contact solver
//! (`contact_wide`) gathers four contacts into these SoA lanes and solves them together.
//!
//! Every op mirrors the C expression tree operand-for-operand (associativity included) so the lanes
//! stay bit-identical to the reference. `mul_add(a, b, c)` is `a + b*c`, never fused.

use crate::simd::FloatW;

/// `min(b, max(-b, a))` — clamp `a` to `[-b, b]` (box3d `b3SymClampW`). Operand order matters: the C
/// computes `b3MaxW(nb, a)` (nb first), which differs from `max(a, nb)` on signed zero / NaN.
#[inline]
pub fn sym_clamp(a: FloatW, b: FloatW) -> FloatW {
    let nb = b.neg();
    let c = nb.max(a);
    b.min(c)
}

#[derive(Clone, Copy)]
pub struct Vec2W {
    pub x: FloatW,
    pub y: FloatW,
}

#[derive(Clone, Copy)]
pub struct Vec3W {
    pub x: FloatW,
    pub y: FloatW,
    pub z: FloatW,
}

#[derive(Clone, Copy)]
pub struct QuatW {
    pub v: Vec3W,
    pub s: FloatW,
}

/// Wide symmetric 2x2 (columns cxx, cxy, cyy).
#[derive(Clone, Copy)]
pub struct SymMatrix2W {
    pub cxx: FloatW,
    pub cxy: FloatW,
    pub cyy: FloatW,
}

/// Wide symmetric 3x3 (columns cxx, cxy, cxz, cyy, cyz, czz).
#[derive(Clone, Copy)]
pub struct SymMatrix3W {
    pub cxx: FloatW,
    pub cxy: FloatW,
    pub cxz: FloatW,
    pub cyy: FloatW,
    pub cyz: FloatW,
    pub czz: FloatW,
}

impl Vec3W {
    #[inline]
    pub fn splat0() -> Self {
        Vec3W { x: FloatW::zero(), y: FloatW::zero(), z: FloatW::zero() }
    }
}

/// `s * a` (b3MulSVW).
#[inline]
pub fn mul_svw(s: FloatW, a: Vec3W) -> Vec3W {
    Vec3W { x: s.mul(a.x), y: s.mul(a.y), z: s.mul(a.z) }
}

/// `a - s*b` (b3MulSubSVW).
#[inline]
pub fn mul_sub_svw(a: Vec3W, s: FloatW, b: Vec3W) -> Vec3W {
    Vec3W {
        x: a.x.sub(s.mul(b.x)),
        y: a.y.sub(s.mul(b.y)),
        z: a.z.sub(s.mul(b.z)),
    }
}

/// `a + s*b` (b3MulAddSVW).
#[inline]
pub fn mul_add_svw(a: Vec3W, s: FloatW, b: Vec3W) -> Vec3W {
    Vec3W {
        x: a.x.add(s.mul(b.x)),
        y: a.y.add(s.mul(b.y)),
        z: a.z.add(s.mul(b.z)),
    }
}

/// `a + b` on Vec2W (b3AddV2W).
#[inline]
pub fn add_v2w(a: Vec2W, b: Vec2W) -> Vec2W {
    Vec2W { x: a.x.add(b.x), y: a.y.add(b.y) }
}

/// `a - b` (b3SubVW).
#[inline]
pub fn sub_vw(a: Vec3W, b: Vec3W) -> Vec3W {
    Vec3W { x: a.x.sub(b.x), y: a.y.sub(b.y), z: a.z.sub(b.z) }
}

/// `a + b` (b3AddVW).
#[inline]
pub fn add_vw(a: Vec3W, b: Vec3W) -> Vec3W {
    Vec3W { x: a.x.add(b.x), y: a.y.add(b.y), z: a.z.add(b.z) }
}

/// `m * a` for symmetric 2x2 (b3MulMV2W).
#[inline]
pub fn mul_mv2w(m: SymMatrix2W, a: Vec2W) -> Vec2W {
    Vec2W {
        x: m.cxx.mul(a.x).add(m.cxy.mul(a.y)),
        y: m.cxy.mul(a.x).add(m.cyy.mul(a.y)),
    }
}

// Row of a symmetric 3x3 times a vec3, matching b3MulMVW associativity: cxx*X + (cxy*Y + cxz*Z).
#[inline]
fn mv_rows(m: SymMatrix3W, b: Vec3W) -> Vec3W {
    Vec3W {
        x: m.cxx.mul(b.x).add(m.cxy.mul(b.y).add(m.cxz.mul(b.z))),
        y: m.cxy.mul(b.x).add(m.cyy.mul(b.y).add(m.cyz.mul(b.z))),
        z: m.cxz.mul(b.x).add(m.cyz.mul(b.y).add(m.czz.mul(b.z))),
    }
}

/// `m * a` for symmetric 3x3 (b3MulMVW).
#[inline]
pub fn mul_mvw(m: SymMatrix3W, a: Vec3W) -> Vec3W {
    mv_rows(m, a)
}

/// `a - m*b` (b3MulSubMVW).
#[inline]
pub fn mul_sub_mvw(a: Vec3W, m: SymMatrix3W, b: Vec3W) -> Vec3W {
    let c = mv_rows(m, b);
    Vec3W { x: a.x.sub(c.x), y: a.y.sub(c.y), z: a.z.sub(c.z) }
}

/// `a + m*b` (b3MulAddMVW).
#[inline]
pub fn mul_add_mvw(a: Vec3W, m: SymMatrix3W, b: Vec3W) -> Vec3W {
    let c = mv_rows(m, b);
    Vec3W { x: a.x.add(c.x), y: a.y.add(c.y), z: a.z.add(c.z) }
}

/// `a . b` (b3DotW): `(aX*bX + aY*bY) + aZ*bZ`, left-associated.
#[inline]
pub fn dot_w(a: Vec3W, b: Vec3W) -> FloatW {
    a.x.mul(b.x).add(a.y.mul(b.y)).add(a.z.mul(b.z))
}

/// `a x b` (b3CrossW).
#[inline]
pub fn cross_w(a: Vec3W, b: Vec3W) -> Vec3W {
    Vec3W {
        x: a.y.mul(b.z).sub(a.z.mul(b.y)),
        y: a.z.mul(b.x).sub(a.x.mul(b.z)),
        z: a.x.mul(b.y).sub(a.y.mul(b.x)),
    }
}

/// Rotate `a` by quaternion `q` (b3RotateVectorW).
#[inline]
pub fn rotate_vector_w(q: QuatW, a: Vec3W) -> Vec3W {
    let t1 = cross_w(q.v, a);
    let t2 = Vec3W {
        x: t1.x.mul_add(q.s, a.x),
        y: t1.y.mul_add(q.s, a.y),
        z: t1.z.mul_add(q.s, a.z),
    };
    let t3 = cross_w(q.v, t2);
    let two = FloatW::splat(2.0);
    Vec3W {
        x: a.x.mul_add(two, t3.x),
        y: a.y.mul_add(two, t3.y),
        z: a.z.mul_add(two, t3.z),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference scalar helpers computed with the same operand order as the wide ops. Per-lane IEEE
    // f32 makes the wide result bit-identical to these, so this pins the associativity choices.
    fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
        (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]
    }

    #[test]
    fn dot_matches_scalar_per_lane() {
        let a = Vec3W {
            x: FloatW::set(1.0, 2.0, -3.0, 0.5),
            y: FloatW::set(4.0, -1.0, 2.0, 1.5),
            z: FloatW::set(0.0, 3.0, 1.0, -2.0),
        };
        let b = Vec3W {
            x: FloatW::set(2.0, 0.5, 1.0, -1.0),
            y: FloatW::set(-1.0, 2.0, 0.0, 3.0),
            z: FloatW::set(3.0, 1.0, -2.0, 0.25),
        };
        let got = dot_w(a, b).to_array();
        let ax = a.x.to_array();
        let ay = a.y.to_array();
        let az = a.z.to_array();
        let bx = b.x.to_array();
        let by = b.y.to_array();
        let bz = b.z.to_array();
        for lane in 0..4 {
            let want = dot3([ax[lane], ay[lane], az[lane]], [bx[lane], by[lane], bz[lane]]);
            assert_eq!(got[lane], want);
        }
    }

    #[test]
    fn cross_matches_scalar_per_lane() {
        let a = Vec3W {
            x: FloatW::set(1.0, 0.0, 2.0, -1.0),
            y: FloatW::set(0.0, 1.0, -1.0, 2.0),
            z: FloatW::set(0.0, 0.0, 3.0, 1.0),
        };
        let b = Vec3W {
            x: FloatW::set(0.0, 0.0, 1.0, 2.0),
            y: FloatW::set(1.0, 0.0, 0.0, -1.0),
            z: FloatW::set(0.0, 1.0, 2.0, 3.0),
        };
        let c = cross_w(a, b);
        let (cx, cy, cz) = (c.x.to_array(), c.y.to_array(), c.z.to_array());
        let (ax, ay, az) = (a.x.to_array(), a.y.to_array(), a.z.to_array());
        let (bx, by, bz) = (b.x.to_array(), b.y.to_array(), b.z.to_array());
        for l in 0..4 {
            assert_eq!(cx[l], ay[l] * bz[l] - az[l] * by[l]);
            assert_eq!(cy[l], az[l] * bx[l] - ax[l] * bz[l]);
            assert_eq!(cz[l], ax[l] * by[l] - ay[l] * bx[l]);
        }
    }

    #[test]
    fn sym_clamp_bounds_each_lane() {
        let a = FloatW::set(-5.0, 0.5, 3.0, -0.25);
        let b = FloatW::splat(1.0);
        assert_eq!(sym_clamp(a, b).to_array(), [-1.0, 0.5, 1.0, -0.25]);
    }
}
