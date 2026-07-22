//! Scalar vector math for the serial contact solver, ported op-for-op from box3d's
//! `math_functions.h/.c` (Erin Catto, MIT) via the tumble.js TS port (`src/math.ts`).
//!
//! Rust `f32` is native IEEE-754 with no FMA contraction (box3d builds `-ffp-contract=off`, and
//! Rust never fuses `a + b * c`), so each C/TS `f32(...)`-wrapped op maps to one Rust `f32` op with
//! the same operand order. The double-rounding theorem makes this bit-identical to the TS
//! `Math.fround`-per-op port: `f32(a ⊙ b)` (round an f64 result) equals the native f32 `⊙` for
//! `+ − × ÷ √` on f32 operands, so no explicit rounding is needed.
//!
//! Two disciplines are load-bearing and mirrored from the port:
//!   - min/max are explicit comparisons (`a < b ? a : b`), never `f32::min`/`max`, which diverge
//!     from box3d's scalar `b3MinFloat`/`b3MaxFloat` on NaN and signed zero.
//!   - `remainderf` (inside the trig unwind) keeps its intermediate in f64 on purpose — the true
//!     remainder is f32-representable and the f64 subtraction is exact, reproducing libm
//!     `remainderf` bit for bit. Every other op stays in f32.

// biome-ignore-parity: the file mirrors src/math.ts; comments there explain the arithmetic.

pub const PI: f32 = 3.141_592_653_59;

pub const FLT_EPSILON: f32 = 1.192_092_895_507_812_5e-7; // 2^-23
pub const FLT_MIN: f32 = 1.175_494_350_822_287_5e-38; // 2^-126, smallest normal
pub const FLT_MAX: f32 = f32::MAX;

const TWO_PI: f32 = 2.0 * PI;

// --- scalar helpers -------------------------------------------------------------------------

#[inline]
pub fn absf(a: f32) -> f32 {
    if a < 0.0 {
        -a
    } else {
        a
    }
}

#[inline]
pub fn minf(a: f32, b: f32) -> f32 {
    if a < b {
        a
    } else {
        b
    }
}

#[inline]
pub fn maxf(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else {
        b
    }
}

/// Clamp to `[lo, hi]` (b3ClampFloat: `a < lo ? lo : (hi < a ? hi : a)`).
#[inline]
pub fn clampf(a: f32, lo: f32, hi: f32) -> f32 {
    if a < lo {
        lo
    } else if hi < a {
        hi
    } else {
        a
    }
}

// --- transcendentals ------------------------------------------------------------------------

// Round to nearest integer, ties to even, in f64 (matches the quotient rounding libm remainderf uses).
fn rint_even(x: f64) -> f64 {
    let fl = x.floor();
    let diff = x - fl;
    if diff < 0.5 {
        return fl;
    }
    if diff > 0.5 {
        return fl + 1.0;
    }
    if fl % 2.0 == 0.0 {
        fl
    } else {
        fl + 1.0
    }
}

// IEEE-754 remainder. The intermediate stays in f64 on purpose: for finite f32 inputs the true
// remainder is f32-representable and the f64 subtraction is exact, so this reproduces libm
// remainderf bit for bit over the sim's range (see the module note).
fn remainderf(x: f32, y: f32) -> f32 {
    let xf = x as f64;
    let yf = y as f64;
    let n = rint_even(xf / yf);
    let r = (xf - n * yf) as f32;
    if r == 0.0 {
        // A zero remainder takes the sign of the dividend.
        if x.is_sign_negative() {
            -0.0
        } else {
            0.0
        }
    } else {
        r
    }
}

/// Convert any angle into the range [-pi, pi].
pub fn unwind_angle(radians: f32) -> f32 {
    remainderf(radians, TWO_PI)
}

const ATAN_P0: f32 = 0.024_840_285;
const ATAN_P1: f32 = 0.186_814_18;
const ATAN_P2: f32 = -0.094_097_948;
const ATAN_P3: f32 = -0.332_130_72;

/// Approximate arctangent in [-pi, pi], deterministic across platforms (b3Atan2).
pub fn atan2(y: f32, x: f32) -> f32 {
    if x == 0.0 && y == 0.0 {
        return 0.0;
    }

    let ax = absf(x);
    let ay = absf(y);
    let mx = maxf(ay, ax);
    let mn = minf(ay, ax);
    let a = mn / mx;

    let s = a * a;
    let c = s * a;
    let q = s * s;
    let mut r = ATAN_P0 * q + ATAN_P1;
    let t = ATAN_P2 * q + ATAN_P3;
    r = r * s + t;
    r = r * c + a;

    if ay > ax {
        r = 1.570_796_37 - r;
    }
    if x < 0.0 {
        r = 3.141_592_74 - r;
    }
    if y < 0.0 {
        r = -r;
    }

    r
}

/// Cosine/sine of an angle, Bhāskara I approximation normalized to unit magnitude (b3ComputeCosSin).
#[derive(Clone, Copy, Debug)]
pub struct CosSin {
    pub cosine: f32,
    pub sine: f32,
}

pub fn compute_cos_sin(radians: f32) -> CosSin {
    let x = unwind_angle(radians);
    let pi2 = PI * PI;

    // cosine needs angle in [-pi/2, pi/2]
    let c = if x < -0.5 * PI {
        let y = x + PI;
        let y2 = y * y;
        -((pi2 - 4.0 * y2) / (pi2 + y2))
    } else if x > 0.5 * PI {
        let y = x - PI;
        let y2 = y * y;
        -((pi2 - 4.0 * y2) / (pi2 + y2))
    } else {
        let y2 = x * x;
        (pi2 - 4.0 * y2) / (pi2 + y2)
    };

    // sine needs angle in [0, pi]
    let s = if x < 0.0 {
        let y = x + PI;
        let pi_minus_y = PI - y;
        let num = 16.0 * y * pi_minus_y;
        let den = 5.0 * pi2 - 4.0 * y * pi_minus_y;
        -(num / den)
    } else {
        let pi_minus_x = PI - x;
        let num = 16.0 * x * pi_minus_x;
        let den = 5.0 * pi2 - 4.0 * x * pi_minus_x;
        num / den
    };

    let mag = (s * s + c * c).sqrt();
    let inv_mag = if mag > 0.0 { 1.0 / mag } else { 0.0 };
    CosSin {
        cosine: c * inv_mag,
        sine: s * inv_mag,
    }
}

// --- vec3 -----------------------------------------------------------------------------------

// repr(C) fixes the field layout so a hull's point pool in wasm linear memory can be reinterpreted as
// `&[Vec3]` (kernel/src/hull.rs, 3c.2 geometry columns).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 { x, y, z }
    }

    #[inline]
    pub fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
    #[inline]
    pub fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
    #[inline]
    pub fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
    /// s * self
    #[inline]
    pub fn scale(self, s: f32) -> Vec3 {
        Vec3::new(s * self.x, s * self.y, s * self.z)
    }
    /// self + s * b
    #[inline]
    pub fn mul_add(self, s: f32, b: Vec3) -> Vec3 {
        Vec3::new(self.x + s * b.x, self.y + s * b.y, self.z + s * b.z)
    }
    /// self - s * b
    #[inline]
    pub fn mul_sub(self, s: f32, b: Vec3) -> Vec3 {
        Vec3::new(self.x - s * b.x, self.y - s * b.y, self.z - s * b.z)
    }
    #[inline]
    pub fn dot(self, o: Vec3) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    #[inline]
    pub fn cross(self, o: Vec3) -> Vec3 {
        Vec3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    /// Component-wise absolute value (b3Abs, via the `b3AbsFloat` ternary).
    #[inline]
    pub fn abs(self) -> Vec3 {
        Vec3::new(absf(self.x), absf(self.y), absf(self.z))
    }
    /// Cross with all-plus signs (b3ModifiedCross): the sweep/rotation-arc bound, not a true cross.
    #[inline]
    pub fn modified_cross(self, o: Vec3) -> Vec3 {
        Vec3::new(
            self.y * o.z + self.z * o.y,
            self.z * o.x + self.x * o.z,
            self.x * o.y + self.y * o.x,
        )
    }
    #[inline]
    pub fn length_sq(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }
    #[inline]
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }
    pub fn normalize(self) -> Vec3 {
        let length_sq = self.length_sq();
        if length_sq > 1000.0 * FLT_MIN {
            let s = 1.0 / length_sq.sqrt();
            self.scale(s)
        } else {
            Vec3::ZERO
        }
    }
    pub fn lerp(self, b: Vec3, alpha: f32) -> Vec3 {
        let t = 1.0 - alpha;
        Vec3::new(
            t * self.x + alpha * b.x,
            t * self.y + alpha * b.y,
            t * self.z + alpha * b.z,
        )
    }
    /// Distance from self to o (b3Distance: `length(o - self)`).
    #[inline]
    pub fn distance(self, o: Vec3) -> f32 {
        o.sub(self).length()
    }
    /// Unit vector perpendicular to self (b3Perp: axis-selecting, always normalized).
    #[inline]
    pub fn perp(self) -> Vec3 {
        let p = if self.x < -0.5 || 0.5 < self.x {
            Vec3::new(self.y, -self.x, 0.0)
        } else {
            Vec3::new(0.0, self.z, -self.y)
        };
        p.normalize()
    }
    /// Is this vector (approximately) unit length? (b3IsNormalized).
    #[inline]
    pub fn is_normalized(self) -> bool {
        let aa = self.dot(self);
        absf(1.0 - aa) < 100.0 * FLT_EPSILON
    }
}

/// dot(a, cross(b, c)) — the scalar triple product (b3ScalarTripleProduct).
pub fn scalar_triple_product(a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let d = Vec3::new(
        b.y * c.z - b.z * c.y,
        b.z * c.x - b.x * c.z,
        b.x * c.y - b.y * c.x,
    );
    a.x * d.x + a.y * d.y + a.z * d.z
}

/// s * a + t * b, componentwise (b3Blend2).
#[inline]
pub fn blend2(s: f32, a: Vec3, t: f32, b: Vec3) -> Vec3 {
    Vec3::new(s * a.x + t * b.x, s * a.y + t * b.y, s * a.z + t * b.z)
}

/// s * a + t * b + u * c, componentwise (b3Blend3).
#[inline]
pub fn blend3(s: f32, a: Vec3, t: f32, b: Vec3, u: f32, c: Vec3) -> Vec3 {
    Vec3::new(
        s * a.x + t * b.x + u * c.x,
        s * a.y + t * b.y + u * c.y,
        s * a.z + t * b.z + u * c.z,
    )
}

/// Unit vector perpendicular to a unit vector, well-conditioned coefficients (b3ArbitraryPerp).
pub fn arbitrary_perp(v: Vec3) -> Vec3 {
    let a: f32 = 0.67;
    let b: f32 = -0.42;
    let p = if v.x < -0.5 || 0.5 < v.x {
        Vec3::new(a * v.y + b * v.z, -a * v.x, -b * v.x)
    } else if v.y < -0.5 || 0.5 < v.y {
        Vec3::new(a * v.y, -a * v.x + b * v.z, -b * v.y)
    } else {
        Vec3::new(a * v.z, b * v.z, -a * v.x - b * v.y)
    };
    p.normalize()
}

// --- vec2 -----------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    #[inline]
    pub fn new(x: f32, y: f32) -> Vec2 {
        Vec2 { x, y }
    }
    #[inline]
    pub fn add(self, o: Vec2) -> Vec2 {
        Vec2::new(self.x + o.x, self.y + o.y)
    }
    #[inline]
    pub fn sub(self, o: Vec2) -> Vec2 {
        Vec2::new(self.x - o.x, self.y - o.y)
    }
    /// s * self (b3MulSV2).
    #[inline]
    pub fn scale(self, s: f32) -> Vec2 {
        Vec2::new(s * self.x, s * self.y)
    }
    #[inline]
    pub fn dot(self, o: Vec2) -> f32 {
        self.x * o.x + self.y * o.y
    }
}

// --- quat -----------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub v: Vec3,
    pub s: f32,
}

impl Quat {
    pub const IDENTITY: Quat = Quat {
        v: Vec3::ZERO,
        s: 1.0,
    };

    #[inline]
    pub fn dot(self, o: Quat) -> f32 {
        self.v.x * o.v.x + self.v.y * o.v.y + self.v.z * o.v.z + self.s * o.s
    }

    /// self * o
    pub fn mul(self, o: Quat) -> Quat {
        let t1 = self.v.cross(o.v);
        let t2 = t1.mul_add(self.s, o.v);
        let t3 = t2.mul_add(o.s, self.v);
        Quat {
            v: t3,
            s: self.s * o.s - self.v.dot(o.v),
        }
    }

    /// inv(self) * o
    pub fn inv_mul(self, o: Quat) -> Quat {
        let t1 = o.v.cross(self.v);
        let t2 = t1.mul_add(self.s, o.v);
        let t3 = t2.mul_sub(o.s, self.v);
        Quat {
            v: t3,
            s: self.s * o.s + self.v.dot(o.v),
        }
    }

    /// Conjugate: negate the vector part, keep the scalar (b3Conjugate). Inverse for a unit quat.
    #[inline]
    pub fn conjugate(self) -> Quat {
        Quat {
            v: self.v.neg(),
            s: self.s,
        }
    }

    #[inline]
    pub fn negate(self) -> Quat {
        Quat {
            v: self.v.neg(),
            s: -self.s,
        }
    }

    pub fn normalize(self) -> Quat {
        let length_sq = self.dot(self);
        if length_sq > 1000.0 * FLT_MIN {
            let s = 1.0 / length_sq.sqrt();
            Quat {
                v: self.v.scale(s),
                s: s * self.s,
            }
        } else {
            Quat::IDENTITY
        }
    }

    /// Rotate v by self: v + 2 * cross(q.v, cross(q.v, v) + q.s * v).
    pub fn rotate(self, v: Vec3) -> Vec3 {
        let t1 = self.v.cross(v);
        let t2 = t1.mul_add(self.s, v);
        let t3 = self.v.cross(t2);
        v.mul_add(2.0, t3)
    }

    /// Inverse rotate v by self: v + 2 * cross(q.v, cross(q.v, v) - q.s * v).
    pub fn inv_rotate(self, v: Vec3) -> Vec3 {
        let t1 = self.v.cross(v);
        let t2 = t1.mul_sub(self.s, v);
        let t3 = self.v.cross(t2);
        v.mul_add(2.0, t3)
    }

    /// Advance `self` by a small-angle rotation vector, then renormalize (b3IntegrateRotation).
    pub fn integrate_rotation(self, delta: Vec3) -> Quat {
        let qd = Quat {
            v: delta.scale(0.5),
            s: 0.0,
        }
        .mul(self);
        let q2 = Quat {
            v: self.v.add(qd.v),
            s: qd.s + self.s,
        };
        q2.normalize()
    }

    pub fn from_axis_angle(axis: Vec3, radians: f32) -> Quat {
        let cs = compute_cos_sin(0.5 * radians);
        Quat {
            v: Vec3::new(cs.sine * axis.x, cs.sine * axis.y, cs.sine * axis.z),
            s: cs.cosine,
        }
    }

    /// Twist angle around the z-axis (revolute angle / twist limit).
    pub fn get_twist_angle(self) -> f32 {
        let twist = if self.s < 0.0 {
            atan2(-self.v.z, -self.s)
        } else {
            atan2(self.v.z, self.s)
        };
        twist * 2.0
    }

    /// Swing angle (cone limit).
    pub fn get_swing_angle(self) -> f32 {
        let x = (self.v.z * self.v.z + self.s * self.s).sqrt();
        let y = (self.v.x * self.v.x + self.v.y * self.v.y).sqrt();
        2.0 * atan2(y, x)
    }

    /// Pseudo angular velocity taking `self` toward `target`: `2 * (target - self) * conj(self)`
    /// (b3DeltaQuatToRotation). Negates `self` first if the two point opposite hemispheres.
    pub fn delta_to_rotation(self, target: Quat) -> Vec3 {
        let s = if self.dot(target) < 0.0 {
            self.negate()
        } else {
            self
        };
        let diff = Quat {
            v: target.v.sub(s.v),
            s: target.s - s.s,
        };
        let product = diff.mul(s.conjugate());
        product.v.scale(2.0)
    }

    /// Interpolate and normalize between two quaternions (b3NLerp).
    pub fn nlerp(self, b: Quat, alpha: f32) -> Quat {
        let mut q1 = self;
        if q1.dot(b) < 0.0 {
            q1 = q1.negate();
        }
        let v = q1.v.lerp(b.v, alpha);
        let s = (1.0 - alpha) * q1.s + alpha * b.s;
        Quat { v, s }.normalize()
    }
}

/// Find a quaternion that rotates unit vector v1 to unit vector v2 (b3ComputeQuatBetweenUnitVectors).
pub fn compute_quat_between_unit_vectors(v1: Vec3, v2: Vec3) -> Quat {
    let m = v1.lerp(v2, 0.5);
    let tolerance = 100.0 * FLT_EPSILON;
    let out = if m.length_sq() > tolerance * tolerance {
        Quat {
            v: v1.cross(m),
            s: v1.dot(m),
        }
    } else if absf(v1.x) > 0.5 {
        Quat {
            v: Vec3::new(v1.y, -v1.x, 0.0),
            s: 0.0,
        }
    } else {
        Quat {
            v: Vec3::new(0.0, v1.z, -v1.y),
            s: 0.0,
        }
    };
    out.normalize()
}

/// Extract a quaternion from a rotation matrix (b3MakeQuatFromMatrix).
pub fn make_quat_from_matrix(m: Mat3) -> Quat {
    let c1 = m.cx;
    let c2 = m.cy;
    let c3 = m.cz;
    let trace = m.cx.x + m.cy.y + m.cz.z;
    let q = if trace >= 0.0 {
        Quat {
            v: Vec3::new(c2.z - c3.y, c3.x - c1.z, c1.y - c2.x),
            s: trace + 1.0,
        }
    } else if c1.x > c2.y && c1.x > c3.z {
        Quat {
            v: Vec3::new(c1.x - c2.y - c3.z + 1.0, c2.x + c1.y, c3.x + c1.z),
            s: c2.z - c3.y,
        }
    } else if c2.y > c3.z {
        Quat {
            v: Vec3::new(c1.y + c2.x, c2.y - c3.z - c1.x + 1.0, c3.y + c2.z),
            s: c3.x - c1.z,
        }
    } else {
        Quat {
            v: Vec3::new(c1.z + c3.x, c2.z + c3.y, c3.z - c1.x - c2.y + 1.0),
            s: c1.y - c2.x,
        }
    };
    q.normalize()
}

// --- mat3 -----------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3 {
    pub cx: Vec3,
    pub cy: Vec3,
    pub cz: Vec3,
}

impl Mat3 {
    pub const ZERO: Mat3 = Mat3 {
        cx: Vec3::ZERO,
        cy: Vec3::ZERO,
        cz: Vec3::ZERO,
    };

    /// self * a, matrix times column vector.
    #[inline]
    pub fn mul_v(self, a: Vec3) -> Vec3 {
        Vec3::new(
            self.cx.x * a.x + self.cy.x * a.y + self.cz.x * a.z,
            self.cx.y * a.x + self.cy.y * a.y + self.cz.y * a.z,
            self.cx.z * a.x + self.cy.z * a.y + self.cz.z * a.z,
        )
    }

    /// self * b
    pub fn mul(self, b: Mat3) -> Mat3 {
        Mat3 {
            cx: self.mul_v(b.cx),
            cy: self.mul_v(b.cy),
            cz: self.mul_v(b.cz),
        }
    }

    /// self + b, componentwise (b3AddMM).
    #[inline]
    pub fn add(self, b: Mat3) -> Mat3 {
        Mat3 {
            cx: self.cx.add(b.cx),
            cy: self.cy.add(b.cy),
            cz: self.cz.add(b.cz),
        }
    }

    /// Componentwise absolute value (b3AbsM); the rotation part of an AABB transform uses it.
    #[inline]
    pub fn abs(self) -> Mat3 {
        Mat3 {
            cx: self.cx.abs(),
            cy: self.cy.abs(),
            cz: self.cz.abs(),
        }
    }

    pub fn transpose(self) -> Mat3 {
        Mat3 {
            cx: Vec3::new(self.cx.x, self.cy.x, self.cz.x),
            cy: Vec3::new(self.cx.y, self.cy.y, self.cz.y),
            cz: Vec3::new(self.cx.z, self.cy.z, self.cz.z),
        }
    }

    /// Negate every component (b3NegateMat3).
    #[inline]
    pub fn neg(self) -> Mat3 {
        Mat3 {
            cx: self.cx.neg(),
            cy: self.cy.neg(),
            cz: self.cz.neg(),
        }
    }

    /// Skew-symmetric cross-product matrix of `v`: `skew(v) * a == cross(v, a)` (b3Skew).
    #[inline]
    pub fn skew(v: Vec3) -> Mat3 {
        Mat3 {
            cx: Vec3::new(0.0, v.z, -v.y),
            cy: Vec3::new(-v.z, 0.0, v.x),
            cz: Vec3::new(v.y, -v.x, 0.0),
        }
    }

    #[inline]
    pub fn det(self) -> f32 {
        self.cx.dot(self.cy.cross(self.cz))
    }

    pub fn invert(self) -> Mat3 {
        let det = self.det();
        if absf(det) > 1000.0 * FLT_MIN {
            let inv_det = 1.0 / det;
            let out = Mat3 {
                cx: self.cy.cross(self.cz).scale(inv_det),
                cy: self.cz.cross(self.cx).scale(inv_det),
                cz: self.cx.cross(self.cy).scale(inv_det),
            };
            out.transpose()
        } else {
            Mat3::ZERO
        }
    }

    /// inv(self) * a, via Cramer's rule (b3Solve3).
    pub fn solve(self, a: Vec3) -> Vec3 {
        let det = self.det();
        if absf(det) > 1000.0 * FLT_MIN {
            let inv_det = 1.0 / det;
            let sx = self.cy.cross(self.cz);
            let sy = self.cz.cross(self.cx);
            let sz = self.cx.cross(self.cy);
            Vec3::new(
                inv_det * sx.dot(a),
                inv_det * sy.dot(a),
                inv_det * sz.dot(a),
            )
        } else {
            Vec3::ZERO
        }
    }

    pub fn from_quat(q: Quat) -> Mat3 {
        let xx = q.v.x * q.v.x;
        let yy = q.v.y * q.v.y;
        let zz = q.v.z * q.v.z;
        let xy = q.v.x * q.v.y;
        let xz = q.v.x * q.v.z;
        let xw = q.v.x * q.s;
        let yz = q.v.y * q.v.z;
        let yw = q.v.y * q.s;
        let zw = q.v.z * q.s;
        Mat3 {
            cx: Vec3::new(1.0 - 2.0 * (yy + zz), 2.0 * (xy + zw), 2.0 * (xz - yw)),
            cy: Vec3::new(2.0 * (xy - zw), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + xw)),
            cz: Vec3::new(2.0 * (xz + yw), 2.0 * (yz - xw), 1.0 - 2.0 * (xx + yy)),
        }
    }
}

// --- mat2 -----------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat2 {
    pub cx: Vec2,
    pub cy: Vec2,
}

impl Mat2 {
    #[inline]
    pub fn det(self) -> f32 {
        self.cx.x * self.cy.y - self.cx.y * self.cy.x
    }

    /// self * a, matrix times column vector (b3MulMV for 2x2).
    #[inline]
    pub fn mul_v(self, a: Vec2) -> Vec2 {
        Vec2::new(
            self.cx.x * a.x + self.cy.x * a.y,
            self.cx.y * a.x + self.cy.y * a.y,
        )
    }

    pub fn invert(self) -> Mat2 {
        let det = self.det();
        if absf(det) > 1000.0 * FLT_MIN {
            let inv_det = 1.0 / det;
            Mat2 {
                cx: Vec2::new(inv_det * self.cy.y, -inv_det * self.cx.y),
                cy: Vec2::new(-inv_det * self.cy.x, inv_det * self.cx.x),
            }
        } else {
            Mat2 {
                cx: Vec2::new(0.0, 0.0),
                cy: Vec2::new(0.0, 0.0),
            }
        }
    }

    /// inv(self) * b, assuming self is positive semi-definite (b3Solve2).
    pub fn solve(self, b: Vec2) -> Vec2 {
        let det = self.det();
        if det > 1000.0 * FLT_MIN {
            let inv_det = 1.0 / det;
            Vec2::new(
                inv_det * self.cy.y * b.x - inv_det * self.cy.x * b.y,
                -inv_det * self.cx.y * b.x + inv_det * self.cx.x * b.y,
            )
        } else {
            Vec2::new(0.0, 0.0)
        }
    }
}

// --- transform ------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub p: Vec3,
    pub q: Quat,
}

impl Transform {
    pub const IDENTITY: Transform = Transform {
        p: Vec3::ZERO,
        q: Quat::IDENTITY,
    };

    /// Apply transform to a point: q rotates, then translate by p.
    pub fn point(self, v: Vec3) -> Vec3 {
        self.q.rotate(v).add(self.p)
    }
    /// Inverse rigid transform (b3InvMulTransforms against identity): conjugate q, rotate -p.
    pub fn invert(self) -> Transform {
        Transform {
            p: self.q.inv_rotate(self.p.neg()),
            q: self.q.conjugate(),
        }
    }
    /// Inverse of `point`.
    pub fn inv_point(self, v: Vec3) -> Vec3 {
        self.q.inv_rotate(v.sub(self.p))
    }
    /// self * b — b's local frame composed into self.
    pub fn mul(self, b: Transform) -> Transform {
        Transform {
            p: self.q.rotate(b.p).add(self.p),
            q: self.q.mul(b.q),
        }
    }
    /// inv(self) * b — b expressed in self's local frame.
    pub fn inv_mul(self, b: Transform) -> Transform {
        Transform {
            p: self.q.inv_rotate(b.p.sub(self.p)),
            q: self.q.inv_mul(b.q),
        }
    }
}

// --- plane ----------------------------------------------------------------------------------

// repr(C) so a hull's plane pool in wasm linear memory reinterprets as `&[Plane]` (kernel/src/hull.rs).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    pub normal: Vec3,
    pub offset: f32,
}

impl Plane {
    /// Plane through `point` with the given normal (b3MakePlaneFromNormalAndPoint).
    #[inline]
    pub fn from_normal_and_point(normal: Vec3, point: Vec3) -> Plane {
        Plane {
            normal,
            offset: normal.dot(point),
        }
    }
    /// Plane through three points, normal = normalize((p2-p1) × (p3-p1)) (b3MakePlaneFromPoints).
    pub fn from_points(p1: Vec3, p2: Vec3, p3: Vec3) -> Plane {
        let normal = p2.sub(p1).cross(p3.sub(p1)).normalize();
        Plane {
            normal,
            offset: normal.dot(p1),
        }
    }
    /// Signed separation of `point` from the plane (b3PlaneSeparation).
    #[inline]
    pub fn separation(self, point: Vec3) -> f32 {
        self.normal.dot(point) - self.offset
    }
    /// Transform a plane by a rigid transform (b3TransformPlane).
    pub fn transform(self, t: Transform) -> Plane {
        let normal = t.q.rotate(self.normal);
        Plane {
            normal,
            offset: self.offset + normal.dot(t.p),
        }
    }
}

// --- segment / line distance ----------------------------------------------------------------

/// Normalize `a` and return its length; zero vector if `a` is tiny (b3GetLengthAndNormalize).
pub fn get_length_and_normalize(a: Vec3) -> (Vec3, f32) {
    let length = a.length();
    if length < FLT_EPSILON {
        return (Vec3::ZERO, length);
    }
    let inv_length = 1.0 / length;
    (a.scale(inv_length), length)
}

/// Closest points on two segments or infinite lines (b3SegmentDistanceResult).
#[derive(Clone, Copy, Debug)]
pub struct SegmentDistanceResult {
    pub point1: Vec3,
    pub fraction1: f32,
    pub point2: Vec3,
    pub fraction2: f32,
}

/// Closest points on the two segments p1-q1 and p2-q2 (b3SegmentDistance).
pub fn segment_distance(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> SegmentDistanceResult {
    let d1 = q1.sub(p1);
    let d2 = q2.sub(p2);
    let r = p1.sub(p2);

    let a = d1.dot(d1);
    let b = d1.dot(d2);
    let c = d1.dot(r);
    let e = d2.dot(d2);
    let f = d2.dot(r);

    let eps = 100.0 * FLT_EPSILON;

    // Both segments degenerate into points.
    if a < eps && e < eps {
        return SegmentDistanceResult {
            point1: p1,
            fraction1: 0.0,
            point2: p2,
            fraction2: 0.0,
        };
    }

    // First segment degenerates into a point.
    if a < eps {
        let s2 = clampf(f / e, 0.0, 1.0);
        return SegmentDistanceResult {
            point1: p1,
            fraction1: 0.0,
            point2: p2.mul_add(s2, d2),
            fraction2: s2,
        };
    }

    // Second segment degenerates into a point.
    if e < eps {
        let s1 = clampf(-c / a, 0.0, 1.0);
        return SegmentDistanceResult {
            point1: p1.mul_add(s1, d1),
            fraction1: s1,
            point2: p2,
            fraction2: 0.0,
        };
    }

    // Non-degenerate case.
    let denom = a * e - b * b;
    let mut s1 = if denom > 1000.0 * FLT_MIN {
        clampf((b * f - c * e) / denom, 0.0, 1.0)
    } else {
        0.0
    };
    let mut s2 = (b * s1 + f) / e;

    // Clamp s2 and recompute s1 if necessary.
    if s2 < 0.0 {
        s1 = clampf(-c / a, 0.0, 1.0);
        s2 = 0.0;
    } else if s2 > 1.0 {
        s1 = clampf((b - c) / a, 0.0, 1.0);
        s2 = 1.0;
    }

    SegmentDistanceResult {
        point1: p1.mul_add(s1, d1),
        fraction1: s1,
        point2: p2.mul_add(s2, d2),
        fraction2: s2,
    }
}

/// Closest point on segment a-b to query point q (b3PointToSegmentDistance).
pub fn point_to_segment_distance(a: Vec3, b: Vec3, q: Vec3) -> Vec3 {
    let ab = b.sub(a);
    let aq = q.sub(a);
    let alpha = ab.dot(aq);
    if alpha <= 0.0 {
        return a;
    }
    let denominator = ab.dot(ab);
    if alpha > denominator {
        return b;
    }
    a.mul_add(alpha / denominator, ab)
}

/// Closest points on the two infinite lines p1+s1*d1 and p2+s2*d2 (b3LineDistance).
pub fn line_distance(p1: Vec3, d1: Vec3, p2: Vec3, d2: Vec3) -> SegmentDistanceResult {
    // Solve A*x = b
    let a11 = d1.dot(d1);
    let a12 = -d1.dot(d2);
    let a21 = d2.dot(d1);
    let a22 = -d2.dot(d2);

    let w = p1.sub(p2);
    let b1 = -d1.dot(w);
    let b2 = -d2.dot(w);

    let det = a11 * a22 - a12 * a21;
    if det * det < 1000.0 * FLT_MIN {
        // Lines are parallel - project p2 onto line L1: x1 = p1 + s1 * d1
        let s1 = p2.sub(p1).dot(d1) / d1.dot(d1);
        let s2 = 0.0;
        return SegmentDistanceResult {
            point1: p1.mul_add(s1, d1),
            fraction1: s1,
            point2: p2.mul_add(s2, d2),
            fraction2: s2,
        };
    }

    let s1 = (a22 * b1 - a12 * b2) / det;
    let s2 = (a11 * b2 - a21 * b1) / det;
    SegmentDistanceResult {
        point1: p1.mul_add(s1, d1),
        fraction1: s1,
        point2: p2.mul_add(s2, d2),
        fraction2: s2,
    }
}

/// Are both closest-point fractions within [0, 1]? (b3IsWithinSegments).
pub fn is_within_segments(result: &SegmentDistanceResult) -> bool {
    0.0 <= result.fraction1
        && result.fraction1 <= 1.0
        && 0.0 <= result.fraction2
        && result.fraction2 <= 1.0
}
