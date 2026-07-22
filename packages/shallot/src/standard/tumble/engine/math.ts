// Vector math, ported op-for-op from Box3D's math_functions.h/.c (Erin Catto, MIT).
//
// Bit-exact discipline: JS numbers are f64, so every f32 arithmetic op is emulated with
// `f32` (Math.fround), one op per wrap, mirroring the C expression tree left to right.
// Box3D builds with -ffp-contract=off and JS has no FMA, so `a*b + c*d` is three wraps,
// never fused. Scalar min/max are explicit comparisons (not Math.min/max, which diverge on
// NaN and signed zero) to match the BOX3D_DISABLE_SIMD scalar path the port targets.
// See the README.
//
// Float mode only: BOX3D_DOUBLE_PRECISION (double world positions for large worlds) is out of
// scope, so b3Pos collapses to Vec3 and b3WorldTransform to Transform.

/** Round to the nearest f32, emulating a single IEEE-754 float operation. */
export const f32 = Math.fround;

// biome-ignore lint/suspicious/noApproximativeNumericConstant: B3_PI verbatim — the port mirrors Box3D's constant, not Math.PI.
export const PI = f32(3.14159265359);
export const DEG_TO_RAD = f32(0.01745329251);
export const RAD_TO_DEG = f32(57.2957795131);
export const MIN_SCALE = f32(0.01);

const FLT_EPSILON = f32(1.1920928955078125e-7); // 2^-23
const FLT_MIN = f32(1.1754943508222875e-38); // 2^-126, smallest normal
const FLT_MAX = f32(3.4028234663852886e38);

const TWO_PI = f32(2 * PI);

export type Vec2 = { x: number; y: number };
export type Vec3 = { x: number; y: number; z: number };

/** A quaternion split into vector part `v` and scalar part `s`, matching Box3D's b3Quat. */
export type Quat = { v: Vec3; s: number };

/** A rigid transform: rotation `q` then translation `p`. */
export type Transform = { p: Vec3; q: Quat };

/** A 3x3 matrix stored as three column vectors. */
export type Mat3 = { cx: Vec3; cy: Vec3; cz: Vec3 };

/** A 2x2 matrix stored as two column vectors. */
export type Mat2 = { cx: Vec2; cy: Vec2 };

/** Axis-aligned bounding box. */
export type AABB = { lowerBound: Vec3; upperBound: Vec3 };

/** A plane: `separation = dot(normal, point) - offset`. */
export type Plane = { normal: Vec3; offset: number };

export type CosSin = { cosine: number; sine: number };

/** In float mode a world position is a plain Vec3. */
export type Pos = Vec3;
/** In float mode a world transform is a plain Transform. */
export type WorldTransform = Transform;

// --- scalar helpers -------------------------------------------------------------------------

/** Is this a finite, non-NaN float? */
export const isValidFloat = (a: number): boolean => Number.isFinite(a);

export const absf = (a: number): number => (a < 0 ? -a : a);
export const minf = (a: number, b: number): number => (a < b ? a : b);
export const maxf = (a: number, b: number): number => (a > b ? a : b);
export const minInt = (a: number, b: number): number => (a < b ? a : b);
export const maxInt = (a: number, b: number): number => (a > b ? a : b);
export const clampInt = (a: number, lo: number, hi: number): number =>
    a < lo ? lo : hi < a ? hi : a;
export const clampf = (a: number, lo: number, hi: number): number =>
    a < lo ? lo : hi < a ? hi : a;

/** (1 - alpha) * a + alpha * b */
export const lerpf = (a: number, b: number, alpha: number): number =>
    f32(f32(f32(1 - alpha) * a) + f32(alpha * b));

/**
 * Round every float field of a user-facing def/config to f32 once, where it crosses into the engine.
 * The C API is f32, so an f64 config value feeds one extra bit into the solver and breaks bit-exact
 * parity (`f32(x - 0.3_f64)` ≠ `x - 0.3_f32`). Recurses into vec/transform fields and arrays
 * (element-wise); `bigint`/`boolean`/`string`/function values pass through, and `userData` is opaque.
 * Returns a fresh deep copy, so the caller's object is never aliased or mutated.
 */
export function froundConfig<T>(cfg: T): T {
    if (typeof cfg === "number") return f32(cfg) as T;
    if (cfg === null || typeof cfg !== "object") return cfg;
    if (Array.isArray(cfg)) return cfg.map((v) => froundConfig(v)) as T;
    const out: Record<string, unknown> = {};
    for (const key in cfg) {
        out[key] =
            key === "userData"
                ? (cfg as Record<string, unknown>)[key]
                : froundConfig((cfg as Record<string, unknown>)[key]);
    }
    return out as T;
}

// --- transcendentals ------------------------------------------------------------------------

// Round to nearest integer, ties to even — matching the quotient rounding IEEE remainderf uses.
function rintEven(x: number): number {
    const fl = Math.floor(x);
    const diff = x - fl;
    if (diff < 0.5) return fl;
    if (diff > 0.5) return fl + 1;
    return fl % 2 === 0 ? fl : fl + 1;
}

// IEEE-754 remainder. For finite f32 inputs the true remainder is itself f32-representable and
// the f64 subtraction below is exact (both operands are small integers-scaled f32 values), so
// this reproduces libm remainderf bit for bit over the range the sim uses.
function remainderf(x: number, y: number): number {
    const n = rintEven(x / y);
    const r = f32(x - n * y);
    // IEEE-754: a zero remainder takes the sign of the dividend (a - a rounds to +0 otherwise).
    if (r === 0) return x < 0 || Object.is(x, -0) ? -0 : 0;
    return r;
}

/** Convert any angle into the range [-pi, pi]. */
export const unwindAngle = (radians: number): number => remainderf(radians, TWO_PI);

// Minimax atan approximation, accurate to ~0.0023 degrees. Hand coded for cross-platform
// determinism (libm atan2f is not). Ported from b3Atan2 in math_functions.c.
const ATAN_P0 = f32(0.024840285);
const ATAN_P1 = f32(0.18681418);
const ATAN_P2 = f32(-0.094097948);
const ATAN_P3 = f32(-0.33213072);

/** Approximate arctangent in [-pi, pi], deterministic across platforms. */
export function atan2(y: number, x: number): number {
    if (x === 0 && y === 0) return 0;

    const ax = absf(x);
    const ay = absf(y);
    const mx = maxf(ay, ax);
    const mn = minf(ay, ax);
    const a = f32(mn / mx);

    const s = f32(a * a);
    const c = f32(s * a);
    const q = f32(s * s);
    let r = f32(f32(ATAN_P0 * q) + ATAN_P1);
    const t = f32(f32(ATAN_P2 * q) + ATAN_P3);
    r = f32(f32(r * s) + t);
    r = f32(f32(r * c) + a);

    if (ay > ax) r = f32(f32(1.57079637) - r);
    if (x < 0) r = f32(f32(3.14159274) - r);
    if (y < 0) r = -r;

    return r;
}

// Bhāskara I sine/cosine approximation, normalized to unit magnitude. Ported from
// b3ComputeCosSin in math_functions.c. Accurate to ~0.1 degrees.
export function computeCosSin(radians: number): CosSin {
    const x = unwindAngle(radians);
    const pi2 = f32(PI * PI);

    // cosine needs angle in [-pi/2, pi/2]
    let c: number;
    if (x < f32(-0.5 * PI)) {
        const y = f32(x + PI);
        const y2 = f32(y * y);
        c = f32(-f32(f32(pi2 - f32(4 * y2)) / f32(pi2 + y2)));
    } else if (x > f32(0.5 * PI)) {
        const y = f32(x - PI);
        const y2 = f32(y * y);
        c = f32(-f32(f32(pi2 - f32(4 * y2)) / f32(pi2 + y2)));
    } else {
        const y2 = f32(x * x);
        c = f32(f32(pi2 - f32(4 * y2)) / f32(pi2 + y2));
    }

    // sine needs angle in [0, pi]
    let s: number;
    if (x < 0) {
        const y = f32(x + PI);
        const piMinusY = f32(PI - y);
        const num = f32(f32(16 * y) * piMinusY);
        const den = f32(f32(5 * pi2) - f32(f32(4 * y) * piMinusY));
        s = f32(-f32(num / den));
    } else {
        const piMinusX = f32(PI - x);
        const num = f32(f32(16 * x) * piMinusX);
        const den = f32(f32(5 * pi2) - f32(f32(4 * x) * piMinusX));
        s = f32(num / den);
    }

    const mag = f32(Math.sqrt(f32(f32(s * s) + f32(c * c))));
    const invMag = mag > 0 ? f32(1 / mag) : 0;
    return { cosine: f32(c * invMag), sine: f32(s * invMag) };
}

/** @deprecated use computeCosSin */
export const sin = (radians: number): number => computeCosSin(radians).sine;
/** @deprecated use computeCosSin */
export const cos = (radians: number): number => computeCosSin(radians).cosine;

// --- vec3 -----------------------------------------------------------------------------------

export const vec3 = {
    zero: (): Vec3 => ({ x: 0, y: 0, z: 0 }),
    axisX: (): Vec3 => ({ x: 1, y: 0, z: 0 }),
    axisY: (): Vec3 => ({ x: 0, y: 1, z: 0 }),
    axisZ: (): Vec3 => ({ x: 0, y: 0, z: 1 }),

    /** Round every component to f32 (the C float-store boundary for f64-valued caller geometry). */
    round: (a: Vec3): Vec3 => ({ x: f32(a.x), y: f32(a.y), z: f32(a.z) }),

    add: (a: Vec3, b: Vec3): Vec3 => ({ x: f32(a.x + b.x), y: f32(a.y + b.y), z: f32(a.z + b.z) }),
    sub: (a: Vec3, b: Vec3): Vec3 => ({ x: f32(a.x - b.x), y: f32(a.y - b.y), z: f32(a.z - b.z) }),
    mul: (a: Vec3, b: Vec3): Vec3 => ({ x: f32(a.x * b.x), y: f32(a.y * b.y), z: f32(a.z * b.z) }),
    neg: (a: Vec3): Vec3 => ({ x: -a.x, y: -a.y, z: -a.z }),

    /** s * a */
    scale: (s: number, a: Vec3): Vec3 => ({ x: f32(s * a.x), y: f32(s * a.y), z: f32(s * a.z) }),
    /** a + s * b */
    mulAdd: (a: Vec3, s: number, b: Vec3): Vec3 => ({
        x: f32(a.x + f32(s * b.x)),
        y: f32(a.y + f32(s * b.y)),
        z: f32(a.z + f32(s * b.z)),
    }),
    /** a - s * b */
    mulSub: (a: Vec3, s: number, b: Vec3): Vec3 => ({
        x: f32(a.x - f32(s * b.x)),
        y: f32(a.y - f32(s * b.y)),
        z: f32(a.z - f32(s * b.z)),
    }),

    // Out-param variants for hot zero-alloc paths: identical f32 expression trees, written into a
    // caller-owned Vec3 instead of a fresh literal. `o` may alias `a` or `b` (each component is read
    // before it is written). Returns `o`.
    subOut: (a: Vec3, b: Vec3, o: Vec3): Vec3 => {
        const x = f32(a.x - b.x);
        const y = f32(a.y - b.y);
        const z = f32(a.z - b.z);
        o.x = x;
        o.y = y;
        o.z = z;
        return o;
    },
    /** a + s * b, written into `o`. */
    mulAddOut: (a: Vec3, s: number, b: Vec3, o: Vec3): Vec3 => {
        const x = f32(a.x + f32(s * b.x));
        const y = f32(a.y + f32(s * b.y));
        const z = f32(a.z + f32(s * b.z));
        o.x = x;
        o.y = y;
        o.z = z;
        return o;
    },
    /** a - s * b, written into `o`. */
    mulSubOut: (a: Vec3, s: number, b: Vec3, o: Vec3): Vec3 => {
        const x = f32(a.x - f32(s * b.x));
        const y = f32(a.y - f32(s * b.y));
        const z = f32(a.z - f32(s * b.z));
        o.x = x;
        o.y = y;
        o.z = z;
        return o;
    },
    /** a + b, written into `o`. */
    addOut: (a: Vec3, b: Vec3, o: Vec3): Vec3 => {
        const x = f32(a.x + b.x);
        const y = f32(a.y + b.y);
        const z = f32(a.z + b.z);
        o.x = x;
        o.y = y;
        o.z = z;
        return o;
    },
    /** Copy `a`'s components into `o` (no rounding — `a` is already f32-valued). */
    copy: (a: Vec3, o: Vec3): Vec3 => {
        o.x = a.x;
        o.y = a.y;
        o.z = a.z;
        return o;
    },
    /** s * a, written into `o` (may alias `a`). */
    scaleOut: (s: number, a: Vec3, o: Vec3): Vec3 => {
        o.x = f32(s * a.x);
        o.y = f32(s * a.y);
        o.z = f32(s * a.z);
        return o;
    },

    dot: (a: Vec3, b: Vec3): number => {
        const xx = f32(a.x * b.x);
        const yy = f32(a.y * b.y);
        const zz = f32(a.z * b.z);
        return f32(f32(xx + yy) + zz);
    },
    cross: (a: Vec3, b: Vec3): Vec3 => ({
        x: f32(f32(a.y * b.z) - f32(a.z * b.y)),
        y: f32(f32(a.z * b.x) - f32(a.x * b.z)),
        z: f32(f32(a.x * b.y) - f32(a.y * b.x)),
    }),
    /** cross(|a|, b) with all-plus signs — used for rotation-arc bounds (b3ModifiedCross). */
    modifiedCross: (a: Vec3, b: Vec3): Vec3 => ({
        x: f32(f32(a.y * b.z) + f32(a.z * b.y)),
        y: f32(f32(a.z * b.x) + f32(a.x * b.z)),
        z: f32(f32(a.x * b.y) + f32(a.y * b.x)),
    }),
    /** a × b, written into `o` (may alias `a` or `b` — components are read before written). */
    crossOut: (a: Vec3, b: Vec3, o: Vec3): Vec3 => {
        const x = f32(f32(a.y * b.z) - f32(a.z * b.y));
        const y = f32(f32(a.z * b.x) - f32(a.x * b.z));
        const z = f32(f32(a.x * b.y) - f32(a.y * b.x));
        o.x = x;
        o.y = y;
        o.z = z;
        return o;
    },
    /** {@link vec3.modifiedCross}, written into `o` (may alias `a` or `b`). */
    modifiedCrossOut: (a: Vec3, b: Vec3, o: Vec3): Vec3 => {
        const x = f32(f32(a.y * b.z) + f32(a.z * b.y));
        const y = f32(f32(a.z * b.x) + f32(a.x * b.z));
        const z = f32(f32(a.x * b.y) + f32(a.y * b.x));
        o.x = x;
        o.y = y;
        o.z = z;
        return o;
    },

    lengthSq: (a: Vec3): number => {
        const xx = f32(a.x * a.x);
        const yy = f32(a.y * a.y);
        const zz = f32(a.z * a.z);
        return f32(f32(xx + yy) + zz);
    },
    length: (v: Vec3): number => f32(Math.sqrt(vec3.dot(v, v))),
    distance: (a: Vec3, b: Vec3): number => vec3.length(vec3.sub(b, a)),
    distanceSquared: (a: Vec3, b: Vec3): number => {
        const d = vec3.sub(b, a);
        return vec3.dot(d, d);
    },
    distanceSq: (a: Vec3, b: Vec3): number => vec3.lengthSq(vec3.sub(b, a)),

    normalize: (a: Vec3): Vec3 => {
        const lengthSq = vec3.lengthSq(a);
        if (lengthSq > f32(1000 * FLT_MIN)) {
            const s = f32(1 / f32(Math.sqrt(lengthSq)));
            return vec3.scale(s, a);
        }
        return { x: 0, y: 0, z: 0 };
    },

    lerp: (a: Vec3, b: Vec3, alpha: number): Vec3 => {
        const t = f32(1 - alpha);
        return {
            x: f32(f32(t * a.x) + f32(alpha * b.x)),
            y: f32(f32(t * a.y) + f32(alpha * b.y)),
            z: f32(f32(t * a.z) + f32(alpha * b.z)),
        };
    },

    /** s * a + t * b (b3Blend2). */
    blend2: (s: number, a: Vec3, t: number, b: Vec3): Vec3 => ({
        x: f32(f32(s * a.x) + f32(t * b.x)),
        y: f32(f32(s * a.y) + f32(t * b.y)),
        z: f32(f32(s * a.z) + f32(t * b.z)),
    }),

    /** s * a + t * b + u * c (b3Blend3). */
    blend3: (s: number, a: Vec3, t: number, b: Vec3, u: number, c: Vec3): Vec3 => ({
        x: f32(f32(f32(s * a.x) + f32(t * b.x)) + f32(u * c.x)),
        y: f32(f32(f32(s * a.y) + f32(t * b.y)) + f32(u * c.y)),
        z: f32(f32(f32(s * a.z) + f32(t * b.z)) + f32(u * c.z)),
    }),

    abs: (a: Vec3): Vec3 => ({ x: absf(a.x), y: absf(a.y), z: absf(a.z) }),
    min: (a: Vec3, b: Vec3): Vec3 => ({ x: minf(a.x, b.x), y: minf(a.y, b.y), z: minf(a.z, b.z) }),
    max: (a: Vec3, b: Vec3): Vec3 => ({ x: maxf(a.x, b.x), y: maxf(a.y, b.y), z: maxf(a.z, b.z) }),
    /** |a|, written into `o` (may alias `a`). */
    absOut: (a: Vec3, o: Vec3): Vec3 => {
        o.x = absf(a.x);
        o.y = absf(a.y);
        o.z = absf(a.z);
        return o;
    },
    /** Component-wise max, written into `o` (may alias `a` or `b`). */
    maxOut: (a: Vec3, b: Vec3, o: Vec3): Vec3 => {
        o.x = maxf(a.x, b.x);
        o.y = maxf(a.y, b.y);
        o.z = maxf(a.z, b.z);
        return o;
    },
    /** Component-wise min, written into `o` (may alias `a` or `b`). */
    minOut: (a: Vec3, b: Vec3, o: Vec3): Vec3 => {
        o.x = minf(a.x, b.x);
        o.y = minf(a.y, b.y);
        o.z = minf(a.z, b.z);
        return o;
    },
    clamp: (a: Vec3, lo: Vec3, hi: Vec3): Vec3 => ({
        x: clampf(a.x, lo.x, hi.x),
        y: clampf(a.y, lo.y, hi.y),
        z: clampf(a.z, lo.z, hi.z),
    }),

    /** Unit vector perpendicular to `a` (b3Perp: axis-selecting, always normalized). */
    perp: (a: Vec3): Vec3 => {
        const p = a.x < -0.5 || 0.5 < a.x ? { x: a.y, y: -a.x, z: 0 } : { x: 0, y: a.z, z: -a.y };
        return vec3.normalize(p);
    },

    isNormalized: (a: Vec3): boolean => {
        const aa = vec3.dot(a, a);
        return absf(f32(1 - aa)) < f32(100 * FLT_EPSILON);
    },

    isValid: (a: Vec3): boolean => isValidFloat(a.x) && isValidFloat(a.y) && isValidFloat(a.z),

    /** Component -1 or 1 (1 if zero). */
    sign: (a: Vec3): Vec3 => ({
        x: a.x >= 0 ? 1 : -1,
        y: a.y >= 0 ? 1 : -1,
        z: a.z >= 0 ? 1 : -1,
    }),
};

/** Normalize `a` and return its length; zero vector if `a` is tiny (b3GetLengthAndNormalize). */
export function getLengthAndNormalize(a: Vec3): { v: Vec3; length: number } {
    const length = vec3.length(a);
    if (length < FLT_EPSILON) return { v: { x: 0, y: 0, z: 0 }, length };
    const invLength = f32(1 / length);
    return { v: vec3.scale(invLength, a), length };
}

/** Index of the largest component (b3MaxElementIndex). */
export function maxElementIndex(v: Vec3): number {
    return v.x < v.y ? (v.y < v.z ? 2 : 1) : v.x < v.z ? 2 : 0;
}

/** dot(a, cross(b, c)) — the scalar triple product, ported from b3ScalarTripleProduct. */
export function scalarTripleProduct(a: Vec3, b: Vec3, c: Vec3): number {
    const d: Vec3 = {
        x: f32(f32(b.y * c.z) - f32(b.z * c.y)),
        y: f32(f32(b.z * c.x) - f32(b.x * c.z)),
        z: f32(f32(b.x * c.y) - f32(b.y * c.x)),
    };
    const xx = f32(a.x * d.x);
    const yy = f32(a.y * d.y);
    const zz = f32(a.z * d.z);
    return f32(f32(xx + yy) + zz);
}

// Unit vector perpendicular to a unit vector, using tuned coefficients that keep the result
// well-conditioned. Ported from b3ArbitraryPerp (math_internal.h).
export function arbitraryPerp(v: Vec3): Vec3 {
    const a = f32(0.67);
    const b = f32(-0.42);
    let p: Vec3;
    if (v.x < -0.5 || 0.5 < v.x) {
        p = { x: f32(f32(a * v.y) + f32(b * v.z)), y: f32(-a * v.x), z: f32(-b * v.x) };
    } else if (v.y < -0.5 || 0.5 < v.y) {
        p = { x: f32(a * v.y), y: f32(f32(-a * v.x) + f32(b * v.z)), z: f32(-b * v.y) };
    } else {
        p = { x: f32(a * v.z), y: f32(b * v.z), z: f32(f32(-a * v.x) - f32(b * v.y)) };
    }
    return vec3.normalize(p);
}

/** Closest points on two segments or infinite lines (b3SegmentDistanceResult). */
export type SegmentDistanceResult = {
    point1: Vec3;
    fraction1: number;
    point2: Vec3;
    fraction2: number;
};

/** Closest points on the two segments p1-q1 and p2-q2 (b3SegmentDistance). */
export function segmentDistance(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3): SegmentDistanceResult {
    const d1 = vec3.sub(q1, p1);
    const d2 = vec3.sub(q2, p2);
    const r = vec3.sub(p1, p2);

    const a = vec3.dot(d1, d1);
    const b = vec3.dot(d1, d2);
    const c = vec3.dot(d1, r);
    const e = vec3.dot(d2, d2);
    const f = vec3.dot(d2, r);

    const eps = f32(100 * FLT_EPSILON);

    // Both segments degenerate into points.
    if (a < eps && e < eps) {
        return { point1: p1, fraction1: 0, point2: p2, fraction2: 0 };
    }

    // First segment degenerates into a point.
    if (a < eps) {
        const s2 = clampf(f32(f / e), 0, 1);
        return { point1: p1, fraction1: 0, point2: vec3.mulAdd(p2, s2, d2), fraction2: s2 };
    }

    // Second segment degenerates into a point.
    if (e < eps) {
        const s1 = clampf(f32(-c / a), 0, 1);
        return { point1: vec3.mulAdd(p1, s1, d1), fraction1: s1, point2: p2, fraction2: 0 };
    }

    // Non-degenerate case.
    const denom = f32(f32(a * e) - f32(b * b));
    let s1 =
        denom > f32(1000 * FLT_MIN) ? clampf(f32(f32(f32(b * f) - f32(c * e)) / denom), 0, 1) : 0;
    let s2 = f32(f32(f32(b * s1) + f) / e);

    // Clamp s2 and recompute s1 if necessary.
    if (s2 < 0) {
        s1 = clampf(f32(-c / a), 0, 1);
        s2 = 0;
    } else if (s2 > 1) {
        s1 = clampf(f32(f32(b - c) / a), 0, 1);
        s2 = 1;
    }

    return {
        point1: vec3.mulAdd(p1, s1, d1),
        fraction1: s1,
        point2: vec3.mulAdd(p2, s2, d2),
        fraction2: s2,
    };
}

/** Closest point on segment a-b to query point q (b3PointToSegmentDistance). */
export function pointToSegmentDistance(a: Vec3, b: Vec3, q: Vec3): Vec3 {
    const ab = vec3.sub(b, a);
    const aq = vec3.sub(q, a);
    const alpha = vec3.dot(ab, aq);
    if (alpha <= 0) {
        // q projects outside interval [a, b] on the side of a.
        return a;
    }
    const denominator = vec3.dot(ab, ab);
    if (alpha > denominator) {
        // q projects outside interval [a, b] on the side of b.
        return b;
    }
    // q projects inside interval [a, b].
    return vec3.mulAdd(a, f32(alpha / denominator), ab);
}

/** Closest points on the two infinite lines p1+s1*d1 and p2+s2*d2 (b3LineDistance). */
export function lineDistance(p1: Vec3, d1: Vec3, p2: Vec3, d2: Vec3): SegmentDistanceResult {
    // Solve A*x = b
    const a11 = vec3.dot(d1, d1);
    const a12 = -vec3.dot(d1, d2);
    const a21 = vec3.dot(d2, d1);
    const a22 = -vec3.dot(d2, d2);

    const w = vec3.sub(p1, p2);
    const b1 = -vec3.dot(d1, w);
    const b2 = -vec3.dot(d2, w);

    const det = f32(f32(a11 * a22) - f32(a12 * a21));
    if (f32(det * det) < f32(1000 * FLT_MIN)) {
        // Lines are parallel - project p2 onto line L1: x1 = p1 + s1 * d1
        const s1 = f32(vec3.dot(vec3.sub(p2, p1), d1) / vec3.dot(d1, d1));
        const s2 = 0;
        return {
            point1: vec3.mulAdd(p1, s1, d1),
            fraction1: s1,
            point2: vec3.mulAdd(p2, s2, d2),
            fraction2: s2,
        };
    }

    const s1 = f32(f32(f32(a22 * b1) - f32(a12 * b2)) / det);
    const s2 = f32(f32(f32(a11 * b2) - f32(a21 * b1)) / det);
    return {
        point1: vec3.mulAdd(p1, s1, d1),
        fraction1: s1,
        point2: vec3.mulAdd(p2, s2, d2),
        fraction2: s2,
    };
}

/** Are both closest-point fractions within [0, 1]? (b3IsWithinSegments). */
export function isWithinSegments(result: SegmentDistanceResult): boolean {
    return (
        0 <= result.fraction1 &&
        result.fraction1 <= 1 &&
        0 <= result.fraction2 &&
        result.fraction2 <= 1
    );
}

// --- vec2 -----------------------------------------------------------------------------------

export const vec2 = {
    add: (a: Vec2, b: Vec2): Vec2 => ({ x: f32(a.x + b.x), y: f32(a.y + b.y) }),
    sub: (a: Vec2, b: Vec2): Vec2 => ({ x: f32(a.x - b.x), y: f32(a.y - b.y) }),
    neg: (a: Vec2): Vec2 => ({ x: -a.x, y: -a.y }),
    scale: (s: number, a: Vec2): Vec2 => ({ x: f32(s * a.x), y: f32(s * a.y) }),
    mulAdd: (a: Vec2, s: number, b: Vec2): Vec2 => ({
        x: f32(a.x + f32(s * b.x)),
        y: f32(a.y + f32(s * b.y)),
    }),
    mulSub: (a: Vec2, s: number, b: Vec2): Vec2 => ({
        x: f32(a.x - f32(s * b.x)),
        y: f32(a.y - f32(s * b.y)),
    }),
    dot: (a: Vec2, b: Vec2): number => f32(f32(a.x * b.x) + f32(a.y * b.y)),
    cross: (a: Vec2, b: Vec2): number => f32(f32(a.x * b.y) - f32(a.y * b.x)),
    lengthSquared: (a: Vec2): number => vec2.dot(a, a),
    length: (a: Vec2): number => f32(Math.sqrt(vec2.dot(a, a))),
    /** Squared distance between two 2D points (b3DistanceSquared2). */
    distanceSquared: (a: Vec2, b: Vec2): number => {
        const dx = f32(b.x - a.x);
        const dy = f32(b.y - a.y);
        return f32(f32(dx * dx) + f32(dy * dy));
    },
};

// --- quat -----------------------------------------------------------------------------------

// Working register for quat.rotateOut/invRotateOut (t1..t3 chain); never live across calls.
const rotateScratch: Vec3 = { x: 0, y: 0, z: 0 };

export const quat = {
    identity: (): Quat => ({ v: { x: 0, y: 0, z: 0 }, s: 1 }),

    dot: (a: Quat, b: Quat): number => {
        const xx = f32(a.v.x * b.v.x);
        const yy = f32(a.v.y * b.v.y);
        const zz = f32(a.v.z * b.v.z);
        const ss = f32(a.s * b.s);
        return f32(f32(f32(xx + yy) + zz) + ss);
    },

    /** q1 * q2 */
    mul: (q1: Quat, q2: Quat): Quat => {
        const t1 = vec3.cross(q1.v, q2.v);
        const t2 = vec3.mulAdd(t1, q1.s, q2.v);
        const t3 = vec3.mulAdd(t2, q2.s, q1.v);
        return { v: t3, s: f32(f32(q1.s * q2.s) - vec3.dot(q1.v, q2.v)) };
    },

    /** inv(q1) * q2 */
    invMul: (q1: Quat, q2: Quat): Quat => {
        const t1 = vec3.cross(q2.v, q1.v);
        const t2 = vec3.mulAdd(t1, q1.s, q2.v);
        const t3 = vec3.mulSub(t2, q2.s, q1.v);
        return { v: t3, s: f32(f32(q1.s * q2.s) + vec3.dot(q1.v, q2.v)) };
    },

    conjugate: (q: Quat): Quat => ({ v: { x: -q.v.x, y: -q.v.y, z: -q.v.z }, s: q.s }),
    negate: (q: Quat): Quat => ({ v: { x: -q.v.x, y: -q.v.y, z: -q.v.z }, s: -q.s }),

    // Out-param variants for hot zero-alloc paths: identical f32 expression trees, written into a
    // caller-owned Quat. `o` must not alias `q1`/`q2` (its `v` is used as the working register).
    /** Copy `q`'s components into `o` (no rounding — `q` is already f32-valued). */
    copy: (q: Quat, o: Quat): Quat => {
        o.v.x = q.v.x;
        o.v.y = q.v.y;
        o.v.z = q.v.z;
        o.s = q.s;
        return o;
    },
    /** {@link quat.conjugate}, written into `o` (may alias `q`). */
    conjugateOut: (q: Quat, o: Quat): Quat => {
        o.v.x = -q.v.x;
        o.v.y = -q.v.y;
        o.v.z = -q.v.z;
        o.s = q.s;
        return o;
    },
    /** q1 * q2, written into `o`. */
    mulOut: (q1: Quat, q2: Quat, o: Quat): Quat => {
        const s = f32(f32(q1.s * q2.s) - vec3.dot(q1.v, q2.v));
        vec3.crossOut(q1.v, q2.v, o.v);
        vec3.mulAddOut(o.v, q1.s, q2.v, o.v);
        vec3.mulAddOut(o.v, q2.s, q1.v, o.v);
        o.s = s;
        return o;
    },
    /** inv(q1) * q2, written into `o`. */
    invMulOut: (q1: Quat, q2: Quat, o: Quat): Quat => {
        const s = f32(f32(q1.s * q2.s) + vec3.dot(q1.v, q2.v));
        vec3.crossOut(q2.v, q1.v, o.v);
        vec3.mulAddOut(o.v, q1.s, q2.v, o.v);
        vec3.mulSubOut(o.v, q2.s, q1.v, o.v);
        o.s = s;
        return o;
    },

    /** Pseudo angular velocity taking q toward target: 2 * (target - q) * conj(q) (b3DeltaQuatToRotation). */
    deltaToRotation: (q: Quat, target: Quat): Vec3 => {
        let s = q;
        if (quat.dot(q, target) < 0) {
            s = quat.negate(q);
        }
        const diff: Quat = { v: vec3.sub(target.v, s.v), s: f32(target.s - s.s) };
        const product = quat.mul(diff, quat.conjugate(s));
        return vec3.scale(f32(2), product.v);
    },

    normalize: (q: Quat): Quat => {
        const lengthSq = quat.dot(q, q);
        if (lengthSq > f32(1000 * FLT_MIN)) {
            const s = f32(1 / f32(Math.sqrt(lengthSq)));
            return { v: vec3.scale(s, q.v), s: f32(s * q.s) };
        }
        return { v: { x: 0, y: 0, z: 0 }, s: 1 };
    },

    /** Rotate v by q: v + 2 * cross(q.v, cross(q.v, v) + q.s * v). */
    rotate: (q: Quat, v: Vec3): Vec3 => {
        const t1 = vec3.cross(q.v, v);
        const t2 = vec3.mulAdd(t1, q.s, v);
        const t3 = vec3.cross(q.v, t2);
        return vec3.mulAdd(v, 2, t3);
    },

    /** Inverse rotate v by q: v + 2 * cross(q.v, cross(q.v, v) - q.s * v). */
    invRotate: (q: Quat, v: Vec3): Vec3 => {
        const t1 = vec3.cross(q.v, v);
        const t2 = vec3.mulSub(t1, q.s, v);
        const t3 = vec3.cross(q.v, t2);
        return vec3.mulAdd(v, 2, t3);
    },

    /** {@link quat.rotate}, written into `o` (may alias `v`, must not alias `q.v`). */
    rotateOut: (q: Quat, v: Vec3, o: Vec3): Vec3 => {
        vec3.crossOut(q.v, v, rotateScratch);
        vec3.mulAddOut(rotateScratch, q.s, v, rotateScratch);
        vec3.crossOut(q.v, rotateScratch, rotateScratch);
        return vec3.mulAddOut(v, 2, rotateScratch, o);
    },

    /** {@link quat.invRotate}, written into `o` (may alias `v`, must not alias `q.v`). */
    invRotateOut: (q: Quat, v: Vec3, o: Vec3): Vec3 => {
        vec3.crossOut(q.v, v, rotateScratch);
        vec3.mulSubOut(rotateScratch, q.s, v, rotateScratch);
        vec3.crossOut(q.v, rotateScratch, rotateScratch);
        return vec3.mulAddOut(v, 2, rotateScratch, o);
    },

    /** Integrate a rotation by a small angular delta and normalize (b3IntegrateRotation). */
    integrateRotation: (q1: Quat, deltaRotation: Vec3): Quat => {
        const qd = quat.mul({ v: vec3.scale(f32(0.5), deltaRotation), s: 0 }, q1);
        const q2 = { v: vec3.add(q1.v, qd.v), s: f32(qd.s + q1.s) };
        return quat.normalize(q2);
    },

    fromAxisAngle: (axis: Vec3, radians: number): Quat => {
        const cs = computeCosSin(f32(0.5 * radians));
        return {
            v: { x: f32(cs.sine * axis.x), y: f32(cs.sine * axis.y), z: f32(cs.sine * axis.z) },
            s: cs.cosine,
        };
    },

    /** Axis and angle (radians) of the quaternion. Assumes it is normalized. */
    getAxisAngle: (q: Quat): { axis: Vec3; radians: number } => {
        const length = f32(Math.sqrt(vec3.lengthSq(q.v)));
        const radians = f32(2 * atan2(length, q.s));
        if (length > 0) {
            const inv = f32(1 / length);
            return { axis: vec3.scale(inv, q.v), radians };
        }
        return { axis: { x: 0, y: 0, z: 0 }, radians };
    },

    getAngle: (q: Quat): number => {
        const length = f32(Math.sqrt(vec3.lengthSq(q.v)));
        return f32(2 * atan2(length, q.s));
    },

    /** Twist angle around the z-axis (revolute angle / twist limit). */
    getTwistAngle: (q: Quat): number => {
        const twist = q.s < 0 ? atan2(-q.v.z, -q.s) : atan2(q.v.z, q.s);
        return f32(twist * 2);
    },

    /** Swing angle (cone limit). */
    getSwingAngle: (q: Quat): number => {
        const x = f32(Math.sqrt(f32(f32(q.v.z * q.v.z) + f32(q.s * q.s))));
        const y = f32(Math.sqrt(f32(f32(q.v.x * q.v.x) + f32(q.v.y * q.v.y))));
        return f32(2 * atan2(y, x));
    },

    /** Interpolate and normalize between two quaternions (b3NLerp). */
    nlerp: (a: Quat, b: Quat, alpha: number): Quat => {
        let q1 = a;
        if (quat.dot(q1, b) < 0) q1 = quat.negate(q1);
        const v = vec3.lerp(q1.v, b.v, alpha);
        const s = f32(f32(f32(1 - alpha) * q1.s) + f32(alpha * b.s));
        return quat.normalize({ v, s });
    },

    isNormalized: (q: Quat): boolean => {
        const xx = f32(q.v.x * q.v.x);
        const yy = f32(q.v.y * q.v.y);
        const zz = f32(q.v.z * q.v.z);
        const ss = f32(q.s * q.s);
        const qq = f32(f32(f32(xx + yy) + zz) + ss);
        return f32(1 - f32(20 * FLT_EPSILON)) < qq && qq < f32(1 + f32(20 * FLT_EPSILON));
    },

    isValid: (q: Quat): boolean =>
        isValidFloat(q.v.x) &&
        isValidFloat(q.v.y) &&
        isValidFloat(q.v.z) &&
        isValidFloat(q.s) &&
        quat.isNormalized(q),
};

/** Find a quaternion that rotates unit vector v1 to unit vector v2 (b3ComputeQuatBetweenUnitVectors). */
export function computeQuatBetweenUnitVectors(v1: Vec3, v2: Vec3): Quat {
    let out: Quat;
    const m = vec3.lerp(v1, v2, 0.5);
    const tolerance = f32(100 * FLT_EPSILON);
    if (vec3.lengthSq(m) > f32(tolerance * tolerance)) {
        out = { v: vec3.cross(v1, m), s: vec3.dot(v1, m) };
    } else if (absf(v1.x) > 0.5) {
        out = { v: { x: v1.y, y: -v1.x, z: 0 }, s: 0 };
    } else {
        out = { v: { x: 0, y: v1.z, z: -v1.y }, s: 0 };
    }
    return quat.normalize(out);
}

/** Extract a quaternion from a rotation matrix (b3MakeQuatFromMatrix). */
export function makeQuatFromMatrix(m: Mat3): Quat {
    const c1 = m.cx;
    const c2 = m.cy;
    const c3 = m.cz;
    const q: Quat = { v: { x: 0, y: 0, z: 0 }, s: 0 };

    const trace = f32(f32(m.cx.x + m.cy.y) + m.cz.z);
    if (trace >= 0) {
        q.v.x = f32(c2.z - c3.y);
        q.v.y = f32(c3.x - c1.z);
        q.v.z = f32(c1.y - c2.x);
        q.s = f32(trace + 1);
    } else if (c1.x > c2.y && c1.x > c3.z) {
        q.v.x = f32(f32(f32(c1.x - c2.y) - c3.z) + 1);
        q.v.y = f32(c2.x + c1.y);
        q.v.z = f32(c3.x + c1.z);
        q.s = f32(c2.z - c3.y);
    } else if (c2.y > c3.z) {
        q.v.x = f32(c1.y + c2.x);
        q.v.y = f32(f32(f32(c2.y - c3.z) - c1.x) + 1);
        q.v.z = f32(c3.y + c2.z);
        q.s = f32(c3.x - c1.z);
    } else {
        q.v.x = f32(c1.z + c3.x);
        q.v.y = f32(c2.z + c3.y);
        q.v.z = f32(f32(f32(c3.z - c1.x) - c2.y) + 1);
        q.s = f32(c1.y - c2.x);
    }
    return quat.normalize(q);
}

// --- mat3 -----------------------------------------------------------------------------------

export const mat3 = {
    zero: (): Mat3 => ({
        cx: { x: 0, y: 0, z: 0 },
        cy: { x: 0, y: 0, z: 0 },
        cz: { x: 0, y: 0, z: 0 },
    }),
    identity: (): Mat3 => ({
        cx: { x: 1, y: 0, z: 0 },
        cy: { x: 0, y: 1, z: 0 },
        cz: { x: 0, y: 0, z: 1 },
    }),

    diagonal: (a: number, b: number, c: number): Mat3 => ({
        cx: { x: a, y: 0, z: 0 },
        cy: { x: 0, y: b, z: 0 },
        cz: { x: 0, y: 0, z: c },
    }),

    /** m * a, matrix times column vector. */
    mulV: (m: Mat3, a: Vec3): Vec3 => ({
        x: f32(f32(f32(m.cx.x * a.x) + f32(m.cy.x * a.y)) + f32(m.cz.x * a.z)),
        y: f32(f32(f32(m.cx.y * a.x) + f32(m.cy.y * a.y)) + f32(m.cz.y * a.z)),
        z: f32(f32(f32(m.cx.z * a.x) + f32(m.cy.z * a.y)) + f32(m.cz.z * a.z)),
    }),

    /** m * a, written into `o` (may alias `a` — each component is read before written). */
    mulVOut: (m: Mat3, a: Vec3, o: Vec3): Vec3 => {
        const x = f32(f32(f32(m.cx.x * a.x) + f32(m.cy.x * a.y)) + f32(m.cz.x * a.z));
        const y = f32(f32(f32(m.cx.y * a.x) + f32(m.cy.y * a.y)) + f32(m.cz.y * a.z));
        const z = f32(f32(f32(m.cx.z * a.x) + f32(m.cy.z * a.y)) + f32(m.cz.z * a.z));
        o.x = x;
        o.y = y;
        o.z = z;
        return o;
    },

    /** a * b */
    mul: (a: Mat3, b: Mat3): Mat3 => ({
        cx: mat3.mulV(a, b.cx),
        cy: mat3.mulV(a, b.cy),
        cz: mat3.mulV(a, b.cz),
    }),

    /** a + b, component-wise (b3AddMM). */
    add: (a: Mat3, b: Mat3): Mat3 => ({
        cx: vec3.add(a.cx, b.cx),
        cy: vec3.add(a.cy, b.cy),
        cz: vec3.add(a.cz, b.cz),
    }),
    /** a - b, component-wise (b3SubMM). */
    sub: (a: Mat3, b: Mat3): Mat3 => ({
        cx: vec3.sub(a.cx, b.cx),
        cy: vec3.sub(a.cy, b.cy),
        cz: vec3.sub(a.cz, b.cz),
    }),
    /** s * m, component-wise (b3MulSM). */
    scale: (s: number, m: Mat3): Mat3 => ({
        cx: vec3.scale(s, m.cx),
        cy: vec3.scale(s, m.cy),
        cz: vec3.scale(s, m.cz),
    }),

    transpose: (m: Mat3): Mat3 => ({
        cx: { x: m.cx.x, y: m.cy.x, z: m.cz.x },
        cy: { x: m.cx.y, y: m.cy.y, z: m.cz.y },
        cz: { x: m.cx.z, y: m.cy.z, z: m.cz.z },
    }),

    /** {@link mat3.transpose}, written into `o`. `o` must not alias `m`. */
    transposeOut: (m: Mat3, o: Mat3): Mat3 => {
        o.cx.x = m.cx.x;
        o.cx.y = m.cy.x;
        o.cx.z = m.cz.x;
        o.cy.x = m.cx.y;
        o.cy.y = m.cy.y;
        o.cy.z = m.cz.y;
        o.cz.x = m.cx.z;
        o.cz.y = m.cy.z;
        o.cz.z = m.cz.z;
        return o;
    },

    /** Component-wise |m|, written into `o` (may alias `m`). */
    absOut: (m: Mat3, o: Mat3): Mat3 => {
        vec3.absOut(m.cx, o.cx);
        vec3.absOut(m.cy, o.cy);
        vec3.absOut(m.cz, o.cz);
        return o;
    },

    /** a * b, written into `o`. `o` must not alias `a` or `b`. */
    mulOut: (a: Mat3, b: Mat3, o: Mat3): Mat3 => {
        mat3.mulVOut(a, b.cx, o.cx);
        mat3.mulVOut(a, b.cy, o.cy);
        mat3.mulVOut(a, b.cz, o.cz);
        return o;
    },

    /** Component-wise negation (b3NegateMat3). */
    neg: (m: Mat3): Mat3 => ({ cx: vec3.neg(m.cx), cy: vec3.neg(m.cy), cz: vec3.neg(m.cz) }),

    /** Skew-symmetric cross-product matrix of `v`: skew(v) * a == cross(v, a) (b3Skew). */
    skew: (v: Vec3): Mat3 => ({
        cx: { x: 0, y: v.z, z: -v.y },
        cy: { x: -v.z, y: 0, z: v.x },
        cz: { x: v.y, y: -v.x, z: 0 },
    }),

    det: (m: Mat3): number => vec3.dot(m.cx, vec3.cross(m.cy, m.cz)),

    abs: (m: Mat3): Mat3 => ({ cx: vec3.abs(m.cx), cy: vec3.abs(m.cy), cz: vec3.abs(m.cz) }),

    invert: (m: Mat3): Mat3 => {
        const det = mat3.det(m);
        if (absf(det) > f32(1000 * FLT_MIN)) {
            const invDet = f32(1 / det);
            const out: Mat3 = {
                cx: vec3.scale(invDet, vec3.cross(m.cy, m.cz)),
                cy: vec3.scale(invDet, vec3.cross(m.cz, m.cx)),
                cz: vec3.scale(invDet, vec3.cross(m.cx, m.cy)),
            };
            return mat3.transpose(out);
        }
        return mat3.zero();
    },

    /**
     * Inverse transpose (b3InvertT): the cofactor columns scaled by 1/det, without the final
     * transpose `invert` applies. For a symmetric matrix (e.g. an inertia tensor) this equals the
     * inverse; the distinct element layout is why the mass path uses it, not `invert`.
     */
    invertT: (m: Mat3): Mat3 => {
        const det = mat3.det(m);
        if (absf(det) > f32(1000 * FLT_MIN)) {
            const invDet = f32(1 / det);
            return {
                cx: vec3.scale(invDet, vec3.cross(m.cy, m.cz)),
                cy: vec3.scale(invDet, vec3.cross(m.cz, m.cx)),
                cz: vec3.scale(invDet, vec3.cross(m.cx, m.cy)),
            };
        }
        return mat3.zero();
    },

    /** inv(m) * a, via Cramer's rule (b3Solve3). */
    solve: (m: Mat3, a: Vec3): Vec3 => {
        const det = mat3.det(m);
        if (absf(det) > f32(1000 * FLT_MIN)) {
            const invDet = f32(1 / det);
            const sx = vec3.cross(m.cy, m.cz);
            const sy = vec3.cross(m.cz, m.cx);
            const sz = vec3.cross(m.cx, m.cy);
            return {
                x: f32(invDet * vec3.dot(sx, a)),
                y: f32(invDet * vec3.dot(sy, a)),
                z: f32(invDet * vec3.dot(sz, a)),
            };
        }
        return { x: 0, y: 0, z: 0 };
    },

    fromQuat: (q: Quat): Mat3 => {
        const xx = f32(q.v.x * q.v.x);
        const yy = f32(q.v.y * q.v.y);
        const zz = f32(q.v.z * q.v.z);
        const xy = f32(q.v.x * q.v.y);
        const xz = f32(q.v.x * q.v.z);
        const xw = f32(q.v.x * q.s);
        const yz = f32(q.v.y * q.v.z);
        const yw = f32(q.v.y * q.s);
        const zw = f32(q.v.z * q.s);
        return {
            cx: {
                x: f32(1 - f32(2 * f32(yy + zz))),
                y: f32(2 * f32(xy + zw)),
                z: f32(2 * f32(xz - yw)),
            },
            cy: {
                x: f32(2 * f32(xy - zw)),
                y: f32(1 - f32(2 * f32(xx + zz))),
                z: f32(2 * f32(yz + xw)),
            },
            cz: {
                x: f32(2 * f32(xz + yw)),
                y: f32(2 * f32(yz - xw)),
                z: f32(1 - f32(2 * f32(xx + yy))),
            },
        };
    },

    /** {@link mat3.fromQuat} written into `o` — identical expression tree, no allocation. */
    fromQuatOut: (q: Quat, o: Mat3): Mat3 => {
        const xx = f32(q.v.x * q.v.x);
        const yy = f32(q.v.y * q.v.y);
        const zz = f32(q.v.z * q.v.z);
        const xy = f32(q.v.x * q.v.y);
        const xz = f32(q.v.x * q.v.z);
        const xw = f32(q.v.x * q.s);
        const yz = f32(q.v.y * q.v.z);
        const yw = f32(q.v.y * q.s);
        const zw = f32(q.v.z * q.s);
        o.cx.x = f32(1 - f32(2 * f32(yy + zz)));
        o.cx.y = f32(2 * f32(xy + zw));
        o.cx.z = f32(2 * f32(xz - yw));
        o.cy.x = f32(2 * f32(xy - zw));
        o.cy.y = f32(1 - f32(2 * f32(xx + zz)));
        o.cy.z = f32(2 * f32(yz + xw));
        o.cz.x = f32(2 * f32(xz + yw));
        o.cz.y = f32(2 * f32(yz - xw));
        o.cz.z = f32(1 - f32(2 * f32(xx + yy)));
        return o;
    },

    isValid: (m: Mat3): boolean => vec3.isValid(m.cx) && vec3.isValid(m.cy) && vec3.isValid(m.cz),
};

// --- inertia --------------------------------------------------------------------------------

/** Solid-sphere inertia tensor about the center (b3SphereInertia). */
export function sphereInertia(mass: number, radius: number): Mat3 {
    // 0.4f is not exactly representable; fround the literal so the product matches C's f32 0.4f.
    const i = f32(f32(f32(f32(0.4) * mass) * radius) * radius);
    return mat3.diagonal(i, i, i);
}

/** Solid-cylinder inertia tensor, axis along y (b3CylinderInertia). */
export function cylinderInertia(mass: number, radius: number, height: number): Mat3 {
    const rr = f32(f32(3 * radius) * radius);
    const hh = f32(height * height);
    const ixx = f32(f32(mass * f32(rr + hh)) / 12);
    const iyy = f32(f32(f32(0.5 * mass) * radius) * radius);
    return mat3.diagonal(ixx, iyy, ixx);
}

/** Solid-box inertia tensor about the center, given its min/max corners (b3BoxInertia). */
export function boxInertia(mass: number, min: Vec3, max: Vec3): Mat3 {
    const d = vec3.sub(max, min);
    const dx = f32(d.x * d.x);
    const dy = f32(d.y * d.y);
    const dz = f32(d.z * d.z);
    const ixx = f32(f32(mass * f32(dy + dz)) / 12);
    const iyy = f32(f32(mass * f32(dx + dz)) / 12);
    const izz = f32(f32(mass * f32(dx + dy)) / 12);
    return mat3.diagonal(ixx, iyy, izz);
}

/** Parallel-axis inertia offset for a mass displaced to `origin` (b3Steiner). */
export function steiner(mass: number, origin: Vec3): Mat3 {
    const ox2 = f32(origin.x * origin.x);
    const oy2 = f32(origin.y * origin.y);
    const oz2 = f32(origin.z * origin.z);
    const ixx = f32(mass * f32(oy2 + oz2));
    const iyy = f32(mass * f32(ox2 + oz2));
    const izz = f32(mass * f32(ox2 + oy2));
    const nm = f32(-mass);
    const ixy = f32(f32(nm * origin.x) * origin.y);
    const ixz = f32(f32(nm * origin.x) * origin.z);
    const iyz = f32(f32(nm * origin.y) * origin.z);
    return {
        cx: { x: ixx, y: ixy, z: ixz },
        cy: { x: ixy, y: iyy, z: iyz },
        cz: { x: ixz, y: iyz, z: izz },
    };
}

/** Rotate a central inertia tensor by q: R * I * Rᵀ (b3RotateInertia). */
export function rotateInertia(q: Quat, centralInertia: Mat3): Mat3 {
    const r = mat3.fromQuat(q);
    return mat3.mul(r, mat3.mul(centralInertia, mat3.transpose(r)));
}

// --- plane ----------------------------------------------------------------------------------

export const plane = {
    /** Plane through `point` with the given normal (b3MakePlaneFromNormalAndPoint). */
    fromNormalAndPoint: (normal: Vec3, point: Vec3): Plane => ({
        normal,
        offset: vec3.dot(normal, point),
    }),

    /** Plane through three points, normal = normalize((p2-p1) × (p3-p1)) (b3MakePlaneFromPoints). */
    fromPoints: (p1: Vec3, p2: Vec3, p3: Vec3): Plane => {
        const normal = vec3.normalize(vec3.cross(vec3.sub(p2, p1), vec3.sub(p3, p1)));
        return { normal, offset: vec3.dot(normal, p1) };
    },

    /** Signed separation of `point` from the plane (b3PlaneSeparation). */
    separation: (pl: Plane, point: Vec3): number => f32(vec3.dot(pl.normal, point) - pl.offset),

    /** Transform a plane by a rigid transform (b3TransformPlane). */
    transform: (t: Transform, pl: Plane): Plane => {
        const normal = quat.rotate(t.q, pl.normal);
        return { normal, offset: f32(pl.offset + vec3.dot(normal, t.p)) };
    },
};

// --- mat2 -----------------------------------------------------------------------------------

export const mat2 = {
    mulV: (m: Mat2, a: Vec2): Vec2 => ({
        x: f32(f32(m.cx.x * a.x) + f32(m.cy.x * a.y)),
        y: f32(f32(m.cx.y * a.x) + f32(m.cy.y * a.y)),
    }),
    mul: (a: Mat2, b: Mat2): Mat2 => ({ cx: mat2.mulV(a, b.cx), cy: mat2.mulV(a, b.cy) }),
    det: (m: Mat2): number => f32(f32(m.cx.x * m.cy.y) - f32(m.cx.y * m.cy.x)),

    invert: (m: Mat2): Mat2 => {
        const det = mat2.det(m);
        if (absf(det) > f32(1000 * FLT_MIN)) {
            const invDet = f32(1 / det);
            return {
                cx: { x: f32(invDet * m.cy.y), y: f32(-invDet * m.cx.y) },
                cy: { x: f32(-invDet * m.cy.x), y: f32(invDet * m.cx.x) },
            };
        }
        return { cx: { x: 0, y: 0 }, cy: { x: 0, y: 0 } };
    },

    /** inv(m) * b, assuming m is positive semi-definite (b3Solve2). */
    solve: (m: Mat2, b: Vec2): Vec2 => {
        const det = mat2.det(m);
        if (det > f32(1000 * FLT_MIN)) {
            const invDet = f32(1 / det);
            return {
                x: f32(f32(f32(invDet * m.cy.y) * b.x) - f32(f32(invDet * m.cy.x) * b.y)),
                y: f32(f32(-f32(invDet * m.cx.y) * b.x) + f32(f32(invDet * m.cx.x) * b.y)),
            };
        }
        return { x: 0, y: 0 };
    },
};

// --- transform ------------------------------------------------------------------------------

export const xf = {
    identity: (): Transform => ({ p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } }),

    /** Apply transform to a point: q rotates, then translate by p. */
    point: (t: Transform, v: Vec3): Vec3 => vec3.add(quat.rotate(t.q, v), t.p),
    /** Inverse of `point`. */
    invPoint: (t: Transform, v: Vec3): Vec3 => quat.invRotate(t.q, vec3.sub(v, t.p)),

    /** a * b — b's local frame composed into a's. */
    mul: (a: Transform, b: Transform): Transform => ({
        p: vec3.add(quat.rotate(a.q, b.p), a.p),
        q: quat.mul(a.q, b.q),
    }),

    /** inv(a) * b — b expressed in a's local frame. */
    invMul: (a: Transform, b: Transform): Transform => ({
        p: quat.invRotate(a.q, vec3.sub(b.p, a.p)),
        q: quat.invMul(a.q, b.q),
    }),

    /** {@link xf.invMul}, written into `o`. `o` must not alias `a` or `b`. */
    invMulOut: (a: Transform, b: Transform, o: Transform): Transform => {
        vec3.subOut(b.p, a.p, o.p);
        quat.invRotateOut(a.q, o.p, o.p);
        quat.invMulOut(a.q, b.q, o.q);
        return o;
    },

    invert: (t: Transform): Transform => ({
        p: quat.invRotate(t.q, vec3.neg(t.p)),
        q: quat.conjugate(t.q),
    }),

    isValid: (t: Transform): boolean => vec3.isValid(t.p) && quat.isValid(t.q),
};

// World-position boundary. In float mode Pos ≡ Vec3 and WorldTransform ≡ Transform, so these
// collapse to their pure-float counterparts; they exist so callers can name the boundary and so
// a later large-world (double) build has a single set of seams to widen.
export const toPos = (v: Vec3): Pos => ({ x: v.x, y: v.y, z: v.z });
export const toVec3 = (p: Pos): Vec3 => ({ x: p.x, y: p.y, z: p.z });
export const subPos = (a: Pos, b: Pos): Vec3 => vec3.sub(a, b);
export const offsetPos = (p: Pos, d: Vec3): Pos => vec3.add(p, d);
export const isValidPosition = (p: Pos): boolean =>
    isValidFloat(p.x) && isValidFloat(p.y) && isValidFloat(p.z);
export const makeWorldTransform = (t: Transform): WorldTransform => ({ p: toPos(t.p), q: t.q });
/** Shift a world transform into the frame of a base position (b3ToRelativeTransform). */
export const toRelativeTransform = (t: WorldTransform, base: Pos): Transform => ({
    q: t.q,
    p: subPos(t.p, base),
});
export const isValidWorldTransform = (t: WorldTransform): boolean =>
    isValidPosition(t.p) && quat.isValid(t.q);
export const transformWorldPoint = (t: WorldTransform, p: Vec3): Pos => xf.point(t, p);
export const invTransformWorldPoint = (t: WorldTransform, p: Pos): Vec3 => xf.invPoint(t, p);
export const invMulWorldTransforms = (a: WorldTransform, b: WorldTransform): Transform =>
    xf.invMul(a, b);
export const invMulWorldTransformsOut = (
    a: WorldTransform,
    b: WorldTransform,
    o: Transform,
): Transform => xf.invMulOut(a, b, o);
export const mulWorldTransforms = (a: WorldTransform, b: Transform): WorldTransform => xf.mul(a, b);
export const isDoublePrecision = (): boolean => false;

// --- aabb -----------------------------------------------------------------------------------

// Working registers for aabb.transformOut; never live across calls.
const aabbCenter: Vec3 = { x: 0, y: 0, z: 0 };
const aabbExtent: Vec3 = { x: 0, y: 0, z: 0 };
const aabbMat: Mat3 = {
    cx: { x: 0, y: 0, z: 0 },
    cy: { x: 0, y: 0, z: 0 },
    cz: { x: 0, y: 0, z: 0 },
};

export const aabb = {
    contains: (a: AABB, b: AABB): boolean => {
        if (a.lowerBound.x > b.lowerBound.x || b.upperBound.x > a.upperBound.x) return false;
        if (a.lowerBound.y > b.lowerBound.y || b.upperBound.y > a.upperBound.y) return false;
        if (a.lowerBound.z > b.lowerBound.z || b.upperBound.z > a.upperBound.z) return false;
        return true;
    },

    area: (a: AABB): number => {
        const d = vec3.sub(a.upperBound, a.lowerBound);
        const xy = f32(d.x * d.y);
        const yz = f32(d.y * d.z);
        const zx = f32(d.z * d.x);
        return f32(2 * f32(f32(xy + yz) + zx));
    },

    /** Surface area used by the tree's SAH cost (b3Perimeter). Term/sum order differs from `area`. */
    perimeter: (a: AABB): number => {
        const wx = f32(a.upperBound.x - a.lowerBound.x);
        const wy = f32(a.upperBound.y - a.lowerBound.y);
        const wz = f32(a.upperBound.z - a.lowerBound.z);
        return f32(2 * f32(f32(f32(wx * wz) + f32(wy * wx)) + f32(wz * wy)));
    },

    /** Grow `a` in place to contain `b`; returns true if it changed (b3EnlargeAABB). */
    enlarge: (a: AABB, b: AABB): boolean => {
        let changed = false;
        if (b.lowerBound.x < a.lowerBound.x) {
            a.lowerBound.x = b.lowerBound.x;
            changed = true;
        }
        if (b.lowerBound.y < a.lowerBound.y) {
            a.lowerBound.y = b.lowerBound.y;
            changed = true;
        }
        if (b.lowerBound.z < a.lowerBound.z) {
            a.lowerBound.z = b.lowerBound.z;
            changed = true;
        }
        if (a.upperBound.x < b.upperBound.x) {
            a.upperBound.x = b.upperBound.x;
            changed = true;
        }
        if (a.upperBound.y < b.upperBound.y) {
            a.upperBound.y = b.upperBound.y;
            changed = true;
        }
        if (a.upperBound.z < b.upperBound.z) {
            a.upperBound.z = b.upperBound.z;
            changed = true;
        }
        return changed;
    },

    clone: (a: AABB): AABB => ({
        lowerBound: { x: a.lowerBound.x, y: a.lowerBound.y, z: a.lowerBound.z },
        upperBound: { x: a.upperBound.x, y: a.upperBound.y, z: a.upperBound.z },
    }),

    center: (a: AABB): Vec3 => vec3.scale(0.5, vec3.add(a.upperBound, a.lowerBound)),
    extents: (a: AABB): Vec3 => vec3.scale(0.5, vec3.sub(a.upperBound, a.lowerBound)),

    union: (a: AABB, b: AABB): AABB => ({
        lowerBound: vec3.min(a.lowerBound, b.lowerBound),
        upperBound: vec3.max(a.upperBound, b.upperBound),
    }),

    /** Transform an AABB, yielding an enclosing (possibly larger) AABB (b3AABB_Transform). */
    transform: (t: Transform, a: AABB): AABB => {
        const center = xf.point(t, aabb.center(a));
        const m = mat3.fromQuat(t.q);
        const extent = mat3.mulV(mat3.abs(m), aabb.extents(a));
        return { lowerBound: vec3.sub(center, extent), upperBound: vec3.add(center, extent) };
    },

    /** {@link aabb.transform}, written into `o` — identical expression tree, no allocation. `o` may
     * alias `a` (both bounds are consumed before `o` is written). */
    transformOut: (t: Transform, a: AABB, o: AABB): AABB => {
        vec3.addOut(a.upperBound, a.lowerBound, aabbCenter);
        vec3.scaleOut(0.5, aabbCenter, aabbCenter);
        quat.rotateOut(t.q, aabbCenter, aabbCenter);
        vec3.addOut(aabbCenter, t.p, aabbCenter);
        vec3.subOut(a.upperBound, a.lowerBound, aabbExtent);
        vec3.scaleOut(0.5, aabbExtent, aabbExtent);
        mat3.fromQuatOut(t.q, aabbMat);
        mat3.absOut(aabbMat, aabbMat);
        mat3.mulVOut(aabbMat, aabbExtent, aabbExtent);
        vec3.subOut(aabbCenter, aabbExtent, o.lowerBound);
        vec3.addOut(aabbCenter, aabbExtent, o.upperBound);
        return o;
    },

    inflate: (a: AABB, extension: number): AABB => {
        const r: Vec3 = { x: extension, y: extension, z: extension };
        return { lowerBound: vec3.sub(a.lowerBound, r), upperBound: vec3.add(a.upperBound, r) };
    },

    /**
     * Translate a local AABB into world space (b3OffsetAABB). The C rounds the bounds outward to
     * contain the f64 box, but that round is identity in the single-precision build the port targets
     * (b3RoundDownFloat/b3RoundUpFloat both reduce to `(float)x`), so it's a plain component add.
     */
    offset: (a: AABB, origin: Vec3): AABB => ({
        lowerBound: vec3.add(origin, a.lowerBound),
        upperBound: vec3.add(origin, a.upperBound),
    }),

    overlaps: (a: AABB, b: AABB): boolean => {
        if (a.upperBound.x < b.lowerBound.x || a.lowerBound.x > b.upperBound.x) return false;
        if (a.upperBound.y < b.lowerBound.y || a.lowerBound.y > b.upperBound.y) return false;
        if (a.upperBound.z < b.lowerBound.z || a.lowerBound.z > b.upperBound.z) return false;
        return true;
    },

    /** AABB enclosing a point cloud, grown by `radius` on every side (b3MakeAABB). */
    make: (points: Vec3[], count: number, radius: number): AABB => {
        let lower = points[0];
        let upper = points[0];
        for (let i = 1; i < count; ++i) {
            lower = vec3.min(lower, points[i]);
            upper = vec3.max(upper, points[i]);
        }
        const r = { x: radius, y: radius, z: radius };
        return { lowerBound: vec3.sub(lower, r), upperBound: vec3.add(upper, r) };
    },

    isValid: (a: AABB): boolean => {
        if (!vec3.isValid(a.lowerBound) || !vec3.isValid(a.upperBound)) return false;
        return (
            a.lowerBound.x <= a.upperBound.x &&
            a.lowerBound.y <= a.upperBound.y &&
            a.lowerBound.z <= a.upperBound.z
        );
    },
};

/**
 * Test a ray (start + delta) against an AABB via SAT edge separation (b3TestBoundsRayOverlap,
 * Gino p80). The scalar path; `Math.min/max`-free.
 */
export function testBoundsRayOverlap(
    nodeMin: Vec3,
    nodeMax: Vec3,
    rayStart: Vec3,
    rayDelta: Vec3,
): boolean {
    const nodeCenter = vec3.scale(0.5, vec3.add(nodeMin, nodeMax));
    const nodeExtent = vec3.sub(nodeMax, nodeCenter);
    const rs = vec3.sub(rayStart, nodeCenter);

    const absCross = vec3.abs(vec3.cross(rayDelta, rs));
    const absDelta = vec3.abs(rayDelta);
    // b3ModifiedCrossV(absDelta, nodeExtent): the cross with all subtractions replaced by additions.
    const modCross: Vec3 = {
        x: f32(f32(absDelta.y * nodeExtent.z) + f32(absDelta.z * nodeExtent.y)),
        y: f32(f32(absDelta.z * nodeExtent.x) + f32(absDelta.x * nodeExtent.z)),
        z: f32(f32(absDelta.x * nodeExtent.y) + f32(absDelta.y * nodeExtent.x)),
    };
    const sep = vec3.sub(absCross, modCross);
    return sep.x <= 0 && sep.y <= 0 && sep.z <= 0;
}

/**
 * Ray vs triangle, returning the hit fraction in [0,1] or 1.0 for a miss (b3IntersectRayTriangle,
 * scalar path). Three per-edge signed-volume tests cull hits outside the triangle and back-facing
 * rays before the plane intersection. The C computes the volumes via a SIMD transpose; scalar-wise
 * each is a plain dot(edgeNormal, rayDelta), so this is `Math.min`-free.
 */
export function intersectRayTriangle(
    rayStart: Vec3,
    rayDelta: Vec3,
    vertex1: Vec3,
    vertex2: Vec3,
    vertex3: Vec3,
): number {
    const edge1 = vec3.sub(vertex3, vertex2);
    const edge2 = vec3.sub(vertex1, vertex3);
    const edge3 = vec3.sub(vertex2, vertex1);

    const midPoint1 = vec3.scale(0.5, vec3.add(vertex2, vertex3));
    const midPoint2 = vec3.scale(0.5, vec3.add(vertex3, vertex1));
    const midPoint3 = vec3.scale(0.5, vec3.add(vertex1, vertex2));

    const normal1 = vec3.cross(edge1, vec3.sub(midPoint1, rayStart));
    const normal2 = vec3.cross(edge2, vec3.sub(midPoint2, rayStart));
    const normal3 = vec3.cross(edge3, vec3.sub(midPoint3, rayStart));

    const volume1 = vec3.dot(normal1, rayDelta);
    const volume2 = vec3.dot(normal2, rayDelta);
    const volume3 = vec3.dot(normal3, rayDelta);
    if (volume1 < 0 || volume2 < 0 || volume3 < 0) return 1;

    const e1 = vec3.sub(vertex2, vertex1);
    const e2 = vec3.sub(vertex3, vertex1);
    const normal = vec3.cross(e1, e2);

    const denominator = vec3.dot(normal, rayDelta);
    if (denominator >= 0) return 1;

    const lambda = f32(vec3.dot(normal, vec3.sub(vertex1, rayStart)) / denominator);
    if (lambda <= 0) return 1;

    return lambda < 1 ? lambda : 1;
}

/**
 * Slab-method ray (segment p1→p2) vs AABB (b3RayCastAABB). Returns whether the segment hits the box
 * and the entry/exit fractions clamped to [0,1]. A near-zero-length ray hits only if p1 is inside.
 * `Math.min/max`-free (scalar b3MinFloat/b3MaxFloat branches).
 */
export function rayCastAABB(
    a: AABB,
    p1: Vec3,
    p2: Vec3,
): { hit: boolean; minFraction: number; maxFraction: number } {
    const d = vec3.sub(p2, p1);
    const rayLength = vec3.length(d);

    if (rayLength < FLT_EPSILON) {
        const inside =
            p1.x >= a.lowerBound.x &&
            p1.x <= a.upperBound.x &&
            p1.y >= a.lowerBound.y &&
            p1.y <= a.upperBound.y &&
            p1.z >= a.lowerBound.z &&
            p1.z <= a.upperBound.z;
        return { hit: inside, minFraction: 0, maxFraction: 0 };
    }

    const rayDir = vec3.scale(f32(1 / rayLength), d);
    const dir = [rayDir.x, rayDir.y, rayDir.z];
    const start = [p1.x, p1.y, p1.z];
    const bmin = [a.lowerBound.x, a.lowerBound.y, a.lowerBound.z];
    const bmax = [a.upperBound.x, a.upperBound.y, a.upperBound.z];

    let tMin = 0;
    let tMax = rayLength;
    for (let i = 0; i < 3; ++i) {
        const comp = dir[i];
        if (absf(comp) < FLT_EPSILON) {
            // Parallel to this slab: miss unless the origin is within it.
            if (start[i] < bmin[i] || start[i] > bmax[i]) {
                return { hit: false, minFraction: 0, maxFraction: 0 };
            }
        } else {
            let t1 = f32(f32(bmin[i] - start[i]) / comp);
            let t2 = f32(f32(bmax[i] - start[i]) / comp);
            if (t1 > t2) {
                const temp = t1;
                t1 = t2;
                t2 = temp;
            }
            tMin = maxf(tMin, t1);
            tMax = minf(tMax, t2);
            if (tMin > tMax) {
                return { hit: false, minFraction: 0, maxFraction: 0 };
            }
        }
    }

    if (tMax < 0) {
        return { hit: false, minFraction: 0, maxFraction: 0 };
    }

    return {
        hit: true,
        minFraction: clampf(f32(tMin / rayLength), 0, 1),
        maxFraction: clampf(f32(tMax / rayLength), 0, 1),
    };
}

// --- constants exposed for callers ----------------------------------------------------------

export { FLT_EPSILON, FLT_MAX, FLT_MIN };
