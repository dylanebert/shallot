// Port of reference/avbd-demo3d/source/maths.h — the math layer the AVBD oracle is
// built on. The reference is the executable spec, so this mirrors it operation for
// operation; the only deliberate departure is f64 (JS numbers) where the C++ is f32.
// That makes the oracle the *exact* sequential reference; f32 round-off is the GPU's
// problem to match, not the spec's. Test scaffolding, kept out of the shipped src/.
//
// Layout matches the C++ structs: float3x3 is row-major (`m[row][col]`), quat is
// [x, y, z, w]. The two non-obvious operators carry their C++ overload here as named
// functions: `qsub` (quat − quat → the 2× angular difference vector) and `qadd`
// (quat + ω vector → integrated, normalized quat).

export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number];
export type Mat3 = [Vec3, Vec3, Vec3];

// ── float3 ──────────────────────────────────────────────────────────

export const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const neg = (v: Vec3): Vec3 => [-v[0], -v[1], -v[2]];
export const scale = (v: Vec3, s: number): Vec3 => [v[0] * s, v[1] * s, v[2] * s];
export const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const lengthSq = (v: Vec3): number => dot(v, v);
export const length = (v: Vec3): number => Math.sqrt(lengthSq(v));
export const normalize = (v: Vec3): Vec3 => scale(v, 1 / length(v));

export const cross = (a: Vec3, b: Vec3): Vec3 => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
];

/** 2D vector length — the reference's `length(float2{x, y})` (friction magnitude) */
export const length2d = (x: number, y: number): number => Math.sqrt(x * x + y * y);

// ── scalar helpers ──────────────────────────────────────────────────

export const sign = (x: number): number => (x < 0 ? -1 : x > 0 ? 1 : 0);
export const clamp = (x: number, a: number, b: number): number => Math.max(a, Math.min(b, x));
export const clamp3 = (v: Vec3, a: number, b: number): Vec3 => [
    clamp(v[0], a, b),
    clamp(v[1], a, b),
    clamp(v[2], a, b),
];

// ── float3x3 (row-major) ────────────────────────────────────────────

export const col = (m: Mat3, i: number): Vec3 => [m[0][i], m[1][i], m[2][i]];
export const addM = (a: Mat3, b: Mat3): Mat3 => [add(a[0], b[0]), add(a[1], b[1]), add(a[2], b[2])];
export const negM = (m: Mat3): Mat3 => [neg(m[0]), neg(m[1]), neg(m[2])];
export const scaleM = (m: Mat3, s: number): Mat3 => [
    scale(m[0], s),
    scale(m[1], s),
    scale(m[2], s),
];
export const mulMV = (m: Mat3, v: Vec3): Vec3 => [dot(m[0], v), dot(m[1], v), dot(m[2], v)];

// row·column directly — avoids allocating a `col` temp 9× per multiply (the solver hot path)
export const mulMM = (a: Mat3, b: Mat3): Mat3 => {
    const [a0, a1, a2] = a;
    const [b0, b1, b2] = b;
    return [
        [
            a0[0] * b0[0] + a0[1] * b1[0] + a0[2] * b2[0],
            a0[0] * b0[1] + a0[1] * b1[1] + a0[2] * b2[1],
            a0[0] * b0[2] + a0[1] * b1[2] + a0[2] * b2[2],
        ],
        [
            a1[0] * b0[0] + a1[1] * b1[0] + a1[2] * b2[0],
            a1[0] * b0[1] + a1[1] * b1[1] + a1[2] * b2[1],
            a1[0] * b0[2] + a1[1] * b1[2] + a1[2] * b2[2],
        ],
        [
            a2[0] * b0[0] + a2[1] * b1[0] + a2[2] * b2[0],
            a2[0] * b0[1] + a2[1] * b1[1] + a2[2] * b2[1],
            a2[0] * b0[2] + a2[1] * b1[2] + a2[2] * b2[2],
        ],
    ];
};

export const transpose = (m: Mat3): Mat3 => [
    [m[0][0], m[1][0], m[2][0]],
    [m[0][1], m[1][1], m[2][1]],
    [m[0][2], m[1][2], m[2][2]],
];

export const diagonal = (a: number, b: number, c: number): Mat3 => [
    [a, 0, 0],
    [0, b, 0],
    [0, 0, c],
];

/** outer product `a ⊗ b` → `M[i][j] = a[i]·b[j]` (maths.h `outer`) — the single-row Hessian block a spring/joint Jacobian stamps */
export const outer = (a: Vec3, b: Vec3): Mat3 => [scale(b, a[0]), scale(b, a[1]), scale(b, a[2])];

/** skew-symmetric (cross-product) matrix of `r` so `skew(r)·v = r × v` (maths.h `skew`) — the joint's angular Jacobian */
export const skew = (r: Vec3): Mat3 => [
    [0, -r[2], r[1]],
    [r[2], 0, -r[0]],
    [-r[1], r[0], 0],
];

/** diagonal of the per-column lengths (maths.h `diagonalize`) — the joint's diagonal geometric-stiffness approximation */
export const diagonalize = (m: Mat3): Mat3 =>
    diagonal(length(col(m, 0)), length(col(m, 1)), length(col(m, 2)));

// ── quat ([x, y, z, w]) ─────────────────────────────────────────────

export const qscale = (q: Quat, s: number): Quat => [q[0] * s, q[1] * s, q[2] * s, q[3] * s];
export const qadd4 = (a: Quat, b: Quat): Quat => [
    a[0] + b[0],
    a[1] + b[1],
    a[2] + b[2],
    a[3] + b[3],
];
export const qconj = (q: Quat): Quat => [-q[0], -q[1], -q[2], q[3]];
export const qlenSq = (q: Quat): number => q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
export const qlen = (q: Quat): number => Math.sqrt(qlenSq(q));
export const qnormalize = (q: Quat): Quat => qscale(q, 1 / qlen(q));
export const qinverse = (q: Quat): Quat => qscale(qconj(q), 1 / qlenSq(q));

export const qmul = (a: Quat, b: Quat): Quat => [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
];

/** quat − quat → the 2× angular-difference vector: `(a · b⁻¹).vec * 2` */
export const qsub = (a: Quat, b: Quat): Vec3 => {
    const r = qmul(a, qinverse(b));
    return [r[0] * 2, r[1] * 2, r[2] * 2];
};

/** quat + ω vector → integrated, renormalized quat: `normalize(a + (ω,0)·a·0.5)` */
export const qadd = (a: Quat, b: Vec3): Quat => {
    const d = qscale(qmul([b[0], b[1], b[2], 0], a), 0.5);
    return qnormalize(qadd4(a, d));
};

export const rotate = (q: Quat, v: Vec3): Vec3 => {
    const u: Vec3 = [q[0], q[1], q[2]];
    const t = scale(cross(u, v), 2);
    return add(add(v, scale(t, q[3])), cross(u, t));
};

/** rotate then translate: `rotate(qAng, v) + qLin` */
export const transform = (qLin: Vec3, qAng: Quat, v: Vec3): Vec3 => add(rotate(qAng, v), qLin);

/** orthonormal basis with `normal` as the first row */
export const orthonormal = (normal: Vec3): Mat3 => {
    let t1: Vec3 =
        Math.abs(normal[0]) > Math.abs(normal[1])
            ? [-normal[2], 0, normal[0]]
            : [0, normal[2], -normal[1]];
    t1 = normalize(t1);
    const t2 = cross(t1, normal);
    return [normal, t1, t2];
};

/**
 * 6×6 SPD solve via LDLᵀ, ported verbatim from maths.h `solve()`. The three input
 * matrices are the lower-triangle blocks: `aLin` (linear), `aAng` (angular), `aCross`
 * (cross-coupling). Solves the cross-coupled lin+ang system for `[xLin, xAng]`.
 */
export const solve = (
    aLin: Mat3,
    aAng: Mat3,
    aCross: Mat3,
    bLin: Vec3,
    bAng: Vec3,
): { xLin: Vec3; xAng: Vec3 } => {
    const A11 = aLin[0][0];
    const A21 = aLin[1][0];
    const A22 = aLin[1][1];
    const A31 = aLin[2][0];
    const A32 = aLin[2][1];
    const A33 = aLin[2][2];
    const A41 = aCross[0][0];
    const A42 = aCross[0][1];
    const A43 = aCross[0][2];
    const A44 = aAng[0][0];
    const A51 = aCross[1][0];
    const A52 = aCross[1][1];
    const A53 = aCross[1][2];
    const A54 = aAng[1][0];
    const A55 = aAng[1][1];
    const A61 = aCross[2][0];
    const A62 = aCross[2][1];
    const A63 = aCross[2][2];
    const A64 = aAng[2][0];
    const A65 = aAng[2][1];
    const A66 = aAng[2][2];

    const L21 = A21 / A11;
    const L31 = A31 / A11;
    const L41 = A41 / A11;
    const L51 = A51 / A11;
    const L61 = A61 / A11;
    const D1 = A11;

    const D2 = A22 - L21 * L21 * D1;
    const L32 = (A32 - L21 * L31 * D1) / D2;
    const L42 = (A42 - L21 * L41 * D1) / D2;
    const L52 = (A52 - L21 * L51 * D1) / D2;
    const L62 = (A62 - L21 * L61 * D1) / D2;

    const D3 = A33 - (L31 * L31 * D1 + L32 * L32 * D2);
    const L43 = (A43 - L31 * L41 * D1 - L32 * L42 * D2) / D3;
    const L53 = (A53 - L31 * L51 * D1 - L32 * L52 * D2) / D3;
    const L63 = (A63 - L31 * L61 * D1 - L32 * L62 * D2) / D3;

    const D4 = A44 - (L41 * L41 * D1 + L42 * L42 * D2 + L43 * L43 * D3);
    const L54 = (A54 - L41 * L51 * D1 - L42 * L52 * D2 - L43 * L53 * D3) / D4;
    const L64 = (A64 - L41 * L61 * D1 - L42 * L62 * D2 - L43 * L63 * D3) / D4;

    const D5 = A55 - (L51 * L51 * D1 + L52 * L52 * D2 + L53 * L53 * D3 + L54 * L54 * D4);
    const L65 = (A65 - L51 * L61 * D1 - L52 * L62 * D2 - L53 * L63 * D3 - L54 * L64 * D4) / D5;

    const D6 =
        A66 - (L61 * L61 * D1 + L62 * L62 * D2 + L63 * L63 * D3 + L64 * L64 * D4 + L65 * L65 * D5);

    const y1 = bLin[0];
    const y2 = bLin[1] - L21 * y1;
    const y3 = bLin[2] - L31 * y1 - L32 * y2;
    const y4 = bAng[0] - L41 * y1 - L42 * y2 - L43 * y3;
    const y5 = bAng[1] - L51 * y1 - L52 * y2 - L53 * y3 - L54 * y4;
    const y6 = bAng[2] - L61 * y1 - L62 * y2 - L63 * y3 - L64 * y4 - L65 * y5;

    const z1 = y1 / D1;
    const z2 = y2 / D2;
    const z3 = y3 / D3;
    const z4 = y4 / D4;
    const z5 = y5 / D5;
    const z6 = y6 / D6;

    const xAng: Vec3 = [0, 0, 0];
    const xLin: Vec3 = [0, 0, 0];
    xAng[2] = z6;
    xAng[1] = z5 - L65 * xAng[2];
    xAng[0] = z4 - L54 * xAng[1] - L64 * xAng[2];
    xLin[2] = z3 - L43 * xAng[0] - L53 * xAng[1] - L63 * xAng[2];
    xLin[1] = z2 - L32 * xLin[2] - L42 * xAng[0] - L52 * xAng[1] - L62 * xAng[2];
    xLin[0] = z1 - L21 * xLin[1] - L31 * xLin[2] - L41 * xAng[0] - L51 * xAng[1] - L61 * xAng[2];

    return { xLin, xAng };
};
