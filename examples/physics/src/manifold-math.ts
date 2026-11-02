const f = Math.fround;
export const vector = (x = 0, y = 0, z = 0) => ({ x: f(x), y: f(y), z: f(z) });
export type Vec = ReturnType<typeof vector>;
export type ManifoldPose = { p: Vec; q: { v: Vec; s: number } };
export const pose = (x = 0, y = 0, z = 0): ManifoldPose => ({
    p: vector(x, y, z),
    q: { v: vector(), s: 1 },
});
// Box3D's quaternion rotation, with float intermediates.
export function manifoldRotate(q: ManifoldPose["q"], p: Vec): Vec {
    const { x, y, z } = q.v,
        w = q.s;
    const tx = f(f(f(y * p.z) - f(z * p.y)) + f(w * p.x)),
        ty = f(f(f(z * p.x) - f(x * p.z)) + f(w * p.y)),
        tz = f(f(f(x * p.y) - f(y * p.x)) + f(w * p.z));
    return vector(
        f(p.x + f(2 * f(f(y * tz) - f(z * ty)))),
        f(p.y + f(2 * f(f(z * tx) - f(x * tz)))),
        f(p.z + f(2 * f(f(x * ty) - f(y * tx)))),
    );
}
export function manifoldTransform(t: ManifoldPose, p: Vec): Vec {
    const r = manifoldRotate(t.q, p);
    return vector(f(t.p.x + r.x), f(t.p.y + r.y), f(t.p.z + r.z));
}
export function relative(a: ManifoldPose, b: ManifoldPose): ManifoldPose {
    const av = a.q.v,
        bv = b.q.v,
        p = vector(b.p.x - a.p.x, b.p.y - a.p.y, b.p.z - a.p.z),
        w = a.q.s;
    const tx = f(f(f(av.y * p.z) - f(av.z * p.y)) - f(w * p.x)),
        ty = f(f(f(av.z * p.x) - f(av.x * p.z)) - f(w * p.y)),
        tz = f(f(f(av.x * p.y) - f(av.y * p.x)) - f(w * p.z));
    return {
        p: vector(
            f(p.x + f(2 * f(f(av.y * tz) - f(av.z * ty)))),
            f(p.y + f(2 * f(f(av.z * tx) - f(av.x * tz)))),
            f(p.z + f(2 * f(f(av.x * ty) - f(av.y * tx)))),
        ),
        q: {
            v: vector(
                f(f(f(f(bv.y * av.z) - f(bv.z * av.y)) + f(w * bv.x)) - f(b.q.s * av.x)),
                f(f(f(f(bv.z * av.x) - f(bv.x * av.z)) + f(w * bv.y)) - f(b.q.s * av.y)),
                f(f(f(f(bv.x * av.y) - f(bv.y * av.x)) + f(w * bv.z)) - f(b.q.s * av.z)),
            ),
            s: f(f(w * b.q.s) + f(f(f(av.x * bv.x) + f(av.y * bv.y)) + f(av.z * bv.z))),
        },
    };
}
export const add = (a: Vec, b: Vec): Vec => vector(f(a.x + b.x), f(a.y + b.y), f(a.z + b.z));
export const sub = (a: Vec, b: Vec): Vec => vector(f(a.x - b.x), f(a.y - b.y), f(a.z - b.z));
export const neg = (a: Vec): Vec => vector(-a.x, -a.y, -a.z);
export const mul = (s: number, a: Vec): Vec => vector(f(s * a.x), f(s * a.y), f(s * a.z));
export const dot = (a: Vec, b: Vec): number => f(f(f(a.x * b.x) + f(a.y * b.y)) + f(a.z * b.z));
export const cross = (a: Vec, b: Vec): Vec =>
    vector(
        f(f(a.y * b.z) - f(a.z * b.y)),
        f(f(a.z * b.x) - f(a.x * b.z)),
        f(f(a.x * b.y) - f(a.y * b.x)),
    );
export const length = (a: Vec): number => f(Math.sqrt(dot(a, a)));
export const normalize = (a: Vec): Vec => {
    const squared = dot(a, a);
    return squared > f(1000 * 1.1754943508222875e-38)
        ? mul(f(1 / f(Math.sqrt(squared))), a)
        : vector();
};
export function invRotate(q: ManifoldPose["q"], p: Vec): Vec {
    const av = q.v,
        w = q.s;
    const tx = f(f(f(av.y * p.z) - f(av.z * p.y)) - f(w * p.x)),
        ty = f(f(f(av.z * p.x) - f(av.x * p.z)) - f(w * p.y)),
        tz = f(f(f(av.x * p.y) - f(av.y * p.x)) - f(w * p.z));
    return vector(
        f(p.x + f(2 * f(f(av.y * tz) - f(av.z * ty)))),
        f(p.y + f(2 * f(f(av.z * tx) - f(av.x * tz)))),
        f(p.z + f(2 * f(f(av.x * ty) - f(av.y * tx)))),
    );
}
export const invTransform = (t: ManifoldPose, a: Vec): Vec => invRotate(t.q, sub(a, t.p));
