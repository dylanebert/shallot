// Port of reference/avbd-demo3d/source/rigid.cpp — the rigid-body state AVBD steps.
// A plain data record, not a class; the solver mutates the pose + cache fields in place
// each step. mass / moment / radius derive from size exactly as `Rigid::Rigid` does;
// `mass <= 0` marks a static / kinematic body (skipped in primal + velocity). f64
// throughout — the oracle is the exact sequential spec, f32 is the GPU's to match. Test
// scaffolding, kept out of the shipped src/.
//
// Phase 6.3 adds rounded shapes (sphere, capsule). The demo3d reference is boxes only, so the
// shape tag + the per-shape moment of inertia + the rounding radius are grounded in the production
// engines that do cover them (Jolt's convex-radius / core-shape idea): a shape is a `size`-defined
// core inflated by `roundRadius`. A sphere's core is a point (`size = 0`), a capsule's a segment
// (`size = (0, 2h, 0)` along Y), a box's the box itself (`roundRadius = 0`). `narrowphase` (rounded.ts)
// reads `shape` to pick the collision routine; the bounding `radius` (broadphase) is shape-aware.

import type { Hull } from "./hull";
import { length, lengthSq, type Quat, scale, type Vec3 } from "./math";

/** collision-shape tag — the narrowphase dispatches on it (rounded.ts). Mirrors the GPU `ShapeKind`. */
export const ShapeKind = { Box: 0, Sphere: 1, Capsule: 2, Hull: 3 } as const;

export interface Body {
    // pose — the solver overwrites these each step
    posLin: Vec3;
    posAng: Quat;
    // per-step caches written in step(): x⁻ (initial), the inertial target
    initialLin: Vec3;
    initialAng: Quat;
    inertialLin: Vec3;
    inertialAng: Quat;
    // velocities (BDF1 recovers them post-solve); prevVelLin feeds the adaptive warmstart
    velLin: Vec3;
    velAng: Vec3;
    prevVelLin: Vec3;
    // shape + constant mass properties
    shape: number; // ShapeKind
    size: Vec3; // box: full widths │ capsule: core segment full length (0, 2h, 0) │ sphere: 0 │ hull: AABB widths
    roundRadius: number; // sphere/capsule rounding radius (0 for box)
    mass: number;
    moment: Vec3;
    friction: number;
    radius: number; // bounding-sphere radius (broadphase) = length(size/2) + roundRadius
    hull?: Hull; // the convex geometry, set only for ShapeKind.Hull (the registry value the GPU packs)
}

/** the solver's static predicate (GPU `solverStatic`): a `mass <= 0` static / kinematic body, skipped
 *  in the primal + velocity passes and the contact / joint all-static dual gates. */
export const solverStatic = (b: Body): boolean => b.mass <= 0;

const clone3 = (v: Vec3): Vec3 => [v[0], v[1], v[2]];
const clone4 = (q: Quat): Quat => [q[0], q[1], q[2], q[3]];

/** box mass from full-width size + density — `sx·sy·sz·density` (rigid.cpp) */
export const massOf = (size: Vec3, density: number): number =>
    size[0] * size[1] * size[2] * density;

/** diagonal moment of a solid box from full-width size + mass (rigid.cpp `Rigid::Rigid`) */
function boxMoment(size: Vec3, mass: number): Vec3 {
    const [sx, sy, sz] = size;
    return [
        ((sy * sy + sz * sz) / 12) * mass,
        ((sx * sx + sz * sz) / 12) * mass,
        ((sx * sx + sy * sy) / 12) * mass,
    ];
}

/** diagonal moment of a solid sphere — `(2/5)·m·r²` on every axis */
function sphereMoment(r: number, mass: number): Vec3 {
    const i = (2 / 5) * mass * r * r;
    return [i, i, i];
}

// diagonal moment of a solid capsule (axis Y): a cylinder (length L = 2h, radius r) capped by two
// hemispheres, the authored mass split by volume. The standard game-physics solid-capsule tensor —
// no AVBD reference covers it. Limits check out: h → 0 gives the sphere (2/5·m·r²), r → 0 the thin
// rod (m·L²/12 perpendicular). The two caps' perpendicular term treats them as a sphere of mass `ms`
// displaced by the cylinder half-length (the L²/4 + 3Lr/8 parallel-axis terms).
function capsuleMoment(h: number, r: number, mass: number): Vec3 {
    const L = 2 * h;
    const vCyl = Math.PI * r * r * L;
    const vSph = (4 / 3) * Math.PI * r * r * r;
    const total = vCyl + vSph;
    const mc = (mass * vCyl) / total;
    const ms = (mass * vSph) / total;
    const iy = mc * 0.5 * r * r + ms * (2 / 5) * r * r;
    const iPerp =
        mc * ((L * L) / 12 + (r * r) / 4) + ms * ((2 / 5) * r * r + (L * L) / 4 + (3 / 8) * L * r);
    return [iPerp, iy, iPerp];
}

function makeBody(
    shape: number,
    size: Vec3,
    roundRadius: number,
    mass: number,
    moment: Vec3,
    friction: number,
    pos: Vec3,
    vel: Vec3,
    quat: Quat,
): Body {
    return {
        posLin: clone3(pos),
        posAng: clone4(quat),
        initialLin: clone3(pos),
        initialAng: clone4(quat),
        inertialLin: clone3(pos),
        inertialAng: clone4(quat),
        velLin: clone3(vel),
        velAng: [0, 0, 0],
        prevVelLin: clone3(vel),
        shape,
        size: clone3(size),
        roundRadius,
        mass,
        moment,
        friction,
        radius: length(scale(size, 0.5)) + roundRadius,
    };
}

/**
 * A rigid box in its initial state. `mass <= 0` is static (use {@link massOf} to specify a box by
 * density). The diagonal moment of inertia + bounding radius derive from size + mass exactly as the
 * reference constructor does; the pose caches start at the initial pose and `prevVelLin` starts at the
 * initial velocity (matching `Rigid::Rigid`).
 */
export function body(
    size: Vec3,
    mass: number,
    friction: number,
    pos: Vec3,
    vel: Vec3 = [0, 0, 0],
    quat: Quat = [0, 0, 0, 1],
): Body {
    return makeBody(ShapeKind.Box, size, 0, mass, boxMoment(size, mass), friction, pos, vel, quat);
}

/** A rigid sphere of `radius`. Core is a point (`size = 0`); the narrowphase collides centre + radius. */
export function sphere(
    radius: number,
    mass: number,
    friction: number,
    pos: Vec3,
    vel: Vec3 = [0, 0, 0],
    quat: Quat = [0, 0, 0, 1],
): Body {
    return makeBody(
        ShapeKind.Sphere,
        [0, 0, 0],
        radius,
        mass,
        sphereMoment(radius, mass),
        friction,
        pos,
        vel,
        quat,
    );
}

/**
 * A rigid capsule: a segment of half-length `halfHeight` along local Y, inflated by `radius`. Orient
 * it along another axis with `quat`. The narrowphase collides the segment ± radius.
 */
export function capsule(
    halfHeight: number,
    radius: number,
    mass: number,
    friction: number,
    pos: Vec3,
    vel: Vec3 = [0, 0, 0],
    quat: Quat = [0, 0, 0, 1],
): Body {
    return makeBody(
        ShapeKind.Capsule,
        [0, 2 * halfHeight, 0],
        radius,
        mass,
        capsuleMoment(halfHeight, radius, mass),
        friction,
        pos,
        vel,
        quat,
    );
}

/**
 * A rigid convex hull (the registry-stored geometry; the narrowphase reads `body.hull`). The bounding-
 * sphere radius is the tightest local-origin sphere (max vertex distance); the geometry is assumed
 * centred on the body origin.
 *
 * Inertia is the **AABB box approximation** — exact for a box-hull (every hull tested so far), an
 * over-estimate for a tet/cone. This is the documented INTERIM: the settled approach (roadmap §6.3
 * "Decided — hull inertia") is exact polyhedral integration baked to the COM-centered principal-axis
 * frame, computed once at registration; it lands when a non-box-hull dynamics test needs it.
 */
export function hull(
    geom: Hull,
    mass: number,
    friction: number,
    pos: Vec3,
    vel: Vec3 = [0, 0, 0],
    quat: Quat = [0, 0, 0, 1],
): Body {
    const mn: Vec3 = [Infinity, Infinity, Infinity];
    const mx: Vec3 = [-Infinity, -Infinity, -Infinity];
    let maxR2 = 0;
    for (const v of geom.verts) {
        for (let i = 0; i < 3; i++) {
            if (v[i] < mn[i]) mn[i] = v[i];
            if (v[i] > mx[i]) mx[i] = v[i];
        }
        maxR2 = Math.max(maxR2, lengthSq(v));
    }
    const size: Vec3 = [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]];
    const b = makeBody(
        ShapeKind.Hull,
        size,
        0,
        mass,
        boxMoment(size, mass),
        friction,
        pos,
        vel,
        quat,
    );
    b.hull = geom;
    b.radius = Math.sqrt(maxR2);
    return b;
}
