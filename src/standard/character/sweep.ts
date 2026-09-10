// CPU character sweep — the runtime collide-and-slide that owns a kinematic capsule's pose, computed on
// the CPU each fixed tick so the player's input → pose path is same-frame (no GPU readback). It is the SOLE
// runtime controller, the f32-tier twin of the f64 oracle `tests/avbd/character.ts` (the spec): a faithful
// port of the oracle's `moveCharacter`, validated against it (`tests/avbd/character-sweep.oracle.ts`). The
// algorithm is unchanged from the oracle — gather (sphere-cull), collide-and-slide along the geometric
// closest-point MTV, ground snap, moving-platform carry, the coyote/jump-buffer timers, the full-speed push
// — only the data it reads is reshaped: runtime poses the caller supplies (the character's own `Body` pose,
// static candidates from the authored slab, dynamic candidates from a `Mirror` snapshot), the `raycast.ts`
// shape of a pure CPU physics primitive over caller-supplied poses.
//
// The controller owns the character's POSE; the broadphase/solver only read it (to push dynamics + carry
// riders), so the sweep is one-way coupled and never touches the GPU here. Reuses the §6.3 closest-point
// narrowphase (the closest-point normal tilts UP at a step edge, so the rounded bottom climbs a sub-radius
// step for free where the SAT face normal would wedge it). Box AND hull statics — the scene's static
// colliders can be either, so a box-only subset would walk through hull geometry.

import { ShapeKind } from "../physics";
import { type Hull, type HullFace, qRotate } from "../physics/core";

type Vec3 = [number, number, number];
type Quat = [number, number, number, number];

// the controller constants — mirror the f64 oracle (`tests/avbd/character.ts`, the spec); the CPU == oracle
// gate (`character-sweep.oracle.ts`) keeps the two homes in sync, the `SPECULATIVE_DISTANCE` shape
// (`physics.md`).
/** depenetration iterations per tick — a corner needs a few pushes to resolve both planes */
const MAX_SLIDE_ITERS = 6;
/** closest-point alternation iterations for the segment-vs-polytope query — converges on a box in a few steps */
const CLOSEST_ITERS = 5;
/** a contact within this gap of a walkable surface counts as ground (snap), so a settled capsule reads
 * grounded at gap 0 without flickering and a small step-down stays glued */
const GROUND_SNAP = 0.05;
/** jump feel windows (seconds): coyote lets a jump fire briefly after leaving a ledge, buffer lets a jump
 * pressed just before landing fire on touchdown — consuming both on launch keeps it a single jump */
const COYOTE_TIME = 0.1;
const JUMP_BUFFER = 0.1;
/** candidate cap per character (the GPU workgroup width): overflow keeps the first 64 in scan order, loud */
export const MAX_CHAR_CANDIDATES = 64;
/** cull slack absorbing the f32-vs-f64 sphere-boundary disagreement — a boundary body contributes to no phase */
const CULL_EPS = 1e-3;
/** below this the closest-point difference is treated as the inside case (collide.ts ROUND_NORMAL_EPS) */
const NORMAL_EPS = 1e-9;
const PLANE_EPS = 1e-7;

// ── vec / quat helpers (self-contained, the raycast.ts pattern) ──────────────────────────────────────
const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (v: Vec3, s: number): Vec3 => [v[0] * s, v[1] * s, v[2] * s];
const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Vec3, b: Vec3): Vec3 => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
];
const lengthSq = (v: Vec3): number => dot(v, v);
const len = (v: Vec3): number => Math.sqrt(lengthSq(v));
const clamp = (x: number, a: number, b: number): number => Math.max(a, Math.min(b, x));
const rotate = (q: Quat, v: Vec3): Vec3 => qRotate(q[0], q[1], q[2], q[3], v[0], v[1], v[2]);
const rotateInv = (q: Quat, v: Vec3): Vec3 => qRotate(-q[0], -q[1], -q[2], q[3], v[0], v[1], v[2]);

/**
 * one candidate body the sweep collides the capsule against: the world pose + collider geometry the caller
 * reads off the runtime sources (a static's authored `Body` slab, a dynamic's `Mirror`-read GPU pose). `half`
 * is the box / hull-AABB half-extents (the capsule core half-height for a capsule); `radius` the rounding
 * (`Body.halfExtents.w`); `hull` the registry geometry for `ShapeKind.Hull`; `vel` the body velocity (a
 * static platform's, read for the carry; a dynamic's, mutated by the push). The {@link RayBody} shape, plus
 * `vel` for the carry/push and `hull` for the convex path.
 */
export interface SweepBody {
    shape: number;
    pos: Vec3;
    quat: Quat;
    half: Vec3;
    radius: number;
    hull?: Hull;
    vel: Vec3;
}

/**
 * a kinematic character's controller state, owned by the CPU sweep across fixed ticks. `pos` / `quat` are
 * the capsule pose the sweep writes (read from the `Body` at tick start, written back for the kinematic
 * upload); `half` / `radius` the capsule core half-height + rounding; `maxSlopeCos` the cos of the walkable
 * cutoff; `jumpSpeed` the launch speed (0 = no jump). `vel` is the persistent controller velocity (gravity
 * accumulates into y), `realizedVel` the swept motion this tick (B_VELL for the upload + carry), and
 * `grounded` / `groundNormal` / `coyote` / `buffer` the per-tick outputs + jump timers the next tick reads.
 */
export interface CharState {
    pos: Vec3;
    quat: Quat;
    half: number;
    radius: number;
    maxSlopeCos: number;
    jumpSpeed: number;
    vel: Vec3;
    realizedVel: Vec3;
    grounded: boolean;
    groundNormal: Vec3;
    coyote: number;
    buffer: number;
}

/** per-tick gather diagnostics: `candidates` gathered, `overflow` past the cap, `guard` past the band budget. */
export interface SweepDiag {
    candidates: number;
    overflow: boolean;
    guard: boolean;
}

// ── geometry ─────────────────────────────────────────────────────────────────────────────────────────

// closest point on segment [a,b] to p (clamped projection) — oracle `closestOnSeg`
function closestOnSeg(p: Vec3, a: Vec3, b: Vec3): Vec3 {
    const ab = sub(b, a);
    const l2 = lengthSq(ab);
    const t = l2 < 1e-12 ? 0 : clamp(dot(sub(p, a), ab) / l2, 0, 1);
    return add(a, scale(ab, t));
}

interface Closest {
    point: Vec3;
    normal: Vec3;
    signedDist: number;
}

// closest point on an OBB (box-local) to a query — the box / sphere / capsule shortcut (collide.ts
// closestPointBox). Clamping into [−half, half] gives the surface point when the query is outside; inside,
// push out along the least-clearance face. `+ signedDist` outside, `−` inside. Geometrically exact — it
// reproduces the oracle's `closestPointOnHull(boxHull)` to float precision (the gym GPU == oracle gate).
function closestPointBox(pl: Vec3, half: Vec3): Closest {
    const d: Vec3 = [
        clamp(pl[0], -half[0], half[0]),
        clamp(pl[1], -half[1], half[1]),
        clamp(pl[2], -half[2], half[2]),
    ];
    const diff = sub(pl, d);
    const distSq = lengthSq(diff);
    if (distSq > NORMAL_EPS * NORMAL_EPS) {
        const dist = Math.sqrt(distSq);
        return { point: d, normal: scale(diff, 1 / dist), signedDist: dist };
    }
    let axis = 0;
    let least = half[0] - Math.abs(pl[0]);
    const cy = half[1] - Math.abs(pl[1]);
    const cz = half[2] - Math.abs(pl[2]);
    if (cy < least) {
        least = cy;
        axis = 1;
    }
    if (cz < least) {
        least = cz;
        axis = 2;
    }
    const normal: Vec3 = [0, 0, 0];
    const point: Vec3 = [...pl] as Vec3;
    const s = pl[axis] >= 0 ? 1 : -1;
    normal[axis] = s;
    point[axis] = s * half[axis];
    return { point, normal, signedDist: -least };
}

// is the in-plane point `p` inside face `f`'s polygon? (on the inner side of every CCW edge) — oracle pointInFace
function pointInFace(h: Hull, f: HullFace, p: Vec3): boolean {
    for (let i = 0; i < f.verts.length; i++) {
        const a = h.verts[f.verts[i]];
        const b = h.verts[f.verts[(i + 1) % f.verts.length]];
        if (dot(cross(sub(b, a), sub(p, a)), f.normal) < -PLANE_EPS) return false;
    }
    return true;
}

// closest point on a convex hull to a LOCAL-frame query — the general convex path (face region → edge →
// vertex outside; least-penetrating face inside). A verbatim port of the oracle `closestPointOnHull`
// (tests/avbd/hull.ts); the WGSL twin is collide.ts `closestPointOnHull`. No GJK/EPA — analytic + exact.
function closestPointHull(h: Hull, q: Vec3): Closest {
    let maxD = -Infinity;
    let maxFace = 0;
    for (let i = 0; i < h.faces.length; i++) {
        const d = dot(h.faces[i].normal, q) - h.faces[i].offset;
        if (d > maxD) {
            maxD = d;
            maxFace = i;
        }
    }
    if (maxD <= 0) {
        const f = h.faces[maxFace];
        return { point: sub(q, scale(f.normal, maxD)), normal: f.normal, signedDist: maxD };
    }
    let bestDist = Infinity;
    let bestPoint: Vec3 = q;
    for (const f of h.faces) {
        const d = dot(f.normal, q) - f.offset;
        if (d <= 0) continue;
        const proj = sub(q, scale(f.normal, d));
        if (pointInFace(h, f, proj) && d < bestDist) {
            bestDist = d;
            bestPoint = proj;
        }
    }
    for (const f of h.faces) {
        for (let i = 0; i < f.verts.length; i++) {
            const cp = closestOnSeg(
                q,
                h.verts[f.verts[i]],
                h.verts[f.verts[(i + 1) % f.verts.length]],
            );
            const dd = len(sub(q, cp));
            if (dd < bestDist) {
                bestDist = dd;
                bestPoint = cp;
            }
        }
    }
    for (const v of h.verts) {
        const dd = len(sub(q, v));
        if (dd < bestDist) {
            bestDist = dd;
            bestPoint = v;
        }
    }
    const diff = sub(q, bestPoint);
    const dist = len(diff);
    const normal = dist > NORMAL_EPS ? scale(diff, 1 / dist) : h.faces[maxFace].normal;
    return { point: bestPoint, normal, signedDist: dist };
}

// closest point on a candidate polytope (LOCAL frame) — box / sphere / capsule use the exact OBB clamp, a
// hull the general convex routine. Mirrors the GPU `closestPointOnPoly` (box → closestPointBox, hull →
// closestPointOnHull) and the oracle `polyOf` (non-hull → boxHull of the core extents).
function closestPointPoly(st: SweepBody, q: Vec3): Closest {
    if (st.shape === ShapeKind.Hull && st.hull) return closestPointHull(st.hull, q);
    return closestPointBox(q, st.half);
}

// the capsule's bounding-sphere radius used by the cull (length(half) + rounding), matching the GPU
// keepBody (`length(bHalf) + bRadius`). For a box (radius 0) this is the half-diagonal.
const boundRadius = (b: SweepBody): number => len(b.half) + b.radius;

// the capsule core segment endpoints at trial pose `pos` (centre ± rotate(quat, halfHeight·Y)) — oracle coreAt
function coreAt(ch: CharState, pos: Vec3): { e0: Vec3; e1: Vec3 } {
    const h = rotate(ch.quat, [0, ch.half, 0]);
    return { e0: sub(pos, h), e1: add(pos, h) };
}

interface Probe {
    normal: Vec3;
    depth: number;
    walkable: boolean;
}

// the capsule (at trial pose `pos`) vs one candidate polytope: the minimum-translation push-out as
// {normal (candidate → capsule = push-out), depth (> 0 = overlap), walkable}. The geometric MTV — the
// closest point between the capsule CORE segment and the polytope, found by alternating closest-on-polytope
// ↔ closest-on-segment until it settles. Verbatim oracle `probe`; the closest-point normal tilts UP at a
// step edge (the free step-up), where the SAT reference-face normal would wedge the capsule horizontally.
function probe(ch: CharState, pos: Vec3, st: SweepBody): Probe {
    const { e0, e1 } = coreAt(ch, pos);
    const a = rotateInv(st.quat, sub(e0, st.pos)); // core segment in the polytope's local frame
    const b = rotateInv(st.quat, sub(e1, st.pos));
    let q = scale(add(a, b), 0.5);
    let cp = closestPointPoly(st, q);
    for (let k = 0; k < CLOSEST_ITERS; k++) {
        q = closestOnSeg(cp.point, a, b);
        cp = closestPointPoly(st, q);
    }
    const normal = rotate(st.quat, cp.normal); // world, polytope → core (the push-out direction)
    const gap = cp.signedDist - ch.radius; // surface gap; < 0 = the capsule overlaps
    const walkable = gap < GROUND_SNAP && normal[1] > ch.maxSlopeCos;
    return { normal, depth: -gap, walkable };
}

// the sphere cull (oracle `gather` keep): a body is a candidate iff its bounding sphere can reach the
// capsule within this tick's motion budget. A contact-set-preserving superset — a culled body's gap stays
// above GROUND_SNAP at every visited pose, and every phase gates on gap < GROUND_SNAP or depth > 0, so it
// contributes to none. Candidates keep scan order (statics first, then push), so the order-dependent
// selections (first-max depth, last-walkable groundNormal) are bit-identical to a full scan; overflow keeps
// the first MAX_CHAR_CANDIDATES + flags loudly. `reach` = the capsule's bounding radius (rotation-invariant).
function gather(
    reach: number,
    start: Vec3,
    motion: number,
    statics: SweepBody[],
    push: SweepBody[],
    diag?: SweepDiag,
): { statics: SweepBody[]; push: SweepBody[] } {
    const pad = 2 * reach + motion + 2 * GROUND_SNAP + CULL_EPS;
    const keep = (st: SweepBody): boolean => {
        const r = pad + boundRadius(st);
        return lengthSq(sub(st.pos, start)) <= r * r;
    };
    const s: SweepBody[] = [];
    const p: SweepBody[] = [];
    let overflow = false;
    for (const st of statics) {
        if (!keep(st)) continue;
        if (s.length >= MAX_CHAR_CANDIDATES) {
            overflow = true;
            break;
        }
        s.push(st);
    }
    if (!overflow)
        for (const d of push) {
            if (!keep(d)) continue;
            if (s.length + p.length >= MAX_CHAR_CANDIDATES) {
                overflow = true;
                break;
            }
            p.push(d);
        }
    if (diag) {
        diag.candidates = s.length + p.length;
        diag.overflow ||= overflow;
    }
    return { statics: s, push: p };
}

/**
 * One controller tick on the CPU, mutating `ch` in place: the runtime twin of the f64 oracle `moveCharacter`.
 * `input` is the desired horizontal velocity (x/z; y ignored, gravity owns the
 * vertical, a jump sets it). Integrates gravity (only while airborne), gathers the sphere-culled candidate
 * set, sweeps the capsule collide-and-slide against `statics` (mass ≤ 0, walls / ground / platforms) AND
 * `push` dynamics (every body blocks; Jolt CharacterVirtual's model), rides a moving platform (carry), snaps
 * to the ground, and shoves touched dynamics at the desired speed (the push mutates their `vel`). Writes the
 * swept pose (`ch.pos`), the realized velocity (`ch.realizedVel`, for the kinematic upload + carry), and the
 * grounded / jump-timer state. `cull: false` is the brute seam (bit-identical output; the cull is a
 * contact-set-preserving superset); `diag` surfaces the gather + displacement-guard diagnostics.
 */
export function sweepCharacter(
    ch: CharState,
    input: Vec3,
    statics: SweepBody[],
    gravity: number,
    dt: number,
    jumpPressed = false,
    push: SweepBody[] = [],
    opts?: { cull?: boolean; diag?: SweepDiag },
): void {
    // jump timers (read last tick's grounded): coyote refills while grounded then decays airborne; buffer is
    // set on a press then decays. A jump fires only when BOTH are positive and CONSUMES both — single jump,
    // a held/spammed button can't re-fire mid-air.
    ch.coyote = ch.grounded ? COYOTE_TIME : Math.max(ch.coyote - dt, 0);
    ch.buffer = jumpPressed ? JUMP_BUFFER : Math.max(ch.buffer - dt, 0);

    // gravity integrates the vertical velocity ONLY while airborne; horizontal is the direct input (no
    // horizontal inertia). Gating gravity on grounded holds a walkable slope without creep; a too-steep slope
    // never grounds, so gravity keeps building → it slides.
    let vy = ch.grounded ? 0 : ch.vel[1] + gravity * dt;
    if (ch.jumpSpeed > 0 && ch.buffer > 0 && ch.coyote > 0) {
        vy = ch.jumpSpeed;
        ch.buffer = 0;
        ch.coyote = 0;
    }
    ch.vel = [input[0], vy, input[2]];

    const reach = ch.half + ch.radius;
    const start: Vec3 = [...ch.pos] as Vec3;
    const cull = opts?.cull !== false;
    const diag = opts?.diag;
    if (diag) {
        diag.candidates = statics.length + push.length;
        diag.overflow = false;
        diag.guard = false;
    }
    const motionOf = (gv: Vec3): number => Math.sqrt(lengthSq(add(ch.vel, gv))) * dt;
    let cand = cull
        ? gather(reach, start, motionOf([0, 0, 0]), statics, push, diag)
        : { statics, push };

    // moving-platform carry: add the supporting body's velocity to the motion so the char rides a
    // translating/descending platform. The ground is the deepest walkable contact at `start`; a true static
    // reads vel 0, so a flat floor never carries. Excluded from ch.vel (transport, not the controller's own
    // velocity). The provisional gather assumes groundVel = 0; a moving support re-gathers with the full band.
    let groundVel: Vec3 = [0, 0, 0];
    let carryDepth = Number.NEGATIVE_INFINITY;
    for (const st of cand.statics) {
        const p = probe(ch, start, st);
        if (p.walkable && p.depth > carryDepth) {
            carryDepth = p.depth;
            groundVel = st.vel;
        }
    }
    if (cull && lengthSq(groundVel) > 0)
        cand = gather(reach, start, motionOf(groundVel), statics, push, diag);
    let pos = add(start, scale(add(ch.vel, groundVel), dt));

    let grounded = false;
    let groundNormal: Vec3 = [0, 0, 0];
    // statics AND dynamics block in every direction (Jolt CharacterVirtual: every body is a contact plane;
    // the char shoves a dynamic via the velocity transfer below, keyed on the DESIRED velocity, so
    // depenetrating here doesn't kill the push).
    const blockers = [...cand.statics, ...cand.push];
    for (let iter = 0; iter < MAX_SLIDE_ITERS; iter++) {
        let depth = 0;
        let normal: Vec3 = [0, 0, 0];
        for (const st of blockers) {
            const p = probe(ch, pos, st);
            if (p.walkable) {
                grounded = true;
                groundNormal = p.normal;
            }
            if (p.depth > depth) {
                depth = p.depth;
                normal = p.normal;
            }
        }
        if (depth <= 0) break;
        pos = add(pos, scale(normal, depth));
    }
    const realized = scale(sub(pos, start), 1 / dt); // before the snap — the actual swept motion

    // ground snap: pull a grounded capsule onto the surface (gap 0) so it rests AT the ground and stays glued
    // walking down a slope/step. Capped at GROUND_SNAP (never yanks across a real drop — a ledge falls).
    // Skipped while RISING (climbing a step the normal tilts diagonally — snapping would pull it off the edge).
    // Excluded from the realized velocity (a cosmetic correction).
    if (grounded && realized[1] <= 0) {
        let gap = Number.POSITIVE_INFINITY;
        for (const st of cand.statics) {
            const p = probe(ch, pos, st);
            if (p.walkable) gap = Math.min(gap, -p.depth); // gap = −depth: < 0 penetrating, > 0 floating
        }
        if (gap > 0 && gap <= GROUND_SNAP) pos = sub(pos, scale(groundNormal, gap));
    }

    // full-speed push (velocity transfer): drive each touched dynamic's velocity along the push normal up to
    // the char's DESIRED speed into it — the sweep leaves the char AT the face (gap ~0), so the trigger is the
    // touch band, not penetration, and the desired velocity (not the zeroed realized one) is the impulse
    // source (Jolt HandleContact). The downward component is cancelled (Jolt's impulse down-cancel — a char
    // landing on a box must not hammer it down; gravity reaches the box through the solver).
    const desired = add(ch.vel, groundVel);
    for (const d of cand.push) {
        const p = probe(ch, pos, d);
        if (-p.depth > GROUND_SNAP) continue; // not in contact (gap beyond the touch band)
        const dir = scale(p.normal, -1); // push normal: char → dynamic
        const into = dot(desired, dir);
        if (into <= 0) continue;
        const cur = dot(d.vel, dir);
        if (cur >= into) continue;
        const dv = scale(dir, into - cur);
        if (dv[1] < 0) dv[1] = 0; // down-cancel
        d.vel = add(d.vel, dv);
    }

    // displacement guard: the band budgets the tick's travel at |motion| + reach (depenetration) +
    // GROUND_SNAP (snap); exceeding it means the band assumption broke (the spawn-inside-geometry class) —
    // flag loudly, never silently.
    if (diag && lengthSq(sub(pos, start)) > (motionOf(groundVel) + reach + GROUND_SNAP) ** 2)
        diag.guard = true;

    ch.pos = pos;
    ch.realizedVel = realized; // realized velocity (wall zeroes x/z, ground zeroes y) for the upload + carry
    ch.grounded = grounded; // next tick gates gravity on this — a walkable slope holds, a steep one slides
    ch.groundNormal = groundNormal;
}
