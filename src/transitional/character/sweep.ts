// CPU character sweep — the runtime collide-and-slide that owns a kinematic capsule's pose, computed on
// the CPU each fixed tick so the player's input → pose path is same-frame (no GPU readback). It is the SOLE
// runtime controller, the f32-tier twin of an f64 controller oracle (the spec): a faithful
// port of the oracle's `moveCharacter`, validated against it. The
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
//
// A tick allocates nothing: every vector result is written into an `out` the caller owns, each function's
// temporaries are module scratch it alone writes, and the controller state is updated in place. The
// arithmetic and its order are the oracle's, operand for operand.

import { type Hull, type HullFace, ShapeKind } from "../physics";

type Vec3 = [number, number, number];
type Quat = [number, number, number, number];

// the controller constants — mirror the f64 controller oracle (the spec); the CPU == oracle
// gate keeps the two homes in sync, the `SPECULATIVE_DISTANCE` shape
//.
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
// Each writes `out` and returns it. The component-wise ones read only the component they write, so `out`
// may alias an input; `cross` and the rotations read every input before writing.
const vec3 = (): Vec3 => [0, 0, 0];
const copy = (out: Vec3, v: Vec3): Vec3 => {
    out[0] = v[0];
    out[1] = v[1];
    out[2] = v[2];
    return out;
};
const zero = (out: Vec3): Vec3 => {
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    return out;
};
const add = (out: Vec3, a: Vec3, b: Vec3): Vec3 => {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
    return out;
};
const sub = (out: Vec3, a: Vec3, b: Vec3): Vec3 => {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
    return out;
};
const scale = (out: Vec3, v: Vec3, s: number): Vec3 => {
    out[0] = v[0] * s;
    out[1] = v[1] * s;
    out[2] = v[2] * s;
    return out;
};
const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (out: Vec3, a: Vec3, b: Vec3): Vec3 => {
    const x = a[1] * b[2] - a[2] * b[1];
    const y = a[2] * b[0] - a[0] * b[2];
    const z = a[0] * b[1] - a[1] * b[0];
    out[0] = x;
    out[1] = y;
    out[2] = z;
    return out;
};
const lengthSq = (v: Vec3): number => dot(v, v);
const len = (v: Vec3): number => Math.sqrt(lengthSq(v));
const clamp = (x: number, a: number, b: number): number => Math.max(a, Math.min(b, x));
// rotate `v` by the unit quaternion `q` — `qRotate`'s arithmetic over arrays: v + w·t + q×t, t = 2(q×v)
function rotate(out: Vec3, q: Quat, v: Vec3): Vec3 {
    const qx = q[0];
    const qy = q[1];
    const qz = q[2];
    const qw = q[3];
    const vx = v[0];
    const vy = v[1];
    const vz = v[2];
    const tx = 2 * (qy * vz - qz * vy);
    const ty = 2 * (qz * vx - qx * vz);
    const tz = 2 * (qx * vy - qy * vx);
    out[0] = vx + qw * tx + qy * tz - qz * ty;
    out[1] = vy + qw * ty + qz * tx - qx * tz;
    out[2] = vz + qw * tz + qx * ty - qy * tx;
    return out;
}

// rotate `v` by the conjugate of `q` (the inverse rotation), the same arithmetic as `rotate`
function rotateInv(out: Vec3, q: Quat, v: Vec3): Vec3 {
    const qx = -q[0];
    const qy = -q[1];
    const qz = -q[2];
    const qw = q[3];
    const vx = v[0];
    const vy = v[1];
    const vz = v[2];
    const tx = 2 * (qy * vz - qz * vy);
    const ty = 2 * (qz * vx - qx * vz);
    const tz = 2 * (qx * vy - qy * vx);
    out[0] = vx + qw * tx + qy * tz - qz * ty;
    out[1] = vy + qw * ty + qz * tx - qx * tz;
    out[2] = vz + qw * tz + qx * ty - qy * tx;
    return out;
}

/**
 * one candidate body the sweep collides the capsule against: the world pose + collider geometry the caller
 * reads off the runtime sources (a static's authored `Body` slab, a dynamic's `Mirror`-read GPU pose). `half`
 * is the box / hull-AABB half-extents (the capsule core half-height for a capsule); `radius` the rounding
 * (`Body.halfExtents.w`); `hull` the registry geometry for `ShapeKind.Hull`; `vel` the body velocity (a
 * static platform's, read for the carry; a dynamic's, mutated in place by the push). The {@link RayBody}
 * shape, plus `vel` for the carry/push and `hull` for the convex path.
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
 * The sweep writes the vectors in place, so each must be its own array.
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

const _segAB = vec3();
const _segAP = vec3();

// closest point on segment [a,b] to p (clamped projection), into `out` — oracle `closestOnSeg`
function closestOnSeg(out: Vec3, p: Vec3, a: Vec3, b: Vec3): Vec3 {
    const ab = sub(_segAB, b, a);
    const l2 = lengthSq(ab);
    const t = l2 < 1e-12 ? 0 : clamp(dot(sub(_segAP, p, a), ab) / l2, 0, 1);
    return add(out, a, scale(ab, ab, t));
}

/** a closest-point query's result; the query writes it and never aliases its own input into it. */
interface Closest {
    point: Vec3;
    normal: Vec3;
    signedDist: number;
}

const _boxDiff = vec3();

// closest point on an OBB (box-local) to a query, into `out` — the box / sphere / capsule shortcut (collide.ts
// closestPointBox). Clamping into [−half, half] gives the surface point when the query is outside; inside,
// push out along the least-clearance face. `+ signedDist` outside, `−` inside. Geometrically exact — it
// reproduces the oracle's `closestPointOnHull(boxHull)` to float precision. `pl` must not be `out.point`.
function closestPointBox(out: Closest, pl: Vec3, half: Vec3): Closest {
    const d = out.point;
    d[0] = clamp(pl[0], -half[0], half[0]);
    d[1] = clamp(pl[1], -half[1], half[1]);
    d[2] = clamp(pl[2], -half[2], half[2]);
    const diff = sub(_boxDiff, pl, d);
    const distSq = lengthSq(diff);
    if (distSq > NORMAL_EPS * NORMAL_EPS) {
        const dist = Math.sqrt(distSq);
        scale(out.normal, diff, 1 / dist);
        out.signedDist = dist;
        return out;
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
    const normal = zero(out.normal);
    const point = copy(d, pl);
    const s = pl[axis] >= 0 ? 1 : -1;
    normal[axis] = s;
    point[axis] = s * half[axis];
    out.signedDist = -least;
    return out;
}

const _faceEdge = vec3();
const _faceToP = vec3();
const _faceCross = vec3();

// is the in-plane point `p` inside face `f`'s polygon? (on the inner side of every CCW edge) — oracle pointInFace
function pointInFace(h: Hull, f: HullFace, p: Vec3): boolean {
    for (let i = 0; i < f.verts.length; i++) {
        const a = h.verts[f.verts[i]];
        const b = h.verts[f.verts[(i + 1) % f.verts.length]];
        const edge = sub(_faceEdge, b, a);
        const toP = sub(_faceToP, p, a);
        if (dot(cross(_faceCross, edge, toP), f.normal) < -PLANE_EPS) return false;
    }
    return true;
}

const _hullBest = vec3();
const _hullProj = vec3();
const _hullSeg = vec3();
const _hullDiff = vec3();

// closest point on a convex hull to a LOCAL-frame query, into `out` — the general convex path (face region →
// edge → vertex outside; least-penetrating face inside). A verbatim port of the oracle `closestPointOnHull`
// (the f64 hull oracle). No GJK/EPA — analytic + exact. `q` must not be `out.point`.
function closestPointHull(out: Closest, h: Hull, q: Vec3): Closest {
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
        sub(out.point, q, scale(out.point, f.normal, maxD));
        copy(out.normal, f.normal);
        out.signedDist = maxD;
        return out;
    }
    let bestDist = Infinity;
    const best = copy(_hullBest, q);
    for (let fi = 0; fi < h.faces.length; fi++) {
        const f = h.faces[fi];
        const d = dot(f.normal, q) - f.offset;
        if (d <= 0) continue;
        const proj = sub(_hullProj, q, scale(_hullProj, f.normal, d));
        if (pointInFace(h, f, proj) && d < bestDist) {
            bestDist = d;
            copy(best, proj);
        }
    }
    for (let fi = 0; fi < h.faces.length; fi++) {
        const f = h.faces[fi];
        for (let i = 0; i < f.verts.length; i++) {
            const cp = closestOnSeg(
                _hullSeg,
                q,
                h.verts[f.verts[i]],
                h.verts[f.verts[(i + 1) % f.verts.length]],
            );
            const dd = len(sub(_hullDiff, q, cp));
            if (dd < bestDist) {
                bestDist = dd;
                copy(best, cp);
            }
        }
    }
    for (let vi = 0; vi < h.verts.length; vi++) {
        const v = h.verts[vi];
        const dd = len(sub(_hullDiff, q, v));
        if (dd < bestDist) {
            bestDist = dd;
            copy(best, v);
        }
    }
    const diff = sub(_hullDiff, q, best);
    const dist = len(diff);
    if (dist > NORMAL_EPS) scale(out.normal, diff, 1 / dist);
    else copy(out.normal, h.faces[maxFace].normal);
    copy(out.point, best);
    out.signedDist = dist;
    return out;
}

// closest point on a candidate polytope (LOCAL frame), into `out` — box / sphere / capsule use the exact OBB
// clamp, a hull the general convex routine. Mirrors the GPU `closestPointOnPoly` (box → closestPointBox, hull →
// closestPointOnHull) and the oracle `polyOf` (non-hull → boxHull of the core extents).
function closestPointPoly(out: Closest, st: SweepBody, q: Vec3): Closest {
    if (st.shape === ShapeKind.Hull && st.hull) return closestPointHull(out, st.hull, q);
    return closestPointBox(out, q, st.half);
}

// the capsule's bounding-sphere radius used by the cull (length(half) + rounding), matching the GPU
// keepBody (`length(bHalf) + bRadius`). For a box (radius 0) this is the half-diagonal.
const boundRadius = (b: SweepBody): number => len(b.half) + b.radius;

const _coreAxis = vec3();
const _coreHalf = vec3();

// the capsule core segment endpoints at trial pose `pos` (centre ± rotate(quat, halfHeight·Y)), into `e0` /
// `e1` — oracle coreAt
function coreAt(ch: CharState, pos: Vec3, e0: Vec3, e1: Vec3): void {
    _coreAxis[1] = ch.half;
    const h = rotate(_coreHalf, ch.quat, _coreAxis);
    sub(e0, pos, h);
    add(e1, pos, h);
}

/** one probe's result; each call overwrites it, so a caller copies what it keeps. */
interface Probe {
    normal: Vec3;
    depth: number;
    walkable: boolean;
}

const _probeA = vec3();
const _probeB = vec3();
const _probeQ = vec3();
const _probeClosest: Closest = { point: vec3(), normal: vec3(), signedDist: 0 };

// the capsule (at trial pose `pos`) vs one candidate polytope, into `out`: the minimum-translation push-out as
// {normal (candidate → capsule = push-out), depth (> 0 = overlap), walkable}. The geometric MTV — the
// closest point between the capsule CORE segment and the polytope, found by alternating closest-on-polytope
// ↔ closest-on-segment until it settles. Verbatim oracle `probe`; the closest-point normal tilts UP at a
// step edge (the free step-up), where the SAT reference-face normal would wedge the capsule horizontally.
function probe(out: Probe, ch: CharState, pos: Vec3, st: SweepBody): Probe {
    const a = _probeA; // core segment in the polytope's local frame
    const b = _probeB;
    coreAt(ch, pos, a, b);
    rotateInv(a, st.quat, sub(a, a, st.pos));
    rotateInv(b, st.quat, sub(b, b, st.pos));
    const q = scale(_probeQ, add(_probeQ, a, b), 0.5);
    const cp = closestPointPoly(_probeClosest, st, q);
    for (let k = 0; k < CLOSEST_ITERS; k++) {
        closestOnSeg(q, cp.point, a, b);
        closestPointPoly(cp, st, q);
    }
    const normal = rotate(out.normal, st.quat, cp.normal); // world, polytope → core (the push-out direction)
    const gap = cp.signedDist - ch.radius; // surface gap; < 0 = the capsule overlaps
    out.walkable = gap < GROUND_SNAP && normal[1] > ch.maxSlopeCos;
    out.depth = -gap;
    return out;
}

const _cullOffset = vec3();
const _motion = vec3();

// the gathered candidates, refilled in place by each gather
const _gatheredStatics: SweepBody[] = [];
const _gatheredPush: SweepBody[] = [];

// fill `list` with its first `count` entries left as written, trimming only a changed length: setting an
// array's length to 0 releases its backing store, so a steady count keeps the store.
function trim(list: SweepBody[], count: number): void {
    if (list.length !== count) list.length = count;
}

// the sphere cull (oracle `gather`), into `_gatheredStatics` / `_gatheredPush`. A contact-set-preserving
// superset — a culled body's gap stays above GROUND_SNAP at every visited pose, and every phase gates on
// gap < GROUND_SNAP or depth > 0, so it contributes to none. Candidates keep scan order (statics first,
// then push), so the order-dependent selections (first-max depth, last-walkable groundNormal) are
// bit-identical to a full scan; overflow keeps the first MAX_CHAR_CANDIDATES + flags loudly. A body is kept
// iff its bounding sphere can reach the capsule within the tick's travel budget |vel + gv|·dt, where `reach`
// is the capsule's bounding radius (rotation-invariant) and `gv` the transport velocity.
function gather(
    ch: CharState,
    start: Vec3,
    gv: Vec3,
    dt: number,
    statics: SweepBody[],
    push: SweepBody[],
    diag?: SweepDiag,
): void {
    const reach = ch.half + ch.radius;
    const motion = Math.sqrt(lengthSq(add(_motion, ch.vel, gv))) * dt;
    const pad = 2 * reach + motion + 2 * GROUND_SNAP + CULL_EPS;
    const s = _gatheredStatics;
    const p = _gatheredPush;
    let ns = 0;
    let np = 0;
    let overflow = false;
    for (let i = 0; i < statics.length; i++) {
        const st = statics[i];
        const r = pad + boundRadius(st);
        if (!(lengthSq(sub(_cullOffset, st.pos, start)) <= r * r)) continue;
        if (ns >= MAX_CHAR_CANDIDATES) {
            overflow = true;
            break;
        }
        s[ns++] = st;
    }
    if (!overflow)
        for (let i = 0; i < push.length; i++) {
            const d = push[i];
            const r = pad + boundRadius(d);
            if (!(lengthSq(sub(_cullOffset, d.pos, start)) <= r * r)) continue;
            if (ns + np >= MAX_CHAR_CANDIDATES) {
                overflow = true;
                break;
            }
            p[np++] = d;
        }
    trim(s, ns);
    trim(p, np);
    if (diag) {
        diag.candidates = ns + np;
        diag.overflow ||= overflow;
    }
}

const ZERO: Vec3 = [0, 0, 0];
const _probe: Probe = { normal: vec3(), depth: 0, walkable: false };
const _start = vec3();
const _groundVel = vec3();
const _pos = vec3();
const _groundNormal = vec3();
const _normal = vec3();
const _shift = vec3();
const _realized = vec3();
const _desired = vec3();
const _dir = vec3();
const _dv = vec3();

/**
 * One controller tick on the CPU, mutating `ch` in place: the runtime twin of the f64 oracle `moveCharacter`.
 * `input` is the desired horizontal velocity (x/z; y ignored, gravity owns the
 * vertical, a jump sets it). Integrates gravity (only while airborne), gathers the sphere-culled candidate
 * set, sweeps the capsule collide-and-slide against `statics` (mass ≤ 0, walls / ground / platforms) AND
 * `push` dynamics (every body blocks; Jolt CharacterVirtual's model), rides a moving platform (carry), snaps
 * to the ground, and shoves touched dynamics at the desired speed (the push mutates their `vel` in place).
 * Writes the swept pose (`ch.pos`), the realized velocity (`ch.realizedVel`, for the kinematic upload +
 * carry), and the grounded / jump-timer state. `cull: false` is the brute seam (bit-identical output; the
 * cull is a contact-set-preserving superset); `diag` surfaces the gather + displacement-guard diagnostics.
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
    const vel = ch.vel;
    vel[0] = input[0];
    vel[1] = vy;
    vel[2] = input[2];

    const reach = ch.half + ch.radius;
    const start = copy(_start, ch.pos);
    const cull = opts?.cull !== false;
    const diag = opts?.diag;
    if (diag) {
        diag.candidates = statics.length + push.length;
        diag.overflow = false;
        diag.guard = false;
    }
    let candStatics = statics;
    let candPush = push;
    if (cull) {
        gather(ch, start, ZERO, dt, statics, push, diag);
        candStatics = _gatheredStatics;
        candPush = _gatheredPush;
    }

    // moving-platform carry: add the supporting body's velocity to the motion so the char rides a
    // translating/descending platform. The ground is the deepest walkable contact at `start`; a true static
    // reads vel 0, so a flat floor never carries. Excluded from ch.vel (transport, not the controller's own
    // velocity). The provisional gather assumes groundVel = 0; a moving support re-gathers with the full band.
    const groundVel = zero(_groundVel);
    let carryDepth = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < candStatics.length; i++) {
        const st = candStatics[i];
        const p = probe(_probe, ch, start, st);
        if (p.walkable && p.depth > carryDepth) {
            carryDepth = p.depth;
            copy(groundVel, st.vel);
        }
    }
    if (cull && lengthSq(groundVel) > 0) gather(ch, start, groundVel, dt, statics, push, diag);
    const pos = add(_pos, start, scale(_pos, add(_pos, vel, groundVel), dt));

    let grounded = false;
    const groundNormal = zero(_groundNormal);
    // statics AND dynamics block in every direction (Jolt CharacterVirtual: every body is a contact plane;
    // the char shoves a dynamic via the velocity transfer below, keyed on the DESIRED velocity, so
    // depenetrating here doesn't kill the push). Blockers are the statics, then the push set.
    const staticCount = candStatics.length;
    const blockerCount = staticCount + candPush.length;
    for (let iter = 0; iter < MAX_SLIDE_ITERS; iter++) {
        let depth = 0;
        const normal = zero(_normal);
        for (let i = 0; i < blockerCount; i++) {
            const st = i < staticCount ? candStatics[i] : candPush[i - staticCount];
            const p = probe(_probe, ch, pos, st);
            if (p.walkable) {
                grounded = true;
                copy(groundNormal, p.normal);
            }
            if (p.depth > depth) {
                depth = p.depth;
                copy(normal, p.normal);
            }
        }
        if (depth <= 0) break;
        add(pos, pos, scale(_shift, normal, depth));
    }
    const realized = scale(_realized, sub(_realized, pos, start), 1 / dt); // before the snap — the actual swept motion

    // ground snap: pull a grounded capsule onto the surface (gap 0) so it rests AT the ground and stays glued
    // walking down a slope/step. Capped at GROUND_SNAP (never yanks across a real drop — a ledge falls).
    // Skipped while RISING (climbing a step the normal tilts diagonally — snapping would pull it off the edge).
    // Excluded from the realized velocity (a cosmetic correction).
    if (grounded && realized[1] <= 0) {
        let gap = Number.POSITIVE_INFINITY;
        for (let i = 0; i < candStatics.length; i++) {
            const p = probe(_probe, ch, pos, candStatics[i]);
            if (p.walkable) gap = Math.min(gap, -p.depth); // gap = −depth: < 0 penetrating, > 0 floating
        }
        if (gap > 0 && gap <= GROUND_SNAP) sub(pos, pos, scale(_shift, groundNormal, gap));
    }

    // full-speed push (velocity transfer): drive each touched dynamic's velocity along the push normal up to
    // the char's DESIRED speed into it — the sweep leaves the char AT the face (gap ~0), so the trigger is the
    // touch band, not penetration, and the desired velocity (not the zeroed realized one) is the impulse
    // source (Jolt HandleContact). The downward component is cancelled (Jolt's impulse down-cancel — a char
    // landing on a box must not hammer it down; gravity reaches the box through the solver).
    const desired = add(_desired, vel, groundVel);
    for (let i = 0; i < candPush.length; i++) {
        const d = candPush[i];
        const p = probe(_probe, ch, pos, d);
        if (-p.depth > GROUND_SNAP) continue; // not in contact (gap beyond the touch band)
        const dir = scale(_dir, p.normal, -1); // push normal: char → dynamic
        const into = dot(desired, dir);
        if (into <= 0) continue;
        const cur = dot(d.vel, dir);
        if (cur >= into) continue;
        const dv = scale(_dv, dir, into - cur);
        if (dv[1] < 0) dv[1] = 0; // down-cancel
        add(d.vel, d.vel, dv);
    }

    // displacement guard: the band budgets the tick's travel at |motion| + reach (depenetration) +
    // GROUND_SNAP (snap); exceeding it means the band assumption broke (the spawn-inside-geometry class) —
    // flag loudly, never silently.
    if (
        diag &&
        lengthSq(sub(_shift, pos, start)) >
            (Math.sqrt(lengthSq(add(_motion, vel, groundVel))) * dt + reach + GROUND_SNAP) ** 2
    )
        diag.guard = true;

    copy(ch.pos, pos);
    copy(ch.realizedVel, realized); // realized velocity (wall zeroes x/z, ground zeroes y) for the upload + carry
    ch.grounded = grounded; // next tick gates gravity on this — a walkable slope holds, a steep one slides
    copy(ch.groundNormal, groundNormal);
}
