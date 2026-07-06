// The kinematic character controller (roadmap §6.4) — a capsule the player drives, moved by a
// collide-and-slide sweep against the scene's bodies (statics block; dynamics block AND get shoved),
// NOT by the rigid solver. No AVBD/webphysics
// reference covers a character (demo3d is boxes only; webphysics has none), so it's grounded in the
// legacy `character.wgsl.ts` behavior (grounded, slope-limit, step-up) minus its bug, and the standard
// kinematic-controller recipe (Unity CharacterController / Godot move_and_slide / Jolt CharacterVirtual):
// integrate the controller's own velocity, sweep the capsule, slide along what it hits, ride a moving
// platform (add the supporting body's velocity), and shove dynamics (velocity transfer). f64; the runtime
// CPU sweep (`standard/character/sweep.ts`) reproduces this. Test scaffolding.
//
// The character body is a `mass <= 0` capsule in the solver's body list, so the AVBD broadphase still
// generates its contacts: a dynamic body the character overlaps is pushed away by its own primal (the
// character, mass <= 0, is never moved by the solve — it has the controller's pose), and a body resting
// on the character is carried by the contact. The character-vs-static contact (both mass <= 0) is the
// one the dual must NOT ramp — the kinematic-pushing fix (manifold.ts updateDual). So the controller
// owns the character's POSE; the solver only reads it to push/carry others.
//
// Collide-and-slide is move-then-depenetrate: advance the capsule by the desired displacement, then
// iteratively push it out of the deepest penetration along the contact normal (each push removes only
// the into-surface motion, leaving the tangential slide). The capsule's rounded bottom climbs a
// sub-radius step for free — the edge contact's normal tilts up, lifting it over (this is why §6.4
// builds on the rounded-box narrowphase: a box character would catch its corner). Grounded keys on a
// walkable contact (normal.y above the slope cutoff) anywhere in the speculative band, so a settled
// character reads grounded without flickering at gap 0.

import { boxHull, closestPointOnHull } from "./hull";
import { add, clamp, dot, lengthSq, qconj, rotate, scale, sub, type Vec3 } from "./math";
import { type Body, ShapeKind } from "./rigid";

/** depenetration iterations per tick — a corner needs a few pushes to resolve both planes (legacy used 4) */
const MAX_SLIDE_ITERS = 6;
/** closest-point iterations for the segment-vs-box query — alternation converges on a box in a few steps */
const CLOSEST_ITERS = 5;
/** a contact within this gap of a walkable surface counts as ground (snap), so a settled capsule reads
 * grounded at gap 0 without flickering and a small step-down stays grounded */
const GROUND_SNAP = 0.05;
/** jump feel windows (seconds), the standard platformer pair: COYOTE_TIME lets a jump fire for a moment
 * AFTER walking off a ledge (grounded just lapsed), JUMP_BUFFER lets a jump pressed just BEFORE landing
 * fire on landing. Together they make "single jump, grounded-gated" forgiving without ever double-jumping:
 * a jump consumes the coyote credit, so spamming the button mid-air can't re-fire until the next landing. */
export const COYOTE_TIME = 0.1;
export const JUMP_BUFFER = 0.1;
/** candidate cap per character — one GPU lane per candidate (the workgroup width); overflow keeps the
 * first MAX_CHAR_CANDIDATES in scan order and flags loudly (the GPU bumps a counter) */
export const MAX_CHAR_CANDIDATES = 64;
/** cull slack absorbing the f32 (GPU) vs f64 (oracle) sphere-predicate disagreement at the boundary —
 * a boundary body's gap stays above GROUND_SNAP all tick, so it contributes to no phase either way */
const CULL_EPS = 1e-3;

/** a kinematic character: its capsule body, the controller velocity (gravity integrates into y), the
 * walkable-slope cutoff, and the jump state. `grounded` / `groundNormal` are the per-tick output the game reads. */
export interface Character {
    body: Body; // a kinematic capsule (mass <= 0); the controller owns its pose
    vel: Vec3; // the controller velocity — gravity accumulates into y, the input sets x/z each tick
    maxSlopeCos: number; // cos of the steepest walkable slope (a contact normal.y above this is ground)
    grounded: boolean;
    groundNormal: Vec3;
    jumpSpeed: number; // the launch velocity a jump sets vel.y to (0 disables jumping)
    coyote: number; // time-since-grounded credit (refilled to COYOTE_TIME while grounded, decays airborne)
    buffer: number; // jump-press buffer (set to JUMP_BUFFER on a press, decays) — fires when coyote > 0
}

/** Wrap a kinematic capsule body as a character. `maxSlopeDeg` is the steepest walkable slope; `jumpSpeed`
 * the launch velocity a buffered+grounded jump sets vel.y to (0 = no jumping). */
export function character(body: Body, maxSlopeDeg = 50, jumpSpeed = 0): Character {
    return {
        body,
        vel: [0, 0, 0],
        maxSlopeCos: Math.cos((maxSlopeDeg * Math.PI) / 180),
        grounded: false,
        groundNormal: [0, 0, 0],
        jumpSpeed,
        coyote: 0,
        buffer: 0,
    };
}

const polyOf = (b: Body) => (b.shape === ShapeKind.Hull ? b.hull! : boxHull(b.size));

/** per-tick cull diagnostics — the runtime CPU sweep's `SweepDiag` mirrors these */
export interface MoveDiag {
    candidates: number; // gathered candidate count (statics + push)
    overflow: boolean; // the set exceeded MAX_CHAR_CANDIDATES (first MAX kept, in scan order)
    guard: boolean; // displacement exceeded the gather band's budget (pathological depenetration)
}

/**
 * The sphere cull: a body is a candidate iff its bounding sphere can reach the capsule within this
 * tick's motion budget — `dist(body, start) ≤ reach + bound + band`, where reach = the capsule's
 * bounding radius (`ch.body.radius` = halfHeight + roundRadius, rotation-invariant), bound = the
 * body's `Body.radius` (the broadphase bound), and band = `motion` (this tick's travel, |vel|·dt) +
 * 2·GROUND_SNAP + reach + CULL_EPS. A contact-set-preserving superset, derived: a culled body's
 * surface gap at `start` exceeds the band, the char's center travels at most `motion` + reach (the
 * depenetration allowance, guarded below) + GROUND_SNAP (the snap), so the gap stays above GROUND_SNAP at every
 * visited pose — and every phase gates on `gap < GROUND_SNAP` (walkable/carry/snap/push touch band)
 * or `depth > 0` (slide), so the culled body contributes to none of them. Candidates keep scan
 * order, so the order-dependent selections (first-max depth, last-walkable groundNormal) are
 * untouched. The GPU runs the identical predicate over the dense eid scan.
 */
function gather(
    ch: Character,
    start: Vec3,
    motion: number,
    statics: Body[],
    push: Body[],
    diag?: MoveDiag,
): { statics: Body[]; push: Body[] } {
    const pad = 2 * ch.body.radius + motion + 2 * GROUND_SNAP + CULL_EPS;
    const keep = (st: Body) => {
        const r = pad + st.radius;
        return lengthSq(sub(st.posLin, start)) <= r * r;
    };
    const s: Body[] = [];
    const p: Body[] = [];
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

/** world endpoints of the capsule's core segment (centre ± rotate(quat, halfHeight·Y)) at a trial pose */
function coreAt(ch: Character, pos: Vec3): { e0: Vec3; e1: Vec3 } {
    const h = rotate(ch.body.posAng, scale(ch.body.size, 0.5));
    return { e0: sub(pos, h), e1: add(pos, h) };
}

/** closest point on segment [a,b] to point p */
function closestOnSeg(p: Vec3, a: Vec3, b: Vec3): Vec3 {
    const ab = sub(b, a);
    const l2 = lengthSq(ab);
    const t = l2 < 1e-12 ? 0 : clamp(dot(sub(p, a), ab) / l2, 0, 1);
    return add(a, scale(ab, t));
}

/**
 * The capsule (at trial pose `pos`) vs one static polytope: the minimum-translation push-out as
 * `{normal, depth}` (normal = static → capsule, the push-out direction; depth > 0 = overlap) and whether
 * it's a walkable surface within snap range (the grounded signal). This is the geometric MTV — the closest
 * point between the capsule's CORE segment and the polytope, found by alternating closest-on-polytope
 * (`closestPointOnHull`, the §6.3 rounded-narrowphase primitive) and closest-on-segment until it settles.
 * Unlike the solver's SAT manifold (whose reference-face normal would wedge the capsule at a step edge),
 * the true closest-point normal tilts UP at an edge, so the rounded bottom climbs a sub-radius step for
 * free; at a tall wall the same query returns the horizontal face normal and the capsule is stopped.
 */
function probe(
    ch: Character,
    pos: Vec3,
    st: Body,
): { normal: Vec3; depth: number; walkable: boolean } {
    const { e0, e1 } = coreAt(ch, pos);
    const h = polyOf(st);
    const qc = qconj(st.posAng);
    const a = rotate(qc, sub(e0, st.posLin)); // core segment in the polytope's local frame
    const b = rotate(qc, sub(e1, st.posLin));
    let q = scale(add(a, b), 0.5);
    let cp = closestPointOnHull(h, q);
    for (let k = 0; k < CLOSEST_ITERS; k++) {
        q = closestOnSeg(cp.point, a, b); // the segment point nearest the current surface point
        cp = closestPointOnHull(h, q); // the surface point nearest that — alternation converges
    }
    const normal = rotate(st.posAng, cp.normal); // world, polytope → core (outward; inside → push-out face)
    const gap = cp.signedDist - ch.body.roundRadius; // surface gap; < 0 = the capsule overlaps
    const walkable = gap < GROUND_SNAP && normal[1] > ch.maxSlopeCos;
    return { normal, depth: -gap, walkable };
}

/**
 * One controller tick. `input` is the desired horizontal velocity (x/z; y ignored — gravity owns the
 * vertical, a jump sets `ch.vel[1]` before the call). Integrates gravity, sweeps the capsule against the
 * `statics` (mass <= 0 — walls, ground, platforms) AND the `push` dynamics collide-and-slide (every body
 * blocks; Jolt CharacterVirtual's model), adds the supporting body's velocity (moving-platform carry), then
 * writes the character body's pose + the realized velocity (the latter feeds the AVBD broadphase pad + the
 * carry of bodies resting ON the character). `push` is the dynamic bodies the char blocks against + shoves
 * at full speed (velocity transfer — see the loop below); `[]` if the scene handles push elsewhere.
 * Every phase runs over the sphere-culled candidate set (`gather`); `opts.cull: false` is the brute
 * test seam (bit-identical output — the cull is a contact-set-preserving superset), `opts.diag`
 * surfaces the gather diagnostics the GPU reports via counters.
 */
export function moveCharacter(
    ch: Character,
    input: Vec3,
    statics: Body[],
    gravity: number,
    dt: number,
    jumpPressed = false,
    push: Body[] = [],
    opts?: { cull?: boolean; diag?: MoveDiag },
): void {
    // jump timers (read last tick's grounded). coyote refills while grounded then decays once airborne;
    // buffer is set by a press then decays. A jump fires only when BOTH are positive — buffered AND within
    // the coyote window — and CONSUMES both, so spamming the button mid-air can't re-fire (single jump): the
    // coyote credit is spent and won't refill until the next landing. A press just before landing rides the
    // buffer and fires on touchdown; a press just after a ledge rides coyote. jumpSpeed 0 disables it.
    ch.coyote = ch.grounded ? COYOTE_TIME : Math.max(ch.coyote - dt, 0);
    ch.buffer = jumpPressed ? JUMP_BUFFER : Math.max(ch.buffer - dt, 0);

    // gravity integrates the vertical velocity ONLY while airborne; horizontal is the direct input (a
    // kinematic controller has no horizontal inertia). Gating gravity on last tick's grounded is what holds
    // a walkable slope without creep — applying gravity every tick leaves a tangential residual after the
    // perpendicular push-out, so the capsule slowly slides even on flat-enough ground. A too-steep slope
    // never grounds, so gravity keeps building there → it slides, the slope-limit behavior.
    let vy = ch.grounded ? 0 : ch.vel[1] + gravity * dt;
    if (ch.jumpSpeed > 0 && ch.buffer > 0 && ch.coyote > 0) {
        vy = ch.jumpSpeed; // launch; consume both credits so a held/spammed button can't double-jump
        ch.buffer = 0;
        ch.coyote = 0;
    }
    ch.vel = [input[0], vy, input[2]];

    const start: Vec3 = [...ch.body.posLin] as Vec3;

    // gather the candidate set once for the whole tick. The provisional band carries groundVel = 0 (it
    // isn't known yet); if the carry below finds a moving support, re-gather with the full motion — the
    // band ↔ groundVel circularity resolved exactly, costing one extra culled scan only on a platform.
    const cull = opts?.cull !== false;
    const diag = opts?.diag;
    if (diag) {
        diag.candidates = statics.length + push.length;
        diag.overflow = false;
        diag.guard = false;
    }
    const motionOf = (gv: Vec3) => Math.sqrt(lengthSq(add(ch.vel, gv))) * dt;
    let cand = cull
        ? gather(ch, start, motionOf([0, 0, 0]), statics, push, diag)
        : { statics, push };

    // moving-platform carry (legacy charMove += groundVelocity): add the supporting body's velocity to the
    // controller's motion so the char rides a translating/descending platform. The ground is the deepest
    // walkable contact at `start`; a true static reads velLin 0, so a flat floor never carries. Excluded from
    // ch.vel (the controller's own velocity, for next-tick gravity) — it's transport, present only in the
    // realized motion below.
    let groundVel: Vec3 = [0, 0, 0];
    let carryDepth = Number.NEGATIVE_INFINITY;
    for (const st of cand.statics) {
        const p = probe(ch, start, st);
        if (p.walkable && p.depth > carryDepth) {
            carryDepth = p.depth;
            groundVel = st.velLin;
        }
    }
    if (cull && lengthSq(groundVel) > 0)
        cand = gather(ch, start, motionOf(groundVel), statics, push, diag);
    let pos = add(start, scale(add(ch.vel, groundVel), dt));

    let grounded = false;
    let groundNormal: Vec3 = [0, 0, 0];
    // statics AND dynamics block in every direction (Jolt CharacterVirtual: every body is a contact
    // plane; the char never penetrates a dynamic — it shoves it via the velocity transfer below, which
    // keys on the DESIRED velocity, so depenetrating here doesn't kill the push). Excluding side dynamic
    // contacts let the capsule sink into a box until the closest-point MTV flipped to its top face
    // (walkable) and the sweep ejected the char upward — the teleport-on-top bug.
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
        pos = add(pos, scale(normal, depth)); // push to the surface (gap 0) along the contact normal
    }
    const realized = scale(sub(pos, start), 1 / dt); // before the snap — the actual swept motion

    // ground snap: pull a grounded capsule down onto the surface (gap 0) so it rests AT the ground rather
    // than hovering up to GROUND_SNAP above where the descent step happened to land it, and stays glued
    // walking down a slope/step. Capped at GROUND_SNAP so it never yanks across a real drop (a ledge:
    // the gap exceeds the snap, so it falls). Excluded from the realized velocity (a cosmetic correction,
    // not motion the broadphase/carry should see). Skipped while RISING (realized.y > 0) — climbing a step
    // the contact normal tilts diagonally, and snapping along it would pull the capsule back off the edge.
    if (grounded && realized[1] <= 0) {
        let gap = Number.POSITIVE_INFINITY;
        for (const st of cand.statics) {
            const p = probe(ch, pos, st);
            if (p.walkable) gap = Math.min(gap, -p.depth); // gap = −depth: < 0 penetrating, > 0 floating
        }
        if (gap > 0 && gap <= GROUND_SNAP) pos = sub(pos, scale(groundNormal, gap));
    }

    // full-speed push (velocity transfer — the standard kinematic shove, roadmap §6.4): for each dynamic the
    // char is in contact with (the sweep leaves it AT the face, gap ~0, so the trigger is the touch band, not
    // penetration), drive its velocity along the push normal up to the char's DESIRED speed into it — the
    // desired velocity, not the realized one, because the sweep above zeroes the realized into-component at
    // the face (Jolt HandleContact: the impulse comes from inVelocity relative to the body). Without this the
    // dynamic is only nudged by the soft solver contact (~0.4 m/s), so a walking char would stall against it.
    // Mass-independent (the kinematic char overpowers) — a max push-force cap is a later refinement. The
    // downward component of the transfer is cancelled (Jolt's impulse down-cancel): a char landing on a box
    // must not hammer it down at fall speed — gravity reaches the box through the solver, not the push.
    const desired = add(ch.vel, groundVel);
    for (const d of cand.push) {
        const p = probe(ch, pos, d);
        if (-p.depth > GROUND_SNAP) continue; // not in contact (gap beyond the touch band)
        const dir = scale(p.normal, -1); // push normal: char → dynamic
        const into = dot(desired, dir); // the char's desired speed toward the dynamic (0 if moving away)
        if (into <= 0) continue;
        const cur = dot(d.velLin, dir);
        if (cur >= into) continue;
        const dv = scale(dir, into - cur);
        if (dv[1] < 0) dv[1] = 0; // down-cancel
        d.velLin = add(d.velLin, dv);
    }

    // displacement guard: the cull band budgets the tick's travel at |motion| + reach (depenetration)
    // + GROUND_SNAP (snap); exceeding it means the band assumption broke (spawn-inside-geometry class)
    // — flag loudly, never silently (the GPU bumps a counter).
    if (
        diag &&
        lengthSq(sub(pos, start)) > (motionOf(groundVel) + ch.body.radius + GROUND_SNAP) ** 2
    )
        diag.guard = true;

    ch.body.posLin = pos;
    ch.body.velLin = realized; // realized velocity (wall zeroes x/z, ground zeroes y) for broadphase + carry
    ch.grounded = grounded; // next tick gates gravity on this — a walkable slope holds, a steep one slides
    ch.groundNormal = groundNormal;
}
