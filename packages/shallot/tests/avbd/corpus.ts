// Phase-4.6 topology stability corpus. The gym `pile` scenario is one jittered grid
// settling into a low heap — a single contact topology, so stability problems (the visible
// wobble) and narrowphase edge cases have nowhere to surface. This module is the corpus that
// gives them somewhere: distinct topologies — tall stack, wide pile, leaning/toppling, mixed
// box sizes, a box dropped onto a stack, a box landing on a corner — each with a statistical
// band (finite, settles, bounded penetration, no tunnel-through; energy non-increasing is the
// oracle tier's per-frame invariant).
//
// The scene builders + the band evaluator are shared so the band is one source of truth: the
// oracle test (corpus.test.ts) runs each through the f64 CPU solver — the reference for what
// "stable" looks like — and the gym `pile` scenario runs the same scenes on
// the real GPU and checks the same band. A scene that wobbles on the GPU where the oracle is
// steady is a GPU bug (physics.md "the oracle is not the suspect"). Test scaffolding; f64.

import { type Box, collide } from "./collide";
import { dot, length, type Quat, qmul, rotate, sub, transform, type Vec3 } from "./math";
import { type Body, body, massOf } from "./rigid";

// ground: a wide static box, top at GROUND_TOP. A unit box (half 0.5) rests at center REST_Y.
export const GROUND_TOP = 0.5;
const REST_Y = GROUND_TOP + 0.5; // 1.0
const ground = (mu = 0.5): Body => body([40, 1, 40], 0, mu, [0, 0, 0]);
const unit = (pos: Vec3, mu = 0.5, vel: Vec3 = [0, 0, 0], quat: Quat = [0, 0, 0, 1]): Body =>
    body([1, 1, 1], massOf([1, 1, 1], 1), mu, pos, vel, quat);

// deterministic sub-mm lateral jitter so a "stack" isn't a perfectly degenerate column (which
// hides rotational coupling) without tipping it into the toppling regime. ±~6 mm.
const jit = (i: number, salt: number): number =>
    (((i * 1327 + salt * 911) % 13) / 13 - 0.5) * 0.012;

/** a vertical stack of unit boxes with 1% gaps — the load-bearing-chain regime the convergence
 *  probe flags (a 10-tall chain wants ~8 iters; at 10 it converges to ~1.1× the margin rest). */
function tallStack(): Body[] {
    const bs = [ground()];
    for (let i = 0; i < 10; i++) bs.push(unit([jit(i, 1), REST_Y + i * 1.01, jit(i, 2)]));
    return bs;
}

/** a 3×3×2 block dropped with 5% gaps — moderate-valence many-contact settling (the gym regime,
 *  generalized to two layers so the bottom layer carries real simultaneous valence). */
function widePile(): Body[] {
    const bs = [ground()];
    for (let layer = 0; layer < 2; layer++)
        for (let gx = 0; gx < 3; gx++)
            for (let gz = 0; gz < 3; gz++) {
                const i = layer * 9 + gx * 3 + gz;
                bs.push(
                    unit([
                        (gx - 1) * 1.05 + jit(i, 3),
                        REST_Y + layer * 1.05,
                        (gz - 1) * 1.05 + jit(i, 4),
                    ]),
                );
            }
    return bs;
}

/** a 5-box stack each offset 0.4 in +x — the cumulative lean puts the upper COM well past the
 *  base, so it topples hard, scatters, and settles. The chaotic stress: a topple must dissipate,
 *  never inject energy. */
function leaning(): Body[] {
    const bs = [ground()];
    for (let i = 0; i < 5; i++) bs.push(unit([i * 0.4, REST_Y + i, 0]));
    return bs;
}

/** mixed masses + sizes: a [2,2,2] box (mass 8) dropped onto four unit boxes (mass 1) perched on
 *  a [4,1,4] slab (mass 16). Big-on-small bridging across four contacts + an 8:1 mass ratio. */
function mixedSizes(): Body[] {
    const bs = [ground()];
    bs.push(body([4, 1, 4], massOf([4, 1, 4], 1), 0.5, [0, 1.0, 0])); // slab, top 1.5
    for (const [sx, sz] of [
        [-1, -1],
        [1, -1],
        [-1, 1],
        [1, 1],
    ] as const)
        bs.push(unit([sx, 2.0, sz])); // four unit boxes on the slab, top 2.5
    bs.push(body([2, 2, 2], massOf([2, 2, 2], 1), 0.5, [0, 3.6, 0])); // big box, dropped 0.1 onto the bridge
    return bs;
}

/** a unit box dropped 5 m onto a settled 3-box stack — high-impact (~10 m/s) + warmstart churn as
 *  the load slams down the chain. */
function dropOnStack(): Body[] {
    const bs = [ground()];
    for (let i = 0; i < 3; i++) bs.push(unit([0, REST_Y + i, 0]));
    bs.push(unit([0, REST_Y + 3 + 5, 0])); // 5 m above the stack top
    return bs;
}

/** a box tilted 45° about x and z, dropped so it lands on a vertex — the hardest narrowphase
 *  (vertex→edge→face), churning feature keys as it tips down to a face. */
function cornerRest(): Body[] {
    const a = Math.PI / 4;
    const qx: Quat = [Math.sin(a / 2), 0, 0, Math.cos(a / 2)];
    const qz: Quat = [0, 0, Math.sin(a / 2), Math.cos(a / 2)];
    return [ground(0.6), unit([0, REST_Y + 1.5, 0], 0.6, [0, 0, 0], qmul(qz, qx))];
}

// `frames` is the horizon to run before reading the settled band — sized so the oracle reaches
// rest (the chaotic topple/drop scenes need the most). All six settle at the shipped iters=10 and the
// below-shipped iters=8 stress (measured offline); the 8:1-ratio mixed-sizes bridge needs
// iters≥6, so iters=8 is the lowest stress the band gates. A GPU scene that doesn't settle is the
// wobble the corpus exists to surface.
export interface Scene {
    name: string;
    bodies: () => Body[];
    frames: number;
}

export const CORPUS: Scene[] = [
    { name: "tall-stack", bodies: tallStack, frames: 200 },
    { name: "wide-pile", bodies: widePile, frames: 200 },
    { name: "leaning", bodies: leaning, frames: 400 },
    { name: "mixed-sizes", bodies: mixedSizes, frames: 220 },
    { name: "drop-on-stack", bodies: dropOnStack, frames: 260 },
    { name: "corner-rest", bodies: cornerRest, frames: 240 },
];

// the derived statistical band — bounds set an order off the oracle's measured steady behavior,
// loose enough to clear the f32 GPU's round-off, tight enough that a wobble/blow-up trips them.
// `BAND_*` is the one source of truth for the deterministic all-six oracle gate (corpus.test.ts).
export const BAND_SETTLE = 0.05; // m/s — < ⅓ of g·dt (0.167); oracle settles ≤ 0.017
export const BAND_ENERGY_EXCESS = 1e-2; // E never exceeds E0 (the drop supremum); oracle ≤ 0 (dissipative)
export const BAND_PENETRATION = 0.1; // m — settled overlap; margin+mg/k rest ~0.02, broken overlap ~0.5
export const BAND_TUNNEL = 0.15; // m a box bottom may dip below the ground top in a hard impact; a true tunnel escapes (≪0)

// ── the band evaluator (shared by both tiers) ────────────────────────────────────────────────

/** the minimal per-body state the band reads — a subset of the oracle Body, also the shape the
 *  GPU readback maps its SoA columns into. `size` is full widths (the SAT's Box.size). */
export interface BandBody {
    pos: Vec3;
    quat: Quat;
    vel: Vec3;
    angVel: Vec3;
    mass: number;
    moment: Vec3;
    size: Vec3;
}

export const toBand = (b: Body): BandBody => ({
    pos: b.posLin,
    quat: b.posAng,
    vel: b.velLin,
    angVel: b.velAng,
    mass: b.mass,
    moment: b.moment,
    size: b.size,
});

/** total mechanical energy (KE_lin + KE_ang + PE) in the solver's world-frame-diagonal-inertia
 *  convention — matches oracle.test.ts `energy`. A dropped-from-rest scene's E(0) is the supremum;
 *  a dissipative implicit solver only loses energy, so E(t) > E(0) is an injection (instability). */
export function energy(bodies: BandBody[], gravity: number): number {
    let E = 0;
    for (const b of bodies) {
        if (b.mass <= 0) continue;
        E += 0.5 * b.mass * dot(b.vel, b.vel);
        const [wx, wy, wz] = b.angVel;
        E += 0.5 * (b.moment[0] * wx * wx + b.moment[1] * wy * wy + b.moment[2] * wz * wz);
        E += b.mass * -gravity * b.pos[1];
    }
    return E;
}

const boxOf = (b: BandBody): Box => ({ pos: b.pos, quat: b.quat, size: b.size });

// the box's lowest world point: center.y minus the half-box's projection onto −y (Σ |rotated half-axis · ŷ|).
function bottom(b: BandBody): number {
    const ex = rotate(b.quat, [b.size[0] / 2, 0, 0]);
    const ey = rotate(b.quat, [0, b.size[1] / 2, 0]);
    const ez = rotate(b.quat, [0, 0, b.size[2] / 2]);
    return b.pos[1] - (Math.abs(ex[1]) + Math.abs(ey[1]) + Math.abs(ez[1]));
}

/** the deepest box-box / box-ground overlap across all dynamic-involving pairs — geometric
 *  penetration depth (−gap along each contact normal). The reference SAT (collide.ts) is the
 *  narrowphase, so this reads exactly the overlap the solver resolves. At a converged rest it sits
 *  at ~COLLISION_MARGIN + mg/k (~0.02); under-convergence sinks it deeper. */
function penetration(bodies: BandBody[]): number {
    let max = 0;
    for (let ia = bodies.length - 1; ia >= 1; ia--) {
        for (let ib = ia - 1; ib >= 0; ib--) {
            const A = bodies[ia];
            const B = bodies[ib];
            if (A.mass <= 0 && B.mass <= 0) continue;
            const { contacts, basis } = collide(boxOf(A), boxOf(B));
            for (const c of contacts) {
                const xA = transform(A.pos, A.quat, c.rA);
                const xB = transform(B.pos, B.quat, c.rB);
                max = Math.max(max, -dot(basis[0], sub(xA, xB)));
            }
        }
    }
    return max;
}

/** a per-state band reading — `finite` (no NaN/Inf), `maxSpeed` (the settle premise), `minBottom`
 *  (tunnel-through guard), `maxPenetration` (overlap bound). The caller aggregates: the oracle
 *  tracks the worst over the trajectory, the gym `pile` reads the settled snapshot + energy samples. */
export interface BandState {
    finite: boolean;
    maxSpeed: number;
    minBottom: number;
    maxPenetration: number;
}

export function bandState(bodies: BandBody[]): BandState {
    let finite = true;
    let maxSpeed = 0;
    let minBottom = Infinity;
    for (const b of bodies) {
        if (b.mass <= 0) continue;
        if (![...b.pos, ...b.vel, ...b.angVel, ...b.quat].every(Number.isFinite)) finite = false;
        maxSpeed = Math.max(maxSpeed, length(b.vel));
        minBottom = Math.min(minBottom, bottom(b));
    }
    return { finite, maxSpeed, minBottom, maxPenetration: penetration(bodies) };
}
