import { describe, expect, test } from "bun:test";
import { type Box, collide } from "./collide";
import { boxHull, collideHull, coneHull, type Hull, tetHull } from "./hull";
import hullGold from "./hull-gold-vectors.json";
import { add, dot, length, type Quat, rotate, scale, sub, type Vec3 } from "./math";
import { body, capsule, hull } from "./rigid";
import { narrowphase } from "./rounded";
import boxGold from "./sat-gold-vectors.json";
import { makeSolver, step } from "./solver";

// Phase 6.3 hull gold. Two independent gates:
//
//   1. box-as-hull == box-box `collide`. A box IS a convex hull, so the polytope SAT (collideHull) fed
//      box-hulls must reproduce the box-box oracle on every face manifold — same normal, same world
//      contact points, same gaps. `collide` is gold-gated against avbd-demo3d (sat.test.ts), so this
//      transitively validates the whole generalization (axis search, projection, reference selection,
//      clip, reduce) against a trusted reference — exactly, on boxes. (Feature keys differ: a box-hull's
//      face indices are 0–5, the box path's axis indices 0–2; the hull path is a separate dispatch, so
//      its keys never need to match box-box. The geometry is what must match.)
//
//   2. bullet3-sat-harness for non-box hulls (tet, cone). No AVBD reference covers them; Bullet's GPU SAT
//      is the independent gold. Bullet's contact points + depths depend on its always-A reference
//      convention, so we gate on the one reference-independent invariant — the separating axis — which is
//      exactly what box-box can't exercise (gate 1 validates the manifold pipeline on boxes).

// ── gate 1: box-as-hull reproduces box-box ───────────────────────────

interface GoldBody {
    size: number[];
    pos: number[];
    quat: number[];
    vel: number[];
}
interface BoxGoldConfig {
    name: string;
    a: GoldBody;
    b: GoldBody;
    numContacts: number;
}

const DT = 1 / 60; // the harness solver dt — the velocity sweep reads (velA−velB)·dt
const box = (s: GoldBody): Box => ({
    size: s.size as Vec3,
    pos: s.pos as Vec3,
    quat: s.quat as Quat,
});
const dRelOf = (a: GoldBody, b: GoldBody): Vec3 => scale(sub(a.vel as Vec3, b.vel as Vec3), DT);

// a contact's two world surface anchors (a box has radius 0, so the arm is the bare surface point)
const worldXA = (pos: Vec3, quat: Quat, rA: Vec3): Vec3 => add(rotate(quat, rA), pos);

describe("hull SAT — box-as-hull reproduces box-box collide", () => {
    for (const cfg of boxGold.configs as BoxGoldConfig[]) {
        test(cfg.name, () => {
            const a = box(cfg.a);
            const b = box(cfg.b);
            const dRel = dRelOf(cfg.a, cfg.b);
            const ref = collide(a, b, dRel);
            const got = collideHull(
                boxHull(a.size),
                a.pos,
                a.quat,
                boxHull(b.size),
                b.pos,
                b.quat,
                dRel,
            );

            // separation status agrees
            expect(got.contacts.length > 0).toBe(ref.contacts.length > 0);
            if (ref.contacts.length === 0) return;

            // the contact normal (basis row 0, B→A) agrees exactly
            for (let i = 0; i < 3; i++) expect(got.basis[0][i]).toBeCloseTo(ref.basis[0][i], 7);

            // the face manifolds (4-point) must match point-for-point; the edge path (box-box uses a
            // closest-segment single point, the hull path face-clips) agrees on the normal only.
            if (cfg.numContacts !== 4) return;
            expect(got.contacts.length).toBe(4);

            // match each reference contact to its hull twin by nearest world anchor, then compare both
            // anchors + the signed gap (order-independent — the reduce may emit a different rotation).
            const refPts = ref.contacts.map((c) => ({
                xA: worldXA(a.pos, a.quat, c.rA),
                xB: worldXA(b.pos, b.quat, c.rB),
            }));
            const gotPts = got.contacts.map((c) => ({
                xA: worldXA(a.pos, a.quat, c.rA),
                xB: worldXA(b.pos, b.quat, c.rB),
            }));
            for (const r of refPts) {
                let best = gotPts[0];
                let bestD = Infinity;
                for (const g of gotPts) {
                    const d =
                        dot(sub(g.xA, r.xA), sub(g.xA, r.xA)) +
                        dot(sub(g.xB, r.xB), sub(g.xB, r.xB));
                    if (d < bestD) {
                        bestD = d;
                        best = g;
                    }
                }
                for (let i = 0; i < 3; i++) {
                    expect(best.xA[i]).toBeCloseTo(r.xA[i], 6);
                    expect(best.xB[i]).toBeCloseTo(r.xB[i], 6);
                }
            }
        });
    }
});

// ── gate 2: bullet3-sat-harness cross-check (tet, cone) ───────────────

interface HullGoldContact {
    point: number[];
    depth: number;
}
interface HullGoldConfig {
    name: string;
    posA: number[];
    posB: number[];
    quatA: number[];
    quatB: number[];
    separated?: boolean;
    separatingNormal?: number[];
    numReducedContacts?: number;
    contacts: HullGoldContact[];
}

// the harness hulls: makeBox(0.5,…) is half-extents 0.5 ⇒ full size [1,1,1]; tet(0.5); cone(r,h,8).
const HULLS: Record<string, [Hull, Hull]> = {
    "cube-cube-axis-overlap-x": [boxHull([1, 1, 1]), boxHull([1, 1, 1])],
    "cube-cube-rotated-edge": [boxHull([1, 1, 1]), boxHull([1, 1, 1])],
    "cube-cube-separated": [boxHull([1, 1, 1]), boxHull([1, 1, 1])],
    "tet-cube-overlap": [tetHull(0.5), boxHull([1, 1, 1])],
    "tet-tet-overlap": [tetHull(0.5), tetHull(0.5)],
    "cone8-cube-overlap": [coneHull(0.4, 1.0, 8), boxHull([1, 1, 1])],
    "cube-cube-full-face": [boxHull([1, 1, 1]), boxHull([1, 1, 1])],
    "cube-cube-asymmetric": [boxHull([1, 1, 1]), boxHull([1, 1, 1])],
    "tet-tet-rotated": [tetHull(0.5), tetHull(0.5)],
    "cone8-tet-overlap": [coneHull(0.3, 0.8, 8), tetHull(0.5)],
};

describe("hull SAT — bullet3-sat-harness cross-check (tet, cone)", () => {
    for (const cfg of hullGold as HullGoldConfig[]) {
        test(cfg.name, () => {
            const [hullA, hullB] = HULLS[cfg.name];
            const { contacts, basis } = collideHull(
                hullA,
                cfg.posA as Vec3,
                cfg.quatA as Quat,
                hullB,
                cfg.posB as Vec3,
                cfg.quatB as Quat,
            );

            if (cfg.separated) {
                expect(contacts.length).toBe(0);
                return;
            }

            // Bullet uses an always-A reference convention and reports depths along its reference *face*
            // normal; we pick the winning-hull reference (so box-as-hull matches box-box, gate 1). The
            // contact points + depths are reference-dependent and not bit-comparable across the two. The
            // reference-INDEPENDENT invariant is the separating axis: it validates the SAT axis search
            // (b3FindSeparatingAxis) on the genuinely non-box geometry (slanted tet faces, the cone's many
            // faces + edges) — the one thing box-box can't exercise (gate 1 validates the manifold pipeline
            // exactly on boxes). Bullet's sepN is B→A like our basis row 0; gate it up to sign.
            const sepN = cfg.separatingNormal as Vec3;
            expect(contacts.length).toBeGreaterThan(0);
            expect(Math.abs(dot(basis[0], sepN))).toBeCloseTo(1, 3);
        });
    }
});

// ── dispatch + capsule segment-clip + through the solver ─────────────

const Z90: Quat = [0, 0, Math.SQRT1_2, Math.SQRT1_2]; // 90° about Z: the +Y capsule axis → −X (horizontal)
const COLLISION_MARGIN = 0.01;

describe("hull narrowphase — dispatch routing", () => {
    // a box-hull body vs a box body routes through the hull SAT (collideHull) and must agree with the
    // box-box SAT on the normal + contact count — the dispatch wiring over the geometry gate 1 already pins.
    test("box-hull vs box (dispatch) agrees with box-box on the normal + count", () => {
        const a = body([1, 1, 1], 1, 0.5, [0, 0.97, 0]); // a unit box resting (shallow, interior) on a ground
        const ground = body([10, 1, 10], 0, 0.5, [0, 0, 0]);
        const groundHull = hull(boxHull([10, 1, 10]), 0, 0.5, [0, 0, 0]);
        const ref = collide(
            { pos: a.posLin, quat: a.posAng, size: a.size },
            {
                pos: ground.posLin,
                quat: ground.posAng,
                size: ground.size,
            },
        );
        const got = narrowphase(a, groundHull);
        expect(ref.contacts.length).toBe(4); // a 4-point face manifold (the realistic rest)
        expect(got.contacts.length).toBe(ref.contacts.length);
        for (let i = 0; i < 3; i++) expect(got.basis[0][i]).toBeCloseTo(ref.basis[0][i], 7);
    });

    test("hull vs hull (two box-hulls) produces a 4-point resting manifold", () => {
        const top = hull(boxHull([1, 1, 1]), 1, 0.5, [0, 0.97, 0]);
        const bottom = hull(boxHull([10, 1, 10]), 0, 0.5, [0, 0, 0]);
        const { contacts, basis } = narrowphase(top, bottom);
        expect(contacts.length).toBe(4);
        for (let i = 0; i < 3; i++) expect(basis[0][i]).toBeCloseTo([0, 1, 0][i], 6); // B→A points up
    });
});

describe("capsule segment-clip — the mid-segment case endpoint sampling misses", () => {
    // a long capsule lying horizontally over a SMALL box: the segment (x ∈ [−1.5, 1.5]) overhangs the box
    // top face (x ∈ [−0.5, 0.5]) on both ends. Endpoint sampling would anchor the contacts at the far
    // overhanging endpoints (off the face); the segment-clip clips the core to the face region, so the two
    // contacts land at the face edges (x ≈ ±0.5) over the top face — the stable rest.
    test("a capsule overhanging a small box rests on the face region, not the overhanging tips", () => {
        const cap = capsule(1.5, 0.4, 1, 0.5, [0, 0.9, 0], [0, 0, 0], Z90); // half-length 1.5, horizontal
        const box = body([1, 1, 1], 0, 0.5, [0, 0, 0]); // top face y = 0.5, spans x ∈ [−0.5, 0.5]
        const { contacts, basis } = narrowphase(cap, box);
        expect(contacts.length).toBe(2);
        for (let i = 0; i < 3; i++) expect(basis[0][i]).toBeCloseTo([0, 1, 0][i], 6); // shared up normal
        // both capsule-core anchors sit over the face (|x| ≤ 0.5 + ε), NOT at the ±1.5 overhang tips
        for (const c of contacts) {
            const core = add(rotate(cap.posAng, c.rA), cap.posLin);
            expect(Math.abs(core[0])).toBeLessThanOrEqual(0.5 + 1e-6);
        }
    });
});

describe("hull — through the solver", () => {
    // a box-hull dropped on a box ground settles at the box-box margin rest; the hull pipeline end to end
    // (the hull bounding radius broadphase → collideHull → the contact Force → BDF1 settle).
    test("a box-hull rests on a box ground at the margin rest", () => {
        const s = makeSolver([
            body([10, 1, 10], 0, 0.5, [0, 0, 0]), // static box ground, top at y = 0.5
            hull(boxHull([1, 1, 1]), 1, 0.5, [0, 3, 0]), // a unit box-hull dropped from above
        ]);
        for (let f = 0; f < 600; f++) step(s);
        const cube = s.bodies[1];
        expect(length(cube.velLin)).toBeLessThan(2e-3);
        // half-height 0.5 above the ground top (0.5), sunk a small mg/k below the margin
        expect(Math.abs(cube.posLin[1] - (0.5 + 0.5 - COLLISION_MARGIN))).toBeLessThan(3e-3);
    });

    test("two box-hulls stack on a box ground", () => {
        const s = makeSolver([
            body([10, 1, 10], 0, 0.5, [0, 0, 0]),
            hull(boxHull([1, 1, 1]), 1, 0.5, [0, 1.2, 0]),
            hull(boxHull([1, 1, 1]), 1, 0.5, [0, 2.4, 0]),
        ]);
        for (let f = 0; f < 800; f++) step(s);
        const lower = s.bodies[1];
        const upper = s.bodies[2];
        expect(length(lower.velLin)).toBeLessThan(5e-3);
        expect(length(upper.velLin)).toBeLessThan(5e-3);
        // resting heights: lower centre ≈ 1.0, upper ≈ 2.0 (each a unit cube), within a few mg/k
        expect(Math.abs(lower.posLin[1] - 1.0)).toBeLessThan(2e-2);
        expect(Math.abs(upper.posLin[1] - 2.0)).toBeLessThan(3e-2);
    });
});
