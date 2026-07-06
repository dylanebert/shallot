import { describe, expect, test } from "bun:test";
import { type Box, type Contact, collide } from "./collide";
import { add, type Quat, scale, sub, transform, type Vec3 } from "./math";
import gold from "./sat-gold-vectors.json";

// The SAT crux gate (scratch.md "AVBD rebuild" → Distributed cruxes). The gold
// vectors are the box-box manifolds the reference `Manifold::collide` produces
// (gen-sat-gold.ts → reference/avbd-demo3d/gold-sat.cpp); collide.ts is the faithful
// f64 port and must reproduce them. The configs cover all four code paths: FACE_A /
// FACE_B reference manifolds, the edge-edge closest-segment contact, and separated.
//
// Tolerance: the only divergence is f32 (C++) vs f64 (oracle) arithmetic on byte-
// identical inputs (the gold poses are stored at full f32 precision). The SAT runs a
// chain of ~30 ops on magnitudes ≲10; f32 ε≈6e-8 ⇒ worst-case accumulated abs error
// ≈ 30·10·6e-8 ≈ 2e-5, plus 2.5-ULP normalize/divide slack. 1e-4 is a ~5× margin —
// derived, not tuned (the test also prints the observed max so drift is visible).

interface GoldContact {
    feature: number;
    rA: [number, number, number];
    rB: [number, number, number];
}
interface GoldBody {
    size: number[];
    pos: number[];
    quat: number[];
    vel: number[];
}
interface GoldConfig {
    name: string;
    a: GoldBody;
    b: GoldBody;
    numContacts: number;
    basis: number[] | null;
    contacts: GoldContact[];
}

const TOL = 1e-4;
// the C++ harness solver's default dt (gold-sat builds a Solver, which sets dt = 1/60); the velocity
// sweep (Phase 4.8.4) reads dRel = (velA − velB)·dt, so the oracle must use the same dt to match.
const DT = 1 / 60;
const box = (s: GoldBody): Box => ({
    size: s.size as Vec3,
    pos: s.pos as Vec3,
    quat: s.quat as Quat,
});
// the relative displacement over the step — the velocity sweep input. Zero for the static configs.
const dRel = (cfg: GoldConfig): Vec3 => scale(sub(cfg.a.vel as Vec3, cfg.b.vel as Vec3), DT);

let maxErr = 0;
const near = (got: number, want: number): void => {
    maxErr = Math.max(maxErr, Math.abs(got - want));
    expect(Math.abs(got - want)).toBeLessThan(TOL);
};

describe("box-box SAT vs C++ gold vectors", () => {
    for (const cfg of gold.configs as GoldConfig[]) {
        test(cfg.name, () => {
            const { contacts, basis } = collide(box(cfg.a), box(cfg.b), dRel(cfg));

            // count is exact — a different count means a different separating axis / clip
            expect(contacts.length).toBe(cfg.numContacts);
            if (cfg.numContacts === 0) return;

            // basis: row-major 9 floats
            const gb = cfg.basis as number[];
            for (let r = 0; r < 3; r++)
                for (let c = 0; c < 3; c++) near(basis[r][c], gb[r * 3 + c]);

            // match each oracle contact to its gold twin by feature key (order-independent),
            // then compare the local arms. Feature keys must be bit-identical.
            for (const want of cfg.contacts) {
                const got = contacts.find((ct) => ct.feature === want.feature);
                expect(got, `missing feature 0x${(want.feature >>> 0).toString(16)}`).toBeDefined();
                if (!got) continue;
                for (let i = 0; i < 3; i++) near(got.rA[i], want.rA[i]);
                for (let i = 0; i < 3; i++) near(got.rB[i], want.rB[i]);
            }
        });
    }

    test("observed max error is well under tolerance", () => {
        console.log(`[sat] max abs error vs gold: ${maxErr.toExponential(2)} (tol ${TOL})`);
        expect(maxErr).toBeLessThan(TOL);
    });
});

// Feature-key continuity (scratch.md "AVBD rebuild" → Distributed cruxes → SAT). The key's low byte
// is the clip-vertex loop index, so a reordered Sutherland-Hodgman output silently reassigns keys —
// a contact that physically persists across a frame would get a NEW key, miss the warmstart merge,
// and jitter (not crash). This is the cross-frame precondition the phase-3 warmstart keying builds
// on, testable now without any keying scheme: perturb a resting face contact by a sub-box ε and
// assert each *surviving* contact (matched by world position) keeps its feature key.
//
// The precondition holds for an INTERIOR contact — a box resting on a larger ground, the realistic
// physics case, where the incident (box) face is fully inside the reference (ground) face, so a
// sub-box perturbation never crosses a clip boundary and the clip order is fixed. A contact whose
// corner sits ON a clip boundary (two equal boxes) *does* reorder under perturbation — the SAT is
// faithful there to the C++ reference, which simply cold-starts that contact (penalty→MIN, λ→0); it
// re-converges in a few iters, no crash. So the warmstart keying must tolerate key loss at boundaries
// (graceful cold-start), and this gate guards the interior case it can actually rely on.
// `collide` returns the SAT min-separation alongside the manifold — the signed overlap depth the gym
// `no-overlap` gate reads to flag severely interpenetrating settled bodies (penetration = −separation).
// Pin the sign + magnitude convention the gate depends on; a flipped sign would read overlap as clearance.
describe("box-box SAT separation (the overlap depth the no-overlap gate reads)", () => {
    test("axis-aligned overlap → separation is the negative penetration depth", () => {
        // unit boxes, centers 0.8 apart on x: spans [-0.5,0.5] and [0.3,1.3] overlap 0.2 → the SAT
        // min-separation = centerΔ − (halfA + halfB) = 0.8 − 1.0 = −0.2 (penetrating by 0.2).
        const a: Box = { size: [1, 1, 1], pos: [0, 0, 0], quat: [0, 0, 0, 1] };
        const b: Box = { size: [1, 1, 1], pos: [0.8, 0, 0], quat: [0, 0, 0, 1] };
        expect(collide(a, b).separation).toBeCloseTo(-0.2, 5);
    });
    test("a non-penetrating near pair → positive separation (the gate reads no penetration)", () => {
        // centers 1.3 apart → a 0.3 gap (≫ the speculative band), so collide reports a positive
        // separation; the gate's `−separation` is then ≤ 0 → contributes no penetration.
        const a: Box = { size: [1, 1, 1], pos: [0, 0, 0], quat: [0, 0, 0, 1] };
        const b: Box = { size: [1, 1, 1], pos: [1.3, 0, 0], quat: [0, 0, 0, 1] };
        expect(collide(a, b).separation).toBeGreaterThan(0);
    });
});

describe("box-box SAT feature-key continuity under sub-box perturbation", () => {
    const baseA: Box = { size: [10, 1, 10], pos: [0, 0, 0], quat: [0, 0, 0, 1] }; // ground
    const baseB: Box = { size: [1, 1, 1], pos: [0, 0.97, 0], quat: [0, 0, 0, 1] }; // box, interior, penetrating

    // world contact location = midpoint of the two local arms transformed to world
    const worldMid = (box: Box, other: Box, c: Contact): Vec3 =>
        scale(add(transform(box.pos, box.quat, c.rA), transform(other.pos, other.quat, c.rB)), 0.5);

    const yaw = (deg: number): Quat => {
        const h = (deg * Math.PI) / 360;
        return [0, Math.sin(h), 0, Math.cos(h)];
    };

    // ε perturbations of B that keep the same face-face contact (corner movement << inter-contact spacing 1.0)
    const cases: { name: string; b: Box }[] = [
        { name: "translate +x", b: { ...baseB, pos: [0.02, 0.97, 0] } },
        { name: "translate -z", b: { ...baseB, pos: [0, 0.97, -0.02] } },
        { name: "sink deeper", b: { ...baseB, pos: [0, 0.95, 0] } },
        { name: "tiny yaw", b: { ...baseB, quat: yaw(0.5) } },
    ];

    const base = collide(baseA, baseB);

    test("base config yields a 4-point face manifold", () => {
        expect(base.contacts.length).toBe(4);
    });

    for (const cs of cases) {
        test(cs.name, () => {
            const perturbed = collide(baseA, cs.b);
            // every base contact still present (matched by world midpoint within << spacing) keeps its key
            for (const b0 of base.contacts) {
                const m0 = worldMid(baseA, baseB, b0);
                let best: Contact | null = null;
                let bestD = 0.1; // >> ε movement (~0.02), << inter-contact spacing (~1.0)
                for (const p of perturbed.contacts) {
                    const mp = worldMid(baseA, cs.b, p);
                    const d = Math.hypot(m0[0] - mp[0], m0[1] - mp[1], m0[2] - mp[2]);
                    if (d < bestD) {
                        bestD = d;
                        best = p;
                    }
                }
                expect(best, `contact at ${m0.map((x) => x.toFixed(2))} survived`).not.toBeNull();
                expect(
                    best?.feature,
                    `feature key 0x${(b0.feature >>> 0).toString(16)} preserved`,
                ).toBe(b0.feature);
            }
        });
    }
});

// Pins the CURRENT warmstart-key choice against accidental drift — not a proven-best decision. We key
// on the stable clip-loop ordinal (a body-fixed corner id), matched by (a,b)+key; webphysics re-ordinals
// the face key to the post-reduction array rank (reference/webphysics/.../contactGeneration.ts:1375).
// The two are mitigated tradeoffs (ours cold-starts a shifted key, wp false-matches a nearby λ that AVBD
// absorbs), and which is better is an OPEN 4.9 A/B (vs Jolt/Bullet PCM position-proximity too —
// roadmap "Phase 4.9 Goal C", scratch.md "Phase 4.8.5" item 4). This guards only against an *accidental*
// re-ordinal while that *deliberate* choice stays open: the continuity test above covers the 4-point
// interior manifold (no reduction); this reaches the OVER-PRODUCED clip the Jolt reduction prunes to 4,
// where each kept contact keeps its ORIGINAL clip ordinal (a non-contiguous subset), so a rank-relabel
// (which would emit exactly [0,1,2,3], the indices of the 4-element output array) goes red.
describe("box-box SAT — reduced manifold keeps clip ordinals, not post-reduction rank", () => {
    // two EQUAL unit boxes, B yawed about the contact normal (Y) and resting on A: the yawed incident
    // square pokes past A's axis-aligned reference square on all four sides, so the Sutherland-Hodgman
    // clip produces an octagon (> 4 candidates) and the reduction runs. (A larger ground would contain
    // the incident face → 4 candidates, no reduction — that is the interior case above.)
    const a: Box = { size: [1, 1, 1], pos: [0, 0, 0], quat: [0, 0, 0, 1] };
    const yawB = (deg: number, sink: number): Box => {
        const h = (deg * Math.PI) / 360;
        return { size: [1, 1, 1], pos: [0, 1 - sink, 0], quat: [0, Math.sin(h), 0, Math.cos(h)] };
    };

    test("reduction over-produces, then the kept keys are body-fixed clip ordinals (≠ array rank)", () => {
        const { contacts } = collide(a, yawB(35, 0.03));
        // the octagon reduced to the 4-point cap
        expect(contacts.length).toBe(4);

        const ordinals = contacts.map((c) => c.feature & 0xff);
        console.log(`[sat] reduced-manifold clip ordinals: [${ordinals.join(", ")}]`);

        // over-production happened: the reduction kept at least one clip index ≥ 4, impossible for a
        // bare 4-candidate manifold — so > 4 candidates existed and were pruned.
        expect(Math.max(...ordinals)).toBeGreaterThanOrEqual(4);
        // and the kept ordinals are NOT the contiguous array rank a re-ordinal would assign.
        expect(ordinals).not.toEqual([0, 1, 2, 3]);
        // every key is a real clip-loop index into the 8-vertex candidate set.
        for (const o of ordinals) expect(o).toBeLessThan(8);

        // the face prefix (high bytes: reference side | refAxis | incAxis) is shared by the manifold —
        // it is the per-contact ordinal in the low byte that distinguishes the points, so two kept
        // contacts never collide on a key.
        const keys = contacts.map((c) => c.feature >>> 0);
        expect(new Set(keys).size).toBe(keys.length);
    });
});
