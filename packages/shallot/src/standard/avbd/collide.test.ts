import { describe, expect, test } from "bun:test";
import * as d from "typegpu/data";
import { boxHull, collideHull as collideHullOracle, coneHull } from "../../../tests/avbd/hull";
import { flat, integerDiscipline, noIntegerDivision, pointerDiscipline } from "../../../tests/wgsl";
import { ShapeKind } from "../physics";
import { Hulls } from "../physics/core";
import {
    boxBoxWgsl,
    collideHull,
    collideWgsl,
    helpersWgsl,
    hullCoreWgsl,
    hullSatWgsl,
    hullWgsl,
    polyMake,
    roundedPolyWgsl,
    roundedWgsl,
    SPECULATIVE_DISTANCE,
    withCpuHullData,
} from "./collide";
import { packHulls } from "./hull";

// The narrowphase's device-free structural seam. What these gate is everything the real-GPU gates (the
// gym `sat` / `pile` kernel gates) can't say cheaply: which chunk owns which definition (a definition
// drifting between chunks is a pipeline that compiles the wrong SAT, or a missing identifier only the
// consumer that skipped a base chunk sees), the TGSL transpile hazards that look correct in the
// TypeScript (signed literals, integer division, `&let`), and the warmstart feature-key packing, which
// must stay bit-identical to the C++ (avbd.md "Reference-fidelity").

const chunks: [string, () => string][] = [
    ["helpersWgsl", helpersWgsl],
    ["boxBoxWgsl", boxBoxWgsl],
    ["roundedWgsl", roundedWgsl],
    ["hullCoreWgsl", hullCoreWgsl],
    ["hullSatWgsl", hullSatWgsl],
    ["roundedPolyWgsl", roundedPolyWgsl],
];

const defs = (wgsl: string): string[] =>
    [...wgsl.matchAll(/^(?:fn|struct)\s+([A-Za-z0-9_]+)/gm)].map((m) => m[1]);

describe("narrowphase chunks", () => {
    test("each chunk owns exactly its own definitions", () => {
        // the split IS the DXC compile budget (avbd.md "Compile discipline"): a box-only pipeline
        // splices helpers + boxBox and must get every definition it calls and nothing more
        expect(defs(helpersWgsl()).sort()).toEqual([
            "Mat3",
            "SatResult",
            "addContact",
            "closestSegments",
            "orthoBasis",
            "preferReduce",
            "pruneContacts",
            "satQConj",
            "satQRotate",
            "satZero",
        ]);
        expect(defs(boxBoxWgsl()).sort()).toEqual([
            "FaceAxes",
            "Obb",
            "Poly",
            "SatAxis",
            "absDot",
            "clip",
            "collideBoxBox",
            "edgeContact",
            "faceManifold",
            "getFaceAxes",
            "incidentAxis",
            "incidentFace",
            "obbAxis",
            "obbHalf",
            "obbOf",
            "supportEdge",
            "supportPoint",
            "testAxis",
        ]);
        expect(defs(roundedWgsl())).toEqual(["collideRounded"]);
        expect(defs(hullSatWgsl()).sort()).toEqual([
            "HullAxis",
            "clipPolyPlane",
            "collideHull",
            "supportPoly",
            "testPolyAxis",
        ]);
        // the closest-point primitives live here, not in helpers: only the polytope half calls them, so a
        // box-only or rounded-only pipeline compiles neither
        expect(defs(hullCoreWgsl())).toEqual(
            expect.arrayContaining(["BoxClosest", "closestPointBox", "Convex", "projectPoly"]),
        );
        expect(defs(helpersWgsl())).not.toContain("closestPointBox");
        expect(defs(boxBoxWgsl())).not.toContain("closestPointBox");
        expect(boxBoxWgsl()).not.toContain("closestPointBox(");
        expect(roundedWgsl()).not.toContain("closestPointBox(");
        expect(defs(roundedPolyWgsl())).toContain("collideRoundedPolytope");
    });

    test("no chunk redefines a base chunk's definition, and every pair splices", () => {
        // the shared `spliceNs` emits a shared dependency into whichever chunk resolves FIRST, so a
        // dependent chunk must force its base (otherwise a consumer gets a duplicate declaration, which
        // Dawn rejects at pipeline creation with nothing pointing at the cause)
        const all = chunks.flatMap(([, c]) => defs(c()));
        expect(all.filter((x, i) => all.indexOf(x) !== i)).toEqual([]);
        // the concrete base-forcing claim: the box SAT calls the helpers' primitives by name
        expect(boxBoxWgsl()).toContain("addContact(");
        expect(boxBoxWgsl()).toContain("pruneContacts(");
        expect(defs(boxBoxWgsl())).not.toContain("addContact");
        // and the hull SATs call the polytope substrate's, never redefining it
        expect(hullSatWgsl()).toContain("projectPoly(");
        expect(defs(hullSatWgsl())).not.toContain("projectPoly");
        expect(roundedPolyWgsl()).toContain("closestPointBox(");
        expect(defs(roundedPolyWgsl())).not.toContain("closestPointBox");
    });

    test("the recomposed chunks are duplicate-free too", () => {
        for (const [name, wgsl] of [
            ["collideWgsl", collideWgsl()],
            ["collideWgsl + hullWgsl", collideWgsl() + hullWgsl()],
        ] as [string, string][]) {
            const all = defs(wgsl);
            expect(
                all.filter((x, i) => all.indexOf(x) !== i),
                name,
            ).toEqual([]);
        }
    });

    test("no chunk emits an anonymous or suffixed definition", () => {
        // typegpu names an unnamed factory-built schema `item` and suffixes a collision (`Mat3_1`) —
        // either way a raw splice site (step.ts declares `var sat: SatResult`) references nothing
        for (const [name, chunk] of chunks)
            for (const def of defs(chunk())) {
                expect(def, `${name} emits ${def}`).not.toMatch(/^item/);
                expect(def, `${name} emits ${def}`).not.toMatch(/_\d+$/);
            }
    });

    test("the entry points a consumer calls keep their names and signatures", () => {
        // step.ts's four collide pipelines and the gym `sat` / `pile` kernels splice these by name
        expect(boxBoxWgsl()).toContain(
            "fn collideBoxBox(posA: vec3f, quatA: vec4f, sizeA: vec3f, posB: vec3f, quatB: vec4f, sizeB: vec3f, dRel: vec3f) -> SatResult",
        );
        expect(roundedWgsl()).toContain(
            "fn collideRounded(posA: vec3f, quatA: vec4f, sizeA: vec3f, radiusA: f32, posB: vec3f, quatB: vec4f, sizeB: vec3f, radiusB: f32, dRel: vec3f) -> SatResult",
        );
        expect(hullSatWgsl()).toContain(
            "fn collideHull(a: Convex, b: Convex, dRel: vec3f) -> SatResult",
        );
        expect(hullCoreWgsl()).toContain(
            "fn polyMake(shape: u32, pos: vec3f, quat: vec4f, size: vec3f, hullId: u32) -> Convex",
        );
        expect(roundedPolyWgsl()).toContain(
            "fn collideRoundedPolytope(aPos: vec3f, aQuat: vec4f, aSize: vec3f, aRad: f32, aShape: u32, aHull: u32, bPos: vec3f, bQuat: vec4f, bSize: vec3f, bRad: f32, bShape: u32, bHull: u32, dRel: vec3f) -> SatResult",
        );
        // SatResult's field names + arity are the decode contract every gate reads back
        expect(helpersWgsl()).toContain(
            "struct SatResult {\n  count: u32,\n  basis: Mat3,\n  feat: array<u32, 4>,\n  rA: array<vec3f, 4>,\n  rB: array<vec3f, 4>,\n}",
        );
    });

    test("no chunk emits a WGSL const", () => {
        // every constant folds to a literal, so a raw consumer that needs one interpolates the TS export
        // (step.ts's sphere filter reads SPECULATIVE_DISTANCE this way)
        for (const [name, chunk] of chunks) expect(chunk(), name).not.toMatch(/^const /m);
    });

    test("the speculative band folds to the exported value at every abort", () => {
        expect(SPECULATIVE_DISTANCE).toBe(0.04);
        const band = `max(${SPECULATIVE_DISTANCE}f, max(0f,`;
        // testAxis (box), testPolyAxis (hull), the two capsule bands, the sphere band
        expect(flat(boxBoxWgsl()).split(band).length - 1).toBe(2); // the abort + the clip band
        expect(flat(hullSatWgsl()).split(band).length - 1).toBe(2);
        expect(flat(roundedPolyWgsl()).split(band).length - 1).toBe(3);
        // the rounded pair takes the same max against a hoisted `closing`
        expect(flat(roundedWgsl())).toContain("let closing = max(0f, -(dot(dRel, normal)));");
        expect(flat(roundedWgsl())).toContain(`let band = max(${SPECULATIVE_DISTANCE}f, closing);`);
    });

    test("the warmstart feature keys pack exactly as the C++ does", () => {
        // a wrong shift silently breaks warmstart persistence (avbd.md "SAT feature keys are
        // bit-identical to the C++"), and nothing downstream reads as wrong — pin the packing verbatim
        expect(flat(boxBoxWgsl())).toContain("var prefix = (select(1u, 0u, referenceIsA) << 24u);");
        expect(flat(boxBoxWgsl())).toContain("prefix = (prefix | ((refAxis & 255u) << 16u));");
        expect(flat(boxBoxWgsl())).toContain("prefix = (prefix | ((incAxis & 255u) << 8u));");
        expect(flat(boxBoxWgsl())).toContain("candFeat[candN] = (prefix | (i & 255u));");
        expect(flat(boxBoxWgsl())).toContain(
            "let feature = (((2u << 24u) | ((axisA & 255u) << 8u)) | (axisB & 255u));",
        );
        expect(flat(hullSatWgsl())).toContain(
            "let featurePrefix = (((select(64u, 65u, best.isEdge) << 24u) | ((refFaceIdx & 255u) << 16u)) | ((incFaceIdx & 255u) << 8u));",
        );
        // the rounded key is the constant 0x03000000, its capsule ordinal in the low bits
        expect(flat(roundedWgsl())).toContain("cB, 50331648u)");
        expect(flat(roundedPolyWgsl())).toContain("(50331648u | hits.ord[k])");
    });

    test("only the hull-reader WGSL leaves touch the consumer's hullData binding", () => {
        // the relocatable boundary: everything above the reader duals is TGSL, so `hullData` appears in
        // exactly the seven raw leaves (a TGSL fn cannot name a global the consumer declares)
        const reads = hullCoreWgsl().match(/hullData\[/g) ?? [];
        expect(reads.length).toBe(20); // 7 header words + 3×3 vec lanes + offset + count + 2 index reads
        expect(hullSatWgsl()).not.toContain("hullData");
        expect(roundedPolyWgsl()).not.toContain("hullData");
        // and the box path never reads it at all, so a box-only pipeline needs no hull binding
        expect(collideWgsl()).not.toContain("hullData");
    });

    test("integer discipline + the integer-division audit hold on every chunk", () => {
        for (const [name, chunk] of chunks) {
            const wgsl = chunk();
            // a bare literal materializes i32 — signed arithmetic in a u32 key is a silent wrong key
            expect(() => integerDiscipline(wgsl), name).not.toThrow();
            // every `/` here divides floats (a reciprocal, a normalize, a clip parameter); the audit is
            // that none of them divides integers, which TGSL would emit as f32(a)/f32(b)
            expect(() => noIntegerDivision(wgsl), name).not.toThrow();
            expect(() => pointerDiscipline(wgsl), name).not.toThrow();
        }
    });

    test("every division divides floats — the operand audit, site by site", () => {
        // the audit is per-site, not a regex: these are the only `/`s in the narrowphase, and each
        // divides an f32 magnitude (the integer indices are all shift/mask/multiply-add)
        // per chunk, so a division that moves or one added while another goes away names its chunk:
        // helpers = closestSegments' 6 segment parameters; boxBox = 1/sqrt + the clip parameter (the hull
        // SAT repeats that pair); rounded = the contact normal; hullCore = the face-normal rescale +
        // closestPointBox's normal; roundedPoly = the closest-point parameter, the hull normal, 1/sqrt,
        // and the capsule segment-clip parameter
        const counts = Object.fromEntries(
            chunks.map(([n, c]) => [n, [...flat(c()).matchAll(/[\w)\]] \/ [\w(-]/g)].length]),
        );
        expect(counts).toEqual({
            helpersWgsl: 6,
            boxBoxWgsl: 2,
            roundedWgsl: 1,
            hullCoreWgsl: 2,
            hullSatWgsl: 2,
            roundedPolyWgsl: 4,
        });
        expect(flat(helpersWgsl())).toContain("clamp((f / e), 0f, 1f)"); // segment params
        expect(flat(hullCoreWgsl())).toContain("(diff / dist)"); // closestPointBox normal
        expect(flat(boxBoxWgsl())).toContain("(axis * (1f / sqrt(lenSq)))"); // axis normalize
        expect(flat(hullCoreWgsl())).toContain("normalize((hFaceNormalL(p.hr, f) / p.scale))");
        expect(flat(roundedPolyWgsl())).toContain("let tc = (-(d0) / dd);"); // segment clip param
    });
});

test("the packed-hull TGSL graph matches the f64 oracle on the symmetric cone manifold", () => {
    const cone = coneHull(0.4, 1, 8);
    const cube = boxHull([1, 1, 1]);
    const coneId = Hulls.register({ name: "__sat_cpu_cone8__", ...cone });
    const cubeId = Hulls.register({ name: "__sat_cpu_cube__", ...cube });
    const quat = d.vec4f(0, 0, 0, 1);
    const oracle = collideHullOracle(
        cone,
        [0, 0.8, 0],
        [0, 0, 0, 1],
        cube,
        [0, 0, 0],
        [0, 0, 0, 1],
    );
    const got = withCpuHullData(packHulls(), () =>
        collideHull(
            polyMake(ShapeKind.Hull, d.vec3f(0, 0.8, 0), quat, d.vec3f(), coneId),
            polyMake(ShapeKind.Hull, d.vec3f(), quat, d.vec3f(), cubeId),
            d.vec3f(),
        ),
    );

    expect(got.count).toBe(oracle.contacts.length);
    const byFeature = new Map<number, number>();
    for (let i = 0; i < got.count; i++) byFeature.set(got.feat[i] >>> 0, i);
    expect([...byFeature.keys()].sort((a, b) => a - b)).toEqual(
        oracle.contacts.map((contact) => contact.feature >>> 0).sort((a, b) => a - b),
    );
    for (const contact of oracle.contacts) {
        const i = byFeature.get(contact.feature >>> 0);
        expect(i).toBeDefined();
        for (let lane = 0; lane < 3; lane++) {
            expect(got.rA[i as number][lane]).toBeCloseTo(contact.rA[lane], 6);
            expect(got.rB[i as number][lane]).toBeCloseTo(contact.rB[lane], 6);
        }
    }
});

test("the packed-hull CPU data scope rejects asynchronous callbacks", () => {
    expect(() => withCpuHullData(new Uint32Array(), () => Promise.resolve())).toThrow(
        "withCpuHullData callback must be synchronous",
    );
});
