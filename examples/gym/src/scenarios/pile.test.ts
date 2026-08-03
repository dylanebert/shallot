import { describe, expect, test } from "bun:test";
import { body as rigidBody } from "../../../../packages/shallot/tests/avbd/rigid";
import {
    body,
    flat,
    integerDiscipline,
    noIntegerDivision,
    pointerDiscipline,
} from "../../../../packages/shallot/tests/wgsl";
import { packRoundedConfigs, roundedCfgFloats, roundedMainWgsl } from "./pile";

describe("pile rounded narrowphase production kernel", () => {
    test("packs rounded configs into the exact typed stride", () => {
        const packed = packRoundedConfigs([
            {
                name: "rounded-address",
                a: {
                    ...rigidBody([8, 9, 10], 1, 0.5, [1, 2, 3], [0, 0, 0], [4, 5, 6, 7]),
                    shape: 1,
                    roundRadius: 0.25,
                },
                b: {
                    ...rigidBody([18, 19, 20], 2, 0.75, [11, 12, 13], [0, 0, 0], [14, 15, 16, 17]),
                    shape: 2,
                    roundRadius: 0.5,
                },
                dRel: [21, 22, 23],
            },
            {
                name: "rounded-stride",
                a: {
                    ...rigidBody(
                        [108, 109, 110],
                        3,
                        0.25,
                        [101, 102, 103],
                        [0, 0, 0],
                        [104, 105, 106, 107],
                    ),
                    shape: 2,
                    roundRadius: 0.125,
                },
                b: {
                    ...rigidBody(
                        [118, 119, 120],
                        4,
                        0.5,
                        [111, 112, 113],
                        [0, 0, 0],
                        [114, 115, 116, 117],
                    ),
                    shape: 1,
                    roundRadius: 0.875,
                },
                dRel: [121, 122, 123],
            },
        ]);

        expect(packed).toHaveLength(2 * roundedCfgFloats);
        expect(Array.from(packed)).toEqual([
            1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 10, 0.25, 11, 12, 13, 0, 14, 15, 16, 17, 18, 19, 20, 0.5,
            21, 22, 23, 0, 1, 2, 0, 0, 101, 102, 103, 0, 104, 105, 106, 107, 108, 109, 110, 0.125,
            111, 112, 113, 0, 114, 115, 116, 117, 118, 119, 120, 0.875, 121, 122, 123, 0, 2, 1, 0,
            0,
        ]);
    });

    test("emits the roundedMain body with exact output addressing", () => {
        const wgsl = roundedMainWgsl();
        expect(wgsl).toContain("fn roundedMain(");
        expect(wgsl).toContain("fn roundedHull(");
        expect(wgsl).toContain("fn roundedMix(");
        expect(wgsl).toContain("fn collideRounded(");
        expect(wgsl).toContain("fn collideRoundedPolytope(");
        expect(flat(body(wgsl, "@compute @workgroup_size(64) fn roundedMain("))).toBe(
            "@compute @workgroup_size(64) fn roundedMain" +
                "(@builtin(global_invocation_id) gid: vec3u) { let i = gid.x; " +
                "if ((i >= params.count)) { return; } let c = cfgs[i]; " +
                "let sA = u32(c.shapes.x); let sB = u32(c.shapes.y); " +
                "let roundedA = ((sA == 1u) || (sA == 2u)); " +
                "let roundedB = ((sB == 1u) || (sB == 2u)); var r = SatResult(); " +
                "if ((roundedA && roundedB)) { r = roundedHull(c.posA.xyz, c.quatA, c.sizeRadA.xyz, c.sizeRadA.w, " +
                "c.posB.xyz, c.quatB, c.sizeRadB.xyz, c.sizeRadB.w, c.dRel.xyz); } else { " +
                "r = roundedMix(c.posA.xyz, c.quatA, c.sizeRadA.xyz, c.sizeRadA.w, sA, 0u, " +
                "c.posB.xyz, c.quatB, c.sizeRadB.xyz, c.sizeRadB.w, sB, 0u, c.dRel.xyz); } " +
                "let base = (i * 32u); out[(base + 0u)] = bitcast<f32>(r.count); " +
                "let normal = (base + 1u); out[normal] = r.basis.r0.x; out[(normal + 1u)] = r.basis.r0.y; " +
                "out[(normal + 2u)] = r.basis.r0.z; for (var k = 0u; (k < 4u); k = (k + 1u)) { " +
                "let contact = ((base + 4u) + (k * 7u)); out[(contact + 0u)] = bitcast<f32>(r.feat[k]); " +
                "let rA = (contact + 1u); out[rA] = r.rA[k].x; out[(rA + 1u)] = r.rA[k].y; " +
                "out[(rA + 2u)] = r.rA[k].z; let rB = (contact + 4u); out[rB] = r.rB[k].x; " +
                "out[(rB + 1u)] = r.rB[k].y; out[(rB + 2u)] = r.rB[k].z; } }",
        );
        integerDiscipline(wgsl);
        noIntegerDivision(wgsl);
        pointerDiscipline(body(wgsl, "@compute @workgroup_size(64) fn roundedMain("));
        pointerDiscipline(body(wgsl, "fn roundedHull("));
        pointerDiscipline(body(wgsl, "fn roundedMix("));
    });
});
