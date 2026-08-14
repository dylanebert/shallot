import { describe, expect, test } from "bun:test";
import { body as rigidBody } from "../../../../packages/shallot/tests/avbd/rigid";
import gold from "../../../../packages/shallot/tests/avbd/sat-gold-vectors.json";
import {
    body,
    flat,
    integerDiscipline,
    noIntegerDivision,
    pointerDiscipline,
} from "../../../../packages/shallot/tests/wgsl";
import {
    boxResult,
    createLateSatPipelineCache,
    diff,
    packBoxConfigs,
    packHullConfigs,
    satEntryWgsl,
    workgroupsFor,
} from "./sat";

function deferred<T>(): {
    promise: Promise<T>;
    resolve: (value: T) => void;
    reject: (cause: Error) => void;
} {
    let resolve = (_value: T) => {};
    let reject = (_cause: Error) => {};
    const promise = new Promise<T>((done, fail) => {
        resolve = done;
        reject = fail;
    });
    return { promise, resolve, reject };
}

describe("SAT production kernel CPU and address reference", () => {
    test("packs every box config into the exact seven-vec4 record", () => {
        const configs = gold.configs.slice(0, 2);
        const packed = packBoxConfigs(configs);

        expect(packed).toHaveLength(2 * 7 * 4);
        expect(Array.from(packed.slice(0, 28))).toEqual([
            ...configs[0].a.pos,
            0,
            ...configs[0].a.quat,
            ...configs[0].a.size,
            0,
            ...configs[0].b.pos,
            0,
            ...configs[0].b.quat,
            ...configs[0].b.size,
            0,
            ...configs[0].a.vel.map((v, i) => Math.fround((v - configs[0].b.vel[i]) / 60)),
            0,
        ]);
        expect(Array.from(packed.slice(28, 56))).toEqual([
            ...configs[1].a.pos,
            0,
            ...configs[1].a.quat,
            ...configs[1].a.size,
            0,
            ...configs[1].b.pos,
            0,
            ...configs[1].b.quat,
            ...configs[1].b.size,
            0,
            ...configs[1].a.vel.map((v, i) => Math.fround((v - configs[1].b.vel[i]) / 60)),
            0,
        ]);
    });

    test("the CPU-callable production leaf matches every C++ gold config", () => {
        for (const config of gold.configs) {
            const got = boxResult(config);
            expect(diff(config, got).detail, config.name).toBe("");
        }
    });

    test("packs hull records without changing shape, hull-id, or vec4 lane addresses", () => {
        const a = {
            ...rigidBody([2, 4, 6], 1, 0.25, [1, 2, 3], [4, 5, 6], [0, 0, 0, 1]),
            shape: 3,
            roundRadius: 0.25,
        };
        const b = {
            ...rigidBody([8, 10, 12], 0, 0.75, [-1, -2, -3], [-4, -5, -6], [0, 1, 0, 0]),
            shape: 2,
            roundRadius: 0.75,
        };
        const packed = packHullConfigs([
            { name: "address", a, b, ha: 17, hb: 23, dRel: [0.25, -0.5, 1] },
            { name: "stride", a: b, b: a, ha: 31, hb: 37, dRel: [-1, 0.5, -0.25] },
        ]);

        expect(packed).toHaveLength(2 * 8 * 4);
        expect(Array.from(packed)).toEqual([
            1, 2, 3, 0, 0, 0, 0, 1, 2, 4, 6, 0.25, -1, -2, -3, 0, 0, 1, 0, 0, 8, 10, 12, 0.75, 0.25,
            -0.5, 1, 0, 3, 2, 17, 23, -1, -2, -3, 0, 0, 1, 0, 0, 8, 10, 12, 0.75, 1, 2, 3, 0, 0, 0,
            0, 1, 2, 4, 6, 0.25, -1, 0.5, -0.25, 0, 2, 3, 31, 37,
        ]);
    });

    test("dispatch coverage rounds up one 64-lane workgroup without dropping the tail", () => {
        expect([0, 1, 64, 65, 129].map(workgroupsFor)).toEqual([0, 1, 1, 2, 3]);
    });
});

describe("SAT production kernel resolution reference", () => {
    const wgsl = satEntryWgsl();

    test("resolves both typed production entries and the certified helper graph", () => {
        for (const name of [
            "fn collideBoxBox(",
            "fn collideRounded(",
            "fn collideHull(",
            "fn collideRoundedPolytope(",
            "fn satMain(",
            "fn satHull(",
        ])
            expect(wgsl).toContain(name);
        expect(wgsl).not.toContain("fn satReference(");
    });

    test("keeps exact output addressing in the two entry bodies", () => {
        expect(flat(body(wgsl, "@compute @workgroup_size(64) fn satMain("))).toBe(
            "@compute @workgroup_size(64) fn satMain" +
                "(@builtin(global_invocation_id) gid: vec3u) { let i = gid.x; " +
                "if ((i >= params.count)) { return; } let c = cfgs[i]; " +
                "let r = boxSat(c.posA.xyz, c.quatA, c.sizeA.xyz, c.posB.xyz, c.quatB, c.sizeB.xyz, c.dRel.xyz); " +
                "let base = (i * 38u); out[(base + 0u)] = bitcast<f32>(r.count); " +
                "let basis = (base + 1u); out[basis] = r.basis.r0.x; " +
                "out[(basis + 1u)] = r.basis.r0.y; out[(basis + 2u)] = r.basis.r0.z; " +
                "out[(basis + 3u)] = r.basis.r1.x; out[(basis + 4u)] = r.basis.r1.y; " +
                "out[(basis + 5u)] = r.basis.r1.z; out[(basis + 6u)] = r.basis.r2.x; " +
                "out[(basis + 7u)] = r.basis.r2.y; out[(basis + 8u)] = r.basis.r2.z; " +
                "for (var k = 0u; (k < 4u); k = (k + 1u)) { " +
                "let contact = ((base + 10u) + (k * 7u)); " +
                "out[(contact + 0u)] = bitcast<f32>(r.feat[k]); let rA = (contact + 1u); " +
                "out[rA] = r.rA[k].x; out[(rA + 1u)] = r.rA[k].y; out[(rA + 2u)] = r.rA[k].z; " +
                "let rB = (contact + 4u); out[rB] = r.rB[k].x; " +
                "out[(rB + 1u)] = r.rB[k].y; out[(rB + 2u)] = r.rB[k].z; } }",
        );
        expect(flat(body(wgsl, "@compute @workgroup_size(64) fn satHull("))).toBe(
            "@compute @workgroup_size(64) fn satHull" +
                "(@builtin(global_invocation_id) gid: vec3u) { let i = gid.x; " +
                "if ((i >= params.count)) { return; } let c = cfgs[i]; " +
                "let sA = u32(c.shapes.x); let sB = u32(c.shapes.y); " +
                "let hA = u32(c.shapes.z); let hB = u32(c.shapes.w); " +
                "let roundedA = ((sA == 1u) || (sA == 2u)); " +
                "let roundedB = ((sB == 1u) || (sB == 2u)); var r = SatResult(); " +
                "if ((roundedA && roundedB)) { " +
                "r = collideRounded(c.posA.xyz, c.quatA, c.sizeRadA.xyz, c.sizeRadA.w, " +
                "c.posB.xyz, c.quatB, c.sizeRadB.xyz, c.sizeRadB.w, c.dRel.xyz); } else { " +
                "if (!((roundedA || roundedB))) { if (((sA == 0u) && (sB == 0u))) { " +
                "r = boxSat(c.posA.xyz, c.quatA, c.sizeRadA.xyz, c.posB.xyz, c.quatB, " +
                "c.sizeRadB.xyz, c.dRel.xyz); } else { " +
                "r = collideHull(polyMake(sA, c.posA.xyz, c.quatA, c.sizeRadA.xyz, hA), " +
                "polyMake(sB, c.posB.xyz, c.quatB, c.sizeRadB.xyz, hB), c.dRel.xyz); } } else { " +
                "r = collideRoundedPolytope(c.posA.xyz, c.quatA, c.sizeRadA.xyz, c.sizeRadA.w, " +
                "sA, hA, c.posB.xyz, c.quatB, c.sizeRadB.xyz, c.sizeRadB.w, sB, hB, c.dRel.xyz); } } " +
                "let base = (i * 32u); " +
                "let hullZero = bitcast<f32>((hullData[0u] & 0u)); " +
                "out[(base + 0u)] = bitcast<f32>(r.count); let normal = (base + 1u); " +
                "out[normal] = (r.basis.r0.x + hullZero); " +
                "out[(normal + 1u)] = r.basis.r0.y; out[(normal + 2u)] = r.basis.r0.z; " +
                "for (var k = 0u; (k < 4u); k = (k + 1u)) { " +
                "let contact = ((base + 4u) + (k * 7u)); " +
                "out[(contact + 0u)] = bitcast<f32>(r.feat[k]); let rA = (contact + 1u); " +
                "out[rA] = r.rA[k].x; out[(rA + 1u)] = r.rA[k].y; out[(rA + 2u)] = r.rA[k].z; " +
                "let rB = (contact + 4u); out[rB] = r.rB[k].x; " +
                "out[(rB + 1u)] = r.rB[k].y; out[(rB + 2u)] = r.rB[k].z; } }",
        );
    });

    test("keeps integer and pointer codegen disciplined", () => {
        integerDiscipline(wgsl);
        noIntegerDivision(wgsl);
        pointerDiscipline(body(wgsl, "@compute @workgroup_size(64) fn satMain("));
        pointerDiscipline(body(wgsl, "@compute @workgroup_size(64) fn satHull("));
    });
});

describe("SAT production pipeline lifecycle", () => {
    test("same-root concurrent and later callers share one initialized pair", async () => {
        const cold = deferred<void>();
        let created = 0;
        let initialized = 0;
        const cache = createLateSatPipelineCache(
            () => `pair-${++created}`,
            async () => {
                initialized++;
                await cold.promise;
            },
        );
        const root = {};
        const first = cache(root, undefined);
        const second = cache(root, undefined);

        expect({ created, initialized }).toEqual({ created: 1, initialized: 1 });
        cold.resolve();
        expect(await Promise.all([first, second])).toEqual(["pair-1", "pair-1"]);
        expect(await cache(root, undefined)).toBe("pair-1");
        expect({ created, initialized }).toEqual({ created: 1, initialized: 1 });
    });

    test("a delayed compile failure rejects every waiter, evicts only its record, and retries", async () => {
        const cold = deferred<void>();
        let created = 0;
        let initialized = 0;
        const cache = createLateSatPipelineCache(
            () => `pair-${++created}`,
            async () => {
                if (++initialized === 1) await cold.promise;
            },
        );
        const root = {};
        const first = cache(root, undefined);
        const second = cache(root, undefined);

        const settled = Promise.allSettled([first, second]);
        expect({ created, initialized }).toEqual({ created: 1, initialized: 1 });
        cold.reject(new Error("compile failed after drain"));
        const failed = await settled;
        expect(failed.map((result) => result.status)).toEqual(["rejected", "rejected"]);
        expect(
            failed.map((result) => (result.status === "rejected" ? result.reason.message : "")),
        ).toEqual(["compile failed after drain", "compile failed after drain"]);
        expect(await cache(root, undefined)).toBe("pair-2");
        expect({ created, initialized }).toEqual({ created: 2, initialized: 2 });
    });
});
