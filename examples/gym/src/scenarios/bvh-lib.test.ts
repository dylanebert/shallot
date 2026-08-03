import { describe, expect, test } from "bun:test";
import {
    body,
    flat,
    integerDiscipline,
    noIntegerDivision,
} from "../../../../packages/shallot/tests/wgsl";
import { createPipelineCache, packRays, traceEntryWgsl } from "./bvh-lib";

function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
    let resolve = (_value: T) => {};
    const promise = new Promise<T>((done) => {
        resolve = done;
    });
    return { promise, resolve };
}

describe("BVH trace helper", () => {
    test("packs each ray into its two vec4 storage slots", () => {
        expect(
            Array.from(
                packRays([
                    { origin: [1, 2, 3], dir: [4, 5, 6] },
                    { origin: [-1, -2, -3], dir: [-4, -5, -6] },
                ]),
            ),
        ).toEqual([1, 2, 3, 0, 4, 5, 6, 0, -1, -2, -3, 0, -4, -5, -6, 0]);
    });

    test("preserves the exact trace entry addressing and result writes", () => {
        const wgsl = traceEntryWgsl();
        const entry = flat(body(wgsl, "@compute"));
        for (const helper of ["fn bvhRoot(", "fn bvhClosestHit(", "fn bvhAnyHit("])
            expect(wgsl).toContain(helper);
        expect(entry).toBe(
            "@compute @workgroup_size(64) " +
                "fn traceBvh(@builtin(global_invocation_id) gid: vec3u) { " +
                "let i = gid.x; if ((i >= params.rayCount)) { return; } " +
                "let ray = (i * 2u); let ro = rayData[ray].xyz; " +
                "let rd = rayData[(ray + 1u)].xyz; let inv = (vec3f(1) / rd); " +
                "let root = bvhRoot(countBuf[0u]); " +
                "let tMax = (1.0000000150474662e+30f + (nodes[0u].x * 0f)); " +
                "if ((params.mode == 1u)) { let occluded = bvhAnyHit(root, ro, inv, tMax); " +
                "hits[i] = vec2u(select(0u, 1u, occluded), 0u); return; } " +
                "let hit = bvhClosestHit(root, ro, inv, tMax); " +
                "hits[i] = vec2u(bitcastF32toU32(hit.t), hit.prim); }",
        );
        integerDiscipline(entry);
        noIntegerDivision(entry);
    });
});

describe("BVH trace pipeline cache", () => {
    test("same-root concurrent and later callers share one ready pipeline", async () => {
        const cold = deferred<void>();
        const created: string[] = [];
        const initialized: string[] = [];
        const cache = createPipelineCache(
            (root: { name: string }) => {
                const pipeline = `pipeline-${root.name}`;
                created.push(pipeline);
                return pipeline;
            },
            async (root, _context: object, pipeline) => {
                initialized.push(`${root.name}:${pipeline}`);
                await cold.promise;
            },
        );
        const root = { name: "a" };
        const first = cache(root, {});
        const second = cache(root, {});

        expect(created).toEqual(["pipeline-a"]);
        expect(initialized).toEqual(["a:pipeline-a"]);
        cold.resolve();
        expect(await Promise.all([first, second])).toEqual(["pipeline-a", "pipeline-a"]);
        expect(await cache(root, {})).toBe("pipeline-a");
        expect(created).toHaveLength(1);
        expect(initialized).toHaveLength(1);
    });

    test("one failed record rejects its waiters, evicts exactly itself, and retries", async () => {
        let created = 0;
        let initialized = 0;
        const cache = createPipelineCache(
            () => `pipeline-${++created}`,
            async () => {
                if (++initialized === 1) throw new Error("compile failed");
            },
        );
        const root = {};
        const first = cache(root, {});
        const second = cache(root, {});

        const failed = await Promise.allSettled([first, second]);
        expect(failed.map((result) => result.status)).toEqual(["rejected", "rejected"]);
        for (const result of failed) {
            expect(result.status === "rejected" ? result.reason.message : "").toBe(
                "compile failed",
            );
        }
        expect(await cache(root, {})).toBe("pipeline-2");
        expect({ created, initialized }).toEqual({ created: 2, initialized: 2 });
    });
});
