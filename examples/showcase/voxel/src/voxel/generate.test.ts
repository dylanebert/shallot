import { describe, expect, test } from "bun:test";
import {
    body,
    flat,
    integerDiscipline,
    noDivision,
} from "../../../../../packages/shallot/tests/wgsl";
import { createDensityRunner, densityWgsl } from "./generate";

describe("density kernel reference", () => {
    const wgsl = densityWgsl();

    test("resolves the complete noise, addressing, and storage graph", () => {
        for (const helper of ["grad2", "perlin2", "fbm2", "voxelIndex"]) {
            expect(wgsl).toContain(`fn ${helper}(`);
        }
        expect(flat(wgsl)).toContain(
            "@group(0) @binding(0) var<storage, read> perm_1: array<u32, 512>;",
        );
        expect(flat(wgsl)).toContain(
            "@group(1) @binding(0) var<storage, read_write> grid: array<f32, 16777216>;",
        );
        expect(wgsl.match(/var<storage/g)).toHaveLength(2);
    });

    test("preserves the exact density dispatch, evaluation, address, and write", () => {
        const kernel = flat(body(wgsl, "@compute"));
        expect(kernel).toBe(
            "@compute @workgroup_size(4, 4, 4) " +
                "fn voxelgeneratedensity(@builtin(global_invocation_id) gid: vec3u) { " +
                "let gid_1 = gid; " +
                "if ((((gid_1.x >= 256u) || (gid_1.y >= 256u)) || (gid_1.z >= 256u))) { return; } " +
                "let surface = (128f + " +
                "(fbm2((vec2f(f32(gid_1.x), f32(gid_1.z)) * 0.012000000104308128f)) * 56f)); " +
                "let field = (surface - f32(gid_1.y)); " +
                "grid[voxelIndex(gid_1.x, gid_1.y, gid_1.z)] = select(0f, 1f, (field > 0f)); }",
        );
        integerDiscipline(kernel);
        noDivision(kernel);
    });
});

interface Root {
    name: string;
}

interface RawGrid {
    name: string;
    writes: { pipeline: string; seed: number; wrapper: number }[];
}

function deferred(): { promise: Promise<void>; resolve: () => void } {
    let resolve = () => {};
    const promise = new Promise<void>((done) => {
        resolve = done;
    });
    return { promise, resolve };
}

describe("density generation lifecycle", () => {
    test("refreshes a borrowed grid per call and replaces the pipeline with the root", async () => {
        let wrapper = 0;
        const pipelines: string[] = [];
        const warms: { root: string; pipeline: string; grid: string; wrapper: number }[] = [];
        const run = createDensityRunner({
            pipeline: ({ root }: { root: Root }) => {
                const pipeline = `pipeline-${root.name}`;
                pipelines.push(pipeline);
                return pipeline;
            },
            wrapGrid: ({ grid }: { grid: RawGrid }) => ({ raw: grid, wrapper: ++wrapper }),
            async precompile({ root }, pipeline, grid) {
                warms.push({
                    root: root.name,
                    pipeline,
                    grid: grid.raw.name,
                    wrapper: grid.wrapper,
                });
            },
            dispatch(_target, pipeline, grid, seed) {
                grid.raw.writes.push({ pipeline, seed, wrapper: grid.wrapper });
            },
        });
        const rootA = { name: "a" };
        const rootB = { name: "b" };
        const gridA1: RawGrid = { name: "a1", writes: [] };
        const gridA2: RawGrid = { name: "a2", writes: [] };
        const gridB: RawGrid = { name: "b", writes: [] };

        await run({ root: rootA, device: {}, grid: gridA1 }, 11);
        await run({ root: rootA, device: {}, grid: gridA2 }, 22);
        await run({ root: rootB, device: {}, grid: gridB }, 33);

        expect(gridA1.writes).toEqual([{ pipeline: "pipeline-a", seed: 11, wrapper: 1 }]);
        expect(gridA2.writes).toEqual([{ pipeline: "pipeline-a", seed: 22, wrapper: 2 }]);
        expect(gridB.writes).toEqual([{ pipeline: "pipeline-b", seed: 33, wrapper: 3 }]);
        expect(pipelines).toEqual(["pipeline-a", "pipeline-b"]);
        expect(warms).toEqual([
            { root: "a", pipeline: "pipeline-a", grid: "a1", wrapper: 1 },
            { root: "b", pipeline: "pipeline-b", grid: "b", wrapper: 3 },
        ]);
    });

    test("serializes concurrent cold calls behind one precompile", async () => {
        const cold = deferred();
        let warms = 0;
        const writes: number[] = [];
        const run = createDensityRunner({
            pipeline: () => "pipeline",
            wrapGrid: ({ grid }: { grid: RawGrid }) => grid,
            precompile: async () => {
                warms++;
                await cold.promise;
            },
            dispatch(_target, _pipeline, _grid, seed) {
                writes.push(seed);
            },
        });
        const root = { name: "a" };
        const grid: RawGrid = { name: "grid", writes: [] };

        const first = run({ root, device: {}, grid }, 1);
        const second = run({ root, device: {}, grid }, 2);
        await Promise.resolve();
        expect(warms).toBe(1);
        expect(writes).toEqual([]);

        cold.resolve();
        await Promise.all([first, second]);
        expect(warms).toBe(1);
        expect(writes).toEqual([1, 2]);
    });

    test("a rejected cold call leaves the runner retryable", async () => {
        let warms = 0;
        const writes: number[] = [];
        const run = createDensityRunner({
            pipeline: () => "pipeline",
            wrapGrid: ({ grid }: { grid: RawGrid }) => grid,
            async precompile() {
                if (++warms === 1) throw new Error("cold compile failed");
            },
            dispatch(_target, _pipeline, _grid, seed) {
                writes.push(seed);
            },
        });
        const root = { name: "a" };
        const grid: RawGrid = { name: "grid", writes: [] };

        await expect(run({ root, device: {}, grid }, 1)).rejects.toThrow("cold compile failed");
        await run({ root, device: {}, grid }, 2);

        expect(warms).toBe(2);
        expect(writes).toEqual([2]);
    });
});
