import { expect, setDefaultTimeout, test } from "bun:test";
import { Meshes, MeshPlugin, RenderingPlugin, type World } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { compileSubjects, initialSsaoCamera, measureCompile } from "./compile.fixture";

setDefaultTimeout(TIMEOUT.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();
// Rendering compositions prepare one upscaling pipeline at every pixel ratio.
const pipelineCounts: Record<string, number> = {
    "engine-only": 0,
    "core rendering": 2, // Stage 6 moved the cluster-grid, light-compact and light-cull pipelines to standard.
    Physics: 0,
    "Physics with core rendering": 2, // Stage 6 moved the three clustered-light pipelines to standard.
    "Character gameplay": 0,
    "Physics profiling": 0,
    Orbit: 0,
    profiling: 0,
    "default plugins": 25, // Unlit became a StandardMaterial branch; it no longer compiles its own
    // color, prepass, point-shadow or cascade-shadow pipeline.
    "producer table": 28,
    ScreenSpaceAmbientOcclusion: 30, // two prepass copies, mip reduction, visibility bitmask AO and denoise
    Fog: 34, // single-sample and multisampled column depth, grid lighting and application; filter and integration
    Outline: 34, // three mask variants, jump flood, composite, reduce, half flood, distance and half composite
    EffectStack: 26,
    Sprite: 30,
};

for (const subject of compileSubjects) {
    test(`${subject.name} compiles exactly ${pipelineCounts[subject.name]} native pipelines`, async () => {
        const labels: string[] = [];
        const result = await measureCompile(subject.config, (label) => labels.push(label));
        expect(result.pipelines).toBe(pipelineCounts[subject.name]);
        // Every nonempty composition here enables rendering.
        expect(labels.filter((label) => label === "upscaling")).toHaveLength(
            result.pipelines ? 1 : 0,
        );
    });
}

test("an optional stream mesh adds one opaque color pipeline and shares depth and shadow pipelines", async () => {
    const labels: string[] = [];
    const result = await measureCompile(
        {
            plugins: [
                {
                    name: "StreamMesh",
                    dependencies: [MeshPlugin],
                    initialize(world: World) {
                        world.resource(Meshes).add({
                            name: "stream",
                            vertices: new Float32Array([
                                -1, -1, 0, 0, 0, 0, 1, 0, 1, -1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
                                1, 1,
                            ]),
                            indices: new Uint32Array([0, 1, 2]),
                            uv1: new Float32Array(6),
                        });
                    },
                },
            ],
        },
        (label) => labels.push(label),
    );
    expect(result.pipelines).toBe(pipelineCounts["default plugins"] + 1);
    expect(labels.filter((label) => label.includes("streams1"))).toEqual([
        "standard-StandardMaterial-0-streams1",
    ]);
});

for (const antialias of [0, 1]) {
    test(`an initial SSAO camera (${antialias ? "4x" : "1x"}) compiles no pipeline after the drain`, async () => {
        const labels: string[] = [];
        const result = await measureCompile(
            initialSsaoCamera(antialias),
            (label) => labels.push(label),
            true,
        );
        expect(result.pipelines).toBe(antialias ? 32 : 36);
        expect(result.latePipelines).toBe(0);
        expect(
            labels.filter(
                (label) => label === `standard-prepass-StandardMaterial-0-${antialias ? 4 : 1}x`,
            ),
        ).toHaveLength(2);
    });
}

test("core rendering prepares the same upscaling pipeline at automatic and fixed pixel ratios", async () => {
    const autoLabels: string[] = [];
    const fixedLabels: string[] = [];
    const config = { defaults: false, plugins: [RenderingPlugin] };
    const auto = await measureCompile(config, (label) => autoLabels.push(label));
    const fixed = await measureCompile({ ...config, pixelRatio: 1 }, (label) =>
        fixedLabels.push(label),
    );
    expect(fixed.pipelines).toBe(auto.pipelines);
    expect(autoLabels.filter((label) => label === "upscaling")).toEqual(["upscaling"]);
    expect(fixedLabels.filter((label) => label === "upscaling")).toEqual(["upscaling"]);
    expect(fixedLabels).toEqual(autoLabels);
});
