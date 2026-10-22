import { expect, setDefaultTimeout, test } from "bun:test";
import { RenderingPlugin } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { compileSubjects, measureCompile } from "./compile.fixture";

setDefaultTimeout(TIMEOUT.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();
// Rendering compositions prepare one upscaling pipeline for the default auto pixel ratio.
const pipelineCounts: Record<string, number> = {
    "engine-only": 0,
    "core rendering": 2, // Stage 6 moved the cluster-grid, light-compact and light-cull pipelines to standard.
    Physics: 0,
    "Physics with core rendering": 2, // Stage 6 moved the three clustered-light pipelines to standard.
    "Character gameplay": 0,
    "Physics profiling": 0,
    Orbit: 0,
    profiling: 0,
    "default plugins": 24, // Unlit became a StandardMaterial branch; it no longer compiles its own
    // color, prepass, point-shadow or cascade-shadow pipeline.
    "producer table": 27,
    Fog: 33, // single-sample and multisampled column depth, grid lighting and application; filter and integration
    Outline: 29, // three mask variants, jump flood and composite
    EffectStack: 25,
    Sprite: 29,
};

for (const subject of compileSubjects) {
    test(`${subject.name} compiles exactly ${pipelineCounts[subject.name]} native pipelines`, async () => {
        const labels: string[] = [];
        const result = await measureCompile(subject.config, (label) => labels.push(label));
        expect(result.pipelines).toBe(pipelineCounts[subject.name]);
        // Every nonempty composition here enables rendering with the default auto pixel ratio.
        expect(labels.filter((label) => label === "upscaling")).toHaveLength(
            result.pipelines ? 1 : 0,
        );
    });
}

test("core rendering's automatic-ratio extra pipeline is exactly upscaling", async () => {
    const autoLabels: string[] = [];
    const fixedLabels: string[] = [];
    const config = { defaults: false, plugins: [RenderingPlugin] };
    const auto = await measureCompile(config, (label) => autoLabels.push(label));
    const fixed = await measureCompile({ ...config, pixelRatio: 1 }, (label) =>
        fixedLabels.push(label),
    );
    expect(auto.pipelines - fixed.pipelines).toBe(1);
    expect(autoLabels.filter((label) => label === "upscaling")).toEqual(["upscaling"]);
    expect(autoLabels.filter((label) => label !== "upscaling")).toEqual(fixedLabels);
    expect(fixedLabels).not.toContain("upscaling");
});
