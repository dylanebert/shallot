import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "@dylanebert/shallot/testing";
import { compileSubjects, measureCompile } from "./compile.fixture";

setDefaultTimeout(CEILING.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();
const pipelineCounts: Record<string, number> = {
    "engine-only": 0,
    "core rendering": 1, // Stage 6 moved the cluster-grid, light-compact and light-cull pipelines to standard.
    Physics: 0,
    "Physics with core rendering": 1, // Stage 6 moved the three clustered-light pipelines to standard.
    "Character gameplay": 0,
    "Player gameplay": 0, // Player authors a camera pose; presenting it belongs to the app's rendering composition.
    "Physics profiling": 0,
    Orbit: 0,
    profiling: 0,
    "default plugins": 23, // Unlit became a StandardMaterial branch; it no longer compiles its own
    // color, prepass, point-shadow or cascade-shadow pipeline.
    Fog: 25, // single-sample and multisampled march
    Outline: 28, // three mask variants, jump flood and composite
    Vignette: 24,
};

for (const subject of compileSubjects) {
    test(`${subject.name} compiles exactly ${pipelineCounts[subject.name]} native pipelines`, async () => {
        const result = await measureCompile(subject.config);
        expect(result.pipelines).toBe(pipelineCounts[subject.name]);
    });
}
