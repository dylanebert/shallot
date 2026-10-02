import { expect, setDefaultTimeout, test } from "bun:test";
import { compileSubjects, measureCompile } from "./compile.fixture";

import { CEILING } from "./test-tiers";

setDefaultTimeout(CEILING.node);
const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();
const pipelineCounts: Record<string, number> = {
    "engine-only": 0,
    "core rendering": 1, // Stage 6 moved the cluster-grid, light-compact and light-cull pipelines to standard.
    Physics: 0,
    "Physics with core rendering": 1, // Stage 6 moved the three clustered-light pipelines to standard.
    "Character gameplay": 0,
    "Player gameplay": 1, // Stage 6 moved the three clustered-light pipelines out of Player's core rendering dependency.
    "Physics profiling": 0,
    Orbit: 0,
    profiling: 0,
    "default plugins": 26, // Presentation also warms the rgba8unorm after-tonemapping output pipeline.
};

for (const subject of compileSubjects) {
    test(`${subject.name} compiles exactly ${pipelineCounts[subject.name]} native pipelines`, async () => {
        const result = await measureCompile(subject.config);
        expect(result.pipelines).toBe(pipelineCounts[subject.name]);
    });
}
