import { expect, setDefaultTimeout, test } from "bun:test";
import { compileSubjects, measureCompile } from "./compile.fixture";

import { CEILING } from "./test-tiers";

setDefaultTimeout(CEILING.node);
const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();
const pipelineCounts: Record<string, number> = {
    "engine-only": 0,
    "core rendering": 4,
    Physics: 0,
    "Physics with core rendering": 4,
    "Character gameplay": 0,
    "Player gameplay": 4,
    "Physics profiling": 0,
    Orbit: 0,
    profiling: 0,
    "default plugins": 25,
};

for (const subject of compileSubjects) {
    test(`${subject.name} compiles exactly ${pipelineCounts[subject.name]} native pipelines`, async () => {
        const result = await measureCompile(subject.config);
        expect(result.pipelines).toBe(pipelineCounts[subject.name]);
    });
}
