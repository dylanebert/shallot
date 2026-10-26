import { expect, setDefaultTimeout, test } from "bun:test";
import { Body, BodyType, createApp, hashPhysics, Shape, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { BenchmarkSamplePlugin, benchmarkScenes } from "../../examples/physics/src/benchmarks";
import { createScene } from "../../examples/physics/src/scenes";
import { sampleGold } from "../../examples/physics/tests/gold";

// Run by path with sample.oracle.ts. Scene parity needs one solver thread; the engine's
// small determinism scenes hold scheduling and cross-host equality.
setDefaultTimeout(0);

for (const scene of Object.keys(benchmarkScenes) as (keyof typeof benchmarkScenes)[]) {
    test(`${scene} equals every recorded native step and rejects a density change`, async () => {
        const text = await Bun.file(new URL(`../../examples/physics/tests/${scene}.gold.txt`, import.meta.url)).text();
        const gold = sampleGold(text);
        expect(gold).toHaveLength(600);
        const start = performance.now();
        for (const mutated of [false, true]) {
            const app = await createApp({
                defaults: false,
                plugins: [StandardPhysicsPlugin, BenchmarkSamplePlugin],
                setup(world) { createScene(world, scene); },
            });
            try {
                let diverged = false;
                for (const [step, expected] of gold.entries()) {
                    app.world.tick();
                    const hash = hashPhysics(app.world);
                    if (mutated && step === 0) {
                        const dynamic = [...app.world.query([Body, Shape])].find(eid => app.world.storage(Body).type.get(eid) === BodyType.Dynamic);
                        expect(dynamic).toBeDefined();
                        app.world.storage(Shape).density.set(dynamic!, 101);
                    }
                    if (!mutated) expect(hash, `step ${step} hash`).toBe(expected.hash);
                    else if (hash !== expected.hash) { diverged = true; break; }
                }
                if (mutated) expect(diverged).toBe(true);
            } finally { app.dispose(); }
        }
        console.log(`${scene}: ${(performance.now() - start).toFixed(0)} ms; ${gold.length} recorded steps; ${text.length} gold bytes; 1 solver thread`);
    });
}
