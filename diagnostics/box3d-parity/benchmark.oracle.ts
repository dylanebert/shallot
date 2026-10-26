import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, hashPhysics, Shape, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { benchmarkScenes } from "../../examples/physics/src/benchmarks";
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
            let bodies: number[] = [];
            const app = await createApp({
                defaults: false,
                plugins: [StandardPhysicsPlugin],
                setup(world) { bodies = createScene(world, scene); },
            });
            try {
                if (mutated) app.world.storage(Shape).density.set(bodies.at(-1)!, 101);
                let diverged = false;
                for (const [step, expected] of gold.entries()) {
                    app.world.tick();
                    const hash = hashPhysics(app.world);
                    if (!mutated) expect(hash, `step ${step} hash`).toBe(expected.hash);
                    else if (hash !== expected.hash) { diverged = true; break; }
                }
                if (mutated) expect(diverged).toBe(true);
            } finally { app.dispose(); }
        }
        console.log(`${scene}: ${(performance.now() - start).toFixed(0)} ms; ${gold.length} recorded steps; ${text.length} gold bytes; 1 solver thread`);
    });
}
