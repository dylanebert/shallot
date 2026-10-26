import { expect, test } from "bun:test";
import { createApp, hashPhysics, Shape, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import type { benchmarkScenes } from "../src/benchmarks";
import { createScene } from "../src/scenes";
import { sampleGold } from "./gold";

export function benchmarkParity(scene: keyof typeof benchmarkScenes): void {
    test(
        `${scene} equals every recorded native step and rejects a density change`,
        async () => {
            const text = await Bun.file(new URL(`./${scene}.gold.txt`, import.meta.url)).text();
            const gold = sampleGold(text);
            const start = performance.now();
            for (const mutated of [false, true]) {
                let bodies: number[] = [];
                const app = await createApp({
                    defaults: false,
                    plugins: [StandardPhysicsPlugin],
                    setup(world) {
                        bodies = createScene(world, scene);
                    },
                });
                try {
                    if (mutated) app.world.storage(Shape).density.set(bodies.at(-1)!, 101);
                    let diverged = false;
                    for (const [step, expected] of gold.entries()) {
                        app.world.tick();
                        const hash = hashPhysics(app.world);
                        if (!mutated) expect(hash, `step ${step} hash`).toBe(expected.hash);
                        else if (hash !== expected.hash) {
                            diverged = true;
                            break;
                        }
                    }
                    if (mutated) expect(diverged).toBe(true);
                } finally {
                    app.dispose();
                }
            }
            console.log(
                `${scene}: ${(performance.now() - start).toFixed(0)} ms; ${gold.length} recorded steps; ${text.length} gold bytes; 1 solver thread`,
            );
        },
        TIMEOUT.node,
    );
}
