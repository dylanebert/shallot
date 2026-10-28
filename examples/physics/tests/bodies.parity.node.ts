import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, hashPhysics, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { BodySamplePlugin, bodyScenes } from "../src/bodies";
import { createScene } from "../src/scenes";
import { sampleGold } from "./gold";

setDefaultTimeout(TIMEOUT.node);
for (const scene of Object.keys(bodyScenes) as (keyof typeof bodyScenes)[]) {
    test(`${scene} equals every native step`, async () => {
        const start = performance.now();
        const gold = sampleGold(
            await Bun.file(new URL(`./${scene}.gold.txt`, import.meta.url)).text(),
        );
        const app = await createApp({
            defaults: false,
            plugins: [StandardPhysicsPlugin, BodySamplePlugin],
            setup(world) {
                createScene(world, scene);
            },
        });
        try {
            expect(gold).toHaveLength(600);
            for (let step = 0; step < gold.length; step++) {
                app.world.tick();
                expect(hashPhysics(app.world), `step ${step} hash`).toBe(gold[step]!.hash);
            }
        } finally {
            app.dispose();
        }
        console.log(`${scene}: ${(performance.now() - start).toFixed(0)} ms; cheap budget 250 ms`);
    });
}
