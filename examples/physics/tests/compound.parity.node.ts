import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, hashPhysics } from "@dylanebert/shallot";
import { StandardPhysicsPlugin } from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { compoundScenes } from "../src/compound";
import { createScene, type Scene } from "../src/scenes";
import { sampleGold } from "./gold";

setDefaultTimeout(TIMEOUT.node);
for (const scene of Object.keys(compoundScenes) as (keyof typeof compoundScenes)[]) {
    test(`${scene} equals every native step`, async () => {
        const gold = sampleGold(
            await Bun.file(new URL(`./${scene}.gold.txt`, import.meta.url)).text(),
        );
        expect(gold).toHaveLength(600);
        const start = performance.now();
        const app = await createApp({
            defaults: false,
            plugins: [StandardPhysicsPlugin],
            setup(world) {
                createScene(world, scene as Scene);
            },
        });
        try {
            for (const [step, expected] of gold.entries()) {
                app.world.tick();
                expect(hashPhysics(app.world), `step ${step}`).toBe(expected.hash);
            }
        } finally {
            app.dispose();
        }
        console.log(`${scene} 600-step parity: ${(performance.now() - start).toFixed(0)} ms`);
    });
}
