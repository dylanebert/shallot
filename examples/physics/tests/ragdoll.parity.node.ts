import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, hashPhysics, Shape } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { RagdollPlugin, ragdollScenes } from "../src/ragdoll";
import { changeControl, controlValues, createScene } from "../src/scenes";
import { sampleGold } from "./gold";

setDefaultTimeout(TIMEOUT.node);
for (const scene of ["ragdollBox", "ragdollMesh"] as const) {
    test(`${scene} controls equal every native step through tuning and respawn`, async () => {
        const gold = sampleGold(
            await Bun.file(new URL(`./${scene}.controls.gold.txt`, import.meta.url)).text(),
        );
        const sequence = (
            await Bun.file(new URL(`./${scene}.controls.txt`, import.meta.url)).text()
        )
            .trim()
            .split("\n")
            .map((line) => {
                const [step, name, value] = line.split("\t");
                return { step: Number(step), name: name!, value: Number(value) };
            });
        let bodies: number[] = [];
        const app = await createApp({
            defaults: false,
            plugins: [RagdollPlugin],
            setup(world) {
                bodies = createScene(world, scene);
            },
        });
        const values = controlValues(scene);
        try {
            for (const [step, expected] of gold.entries()) {
                for (const change of sequence)
                    if (change.step === step)
                        changeControl(app.world, scene, bodies, values, change.name, change.value);
                app.world.tick();
                expect(hashPhysics(app.world), `step ${step}`).toBe(expected.hash);
            }
        } finally {
            app.dispose();
        }
    });
}
for (const scene of Object.keys(ragdollScenes) as (keyof typeof ragdollScenes)[]) {
    test(`${scene} equals every native step and rejects a density change`, async () => {
        const gold = sampleGold(
            await Bun.file(new URL(`./${scene}.gold.txt`, import.meta.url)).text(),
        );
        const start = performance.now();
        for (const mutated of [false, true]) {
            let bodies: number[] = [];
            const app = await createApp({
                defaults: false,
                plugins: [RagdollPlugin],
                setup(world) {
                    bodies = createScene(world, scene);
                },
            });
            try {
                if (mutated) app.world.storage(Shape).density.set(bodies[0]!, 1001);
                let diverged = false;
                for (const [step, expected] of gold.entries()) {
                    app.world.tick();
                    const hash = hashPhysics(app.world);
                    if (!mutated) expect(hash, `step ${step}`).toBe(expected.hash);
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
        console.log(`${scene}: ${(performance.now() - start).toFixed(0)} ms`);
    });
}
