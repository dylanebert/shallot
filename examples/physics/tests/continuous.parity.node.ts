import { expect, setDefaultTimeout, test } from "bun:test";
import { Body, createApp, hashPhysics } from "@dylanebert/shallot";
import { PhysicsWorldDefinition } from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import {
    ContinuousSamplePlugin,
    ContinuousState,
    ContinuousStepObserver,
    continuousScenes,
} from "../src/continuous";
import { sceneSwitcher } from "../src/sample";
import { changeControl, controlValues, createScene } from "../src/scenes";
import { sampleGold } from "./gold";

setDefaultTimeout(TIMEOUT.node);
for (const scene of Object.keys(continuousScenes) as (keyof typeof continuousScenes)[]) {
    test(`${scene} equals all 600 native steps and rejects a gravity change`, async () => {
        const gold = sampleGold(
            await Bun.file(new URL(`./${scene}.gold.txt`, import.meta.url)).text(),
        );
        expect(gold).toHaveLength(600);
        const start = performance.now();
        for (const mutated of [false, true]) {
            let bodies: number[] = [];
            const app = await createApp({
                defaults: false,
                plugins: [ContinuousSamplePlugin],
                setup(world) {
                    world.resource(ContinuousState).seed = 12345;
                    bodies = createScene(world, scene);
                },
            });
            try {
                if (mutated) app.world.storage(Body).gravityScale.set(bodies[0]!, 2);
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
        console.log(`${scene} parity: ${(performance.now() - start).toFixed(0)} ms`);
    });
}
for (const [scene, recording, budget] of [
    ["bulletVersusStack", "bulletVersusStack", false],
    ["meshDrop", "meshDrop", true],
    ["meshDrop", "meshDrop.regeneration", false],
] as const) {
    test(`${recording} controls equal every native world step through DrawControls`, async () => {
        const gold = sampleGold(
            await Bun.file(new URL(`./${recording}.controls.gold.txt`, import.meta.url)).text(),
        );
        const changes = (
            await Bun.file(new URL(`./${recording}.controls.txt`, import.meta.url)).text()
        )
            .trim()
            .split("\n")
            .map((line) => {
                const [step, name, value] = line.split("\t");
                return { step: Number(step), name: name!, value: Number(value) };
            });
        const start = performance.now();
        const observed: bigint[] = [];
        const controlled = await createApp({
            defaults: false,
            plugins: [ContinuousSamplePlugin],
            setup(world) {
                const state = world.resource(ContinuousState);
                // Sleeping drops regenerate and reset the budget; keep this run moving past it.
                if (budget) world.resource(PhysicsWorldDefinition).enableSleep = false;
                state.seed = 12345;
                world.resource(ContinuousStepObserver).observe = (world) =>
                    observed.push(hashPhysics(world));
                createScene(world, scene);
            },
        });
        try {
            const values = controlValues(scene);
            for (let tick = 0; tick < 600; tick++) {
                for (const change of changes)
                    if (change.step === tick)
                        changeControl(
                            controlled.world,
                            scene,
                            [],
                            values,
                            change.name,
                            change.value,
                        );
                controlled.world.tick();
                if (budget && tick === 269) {
                    expect(controlled.world.resource(ContinuousState).steps).toBeGreaterThan(1000);
                    expect(controlled.world.resource(ContinuousState).paused).toBe(true);
                }
                if (scene !== "meshDrop") observed.push(hashPhysics(controlled.world));
            }
            if (budget) {
                const state = controlled.world.resource(ContinuousState);
                expect(state.paused).toBe(true);
                expect(state.failure).toBe(true);
                expect(state.auto).toBe(false);
                expect(state.steps).toBe(0); // Generate after the budget preserves the pause.
            }
            expect(observed).toHaveLength(gold.length);
            for (const [step, expected] of gold.entries())
                expect(observed[step], `world step ${step}`).toBe(expected.hash);
        } finally {
            controlled.dispose();
        }
        console.log(`${recording} controls parity: ${(performance.now() - start).toFixed(0)} ms`);
    });
}
test("Stall Launch replaces the bullet through the sample's DrawControls sequence", async () => {
    const gold = sampleGold(
        await Bun.file(new URL("./stall.controls.gold.txt", import.meta.url)).text(),
    );
    const sequence = (await Bun.file(new URL("./stall.controls.txt", import.meta.url)).text())
        .trimEnd()
        .split("\n")
        .map((line) => Number(line.split("\t")[0]));
    const app = await createApp({ defaults: false, plugins: [ContinuousSamplePlugin] });
    try {
        const bodies = sceneSwitcher(app.world)("stall");
        const values = controlValues("stall");
        for (const [step, expected] of gold.entries()) {
            if (sequence.includes(step))
                changeControl(app.world, "stall", bodies, values, "Launch", 1);
            app.world.tick();
            expect(hashPhysics(app.world), `step ${step} hash`).toBe(expected.hash);
        }
    } finally {
        app.dispose();
    }
});
