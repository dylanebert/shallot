import { expect, setDefaultTimeout, test } from "bun:test";
import { Body, createApp, hashPhysics } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { ContinuousSamplePlugin, continuousScenes } from "../src/continuous";
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
