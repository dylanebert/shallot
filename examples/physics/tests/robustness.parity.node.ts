import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, hashPhysics, Shape } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { RobustnessSamplePlugin, robustnessScenes } from "../src/robustness";
import { sceneSwitcher } from "../src/sample";
import { changeControl, controlValues, createScene, type Scene, scenes } from "../src/scenes";
import { sampleGold } from "./gold";

setDefaultTimeout(TIMEOUT.node);
for (const scene of Object.keys(robustnessScenes) as (keyof typeof robustnessScenes)[]) {
    test(`${scene} equals every native step and rejects a density change`, async () => {
        const gold = sampleGold(
            await Bun.file(new URL(`./${scene}.gold.txt`, import.meta.url)).text(),
        );
        const readouts =
            scene === "overflowColorPile"
                ? (
                      await Bun.file(
                          new URL("./overflowColorPile.readouts.gold.txt", import.meta.url),
                      ).text()
                  )
                      .trim()
                      .split("\n")
                      .map((line) => line.split(" ").map(Number))
                : [];
        const start = performance.now();
        for (const mutated of [false, true]) {
            let bodies: number[] = [];
            const app = await createApp({
                defaults: false,
                plugins: [RobustnessSamplePlugin],
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
                    if (!mutated) {
                        expect(hash, `step ${step}`).toBe(expected.hash);
                        if (scene === "overflowColorPile")
                            expect(
                                Object.values(scenes.overflowColorPile.controls).map((row) =>
                                    row.read(app.world),
                                ),
                                `readouts ${step}`,
                            ).toEqual(readouts[step]);
                    } else if (hash !== expected.hash) {
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

test("Overlap Recovery controls rebuild through DrawControls' native sequence", async () => {
    const gold = sampleGold(
        await Bun.file(new URL("./overlapRecovery.controls.gold.txt", import.meta.url)).text(),
    );
    const sequence = (
        await Bun.file(new URL("./overlapRecovery.controls.txt", import.meta.url)).text()
    )
        .trim()
        .split("\n")
        .map((line) => {
            const [step, name, value] = line.split("\t");
            return { step: Number(step), name: name!, value: Number(value) };
        });
    const app = await createApp({ defaults: false, plugins: [RobustnessSamplePlugin] });
    try {
        const scene: Scene = "overlapRecovery";
        const bodies = sceneSwitcher(app.world)(scene);
        const values = controlValues(scene);
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
