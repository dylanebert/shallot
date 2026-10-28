import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, hashPhysics, Shape } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { sceneSwitcher } from "../src/sample";
import { changeControl, controlValues, scenes } from "../src/scenes";
import { WorldSamplePlugin, WorldSampleState, worldScenes } from "../src/world";
import { sampleGold } from "./gold";

setDefaultTimeout(TIMEOUT.node);
for (const scene of Object.keys(worldScenes) as (keyof typeof worldScenes)[]) {
    for (const controlled of scene === "farStack" ? [false, true] : [false]) {
        test(`${scene}${controlled ? " controls" : ""} equals native at every step and rejects a density change`, async () => {
            const stem = `${scene}${controlled ? ".controls" : ""}`;
            const gold = sampleGold(
                await Bun.file(new URL(`./${stem}.gold.txt`, import.meta.url)).text(),
            );
            const rows = (
                await Bun.file(new URL(`./${stem}.readouts.gold.txt`, import.meta.url)).text()
            )
                .trim()
                .split("\n")
                .map((row) => row.split(" ").map(Number));
            const sequence = controlled
                ? (await Bun.file(new URL("./farStack.controls.txt", import.meta.url)).text())
                      .trim()
                      .split("\n")
                      .map((line) => {
                          const [step, name, value] = line.split("\t");
                          return { step: Number(step), name: name!, value: Number(value) };
                      })
                : [];
            const start = performance.now();
            for (const mutated of controlled ? [false] : [false, true]) {
                const app = await createApp({ defaults: false, plugins: [WorldSamplePlugin] });
                try {
                    const bodies = sceneSwitcher(app.world)(scene),
                        values = controlValues(scene);
                    if (mutated) app.world.storage(Shape).density.set(bodies[0]!, 1001);
                    let diverged = false;
                    for (const [step, expected] of gold.entries()) {
                        for (const change of sequence)
                            if (change.step === step)
                                changeControl(
                                    app.world,
                                    scene,
                                    bodies,
                                    values,
                                    change.name,
                                    change.value,
                                );
                        app.world.tick();
                        const hash = hashPhysics(app.world);
                        if (mutated) {
                            if (hash !== expected.hash) {
                                diverged = true;
                                break;
                            }
                        } else {
                            expect(hash, `step ${step}`).toBe(expected.hash);
                            const state = app.world.resource(WorldSampleState);
                            const actual = [state.offset];
                            if (scene === "farStack")
                                actual.push(
                                    Number(
                                        scenes.farStack.controls.Height.read(app.world).toFixed(4),
                                    ),
                                );
                            if (scene === "farMeshDrop") actual.push(Number(state.failed));
                            expect(actual, `readouts ${step}`).toEqual(rows[step]!);
                        }
                    }
                    if (mutated) expect(diverged).toBe(true);
                } finally {
                    app.dispose();
                }
            }
            console.log(`${stem}: ${(performance.now() - start).toFixed(0)} ms`);
        });
    }
}
