import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, hashPhysics } from "@dylanebert/shallot";
import { StandardPhysicsPlugin } from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { MeshSamplePlugin, meshScenes } from "../src/mesh";
import { type Controls, controlValues, createScene } from "../src/scenes";
import { sampleGold } from "./gold";

setDefaultTimeout(TIMEOUT.node);
for (const scene of Object.keys(meshScenes) as (keyof typeof meshScenes)[]) {
    for (const controlled of meshScenes[scene].controls ? [false, true] : [false]) {
        test(`${scene}${controlled ? " controls" : ""} equals every native step`, async () => {
            const gold = sampleGold(
                await Bun.file(
                    new URL(`./${scene}${controlled ? ".controls" : ""}.gold.txt`, import.meta.url),
                ).text(),
            );
            expect(gold).toHaveLength(600);
            const sequence = controlled
                ? (await Bun.file(new URL(`./${scene}.controls.txt`, import.meta.url)).text())
                      .trim()
                      .split("\n")
                      .map((line) => {
                          const [step, name, value] = line.split("\t");
                          return { step: Number(step), name: name!, value: Number(value) };
                      })
                : [];
            const start = performance.now();
            const app = await createApp({
                defaults: false,
                plugins: [StandardPhysicsPlugin, MeshSamplePlugin],
                setup(world) {
                    createScene(world, scene);
                },
            });
            const values = controlValues(scene);
            try {
                for (const [step, expected] of gold.entries()) {
                    for (const change of sequence.filter((c) => c.step === step)) {
                        const type = ["Sphere", "Capsule", "Box", "Cylinder"].indexOf(change.name);
                        const name = type < 0 ? change.name : "Shape";
                        const value = type < 0 ? change.value : type;
                        values[name] = value;
                        (meshScenes[scene].controls as Controls)[name]!.set!(
                            app.world,
                            [],
                            value,
                            values,
                        );
                    }
                    app.world.tick();
                    expect(hashPhysics(app.world), `step ${step}`).toBe(expected.hash);
                }
            } finally {
                app.dispose();
            }
            console.log(
                `${scene}${controlled ? " controls" : ""} 600-step parity: ${(performance.now() - start).toFixed(0)} ms`,
            );
        });
    }
}
