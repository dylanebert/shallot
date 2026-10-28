import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, hashPhysics, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { BodySamplePlugin, bodyScenes } from "../src/bodies";
import { type Controls, changeControl, controlValues, createScene, scenes } from "../src/scenes";
import { sampleGold } from "./gold";

setDefaultTimeout(TIMEOUT.node);
for (const scene of Object.keys(bodyScenes) as (keyof typeof bodyScenes)[]) {
    for (const controlled of Object.values(scenes[scene].controls ?? {}).some(
        (control) => !control.read,
    )
        ? [false, true]
        : [false]) {
        test(`${scene} equals every native step${controlled ? " through all controls" : ""}`, async () => {
            const start = performance.now();
            const gold = sampleGold(
                await Bun.file(
                    new URL(`./${scene}${controlled ? ".controls" : ""}.gold.txt`, import.meta.url),
                ).text(),
            );
            const sequence = controlled
                ? (await Bun.file(new URL(`./${scene}.controls.txt`, import.meta.url)).text())
                      .trim()
                      .split("\n")
                      .map((line) => {
                          const [step, name, value] = line.split("\t");
                          return { step: Number(step), name: name!, value: Number(value) };
                      })
                : [];
            const values = controlValues(scene);
            const used = new Set<string>();
            let bodies: number[] = [];
            const app = await createApp({
                defaults: false,
                plugins: [StandardPhysicsPlugin, BodySamplePlugin],
                setup(world) {
                    bodies = createScene(world, scene);
                },
            });
            try {
                expect(gold).toHaveLength(600);
                for (let step = 0; step < gold.length; step++) {
                    for (const change of sequence)
                        if (change.step === step) {
                            const option =
                                scene === "bodyType"
                                    ? scenes.bodyType.controls.Type.options.find(
                                          (option) => option.label === change.name,
                                      )
                                    : undefined;
                            const name = option ? "Type" : change.name;
                            changeControl(
                                app.world,
                                scene,
                                bodies,
                                values,
                                name,
                                option?.value ?? change.value,
                            );
                            used.add(name);
                        }
                    app.world.tick();
                    expect(hashPhysics(app.world), `step ${step} hash`).toBe(gold[step]!.hash);
                }
                if (controlled)
                    expect([...used].sort()).toEqual(
                        Object.keys(scenes[scene].controls as Controls).sort(),
                    );
            } finally {
                app.dispose();
            }
            console.log(
                `${scene}${controlled ? " controls" : " default"}: ${(performance.now() - start).toFixed(0)} ms; cheap budget 250 ms`,
            );
        });
    }
}
