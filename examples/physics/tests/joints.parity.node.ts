import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, hashPhysics, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import {
    type Controls,
    changeControl,
    controlValues,
    createScene,
    type Scene,
    scenes,
} from "../src/scenes";

import { sampleGold } from "./gold";

setDefaultTimeout(TIMEOUT.node);

for (const scene of ["revolute", "prismatic", "wheel", "filter", "parallel"] satisfies Scene[]) {
    for (const controlled of [false, true]) {
        test(`${scene} equals every native step ${controlled ? "through all controls" : "by default"}`, async () => {
            const start = performance.now();
            const gold = sampleGold(
                await Bun.file(
                    new URL(
                        `./${scene}${controlled && scene !== "filter" ? ".controls" : ""}.gold.txt`,
                        import.meta.url,
                    ),
                ).text(),
            );
            const sequence =
                controlled && scene !== "filter"
                    ? (await Bun.file(new URL(`./${scene}.controls.txt`, import.meta.url)).text())
                          .trim()
                          .split("\n")
                          .map((line) => {
                              const [step, name, value] = line.split("\t");
                              return { step: Number(step), name: name!, value: Number(value) };
                          })
                    : [];
            const values = controlValues(scene);
            let bodies: number[] = [];
            const app = await createApp({
                defaults: false,
                plugins: [StandardPhysicsPlugin],
                setup(world) {
                    bodies = createScene(world, scene);
                },
            });
            try {
                const used = new Set<string>();
                for (let step = 0; step < gold.length; step++) {
                    for (const change of sequence)
                        if (change.step === step) {
                            const controls: Controls | undefined = scenes[scene].controls;
                            const control = controls?.[change.name];
                            expect(control).toBeDefined();
                            for (const name of control?.when ?? []) expect(values[name]).toBe(1);
                            changeControl(
                                app.world,
                                scene,
                                bodies,
                                values,
                                change.name,
                                change.value,
                            );
                            used.add(change.name);
                        }
                    app.world.tick();
                    expect(hashPhysics(app.world), `step ${step} hash`).toBe(gold[step]!.hash);
                }
                if (controlled)
                    expect([...used].sort()).toEqual(
                        Object.keys(scenes[scene].controls ?? {}).sort(),
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
