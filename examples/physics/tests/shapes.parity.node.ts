import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, hashPhysics, Shape } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import {
    type Controls,
    changeControl,
    controlValues,
    createScene,
    type Scene,
    scenes,
} from "../src/scenes";
import { ShapesSamplePlugin, shapeScenes } from "../src/shapes";
import { sampleGold } from "./gold";

setDefaultTimeout(TIMEOUT.node);
for (const scene of Object.keys(shapeScenes) as Scene[]) {
    if (scenes[scene].controls)
        test(`${scene} equals every native step through every control`, async () => {
            const start = performance.now();
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
            const values = controlValues(scene);
            const used = new Set<string>();
            let bodies: number[] = [];
            const app = await createApp({
                defaults: false,
                plugins: [ShapesSamplePlugin],
                setup(world) {
                    bodies = createScene(world, scene);
                },
            });
            try {
                expect(gold).toHaveLength(600);
                for (let step = 0; step < gold.length; step++) {
                    for (const change of sequence.filter((change) => change.step === step)) {
                        const controls = scenes[scene].controls as Controls;
                        const row = Object.entries(controls).find(([, row]) =>
                            row.options?.some((option) => option.label === change.name),
                        );
                        const name = row?.[0] ?? change.name;
                        const value =
                            row?.[1].options?.find((option) => option.label === change.name)
                                ?.value ?? change.value;
                        changeControl(app.world, scene, bodies, values, name, value);
                        used.add(name);
                    }
                    app.world.tick();
                    expect(hashPhysics(app.world), `step ${step} hash`).toBe(gold[step]!.hash);
                }
                expect([...used].sort()).toEqual(Object.keys(scenes[scene].controls!).sort());
            } finally {
                app.dispose();
            }
            console.log(`${scene} controls: ${(performance.now() - start).toFixed(0)} ms`);
        });
    test(`${scene} equals every native step and rejects a density change`, async () => {
        const gold = sampleGold(
            await Bun.file(new URL(`./${scene}.gold.txt`, import.meta.url)).text(),
        );
        expect(gold).toHaveLength(600);
        const start = performance.now();
        for (const mutated of [false, true]) {
            let bodies: number[] = [];
            const app = await createApp({
                defaults: false,
                plugins: [ShapesSamplePlugin],
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
        console.log(`${scene}: ${(performance.now() - start).toFixed(0)} ms`);
    });
}
