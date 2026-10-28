import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, hashPhysics } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { ManifoldPlugin, manifoldNames } from "../src/manifold";
import { manifoldDrag } from "../src/manifold-drag";
import { controlValues, createScene } from "../src/scenes";
import { drawingDigest } from "./digest";
import { manifoldInput } from "./manifold-input";
import { manifoldRecords } from "./manifold-records";

setDefaultTimeout(TIMEOUT.node);
for (const scene of manifoldNames)
    test(`${scene}: 600 unstepped native manifolds, caches, geometry and drag controls`, async () => {
        const start = performance.now();
        for (const controlled of [false, true]) {
            const stem = scene + (controlled ? ".controls" : "");
            const gold = async (suffix: string) =>
                (await Bun.file(new URL(`./${stem}.${suffix}`, import.meta.url)).text())
                    .trimEnd()
                    .split("\n");
            const hashes = await gold("gold.txt"),
                records = await gold("manifold.gold.txt");
            const changes = controlled
                ? (await Bun.file(new URL(`./${scene}.controls.txt`, import.meta.url)).text())
                      .trim()
                      .split("\n")
                      .map((l) => {
                          const [step, name, value] = l.split("\t");
                          return { step: Number(step), name: name!, value: Number(value) };
                      })
                : [];
            const app = await createApp({
                defaults: false,
                plugins: [ManifoldPlugin],
                setup: (w) => {
                    createScene(w, scene);
                },
            });
            const values = controlValues(scene),
                ray = { origin: { x: 0, y: 0, z: 0 }, held: false };
            try {
                expect(hashes).toHaveLength(600);
                expect(records).toHaveLength(600);
                for (let step = 0; step < 600; step++) {
                    for (const c of changes)
                        if (c.step === step)
                            manifoldInput(app.world, scene, values, ray, c.name, c.value);
                    if (controlled)
                        manifoldDrag(
                            app.world,
                            ray.origin,
                            { x: 0, y: 0, z: -20 },
                            false,
                            ray.held,
                        );
                    app.world.tick();
                    expect(
                        hashPhysics(app.world).toString(16).padStart(16, "0"),
                        `${stem} step ${step} hash`,
                    ).toBe(hashes[step]);
                    expect(
                        drawingDigest(manifoldRecords(app.world)),
                        `${stem} step ${step} manifold`,
                    ).toBe(records[step]);
                }
            } finally {
                app.dispose();
            }
        }
        console.log(`${scene} parity: ${(performance.now() - start).toFixed(0)} ms`);
    });
