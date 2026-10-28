import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, hashPhysics } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { GeometryPlugin, geometryNames } from "../src/geometry";
import { changeControl, controlValues, createScene } from "../src/scenes";
import { drawingDigest } from "./digest";
import { geometryRecords } from "./geometry-records";

setDefaultTimeout(TIMEOUT.node);
for (const scene of geometryNames)
    test(`${scene}: 600 unstepped native geometry and mass records through DrawControls`, async () => {
        const start = performance.now();
        for (const controlled of [false, true]) {
            const sequence = Bun.file(new URL(`./${scene}.controls.txt`, import.meta.url));
            if (controlled && !(await sequence.exists())) continue;
            const stem = `${scene}${controlled ? ".controls" : ""}`;
            const hashes = (await Bun.file(new URL(`./${stem}.gold.txt`, import.meta.url)).text())
                .trimEnd()
                .split("\n");
            const records = (
                await Bun.file(new URL(`./${stem}.geometry.gold.txt`, import.meta.url)).text()
            )
                .trimEnd()
                .split("\n");
            const changes = controlled
                ? (await sequence.text())
                      .trimEnd()
                      .split("\n")
                      .map((line) => {
                          const [step, name, value] = line.split("\t");
                          return { step: Number(step), name: name!, value: Number(value) };
                      })
                : [];
            const values = controlValues(scene);
            const app = await createApp({
                defaults: false,
                plugins: [GeometryPlugin],
                setup(w) {
                    createScene(w, scene);
                },
            });
            try {
                expect(hashes).toHaveLength(600);
                expect(records).toHaveLength(600);
                for (let step = 0; step < 600; step++) {
                    for (const c of changes)
                        if (c.step === step) {
                            if (c.name === "Box" || c.name === "Sphere")
                                changeControl(
                                    app.world,
                                    scene,
                                    [],
                                    values,
                                    "type",
                                    c.name === "Box" ? 0 : 1,
                                );
                            else changeControl(app.world, scene, [], values, c.name, c.value);
                        }
                    app.world.tick();
                    expect(
                        hashPhysics(app.world).toString(16).padStart(16, "0"),
                        `${stem} step ${step} hash`,
                    ).toBe(hashes[step]);
                    expect(
                        drawingDigest(geometryRecords(app.world)),
                        `${stem} step ${step} geometry`,
                    ).toBe(records[step]);
                }
            } finally {
                app.dispose();
            }
        }
        console.log(`${scene} parity: ${(performance.now() - start).toFixed(0)} ms`);
    });
