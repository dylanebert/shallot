import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, hashPhysics } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { ManifoldPlugin, ManifoldSample, manifoldNames } from "../src/manifold";
import { controlValues, createScene } from "../src/scenes";
import { drawingDigest } from "./digest";
import { manifoldInput } from "./manifold-input";
import { manifoldPoses } from "./manifold-poses";
import { manifoldRecords } from "./manifold-records";

setDefaultTimeout(TIMEOUT.node);
for (const scene of manifoldNames)
    test(`${scene}: 600 native manifolds, caches and geometry from recorded world poses`, async () => {
        const start = performance.now();
        for (const controlled of [false, true]) {
            const stem = scene + (controlled ? ".controls" : "");
            const gold = async (suffix: string) =>
                (await Bun.file(new URL(`./${stem}.${suffix}`, import.meta.url)).text())
                    .trimEnd()
                    .split("\n");
            const hashes = await gold("gold.txt"),
                records = await gold("manifold.gold.txt");
            const poses = controlled
                ? manifoldPoses(
                      await Bun.file(new URL(`./${scene}.poses.gold.txt`, import.meta.url)).text(),
                  )
                : [];
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
                if (controlled) expect(poses).toHaveLength(600);
                for (let step = 0; step < 600; step++) {
                    for (const c of changes)
                        if (c.step === step && !c.name.startsWith("drag"))
                            manifoldInput(app.world, scene, values, ray, c.name, c.value);
                    if (controlled) app.world.resource(ManifoldSample).b = poses[step]!;
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
