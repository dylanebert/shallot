import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, hashPhysics, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { CharacterSample, CharacterSamplePlugin, characterDrag } from "../src/character";
import { characterDrawing } from "../src/character-presentation";
import { changeControl, controlValues, createScene } from "../src/scenes";
import { drawingDigest } from "./digest";
import { sampleGold } from "./gold";

setDefaultTimeout(TIMEOUT.node);
for (const scene of ["capsulePlane", "moverOverlap"] as const)
    for (const controlled of [false, true]) {
        test(`${scene} equals all 600 native hashes and drawing records${controlled ? " through dragging and controls" : ""}`, async () => {
            const start = performance.now();
            const stem = `${scene}${controlled ? ".controls" : ""}`;
            const gold = sampleGold(
                await Bun.file(new URL(`./${stem}.gold.txt`, import.meta.url)).text(),
            );
            const drawing = (
                await Bun.file(new URL(`./${stem}.drawing.gold.txt`, import.meta.url)).text()
            )
                .trimEnd()
                .split("\n");
            const sequence = controlled
                ? (await Bun.file(new URL(`./${scene}.controls.txt`, import.meta.url)).text())
                      .trimEnd()
                      .split("\n")
                      .map((line) => {
                          const [step, name, value] = line.split("\t");
                          return {
                              step: Number(step),
                              name: name!,
                              value: Math.fround(Number(value)),
                          };
                      })
                : [];
            const app = await createApp({
                defaults: false,
                plugins: [StandardPhysicsPlugin, CharacterSamplePlugin],
            });
            try {
                const bodies = createScene(app.world, scene);
                const values = controlValues(scene);
                const origin = { x: 0, y: 0, z: 0 },
                    translation = { x: 0, y: 0, z: -20 };
                let held = false;
                expect(gold).toHaveLength(600);
                expect(drawing).toHaveLength(600);
                for (let step = 0; step < 600; step++) {
                    for (const change of sequence)
                        if (change.step === step) {
                            if (change.name === "dragX") origin.x = change.value;
                            else if (change.name === "dragY") origin.y = change.value;
                            else if (change.name === "dragZ") origin.z = change.value;
                            else if (change.name === "drag") {
                                held = change.value !== 0;
                                characterDrag(app.world, origin, translation, held, held);
                            }
                        }
                    characterDrag(app.world, origin, translation, false, held);
                    for (const change of sequence)
                        if (change.step === step && change.name === "Solve")
                            changeControl(app.world, scene, bodies, values, "Solve", change.value);
                    app.world.tick();
                    expect(hashPhysics(app.world), `${stem} step ${step} hash`).toBe(
                        gold[step]!.hash,
                    );
                    const records = characterDrawing(app.world).records;
                    expect(drawingDigest(records), `${stem} step ${step} drawing`).toBe(
                        drawing[step]!,
                    );
                    expect(app.world.resource(CharacterSample).zeroNormalCount).toBe(0);
                }
            } finally {
                app.dispose();
            }
            console.log(`${stem}: ${(performance.now() - start).toFixed(0)} ms`);
        });
    }
