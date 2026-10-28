import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { heightFieldDrawing } from "../src/height-field-presentation";
import { MeshSamplePlugin } from "../src/mesh";
import { type Controls, controlValues, createScene, scenes } from "../src/scenes";
import { drawingDigest } from "./digest";

setDefaultTimeout(TIMEOUT.node);
for (const controlled of [false, true])
    test(`Height Field ${controlled ? "controls " : ""}draws every native ray, endpoint, sphere and hit`, async () => {
        const stem = `meshHeightField${controlled ? ".controls" : ""}`;
        const gold = (await Bun.file(new URL(`./${stem}.drawing.gold.txt`, import.meta.url)).text())
            .trim()
            .split("\n");
        const sequence = controlled
            ? (await Bun.file(new URL("./meshHeightField.controls.txt", import.meta.url)).text())
                  .trim()
                  .split("\n")
                  .map((line) => {
                      const [step, name, value] = line.split("\t");
                      return { step: Number(step), name: name!, value: Number(value) };
                  })
            : [];
        const app = await createApp({
            defaults: false,
            plugins: [MeshSamplePlugin],
            setup(world) {
                createScene(world, "meshHeightField");
            },
        });
        const values = controlValues("meshHeightField");
        try {
            expect(gold).toHaveLength(600);
            for (const [step, expected] of gold.entries()) {
                for (const change of sequence.filter((c) => c.step === step)) {
                    values[change.name] = change.value;
                    (scenes.meshHeightField.controls as Controls)[change.name]!.set(
                        app.world,
                        [],
                        change.value,
                        values,
                    );
                }
                app.world.tick();
                expect(drawingDigest(heightFieldDrawing(app.world)), `step ${step}`).toBe(expected);
            }
        } finally {
            app.dispose();
        }
    });
