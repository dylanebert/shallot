import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, hashPhysics } from "@dylanebert/shallot";
import { StandardPhysicsPlugin } from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { MeshSamplePlugin, meshScenes } from "../src/mesh";
import { MeshInspectorState } from "../src/mesh-inspector";
import { meshViewerBounds } from "../src/mesh-inspector-presentation";
import { type Controls, controlValues, createScene } from "../src/scenes";
import { drawingDigest } from "./digest";
import { sampleGold } from "./gold";
import { manualParityScenes } from "./manual-parity";
import { meshControl } from "./mesh-controls";

setDefaultTimeout(TIMEOUT.node);
for (const scene of Object.keys(meshScenes) as (keyof typeof meshScenes)[]) {
    if (manualParityScenes.has(scene)) continue;
    const controls = meshScenes[scene].controls as Controls | undefined;
    for (const controlled of controls && Object.values(controls).some((control) => !control.read)
        ? [false, true]
        : [false]) {
        test(`${scene}${controlled ? " controls" : ""} equals every native step`, async () => {
            const gold = sampleGold(
                await Bun.file(
                    new URL(`./${scene}${controlled ? ".controls" : ""}.gold.txt`, import.meta.url),
                ).text(),
            );
            expect(gold).toHaveLength(600);
            const inspector = scene === "meshViewer";
            const stem = `${scene}${controlled ? ".controls" : ""}`;
            const readouts = inspector
                ? (await Bun.file(new URL(`./${stem}.readouts.gold.txt`, import.meta.url)).text())
                      .trim()
                      .split("\n")
                : [];
            const bounds =
                scene === "meshViewer"
                    ? (await Bun.file(new URL(`./${stem}.bounds.gold.txt`, import.meta.url)).text())
                          .trim()
                          .split("\n")
                    : [];
            if (inspector) expect(readouts).toHaveLength(600);
            if (scene === "meshViewer") expect(bounds).toHaveLength(600);
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
                        const { name, value } = meshControl(change.name, change.value);
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
                    if (inspector) {
                        const names = [
                            "triangle count",
                            "vertex count",
                            "degenerate count",
                            "node area",
                            "height",
                        ];
                        const row = names.map((name) => controls![name]!.read!(app.world)!);
                        row.push(app.world.resource(MeshInspectorState).values["draw level"]);
                        expect(drawingDigest([row]), `readouts step ${step}`).toBe(readouts[step]!);
                    }
                    if (scene === "meshViewer")
                        expect(
                            drawingDigest(meshViewerBounds(app.world)),
                            `node bounds step ${step}`,
                        ).toBe(bounds[step]!);
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
