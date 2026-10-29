import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, hashPhysics } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { changeControl, controlValues, createScene } from "../src/scenes";
import { TreeBenchmarkPlugin, treeRecords } from "../src/tree";
import { treeBounds, treeQueryDrawing } from "../src/tree-presentation";
import { drawingDigest } from "./digest";
import { sampleGold } from "./gold";

setDefaultTimeout(TIMEOUT.node);
for (const controlled of [false, true])
    test(`Tree Benchmark${controlled ? " controls" : ""} equals all 600 native world and independent-tree records`, async () => {
        const stem = `treeBenchmark${controlled ? ".controls" : ""}`;
        const gold = sampleGold(
            await Bun.file(new URL(`./${stem}.gold.txt`, import.meta.url)).text(),
        );
        const records = (await Bun.file(new URL(`./${stem}.tree.gold.txt`, import.meta.url)).text())
            .trim()
            .split("\n");
        const bounds = (
            await Bun.file(new URL(`./${stem}.bounds.gold.txt`, import.meta.url)).text()
        )
            .trim()
            .split("\n");
        const sequence = controlled
            ? (await Bun.file(new URL("./treeBenchmark.controls.txt", import.meta.url)).text())
                  .trim()
                  .split("\n")
                  .map((line) => {
                      const [step, name, value] = line.split("\t");
                      return { step: Number(step), name: name!, value: Number(value) };
                  })
            : [];
        expect(gold).toHaveLength(600);
        expect(records).toHaveLength(600);
        expect(bounds).toHaveLength(600);
        const start = performance.now();
        const app = await createApp({
            defaults: false,
            plugins: [TreeBenchmarkPlugin],
            setup(world) {
                createScene(world, "treeBenchmark");
            },
        });
        const values = controlValues("treeBenchmark");
        try {
            for (const [step, expected] of gold.entries()) {
                for (const c of sequence.filter((c) => c.step === step))
                    changeControl(app.world, "treeBenchmark", [], values, c.name, c.value);
                app.world.tick();
                expect(hashPhysics(app.world), `step ${step} world`).toBe(expected.hash);
                expect(drawingDigest(treeRecords(app.world)), `step ${step} tree`).toBe(
                    records[step]!,
                );
                const drawn = [
                    ...treeBounds(app.world, [0, 0, 0]),
                    ...treeQueryDrawing(app.world).boxes,
                ].map((row) => row.slice(0, 6));
                expect(drawingDigest(drawn), `step ${step} drawn bounds`).toBe(bounds[step]!);
            }
        } finally {
            app.dispose();
        }
        console.log(
            `Tree Benchmark${controlled ? " controls" : ""} 600-step parity: ${(performance.now() - start).toFixed(0)} ms`,
        );
    });
