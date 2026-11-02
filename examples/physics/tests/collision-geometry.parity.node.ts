import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, hashPhysics } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import {
    CollisionGeometry,
    CollisionGeometryPlugin,
    collisionGeometryScenes,
} from "../src/collision-geometry";
import { sceneSwitcher } from "../src/sample";
import { type Controls, changeControl, controlValues, createScene, scenes } from "../src/scenes";
import { collisionGeometryRecords } from "./collision-geometry-records";
import { drawingDigest } from "./digest";
import { sampleGold } from "./gold";

setDefaultTimeout(TIMEOUT.node);
for (const scene of Object.keys(
    collisionGeometryScenes,
) as (keyof typeof collisionGeometryScenes)[])
    for (const controlled of [false, true])
        test(`${scene} retains every native geometry and query record${controlled ? " through all controls" : ""}`, async () => {
            const stem = `${scene}${controlled ? ".controls" : ""}`;
            const gold = sampleGold(
                await Bun.file(new URL(`./${stem}.gold.txt`, import.meta.url)).text(),
            );
            const records = (
                await Bun.file(new URL(`./${stem}.geometry.gold.txt`, import.meta.url)).text()
            )
                .trimEnd()
                .split("\n");
            expect(gold).toHaveLength(600);
            expect(records).toHaveLength(600);
            const changes = controlled
                ? (await Bun.file(new URL(`./${scene}.controls.txt`, import.meta.url)).text())
                      .trimEnd()
                      .split("\n")
                      .map((line) => {
                          const [step, label, value] = line.split("\t");
                          return { step: Number(step), label, value: Number(value) };
                      })
                : [];
            const app = await createApp({
                defaults: false,
                plugins: [CollisionGeometryPlugin],
                setup(world) {
                    createScene(world, scene);
                },
            });
            try {
                const controls: Controls = scenes[scene].controls;
                const values = controlValues(scene);
                const used = new Set<string>();
                for (let step = 0; step < 600; step++) {
                    for (const change of changes)
                        if (change.step === step) {
                            const key = Object.entries(controls).find(
                                ([, c]) => c.label === change.label,
                            )![0];
                            changeControl(app.world, scene, [], values, key, change.value);
                            used.add(key);
                        }
                    app.world.tick();
                    expect(hashPhysics(app.world), `step ${step} world`).toBe(gold[step]!.hash);
                    expect(
                        drawingDigest(collisionGeometryRecords(app.world)),
                        `step ${step} geometry`,
                    ).toBe(records[step]!);
                }
                if (controlled) expect([...used].sort()).toEqual(Object.keys(controls).sort());
            } finally {
                app.dispose();
            }
        });
test("leaving Collision geometry clears its query, labels and generated hull state", async () => {
    const app = await createApp({ defaults: false, plugins: [CollisionGeometryPlugin] });
    try {
        const select = sceneSwitcher(app.world);
        for (const scene of ["hull2D", "encroachCast"] as const) {
            select(scene);
            app.world.tick();
            const first = (
                await Bun.file(new URL(`./${scene}.geometry.gold.txt`, import.meta.url)).text()
            ).split("\n")[0]!;
            expect(drawingDigest(collisionGeometryRecords(app.world))).toBe(first);
            select("boxStack");
            expect(app.world.resource(CollisionGeometry)).toEqual(CollisionGeometry.create());
            app.world.tick();
        }
    } finally {
        app.dispose();
    }
});
