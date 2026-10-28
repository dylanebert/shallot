import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, hashPhysics } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { CollisionPlugin, CollisionState, collisionNames } from "../src/collision";
import {
    CollisionDebug,
    CollisionDebugPlugin,
    debugDrag,
    debugNames,
} from "../src/collision-debug";
import { changeControl, controlValues, createScene } from "../src/scenes";
import { debugRecords } from "./collision-debug-records";
import { collisionInput } from "./collision-input";
import { collisionRecords } from "./collision-records";
import { recordDigest } from "./digest";

setDefaultTimeout(TIMEOUT.node);
for (const scene of collisionNames) {
    test(`${scene}: 600 native hashes and every query invocation, including DrawControls`, async () => {
        const start = performance.now();
        for (const controlled of [false, true]) {
            const sequence = Bun.file(new URL(`./${scene}.controls.txt`, import.meta.url));
            if (controlled && !(await sequence.exists())) continue;
            const stem = `${scene}${controlled ? ".controls" : ""}`;
            const hashes = (await Bun.file(new URL(`./${stem}.gold.txt`, import.meta.url)).text())
                .trimEnd()
                .split("\n");
            const queries = (
                await Bun.file(new URL(`./${stem}.queries.gold.txt`, import.meta.url)).text()
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
            let bodies: number[] = [];
            const app = await createApp({
                defaults: false,
                plugins: [CollisionPlugin],
                setup(world) {
                    bodies = createScene(world, scene);
                },
            });
            try {
                expect(hashes).toHaveLength(600);
                expect(queries).toHaveLength(600);
                for (let step = 0; step < 600; step++) {
                    for (const change of changes)
                        if (change.step === step)
                            collisionInput(
                                app.world,
                                scene,
                                bodies,
                                values,
                                change.name,
                                change.value,
                            );
                    app.world.tick();
                    expect(
                        hashPhysics(app.world).toString(16).padStart(16, "0"),
                        `${stem} step ${step} hash`,
                    ).toBe(hashes[step]);
                    const records = collisionRecords(app.world.resource(CollisionState).queries);
                    expect(recordDigest(records), `${stem} step ${step} query records`).toBe(
                        queries[step],
                    );
                }
            } finally {
                app.dispose();
            }
        }
        console.log(`${scene} parity: ${(performance.now() - start).toFixed(0)} ms`);
    });
}
for (const scene of debugNames) {
    test(`${scene}: 600 native hashes and algorithm records, including DrawControls`, async () => {
        const start = performance.now();
        for (const controlled of [false, true]) {
            const sequence = Bun.file(new URL(`./${scene}.controls.txt`, import.meta.url));
            if (controlled && !(await sequence.exists())) continue;
            const stem = `${scene}${controlled ? ".controls" : ""}`;
            const hashes = (await Bun.file(new URL(`./${stem}.gold.txt`, import.meta.url)).text())
                .trimEnd()
                .split("\n");
            const records = (
                await Bun.file(new URL(`./${stem}.queries.gold.txt`, import.meta.url)).text()
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
                plugins: [CollisionDebugPlugin],
                setup(w) {
                    createScene(w, scene);
                },
            });
            try {
                expect(hashes).toHaveLength(600);
                expect(records).toHaveLength(600);
                const origin = { x: 0, y: 0, z: 0 };
                let held = false;
                for (let step = 0; step < 600; step++) {
                    let press = false;
                    for (const c of changes)
                        if (c.step === step) {
                            if (c.name === "dragX") origin.x = c.value;
                            else if (c.name === "dragY") origin.y = c.value;
                            else if (c.name === "dragZ") origin.z = c.value;
                            else if (c.name === "drag") {
                                held = !!c.value;
                                press = held;
                            } else changeControl(app.world, scene, [], values, c.name, c.value);
                        }
                    if (scene === "shapeDistance")
                        debugDrag(app.world, origin, { x: 0, y: 0, z: -20 }, press, held);
                    app.world.tick();
                    expect(
                        hashPhysics(app.world).toString(16).padStart(16, "0"),
                        `${stem} step ${step} hash`,
                    ).toBe(hashes[step]);
                    expect(
                        recordDigest(debugRecords(app.world.resource(CollisionDebug))),
                        `${stem} step ${step} records`,
                    ).toBe(records[step]);
                }
            } finally {
                app.dispose();
            }
        }
        console.log(`${scene} parity: ${(performance.now() - start).toFixed(0)} ms`);
    });
}
