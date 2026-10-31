import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp } from "../../src/engine";
import { TIMEOUT } from "../../src/project/testing.js";
import { StandardPhysicsPlugin } from "../../src/standard/physics";
import { BodyType } from "../../src/standard/physics/common/types";
import { makeBoxHull } from "../../src/standard/physics/shapes/build-hull";
import { createHullShape, createSphereShape } from "../../src/standard/physics/shapes/shape";
import { manifoldRecords, manifoldSubject } from "../../src/standard/physics/standalone-manifold.fixture";
import { createTestBody, createTestWorld, destroyTestWorld, stepTestWorld, testShapeDef } from "../../src/standard/physics/testing";
import { worldSetPreSolveCallback } from "../../src/standard/physics/world/world";
import { WorldCountersBuffer } from "../../src/standard/physics/world/counters";
import { kernel } from "../../src/standard/physics/kernel/kernel";
import { nativeSseOutput } from "./native-evidence";
import { assertPublicOracleKernel } from "./oracle-kernel";

setDefaultTimeout(TIMEOUT.node);

for (const [op, name] of [[10, "sphere-sphere"], [11, "capsule-sphere"]] as const) {
    test(`${name} includes separated points within the speculative distance (#187)`, async () => {
        const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
        try {
            const subject = manifoldSubject(app.world);
            subject.transform.p.y = 1.01;
            expect(subject.call(op).pointCount).toBe(1);
            const records = manifoldRecords(app.world);
            const inputs = records.input.trim().split("\n");
            const outputs = records.output.trim().split("\n");
            const selected = inputs.flatMap((line, i) => line.startsWith(`${op} `) ? [i] : []);
            expect(selected.length).toBeGreaterThan(0);
            expect(selected.map(i => outputs[i]).join("\n") + "\n").toBe(
                nativeSseOutput("functions.c", selected.map(i => inputs[i]).join("\n") + "\n"),
            );
        } finally { app.dispose(); }
    });
}

for (const mode of [0, 1, 2]) {
    test(`${mode === 2 ? "rejected pre-solve invalidates the recycle cache" : `speculative points ${mode ? "allow" : "disable"} recycling`} (#187)`, async () => {
        await assertPublicOracleKernel();
        const world = createTestWorld({ enableSleep: false });
        try {
            const ground = createTestBody(world, { position: { x: 0, y: -0.5, z: 0 } });
            createHullShape(world, ground, testShapeDef(), makeBoxHull(10, 0.5, 10));
            const body = createTestBody(world, {
                type: BodyType.Dynamic,
                position: { x: 0, y: mode === 2 ? 0.25 : 0.5, z: 0 },
            });
            let calls = 0;
            let accept = false;
            if (mode === 2) {
                createSphereShape(world, body, testShapeDef({ enablePreSolveEvents: true }), {
                    center: { x: 0, y: 0, z: 0 }, radius: 0.25,
                });
                worldSetPreSolveCallback(world, () => { ++calls; return accept; });
            } else {
                createHullShape(world, body, testShapeDef({ enableSpeculativeContact: mode === 1 }), makeBoxHull(0.5, 0.5, 0.5));
            }
            const rows: string[] = [];
            const counters = new WorldCountersBuffer();
            for (let i = 0; i < 60; ++i) {
                accept = i !== 0;
                stepTestWorld(world, 1 / 60, 4);
                rows.push(`${calls} ${counters.read(kernel(world.ecsState), world.worldId).recycledContactCount}`);
            }
            expect(rows.join("\n") + "\n").toBe(nativeSseOutput("contact-fixes.c", `${mode}\n`));
        } finally { destroyTestWorld(world); }
    });
}
