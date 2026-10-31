import { expect, setDefaultTimeout, test } from "bun:test";
import { Body, BodyType, Shape, ShapeKind, Hulls } from "../../src/core/physics";
import { Transform } from "../../src/core/transform";
import { createApp } from "../../src/engine";
import { hashPhysics, PhysicsWorldDefinition, StandardPhysicsPlugin } from "../../src/standard/physics";
import { nativeBinary, run } from "./native";

setDefaultTimeout(300_000);
const cases = [
    [0, false, false],
    [1, false, false],
    [2, false, false],
    [63, false, false],
    [2, true, false],
    [-1, false, false],
    [100, true, false],
    [2, false, true],
    [0, true, true],
] as const;
const binary = nativeBinary("restitution-settings.c", "x86_64");
const outputs = cases.map(([iterations, propagation, change]) =>
    run([binary, String(iterations), String(Number(propagation)), String(Number(change))])
        .trim().split("\n").map((line) => line.split(" ")),
);

for (const [index, [iterations, propagation, change]] of cases.entries()) {
    for (const workerCount of [1, 4]) {
        test(`restitution ${iterations}/${propagation}, live change ${change}, ${workerCount} workers equals native`, async () => {
            const app = await createApp({
                defaults: false,
                plugins: [StandardPhysicsPlugin],
                setup(world) {
                    Object.assign(world.resource(PhysicsWorldDefinition), {
                        workerCount,
                        restitutionIterations: iterations,
                        enableRestitutionPropagation: propagation,
                    });
                },
            });
            try {
                const world = app.world;
                const box = world.resource(Hulls).add({ halfExtents: [1, 1, 1] });
                for (let i = 0; i < 5; i++) {
                    const eid = world.create();
                    world.add(eid, Transform, { translation: [i === 4 ? 0.25 : 0, i === 4 ? 11 : 2 * i, 0] });
                    world.add(eid, Body, {
                        type: i === 0 ? BodyType.Static : BodyType.Dynamic,
                        linearVelocity: [0, i === 4 ? -12 : 0, 0],
                        angularVelocity: [0, 0, i === 4 ? 0.3 : 0],
                    });
                    world.add(eid, Shape, { kind: i === 4 ? ShapeKind.Hull : ShapeKind.Sphere, geometry: box, radius: 1, friction: 0, restitution: i === 4 ? 0.75 : 0, density: 1 });
                }
                for (let step = 0; step < 180; step++) {
                    if (change && step === 5) {
                        const definition = world.resource(PhysicsWorldDefinition);
                        definition.restitutionIterations = iterations === 0 ? 3 : 0;
                        definition.enableRestitutionPropagation = !propagation;
                    }
                    world.tick();
                    expect(hashPhysics(world).toString(16).padStart(16, "0"), `step ${step}`).toBe(outputs[index]![step]![2]!);
                }
            } finally { app.dispose(); }
        });
    }
}

test("the fixture distinguishes iteration count, propagation and live changes", () => {
    const hashes = outputs.map((rows) => rows.map((row) => row[2]));
    expect(hashes[0]).not.toEqual(hashes[2]);
    expect(hashes[1]).not.toEqual(hashes[2]);
    expect(hashes[4]).not.toEqual(hashes[2]);
    expect(hashes[7]).not.toEqual(hashes[2]);
    expect(hashes[8]).not.toEqual(hashes[0]);
    for (const [i, [iterations, propagation, change]] of cases.entries()) {
        for (const [step, row] of outputs[i]!.entries()) {
            const live = change && step >= 5;
            expect(Number(row[0])).toBe(live ? iterations === 0 ? 3 : 0 : Math.max(0, Math.min(63, iterations)));
            expect(Number(row[1])).toBe(Number(live ? !propagation : propagation));
        }
    }
});
