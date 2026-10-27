import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Body, Hulls, Shape } from "../../core/physics";
import { createApp } from "../../engine/app";
import { PhysicsWorldDefinition, StandardPhysicsPlugin } from ".";
import { artifact, kernel } from "./kernel/kernel";

setDefaultTimeout(TIMEOUT.node);

for (const maximumMemory of [undefined, 4294967296]) {
    test(`shared world maximum is ${maximumMemory ?? 1073741824} bytes`, async () => {
        const app = await createApp({
            defaults: false,
            plugins: [StandardPhysicsPlugin],
            setup(world) {
                Object.assign(world.resource(PhysicsWorldDefinition), {
                    workerCount: 1,
                    maximumMemory,
                });
            },
        });
        try {
            expect(artifact(app.world)).toBe("shared");
            const memory = kernel(app.world).memory;
            const maximum = maximumMemory ?? 1073741824;
            memory.grow((maximum - memory.buffer.byteLength) / 65536);
            expect(memory.buffer.byteLength).toBe(maximum);
            expect(() => memory.grow(1)).toThrow();
        } finally {
            app.dispose();
        }
    });
}

test("a maximum below the shared module's minimum is refused rather than falling back", async () => {
    await expect(
        createApp({
            defaults: false,
            plugins: [StandardPhysicsPlugin],
            setup(world) {
                Object.assign(world.resource(PhysicsWorldDefinition), {
                    workerCount: 1,
                    maximumMemory: 65536,
                });
            },
        }),
    ).rejects.toThrow("PhysicsWorldDefinition.maximumMemory must be at least");
});

test("shapes exceeding the shared ceiling name the ceiling and definition field", async () => {
    const maximumMemory = 16 * 1024 * 1024;
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin],
        setup(world) {
            Object.assign(world.resource(PhysicsWorldDefinition), {
                workerCount: 1,
                maximumMemory,
            });
        },
    });
    try {
        expect(artifact(app.world)).toBe("shared");
        const world = app.world;
        const geometry = world.resource(Hulls).add({ name: "box", halfExtents: [1, 1, 1] });
        expect(() => {
            for (let i = 0; i < 100000; i++) {
                const eid = world.create();
                world.add(eid, Body);
                world.add(eid, Shape, { geometry });
            }
            world.tick();
        }).toThrow(
            `physics: shared kernel allocation failed at the ${maximumMemory} byte ceiling; increase PhysicsWorldDefinition.maximumMemory`,
        );
    } finally {
        app.dispose();
    }
});
