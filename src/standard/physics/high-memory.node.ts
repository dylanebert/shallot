import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Body, Hulls, Shape } from "../../core/physics";
import { Transform } from "../../core/transform";
import { createApp } from "../../engine/app";
import { PhysicsWorldDefinition, StandardPhysicsPlugin } from ".";
import { artifact, kernel } from "./kernel/kernel";

setDefaultTimeout(TIMEOUT.node);

/** Falling boxes' heights after a few ticks, with the shared memory first grown to `floor` bytes. */
async function heights(floor: number): Promise<{ y: number[]; bytes: number }> {
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin],
        setup(world) {
            Object.assign(world.resource(PhysicsWorldDefinition), {
                workerCount: 1,
                maximumMemory: 2 ** 32,
            });
        },
    });
    try {
        const world = app.world;
        expect(artifact(world)).toBe("shared");
        const memory = kernel(world).memory;
        if (memory.buffer.byteLength < floor)
            memory.grow((floor - memory.buffer.byteLength) / 65536);
        const geometry = world.resource(Hulls).add({ name: "box", halfExtents: [0.5, 0.5, 0.5] });
        const bodies: number[] = [];
        for (let i = 0; i < 4000; i++) {
            const eid = world.create();
            world.add(eid, Transform, {
                translation: [(i % 64) * 2, 2 + Math.floor(i / 64) * 2, 0],
            });
            world.add(eid, Body);
            world.add(eid, Shape, { geometry });
            bodies.push(eid);
        }
        for (let i = 0; i < 10; i++) world.tick();
        return {
            y: bodies.map((eid) => world.get(eid, Transform)!.translation[1]),
            bytes: memory.buffer.byteLength,
        };
    } finally {
        app.dispose();
    }
}

test("the shared kernel reads columns it allocates above 2 GiB", async () => {
    const low = await heights(0);
    // Memory grown past 2 GiB first, so every allocation the scene grows lands above it, at
    // addresses its exports return as negative int32.
    const high = await heights(2 ** 31);
    expect(high.bytes).toBeGreaterThan(2 ** 31);
    expect(high.y).toEqual(low.y);
    expect(high.y.some((y) => y !== 0)).toBe(true);
});
