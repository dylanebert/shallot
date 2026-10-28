import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { World } from "../../engine";
import {
    computeCapsuleMass,
    computeHullMass,
    makeScaledBoxHull,
    StandardPhysicsPlugin,
} from "./index";
import { kernel } from "./kernel/kernel";

setDefaultTimeout(TIMEOUT.node);

async function released(retain: boolean): Promise<number> {
    const world = new World();
    await StandardPhysicsPlugin.initialize!(world);
    let count = 0;
    const registry = new FinalizationRegistry(() => count++);
    const held: object[] = [];
    const transform = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };
    const scale = { x: 1, y: 1, z: 1 };
    const capsule = { center1: { x: -1, y: 0, z: 0 }, center2: { x: 1, y: 0, z: 0 }, radius: 1 };
    function build() {
        const hull = makeScaledBoxHull(scale, transform, scale, world);
        const mass = computeHullMass(hull, 1, world);
        const capMass = computeCapsuleMass(capsule, 1, world);
        registry.register(hull, 0);
        registry.register(mass, 0);
        registry.register(capMass, 0);
        if (retain) held.push(hull, mass, capMass);
    }
    try {
        build();
        const bytes = kernel(world).memory.buffer.byteLength;
        for (let i = 0; i < 1000; i++) build();
        expect(kernel(world).memory.buffer.byteLength).toBe(bytes);
        for (let i = 0; i < 20; i++) {
            Bun.gc(true);
            await Bun.sleep(0);
        }
        // Keep the control's owners live through collection.
        expect(held.length).toBe(retain ? 3003 : 0);
        return count;
    } finally {
        await StandardPhysicsPlugin.dispose!(world);
        world.dispose();
    }
}

test("construction geometry releases owned JS results and bounds kernel scratch after warmup", async () => {
    expect(await released(false)).toBe(3003);
});
test("geometry retention proof rejects deliberately retained construction results", async () => {
    expect(await released(true)).toBe(0);
});
