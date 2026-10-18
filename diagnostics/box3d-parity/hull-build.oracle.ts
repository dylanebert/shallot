import { expect, test } from "bun:test";
import { init, kernel } from "../../src/standard/physics/kernel/kernel";
import gold from "../../crates/physics/tests/hull_build.gold.json";
import { nativeBinary, run } from "./native";
import { Hulls } from "../../src/core/physics";
import { World } from "../../src/engine";
import { buildHull } from "../../src/standard/physics/geometry-builds";
import { StandardPhysicsPlugin } from "../../src/standard/physics";
import { drawHull } from "../../src/standard/physics/shapes/readgeometry";

test("hull construction and every generator match native byte images", async () => {
    const native = JSON.parse(run([nativeBinary("hull-build.c")])) as typeof gold;
    expect(native).toEqual(gold);
    await init(undefined, { threads: 0 });
    const k = kernel(undefined);
    let compared = 0;
    for (const item of native) {
        const count = item.points.length;
        const inputPtr = k.hullBuildInputPtr(count);
        const input = new Float32Array(k.memory.buffer, inputPtr, 10 + 3 * count);
        const words = new Uint32Array(input.buffer, input.byteOffset, input.length);
        for (let i = 0; i < item.args.length; ++i) words[i] = item.args[i];
        for (let i = 0; i < count; ++i) words.set(item.points[i], 10 + 3 * i);
        const ptr = k.hullBuild(item.operation, count, item.budget);
        if (item.words === null) {
            expect(ptr, item.name).toBe(0);
        } else {
            expect(ptr, item.name).not.toBe(0);
            const output = new Uint32Array(k.memory.buffer, ptr, item.words.length);
            expect(Array.from(output), item.name).toEqual(item.words);
            ++compared;
        }
    }
    const world = new World();
    await StandardPhysicsPlugin.initialize!(world);
    try {
        for (const item of native.filter((item) => item.operation === 1 || item.operation === 7)) {
            const a = new Float32Array(new Uint32Array(item.args).buffer);
            const definition = item.operation === 1
                ? { name: item.name, cylinder: { height: a[0]!, radius: a[1]!, yOffset: a[2]!, sides: a[3]! } }
                : { name: item.name, halfExtents: [a[0]!, a[1]!, a[2]!] as const, transform: { position: [a[3]!, a[4]!, a[5]!] as const, rotation: [a[6]!, a[7]!, a[8]!, a[9]!] as const } };
            const handle = world.resource(Hulls).add(definition);
            expect(buildHull(world, world.resource(Hulls).get(handle)!), item.name).toEqual(drawHull(new DataView(new Uint32Array(item.words!).buffer), 0));
        }
    } finally {
        await StandardPhysicsPlugin.dispose!(world);
        world.dispose();
    }
    console.log(JSON.stringify({ cases: native.length, images: compared }));
});
