import { expect, test } from "bun:test";
import { dirname, join } from "node:path";
import { nativeBinary, run } from "./native";
import { PhysicsWorld } from "../../src/standard/physics/api/world";
import { BodyType } from "../../src/standard/physics/common/types";
import { init } from "../../src/standard/physics/kernel/kernel";

await init(undefined, { threads: 0 });
test("native warmed mutation allocation denominator", () => {
    const cache = dirname(nativeBinary());
    const binary = join(cache, "mutations");
    run(["cc", "-O2", "-std=c17", "-ffp-contract=off", `-I${process.env.BOX3D}/include`, join(import.meta.dir, "mutations.c"), join(cache, "cmake/src/libbox3d.a"), "-o", binary]);
    const output = run([binary]);
    console.log(output.trim());
    expect(output).toBe("native warmed mutation allocations: 1200 for 600 iterations\n");
});
test("cold world mutation timings on warmed records", () => {
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    const body = world.createBody({ type: BodyType.Dynamic });
    const shape = body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 1 });
    const filters = [{ categoryBits: 1n, maskBits: 1n, groupIndex: 0 }, { categoryBits: 2n, maskBits: 2n, groupIndex: 0 }];
    const definition = { type: BodyType.Dynamic };
    const actions = {
        type: (i: number) => body.setType(i & 1 ? BodyType.Static : BodyType.Dynamic),
        filter: (i: number) => shape.setFilter(filters[i & 1]!),
        destroyCreate: () => world.createBody(definition).destroy(),
    };
    try {
        for (const [name, action] of Object.entries(actions)) {
            for (let i = 0; i < 2000; ++i) action(i);
            const samples: number[] = [];
            for (let run = 0; run < 7; ++run) {
                const start = performance.now();
                for (let i = 0; i < 10000; ++i) action(i);
                samples.push((performance.now() - start) * 1000 / 10000);
            }
            samples.sort((a, b) => a - b);
            console.log(`${name}: ${samples[3]!.toFixed(3)} us per mutation (median of 7 x 10000)`);
        }
    } finally { world.destroy(); }
});
