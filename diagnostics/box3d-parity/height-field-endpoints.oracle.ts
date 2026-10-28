import { expect, setDefaultTimeout, test } from "bun:test";
import { Body, HeightFields, Shape, ShapeKind, createApp } from "../../src";
import { castRayClosest, emptyQueryHit, hashPhysics, StandardPhysicsPlugin } from "../../src/standard/physics";
import { physicsState } from "../../src/standard/physics/runtime";
import { worldCounters } from "../../src/standard/physics/world/world";
import { nativeBinary, run } from "./native";

setDefaultTimeout(0);
const bits = (value: number) => new Uint32Array(new Float32Array([value]).buffer)[0]!;

test("assert-enabled native height-field construction holds the height-count and bounds preconditions", () => {
    const binary = nativeBinary("height-field-preconditions.c");
    for (const [x, z, minimum, maximum, condition] of [
        [0, 4, 0, 1, "heightCount >= 4"],
        [1, 1, 0, 1, "heightCount >= 4"],
        [1, 3, 0, 1, "heightCount >= 4"],
        [3, 1, 0, 1, "heightCount >= 4"],
        [2, 2, 1, 0, "data->globalMinimumHeight <= data->globalMaximumHeight"],
        [1, 4, 0, 1, ""], [4, 1, 0, 1, ""], [2, 2, 0, 0, ""],
    ] as const) {
        const result = Bun.spawnSync([binary, String(x), String(z), String(minimum), String(maximum)]);
        expect(result.exitCode).toBe(condition ? 77 : 0);
        if (condition) expect(result.stderr.toString()).toContain(condition);
    }
});

test("height-field sample endpoints create, step and ray-cast exactly as native", async () => {
    const native = run([nativeBinary("height-field-endpoints.c")]).trim().split("\n");
    const actual: string[] = [];
    for (const [index, [rows, columns]] of [[10, 10], [4, 1], [1, 4], [2, 2]].entries()) {
        const app = await createApp({
            defaults: false,
            plugins: [StandardPhysicsPlugin],
            setup(world) {
                const eid = world.create();
                world.add(eid, Body);
                world.add(eid, Shape, {
                    kind: ShapeKind.HeightField,
                    geometry: world.resource(HeightFields).add({
                        name: "endpoint",
                        countX: columns!,
                        countZ: rows!,
                        heights: new Float32Array(rows! * columns!),
                        scale: [2, index === 0 ? 0 : 1.5, 2],
                        globalMinimumHeight: -256,
                        globalMaximumHeight: 256,
                    }),
                });
            },
        });
        try {
            for (let step = 0; step < 3; step++) {
                app.world.tick();
                const out = emptyQueryHit();
                const hit = castRayClosest(app.world, { x: 0, y: 4, z: 0 }, { x: 0, y: -8, z: 0 }, out);
                const ray = hit
                    ? [1, ...[out.point.x, out.point.y, out.point.z, out.normal.x, out.normal.y, out.normal.z, out.fraction].map(bits)]
                    : [0];
                const hash = hashPhysics(app.world).toString(16).padStart(16, "0");
                const count = worldCounters(physicsState(app.world)!).shapeCount;
                actual.push(`${hash} ${count} ${JSON.stringify(ray)}`);
            }
        } finally {
            app.dispose();
        }
    }
    expect(actual).toEqual(native);
});
