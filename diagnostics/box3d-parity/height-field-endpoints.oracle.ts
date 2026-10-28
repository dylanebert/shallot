import { expect, setDefaultTimeout, test } from "bun:test";
import { Body, BodyType, GlobalTransform, HeightFields, Shape, ShapeKind, Transform, createApp } from "../../src";
import { castRayClosest, emptyQueryHit, hashPhysics, isAwake, StandardPhysicsPlugin } from "../../src/standard/physics";
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

test("height-field endpoint bodies rest on surfaces or fall through cell-free fields exactly as native for 600 steps", async () => {
    const native = run([nativeBinary("height-field-endpoints.c")]).trim().split("\n");
    const actual: string[] = [];
    for (const [index, [rows, columns]] of [[10, 10], [4, 1], [1, 4], [2, 2]].entries()) {
        let dynamic = 0;
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
                dynamic = world.create();
                world.add(dynamic, Transform, {translation: [columns! - 1, 2, rows! - 1]});
                world.add(dynamic, Body, {type: BodyType.Dynamic});
                world.add(dynamic, Shape, {kind: ShapeKind.Sphere, radius: 0.5, filterCategoryLow: 2, filterCategoryHigh: 0});
            },
        });
        try {
            for (let step = 0; step < 600; step++) {
                app.world.tick();
                const out = emptyQueryHit();
                const hit = castRayClosest(app.world, { x: columns! - 1, y: 4, z: rows! - 1 }, { x: 0, y: -8, z: 0 }, out, {categoryBits: 0xffffffffffffffffn, maskBits: 1n});
                const ray = hit
                    ? [1, ...[out.point.x, out.point.y, out.point.z, out.normal.x, out.normal.y, out.normal.z, out.fraction].map(bits)]
                    : [0];
                const hash = hashPhysics(app.world).toString(16).padStart(16, "0");
                const count = worldCounters(physicsState(app.world)!).shapeCount;
                actual.push(`${hash} ${count} ${JSON.stringify(ray)}`);
            }
            const pose = app.world.get(dynamic, GlobalTransform)!;
            const y = pose.translation[1];
            const velocity = pose.linearVelocity[1];
            const awake = isAwake(app.world, dynamic);
            actual.push(`sphere ${bits(y)} ${bits(velocity)} ${Number(awake)}`);
            if (columns === 1 || rows === 1) {
                expect(y).toBeLessThan(-100);
                expect(awake).toBe(true);
            } else {
                expect(y).toBeGreaterThan(0.4);
                expect(y).toBeLessThan(0.6);
                expect(velocity).toBe(0);
                expect(awake).toBe(false);
            }
        } finally {
            app.dispose();
        }
    }
    expect(actual).toEqual(native);
});
