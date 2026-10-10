import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp } from "../../src/engine";
import { Body, BodyType, Shape, ShapeKind } from "../../src/core/physics";
import { GlobalTransform } from "../../src/core/transform";
import { hashPhysics, StandardPhysicsPlugin } from "../../src/standard/physics";
import { nativeSseOutput } from "./native-evidence";
import { assertPublicOracleKernel } from "./oracle-kernel";

setDefaultTimeout(180_000);
await assertPublicOracleKernel();
const [nativeHash, nativeShapeCount, nativeY] = nativeSseOutput("shape-compound.c", "")
    .trim()
    .split(/\s+/);

// Build only with exported ECS definitions: one body entity owns a floor Shape and two separate
// Shape entities reference the dynamic body. No Body collider fields or solver handles author them.
test("two ECS Shape entities form one Box3D body with native compound collision and hash", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        const floor = world.create();
        world.add(floor, Body, { position: [0, -0.5, 0, 0] });
        world.add(floor, Shape, { scale: [5, 0.5, 5, 0] });

        const body = world.create();
        world.add(body, Body, { type: BodyType.Dynamic, position: [0, 0.4, 0, 0] });
        const left = world.create();
        world.add(left, Shape, { body, kind: ShapeKind.Hull, scale: [0.5, 0.5, 0.5, 0] });
        const right = world.create();
        world.add(right, Shape, { body, kind: ShapeKind.Hull, scale: [0.5, 0.5, 0.5, 0] });

        world.tick();

        expect(world.has(left, Shape)).toBe(true);
        expect(world.has(right, Shape)).toBe(true);
        expect(hashPhysics(world).toString(16).padStart(16, "0")).toBe(nativeHash);
        expect(world.storage(Shape).body.get(left)).toBe(body);
        expect(world.storage(Shape).body.get(right)).toBe(body);
        expect(nativeShapeCount).toBe("3");
        const y = world.storage(GlobalTransform).translation.y.get(body);
        expect(y).toBeCloseTo(Number(nativeY), 6);
        expect(y).toBeGreaterThan(0.4);
    } finally {
        app.dispose();
    }
});
