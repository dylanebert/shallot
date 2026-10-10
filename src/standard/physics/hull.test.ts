import { expect, test } from "bun:test";
import {
    Body,
    BodyType,
    Hulls,
    PhysicsPlugin,
    Shape,
    ShapeKind,
    UNIT_CUBE_ID,
} from "../../core/physics";
import { World } from "../../engine";
import { PhysicsWorld } from "./api";
import { marshalShape } from "./marshal-shape";

test("two worlds marshal their own hull registered under the same name", () => {
    const worlds = [new World(), new World()];
    const solver = new PhysicsWorld();
    try {
        const bodies = worlds.map((world, i) => {
            for (const component of PhysicsPlugin.components!)
                world.registry.register(component, PhysicsPlugin.name);
            const hulls = world.resource(Hulls);
            expect(hulls.id("__unit_cube__")).toBe(UNIT_CUBE_ID);
            const cube = hulls.get("__unit_cube__")!;
            const scale = i + 1;
            const id = hulls.register({
                ...cube,
                name: "shared-name",
                verts: cube.verts.map(([x, y, z]) => [x * scale, y * scale, z * scale]),
            });
            expect(id).toBe(1);
            const eid = world.create();
            world.add(eid, Body, { type: BodyType.Dynamic });
            world.add(eid, Shape, {
                kind: ShapeKind.Hull,
                geometry: id,
                scale: [1, 1, 1, 0],
                density: 1 / (8 * scale ** 3),
            });
            const body = solver.createBody({ type: BodyType.Dynamic });
            marshalShape(world, body, eid);
            return body;
        });
        expect(bodies[0]!.getMassData().inertia.cx.x).toBeCloseTo(2 / 3, 5);
        expect(bodies[1]!.getMassData().inertia.cx.x).toBeCloseTo(8 / 3, 5);
    } finally {
        solver.destroy();
        for (const world of worlds) world.dispose();
    }
});
