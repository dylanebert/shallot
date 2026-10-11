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
import { marshalShape } from "./marshal-shape";
import { createTestBody, createTestWorld, destroyTestWorld } from "./testing";
import { getMassData, makeBodyId } from "./world/body";

test("two worlds marshal their own hull registered under the same name", () => {
    const worlds = [new World(), new World()];
    const solvers = worlds.map((world) => createTestWorld({}, world));
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
            const state = solvers[i]!;
            const body = createTestBody(state, { type: BodyType.Dynamic });
            marshalShape(world, state, makeBodyId(state, body), eid);
            return { state, body };
        });
        expect(getMassData(bodies[0]!.state, bodies[0]!.body).inertia.cx.x).toBeCloseTo(2 / 3, 5);
        expect(getMassData(bodies[1]!.state, bodies[1]!.body).inertia.cx.x).toBeCloseTo(8 / 3, 5);
    } finally {
        for (const solver of solvers) destroyTestWorld(solver);
        for (const world of worlds) world.dispose();
    }
});
