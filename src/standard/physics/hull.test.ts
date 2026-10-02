import { expect, test } from "bun:test";
import { Body, Hulls, ShapeKind, UNIT_CUBE_ID } from "../../core/physics";
import { World } from "../../engine";
import { PhysicsWorld } from "./api";
import { marshalBody } from "./marshal";

test("two worlds marshal their own hull registered under the same name", () => {
    const worlds = [new World(), new World()];
    const solver = new PhysicsWorld();
    try {
        const bodies = worlds.map((world, i) => {
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
            world.add(eid, Body);
            world.storage(Body).shape.set(eid, ShapeKind.Hull);
            world.storage(Body).halfExtents.w.set(eid, id);
            world.storage(Body).mass.set(eid, 1);
            return marshalBody(world, solver, eid)!;
        });
        expect(bodies[0]!.getMassData().inertia.cx.x).toBeCloseTo(2 / 3, 5);
        expect(bodies[1]!.getMassData().inertia.cx.x).toBeCloseTo(8 / 3, 5);
    } finally {
        solver.destroy();
        for (const world of worlds) world.dispose();
    }
});
