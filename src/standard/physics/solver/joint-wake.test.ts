import { expect, test } from "bun:test";
import { BodyType, PhysicsWorld } from "../api";
import { BodyField, bodyField } from "../kernel/bodyrecords";
import { setSplitIslandCandidate, splitIslandCandidate } from "../kernel/islandcolumns";

test("joint creation and destruction wake every sleeping endpoint-set body and keep their shape-query outputs current", () => {
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        const bodies = [0, 10, 20].map((x) =>
            world.createBody({ type: BodyType.Dynamic, position: { x, y: 0, z: 0 } }),
        );
        const shapes = bodies.map((body) =>
            body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 }),
        );
        world.createFilterJoint(bodies[0], bodies[1]);
        bodies[0].setAwake(false);
        expect(bodies[0].isAwake()).toBe(false);
        expect(bodies[1].isAwake()).toBe(false);
        const connector = world.createFilterJoint(bodies[1], bodies[2]);
        const verify = (offset: number) => {
            for (const body of bodies) {
                expect(body.isAwake()).toBe(true);
                body.setLinearVelocity({ x: 60, y: 0, z: 0 });
            }
            world.step(1 / 60);
            for (let i = 0; i < bodies.length; ++i) {
                const ray = world.castRayClosest(
                    { x: i * 10 + offset, y: -2, z: 0 },
                    { x: 0, y: 4, z: 0 },
                );
                expect({
                    hit: ray.hit,
                    shape: ray.shape?.id,
                    point: ray.point,
                    fraction: ray.fraction,
                }).toEqual({
                    hit: true,
                    shape: shapes[i].id,
                    point: { x: i * 10 + offset, y: -0.5, z: 0 },
                    fraction: 0.375,
                });
            }
        };
        verify(1);
        bodies[0].setAwake(false);
        for (const body of bodies) expect(body.isAwake()).toBe(false);
        connector.destroy(true);
        verify(2);
    } finally {
        world.destroy();
    }
});

test("a kernel joint merge clears a destroyed island's split candidate and snapshots retain the candidate without touching a sibling", () => {
    const world = new PhysicsWorld();
    const sibling = new PhysicsWorld();
    try {
        const a = world.createBody({ type: BodyType.Dynamic });
        const b = world.createBody({ type: BodyType.Dynamic });
        const other = sibling.createBody({ type: BodyType.Dynamic });
        const candidate = bodyField(world.state, b.id.index1 - 1, BodyField.islandId);
        const otherCandidate = bodyField(sibling.state, other.id.index1 - 1, BodyField.islandId);
        setSplitIslandCandidate(world.state, candidate);
        setSplitIslandCandidate(sibling.state, otherCandidate);
        const saved = world.snapshot();
        world.createFilterJoint(a, b);
        expect(splitIslandCandidate(world.state)).toBe(-1);
        expect(splitIslandCandidate(sibling.state)).toBe(otherCandidate);
        world.restore(saved);
        expect(splitIslandCandidate(world.state)).toBe(candidate);
        expect(splitIslandCandidate(sibling.state)).toBe(otherCandidate);
        world.createFilterJoint(a, b);
        expect(splitIslandCandidate(world.state)).toBe(-1);
    } finally {
        world.destroy();
        sibling.destroy();
    }
});
