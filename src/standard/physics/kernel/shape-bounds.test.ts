import { expect, test } from "bun:test";
import { BodyType, PhysicsWorld } from "../api";
import { readFatAabb, readShapeAabb, writeTightAabb } from "./shapecolumns";

const box = () => ({ lowerBound: { x: 0, y: 0, z: 0 }, upperBound: { x: 0, y: 0, z: 0 } });

test("public shape bounds read the kernel home and return independent snapshots", () => {
    const world = new PhysicsWorld({ enableSleep: false });
    try {
        const body = world.createBody({ type: BodyType.Dynamic });
        const shape = body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 1 });
        const id = shape.id.index1 - 1;
        const old = shape.getAABB();
        const next = { lowerBound: { x: -2, y: -3, z: -4 }, upperBound: { x: 2, y: 3, z: 4 } };
        world.state.shapeStore.refreshViews();
        writeTightAabb(world.state.shapeStore.shapeF, id, next);
        expect(shape.getAABB()).toEqual(next);
        expect(readShapeAabb(world.state, id, box())).toEqual(next);
        expect(old).not.toEqual(next);
        world.state.shapeStore.writeFatAabb(id, next);
        expect(readFatAabb(world.state, id, box())).toEqual(next);
    } finally {
        world.destroy();
    }
});
