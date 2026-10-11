import { expect, test } from "bun:test";
import { BodyType } from "../common/types";
import { createSphereShape } from "../shapes/shape";
import { createTestBody, createTestWorld, destroyTestWorld, testShapeDef } from "../testing";
import { readFatAabb, readShapeAabb, writeTightAabb } from "./shapecolumns";

const box = () => ({ lowerBound: { x: 0, y: 0, z: 0 }, upperBound: { x: 0, y: 0, z: 0 } });

test("shape bounds read the kernel home and return independent snapshots", () => {
    const world = createTestWorld({ enableSleep: false });
    try {
        const body = createTestBody(world, { type: BodyType.Dynamic });
        const id = createSphereShape(world, body, testShapeDef(), {
            center: { x: 0, y: 0, z: 0 },
            radius: 1,
        })!;
        const old = readShapeAabb(world, id, box());
        const next = { lowerBound: { x: -2, y: -3, z: -4 }, upperBound: { x: 2, y: 3, z: 4 } };
        world.shapeStore.refreshViews();
        writeTightAabb(world.shapeStore.shapeF, id, next);
        expect(readShapeAabb(world, id, box())).toEqual(next);
        expect(old).not.toEqual(next);
        world.shapeStore.writeFatAabb(id, next);
        expect(readFatAabb(world, id, box())).toEqual(next);
    } finally {
        destroyTestWorld(world);
    }
});
