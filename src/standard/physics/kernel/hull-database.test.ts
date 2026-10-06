import { expect, test } from "bun:test";
import { PhysicsWorld } from "../api/world";
import { BodyType } from "../common/types";
import { cloneHull, makeBoxHull, readShapeHull } from "../shapes/hull";
import { hullDatabaseIndex, stageHullUpload } from "./geocolumns";
import { kernel } from "./kernel";
import { S_GEO_REFERENCE, SHAPE_STRIDE } from "./shapecolumns";

test("kernel hull references share byte-identical data, distinguish equal hashes with different bytes, and snapshot their lifetime", () => {
    const world = new PhysicsWorld();
    const other = new PhysicsWorld();
    const hull = makeBoxHull(1, 1, 1);
    try {
        const body = world.createBody({ type: BodyType.Dynamic });
        const first = body.createHull({ density: 1 }, hull);
        const second = body.createHull({ density: 1 }, cloneHull(hull));
        const k = kernel(world.state.ecsState);
        const handle = hullDatabaseIndex(world.state, hull);
        expect(k.hullDatabaseCount(world.state.worldId)).toBe(1);
        expect(k.hullDatabaseRefs(world.state.worldId, handle)).toBe(2);
        const sameHash = cloneHull(hull);
        sameHash.innerRadius = 0.5;
        const different = body.createHull({}, sameHash);
        const differentHandle = hullDatabaseIndex(world.state, sameHash);
        expect(differentHandle).not.toBe(handle);
        expect(k.hullDatabaseCount(world.state.worldId)).toBe(2);
        different.destroy();
        const snapshot = world.snapshot();
        first.destroy();
        expect(k.hullDatabaseRefs(world.state.worldId, handle)).toBe(1);
        second.destroy();
        expect(k.hullDatabaseCount(world.state.worldId)).toBe(0);
        world.restore(snapshot);
        expect(k.hullDatabaseCount(world.state.worldId)).toBe(1);
        expect(k.hullDatabaseRefs(world.state.worldId, handle)).toBe(2);
        const id = first.id.index1 - 1;
        expect(world.state.shapeStore.shapeU[id * SHAPE_STRIDE + S_GEO_REFERENCE]).toBe(handle);
        expect(first.computeMassData().mass).toBe(8);
        const output = readShapeHull(world.state, id);
        expect(output.hash).toBe(hull.hash);
        expect(output.points).toEqual(hull.points);
        output.points[0].x = 123;
        expect(readShapeHull(world.state, id).points).toEqual(hull.points);
        const sibling = other
            .createBody({ type: BodyType.Dynamic })
            .createHull({ density: 1 }, hull);
        const siblingHandle = hullDatabaseIndex(other.state, hull);
        expect(k.hullDataPtr(world.state.worldId, handle)).not.toBe(
            k.hullDataPtr(other.state.worldId, siblingHandle),
        );
        world.destroy();
        expect(sibling.computeMassData().mass).toBe(8);
        const size = stageHullUpload(other.state, hull);
        expect(k.hullDatabaseLookup(other.state.worldId, size) >>> 0).toBe(siblingHandle);
    } finally {
        if (world.state.inUse) world.destroy();
        other.destroy();
    }
});
