import { expect, test } from "bun:test";
import { PhysicsWorld } from "../api/world";
import { f32, xf } from "../common/math";
import { BodyType, defaultSurfaceMaterial } from "../common/types";
import { createCompound } from "../shapes/compound";
import { cloneHull, makeBoxHull, readShapeHull } from "../shapes/hull";
import { hullDatabaseIndex, stageHullUpload } from "./geocolumns";
import { kernel } from "./kernel";
import { S_GEO_REFERENCE, SHAPE_STRIDE } from "./shapecolumns";

function compoundHullPointer(world: PhysicsWorld, shape: number): number {
    const store = world.state.shapeStore;
    store.refreshViews();
    const pointer = store.shapeU[shape * SHAPE_STRIDE + S_GEO_REFERENCE];
    const view = new DataView(store.materialU.buffer);
    const hullArray = view.getUint32(pointer + 84, true);
    const hullImage = view.getUint32(pointer + hullArray + 28, true);
    return pointer + hullImage;
}

test("compound hull images relocate with caller-owned compound identity and survive source destruction", () => {
    const source = new PhysicsWorld(),
        target = new PhysicsWorld();
    try {
        const hull = makeBoxHull(1, 1, 1);
        const data = createCompound({
            hulls: [{ hull, transform: xf.identity(), material: defaultSurfaceMaterial() }],
        })!;
        const body = source.createBody({ type: BodyType.Static });
        const shape = body.createCompound({}, data)!;
        body.createCompound({}, data);
        const old = compoundHullPointer(source, shape.id.index1 - 1);
        target.restore(source.snapshot());
        const pointer = compoundHullPointer(target, shape.id.index1 - 1);
        expect(pointer).not.toBe(old);
        expect(
            new DataView(target.state.shapeStore.materialU.buffer).getBigUint64(pointer, true),
        ).toBe(0x4a4c9587de57485cn);
        expect([...target.state.geometryIdentityValues.values()][0]).toBe(data);
        source.destroy();
        const hit = target.castRayClosest({ x: -3, y: 0, z: 0 }, { x: 6, y: 0, z: 0 });
        expect(hit.hit).toBe(true);
        expect(hit.fraction).toBe(f32(1 / 3));
    } finally {
        if (source.state.inUse) source.destroy();
        target.destroy();
    }
});

test("kernel hull references share byte-identical data, distinguish equal hashes with different bytes, and snapshot their lifetime", () => {
    const world = new PhysicsWorld();
    const other = new PhysicsWorld();
    const hull = makeBoxHull(1, 1, 1);
    try {
        const body = world.createBody({ type: BodyType.Dynamic });
        const first = body.createHull({ density: 1 }, hull);
        const second = body.createHull({ density: 1 }, cloneHull(hull));
        const k = kernel(world.state.ecsState);
        let handle = hullDatabaseIndex(world.state, hull);
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
        handle = hullDatabaseIndex(world.state, hull);
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
        expect(handle).not.toBe(siblingHandle);
        world.destroy();
        expect(sibling.computeMassData().mass).toBe(8);
        const size = stageHullUpload(other.state, hull);
        expect(k.hullDatabaseLookup(other.state.worldId, size) >>> 0).toBe(siblingHandle);
    } finally {
        if (world.state.inUse) world.destroy();
        other.destroy();
    }
});
