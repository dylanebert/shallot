import { expect, test } from "bun:test";
import { restore, snapshot } from "../api/snapshot";
import { transform } from "./math";

const f32 = Math.fround;

import { BodyType, defaultSurfaceMaterial } from "../common/types";
import { makeBoxHull } from "../shapes/build-hull";
import { createCompound } from "../shapes/compound";
import { cloneHull, readShapeHull } from "../shapes/hull";
import {
    computeShapeMass,
    createCompoundShape,
    createHullShape,
    destroyShape,
} from "../shapes/shape";
import { createTestBody, createTestWorld, destroyTestWorld, testShapeDef } from "../testing";
import type { WorldState } from "../world/world";
import { hullDatabaseIndex, stageHullUpload } from "./geocolumns";
import { kernel } from "./kernel";
import { queryColumns } from "./querycolumns";
import { S_GEO_REFERENCE, SHAPE_STRIDE } from "./shapecolumns";

function compoundHullPointer(world: WorldState, shape: number): number {
    const store = world.shapeStore;
    store.refreshViews();
    const pointer = store.shapeU[shape * SHAPE_STRIDE + S_GEO_REFERENCE];
    const view = new DataView(store.materialU.buffer);
    const hullArray = view.getUint32(pointer + 84, true);
    const hullImage = view.getUint32(pointer + hullArray + 28, true);
    return pointer + hullImage;
}

test("compound hull images relocate with caller-owned compound identity and survive source destruction", () => {
    const source = createTestWorld();
    const target = createTestWorld();
    try {
        const hull = makeBoxHull(1, 1, 1);
        const data = createCompound({
            hulls: [{ hull, transform: transform.identity(), material: defaultSurfaceMaterial() }],
        })!;
        const body = createTestBody(source, { type: BodyType.Static });
        const shape = createCompoundShape(source, body, testShapeDef(), data)!;
        createCompoundShape(source, body, testShapeDef(), data);
        const old = compoundHullPointer(source, shape);
        restore(target, snapshot(source));
        const pointer = compoundHullPointer(target, shape);
        expect(pointer).not.toBe(old);
        expect(new DataView(target.shapeStore.materialU.buffer).getBigUint64(pointer, true)).toBe(
            0x4a4c9587de57485cn,
        );
        expect([...target.geometryIdentityValues.values()][0]).toBe(data);
        destroyTestWorld(source);
        const query = queryColumns(target);
        const targetKernel = kernel(target.ecsState);
        query.prepare({ x: -3, y: 0, z: 0 });
        query.translation({ x: 6, y: 0, z: 0 });
        targetKernel.worldQuery(target.worldId, 3, 0);
        expect(query.resultU[0]).not.toBe(0xffffffff);
        expect(query.resultF[5]).toBe(f32(1 / 3));
    } finally {
        if (source.inUse) destroyTestWorld(source);
        destroyTestWorld(target);
    }
});

test("kernel hull references share byte-identical data, distinguish equal hashes with different bytes, and snapshot their lifetime", () => {
    const world = createTestWorld();
    const other = createTestWorld();
    const hull = makeBoxHull(1, 1, 1);
    try {
        const body = createTestBody(world, { type: BodyType.Dynamic });
        const first = createHullShape(world, body, testShapeDef({ density: 1 }), hull)!;
        const second = createHullShape(world, body, testShapeDef({ density: 1 }), cloneHull(hull))!;
        const k = kernel(world.ecsState);
        let handle = hullDatabaseIndex(world, hull);
        expect(k.hullDatabaseCount(world.worldId)).toBe(1);
        expect(k.hullDatabaseRefs(world.worldId, handle)).toBe(2);
        const sameHash = cloneHull(hull);
        sameHash.innerRadius = 0.5;
        const different = createHullShape(world, body, testShapeDef(), sameHash)!;
        const differentHandle = hullDatabaseIndex(world, sameHash);
        expect(differentHandle).not.toBe(handle);
        expect(k.hullDatabaseCount(world.worldId)).toBe(2);
        destroyShape(world, different, true);
        const saved = snapshot(world);
        destroyShape(world, first, true);
        expect(k.hullDatabaseRefs(world.worldId, handle)).toBe(1);
        destroyShape(world, second, true);
        expect(k.hullDatabaseCount(world.worldId)).toBe(0);
        restore(world, saved);
        handle = hullDatabaseIndex(world, hull);
        expect(k.hullDatabaseCount(world.worldId)).toBe(1);
        expect(k.hullDatabaseRefs(world.worldId, handle)).toBe(2);
        expect(world.shapeStore.shapeU[first * SHAPE_STRIDE + S_GEO_REFERENCE]).toBe(handle);
        expect(computeShapeMass(world, first).mass).toBe(8);
        const output = readShapeHull(world, first);
        expect(output.hash).toBe(hull.hash);
        expect(output.points).toEqual(hull.points);
        output.points[0]!.x = 123;
        expect(readShapeHull(world, first).points).toEqual(hull.points);
        const siblingBody = createTestBody(other, { type: BodyType.Dynamic });
        const sibling = createHullShape(other, siblingBody, testShapeDef({ density: 1 }), hull)!;
        const siblingHandle = hullDatabaseIndex(other, hull);
        expect(handle).not.toBe(siblingHandle);
        destroyTestWorld(world);
        expect(computeShapeMass(other, sibling).mass).toBe(8);
        const size = stageHullUpload(other, hull);
        expect(k.hullDatabaseLookup(other.worldId, size) >>> 0).toBe(siblingHandle);
    } finally {
        if (world.inUse) destroyTestWorld(world);
        destroyTestWorld(other);
    }
});
