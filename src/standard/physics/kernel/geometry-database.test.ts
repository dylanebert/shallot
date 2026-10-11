import { expect, test } from "bun:test";
import { restore, snapshot } from "../api/snapshot";
import { BodyType, defaultSurfaceMaterial, ShapeType } from "../common/types";
import { createCompound } from "../shapes/compound";
import { createGrid } from "../shapes/heightfield";
import { createGridMesh } from "../shapes/mesh";
import {
    createCompoundShape,
    createHeightFieldShape,
    createMeshShape,
    destroyShape,
} from "../shapes/shape";
import { createTestBody, createTestWorld, destroyTestWorld, testShapeDef } from "../testing";
import { kernel } from "./kernel";
import { S_GEO_REFERENCE, SHAPE_STRIDE } from "./shapecolumns";

test("kernel geometry uploads are caller-identity keyed, refcounted, world-local and snapshot-stable", () => {
    const source = createTestWorld();
    const target = createTestWorld();
    const mesh = createGridMesh(3, 3, 1, 0, true);
    const field = createGrid(3, 3, { x: 1, y: 1, z: 1 }, false);
    const compound = createCompound({
        spheres: [
            {
                sphere: { center: { x: 0, y: 0, z: 0 }, radius: 0.5 },
                material: defaultSurfaceMaterial(),
            },
        ],
    })!;
    try {
        const k = kernel(source.ecsState);
        const sourceWorldId = source.worldId;
        const startingBytes = k.geometryDatabaseAllocationBytes(sourceWorldId);
        expect(startingBytes).toBe(0);
        const body = createTestBody(source, { type: BodyType.Static });
        const meshA = createMeshShape(source, body, testShapeDef(), mesh, { x: 1, y: 1, z: 1 })!;
        const meshB = createMeshShape(source, body, testShapeDef(), mesh, { x: 1, y: 1, z: 1 })!;
        const heightA = createHeightFieldShape(source, body, testShapeDef(), field)!;
        const heightB = createHeightFieldShape(source, body, testShapeDef(), field)!;
        const compoundA = createCompoundShape(source, body, testShapeDef(), compound)!;
        const compoundB = createCompoundShape(source, body, testShapeDef(), compound)!;
        const ptr = (world: typeof source, shape: number) =>
            world.shapeStore.shapeU[shape * SHAPE_STRIDE + S_GEO_REFERENCE];
        const meshPtr = ptr(source, meshA);
        const heightPtr = ptr(source, heightA);
        const compoundPtr = ptr(source, compoundA);
        expect(ptr(source, meshB)).toBe(meshPtr);
        expect(ptr(source, heightB)).toBe(heightPtr);
        expect(ptr(source, compoundB)).toBe(compoundPtr);
        expect(k.geometryDatabaseRefs(source.worldId, ShapeType.Mesh, meshPtr)).toBe(2);
        expect(k.geometryDatabaseRefs(source.worldId, ShapeType.HeightField, heightPtr)).toBe(2);
        expect(k.geometryDatabaseRefs(source.worldId, ShapeType.Compound, compoundPtr)).toBe(2);
        expect(k.geometryDatabaseAllocationBytes(sourceWorldId)).toBeGreaterThan(startingBytes);

        restore(target, snapshot(source));
        const targetKernel = kernel(target.ecsState);
        const targetPtr = (shape: number) => ptr(target, shape);
        const targetMeshPtr = targetPtr(meshA);
        const targetHeightPtr = targetPtr(heightA);
        const targetCompoundPtr = targetPtr(compoundA);
        expect(targetMeshPtr).not.toBe(meshPtr);
        expect(targetHeightPtr).not.toBe(heightPtr);
        expect(targetCompoundPtr).not.toBe(compoundPtr);
        expect(
            targetKernel.geometryDatabaseRefs(target.worldId, ShapeType.Mesh, targetMeshPtr),
        ).toBe(2);
        expect(
            targetKernel.geometryDatabaseRefs(
                target.worldId,
                ShapeType.HeightField,
                targetHeightPtr,
            ),
        ).toBe(2);
        expect(
            targetKernel.geometryDatabaseRefs(
                target.worldId,
                ShapeType.Compound,
                targetCompoundPtr,
            ),
        ).toBe(2);
        expect([...target.geometryIdentityValues.values()]).toEqual([mesh, field, compound]);
        const targetBody = createTestBody(target, { type: BodyType.Static });
        const reused = createMeshShape(target, targetBody, testShapeDef(), mesh, {
            x: 1,
            y: 1,
            z: 1,
        })!;
        expect(targetPtr(reused)).toBe(targetMeshPtr);
        expect(
            targetKernel.geometryDatabaseRefs(target.worldId, ShapeType.Mesh, targetMeshPtr),
        ).toBe(3);
        destroyShape(target, reused, true);

        destroyTestWorld(source);
        expect(k.geometryDatabaseAllocationBytes(sourceWorldId)).toBe(startingBytes);
        expect(
            targetKernel.geometryDatabaseRefs(target.worldId, ShapeType.Mesh, targetMeshPtr),
        ).toBe(2);
    } finally {
        if (source.inUse) destroyTestWorld(source);
        destroyTestWorld(target);
    }
});
