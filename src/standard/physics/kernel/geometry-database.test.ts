import { expect, test } from "bun:test";
import { PhysicsWorld } from "../api/world";
import { BodyType, defaultSurfaceMaterial, ShapeType } from "../common/types";
import { createCompound } from "../shapes/compound";
import { createGrid } from "../shapes/heightfield";
import { createGridMesh } from "../shapes/mesh";
import { kernel } from "./kernel";
import { SHAPE_STRIDE } from "./shapecolumns";

test("kernel geometry uploads are caller-identity keyed, refcounted, world-local and snapshot-stable", () => {
    const source = new PhysicsWorld();
    const target = new PhysicsWorld();
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
        const body = source.createBody({ type: BodyType.Static });
        const meshA = body.createMesh({}, mesh)!;
        const meshB = body.createMesh({}, mesh)!;
        const heightA = body.createHeightField({}, field)!;
        const heightB = body.createHeightField({}, field)!;
        const compoundA = body.createCompound({}, compound)!;
        const compoundB = body.createCompound({}, compound)!;
        const k = kernel(source.state.ecsState);
        const ptr = (shape: number) => source.state.shapeStore.shapeU[shape * SHAPE_STRIDE + 2];
        const meshPtr = ptr(meshA.id.index1 - 1);
        const heightPtr = ptr(heightA.id.index1 - 1);
        const compoundPtr = ptr(compoundA.id.index1 - 1);
        expect(ptr(meshB.id.index1 - 1)).toBe(meshPtr);
        expect(ptr(heightB.id.index1 - 1)).toBe(heightPtr);
        expect(ptr(compoundB.id.index1 - 1)).toBe(compoundPtr);
        expect(k.geometryDatabaseRefs(source.state.worldId, ShapeType.Mesh, meshPtr)).toBe(2);
        expect(k.geometryDatabaseRefs(source.state.worldId, ShapeType.HeightField, heightPtr)).toBe(
            2,
        );
        expect(k.geometryDatabaseRefs(source.state.worldId, ShapeType.Compound, compoundPtr)).toBe(
            2,
        );

        target.restore(source.snapshot());
        const targetKernel = kernel(target.state.ecsState);
        const targetPtr = (shape: number) =>
            target.state.shapeStore.shapeU[shape * SHAPE_STRIDE + 2];
        const targetMeshPtr = targetPtr(meshA.id.index1 - 1);
        const targetHeightPtr = targetPtr(heightA.id.index1 - 1);
        const targetCompoundPtr = targetPtr(compoundA.id.index1 - 1);
        expect(targetMeshPtr).not.toBe(meshPtr);
        expect(targetHeightPtr).not.toBe(heightPtr);
        expect(targetCompoundPtr).not.toBe(compoundPtr);
        expect(
            targetKernel.geometryDatabaseRefs(target.state.worldId, ShapeType.Mesh, targetMeshPtr),
        ).toBe(2);
        expect(
            targetKernel.geometryDatabaseRefs(
                target.state.worldId,
                ShapeType.HeightField,
                targetHeightPtr,
            ),
        ).toBe(2);
        expect(
            targetKernel.geometryDatabaseRefs(
                target.state.worldId,
                ShapeType.Compound,
                targetCompoundPtr,
            ),
        ).toBe(2);
        expect([...target.state.meshDatabase.keys()][0]).toBe(mesh);
        expect([...target.state.heightFieldDatabase.keys()][0]).toBe(field);
        expect([...target.state.compoundDatabase.keys()][0]).toBe(compound);
        expect(target.state.shapeGeometry[meshA.id.index1 - 1].mesh?.data).toBe(mesh);
        expect(target.state.shapeGeometry[heightA.id.index1 - 1].heightField).toBe(field);
        expect(target.state.shapeGeometry[compoundA.id.index1 - 1].compound).toBe(compound);

        source.destroy();
        expect(target.castRayClosest({ x: 0, y: 2, z: 0 }, { x: 0, y: -4, z: 0 }).hit).toBe(true);
    } finally {
        if (source.state.inUse) source.destroy();
        target.destroy();
    }
});
