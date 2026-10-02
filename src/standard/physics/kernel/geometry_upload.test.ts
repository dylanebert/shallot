import { expect, test } from "bun:test";
import {
    BodyType,
    defaultBodyDef,
    defaultShapeDef,
    defaultSurfaceMaterial,
    defaultWorldDef,
} from "../common/types";
import { createCompound } from "../shapes/compound";
import { createGrid } from "../shapes/heightfield";
import { makeBoxHull } from "../shapes/hull";
import { createGridMesh } from "../shapes/mesh";
import {
    createCapsuleShape,
    createCompoundShape,
    createHeightFieldShape,
    createHullShape,
    createMeshShape,
    createSphereShape,
} from "../shapes/shape";
import { step } from "../solver/step";
import { createBody, destroyBody } from "../world/body";
import { createWorld, destroyWorld, getWorld, type WorldState } from "../world/world";
import { init } from "./kernel";
import { S_GEO_REFERENCE, SHAPE_STRIDE } from "./shapecolumns";

await init(undefined, { threads: 0 });
const unit = { x: 1, y: 1, z: 1 };
const advance = (world: WorldState): void => step(world, 1 / 60, 4);
const body = (world: WorldState) =>
    world.bodies[createBody(world, { ...defaultBodyDef(), position: { x: 100, y: 100, z: 100 } })];

test("sphere and capsule body churn uploads no geometry, and only a mesh datum entering or leaving the set uploads", () => {
    const world = getWorld(
        createWorld(undefined, { ...defaultWorldDef(), gravity: { x: 0, y: 0, z: 0 } }),
    ) as WorldState;
    try {
        const ground = body(world);
        createMeshShape(world, ground, defaultShapeDef(), createGridMesh(2, 2, 1, 0, true), unit);
        advance(world);
        const mesh = createGridMesh(4, 4, 1, 0, true);
        const first = body(world);
        const before = world.geometryUploadCount;
        const shape = createMeshShape(world, first, defaultShapeDef(), mesh, unit)!;
        advance(world);
        expect(world.geometryUploadCount - before).toBe(1);
        const resident = world.geometryUploadCount;
        for (const kind of ["sphere", "capsule"]) {
            const ball =
                world.bodies[
                    createBody(world, {
                        ...defaultBodyDef(),
                        type: BodyType.Dynamic,
                        position: { x: 200, y: 200, z: 200 },
                    })
                ];
            if (kind === "sphere")
                createSphereShape(world, ball, defaultShapeDef(), {
                    center: { x: 0, y: 0, z: 0 },
                    radius: 1,
                });
            else
                createCapsuleShape(world, ball, defaultShapeDef(), {
                    center1: { x: 0, y: 0, z: 0 },
                    center2: { x: 0, y: 1, z: 0 },
                    radius: 1,
                });
            advance(world);
            expect(world.geometryUploadCount).toBe(resident);
            destroyBody(world, ball);
            advance(world);
            expect(world.geometryUploadCount).toBe(resident);
        }
        const second = body(world);
        const duplicate = createMeshShape(world, second, defaultShapeDef(), mesh, unit)!;
        const reference = world.shapeStore.shapeU[shape.id * SHAPE_STRIDE + S_GEO_REFERENCE];
        expect(reference).toBeGreaterThan(0);
        expect(world.shapeStore.shapeU[duplicate.id * SHAPE_STRIDE + S_GEO_REFERENCE]).toBe(
            reference,
        );
        advance(world);
        expect(world.geometryUploadCount).toBe(resident);
        destroyBody(world, second);
        advance(world);
        expect(world.geometryUploadCount).toBe(resident);
        destroyBody(world, first);
        advance(world);
        expect(world.geometryUploadCount).toBe(resident + 1);
    } finally {
        destroyWorld(world);
    }
});

test("height-field and compound instances retain resident records, including a compound's shared hull and mesh dependencies", () => {
    const world = getWorld(createWorld(undefined, defaultWorldDef())) as WorldState;
    try {
        const mesh = createGridMesh(2, 2, 1, 0, true);
        const ground = body(world);
        createMeshShape(world, ground, defaultShapeDef(), mesh, unit);
        const hull = makeBoxHull(1, 1, 1);
        createHullShape(world, ground, defaultShapeDef(), hull);
        advance(world);
        const field = createGrid(4, 4, unit, false);
        const compound = createCompound({
            meshes: [
                {
                    meshData: mesh,
                    transform: { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
                    scale: unit,
                    materials: [defaultSurfaceMaterial()],
                    materialCount: 1,
                },
            ],
            hulls: [
                {
                    hull,
                    transform: { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
                    material: defaultSurfaceMaterial(),
                },
            ],
        });
        if (!compound) throw new Error("compound creation failed");
        for (const kind of ["height", "compound"]) {
            const first = body(world);
            const create = (b: typeof first) =>
                kind === "height"
                    ? createHeightFieldShape(world, b, defaultShapeDef(), field)
                    : createCompoundShape(world, b, defaultShapeDef(), compound);
            const shape = create(first)!;
            const before = world.geometryUploadCount;
            advance(world);
            expect(world.geometryUploadCount).toBe(before + 1);
            const second = body(world);
            const duplicate = create(second)!;
            expect(world.shapeStore.shapeU[duplicate.id * SHAPE_STRIDE + S_GEO_REFERENCE]).toBe(
                world.shapeStore.shapeU[shape.id * SHAPE_STRIDE + S_GEO_REFERENCE],
            );
            advance(world);
            expect(world.geometryUploadCount).toBe(before + 1);
            destroyBody(world, first);
            advance(world);
            expect(world.geometryUploadCount).toBe(before + 1);
            destroyBody(world, second);
            advance(world);
            expect(world.geometryUploadCount).toBe(before + 2);
        }
        expect(world.meshDatabase.get(mesh)?.refCount).toBe(1);
        expect(world.hullDatabase.get(hull.hash)?.refCount).toBe(1);
        expect(world.heightFieldDatabase.size).toBe(0);
        expect(world.compoundDatabase.size).toBe(0);
    } finally {
        destroyWorld(world);
    }
});
