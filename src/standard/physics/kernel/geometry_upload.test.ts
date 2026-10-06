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
import { uploadGeometry } from "./geocolumns";
import { init, kernel } from "./kernel";
import { S_GEO_REFERENCE, SHAPE_STRIDE } from "./shapecolumns";

await init(undefined, { threads: 0 });
const unit = { x: 1, y: 1, z: 1 };
const advance = (world: WorldState): void => step(world, 1 / 60, 4);
const body = (world: WorldState) =>
    createBody(world, { ...defaultBodyDef(), position: { x: 100, y: 100, z: 100 } });

test("hull upload derives padded SoA vertices and normals from authoring, without overlapping adjacent hulls", () => {
    const box = makeBoxHull(1, 2, 3);
    const hulls = [5, 8].map((count) => ({
        ...box,
        vertexCount: count,
        faceCount: 5,
        points: box.points.slice(0, count),
        planes: box.planes.slice(0, 5),
    }));
    uploadGeometry(undefined, hulls);
    const k = kernel(undefined);
    const layout = new Uint32Array(k.memory.buffer, k.geoLayoutPtr(), 8);
    const directory = new Uint32Array(k.memory.buffer, layout[0], hulls.length);
    for (const [i, h] of hulls.entries()) {
        const nv = (h.vertexCount + 3) & ~3;
        const nf = (h.faceCount + 3) & ~3;
        const base = layout[0] + directory[i];
        const record = new Uint32Array(k.memory.buffer, base, 36);
        const soa = new Float32Array(k.memory.buffer, base + record[33], 3 * (nv + nf));
        expect(record[34]).toBe(record[33] + 12 * nv);
        const floats = new Float32Array(k.memory.buffer, base, 36);
        expect(Array.from(record.slice(0, 4))).toEqual([0xde57485c, 0x4a4c9587, h.hash >>> 0, 0]);
        expect(Array.from(floats.slice(4, 25))).toEqual([
            h.aabb.lowerBound.x,
            h.aabb.lowerBound.y,
            h.aabb.lowerBound.z,
            h.aabb.upperBound.x,
            h.aabb.upperBound.y,
            h.aabb.upperBound.z,
            h.surfaceArea,
            h.volume,
            h.innerRadius,
            h.center.x,
            h.center.y,
            h.center.z,
            h.centralInertia.cx.x,
            h.centralInertia.cx.y,
            h.centralInertia.cx.z,
            h.centralInertia.cy.x,
            h.centralInertia.cy.y,
            h.centralInertia.cy.z,
            h.centralInertia.cz.x,
            h.centralInertia.cz.y,
            h.centralInertia.cz.z,
        ]);
        const bytes = new Uint8Array(k.memory.buffer, base, record[35]);
        expect(Array.from(bytes.slice(record[26], record[26] + h.vertexCount))).toEqual(
            h.vertices.slice(0, h.vertexCount).map((v) => v.edge),
        );
        expect(Array.from(bytes.slice(record[29], record[29] + 4 * h.edgeCount))).toEqual(
            h.edges.flatMap((e) => [e.next, e.twin, e.origin, e.face]),
        );
        expect(Array.from(bytes.slice(record[32], record[32] + h.faceCount))).toEqual(
            h.faces.slice(0, h.faceCount).map((v) => v.edge),
        );
        for (const lane of [26, 27, 29, 31, 32, 33, 34, 35]) {
            expect(record[lane] % 8).toBe(0);
        }
        for (let lane = 0; lane < nv; ++lane) {
            const p = h.points[lane < h.vertexCount ? lane : 0];
            expect([soa[lane], soa[nv + lane], soa[2 * nv + lane]]).toEqual([p.x, p.y, p.z]);
        }
        for (let lane = 0; lane < nf; ++lane) {
            const n = lane < h.faceCount ? h.planes[lane].normal : { x: 0, y: 0, z: 0 };
            expect([
                soa[3 * nv + lane],
                soa[3 * nv + nf + lane],
                soa[3 * nv + 2 * nf + lane],
            ]).toEqual([n.x, n.y, n.z]);
        }
    }
});

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
            const ball = createBody(world, {
                ...defaultBodyDef(),
                type: BodyType.Dynamic,
                position: { x: 200, y: 200, z: 200 },
            });
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
        const reference = world.shapeStore.shapeU[shape * SHAPE_STRIDE + S_GEO_REFERENCE];
        expect(reference).toBeGreaterThan(0);
        expect(world.shapeStore.shapeU[duplicate * SHAPE_STRIDE + S_GEO_REFERENCE]).toBe(reference);
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
            const before = world.geometryUploadCount;
            const shape = create(first)!;
            advance(world);
            expect(world.geometryUploadCount).toBe(before + 1);
            const second = body(world);
            const duplicate = create(second)!;
            expect(world.shapeStore.shapeU[duplicate * SHAPE_STRIDE + S_GEO_REFERENCE]).toBe(
                world.shapeStore.shapeU[shape * SHAPE_STRIDE + S_GEO_REFERENCE],
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
        expect(world.hullDatabase.get(hull.hash | 0)?.refCount).toBe(1);
        expect(world.heightFieldDatabase.size).toBe(0);
        expect(world.compoundDatabase.size).toBe(0);
    } finally {
        destroyWorld(world);
    }
});
