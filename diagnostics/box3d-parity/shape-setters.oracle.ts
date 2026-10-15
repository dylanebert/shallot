import { expect, setDefaultTimeout, test } from "bun:test";
import { nativeSseOutput } from "./native-evidence";
import { BodyType } from "../../src/standard/physics/common/types";
import { defaultSurfaceMaterial, type SurfaceMaterial } from "../../src/standard/physics/common/types";
import { createHull } from "../../src/standard/physics/shapes/build-hull";
import { readShapeHull } from "../../src/standard/physics/shapes/hull";
import { createMesh } from "../../src/standard/physics/shapes/mesh";
import {
    createCapsuleShape,
    createHullShape,
    createMeshShape,
    createSphereShape,
    getShapeFilter,
    getShapeGeometryScale,
    getShapeMaterial,
    getShapeMaterials,
    getShapeDensity,
    readShapeCapsule,
    readShapeSphere,
    setShapeCapsule,
    setShapeDensity,
    setShapeFilter,
    setShapeFriction,
    setShapeHull,
    setShapeMesh,
    setShapeMeshMaterial,
    setShapeRestitution,
    setShapeSphere,
    setShapeSurfaceMaterial,
} from "../../src/standard/physics/shapes/shape";
import { ShapeFlags, setShapeFlag, shapeFlag } from "../../src/standard/physics/kernel/shaperecords";
import { readShapeMesh } from "../../src/standard/physics/shapes/readgeometry";
import { init } from "../../src/standard/physics/kernel/kernel";
import { createTestBody, createTestWorld, destroyTestWorld, testShapeDef } from "../../src/standard/physics/testing";
import type { WorldState } from "../../src/standard/physics/world/world";

setDefaultTimeout(180_000);
await init(undefined, { threads: 0 });
const rows = nativeSseOutput("shape-setters.c", "").trim().split("\n");
const expected = new Map(rows.map((row) => [row.slice(0, row.indexOf(" ")), row]));
function values(name: string): string[] {
    const row = expected.get(name);
    if (!row) throw new Error(`missing native Shape setter row ${name}`);
    return row.split(/\s+/).slice(1);
}
function floats(name: string, actual: number[]): void {
    expect(actual.map(Math.fround), name).toEqual(
        values(name).slice(0, actual.length).map(Number).map(Math.fround),
    );
}
function material(name: string, value: SurfaceMaterial): void {
    const row = values(name);
    floats(name, [
        value.friction,
        value.restitution,
        value.rollingResistance,
        value.tangentVelocity[0],
        value.tangentVelocity[1],
        value.tangentVelocity[2],
    ]);
    expect(value.userMaterialId).toBe(BigInt(row[6]!));
    expect(value.customColor >>> 0).toBe(Number(row[7]) >>> 0);
}
function scene(type = BodyType.Dynamic): { world: WorldState; body: number } {
    const world = createTestWorld({
        gravity: { x: 0, y: 0, z: 0 },
        workerCount: 1,
    });
    return { world, body: createTestBody(world, { type }) };
}
function sphereShape(world: WorldState, body: number): number {
    return createSphereShape(
        world,
        body,
        testShapeDef(),
        { center: { x: 0, y: 0, z: 0 }, radius: 0.5 },
    );
}
function emptyMaterial(): SurfaceMaterial {
    return {
        friction: 0,
        restitution: 0,
        rollingResistance: 0,
        tangentVelocity: [0, 0, 0],
        userMaterialId: 0n,
        customColor: 0,
    };
}
function meshData(offset: number) {
    return createMesh({
        vertices: [
            { x: offset, y: 0, z: 0 },
            { x: offset + 1, y: 0, z: 0 },
            { x: offset, y: 0, z: 1 },
        ],
        indices: [0, 2, 1],
        useMedianSplit: false,
        identifyEdges: false,
    })!;
}

for (const [name, setter, getter] of [
    ["density", (world: WorldState, shape: number) => setShapeDensity(world, shape, 2.75, false), (world: WorldState, shape: number) => getShapeDensity(world, shape)],
    ["friction", (world: WorldState, shape: number) => setShapeFriction(world, shape, 0.37), (world: WorldState, shape: number) => getShapeMaterial(world, shape, emptyMaterial()).friction],
    ["restitution", (world: WorldState, shape: number) => setShapeRestitution(world, shape, 0.61), (world: WorldState, shape: number) => getShapeMaterial(world, shape, emptyMaterial()).restitution],
] as const) {
    test(`Shape.${name} setter/getter matches Box3D`, () => {
        const { world, body } = scene();
        try {
            const shape = sphereShape(world, body);
            setter(world, shape);
            floats(name, [getter(world, shape)]);
        } finally {
            destroyTestWorld(world);
        }
    });
}

test("Shape.setSurfaceMaterial/getSurfaceMaterial matches Box3D", () => {
    const { world, body } = scene();
    try {
        const shape = sphereShape(world, body);
        const next: SurfaceMaterial = {
            friction: 0.23,
            restitution: 0.34,
            rollingResistance: 0.45,
            tangentVelocity: [1.25, -2.5, 3.75],
            userMaterialId: 0x123456789abcdef0n,
            customColor: 0x9a654321,
        };
        setShapeSurfaceMaterial(world, shape, next);
        material("surface", getShapeMaterial(world, shape, emptyMaterial()));
    } finally {
        destroyTestWorld(world);
    }
});

test("Shape.setFilter/getFilter matches Box3D", () => {
    const { world, body } = scene();
    try {
        const shape = sphereShape(world, body);
        setShapeFilter(world, shape, {
            categoryBits: 0x123456789abcdef0n,
            maskBits: 0xfedcba9876543210n,
            groupIndex: -7,
        });
        const actual = getShapeFilter(world, shape);
        const row = values("filter");
        expect([actual.categoryBits, actual.maskBits, actual.groupIndex]).toEqual([
            BigInt(row[0]!),
            BigInt(row[1]!),
            Number(row[2]),
        ]);
    } finally {
        destroyTestWorld(world);
    }
});

for (const [name, flag] of [
    ["sensorEvents", ShapeFlags.enableSensorEvents],
    ["contactEvents", ShapeFlags.enableContactEvents],
    ["hitEvents", ShapeFlags.enableHitEvents],
    ["preSolveEvents", ShapeFlags.enablePreSolveEvents],
] as const) {
    test(`Shape.${name} setter/getter matches Box3D`, () => {
        const { world, body } = scene();
        try {
            const shape = sphereShape(world, body);
            setShapeFlag(world, shape, flag, true);
            expect(Number(shapeFlag(world, shape, flag)), name).toBe(Number(values(name)[0]));
        } finally {
            destroyTestWorld(world);
        }
    });
}

test("Shape.setSphere/getSphere matches Box3D", () => {
    const { world, body } = scene();
    try {
        const shape = sphereShape(world, body);
        setShapeSphere(world, shape, { center: { x: 0.25, y: -0.5, z: 0.75 }, radius: 0.8 });
        const got = readShapeSphere(world, shape);
        floats("sphere", [got.center.x, got.center.y, got.center.z, got.radius]);
    } finally {
        destroyTestWorld(world);
    }
});

test("Shape.setCapsule/getCapsule matches Box3D", () => {
    const { world, body } = scene();
    try {
        const shape = createCapsuleShape(
            world,
            body,
            testShapeDef(),
            {
                center1: { x: 0, y: -0.5, z: 0 },
                center2: { x: 0, y: 0.5, z: 0 },
                radius: 0.25,
            },
        );
        setShapeCapsule(world, shape, {
            center1: { x: -0.75, y: 0.25, z: 1.5 },
            center2: { x: 0.5, y: 1.25, z: -0.25 },
            radius: 0.4,
        });
        const got = readShapeCapsule(world, shape);
        floats("capsule", [
            got.center1.x, got.center1.y, got.center1.z,
            got.center2.x, got.center2.y, got.center2.z,
            got.radius,
        ]);
    } finally {
        destroyTestWorld(world);
    }
});

test("Shape.setHull/readShapeHull matches Box3D", () => {
    const { world, body } = scene();
    const initial = createHull([
        { x: -1, y: -1, z: -1 }, { x: 1, y: -1, z: -1 },
        { x: 1, y: 1, z: -1 }, { x: -1, y: 1, z: -1 },
        { x: -1, y: -1, z: 1 }, { x: 1, y: -1, z: 1 },
        { x: 1, y: 1, z: 1 }, { x: -1, y: 1, z: 1 },
    ], 8)!;
    const next = createHull([
        { x: 0, y: 0, z: 0 }, { x: 1, y: 0, z: 0 },
        { x: 0, y: 1, z: 0 }, { x: 0, y: 0, z: 1 },
    ], 4)!;
    try {
        const shape = createHullShape(world, body, testShapeDef(), initial);
        setShapeHull(world, shape, next);
        const got = readShapeHull(world, shape);
        const expectedRow = values("hull").map(Number);
        const actual = [
            got.vertexCount,
            got.center.x, got.center.y, got.center.z,
            ...got.points.flatMap(({ x, y, z }) => [x, y, z]),
        ];
        expect(actual.map(Math.fround), "hull").toEqual(expectedRow.map(Math.fround));
    } finally {
        destroyTestWorld(world);
    }
});

test("Shape.setMesh/getGeometryScale/readShapeMesh matches Box3D", () => {
    const { world, body } = scene(BodyType.Static);
    try {
        const shape = createMeshShape(world, body, testShapeDef(), meshData(0));
        setShapeMesh(world, shape, meshData(2), { x: 2, y: 0.5, z: -1.5 });
        const got = readShapeMesh(world, shape);
        const expectedRow = values("mesh").map(Number);
        const actual = [
            ...Object.values(getShapeGeometryScale(world, shape)),
            got.data.vertices.length,
            got.data.triangles.length,
            ...got.data.vertices.flatMap(({ x, y, z }) => [x, y, z]),
        ];
        expect(actual.map(Math.fround), "mesh").toEqual(expectedRow.map(Math.fround));
    } finally {
        destroyTestWorld(world);
    }
});

test("Shape.setMeshMaterial/getMaterials matches Box3D", () => {
    const { world, body } = scene(BodyType.Static);
    try {
        const shape = createMeshShape(world, body, testShapeDef(), meshData(0));
        const next: SurfaceMaterial = {
            friction: 0.78,
            restitution: 0.19,
            rollingResistance: 0.28,
            tangentVelocity: [-1, 2, -3],
            userMaterialId: 0x0fedcba987654321n,
            customColor: 0x87654321,
        };
        setShapeMeshMaterial(world, shape, next, 0);
        material("meshMaterial", getShapeMaterials(world, shape)[0]!);
    } finally {
        destroyTestWorld(world);
    }
});
