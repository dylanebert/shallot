import { expect, setDefaultTimeout, test } from "bun:test";
import { nativeSseOutput } from "./native-evidence";
import { PhysicsWorld } from "../../src/standard/physics/api/world";
import { Shape } from "../../src/standard/physics/api/shape";
import { BodyType } from "../../src/standard/physics/common/types";
import { createHull, readShapeHull } from "../../src/standard/physics/shapes/hull";
import { createMesh } from "../../src/standard/physics/shapes/mesh";
import { readShapeMesh } from "../../src/standard/physics/shapes/readgeometry";
import { init } from "../../src/standard/physics/kernel/kernel";
import type { SurfaceMaterial } from "../../src/standard/physics/common/types";

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
        value.tangentVelocity.x,
        value.tangentVelocity.y,
        value.tangentVelocity.z,
    ]);
    expect(value.userMaterialId).toBe(BigInt(row[6]!));
    expect(value.customColor >>> 0).toBe(Number(row[7]) >>> 0);
}
function scene(type = BodyType.Dynamic): { world: PhysicsWorld; body: ReturnType<PhysicsWorld["createBody"]> } {
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 }, workerCount: 1 });
    return { world, body: world.createBody({ type }) };
}
function sphereShape(world: PhysicsWorld, body: ReturnType<PhysicsWorld["createBody"]>): Shape {
    return body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
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
    ["density", (shape: Shape) => shape.setDensity(2.75, false), (shape: Shape) => shape.getDensity()],
    ["friction", (shape: Shape) => shape.setFriction(0.37), (shape: Shape) => shape.getFriction()],
    ["restitution", (shape: Shape) => shape.setRestitution(0.61), (shape: Shape) => shape.getRestitution()],
] as const) {
    test(`Shape.${name} setter/getter matches Box3D`, () => {
        const { world, body } = scene();
        try {
            const shape = sphereShape(world, body);
            setter(shape);
            floats(name, [getter(shape)]);
        } finally {
            world.destroy();
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
            tangentVelocity: { x: 1.25, y: -2.5, z: 3.75 },
            userMaterialId: 0x123456789abcdef0n,
            customColor: 0x9a654321,
        };
        shape.setSurfaceMaterial(next);
        material("surface", shape.getSurfaceMaterial());
    } finally {
        world.destroy();
    }
});

test("Shape.setFilter/getFilter matches Box3D", () => {
    const { world, body } = scene();
    try {
        const shape = sphereShape(world, body);
        shape.setFilter({
            categoryBits: 0x123456789abcdef0n,
            maskBits: 0xfedcba9876543210n,
            groupIndex: -7,
        });
        const actual = shape.getFilter();
        const row = values("filter");
        expect([actual.categoryBits, actual.maskBits, actual.groupIndex]).toEqual([
            BigInt(row[0]!),
            BigInt(row[1]!),
            Number(row[2]),
        ]);
    } finally {
        world.destroy();
    }
});

for (const [name, enable, getter] of [
    ["sensorEvents", (shape: Shape) => shape.enableSensorEvents(true), (shape: Shape) => shape.areSensorEventsEnabled()],
    ["contactEvents", (shape: Shape) => shape.enableContactEvents(true), (shape: Shape) => shape.areContactEventsEnabled()],
    ["hitEvents", (shape: Shape) => shape.enableHitEvents(true), (shape: Shape) => shape.areHitEventsEnabled()],
    ["preSolveEvents", (shape: Shape) => shape.enablePreSolveEvents(true), (shape: Shape) => shape.arePreSolveEventsEnabled()],
] as const) {
    test(`Shape.${name} setter/getter matches Box3D`, () => {
        const { world, body } = scene();
        try {
            const shape = sphereShape(world, body);
            enable(shape);
            expect(Number(getter(shape)), name).toBe(Number(values(name)[0]));
        } finally {
            world.destroy();
        }
    });
}

test("Shape.setSphere/getSphere matches Box3D", () => {
    const { world, body } = scene();
    try {
        const shape = sphereShape(world, body);
        shape.setSphere({ center: { x: 0.25, y: -0.5, z: 0.75 }, radius: 0.8 });
        const got = shape.getSphere();
        floats("sphere", [got.center.x, got.center.y, got.center.z, got.radius]);
    } finally {
        world.destroy();
    }
});

test("Shape.setCapsule/getCapsule matches Box3D", () => {
    const { world, body } = scene();
    try {
        const shape = body.createCapsule({}, {
            center1: { x: 0, y: -0.5, z: 0 },
            center2: { x: 0, y: 0.5, z: 0 },
            radius: 0.25,
        });
        shape.setCapsule({
            center1: { x: -0.75, y: 0.25, z: 1.5 },
            center2: { x: 0.5, y: 1.25, z: -0.25 },
            radius: 0.4,
        });
        const got = shape.getCapsule();
        floats("capsule", [
            got.center1.x, got.center1.y, got.center1.z,
            got.center2.x, got.center2.y, got.center2.z,
            got.radius,
        ]);
    } finally {
        world.destroy();
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
        const shape = body.createHull({}, initial);
        shape.setHull(next);
        const got = readShapeHull(world.state, shape.id.index1 - 1);
        const expectedRow = values("hull").map(Number);
        const actual = [
            got.vertexCount,
            got.center.x, got.center.y, got.center.z,
            ...got.points.flatMap(({ x, y, z }) => [x, y, z]),
        ];
        expect(actual.map(Math.fround), "hull").toEqual(expectedRow.map(Math.fround));
    } finally {
        world.destroy();
    }
});

test("Shape.setMesh/getGeometryScale/readShapeMesh matches Box3D", () => {
    const { world, body } = scene(BodyType.Static);
    try {
        const shape = body.createMesh({}, meshData(0));
        shape.setMesh(meshData(2), { x: 2, y: 0.5, z: -1.5 });
        const got = readShapeMesh(world.state, shape.id.index1 - 1);
        const expectedRow = values("mesh").map(Number);
        const actual = [
            ...Object.values(shape.getGeometryScale()),
            got.data.vertices.length,
            got.data.triangles.length,
            ...got.data.vertices.flatMap(({ x, y, z }) => [x, y, z]),
        ];
        expect(actual.map(Math.fround), "mesh").toEqual(expectedRow.map(Math.fround));
    } finally {
        world.destroy();
    }
});

test("Shape.setMeshMaterial/getMaterials matches Box3D", () => {
    const { world, body } = scene(BodyType.Static);
    try {
        const shape = body.createMesh({}, meshData(0));
        const next: SurfaceMaterial = {
            friction: 0.78,
            restitution: 0.19,
            rollingResistance: 0.28,
            tangentVelocity: { x: -1, y: 2, z: -3 },
            userMaterialId: 0x0fedcba987654321n,
            customColor: 0x87654321,
        };
        shape.setMeshMaterial(next, 0);
        material("meshMaterial", shape.getMaterials()[0]!);
    } finally {
        world.destroy();
    }
});
