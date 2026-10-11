import { expect, test } from "bun:test";
import { ContactField, contactField, contactIds, contactIsValid } from "../collision/contact";
import { updateBroadPhasePairs } from "../collision/pairs";
import { f32, mat3, vec3, xf } from "../common/math";
import { BodyType, defaultShapeDef, defaultSurfaceMaterial, ShapeType } from "../common/types";
import { readShapeAabb } from "../kernel/shapecolumns";
import { ShapeField, shapeField } from "../kernel/shaperecords";
import {
    createTestBody,
    createTestWorld,
    destroyTestWorld,
    stepTestWorld,
    testShapeDef,
} from "../testing";
import { createCompound } from "./compound";
import gold from "./geometry.gold.json";
import { createGrid } from "./heightfield";
import { makeBoxHull } from "./hull";
import { createGridMesh } from "./mesh";
import {
    computeFatShapeAABBOut,
    computeShapeAABB,
    computeShapeAABBOut,
    computeShapeExtent,
    computeShapeMass,
    createCapsuleShape,
    createCompoundShape,
    createHeightFieldShape,
    createHullShape,
    createMeshShape,
    createSphereShape,
    getShapeCentroid,
    setShapeFilter,
} from "./shape";

const bits = new DataView(new ArrayBuffer(4));
function hex(x: number): string {
    bits.setFloat32(0, x);
    return `0x${bits.getUint32(0).toString(16).padStart(8, "0")}`;
}
const v = (x: number, y: number, z: number) => ({ x, y, z });

test("assigning an equal filter preserves native contacts", () => {
    const world = createTestWorld({ gravity: v(0, 0, 0) });
    try {
        const ground = createTestBody(world);
        const visitor = createTestBody(world, { type: BodyType.Dynamic, position: v(0, 1.5, 0) });
        const filter = defaultShapeDef().filter;
        createSphereShape(world, ground, testShapeDef({ enableContactEvents: true }), {
            center: v(0, 0, 0),
            radius: 1,
        });
        const shape = createSphereShape(
            world,
            visitor,
            testShapeDef({ enableContactEvents: true, filter }),
            { center: v(0, 0, 0), radius: 1 },
        )!;
        updateBroadPhasePairs(world);
        stepTestWorld(world, 1 / 60);
        const contact = contactIds(world)[0]!;
        expect(
            contactIsValid(world, {
                index1: contact + 1,
                world0: world.worldId,
                generation: contactField(world, contact, ContactField.generation),
            }),
        ).toBe(true);
        setShapeFilter(world, shape, filter);
        expect(
            contactIsValid(world, {
                index1: contact + 1,
                world0: world.worldId,
                generation: contactField(world, contact, ContactField.generation),
            }),
        ).toBe(true);
    } finally {
        destroyTestWorld(world);
    }
});

test("kernel shape mass getters retain native sphere/capsule gold bits and return independent results", () => {
    const world = createTestWorld();
    try {
        const body = createTestBody(world, { type: BodyType.Dynamic });
        const shapes = [
            createSphereShape(world, body, testShapeDef({ density: 1 }), {
                center: v(0, 0, 0),
                radius: 1,
            })!,
            createSphereShape(world, body, testShapeDef({ density: 2.5 }), {
                center: v(0.5, -1, 2),
                radius: 0.35,
            })!,
            createCapsuleShape(world, body, testShapeDef({ density: 1 }), {
                center1: v(0, -1, 0),
                center2: v(0, 1, 0),
                radius: 0.5,
            })!,
            createCapsuleShape(world, body, testShapeDef({ density: 3 }), {
                center1: v(-1, 0.5, 0.25),
                center2: v(1.5, -0.5, 0.75),
                radius: 0.3,
            })!,
            createCapsuleShape(world, body, testShapeDef({ density: 1000 }), {
                center1: v(0.06, 0, 0),
                center2: v(-0.06, 0, 0),
                radius: 0.12,
            })!,
        ];
        const vectors = [...gold.spheres, ...gold.capsules];
        for (let i = 0; i < shapes.length; ++i) {
            const shape = shapes[i]!;
            const m = computeShapeMass(world, shape);
            const g = vectors[i]!;
            expect(hex(m.mass)).toBe(g.mass);
            expect([m.center.x, m.center.y, m.center.z].map(hex)).toEqual(g.center);
            expect(
                [
                    m.inertia.cx.x,
                    m.inertia.cx.y,
                    m.inertia.cx.z,
                    m.inertia.cy.x,
                    m.inertia.cy.y,
                    m.inertia.cy.z,
                    m.inertia.cz.x,
                    m.inertia.cz.y,
                    m.inertia.cz.z,
                ].map(hex),
            ).toEqual(g.inertia);
            const out = { mass: 0, center: vec3.zero(), inertia: mat3.zero() };
            expect(computeShapeMass(world, shape, out)).toBe(out);
            expect(out).toEqual(m);
            m.center.x = 99;
            m.inertia.cx.x = 99;
            expect(computeShapeMass(world, shape)).toEqual(out);
        }
    } finally {
        destroyTestWorld(world);
    }
});

test("kernel AABB, fat AABB, centroid, hull mass and extent results are independent of retained TypeScript geometry", () => {
    const world = createTestWorld();
    try {
        const body = createTestBody(world, { type: BodyType.Dynamic, position: v(3, 4, 5) });
        const authoredSphere = { center: v(-2, 1, 0), radius: 0.25 };
        const sphere = createSphereShape(
            world,
            body,
            testShapeDef({ density: 2 }),
            authoredSphere,
        )!;
        const capsule = createCapsuleShape(world, body, testShapeDef(), {
            center1: v(-3, -1, 0),
            center2: v(-1, 2, 0),
            radius: 0.5,
        })!;
        const hull = createHullShape(
            world,
            body,
            testShapeDef({ density: 2 }),
            makeBoxHull(1, 2, 3),
        )!;
        const pose = xf.identity();
        pose.p = v(3, 4, 5);
        const expected = { lowerBound: v(0.75, 4.75, 4.75), upperBound: v(1.25, 5.25, 5.25) };
        const box = computeShapeAABB(world, sphere, pose);
        expect(box).toEqual(expected);
        const published = readShapeAabb(world, sphere, {
            lowerBound: v(0, 0, 0),
            upperBound: v(0, 0, 0),
        });
        expect(published).toEqual({
            lowerBound: v(f32(0.75 - 0.02), f32(4.75 - 0.02), f32(4.75 - 0.02)),
            upperBound: v(f32(1.25 + 0.02), f32(5.25 + 0.02), f32(5.25 + 0.02)),
        });
        published.lowerBound.x = 99;
        expect(
            readShapeAabb(world, sphere, { lowerBound: v(0, 0, 0), upperBound: v(0, 0, 0) })
                .lowerBound.x,
        ).toBe(f32(0.75 - 0.02));
        authoredSphere.center.x = 99;
        expect(computeShapeAABBOut(world, sphere, pose, box)).toBe(box);
        expect(box).toEqual(expected);
        const centroid = getShapeCentroid(world, sphere);
        expect(centroid).toEqual(v(-2, 1, 0));
        centroid.x = 99;
        expect(getShapeCentroid(world, sphere)).toEqual(v(-2, 1, 0));
        expect(computeFatShapeAABBOut(world, sphere, pose, 0.125, box)).toBe(box);
        expect(box).toEqual({
            lowerBound: v(0.625, 4.625, 4.625),
            upperBound: v(1.375, 5.375, 5.375),
        });
        expect(computeShapeExtent(world, sphere, v(1, 2, 0))).toEqual({
            minExtent: 0.25,
            maxExtent: v(3.25, 1.25, 0.25),
        });
        expect(computeShapeExtent(world, capsule, v(0, 0, 0))).toEqual({
            minExtent: 0.5,
            maxExtent: v(3.5, 2.5, 0.5),
        });
        expect(computeShapeMass(world, hull)).toEqual({
            mass: 96,
            center: v(0, 0, 0),
            inertia: { cx: v(416, 0, 0), cy: v(0, 320, 0), cz: v(0, 0, 160) },
        });
        expect(getShapeCentroid(world, hull)).toEqual(v(0, 0, 0));
        expect(computeShapeAABB(world, hull, pose)).toEqual({
            lowerBound: v(2, 2, 2),
            upperBound: v(4, 6, 8),
        });
    } finally {
        destroyTestWorld(world);
    }
});

test("public geometry results for mesh, height field and compound shapes come from retained kernel images", () => {
    const world = createTestWorld();
    try {
        const body = createTestBody(world, { type: BodyType.Static });
        const meshData = createGridMesh(3, 3, 1, 0, true);
        const mesh = createMeshShape(world, body, testShapeDef(), meshData, { x: 1, y: 1, z: 1 })!;
        const meshBox = readShapeAabb(world, mesh, {
            lowerBound: v(0, 0, 0),
            upperBound: v(0, 0, 0),
        });
        meshData.vertices[0]!.x += 100;
        expect(
            readShapeAabb(world, mesh, { lowerBound: v(0, 0, 0), upperBound: v(0, 0, 0) }),
        ).toEqual(meshBox);

        const fieldData = createGrid(3, 3, { x: 1, y: 1, z: 1 }, false);
        const field = createHeightFieldShape(world, body, testShapeDef(), fieldData)!;
        const fieldBox = readShapeAabb(world, field, {
            lowerBound: v(0, 0, 0),
            upperBound: v(0, 0, 0),
        });
        fieldData.compressedHeights.fill(0xffff);
        expect(
            readShapeAabb(world, field, { lowerBound: v(0, 0, 0), upperBound: v(0, 0, 0) }),
        ).toEqual(fieldBox);

        const compoundData = createCompound({
            spheres: [
                {
                    sphere: { center: v(1, 2, 3), radius: 0.75 },
                    material: defaultSurfaceMaterial(),
                },
            ],
        })!;
        const compound = createCompoundShape(world, body, testShapeDef(), compoundData)!;
        const compoundBox = readShapeAabb(world, compound, {
            lowerBound: v(0, 0, 0),
            upperBound: v(0, 0, 0),
        });
        const mass = computeShapeMass(world, compound);
        compoundData.spheres[0]!.sphere.center.x = 100;
        compoundData.spheres[0]!.sphere.radius = 10;
        expect(
            readShapeAabb(world, compound, { lowerBound: v(0, 0, 0), upperBound: v(0, 0, 0) }),
        ).toEqual(compoundBox);
        expect(computeShapeMass(world, compound)).toEqual(mass);
        mass.center.x = 99;
        expect(computeShapeMass(world, compound).center.x).not.toBe(99);
    } finally {
        destroyTestWorld(world);
    }
});

test("kernel creation collapses short capsules and rejects nonstatic compounds, while centroid fills caller storage", () => {
    const world = createTestWorld();
    try {
        const body = createTestBody(world, { type: BodyType.Dynamic });
        const shape = createCapsuleShape(world, body, testShapeDef(), {
            center1: v(0, 0, 0),
            center2: v(0.002, 0, 0),
            radius: 0.3,
        })!;
        expect(shapeField(world, shape, ShapeField.type)).toBe(ShapeType.Sphere);
        expect(getShapeCentroid(world, shape)).toEqual(v(f32(0.001), 0, 0));
        const compound = createCompound({
            spheres: [
                { sphere: { center: v(0, 0, 0), radius: 1 }, material: defaultSurfaceMaterial() },
            ],
        })!;
        expect(createCompoundShape(world, body, testShapeDef(), compound)).toBeNull();
        const ground = createTestBody(world, { type: BodyType.Static });
        const staticShape = createCompoundShape(world, ground, testShapeDef(), compound)!;
        const out = vec3.zero();
        expect(getShapeCentroid(world, staticShape, out)).toBe(out);
        expect(out).toEqual(v(0, 0, 0));
    } finally {
        destroyTestWorld(world);
    }
});
