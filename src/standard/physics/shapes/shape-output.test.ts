import { expect, test } from "bun:test";
import { PhysicsWorld } from "../api/world";
import { f32, mat3, vec3, xf } from "../common/math";
import { BodyType, defaultShapeDef, defaultSurfaceMaterial, ShapeType } from "../common/types";
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
    createCompoundShape,
    getShapeCentroid,
} from "./shape";

const bits = new DataView(new ArrayBuffer(4));
function hex(x: number): string {
    bits.setFloat32(0, x);
    return `0x${bits.getUint32(0).toString(16).padStart(8, "0")}`;
}
const v = (x: number, y: number, z: number) => ({ x, y, z });

test("assigning an equal filter preserves native contacts", () => {
    const world = new PhysicsWorld({ gravity: v(0, 0, 0) });
    try {
        const ground = world.createBody();
        const visitor = world.createBody({ type: BodyType.Dynamic, position: v(0, 1.5, 0) });
        const filter = defaultShapeDef().filter;
        ground.createSphere({ enableContactEvents: true }, { center: v(0, 0, 0), radius: 1 });
        const shape = visitor.createSphere(
            { enableContactEvents: true, filter },
            { center: v(0, 0, 0), radius: 1 },
        );
        world.step(1 / 60);
        const contact = world.getContactEvents().beginEvents[0].contact;
        expect(contact.isValid()).toBe(true);
        shape.setFilter(filter);
        expect(contact.isValid()).toBe(true);
    } finally {
        world.destroy();
    }
});

test("kernel shape mass getters retain native sphere/capsule gold bits and return independent results", () => {
    const world = new PhysicsWorld();
    try {
        const body = world.createBody({ type: BodyType.Dynamic });
        const shapes = [
            body.createSphere({ density: 1 }, { center: v(0, 0, 0), radius: 1 }),
            body.createSphere({ density: 2.5 }, { center: v(0.5, -1, 2), radius: 0.35 }),
            body.createCapsule(
                { density: 1 },
                { center1: v(0, -1, 0), center2: v(0, 1, 0), radius: 0.5 },
            ),
            body.createCapsule(
                { density: 3 },
                { center1: v(-1, 0.5, 0.25), center2: v(1.5, -0.5, 0.75), radius: 0.3 },
            ),
            body.createCapsule(
                { density: 1000 },
                { center1: v(0.06, 0, 0), center2: v(-0.06, 0, 0), radius: 0.12 },
            ),
        ];
        const vectors = [...gold.spheres, ...gold.capsules];
        for (let i = 0; i < shapes.length; ++i) {
            const shape = shapes[i];
            const m = shape.computeMassData();
            const g = vectors[i];
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
            expect(computeShapeMass(world.state, shape.id.index1 - 1, out)).toBe(out);
            expect(out).toEqual(m);
            m.center.x = 99;
            m.inertia.cx.x = 99;
            expect(shape.computeMassData()).toEqual(out);
        }
    } finally {
        world.destroy();
    }
});

test("kernel AABB, fat AABB, centroid, hull mass and extent results are independent of retained TypeScript geometry", () => {
    const world = new PhysicsWorld();
    try {
        const body = world.createBody({ type: BodyType.Dynamic, position: v(3, 4, 5) });
        const authoredSphere = { center: v(-2, 1, 0), radius: 0.25 };
        const sphere = body.createSphere({ density: 2 }, authoredSphere);
        const capsule = body.createCapsule(
            {},
            { center1: v(-3, -1, 0), center2: v(-1, 2, 0), radius: 0.5 },
        );
        const hull = body.createHull({ density: 2 }, makeBoxHull(1, 2, 3));
        const id = sphere.id.index1 - 1;
        const pose = xf.identity();
        pose.p = v(3, 4, 5);
        const expected = { lowerBound: v(0.75, 4.75, 4.75), upperBound: v(1.25, 5.25, 5.25) };
        const box = computeShapeAABB(world.state, id, pose);
        expect(box).toEqual(expected);
        const published = sphere.getAABB();
        expect(published).toEqual({
            lowerBound: v(f32(0.75 - 0.02), f32(4.75 - 0.02), f32(4.75 - 0.02)),
            upperBound: v(f32(1.25 + 0.02), f32(5.25 + 0.02), f32(5.25 + 0.02)),
        });
        published.lowerBound.x = 99;
        expect(sphere.getAABB().lowerBound.x).toBe(f32(0.75 - 0.02));
        authoredSphere.center.x = 99;
        expect(computeShapeAABBOut(world.state, id, pose, box)).toBe(box);
        expect(box).toEqual(expected);
        const centroid = getShapeCentroid(world.state, id);
        expect(centroid).toEqual(v(-2, 1, 0));
        centroid.x = 99;
        expect(getShapeCentroid(world.state, id)).toEqual(v(-2, 1, 0));
        expect(computeFatShapeAABBOut(world.state, id, pose, 0.125, box)).toBe(box);
        expect(box).toEqual({
            lowerBound: v(0.625, 4.625, 4.625),
            upperBound: v(1.375, 5.375, 5.375),
        });
        expect(computeShapeExtent(world.state, id, v(1, 2, 0))).toEqual({
            minExtent: 0.25,
            maxExtent: v(3.25, 1.25, 0.25),
        });
        expect(computeShapeExtent(world.state, capsule.id.index1 - 1, v(0, 0, 0))).toEqual({
            minExtent: 0.5,
            maxExtent: v(3.5, 2.5, 0.5),
        });
        expect(hull.computeMassData()).toEqual({
            mass: 96,
            center: v(0, 0, 0),
            inertia: { cx: v(416, 0, 0), cy: v(0, 320, 0), cz: v(0, 0, 160) },
        });
        expect(getShapeCentroid(world.state, hull.id.index1 - 1)).toEqual(v(0, 0, 0));
        expect(computeShapeAABB(world.state, hull.id.index1 - 1, pose)).toEqual({
            lowerBound: v(2, 2, 2),
            upperBound: v(4, 6, 8),
        });
    } finally {
        world.destroy();
    }
});

test("public geometry results for mesh, height field and compound shapes come from retained kernel images", () => {
    const world = new PhysicsWorld();
    try {
        const body = world.createBody({ type: BodyType.Static });
        const meshData = createGridMesh(3, 3, 1, 0, true);
        const mesh = body.createMesh({}, meshData)!;
        const meshBox = mesh.getAABB();
        meshData.vertices[0].x += 100;
        expect(mesh.getAABB()).toEqual(meshBox);

        const fieldData = createGrid(3, 3, { x: 1, y: 1, z: 1 }, false);
        const field = body.createHeightField({}, fieldData)!;
        const fieldBox = field.getAABB();
        fieldData.compressedHeights.fill(0xffff);
        expect(field.getAABB()).toEqual(fieldBox);

        const compoundData = createCompound({
            spheres: [
                {
                    sphere: { center: v(1, 2, 3), radius: 0.75 },
                    material: defaultSurfaceMaterial(),
                },
            ],
        })!;
        const compound = body.createCompound({}, compoundData)!;
        const compoundBox = compound.getAABB();
        const mass = compound.computeMassData();
        compoundData.spheres[0].sphere.center.x = 100;
        compoundData.spheres[0].sphere.radius = 10;
        expect(compound.getAABB()).toEqual(compoundBox);
        expect(compound.computeMassData()).toEqual(mass);
        mass.center.x = 99;
        expect(compound.computeMassData().center.x).not.toBe(99);
    } finally {
        world.destroy();
    }
});

test("kernel creation collapses short capsules and rejects nonstatic compounds, while centroid fills caller storage", () => {
    const world = new PhysicsWorld();
    try {
        const body = world.createBody({ type: BodyType.Dynamic });
        const shape = body.createCapsule(
            {},
            { center1: v(0, 0, 0), center2: v(0.002, 0, 0), radius: 0.3 },
        );
        expect(shape.getType()).toBe(ShapeType.Sphere);
        expect(getShapeCentroid(world.state, shape.id.index1 - 1)).toEqual(v(f32(0.001), 0, 0));
        const compound = createCompound({
            spheres: [
                { sphere: { center: v(0, 0, 0), radius: 1 }, material: defaultSurfaceMaterial() },
            ],
        })!;
        expect(
            createCompoundShape(world.state, body.id.index1 - 1, defaultShapeDef(), compound),
        ).toBeNull();
        const ground = world.createBody({ type: BodyType.Static });
        const staticShape = ground.createCompound({}, compound)!;
        const out = vec3.zero();
        expect(getShapeCentroid(world.state, staticShape.id.index1 - 1, out)).toBe(out);
        expect(out).toEqual(v(0, 0, 0));
    } finally {
        world.destroy();
    }
});
