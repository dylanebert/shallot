import { expect, test } from "bun:test";
import {
    Body,
    BodyType,
    Hulls,
    PhysicsPlugin,
    Shape,
    ShapeKind,
    UNIT_CUBE_ID,
} from "../../core/physics";
import { World } from "../../engine";
import { marshalShape, shapeHullGeometry } from "./marshal-shape";
import { createTestBody, createTestWorld, destroyTestWorld, stepTestWorld } from "./testing";
import { getMassData, makeBodyId } from "./world/body";
import { hashWorldState } from "./world/hash";

test("two worlds marshal their own hull registered under the same name", () => {
    const worlds = [new World(), new World()];
    const solvers = worlds.map((world) => createTestWorld({}, world));
    try {
        const bodies = worlds.map((world, i) => {
            for (const component of PhysicsPlugin.components!)
                world.registry.register(component, PhysicsPlugin.name);
            const hulls = world.resource(Hulls);
            expect(hulls.id("__unit_cube__")).toBe(UNIT_CUBE_ID);
            const cube = hulls.get("__unit_cube__")!;
            const scale = i + 1;
            const id = hulls.register({
                ...cube,
                name: "shared-name",
                points: cube.points.map((coordinate) => coordinate * scale),
            });
            expect(id).toBe(1);
            const eid = world.create();
            world.add(eid, Body, { type: BodyType.Dynamic });
            world.add(eid, Shape, {
                kind: ShapeKind.Hull,
                geometry: id,
                scale: [1, 1, 1, 0],
                density: 1 / scale ** 3,
            });
            const state = solvers[i]!;
            const body = createTestBody(state, { type: BodyType.Dynamic });
            marshalShape(world, state, makeBodyId(state, body), eid);
            return { state, body };
        });
        expect(getMassData(bodies[0]!.state, bodies[0]!.body).inertia.cx.x).toBeCloseTo(1 / 6, 5);
        expect(getMassData(bodies[1]!.state, bodies[1]!.body).inertia.cx.x).toBeCloseTo(2 / 3, 5);
    } finally {
        for (const solver of solvers) destroyTestWorld(solver);
        for (const world of worlds) world.dispose();
    }
});

// The unit cube spans ±0.5, so a box's Shape.scale is its full lengths: scaled by 2h it builds the
// same hull, and steps to the same world hash, as a registered ±1 cube scaled by h, the box a ±1 unit
// cube gave at half-extents h. fround(2h) is 2 * fround(h), so the two meet bit for bit.
test("the unit cube at scale 2h builds the hull and world hash of a registered ±1 cube at scale h", () => {
    const half = [0.3, 0.7, 1.1] as const;
    const groundHalf = [10, 0.5, 10] as const;
    const run = (unitCube: boolean) => {
        const world = new World();
        const state = createTestWorld({}, world);
        try {
            for (const component of PhysicsPlugin.components!)
                world.registry.register(component, PhysicsPlugin.name);
            const geometry = unitCube
                ? UNIT_CUBE_ID
                : world.resource(Hulls).register({
                      name: "cube-2",
                      points: world.resource(Hulls).get("__unit_cube__")!.points.map(Math.sign),
                  });
            const shape = (h: readonly number[], body: number) => {
                const eid = world.create();
                const k = unitCube ? 2 : 1;
                world.add(eid, Shape, {
                    kind: ShapeKind.Hull,
                    geometry,
                    scale: [k * h[0]!, k * h[1]!, k * h[2]!],
                });
                marshalShape(world, state, makeBodyId(state, body), eid);
                return eid;
            };
            shape(groundHalf, createTestBody(state));
            const box = shape(
                half,
                createTestBody(state, {
                    type: BodyType.Dynamic,
                    position: { x: 0, y: 2, z: 0 },
                    rotation: { v: { x: 0.2, y: 0.1, z: 0 }, s: Math.sqrt(0.95) },
                }),
            );
            const hull = shapeHullGeometry(world, box);
            const hashes: bigint[] = [];
            for (let i = 0; i < 60; i++) {
                stepTestWorld(state, 1 / 60);
                hashes.push(hashWorldState(state));
            }
            return { hull, hashes };
        } finally {
            destroyTestWorld(state);
            world.dispose();
        }
    };
    const [unit, registered] = [run(true), run(false)];
    expect(unit.hull.aabb.upperBound).toEqual({
        x: Math.fround(half[0]),
        y: Math.fround(half[1]),
        z: Math.fround(half[2]),
    });
    expect(unit.hull).toEqual(registered.hull);
    expect(unit.hashes).toEqual(registered.hashes);
});

test("a hull's maxVertexCount bounds the vertices its built hull keeps, and its absence keeps every extreme point", () => {
    const world = new World();
    try {
        for (const component of PhysicsPlugin.components!)
            world.registry.register(component, PhysicsPlugin.name);
        const points = new Float32Array(48);
        for (let i = 0; i < 16; i++) {
            const angle = (2 * Math.PI * (i % 8)) / 8;
            points.set([Math.cos(angle), i < 8 ? -1 : 1, Math.sin(angle)], 3 * i);
        }
        const hulls = world.resource(Hulls);
        const vertexCount = (geometry: number) => {
            const eid = world.create();
            world.add(eid, Shape, { kind: ShapeKind.Hull, geometry, scale: [1, 1, 1] });
            return shapeHullGeometry(world, eid).vertexCount;
        };
        expect(vertexCount(hulls.register({ name: "prism", points }))).toBe(16);
        expect(vertexCount(hulls.register({ name: "prism-6", points, maxVertexCount: 6 }))).toBe(6);
    } finally {
        world.dispose();
    }
});
