import { afterAll, beforeAll, expect, test } from "bun:test";
import { Hulls, PhysicsMeshes, PhysicsPlugin, Shape, ShapeKind } from "../../core/physics";
import { World } from "../../engine";
import { StandardPhysicsPlugin, shapeMesh } from "./index";

let world: World;
beforeAll(async () => {
    world = new World();
    for (const component of PhysicsPlugin.components!)
        world.registry.register(component, PhysicsPlugin.name);
    await StandardPhysicsPlugin.initialize!(world);
});
afterAll(async () => {
    await StandardPhysicsPlugin.dispose!(world);
    world.dispose();
});

function verify(
    eid: number,
    lower: number[],
    upper: number[],
    surface: (p: number[]) => number,
): void {
    const { vertices, indices } = shapeMesh(world, eid);
    expect(indices.length).toBeGreaterThan(0);
    const min = [Infinity, Infinity, Infinity],
        max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < vertices.length; i += 8) {
        const p = Array.from(vertices.subarray(i, i + 3));
        expect(surface(p)).toBeLessThan(2e-6);
        expect(Math.hypot(vertices[i + 4]!, vertices[i + 5]!, vertices[i + 6]!)).toBeCloseTo(1, 5);
        for (let axis = 0; axis < 3; axis++) {
            min[axis] = Math.min(min[axis]!, p[axis]!);
            max[axis] = Math.max(max[axis]!, p[axis]!);
        }
    }
    for (let axis = 0; axis < 3; axis++) {
        expect(min[axis]).toBeCloseTo(lower[axis]!, 5);
        expect(max[axis]).toBeCloseTo(upper[axis]!, 5);
    }
    for (const index of indices) expect(index).toBeLessThan(vertices.length / 8);
    for (let i = 0; i < indices.length; i += 3) {
        const a = indices[i]! * 8,
            b = indices[i + 1]! * 8,
            c = indices[i + 2]! * 8;
        const u = [0, 1, 2].map((axis) => vertices[b + axis]! - vertices[a + axis]!);
        const v = [0, 1, 2].map((axis) => vertices[c + axis]! - vertices[a + axis]!);
        const cross = [
            u[1]! * v[2]! - u[2]! * v[1]!,
            u[2]! * v[0]! - u[0]! * v[2]!,
            u[0]! * v[1]! - u[1]! * v[0]!,
        ];
        expect(
            cross.reduce((sum, value, axis) => sum + value * vertices[a + 4 + axis]!, 0),
        ).toBeGreaterThanOrEqual(-1e-7);
    }
}

test("box hull vertices and bounds match scaled collision geometry", () => {
    const geometry = world.resource(Hulls).add({ name: "box", halfExtents: [1, 2, 3] });
    const eid = world.create();
    world.add(eid, Shape, { geometry, scale: [2, 0.5, 3] });
    verify(eid, [-2, -1, -9], [2, 1, 9], (p) =>
        Math.abs(Math.max(Math.abs(p[0]!) / 2, Math.abs(p[1]!), Math.abs(p[2]!) / 9) - 1),
    );
});

test("point hull removes interior points and triangulates its surface", () => {
    const geometry = world.resource(Hulls).add({
        name: "tetrahedron",
        points: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0.1, 0.1, 0.1]),
    });
    const eid = world.create();
    world.add(eid, Shape, { geometry, scale: [2, 3, 4] });
    verify(eid, [0, 0, 0], [2, 3, 4], (p) =>
        Math.min(
            Math.abs(p[0]!),
            Math.abs(p[1]!),
            Math.abs(p[2]!),
            Math.abs(p[0]! / 2 + p[1]! / 3 + p[2]! / 4 - 1),
        ),
    );
});

test("offset sphere vertices lie on its radius and reach all six bounds", () => {
    const eid = world.create();
    world.add(eid, Shape, { kind: ShapeKind.Sphere, sphereCenter: [1, -2, 3], radius: 0.7 });
    verify(eid, [0.3, -2.7, 2.3], [1.7, -1.3, 3.7], (p) =>
        Math.abs(Math.hypot(p[0]! - 1, p[1]! + 2, p[2]! - 3) - 0.7),
    );
});

for (const b of [
    [1, 5, 1],
    [5, 1, 1],
    [1, 2, 3],
    [-2, 3, 1],
    [1, 1, 1],
]) {
    test(`capsule ${b} vertices lie on the segment's radius and reach all bounds`, () => {
        const a = [1, 1, 1];
        const eid = world.create();
        world.add(eid, Shape, {
            kind: ShapeKind.Capsule,
            capsuleCenter1: a as [number, number, number],
            capsuleCenter2: b as [number, number, number],
            radius: 0.5,
        });
        const d = b.map((v, i) => v - a[i]!);
        const length2 = d.reduce((sum, v) => sum + v * v, 0);
        verify(
            eid,
            a.map((v, i) => Math.min(v, b[i]!) - 0.5),
            a.map((v, i) => Math.max(v, b[i]!) + 0.5),
            (p) => {
                const t = length2
                    ? Math.max(
                          0,
                          Math.min(
                              1,
                              d.reduce((sum, v, i) => sum + v * (p[i]! - a[i]!), 0) / length2,
                          ),
                      )
                    : 0;
                return Math.abs(Math.hypot(...p.map((v, i) => v - a[i]! - t * d[i]!)) - 0.5);
            },
        );
    });
}

test("strided reflected triangle mesh uses built triangles and scaled bounds", () => {
    const geometry = world.resource(PhysicsMeshes).add({
        name: "triangle",
        vertices: new Float32Array([0, 0, 0, 99, 1, 0, 0, 99, 0, 1, 0]),
        stride: 4,
        indices: new Uint32Array([0, 1, 2, 0, 0, 0]),
        clockwiseWinding: true,
    });
    const eid = world.create();
    world.add(eid, Shape, { kind: ShapeKind.Mesh, geometry, scale: [-2, 3, 1] });
    verify(eid, [-2, 0, 0], [0, 3, 0], (p) => Math.abs(p[2]!));
    expect(shapeMesh(world, eid).indices.length).toBe(3);
});

test("unsupported kinds and stale geometry are refused", () => {
    const eid = world.create();
    expect(() => shapeMesh(world, eid)).toThrow("no Shape");
    world.add(eid, Shape, { kind: ShapeKind.Compound });
    expect(() => shapeMesh(world, eid)).toThrow("unsupported");
    const hulls = world.resource(Hulls);
    const geometry = hulls.add({ name: "removed", halfExtents: [1, 1, 1] });
    hulls.remove(geometry);
    hulls.add({ name: "replacement", halfExtents: [2, 2, 2] });
    world.storage(Shape).kind.set(eid, ShapeKind.Hull);
    world.storage(Shape).geometry.set(eid, geometry);
    expect(() => shapeMesh(world, eid)).toThrow("missing hull");
});
