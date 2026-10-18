import { afterAll, beforeAll, expect, test } from "bun:test";
import { Hulls, PhysicsMeshes, ShapeKind } from "../../core/physics";
import { World } from "../../engine";
import { StandardPhysicsPlugin, shapeMesh } from "./index";

type Point = [number, number, number];
type Geometry = ReturnType<typeof shapeMesh>;
type Definition = Parameters<typeof shapeMesh>[1];
let world: World;
beforeAll(async () => {
    world = new World();
    await StandardPhysicsPlugin.initialize!(world);
});
afterAll(async () => {
    await StandardPhysicsPlugin.dispose!(world);
    world.dispose();
});

function definition(values: Partial<Definition>): Definition {
    return {
        kind: ShapeKind.Hull,
        geometry: 0,
        scale: [1, 1, 1],
        radius: 0.5,
        sphereCenter: [0, 0, 0],
        capsuleCenter1: [0, -0.5, 0],
        capsuleCenter2: [0, 0.5, 0],
        ...values,
    };
}
const unit = (p: Point): Point => p.map((v) => v / Math.hypot(...p)) as Point;
const dot = (a: Point, b: Point) => a.reduce((sum, v, i) => sum + v * b[i]!, 0);

interface Surface {
    lower: Point;
    upper: Point;
    distance(p: Point): number;
    normal(p: Point): Point;
    smooth?: boolean;
    area: number;
    areaTolerance?: number;
}

function verify({ vertices, indices }: Geometry, surface: Surface): void {
    expect(indices.length).toBeGreaterThan(0);
    const point = (index: number): Point =>
        Array.from(vertices.subarray(index * 8, index * 8 + 3)) as Point;
    const normal = (index: number): Point =>
        Array.from(vertices.subarray(index * 8 + 4, index * 8 + 7)) as Point;
    const min = [Infinity, Infinity, Infinity],
        max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < vertices.length / 8; i++) {
        const p = point(i);
        expect(surface.distance(p)).toBeLessThan(2e-6);
        for (let axis = 0; axis < 3; axis++) {
            min[axis] = Math.min(min[axis]!, p[axis]!);
            max[axis] = Math.max(max[axis]!, p[axis]!);
        }
    }
    for (let axis = 0; axis < 3; axis++) {
        expect(min[axis]).toBeCloseTo(surface.lower[axis]!, 5);
        expect(max[axis]).toBeCloseTo(surface.upper[axis]!, 5);
    }
    let area = 0;
    for (const index of indices) expect(index).toBeLessThan(vertices.length / 8);
    for (let i = 0; i < indices.length; i += 3) {
        const ids = Array.from(indices.subarray(i, i + 3));
        const [a, b, c] = ids.map(point) as [Point, Point, Point];
        const u = b.map((v, axis) => v - a[axis]!) as Point;
        const v = c.map((v, axis) => v - a[axis]!) as Point;
        const cross: Point = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        const magnitude = Math.hypot(...cross);
        area += magnitude / 2;
        if (magnitude < 1e-10) continue;
        const center = a.map((v, axis) => (v + b[axis]! + c[axis]!) / 3) as Point;
        const expected = surface.normal(center);
        if (dot(unit(cross), expected) < 0.95) throw new Error("outward triangle winding");
        for (const id of ids) {
            const expected = surface.normal(surface.smooth ? point(id) : center);
            if (
                dot(normal(id), expected) < 0.99999 ||
                Math.abs(Math.hypot(...normal(id)) - 1) > 1e-5
            )
                throw new Error("outward vertex normal");
        }
    }
    if (Math.abs(area - surface.area) > (surface.areaTolerance ?? 2e-5))
        throw new Error(`surface area: ${area}, expected ${surface.area}`);
}

const box: Surface = {
    lower: [-2, -1, -9],
    upper: [2, 1, 9],
    area: 232,
    distance: (p) => Math.abs(Math.max(Math.abs(p[0]) / 2, Math.abs(p[1]), Math.abs(p[2]) / 9) - 1),
    normal: (p) => {
        const axis = [Math.abs(p[0]) / 2, Math.abs(p[1]), Math.abs(p[2]) / 9];
        const face = axis.indexOf(Math.max(...axis));
        return p.map((v, i) => (i === face ? Math.sign(v) : 0)) as Point;
    },
};
function boxMesh(): Geometry {
    const geometry = world.resource(Hulls).add({ name: "box", halfExtents: [1, 2, 3] });
    return shapeMesh(world, definition({ geometry, scale: [2, 0.5, 3] }));
}

test("unspawned box hull covers all faces with outward normals, winding and exact area", () => {
    verify(boxMesh(), box);
});

for (const scale of [
    [1, 1, 1],
    [-2, 0.5, 3],
] satisfies Point[]) {
    test(`transformed box at scale ${scale} has analytic face normals, bounds and area`, () => {
        const geometry = world.resource(Hulls).add({
            name: "transformed box",
            halfExtents: [1, 2, 3],
            transform: { position: [1, 2, 3], rotation: [0, 0, Math.SQRT1_2, Math.SQRT1_2] },
        });
        const center = [1, 2, 3].map((v, i) => v * scale[i]!);
        const extent = [2, 1, 3].map((v, i) => v * Math.abs(scale[i]!));
        const [x, y, z] = extent as Point;
        const surface: Surface = {
            lower: center.map((v, i) => v - extent[i]!) as Point,
            upper: center.map((v, i) => v + extent[i]!) as Point,
            area: 8 * (x * y + x * z + y * z),
            areaTolerance: 1e-4,
            distance: (p) =>
                Math.abs(Math.max(...p.map((v, i) => Math.abs(v - center[i]!) / extent[i]!)) - 1),
            normal: (p) => {
                const offset = p.map((v, i) => v - center[i]!);
                const distances = offset.map((v, i) => Math.abs(v) / extent[i]!);
                const face = distances.indexOf(Math.max(...distances));
                return offset.map((v, i) => (i === face ? Math.sign(v) : 0)) as Point;
            },
        };
        verify(shapeMesh(world, definition({ geometry, scale })), surface);
    });
}

for (const scale of [
    [1, 1, 1],
    [-2, 3, 0.5],
] satisfies Point[]) {
    test(`four-sided cylinder at scale ${scale} has analytic prism normals, caps and area`, () => {
        const geometry = world.resource(Hulls).add({
            name: "cylinder",
            cylinder: { height: 0.25, radius: 0.4, yOffset: 0.3, sides: 4 },
        });
        const x = 0.4 * Math.abs(scale[0]),
            z = 0.4 * Math.abs(scale[2]),
            lo = 0.3 * scale[1],
            hi = 0.55 * scale[1];
        verify(shapeMesh(world, definition({ geometry, scale })), {
            lower: [-x, lo, -z],
            upper: [x, hi, z],
            area: 4 * x * z + 4 * Math.hypot(x, z) * (hi - lo),
            distance: (p) =>
                Math.min(
                    Math.abs(p[1] - lo),
                    Math.abs(p[1] - hi),
                    Math.abs(Math.abs(p[0]) / x + Math.abs(p[2]) / z - 1),
                ),
            normal: (p) =>
                Math.abs(p[1] - lo) < 1e-6
                    ? [0, -1, 0]
                    : Math.abs(p[1] - hi) < 1e-6
                      ? [0, 1, 0]
                      : unit([Math.sign(p[0]) / x, 0, Math.sign(p[2]) / z]),
        });
    });
}

test("reversed winding is rejected even when its normals are recomputed to match", () => {
    const mesh = boxMesh();
    for (let i = 0; i < mesh.indices.length; i += 3) {
        const second = mesh.indices[i + 1]!;
        mesh.indices[i + 1] = mesh.indices[i + 2]!;
        mesh.indices[i + 2] = second;
    }
    expect(() => verify(mesh, box)).toThrow("outward triangle winding");
    for (let i = 0; i < mesh.vertices.length; i += 8)
        for (let axis = 4; axis < 7; axis++) mesh.vertices[i + axis] *= -1;
    expect(() => verify(mesh, box)).toThrow("outward triangle winding");
});

test("missing box face is rejected despite unchanged surface vertices and bounds", () => {
    const mesh = boxMesh();
    mesh.indices = mesh.indices.slice(6);
    expect(() => verify(mesh, box)).toThrow("surface area");
});

test("point hull removes interior points and covers the tetrahedron's analytic area", () => {
    const geometry = world.resource(Hulls).add({
        name: "tetrahedron",
        points: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0.1, 0.1, 0.1]),
    });
    verify(shapeMesh(world, definition({ geometry, scale: [2, 3, 4] })), {
        lower: [0, 0, 0],
        upper: [2, 3, 4],
        area: 13 + Math.sqrt(61),
        distance: (p) => Math.min(...p.map(Math.abs), Math.abs(p[0] / 2 + p[1] / 3 + p[2] / 4 - 1)),
        normal: (p) =>
            p[0] < 1e-6
                ? [-1, 0, 0]
                : p[1] < 1e-6
                  ? [0, -1, 0]
                  : p[2] < 1e-6
                    ? [0, 0, -1]
                    : unit([0.5, 1 / 3, 0.25]),
    });
});

function roundSurface(a: Point, b: Point, radius: number): Surface {
    const d = b.map((v, i) => v - a[i]!) as Point;
    const length = Math.hypot(...d);
    const radial = (p: Point): Point => {
        const offset = p.map((v, i) => v - a[i]!) as Point;
        const t = length ? Math.max(0, Math.min(1, dot(d, offset) / (length * length))) : 0;
        return offset.map((v, i) => v - t * d[i]!) as Point;
    };
    const area = 4 * Math.PI * radius * radius + 2 * Math.PI * radius * length;
    return {
        lower: a.map((v, i) => Math.min(v, b[i]!) - radius) as Point,
        upper: a.map((v, i) => Math.max(v, b[i]!) + radius) as Point,
        distance: (p) => Math.abs(Math.hypot(...radial(p)) - radius),
        normal: (p) => unit(radial(p)),
        smooth: true,
        area,
        // Inscribed triangles approximate curved area; 32 azimuths and 16 latitudes stay within 2%.
        areaTolerance: area * 0.02,
    };
}

test("offset sphere has radial normals, outward winding and the analytic surface area", () => {
    const center: Point = [1, -2, 3];
    verify(
        shapeMesh(world, definition({ kind: ShapeKind.Sphere, sphereCenter: center, radius: 0.7 })),
        roundSurface(center, center, 0.7),
    );
});

for (const b of [
    [1, 5, 1],
    [5, 1, 1],
    [1, 2, 3],
    [-2, 3, 1],
    [1, 1, 1],
] satisfies Point[]) {
    test(`capsule ${b} covers both caps and cylinder with outward normals and winding`, () => {
        const a: Point = [1, 1, 1];
        verify(
            shapeMesh(
                world,
                definition({ kind: ShapeKind.Capsule, capsuleCenter1: a, capsuleCenter2: b }),
            ),
            roundSurface(a, b, 0.5),
        );
    });
}

test("clockwise reflected strided mesh preserves the independently expected -Z normal and area", () => {
    const geometry = world.resource(PhysicsMeshes).add({
        name: "triangle",
        vertices: new Float32Array([0, 0, 0, 99, 1, 0, 0, 99, 0, 1, 0]),
        stride: 4,
        indices: new Uint32Array([0, 1, 2, 0, 0, 0]),
        clockwiseWinding: true,
    });
    const mesh = shapeMesh(
        world,
        definition({ kind: ShapeKind.Mesh, geometry, scale: [-2, 3, 1] }),
    );
    verify(mesh, {
        lower: [-2, 0, 0],
        upper: [0, 3, 0],
        distance: (p) => Math.abs(p[2]),
        normal: () => [0, 0, -1],
        area: 3,
    });
    expect(mesh.indices.length).toBe(3);
});

test("unsupported kinds and stale geometry are refused without creating an entity", () => {
    expect(() => shapeMesh(world, definition({ kind: ShapeKind.Compound }))).toThrow("unsupported");
    const hulls = world.resource(Hulls);
    const geometry = hulls.add({ name: "removed", halfExtents: [1, 1, 1] });
    hulls.remove(geometry);
    hulls.add({ name: "replacement", halfExtents: [2, 2, 2] });
    expect(() => shapeMesh(world, definition({ geometry }))).toThrow("missing hull");
});
