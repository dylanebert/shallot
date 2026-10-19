import { sphere } from "../../core/mesh";
import { Hulls, type Shape, ShapeKind } from "../../core/physics";
import type { ComponentRecord, World } from "../../engine";
import type { Vec3 } from "./common/records";
import { buildPlacedHull, meshBuild } from "./geometry-builds";

type Point = [number, number, number];

function roundMesh(radius: number, a: Point, b: Point) {
    const axis = b.map((v, i) => v - a[i]!) as Point;
    const length = Math.hypot(...axis);
    if (!length) {
        const result = sphere(radius);
        for (let i = 0; i < result.vertices.length; i += 8)
            for (let j = 0; j < 3; j++) result.vertices[i + j] += a[j]!;
        return result;
    }
    const y = axis.map((v) => v / length) as Point;
    const cross = (a: Point, b: Point): Point => [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    const x = cross(y, Math.abs(y[0]) < 0.9 ? [1, 0, 0] : [0, 0, 1]);
    const xLength = Math.hypot(...x);
    for (let i = 0; i < 3; i++) x[i] /= xLength;
    const z = cross(x, y);
    const latitudes = new Set(Array.from({ length: 17 }, (_, i) => (i * Math.PI) / 16));
    const longitudes = new Set(Array.from({ length: 32 }, (_, i) => (i * 2 * Math.PI) / 32));
    // Include world-axis supports so an oblique capsule keeps its exact bounds, not just sampled bounds.
    for (let i = 0; i < 3; i++)
        for (const sign of [-1, 1]) {
            latitudes.add(Math.acos(Math.max(-1, Math.min(1, sign * y[i]!))));
            longitudes.add((Math.atan2(sign * z[i]!, sign * x[i]!) + 2 * Math.PI) % (2 * Math.PI));
        }
    const angles = [...longitudes].sort((a, b) => a - b);
    const rows = [...latitudes]
        .sort((a, b) => a - b)
        .flatMap((theta) =>
            theta === Math.PI / 2
                ? [
                      { theta, center: b },
                      { theta, center: a },
                  ]
                : [{ theta, center: theta < Math.PI / 2 ? b : a }],
        );
    const vertices: number[] = [],
        indices: number[] = [];
    for (const { theta, center } of rows)
        for (const phi of angles) {
            const normal = y.map(
                (v, i) =>
                    v * Math.cos(theta) +
                    Math.sin(theta) * (x[i]! * Math.cos(phi) + z[i]! * Math.sin(phi)),
            );
            vertices.push(
                ...normal.map((v, i) => radius * v + center[i]!),
                phi / (2 * Math.PI),
                ...normal,
                theta / Math.PI,
            );
        }
    const stride = angles.length;
    for (let row = 0; row + 1 < rows.length; row++)
        for (let col = 0; col < stride; col++) {
            const next = (col + 1) % stride;
            const a = row * stride + col,
                b = row * stride + next;
            const c = (row + 1) * stride + col,
                d = (row + 1) * stride + next;
            if (row > 0) indices.push(a, b, c);
            if (row + 2 < rows.length) indices.push(b, d, c);
        }
    return { vertices: new Float32Array(vertices), indices: new Uint32Array(indices) };
}

/**
 * Fresh CPU triangle geometry in body-local meters, suitable for Meshes.add. Bakes Shape.scale
 * for hulls and meshes, plus hullPosition/hullRotation for hulls (identity when omitted),
 * and the centers and radius for spheres and capsules; the body's
 * Transform is not baked. Requires an initialized StandardPhysicsPlugin for geometry builds.
 * Throws for missing geometry, or an unsupported compound or height field.
 * This is an authoring-time conversion, not a live binding to subsequent Shape edits.
 */
export function shapeMesh(
    world: World,
    shape: Pick<
        ComponentRecord<typeof Shape>,
        | "kind"
        | "geometry"
        | "scale"
        | "radius"
        | "sphereCenter"
        | "capsuleCenter1"
        | "capsuleCenter2"
    > &
        Partial<Pick<ComponentRecord<typeof Shape>, "hullPosition" | "hullRotation">>,
): {
    vertices: Float32Array;
    indices: Uint32Array;
} {
    const { kind, geometry, scale } = shape;
    if (kind === ShapeKind.Sphere || kind === ShapeKind.Capsule) {
        const radius = shape.radius;
        const a = kind === ShapeKind.Sphere ? shape.sphereCenter : shape.capsuleCenter1;
        const b = kind === ShapeKind.Sphere ? a : shape.capsuleCenter2;
        return roundMesh(radius, a, b);
    }
    const positions: number[] = [];
    const indices: number[] = [];
    const triangle = (a: Vec3, b: Vec3, c: Vec3): void => {
        const ux = b.x - a.x,
            uy = b.y - a.y,
            uz = b.z - a.z;
        const vx = c.x - a.x,
            vy = c.y - a.y,
            vz = c.z - a.z;
        const nx = uy * vz - uz * vy,
            ny = uz * vx - ux * vz,
            nz = ux * vy - uy * vx;
        const length = Math.hypot(nx, ny, nz);
        if (!length) return;
        const start = positions.length / 8;
        for (const p of [a, b, c])
            positions.push(p.x, p.y, p.z, 0, nx / length, ny / length, nz / length, 0);
        indices.push(start, start + 1, start + 2);
    };
    if (kind === ShapeKind.Hull) {
        const definition = world.resource(Hulls).get(geometry);
        if (!definition) throw new Error(`shapeMesh: missing hull ${geometry}`);
        const hull = buildPlacedHull(
            world,
            definition,
            scale,
            shape.hullPosition,
            shape.hullRotation,
        );
        for (const face of hull.faces) {
            const points: Vec3[] = [];
            let edge = face.edge;
            do {
                points.push(hull.points[hull.edges[edge]!.origin]!);
                edge = hull.edges[edge]!.next;
            } while (edge !== face.edge);
            for (let i = 1; i + 1 < points.length; i++)
                triangle(points[0]!, points[i]!, points[i + 1]!);
        }
    } else if (kind === ShapeKind.Mesh) {
        const mesh = meshBuild(world, geometry);
        if (!mesh) throw new Error(`shapeMesh: missing mesh ${geometry}`);
        const points = mesh.vertices.map((p) => ({
            x: p.x * scale[0],
            y: p.y * scale[1],
            z: p.z * scale[2],
        }));
        const reflected = scale[0] * scale[1] * scale[2] < 0;
        for (const t of mesh.triangles)
            triangle(
                points[t.index1]!,
                points[reflected ? t.index3 : t.index2]!,
                points[reflected ? t.index2 : t.index3]!,
            );
    } else {
        throw new Error(`shapeMesh: unsupported Shape kind ${kind}`);
    }
    return { vertices: new Float32Array(positions), indices: new Uint32Array(indices) };
}
