import { sphere } from "../../core/mesh";
import { Hulls, type Shape, ShapeKind } from "../../core/physics";
import type { ComponentRecord, World } from "../../engine";
import type { Vec3 } from "./common/records";
import { buildPlacedHull, heightFieldBuild, meshBuild } from "./geometry-builds";
import { getHeightFieldTriangle, HEIGHT_FIELD_HOLE } from "./shapes/heightfield";

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

function placedHull(world: World, shape: Parameters<typeof shapeMesh>[1]) {
    const definition = world.resource(Hulls).get(shape.geometry);
    if (!definition) throw new Error(`shapeMesh: missing hull ${shape.geometry}`);
    return buildPlacedHull(
        world,
        definition,
        shape.scale,
        shape.hullPosition,
        shape.hullRotation,
        undefined,
        !!shape.enableHullTransform,
    );
}

/**
 * Fresh CPU triangle geometry in body-local meters, suitable for Meshes.add. Bakes Shape.scale
 * for hulls and meshes, plus hullPosition/hullRotation and enableHullTransform for hulls (identity
 * placement and no forced transform when omitted),
 * and the centers and radius for spheres and capsules; the body's
 * Transform is not baked. Requires an initialized StandardPhysicsPlugin for geometry builds.
 * Height fields use their definition scale (Shape.scale does not apply), decoded after
 * quantization, and omit holes. Throws for missing geometry or an unsupported compound.
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
        Partial<
            Pick<
                ComponentRecord<typeof Shape>,
                "hullPosition" | "hullRotation" | "enableHullTransform"
            >
        >,
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
        const hull = placedHull(world, shape);
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
    } else if (kind === ShapeKind.HeightField) {
        const field = heightFieldBuild(world, geometry);
        if (!field) throw new Error(`shapeMesh: missing height field ${geometry}`);
        for (let i = 0; i < field.flags.length; i++) {
            if (field.materialIndices[i >> 1] === HEIGHT_FIELD_HOLE) continue;
            const t = getHeightFieldTriangle(field, i);
            triangle(...t.vertices);
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

/**
 * Fresh pairs of endpoints in body-local meters, with the same placement and refusal
 * rules as shapeMesh. Hulls have their faces' boundaries, one per half-edge pair as in
 * Box3D's hull drawing; meshes omit coplanar triangle diagonals and retain open
 * boundaries. Spheres have three great circles; capsules have two perpendicular
 * longitudinal rings (semicircular caps joined by straight sides), sampled at 32
 * segments per circle. This is an authoring-time conversion, not a live binding.
 */
export function shapeEdges(world: World, shape: Parameters<typeof shapeMesh>[1]): [Point, Point][] {
    const edges: [Point, Point][] = [];
    if (shape.kind === ShapeKind.Sphere || shape.kind === ShapeKind.Capsule) {
        const a = shape.kind === ShapeKind.Sphere ? shape.sphereCenter : shape.capsuleCenter1;
        const b = shape.kind === ShapeKind.Sphere ? a : shape.capsuleCenter2;
        const axis = b.map((v, i) => v - a[i]!) as Point;
        const length = Math.hypot(...axis);
        const y = length ? (axis.map((v) => v / length) as Point) : ([0, 1, 0] as Point);
        const seed: Point = Math.abs(y[0]) < 0.9 ? [1, 0, 0] : [0, 0, 1];
        const projection = seed.reduce((sum, v, i) => sum + v * y[i]!, 0);
        const x = seed.map((v, i) => v - projection * y[i]!) as Point;
        const magnitude = Math.hypot(...x);
        for (let i = 0; i < 3; i++) x[i] /= magnitude;
        const z: Point = [
            y[1] * x[2] - y[2] * x[1],
            y[2] * x[0] - y[0] * x[2],
            y[0] * x[1] - y[1] * x[0],
        ];
        const ring = (u: Point, v: Point, capsule: boolean) => {
            const points: Point[] = [];
            for (let i = 0; i < 32; i++) {
                const theta = (i * Math.PI) / 16;
                const point = (center: Point): Point =>
                    center.map(
                        (c, j) =>
                            c + shape.radius * (u[j]! * Math.cos(theta) + v[j]! * Math.sin(theta)),
                    ) as Point;
                if (capsule && (i === 0 || i === 16)) {
                    points.push(point(i === 0 ? a : b), point(i === 0 ? b : a));
                } else points.push(point(capsule && i < 16 ? b : a));
            }
            for (let i = 0; i < points.length; i++)
                edges.push([points[i]!, points[(i + 1) % points.length]!]);
        };
        ring(x, y, length > 0);
        ring(z, y, length > 0);
        if (!length) ring(x, z, false);
        return edges;
    }
    if (shape.kind === ShapeKind.Hull) {
        const hull = placedHull(world, shape);
        const point = ({ x, y, z }: Vec3): Point => [x, y, z];
        for (const [i, edge] of hull.edges.entries())
            if (i < edge.twin)
                edges.push([
                    point(hull.points[edge.origin]!),
                    point(hull.points[hull.edges[edge.twin]!.origin]!),
                ]);
        return edges;
    }
    const { vertices, indices } = shapeMesh(world, shape);
    const candidates = new Map<
        string,
        { edge: [Point, Point]; third: Point; normal: Point; crease: boolean; count: number }
    >();
    const faces = new Set<string>();
    for (let i = 0; i < indices.length; i += 3) {
        const points = Array.from(
            indices.subarray(i, i + 3),
            (index) => Array.from(vertices.subarray(index * 8, index * 8 + 3)) as Point,
        );
        const face = points
            .map((p) => p.join(","))
            .sort()
            .join(";");
        if (faces.has(face)) continue;
        faces.add(face);
        const index = indices[i]! * 8;
        const normal = Array.from(vertices.subarray(index + 4, index + 7)) as Point;
        for (let j = 0; j < 3; j++) {
            const a = points[j]!,
                b = points[(j + 1) % 3]!;
            const keys = [a.join(","), b.join(",")].sort();
            const key = keys.join(";");
            const previous = candidates.get(key);
            const third = points[(j + 2) % 3]!;
            if (previous) {
                previous.count++;
                const [origin, end] = previous.edge;
                const axis = end.map((v, k) => v - origin[k]!);
                const u = previous.third.map((v, k) => v - origin[k]!);
                const v = third.map((value, k) => value - origin[k]!);
                const dot = (a: number[], b: number[]) =>
                    a.reduce((sum, value, k) => sum + value * b[k]!, 0);
                // Coplanar faces only hide an edge when they occupy opposite sides of it.
                const sameSide = dot(u, v) * dot(axis, axis) - dot(u, axis) * dot(v, axis) > 0;
                previous.crease ||= sameSide || Math.abs(dot(normal, previous.normal)) < 1 - 1e-6;
            } else candidates.set(key, { edge: [a, b], third, normal, crease: false, count: 1 });
        }
    }
    for (const candidate of candidates.values())
        if (candidate.count === 1 || candidate.crease) edges.push(candidate.edge);
    return edges;
}
