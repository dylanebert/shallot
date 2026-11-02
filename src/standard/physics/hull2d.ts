import type { World } from "../../engine";
import { kernel } from "./kernel/kernel";

/** Box3D's b3Point2D: planar coordinates, contact separation metadata and original input identity. */
export interface Point2D {
    p: { x: number; y: number };
    separation: number;
    originalIndex: number;
}
function compute(world: World, points: readonly Point2D[], target?: number): Point2D[] {
    if (target !== undefined && (!Number.isInteger(target) || target < 3 || target > 0x7fffffff))
        throw new Error("simplifyHull2D target must be an integer from 3 to 2147483647");
    for (let i = 0; i < points.length; i++) {
        const point = points[i]!;
        if (
            ![point.p.x, point.p.y, point.separation].every((value) =>
                Number.isFinite(Math.fround(value)),
            ) ||
            !Number.isInteger(point.originalIndex) ||
            point.originalIndex < -0x80000000 ||
            point.originalIndex > 0x7fffffff
        )
            throw new Error(
                `Hull2D point ${i} requires finite f32 coordinates and separation, and an int32 originalIndex`,
            );
    }
    const k = kernel(world);
    const input = k.hull2DInputPtr(points.length);
    const data = new DataView(k.memory.buffer);
    for (let i = 0; i < points.length; i++) {
        const p = points[i]!;
        const at = input + 16 * i;
        data.setFloat32(at, p.p.x, true);
        data.setFloat32(at + 4, p.p.y, true);
        data.setFloat32(at + 8, p.separation, true);
        data.setInt32(at + 12, p.originalIndex, true);
    }
    const result = k.hull2D(Number(target !== undefined), points.length, target ?? 0);
    const header = new Uint32Array(k.memory.buffer, result, 2);
    const output = new DataView(k.memory.buffer);
    const pointsOut: Point2D[] = [];
    for (let i = 0; i < header[1]!; i++) {
        const at = header[0]! + i * 16;
        pointsOut.push({
            p: { x: output.getFloat32(at, true), y: output.getFloat32(at + 4, true) },
            separation: output.getFloat32(at + 8, true),
            originalIndex: output.getInt32(at + 12, true),
        });
    }
    return pointsOut;
}
/** Counterclockwise Box3D planar hull, with near-duplicate welding and collinear removal.
 * Coordinates and separation must fit finite f32, and originalIndex must fit int32.
 * Input is unchanged; returned points belong to the caller and retain separation/index metadata. */
export function hull2D(world: World, points: readonly Point2D[]): Point2D[] {
    return compute(world, points);
}
/** Simplify a counterclockwise convex hull to at most target vertices (an int32 >= 3).
 * Near-collinear vertices are removed even below target; hulls with at most three points are retained.
 * Input has hull2D's field constraints and is unchanged; returned records belong to the caller. */
export function simplifyHull2D(
    world: World,
    points: readonly Point2D[],
    target: number,
): Point2D[] {
    return compute(world, points, target);
}
