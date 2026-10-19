import type { World } from "../../../engine";
import type { Transform, Vec3 } from "../common/records";
import { type Kernel, kernel } from "../kernel/kernel";
import type { HullData } from "./hull";
import { hullImage } from "./hullbytes";
import { drawHull } from "./readgeometry";

type Scratch = { input: Float32Array; view: DataView };
const scratch = new WeakMap<Kernel, Scratch>();
function prepare(world: World | undefined, count = 0): [Kernel, Float32Array] {
    const k = kernel(world);
    const ptr = k.hullBuildInputPtr(count);
    let views = scratch.get(k);
    if (!views) {
        views = { input: new Float32Array(0), view: new DataView(k.memory.buffer) };
        scratch.set(k, views);
    }
    const length = 10 + 3 * count;
    if (
        views.input.buffer !== k.memory.buffer ||
        views.input.byteOffset !== ptr ||
        views.input.length < length
    )
        views.input = new Float32Array(k.memory.buffer, ptr, length);
    return [k, views.input];
}
function result(k: Kernel, operation: number, count = 0, budget = 0): HullData | null {
    const ptr = k.hullBuild(operation, count, budget);
    if (ptr === 0) return null;
    const views = scratch.get(k)!;
    if (views.view.buffer !== k.memory.buffer) views.view = new DataView(k.memory.buffer);
    return drawHull(views.view, ptr);
}

/** Clones a hull's topology and rebuilds its planes and bulk properties after scale, then placement. */
export function cloneAndTransformHull(
    hull: HullData,
    transform: Transform,
    scale: Vec3,
    world?: World,
): HullData | null {
    const image = hullImage(hull);
    const [k, input] = prepare(world, Math.ceil((image.length / 4 + 6) / 3));
    input.set([
        transform.p.x,
        transform.p.y,
        transform.p.z,
        transform.q.v.x,
        transform.q.v.y,
        transform.q.v.z,
        transform.q.s,
        scale.x,
        scale.y,
        scale.z,
    ]);
    new Uint8Array(k.memory.buffer, input.byteOffset + 64, image.length).set(image);
    return result(k, 8, image.length);
}

/** Builds a hull using Box3D's [4, 128] vertex budget; degenerate or oversized hulls return null. */
export function createHull(points: Vec3[], maxVertexCount: number, world?: World): HullData | null {
    const [k, input] = prepare(world, points.length);
    for (let i = 0; i < points.length; ++i) {
        input[10 + 3 * i] = points[i].x;
        input[11 + 3 * i] = points[i].y;
        input[12 + 3 * i] = points[i].z;
    }
    return result(k, 0, points.length, maxVertexCount);
}

export function createCylinder(
    height: number,
    radius: number,
    yOffset: number,
    sides: number,
    world?: World,
): HullData {
    const [k, input] = prepare(world);
    input[0] = height;
    input[1] = radius;
    input[2] = yOffset;
    input[3] = sides;
    return result(k, 1) as HullData;
}
export function createCone(
    height: number,
    radius1: number,
    radius2: number,
    slices: number,
    world?: World,
): HullData {
    const [k, input] = prepare(world);
    input[0] = height;
    input[1] = radius1;
    input[2] = radius2;
    input[3] = slices;
    return result(k, 2) as HullData;
}
export function createRock(radius: number, world?: World): HullData {
    const [k, input] = prepare(world);
    input[0] = radius;
    return result(k, 3) as HullData;
}
export function makeBoxHull(hx: number, hy: number, hz: number, world?: World): HullData {
    const [k, input] = prepare(world);
    input[0] = hx;
    input[1] = hy;
    input[2] = hz;
    return result(k, 4) as HullData;
}
export function makeCubeHull(halfWidth: number, world?: World): HullData {
    const [k, input] = prepare(world);
    input[0] = halfWidth;
    return result(k, 5) as HullData;
}
export function makeOffsetBoxHull(
    hx: number,
    hy: number,
    hz: number,
    offset: Vec3,
    world?: World,
): HullData {
    const [k, input] = prepare(world);
    input[0] = hx;
    input[1] = hy;
    input[2] = hz;
    input[3] = offset.x;
    input[4] = offset.y;
    input[5] = offset.z;
    return result(k, 6) as HullData;
}
export function makeTransformedBoxHull(
    hx: number,
    hy: number,
    hz: number,
    transform: Transform,
    world?: World,
): HullData {
    const [k, input] = prepare(world);
    input[0] = hx;
    input[1] = hy;
    input[2] = hz;
    input[3] = transform.p.x;
    input[4] = transform.p.y;
    input[5] = transform.p.z;
    input[6] = transform.q.v.x;
    input[7] = transform.q.v.y;
    input[8] = transform.q.v.z;
    input[9] = transform.q.s;
    return result(k, 7) as HullData;
}
