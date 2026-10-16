import type { StorageFlag, TgpuBuffer } from "typegpu";
import * as d from "typegpu/data";
import type { World } from "../../engine";
import { packColor } from "../../engine/utils";
import { SEGMENT_BYTES, SEGMENT_FLOATS, Segment } from "./pipeline";

// initial segment capacity; the CPU staging + GPU buffer double on demand for large wireframe workloads
const INITIAL = 1 << 14;

interface SegmentState {
    buffer: (TgpuBuffer<d.WgslArray<typeof Segment>> & StorageFlag) | null;
    staging: ArrayBuffer;
    f32: Float32Array;
    u32: Uint32Array;
    stableIds: Uint32Array;
    stableParts: Uint32Array;
    capacity: number;
    count: number;
    publishedCount: number;
}

const segmentStateKey = { create: createSegmentState };

function createSegmentState(): SegmentState {
    const staging = new ArrayBuffer(INITIAL * SEGMENT_BYTES);
    return {
        buffer: null,
        staging,
        f32: new Float32Array(staging),
        u32: new Uint32Array(staging),
        stableIds: new Uint32Array(INITIAL),
        stableParts: new Uint32Array(INITIAL),
        capacity: INITIAL,
        count: 0,
        publishedCount: 0,
    };
}

function segmentState(world: World): SegmentState {
    return world.resource(segmentStateKey);
}

export function initializeSegmentState(world: World): void {
    world.resource(segmentStateKey);
}

// the producer's GPU publication: the segment storage buffer the transparent phase binds.
interface Lines {
    buffer: (TgpuBuffer<d.WgslArray<typeof Segment>> & StorageFlag) | null;
}

export const Lines: import("../../engine").Resource<Lines> = {
    create: (world) => world.resource(segmentStateKey),
};

function grow(world: World, min: number): void {
    let cap = segmentState(world).capacity;
    while (cap < min) cap *= 2;
    const next = new ArrayBuffer(cap * SEGMENT_BYTES);
    new Uint8Array(next).set(
        new Uint8Array(segmentState(world).staging, 0, segmentState(world).count * SEGMENT_BYTES),
    );
    const state = segmentState(world);
    const stableIds = new Uint32Array(cap);
    const stableParts = new Uint32Array(cap);
    stableIds.set(state.stableIds.subarray(0, state.count));
    stableParts.set(state.stableParts.subarray(0, state.count));
    state.staging = next;
    state.f32 = new Float32Array(next);
    state.u32 = new Uint32Array(next);
    state.stableIds = stableIds;
    state.stableParts = stableParts;
    state.capacity = cap;
}

export function push(
    world: World,
    ax: number,
    ay: number,
    az: number,
    bx: number,
    by: number,
    bz: number,
    width: number,
    color: number,
    stableId = 0,
    stablePart?: number,
): void {
    const state = segmentState(world);
    if (state.count >= state.capacity) grow(world, state.count + 1);
    const index = state.count;
    const o = index * SEGMENT_FLOATS;
    state.f32[o] = ax;
    state.f32[o + 1] = ay;
    state.f32[o + 2] = az;
    state.f32[o + 3] = width;
    state.f32[o + 4] = bx;
    state.f32[o + 5] = by;
    state.f32[o + 6] = bz;
    state.u32[o + 7] = color;
    state.u32[o + 11] = 0;
    state.stableIds[index] = stableId;
    state.stableParts[index] = stablePart ?? index;
    state.count++;
}

// Stage the whole triangle once: every view receives the same world-space head.
export function head(
    world: World,
    tx: number,
    ty: number,
    tz: number,
    fromX: number,
    fromY: number,
    fromZ: number,
    tipLength: number,
    width: number,
    color: number,
    stableId = 0,
    firstPart?: number,
    nx = 0,
    ny = 1,
    nz = 0,
): number {
    let dx = tx - fromX;
    let dy = ty - fromY;
    let dz = tz - fromZ;
    const len = Math.hypot(dx, dy, dz);
    if (len < 1e-6) return 0;
    dx /= len;
    dy /= len;
    dz /= len;
    const back = tipLength < 0 ? len / 10 : tipLength;
    if (back <= 0) return 0;
    const normalLength = Math.hypot(nx, ny, nz);
    if (
        !Number.isFinite(normalLength) ||
        normalLength === 0 ||
        Math.abs(dx * nx + dy * ny + dz * nz) > normalLength * 1e-5
    )
        throw new RangeError(
            "Arrow normal must be nonzero and perpendicular to its world-space shaft",
        );
    const halfWidth = back * 0.5;
    const sx = ((ny * dz - nz * dy) / normalLength) * halfWidth;
    const sy = ((nz * dx - nx * dz) / normalLength) * halfWidth;
    const sz = ((nx * dy - ny * dx) / normalLength) * halfWidth;
    push(
        world,
        tx,
        ty,
        tz,
        tx - dx * back + sx,
        ty - dy * back + sy,
        tz - dz * back + sz,
        width,
        color,
        stableId,
        firstPart,
    );
    const state = segmentState(world);
    const o = (state.count - 1) * SEGMENT_FLOATS;
    state.f32[o + 8] = tx - dx * back - sx;
    state.f32[o + 9] = ty - dy * back - sy;
    state.f32[o + 10] = tz - dz * back - sz;
    state.u32[o + 11] = 1;
    return 1;
}

/** draw one world-space segment this frame (cleared next frame). `width` in pixels, `color` hex sRGB */
export function drawLine(
    world: World,
    a: ArrayLike<number>,
    b: ArrayLike<number>,
    color: number,
    width = 1,
): void {
    push(world, a[0], a[1], a[2], b[0], b[1], b[2], width, packColor(color, 1));
}

/** draw the 12 wireframe edges of an axis-aligned box this frame */
export function drawWireBox(
    world: World,
    min: ArrayLike<number>,
    max: ArrayLike<number>,
    color: number,
    width = 1,
): void {
    const c = packColor(color, 1);
    const x0 = min[0];
    const y0 = min[1];
    const z0 = min[2];
    const x1 = max[0];
    const y1 = max[1];
    const z1 = max[2];
    // Inline edges to avoid a per-call closure on the immediate path.
    push(world, x0, y0, z0, x1, y0, z0, width, c);
    push(world, x1, y0, z0, x1, y0, z1, width, c);
    push(world, x1, y0, z1, x0, y0, z1, width, c);
    push(world, x0, y0, z1, x0, y0, z0, width, c);
    push(world, x0, y1, z0, x1, y1, z0, width, c);
    push(world, x1, y1, z0, x1, y1, z1, width, c);
    push(world, x1, y1, z1, x0, y1, z1, width, c);
    push(world, x0, y1, z1, x0, y1, z0, width, c);
    push(world, x0, y0, z0, x0, y1, z0, width, c);
    push(world, x1, y0, z0, x1, y1, z0, width, c);
    push(world, x1, y0, z1, x1, y1, z1, width, c);
    push(world, x0, y0, z1, x0, y1, z1, width, c);
}

const WORLD_UP = [0, 1, 0] as const;

/** Draw a shaft and one filled triangle at `b`, in its colour and plane normal to `normal` (world up by default). The normal must be nonzero and perpendicular to the world-space shaft. Shaft width is in pixels; tipLength is in world units, with a negative value choosing a tenth of the shaft length. Zero omits the head. */
export function drawArrow(
    world: World,
    a: ArrayLike<number>,
    b: ArrayLike<number>,
    color: number,
    width = 1,
    tipLength = -1,
    normal: ArrayLike<number> = WORLD_UP,
): void {
    const c = packColor(color, 1);
    push(world, a[0], a[1], a[2], b[0], b[1], b[2], width, c);
    head(
        world,
        b[0],
        b[1],
        b[2],
        a[0],
        a[1],
        a[2],
        tipLength,
        width,
        c,
        0,
        undefined,
        normal[0],
        normal[1],
        normal[2],
    );
}

/** true once the GPU buffer is allocated (`warmSegments` ran with a device) */
export function ready(world: World): boolean {
    return !!segmentState(world).buffer;
}

/** Segment data published by the latest draw update, indexed by `instance_index`. */
export function lineStaging(world: World): Float32Array {
    return segmentState(world).f32;
}

/** Number of segments published by the latest draw update. */
export function lineCount(world: World): number {
    return segmentState(world).publishedCount;
}

/** Stable owner eids for the currently published segment indices. */
export function lineStableIds(world: World): Uint32Array {
    return segmentState(world).stableIds;
}

/** Stable per-owner segment indices for the currently published segment indices. */
export function lineStableParts(world: World): Uint32Array {
    return segmentState(world).stableParts;
}

/** reset the segment count without touching the GPU buffers (reload-safe pre-warm init) */
export function resetCount(world: World): void {
    segmentState(world).count = 0;
    segmentState(world).publishedCount = 0;
}

/** allocate the segment storage buffer and publish `lineSegments` */
export function warmSegments(world: World): void {
    segmentState(world).capacity = INITIAL;
    segmentState(world).staging = new ArrayBuffer(INITIAL * SEGMENT_BYTES);
    segmentState(world).f32 = new Float32Array(segmentState(world).staging);
    segmentState(world).u32 = new Uint32Array(segmentState(world).staging);
    segmentState(world).stableIds = new Uint32Array(INITIAL);
    segmentState(world).stableParts = new Uint32Array(INITIAL);
    segmentState(world).count = 0;
    segmentState(world).publishedCount = 0;
    const resources = segmentState(world);
    const buffer = world.gpu.root
        .createBuffer(d.arrayOf(Segment, INITIAL))
        .$usage("storage")
        .$name("shallot-line-segments");
    resources.buffer = buffer;
    world.gpu.buffers.set("lineSegments", world.gpu.root.unwrap(buffer));
    world.gpu.typed.set("lineSegments", buffer);
}

// Republish a grown buffer so its consumers re-resolve the binding.
export function flushSegments(world: World, device: GPUDevice): void {
    const resources = segmentState(world);
    if (!resources.buffer) return;
    if (resources.capacity * SEGMENT_BYTES > world.gpu.root.unwrap(resources.buffer).size) {
        const stale = resources.buffer;
        const buffer = world.gpu.root
            .createBuffer(d.arrayOf(Segment, resources.capacity))
            .$usage("storage")
            .$name("shallot-line-segments");
        resources.buffer = buffer;
        world.gpu.buffers.set("lineSegments", world.gpu.root.unwrap(buffer));
        world.gpu.typed.set("lineSegments", buffer);
        device.queue.onSubmittedWorkDone().then(() => stale.destroy());
    }
    const buffer = resources.buffer;
    if (resources.count > 0)
        device.queue.writeBuffer(
            world.gpu.root.unwrap(buffer),
            0,
            resources.staging,
            0,
            resources.count * SEGMENT_BYTES,
        );
    segmentState(world).publishedCount = resources.count;
    segmentState(world).count = 0;
}

export function disposeSegments(world: World): void {
    segmentState(world).buffer?.destroy();
    segmentState(world).buffer = null;
    segmentState(world).count = 0;
}
