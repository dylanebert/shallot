import type { StorageFlag, TgpuBuffer } from "typegpu";
import * as d from "typegpu/data";
import type { World } from "../../engine";
import { packColor } from "../../engine/utils";
import { SEGMENT_BYTES, SEGMENT_FLOATS, Segment } from "./pipeline";

// initial segment capacity; the CPU staging + GPU buffer double on demand for large wireframe workloads
const INITIAL = 1 << 14;

export interface SegmentState {
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

const segmentStateKey = { create: () => createSegmentState() };

function createSegmentState(capacity = INITIAL): SegmentState {
    const staging = new ArrayBuffer(capacity * SEGMENT_BYTES);
    return {
        buffer: null,
        staging,
        f32: new Float32Array(staging),
        u32: new Uint32Array(staging),
        stableIds: new Uint32Array(capacity),
        stableParts: new Uint32Array(capacity),
        capacity,
        count: 0,
        publishedCount: 0,
    };
}

/** Retained local-space geometry, built with the immediate drawing functions. */
export class GizmoAsset {
    readonly segments = createSegmentState(16);
    private _packed: { f32: Float32Array; u32: Uint32Array; count: number } | undefined;
    private _rebuildCount = 0;

    /** Number of times the engine has packed this asset's geometry for rendering. */
    get rebuilds(): number {
        return this._rebuildCount;
    }

    /** @internal */
    geometry() {
        if (!this._packed || this._packed.count !== this.segments.count) this.rebuildGeometry();
        return this._packed!;
    }

    /** @internal */
    rebuildGeometry(): void {
        const length = this.segments.count * SEGMENT_FLOATS;
        this._packed = {
            f32: this.segments.f32.slice(0, length),
            u32: this.segments.u32.slice(0, length),
            count: this.segments.count,
        };
        this._rebuildCount++;
    }
}

export function segmentState(target: World | GizmoAsset): SegmentState {
    return target instanceof GizmoAsset ? target.segments : target.resource(segmentStateKey);
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

export function grow(state: SegmentState, min: number): void {
    let cap = state.capacity;
    while (cap < min) cap *= 2;
    const next = new ArrayBuffer(cap * SEGMENT_BYTES);
    new Uint8Array(next).set(new Uint8Array(state.staging, 0, state.count * SEGMENT_BYTES));
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
    state: SegmentState,
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
    if (state.count >= state.capacity) grow(state, state.count + 1);
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

// Stage the whole triangle once: every view receives the same head geometry.
export function head(state: SegmentState, tipLength: number, normal: ArrayLike<number>): number {
    const shaft = (state.count - 1) * SEGMENT_FLOATS;
    const tx = state.f32[shaft + 4];
    const ty = state.f32[shaft + 5];
    const tz = state.f32[shaft + 6];
    let dx = tx - state.f32[shaft];
    let dy = ty - state.f32[shaft + 1];
    let dz = tz - state.f32[shaft + 2];
    const nx = normal[0],
        ny = normal[1],
        nz = normal[2];
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (len < 1e-6) return 0;
    dx /= len;
    dy /= len;
    dz /= len;
    // Both branches stay numeric so the default length needs no boxed number.
    const back = tipLength < 0 ? len / 10 : Math.abs(tipLength);
    if (back <= 0) return 0;
    const normalLength = Math.sqrt(nx * nx + ny * ny + nz * nz);
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
    if (state.count >= state.capacity) grow(state, state.count + 1);
    const index = state.count++;
    const o = index * SEGMENT_FLOATS;
    state.f32[o] = tx;
    state.f32[o + 1] = ty;
    state.f32[o + 2] = tz;
    state.f32[o + 3] = state.f32[shaft + 3];
    state.f32[o + 4] = tx - dx * back + sx;
    state.f32[o + 5] = ty - dy * back + sy;
    state.f32[o + 6] = tz - dz * back + sz;
    state.u32[o + 7] = state.u32[shaft + 7];
    state.stableIds[index] = state.stableIds[index - 1];
    state.stableParts[index] = state.stableParts[index - 1] + 1;
    state.f32[o + 8] = tx - dx * back - sx;
    state.f32[o + 9] = ty - dy * back - sy;
    state.f32[o + 10] = tz - dz * back - sz;
    state.u32[o + 11] = 1;
    return 1;
}

/** Draws one segment this frame on a World, or appends local-space geometry to an asset. Width is in pixels; color is hex sRGB. */
export function drawLine(
    world: World,
    a: ArrayLike<number>,
    b: ArrayLike<number>,
    color: number,
    width?: number,
): void;
export function drawLine(
    asset: GizmoAsset,
    a: ArrayLike<number>,
    b: ArrayLike<number>,
    color: number,
): void;
export function drawLine(
    world: World | GizmoAsset,
    a: ArrayLike<number>,
    b: ArrayLike<number>,
    color: number,
    width = 1,
): void {
    push(
        segmentState(world),
        a[0],
        a[1],
        a[2],
        b[0],
        b[1],
        b[2],
        world instanceof GizmoAsset ? 0 : width,
        packColor(color, 1),
    );
}

/** Draws twelve axis-aligned box edges this frame on a World, or appends local-space edges to an asset. */
export function drawWireBox(
    world: World,
    min: ArrayLike<number>,
    max: ArrayLike<number>,
    color: number,
    width?: number,
): void;
export function drawWireBox(
    asset: GizmoAsset,
    min: ArrayLike<number>,
    max: ArrayLike<number>,
    color: number,
): void;
export function drawWireBox(
    world: World | GizmoAsset,
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
    const state = segmentState(world);
    if (world instanceof GizmoAsset) width = 0;
    for (let i = 0; i < BOX_EDGES.length; i += 2) {
        const a = BOX_EDGES[i];
        const b = BOX_EDGES[i + 1];
        push(
            state,
            a & 1 ? x1 : x0,
            a & 2 ? y1 : y0,
            a & 4 ? z1 : z0,
            b & 1 ? x1 : x0,
            b & 2 ? y1 : y0,
            b & 4 ? z1 : z0,
            width,
            c,
        );
    }
}

const BOX_EDGES = [0, 1, 1, 5, 5, 4, 4, 0, 2, 3, 3, 7, 7, 6, 6, 2, 0, 2, 1, 3, 5, 7, 4, 6];
const WORLD_UP = [0, 1, 0] as const;

/** Draws a shaft and filled triangle at `b`, normal to `normal` (up by default). Coordinates, tipLength and normal are world-space on a World, local-space on an asset. The normal must be nonzero and perpendicular to the shaft. Width is in pixels on a World; asset calls must leave it undefined and use Gizmo.width. A negative tipLength chooses a tenth of the shaft length; zero omits the head. */
export function drawArrow(
    world: World,
    a: ArrayLike<number>,
    b: ArrayLike<number>,
    color: number,
    width?: number,
    tipLength?: number,
    normal?: ArrayLike<number>,
): void;
export function drawArrow(
    asset: GizmoAsset,
    a: ArrayLike<number>,
    b: ArrayLike<number>,
    color: number,
    width?: never,
    tipLength?: number,
    normal?: ArrayLike<number>,
): void;
export function drawArrow(
    world: World | GizmoAsset,
    a: ArrayLike<number>,
    b: ArrayLike<number>,
    color: number,
    width = 1,
    tipLength = -1,
    normal: ArrayLike<number> = WORLD_UP,
): void {
    const c = packColor(color, 1);
    const state = segmentState(world);
    push(state, a[0], a[1], a[2], b[0], b[1], b[2], world instanceof GizmoAsset ? 0 : width, c);
    head(state, tipLength, normal);
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
