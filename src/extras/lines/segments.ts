// The line producer's segment buffer: one shared CPU staging array fed by the immediate API
// (`segment` / `box` / `arrow`) and the retained-component expansion, doubled on demand, uploaded to a
// GPU storage buffer each frame and drawn as one instanced quad per segment. Internal —
// `segment` / `box` / `arrow` re-export through the barrel; the staging, the `Lines` handle, and the GPU
// lifecycle stay off it (`head` / `push` are shared with the retained expansion in `index.ts`).

import type { StorageFlag, TgpuBuffer } from "typegpu";
import * as d from "typegpu/data";
import { DrawIndexedIndirect } from "../../core/rendering";
import type { World } from "../../engine";

import { packColor } from "../../engine/utils";
import { Segment } from "./surface";

// one segment = two world endpoints + a pixel width + a packed sRGBA color, 32 bytes / two vec4 reads
// (read-all per instance coalesces near the floor). `a.xyz` shares its 16-byte slot with `width`,
// `b.xyz` with `color`. Stride derived from the schema (a second hand-authored stride is layout
// drift waiting to happen).
const SEGMENT_BYTES = d.sizeOf(Segment);
const SEGMENT_FLOATS = SEGMENT_BYTES / 4;
// initial segment capacity; the CPU staging + GPU buffer double on demand (BVH wireframes push thousands)
const INITIAL = 1 << 14;

interface SegmentState {
    buffer: (TgpuBuffer<d.WgslArray<typeof Segment>> & StorageFlag) | null;
    staging: ArrayBuffer;
    f32: Float32Array;
    u32: Uint32Array;
    capacity: number;
    count: number;
    args: (TgpuBuffer<typeof DrawIndexedIndirect> & { usableAsIndirect: true }) | null;
}

const segmentStateKey = { create: createSegmentState };

function createSegmentState(): SegmentState {
    const staging = new ArrayBuffer(INITIAL * SEGMENT_BYTES);
    return {
        buffer: null,
        staging,
        f32: new Float32Array(staging),
        u32: new Uint32Array(staging),
        capacity: INITIAL,
        count: 0,
        args: null,
    };
}

function segmentState(state: World): SegmentState {
    return state.resource(segmentStateKey);
}

export function initializeSegmentState(state: World): void {
    state.resource(segmentStateKey);
}

// the producer's GPU publication. `count` is the segments packed this frame (reset after the upload);
// `args` is the `DrawIndexedIndirect` buffer whose `instanceCount` lane the live segment count drives.
// Internal — the unit test reads `count` to check the immediate-API expansion; `args` is COPY_SRC so a
// A one-shot probe can read back the produced instance count
interface Lines {
    readonly count: number;
    args: (TgpuBuffer<typeof DrawIndexedIndirect> & { usableAsIndirect: true }) | null;
}

export const Lines: import("../../engine").Resource<Lines> = {
    create: (state) => state.resource(segmentStateKey),
};

function grow(state: World, min: number): void {
    let cap = segmentState(state).capacity;
    while (cap < min) cap *= 2;
    const next = new ArrayBuffer(cap * SEGMENT_BYTES);
    new Uint8Array(next).set(
        new Uint8Array(segmentState(state).staging, 0, segmentState(state).count * SEGMENT_BYTES),
    );
    segmentState(state).staging = next;
    segmentState(state).f32 = new Float32Array(next);
    segmentState(state).u32 = new Uint32Array(next);
    segmentState(state).capacity = cap;
}

export function push(
    state: World,
    ax: number,
    ay: number,
    az: number,
    bx: number,
    by: number,
    bz: number,
    width: number,
    color: number,
): void {
    if (segmentState(state).count >= segmentState(state).capacity)
        grow(state, segmentState(state).count + 1);
    const o = segmentState(state).count * SEGMENT_FLOATS;
    segmentState(state).f32[o] = ax;
    segmentState(state).f32[o + 1] = ay;
    segmentState(state).f32[o + 2] = az;
    segmentState(state).f32[o + 3] = width;
    segmentState(state).f32[o + 4] = bx;
    segmentState(state).f32[o + 5] = by;
    segmentState(state).f32[o + 6] = bz;
    segmentState(state).u32[o + 7] = color;
    segmentState(state).count++;
}

// four world-space fins from the tip back along the shaft. perpendicular basis off an up reference that
// flips near-vertical shafts; fins go back `0.2 * shaftLen * size` and out half that along ±e1/±e2
export function head(
    state: World,
    tx: number,
    ty: number,
    tz: number,
    fromX: number,
    fromY: number,
    fromZ: number,
    size: number,
    width: number,
    color: number,
): void {
    let dx = tx - fromX;
    let dy = ty - fromY;
    let dz = tz - fromZ;
    const len = Math.hypot(dx, dy, dz);
    if (len < 1e-6) return;
    dx /= len;
    dy /= len;
    dz /= len;
    const ux = Math.abs(dy) < 0.99 ? 0 : 1;
    const uy = Math.abs(dy) < 0.99 ? 1 : 0;
    let e1x = dy * 0 - dz * uy;
    let e1y = dz * ux - dx * 0;
    let e1z = dx * uy - dy * ux;
    const e1l = Math.hypot(e1x, e1y, e1z);
    e1x /= e1l;
    e1y /= e1l;
    e1z /= e1l;
    const e2x = dy * e1z - dz * e1y;
    const e2y = dz * e1x - dx * e1z;
    const e2z = dx * e1y - dy * e1x;
    const back = 0.2 * len * size;
    const out = back * 0.5;
    // fin base, one step back from the tip along the shaft; four fins splay ±e1 / ±e2 from it
    const bx = tx - dx * back;
    const by = ty - dy * back;
    const bz = tz - dz * back;
    push(state, tx, ty, tz, bx + e1x * out, by + e1y * out, bz + e1z * out, width, color);
    push(state, tx, ty, tz, bx - e1x * out, by - e1y * out, bz - e1z * out, width, color);
    push(state, tx, ty, tz, bx + e2x * out, by + e2y * out, bz + e2z * out, width, color);
    push(state, tx, ty, tz, bx - e2x * out, by - e2y * out, bz - e2z * out, width, color);
}

/** draw one world-space segment this frame (cleared next frame). `width` in pixels, `color` hex sRGB */
export function segment(
    state: World,
    a: ArrayLike<number>,
    b: ArrayLike<number>,
    color: number,
    width = 1,
): void {
    push(state, a[0], a[1], a[2], b[0], b[1], b[2], width, packColor(color, 1));
}

/** draw the 12 wireframe edges of an axis-aligned box this frame */
export function box(
    state: World,
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
    // 4 bottom edges, 4 top, 4 verticals — inlined (no per-call closure: box() is on the scale path)
    push(state, x0, y0, z0, x1, y0, z0, width, c);
    push(state, x1, y0, z0, x1, y0, z1, width, c);
    push(state, x1, y0, z1, x0, y0, z1, width, c);
    push(state, x0, y0, z1, x0, y0, z0, width, c);
    push(state, x0, y1, z0, x1, y1, z0, width, c);
    push(state, x1, y1, z0, x1, y1, z1, width, c);
    push(state, x1, y1, z1, x0, y1, z1, width, c);
    push(state, x0, y1, z1, x0, y1, z0, width, c);
    push(state, x0, y0, z0, x0, y1, z0, width, c);
    push(state, x1, y0, z0, x1, y1, z0, width, c);
    push(state, x1, y0, z1, x1, y1, z1, width, c);
    push(state, x0, y0, z1, x0, y1, z1, width, c);
}

/** draw a world-space arrow (shaft + a fletched head at `b`) this frame */
export function arrow(
    state: World,
    a: ArrayLike<number>,
    b: ArrayLike<number>,
    color: number,
    width = 1,
    size = 1,
): void {
    const c = packColor(color, 1);
    push(state, a[0], a[1], a[2], b[0], b[1], b[2], width, c);
    head(state, b[0], b[1], b[2], a[0], a[1], a[2], size, width, c);
}

/** true once the GPU buffers are allocated (`warmSegments` ran with a device) */
export function ready(state: World): boolean {
    return !!segmentState(state).buffer && !!state.resource(Lines).args;
}

/** reset the segment count without touching the GPU buffers (reload-safe pre-warm init) */
export function resetCount(state: World): void {
    segmentState(state).count = 0;
}

/** allocate the segment storage + indirect-args buffers and publish `lineSegments` */
export function warmSegments(state: World, _device: GPUDevice): void {
    segmentState(state).capacity = INITIAL;
    segmentState(state).staging = new ArrayBuffer(INITIAL * SEGMENT_BYTES);
    segmentState(state).f32 = new Float32Array(segmentState(state).staging);
    segmentState(state).u32 = new Uint32Array(segmentState(state).staging);
    segmentState(state).count = 0;
    const resources = segmentState(state);
    const buffer = state.gpu.root
        .createBuffer(d.arrayOf(Segment, INITIAL))
        .$usage("storage")
        .$name("shallot-line-segments");
    resources.buffer = buffer;
    state.gpu.buffers.set("lineSegments", state.gpu.root.unwrap(buffer));
    state.gpu.typed.set("lineSegments", buffer);
    state.resource(Lines).args = state.gpu.root
        .createBuffer(DrawIndexedIndirect)
        .$usage("indirect")
        .$name("shallot-line-args");
}

// grow the GPU buffer to match the CPU staging (rare); republish so sear re-resolves the binding, then
// upload this frame's segments, write the indirect record (instanceCount = live count), and clear
export function flushSegments(state: World, device: GPUDevice, quadBase: number): void {
    const _lines = state.resource(Lines);

    const resources = segmentState(state);
    if (!resources.buffer || !_lines.args) return;
    if (resources.capacity * SEGMENT_BYTES > state.gpu.root.unwrap(resources.buffer).size) {
        const stale = resources.buffer;
        const buffer = state.gpu.root
            .createBuffer(d.arrayOf(Segment, resources.capacity))
            .$usage("storage")
            .$name("shallot-line-segments");
        resources.buffer = buffer;
        state.gpu.buffers.set("lineSegments", state.gpu.root.unwrap(buffer));
        state.gpu.typed.set("lineSegments", buffer);
        device.queue.onSubmittedWorkDone().then(() => stale.destroy());
    }
    const buffer = resources.buffer;
    const args = _lines.args;
    if (!buffer || !args) return;
    if (resources.count > 0)
        device.queue.writeBuffer(
            state.gpu.root.unwrap(buffer),
            0,
            resources.staging,
            0,
            resources.count * SEGMENT_BYTES,
        );
    args.write({
        indexCount: 6,
        instanceCount: resources.count,
        firstIndex: quadBase,
        baseVertex: 0,
        firstInstance: 0,
    });
    segmentState(state).count = 0;
}

export function disposeSegments(state: World): void {
    const _lines = state.resource(Lines);

    segmentState(state).buffer?.destroy();
    _lines.args?.destroy();
    segmentState(state).buffer = null;
    _lines.args = null;
    segmentState(state).count = 0;
}
