// The line producer's segment buffer: one shared CPU staging array fed by the immediate API
// (`segment` / `box` / `arrow`) and the retained-component expansion, doubled on demand, uploaded to a
// GPU storage buffer each frame and drawn as one instanced quad per segment. Internal —
// `segment` / `box` / `arrow` re-export through the barrel; the staging, the `Lines` handle, and the GPU
// lifecycle stay off it (`head` / `push` are shared with the retained expansion in `index.ts`).

import type { StorageFlag, TgpuBuffer } from "typegpu";
import * as d from "typegpu/data";
import { DrawIndexedIndirect } from "../../core/rendering";
import { Compute, type State } from "../../engine";
import { worldResource } from "../../engine/runtime";
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

const segmentStateKey = Symbol("shallot.line-segments");

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

function segmentState(): SegmentState {
    return worldResource(segmentStateKey, createSegmentState);
}

export function initializeSegmentState(state: State): void {
    state.resource(segmentStateKey, createSegmentState);
}

// the producer's GPU publication. `count` is the segments packed this frame (reset after the upload);
// `args` is the `DrawIndexedIndirect` buffer whose `instanceCount` lane the live segment count drives.
// Internal — the unit test reads `count` to check the immediate-API expansion; `args` is COPY_SRC so a
// A one-shot probe can read back the produced instance count
interface Lines {
    readonly count: number;
    args: (TgpuBuffer<typeof DrawIndexedIndirect> & { usableAsIndirect: true }) | null;
}

export const Lines: Lines = new Proxy({} as Lines, {
    get(_target, key) {
        const state = segmentState();
        return key === "count" ? state.count : state[key as keyof SegmentState];
    },
    set(_target, key, value) {
        (segmentState() as unknown as Record<PropertyKey, unknown>)[key] = value;
        return true;
    },
});

function grow(min: number): void {
    let cap = segmentState().capacity;
    while (cap < min) cap *= 2;
    const next = new ArrayBuffer(cap * SEGMENT_BYTES);
    new Uint8Array(next).set(
        new Uint8Array(segmentState().staging, 0, segmentState().count * SEGMENT_BYTES),
    );
    segmentState().staging = next;
    segmentState().f32 = new Float32Array(next);
    segmentState().u32 = new Uint32Array(next);
    segmentState().capacity = cap;
}

export function push(
    ax: number,
    ay: number,
    az: number,
    bx: number,
    by: number,
    bz: number,
    width: number,
    color: number,
): void {
    if (segmentState().count >= segmentState().capacity) grow(segmentState().count + 1);
    const o = segmentState().count * SEGMENT_FLOATS;
    segmentState().f32[o] = ax;
    segmentState().f32[o + 1] = ay;
    segmentState().f32[o + 2] = az;
    segmentState().f32[o + 3] = width;
    segmentState().f32[o + 4] = bx;
    segmentState().f32[o + 5] = by;
    segmentState().f32[o + 6] = bz;
    segmentState().u32[o + 7] = color;
    segmentState().count++;
}

// four world-space fins from the tip back along the shaft. perpendicular basis off an up reference that
// flips near-vertical shafts; fins go back `0.2 * shaftLen * size` and out half that along ±e1/±e2
export function head(
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
    push(tx, ty, tz, bx + e1x * out, by + e1y * out, bz + e1z * out, width, color);
    push(tx, ty, tz, bx - e1x * out, by - e1y * out, bz - e1z * out, width, color);
    push(tx, ty, tz, bx + e2x * out, by + e2y * out, bz + e2z * out, width, color);
    push(tx, ty, tz, bx - e2x * out, by - e2y * out, bz - e2z * out, width, color);
}

/** draw one world-space segment this frame (cleared next frame). `width` in pixels, `color` hex sRGB */
export function segment(
    a: ArrayLike<number>,
    b: ArrayLike<number>,
    color: number,
    width = 1,
): void {
    push(a[0], a[1], a[2], b[0], b[1], b[2], width, packColor(color, 1));
}

/** draw the 12 wireframe edges of an axis-aligned box this frame */
export function box(
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
    push(x0, y0, z0, x1, y0, z0, width, c);
    push(x1, y0, z0, x1, y0, z1, width, c);
    push(x1, y0, z1, x0, y0, z1, width, c);
    push(x0, y0, z1, x0, y0, z0, width, c);
    push(x0, y1, z0, x1, y1, z0, width, c);
    push(x1, y1, z0, x1, y1, z1, width, c);
    push(x1, y1, z1, x0, y1, z1, width, c);
    push(x0, y1, z1, x0, y1, z0, width, c);
    push(x0, y0, z0, x0, y1, z0, width, c);
    push(x1, y0, z0, x1, y1, z0, width, c);
    push(x1, y0, z1, x1, y1, z1, width, c);
    push(x0, y0, z1, x0, y1, z1, width, c);
}

/** draw a world-space arrow (shaft + a fletched head at `b`) this frame */
export function arrow(
    a: ArrayLike<number>,
    b: ArrayLike<number>,
    color: number,
    width = 1,
    size = 1,
): void {
    const c = packColor(color, 1);
    push(a[0], a[1], a[2], b[0], b[1], b[2], width, c);
    head(b[0], b[1], b[2], a[0], a[1], a[2], size, width, c);
}

/** true once the GPU buffers are allocated (`warmSegments` ran with a device) */
export function ready(): boolean {
    return !!segmentState().buffer && !!Lines.args;
}

/** reset the segment count without touching the GPU buffers (reload-safe pre-warm init) */
export function resetCount(): void {
    segmentState().count = 0;
}

/** allocate the segment storage + indirect-args buffers and publish `lineSegments` */
export function warmSegments(_device: GPUDevice): void {
    segmentState().capacity = INITIAL;
    segmentState().staging = new ArrayBuffer(INITIAL * SEGMENT_BYTES);
    segmentState().f32 = new Float32Array(segmentState().staging);
    segmentState().u32 = new Uint32Array(segmentState().staging);
    segmentState().count = 0;
    const state = segmentState();
    const buffer = Compute.root
        .createBuffer(d.arrayOf(Segment, INITIAL))
        .$usage("storage")
        .$name("shallot-line-segments");
    state.buffer = buffer;
    Compute.buffers.set("lineSegments", Compute.root.unwrap(buffer));
    Compute.typed.set("lineSegments", buffer);
    Lines.args = Compute.root
        .createBuffer(DrawIndexedIndirect)
        .$usage("indirect")
        .$name("shallot-line-args");
}

// grow the GPU buffer to match the CPU staging (rare); republish so sear re-resolves the binding, then
// upload this frame's segments, write the indirect record (instanceCount = live count), and clear
export function flushSegments(device: GPUDevice, quadBase: number): void {
    const state = segmentState();
    if (!state.buffer || !Lines.args) return;
    if (state.capacity * SEGMENT_BYTES > Compute.root.unwrap(state.buffer).size) {
        const stale = state.buffer;
        const buffer = Compute.root
            .createBuffer(d.arrayOf(Segment, state.capacity))
            .$usage("storage")
            .$name("shallot-line-segments");
        state.buffer = buffer;
        Compute.buffers.set("lineSegments", Compute.root.unwrap(buffer));
        Compute.typed.set("lineSegments", buffer);
        device.queue.onSubmittedWorkDone().then(() => stale.destroy());
    }
    const buffer = state.buffer;
    const args = Lines.args;
    if (!buffer || !args) return;
    if (state.count > 0)
        device.queue.writeBuffer(
            Compute.root.unwrap(buffer),
            0,
            state.staging,
            0,
            state.count * SEGMENT_BYTES,
        );
    args.write({
        indexCount: 6,
        instanceCount: state.count,
        firstIndex: quadBase,
        baseVertex: 0,
        firstInstance: 0,
    });
    segmentState().count = 0;
}

export function disposeSegments(): void {
    segmentState().buffer?.destroy();
    Lines.args?.destroy();
    segmentState().buffer = null;
    Lines.args = null;
    segmentState().count = 0;
}
