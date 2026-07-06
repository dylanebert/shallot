// The line producer's segment buffer: one shared CPU staging array fed by the immediate API
// (`segment` / `box` / `arrow`) and the retained-component expansion, doubled on demand, uploaded to a
// GPU storage buffer each frame and drawn as one instanced quad per segment. Internal —
// `segment` / `box` / `arrow` re-export through the barrel; the staging, the `Lines` handle, and the GPU
// lifecycle stay off it (`head` / `push` are shared with the retained expansion in `index.ts`).

import { Compute } from "../../engine";
import { packColor } from "../../engine/utils/core";

// one segment = two world endpoints + a pixel width + a packed sRGBA color, 32 bytes / two vec4 reads
// (read-all per instance coalesces near the floor, gpu.md). `a.xyz` shares its 16-byte slot with `width`,
// `b.xyz` with `color`
const SEGMENT_FLOATS = 8;
const SEGMENT_BYTES = 32;
// initial segment capacity; the CPU staging + GPU buffer double on demand (BVH wireframes push thousands)
const INITIAL = 1 << 14;

let _segBuf: GPUBuffer | null = null;
let _staging = new ArrayBuffer(INITIAL * SEGMENT_BYTES);
let _f32 = new Float32Array(_staging);
let _u32 = new Uint32Array(_staging);
let _cap = INITIAL;
let _count = 0;
const _args = new Uint32Array([6, 0, 0, 0, 0]);

// the producer's GPU publication. `count` is the segments packed this frame (reset after the upload);
// `args` is the `DrawIndexedIndirect` buffer whose `instanceCount` lane the live segment count drives.
// Internal — the unit test reads `count` to check the immediate-API expansion; `args` is COPY_SRC so a
// gym Mirror can read back the produced instance count
interface Lines {
    readonly count: number;
    args: GPUBuffer | null;
}

export const Lines: Lines = {
    get count(): number {
        return _count;
    },
    args: null,
};

function grow(min: number): void {
    let cap = _cap;
    while (cap < min) cap *= 2;
    const next = new ArrayBuffer(cap * SEGMENT_BYTES);
    new Uint8Array(next).set(new Uint8Array(_staging, 0, _count * SEGMENT_BYTES));
    _staging = next;
    _f32 = new Float32Array(next);
    _u32 = new Uint32Array(next);
    _cap = cap;
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
    if (_count >= _cap) grow(_count + 1);
    const o = _count * SEGMENT_FLOATS;
    _f32[o] = ax;
    _f32[o + 1] = ay;
    _f32[o + 2] = az;
    _f32[o + 3] = width;
    _f32[o + 4] = bx;
    _f32[o + 5] = by;
    _f32[o + 6] = bz;
    _u32[o + 7] = color;
    _count++;
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
    return !!_segBuf && !!Lines.args;
}

/** reset the segment count without touching the GPU buffers (reload-safe pre-warm init) */
export function resetCount(): void {
    _count = 0;
}

/** allocate the segment storage + indirect-args buffers and publish `lineSegments` */
export function warmSegments(device: GPUDevice): void {
    _cap = INITIAL;
    _staging = new ArrayBuffer(INITIAL * SEGMENT_BYTES);
    _f32 = new Float32Array(_staging);
    _u32 = new Uint32Array(_staging);
    _count = 0;
    _segBuf = device.createBuffer({
        label: "kitchen-line-segments",
        size: INITIAL * SEGMENT_BYTES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    Compute.buffers.set("lineSegments", _segBuf);
    Lines.args = device.createBuffer({
        label: "kitchen-line-args",
        // COPY_SRC so a gym Mirror can read back instanceCount (gpu.md)
        size: 20,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
}

// grow the GPU buffer to match the CPU staging (rare); republish so sear re-resolves the binding, then
// upload this frame's segments, write the indirect record (instanceCount = live count), and clear
export function flushSegments(device: GPUDevice, quadBase: number): void {
    if (!_segBuf || !Lines.args) return;
    if (_cap * SEGMENT_BYTES > _segBuf.size) {
        const stale = _segBuf;
        _segBuf = device.createBuffer({
            label: "kitchen-line-segments",
            size: _cap * SEGMENT_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        Compute.buffers.set("lineSegments", _segBuf);
        device.queue.onSubmittedWorkDone().then(() => stale.destroy());
    }
    if (_count > 0) device.queue.writeBuffer(_segBuf, 0, _staging, 0, _count * SEGMENT_BYTES);
    _args[1] = _count;
    _args[2] = quadBase;
    device.queue.writeBuffer(Lines.args, 0, _args);
    _count = 0;
}

export function disposeSegments(): void {
    _segBuf?.destroy();
    Lines.args?.destroy();
    _segBuf = null;
    Lines.args = null;
    _count = 0;
}
