// Edit-time heightfield flattening under the road network — the spec's Locked decision's "second half of
// the anti-decal argument" (UE Landscape Splines' "Apply Splines to Landscape" semantics): a road's own
// centreline follows the natural terrain height (so a straight XZ path still reads as terrain-conformed),
// but the ground is flattened *across* the road's width, cosine-eased back out to unmodified terrain by
// {@link FALLOFF}. `generate.ts`'s height kernel calls {@link flattenedHeightAt} in place of a bare
// `heightAt` at the vertex and its four finite-difference neighbours, so the emitted normal reflects the
// flattened surface too ("affected-region remesh" — every reseed/regenerate re-dispatches the whole
// kernel, so there's no separate patch step).
//
// Two independently-derived cosine-ease forms (`checks.md`'s two-derivations discipline, the same split
// `overlay/document.ts`/`overlay/rasterize.ts` use for distance): {@link flattenHeight} is the CPU
// reference `flatten.test.ts` pins directly; {@link flattenHeightGpu} is TGSL's own re-authoring of the
// same formula (it *can't* call the CPU one — TGSL has no FFI into plain JS), checked against it by a
// randomized differential test.

import { Compute, type State } from "@dylanebert/shallot";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { flattenSegments, type StrokeDocument } from "../overlay/document";
import { SPACING } from "./grid";
import { heightAt } from "./noise";

// FALLOFF derivation: `overlay/stroke.ts`'s OFF_ROAD_POINT is deliberately built "one grid cell (SPACING)
// beyond the road edge" — i.e. at `coreDist === SPACING` on the hand-authored stroke — so its own comment
// already promises "grid-aligned for an exact readVertices height" against *unmodified* terrain. Setting
// FALLOFF to exactly SPACING makes that promise true under flattening too: OFF_ROAD_POINT lands precisely
// at the falloff's own terminus (`flattenHeight`'s `coreDist >= falloff` arm), reading fully natural
// height with no coincidence required — not a value tuned to the capture, a value equal to a margin
// stage 4 already chose for an unrelated reason.
export const FALLOFF = SPACING; // 4 m

/** cosine-eased blend from `target` (the flattened plateau, at `coreDist <= 0`) out to `natural`
 *  (unmodified terrain, at `coreDist >= falloff`) — monotone in `coreDist` since the ease term's own
 *  derivative (`0.5·π·sin(π·t)`) never goes negative over `t ∈ [0, 1]`, so the blend always moves *toward*
 *  natural as `coreDist` grows, never overshooting either endpoint.
 * @example flattenHeight(10, 4, -2, 4) // 4 — inside the core, the target height wins outright
 * @example flattenHeight(10, 4, 4, 4) // 10 — at/past the falloff distance, fully natural */
export function flattenHeight(
    natural: number,
    targetHeight: number,
    coreDist: number,
    falloff: number,
): number {
    if (coreDist <= 0) return targetHeight;
    if (coreDist >= falloff) return natural;
    const t = coreDist / falloff;
    const ease = 0.5 - 0.5 * Math.cos(Math.PI * t);
    return targetHeight + (natural - targetHeight) * ease;
}

/** the GPU kernel's own cosine-ease derivation — written independently of {@link flattenHeight} (TGSL
 *  can't call a plain JS function anyway), CPU-callable so `flatten.test.ts`'s differential oracle can
 *  compare the two forms with no device (`testing.md`'s logic tier). */
export const flattenHeightGpu = tgpu.fn(
    [d.f32, d.f32, d.f32, d.f32],
    d.f32,
)((natural, targetHeight, coreDist, falloff) => {
    "use gpu";
    if (coreDist <= 0) return targetHeight;
    if (coreDist >= falloff) return natural;
    const t = coreDist / falloff;
    const ease = d.f32(0.5) - d.f32(0.5) * std.cos(d.f32(Math.PI) * t);
    return targetHeight + (natural - targetHeight) * ease;
});

/** one flattened polyline segment, GPU-side. */
const NetworkSegment = d.struct({ a: d.vec2f, b: d.vec2f, halfWidth: d.f32 });
/** one polygon stamp's vertex span within the shared `polyVerts` buffer, plus its precomputed centroid —
 *  the single flat target height under the whole footprint (a real carpark is one flat plane, not a
 *  cross-section following its own nearest edge, which is what a per-edge target would produce). */
const NetworkPolygon = d.struct({ start: d.u32, count: d.u32, centroid: d.vec2f });

const NetworkParams = d.struct({ segmentCount: d.u32, polygonCount: d.u32 });

export const networkLayout = tgpu.bindGroupLayout({
    segments: { storage: d.arrayOf(NetworkSegment), access: "readonly" },
    polygons: { storage: d.arrayOf(NetworkPolygon), access: "readonly" },
    polyVerts: { storage: d.arrayOf(d.vec2f), access: "readonly" },
    params: { uniform: NetworkParams },
});

const NetworkCore = d.struct({ coreDist: d.f32, targetHeight: d.f32 });

/** the nearest network primitive's core boundary distance + target height at (px, pz) — the GPU-side
 *  geometry half of the flatten pipeline. Segments: clamped-projection distance to the core boundary
 *  (the same clamped form `overlay/rasterize.ts`'s `segmentDistanceGpu` uses — an independent derivation
 *  from `overlay/document.ts`'s cross-product form) minus half-width; target is the *natural* height at
 *  the projected point, so the road's longitudinal profile still tracks the terrain. Polygons: ray-cast
 *  winding for the sign, nearest-edge distance for the magnitude (mirrors `rasterize.ts`'s
 *  `polygonDistanceGpu`); target is the natural height at the polygon's own centroid. */
const networkCore = tgpu.fn(
    [d.f32, d.f32],
    NetworkCore,
)((px, pz) => {
    "use gpu";
    let bestCore = d.f32(3.402823e38);
    let bestTarget = d.f32(0);

    let i = d.u32(0);
    for (; i < networkLayout.$.params.segmentCount; i = i + d.u32(1)) {
        const seg = networkLayout.$.segments[i];
        const abx = seg.b.x - seg.a.x;
        const abz = seg.b.y - seg.a.y;
        const apx = px - seg.a.x;
        const apz = pz - seg.a.y;
        const dd = abx * abx + abz * abz;
        let t = d.f32(0);
        if (dd > 0) t = std.clamp((apx * abx + apz * abz) / dd, 0, 1);
        const cx = seg.a.x + t * abx;
        const cz = seg.a.y + t * abz;
        const core = std.distance(d.vec2f(px, pz), d.vec2f(cx, cz)) - seg.halfWidth;
        if (core < bestCore) {
            bestCore = core;
            bestTarget = heightAt(cx, cz);
        }
    }

    let p = d.u32(0);
    for (; p < networkLayout.$.params.polygonCount; p = p + d.u32(1)) {
        const poly = networkLayout.$.polygons[p];
        let inside = false;
        let nearestEdge = d.f32(3.402823e38);
        let k = d.u32(0);
        for (; k < poly.count; k = k + d.u32(1)) {
            const a = networkLayout.$.polyVerts[poly.start + k];
            const b = networkLayout.$.polyVerts[poly.start + ((k + d.u32(1)) % poly.count)];
            if (a.y > pz !== b.y > pz) {
                const xCross = a.x + ((pz - a.y) / (b.y - a.y)) * (b.x - a.x);
                if (px < xCross) inside = !inside;
            }
            const eAbx = b.x - a.x;
            const eAbz = b.y - a.y;
            const eApx = px - a.x;
            const eApz = pz - a.y;
            const eDd = eAbx * eAbx + eAbz * eAbz;
            let et = d.f32(0);
            if (eDd > 0) et = std.clamp((eApx * eAbx + eApz * eAbz) / eDd, 0, 1);
            const ecx = a.x + et * eAbx;
            const ecz = a.y + et * eAbz;
            const edgeDist = std.distance(d.vec2f(px, pz), d.vec2f(ecx, ecz));
            if (edgeDist < nearestEdge) nearestEdge = edgeDist;
        }
        const core = std.select(nearestEdge, -nearestEdge, inside);
        if (core < bestCore) {
            bestCore = core;
            bestTarget = heightAt(poly.centroid.x, poly.centroid.y);
        }
    }

    return NetworkCore({ coreDist: bestCore, targetHeight: bestTarget });
});

/** the flattened height at world (x, z): natural terrain height blended toward the nearest network
 *  primitive's core target via {@link flattenHeightGpu}'s cosine ease. `generate.ts`'s height kernel calls
 *  this in place of a bare `heightAt`, at the vertex and its four finite-difference neighbours alike, so
 *  the emitted normal reflects the flattened surface. An empty network (no segments, no polygons) leaves
 *  `bestCore` at its f32-max sentinel, always `>= FALLOFF`, so this degrades to plain `heightAt`. */
export const flattenedHeightAt = tgpu.fn(
    [d.f32, d.f32],
    d.f32,
)((x, z) => {
    "use gpu";
    const natural = heightAt(x, z);
    const core = networkCore(x, z);
    return flattenHeightGpu(natural, core.targetHeight, core.coreDist, d.f32(FALLOFF));
});

type FlattenRoot = typeof Compute.root;

let segRaw: GPUBuffer | null = null;
let polyRaw: GPUBuffer | null = null;
let vertsRaw: GPUBuffer | null = null;
let paramsRaw: GPUBuffer | null = null;
let bindGroup: ReturnType<FlattenRoot["createBindGroup"]> | null = null;

function teardownNetworkBuffers(): void {
    segRaw?.destroy();
    polyRaw?.destroy();
    vertsRaw?.destroy();
    paramsRaw?.destroy();
    segRaw = null;
    polyRaw = null;
    vertsRaw = null;
    paramsRaw = null;
    bindGroup = null;
}

/** register the network buffers' cleanup against `state.onDispose` — the same in-place-rebuild-safe
 *  pattern `overlay/atlas.ts`/`overlay/rasterize.ts` use. Call once, from `terrain.ts`'s own `warm()`. */
export function warmNetwork(state: State): void {
    teardownNetworkBuffers();
    state.onDispose(teardownNetworkBuffers);
}

/** (re)build the network's GPU geometry buffers + bind group from `doc` — called once at warm and again
 *  on every reseed (`terrain.ts`'s `regenerate`), since a new network has a different segment/polygon
 *  count. Mirrors `overlay/rasterize.ts`'s per-tile buffer-rebuild shape, but persists across the height
 *  kernel's own dispatches instead of being rebuilt per redraw — the kernel dispatches once per reseed,
 *  not once per throttled tile. */
export function setNetwork(doc: StrokeDocument): void {
    teardownNetworkBuffers();
    const { device, root } = Compute;

    const segments = flattenSegments(doc);
    const segmentsData =
        segments.length > 0 ? segments : [{ ax: 0, az: 0, bx: 0, bz: 0, halfWidth: 0 }];
    segRaw = device.createBuffer({
        label: "network-segments",
        size: segmentsData.length * d.sizeOf(NetworkSegment),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const segBuf = root
        .createBuffer(d.arrayOf(NetworkSegment, segmentsData.length), segRaw)
        .$usage("storage");
    segBuf.write(
        segmentsData.map((s) => ({
            a: d.vec2f(s.ax, s.az),
            b: d.vec2f(s.bx, s.bz),
            halfWidth: s.halfWidth,
        })),
    );

    const polyVerts: { x: number; y: number }[] = [];
    const polygonsData = doc.polygons.map((poly) => {
        const start = polyVerts.length;
        let cx = 0;
        let cz = 0;
        for (const [x, z] of poly.points) {
            polyVerts.push({ x, y: z });
            cx += x;
            cz += z;
        }
        cx /= poly.points.length;
        cz /= poly.points.length;
        return { start, count: poly.points.length, centroid: { x: cx, y: cz } };
    });
    const polygonsSafe =
        polygonsData.length > 0 ? polygonsData : [{ start: 0, count: 0, centroid: { x: 0, y: 0 } }];
    polyRaw = device.createBuffer({
        label: "network-polygons",
        size: polygonsSafe.length * d.sizeOf(NetworkPolygon),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const polyBuf = root
        .createBuffer(d.arrayOf(NetworkPolygon, polygonsSafe.length), polyRaw)
        .$usage("storage");
    polyBuf.write(
        polygonsSafe.map((p) => ({
            start: p.start,
            count: p.count,
            centroid: d.vec2f(p.centroid.x, p.centroid.y),
        })),
    );

    const vertsSafe = polyVerts.length > 0 ? polyVerts : [{ x: 0, y: 0 }];
    vertsRaw = device.createBuffer({
        label: "network-polyverts",
        size: vertsSafe.length * d.sizeOf(d.vec2f),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const vertsBuf = root
        .createBuffer(d.arrayOf(d.vec2f, vertsSafe.length), vertsRaw)
        .$usage("storage");
    vertsBuf.write(vertsSafe.map((v) => d.vec2f(v.x, v.y)));

    paramsRaw = device.createBuffer({
        label: "network-params",
        size: d.sizeOf(NetworkParams),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const paramsBuf = root.createBuffer(NetworkParams, paramsRaw).$usage("uniform");
    paramsBuf.write({ segmentCount: segments.length, polygonCount: doc.polygons.length });

    bindGroup = root.createBindGroup(networkLayout, {
        segments: segBuf,
        polygons: polyBuf,
        polyVerts: vertsBuf,
        params: paramsBuf,
    });
}

/** the live network bind group `generate.ts`'s height kernel dispatch attaches — throws before the first
 *  {@link setNetwork} call, the same read-before-warm guard `overlay/atlas.ts`'s `bindings()` uses. */
export function networkBindGroup(): NonNullable<typeof bindGroup> {
    if (!bindGroup) throw new Error("flatten: networkBindGroup read before setNetwork()");
    return bindGroup;
}

/** the emitted flatten WGSL — the device-free structural seam `flatten.test.ts` resolves. */
export function flattenWgsl(): string {
    return tgpu.resolve([flattenHeightGpu, networkCore, flattenedHeightAt], { names: "strict" });
}
