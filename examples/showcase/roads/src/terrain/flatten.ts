// Edit-time heightfield flattening under the road network — the spec's Locked decision's "second half of
// the anti-decal argument" (UE Landscape Splines' "Apply Splines to Landscape" semantics): the ground is
// flattened *across* the road's width, cosine-eased back out to unmodified terrain over {@link
// computeFalloff}'s derived distance. `generate.ts`'s height kernel calls {@link flattenedHeightAt} in
// place of a bare `heightAt` at the vertex and its four finite-difference neighbours, so the emitted
// normal reflects the flattened surface too ("affected-region remesh" — every reseed/regenerate
// re-dispatches the whole kernel, so there's no separate patch step).
//
// Stage 8 (2026-08-18, road profile smoothing): the road's own *longitudinal* target height used to be a
// bare `heightAt` at the projected centreline point — every noise octave at the 4 m vertex scale rode
// straight into the road surface. `terrain/profile.ts`'s {@link buildPolylineProfile} now pre-samples,
// low-passes, and grade-limits each polyline's own elevation once per `setNetwork` call; `networkCore`
// interpolates between the resulting control points' stored heights by the same clamped projection
// parameter `t` it already computes, instead of re-sampling `heightAt` at every query point.
//
// Two independently-derived cosine-ease forms (the same split `overlay/document.ts`/`overlay/rasterize.ts`
// use for distance): {@link flattenHeight} is the CPU reference `flatten.test.ts` pins directly;
// {@link flattenHeightGpu} is TGSL's own re-authoring of the
// same formula (it *can't* call the CPU one — TGSL has no FFI into plain JS), checked against it by a
// randomized differential test.

import { Compute, type State } from "@dylanebert/shallot";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import type { StrokeDocument } from "../overlay/document";
import { SPACING } from "./grid";
import { heightAt, makePermutation } from "./noise";
import { buildPolylineProfile, heightAtCpu } from "./profile";

// FALLOFF is no longer `= SPACING` (stage 6's choice, derived only from a capture probe's own grid
// alignment, not a road-geometry argument). Once the profile is smoothed the cut depth at a given point
// can grow past what a fixed 4 m band eases gracefully — a deep cut eased back over 4 m reads as a cliff
// wall, not a shoulder. {@link computeFalloff} re-derives it per network, from the network's own measured
// cut depth (the largest |natural - target| the flatten pipeline actually produces this reseed) and a
// cited side-slope limit, so the transition is only ever as wide as the deepest cut demands.
//
// AASHTO's Roadside Design Guide draws the line between a "traversable, non-recoverable" and a "critical,
// non-traversable" roadside slope at 1V:3H (~33%) — steeper reads as a cliff face to a driver leaving the
// shoulder. That's the target slope for the flatten transition's own *steepest* point, not its average:
// the cosine ease `0.5 - 0.5·cos(π·t)` has derivative `0.5·π·sin(π·t)`, peaking at `t = 0.5` (the
// transition's midpoint) at `π/2` — so a cut of depth `D` eased over a falloff distance `F` peaks at slope
// `D·(π/2)/F`. Solving for `F` at the side-slope limit gives the derivation below.
export const SIDE_SLOPE_LIMIT = 1 / 3; // AASHTO Roadside Design Guide, 1V:3H, rise/run

// Stage 11 (2026-08-18): the previous floor, a bare `SPACING`, was one quad wide — the mesh only carries a
// height *value* at each vertex and linearly interpolates between them, so a falloff one quad wide has
// exactly one interior region and reads as a straight-line ramp between two heights, not a cosine. That's
// the root cause stage 9 traced the metre-scale staircase to.
//
// How many vertices does the transition need before its piecewise-linear reconstruction can even *look*
// like the cosine ease rather than a plain ramp? Any 2-point chord is a straight line by definition, so a
// single segment across a curved arc can never register that the arc bows away from its own endpoints —
// revealing an arc's curvature needs an interior sample point splitting it into (at least) two
// sub-segments. The ease `0.5 - 0.5·cos(π·t)` is exactly two such arcs, concave on `t ∈ [0, 0.5]` and
// convex on `t ∈ [0.5, 1]`, meeting at the single inflection its own point symmetry locates at `t = 0.5`
// (`ease(t) + ease(1 - t) = 1` for every `t`, since `cos(π·t)` and `cos(π - π·t) = -cos(π·t)` cancel —
// which is also *why* a coarser, 2-segment sampling fails outright: its only interior point sits exactly
// on that symmetric midpoint, where `ease(0.5) = 0.5` lands precisely on the endpoint-to-endpoint chord,
// so both segments measure the same slope and the reconstruction is indistinguishable from a bare linear
// ramp — proven directly, not just for this cosine, by the identity above).
//
// Applying "an arc needs an interior sample to show it isn't straight" to each of the two monotone arcs
// separately — not just to the whole curve — takes two sub-segments per arc, four total. `flatten.test.ts`
// pins this as an observable property of {@link flattenHeight}'s own sampled output (each arc's sampled
// slopes differ from each other), not the arithmetic restated.
export const FALLOFF_SAMPLE_SEGMENTS = 4;

// The stage's live handover (spec Approach, stage 11): a wider falloff is a visible look change — roads
// sit in visibly wider cuttings — and that width is a taste call inside a locked technique
// (`taste.md` row 2), the same shape as stage 8's smoothing radius. `scale` is the live control's own
// entry point: it multiplies the *result* of the two derived constraints above, never substitutes for
// either, so `MIN_FALLOFF_SCALE = 1` (`terrain.ts`'s `setFalloffScale` clamp) can never narrow the
// transition below what either constraint demands — only widen it further, which is the only direction a
// taste call over this parameter has room to move without contradicting the derivations.
export const MIN_FALLOFF_SCALE = 1;
export const MAX_FALLOFF_SCALE = 3; // an arbitrary live-control ceiling, not a derived bound — 3x the
// technique's own derived width is well past where a cutting reads as a road rather than a valley
export const DEFAULT_FALLOFF_SCALE = 1; // ships at the derived width itself, not a verdict

/** the falloff distance (metres) whose cosine-ease transition peaks at exactly {@link SIDE_SLOPE_LIMIT}
 *  for a cut of `cutDepth` metres — floored at the max of two independently justified constraints on the
 *  same quantity: {@link FALLOFF_SAMPLE_SEGMENTS} × {@link SPACING} (a transition narrower than that can't
 *  be resolved by the heightfield's own piecewise-linear reconstruction, regardless of what the AASHTO
 *  slope term asks for) and the AASHTO side-slope term below (the slope a driver survives). AASHTO sets
 *  the width the road demands; sampling sets the width the mesh can represent — whichever is wider wins.
 *  `scale` is the live handover's own multiplier on top of both, defaulting to 1 (untouched).
 * @example computeFalloff(0) // FALLOFF_SAMPLE_SEGMENTS * SPACING — no cut, the sampling floor wins
 * @example computeFalloff(40) // (Math.PI / 2) * 40 / SIDE_SLOPE_LIMIT, well past either floor */
export function computeFalloff(cutDepth: number, scale = 1): number {
    return (
        Math.max(FALLOFF_SAMPLE_SEGMENTS * SPACING, ((Math.PI / 2) * cutDepth) / SIDE_SLOPE_LIMIT) *
        scale
    );
}

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
 *  compare the two forms with no device. */
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

/** one flattened profile sub-segment, GPU-side — finer than the road's own two endpoints
 *  (`terrain/profile.ts`'s `buildPolylineProfile` resamples every polyline at <= vertex-spacing arc
 *  length), each carrying its own smoothed, grade-limited elevation at `aHeight`/`bHeight` so
 *  `networkCore` interpolates between them instead of re-sampling `heightAt`. */
const NetworkSegment = d.struct({
    a: d.vec2f,
    b: d.vec2f,
    halfWidth: d.f32,
    aHeight: d.f32,
    bHeight: d.f32,
});
/** one polygon stamp's vertex span within the shared `polyVerts` buffer, plus its precomputed centroid —
 *  the single flat target height under the whole footprint (a real carpark is one flat plane, not a
 *  cross-section following its own nearest edge, which is what a per-edge target would produce). No
 *  profile: a carpark has no length to smooth along, so its target still samples `heightAt` directly. */
const NetworkPolygon = d.struct({ start: d.u32, count: d.u32, centroid: d.vec2f });

/** `falloff` is {@link computeFalloff}'s output for the current network — re-derived, not a compile-time
 *  constant, since it depends on the actual cut depth this reseed produced. */
const NetworkParams = d.struct({ segmentCount: d.u32, polygonCount: d.u32, falloff: d.f32 });

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
 *  from `overlay/document.ts`'s cross-product form) minus half-width; target is the segment's own
 *  pre-smoothed profile height, linearly interpolated by the same clamped projection parameter `t` the
 *  distance calculation already computes — never a fresh `heightAt` sample at the projected point (stage
 *  8's fix: that was every noise octave riding straight into the road surface). Polygons: ray-cast winding
 *  for the sign, nearest-edge distance for the magnitude (mirrors `rasterize.ts`'s `polygonDistanceGpu`);
 *  target is the natural height at the polygon's own centroid — a carpark is one flat plane, no profile to
 *  smooth. */
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
            bestTarget = std.mix(seg.aHeight, seg.bHeight, t);
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
 *  primitive's core target via {@link flattenHeightGpu}'s cosine ease, over the network's own
 *  per-reseed {@link computeFalloff} distance (`networkLayout.$.params.falloff`, written by `setNetwork`).
 *  `generate.ts`'s height kernel calls this in place of a bare `heightAt`, at the vertex and its four
 *  finite-difference neighbours alike, so the emitted normal reflects the flattened surface. An empty
 *  network (no segments, no polygons) leaves `bestCore` at its f32-max sentinel, always `>= falloff`, so
 *  this degrades to plain `heightAt`. */
export const flattenedHeightAt = tgpu.fn(
    [d.f32, d.f32],
    d.f32,
)((x, z) => {
    "use gpu";
    const natural = heightAt(x, z);
    const core = networkCore(x, z);
    return flattenHeightGpu(
        natural,
        core.targetHeight,
        core.coreDist,
        networkLayout.$.params.falloff,
    );
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

/** one flattened profile sub-segment — a `flatten.ts`-local extension of `overlay/document.ts`'s
 *  `Segment` shape with the two elevation control-point heights `networkCore` interpolates between. */
export interface ProfileSegment {
    readonly ax: number;
    readonly az: number;
    readonly bx: number;
    readonly bz: number;
    readonly halfWidth: number;
    readonly aHeight: number;
    readonly bHeight: number;
}

/** {@link buildNetworkGeometry}'s output: the GPU-bound sub-segment list plus the measured cut depth
 *  {@link computeFalloff} derives the falloff from. */
export interface NetworkGeometry {
    readonly segments: readonly ProfileSegment[];
    readonly cutDepth: number;
}

/**
 * pure CPU builder: every polyline in `doc`, resampled into its own smoothed-profile sub-segments
 * (`terrain/profile.ts`'s `buildPolylineProfile`), plus the measured cut depth — the largest
 * `|natural - target|` over every profile point this network actually produces, the real input
 * {@link computeFalloff} re-derives the falloff from (not a network-independent worst case). Deliberately
 * builds each polyline directly from `doc.polylines` rather than reusing `overlay/document.ts`'s
 * `flattenSegments`: that helper flattens a polyline to its own bare endpoint-to-endpoint segments, which
 * carry no per-point elevation to interpolate — this needs the finer, height-bearing resampling instead.
 * Device-free (`setNetwork` below is the only GPU-touching caller), so `flatten.test.ts` pins the
 * profile-to-segment marshaling and the cut-depth measurement without a device — the split
 * `overlay/document.ts` uses for its own geometry math.
 */
export function buildNetworkGeometry(
    doc: StrokeDocument,
    seed: number,
    smoothRadius: number,
): NetworkGeometry {
    const perm = makePermutation(seed);
    const segments: ProfileSegment[] = [];
    let cutDepth = 0;
    for (const line of doc.polylines) {
        const profile = buildPolylineProfile(line.points, perm, smoothRadius);
        for (let i = 0; i < profile.length - 1; i++) {
            const a = profile[i];
            const b = profile[i + 1];
            segments.push({
                ax: a.x,
                az: a.z,
                bx: b.x,
                bz: b.z,
                halfWidth: line.halfWidth,
                aHeight: a.height,
                bHeight: b.height,
            });
        }
        for (const p of profile) {
            cutDepth = Math.max(cutDepth, Math.abs(heightAtCpu(p.x, p.z, perm) - p.height));
        }
    }
    return { segments, cutDepth };
}

/** (re)build the network's GPU geometry buffers + bind group from `doc` at `seed` (the same seed the
 *  height kernel's own permutation buffer uses, so the profile's natural samples and the falloff
 *  terminus's `heightAt` agree) with `smoothRadius`'s longitudinal profile smoothing — called once at warm
 *  and again on every reseed or smoothing-strength change (`terrain.ts`'s `regenerate`/`setSmoothRadius`),
 *  since a new network or radius produces a different segment list and falloff. Mirrors
 *  `overlay/rasterize.ts`'s per-tile buffer-rebuild shape, but persists across the height kernel's own
 *  dispatches instead of being rebuilt per redraw — the kernel dispatches once per reseed, not once per
 *  throttled tile. `falloffScale` is the live handover's own multiplier ({@link computeFalloff}'s `scale`
 *  argument, `terrain.ts`'s `setFalloffScale`) — defaults to 1 (untouched) for every caller that doesn't
 *  pass one. */
export function setNetwork(
    doc: StrokeDocument,
    seed: number,
    smoothRadius: number,
    falloffScale = 1,
): void {
    teardownNetworkBuffers();
    const { device, root } = Compute;

    const { segments, cutDepth } = buildNetworkGeometry(doc, seed, smoothRadius);
    const falloff = computeFalloff(cutDepth, falloffScale);
    const segmentsData: readonly ProfileSegment[] =
        segments.length > 0
            ? segments
            : [{ ax: 0, az: 0, bx: 0, bz: 0, halfWidth: 0, aHeight: 0, bHeight: 0 }];
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
            aHeight: s.aHeight,
            bHeight: s.bHeight,
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
    paramsBuf.write({ segmentCount: segments.length, polygonCount: doc.polygons.length, falloff });

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
