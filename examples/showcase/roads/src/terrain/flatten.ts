// Edit-time heightfield flattening under the road network — the spec's Locked decision's "second half of
// the anti-decal argument" (UE Landscape Splines' "Apply Splines to Landscape" semantics): the ground is
// flattened *across* the road's width, cosine-eased back out to unmodified terrain over {@link
// computeFalloff}'s derived distance. `generate.ts`'s height kernel calls {@link flattenedHeightAt} in
// place of a bare `heightAt` at the vertex and its four finite-difference neighbours, so the emitted
// normal reflects the flattened surface too ("affected-region remesh" — every reseed/regenerate
// re-dispatches the whole kernel, so there's no separate patch step).
//
// Stage 8 (2026-08-18, road profile linearization): the road's own *longitudinal* target height used to be a
// bare `heightAt` at the projected centreline point — every noise octave at the 4 m vertex scale rode
// straight into the road surface. `terrain/profile.ts`'s {@link buildPolylineProfile} now returns one
// straight chord per polyline; `networkCore` interpolates between its endpoint heights by the same
// clamped projection parameter `t` it already computes, instead of re-sampling `heightAt` at every query
// point.
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
import { buildPolylineProfile, heightAtCpu, PROFILE_STEP } from "./profile";

// FALLOFF is no longer `= SPACING` (stage 6's choice, derived only from a capture probe's own grid
// alignment, not a road-geometry argument). Once the profile is the straight chord the cut depth at a given point
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

// Stage 15 (2026-08-19, the reconstruction-axis fix `flatness.ts` diagnosed): a triangle straddling the
// footprint edge can have a corner *outside* the old `halfWidth`-radius flat core, eased toward off-road
// terrain that varies by metres over one grid cell — the mesh reads that blend at a point whose analytic
// position is still inside the road. Widening the flat core (the `coreDist <= 0` region `flattenHeight`
// hands `targetHeight` outright) by a grid cell's own diagonal guarantees every vertex that can support a
// footprint-intersecting triangle sits inside it: a cell is `SPACING` square, so the farthest any of a
// straddling triangle's corners can sit from the footprint edge is that cell's diagonal, `√2·SPACING`.
// Cell geometry, not a fitted number.
export const FLAT_CORE_MARGIN = Math.SQRT2 * SPACING;

/** the falloff distance (metres) whose cosine-ease transition peaks at exactly {@link SIDE_SLOPE_LIMIT}
 *  for a cut of `cutDepth` metres — floored at {@link SPACING}, since a transition narrower than the mesh's
 *  own vertex spacing can't be resolved by the heightfield regardless of what the formula asks for.
 * @example computeFalloff(0) // SPACING — no cut, the floor wins
 * @example computeFalloff(4) // (Math.PI / 2) * 4 / SIDE_SLOPE_LIMIT, well past the floor */
export function computeFalloff(cutDepth: number): number {
    return Math.max(SPACING, ((Math.PI / 2) * cutDepth) / SIDE_SLOPE_LIMIT);
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

/** one flattened profile sub-segment, GPU-side — the road's own two chord endpoints
 *  (`terrain/profile.ts`'s `buildPolylineProfile` returns one straight chord per polyline), each
 *  carrying its own natural elevation at `aHeight`/`bHeight` so
 *  `networkCore` interpolates between them instead of re-sampling `heightAt`. `road` is the owning
 *  polyline's own index in `doc.polylines` (stage 15) — segments are emitted contiguously per road
 *  (`buildNetworkGeometry`'s own per-line loop), which `networkCore` relies on to fold a road's own chain
 *  of sub-segments (continuous by construction — consecutive sub-segments share an endpoint height) down
 *  to *one* nearest-point contribution before blending across distinct primitives, rather than blending
 *  every fine sub-segment as if it were its own primitive. */
const NetworkSegment = d.struct({
    a: d.vec2f,
    b: d.vec2f,
    halfWidth: d.f32,
    aHeight: d.f32,
    bHeight: d.f32,
    road: d.u32,
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

/** the network's own core boundary distance (the union of every primitive's, still a plain nearest-SDF
 *  min — continuous by construction) + a **blended** target height at (px, pz) — the GPU-side geometry
 *  half of the flatten pipeline. Segments: clamped-projection distance to the core boundary (the same
 *  clamped form `overlay/rasterize.ts`'s `segmentDistanceGpu` uses — an independent derivation from
 *  `overlay/document.ts`'s cross-product form) minus half-width, widened by {@link FLAT_CORE_MARGIN};
 *  target is the segment's own chord profile height, linearly interpolated by the same clamped
 *  projection parameter `t` the distance calculation already computes — never a fresh `heightAt` sample at
 *  the projected point (stage 8's fix: that was every noise octave riding straight into the road surface).
 *  Polygons: ray-cast winding for the sign, nearest-edge distance for the magnitude (mirrors
 *  `rasterize.ts`'s `polygonDistanceGpu`), same margin; target is the natural height at the polygon's own
 *  centroid — a carpark is one flat plane, no profile to smooth.
 *
 *  Stage 15: `bestTarget` used to be the *nearest* primitive's own target, picked by a hard `core <
 *  bestCore` switch — continuous in `bestCore` (a min of SDFs) but discontinuous in the target it carries,
 *  since crossing the bisector between two primitives flips which one's target wins outright. This now
 *  blends every *primitive*'s target by the same cosine-ease weight {@link flattenHeightGpu} already uses
 *  to fade a single primitive's target toward natural (`1 - ease(coreDist / falloff)`, clamped to `[0,
 *  1]`) — a primitive fully inside its own core contributes weight 1, one past its own falloff contributes
 *  0, so away from any overlap exactly one weight survives and this reduces to the old nearest-primitive
 *  target exactly; where two primitives' falloff bands overlap (a bisector, a junction), both contribute
 *  and the target cross-fades instead of switching.
 *
 *  Stage 15b: each primitive's blend weight is multiplied by a relative-depth suppression factor
 *  `1 − ease(clamp((core_i − bestCore) / FLAT_CORE_MARGIN, 0, 1))` — a primitive at the same depth as
 *  the nearest (core_i === bestCore) keeps full weight, one deeper by ≥ FLAT_CORE_MARGIN is fully
 *  suppressed. This requires `bestCore` to be known before any weight is computed, so the function is now
 *  two passes: pass 1 finds `bestCore` (the global minimum core across all sub-segments and polygons — no
 *  road grouping needed, since the min of per-road minima equals the min of all sub-segments); pass 2
 *  computes the weighted blend, each weight carrying the depth factor. Contamination is exactly zero
 *  outside designed junction zones by construction, not by sample — the host's target wins outright
 *  wherever every other primitive's core exceeds the nearest's by ≥ the band.
 *
 *  A *primitive* is a whole road or a whole carpark, never one fine profile sub-segment: a road's own
 *  chain of sub-segments is already continuous by construction (`buildNetworkGeometry`'s per-point
 *  interpolation chains `aHeight`/`bHeight` endpoint to endpoint), so blending across sub-segments the way
 *  distinct primitives blend would reintroduce noise along a single, already-smooth road — measured
 *  directly (2026-08-19): treating every sub-segment as its own blend contributor left ~1078 cross-section
 *  violations up to 0.44 m, entirely on interior stretches of a single curving road with no other primitive
 *  within its falloff. `NetworkSegment.road` groups sub-segments back into their own road: the loop below
 *  takes a hard nearest-sub-segment reduction *within* a road (`roadBestCore`/`roadBestTarget`, flushed
 *  whenever `seg.road` changes — segments are emitted contiguously per road, `NetworkSegment`'s own
 *  docstring), then folds each road's single reduced (core, target) into the cross-primitive blend the same
 *  way a polygon's own single (core, target) does. */
const networkCore = tgpu.fn(
    [d.f32, d.f32],
    NetworkCore,
)((px, pz) => {
    "use gpu";
    const falloff = networkLayout.$.params.falloff;

    // Pass 1: find bestCore — the global minimum core distance across all sub-segments and polygons.
    // No road grouping needed: the min of per-road minima equals the min of all sub-segments.
    let bestCore = d.f32(3.402823e38);
    let si = d.u32(0);
    for (; si < networkLayout.$.params.segmentCount; si = si + d.u32(1)) {
        const seg = networkLayout.$.segments[si];
        const abx = seg.b.x - seg.a.x;
        const abz = seg.b.y - seg.a.y;
        const apx = px - seg.a.x;
        const apz = pz - seg.a.y;
        const dd = abx * abx + abz * abz;
        let t = d.f32(0);
        if (dd > 0) t = std.clamp((apx * abx + apz * abz) / dd, 0, 1);
        const cx = seg.a.x + t * abx;
        const cz = seg.a.y + t * abz;
        const core =
            std.distance(d.vec2f(px, pz), d.vec2f(cx, cz)) -
            (seg.halfWidth + d.f32(FLAT_CORE_MARGIN));
        if (core < bestCore) bestCore = core;
    }
    let pi = d.u32(0);
    for (; pi < networkLayout.$.params.polygonCount; pi = pi + d.u32(1)) {
        const poly = networkLayout.$.polygons[pi];
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
        const core = std.select(nearestEdge, -nearestEdge, inside) - d.f32(FLAT_CORE_MARGIN);
        if (core < bestCore) bestCore = core;
    }

    // Pass 2: weighted blend across primitives, each weight multiplied by the relative-depth suppression
    // factor 1 − ease(clamp((core_i − bestCore) / FLAT_CORE_MARGIN, 0, 1)).
    let sumWeight = d.f32(0);
    let sumWeightedTarget = d.f32(0);

    let haveRoad = false;
    let curRoad = d.u32(0);
    let roadBestCore = d.f32(3.402823e38);
    let roadBestTarget = d.f32(0);

    let i = d.u32(0);
    for (; i < networkLayout.$.params.segmentCount; i = i + d.u32(1)) {
        const seg = networkLayout.$.segments[i];
        if (haveRoad && seg.road !== curRoad) {
            const rt = std.clamp(roadBestCore / falloff, 0, 1);
            const rw = d.f32(1) - (d.f32(0.5) - d.f32(0.5) * std.cos(d.f32(Math.PI) * rt));
            const dt = std.clamp((roadBestCore - bestCore) / d.f32(FLAT_CORE_MARGIN), 0, 1);
            const df = d.f32(1) - (d.f32(0.5) - d.f32(0.5) * std.cos(d.f32(Math.PI) * dt));
            const w = rw * df;
            sumWeight = sumWeight + w;
            sumWeightedTarget = sumWeightedTarget + w * roadBestTarget;
            roadBestCore = d.f32(3.402823e38);
            roadBestTarget = d.f32(0);
        }
        curRoad = seg.road;
        haveRoad = true;

        const abx = seg.b.x - seg.a.x;
        const abz = seg.b.y - seg.a.y;
        const apx = px - seg.a.x;
        const apz = pz - seg.a.y;
        const dd = abx * abx + abz * abz;
        let t = d.f32(0);
        if (dd > 0) t = std.clamp((apx * abx + apz * abz) / dd, 0, 1);
        const cx = seg.a.x + t * abx;
        const cz = seg.a.y + t * abz;
        const core =
            std.distance(d.vec2f(px, pz), d.vec2f(cx, cz)) -
            (seg.halfWidth + d.f32(FLAT_CORE_MARGIN));
        if (core < roadBestCore) {
            roadBestCore = core;
            roadBestTarget = std.mix(seg.aHeight, seg.bHeight, t);
        }
    }
    if (haveRoad) {
        const rt = std.clamp(roadBestCore / falloff, 0, 1);
        const rw = d.f32(1) - (d.f32(0.5) - d.f32(0.5) * std.cos(d.f32(Math.PI) * rt));
        const dt = std.clamp((roadBestCore - bestCore) / d.f32(FLAT_CORE_MARGIN), 0, 1);
        const df = d.f32(1) - (d.f32(0.5) - d.f32(0.5) * std.cos(d.f32(Math.PI) * dt));
        const w = rw * df;
        sumWeight = sumWeight + w;
        sumWeightedTarget = sumWeightedTarget + w * roadBestTarget;
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
        const core = std.select(nearestEdge, -nearestEdge, inside) - d.f32(FLAT_CORE_MARGIN);
        const target = heightAt(poly.centroid.x, poly.centroid.y);
        const pt = std.clamp(core / falloff, 0, 1);
        const pw = d.f32(1) - (d.f32(0.5) - d.f32(0.5) * std.cos(d.f32(Math.PI) * pt));
        const dt = std.clamp((core - bestCore) / d.f32(FLAT_CORE_MARGIN), 0, 1);
        const df = d.f32(1) - (d.f32(0.5) - d.f32(0.5) * std.cos(d.f32(Math.PI) * dt));
        const w = pw * df;
        sumWeight = sumWeight + w;
        sumWeightedTarget = sumWeightedTarget + w * target;
    }

    const bestTarget = std.select(d.f32(0), sumWeightedTarget / sumWeight, sumWeight > 0);
    return NetworkCore({ coreDist: bestCore, targetHeight: bestTarget });
});

/** the flattened height at world (x, z): natural terrain height blended toward {@link networkCore}'s own
 *  (now-blended) target via {@link flattenHeightGpu}'s cosine ease, over the network's own
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
 *  `Segment` shape with the two elevation control-point heights `networkCore` interpolates between, plus
 *  `road` (stage 15): the owning polyline's own index in `doc.polylines`, {@link NetworkSegment}'s CPU
 *  twin. */
export interface ProfileSegment {
    readonly ax: number;
    readonly az: number;
    readonly bx: number;
    readonly bz: number;
    readonly halfWidth: number;
    readonly aHeight: number;
    readonly bHeight: number;
    readonly road: number;
}

/** {@link buildNetworkGeometry}'s output: the GPU-bound sub-segment list plus the measured cut depth
 *  {@link computeFalloff} derives the falloff from. */
export interface NetworkGeometry {
    readonly segments: readonly ProfileSegment[];
    readonly cutDepth: number;
}

/**
 * pure CPU builder: every polyline in `doc`, built into its own straight-chord profile segments
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
export function buildNetworkGeometry(doc: StrokeDocument, seed: number): NetworkGeometry {
    const perm = makePermutation(seed);
    const segments: ProfileSegment[] = [];
    let cutDepth = 0;
    for (const [road, line] of doc.polylines.entries()) {
        const profile = buildPolylineProfile(line.points, perm);
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
                road,
            });
        }
        // cutDepth measured along the chord: the chord's target height at the endpoints equals the
        // natural height there (by construction), so the profile points themselves read zero — the
        // actual cut lives *between* the endpoints, where natural terrain deviates from the straight
        // chord. Sample at <= PROFILE_STEP arc-length increments (the same resolution the pre-chord
        // profile used) and measure |natural - chord_target| at each, the same formula as today.
        const first = line.points[0];
        const last = line.points[line.points.length - 1];
        const chordLen = Math.hypot(last[0] - first[0], last[1] - first[1]);
        const aH = profile[0].height;
        const bH = profile[profile.length - 1].height;
        const n = Math.max(1, Math.ceil(chordLen / PROFILE_STEP));
        for (let s = 0; s <= n; s++) {
            const t = s / n;
            const x = first[0] + (last[0] - first[0]) * t;
            const z = first[1] + (last[1] - first[1]) * t;
            const chordHeight = aH + (bH - aH) * t;
            cutDepth = Math.max(cutDepth, Math.abs(heightAtCpu(x, z, perm) - chordHeight));
        }
    }
    return { segments, cutDepth };
}

/** (re)build the network's GPU geometry buffers + bind group from `doc` at `seed` (the same seed the
 *  height kernel's own permutation buffer uses, so the profile's natural samples and the falloff
 *  terminus's `heightAt` agree) — called once at warm and again on every reseed
 *  (`terrain.ts`'s `regenerate`), since a new network produces a different segment list and falloff. Mirrors
 *  `overlay/rasterize.ts`'s per-tile buffer-rebuild shape, but persists across the height kernel's own
 *  dispatches instead of being rebuilt per redraw — the kernel dispatches once per reseed, not once per
 *  throttled tile. */
export function setNetwork(doc: StrokeDocument, seed: number): void {
    teardownNetworkBuffers();
    const { device, root } = Compute;

    const { segments, cutDepth } = buildNetworkGeometry(doc, seed);
    const falloff = computeFalloff(cutDepth);
    const segmentsData: readonly ProfileSegment[] =
        segments.length > 0
            ? segments
            : [{ ax: 0, az: 0, bx: 0, bz: 0, halfWidth: 0, aHeight: 0, bHeight: 0, road: 0 }];
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
            road: s.road,
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
