// The surface-flatness oracle — the spec's Validation "Surface flatness in the corridor — exactly zero,
// unconditional", the unit's defect gate. It measures the mesh's piecewise-linear *reconstruction* across
// the footprint edge, not the boundary's own width or position.
//
// A vertex-only check is insufficient by construction: footprint vertices already carry the flattened
// chord profile (`terrain/flatten.ts`'s `networkCore` interpolates `mix(aHeight, bHeight, t)`), so
// away from junctions they're flat to a few centimetres — a vertex-only oracle would read green over the
// live defect. The predicted defect lives in the *reconstruction*: a triangle straddling the footprint
// edge has one or more corners *outside* the flattened core, cosine-eased toward off-road terrain that can
// vary by metres over one 4 m cell — so a point sampled right at the edge, inside a triangle whose other
// corners sit outside it, reads a height blended toward that off-road variation, not the flat corridor its
// own analytic position implies. Reading height from `capture.ts`'s `meshHeightAt` (a two-triangle-per-quad
// reconstruction over a raw vertex buffer) rather than the continuous `flattenHeight` function is what
// makes this oracle able to see that — the continuous function has no triangles to straddle.
//
// Three vertex-buffer sources feed the one oracle below (`checkSurfaceFlatness`), which only ever touches a
// `Uint32Array` + `meshHeightAt` — never which device produced it:
//   - {@link buildBandedLatticeVertices} — the same full-resolution lattice with **only** the vertices
//     within one cell diagonal of the chords filled, so it is valid for a footprint-interior-only reader
//     and for nothing else (its own docblock carries the derivation and the caveat). It is the oracle's
//     main substrate: six default-suite arms in `flatness.test.ts` build from it (the real-generator
//     exactness pair, the banded-vs-full null control pair, the synthetic 30°-network exactness pair),
//     plus `edit.test.ts`'s corridor-flatness sentinel and the whole 200-drag corpus in
//     `editCorridor.tier.ts` — both of those through `dragCorpus.ts`'s `scanDrag`.
//   - {@link buildDeviceFreeVertices} — a CPU-only mirror of `terrain/generate.ts`'s height-kernel loop
//     (no GPU dispatch, the full {@link buildLatticeVertices} fill at `SPACING`/`CELLS`). It feeds
//     `checkSurfaceFlatness` from one default-suite `describe` only — the shipped-pipeline reading
//     (`flatness.test.ts`, arm i), whose `result` two tests then read: the first *reports* the readings
//     (a `console.log`, no `expect` — deliberate, Blocker 3's "no absolute amplitude bound is
//     admissible", and labelled in its own body), the second asserts every violation sits inside the road
//     footprint. So the substrate feeds one reading and one assertion, not one arm — and is otherwise
//     {@link reconstructionAgreement}'s own CPU side, where every vertex is needed because the comparison
//     is against a real readback.
//   - the real `readVertices()` (`terrain/terrain.ts`) — the device arms inside `bun run gate`, two of
//     them: `gate.ts`'s `reconstructionAgreement` pins the CPU builder's *fidelity* against the real GPU
//     output, and its `surface-flatness-property-on-device` check re-asserts the *property* device-side —
//     the same unconditional exact zero the default suite asserts, over real `readVertices()`.
//
// What this oracle cannot see (name the blind axes, `checks.md`'s granularity clause):
//   - albedo registration — the overlay's own texel/coverage crispness is a separate render-half property
//     (the spec's "Fs composite" criterion, `terrain.ts`'s fs), untouched by anything read here.
//   - shading normals — the finite-difference normal `generate.ts`'s kernel emits alongside height is
//     never read; a normal can look wrong while every height sampled here reads within tolerance.
//   - everything outside the road footprint — `documentDistance(x, z, doc) > 0` terrain is natural and
//     unconstrained by design; this oracle only walks centreline + edge lines *inside* a footprint.
//   - reseed staleness — this always samples the *live* document's own current geometry; a stale
//     atlas/dirty-tile residue leaves no trace on the heightfield this oracle reads.
//   - the segment endpoint cap — every sampled line excises an arc-length margin of `halfWidth +
//     √2·SPACING` (9.66 m on this network) at *both* ends of every segment (`endpointMargin`, below):
//     past that margin, `networkCoreCpu`'s own nearest-*point* clamp is a different geometric case from
//     the nearest-*line* interior this oracle checks, and the reconstruction reaches one cell diagonal
//     past its sample point. A green run says nothing about the geometry within that margin of any road
//     endpoint. Both terms are treatment-free: `halfWidth` is document geometry, `SPACING` is the mesh
//     constant. This cap is the criterion's *only* remaining exclusion — the junction carve-out, the
//     partition invariant, and the exclusion fraction are all gone.
//
// The oracle's sampling window is the per-segment endpoint cap (`halfWidth + √2·SPACING` at each
// end, derived from the affine region and the lattice — see `endpointMargin` below); beyond that, the
// blind axes (albedo, normals, off-footprint, reseed staleness) are out-of-instrument by construction,
// and the footprint interior is never sampled (the oracle only walks centreline + edge lines inside a
// footprint). The criterion is unconditional — no junction zone, no partition, no exclusion fraction.

import { encodePos } from "@dylanebert/shallot/utils/core";
import * as d from "typegpu/data";
import { meshHeightAt } from "./capture";
import type { StrokeDocument } from "./overlay/document";
import {
    buildNetworkGeometry,
    computeFalloff,
    FLAT_CORE_MARGIN,
    flattenHeight,
    type ProfileSegment,
} from "./terrain/flatten";
import { TERRAIN_QUANT } from "./terrain/generate";
import { CELLS, SPACING } from "./terrain/grid";
import { makePermutation } from "./terrain/noise";
import { heightAtCpu, MAX_GRADE } from "./terrain/profile";

// --- derived window + tolerance — no candidate treatment (falloff) appears in any of these,
// per the spec's own non-coupling requirement: a criterion whose window or threshold is a function of the
// parameter under test is not a differential over that parameter.

/** the spec's own longitudinal sample spacing bound: no coarser than a quarter of the mesh's own vertex
 *  spacing, so no triangle edge along a sampled line is ever skipped between two adjacent samples. */
export const SAMPLE_STEP = SPACING / 4;

/** height-quantization noise on one decoded reconstruction sample — `TERRAIN_QUANT`'s own AABB divided by
 *  the unorm16 range, the same derivation `capture.test.ts`'s `meshHeightAt` oracle uses. */
const QUANT_TOL = TERRAIN_QUANT.posScale.y / 65535;

/** the longitudinal grade bound over a `ds`-metre step: the road-design grade limit (`MAX_GRADE`) plus two
 *  independent quantized height reads (one at each end of the step). Neither `MAX_GRADE`/`PROFILE_STEP`
 *  (road-design + mesh constants) nor `QUANT_TOL` (the codec's own AABB) depends on any candidate
 *  treatment. */
export function gradeBound(ds: number): number {
    return MAX_GRADE * ds + 2 * QUANT_TOL;
}

/** the cross-section bound: two footprint points at the same longitudinal station should read the same
 *  height up to two independent reconstructions' quantization noise. The flatten *field* is a pure
 *  function of station (`networkCoreCpu`'s own `t`-only interpolation), but the piecewise-linear
 *  *reconstruction* this oracle reads is not — a triangle straddling the footprint edge blends one
 *  corner's off-road height into a point whose analytic position is still inside the road, so two
 *  same-station footprint points can read different heights by mesh construction, not by blend design.
 *  This is the distinction Leg A (the field, exactly zero) and Leg B (the mesh,
 *  convergence) separate — a single absolute bound over the reconstruction reading is fitted whichever
 *  number it takes. */
export const CROSS_SECTION_TOL = 2 * QUANT_TOL;

/** how far inside the road edge the two edge lines sit — small relative to `SPACING` so the probe stays
 *  within the single grid cell the predicted defect straddles (the edge itself), rather than retreating
 *  into the core's interior where the defect can't reach. Not derived from any falloff/treatment
 *  quantity — a fixed fraction of the mesh's own cell size. */
export const EDGE_EPSILON = SPACING / 100;

// --- CPU-only lattice reconstruction — no GPU, no `@dylanebert/shallot/render/core` import, the
// default-suite arm's own substrate.

/** the network's own core distance (still a plain min — continuous) + a **blended** target height at
 *  (px, pz) — a plain-JS re-authoring of `terrain/flatten.ts`'s GPU `networkCore`, deliberately the
 *  *same* formula (not an independent derivation the way `overlay/rasterize.ts`'s distance math is):
 *  this function's job is to reconstruct exactly what the real mesh does, not to provide a second opinion
 *  on it — the property under test is the mesh's own triangle reconstruction, and the height *per vertex*
 *  has to match production's own `flattenedHeightAt` or the built lattice isn't the mesh this oracle is
 *  meant to read.
 *
 *  Each primitive's core distance is widened by `terrain/flatten.ts`'s {@link FLAT_CORE_MARGIN}
 *  (a grid cell's own diagonal, so every vertex that can support a footprint-intersecting triangle reads
 *  the flat plateau), and `bestTarget` is a weighted blend across every *primitive* (a whole road, never
 *  one fine profile sub-segment) whose falloff band reaches (px, pz) — weight `1 - ease(coreDist /
 *  falloff)`, the same cosine ease {@link flattenHeight} fades a single primitive's target toward natural
 *  with — rather than a hard `core < bestCore` switch, whose discontinuity across two primitives'
 *  bisector would be a visible step. A road's own sub-segments are reduced to one (core, target)
 *  pair by a hard nearest-sub-segment min *first* (`roadBest`, below) — they're already continuous by
 *  construction (chained endpoint to endpoint), so blending them as if they were separate primitives
 *  reintroduces noise along a single, already-smooth road: measured directly (2026-08-19), blending every
 *  sub-segment left ~1078 cross-section violations up to 0.44 m on interior stretches of one curving road
 *  with no other primitive nearby. The carpark-polygon primitive kind and its own ray-cast/nearest-edge
 *  leg are not part of this path — it mirrors `terrain/flatten.ts`'s GPU `networkCore`. */
export function networkCoreCpu(
    px: number,
    pz: number,
    segments: readonly ProfileSegment[],
    falloff: number,
): { coreDist: number; targetHeight: number } {
    // Build per-primitive (core, target) pairs and find bestCore (the global minimum core, returned
    // as coreDist).
    const roadBest = new Map<number, { core: number; target: number }>();
    for (const seg of segments) {
        const abx = seg.bx - seg.ax;
        const abz = seg.bz - seg.az;
        const apx = px - seg.ax;
        const apz = pz - seg.az;
        const dd = abx * abx + abz * abz;
        const t = dd > 0 ? Math.min(1, Math.max(0, (apx * abx + apz * abz) / dd)) : 0;
        const cx = seg.ax + t * abx;
        const cz = seg.az + t * abz;
        const core = Math.hypot(px - cx, pz - cz) - (seg.halfWidth + FLAT_CORE_MARGIN);
        const target = seg.aHeight + t * (seg.bHeight - seg.aHeight);
        const prev = roadBest.get(seg.road);
        if (!prev || core < prev.core) roadBest.set(seg.road, { core, target });
    }

    let bestCore = Number.POSITIVE_INFINITY;
    for (const best of roadBest.values()) {
        if (best.core < bestCore) bestCore = best.core;
    }

    // Weighted blend across primitives — each weight is the cosine-ease fade of the primitive's own
    // target toward natural (`1 - ease(core / falloff)`), with no depth suppression: non-overlapping
    // primitives never contend, so the blend carries no depth-suppression factor — there is nothing
    // left to suppress.
    let sumWeight = 0;
    let sumWeightedTarget = 0;
    for (const best of roadBest.values()) {
        const t = Math.min(1, Math.max(0, best.core / falloff));
        const weight = 1 - (0.5 - 0.5 * Math.cos(Math.PI * t));
        sumWeight += weight;
        sumWeightedTarget += weight * best.target;
    }

    const bestTarget = sumWeight > 0 ? sumWeightedTarget / sumWeight : 0;
    return { coreDist: bestCore, targetHeight: bestTarget };
}

/** the continuous flattened field at (px, pz) — {@link networkCoreCpu}'s blended target eased toward
 *  natural via {@link flattenHeight}, with no mesh at all. This is the analytic limit Leg A (spec
 *  Validation, 2026-08-19 second consult) samples: the field is exactly flat by construction
 *  (non-overlapping primitives never contend, so exactly one weight survives at every sampled
 *  point), so a non-zero reading is the blend's design, not the mesh's discretization. */
export function flattenFieldAt(
    px: number,
    pz: number,
    segments: readonly ProfileSegment[],
    falloff: number,
    naturalHeightAt: (x: number, z: number) => number,
): number {
    const natural = naturalHeightAt(px, pz);
    const { coreDist, targetHeight } = networkCoreCpu(px, pz, segments, falloff);
    return flattenHeight(natural, targetHeight, coreDist, falloff);
}

/** discretize the continuous flatten field (`segments`/`falloff`, `naturalHeightAt`) onto a
 *  `spacing`-metre, `cells`-quad lattice, encoded the same way `terrain/generate.ts`'s height kernel
 *  encodes its own vertex stream (`encodePos` against the live `TERRAIN_QUANT`) — the output is a
 *  `meshHeightAt`-compatible raw buffer regardless of resolution, so the same reconstruction function reads
 *  both the production-resolution lattice and one built at another `spacing`/`cells`: `flatness.test.ts`'s
 *  banded-vs-full null control calls this builder directly at `SPACING` and at `SPACING/2`, while {@link
 *  buildDeviceFreeVertices} — its only non-test caller — always builds at production resolution. Normal/UV words
 *  are left at zero — nothing downstream of this buffer reads them (`meshHeightAt` only decodes the
 *  position words). */
export function buildLatticeVertices(
    spacing: number,
    cells: number,
    segments: readonly ProfileSegment[],
    falloff: number,
    naturalHeightAt: (x: number, z: number) => number,
): Uint32Array {
    const verts = cells + 1;
    const half = cells / 2;
    const raw = new Uint32Array(verts * verts * 4);
    for (let iz = 0; iz < verts; iz++) {
        for (let ix = 0; ix < verts; ix++) {
            const x = (ix - half) * spacing;
            const z = (iz - half) * spacing;
            const natural = naturalHeightAt(x, z);
            const { coreDist, targetHeight } = networkCoreCpu(x, z, segments, falloff);
            const y = flattenHeight(natural, targetHeight, coreDist, falloff);
            const idx = (iz * verts + ix) * 4;
            const m = encodePos(d.vec3f(x, y, z), 0, TERRAIN_QUANT);
            raw[idx] = m.x;
            raw[idx + 1] = m.y;
        }
    }
    return raw;
}

/**
 * the same lattice as {@link buildLatticeVertices}, filled **only** inside a band of `halfWidth + √2 ·
 * spacing` metres around `segments`, every other vertex left zeroed.
 *
 * Valid only for a reader that samples strictly inside the road footprint — {@link
 * checkSurfaceFlatness} is the one such reader (it walks the centreline and the two edge lines inset
 * by {@link EDGE_EPSILON}, never off-road terrain). Anything that compares this buffer against a real
 * `readVertices()` readback needs every vertex and must keep using the full builder: {@link
 * buildDeviceFreeVertices} and `gate.ts`'s device arms therefore do not call this.
 *
 * Why `halfWidth + √2 · spacing` is the band's derivation and not a tuned number: a sampled point
 * sits at perpendicular offset `< halfWidth` from its segment, and `capture.ts`'s `meshHeightAt`
 * interpolates it over the three corners of one lattice triangle, displaced from the sample point by
 * `(-tx, -tz)`, `(1 - tx, -tz)` and `(-tx, 1 - tz)` cells (the upper triangle is the same by
 * symmetry). A displacement whose two components can carry opposite signs projects onto the chord
 * normal as much as `spacing · (|nx| + |nz|) <= √2 · spacing` — the *diagonal* of a cell, reached
 * when the chord runs at 45° to the lattice, so the band is one cell diagonal wide beyond the
 * footprint and not one cell. Along the chord the corners stay inside the segment's own station
 * range, since `sampleWindow` already excises {@link endpointMargin} (`halfWidth + √2·SPACING`) at
 * each end and `√2 · spacing <= √2 · SPACING`; so a used corner's distance *to the segment* is its
 * perpendicular offset, `< halfWidth + √2 · spacing`. The bound is attained in the limit, so it is
 * tight: `halfWidth + spacing` — one cell instead of its diagonal — already drops corners the oracle
 * interpolates from on a 45° chord, and a zeroed corner decodes to a height nowhere near the
 * corridor, so an over-narrowed band reads as violations rather than as a silent pass
 * (`flatness.test.ts`'s "the banded lattice reads what the full lattice reads" carries that red).
 */
export function buildBandedLatticeVertices(
    spacing: number,
    cells: number,
    segments: readonly ProfileSegment[],
    falloff: number,
    naturalHeightAt: (x: number, z: number) => number,
): Uint32Array {
    const verts = cells + 1;
    const half = cells / 2;
    const raw = new Uint32Array(verts * verts * 4);
    for (let iz = 0; iz < verts; iz++) {
        const z = (iz - half) * spacing;
        for (let ix = 0; ix < verts; ix++) {
            const x = (ix - half) * spacing;
            if (!withinBand(x, z, segments, spacing)) continue;
            const natural = naturalHeightAt(x, z);
            const { coreDist, targetHeight } = networkCoreCpu(x, z, segments, falloff);
            const y = flattenHeight(natural, targetHeight, coreDist, falloff);
            const idx = (iz * verts + ix) * 4;
            const m = encodePos(d.vec3f(x, y, z), 0, TERRAIN_QUANT);
            raw[idx] = m.x;
            raw[idx + 1] = m.y;
        }
    }
    return raw;
}

/** whether (px, pz) is within `seg.halfWidth + √2 · spacing` of any segment — the band
 *  {@link buildBandedLatticeVertices} fills, tested against the capsule (distance to the segment,
 *  endpoints included), not the infinite line. */
function withinBand(
    px: number,
    pz: number,
    segments: readonly ProfileSegment[],
    spacing: number,
): boolean {
    for (const seg of segments) {
        const abx = seg.bx - seg.ax;
        const abz = seg.bz - seg.az;
        const apx = px - seg.ax;
        const apz = pz - seg.az;
        const dd = abx * abx + abz * abz;
        const t = dd > 0 ? Math.min(1, Math.max(0, (apx * abx + apz * abz) / dd)) : 0;
        const dx = apx - t * abx;
        const dz = apz - t * abz;
        const reach = seg.halfWidth + Math.SQRT2 * spacing;
        if (dx * dx + dz * dz <= reach * reach) return true;
    }
    return false;
}

/** the production-shape reconstruction: `doc`'s own network flattened at `seed`, at the
 *  mesh's real `SPACING`/`CELLS` — a drop-in `readVertices()` substitute with no device, `flatness.test.ts`'s
 *  and `gate.ts`'s shared entry point. `flattenDoc` and `sampleDoc` (the caller's own footprint definition,
 *  `checkSurfaceFlatness`) are deliberately the same `doc` here; {@link buildLatticeVertices} lets a caller
 *  split them (a null-control arm: an empty `flattenDoc` with the real network's footprint still
 *  standing, `flatness.test.ts`'s "no-cut" arm). */
export function buildDeviceFreeVertices(flattenDoc: StrokeDocument, seed: number): Uint32Array {
    const perm = makePermutation(seed);
    const { segments, cutDepth } = buildNetworkGeometry(flattenDoc, seed);
    const falloff = computeFalloff(cutDepth);
    return buildLatticeVertices(SPACING, CELLS, segments, falloff, (x, z) =>
        heightAtCpu(x, z, perm),
    );
}

// --- the property: cross-section + longitudinal grade over the real reconstruction.

interface RoadFrame {
    readonly ax: number;
    readonly az: number;
    readonly ux: number;
    readonly uz: number;
    readonly len: number;
    readonly nx: number;
    readonly nz: number;
    readonly halfWidth: number;
}

/** every polyline segment in `doc`, resolved into the frame the sampler walks — every consecutive point
 *  pair, not just `doc.polylines[0]`'s first segment (unlike `boundaryAnchors.ts`'s single-road framing,
 *  this oracle covers the whole network, since the reconstruction defect is predicted on every road). */
function segmentFrames(doc: StrokeDocument): RoadFrame[] {
    const frames: RoadFrame[] = [];
    for (const line of doc.polylines) {
        for (let i = 0; i < line.points.length - 1; i++) {
            const [ax, az] = line.points[i];
            const [bx, bz] = line.points[i + 1];
            const dx = bx - ax;
            const dz = bz - az;
            const len = Math.hypot(dx, dz) || 1;
            const ux = dx / len;
            const uz = dz / len;
            frames.push({ ax, az, ux, uz, len, nx: -uz, nz: ux, halfWidth: line.halfWidth });
        }
    }
    return frames;
}

// stay off each segment's own endpoint cap (a distinct geometric case — nearest-*point* rather than
// nearest-*line*, `networkCoreCpu`'s own clamp-to-[0,1] projection: past either endpoint the nearest
// primitive distance is measured to that endpoint, not perpendicular to the segment, and the chord target
// is the endpoint height (constant), not the affine chord extrapolation — the affine region is strictly
// *between* the endpoints). The excluded band is an *arc-length* margin of `halfWidth + √2·SPACING`
// metres at each end. The `halfWidth` term is the same document-geometry quantity the spec already names
// for the window; the `√2·SPACING` term (the coarse lattice's own cell diagonal — the coarser lattice is
// the weaker case, so it bounds both `SPACING` and `SPACING/2`) is the mesh constant the piecewise-linear
// reconstruction reaches: a sample point's enclosing triangle has vertices up to one cell diagonal away,
// so a vertex can sit `√2·SPACING` past the endpoint while the sample point is still inside the window,
// reading the clamped (non-affine) target and contaminating the interpolation. Widening the margin by that
// same diagonal keeps every vertex the oracle interpolates from inside the affine region. This is the
// longitudinal-direction twin of the perpendicular-direction `√2·SPACING` clearance: the same
// lattice-vertex argument, applied to the endpoint edge rather than the side edge. Treatment-free:
// only `halfWidth` and `SPACING` (the document and mesh constants), never `computeFalloff` or a falloff
// scale — the generator's own placement/clearance appears in neither the window
// nor the threshold.
function endpointMargin(frame: RoadFrame): number {
    return frame.halfWidth + Math.SQRT2 * SPACING;
}

/** the sampled `[tLo, tHi]` interior of `frame` after excising {@link endpointMargin} at each end.
 *  Degenerates to a single point at the segment's own midpoint (`tLo === tHi === 0.5`) when the two
 *  margins would cross — a segment shorter than `2 * halfWidth` has no interior left once both endpoint
 *  caps are removed. Not reachable by this project's own generator (`ROAD_MIN_LENGTH = 80` against
 *  `halfWidth = 4`), but a reused caller could feed a shorter segment, and the alternative (sampling
 *  nothing, silently) is exactly the failure mode this guard exists for — so this returns one
 *  real sample point instead of an empty line. */
function sampleWindow(frame: RoadFrame): { tLo: number; tHi: number } {
    const margin = endpointMargin(frame);
    const tLo = margin / frame.len;
    const tHi = 1 - tLo;
    if (tLo >= tHi) return { tLo: 0.5, tHi: 0.5 };
    return { tLo, tHi };
}

interface LineSample {
    readonly t: number;
    readonly x: number;
    readonly z: number;
    readonly h: number;
}

function sampleLine(
    frame: RoadFrame,
    offset: number,
    sampleAt: (x: number, z: number) => number,
    tLo: number,
    tHi: number,
): LineSample[] {
    const span = tHi - tLo;
    const n = span <= 0 ? 0 : Math.max(1, Math.ceil((span * frame.len) / SAMPLE_STEP));
    const out: LineSample[] = [];
    for (let s = 0; s <= n; s++) {
        const t = n === 0 ? tLo : tLo + (span * s) / n;
        const station = t * frame.len;
        const x = frame.ax + frame.ux * station + frame.nx * offset;
        const z = frame.az + frame.uz * station + frame.nz * offset;
        out.push({ t, x, z, h: sampleAt(x, z) });
    }
    return out;
}

export type FlatnessLine = "centre" | "edgePos" | "edgeNeg";

export interface LongitudinalViolation {
    readonly line: FlatnessLine;
    readonly roadIndex: number;
    readonly t: number;
    readonly x: number;
    readonly z: number;
    readonly delta: number;
    readonly bound: number;
}

export interface CrossSectionViolation {
    readonly line: FlatnessLine;
    readonly roadIndex: number;
    readonly t: number;
    readonly x: number;
    readonly z: number;
    readonly deltaFromCentre: number;
    readonly bound: number;
}

export interface FlatnessResult {
    readonly longitudinal: readonly LongitudinalViolation[];
    /** cross-section violations — the oracle's own gate. */
    readonly crossSection: readonly CrossSectionViolation[];
    /** the worst longitudinal excess over its own step's bound (0 when every step is within tolerance). */
    readonly maxLongitudinalExcess: number;
    /** the worst cross-section excess (0 when every station agrees). */
    readonly maxCrossSectionExcess: number;
    readonly sampleCount: number;
}

/**
 * the surface-flatness property (spec Validation, "Surface flatness along the road"): within the road
 * footprint, rendered mesh height is a bounded-grade function of longitudinal station alone.
 *
 * (a) *cross-section* — at each sampled station, the centreline and both edge lines (`±(halfWidth −
 *     {@link EDGE_EPSILON})`) must read the same height within {@link CROSS_SECTION_TOL}.
 * (b) *longitudinal* — along each of those three lines independently, adjacent samples (`Δs <=`
 *     {@link SAMPLE_STEP}) must differ by no more than {@link gradeBound}`(Δs)`.
 *
 * `sampleAt` is `meshHeightAt` closed over a vertex buffer — CPU-built ({@link buildDeviceFreeVertices}) or
 * the real `readVertices()`, this function never knows which. The window (which stations/lines get
 * sampled) comes only from `doc`'s own geometry (`halfWidth`, segment endpoints); the threshold comes only
 * from road-design constants and the codec's own quantization — {@link gradeBound}/{@link
 * CROSS_SECTION_TOL} take no falloff input, so no candidate treatment can move the
 * window or the threshold (the non-coupling requirement).
 */
export function checkSurfaceFlatness(
    sampleAt: (x: number, z: number) => number,
    doc: StrokeDocument,
): FlatnessResult {
    const longitudinal: LongitudinalViolation[] = [];
    const crossSection: CrossSectionViolation[] = [];
    let sampleCount = 0;
    let maxLongitudinalExcess = 0;
    let maxCrossSectionExcess = 0;

    const frames = segmentFrames(doc);
    for (let roadIndex = 0; roadIndex < frames.length; roadIndex++) {
        const frame = frames[roadIndex];
        const inset = frame.halfWidth - EDGE_EPSILON;
        const { tLo, tHi } = sampleWindow(frame);
        const centre = sampleLine(frame, 0, sampleAt, tLo, tHi);
        const edgePos = sampleLine(frame, inset, sampleAt, tLo, tHi);
        const edgeNeg = sampleLine(frame, -inset, sampleAt, tLo, tHi);
        sampleCount += centre.length + edgePos.length + edgeNeg.length;

        // cross-section: same station (same t by construction — sampleLine shares tLo/tHi/n), every
        // footprint line should read the centreline's own height.
        for (let i = 0; i < centre.length; i++) {
            for (const [line, samples] of [
                ["edgePos", edgePos],
                ["edgeNeg", edgeNeg],
            ] as const) {
                const delta = Math.abs(samples[i].h - centre[i].h);
                const excess = delta - CROSS_SECTION_TOL;
                if (excess > 0) {
                    maxCrossSectionExcess = Math.max(maxCrossSectionExcess, excess);
                    crossSection.push({
                        line,
                        roadIndex,
                        t: samples[i].t,
                        x: samples[i].x,
                        z: samples[i].z,
                        deltaFromCentre: delta,
                        bound: CROSS_SECTION_TOL,
                    });
                }
            }
        }

        // longitudinal: adjacent-sample grade bound, walked independently along each of the three lines.
        const ds = ((tHi - tLo) * frame.len) / Math.max(1, centre.length - 1);
        const bound = gradeBound(ds);
        for (const [line, samples] of [
            ["centre", centre],
            ["edgePos", edgePos],
            ["edgeNeg", edgeNeg],
        ] as const) {
            for (let i = 1; i < samples.length; i++) {
                const delta = Math.abs(samples[i].h - samples[i - 1].h);
                const excess = delta - bound;
                if (excess > 0) {
                    maxLongitudinalExcess = Math.max(maxLongitudinalExcess, excess);
                    longitudinal.push({
                        line,
                        roadIndex,
                        t: samples[i].t,
                        x: samples[i].x,
                        z: samples[i].z,
                        delta,
                        bound,
                    });
                }
            }
        }
    }

    return {
        longitudinal,
        crossSection,
        maxLongitudinalExcess,
        maxCrossSectionExcess,
        sampleCount,
    };
}

/**
 * pins the default-suite CPU reconstruction against the real device: rebuilds the identical lattice
 * ({@link buildDeviceFreeVertices}) at the live network's own `seed` and compares its
 * footprint-line samples against `deviceRaw` (a real `readVertices()` readback) point for point. This is
 * the "device arm" the spec asks for — it validates the CPU builder's *fidelity* against the real GPU
 * output (a reference/differential check), not the surface-flatness property itself — that is a separate
 * `gate.ts` check (`surface-flatness-property-on-device`), which asserts exact zero over the same
 * `deviceRaw` because the property reads exactly zero on the shipped pipeline. Keep the
 * two apart: this one is a tolerance-bounded differential and must stay one, since a fidelity tolerance
 * cannot express an unconditional zero. Tolerance is quantization noise
 * only, doubled for two independent codec round-trips plus float-precision drift between the CPU (f64) and
 * GPU (f32) paths (`terrain/profile.ts`'s own module header names this same drift as expected and
 * harmless).
 */
export function reconstructionAgreement(
    deviceRaw: Uint32Array,
    doc: StrokeDocument,
    seed: number,
): { maxDiffM: number; sampleCount: number } {
    const cpuRaw = buildDeviceFreeVertices(doc, seed);
    const deviceSample = (x: number, z: number) => meshHeightAt(deviceRaw, x, z);
    const cpuSample = (x: number, z: number) => meshHeightAt(cpuRaw, x, z);

    let maxDiffM = 0;
    let sampleCount = 0;
    for (const frame of segmentFrames(doc)) {
        const inset = frame.halfWidth - EDGE_EPSILON;
        const { tLo, tHi } = sampleWindow(frame);
        for (const offset of [0, inset, -inset]) {
            for (const s of sampleLine(frame, offset, deviceSample, tLo, tHi)) {
                const cpuH = cpuSample(s.x, s.z);
                maxDiffM = Math.max(maxDiffM, Math.abs(s.h - cpuH));
                sampleCount++;
            }
        }
    }
    return { maxDiffM, sampleCount };
}

/** the tolerance {@link reconstructionAgreement} reads its own `maxDiffM` against — quantization noise on
 *  two independent decoded reads, ×4 for CPU/GPU float-precision drift (f64 vs f32 accumulation through
 *  five fbm octaves, a wider margin than the ×2 same-precision case since both encoders round separately
 *  off slightly different intermediate values, not a value fitted to one observed run). */
export const RECONSTRUCTION_AGREEMENT_TOL = QUANT_TOL * 4;
