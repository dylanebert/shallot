// Stage 12's surface-flatness oracle — the spec's Validation "Surface flatness along the road (the
// reconstruction axis)", the unit's live defect gate. Stage 9-11 all measured a *boundary* axis (albedo
// straightness, then height straightness, then a falloff-width knob 11c proved orthogonal to the reported
// defect); the spec's own diagnosis routes this stage to a different axis entirely — the mesh's
// piecewise-linear *reconstruction* across the footprint edge, not the boundary's own width or position.
//
// A vertex-only check is insufficient by construction: footprint vertices already carry the smoothed,
// grade-limited profile (`terrain/flatten.ts`'s `networkCore` interpolates `mix(aHeight, bHeight, t)`), so
// away from junctions they're flat to a few centimetres — a vertex-only oracle would read green over the
// live defect, the sixth instrument failing the way the first five did (stage 9's screen-space albedo
// probe, stage 10's un-anchored height probe, stage 11's width-only floor). The predicted defect lives in
// the *reconstruction*: a triangle straddling the footprint edge has one or more corners *outside* the
// flattened core, cosine-eased toward off-road terrain that can vary by metres over one 4 m cell — so a
// point sampled right at the edge, inside a triangle whose other corners sit outside it, reads a height
// blended toward that off-road variation, not the flat corridor its own analytic position implies. Reading
// height from `capture.ts`'s `meshHeightAt` (a two-triangle-per-quad reconstruction over a raw vertex
// buffer) rather than the continuous `flattenHeight` function is what makes this oracle able to see that —
// the continuous function has no triangles to straddle.
//
// Two vertex-buffer sources feed the one oracle below (`checkSurfaceFlatness`), which only ever touches a
// `Uint32Array` + `meshHeightAt` — never which device produced it:
//   - {@link buildDeviceFreeVertices} — a CPU-only mirror of `terrain/generate.ts`'s height-kernel loop
//     (no GPU dispatch), the default-suite arm (`flatness.test.ts`, `bun test ./src -t "surface flatness"`).
//   - the real `readVertices()` (`terrain/terrain.ts`) — the device arm inside `bun run gate`
//     (`gate.ts`'s `reconstructionAgreement`, which pins the CPU builder's fidelity against the real GPU
//     output rather than re-asserting the property device-side).
//
// What this oracle cannot see (name the blind axes, `checks.md`'s granularity clause):
//   - albedo registration — the overlay's own texel/coverage crispness is a separate render-half property
//     (the spec's "Fs composite" criterion, `terrain.ts`'s fs), untouched by anything read here.
//   - shading normals — the finite-difference normal `generate.ts`'s kernel emits alongside height is
//     never read; a normal can look wrong while every height sampled here reads within tolerance.
//   - everything outside the road footprint — `documentDistance(x, z, doc) > 0` terrain is natural and
//     unconstrained by design; this oracle only walks centreline + edge lines *inside* a footprint.
//   - reseed staleness — this always samples the *live* document's own current geometry; a stale
//     atlas/dirty-tile residue (stage 14's subject) leaves no trace on the heightfield this oracle reads.

import { encodePos } from "@dylanebert/shallot/utils/core";
import * as d from "typegpu/data";
import { meshHeightAt } from "./capture";
import { documentDistance, type PolygonStamp, type StrokeDocument } from "./overlay/document";
import {
    buildNetworkGeometry,
    computeFalloff,
    flattenHeight,
    type ProfileSegment,
} from "./terrain/flatten";
import { TERRAIN_QUANT } from "./terrain/generate";
import { CELLS, SPACING } from "./terrain/grid";
import { makePermutation } from "./terrain/noise";
import { heightAtCpu, MAX_GRADE, MAX_GRADE_BREAK, PROFILE_STEP } from "./terrain/profile";

// --- derived window + tolerance — no candidate treatment (falloff, smoothRadius, falloffScale) appears
// in any of these, per the spec's own non-coupling requirement (stage 11's whole lesson: a criterion whose
// window or threshold is a function of the parameter under test is not a differential over that parameter).

/** the spec's own longitudinal sample spacing bound: no coarser than a quarter of the mesh's own vertex
 *  spacing, so no triangle edge along a sampled line is ever skipped between two adjacent samples. */
export const SAMPLE_STEP = SPACING / 4;

/** height-quantization noise on one decoded reconstruction sample — `TERRAIN_QUANT`'s own AABB divided by
 *  the unorm16 range, the same derivation `capture.test.ts`'s `meshHeightAt` oracle uses. */
const QUANT_TOL = TERRAIN_QUANT.posScale.y / 65535;

/** the profile's own piecewise-linearity error: half a grade-break step over one arc-length increment
 *  (`terrain/profile.ts`'s own derivation, ≈ 2 cm at the shipped constants) — the profile control points
 *  themselves are a grade-*limited*, not grade-*continuous*, sequence, so even a perfect reconstruction of
 *  them carries this much slack against the idealized continuous grade bound. */
const PROFILE_LINEARITY_TOL = 0.5 * MAX_GRADE_BREAK * PROFILE_STEP;

/** the longitudinal grade bound over a `ds`-metre step: the road-design grade limit (`MAX_GRADE`) plus the
 *  profile's own linearity slack plus two independent quantized height reads (one at each end of the
 *  step). Neither `MAX_GRADE`/`MAX_GRADE_BREAK`/`PROFILE_STEP` (road-design + mesh constants) nor
 *  `QUANT_TOL` (the codec's own AABB) depends on any candidate treatment. */
export function gradeBound(ds: number): number {
    return MAX_GRADE * ds + PROFILE_LINEARITY_TOL + 2 * QUANT_TOL;
}

/** the cross-section bound: two footprint points at the same longitudinal station should read the same
 *  height (the flatten target is a pure function of station, `networkCoreCpu`'s own `t`-only interpolation
 *  below), up to two independent reconstructions' quantization noise. */
export const CROSS_SECTION_TOL = 2 * QUANT_TOL;

/** how far inside the road edge the two edge lines sit — small relative to `SPACING` so the probe stays
 *  within the single grid cell the predicted defect straddles (the edge itself), rather than retreating
 *  into the core's interior where the defect can't reach. Not derived from any falloff/treatment
 *  quantity — a fixed fraction of the mesh's own cell size. */
export const EDGE_EPSILON = SPACING / 100;

// --- CPU-only lattice reconstruction — no GPU, no `@dylanebert/shallot/render/core` import, the
// default-suite arm's own substrate.

/** the nearest-network-primitive core distance + target height at (px, pz) — a plain-JS re-authoring of
 *  `terrain/flatten.ts`'s GPU `networkCore`, deliberately the *same* formula (not an independent
 *  derivation the way `overlay/rasterize.ts`'s distance math is): this function's job is to reconstruct
 *  exactly what the real mesh does, not to provide a second opinion on it — the property under test is the
 *  mesh's own triangle reconstruction, and the height *per vertex* has to match production's own
 *  `flattenedHeightAt` or the built lattice isn't the mesh this oracle is meant to read. */
function networkCoreCpu(
    px: number,
    pz: number,
    segments: readonly ProfileSegment[],
    polygons: readonly PolygonStamp[],
    naturalHeightAt: (x: number, z: number) => number,
): { coreDist: number; targetHeight: number } {
    let bestCore = Number.POSITIVE_INFINITY;
    let bestTarget = 0;

    for (const seg of segments) {
        const abx = seg.bx - seg.ax;
        const abz = seg.bz - seg.az;
        const apx = px - seg.ax;
        const apz = pz - seg.az;
        const dd = abx * abx + abz * abz;
        const t = dd > 0 ? Math.min(1, Math.max(0, (apx * abx + apz * abz) / dd)) : 0;
        const cx = seg.ax + t * abx;
        const cz = seg.az + t * abz;
        const core = Math.hypot(px - cx, pz - cz) - seg.halfWidth;
        if (core < bestCore) {
            bestCore = core;
            bestTarget = seg.aHeight + t * (seg.bHeight - seg.aHeight);
        }
    }

    for (const poly of polygons) {
        const pts = poly.points;
        let inside = false;
        let nearestEdge = Number.POSITIVE_INFINITY;
        let cx = 0;
        let cz = 0;
        for (const [x, z] of pts) {
            cx += x;
            cz += z;
        }
        cx /= pts.length;
        cz /= pts.length;
        for (let i = 0; i < pts.length; i++) {
            const [ax, az] = pts[i];
            const [bx, bz] = pts[(i + 1) % pts.length];
            if (az > pz !== bz > pz) {
                const xCross = ax + ((pz - az) / (bz - az)) * (bx - ax);
                if (px < xCross) inside = !inside;
            }
            const abx = bx - ax;
            const abz = bz - az;
            const apx = px - ax;
            const apz = pz - az;
            const dd = abx * abx + abz * abz;
            const t = dd > 0 ? Math.min(1, Math.max(0, (apx * abx + apz * abz) / dd)) : 0;
            const ecx = ax + t * abx;
            const ecz = az + t * abz;
            const edgeDist = Math.hypot(px - ecx, pz - ecz);
            if (edgeDist < nearestEdge) nearestEdge = edgeDist;
        }
        const core = inside ? -nearestEdge : nearestEdge;
        if (core < bestCore) {
            bestCore = core;
            bestTarget = naturalHeightAt(cx, cz);
        }
    }

    return { coreDist: bestCore, targetHeight: bestTarget };
}

/** discretize the continuous flatten field (`segments`/`polygons`/`falloff`, `naturalHeightAt`) onto a
 *  `spacing`-metre, `cells`-quad lattice, encoded the same way `terrain/generate.ts`'s height kernel
 *  encodes its own vertex stream (`encodePos` against the live `TERRAIN_QUANT`) — the output is a
 *  `meshHeightAt`-compatible raw buffer regardless of resolution, so the same reconstruction function reads
 *  both the production-resolution lattice and {@link buildDeviceFreeVertices}'s discrimination arm at a
 *  different resolution (`flatness.test.ts`'s `SPACING/2` arm). Normal/UV words are left at zero — nothing
 *  downstream of this buffer reads them (`meshHeightAt` only decodes the position words). */
export function buildLatticeVertices(
    spacing: number,
    cells: number,
    segments: readonly ProfileSegment[],
    polygons: readonly PolygonStamp[],
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
            const { coreDist, targetHeight } = networkCoreCpu(
                x,
                z,
                segments,
                polygons,
                naturalHeightAt,
            );
            const y = flattenHeight(natural, targetHeight, coreDist, falloff);
            const idx = (iz * verts + ix) * 4;
            const m = encodePos(d.vec3f(x, y, z), 0, TERRAIN_QUANT);
            raw[idx] = m.x;
            raw[idx + 1] = m.y;
        }
    }
    return raw;
}

/** the production-shape reconstruction: `doc`'s own network flattened at `seed`/`smoothRadius`
 *  (optionally `falloffScale`, stage 11a's still-live multiplier on this branch), at the mesh's real
 *  `SPACING`/`CELLS` — a drop-in `readVertices()` substitute with no device, `flatness.test.ts`'s and
 *  `gate.ts`'s shared entry point. `flattenDoc` and `sampleDoc` (the caller's own footprint definition,
 *  `checkSurfaceFlatness`) are deliberately the same `doc` here; {@link buildLatticeVertices} lets a caller
 *  split them (stage 12's own null-control arm: an empty `flattenDoc` with the real network's footprint
 *  still standing, `flatness.test.ts`'s "no-cut" arm). */
export function buildDeviceFreeVertices(
    flattenDoc: StrokeDocument,
    seed: number,
    smoothRadius: number,
    falloffScale = 1,
): Uint32Array {
    const perm = makePermutation(seed);
    const { segments, cutDepth } = buildNetworkGeometry(flattenDoc, seed, smoothRadius);
    const falloff = computeFalloff(cutDepth, falloffScale);
    return buildLatticeVertices(SPACING, CELLS, segments, flattenDoc.polygons, falloff, (x, z) =>
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

// stay off each segment's own endpoint cap (a distinct geometric case — nearest-point rather than
// nearest-line — out of this oracle's scope) by sampling only the segment's interior.
const T_LO = 0.05;
const T_HI = 0.95;

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
): LineSample[] {
    const n = Math.max(1, Math.ceil(((T_HI - T_LO) * frame.len) / SAMPLE_STEP));
    const out: LineSample[] = [];
    for (let s = 0; s <= n; s++) {
        const t = T_LO + ((T_HI - T_LO) * s) / n;
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
    readonly crossSection: readonly CrossSectionViolation[];
    /** the worst longitudinal excess over its own step's bound (0 when every step is within tolerance). */
    readonly maxLongitudinalExcess: number;
    /** the worst cross-section excess over `CROSS_SECTION_TOL` (0 when every station agrees). */
    readonly maxCrossSectionExcess: number;
    readonly sampleCount: number;
}

/**
 * the stage 12 property (spec Validation, "Surface flatness along the road"): within the road footprint,
 * rendered mesh height is a grade-limited function of longitudinal station alone.
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
 * CROSS_SECTION_TOL} take no falloff, smoothing radius, or falloff-scale input, so no candidate treatment
 * can move the window or the threshold (the non-coupling stage 11's whole investigation turned on).
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
        const centre = sampleLine(frame, 0, sampleAt);
        const edgePos = sampleLine(frame, inset, sampleAt);
        const edgeNeg = sampleLine(frame, -inset, sampleAt);
        sampleCount += centre.length + edgePos.length + edgeNeg.length;

        // cross-section: same station (same t by construction — sampleLine shares T_LO/T_HI/n), every
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
        const ds = ((T_HI - T_LO) * frame.len) / Math.max(1, centre.length - 1);
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

/** a shared derivation `checkSurfaceFlatness` doesn't otherwise need: whether (x, z) sits in *any*
 *  footprint at all — exported only for tests that want to sanity-check a sampled point's own membership,
 *  never consulted by the oracle above (which walks known-on-footprint lines by construction). */
export function inFootprint(x: number, z: number, doc: StrokeDocument): boolean {
    return documentDistance(x, z, doc) <= 0;
}

/**
 * pins the default-suite CPU reconstruction against the real device: rebuilds the identical lattice
 * ({@link buildDeviceFreeVertices}) at the live network's own `seed`/`smoothRadius`/`falloffScale` and
 * compares its footprint-line samples against `deviceRaw` (a real `readVertices()` readback) point for
 * point. This is the "device arm" the spec asks for — it validates the CPU builder's *fidelity* against
 * the real GPU output (a reference/differential check), not the surface-flatness property itself (which
 * `bun test`'s default-suite arm already checks device-free, and is allowed to read red on the shipped
 * pipeline, `gate.ts` would break every run if it re-asserted "no violations" here). Tolerance is
 * quantization noise only, doubled for two independent codec round-trips plus float-precision drift
 * between the CPU (f64) and GPU (f32) paths (`terrain/profile.ts`'s own module header names this same
 * drift as expected and harmless).
 */
export function reconstructionAgreement(
    deviceRaw: Uint32Array,
    doc: StrokeDocument,
    seed: number,
    smoothRadius: number,
    falloffScale: number,
): { maxDiffM: number; sampleCount: number } {
    const cpuRaw = buildDeviceFreeVertices(doc, seed, smoothRadius, falloffScale);
    const deviceSample = (x: number, z: number) => meshHeightAt(deviceRaw, x, z);
    const cpuSample = (x: number, z: number) => meshHeightAt(cpuRaw, x, z);

    let maxDiffM = 0;
    let sampleCount = 0;
    for (const frame of segmentFrames(doc)) {
        const inset = frame.halfWidth - EDGE_EPSILON;
        for (const offset of [0, inset, -inset]) {
            for (const s of sampleLine(frame, offset, deviceSample)) {
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
