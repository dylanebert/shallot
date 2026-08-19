import { Orbit } from "@dylanebert/shallot/extras";
import { Views } from "@dylanebert/shallot/render/core";
import {
    edgeAnchorPoints,
    heightMidpointAnchor,
    roadFrame,
    sideSlopeWindow,
    worldEdgeAnchors,
} from "./boundaryAnchors";
import { meshHeightAt, withHeight, worldToScreen } from "./capture";
import { frames } from "./harness";
import { generateNetwork } from "./overlay/network";
import { detectEdgeOffset, raggedness } from "./straightness";
import { buildNetworkGeometry, computeFalloff } from "./terrain/flatten";
import { TERRAIN_QUANT } from "./terrain/generate";
import { readVertices, SEED } from "./terrain/terrain";

// Stage 9's discriminator — the grazing-view half. The release-look screenshot that opened this stage was
// a hand-orbited shot; this reproduces it as a *fixed, reproducible* camera pose so the two-resolution
// reading is a real comparison and not two different framings. A near-horizontal grazing angle is the
// point, not an aesthetic choice: a small world-space deviation projects to a much larger screen-pixel
// deviation the closer the view direction runs parallel to the surface (the same reason raking light
// reveals machining marks a top-down look hides) — the straightness signal this stage measures needs that
// magnification to be visible in a handful of screen pixels at all. Kept for the evidence screenshot
// `test/straightness.spec.ts` still saves; the actual reading (`heightSilhouette`, below) is now
// camera-independent — it lives entirely in the heightfield, not in what a camera happens to project.
//
// The analytic anchor derivation (road frame, per-probe boundary point, world-space anchor) lives in
// `boundaryAnchors.ts` — this module only adds the camera-dependent half (`grazingCapture`, needs
// `Orbit`/`Views`) and the height-axis reading (`heightSilhouette`, needs the mesh readback).
//
// Stage 10: stage 9's `detectEdgeOffset`/`raggedness` (`straightness.ts`) are dimension-agnostic — a 1D
// sampler walked over a search axis, looking for a crossing — so the height-axis criterion below reuses
// them unchanged, only swapping what's sampled (real mesh height, `capture.ts`'s `meshHeightAt`, in place
// of screen-pixel luminance) and the axis it's sampled along (the world-space normal to the road
// centreline, in metres, in place of a screen-space search direction). The defect stage 9 found lives in
// the heightfield itself, so the corrected instrument measures the heightfield directly rather than a
// camera's projection of it — the same reason it needs no camera pose at all.
//
// Stage 11b: stage 10's window and threshold scanned `computeFalloff(cutDepth)` directly — the real
// rendered transition width, floor included — so widening the floor widened the search window and moved
// the threshold with no real change on the ground, making the reading non-comparable across two floor
// derivations (spec Approach, stage 11b; `boundaryAnchors.ts`'s `sideSlopeWindow` doc comment has the full
// argument). `heightSilhouette` below now scans and anchors on `sideSlopeWindow(cutDepth)` instead — a
// pure function of the network's own measured cut, upstream of any floor — and reports the real
// `computeFalloff` output separately, as `falloffM`, for evidence only.

const TARGET_T = 0.08; // fraction along the road from its start — near one end, so the far ~90% recedes
const PITCH = 0.06; // radians (~3.4°) — near-horizontal, the grazing angle itself
const DISTANCE = 15; // world metres — close to the scene's own min-distance (10), low and near the edge
const EYE_MARGIN = 0.4; // metres above the read terrain height, clearing z-fighting/embedding

export interface GrazingAnchor {
    /** the analytic (ground-truth) boundary point's screen position, as a fraction of canvas size. */
    x: number;
    y: number;
    /** unit search direction in screen-fraction space, perpendicular to the projected road boundary at
     *  this anchor — the axis `straightness.ts`'s `detectEdgeOffset` walks. */
    dirX: number;
    dirY: number;
}

/** the canvas-presenting camera's eid — the same `Views` disambiguation `capture.ts`'s `cameraEid` uses
 *  (a shadow-casting light's pooled shadow camera also carries `Camera`, so only a live `canvas` picks the
 *  display camera out). Re-derived here rather than imported since `capture.ts` keeps it private to its
 *  own read-only probe bridge; this module's job is to *write* the pose, a different responsibility. */
function cameraEid(): number {
    for (const [eid, view] of Views) {
        if (view.canvas) return eid;
    }
    throw new Error("grazingCapture: no canvas-presenting camera in Views");
}

/**
 * repositions the scene's orbit camera to a fixed grazing pose looking along the boot document's first
 * road (near one end, looking toward the other), waits for the pose to settle, and returns the analytic
 * boundary anchors (screen position + local search direction) the real screenshot's straightness probe
 * should search around. Exported as `window.__roadsGrazingCapture` (`boot.ts`).
 */
export async function grazingCapture(): Promise<{ anchors: GrazingAnchor[] }> {
    const eid = cameraEid();
    const frame = roadFrame();
    const { ax, az, ux, uz, len } = frame;

    const targetX = ax + ux * len * TARGET_T;
    const targetZ = az + uz * len * TARGET_T;
    const [, targetYRaw] = await withHeight(targetX, targetZ);
    const targetY = targetYRaw + EYE_MARGIN;

    // forward = (ux, uz); Orbit's own convention (extras/orbit/index.ts) places the camera at
    // target + distance*(sin(yaw)cos(pitch), sin(pitch), cos(yaw)cos(pitch)) and looks back at target, so
    // the camera sits *behind* target along -forward when sin(yaw) = -ux, cos(yaw) = -uz.
    const yaw = Math.atan2(-ux, -uz);

    Orbit.smoothness.set(eid, 1); // snap in one frame instead of easing over several
    Orbit.pitch.set(eid, PITCH);
    Orbit.yaw.set(eid, yaw);
    Orbit.distance.set(eid, DISTANCE);
    Orbit.pan.set(eid, targetX, targetY, targetZ, 0); // target defaults to world origin (entity 0), so
    // pan *is* the absolute look-at point (extras/orbit/index.ts's targetX/Y/Z derivation)

    await frames(3); // let OrbitSystem (mode: always) re-derive Transform from the snapped pose

    const anchors: GrazingAnchor[] = [];
    const doc = generateNetwork(SEED);
    for (const { ex, ez } of edgeAnchorPoints(frame, doc)) {
        const [, ey] = await withHeight(ex, ez);
        const [pt] = worldToScreen([[ex, ey, ez]]);

        // local screen search direction: perpendicular to the projected road boundary at this anchor,
        // derived from two more projected points along the boundary rather than assumed horizontal — a
        // near-grazing view can still tilt the boundary's screen-space run away from purely horizontal.
        const alongEx = ex + ux * 1;
        const alongEz = ez + uz * 1;
        const [, alongEy] = await withHeight(alongEx, alongEz);
        const [alongPt] = worldToScreen([[alongEx, alongEy, alongEz]]);
        const adx = alongPt.x - pt.x;
        const ady = alongPt.y - pt.y;
        const alen = Math.hypot(adx, ady) || 1;
        // perpendicular to the along-boundary screen direction, in screen-fraction space
        const dirX = -ady / alen;
        const dirY = adx / alen;

        anchors.push({ x: pt.x, y: pt.y, dirX, dirY });
    }

    return { anchors };
}

/** one anchor's height-axis reading: the world-metre offset (along the anchor's own outward normal)
 *  between the analytic boundary's {@link heightMidpointAnchor} and where the *real* mesh height actually
 *  crosses from the flattened corridor to natural terrain — `null` when no crossing was found (the search
 *  radius missed it) or the endpoints didn't differ enough to trust as a real transition (`reason:
 *  "low-contrast"`, the flat-ground control case: no cut, no differential, nothing to find). */
export interface HeightReading {
    i: number;
    offset: number | null;
    contrastM: number;
    reason: "found" | "no-crossing" | "low-contrast";
}

export interface HeightSilhouetteResult {
    /** `computeFalloff`'s output for this network — the *real* rendered transition width, floor included.
     *  Informational only as of stage 11b: neither the search window nor the threshold anchor below reads
     *  this value (they read {@link HeightSilhouetteResult.windowM} instead), so it no longer bounds
     *  `rmsM`/`maxM`. */
    falloffM: number;
    /** stage 11b's decoupled measurement window (`boundaryAnchors.ts`'s `sideSlopeWindow(cutDepthM)`) —
     *  what the search radius and the threshold anchor actually read, comparable across two networks whose
     *  `computeFalloff` floor differs even when `falloffM` isn't. */
    windowM: number;
    cutDepthM: number;
    anchorCount: number;
    readings: HeightReading[];
    rmsM: number;
    maxM: number;
    foundCount: number;
}

// noise floor for a real road/terrain height differential: 100x the Y-axis quantization step
// (`TERRAIN_QUANT.posScale.y / 65535`, ~1.2 mm at RELIEF=40) puts the floor around 12 cm — comfortably
// above quantization/interpolation noise, comfortably below the metre-scale defect this criterion exists
// to catch, so a genuine transition is never mistaken for noise and a flat/no-cut control (no differential
// at all) reads as "nothing to find" rather than a spurious crossing on numerical noise.
const MIN_CONTRAST_M = 100 * (TERRAIN_QUANT.posScale.y / 65535);
const STEPS_PER_M = 8; // sub-decimetre crossing resolution — cheap at this anchor/radius count

/** stage 10's height-axis straightness criterion, stage 11b's window: walks the *real* rendered mesh
 *  height (`capture.ts`'s `meshHeightAt`) outward from each analytic boundary anchor's
 *  `boundaryAnchors.ts`'s {@link heightMidpointAnchor} over `sideSlopeWindow(cutDepth)` — not
 *  `computeFalloff(cutDepth)` — and finds where it actually crosses from the flattened plateau to natural
 *  terrain; the deviation from that ground-truth midpoint is the height-silhouette offset, in world
 *  metres. Stage 10's version scanned and anchored on `computeFalloff`'s output directly, which bundles
 *  the AASHTO side-slope term with a mesh-sampling floor denominated in `SPACING` — so widening that floor
 *  widened the search window into raw terrain *and* moved the threshold, without a real change on the
 *  ground the window was supposed to measure (spec Approach, stage 11b). `sideSlopeWindow` is a pure
 *  function of `cutDepth` alone, upstream of the floor entirely, so a
 *  floor change can't reach either the window or the threshold — the property that makes two falloff
 *  floors comparable. `computeFalloff`'s real output still rides along as `falloffM`, informational only.
 *  `raggedness` (`straightness.ts`) aggregates the same way stage 9's screen-space instrument did; what
 *  changed since then is only the sampled quantity and its axis (world height, not screen luminance) — the
 *  corrected criterion `boot.ts` exposes as `window.__roadsHeightSilhouette`. */
export async function heightSilhouette(): Promise<HeightSilhouetteResult> {
    const anchors = worldEdgeAnchors();
    const raw = await readVertices();
    const doc = generateNetwork(SEED);
    const { cutDepth } = buildNetworkGeometry(doc, SEED);
    const falloff = computeFalloff(cutDepth); // informational only — see HeightSilhouetteResult.falloffM
    const win = sideSlopeWindow(cutDepth); // what the search + threshold below actually read

    const sampleAt = (x: number, z: number): number => meshHeightAt(raw, x, z);

    const readings: HeightReading[] = anchors.map((a, i) => {
        const s0 = sampleAt(a.ex - a.nx * win, a.ez - a.nz * win);
        const s1 = sampleAt(a.ex + a.nx * win, a.ez + a.nz * win);
        const lo = Math.min(s0, s1);
        const hi = Math.max(s0, s1);
        const contrastM = hi - lo;
        if (contrastM < MIN_CONTRAST_M) {
            return { i, offset: null, contrastM, reason: "low-contrast" as const };
        }
        const { mx, mz } = heightMidpointAnchor(a, win);
        const offset = detectEdgeOffset(sampleAt, mx, mz, a.nx, a.nz, lo, hi, win, STEPS_PER_M);
        return {
            i,
            offset,
            contrastM,
            reason: offset === null ? ("no-crossing" as const) : ("found" as const),
        };
    });

    const r = raggedness(readings.map((d) => d.offset));
    return {
        falloffM: falloff,
        windowM: win,
        cutDepthM: cutDepth,
        anchorCount: anchors.length,
        readings,
        rmsM: r.rms,
        maxM: r.max,
        foundCount: r.n,
    };
}
