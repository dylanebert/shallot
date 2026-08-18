import { Orbit } from "@dylanebert/shallot/extras";
import { Views } from "@dylanebert/shallot/render/core";
import { meshHeightAt, withHeight, worldToScreen } from "./capture";
import { frames } from "./harness";
import { documentDistance } from "./overlay/document";
import { generateNetwork } from "./overlay/network";
import { detectEdgeOffset, raggedness } from "./straightness";
import { buildNetworkGeometry, computeFalloff } from "./terrain/flatten";
import { TERRAIN_QUANT } from "./terrain/generate";
import { getSmoothRadius, readVertices, SEED } from "./terrain/terrain";

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
// Reuses the *same* boot document every run (`generateNetwork(SEED)`, `terrain.ts`'s own boot document) —
// road index 0's own first segment, deterministic — so a SPACING/TILE_RES/DIST_RANGE edit between runs
// changes only the mesh/atlas resolution under test, never the network geometry the camera is framed to.
//
// Stage 10: stage 9's `detectEdgeOffset`/`raggedness` (`straightness.ts`) are dimension-agnostic — a 1D
// sampler walked over a search axis, looking for a crossing — so the height-axis criterion below reuses
// them unchanged, only swapping what's sampled (real mesh height, `capture.ts`'s `meshHeightAt`, in place
// of screen-pixel luminance) and the axis it's sampled along (the world-space normal to the road
// centreline, in metres, in place of a screen-space search direction). The defect stage 9 found lives in
// the heightfield itself, so the corrected instrument measures the heightfield directly rather than a
// camera's projection of it — the same reason it needs no camera pose at all.

const TARGET_T = 0.08; // fraction along the road from its start — near one end, so the far ~90% recedes
const PROBE_T_LO = 0.2;
const PROBE_T_HI = 0.8;
const PROBE_COUNT = 16; // anchors along the visible run, avoiding both ends' falloff/off-frame risk
const PITCH = 0.06; // radians (~3.4°) — near-horizontal, the grazing angle itself
const DISTANCE = 15; // world metres — close to the scene's own min-distance (10), low and near the edge
const EYE_MARGIN = 0.4; // metres above the read terrain height, clearing z-fighting/embedding
const ON_BOUNDARY_TOLERANCE = 0.02; // metres — the analytic edge point must sit this close to dist=0

export interface GrazingAnchor {
    /** the analytic (ground-truth) boundary point's screen position, as a fraction of canvas size. */
    x: number;
    y: number;
    /** unit search direction in screen-fraction space, perpendicular to the projected road boundary at
     *  this anchor — the axis `straightness.ts`'s `detectEdgeOffset` walks. */
    dirX: number;
    dirY: number;
}

/** the boot document's first road, flattened to its own single segment — the fixed geometry every grazing
 *  capture (any SPACING/TILE_RES/DIST_RANGE run) frames against. */
function roadSegment(): { ax: number; az: number; bx: number; bz: number; halfWidth: number } {
    const doc = generateNetwork(SEED);
    const [[ax, az], [bx, bz]] = doc.polylines[0].points;
    return { ax, az, bx, bz, halfWidth: doc.polylines[0].halfWidth };
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
    const { ax, az, bx, bz, halfWidth } = roadSegment();
    const dx = bx - ax;
    const dz = bz - az;
    const len = Math.hypot(dx, dz) || 1;
    const ux = dx / len;
    const uz = dz / len;
    const nx = -uz; // unit normal, network.ts's own convention
    const nz = ux;

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
    for (let i = 0; i < PROBE_COUNT; i++) {
        const t = PROBE_T_LO + ((PROBE_T_HI - PROBE_T_LO) * i) / (PROBE_COUNT - 1);
        const cx = ax + ux * len * t;
        const cz = az + uz * len * t;
        // the far side from the carpark (network.ts anchors it on the +normal side of road 0), so the
        // probe's search corridor never crosses carpark-rasterized tiles.
        const ex = cx - nx * halfWidth;
        const ez = cz - nz * halfWidth;
        const edgeDist = documentDistance(ex, ez, doc);
        if (Math.abs(edgeDist) > ON_BOUNDARY_TOLERANCE) {
            // the analytic edge point should sit within quantization noise of dist=0 by construction; a
            // gross miss here means the road geometry assumption drifted and the probe would be silently
            // wrong rather than loudly wrong — fail fast instead of returning a garbage anchor.
            throw new Error(
                `grazingCapture: probe ${i} analytic edge point isn't on the boundary (dist ${edgeDist.toFixed(3)})`,
            );
        }
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

/** one boundary anchor for the height-axis criterion: the same analytic edge point
 *  {@link grazingCapture}'s anchors project to screen space, plus a world-space unit search direction
 *  (perpendicular to the road centreline, metres) instead of a screen one. */
export interface WorldEdgeAnchor {
    ex: number;
    ez: number;
    nx: number;
    nz: number;
}

/** the same {@link PROBE_COUNT} boundary anchors {@link grazingCapture} derives, in raw world coordinates
 *  rather than a screen projection — stage 10's height-axis criterion needs the point and a world-space
 *  search axis, never a camera. Exported separately since a camera pose plays no role in this reading. */
export function worldEdgeAnchors(): WorldEdgeAnchor[] {
    const { ax, az, bx, bz, halfWidth } = roadSegment();
    const dx = bx - ax;
    const dz = bz - az;
    const len = Math.hypot(dx, dz) || 1;
    const ux = dx / len;
    const uz = dz / len;
    const nx = -uz;
    const nz = ux;
    const doc = generateNetwork(SEED);

    const anchors: WorldEdgeAnchor[] = [];
    for (let i = 0; i < PROBE_COUNT; i++) {
        const t = PROBE_T_LO + ((PROBE_T_HI - PROBE_T_LO) * i) / (PROBE_COUNT - 1);
        const cx = ax + ux * len * t;
        const cz = az + uz * len * t;
        const ex = cx - nx * halfWidth;
        const ez = cz - nz * halfWidth;
        const edgeDist = documentDistance(ex, ez, doc);
        if (Math.abs(edgeDist) > ON_BOUNDARY_TOLERANCE) {
            throw new Error(
                `worldEdgeAnchors: probe ${i} analytic edge point isn't on the boundary (dist ${edgeDist.toFixed(3)})`,
            );
        }
        // the outward search axis: away from the corridor's own centreline, toward natural terrain — the
        // sign only labels which side of the anchor is "positive", detection works either way (`detectEdgeOffset`
        // scans symmetrically), but a consistent outward convention keeps the sign of a returned offset legible.
        anchors.push({ ex, ez, nx: -nx, nz: -nz });
    }
    return anchors;
}

/** one anchor's height-axis reading: the world-metre offset (along {@link WorldEdgeAnchor.nx}/`nz`) between
 *  the analytic boundary point and where the *real* mesh height actually crosses from the flattened
 *  corridor to natural terrain — `null` when no crossing was found (the search radius missed it) or the
 *  endpoints didn't differ enough to trust as a real transition (`reason: "low-contrast"`, the flat-ground
 *  control case: no cut, no differential, nothing to find). */
export interface HeightReading {
    i: number;
    offset: number | null;
    contrastM: number;
    reason: "found" | "no-crossing" | "low-contrast";
}

export interface HeightSilhouetteResult {
    falloffM: number;
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

/** stage 10's height-axis straightness criterion: walks the *real* rendered mesh height
 *  (`capture.ts`'s `meshHeightAt`) outward from each analytic boundary anchor over the network's own
 *  derived falloff distance, and finds where it actually crosses from the flattened plateau to natural
 *  terrain — the deviation from the anchor (which sits exactly on the analytic boundary by construction)
 *  is the height-silhouette offset, in world metres. `raggedness` (`straightness.ts`) aggregates the same
 *  way stage 9's screen-space instrument did; what changed is only the sampled quantity and its axis
 *  (world height, not screen luminance) — the corrected criterion `boot.ts` exposes as
 *  `window.__roadsHeightSilhouette`. */
export async function heightSilhouette(): Promise<HeightSilhouetteResult> {
    const anchors = worldEdgeAnchors();
    const raw = await readVertices();
    const doc = generateNetwork(SEED);
    const { cutDepth } = buildNetworkGeometry(doc, SEED, getSmoothRadius());
    const falloff = computeFalloff(cutDepth);

    const sampleAt = (x: number, z: number): number => meshHeightAt(raw, x, z);

    const readings: HeightReading[] = anchors.map((a, i) => {
        const s0 = sampleAt(a.ex - a.nx * falloff, a.ez - a.nz * falloff);
        const s1 = sampleAt(a.ex + a.nx * falloff, a.ez + a.nz * falloff);
        const lo = Math.min(s0, s1);
        const hi = Math.max(s0, s1);
        const contrastM = hi - lo;
        if (contrastM < MIN_CONTRAST_M) {
            return { i, offset: null, contrastM, reason: "low-contrast" as const };
        }
        const offset = detectEdgeOffset(
            sampleAt,
            a.ex,
            a.ez,
            a.nx,
            a.nz,
            lo,
            hi,
            falloff,
            STEPS_PER_M,
        );
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
        cutDepthM: cutDepth,
        anchorCount: anchors.length,
        readings,
        rmsM: r.rms,
        maxM: r.max,
        foundCount: r.n,
    };
}
