import { documentDistance, type StrokeDocument } from "./overlay/document";
import { generateNetwork } from "./overlay/network";
import { SIDE_SLOPE_LIMIT } from "./terrain/flatten";
import { SEED } from "./terrain/terrain";

// The analytic (ground-truth) boundary anchor derivation `grazingCapture.ts`'s two straightness
// instruments both need — `grazingCapture()` (screen-space, needs a live camera/Orbit) projects these to
// screen pixels, `heightSilhouette()` (world-space, device-free apart from the mesh readback) walks them
// directly. Pulled into its own module, with zero `@dylanebert/shallot/extras`/`render/core` imports, so
// it stays importable under plain `bun test` — `grazingCapture.ts` imports `Orbit` at module scope, which
// throws under bun (`GPUTextureUsage is not defined`, `@dylanebert/shallot/extras`'s own eager surface
// registration), so this module is the only path to a device-free unit test over the shared anchor math
// (`boundaryAnchors.test.ts`).
//
// Reuses the *same* boot document every run (`generateNetwork(SEED)`, `terrain.ts`'s own boot document) —
// road index 0's own first segment, deterministic — so a SPACING/TILE_RES/DIST_RANGE edit between runs
// changes only the mesh/atlas resolution under test, never the network geometry the probes are framed to.

const PROBE_T_LO = 0.2;
const PROBE_T_HI = 0.8;
const PROBE_COUNT = 16; // anchors along the visible run, avoiding both ends' falloff/off-frame risk
const ON_BOUNDARY_TOLERANCE = 0.02; // metres — the analytic edge point must sit this close to dist=0

/** the boot document's first road, flattened to its own single segment — the fixed geometry every probe
 *  (any SPACING/TILE_RES/DIST_RANGE run) frames against. */
function roadSegment(): { ax: number; az: number; bx: number; bz: number; halfWidth: number } {
    const doc = generateNetwork(SEED);
    const [[ax, az], [bx, bz]] = doc.polylines[0].points;
    return { ax, az, bx, bz, halfWidth: doc.polylines[0].halfWidth };
}

export interface RoadFrame {
    ax: number;
    az: number;
    ux: number;
    uz: number;
    len: number;
    nx: number;
    nz: number;
    halfWidth: number;
}

/** {@link roadSegment}'s endpoints resolved into the frame every boundary probe anchors to: unit tangent
 *  (`ux`/`uz`), unit outward-from-carpark normal (`nx`/`nz`, `network.ts`'s own convention), and the
 *  segment length. */
export function roadFrame(): RoadFrame {
    const { ax, az, bx, bz, halfWidth } = roadSegment();
    const dx = bx - ax;
    const dz = bz - az;
    const len = Math.hypot(dx, dz) || 1;
    const ux = dx / len;
    const uz = dz / len;
    const nx = -uz; // unit normal, network.ts's own convention
    const nz = ux;
    return { ax, az, ux, uz, len, nx, nz, halfWidth };
}

export interface EdgeAnchorPoint {
    ex: number;
    ez: number;
}

/** probe `i`'s analytic boundary point along `frame` — `PROBE_T_LO..PROBE_T_HI` fraction along the
 *  centreline, offset by `halfWidth` toward the far side from the carpark (network.ts anchors it on the
 *  +normal side of road 0, so the probe's search corridor never crosses carpark-rasterized tiles). */
function edgeAnchorPoint(i: number, frame: RoadFrame): EdgeAnchorPoint {
    const t = PROBE_T_LO + ((PROBE_T_HI - PROBE_T_LO) * i) / (PROBE_COUNT - 1);
    const cx = frame.ax + frame.ux * frame.len * t;
    const cz = frame.az + frame.uz * frame.len * t;
    const ex = cx - frame.nx * frame.halfWidth;
    const ez = cz - frame.nz * frame.halfWidth;
    return { ex, ez };
}

/** all {@link PROBE_COUNT} analytic boundary points along `frame`, each verified within
 *  {@link ON_BOUNDARY_TOLERANCE} of the analytic boundary (`documentDistance`) — the analytic edge point
 *  should sit within quantization noise of `dist = 0` by construction; a gross miss means the road
 *  geometry assumption drifted, so this fails loud rather than returning a garbage anchor. Shared by
 *  `grazingCapture.ts`'s `grazingCapture()` (needs the screen projection too) and {@link worldEdgeAnchors}
 *  (needs only the raw point) — one derivation, so a fix here lands in both instruments at once. */
export function edgeAnchorPoints(frame: RoadFrame, doc: StrokeDocument): EdgeAnchorPoint[] {
    return Array.from({ length: PROBE_COUNT }, (_, i) => {
        const p = edgeAnchorPoint(i, frame);
        const edgeDist = documentDistance(p.ex, p.ez, doc);
        if (Math.abs(edgeDist) > ON_BOUNDARY_TOLERANCE) {
            throw new Error(
                `edgeAnchorPoints: probe ${i} analytic edge point isn't on the boundary (dist ${edgeDist.toFixed(3)})`,
            );
        }
        return p;
    });
}

/** one boundary anchor for the height-axis criterion: the analytic edge point plus a world-space unit
 *  search direction (perpendicular to the road centreline, metres, pointing outward toward natural
 *  terrain) instead of a screen one. */
export interface WorldEdgeAnchor {
    ex: number;
    ez: number;
    nx: number;
    nz: number;
}

/** the {@link PROBE_COUNT} boundary anchors in raw world coordinates rather than a screen projection —
 *  the height-axis criterion needs the point and a world-space search axis, never a camera. */
export function worldEdgeAnchors(): WorldEdgeAnchor[] {
    const frame = roadFrame();
    const doc = generateNetwork(SEED);
    // the outward search axis: away from the corridor's own centreline, toward natural terrain — the sign
    // only labels which side of the anchor is "positive", detection works either way (`detectEdgeOffset`
    // scans symmetrically), but a consistent outward convention keeps the sign of a returned offset legible.
    return edgeAnchorPoints(frame, doc).map(({ ex, ez }) => ({
        ex,
        ez,
        nx: -frame.nx,
        nz: -frame.nz,
    }));
}

/** the height-axis instrument's own ground-truth zero point: not the road edge itself
 *  ({@link WorldEdgeAnchor.ex}/`ez`, `coreDist = 0`) but the cosine ease's own midpoint
 *  (`coreDist = window / 2`, `window` being whatever transition width the caller is measuring against —
 *  `terrain/flatten.ts`'s `flattenHeight` reaches `0.5 - 0.5·cos(π·t) = 0.5` at `t = 0.5`).
 *  `straightness.ts`'s `detectEdgeOffset` reports where the sampled signal crosses the *midpoint* between
 *  its `lo`/`hi` brackets, so that ease-midpoint, not the road edge, is where a perfectly straight edge's
 *  detected crossing sits by construction. Anchoring the reading at `coreDist = 0` instead reports a
 *  constant `window / 2` bias on every straight edge, indistinguishable from boundary raggedness
 *  (`boundaryAnchors.test.ts` proves the bias and the fix against a synthetic straight edge). Generic in
 *  `window` rather than hard-coded to {@link sideSlopeWindow}'s output: the caller decides which
 *  transition width it's measuring against. */
export function heightMidpointAnchor(
    a: WorldEdgeAnchor,
    window: number,
): { mx: number; mz: number } {
    const half = window / 2;
    return { mx: a.ex + a.nx * half, mz: a.ez + a.nz * half };
}

/**
 * stage 11b's decoupled measurement window (metres): the AASHTO side-slope term alone —
 * `(π/2)·cutDepth/SIDE_SLOPE_LIMIT`, the same derivation `terrain/flatten.ts`'s `computeFalloff` uses for
 * its own non-floor branch — deliberately omitting `computeFalloff`'s sampling floor (denominated in
 * `SPACING`, or a differently-derived floor on stage 11a's still-unshipped branch). The floor exists only
 * to keep the *mesh's* piecewise-linear reconstruction resolvable; it carries no side-slope-safety meaning
 * and is exactly the term stage 11 gates (the floor is what widens when 11a's falloff scale grows). This
 * function's only input is `cutDepth` — the network's own measured cut, `buildNetworkGeometry`'s output,
 * upstream of `computeFalloff` entirely — so by construction no floor-widening treatment can reach it: the
 * treatment appears in neither the search window nor (via {@link heightMidpointAnchor}) the threshold
 * anchor built from it. `grazingCapture.ts`'s `heightSilhouette` scans and anchors against this instead of
 * `computeFalloff`'s output, so two networks whose floor differs (today's `SPACING`, 11a's derived
 * multiple of it, any future floor) read on the same window and are comparable. Known limit: whenever a
 * floor actually binds (`computeFalloff`'s output exceeds this), the *real* rendered transition is wider
 * than this window measures against — an accepted scope boundary (`shallot-roads.md` stage 11b). **Live on
 * this branch**, where 11a's sampling floor (`FALLOFF_SAMPLE_SEGMENTS · SPACING` = 16 m) binds against this
 * network's side-slope term of 14.026 m. The consequence is quantified: `heightMidpointAnchor` places the
 * ground-truth anchor at `window / 2`, but the real ease reaches its midpoint at `computeFalloff / 2`, so a
 * bound floor offsets the anchor outward by `(computeFalloff − sideSlopeWindow) / 2` — 0.988 m here. That is
 * stage 10's anchor-offset bias recurring at reduced magnitude, and it inflates the reading rather than
 * deflating it, so a floor measured this way is scored against itself. Readings stay comparable across floor
 * derivations (the point of this window) but are *absolute* only while no floor binds.
 * @example sideSlopeWindow(0) // 0 — no cut, no transition to measure
 * @example sideSlopeWindow(2.976) // 14.026, below computeFalloff(2.976) = 16 — 11a's floor binds here */
export function sideSlopeWindow(cutDepth: number): number {
    return ((Math.PI / 2) * cutDepth) / SIDE_SLOPE_LIMIT;
}
