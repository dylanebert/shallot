import { Orbit } from "@dylanebert/shallot/extras";
import { Views } from "@dylanebert/shallot/render/core";
import { withHeight, worldToScreen } from "./capture";
import { frames } from "./harness";
import { documentDistance } from "./overlay/document";
import { generateNetwork } from "./overlay/network";
import { SEED } from "./terrain/terrain";

// Stage 9's discriminator — the grazing-view half. The release-look screenshot that opened this stage was
// a hand-orbited shot; this reproduces it as a *fixed, reproducible* camera pose so the two-resolution
// reading is a real comparison and not two different framings. A near-horizontal grazing angle is the
// point, not an aesthetic choice: a small world-space deviation projects to a much larger screen-pixel
// deviation the closer the view direction runs parallel to the surface (the same reason raking light
// reveals machining marks a top-down look hides) — the straightness signal this stage measures needs that
// magnification to be visible in a handful of screen pixels at all.
//
// Reuses the *same* boot document every run (`generateNetwork(SEED)`, `terrain.ts`'s own boot document) —
// road index 0's own first segment, deterministic — so a SPACING/TILE_RES/DIST_RANGE edit between runs
// changes only the mesh/atlas resolution under test, never the network geometry the camera is framed to.

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
