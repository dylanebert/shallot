// Stage 24a's corridor-pose capture — repositions the scene's orbit camera to the derived corridor
// pose (corridorPose.ts) so a screenshot can be taken for 24b's release look.
//
// Exposed as `window.__roadsCorridorCapture` (`boot.ts`); the Playwright driver
// (`test/roads.playwright.ts`) calls it, waits, and saves the screenshot to a second file alongside the
// gate's own `test-results/roads-capture.png` (which stays unchanged in pose — it feeds the
// fs-composite pixel probes and must not move).

import { Orbit } from "@dylanebert/shallot/extras";
import { Views } from "@dylanebert/shallot/render/core";
import { roadFrame } from "./boundaryAnchors";
import { withHeight } from "./capture";
import { CORRIDOR_DISTANCE, CORRIDOR_PITCH } from "./corridorPose";
import { frames } from "./harness";

// Target at the road's midpoint (t = 0.5) — the corridor is visible in both directions with
// terrain flanking on both sides.
const TARGET_T = 0.5;
const EYE_MARGIN = 0.4; // metres above the read terrain height, clearing z-fighting/embedding

/** the canvas-presenting camera's eid — the same `Views` disambiguation `capture.ts`'s `cameraEid`
 *  uses (a shadow-casting light's pooled shadow camera also carries `Camera`, so only a live
 *  `canvas` picks the display camera out). */
function cameraEid(): number {
    for (const [eid, view] of Views) {
        if (view.canvas) return eid;
    }
    throw new Error("corridorCapture: no canvas-presenting camera in Views");
}

/**
 * repositions the scene's orbit camera to the derived corridor pose (looking along the boot
 * document's first road at the derived pitch and distance), waits for the pose to settle, and
 * returns. The caller (`test/roads.playwright.ts` via `window.__roadsCorridorCapture`) takes the
 * screenshot after this resolves.
 */
export async function corridorCapture(): Promise<void> {
    const eid = cameraEid();
    const frame = roadFrame();
    const { ax, az, ux, uz, len } = frame;

    const targetX = ax + ux * len * TARGET_T;
    const targetZ = az + uz * len * TARGET_T;
    const [, targetYRaw] = await withHeight(targetX, targetZ);
    const targetY = targetYRaw + EYE_MARGIN;

    // forward = (ux, uz); Orbit's convention places the camera at
    // target + distance*(sin(yaw)cos(pitch), sin(pitch), cos(yaw)cos(pitch)) looking back at target,
    // so the camera sits behind target along -forward when sin(yaw) = -ux, cos(yaw) = -uz.
    const yaw = Math.atan2(-ux, -uz);

    Orbit.smoothness.set(eid, 1); // snap in one frame instead of easing
    Orbit.pitch.set(eid, CORRIDOR_PITCH);
    Orbit.yaw.set(eid, yaw);
    Orbit.distance.set(eid, CORRIDOR_DISTANCE);
    Orbit.pan.set(eid, targetX, targetY, targetZ, 0);

    await frames(3); // let OrbitSystem (mode: always) re-derive Transform from the snapped pose
}
