// The corridor-pose derivation — the admissible artifact for the release look.
//
// The gate's default-orbit capture (`test-results/roads-capture.png`) is the scene's own orbit
// (`public/scenes/roads.scene`). **At the pose this module was derived against — the earlier
// default, distance 900, pitch 0.5 — the post-selection earthwork's 1.4404 m vertical excursion
// projects to ≈0.9 px**, under the frame's FBM mottling and an order of magnitude below the ~8–10 px
// of natural-relief silhouette waviness, so 24b's hostile read against "the earthwork reads as an
// artificial trench/embankment rather than terrain" was unreadable at that pose whether or not the
// failure was present. **Every 900 m figure in this module and its arm is that historical
// measurement point, not the scene's current default** — the scene default is 120 m
// (where the same excursion projects to ≈6.6 px) for presentation reasons, and this module is
// deliberately not re-anchored on it: the pose and its anchors are fixed literals precisely so they
// cannot drift with an unrelated change. A future reader must not "update" the 900s to
// track the scene — that would re-fit the anchors to the treatment. This module derives a second pose that makes the
// earthwork's vertical excursion resolvable (≥5 px) while keeping ≥30 m of unflattened terrain
// flanking the corridor in frame (the corridor must read *set into* terrain, or the look loses its
// comparison).
//
// This module is pure: it imports nothing. The pose and the document constants are fixed literals
// (the derivation written in comments beside them), so the CPU arm (`corridorPose.test.ts`) can
// import this module under plain `bun test` and independently verify the budget by computing cutDepth
// from the real network geometry. A future stage that moves cutDepth reds the arm because the
// fixed-literal pose no longer satisfies the budget at the new cutDepth — the pose does not
// auto-adjust, which is the whole point (per checks.md's "re-derives its own rule" and this unit's
// Residue: "re-run that arithmetic whenever any earlier stage moves the subject's magnitude").

// ─── Scene constants (Playwright defaults and camera defaults — see sourcing arm) ─────────────
//
// The viewport is Playwright's default: 1280 × 720 (`playwright.config.ts` sets no `viewport`).
// The camera's FOV is 60° (`roads.scene` sets no `fov`; `render/index.ts`'s Camera trait defaults
// `fov: 60`). These are assumed, not verified by this module — the sourcing arm in
// `corridorPose.test.ts` checks that neither config file overrides the default.

/** Playwright's default viewport height in pixels (`playwright.config.ts` sets no `viewport`). */
export const VIEWPORT_HEIGHT = 720;

/** Playwright's default viewport width in pixels. */
export const VIEWPORT_WIDTH = 1280;

/** The camera's vertical field of view in degrees (`roads.scene` sets no `fov`; default 60). */
export const CAMERA_FOV_DEG = 60;

// ─── Document constants (measured from the network geometry at seed 1337) ──────────────────────
//
// cutDepth and falloff are the network's own measured quantities on the boot document — the fixed
// standard chord — at the terrain's own
// pinned noise seed 1337 (`buildNetworkGeometry(generateNetwork(), 1337)`; the noise seed feeds
// `heightAtCpu` alone, the chord itself is no longer seeded). halfWidth is
// `ROAD_HALF_WIDTH` from `overlay/network.ts` (= 4).
//
// These are fixed literals so the pose (below) does not auto-adjust when cutDepth changes — the
// arm independently re-derives cutDepth from the real network and asserts the budget against the
// fixed-literal pose, so a future stage that moves cutDepth reds the arm.

/** The network's measured cut depth on the boot document, in metres. */
export const CUT_DEPTH = 3.872;

/** The network's falloff distance on the boot document, in metres (computeFalloff(CUT_DEPTH)). */
export const FALLOFF = 18.2464;

/** The road's half-width in metres (`ROAD_HALF_WIDTH` from `overlay/network.ts`). */
export const HALF_WIDTH = 4;

// ─── The derived pose (fixed literals — derivation in comments) ────────────────────────────────
//
// f_px (focal length in pixels) = (h/2) / tan(fov/2):
//   f_px = (720/2) / tan(30°) = 360 / 0.57735 = 623.54 px
//
// A vertical world displacement Δh at orbit distance D, pitch θ projects to
//   screen_px = Δh × cos(θ) × f_px / D
//
// Constraint 1 — cutDepth ≥ 5 px vertical:
//   cutDepth × cos(θ) × f_px / D ≥ 5
//   D ≤ cutDepth × cos(θ) × f_px / 5 = 3.8720 × cos(θ) × 623.54 / 5 = 483.06 × cos(θ)
//
// The 5 px anchor is the road's own already-resolved on-screen width at the *measurement point*
// (the earlier scene default, 900 m): 8 m (2 × halfWidth) × f_px / 900 = 5.54 px — the smallest
// extent this artifact was demonstrated to resolve, so the threshold is anchored on a resolved
// quantity in that frame rather than picked to make something pass. The anchor stays at 900 m by
// design: it is the frame the ≈0.9 px inadmissibility reading came from, and re-deriving it at the
// 120 m default (where the road is ~41.6 px) would raise the floor to track a presentation
// choice rather than a resolved quantity. It depends only on halfWidth, f_px, and the 900 m
// measurement point, none of which moved.
//
// Constraint 2 — ≥30 m of unflattened terrain flanking the corridor in frame:
//   D × tan(fov_h/2) − (halfWidth + falloff) ≥ 30
//   D ≥ (halfWidth + falloff + 30) / tan(fov_h/2)
// where fov_h = 2 × atan(tan(fov/2) × aspect) = 2 × atan(tan(30°) × 1280/720) = 91.49°
//   → tan(fov_h/2) = tan(30°) × 1280/720 = 1.0264
//   D ≥ (4 + 18.2464 + 30) / 1.0264 = 50.90 m
//
// The admissible interval is [50.90 m, 483.06 × cos(θ)].
//
// Pitch: half the corridor's average side-slope angle: atan(cutDepth / falloff) / 2 =
//   atan(3.8720 / 18.2464) / 2 = atan(0.21219) / 2 = 0.20911 / 2 = 0.10455 rad (6.0°) — the same
// ratio as under the retired route-selected chord (`computeFalloff`'s own side-slope-limit
// derivation holds cutDepth/falloff roughly constant), so the pitch barely moved despite cutDepth
// and falloff both roughly tripling.
// Below the side-slope angle so the camera sees the transition as a surface (not edge-on), with
// enough elevation to read terrain on both sides while keeping cos(θ) ≈ 1 (0.9945, losing <1%).
//
// Operating point: D = 120 m — inside the interval [50.90, 480.23], carrying margin on the axis that
// matters (the earthwork's vertical extent). At D = 120:
//   earthwork_px = 3.8720 × cos(0.10455) × 623.54 / 120 = 20.01 px (≥5, 4.0× margin over the floor)
//   flanking     = 120 × 1.0264 − (4 + 18.2464) = 100.92 m (≥30, 3.4× margin)
//
// The confounder ratio (earthwork_px / confounder_px) is constant across the
// interval — both earthwork and confounder scale the same way with D and θ — so it does not
// constrain the distance. The absolute 5 px floor is what constrained it. The confounder comparison
// is an assertion (see corridorPose.test.ts), not what selected the distance.

/** Half the corridor's average side-slope angle: atan(CUT_DEPTH / FALLOFF) / 2 = 0.10455 rad.
 *  Fixed literal — does not auto-adjust if cutDepth changes. */
export const CORRIDOR_PITCH = 0.1046;

/** Orbit distance in metres. Fixed literal: 120, inside the admissible interval [50.90, 483.06].
 *  At this distance cutDepth projects to 20.01 px (4.0× margin over the 5 px floor) and 100.92 m of
 *  unflattened terrain flanks the corridor (3.4× margin over the 30 m floor). */
export const CORRIDOR_DISTANCE = 120;

// ─── Budget computation (used by the CPU arm and re-used by the capture) ───────────────────────

/** Focal length in pixels: (viewport_height / 2) / tan(fov / 2). */
export function fPx(): number {
    return VIEWPORT_HEIGHT / 2 / Math.tan((CAMERA_FOV_DEG * Math.PI) / 180 / 2);
}

/** Horizontal field of view in radians: 2 × atan(tan(fov_v/2) × aspect). */
export function fovHRad(): number {
    const fovVRad = (CAMERA_FOV_DEG * Math.PI) / 180;
    const aspect = VIEWPORT_WIDTH / VIEWPORT_HEIGHT;
    return 2 * Math.atan(Math.tan(fovVRad / 2) * aspect);
}

/** The vertical screen-pixel extent of a vertical displacement `heightM` at the derived pose. */
export function verticalPx(heightM: number): number {
    return (heightM * Math.cos(CORRIDOR_PITCH) * fPx()) / CORRIDOR_DISTANCE;
}

/** The metres of unflattened terrain flanking the corridor at the derived pose.
 *  Lateral half-extent at the target distance minus the corridor half-width. */
export function flankingTerrainM(): number {
    return CORRIDOR_DISTANCE * Math.tan(fovHRad() / 2) - (HALF_WIDTH + FALLOFF);
}
