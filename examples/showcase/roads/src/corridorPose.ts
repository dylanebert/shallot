// Stage 24a's corridor-pose derivation — the admissible artifact for the release look (24b).
//
// The gate's default-orbit capture (`test-results/roads-capture.png`) is the scene's own orbit
// (`public/scenes/roads.scene`: distance 900, pitch 0.5), where the post-selection earthwork's
// 1.4404 m vertical excursion projects to ≈0.9 px — under the frame's FBM mottling and an order of
// magnitude below the ~8–10 px of natural-relief silhouette waviness. 24b's hostile read against
// "the earthwork reads as an artificial trench/embankment rather than terrain" is unreadable at that
// pose whether or not the failure is present. This module derives a second pose that makes the
// earthwork's vertical excursion resolvable (≥5 px) while keeping ≥30 m of unflattened terrain
// flanking the corridor in frame (the corridor must read *set into* terrain, or the look loses its
// comparison).
//
// This module is pure (no `@dylanebert/shallot` imports) so the CPU arm (`corridorPose.test.ts`)
// can import it under `bun test` and the browser-side `corridorCapture.ts` can import it alongside
// its `Orbit.*` writes — the same separation as `flatten-math.ts` / `flatten.ts`.

import { generateNetwork, ROAD_HALF_WIDTH } from "./overlay/network";
import { buildNetworkGeometry } from "./terrain/flatten";
import { computeFalloff } from "./terrain/flatten-math";

// ─── Scene constants (from the scene file and the camera defaults) ────────────────────────────
//
// The viewport is Playwright's default: 1280 × 720 (`playwright.config.ts` sets no `viewport`).
// The camera's FOV is 60° (`roads.scene` sets no `fov`; `render/index.ts`'s Camera trait defaults
// `fov: 60`). These are the scene's own quantities, not fitted to this arm.

/** Playwright's default viewport height in pixels (`playwright.config.ts` sets no `viewport`). */
export const VIEWPORT_HEIGHT = 720;

/** Playwright's default viewport width in pixels. */
export const VIEWPORT_WIDTH = 1280;

/** The camera's vertical field of view in degrees (`roads.scene` sets no `fov`; default 60). */
export const CAMERA_FOV_DEG = 60;

// ─── Document constants (from the network geometry and the flatten math) ──────────────────────
//
// cutDepth and falloff are the network's own measured quantities at the pinned seed 1337
// (`buildNetworkGeometry(generateNetwork(1337), 1337)`), the same figures the spec's Live log
// records (1.4404 m → 6.7876 m). halfWidth is `ROAD_HALF_WIDTH` from `overlay/network.ts`.

const doc = generateNetwork(1337);
const { cutDepth } = buildNetworkGeometry(doc, 1337);
const falloff = computeFalloff(cutDepth);
const halfWidth = ROAD_HALF_WIDTH;

/** The network's measured cut depth at seed 1337, in metres. */
export const CUT_DEPTH = cutDepth;

/** The network's falloff distance at seed 1337, in metres. */
export const FALLOFF = falloff;

/** The road's half-width in metres (`ROAD_HALF_WIDTH` from `overlay/network.ts`). */
export const HALF_WIDTH = halfWidth;

// ─── The derived pose ─────────────────────────────────────────────────────────────────────────
//
// f_px (focal length in pixels) = (h/2) / tan(fov/2):
//   f_px = (720/2) / tan(30°) = 360 / 0.57735 = 623.54 px
//
// A vertical world displacement Δh at orbit distance D, pitch θ projects to
//   screen_px = Δh × cos(θ) × f_px / D
// (the component of the vertical displacement perpendicular to the view direction at pitch θ).
//
// Constraint 1 — cutDepth ≥ 5 px vertical:
//   cutDepth × cos(θ) × f_px / D ≥ 5
//   D ≤ cutDepth × cos(θ) × f_px / 5 = 1.4404 × cos(θ) × 623.54 / 5 = 179.6 × cos(θ)
//
// The 5 px anchor is the road's own already-resolved on-screen width at the current pose:
// 8 m (2 × halfWidth) × f_px / 900 = 5.54 px — the smallest extent this artifact demonstrably
// resolves, so the threshold is anchored on a resolved quantity in the same frame rather than
// picked to make something pass.
//
// Constraint 2 — ≥30 m of unflattened terrain flanking the corridor in frame:
//   D × tan(fov_h/2) − (halfWidth + falloff) ≥ 30
//   D ≥ (halfWidth + falloff + 30) / tan(fov_h/2)
// where fov_h = 2 × atan(tan(fov/2) × aspect) = 2 × atan(tan(30°) × 1280/720) = 91.48°
//   → tan(fov_h/2) = tan(30°) × 1280/720 = 1.0264
//   D ≥ (4 + 6.7876 + 30) / 1.0264 = 39.74 m
//
// Pitch derivation: the corridor's average side-slope angle is atan(cutDepth / falloff) =
// atan(1.4404 / 6.7876) = 0.2094 rad (12.0°). The pitch is set to half this angle — below the
// side-slope angle so the camera sees the transition as a surface (not edge-on), with enough
// elevation to read terrain on both sides while keeping the vertical excursion maximally
// projected (cos(0.1047) = 0.9945, losing <1% of the vertical signal).
//
// Distance: set to the maximum satisfying cutDepth = 5 px at the derived pitch
// (maximum terrain context at the resolution threshold):
//   D = 179.6 × cos(0.1047) = 178.6 m
//
// Verification:
//   cutDepth px = 1.4404 × cos(0.1047) × 623.54 / 178.6 = 5.00 px ≥ 5  ✓
//   flanking    = 178.6 × 1.0264 − (4 + 6.7876) = 172.6 m ≥ 30        ✓
//
// A reviewer at stage 24's split prescribed distance ~120, pitch ~0.12 — taken as measurement,
// not remedy. This derivation's pitch (0.1047) lands near that band; the distance (178.6) lands
// ~50% beyond it. The difference: this derivation sets cutDepth to exactly 5 px (the resolution
// threshold), so D is the maximum distance that still resolves the excursion. D = 120 would give
// cutDepth ≈7.4 px — more margin, but the threshold is 5, not 7, and no derivation selects 7.

/** Half the corridor's average side-slope angle: atan(cutDepth / falloff) / 2.
 *  Below the side-slope angle so the camera sees the transition as a surface; enough elevation
 *  to read terrain on both sides while keeping cos(θ) ≈ 1. */
export const CORRIDOR_PITCH = Math.atan(CUT_DEPTH / FALLOFF) / 2;

/** Maximum orbit distance at which cutDepth projects to ≥5 px at {@link CORRIDOR_PITCH}.
 *  Derived: cutDepth × cos(pitch) × f_px / 5. */
export const CORRIDOR_DISTANCE = (CUT_DEPTH * Math.cos(CORRIDOR_PITCH) * fPx()) / 5;

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

/** The vertical screen-pixel extent of {@link CUT_DEPTH} at the derived pose. */
export function cutDepthPx(): number {
    return (CUT_DEPTH * Math.cos(CORRIDOR_PITCH) * fPx()) / CORRIDOR_DISTANCE;
}

/** The metres of unflattened terrain flanking the corridor at the derived pose.
 *  Lateral half-extent at the target distance minus the corridor half-width. */
export function flankingTerrainM(): number {
    return CORRIDOR_DISTANCE * Math.tan(fovHRad() / 2) - (HALF_WIDTH + FALLOFF);
}
