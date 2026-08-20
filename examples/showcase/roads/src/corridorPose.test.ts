// Stage 24a's px/m budget arm — asserts the corridor-pose capture's admissibility from scene and
// document constants, never measured off the image. Both halves of the budget:
//   1. cutDepth ≥ 5 px of vertical extent (the resolution threshold, anchored on the road's own
//      resolved on-screen width at the default pose: 8 m × f_px / 900 ≈ 5.5 px)
//   2. ≥30 m of unflattened terrain flanking the corridor in frame (the corridor must read *set
//      into* terrain, or the look loses its comparison)
//
// Every constant is justified by a derivation or a document/scene quantity — none by "the arm
// passes here" (per checks.md and this unit's residue on fitted floors). See `corridorPose.ts`'s
// header comment for the full derivation.

import { describe, expect, test } from "bun:test";
import {
    CAMERA_FOV_DEG,
    CORRIDOR_DISTANCE,
    CORRIDOR_PITCH,
    CUT_DEPTH,
    cutDepthPx,
    FALLOFF,
    flankingTerrainM,
    fovHRad,
    fPx,
    HALF_WIDTH,
    VIEWPORT_HEIGHT,
    VIEWPORT_WIDTH,
} from "./corridorPose";

describe("corridor-pose px/m budget (stage 24a)", () => {
    test("f_px is derived from the viewport height and the camera's default FOV", () => {
        // f_px = (h/2) / tan(fov/2) — the perspective focal length in pixels.
        // h = 720 (Playwright default viewport height, playwright.config.ts sets no viewport).
        // fov = 60° (roads.scene sets no fov; render/index.ts Camera trait defaults fov: 60).
        const expected = 720 / 2 / Math.tan((60 * Math.PI) / 180 / 2);
        expect(fPx()).toBeCloseTo(expected, 1);
        expect(fPx()).toBeCloseTo(623.5, 0);
    });

    test("cutDepth projects to ≥5 px of vertical extent at the derived pose", () => {
        // The 5 px anchor: the road's own on-screen width at the default pose (900 m).
        // 8 m (2 × ROAD_HALF_WIDTH) × f_px / 900 = 5.54 px — the smallest extent this
        // artifact demonstrably resolves, so the threshold is a resolved quantity, not a
        // number picked to make something pass.
        const roadWidthPxAtDefault = (2 * HALF_WIDTH * fPx()) / 900;
        expect(roadWidthPxAtDefault).toBeGreaterThanOrEqual(5);
        expect(roadWidthPxAtDefault).toBeLessThanOrEqual(6);

        // The corridor pose must make cutDepth at least as resolvable as the road width.
        expect(cutDepthPx()).toBeGreaterThanOrEqual(5);

        // Re-derive independently from the constants (not via cutDepthPx's shortcut):
        // screen_px = cutDepth × cos(pitch) × f_px / distance
        const independent = (CUT_DEPTH * Math.cos(CORRIDOR_PITCH) * fPx()) / CORRIDOR_DISTANCE;
        expect(independent).toBeGreaterThanOrEqual(5);
        expect(independent).toBeCloseTo(cutDepthPx(), 5);
    });

    test("≥30 m of unflattened terrain flanks the corridor in frame at the derived pose", () => {
        // The corridor half-width is halfWidth + falloff (the road plus its eased transition).
        // The visible lateral half-extent at the target distance is D × tan(fov_h/2).
        // Flanking = lateral half-extent − corridor half-width.
        expect(flankingTerrainM()).toBeGreaterThanOrEqual(30);

        // Re-derive independently:
        const corridorHalfWidth = HALF_WIDTH + FALLOFF;
        const lateralHalfExtent = CORRIDOR_DISTANCE * Math.tan(fovHRad() / 2);
        expect(lateralHalfExtent - corridorHalfWidth).toBeGreaterThanOrEqual(30);
        expect(lateralHalfExtent - corridorHalfWidth).toBeCloseTo(flankingTerrainM(), 5);
    });

    test("the derived distance is the maximum satisfying cutDepth = 5 px at the derived pitch", () => {
        // D = cutDepth × cos(pitch) × f_px / 5 — by construction, so cutDepth reads exactly 5 px.
        const maxDistance = (CUT_DEPTH * Math.cos(CORRIDOR_PITCH) * fPx()) / 5;
        expect(CORRIDOR_DISTANCE).toBeCloseTo(maxDistance, 5);
    });

    test("the pitch is half the corridor's average side-slope angle", () => {
        // atan(cutDepth / falloff) / 2 — below the side-slope angle so the camera sees the
        // transition as a surface, with enough elevation for terrain context.
        const sideSlopeAngle = Math.atan(CUT_DEPTH / FALLOFF);
        expect(CORRIDOR_PITCH).toBeCloseTo(sideSlopeAngle / 2, 5);
    });

    test("scene constants are what the derivation cites (not fitted)", () => {
        // Each constant is named and sourced — none justified by "the arm passes here."
        expect(VIEWPORT_HEIGHT).toBe(720); // Playwright default
        expect(VIEWPORT_WIDTH).toBe(1280); // Playwright default
        expect(CAMERA_FOV_DEG).toBe(60); // camera default (roads.scene sets no fov)
        expect(HALF_WIDTH).toBe(4); // ROAD_HALF_WIDTH (overlay/network.ts)
        expect(CUT_DEPTH).toBeCloseTo(1.4404, 3); // buildNetworkGeometry at seed 1337
        expect(FALLOFF).toBeCloseTo(6.7876, 3); // computeFalloff(cutDepth)
    });
});
