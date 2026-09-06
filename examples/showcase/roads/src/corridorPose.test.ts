// The corridor-pose px/m budget arm — asserts the corridor-pose capture's admissibility from scene and
// document constants, never measured off the image. Three assertions:
//   1. cutDepth ≥ 5 px of vertical extent (the resolution threshold, anchored on the road's own
//      resolved on-screen width at the *measurement point* — the earlier scene default,
//      900 m: 8 m × f_px / 900 ≈ 5.5 px. Every 900 in this file is the measurement point the
//      derivation is anchored on, never the scene's current default (120 m); the literals are
//      fixed on purpose and re-fitting them to the scene would track a presentation choice)
//   2. ≥30 m of unflattened terrain flanking the corridor in frame (the corridor must read *set
//      into* terrain, or the look loses its comparison)
//   3. earthwork px vs confounder px in the same frame — the failure mode's magnitude compared
//      against the loudest confounder (natural relief), per Validation's own comparison
//
// The pose (CORRIDOR_PITCH, CORRIDOR_DISTANCE) is fixed literals in corridorPose.ts. This arm
// independently re-derives cutDepth from the real network geometry (buildNetworkGeometry) and
// computes the projected extents from that independent cutDepth plus the fixed-literal pose. So a
// future stage that moves cutDepth reds this arm — the pose does not auto-adjust, which is the
// whole point.

import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
    CAMERA_FOV_DEG,
    CORRIDOR_DISTANCE,
    CORRIDOR_PITCH,
    CUT_DEPTH,
    FALLOFF,
    flankingTerrainM,
    fovHRad,
    fPx,
    HALF_WIDTH,
    VIEWPORT_HEIGHT,
    VIEWPORT_WIDTH,
    verticalPx,
} from "./corridorPose";
import { generateNetwork, ROAD_HALF_WIDTH } from "./overlay/network";
import { buildNetworkGeometry, computeFalloff } from "./terrain/flatten";

// The default orbit's pitch in radians (roads.scene: pitch: 0.5) — used to scale the confounder
// from the default pose to the corridor pose.
const DEFAULT_PITCH = 0.5;

// The confounder's measured pixel extent at the measurement point (900 m, pitch 0.5 — the
// earlier scene default, not the current one): the spec records
// ~8–10 px of natural-relief silhouette waviness in the same frame.
// Midpoint of the recorded range.
const CONFUNDER_PX_AT_DEFAULT = 9;

const __dirname = dirname(fileURLToPath(import.meta.url));

describe("corridor-pose px/m budget (stage 24a)", () => {
    // Independently derive cutDepth and falloff from the real network — not from the fixed literals.
    // `generateNetwork` takes no seed — the boot road is a fixed standard chord; 1337 here is only the
    // terrain's noise seed `buildNetworkGeometry` samples natural height against.
    const doc = generateNetwork();
    const { cutDepth: independentCutDepth } = buildNetworkGeometry(doc, 1337);
    const independentFalloff = computeFalloff(independentCutDepth);

    test("the fixed-literal document constants match the independently derived network geometry", () => {
        // If a future stage changes route selection or the generator, cutDepth changes and these
        // pins red — which is the trigger to re-derive the pose, not to silently auto-adjust it.
        expect(independentCutDepth).toBeCloseTo(CUT_DEPTH, 3);
        expect(independentFalloff).toBeCloseTo(FALLOFF, 3);
        expect(HALF_WIDTH).toBe(ROAD_HALF_WIDTH);
    });

    test("f_px is derived from the viewport height and the camera's default FOV", () => {
        const expected = VIEWPORT_HEIGHT / 2 / Math.tan((CAMERA_FOV_DEG * Math.PI) / 180 / 2);
        expect(fPx()).toBeCloseTo(expected, 1);
        expect(fPx()).toBeCloseTo(623.5, 0);
    });

    test("cutDepth projects to ≥5 px of vertical extent at the fixed-literal pose", () => {
        // The 5 px anchor: the road's own on-screen width at the measurement point (900 m, the
        // earlier scene default). The literal is provenance, not the live scene value.
        const roadWidthPxAtDefault = (2 * HALF_WIDTH * fPx()) / 900;
        expect(roadWidthPxAtDefault).toBeGreaterThanOrEqual(5);
        expect(roadWidthPxAtDefault).toBeLessThanOrEqual(6);

        // Independently compute earthwork px from the real cutDepth + the fixed-literal pose.
        // If cutDepth shrinks (a future stage moves the subject), this reds because
        // CORRIDOR_DISTANCE is a fixed literal that does not auto-adjust.
        const earthworkPx = verticalPx(independentCutDepth);
        expect(earthworkPx).toBeGreaterThanOrEqual(5);

        // Re-derive independently from the constants (not via verticalPx):
        const independent =
            (independentCutDepth * Math.cos(CORRIDOR_PITCH) * fPx()) / CORRIDOR_DISTANCE;
        expect(independent).toBeCloseTo(earthworkPx, 5);
    });

    test("≥30 m of unflattened terrain flanks the corridor in frame at the fixed-literal pose", () => {
        expect(flankingTerrainM()).toBeGreaterThanOrEqual(30);

        const corridorHalfWidth = HALF_WIDTH + FALLOFF;
        const lateralHalfExtent = CORRIDOR_DISTANCE * Math.tan(fovHRad() / 2);
        expect(lateralHalfExtent - corridorHalfWidth).toBeGreaterThanOrEqual(30);
        expect(lateralHalfExtent - corridorHalfWidth).toBeCloseTo(flankingTerrainM(), 5);
    });

    test("earthwork px vs confounder px in the same frame (Validation's comparison)", () => {
        // Scale the confounder from the measurement point (900 m, pitch 0.5) to the corridor pose.
        // Both earthwork and confounder are vertical displacements projected through f_px/D/cos(θ),
        // so the ratio is independent of D and θ — it is cutDepth / effective_confounder_magnitude,
        // a property of the subject, not the pose. Pinning it makes a cutDepth change visible.
        const confounderPx =
            CONFUNDER_PX_AT_DEFAULT *
            (900 / CORRIDOR_DISTANCE) *
            (Math.cos(CORRIDOR_PITCH) / Math.cos(DEFAULT_PITCH));

        const earthworkPx = verticalPx(independentCutDepth);

        // The earthwork is a non-zero fraction of the confounder — it is resolvable, not drowned.
        expect(earthworkPx).toBeGreaterThan(0);
        expect(confounderPx).toBeGreaterThan(0);

        // Pin the ratio against the FROZEN LITERAL CUT_DEPTH (not independentCutDepth) so a future
        // stage that moves the real cutDepth reds this arm. The ratio is
        // (cutDepth × f_px × cos(0.5)) / (CONFUNDER_PX_AT_DEFAULT × 900) — constant across the
        // interval, so it does not select the distance; it records the comparison. Deriving the
        // expectation from independentCutDepth (as the first attempt did) makes the arm a tautology:
        // CORRIDOR_PITCH and CORRIDOR_DISTANCE cancel between earthworkPx and confounderPx, so the
        // ratio always equals the expectation by construction and no change to the subject can red it
        // — the exact defect the spec's fifth fitted-constant instance names. Using the frozen literal
        // breaks the tautology: if independentCutDepth drifts from CUT_DEPTH the two sides diverge.
        const ratio = earthworkPx / confounderPx;
        const expectedRatio =
            (CUT_DEPTH * fPx() * Math.cos(DEFAULT_PITCH)) / (CONFUNDER_PX_AT_DEFAULT * 900);
        expect(ratio).toBeCloseTo(expectedRatio, 2);
    });

    test("the pitch is half the corridor's average side-slope angle", () => {
        const sideSlopeAngle = Math.atan(independentCutDepth / independentFalloff);
        expect(CORRIDOR_PITCH).toBeCloseTo(sideSlopeAngle / 2, 1);
    });

    test("scene/viewport sourcing: neither config overrides the defaults the pose derives from", () => {
        // The pose derives from VIEWPORT_HEIGHT (720), VIEWPORT_WIDTH (1280), and CAMERA_FOV_DEG
        // (60) — Playwright's default viewport and the camera trait's default FOV. If either config
        // file gains an explicit setting, the pose goes silently wrong while the literals stay
        // green. This arm reads the config files and asserts they set neither, so adding one reds.
        const configText = readFileSync(
            join(__dirname, "..", "playwright.config.ts"),
            "utf-8",
        ).toLowerCase();
        const sceneText = readFileSync(
            join(__dirname, "..", "public", "scenes", "roads.scene"),
            "utf-8",
        ).toLowerCase();

        // Playwright's default viewport is 1280 × 720 — the config must not override it.
        expect(configText).not.toContain("viewport");

        // The camera's default FOV is 60° — the scene must not override it.
        expect(sceneText).not.toContain("fov");

        // The literals must match the defaults they claim.
        expect(VIEWPORT_HEIGHT).toBe(720);
        expect(VIEWPORT_WIDTH).toBe(1280);
        expect(CAMERA_FOV_DEG).toBe(60);
    });
});
