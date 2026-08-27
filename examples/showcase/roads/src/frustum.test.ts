// The frustum-coverage arm — Validation's "Camera frustum covers the world it can reach".
//
// The engine's Camera trait defaults `far: 1000` (`packages/shallot/src/standard/render/index.ts:369`).
// The scene's orbit can reach `max-distance` from the target, and the grid's far corner sits
// `WORLD_HALF·√2` beyond the world origin — so the farthest reachable point is `max-distance + WORLD_HALF·√2`
// from the camera. If `far` is below that, the far grid corner clips behind the far plane at some
// orbit position the user can actually reach.
//
// This arm parses the scene file's own `camera` and `orbit` attributes and asserts
// `far ≥ max-distance + WORLD_HALF·√2`. The orbit numbers are read from the scene file, never
// duplicated into the arm, and the treatment (`far`'s value) appears on neither side of the
// derivation — so it is a claim about the mechanism (nothing the camera can orbit to may sit behind
// the far plane), not a restatement of the value it polices. If `far` is absent from the scene's
// camera attribute, the engine default (1000) applies and the arm reds — which is the bug this
// arm polices.

import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { WORLD_HALF } from "./terrain/grid";

const __dirname = dirname(fileURLToPath(import.meta.url));

const SCENE_PATH = join(__dirname, "..", "public", "scenes", "roads.scene");

/** The engine's Camera trait default for `far` (`render/index.ts:369`). */
const ENGINE_FAR_DEFAULT = 1000;

/** Parse a semicolon-separated `key: value` attribute string into a Map. */
function parseAttributePairs(attr: string): Map<string, string> {
    const map = new Map<string, string>();
    for (const pair of attr.split(";")) {
        const colon = pair.indexOf(":");
        if (colon === -1) continue;
        const key = pair.slice(0, colon).trim();
        const value = pair.slice(colon + 1).trim();
        if (key) map.set(key, value);
    }
    return map;
}

/** Extract the value of a quoted attribute (`name="value"`) from the scene file text. */
function extractAttribute(sceneText: string, attrName: string): string | null {
    const match = sceneText.match(new RegExp(`${attrName}="([^"]*)"`));
    return match ? match[1] : null;
}

describe("camera frustum covers the world it can reach (stage 25)", () => {
    const sceneText = readFileSync(SCENE_PATH, "utf-8");

    const cameraAttr = extractAttribute(sceneText, "camera");
    const orbitAttr = extractAttribute(sceneText, "orbit");

    test("the scene defines both camera and orbit attributes", () => {
        expect(cameraAttr).not.toBeNull();
        expect(orbitAttr).not.toBeNull();
    });

    test("far ≥ max-distance + WORLD_HALF·√2 (nothing the camera can reach clips behind the far plane)", () => {
        const camera = parseAttributePairs(cameraAttr!);
        const orbit = parseAttributePairs(orbitAttr!);

        // far: read from the scene's camera attribute, or fall back to the engine default.
        const far = camera.has("far")
            ? Number.parseInt(camera.get("far")!, 10)
            : ENGINE_FAR_DEFAULT;

        // max-distance: read from the scene's orbit attribute — never duplicated into the arm.
        expect(orbit.has("max-distance")).toBe(true);
        const maxDistance = Number.parseInt(orbit.get("max-distance")!, 10);

        // The derived bound: the farthest reachable point from the camera is the orbit's
        // max-distance plus the grid's half-diagonal (WORLD_HALF·√2).
        const bound = maxDistance + WORLD_HALF * Math.sqrt(2);

        expect(
            far,
            `far ${far} must be ≥ max-distance ${maxDistance} + WORLD_HALF·√2 ${WORLD_HALF}·${Math.sqrt(2).toFixed(4)} = ${bound.toFixed(1)}`,
        ).toBeGreaterThanOrEqual(bound);
    });
});
