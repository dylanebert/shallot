import type { Plugin, System } from "@dylanebert/shallot";
import { Meshes } from "@dylanebert/shallot/render/core";
import {
    capturePoints,
    type ScreenPoint,
    TRANSITION_TOLERANCE_PX,
    withHeight,
    worldToScreen,
} from "./capture";
import { corridorCapture } from "./corridorCapture";
import { gate } from "./gate";
import {
    type GrazingAnchor,
    grazingCapture,
    type HeightSilhouetteResult,
    heightSilhouette,
} from "./grazingCapture";
import type { Check } from "./harness";
import { DIST_RANGE, TILE_RES } from "./overlay/tiles";
import { SPACING } from "./terrain/grid";
import { overlayIdle, regenerate } from "./terrain/terrain";

// The roads showcase's boot orchestration, as a plugin (a manifest project has no `main.ts` entry) —
// the same shape as voxel's `boot.ts`. Terrain generation itself runs inside `terrain.ts`'s own `warm()`
// (the grid's topology is fixed, so there's no separate "wait for buffers, then generate" step the way
// voxel's carve-capable mesher needs); this plugin's only job is installing the device gate + the capture
// bridge once the terrain mesh is registered, and the seed control's F9 key (the spec's "a seed control in
// the voxel-toolbar idiom at most" — a key, not a toolbar: this example has no drawing tool for a toolbar
// to switch between). `mode: always` so the poll runs in edit mode too, not just play.

declare global {
    interface Window {
        // the device gate, driven by the project's own Playwright on a real GPU (test/roads.spec.ts).
        __roadsGate?: () => Promise<Check[]>;
        // the pixel-probe capture's world→screen bridge (capture.ts), its on-road/off-road world points
        // (real generated terrain height), and an overlay-drained poll — all real ECS/GPU state Playwright
        // can't compute itself.
        __roadsProbe?: (points: ReadonlyArray<readonly [number, number, number]>) => ScreenPoint[];
        __roadsCapturePoints?: () => ReturnType<typeof capturePoints>;
        __roadsOverlayIdle?: () => boolean;
        // a plain re-export of capture.ts's derived constant — never imported directly into the Playwright
        // driver (this project's src/ runs in the browser via the dev server; a direct Node-side import of
        // it pulls in the published `@dylanebert/shallot` package graph under Playwright's own loader,
        // which chokes on the package's `exports` map — `test/roads.spec.ts` stays bridge-only).
        __roadsTransitionTolerancePx?: number;
        // stage 9's straightness discriminator (`grazingCapture.ts`) — repositions the camera to a fixed
        // grazing pose and returns the analytic boundary anchors a straightness probe scans around. The
        // same bridge-only shape as the rest of this file: `test/straightness.spec.ts` never imports
        // engine/ECS code directly, only reads this window global.
        __roadsGrazingCapture?: () => Promise<{ anchors: GrazingAnchor[] }>;
        // stage 10's corrected straightness criterion (`grazingCapture.ts`'s `heightSilhouette`) — the
        // same anchors, but the world-space height axis instead of the screen-space albedo edge. No camera
        // pose to set up first, unlike `__roadsGrazingCapture` above.
        __roadsHeightSilhouette?: () => Promise<HeightSilhouetteResult>;
        // the live mesh/atlas resolution constants under test — read by `test/straightness.spec.ts` so the
        // two-resolution reading is self-labeling rather than trusting whatever the driver assumed was
        // edited before the run.
        __roadsMeshParams?: { spacing: number; tileRes: number; distRange: number };
        // stage 24a's corridor-pose capture (`corridorCapture.ts`) — repositions the orbit to the
        // derived corridor pose for 24b's release-look screenshot. The Playwright driver calls this,
        // waits, and saves the screenshot to a second file alongside the gate's own.
        __roadsCorridorCapture?: () => Promise<void>;
        // stage 14's reseed-integrity device arm (`test/roads.spec.ts`) — a deterministic bridge onto the
        // real F9 handler's own `regenerate` call, so the check can drive two reseeds by fixed seed
        // instead of `Math.random()`'s live keypress (which could coincidentally re-touch the probed
        // tile and make a still-stale read look correct by accident). `__roadsHeightAt` re-derives a
        // world point's real generated surface height after a reseed changes it (`capture.ts`'s
        // `withHeight`), since the probe's world (x, z) is fixed but its height isn't once the old
        // network's flatten target is gone.
        __roadsRegenerate?: (seed: number) => Promise<void>;
        __roadsHeightAt?: (x: number, z: number) => Promise<[number, number, number]>;
    }
}

let armed = true;

const BootSystem: System = {
    name: "roads-boot",
    group: "simulation",
    annotations: { mode: "always" },
    setup() {
        armed = true; // setup runs once per State build — re-arm so a rebuild re-installs the gate
    },
    update(state) {
        if (!armed || !Meshes.get("terrain")) return;
        armed = false;
        window.__roadsGate = () => gate();
        window.__roadsProbe = (points) => worldToScreen(points);
        window.__roadsCapturePoints = () => capturePoints();
        window.__roadsOverlayIdle = () => overlayIdle();
        window.__roadsTransitionTolerancePx = TRANSITION_TOLERANCE_PX;
        window.__roadsCorridorCapture = () => corridorCapture();
        window.__roadsGrazingCapture = () => grazingCapture();
        window.__roadsHeightSilhouette = () => heightSilhouette();
        window.__roadsMeshParams = { spacing: SPACING, tileRes: TILE_RES, distRange: DIST_RANGE };
        window.__roadsRegenerate = (seed) => regenerate(seed);
        window.__roadsHeightAt = (x, z) => withHeight(x, z);
        // F9 reseeds the live procedural network (`terrain.ts`'s `regenerate`) — the same key voxel's own
        // reseed uses. `{ signal: state.signal }` detaches the listener at `state.dispose()`, no removal
        // code needed.
        window.addEventListener(
            "keydown",
            (e) => {
                if (e.key === "F9") {
                    e.preventDefault();
                    regenerate((Math.random() * 0x1_0000_0000) >>> 0).catch((err) => {
                        console.error("roads: F9 reseed failed", err);
                    });
                }
            },
            { signal: state.signal },
        );
    },
    dispose() {
        delete window.__roadsGate;
        delete window.__roadsProbe;
        delete window.__roadsCapturePoints;
        delete window.__roadsOverlayIdle;
        delete window.__roadsTransitionTolerancePx;
        delete window.__roadsCorridorCapture;
        delete window.__roadsGrazingCapture;
        delete window.__roadsHeightSilhouette;
        delete window.__roadsMeshParams;
        delete window.__roadsRegenerate;
        delete window.__roadsHeightAt;
    },
};

const RoadsBootPlugin: Plugin = {
    name: "RoadsBoot",
    systems: [BootSystem],
};

export default RoadsBootPlugin;
