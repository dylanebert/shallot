import type { Plugin, System } from "@dylanebert/shallot";
import { Meshes } from "@dylanebert/shallot/render/core";
import {
    capturePoints,
    markingProbePoints,
    meshHeightAt,
    type ScreenPoint,
    TRANSITION_TOLERANCE_PX,
    withHeight,
    worldToScreen,
} from "./capture";
import { corridorCapture } from "./corridorCapture";
import { applyEdit, clampDragTarget, clampToBound, handlePositions } from "./edit";
import { checkSurfaceFlatness } from "./flatness";
import { gate } from "./gate";
import type { Check } from "./harness";
import { checkPosts } from "./posts";
import {
    editDocument,
    getDocument,
    overlayIdle,
    readVertices,
    regenerate,
} from "./terrain/terrain";

// The roads showcase's boot orchestration, as a plugin (a manifest project has no `main.ts` entry) —
// the same shape as voxel's `boot.ts`. Terrain generation itself runs inside `terrain.ts`'s own `warm()`
// (the grid's topology is fixed, so there's no separate "wait for buffers, then generate" step the way
// voxel's carve-capable mesher needs); this plugin's only job is installing the device gate + the capture
// bridge once the terrain mesh is registered, and the seed control's F9 key (the spec's "a seed control in
// the voxel-toolbar idiom at most" — a key, not a toolbar: this example has no drawing tool for a toolbar
// to switch between). `mode: always` so the poll runs in edit mode too, not just play.

declare global {
    interface Window {
        // the device gate, driven by the project's own Playwright on a real GPU (test/roads.playwright.ts).
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
        // which reaches `standard/sear/codegen.ts` reading `GPUTextureUsage` at module scope — a
        // `ReferenceError` under Node; `test/roads.playwright.ts` stays bridge-only).
        __roadsTransitionTolerancePx?: number;
        // stage 24a's corridor-pose capture (`corridorCapture.ts`) — repositions the orbit to the
        // derived corridor pose for 24b's release-look screenshot. The Playwright driver calls this,
        // waits, and saves the screenshot to a second file alongside the gate's own.
        __roadsCorridorCapture?: () => Promise<void>;
        // stage 14's reseed-integrity device arm (`test/roads.playwright.ts`) — a deterministic bridge onto the
        // real F9 handler's own `regenerate` call, so the check can drive two reseeds by fixed seed
        // instead of `Math.random()`'s live keypress (which could coincidentally re-touch the probed
        // tile and make a still-stale read look correct by accident). `__roadsHeightAt` re-derives a
        // world point's real generated surface height after a reseed changes it (`capture.ts`'s
        // `withHeight`), since the probe's world (x, z) is fixed but its height isn't once the old
        // network's flatten target is gone.
        __roadsRegenerate?: (seed: number) => Promise<void>;
        __roadsHeightAt?: (x: number, z: number) => Promise<[number, number, number]>;
        // stage 3's marking device probes — four world points derived from the document, each on a
        // distinct marking class (edge line, asphalt, dash gap, dash). The device gate asserts the
        // luminance bands at these points are disjoint.
        __roadsMarkingProbePoints?: () => Promise<{
            edgeLine: [number, number, number];
            asphalt: [number, number, number];
            dashGap: [number, number, number];
            dash: [number, number, number];
        }>;
        // stage 4's edit bridge — drive `__roadsEdit(end, x, z)` to move an endpoint. The target is
        // clamped to the world bounds and then to the `ROAD_MIN_LENGTH` floor (never refused — a
        // constraint on a dragged quantity clamps, never no-ops). Returns true once the edit is applied.
        // The device gate drives this, waits for overlay idle, and reads the flatness and handle
        // position back.
        __roadsEdit?: (end: number, x: number, z: number) => Promise<boolean>;
        // stage 4's flatness bridge — reads `readVertices()` and runs `checkSurfaceFlatness` against the
        // live document, returning the violation counts on both axes.
        __roadsFlatnessViolations?: () => Promise<{
            longitudinal: number;
            crossSection: number;
        }>;
        // stage 4's handle position bridge — returns the two handle entities' world (x, y, z).
        __roadsHandlePos?: () => [[number, number, number], [number, number, number]];
        // stage 5's posts check bridge — reads back the posts buffer and verifies every Validation
        // criterion (y = flattenFieldAt, lateral pinned at exactly `halfWidth + POST_OFFSET` — the kerb
        // line — with the footing ±POST_RADIUS inside the flat core, lateral sign matches
        // postLateralSign(i), live-slot count, scale-0). The device gate drives this after an
        // __roadsEdit to re-verify post placement on the edited chord.
        __roadsPostsCheck?: () => Promise<{ name: string; pass: boolean; detail: string }>;
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
        window.__roadsRegenerate = (seed) => regenerate(seed);
        window.__roadsHeightAt = (x, z) => withHeight(x, z);
        window.__roadsMarkingProbePoints = () => markingProbePoints();
        window.__roadsEdit = async (end, x, z) => {
            const doc = getDocument();
            const [cx, cz] = clampToBound(x, z);
            const [fx, fz] = clampDragTarget(doc, end as 0 | 1, cx, cz);
            const newDoc = applyEdit(doc, end as 0 | 1, fx, fz);
            await editDocument(newDoc);
            return true;
        };
        window.__roadsFlatnessViolations = async () => {
            const raw = await readVertices();
            const doc = getDocument();
            const result = checkSurfaceFlatness((x, z) => meshHeightAt(raw, x, z), doc);
            return {
                longitudinal: result.longitudinal.length,
                crossSection: result.crossSection.length,
            };
        };
        window.__roadsHandlePos = () => handlePositions();
        window.__roadsPostsCheck = () => checkPosts();
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
        delete window.__roadsRegenerate;
        delete window.__roadsHeightAt;
        delete window.__roadsMarkingProbePoints;
        delete window.__roadsEdit;
        delete window.__roadsFlatnessViolations;
        delete window.__roadsHandlePos;
        delete window.__roadsPostsCheck;
    },
};

const RoadsBootPlugin: Plugin = {
    name: "RoadsBoot",
    systems: [BootSystem],
};

export default RoadsBootPlugin;
