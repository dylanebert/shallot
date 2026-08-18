import type { Plugin, System } from "@dylanebert/shallot";
import { Meshes } from "@dylanebert/shallot/render/core";
import { capturePoints, type ScreenPoint, TRANSITION_TOLERANCE_PX, worldToScreen } from "./capture";
import { gate } from "./gate";
import type { Check } from "./harness";
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
        // F9 reseeds the live procedural network (`terrain.ts`'s `regenerate`) — the same key voxel's own
        // reseed uses. `{ signal: state.signal }` detaches it at `state.dispose()`, no removal code needed.
        window.addEventListener(
            "keydown",
            (e) => {
                if (e.key !== "F9") return;
                e.preventDefault();
                void regenerate((Math.random() * 0x1_0000_0000) >>> 0);
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
    },
};

const RoadsBootPlugin: Plugin = {
    name: "RoadsBoot",
    systems: [BootSystem],
};

export default RoadsBootPlugin;
