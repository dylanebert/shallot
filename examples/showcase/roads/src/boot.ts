import type { Plugin, System } from "@dylanebert/shallot";
import { Meshes } from "@dylanebert/shallot/render/core";
import { gate } from "./gate";
import type { Check } from "./harness";

// The roads showcase's boot orchestration, as a plugin (a manifest project has no `main.ts` entry) —
// the same shape as voxel's `boot.ts`. Terrain generation itself runs inside `terrain.ts`'s own `warm()`
// (the grid's topology is fixed, so there's no separate "wait for buffers, then generate" step the way
// voxel's carve-capable mesher needs); this plugin's only job is installing the device gate once the
// terrain mesh is registered. `mode: always` so the poll runs in edit mode too, not just play.

declare global {
    interface Window {
        // the device gate, driven by the project's own Playwright on a real GPU (test/roads.spec.ts).
        __roadsGate?: () => Promise<Check[]>;
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
    update() {
        if (!armed || !Meshes.get("terrain")) return;
        armed = false;
        window.__roadsGate = () => gate();
    },
    dispose() {
        delete window.__roadsGate;
    },
};

const RoadsBootPlugin: Plugin = {
    name: "RoadsBoot",
    systems: [BootSystem],
};

export default RoadsBootPlugin;
