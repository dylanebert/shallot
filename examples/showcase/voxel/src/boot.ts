import { type Mirror, mirror, type Plugin, type State, type System } from "@dylanebert/shallot";
import { initCarve, mountToolbar, setSeed } from "./carve";
import { gate } from "./gate";
import type { Check } from "./harness";
import { type EmitDispatchSample, emitDispatchSample } from "./perf";
import { generate } from "./voxel/generate";
import { Voxels } from "./voxel/mesher";

// The voxel showcase's boot orchestration, as a plugin (a manifest project has no `main.ts` entry).
// The mesher allocates `Voxels.grid` + `.cursor` in its first-frame setup, so the boot can only run once
// they exist: it fills the grid on the GPU (FBM terrain — the live visual), syncs the CPU mirror so the carve
// path can march it, mounts the toolbar + keys, and installs the device gate. The gate + perf probe both
// mirror `Voxels.cursor` (the per-chunk atomic face count) rather than `.indirect` — the CPU-authored draw
// record would make the gate circular. `mode: always` so the terrain meshes
// in edit mode too, not just play. Idempotent per State — `setup` re-arms it each build (ecs.md
// "Reload-safety"); `dispose` tears the UI down so a rebuild doesn't stack overlays.

declare global {
    interface Window {
        // the device gate, driven by the project's own Playwright on a real GPU (test/voxel.playwright.ts).
        __voxelGate?: () => Promise<Check[]>;
        // the structural perf probe, driven by test/perf.playwright.ts.
        __voxelPerf?: () => Promise<EmitDispatchSample>;
    }
}

const SEED = 1337;

let armed = true;
let cursorMirror: Mirror | null = null;
let toolbar: { setTool: (t: "pointer" | "terrain") => void; dispose: () => void } | null = null;

const BootSystem: System = {
    name: "voxel-boot",
    group: "simulation",
    annotations: { mode: "always" },
    setup() {
        armed = true; // setup runs once per State build — re-arm so a rebuild re-boots
    },
    update(state) {
        if (!armed || !Voxels.grid || !Voxels.cursor) return;
        armed = false;
        cursorMirror = mirror(Voxels.cursor);
        const m = cursorMirror;
        void boot(state, m);
    },
    dispose() {
        teardownUi();
        cursorMirror?.dispose();
        cursorMirror = null;
        delete window.__voxelGate;
        delete window.__voxelPerf;
    },
};

async function boot(state: State, m: Mirror): Promise<void> {
    // generate() syncs the CPU mirror itself (the mesher's chunk allocation is CPU-exact), so there's no
    // separate syncGrid() step to sequence here.
    await generate(SEED);
    if (state.signal.aborted) return;
    initCarve(state, document.querySelector("canvas"), SEED);
    mountUi(state);
    window.__voxelGate = () => gate(m);
    window.__voxelPerf = () => emitDispatchSample(m);
}

function mountUi(state: State): void {
    teardownUi();
    toolbar = mountToolbar();
    // the shortcut keys ride a `window` listener (global, outside the canvas) — `{ signal: state.signal }`
    // detaches it at `state.dispose()` with no removal code, so teardownUi only unwinds the toolbar.
    window.addEventListener(
        "keydown",
        (e) => {
            if (e.key === "v" || e.key === "V") toolbar?.setTool("pointer");
            else if (e.key === "b" || e.key === "B") toolbar?.setTool("terrain");
            else if (e.key === "F9") {
                e.preventDefault();
                setSeed((Math.random() * 0x1_0000_0000) >>> 0);
            }
        },
        { signal: state.signal },
    );
}

function teardownUi(): void {
    toolbar?.dispose();
    toolbar = null;
}

const VoxelBootPlugin: Plugin = {
    name: "VoxelBoot",
    systems: [BootSystem],
};

export default VoxelBootPlugin;
