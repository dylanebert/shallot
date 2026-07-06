import { type Mirror, mirror, type Plugin, type State, type System } from "@dylanebert/shallot";
import { initCarve, mountToolbar, setSeed } from "./carve";
import { gate } from "./gate";
import type { Check } from "./harness";
import { generate } from "./voxel/generate";
import { syncGrid, Voxels } from "./voxel/mesher";

// The voxel showcase's boot orchestration, as a plugin (a manifest project has no `main.ts` entry).
// The mesher allocates `Voxels.grid` + `.indirect` in its first-frame setup, so the boot can only run once
// they exist: it fills the grid on the GPU (FBM terrain — the live visual), syncs the CPU mirror so the carve
// path can march it, mounts the toolbar + keys, and installs the device gate. `mode: always` so the terrain
// meshes in the editor viewport, not just play. Idempotent per State — `setup` re-arms it each build (ecs.md
// "Reload-safety"); `dispose` tears the UI down so a rebuild doesn't stack overlays.

declare global {
    interface Window {
        // the device gate, driven by the project's own Playwright on a real GPU (test/voxel.spec.ts).
        __voxelGate?: () => Promise<Check[]>;
    }
}

const SEED = 1337;

let armed = true;
let indirect: Mirror | null = null;
let toolbar: { setTool: (t: "pointer" | "terrain") => void; dispose: () => void } | null = null;
let onKey: ((e: KeyboardEvent) => void) | null = null;

const BootSystem: System = {
    name: "voxel-boot",
    group: "simulation",
    annotations: { mode: "always" },
    setup() {
        armed = true; // setup runs once per State build — re-arm so a rebuild re-boots
    },
    update(state) {
        if (!armed || !Voxels.grid || !Voxels.indirect) return;
        armed = false;
        indirect = mirror(Voxels.indirect);
        const m = indirect;
        window.__voxelGate = () => gate(m);
        void boot(state);
    },
    dispose() {
        teardownUi();
        indirect?.dispose();
        indirect = null;
        delete window.__voxelGate;
    },
};

async function boot(state: State): Promise<void> {
    await generate(SEED);
    await syncGrid();
    initCarve(state, document.querySelector("canvas"), SEED);
    mountUi();
}

function mountUi(): void {
    teardownUi();
    toolbar = mountToolbar();
    onKey = (e) => {
        if (e.key === "v" || e.key === "V") toolbar?.setTool("pointer");
        else if (e.key === "b" || e.key === "B") toolbar?.setTool("terrain");
        else if (e.key === "F9") {
            e.preventDefault();
            setSeed((Math.random() * 0x1_0000_0000) >>> 0);
        }
    };
    window.addEventListener("keydown", onKey);
}

function teardownUi(): void {
    toolbar?.dispose();
    toolbar = null;
    if (onKey) window.removeEventListener("keydown", onKey);
    onKey = null;
}

const VoxelBootPlugin: Plugin = {
    name: "VoxelBoot",
    systems: [BootSystem],
};

export default VoxelBootPlugin;
