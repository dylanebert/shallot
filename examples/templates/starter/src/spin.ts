import { type Plugin, type State, type System, Part, quat, Transform } from "@dylanebert/shallot";

// A plugin is plain data: components + systems the engine runs. This system spins every Part
// around Y. It runs when the project plays (the editor's preview, or a shipped build) but not
// while you edit — the editor (`bunx shallot`) builds in edit mode, where unannotated systems
// are skipped, so the cube stays put and editable there. Delete this file (and its
// `shallot.json` entry) for a static scene.
const SpinSystem: System = {
    group: "simulation",
    update(state: State) {
        // Derive the angle from elapsed time, not a module-level accumulator, so it stays
        // correct after a hot reload or State rebuild rather than carrying stale rotation.
        const q = quat(0, (state.time.elapsed * 45) % 360, 0);
        for (const eid of state.query([Part, Transform])) {
            Transform.rot.set(eid, q.x, q.y, q.z, q.w);
        }
    },
};

// The default export is the plugin — `shallot.json` references this file by path and imports
// its default. The name ("Spin") is how the plugin menu lists it.
const SpinPlugin: Plugin = { name: "Spin", systems: [SpinSystem] };
export default SpinPlugin;
