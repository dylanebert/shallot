import { build, CharacterPlugin, InputPlugin, PhysicsPlugin } from "@dylanebert/shallot";
import { getComponent } from "@dylanebert/shallot/ecs";
import { Demo } from "./demo";

// A compile-time constant equal to Time.FIXED_DT: that one is an object property, so passing it
// across the arrow boxes a double every frame, which is the harness's cost, not the scene's.
const FIXED_DT = 1 / 60;

// The allocation row's control: one known literal per call, kept live through a module binding, run
// and attributed exactly as `step` is, so an empty site set is not a dead probe.
export let controlSink: { frame: number } | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};

// Bundled for Node by the allocation row, which passes the scene as XML text since the runtime's
// file loader is Bun or fetch. Same CPU composition as the demo rows.
export default async function create(scene: string) {
    const app = await build({
        defaults: false,
        plugins: [PhysicsPlugin, CharacterPlugin, InputPlugin, Demo],
        scene,
    });
    const state = app.state;
    // Pose is the composition's one non-Body slab component; the package root does not export it.
    const pose = getComponent("pose");
    if (!pose) throw new Error("allocation entry: the composition registers no `pose` component");
    let eid = 0;
    return {
        step: () => state.step(FIXED_DT),
        // The transition row's event frames, an ECS entity cycle with no Body: create an entity carrying the
        // composition's non-Body slab component, step its frame, destroy it, step its frame.
        spawn: () => {
            eid = state.create();
            state.add(eid, pose);
            state.step(FIXED_DT);
        },
        despawn: () => {
            state.destroy(eid);
            state.step(FIXED_DT);
        },
        dispose: () => app.dispose(),
    };
}
