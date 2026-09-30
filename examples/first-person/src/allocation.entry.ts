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
// file loader is Bun or fetch. Same GPU composition as the demo rows.
export default async function create(scene: string) {
    const app = await build({
        defaults: false,
        plugins: [PhysicsPlugin, CharacterPlugin, InputPlugin, Demo],
        scene,
    });
    const state = app.state;
    const global = getComponent(state, "global-transform");
    if (!global) throw new Error("allocation entry: the composition registers no GlobalTransform");
    let eid = 0;
    return {
        step: () => state.step(FIXED_DT),
        wait: () => state.gpu.device.queue.onSubmittedWorkDone(),
        // The transition row's event frames, an ECS entity cycle with no Body: create an entity carrying the
        // composition's non-Body slab component, step its frame, destroy it, step its frame.
        spawn: () => {
            eid = state.create();
            state.add(eid, global);
            state.step(FIXED_DT);
        },
        despawn: () => {
            state.destroy(eid);
            state.step(FIXED_DT);
        },
        dispose: () => app.dispose(),
    };
}
