import {
    CharacterPlugin,
    createApp,
    GlobalTransform,
    InputPlugin,
    PhysicsPlugin,
} from "@dylanebert/shallot";

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

// Bundled for Node by the allocation row. Same GPU composition as the demo rows.
export default async function create() {
    const app = await createApp({
        defaults: false,
        plugins: [PhysicsPlugin, CharacterPlugin, InputPlugin, Demo],
    });
    const world = app.world;
    let eid = 0;
    return {
        step: () => world.step(FIXED_DT),
        wait: () => world.gpu.device.queue.onSubmittedWorkDone(),
        // The transition row's event frames, an ECS entity cycle with no Body: create an entity carrying the
        // composition's non-Body slab component, step its frame, destroy it, step its frame.
        spawn: () => {
            eid = world.create();
            world.add(eid, GlobalTransform);
            world.step(FIXED_DT);
        },
        despawn: () => {
            world.destroy(eid);
            world.step(FIXED_DT);
        },
        dispose: () => app.dispose(),
    };
}
