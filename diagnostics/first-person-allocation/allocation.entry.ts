import {
    CharacterPlugin,
    createApp,
    GlobalTransform,
    InputPlugin,
    StandardPhysicsPlugin,
    Transform,
} from "@dylanebert/shallot";
import { internText, Text, TextPlugin } from "@dylanebert/shallot/extras";

import { gpuAllocationExpressions } from "../../src/engine/runtime/allocation-site";
import { Demo } from "../../examples/first-person/src/demo";

// A compile-time constant equal to Time.FIXED_DT: that one is an object property, so passing it
// across the arrow boxes a double every frame, which is the harness's cost, not the scene's.
const FIXED_DT = 1 / 60;

// The allocation row's control: one known literal per call, kept live through a module binding, run
// and attributed exactly as `step` is, so an empty site set is not a dead probe.
export let controlSink: { frame: number } | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};

// Bundled for Node by the allocation row. Same GPU composition as the demo rows; input "text" adds
// one world label, whose core pipeline records and submits a GPU frame every step.
export default async function create(input = "") {
    const plugins = [StandardPhysicsPlugin, CharacterPlugin, InputPlugin, Demo];
    if (input === "text") plugins.push(TextPlugin);
    const app = await createApp({ defaults: false, plugins });
    const world = app.world;
    if (input === "text") {
        const label = world.create();
        world.add(label, Transform, { translation: [0, 0.505, 7.5] });
        world.add(label, Text, { content: internText(world, "Text") });
    }
    let eid = 0;
    return {
        allocationExpressions: gpuAllocationExpressions,
        // Keep the frame fence visible so its WebGPU calls are not attributed to the step that inlines it.
        allocationFunctions: [world.gpu.sync],
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
