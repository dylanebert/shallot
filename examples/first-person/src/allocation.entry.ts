import {
    Body,
    build,
    CharacterPlugin,
    InputPlugin,
    PhysicsPlugin,
    ShapeKind,
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

// Bundled for Node by the allocation row, which passes the scene as XML text since the runtime's
// file loader is Bun or fetch. Same CPU composition as the demo rows.
export default async function create(scene: string) {
    const app = await build({
        defaults: false,
        plugins: [PhysicsPlugin, CharacterPlugin, InputPlugin, Demo],
        scene,
    });
    const state = app.state;
    let crate = 0;
    return {
        step: () => state.step(FIXED_DT),
        // The transition row's event frames: a dynamic box spawned far above and aside the course, so it
        // falls free of every contact while it lives, then despawned; each steps its one frame.
        spawn: () => {
            crate = state.create();
            state.add(crate, Body);
            Body.shape.set(crate, ShapeKind.Box);
            Body.pos.set(crate, 60, 40, 0, 0);
            Body.quat.set(crate, 0, 0, 0, 1);
            Body.halfExtents.set(crate, 0.5, 0.5, 0.5, 0);
            Body.mass.set(crate, 1);
            Body.friction.set(crate, 0.5);
            state.step(FIXED_DT);
        },
        despawn: () => {
            state.destroy(crate);
            state.step(FIXED_DT);
        },
        dispose: () => app.dispose(),
    };
}
