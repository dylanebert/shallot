import { build, CharacterPlugin, InputPlugin, PhysicsPlugin } from "../../../src/index";
import { Demo } from "../../../examples/first-person/src/demo";
import { Compute } from "../../../src/engine/runtime";
const FIXED_DT = 1 / 60;
export let controlSink: { frame: number } | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};
let subjectSink: { frame: number } | undefined;
export default async function create(scene: string) {
    const app = await build({ defaults: false, plugins: [PhysicsPlugin, CharacterPlugin, InputPlugin, Demo], scene });
    const state = app.state;
    return {
        step: () => {
            subjectSink = { frame: 0 };
            state.step(FIXED_DT);
        },
        wait: () => Compute.device.queue.onSubmittedWorkDone(),
        dispose: () => app.dispose(),
    };
}
