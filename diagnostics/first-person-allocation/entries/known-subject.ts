import { createApp, CharacterPlugin, InputPlugin, PhysicsPlugin } from "../../../src/index";
import { Demo } from "../../../examples/first-person/src/demo";
const FIXED_DT = 1 / 60;
export let controlSink: { frame: number } | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};
let subjectSink: { frame: number } | undefined;
export default async function create(scene: string) {
    const app = await createApp({ defaults: false, plugins: [PhysicsPlugin, CharacterPlugin, InputPlugin, Demo], scene });
    const world = app.world;
    return {
        step: () => {
            subjectSink = { frame: 0 };
            world.step(FIXED_DT);
        },
        wait: () => world.gpu.device.queue.onSubmittedWorkDone(),
        dispose: () => app.dispose(),
    };
}
