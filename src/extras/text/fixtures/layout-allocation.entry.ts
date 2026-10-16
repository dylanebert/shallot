import { Transform } from "../../../core/transform";
import { createApp } from "../../../engine/app";
import { internText, Text, TextPlugin } from "../index";
export let controlSink: object;
export function control() {
    controlSink = { frame: 0 };
}
export default async function create(_input = "", device?: GPUDevice) {
    const app = await createApp({ defaults: false, plugins: [TextPlugin], device });
    const { world } = app;
    const content = internText(world, "label");
    for (let i = 0; i < 1000; i++) {
        const eid = world.create();
        world.add(eid, Transform);
        world.add(eid, Text, { content });
    }
    world.step(0);
    return {
        world,
        warmSamplingSessions: true,
        step: () => world.step(0),
        wait: () => world.gpu.device.queue.onSubmittedWorkDone(),
        dispose: () => app.dispose(),
    };
}
