import { createApp, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { physicsWorld } from "@dylanebert/shallot/standard/physics";
import { shapeAllocationSubject } from "../shape-allocation.fixture";

export let controlSink: object | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};
export default async function create(input: string) {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const step = shapeAllocationSubject(physicsWorld(app.world)!);
    const allocating = input === "allocating";
    return {
        step: () => {
            step();
            if (allocating) controlSink = { frame: 0 };
        },
        wait: () => app.world.gpu.device.queue.onSubmittedWorkDone(),
        dispose: () => app.dispose(),
    };
}
