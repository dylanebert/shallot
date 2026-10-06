import { createApp, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { physicsWorld } from "@dylanebert/shallot/standard/physics";
import { mutationAllocationSubject } from "../mutation-allocation.fixture";

export let controlSink: object | undefined;
export const control = () => {
    controlSink = { mutation: 0 };
};

export default async function create() {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const step = mutationAllocationSubject(physicsWorld(app.world)!);
    return {
        step,
        wait: () => app.world.gpu.device.queue.onSubmittedWorkDone(),
        dispose: () => app.dispose(),
    };
}
