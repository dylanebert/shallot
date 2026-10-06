import { createApp, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { physicsWorld } from "@dylanebert/shallot/standard/physics";
import { jointAllocationSubject } from "../joint-allocation.fixture";

export let controlSink: { frame: number } | undefined;
export function control() {
    controlSink = { frame: 0 };
}

export default async function create(input: string) {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const ecs = app.world;
    return {
        step: jointAllocationSubject(
            physicsWorld(ecs)!,
            input === "allocating" ? control : undefined,
        ),
        wait: () => ecs.gpu.device.queue.onSubmittedWorkDone(),
        dispose: () => app.dispose(),
    };
}
