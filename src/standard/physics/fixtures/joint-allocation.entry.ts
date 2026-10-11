import { createApp, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { jointAllocationSubject } from "../joint-allocation.fixture";
import { physicsState } from "../runtime";

export let controlSink: { frame: number } | undefined;
export function control() {
    controlSink = { frame: 0 };
}

export default async function create(input: string) {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const ecs = app.world;
    return {
        step: jointAllocationSubject(
            physicsState(ecs)!,
            input === "allocating" ? control : undefined,
        ),
        dispose: () => app.dispose(),
    };
}
