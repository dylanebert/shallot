import { createApp, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { mutationAllocationSubject } from "../mutation-allocation.fixture";
import { physicsState } from "../runtime";

export let controlSink: object | undefined;
export const control = () => {
    controlSink = { mutation: 0 };
};

export default async function create() {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const state = physicsState(app.world);
    if (!state) throw new Error("physics world unavailable after app creation");
    const step = mutationAllocationSubject(state);
    return {
        step,
        dispose: () => app.dispose(),
    };
}
