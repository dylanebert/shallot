import { createApp, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { physicsState } from "../runtime";
import { shapeAllocationSubject } from "../shape-allocation.fixture";

export let controlSink: object | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};
export default async function create(input: string) {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const step = shapeAllocationSubject(physicsState(app.world)!, true, input === "upload");
    const allocating = input === "allocating";
    return {
        step: () => {
            step();
            if (allocating) controlSink = { frame: 0 };
        },
        dispose: () => app.dispose(),
    };
}
