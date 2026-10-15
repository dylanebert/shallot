import { createApp, PhysicsMeshes, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { meshBuild } from "../geometry-builds";
import { physicsState } from "../runtime";
import { shapeAllocationSubject } from "../shape-allocation.fixture";

export let controlSink: object | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};
export default async function create(input: string) {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    if (input === "mesh-lookup") {
        const world = app.world;
        const id = world.resource(PhysicsMeshes).register({
            name: "lookup",
            vertices: new Float32Array([0, 0, 0, 1, 0, 0, 0, 0, 1]),
            indices: new Uint32Array([0, 2, 1]),
        });
        const built = meshBuild(world, id);
        return {
            step: () => {
                for (let i = 0; i < 64; i++)
                    if (meshBuild(world, id) !== built)
                        throw new Error("mesh lookup missed its build");
            },
            dispose: () => app.dispose(),
        };
    }
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
