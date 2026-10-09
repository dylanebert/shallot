import { createApp } from "@dylanebert/shallot";
import {
    clipVector,
    StandardPhysicsPlugin,
    solvePlanes,
} from "@dylanebert/shallot/standard/physics";
export let controlSink: { frame: number } | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};
export const sink = new Float32Array(1);
export let resultSink: ReturnType<typeof solvePlanes> | undefined;
export let vectorSink: ReturnType<typeof clipVector> | undefined;
export default async function create() {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const target = { x: -1, y: -1, z: 1 };
    const velocity = { x: -1, y: -1, z: 1 };
    const out = { delta: { x: 0, y: 0, z: 0 }, iterationCount: 0 };
    const clipped = { x: 0, y: 0, z: 0 };
    const planes = [
        {
            plane: { normal: { x: 1, y: 0, z: 0 }, offset: 0 },
            pushLimit: 3.4028234663852886e38,
            push: 0,
            clipVelocity: true,
        },
        {
            plane: { normal: { x: 0, y: 1, z: 0 }, offset: 0 },
            pushLimit: 3.4028234663852886e38,
            push: 0,
            clipVelocity: true,
        },
    ];
    return {
        step: () => {
            target.x = -1 - (sink[0] % 0.1);
            const result = solvePlanes(target, planes, 2, out);
            const vector = clipVector(velocity, planes, 2, clipped);
            resultSink = result;
            vectorSink = vector;
            sink[0] =
                result.delta.x + result.delta.y + result.delta.z + vector.x + vector.y + vector.z;
        },
        wait: async () => {
            await app.world.gpuIfAvailable?.device.queue.onSubmittedWorkDone();
        },
        dispose: () => app.dispose(),
    };
}
