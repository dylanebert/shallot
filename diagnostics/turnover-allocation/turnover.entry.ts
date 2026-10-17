import { createApp, Meshes } from "@dylanebert/shallot";
import { attachTexture } from "@dylanebert/shallot/rendering";
import { gpuAllocationExpressions } from "../../src/engine/runtime/allocation-site";
import { Terrain, terrainPlugin } from "../../examples/voxel/src/terrain";

let grow: () => void;
export function control() { grow(); }

export default async function create() {
    const config = { chunkSize: 8, viewDistance: 1, height: 16, relief: 4 };
    const app = await createApp({ plugins: [terrainPlugin(config)] });
    const world = app.world;
    const terrain = world.resource(Terrain);
    attachTexture(world, terrain.camera, { width: 16, height: 16 });
    const meshes = world.resource(Meshes);
    grow = () => {
        meshes.add(terrain.slots[0]!.geometry.mesh);
        world.step(0);
    };
    return {
        world,
        terrain,
        warmSamplingSessions: true,
        samplingWarmFrames: 2_400,
        // Unoptimized wrappers keep their WebGPU allocations off the callers that would inline them.
        allocationExpressions: gpuAllocationExpressions,
        allocationFunctions: [world.gpu.sync],
        step() {
            terrain.viewer += config.chunkSize;
            world.tick();
            world.step(0);
        },
        wait: () => world.gpu.sync(),
        dispose: () => app.dispose(),
    };
}
