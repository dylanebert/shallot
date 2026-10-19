import { cube, Meshes, MeshInstance } from "../../src/core/mesh";
import { Transform } from "../../src/core/transform";
import { createApp } from "../../src/engine";
import { gpuAllocationExpressions } from "../../src/engine/runtime/allocation-site";
import {
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    StandardMaterial,
    StandardRenderingPlugin,
} from "../../src/standard/rendering";

let grow: () => void;
export const control = () => grow();

// No view: rendering a view costs the software adapter four times this subject's membership work.
export default async function create() {
    const app = await createApp({
        defaults: false,
        plugins: [MeshRenderPlugin, StandardRenderingPlugin],
    });
    const world = app.world;
    const meshes = world.resource(Meshes);
    let handle = meshes.add({ name: "turnover", ...cube() });
    const mesh = meshes.get(handle)!;
    const eid = world.create();
    world.add(eid, Transform);
    world.add(eid, MeshInstance, { mesh: handle });
    world.add(eid, MeshMaterial, world.resource(Materials).add(StandardMaterial()));
    world.step(0);
    grow = () => {
        meshes.add(mesh);
        world.step(0);
    };
    return {
        world,
        warmSamplingSessions: true,
        samplingWarmFrames: 2_400,
        allocationExpressions: gpuAllocationExpressions,
        allocationFunctions: [world.gpu.sync],
        step() {
            meshes.remove(handle);
            handle = meshes.add(mesh);
            world.storage(MeshInstance).mesh.set(eid, handle);
            world.step(0);
        },
        wait: () => world.gpu.sync(),
        dispose: () => app.dispose(),
    };
}
