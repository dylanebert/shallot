import { expect, setDefaultTimeout, test } from "bun:test";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { CEILING } from "../../scripts/test-tiers";
import { Meshes, MeshInstance, MeshPlugin, registerMesh } from "../../src/core/mesh";
import { AmbientLight, attachTexture, Camera } from "../../src/core/rendering";
import { Transform } from "../../src/core/transform";
import type { World } from "../../src/engine";
import { createApp } from "../../src/engine";
import { Profile, ProfilePlugin } from "../../src/extras/profile";
import {
    MeshRenderPlugin,
    StandardRenderer,
    StandardRenderingPlugin,
} from "../../src/standard/rendering";

setDefaultTimeout(CEILING.node);

const vertices = new Float32Array([
    -2, -2, 0, 0, 0, 0, 1, 0, 2, -2, 0, 1, 0, 0, 1, 0, 2, 2, 0, 1, 0, 0, 1, 1, -2, 2, 0, 0, 0, 0, 1,
    1,
]);
const indices = new Uint32Array([0, 1, 2, 0, 2, 3]);
const initialize = (world: World) => registerMesh(world, { name: "slab", vertices, indices });

test("measure register.node first-frame GPU passes", async () => {
    await setupGlobals();
    const app = await createApp({
        defaults: false,
        plugins: [
            ProfilePlugin,
            StandardRenderingPlugin,
            MeshRenderPlugin,
            { name: "InitializeSlab", dependencies: [MeshPlugin], initialize },
        ],
    });
    try {
        const { world } = app;
        const camera = world.create();
        world.add(camera, Transform, { translation: [0, 0, 5, 0] });
        world.add(camera, Camera);
        world.add(camera, StandardRenderer);
        attachTexture(world, camera, { width: 32, height: 32 });
        world.add(world.create(), AmbientLight, { intensity: 1 });
        const slab = world.create();
        world.add(slab, Transform);
        world.add(slab, MeshInstance, { mesh: world.resource(Meshes).id("slab") });

        world.step(0);
        await world.gpu.device.queue.onSubmittedWorkDone();
        world.step(0);
        await world.gpu.device.queue.onSubmittedWorkDone();
        await new Promise((resolve) => setTimeout(resolve, 25));
        world.step(0);
        await world.gpu.device.queue.onSubmittedWorkDone();

        const profile = world.resource(Profile);
        console.info(
            `[portable-tiers-pass-measure] adapter=${JSON.stringify(world.gpu.adapter)} timestamp-query=${world.gpu.device.features.has("timestamp-query")}`,
        );
        console.info(
            `[portable-tiers-pass-measure] first-frame=${JSON.stringify(
                [...profile.gpuTime].map(([name, ms]) => ({
                    name,
                    ms,
                    fires: profile.gpuFires.get(name),
                })),
            )}`,
        );
        expect(profile.gpuTime.size).toBeGreaterThan(0);
        expect([...profile.gpuFires.values()].every((fires) => fires === 1)).toBe(true);
    } finally {
        app.dispose();
    }
});
