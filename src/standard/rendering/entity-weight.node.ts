import { heapStats } from "bun:jsc";
import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Camera,
    DirectionalLight,
    Materials,
    Meshes,
    type MeshHandle,
    MeshInstance,
    MeshMaterial,
    type Plugin,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
    sphere,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { attachTexture } from "@dylanebert/shallot/rendering";
import type { MaterialHandle } from "@dylanebert/shallot/standard/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";

setDefaultTimeout(TIMEOUT.node);

const POPULATION = 3 * 2 ** 16;

type Part = { mesh: MeshHandle; material: MaterialHandle };

function spawn(world: World, part: Part): number {
    const eid = world.create();
    world.add(eid, Transform, { translation: [0, 0, -10] });
    world.add(eid, MeshInstance, { mesh: part.mesh });
    world.add(eid, MeshMaterial, part.material);
    return eid;
}

function retainedHeap(): number {
    Bun.gc(true);
    Bun.gc(true);
    return heapStats().heapSize;
}

// At a power-of-two population, late camera/light eids double eid-indexed columns and tables;
// 3 × 2^16 sits midway, measuring entity weight rather than that doubling slack.
test("196608 rendered entities bound retained bytes after their first standard-rendered step", async () => {
    const baseline = retainedHeap();
    const scene: Plugin = {
        name: "EntityWeight",
        dependencies: [StandardRenderingPlugin],
        initialize(world) {
            const meshes = world.resource(Meshes);
            const materials = world.resource(Materials);
            const parts: Part[] = [
                {
                    mesh: meshes.add({ name: "sphere", ...sphere(0.5, 8, 4) }),
                    material: materials.add(StandardMaterial()),
                },
                {
                    mesh: meshes.add({ name: "small sphere", ...sphere(0.25, 8, 4) }),
                    material: materials.add(StandardMaterial({ baseColor: [1, 0, 0, 1] })),
                },
            ];
            for (let i = 0; i < POPULATION; i++) spawn(world, parts[i % parts.length]!);
            const camera = world.create();
            world.add(camera, Camera);
            world.add(camera, StandardRenderer);
            world.add(camera, Transform);
            attachTexture(world, camera, { width: 4, height: 4 });
            const sun = world.create();
            world.add(sun, Transform);
            world.add(sun, DirectionalLight);
        },
    };
    const app = await openGpuApp({ plugins: [scene] });
    try {
        app.world.step();
        await app.world.gpu.device.queue.onSubmittedWorkDone();
        const bytes = (retainedHeap() - baseline) / POPULATION;
        console.info(`retained bytes/rendered entity: ${bytes.toFixed(3)}`);
        // Includes each part's components/table rows and late camera/light one-holder columns:
        // eid-indexed raw columns must span their highest holder (CONTRIBUTING: Device and world ownership).
        expect(bytes).toBeLessThanOrEqual(780);
    } finally {
        app.dispose();
    }
});
