import { expect, setDefaultTimeout, test } from "bun:test";
import { lookAtRotation } from "@dylanebert/shallot";
import { Meshes, MeshInstance, sphere } from "@dylanebert/shallot/mesh";
import {
    attachTexture,
    Camera,
    captureTexture,
    DirectionalLight,
    Tonemapping,
    TonemappingMethod,
} from "@dylanebert/shallot/rendering";
import {
    MaterialPlugin,
    MeshMaterial,
    MeshRenderPlugin,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "@dylanebert/shallot/standard/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { Transform } from "@dylanebert/shallot/transform";
import { QuantizedMaterial } from "./materials";

setDefaultTimeout(TIMEOUT.node);

test("the quantized PBR exhibit has flat lighting bands across its curved surface", async () => {
    const app = await openGpuApp({
        defaults: false,
        plugins: [StandardRenderingPlugin, MeshRenderPlugin, MaterialPlugin(QuantizedMaterial)],
    });
    try {
        const { world } = app;
        const camera = world.create();
        world.add(camera, Transform, { translation: [0, 0, 3] });
        world.add(camera, Camera, { antialias: 0 });
        world.add(camera, StandardRenderer);
        world.add(camera, Tonemapping, { method: TonemappingMethod.None });
        attachTexture(world, camera, { width: 64, height: 64 });
        const sun = world.create();
        world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, -0.5, -1, -0.6) });
        world.add(sun, DirectionalLight, { color: 0xffe8cc, illuminance: 3200 });
        const eid = world.create();
        world.add(eid, Transform);
        world.add(eid, MeshInstance, {
            mesh: world.resource(Meshes).add({ name: "Exhibit sphere", ...sphere(0.65, 64, 32) }),
        });
        world.add(
            eid,
            MeshMaterial,
            world
                .resource(QuantizedMaterial)
                .add(
                    StandardMaterial({ baseColor: [0.72, 0.4, 0.05, 1], perceptualRoughness: 0.8 }),
                ),
        );
        world.step(0);
        world.step(0);
        const { rgba } = await captureTexture(world, camera);
        for (const [channel, name] of ["red", "green", "blue"].entries()) {
            const levels = new Set<number>();
            let neighbors = 0;
            for (let y = 24; y < 40; y++)
                for (let x = 24; x < 40; x++) {
                    const offset = (y * 64 + x) * 4 + channel;
                    levels.add(rgba[offset]!);
                    if (rgba[offset] === rgba[offset + 4]) neighbors++;
                }
            if (channel === 0) expect(levels.size, "multiple lighting bands").toBeGreaterThan(1);
            expect(levels.size, `${name} band levels`).toBeLessThanOrEqual(4);
            expect(neighbors, `${name} flat neighbors`).toBeGreaterThan(200);
        }
    } finally {
        app.dispose();
    }
});
