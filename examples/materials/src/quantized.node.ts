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

test("the Banded lighting exhibit has flat lighting bands across its curved surface", async () => {
    let camera = 0;
    let subject = 0;
    let light = 0;
    const app = await openGpuApp({
        defaults: false,
        plugins: [
            {
                name: "BandedLightingScene",
                dependencies: [
                    StandardRenderingPlugin,
                    MeshRenderPlugin,
                    MaterialPlugin(QuantizedMaterial),
                ],
                initialize(world) {
                    camera = world.create();
                    world.add(camera, Transform, { translation: [0, 0, 3] });
                    world.add(camera, Camera, { antialias: 0 });
                    world.add(camera, StandardRenderer);
                    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
                    attachTexture(world, camera, { width: 64, height: 64 });
                    const sun = world.create();
                    light = sun;
                    world.add(sun, Transform, {
                        rotation: lookAtRotation(0, 0, 0, -0.5, -1, -0.6),
                    });
                    world.add(sun, DirectionalLight, { color: 0xffe8cc, illuminance: 3200 });
                    const eid = world.create();
                    subject = eid;
                    world.add(eid, Transform);
                    world.add(eid, MeshInstance, {
                        mesh: world
                            .resource(Meshes)
                            .add({ name: "Exhibit sphere", ...sphere(0.65, 64, 32) }),
                    });
                    world.add(
                        eid,
                        MeshMaterial,
                        world.resource(QuantizedMaterial).add({
                            ...StandardMaterial({
                                baseColor: [0.72, 0.4, 0.05, 1],
                                perceptualRoughness: 0.8,
                            }),
                            steps: 4,
                        }),
                    );
                },
            },
        ],
    });
    try {
        const { world } = app;
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
        world.storage(DirectionalLight).color.set(light, 0xffffff);
        world.storage(DirectionalLight).illuminance.set(light, 10000);
        const assets = world.resource(QuantizedMaterial);
        const rows = [2, 8].map((steps) =>
            assets.add({
                ...StandardMaterial({ baseColor: [1, 1, 1, 1], perceptualRoughness: 1 }),
                steps,
            }),
        );
        const counts: number[] = [];
        for (const row of rows) {
            world.remove(subject, MeshMaterial);
            world.add(subject, MeshMaterial, row);
            world.step(0);
            world.step(0);
            const shot = await captureTexture(world, camera);
            const levels = new Set<number>();
            for (let y = 24; y < 40; y++)
                for (let x = 24; x < 40; x++) levels.add(shot.rgba[(y * 64 + x) * 4]!);
            counts.push(levels.size);
        }
        console.log("QuantizedMaterial row measurements", counts);
        expect(counts[0]).toBeLessThanOrEqual(3);
        expect(counts[1]).toBeGreaterThan(4);
    } finally {
        app.dispose();
    }
});
